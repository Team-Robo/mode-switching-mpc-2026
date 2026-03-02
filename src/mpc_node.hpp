#ifndef MPC_NODE_HPP
#define MPC_NODE_HPP
// mpc_node.hpp

#include <ros/ros.h>
#include <nav_msgs/Path.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/PoseStamped.h>
#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/Marker.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <std_msgs/Float64MultiArray.h>
#include <std_srvs/Empty.h>
#include <obstacle_detector/Obstacles.h>
#include <vector>
#include <string>
#include <cmath>
#include <memory>
#include <mutex>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>

// ACADOS interface
extern "C" {
    #include "acados/utils/print.h"
    #include "acados_c/ocp_nlp_interface.h"
    #include "acados_c/external_function_interface.h"
    #include "acados_solver_jackal_diff_drive.h"
}

namespace mpc_controller {

struct DynamicObstacle {
    double x, y;
    double vx, vy;
    double radius;
};

struct PredictedObstacle {
    std::vector<double> x_predicted;
    std::vector<double> y_predicted;
    std::vector<double> vx_predicted;
    std::vector<double> vy_predicted;
    std::vector<double> radius_predicted;
};

// Priority (highest → lowest):
//   DYNAMIC_OBS > RECOVERY > ROTATION_SHIM > RUSH_GOAL > STATIC_OBS > NORMAL
//
// RECOVERY fires when the robot is detected as stuck (position has not changed
// by more than stuck_dist_threshold_ over stuck_timeout_ seconds while a
// non-zero cmd_vel is being sent).  It then executes a finite sequence:
//   ESCAPE_REVERSE  → drive backwards recovery_reverse_dist_ metres
//   ESCAPE_ROTATE   → rotate in place by recovery_rotate_deg_
//   ESCAPE_REPLAN   → stop and request a global replan via move_base
//   (repeat up to recovery_max_attempts_ times)
//   HARD_RESET      → stop and request a replan; if repeated replans fail,
//                     the robot gives up and waits for a new goal.
enum class ControlMode {
    NORMAL,          // No obstacles nearby — full speed cap (v_linear_max_)
    STATIC_OBS,      // Static obstacles detected — hard-capped at v_static_obs_max_
    DYNAMIC_OBS,     // Dynamic obstacles detected — full speed, higher accel weight
    RUSH_GOAL,       // Near goal, no obstacles — blast at v_linear_max_
    ROTATION_SHIM,   // Reversal needed but goal in lidar blind spot — spin in place
    RECOVERY         // Robot is stuck — execute escape sequence then replan
};

// Sub-states of RECOVERY mode
enum class RecoveryStep {
    IDLE,            // Not in recovery
    ESCAPE_REVERSE,  // Reversing away from obstacle
    ESCAPE_ROTATE,   // Rotating 45° to find a new heading
    ESCAPE_REPLAN,   // Waiting for move_base to deliver a new global plan
    HARD_RESET       // Hard stop + replan after max attempts exhausted
};

class MPCNode {
public:
    MPCNode(ros::NodeHandle& nh, ros::NodeHandle& nh_private);
    ~MPCNode();

    void run();

private:
    ros::NodeHandle nh_;
    ros::NodeHandle nh_private_;

    // Publishers
    ros::Publisher pub_vel_;
    ros::Publisher pub_mpc_plan_;
    ros::Publisher pub_marker_;

    std::mutex solver_mutex_;

    // Subscribers
    ros::Subscriber sub_odom_;
    ros::Subscriber sub_global_plan_;
    ros::Subscriber sub_cloud_;
    ros::Subscriber sub_map_cloud_;
    ros::Subscriber sub_dynamic_obstacle_;

    // Service clients
    ros::ServiceClient srv_clear_costmaps_;   // /move_base/clear_costmaps
    ros::ServiceClient srv_make_plan_;        // (optional) not used directly

    // Callbacks
    void callbackOdom(const nav_msgs::Odometry::ConstPtr& msg);
    void callbackGlobalPlan(const nav_msgs::Path::ConstPtr& msg);
    void callbackCloud(const sensor_msgs::PointCloud2::ConstPtr& msg);
    void callbackMapCloud(const sensor_msgs::PointCloud2::ConstPtr& msg);
    void callbackTrackDynamicObstacle(const obstacle_detector::Obstacles::ConstPtr& msg);

    // ACADOS
    jackal_diff_drive_solver_capsule* acados_ocp_capsule_ = nullptr;
    ocp_nlp_config* nlp_config_;
    ocp_nlp_dims*   nlp_dims_;
    ocp_nlp_in*     nlp_in_;
    ocp_nlp_out*    nlp_out_;
    ocp_nlp_solver* nlp_solver_;
    void*           nlp_opts_;

    void initializeAcadosSolver();
    void cleanupAcadosSolver();

    bool solveOCP(const std::vector<double>& x_ref,
                  const std::vector<double>& y_ref,
                  const std::vector<double>& theta_ref,
                  const std::vector<double>& current_state,
                  const std::vector<double>& obs_x,
                  const std::vector<double>& obs_y);

    void publishVelocity(double v, double w);
    void publishTrajectory(const std::vector<double>& x_traj,
                           const std::vector<double>& y_traj);
    void publishMarker();

    // Utility
    double quaternionToYaw(const geometry_msgs::Quaternion& q);
    double headingPreprocess(double center, double target);
    double diffAngle(double a1, double a2) const;
    void   findClosestPoint(const std::vector<double>& x_ref,
                            const std::vector<double>& y_ref,
                            double curr_x, double curr_y, int& min_idx);
    bool   isLeft(double rx, double ry, double rtheta, double ox, double oy);

    // Reversal
    bool checkReversalNeeded(const std::vector<double>& theta_ref, double current_theta);
    std::vector<double> computeReverseThetaRef(const std::vector<double>& x_ref,
                                               const std::vector<double>& y_ref,
                                               double current_theta);

    // Dynamic obstacle prediction
    std::vector<PredictedObstacle> predicted_obstacles_;
    std::vector<PredictedObstacle> predictObstaclesTrajectory(
        const std::vector<DynamicObstacle>& obstacles, double dt, int N);

    // =========================================================================
    // Obstacle selection
    // =========================================================================
    void selectTwoObstacles(
        const std::vector<double>& obs_x,
        const std::vector<double>& obs_y,
        const std::vector<PredictedObstacle>& predicted_obstacles,
        double rx, double ry,
        int stage,
        double search_radius_sq,
        double p_data[4]) const;

    // Emergency stop for 3rd+ dynamic obstacle
    bool checkEmergencyStop(
        const std::vector<PredictedObstacle>& predicted_obstacles,
        const std::vector<double>& current_state) const;

    // =========================================================================
    // RUSH_GOAL helpers
    // =========================================================================
    double distToGoal(double rx, double ry) const;
    void warmStartFromCurrentState(const std::vector<double>& current_state);

    // =========================================================================
    // ROTATION_SHIM helpers
    // =========================================================================
    bool isGoalInBlindSpot(double robot_theta,
                           double goal_x, double goal_y,
                           double robot_x, double robot_y) const;

    // =========================================================================
    // RECOVERY helpers
    // =========================================================================
    // Returns true if the robot is considered stuck.
    // Condition: position has not changed by more than stuck_dist_threshold_
    // within the last stuck_timeout_ seconds, AND we are commanding motion.
    bool isRobotStuck() const;

    // Tick the recovery FSM; sets v_opt_ / w_opt_ and returns the step taken.
    // Returns true while recovery is still in progress, false once it should
    // hand back to normal operation (new plan received after replan).
    bool tickRecovery();

    // Call move_base's clear_costmaps service and republish the current goal
    // so the global planner generates a fresh plan.
    void requestReplan();

    // =========================================================================
    // Robot constants (fixed)
    // =========================================================================
    static constexpr double WHEELBASE = 0.37558;  // [m]

    // =========================================================================
    // MPC dimensions (must match code-gen)
    // =========================================================================
    int N_  = 20;
    int nx_ = 5;
    int nu_ = 2;
    
    // =========================================================================
    // Path progress tracking
    // =========================================================================
    int path_progress_idx_ = 0;

    // =========================================================================
    // Optuna-tunable parameters — all loaded via nh_private_ / ROS param server
    // =========================================================================

    // --- Velocity limits ---
    double v_linear_max_      = 2.0;
    double v_static_obs_max_  = 0.9;
    double omega_max_         = 1.8;
    double omega_static_obs_max_ = 0.8;

    // --- Stage cost weights ---
    double weight_position_error_ = 49.0;
    double weight_heading_error_  = 37.0;
    double weight_acceleration_   = 0.0021;
    double weight_velocity_       = 10.0;

    double accel_weight_mult_static_  = 5.0;
    double accel_weight_mult_dynamic_ = 3.0;

    // --- Mode trigger distances ---
    double static_obs_safe_dist_   = 1.25;
    double dynamic_obs_safe_dist_  = 2.5;

    // --- Obstacle geometry ---
    double robot_radius_       = 0.35;
    double dynamic_obs_radius_ = 0.5;
    double safety_margin_      = 0.1;
    double obs_search_radius_  = 4.0;

    // --- Reversal detection ---
    double reversal_threshold_ = 0.7;
    double reversal_angle_deg_ = 90.0;

    // --- RUSH_GOAL ---
    double rush_goal_dist_      = 4.0;
    double rush_goal_exit_dist_ = 0.5;
    double rush_weight_position_   = 5000.0;
    double rush_weight_heading_    = 6000.0;
    double rush_weight_velocity_   = 50000.0;
    double rush_weight_accel_      = 0.00001;
    double rush_vref_              = 2.0;
    bool   prev_was_rush_goal_  = false;
    bool   rush_goal_latched_   = false;

    // --- ROTATION_SHIM ---
    double lidar_blind_angle_deg_ = 45.0;
    double shim_exit_heading_deg_ = 30.0;
    double shim_omega_            = 1.2;
    bool   shim_active_     = false;
    bool   shim_turn_left_  = true;

    // --- RECOVERY ---
    // Stuck detection
    double stuck_timeout_           = 3.0;   // [s]  time window to check for movement
    double stuck_dist_threshold_    = 0.05;  // [m]  minimum displacement to be "moving"
    double stuck_cmd_vel_threshold_ = 0.05;  // [m/s or rad/s] ignore if barely commanding

    // Escape sequence parameters
    double recovery_reverse_speed_  = 0.2;   // [m/s]   speed while reversing
    double recovery_reverse_dist_   = 0.15;  // [m]     distance to reverse
    double recovery_rotate_speed_   = 0.6;   // [rad/s] speed while rotating
    double recovery_rotate_deg_     = 45.0;  // [deg]   angle to rotate
    double recovery_replan_timeout_ = 5.0;   // [s]     wait for new plan before hard reset
    int    recovery_max_attempts_   = 3;     // number of escape cycles before HARD_RESET

    // =========================================================================
    // Runtime state
    // =========================================================================
    std::vector<double> current_state_;   // [x, y, theta, vr, vl]

    std::vector<double> og_x_ref_, og_y_ref_, theta_ref_;
    std::vector<double> x_ref_, y_ref_;

    std::vector<DynamicObstacle> dynamic_obstacles_;
    std::vector<double> obs_x_, obs_y_;
    std::vector<double> map_x_, map_y_;

    // Cached KD-tree for static obstacles
    pcl::PointCloud<pcl::PointXYZ>::Ptr static_obs_cloud_;
    pcl::KdTreeFLANN<pcl::PointXYZ>    static_obs_kdtree_;
    bool                                static_obs_kdtree_valid_ = false;
    void rebuildStaticKdtree();

    ros::Time last_dynamic_obs_time_;
    double dynamic_obs_timeout_ = 0.5;

    ControlMode mode_        = ControlMode::NORMAL;
    bool        in_reversal_ = false;
    std::string display_text_;

    double v_opt_ = 0.0;
    double w_opt_ = 0.0;

    std::vector<double> reverse_theta_ref_;

    // --- RECOVERY runtime state ---
    RecoveryStep recovery_step_        = RecoveryStep::IDLE;
    int          recovery_attempt_     = 0;      // current attempt count
    ros::Time    recovery_step_start_;           // when the current sub-step began
    bool         recovery_rotate_left_ = true;   // direction chosen at entry
    bool         recovery_new_plan_received_ = false; // set by callbackGlobalPlan

    // Stuck detection sliding window
    ros::Time    stuck_window_start_;            // start of the current check window
    double       stuck_window_start_x_ = 0.0;
    double       stuck_window_start_y_ = 0.0;
    bool         stuck_window_init_    = false;
};

} // namespace mpc_controller

#endif // MPC_NODE_HPP