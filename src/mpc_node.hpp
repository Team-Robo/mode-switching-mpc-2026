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
#include <obstacle_detector/Obstacles.h>
#include <vector>
#include <string>
#include <cmath>
#include <memory>
#include <mutex>

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

// Priority: DYNAMIC_OBS > STATIC_OBS > NORMAL
enum class ControlMode {
    NORMAL,       // No obstacles nearby — full speed cap (v_linear_max_)
    STATIC_OBS,   // Static obstacles detected — hard-capped at v_static_obs_max_
    DYNAMIC_OBS   // Dynamic obstacles detected — full speed, higher accel weight
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
    double diffAngle(double a1, double a2);
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
    // Obstacle selection — replaces L/R binary classification
    // =========================================================================
    // Selects the 2 best obstacles to fill the ACADOS parameter slots at each
    // stage. "Best" means: slot1 = closest, slot2 = closest with >30° angular
    // separation from slot1 (to avoid constraint rank-deficiency in corridors).
    // Both static and dynamic (predicted) obstacles are considered together.
    void selectTwoObstacles(
        const std::vector<double>& obs_x,
        const std::vector<double>& obs_y,
        const std::vector<PredictedObstacle>& predicted_obstacles,
        double rx, double ry,
        int stage,
        double search_radius_sq,
        double p_data[4]) const;

    // Emergency stop for 3rd+ dynamic obstacle (invisible to ACADOS 2-slot model)
    bool checkEmergencyStop(
        const std::vector<PredictedObstacle>& predicted_obstacles,
        const std::vector<double>& current_state) const;

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
    // Optuna-tunable parameters — all loaded via nh_private_ / ROS param server
    // =========================================================================

    // --- Velocity limits ---
    double v_linear_max_      = 2.0;   // [m/s] cap for NORMAL & DYNAMIC_OBS
    double v_static_obs_max_  = 1.0;   // [m/s] cap for STATIC_OBS
    double omega_max_         = 1.8;   // [rad/s] shared limit

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

    // =========================================================================
    // Runtime state
    // =========================================================================
    std::vector<double> current_state_;   // [x, y, theta, vr, vl]

    std::vector<double> og_x_ref_, og_y_ref_, theta_ref_;
    std::vector<double> x_ref_, y_ref_;

    std::vector<DynamicObstacle> dynamic_obstacles_;
    std::vector<double> obs_x_, obs_y_;
    std::vector<double> map_x_, map_y_;

    ControlMode mode_        = ControlMode::NORMAL;
    bool        in_reversal_ = false;
    std::string display_text_;

    double v_opt_ = 0.0;
    double w_opt_ = 0.0;

    std::vector<double> reverse_theta_ref_;
};

} // namespace mpc_controller

#endif // MPC_NODE_HPP