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

enum class ControlMode {
    SAFE,
    OBSTACLE
};

class MPCNode {
public:
    MPCNode(ros::NodeHandle& nh, ros::NodeHandle& nh_private);
    ~MPCNode();
    
    void run();

private:
    // ROS communication
    ros::NodeHandle nh_;
    ros::NodeHandle nh_private_;
    
    // Publishers
    ros::Publisher pub_vel_;
    ros::Publisher pub_mpc_plan_;
    ros::Publisher pub_marker_;

    // Mutex for solver
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
    
    // MPC solver
    jackal_diff_drive_solver_capsule* acados_ocp_capsule_ = nullptr;
    ocp_nlp_config* nlp_config_;
    ocp_nlp_dims* nlp_dims_;
    ocp_nlp_in* nlp_in_;
    ocp_nlp_out* nlp_out_;
    ocp_nlp_solver* nlp_solver_;
    void* nlp_opts_;
    
    // Helper functions
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
    
    double quaternionToYaw(const geometry_msgs::Quaternion& q);
    double headingPreprocess(double center, double target);
    double diffAngle(double a1, double a2);
    void findClosestPoint(const std::vector<double>& x_ref,
                         const std::vector<double>& y_ref,
                         double curr_x, double curr_y,
                         int& min_idx);
    bool isLeft(double rx, double ry, double rtheta, double ox, double oy);
    
    // Reversal detection and handling
    bool checkReversalNeeded(const std::vector<double>& theta_ref,
                             double current_theta);
    std::vector<double> computeReverseThetaRef(const std::vector<double>& x_ref,
                                                const std::vector<double>& y_ref,
                                                double current_theta);
    
    std::vector<PredictedObstacle> predicted_obstacles_;
    // Prediction of dynamic obs
    std::vector<PredictedObstacle> predictObstaclesTrajectory(
        const std::vector<DynamicObstacle>& obstacles, double dt, int N
    );
    
    // Robot parameters
    static constexpr double WHEELBASE = 0.37558;  // Distance between wheels [m]
    
    double SAFE_DISTANCE = 1.25;  // Distance threshold to enter OBSTACLE mode [m]
    
    // Velocity limits (absolute physical limits)
    double v_linear_max_ = 2.0;   // Maximum linear velocity [m/s]
    double omega_max_ = 1.8;      // Maximum angular velocity [rad/s] — must match Python codegen
    double a_max_ = 3.0;          // Maximum acceleration [m/s^2]
    
    // MPC parameters
    int N_ = 20;  // Prediction horizon
    int nx_ = 5;  // State dimension (5: x, y, theta, vr, vl)
    int nu_ = 2;  // Control dimension (2: ar, al)
    
    // Current state [x, y, theta, vr, vl]
    std::vector<double> current_state_;
    
    // Reference trajectories
    std::vector<double> og_x_ref_;
    std::vector<double> og_y_ref_;
    std::vector<double> theta_ref_;
    std::vector<double> x_ref_;
    std::vector<double> y_ref_;
    
    // Obstacles
    std::vector<DynamicObstacle> dynamic_obstacles_;
    std::vector<double> obs_x_;
    std::vector<double> obs_y_;
    std::vector<double> map_x_;
    std::vector<double> map_y_;
    
    // Control mode
    ControlMode mode_ = ControlMode::SAFE;
    std::string display_text_;
    
    // Optimal controls
    double v_opt_ = 0.0; // Linear velocity
    double w_opt_ = 0.0; // Angular velocity
    
    // Reversal state
    bool reverse_mode_ = false;
    std::vector<double> reverse_theta_ref_;
    
    double weight_position_error_ = 50.0;
    double weight_heading_error_ = 20.0;
    double weight_acceleration_ = 0.0001;
    // Tuning parameters
    double reversa_alpha = 0.7;  
    double obs_search_radius_ = 4.0;  
    double min_obstacle_distance_ = 0.40;
};

} // namespace mpc_controller

#endif // MPC_NODE_HPP