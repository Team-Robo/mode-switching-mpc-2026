#include "mpc_node.hpp"
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <algorithm>
#include <limits>

// mpc_node.cpp

namespace mpc_controller {

MPCNode::MPCNode(ros::NodeHandle& nh, ros::NodeHandle& nh_private)
    : nh_(nh), nh_private_(nh_private) {
    
    // Load parameters from parameter server
    nh_private_.param<double>("v_linear_max", v_linear_max_, 2.0);
    nh_private_.param<double>("reversal_threshold", reversal_threshold_, 0.5);
    nh_private_.param<double>("reversal_angle_deg", reversal_angle_deg_, 90.0);
    nh_private_.param<double>("obs_search_radius", obs_search_radius_, 4.6);
    nh_private_.param<double>("SAFE_DISTANCE", SAFE_DISTANCE, 1.25);
    
    nh_private_.param<double>("weight_position_error", weight_position_error_, 49.0);
    nh_private_.param<double>("weight_heading_error", weight_heading_error_, 37.0);
    nh_private_.param<double>("weight_acceleration", weight_acceleration_, 0.0021);
    
    ROS_INFO("MPC Parameters: N=%d, v_linear_max=%.2f m/s, SAFE_DISTANCE=%.2f m",
             N_, v_linear_max_, SAFE_DISTANCE);
    ROS_INFO("MPC Weights: pos=%.2f, heading=%.2f, accel=%.2f",
             weight_position_error_, weight_heading_error_, weight_acceleration_);
    ROS_INFO("Reversal: threshold=%.2f, angle=%.1f deg",
             reversal_threshold_, reversal_angle_deg_);
    
    // Initialize publishers
    pub_vel_ = nh_.advertise<geometry_msgs::Twist>("/cmd_vel", 10, true);
    pub_mpc_plan_ = nh_.advertise<nav_msgs::Path>("/mpc_plan", 1);
    pub_marker_ = nh_.advertise<visualization_msgs::Marker>("/mode", 1);
    
    // Initialize subscribers
    sub_odom_ = nh_.subscribe("/odometry/filtered", 1, &MPCNode::callbackOdom, this);
    sub_global_plan_ = nh_.subscribe("/move_base/TrajectoryPlannerROS/global_plan", 1, 
                                     &MPCNode::callbackGlobalPlan, this);
    sub_cloud_ = nh_.subscribe("/front/odom/cloud", 1, &MPCNode::callbackCloud, this);
    sub_map_cloud_ = nh_.subscribe("/map/cloud", 1, &MPCNode::callbackMapCloud, this);
    sub_dynamic_obstacle_ = nh_.subscribe("/obstacles", 10, &MPCNode::callbackTrackDynamicObstacle, this);

    // Initialize state
    current_state_.resize(nx_, 0.0);
    
    // Initialize ACADOS solver
    initializeAcadosSolver();
    
    ROS_INFO("MPC Node initialized with ACADOS solver");
}

MPCNode::~MPCNode() {
    cleanupAcadosSolver();
}

void MPCNode::initializeAcadosSolver() {
    acados_ocp_capsule_ = jackal_diff_drive_acados_create_capsule();
    
    int status = jackal_diff_drive_acados_create(acados_ocp_capsule_);
    if (status != 0) {
        ROS_ERROR("Failed to create ACADOS solver");
        return;
    }
    
    nlp_config_ = jackal_diff_drive_acados_get_nlp_config(acados_ocp_capsule_);
    nlp_dims_ = jackal_diff_drive_acados_get_nlp_dims(acados_ocp_capsule_);
    nlp_in_ = jackal_diff_drive_acados_get_nlp_in(acados_ocp_capsule_);
    nlp_out_ = jackal_diff_drive_acados_get_nlp_out(acados_ocp_capsule_);
    nlp_solver_ = jackal_diff_drive_acados_get_nlp_solver(acados_ocp_capsule_);
    nlp_opts_ = jackal_diff_drive_acados_get_nlp_opts(acados_ocp_capsule_);
    
    ROS_INFO("ACADOS solver created successfully");
}

void MPCNode::cleanupAcadosSolver() {
    if (acados_ocp_capsule_ != nullptr) {
        int status = jackal_diff_drive_acados_free(acados_ocp_capsule_);
        if (status != 0) {
            ROS_WARN("Failed to free ACADOS solver");
        }
        status = jackal_diff_drive_acados_free_capsule(acados_ocp_capsule_);
        if (status != 0) {
            ROS_WARN("Failed to free ACADOS capsule");
        }
    }
}

// =============================================================================
// Callbacks
// =============================================================================
void MPCNode::callbackOdom(const nav_msgs::Odometry::ConstPtr& msg) {
    double yaw = quaternionToYaw(msg->pose.pose.orientation);
    double x = msg->pose.pose.position.x;
    double y = msg->pose.pose.position.y;
    double v = msg->twist.twist.linear.x;
    double w = msg->twist.twist.angular.z;
    
    double vr = v + w * WHEELBASE / 2.0;
    double vl = v - w * WHEELBASE / 2.0;
    
    current_state_[0] = x;
    current_state_[1] = y;
    current_state_[2] = yaw;
    current_state_[3] = vr;
    current_state_[4] = vl;
    
    publishMarker();
}

void MPCNode::callbackGlobalPlan(const nav_msgs::Path::ConstPtr& msg) {
    if (msg->poses.empty()) return;
    
    og_x_ref_.clear();
    og_y_ref_.clear();
    theta_ref_.clear();
    
    int skip = (msg->poses.size() <= 2 * (N_ + 5)) ? 1 : 2;
    
    for (size_t i = 0; i < msg->poses.size(); i += skip) {
        og_x_ref_.push_back(msg->poses[i].pose.position.x);
        og_y_ref_.push_back(msg->poses[i].pose.position.y);
    }
    
    double center_heading = current_state_[2];
    for (size_t i = 0; i < og_x_ref_.size() - 1; ++i) {
        double dx = og_x_ref_[i+1] - og_x_ref_[i];
        double dy = og_y_ref_[i+1] - og_y_ref_[i];
        double theta = std::atan2(dy, dx);
        double theta_proc = headingPreprocess(center_heading, theta);        
        if (i == 0) {
            theta_ref_.push_back(theta_proc);  // duplicate first so indexing aligns
        }
        theta_ref_.push_back(theta_proc);
        center_heading = theta_proc;
    }
}

void MPCNode::callbackCloud(const sensor_msgs::PointCloud2::ConstPtr& msg) {
    obs_x_.clear();
    obs_y_.clear();
    
    sensor_msgs::PointCloud2ConstIterator<float> iter_x(*msg, "x");
    sensor_msgs::PointCloud2ConstIterator<float> iter_y(*msg, "y");
    
    for (; iter_x != iter_x.end(); ++iter_x, ++iter_y) {
        // Skip NaN / inf points to prevent numerical issues in the solver
        if (!std::isfinite(*iter_x) || !std::isfinite(*iter_y)) continue;
        obs_x_.push_back(*iter_x);
        obs_y_.push_back(*iter_y);
    }
}

void MPCNode::callbackMapCloud(const sensor_msgs::PointCloud2::ConstPtr& msg) {
    map_x_.clear();
    map_y_.clear();
    
    sensor_msgs::PointCloud2ConstIterator<float> iter_x(*msg, "x");
    sensor_msgs::PointCloud2ConstIterator<float> iter_y(*msg, "y");
    
    for (; iter_x != iter_x.end(); ++iter_x, ++iter_y) {
        // Skip NaN / inf points to prevent numerical issues in the solver
        if (!std::isfinite(*iter_x) || !std::isfinite(*iter_y)) continue;
        map_x_.push_back(*iter_x);
        map_y_.push_back(*iter_y);
    }
}

void MPCNode::callbackTrackDynamicObstacle(const obstacle_detector::Obstacles::ConstPtr& msg) {
    dynamic_obstacles_.clear();
    
    for (const auto& circle : msg->circles) {
        DynamicObstacle obs;
        obs.x = circle.center.x;
        obs.y = circle.center.y;
        obs.vx = circle.velocity.x;
        obs.vy = circle.velocity.y;
        obs.radius = circle.true_radius;
        
        dynamic_obstacles_.push_back(obs);
    }
    ROS_INFO("Received %lu dynamic obstacles", dynamic_obstacles_.size());
}

// =============================================================================
// Utility
// =============================================================================
double MPCNode::quaternionToYaw(const geometry_msgs::Quaternion& q) {
    double q0 = q.x, q1 = q.y, q2 = q.z, q3 = q.w;
    return std::atan2(2.0 * (q2 * q3 + q0 * q1), 1.0 - 2.0 * (q1 * q1 + q2 * q2));
}

double MPCNode::headingPreprocess(double center, double target) {
    double lo = center - M_PI;
    double hi = center + M_PI;
    while (target < lo) target += 2.0 * M_PI;
    while (target > hi) target -= 2.0 * M_PI;
    return target;
}

double MPCNode::diffAngle(double a1, double a2) {
    double diff = std::max(a1, a2) - std::min(a1, a2);
    if (diff > M_PI) diff = 2.0 * M_PI - diff;
    return diff;
}

bool MPCNode::isLeft(double rx, double ry, double rtheta, double ox, double oy) {
    double dx = ox - rx;
    double dy = oy - ry;
    double local_y = -std::sin(rtheta) * dx + std::cos(rtheta) * dy;
    return local_y > 0;
}

bool MPCNode::checkReversalNeeded(const std::vector<double>& theta_ref,
                                   double current_theta) {
    if (theta_ref.empty()) return false;
    
    int count  = 0;
    int length = std::min(static_cast<int>(theta_ref.size()), N_);
    double angle_threshold_rad = reversal_angle_deg_ * M_PI / 180.0;
    
    for (int i = 1; i < length; ++i) {
        if (diffAngle(current_theta, theta_ref[i]) > angle_threshold_rad) {
            count++;
        }
    }
    
    return (static_cast<double>(count) / length) > reversal_threshold_;
}

std::vector<double> MPCNode::computeReverseThetaRef(const std::vector<double>& x_ref,
                                                     const std::vector<double>& y_ref,
                                                     double current_theta) {
    std::vector<double> reverse_theta;
    if (x_ref.size() < 2) return reverse_theta;
    
    double center_heading = current_theta;
    
    for (size_t i = 0; i < std::min(static_cast<size_t>(N_), x_ref.size() - 1); ++i) {
        double dx = x_ref[i] - x_ref[i + 1]; 
        double dy = y_ref[i] - y_ref[i + 1];
        double theta = std::atan2(dy, dx);
        double theta_preprocessed = headingPreprocess(center_heading, theta);
        reverse_theta.push_back(theta_preprocessed);
        center_heading = theta_preprocessed;
    }
    
    if (!reverse_theta.empty()) {
        reverse_theta.push_back(reverse_theta.back());
    }
    
    return reverse_theta;
}

void MPCNode::findClosestPoint(const std::vector<double>& x_ref,
                               const std::vector<double>& y_ref,
                               double curr_x, double curr_y,
                               int& min_idx) {
    double min_dist = std::numeric_limits<double>::max();
    min_idx = 0;
    
    for (size_t i = 0; i < x_ref.size(); ++i) {
        double dx = x_ref[i] - curr_x;
        double dy = y_ref[i] - curr_y;
        double dist = std::sqrt(dx*dx + dy*dy);
        if (dist < min_dist) {
            min_dist = dist;
            min_idx  = static_cast<int>(i);
        }
    }
}

std::vector<PredictedObstacle> MPCNode::predictObstaclesTrajectory(
    const std::vector<DynamicObstacle>& obstacles, double dt, int N) {
    
    std::vector<PredictedObstacle> predictions;

    for (const auto& obs : obstacles) {
        PredictedObstacle pred;
        pred.x_predicted.resize(N + 1);
        pred.y_predicted.resize(N + 1);
        pred.vx_predicted.resize(N + 1);
        pred.vy_predicted.resize(N + 1);
        pred.radius_predicted.resize(N + 1);
        
        // Constant velocity model
        for (int i = 0; i <= N; ++i) {
            double t = i * dt;
            
            // Constant velocity
            pred.vx_predicted[i] = obs.vx;
            pred.vy_predicted[i] = obs.vy;
            
            // Linear extrapolation
            pred.x_predicted[i] = obs.x + obs.vx * t;
            pred.y_predicted[i] = obs.y + obs.vy * t;
            
            // Constant radius
            pred.radius_predicted[i] = 0.5;  // or obs.radius if variable
        }
        predictions.push_back(pred);
    }
    return predictions;
}

// =============================================================================
// OCP Solver
// =============================================================================
bool MPCNode::solveOCP(const std::vector<double>& x_ref,
                       const std::vector<double>& y_ref,
                       const std::vector<double>& theta_ref,
                       const std::vector<double>& current_state,
                       const std::vector<double>& obs_x,
                       const std::vector<double>& obs_y) {
    
    // =========================================================================
    // 1. SET INITIAL STATE CONSTRAINT
    // =========================================================================
    ocp_nlp_constraints_model_set(nlp_config_, nlp_dims_, nlp_in_, nlp_out_, 0, 
                                  "lbx", (void*)current_state.data());
    ocp_nlp_constraints_model_set(nlp_config_, nlp_dims_, nlp_in_, nlp_out_, 0, 
                                  "ubx", (void*)current_state.data());
    
    // =========================================================================
    // 2. PREDICT DYNAMIC OBSTACLES (before mode detection)
    // =========================================================================
    double Tf = 2.5;
    double dt = Tf / N_;
    predicted_obstacles_ = predictObstaclesTrajectory(dynamic_obstacles_, dt, N_);
    
    // =========================================================================
    // 3. DETERMINE CONTROL MODE & COMPUTE EFFECTIVE WEIGHTS
    // =========================================================================
    bool has_close_obstacles = false;
    double closest_obstacle_dist = std::numeric_limits<double>::max();

    // Check static obstacles
    if (!obs_x.empty()) {
        for (size_t i = 0; i < obs_x.size(); ++i) {
            double dx = obs_x[i] - current_state[0];
            double dy = obs_y[i] - current_state[1];
            double dist = std::sqrt(dx*dx + dy*dy);
            if (dist < closest_obstacle_dist) closest_obstacle_dist = dist;
            if (dist < SAFE_DISTANCE) has_close_obstacles = true;
        }
    }

    // Check dynamic obstacles at current position
    for (const auto& pred_obs : predicted_obstacles_) {
        if (!pred_obs.x_predicted.empty()) {
            double dx = pred_obs.x_predicted[0] - current_state[0];
            double dy = pred_obs.y_predicted[0] - current_state[1];
            double dist = std::sqrt(dx*dx + dy*dy) - pred_obs.radius_predicted[0];
            if (dist < closest_obstacle_dist) closest_obstacle_dist = dist;
            if (dist < SAFE_DISTANCE) has_close_obstacles = true;
        }
    }
    
    // Start with the base acceleration weight
    double effective_accel_weight = weight_acceleration_;
    
    // Increase accel weight for smoother control near obstacles
    if (has_close_obstacles) {
        effective_accel_weight = weight_acceleration_ * 5.0;
        ROS_INFO_THROTTLE(2.0, "Obstacles detected (closest: %.2fm)", closest_obstacle_dist);
    }
    
    // --- Reversal logic ---
    std::vector<double> effective_theta_ref = theta_ref;
    
    if (checkReversalNeeded(theta_ref, current_state[2])) {
        mode_ = ControlMode::REVERSAL;
        reverse_theta_ref_ = computeReverseThetaRef(x_ref, y_ref, current_state[2]);
        effective_theta_ref = reverse_theta_ref_;
        display_text_ = "REVERSAL";
        ROS_INFO_THROTTLE(1.0, "Reversal mode activated");
    } else {
        mode_ = ControlMode::NORMAL;
        display_text_ = "NORMAL";
        ROS_INFO_THROTTLE(2.0, "Mode: NORMAL");
    }

    // =========================================================================
    // 4. UPDATE CONSTRAINT BOUNDS
    // =========================================================================
    double omega_limit  = 1.8;
    double min_dist_from_center = robot_radius_ + dynamic_obs_radius_ + safety_margin_;
    double dist_bound = min_dist_from_center * min_dist_from_center;
    
    double lh[4] = {-v_linear_max_, -omega_limit, dist_bound, dist_bound};
    double uh[4] = { v_linear_max_, omega_limit, 1.0e9, 1.0e9};

    for (int i = 0; i < N_; ++i) {
        ocp_nlp_constraints_model_set(nlp_config_, nlp_dims_, nlp_in_, nlp_out_, i, "lh", lh);
        ocp_nlp_constraints_model_set(nlp_config_, nlp_dims_, nlp_in_, nlp_out_, i, "uh", uh);
    }

    // =========================================================================
    // 5. UPDATE COST WEIGHTS (W Matrix)
    // =========================================================================
    int ny = 3 + nu_;   // 5: [x, y, theta, ar, al]
    std::vector<double> W(ny * ny, 0.0);

    W[0 * ny + 0] = weight_position_error_;   // x
    W[1 * ny + 1] = weight_position_error_;   // y
    W[2 * ny + 2] = weight_heading_error_;    // theta
    W[3 * ny + 3] = effective_accel_weight;   // ar
    W[4 * ny + 4] = effective_accel_weight;   // al

    for (int i = 0; i < N_; ++i) {
        ocp_nlp_cost_model_set(nlp_config_, nlp_dims_, nlp_in_, i, "W", W.data());
    }

    // =========================================================================
    // 6. PER-STAGE: REFERENCE & OBSTACLE PARAMETERS
    // =========================================================================
    double search_radius_sq = obs_search_radius_ * obs_search_radius_;

    for (int i = 0; i <= N_; ++i) {
        int ref_idx = std::min(i, static_cast<int>(x_ref.size()) - 1);
        int theta_ref_idx = std::min(i, static_cast<int>(effective_theta_ref.size()) - 1);
        
        double stage_x = x_ref[ref_idx];
        double stage_y = y_ref[ref_idx];
        double stage_theta = effective_theta_ref[theta_ref_idx];

        // --- Cost reference ---
        if (i < N_) {
            double y_ref_stage[5];
            y_ref_stage[0] = stage_x;
            y_ref_stage[1] = stage_y;
            y_ref_stage[2] = stage_theta;
            y_ref_stage[3] = 0.0;   // ar reference = 0 (minimise acceleration)
            y_ref_stage[4] = 0.0;   // al reference = 0
            ocp_nlp_cost_model_set(nlp_config_, nlp_dims_, nlp_in_, i, "yref", y_ref_stage);
        } else {
            double y_ref_e[3];
            y_ref_e[0] = stage_x;
            y_ref_e[1] = stage_y;
            y_ref_e[2] = stage_theta;
            ocp_nlp_cost_model_set(nlp_config_, nlp_dims_, nlp_in_, N_, "yref", y_ref_e);
        }

        // --- Closest left / right obstacles within search radius ---
        std::vector<double> obs_L = {1000.0, 1000.0}; 
        std::vector<double> obs_R = {1000.0, 1000.0}; 
        double min_dist_L = std::numeric_limits<double>::max();
        double min_dist_R = std::numeric_limits<double>::max();

        // 1. Check STATIC obstacles from obs_x, obs_y
        if (!obs_x.empty()) {
            for (size_t j = 0; j < obs_x.size(); ++j) {
                double dx = obs_x[j] - stage_x;
                double dy = obs_y[j] - stage_y;
                double d2 = dx*dx + dy*dy;
                
                if (d2 > search_radius_sq) continue;
                
                if (isLeft(stage_x, stage_y, stage_theta, obs_x[j], obs_y[j])) {
                    if (d2 < min_dist_L) { 
                        min_dist_L = d2; 
                        obs_L[0] = obs_x[j]; 
                        obs_L[1] = obs_y[j]; 
                    }
                } else {
                    if (d2 < min_dist_R) { 
                        min_dist_R = d2; 
                        obs_R[0] = obs_x[j]; 
                        obs_R[1] = obs_y[j]; 
                    }
                }
            }
        }

        // 2. Check DYNAMIC obstacles at their predicted positions
        for (const auto& pred_obs : predicted_obstacles_) {
            if (i >= static_cast<int>(pred_obs.x_predicted.size())) continue;
            
            double obs_x_pred = pred_obs.x_predicted[i];
            double obs_y_pred = pred_obs.y_predicted[i];
            double obs_radius = pred_obs.radius_predicted[i];
            
            // Distance to obstacle center (ACADOS will compute this distance)
            double dx = obs_x_pred - stage_x;
            double dy = obs_y_pred - stage_y;
            double dist_to_center_sq = dx*dx + dy*dy;
            double dist_to_center = std::sqrt(dist_to_center_sq);
            
            if (dist_to_center > obs_search_radius_) continue;
            
            // For closest obstacle selection, consider distance to surface
            // but pass CENTER position to ACADOS (it will enforce constraint on center distance)
            double dist_to_surface = dist_to_center - obs_radius;
            double effective_d2 = dist_to_surface * dist_to_surface;
            
            if (isLeft(stage_x, stage_y, stage_theta, obs_x_pred, obs_y_pred)) {
                if (effective_d2 < min_dist_L) { 
                    min_dist_L = effective_d2; 
                    obs_L[0] = obs_x_pred;  // Pass CENTER position
                    obs_L[1] = obs_y_pred; 
                }
            } else {
                if (effective_d2 < min_dist_R) { 
                    min_dist_R = effective_d2; 
                    obs_R[0] = obs_x_pred;  // Pass CENTER position
                    obs_R[1] = obs_y_pred; 
                }
            }
        }

        // --- Update ACADOS parameters ---
        double p_data[4] = { obs_L[0], obs_L[1], obs_R[0], obs_R[1] };
        
        if (ocp_nlp_dims_get_from_attr(nlp_config_, nlp_dims_, nlp_out_, i, "np") > 0) {
            jackal_diff_drive_acados_update_params(acados_ocp_capsule_, i, p_data, 4);
        }
    }
    
    // =========================================================================
    // 7. SOLVE
    // =========================================================================
    int status = jackal_diff_drive_acados_solve(acados_ocp_capsule_);
    
    if (status != 0) {
        ROS_WARN("ACADOS solver failed with status %d", status);
        return false;
    }
    
    // =========================================================================
    // 8. EXTRACT SOLUTION
    // =========================================================================
    
    double u_opt[2];
    ocp_nlp_out_get(nlp_config_, nlp_dims_, nlp_out_, 0, "u", u_opt);
    
    double x_opt[5];
    ocp_nlp_out_get(nlp_config_, nlp_dims_, nlp_out_, 1, "x", x_opt);
    
    double vr_opt = x_opt[3];
    double vl_opt = x_opt[4];
    v_opt_ = (vr_opt + vl_opt) / 2.0;
    w_opt_ = (vr_opt - vl_opt) / WHEELBASE;
    
    // Publish predicted trajectory for visualisation
    std::vector<double> x_traj, y_traj;
    for (int i = 0; i < N_; ++i) {
        double x_stage[5];
        ocp_nlp_out_get(nlp_config_, nlp_dims_, nlp_out_, i, "x", x_stage);
        x_traj.push_back(x_stage[0]);
        y_traj.push_back(x_stage[1]);
    }
    publishTrajectory(x_traj, y_traj);
    
    return true;
}

// =============================================================================
// Publishers
// =============================================================================
void MPCNode::publishVelocity(double v, double w) {
    geometry_msgs::Twist vel_msg;
    vel_msg.linear.x = v;
    vel_msg.angular.z = w;
    pub_vel_.publish(vel_msg);
}

void MPCNode::publishTrajectory(const std::vector<double>& x_traj,
                                const std::vector<double>& y_traj) {
    nav_msgs::Path path_msg;
    path_msg.header.stamp = ros::Time::now();
    path_msg.header.frame_id = "odom";
    
    for (size_t i = 0; i < x_traj.size(); ++i) {
        geometry_msgs::PoseStamped pose;
        pose.pose.position.x = x_traj[i];
        pose.pose.position.y = y_traj[i];
        pose.pose.orientation.w = 1.0;
        path_msg.poses.push_back(pose);
    }
    
    pub_mpc_plan_.publish(path_msg);
}

void MPCNode::publishMarker() {
    visualization_msgs::Marker marker;
    marker.header.frame_id = "odom";
    marker.header.stamp = ros::Time::now();
    marker.type = visualization_msgs::Marker::SPHERE;
    marker.action = visualization_msgs::Marker::ADD;
    
    marker.pose.position.x = current_state_[0] + 0.5;
    marker.pose.position.y = current_state_[1];
    marker.pose.position.z = 0.0;
    marker.pose.orientation.w = 1.0;
    
    marker.scale.x = 0.2;
    marker.scale.y = 0.2;
    marker.scale.z = 0.2;
    
    marker.color.a = 1.0;
    if (mode_ == ControlMode::REVERSAL) {
        marker.color.r = 1.0; marker.color.g = 0.0; marker.color.b = 0.0;  // Red
    } else {
        marker.color.r = 0.0; marker.color.g = 1.0; marker.color.b = 0.0;  // Green
    }
    
    pub_marker_.publish(marker);
    
    // Text marker
    marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
    marker.text = "V: " + std::to_string(v_opt_).substr(0, 5) + 
                    " W: " + std::to_string(w_opt_).substr(0, 5) + 
                    "\n" + display_text_;
    marker.scale.z = 0.2;
    pub_marker_.publish(marker);
}

// =============================================================================
// Main loop
// =============================================================================
void MPCNode::run() {
    std::lock_guard<std::mutex> lock(solver_mutex_);
    try {
        if (og_x_ref_.empty() || theta_ref_.empty()) return;
        
        // Find closest point on global reference
        int min_idx = 0;
        findClosestPoint(og_x_ref_, og_y_ref_, 
                        current_state_[0], current_state_[1], min_idx);
        
        // Extract reference from closest point
        x_ref_.clear();
        y_ref_.clear();
        for (size_t i = min_idx; i < og_x_ref_.size(); ++i) {
            x_ref_.push_back(og_x_ref_[i]);
            y_ref_.push_back(og_y_ref_[i]);
        }
        
        // Check if we have enough reference points
        if (x_ref_.size() <= static_cast<size_t>(N_)) {
            if (x_ref_.empty()) {
                publishVelocity(0.0, 0.0);
                return;
            }
            
            // Pad with the goal point
            double goal_x = x_ref_.back();
            double goal_y = y_ref_.back();
            while (x_ref_.size() <= static_cast<size_t>(N_)) {
                x_ref_.push_back(goal_x);
                y_ref_.push_back(goal_y);
            }
            ROS_INFO("Extended path to goal: %zu points", x_ref_.size());
        }
        
        // Prepare theta reference from closest point
        std::vector<double> theta_ref_subset;
        for (size_t i = min_idx; i < theta_ref_.size(); ++i) {
            theta_ref_subset.push_back(theta_ref_[i]);
        }

        // Pad theta if needed
        while (theta_ref_subset.size() < x_ref_.size()) {
            theta_ref_subset.push_back(theta_ref_subset.back());
        }
        
        // Combine obstacles
        std::vector<double> all_obs_x;
        std::vector<double> all_obs_y;

        // Add dynamic obstacles
        for (const auto& pred_obs : predicted_obstacles_) {
            if (!pred_obs.x_predicted.empty()) {
                all_obs_x.push_back(pred_obs.x_predicted[0]);
                all_obs_y.push_back(pred_obs.y_predicted[0]);
            }
        }

        // Add static map obstacles
        all_obs_x.insert(all_obs_x.end(), map_x_.begin(), map_x_.end());
        all_obs_y.insert(all_obs_y.end(), map_y_.begin(), map_y_.end());
        
        // Solve MPC
        bool success = solveOCP(x_ref_, y_ref_, theta_ref_subset, 
                               current_state_, all_obs_x, all_obs_y);
        
        if (success) {
            publishVelocity(v_opt_, w_opt_);
            ROS_INFO("V: %.3f, W: %.3f, Mode: %s", v_opt_, w_opt_, display_text_.c_str());
        } else {
            v_opt_ = 0.0;
            w_opt_ = 0.0;
            publishVelocity(0.0, 0.0);
            ROS_WARN("MPC solve failed, stopping");
        }
        
    } catch (const std::exception& e) {
        ROS_ERROR("Exception in MPC run: %s", e.what());
        v_opt_ = 0.0;
        w_opt_ = 0.0;
        publishVelocity(0.0, 0.0);
    }
}

} // namespace mpc_controller

int main(int argc, char** argv) {
    ros::init(argc, argv, "nmpc");
    
    ros::NodeHandle nh;
    ros::NodeHandle nh_private("~");
    
    mpc_controller::MPCNode mpc_node(nh, nh_private);

    double mpc_rate_;
    nh_private.param<double>("mpc_rate", mpc_rate_, 27.5);
    ros::Rate rate(mpc_rate_);
    ros::Duration(1.0).sleep();
    
    ROS_INFO("Non-Linear MPC Node running with ACADOS");
    
    while (ros::ok()) {
        ros::spinOnce();
        mpc_node.run();
        rate.sleep();
    }
    
    return 0;
}