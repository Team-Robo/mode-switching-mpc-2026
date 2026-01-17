#include "mpc_node.hpp"
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <algorithm>
#include <limits>

namespace mpc_controller {

MPCNode::MPCNode(ros::NodeHandle& nh, ros::NodeHandle& nh_private)
    : nh_(nh), nh_private_(nh_private),
      acados_ocp_capsule_(nullptr),
      N_(25), nx_(5), nu_(2), rate_(20.0),
      v_max_indiv_(1.0), v_min_indiv_(-1.0),
      v_max_total_(1.0), v_min_total_(-1.0),
      a_max_(1.0), w_max_(0.8), w_min_(-0.8),
      mode_(ControlMode::SAFE),
      v_opt_(0.0), w_opt_(0.0),
      reverse_mode_(false),
      weight_velocity_ref_(0.1), weight_position_error_(5.0),
      weight_acceleration_(1.0), v_ref_(0.8) {
    
    // Initialize publishers
    pub_vel_ = nh_.advertise<geometry_msgs::Twist>("/cmd_vel", 10, true);
    pub_mpc_plan_ = nh_.advertise<nav_msgs::Path>("/mpc_plan", 1);
    pub_marker_ = nh_.advertise<visualization_msgs::Marker>("/mode", 1);
    
    // Initialize subscribers
    sub_odom_ = nh_.subscribe("/odometry/filtered", 1, &MPCNode::callbackOdom, this);
    sub_global_plan_ = nh_.subscribe("/move_base/TrajectoryPlannerROS/global_plan", 1, 
                                     &MPCNode::callbackGlobalPlan, this);
    sub_local_plan_ = nh_.subscribe("/move_base/TrajectoryPlannerROS/local_plan", 1,
                                    &MPCNode::callbackLocalPlan, this);
    sub_cloud_ = nh_.subscribe("/front/odom/cloud", 1, &MPCNode::callbackCloud, this);
    sub_map_cloud_ = nh_.subscribe("/map/cloud", 1, &MPCNode::callbackMapCloud, this);
    
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
    // Create ACADOS solver capsule
    acados_ocp_capsule_ = jackal_diff_drive_acados_create_capsule();
    
    // Create the solver
    int status = jackal_diff_drive_acados_create(acados_ocp_capsule_);
    if (status != 0) {
        ROS_ERROR("Failed to create ACADOS solver");
        return;
    }
    
    // Get pointers to solver components
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

void MPCNode::callbackOdom(const nav_msgs::Odometry::ConstPtr& msg) {
    double yaw = quaternionToYaw(msg->pose.pose.orientation);
    double x = msg->pose.pose.position.x;
    double y = msg->pose.pose.position.y;
    double v = msg->twist.twist.linear.x;
    double w = msg->twist.twist.angular.z;
    
    // Convert to wheel velocities
    double vr = v + w * WHEELBASE / 2.0;
    double vl = v - w * WHEELBASE / 2.0;
    
    // Update current state [x, y, theta, vr, vl]
    current_state_[0] = x;
    current_state_[1] = y;
    current_state_[2] = yaw;
    current_state_[3] = vr;
    current_state_[4] = vl;
    
    // Publish marker
    publishMarker();
}

void MPCNode::callbackGlobalPlan(const nav_msgs::Path::ConstPtr& msg) {
    if (msg->poses.empty()) return;
    
    og_x_ref_.clear();
    og_y_ref_.clear();
    theta_ref_.clear();
    
    // Downsample if too many points
    int skip = (msg->poses.size() <= 2 * (N_ + 5)) ? 1 : 2;
    
    for (size_t i = 0; i < msg->poses.size(); i += skip) {
        og_x_ref_.push_back(msg->poses[i].pose.position.x);
        og_y_ref_.push_back(msg->poses[i].pose.position.y);
    }
    
    // Compute heading angles
    double center_heading = current_state_[2];
    for (size_t i = 0; i < og_x_ref_.size() - 1; ++i) {
        double dx = og_x_ref_[i+1] - og_x_ref_[i];
        double dy = og_y_ref_[i+1] - og_y_ref_[i];
        double theta = std::atan2(dy, dx);
        double theta_proc = headingPreprocess(center_heading, theta);
        theta_ref_.push_back(theta_proc);
        
        if (i == 0) {
            theta_ref_.push_back(theta_proc);
        }
        center_heading = theta_proc;
    }
}

void MPCNode::callbackLocalPlan(const nav_msgs::Path::ConstPtr& msg) {
    // Optional: process local plan if needed
}

void MPCNode::callbackCloud(const sensor_msgs::PointCloud2::ConstPtr& msg) {
    obs_x_.clear();
    obs_y_.clear();
    
    sensor_msgs::PointCloud2ConstIterator<float> iter_x(*msg, "x");
    sensor_msgs::PointCloud2ConstIterator<float> iter_y(*msg, "y");
    
    for (; iter_x != iter_x.end(); ++iter_x, ++iter_y) {
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
        map_x_.push_back(*iter_x);
        map_y_.push_back(*iter_y);
    }
}

double MPCNode::quaternionToYaw(const geometry_msgs::Quaternion& q) {
    double q0 = q.x;
    double q1 = q.y;
    double q2 = q.z;
    double q3 = q.w;
    return std::atan2(2.0 * (q2 * q3 + q0 * q1), 1.0 - 2.0 * (q1 * q1 + q2 * q2));
}

double MPCNode::headingPreprocess(double center, double target) {
    double min = center - M_PI;
    double max = center + M_PI;
    
    while (target < min) {
        target += 2.0 * M_PI;
    }
    while (target > max) {
        target -= 2.0 * M_PI;
    }
    
    return target;
}

double MPCNode::diffAngle(double a1, double a2) {
    double diff = std::max(a1, a2) - std::min(a1, a2);
    if (diff > M_PI) {
        diff = 2.0 * M_PI - diff;
    }
    return diff;
}

bool MPCNode::isLeft(double rx, double ry, double rtheta, double ox, double oy) {
    // Transform obstacle to robot's local frame
    double dx = ox - rx;
    double dy = oy - ry;

    // Rotate by -rtheta. We only care about the new Y coordinate (Lateral distance)
    // local_y = -sin(theta)*dx + cos(theta)*dy
    double local_y = -std::sin(rtheta) * dx + std::cos(rtheta) * dy;

    return local_y > 0; // Positive Y is Left
}

bool MPCNode::checkReversalNeeded(const std::vector<double>& theta_ref,
                                   double current_theta) {
    // Check if more than 50% of trajectory points require reversal
    // (angle difference > pi/2 from current heading)
    if (theta_ref.empty()) return false;
    
    int count = 0;
    int length = std::min(static_cast<int>(theta_ref.size()), N_);
    
    for (int i = 1; i < length; ++i) {
        if (diffAngle(current_theta, theta_ref[i]) > M_PI / 2.0) {
            count++;
        }
    }
    
    return (static_cast<double>(count) / length) > 0.5;
}

std::vector<double> MPCNode::computeReverseThetaRef(const std::vector<double>& x_ref,
                                                     const std::vector<double>& y_ref,
                                                     double current_theta) {
    // Compute reversed theta references by looking at path backwards
    // (from point i to point i-1 instead of i to i+1)
    std::vector<double> reverse_theta;
    
    if (x_ref.size() < 2) {
        return reverse_theta;
    }
    
    double center_heading = current_theta;
    
    for (size_t i = 0; i < std::min(static_cast<size_t>(N_), x_ref.size() - 1); ++i) {
        // Compute angle from current point to previous point (reversed direction)
        double dx = x_ref[i] - x_ref[i + 1];
        double dy = y_ref[i] - y_ref[i + 1];
        double theta = std::atan2(dy, dx);
        
        // Preprocess to keep angle continuous
        double theta_preprocessed = headingPreprocess(center_heading, theta);
        reverse_theta.push_back(theta_preprocessed);
        
        center_heading = theta_preprocessed;
    }
    
    // Add one more element for terminal reference
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
            min_idx = i;
        }
    }
}

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
    // 2. DETERMINE CONTROL MODE (SAFE / CAREFUL / OBSTACLE)
    // =========================================================================
    bool has_close_obstacles = false;
    if (!obs_x.empty()) {
        for (size_t i = 0; i < obs_x.size(); ++i) {
            double dx = obs_x[i] - current_state[0];
            double dy = obs_y[i] - current_state[1];
            double dist = std::sqrt(dx*dx + dy*dy);
            if (dist < SAFE_DISTANCE) {
                has_close_obstacles = true;
                break;
            }
        }
    }
    
    // Define max velocity variable for this iteration
    double current_v_max_total = 0.8; // Default

    // Update logic based on obstacles
    if (obs_x.empty()) {
        mode_ = ControlMode::SAFE;
        current_v_max_total = 0.8; //
        v_ref_ = 0.8;
        weight_acceleration_ = 1.0;
        display_text_ = "SAFE";
    } else if (has_close_obstacles) {
        mode_ = ControlMode::CAREFUL;
        current_v_max_total = 0.4; //
        v_ref_ = 0.3;
        weight_acceleration_ = 0.1;
        display_text_ = "CAREFUL";
    } else {
        mode_ = ControlMode::OBSTACLE;
        current_v_max_total = 0.7; //
        v_ref_ = 0.5;
        weight_acceleration_ = 0.1;
        display_text_ = "OBSTACLE";
    }
    
    // Handle Reversal Mode Logic
    reverse_mode_ = false;
    std::vector<double> effective_theta_ref = theta_ref;
    
    if (mode_ == ControlMode::CAREFUL && checkReversalNeeded(theta_ref, current_state[2])) {
        reverse_mode_ = true;
        reverse_theta_ref_ = computeReverseThetaRef(x_ref, y_ref, current_state[2]);
        effective_theta_ref = reverse_theta_ref_;
        display_text_ = "REVERSING";
        ROS_INFO_THROTTLE(1.0, "Reversal mode activated");
    }

    // =========================================================================
    // 3. UPDATE CONSTRAINT BOUNDS (v_max_total)
    // =========================================================================
    // We must update the "lh" and "uh" arrays to enforce the new v_max_total.
    // The h vector is defined as: [ (vr+vl), omega, dist_L_sq, dist_R_sq ]
    
    double lin_vel_bound = 2.0 * current_v_max_total; // vr + vl = 2*v
    double ang_vel_bound = 0.8;                       // w_max
    double dist_bound = 0.37 * 0.37;                  // min_dist_sq
    
    // Construct the bounds arrays (Size 4)
    double lh[4] = {-lin_vel_bound, -ang_vel_bound, dist_bound, dist_bound};
    double uh[4] = {lin_vel_bound, ang_vel_bound, 1.0e9, 1.0e9};

    // Update bounds for stages 0 to N-1
    for (int i = 0; i < N_; ++i) {
        ocp_nlp_constraints_model_set(nlp_config_, nlp_dims_, nlp_in_, nlp_out_, i, "lh", lh);
        ocp_nlp_constraints_model_set(nlp_config_, nlp_dims_, nlp_in_, nlp_out_, i, "uh", uh);
    }

    // =========================================================================
    // 4. UPDATE COST WEIGHTS (W Matrix)
    // =========================================================================
    // The W matrix is (nx + nu) x (nx + nu). For this robot: 5 + 2 = 7x7.
    int ny = nx_ + nu_;
    std::vector<double> W(ny * ny, 0.0);
    
    // Fill the diagonal.
    // Index mapping: 0:x, 1:y, 2:theta, 3:vr, 4:vl, 5:ar, 6:al
    W[0 * ny + 0] = weight_position_error_;
    W[1 * ny + 1] = weight_position_error_;
    W[2 * ny + 2] = 1.0;                    
    W[3 * ny + 3] = weight_velocity_ref_;   
    W[4 * ny + 4] = weight_velocity_ref_;   
    W[5 * ny + 5] = weight_acceleration_;   // Dynamic!
    W[6 * ny + 6] = weight_acceleration_;   // Dynamic!
    
    for (int i = 0; i < N_; ++i) {
        ocp_nlp_cost_model_set(nlp_config_, nlp_dims_, nlp_in_, i, "W", W.data());
    }

    // =========================================================================
    // 5. PER-STAGE UPDATE: REFERENCE & INDEPENDENT OBSTACLES
    // =========================================================================
    double search_radius_sq = 2.5 * 2.5; 

    for (int i = 0; i <= N_; ++i) {
        // --- A. GET REFERENCE POINT FOR THIS STAGE ---
        int ref_idx = std::min(i, static_cast<int>(x_ref.size()) - 1);
        int theta_ref_idx = std::min(i, static_cast<int>(effective_theta_ref.size()) - 1);
        
        double stage_x = x_ref[ref_idx];
        double stage_y = y_ref[ref_idx];
        double stage_theta = effective_theta_ref[theta_ref_idx];

        // --- B. SET COST REFERENCE (yref) ---
        if (i < N_) {
            double v_des = reverse_mode_ ? -v_ref_ : v_ref_;
            double y_ref_stage[7];
            y_ref_stage[0] = stage_x;
            y_ref_stage[1] = stage_y;
            y_ref_stage[2] = stage_theta;
            y_ref_stage[3] = v_des;
            y_ref_stage[4] = v_des;
            y_ref_stage[5] = 0.0;
            y_ref_stage[6] = 0.0;
            ocp_nlp_cost_model_set(nlp_config_, nlp_dims_, nlp_in_, i, "yref", y_ref_stage);
        } else {
            double y_ref_e[5];
            y_ref_e[0] = stage_x;
            y_ref_e[1] = stage_y;
            y_ref_e[2] = stage_theta;
            y_ref_e[3] = 0.0;
            y_ref_e[4] = 0.0;
            ocp_nlp_cost_model_set(nlp_config_, nlp_dims_, nlp_in_, N_, "yref", y_ref_e);
        }

        // --- C. FIND CLOSEST LEFT & RIGHT OBSTACLES ---
        std::vector<double> obs_L = {1000.0, 1000.0}; 
        std::vector<double> obs_R = {1000.0, 1000.0}; 
        double min_dist_L = std::numeric_limits<double>::max();
        double min_dist_R = std::numeric_limits<double>::max();

        if (!obs_x.empty()) {
            for(size_t j = 0; j < obs_x.size(); ++j) {
                double d2 = std::pow(obs_x[j] - stage_x, 2) + std::pow(obs_y[j] - stage_y, 2);
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

        // --- D. UPDATE ACADOS PARAMETERS ---
        double p_data[4];
        p_data[0] = obs_L[0];
        p_data[1] = obs_L[1];
        p_data[2] = obs_R[0];
        p_data[3] = obs_R[1];
        
        jackal_diff_drive_acados_update_params(acados_ocp_capsule_, i, p_data, 4);
    }
    
    // =========================================================================
    // 6. SOLVE
    // =========================================================================
    int status = jackal_diff_drive_acados_solve(acados_ocp_capsule_);
    
    if (status != 0) {
        ROS_WARN("ACADOS solver failed with status %d", status);
        return false;
    }
    
    // =========================================================================
    // 7. EXTRACT SOLUTION
    // =========================================================================
    double u_opt[2];
    ocp_nlp_out_get(nlp_config_, nlp_dims_, nlp_out_, 0, "u", u_opt);
    
    double x_opt[5];
    ocp_nlp_out_get(nlp_config_, nlp_dims_, nlp_out_, 1, "x", x_opt);
    
    double vr_opt = x_opt[3];
    double vl_opt = x_opt[4];
    v_opt_ = (vr_opt + vl_opt) / 2.0;
    w_opt_ = (vr_opt - vl_opt) / WHEELBASE;
    
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
    if (reverse_mode_) {
        // Blue for reversal mode
        marker.color.r = 0.0;
        marker.color.g = 0.0;
        marker.color.b = 1.0;
    } else if (mode_ == ControlMode::SAFE) {
        // Green for safe mode
        marker.color.r = 0.0;
        marker.color.g = 1.0;
        marker.color.b = 0.0;
    } else if (mode_ == ControlMode::OBSTACLE) {
        // Yellow for obstacle mode
        marker.color.r = 1.0;
        marker.color.g = 1.0;
        marker.color.b = 0.0;
    } else {
        // Red for careful mode
        marker.color.r = 1.0;
        marker.color.g = 0.0;
        marker.color.b = 0.0;
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

void MPCNode::run() {
    try {
        // Find closest point on reference trajectory
        if (og_x_ref_.empty() || theta_ref_.empty()) {
            return;
        }
        
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
            ROS_INFO("Stopped: insufficient reference points (%zu)", x_ref_.size());
            publishVelocity(0.0, 0.0);
            return;
        }
        
        // Prepare theta reference from closest point
        std::vector<double> theta_ref_subset;
        for (size_t i = min_idx; i < theta_ref_.size(); ++i) {
            theta_ref_subset.push_back(theta_ref_[i]);
        }
        
        // Combine obstacles
        std::vector<double> all_obs_x = obs_x_;
        std::vector<double> all_obs_y = obs_y_;
        all_obs_x.insert(all_obs_x.end(), map_x_.begin(), map_x_.end());
        all_obs_y.insert(all_obs_y.end(), map_y_.begin(), map_y_.end());
        
        // Solve MPC
        bool success = solveOCP(x_ref_, y_ref_, theta_ref_subset, 
                               current_state_, all_obs_x, all_obs_y);
        
        if (success) {
            publishVelocity(v_opt_, w_opt_);
            ROS_INFO("V: %.3f, W: %.3f, Mode: %s", v_opt_, w_opt_, display_text_.c_str());
        } else {
            publishVelocity(0.0, 0.0);
            ROS_WARN("MPC solve failed, stopping");
        }
        
    } catch (const std::exception& e) {
        ROS_ERROR("Exception in MPC run: %s", e.what());
    }
}

} // namespace mpc_controller

int main(int argc, char** argv) {
    ros::init(argc, argv, "nmpc");
    
    ros::NodeHandle nh;
    ros::NodeHandle nh_private("~");
    
    mpc_controller::MPCNode mpc_node(nh, nh_private);
    
    ros::Rate rate(20);  // 20 Hz
    ros::Duration(1.0).sleep();  // Initial sleep
    
    ROS_INFO("Non-Linear MPC Node running with ACADOS (N=25, 20Hz)");
    
    while (ros::ok()) {
        ros::spinOnce();
        mpc_node.run();
        rate.sleep();
    }
    
    return 0;
}
