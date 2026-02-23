#include "mpc_node.hpp"
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <algorithm>
#include <limits>

// mpc_node.cpp

namespace mpc_controller {

MPCNode::MPCNode(ros::NodeHandle& nh, ros::NodeHandle& nh_private)
    : nh_(nh), nh_private_(nh_private)
{
    nh_private_.param<double>("v_linear_max",      v_linear_max_,      2.0);
    nh_private_.param<double>("v_static_obs_max",  v_static_obs_max_,  0.9);
    nh_private_.param<double>("omega_max",          omega_max_,         1.8);

    nh_private_.param<double>("weight_position_error",  weight_position_error_,  130.0);
    nh_private_.param<double>("weight_heading_error",   weight_heading_error_,   49.0);
    nh_private_.param<double>("weight_velocity",        weight_velocity_,        27.0);
    nh_private_.param<double>("weight_acceleration",    weight_acceleration_,    0.0011130334330777412);

    nh_private_.param<double>("accel_weight_mult_static",   accel_weight_mult_static_,  3.0);
    nh_private_.param<double>("accel_weight_mult_dynamic",  accel_weight_mult_dynamic_, 5.5);

    nh_private_.param<double>("static_obs_safe_dist",  static_obs_safe_dist_,  1.1);
    nh_private_.param<double>("dynamic_obs_safe_dist", dynamic_obs_safe_dist_, 2.7);

    nh_private_.param<double>("robot_radius",       robot_radius_,       0.37);
    nh_private_.param<double>("dynamic_obs_radius", dynamic_obs_radius_, 0.5);
    nh_private_.param<double>("safety_margin",      safety_margin_,      0.01);
    nh_private_.param<double>("obs_search_radius",  obs_search_radius_,  4.0);

    nh_private_.param<double>("reversal_threshold", reversal_threshold_, 0.85);
    nh_private_.param<double>("reversal_angle_deg", reversal_angle_deg_, 90.0);

    ROS_INFO("=== MPC Node Parameters ===");
    ROS_INFO("  Velocity: NORMAL/DYN=%.2f m/s  STATIC=%.2f m/s  omega=%.2f rad/s",
             v_linear_max_, v_static_obs_max_, omega_max_);
    ROS_INFO("  Weights: pos=%.2f  heading=%.2f  accel=%.4f  velocity=%.2f",
             weight_position_error_, weight_heading_error_, weight_acceleration_, weight_velocity_);
    ROS_INFO("  Accel mults: static=%.1f  dynamic=%.1f",
             accel_weight_mult_static_, accel_weight_mult_dynamic_);
    ROS_INFO("  Trigger dists: static=%.2f m  dynamic=%.2f m",
             static_obs_safe_dist_, dynamic_obs_safe_dist_);
    ROS_INFO("  Reversal: threshold=%.2f  angle=%.1f deg",
             reversal_threshold_, reversal_angle_deg_);

    pub_vel_      = nh_.advertise<geometry_msgs::Twist>("/cmd_vel", 10, true);
    pub_mpc_plan_ = nh_.advertise<nav_msgs::Path>("/mpc_plan", 1);
    pub_marker_   = nh_.advertise<visualization_msgs::Marker>("/mode", 1);

    sub_odom_       = nh_.subscribe("/odometry/filtered", 1, &MPCNode::callbackOdom, this);
    sub_global_plan_= nh_.subscribe("/move_base/TrajectoryPlannerROS/global_plan", 1,
                                    &MPCNode::callbackGlobalPlan, this);
    sub_cloud_      = nh_.subscribe("/front/odom/cloud", 1, &MPCNode::callbackCloud, this);
    sub_map_cloud_  = nh_.subscribe("/map/cloud", 1, &MPCNode::callbackMapCloud, this);
    sub_dynamic_obstacle_ = nh_.subscribe("/obstacles", 10,
                                          &MPCNode::callbackTrackDynamicObstacle, this);

    current_state_.resize(nx_, 0.0);
    initializeAcadosSolver();

    ROS_INFO("MPC Node initialized with ACADOS solver");
}

MPCNode::~MPCNode() {
    cleanupAcadosSolver();
}

void MPCNode::initializeAcadosSolver() {
    acados_ocp_capsule_ = jackal_diff_drive_acados_create_capsule();
    int status = jackal_diff_drive_acados_create(acados_ocp_capsule_);
    if (status != 0) { ROS_ERROR("Failed to create ACADOS solver"); return; }
    nlp_config_ = jackal_diff_drive_acados_get_nlp_config(acados_ocp_capsule_);
    nlp_dims_   = jackal_diff_drive_acados_get_nlp_dims(acados_ocp_capsule_);
    nlp_in_     = jackal_diff_drive_acados_get_nlp_in(acados_ocp_capsule_);
    nlp_out_    = jackal_diff_drive_acados_get_nlp_out(acados_ocp_capsule_);
    nlp_solver_ = jackal_diff_drive_acados_get_nlp_solver(acados_ocp_capsule_);
    nlp_opts_   = jackal_diff_drive_acados_get_nlp_opts(acados_ocp_capsule_);
    ROS_INFO("ACADOS solver created successfully");
}

void MPCNode::cleanupAcadosSolver() {
    if (acados_ocp_capsule_ != nullptr) {
        if (jackal_diff_drive_acados_free(acados_ocp_capsule_) != 0)
            ROS_WARN("Failed to free ACADOS solver");
        if (jackal_diff_drive_acados_free_capsule(acados_ocp_capsule_) != 0)
            ROS_WARN("Failed to free ACADOS capsule");
    }
}

// =============================================================================
// Callbacks
// =============================================================================
void MPCNode::callbackOdom(const nav_msgs::Odometry::ConstPtr& msg) {
    double yaw = quaternionToYaw(msg->pose.pose.orientation);
    double v   = msg->twist.twist.linear.x;
    double w   = msg->twist.twist.angular.z;
    current_state_[0] = msg->pose.pose.position.x;
    current_state_[1] = msg->pose.pose.position.y;
    current_state_[2] = yaw;
    current_state_[3] = v + w * WHEELBASE / 2.0;  // vr
    current_state_[4] = v - w * WHEELBASE / 2.0;  // vl
    publishMarker();
}

void MPCNode::callbackGlobalPlan(const nav_msgs::Path::ConstPtr& msg) {
    if (msg->poses.empty()) return;
    og_x_ref_.clear(); og_y_ref_.clear(); theta_ref_.clear();
    int skip = (msg->poses.size() <= 2 * (N_ + 5)) ? 1 : 2;
    for (size_t i = 0; i < msg->poses.size(); i += skip) {
        og_x_ref_.push_back(msg->poses[i].pose.position.x);
        og_y_ref_.push_back(msg->poses[i].pose.position.y);
    }
    double center_heading = current_state_[2];
    for (size_t i = 0; i < og_x_ref_.size() - 1; ++i) {
        double dx = og_x_ref_[i+1] - og_x_ref_[i];
        double dy = og_y_ref_[i+1] - og_y_ref_[i];
        double theta_proc = headingPreprocess(center_heading, std::atan2(dy, dx));
        if (i == 0) theta_ref_.push_back(theta_proc);
        theta_ref_.push_back(theta_proc);
        center_heading = theta_proc;
    }
}

void MPCNode::callbackCloud(const sensor_msgs::PointCloud2::ConstPtr& msg) {
    obs_x_.clear(); obs_y_.clear();
    sensor_msgs::PointCloud2ConstIterator<float> ix(*msg, "x"), iy(*msg, "y");
    for (; ix != ix.end(); ++ix, ++iy) {
        if (!std::isfinite(*ix) || !std::isfinite(*iy)) continue;
        obs_x_.push_back(*ix); obs_y_.push_back(*iy);
    }
}

void MPCNode::callbackMapCloud(const sensor_msgs::PointCloud2::ConstPtr& msg) {
    map_x_.clear(); map_y_.clear();
    sensor_msgs::PointCloud2ConstIterator<float> ix(*msg, "x"), iy(*msg, "y");
    for (; ix != ix.end(); ++ix, ++iy) {
        if (!std::isfinite(*ix) || !std::isfinite(*iy)) continue;
        map_x_.push_back(*ix); map_y_.push_back(*iy);
    }
}

void MPCNode::callbackTrackDynamicObstacle(const obstacle_detector::Obstacles::ConstPtr& msg) {
    dynamic_obstacles_.clear();
    for (const auto& circle : msg->circles) {
        DynamicObstacle obs;
        obs.x = circle.center.x; obs.y = circle.center.y;
        obs.vx = circle.velocity.x; obs.vy = circle.velocity.y;
        obs.radius = circle.true_radius;
        dynamic_obstacles_.push_back(obs);
    }
}

// =============================================================================
// Utility
// =============================================================================
double MPCNode::quaternionToYaw(const geometry_msgs::Quaternion& q) {
    return std::atan2(2.0*(q.z*q.w + q.x*q.y), 1.0 - 2.0*(q.y*q.y + q.z*q.z));
}

double MPCNode::headingPreprocess(double center, double target) {
    while (target < center - M_PI) target += 2.0 * M_PI;
    while (target > center + M_PI) target -= 2.0 * M_PI;
    return target;
}

double MPCNode::diffAngle(double a1, double a2) {
    double diff = std::fabs(a1 - a2);
    if (diff > M_PI) diff = 2.0 * M_PI - diff;
    return diff;
}

bool MPCNode::isLeft(double rx, double ry, double rtheta, double ox, double oy) {
    double dx = ox - rx, dy = oy - ry;
    return (-std::sin(rtheta)*dx + std::cos(rtheta)*dy) > 0;
}

bool MPCNode::checkReversalNeeded(const std::vector<double>& theta_ref, double current_theta) {
    if (theta_ref.empty()) return false;
    int count  = 0;
    int length = std::min(static_cast<int>(theta_ref.size()), N_);
    double thresh = reversal_angle_deg_ * M_PI / 180.0;
    for (int i = 1; i < length; ++i)
        if (diffAngle(current_theta, theta_ref[i]) > thresh) count++;
    return (static_cast<double>(count) / length) > reversal_threshold_;
}

std::vector<double> MPCNode::computeReverseThetaRef(const std::vector<double>& x_ref,
                                                     const std::vector<double>& y_ref,
                                                     double current_theta) {
    std::vector<double> rev;
    if (x_ref.size() < 2) return rev;
    double center = current_theta;
    for (size_t i = 0; i < std::min(static_cast<size_t>(N_), x_ref.size()-1); ++i) {
        double dx = x_ref[i] - x_ref[i+1], dy = y_ref[i] - y_ref[i+1];
        double t  = headingPreprocess(center, std::atan2(dy, dx));
        rev.push_back(t); center = t;
    }
    if (!rev.empty()) rev.push_back(rev.back());
    return rev;
}

void MPCNode::findClosestPoint(const std::vector<double>& x_ref,
                               const std::vector<double>& y_ref,
                               double curr_x, double curr_y, int& min_idx) {
    double min_dist = std::numeric_limits<double>::max();
    min_idx = 0;
    for (size_t i = 0; i < x_ref.size(); ++i) {
        double dx = x_ref[i]-curr_x, dy = y_ref[i]-curr_y;
        double d  = dx*dx + dy*dy;
        if (d < min_dist) { min_dist = d; min_idx = static_cast<int>(i); }
    }
}

std::vector<PredictedObstacle> MPCNode::predictObstaclesTrajectory(
    const std::vector<DynamicObstacle>& obstacles, double dt, int N)
{
    std::vector<PredictedObstacle> preds;
    for (const auto& obs : obstacles) {
        PredictedObstacle pred;
        pred.x_predicted.resize(N+1);
        pred.y_predicted.resize(N+1);
        pred.vx_predicted.resize(N+1, obs.vx);
        pred.vy_predicted.resize(N+1, obs.vy);
        pred.radius_predicted.resize(N+1, dynamic_obs_radius_);
        for (int i = 0; i <= N; ++i) {
            pred.x_predicted[i] = obs.x + obs.vx * i * dt;
            pred.y_predicted[i] = obs.y + obs.vy * i * dt;
        }
        preds.push_back(pred);
    }
    return preds;
}

// =============================================================================
// OCP Solver
// =============================================================================
bool MPCNode::solveOCP(const std::vector<double>& x_ref,
                       const std::vector<double>& y_ref,
                       const std::vector<double>& theta_ref,
                       const std::vector<double>& current_state,
                       const std::vector<double>& obs_x,
                       const std::vector<double>& obs_y)
{
    // =========================================================================
    // 1. INITIAL STATE CONSTRAINT
    // =========================================================================
    ocp_nlp_constraints_model_set(nlp_config_, nlp_dims_, nlp_in_, nlp_out_, 0,
                                  "lbx", (void*)current_state.data());
    ocp_nlp_constraints_model_set(nlp_config_, nlp_dims_, nlp_in_, nlp_out_, 0,
                                  "ubx", (void*)current_state.data());

    // =========================================================================
    // 2. PREDICT DYNAMIC OBSTACLES
    // =========================================================================
    constexpr double Tf = 2.0;
    double dt = Tf / N_;
    predicted_obstacles_ = predictObstaclesTrajectory(dynamic_obstacles_, dt, N_);

    // =========================================================================
    // 3. MODE DETECTION
    // =========================================================================
    bool has_dynamic_obs = false;
    double closest_dynamic_dist = std::numeric_limits<double>::max();
    for (const auto& pred : predicted_obstacles_) {
        if (pred.x_predicted.empty()) continue;
        double dx   = pred.x_predicted[0] - current_state[0];
        double dy   = pred.y_predicted[0] - current_state[1];
        double dist = std::sqrt(dx*dx + dy*dy) - pred.radius_predicted[0];
        if (dist < closest_dynamic_dist) closest_dynamic_dist = dist;
        if (dist < dynamic_obs_safe_dist_) has_dynamic_obs = true;
    }

    bool has_static_obs = false;
    double closest_static_dist = std::numeric_limits<double>::max();
    for (size_t i = 0; i < obs_x.size(); ++i) {
        double dx   = obs_x[i] - current_state[0];
        double dy   = obs_y[i] - current_state[1];
        double dist = std::sqrt(dx*dx + dy*dy);
        if (dist < closest_static_dist) closest_static_dist = dist;
        if (dist < static_obs_safe_dist_) has_static_obs = true;
    }

    if (has_dynamic_obs) {
        mode_ = ControlMode::DYNAMIC_OBS;
        display_text_ = "DYNAMIC_OBS";
        ROS_INFO_THROTTLE(1.0, "Mode: DYNAMIC_OBS (closest=%.2fm)", closest_dynamic_dist);
    } else if (has_static_obs) {
        mode_ = ControlMode::STATIC_OBS;
        display_text_ = "STATIC_OBS";
        ROS_INFO_THROTTLE(1.0, "Mode: STATIC_OBS (closest=%.2fm)", closest_static_dist);
    } else {
        mode_ = ControlMode::NORMAL;
        display_text_ = "NORMAL";
        ROS_INFO_THROTTLE(2.0, "Mode: NORMAL");
    }

    // =========================================================================
    // 4. MODE-SPECIFIC CONFIG
    // =========================================================================
    double v_cap = (mode_ == ControlMode::STATIC_OBS) ? v_static_obs_max_ : v_linear_max_;

    double accel_mult = 1.0;
    if (mode_ == ControlMode::STATIC_OBS)  accel_mult = accel_weight_mult_static_;
    if (mode_ == ControlMode::DYNAMIC_OBS) accel_mult = accel_weight_mult_dynamic_;
    double effective_accel_weight = weight_acceleration_ * accel_mult;

    // =========================================================================
    // 5. REVERSAL OVERLAY
    // =========================================================================
    std::vector<double> effective_theta_ref = theta_ref;
    in_reversal_ = checkReversalNeeded(theta_ref, current_state[2]);
    if (in_reversal_) {
        reverse_theta_ref_ = computeReverseThetaRef(x_ref, y_ref, current_state[2]);
        effective_theta_ref = reverse_theta_ref_;
        display_text_ = "REV+" + display_text_;
        ROS_INFO_THROTTLE(1.0, "Reversal overlay active");
    }

    // =========================================================================
    // 6. CONSTRAINT BOUNDS
    // =========================================================================
    double min_dist_sq;
    if (mode_ == ControlMode::DYNAMIC_OBS) {
        min_dist_sq = std::pow(robot_radius_ + dynamic_obs_radius_ + safety_margin_, 2.0);
    } else {
        min_dist_sq = std::pow(robot_radius_ + safety_margin_, 2.0);
    }
    double lh[4] = { -v_cap, -omega_max_, min_dist_sq, min_dist_sq };
    double uh[4] = {  v_cap,  omega_max_, 1.0e9,       1.0e9 };

    for (int i = 0; i < N_; ++i) {
        ocp_nlp_constraints_model_set(nlp_config_, nlp_dims_, nlp_in_, nlp_out_, i, "lh", lh);
        ocp_nlp_constraints_model_set(nlp_config_, nlp_dims_, nlp_in_, nlp_out_, i, "uh", uh);
    }

    // =========================================================================
    // 7. COST WEIGHTS — ny = 6: [x, y, theta, v_linear, ar, al]
    // =========================================================================
    int ny = 4 + nu_;  // 6
    std::vector<double> W(ny * ny, 0.0);
    W[0*ny+0] = weight_position_error_;
    W[1*ny+1] = weight_position_error_;
    W[2*ny+2] = weight_heading_error_;
    W[3*ny+3] = weight_velocity_;
    W[4*ny+4] = effective_accel_weight;
    W[5*ny+5] = effective_accel_weight;

    for (int i = 0; i < N_; ++i)
        ocp_nlp_cost_model_set(nlp_config_, nlp_dims_, nlp_in_, i, "W", W.data());

    // =========================================================================
    // 8. PER-STAGE: REFERENCE & OBSTACLE PARAMETERS
    // =========================================================================
    double search_radius_sq = obs_search_radius_ * obs_search_radius_;

    for (int i = 0; i <= N_; ++i) {
        int ref_idx   = std::min(i, static_cast<int>(x_ref.size())-1);
        int theta_idx = std::min(i, static_cast<int>(effective_theta_ref.size())-1);

        double sx = x_ref[ref_idx];
        double sy = y_ref[ref_idx];
        double st = effective_theta_ref[theta_idx];

        // -----------------------------------------------------------------
        // FIX 3: Use predicted robot state for obstacle geometry, not the
        // reference waypoint. At stage 0 we use current_state directly;
        // for i > 0 we pull from the previous solve's warm-start trajectory.
        // This means left/right classification and distance checks reflect
        // where the robot will actually be, not where the path is.
        // -----------------------------------------------------------------
        double rx, ry, rt;
        if (i == 0) {
            rx = current_state[0];
            ry = current_state[1];
            rt = current_state[2];
        } else {
            double xs[5];
            ocp_nlp_out_get(nlp_config_, nlp_dims_, nlp_out_, i, "x", xs);
            rx = xs[0];
            ry = xs[1];
            rt = xs[2];
        }

        if (i < N_) {
            double v_ref = (mode_ == ControlMode::STATIC_OBS) ? v_static_obs_max_ : v_linear_max_;
            double yref[6] = { sx, sy, st, v_ref, 0.0, 0.0 };
            ocp_nlp_cost_model_set(nlp_config_, nlp_dims_, nlp_in_, i, "yref", yref);
        } else {
            double yref_e[3] = { sx, sy, st };
            ocp_nlp_cost_model_set(nlp_config_, nlp_dims_, nlp_in_, N_, "yref", yref_e);
        }

        // --- Closest left/right obstacles relative to predicted robot pose ---
        double obs_L[2] = {1000.0, 1000.0};
        double obs_R[2] = {1000.0, 1000.0};
        double min_dL = std::numeric_limits<double>::max();
        double min_dR = std::numeric_limits<double>::max();

        for (size_t j = 0; j < obs_x.size(); ++j) {
            double dx = obs_x[j] - rx, dy = obs_y[j] - ry;
            double d2 = dx*dx + dy*dy;
            if (d2 > search_radius_sq) continue;
            if (isLeft(rx, ry, rt, obs_x[j], obs_y[j])) {
                if (d2 < min_dL) { min_dL = d2; obs_L[0] = obs_x[j]; obs_L[1] = obs_y[j]; }
            } else {
                if (d2 < min_dR) { min_dR = d2; obs_R[0] = obs_x[j]; obs_R[1] = obs_y[j]; }
            }
        }

        for (const auto& pred : predicted_obstacles_) {
            if (i >= static_cast<int>(pred.x_predicted.size())) continue;
            double px = pred.x_predicted[i], py = pred.y_predicted[i];
            double pr = pred.radius_predicted[i];
            double dx = px - rx, dy = py - ry;
            double dist_center = std::sqrt(dx*dx + dy*dy);
            if (dist_center > obs_search_radius_) continue;
            double eff_d2 = (dist_center - pr) * (dist_center - pr);
            if (isLeft(rx, ry, rt, px, py)) {
                if (eff_d2 < min_dL) { min_dL = eff_d2; obs_L[0] = px; obs_L[1] = py; }
            } else {
                if (eff_d2 < min_dR) { min_dR = eff_d2; obs_R[0] = px; obs_R[1] = py; }
            }
        }

        double p_data[4] = { obs_L[0], obs_L[1], obs_R[0], obs_R[1] };
        if (ocp_nlp_dims_get_from_attr(nlp_config_, nlp_dims_, nlp_out_, i, "np") > 0)
            jackal_diff_drive_acados_update_params(acados_ocp_capsule_, i, p_data, 4);
    }

    // =========================================================================
    // 9. SOLVE
    // =========================================================================
    int status = jackal_diff_drive_acados_solve(acados_ocp_capsule_);
    if (status != 0) {
        ROS_WARN("ACADOS solver failed with status %d", status);
        return false;
    }

    // =========================================================================
    // 10. EXTRACT SOLUTION
    // =========================================================================
    double x_opt[5];
    ocp_nlp_out_get(nlp_config_, nlp_dims_, nlp_out_, 1, "x", x_opt);
    v_opt_ = (x_opt[3] + x_opt[4]) / 2.0;
    w_opt_ = (x_opt[3] - x_opt[4]) / WHEELBASE;

    std::vector<double> x_traj, y_traj;
    for (int i = 0; i < N_; ++i) {
        double xs[5];
        ocp_nlp_out_get(nlp_config_, nlp_dims_, nlp_out_, i, "x", xs);
        x_traj.push_back(xs[0]); y_traj.push_back(xs[1]);
    }
    publishTrajectory(x_traj, y_traj);
    return true;
}

// =============================================================================
// Publishers
// =============================================================================
void MPCNode::publishVelocity(double v, double w) {
    geometry_msgs::Twist msg;
    msg.linear.x  = v;
    msg.angular.z = w;
    pub_vel_.publish(msg);
}

void MPCNode::publishTrajectory(const std::vector<double>& x_traj,
                                const std::vector<double>& y_traj) {
    nav_msgs::Path path;
    path.header.stamp    = ros::Time::now();
    path.header.frame_id = "odom";
    for (size_t i = 0; i < x_traj.size(); ++i) {
        geometry_msgs::PoseStamped ps;
        ps.pose.position.x = x_traj[i];
        ps.pose.position.y = y_traj[i];
        ps.pose.orientation.w = 1.0;
        path.poses.push_back(ps);
    }
    pub_mpc_plan_.publish(path);
}

void MPCNode::publishMarker() {
    visualization_msgs::Marker m;
    m.header.frame_id = "odom";
    m.header.stamp    = ros::Time::now();
    m.ns = "mpc_mode"; m.id = 0;
    m.type   = visualization_msgs::Marker::SPHERE;
    m.action = visualization_msgs::Marker::ADD;
    m.pose.position.x = current_state_[0] + 0.5;
    m.pose.position.y = current_state_[1];
    m.pose.position.z = 0.0;
    m.pose.orientation.w = 1.0;
    m.scale.x = m.scale.y = m.scale.z = 0.3;
    m.color.a = 1.0;
    switch (mode_) {
        case ControlMode::NORMAL:      m.color.r=0.0; m.color.g=1.0; m.color.b=0.0; break;
        case ControlMode::STATIC_OBS:  m.color.r=1.0; m.color.g=0.5; m.color.b=0.0; break;
        case ControlMode::DYNAMIC_OBS: m.color.r=1.0; m.color.g=0.0; m.color.b=0.0; break;
    }
    pub_marker_.publish(m);
    m.id = 1;
    m.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
    m.pose.position.z = 0.5;
    m.scale.x = m.scale.y = 0.0; m.scale.z = 0.25;
    m.text = "V:" + std::to_string(v_opt_).substr(0,5) +
             " W:" + std::to_string(w_opt_).substr(0,5) +
             "\n" + display_text_;
    pub_marker_.publish(m);
}

// =============================================================================
// Main loop
// =============================================================================
void MPCNode::run() {
    std::lock_guard<std::mutex> lock(solver_mutex_);
    try {
        if (og_x_ref_.empty() || theta_ref_.empty()) return;

        int min_idx = 0;
        findClosestPoint(og_x_ref_, og_y_ref_, current_state_[0], current_state_[1], min_idx);

        x_ref_.clear(); y_ref_.clear();
        for (size_t i = min_idx; i < og_x_ref_.size(); ++i) {
            x_ref_.push_back(og_x_ref_[i]);
            y_ref_.push_back(og_y_ref_[i]);
        }
        if (x_ref_.empty()) { publishVelocity(0.0, 0.0); return; }

        double gx = x_ref_.back(), gy = y_ref_.back();
        while (x_ref_.size() <= static_cast<size_t>(N_)) {
            x_ref_.push_back(gx); y_ref_.push_back(gy);
        }

        std::vector<double> theta_sub;
        for (size_t i = min_idx; i < theta_ref_.size(); ++i)
            theta_sub.push_back(theta_ref_[i]);
        while (theta_sub.size() < x_ref_.size())
            theta_sub.push_back(theta_sub.back());

        std::vector<double> all_obs_x, all_obs_y;
        all_obs_x.insert(all_obs_x.end(), obs_x_.begin(), obs_x_.end());
        all_obs_x.insert(all_obs_x.end(), map_x_.begin(), map_x_.end());
        all_obs_y.insert(all_obs_y.end(), obs_y_.begin(), obs_y_.end());
        all_obs_y.insert(all_obs_y.end(), map_y_.begin(), map_y_.end());

        bool success = solveOCP(x_ref_, y_ref_, theta_sub,
                                current_state_, all_obs_x, all_obs_y);
        if (success) {
            publishVelocity(v_opt_, w_opt_);
            ROS_INFO_THROTTLE(0.5, "[%s] V=%.3f W=%.3f", display_text_.c_str(), v_opt_, w_opt_);
        } else {
            v_opt_ = 0.0; w_opt_ = 0.0;
            publishVelocity(0.0, 0.0);
            ROS_WARN("MPC solve failed — stopping robot");
        }
    } catch (const std::exception& e) {
        ROS_ERROR("Exception in MPC run: %s", e.what());
        publishVelocity(0.0, 0.0);
    }
}

} // namespace mpc_controller

// =============================================================================
// main
// =============================================================================
int main(int argc, char** argv) {
    ros::init(argc, argv, "nmpc");
    ros::NodeHandle nh;
    ros::NodeHandle nh_private("~");
    mpc_controller::MPCNode mpc_node(nh, nh_private);
    double mpc_rate = 30.0;
    nh_private.param<double>("mpc_rate", mpc_rate, 30.0);
    ros::Rate rate(mpc_rate);
    ros::Duration(1.0).sleep();
    ROS_INFO("Non-Linear MPC Node running at %.1f Hz", mpc_rate);
    while (ros::ok()) {
        ros::spinOnce();
        mpc_node.run();
        rate.sleep();
    }
    return 0;
}