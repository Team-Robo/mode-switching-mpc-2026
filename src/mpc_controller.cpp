#include <teamrobo2026/mpc_controller.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2/utils.h>
#include <algorithm>
#include <limits>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>

// mpc_node.cpp

namespace mpc_controller {

namespace {
inline bool checkAcadosSetterStatus(int status, const char* api, const char* field, int stage) {
    if (status == 0) {
        return true;
    }
    ROS_ERROR("MPC_FAIL_SET api=%s field=%s stage=%d status=%d", api, field, stage, status);
    return false;
}

inline void normalize2D(double& x, double& y) {
    const double n = std::sqrt(x * x + y * y);
    if (n > 1e-9) {
        x /= n;
        y /= n;
    } else {
        x = 1.0;
        y = 0.0;
    }
}
}  // namespace

MpcController::MpcController(ros::NodeHandle& nh, ros::NodeHandle& nh_private)
    : nh_(nh), nh_private_(nh_private)
{
    nh_private_.param<double>("v_linear_max",      v_linear_max_,      2.0);
    nh_private_.param<double>("v_ref_static",      v_ref_static_,      1.0);
    nh_private_.param<double>("omega_max",          omega_max_,         1.8);

    nh_private_.param<double>("weight_position_error",  weight_position_error_,  128.0);
    nh_private_.param<double>("weight_heading_error",   weight_heading_error_,   57.0);
    nh_private_.param<double>("weight_velocity",        weight_velocity_,        33.0);
    nh_private_.param<double>("weight_acceleration",    weight_acceleration_,    0.01803665193219243);

    nh_private_.param<double>("accel_weight_mult_static",   accel_weight_mult_static_,  3.0);
    nh_private_.param<double>("accel_weight_mult_dynamic",  accel_weight_mult_dynamic_, 4.9711669468300554);
    nh_private_.param<double>("position_weight_mult_dynamic", position_weight_mult_dynamic_, 0.7000000000000001);
    nh_private_.param<double>("heading_weight_mult_dynamic",  heading_weight_mult_dynamic_,  0.44999999999999996);
    nh_private_.param<double>("vel_weight_mult_dynamic",      vel_weight_mult_dynamic_,      1.113057650495854);

    nh_private_.param<double>("static_obs_safe_dist",  static_obs_safe_dist_,  1.1);
    nh_private_.param<double>("dynamic_obs_safe_dist", dynamic_obs_safe_dist_, 8.8);

    nh_private_.param<double>("robot_radius",       robot_radius_,       0.37);
    nh_private_.param<double>("dynamic_obs_radius", dynamic_obs_radius_, 0.5);
    nh_private_.param<double>("safety_margin",      safety_margin_,      0.01);
    nh_private_.param<double>("obs_search_radius",  obs_search_radius_,  5.0);

    nh_private_.param<double>("reversal_threshold", reversal_threshold_, 0.9);
    nh_private_.param<double>("reversal_angle_deg", reversal_angle_deg_, 85.0);

    // ROTATION_SHIM params
    nh_private_.param<double>("lidar_blind_angle_deg", lidar_blind_angle_deg_, 39.0);
    nh_private_.param<double>("shim_exit_heading_deg", shim_exit_heading_deg_, 15.0);
    nh_private_.param<double>("shim_omega",             shim_omega_,            0.8);

    // RUSH_GOAL params — direct absolute weights, no multipliers
    nh_private_.param<double>("rush_goal_dist",       rush_goal_dist_,       4.6);
    nh_private_.param<double>("rush_goal_exit_dist",  rush_goal_exit_dist_,  0.8);
    nh_private_.param<double>("rush_weight_position", rush_weight_position_, 0.0);
    nh_private_.param<double>("rush_weight_heading",  rush_weight_heading_,  3030.0);
    nh_private_.param<double>("rush_weight_velocity", rush_weight_velocity_, 3050.0);
    nh_private_.param<double>("rush_weight_accel",    rush_weight_accel_,    0.0);
    nh_private_.param<double>("rush_vref",            rush_vref_,            2.0);

    nh_private_.param<bool>("enable_startup_scan", enable_startup_scan_, true);

    // Dynamic obstacle hysteresis timeout
    nh_private_.param<double>("dynamic_obs_timeout", dynamic_obs_timeout_, 0.35);

    // Dynamic behavior-planning knobs
    nh_private_.param<double>("dyn_plan_influence_dist", dyn_plan_influence_dist_, 3.5);
    nh_private_.param<double>("dyn_plan_margin", dyn_plan_margin_, 0.25);
    nh_private_.param<double>("dyn_plan_max_lateral_shift", dyn_plan_max_lateral_shift_, 1.2);
    nh_private_.param<double>("dyn_plan_smoothing", dyn_plan_smoothing_, 0.35);

    nh_private_.param<double>("min_spacing_global_plan", min_spacing_global_plan_, 0.14);
    nh_private_.param<bool>("retry_profile_enabled", retry_profile_enabled_, true);
    nh_private_.param<int>("retry_profile_max_attempts", retry_profile_max_attempts_, 1);
    nh_private_.param<double>("retry_v_ref_scale", retry_v_ref_scale_, 0.7);
    nh_private_.param<double>("retry_heading_weight_scale", retry_heading_weight_scale_, 0.7);
    nh_private_.param<double>("retry_accel_weight_scale", retry_accel_weight_scale_, 1.5);
    nh_private_.param<std::string>("odom_frame", odom_frame_, std::string("odom"));
    nh_private_.param<bool>("bench_log_enabled", bench_log_enabled_, true);
    nh_private_.param<double>("bench_log_period_s", bench_log_period_s_, 1.0);

    ROS_INFO("=== MPC Node Parameters ===");
    ROS_INFO("  Startup scan: %s (fixed sweep ±45° / 90° @ %.1f rad/s)",
            enable_startup_scan_ ? "ENABLED" : "disabled", STARTUP_SCAN_OMEGA);
    ROS_INFO("  Velocity: cap(NORMAL/DYN/STATIC)=%.2f m/s  STATIC v_ref=%.2f m/s  omega_cap=%.2f rad/s",
             v_linear_max_, v_ref_static_, omega_max_);
    ROS_INFO("  Weights: pos=%.2f  heading=%.2f  accel=%.4f  velocity=%.2f",
             weight_position_error_, weight_heading_error_, weight_acceleration_, weight_velocity_);
    ROS_INFO("  Accel mults: static=%.1f  dynamic=%.1f",
             accel_weight_mult_static_, accel_weight_mult_dynamic_);
    ROS_INFO("  Dynamic weight mults: pos=%.3f  heading=%.3f  vel=%.3f",
             position_weight_mult_dynamic_, heading_weight_mult_dynamic_, vel_weight_mult_dynamic_);
    ROS_INFO("  Trigger dists: static=%.2f m  dynamic=%.2f m",
             static_obs_safe_dist_, dynamic_obs_safe_dist_);
    ROS_INFO("  Reversal: threshold=%.2f  angle=%.1f deg",
             reversal_threshold_, reversal_angle_deg_);
    ROS_INFO("  ROTATION_SHIM: blind=%.1f deg  exit=%.1f deg  omega=%.2f rad/s",
             lidar_blind_angle_deg_, shim_exit_heading_deg_, shim_omega_);
    ROS_INFO("  RUSH_GOAL: dist=%.1f m  pos=%.0f  hdg=%.0f  vel=%.0f  accel=%.5f  vref=%.1f",
             rush_goal_dist_, rush_weight_position_, rush_weight_heading_,
             rush_weight_velocity_, rush_weight_accel_, rush_vref_);
    ROS_INFO("  Dynamic obs hysteresis timeout: %.2f s", dynamic_obs_timeout_);
    ROS_INFO("  Dynamic behavior plan: influence=%.2f m  margin=%.2f m  max_shift=%.2f m  smooth=%.2f",
             dyn_plan_influence_dist_, dyn_plan_margin_, dyn_plan_max_lateral_shift_, dyn_plan_smoothing_);
    ROS_INFO("  Global plan min spacing: %.2f m", min_spacing_global_plan_);
    ROS_INFO("  Retry profile: enabled=%s attempts=%d vref_scale=%.2f heading_scale=%.2f accel_scale=%.2f",
             retry_profile_enabled_ ? "true" : "false",
             retry_profile_max_attempts_, retry_v_ref_scale_,
             retry_heading_weight_scale_, retry_accel_weight_scale_);
    ROS_INFO("  Safety settings: robot_radius=%.2f m  dynamic_obs_radius=%.2f m  safety_margin=%.2f m obs_search_radius=%.2f m",
             robot_radius_, dynamic_obs_radius_, safety_margin_, obs_search_radius_);
    ROS_INFO("  Bench logging: enabled=%s  period=%.2f s",
             bench_log_enabled_ ? "true" : "false", bench_log_period_s_);

    pub_mpc_plan_ = nh_.advertise<nav_msgs::Path>("/mpc_plan", 1);
    pub_marker_   = nh_.advertise<visualization_msgs::Marker>("/mode", 1);

    sub_odom_       = nh_.subscribe("/odometry/filtered", 1, &MpcController::callbackOdom, this);
    sub_cloud_      = nh_.subscribe("/front/odom/cloud", 1, &MpcController::callbackCloud, this);
    sub_map_cloud_  = nh_.subscribe("/map/cloud", 1, &MpcController::callbackMapCloud, this);
    //sub_dynamic_obstacle_ = nh_.subscribe("/obstacles", 10,
    //                                      &MpcController::callbackTrackDynamicObstacle, this);

    current_state_.resize(nx_, 0.0);

    // Initialise hysteresis timestamp to a time far in the past
    last_dynamic_obs_time_ = ros::Time(0);

    initializeAcadosSolver();

    ROS_INFO("MPC Node initialized with ACADOS solver");
}

MpcController::~MpcController() {
    cleanupAcadosSolver();
}

void MpcController::initializeAcadosSolver() {
    acados_ocp_capsule_ = jackal_diff_drive_acados_create_capsule();
    if (acados_ocp_capsule_ == nullptr) {
        ROS_ERROR("ACADOS capsule allocation returned null");
        solver_ready_ = false;
        return;
    }
    int status = jackal_diff_drive_acados_create(acados_ocp_capsule_);
    if (status != 0) {
        ROS_ERROR("Failed to create ACADOS solver (status=%d)", status);
        solver_ready_ = false;
        return;
    }
    solver_ready_ = true;
    nlp_config_ = jackal_diff_drive_acados_get_nlp_config(acados_ocp_capsule_);
    nlp_dims_   = jackal_diff_drive_acados_get_nlp_dims(acados_ocp_capsule_);
    nlp_in_     = jackal_diff_drive_acados_get_nlp_in(acados_ocp_capsule_);
    nlp_out_    = jackal_diff_drive_acados_get_nlp_out(acados_ocp_capsule_);
    nlp_solver_ = jackal_diff_drive_acados_get_nlp_solver(acados_ocp_capsule_);
    nlp_opts_   = jackal_diff_drive_acados_get_nlp_opts(acados_ocp_capsule_);
    ROS_INFO("ACADOS solver created successfully");
}

void MpcController::cleanupAcadosSolver() {
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
void MpcController::callbackOdom(const nav_msgs::Odometry::ConstPtr& msg) {
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

void MpcController::ingestGlobalPlan(const std::vector<double>& xs, const std::vector<double>& ys) {
    if (xs.empty() || xs.size() != ys.size()) return;
    og_x_ref_ = xs;
    og_y_ref_ = ys;
    theta_ref_.clear();
    path_progress_idx_ = 0;

    double center_heading = current_state_[2];
    for (size_t i = 0; i < og_x_ref_.size() - 1; ++i) {
        double dx = og_x_ref_[i + 1] - og_x_ref_[i];
        double dy = og_y_ref_[i + 1] - og_y_ref_[i];
        double theta_proc = headingPreprocess(center_heading, std::atan2(dy, dx));
        if (i == 0) theta_ref_.push_back(theta_proc);
        theta_ref_.push_back(theta_proc);
        center_heading = theta_proc;
    }
}

bool MpcController::setPlan(const std::vector<geometry_msgs::PoseStamped>& plan,
                            tf2_ros::Buffer* tf) {
    if (plan.empty() || tf == nullptr) return false;

    geometry_msgs::PoseStamped goal_out;
    try {
        geometry_msgs::PoseStamped goal_in = plan.back();
        if (goal_in.header.frame_id.empty()) {
            ROS_WARN_THROTTLE(2.0, "setPlan: goal pose has empty frame_id");
            return false;
        }
        if (goal_in.header.frame_id == odom_frame_) {
            goal_out = goal_in;
        } else {
            goal_out = tf->transform(goal_in, odom_frame_, ros::Duration(0.2));
        }
    } catch (const tf2::TransformException& ex) {
        ROS_WARN_THROTTLE(2.0, "setPlan goal TF: %s", ex.what());
        return false;
    }

    std::vector<double> xs, ys;
    const int skip = (plan.size() <= static_cast<size_t>(2 * (N_ + 5))) ? 1 : 2;

    for (size_t i = 0; i < plan.size(); i += static_cast<size_t>(skip)) {
        geometry_msgs::PoseStamped in = plan[i];
        geometry_msgs::PoseStamped out;
        try {
            if (in.header.frame_id.empty()) {
                ROS_WARN_THROTTLE(2.0, "setPlan: pose has empty frame_id");
                return false;
            }
            if (in.header.frame_id == odom_frame_) {
                out = in;
            } else {
                out = tf->transform(in, odom_frame_, ros::Duration(0.2));
            }
        } catch (const tf2::TransformException& ex) {
            ROS_WARN_THROTTLE(2.0, "setPlan TF: %s", ex.what());
            return false;
        }
        xs.push_back(out.pose.position.x);
        ys.push_back(out.pose.position.y);
    }

    goal_x_ = goal_out.pose.position.x;
    goal_y_ = goal_out.pose.position.y;
    goal_yaw_ = quaternionToYaw(goal_out.pose.orientation);
    goal_pose_valid_ = true;

    if (xs.empty()) {
        xs.push_back(goal_x_);
        ys.push_back(goal_y_);
    } else {
        const double dx_goal = xs.back() - goal_x_;
        const double dy_goal = ys.back() - goal_y_;
        if ((dx_goal * dx_goal + dy_goal * dy_goal) > 1e-8) {
            xs.push_back(goal_x_);
            ys.push_back(goal_y_);
        }
    }

    ingestGlobalPlan(xs, ys);
    if (goal_pose_valid_) {
        if (theta_ref_.empty()) {
            theta_ref_.push_back(goal_yaw_);
        } else {
            theta_ref_.back() = headingPreprocess(theta_ref_.back(), goal_yaw_);
        }
    }
    return !og_x_ref_.empty();
}

void MpcController::updateRobotPose(const geometry_msgs::PoseStamped& pose) {
    current_state_[0] = pose.pose.position.x;
    current_state_[1] = pose.pose.position.y;
    current_state_[2] = quaternionToYaw(pose.pose.orientation);
}

bool MpcController::getGoalPose(double& gx, double& gy, double& gyaw) const {
    if (!goal_pose_valid_) {
        return false;
    }
    gx = goal_x_;
    gy = goal_y_;
    gyaw = goal_yaw_;
    return true;
}

bool MpcController::isGoalReached(double xy_tolerance, double yaw_tolerance) const {
    if (og_x_ref_.empty() || current_state_.size() < 3) return false;
    const double gx = goal_pose_valid_ ? goal_x_ : og_x_ref_.back();
    const double gy = goal_pose_valid_ ? goal_y_ : og_y_ref_.back();
    const double dx = gx - current_state_[0];
    const double dy = gy - current_state_[1];
    const double dist = std::sqrt(dx * dx + dy * dy);
    if (dist > xy_tolerance) return false;

    if (!goal_pose_valid_ && theta_ref_.empty()) return true;
    const double goal_yaw = goal_pose_valid_ ? goal_yaw_ : theta_ref_.back();
    const double yaw_err = std::abs(diffAngle(current_state_[2], goal_yaw));
    return yaw_err <= yaw_tolerance;
}

void MpcController::callbackCloud(const sensor_msgs::PointCloud2::ConstPtr& msg) {
    obs_x_.clear(); obs_y_.clear();
    sensor_msgs::PointCloud2ConstIterator<float> ix(*msg, "x"), iy(*msg, "y");
    for (; ix != ix.end(); ++ix, ++iy) {
        if (!std::isfinite(*ix) || !std::isfinite(*iy)) continue;
        obs_x_.push_back(*ix); obs_y_.push_back(*iy);
    }
}

void MpcController::callbackMapCloud(const sensor_msgs::PointCloud2::ConstPtr& msg) {
    map_x_.clear(); map_y_.clear();
    sensor_msgs::PointCloud2ConstIterator<float> ix(*msg, "x"), iy(*msg, "y");
    for (; ix != ix.end(); ++ix, ++iy) {
        if (!std::isfinite(*ix) || !std::isfinite(*iy)) continue;
        map_x_.push_back(*ix); map_y_.push_back(*iy);
    }
}

void MpcController::callbackTrackDynamicObstacle(const obstacle_detector::Obstacles::ConstPtr& msg) {
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
double MpcController::quaternionToYaw(const geometry_msgs::Quaternion& q) {
    return std::atan2(2.0*(q.z*q.w + q.x*q.y), 1.0 - 2.0*(q.y*q.y + q.z*q.z));
}

double MpcController::headingPreprocess(double center, double target) const {
    while (target < center - M_PI) target += 2.0 * M_PI;
    while (target > center + M_PI) target -= 2.0 * M_PI;
    return target;
}

double MpcController::diffAngle(double a1, double a2) const {
    double diff = std::fabs(a1 - a2);
    if (diff > M_PI) diff = 2.0 * M_PI - diff;
    return diff;
}

bool MpcController::isLeft(double rx, double ry, double rtheta, double ox, double oy) const {
    double dx = ox - rx, dy = oy - ry;
    return (-std::sin(rtheta)*dx + std::cos(rtheta)*dy) > 0;
}

bool MpcController::checkReversalNeeded(const std::vector<double>& theta_ref, double current_theta) {
    if (theta_ref.empty()) return false;
    int count  = 0;
    int length = std::min(static_cast<int>(theta_ref.size()), N_);
    double thresh = reversal_angle_deg_ * M_PI / 180.0;
    for (int i = 1; i < length; ++i)
        if (diffAngle(current_theta, theta_ref[i]) > thresh) count++;
    return (static_cast<double>(count) / length) > reversal_threshold_;
}

std::vector<double> MpcController::computeReverseThetaRef(const std::vector<double>& x_ref,
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

// =============================================================================
// ROTATION_SHIM helper
// =============================================================================
bool MpcController::isGoalInBlindSpot(double robot_theta,
                                double goal_x, double goal_y,
                                double robot_x, double robot_y) const
{
    double dx = goal_x - robot_x;
    double dy = goal_y - robot_y;
    double angle_to_goal = std::atan2(dy, dx);

    // Blind zone is centred on the robot's REAR direction.
    double rear_dir = robot_theta + M_PI;
    double diff = diffAngle(angle_to_goal, rear_dir);

    double blind_half = lidar_blind_angle_deg_ * M_PI / 180.0;
    return (diff < blind_half);
}

void MpcController::findClosestPoint(const std::vector<double>& x_ref,
                               const std::vector<double>& y_ref,
                               double curr_x, double curr_y, int& min_idx) {
    double min_dist = std::numeric_limits<double>::max();
    min_idx = path_progress_idx_;
    for (size_t i = path_progress_idx_; i < x_ref.size(); ++i) {
        double dx = x_ref[i]-curr_x, dy = y_ref[i]-curr_y;
        double d  = dx*dx + dy*dy;
        if (d < min_dist) { min_dist = d; min_idx = (int)i; }
    }
    path_progress_idx_ = min_idx;
}

std::vector<PredictedObstacle> MpcController::predictObstaclesTrajectory(
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

void MpcController::applyDynamicBehaviorPlanning(
    std::vector<double>& x_ref,
    std::vector<double>& y_ref,
    const std::vector<PredictedObstacle>& predicted_obstacles,
    const std::vector<double>& current_state,
    double dt) const
{
    (void)dt;
    if (x_ref.size() < 3 || y_ref.size() < 3 || predicted_obstacles.empty()) {
        return;
    }

    const size_t horizon_pts = std::min(static_cast<size_t>(N_ + 1), x_ref.size());
    const double influence = std::max(dyn_plan_influence_dist_, 0.1);
    const double max_shift = std::max(dyn_plan_max_lateral_shift_, 0.0);
    const double smoothing = std::min(0.9, std::max(0.0, dyn_plan_smoothing_));

    std::vector<double> shifted_x = x_ref;
    std::vector<double> shifted_y = y_ref;

    for (size_t i = 0; i < horizon_pts; ++i) {
        double tx, ty;
        if (i + 1 < x_ref.size()) {
            tx = x_ref[i + 1] - x_ref[i];
            ty = y_ref[i + 1] - y_ref[i];
        } else {
            tx = x_ref[i] - x_ref[i - 1];
            ty = y_ref[i] - y_ref[i - 1];
        }
        normalize2D(tx, ty);
        const double nx = -ty;
        const double ny = tx;

        double lateral_shift = 0.0;
        for (const auto& pred : predicted_obstacles) {
            if (i >= pred.x_predicted.size() || i >= pred.y_predicted.size() || i >= pred.radius_predicted.size()) {
                continue;
            }

            const double ox = pred.x_predicted[i];
            const double oy = pred.y_predicted[i];
            const double dx = ox - x_ref[i];
            const double dy = oy - y_ref[i];
            const double dist = std::sqrt(dx * dx + dy * dy);

            const double hard_clear = robot_radius_ + pred.radius_predicted[i] + safety_margin_ + dyn_plan_margin_;
            if (dist >= influence) {
                continue;
            }

            const double denom = std::max(1e-3, influence - hard_clear);
            const double strength = std::min(1.0, std::max(0.0, (influence - dist) / denom));
            if (strength <= 0.0) {
                continue;
            }

            const double side = dx * nx + dy * ny;
            double sign = 0.0;
            if (std::fabs(side) > 1e-3) {
                sign = (side > 0.0) ? -1.0 : 1.0;
            } else {
                const double robot_side = (current_state[0] - x_ref[i]) * nx +
                                          (current_state[1] - y_ref[i]) * ny;
                sign = (robot_side >= 0.0) ? -1.0 : 1.0;
            }

            lateral_shift += sign * strength * 0.8;
        }

        lateral_shift = std::max(-max_shift, std::min(max_shift, lateral_shift));
        shifted_x[i] = x_ref[i] + nx * lateral_shift;
        shifted_y[i] = y_ref[i] + ny * lateral_shift;
    }

    // Keep the tail fixed at the nominal goal to preserve terminal convergence.
    const size_t tail_keep = std::min(static_cast<size_t>(3), horizon_pts);
    for (size_t i = 0; i < tail_keep; ++i) {
        const size_t idx = horizon_pts - 1 - i;
        shifted_x[idx] = x_ref[idx];
        shifted_y[idx] = y_ref[idx];
    }

    // One-pass smoothing to avoid zig-zag references.
    if (smoothing > 0.0 && horizon_pts > 2) {
        std::vector<double> smooth_x = shifted_x;
        std::vector<double> smooth_y = shifted_y;
        for (size_t i = 1; i + 1 < horizon_pts; ++i) {
            const double lap_x = 0.25 * shifted_x[i - 1] + 0.5 * shifted_x[i] + 0.25 * shifted_x[i + 1];
            const double lap_y = 0.25 * shifted_y[i - 1] + 0.5 * shifted_y[i] + 0.25 * shifted_y[i + 1];
            smooth_x[i] = (1.0 - smoothing) * shifted_x[i] + smoothing * lap_x;
            smooth_y[i] = (1.0 - smoothing) * shifted_y[i] + smoothing * lap_y;
        }
        shifted_x.swap(smooth_x);
        shifted_y.swap(smooth_y);
    }

    x_ref.swap(shifted_x);
    y_ref.swap(shifted_y);
}

std::vector<double> MpcController::buildHeadingRefFromPath(
    const std::vector<double>& x_ref,
    const std::vector<double>& y_ref,
    double current_heading) const
{
    std::vector<double> theta_ref;
    if (x_ref.empty() || y_ref.empty()) {
        return theta_ref;
    }

    theta_ref.reserve(x_ref.size());
    double center = current_heading;
    if (x_ref.size() == 1) {
        theta_ref.push_back(center);
        return theta_ref;
    }

    for (size_t i = 0; i + 1 < x_ref.size(); ++i) {
        const double dx = x_ref[i + 1] - x_ref[i];
        const double dy = y_ref[i + 1] - y_ref[i];
        const double th = headingPreprocess(center, std::atan2(dy, dx));
        theta_ref.push_back(th);
        center = th;
    }
    theta_ref.push_back(theta_ref.back());
    return theta_ref;
}

// =============================================================================
// RUSH_GOAL helpers
// =============================================================================
double MpcController::distToGoal(double rx, double ry) const {
    if (og_x_ref_.empty()) return std::numeric_limits<double>::max();
    double dx = og_x_ref_.back() - rx;
    double dy = og_y_ref_.back() - ry;
    return std::sqrt(dx*dx + dy*dy);
}

void MpcController::warmStartFromCurrentState(const std::vector<double>& current_state) {
    constexpr double Tf = 2.0;
    double dt = Tf / N_;

    double vr_full = rush_vref_;
    double vl_full = rush_vref_;
    double v_full  = (vr_full + vl_full) / 2.0;

    double x_k   = current_state[0];
    double y_k   = current_state[1];
    double th_k  = current_state[2];

    double ar = (rush_vref_ - current_state[3]) / dt;
    double al = (rush_vref_ - current_state[4]) / dt;
    ar = std::max(-4.0, std::min(4.0, ar));
    al = std::max(-4.0, std::min(4.0, al));
    double u_full[2] = {ar, al};

    for (int i = 0; i <= N_; ++i) {
        double stage_state[5] = {x_k, y_k, th_k, vr_full, vl_full};
        if (i == 0) {
            ocp_nlp_out_set(nlp_config_, nlp_dims_, nlp_out_, nlp_in_, 0, "x",
                            const_cast<double*>(current_state.data()));
        } else {
            ocp_nlp_out_set(nlp_config_, nlp_dims_, nlp_out_, nlp_in_, i, "x", stage_state);
        }
        x_k  += v_full * std::cos(th_k) * dt;
        y_k  += v_full * std::sin(th_k) * dt;
    }
    for (int i = 0; i < N_; ++i) {
        ocp_nlp_out_set(nlp_config_, nlp_dims_, nlp_out_, nlp_in_, i, "u", u_full);
    }
    ROS_INFO("RUSH_GOAL warm-start: seeded at vr=%.2f vl=%.2f (ar=%.2f al=%.2f)",
             vr_full, vl_full, ar, al);
}

// =============================================================================
// selectObstacles — 1 closest LEFT static + 1 closest RIGHT static + up to 10 dynamic obstacles
//   p_data layout: [x0,y0, x1,y1,   x2,y2, ..., x11,y11]
//                   ^---static---^   ^-------dynamic-------^
// =============================================================================
void MpcController::selectObstacles(
    const std::vector<double>& obs_x,
    const std::vector<double>& obs_y,
    const std::vector<PredictedObstacle>& predicted_obstacles,
    double rx, double ry, double rtheta,
    int stage,
    double search_radius_sq,
    double p_data[24]) const
{
    constexpr int N_STATIC  = 2;
    constexpr int N_DYNAMIC = 10;

    // --- static obstacles: one closest on LEFT, one closest on RIGHT ---
    double left_x = 1000.0;
    double left_y = 1000.0;
    double right_x = 1000.0;
    double right_y = 1000.0;
    double left_dist_sq = std::numeric_limits<double>::max();
    double right_dist_sq = std::numeric_limits<double>::max();
    for (size_t j = 0; j < obs_x.size(); ++j) {
        double dx = obs_x[j] - rx, dy = obs_y[j] - ry;
        double d2 = dx*dx + dy*dy;
        if (d2 > search_radius_sq) continue;
        const bool on_left = isLeft(rx, ry, rtheta, obs_x[j], obs_y[j]);
        if (on_left) {
            if (d2 < left_dist_sq) {
                left_dist_sq = d2;
                left_x = obs_x[j];
                left_y = obs_y[j];
            }
        } else {
            if (d2 < right_dist_sq) {
                right_dist_sq = d2;
                right_x = obs_x[j];
                right_y = obs_y[j];
            }
        }
    }
    p_data[0] = left_x;
    p_data[1] = left_y;
    p_data[2] = right_x;
    p_data[3] = right_y;

    // --- up to 10 dynamic obstacles (sorted by effective distance) ---
    struct DynCand { double x, y, eff_dist; };
    std::vector<DynCand> dyn_cands;
    dyn_cands.reserve(predicted_obstacles.size());
    for (const auto& pred : predicted_obstacles) {
        if (stage >= static_cast<int>(pred.x_predicted.size())) continue;
        double px = pred.x_predicted[stage];
        double py = pred.y_predicted[stage];
        double pr = pred.radius_predicted[stage];
        double dx = px - rx, dy = py - ry;
        double dist_center = std::sqrt(dx*dx + dy*dy);
        if (dist_center > obs_search_radius_) continue;
        dyn_cands.push_back({px, py, std::max(0.0, dist_center - pr)});
    }
    std::sort(dyn_cands.begin(), dyn_cands.end(),
              [](const DynCand& a, const DynCand& b){ return a.eff_dist < b.eff_dist; });
    for (int k = 0; k < N_DYNAMIC; ++k) {
        int base = (N_STATIC + k) * 2;
        if (k < (int)dyn_cands.size()) {
            p_data[base+0] = dyn_cands[k].x;
            p_data[base+1] = dyn_cands[k].y;
        } else {
            p_data[base+0] = 1000.0;
            p_data[base+1] = 1000.0;
        }
    }
}

// =============================================================================
// checkEmergencyStop
// =============================================================================
bool MpcController::checkEmergencyStop(
    const std::vector<PredictedObstacle>& predicted_obstacles,
    const std::vector<double>& current_state) const
{
    const double hard_stop_dist = robot_radius_ + dynamic_obs_radius_ + 0.3;

    int slot = 0;
    for (const auto& pred : predicted_obstacles) {
        if (pred.x_predicted.empty()) continue;
        double dx   = pred.x_predicted[0] - current_state[0];
        double dy   = pred.y_predicted[0] - current_state[1];
        double dist = std::sqrt(dx*dx + dy*dy) - pred.radius_predicted[0];
        if (dist < hard_stop_dist) {
            ++slot;
            if (slot > 10) {
                ROS_WARN_THROTTLE(0.5,
                    "Emergency stop: 11th+ dynamic obstacle at %.2fm (threshold %.2fm)",
                    dist, hard_stop_dist);
                return true;
            }
        }
    }
    return false;
}

// =============================================================================
// OCP Solver
// =============================================================================
bool MpcController::solveOCP(const std::vector<double>& x_ref,
                       const std::vector<double>& y_ref,
                       const std::vector<double>& theta_ref,
                       const std::vector<double>& current_state,
                       const std::vector<double>& obs_x,
                       const std::vector<double>& obs_y
                    )
{
    const auto t_start = ros::WallTime::now();
    ros::WallTime t_after_init = t_start;
    ros::WallTime t_after_predict = t_start;
    ros::WallTime t_after_mode = t_start;
    ros::WallTime t_after_warmstart = t_start;
    ros::WallTime t_after_constraints = t_start;
    ros::WallTime t_after_cost = t_start;
    ros::WallTime t_after_params = t_start;
    ros::WallTime t_after_solve = t_start;
    constexpr double kWheelSpeedLimit = 2.0;
    auto check_set = [&](int status, const char* api, const char* field, int stage) -> bool {
        return checkAcadosSetterStatus(status, api, field, stage);
    };

    // =========================================================================
    // 1. INITIAL STATE CONSTRAINT
    // =========================================================================
    if (!check_set(
            ocp_nlp_constraints_model_set(nlp_config_, nlp_dims_, nlp_in_, nlp_out_, 0,
                                          "lbx", (void*)current_state.data()),
            "ocp_nlp_constraints_model_set", "lbx", 0)) {
        return false;
    }
    if (!check_set(
            ocp_nlp_constraints_model_set(nlp_config_, nlp_dims_, nlp_in_, nlp_out_, 0,
                                          "ubx", (void*)current_state.data()),
            "ocp_nlp_constraints_model_set", "ubx", 0)) {
        return false;
    }
    t_after_init = ros::WallTime::now();

    // =========================================================================
    // 2. PREDICT DYNAMIC OBSTACLES
    // =========================================================================
    constexpr double Tf = 2.0;
    double dt = Tf / N_;
    predicted_obstacles_ = predictObstaclesTrajectory(dynamic_obstacles_, dt, N_);
    t_after_predict = ros::WallTime::now();

    // =========================================================================
    // 3. EMERGENCY STOP CHECK
    // =========================================================================
    if (checkEmergencyStop(predicted_obstacles_, current_state)) {
        if (bench_log_enabled_) {
            const double total_ms = (ros::WallTime::now() - t_start).toSec() * 1e3;
            ROS_INFO_STREAM_THROTTLE(bench_log_period_s_,
                "[BENCH][solveOCP] early-exit(emergency_stop) total="
                << total_ms << " ms"
                << " | init_state=" << (t_after_init - t_start).toSec() * 1e3
                << " ms"
                << " | predict_dyn=" << (t_after_predict - t_after_init).toSec() * 1e3
                << " ms");
        }
        return false;
    }

    // =========================================================================
    // 4. MODE DETECTION
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

    // --- Dynamic obstacle hysteresis latch ---
    ros::Time now = ros::Time::now();
    // if (has_dynamic_obs) {
    //     last_dynamic_obs_time_ = now;
    // }
    // const bool dynamic_obs_active =
    //     has_dynamic_obs ||
    //     ((now - last_dynamic_obs_time_).toSec() < dynamic_obs_timeout_);
    const bool dynamic_obs_active = false;

    bool has_static_obs = false;
    double closest_static_dist = std::numeric_limits<double>::max();
    for (size_t i = 0; i < obs_x.size(); ++i) {
        double dx   = obs_x[i] - current_state[0];
        double dy   = obs_y[i] - current_state[1];
        double dist = std::sqrt(dx*dx + dy*dy);
        if (dist < closest_static_dist) closest_static_dist = dist;
        if (dist < static_obs_safe_dist_) has_static_obs = true;
    }

    double goal_dist = distToGoal(current_state[0], current_state[1]);
    bool near_goal   = (goal_dist < rush_goal_dist_);

    // --- RUSH_GOAL latch management ---
    // bool facing_goal = false;
    // if (!og_x_ref_.empty()) {
    //     double dx_goal = og_x_ref_.back() - current_state[0];
    //     double dy_goal = og_y_ref_.back() - current_state[1];
    //     double angle_to_goal = std::atan2(dy_goal, dx_goal);
    //     double heading_err = diffAngle(current_state[2], angle_to_goal);
    //     facing_goal = (heading_err < (45.0 * M_PI / 180.0));
    // }
    // const bool trigger_rush = near_goal && !dynamic_obs_active && !has_static_obs && facing_goal;
    //
    // if (dynamic_obs_active) {
    //     rush_goal_latched_ = false;
    // } else if (has_static_obs) {
    //     rush_goal_latched_ = false;  // let STATIC_OBS take over
    // } else if (trigger_rush) {
    //     rush_goal_latched_ = true;
    // } else if (rush_goal_latched_ && goal_dist < rush_goal_exit_dist_) {
    //     rush_goal_latched_ = false;
    // }
    rush_goal_latched_ = false;

    // --- Primary mode assignment (priority order) ---
    if (dynamic_obs_active) {
        mode_ = ControlMode::DYNAMIC_OBS;
        display_text_ = "DYNAMIC_OBS";
        if (has_dynamic_obs) {
            ROS_INFO_THROTTLE(1.0, "Mode: DYNAMIC_OBS (closest=%.2fm)", closest_dynamic_dist);
        } else {
            ROS_INFO_THROTTLE(1.0, "Mode: DYNAMIC_OBS [hysteresis, last seen %.2fs ago]",
                              (now - last_dynamic_obs_time_).toSec());
        }
    } else if (rush_goal_latched_) {
        mode_ = ControlMode::RUSH_GOAL;
        display_text_ = "RUSH_GOAL";
        ROS_INFO_THROTTLE(0.5, "Mode: RUSH_GOAL (goal=%.2fm, static_obs=%.2fm)",
                          goal_dist, closest_static_dist);
    } else if (has_static_obs) {
        mode_ = ControlMode::STATIC_OBS;
        display_text_ = "STATIC_OBS";
        ROS_INFO_THROTTLE(1.0, "Mode: STATIC_OBS (closest=%.2fm)", closest_static_dist);
    } else {
        mode_ = ControlMode::NORMAL;
        display_text_ = "NORMAL";
        ROS_INFO_THROTTLE(2.0, "Mode: NORMAL");
    }

    // =====================================================================
    // 4b. ROTATION_SHIM — overrides all non-DYNAMIC_OBS modes whenever the
    //     robot heading is too far off the path heading.
    // =====================================================================
    // if (mode_ != ControlMode::DYNAMIC_OBS) {
    //     // Angular error from robot heading to first meaningful path heading.
    //     // theta_ref is the forward path heading — during reversal this will
    //     // naturally be ~180° off, so reversal is covered automatically.
    //     const double path_heading = (theta_ref.size() > 1) ? theta_ref[1] : 
    //                                 (!theta_ref.empty() ? theta_ref[0] : current_state[2]);
    //     double angular_err = path_heading - current_state[2];
    //     while (angular_err >  M_PI) angular_err -= 2.0 * M_PI;
    //     while (angular_err < -M_PI) angular_err += 2.0 * M_PI;

    //     // shim_exit_heading_deg_ reused as the ENGAGE threshold (e.g. 90°).
    //     // Disengage at half to avoid chattering at the boundary.
    //     const double engage_rad    = shim_exit_heading_deg_ * M_PI / 180.0;
    //     const double disengage_rad = engage_rad * 0.5;

    //     if (!shim_active_ && std::fabs(angular_err) > engage_rad) {
    //         shim_active_    = true;
    //         shim_turn_left_ = (angular_err > 0.0);
    //         ROS_INFO("ROTATION_SHIM ON: err=%.1f deg → turning %s",
    //                 angular_err * 180.0 / M_PI, shim_turn_left_ ? "LEFT" : "RIGHT");
    //     }
    //     if (shim_active_ && std::fabs(angular_err) < disengage_rad) {
    //         shim_active_ = false;
    //         ROS_INFO("ROTATION_SHIM OFF: aligned to %.1f deg (< %.1f deg threshold)",
    //                 std::fabs(angular_err) * 180.0 / M_PI,
    //                 disengage_rad * 180.0 / M_PI);
    //     }

    //     if (shim_active_) {
    //         mode_         = ControlMode::ROTATION_SHIM;
    //         display_text_ = "ROT_SHIM";
    //         ROS_INFO_THROTTLE(0.5, "ROTATION_SHIM: spinning %s (err=%.1f deg)",
    //                         shim_turn_left_ ? "LEFT" : "RIGHT",
    //                         angular_err * 180.0 / M_PI);
    //     }
    // } else {
    //     // Dynamic obstacle clears the shim so it re-evaluates once it clears.
    //     shim_active_ = false;
    // }

    // =========================================================================
    // 4c. ROTATION_SHIM EARLY-EXIT — bypass the ACADOS solver completely.
    //
    //     Running the solver with v_cap=0 against a warm solution that has
    //     nonzero wheel velocities produces an infeasible QP (ACADOS_MINSTEP).
    //     Since run() already overrides the solver output with a direct spin
    //     command, there is zero benefit in solving — just set the outputs and
    //     return immediately so the warm solution stays undisturbed for when
    //     normal mode resumes.
    // =========================================================================
    // if (mode_ == ControlMode::ROTATION_SHIM) {
    //     v_opt_ = 0.0;
    //     w_opt_ = shim_turn_left_ ? shim_omega_ : -shim_omega_;
    //     if (bench_log_enabled_) {
    //         const double total_ms = (ros::WallTime::now() - t_start).toSec() * 1e3;
    //         ROS_INFO_STREAM_THROTTLE(bench_log_period_s_,
    //             "[BENCH][solveOCP] early-exit(rotation_shim) total="
    //             << total_ms << " ms"
    //             << " | init_state=" << (t_after_init - t_start).toSec() * 1e3
    //             << " ms"
    //             << " | predict_dyn=" << (t_after_predict - t_after_init).toSec() * 1e3
    //             << " ms"
    //             << " | mode_detection=" << (t_after_mode - t_after_predict).toSec() * 1e3
    //             << " ms");
    //     }
    //     return true;
    // }

    // =========================================================================
    // 5. RUSH_GOAL WARM-START
    // =========================================================================
    if (mode_ == ControlMode::RUSH_GOAL && !prev_was_rush_goal_) {
        ROS_INFO("RUSH_GOAL: warm-starting solver from current state");
        warmStartFromCurrentState(current_state);
    }
    prev_was_rush_goal_ = (mode_ == ControlMode::RUSH_GOAL);
    t_after_warmstart = ros::WallTime::now();

    // =========================================================================
    // 6. MODE-SPECIFIC CONFIG
    // =========================================================================
    double v_cap, omega_cap, effective_accel_weight;
    double eff_pos_weight, eff_heading_weight, eff_velocity_weight;

    switch (mode_) {
        case ControlMode::RUSH_GOAL:
            v_cap                  = v_linear_max_;
            omega_cap              = omega_max_;
            eff_pos_weight         = rush_weight_position_;
            eff_heading_weight     = rush_weight_heading_;
            eff_velocity_weight    = rush_weight_velocity_;
            effective_accel_weight = rush_weight_accel_;
            break;

        case ControlMode::ROTATION_SHIM:
            // Pure in-place rotation — zero forward velocity, constrained omega.
            // Position weight zeroed so the solver doesn't fight the spin with
            // xy-tracking cost.  Heading weight boosted to help the solver converge
            // even though we bypass its output in run().
            v_cap                  = 0.0;
            omega_cap              = shim_omega_;
            effective_accel_weight = weight_acceleration_;
            eff_pos_weight         = 0.0;
            eff_heading_weight     = weight_heading_error_ * 3.0;
            eff_velocity_weight    = weight_velocity_;
            break;

        case ControlMode::STATIC_OBS:
            v_cap                  = v_linear_max_;
            omega_cap              = omega_max_;
            effective_accel_weight = weight_acceleration_ * accel_weight_mult_static_;
            eff_pos_weight         = weight_position_error_;
            eff_heading_weight     = weight_heading_error_;
            eff_velocity_weight    = weight_velocity_;
            break;

        case ControlMode::DYNAMIC_OBS:
            v_cap                  = v_linear_max_;
            omega_cap              = omega_max_;
            effective_accel_weight = weight_acceleration_ * accel_weight_mult_dynamic_;
            eff_pos_weight         = weight_position_error_ * position_weight_mult_dynamic_;
            eff_heading_weight     = weight_heading_error_ * heading_weight_mult_dynamic_;
            eff_velocity_weight    = weight_velocity_ * vel_weight_mult_dynamic_;
            break;

        default: // NORMAL
            v_cap                  = v_linear_max_;
            omega_cap              = omega_max_;
            effective_accel_weight = weight_acceleration_;
            eff_pos_weight         = weight_position_error_;
            eff_heading_weight     = weight_heading_error_;
            eff_velocity_weight    = weight_velocity_;
            break;
    }

    const bool apply_retry_profile =
        retry_profile_active_ &&
        (mode_ == ControlMode::STATIC_OBS || mode_ == ControlMode::NORMAL);
    if (apply_retry_profile) {
        eff_heading_weight *= std::max(0.0, retry_heading_weight_scale_);
        effective_accel_weight *= std::max(0.0, retry_accel_weight_scale_);
    }

    // =========================================================================
    // 7. REVERSAL OVERLAY
    // =========================================================================
    std::vector<double> effective_theta_ref = theta_ref;
    // in_reversal_ = checkReversalNeeded(theta_ref, current_state[2]);
    // if (in_reversal_) {
    //     reverse_theta_ref_ = computeReverseThetaRef(x_ref, y_ref, current_state[2]);
    //     effective_theta_ref = reverse_theta_ref_;
    //     display_text_ = "REV+" + display_text_;
    //     ROS_INFO_THROTTLE(1.0, "Reversal overlay active");
    // }

    // =========================================================================
    // 8. CONSTRAINT BOUNDS
    // =========================================================================
    double min_dist_sq;
    if (mode_ == ControlMode::DYNAMIC_OBS) {
        min_dist_sq = std::pow(robot_radius_ + dynamic_obs_radius_ + safety_margin_, 2.0);
    } else {
        min_dist_sq = std::pow(robot_radius_ + safety_margin_, 2.0);
    }
    double lh[14] = {
        -v_cap,
        -omega_cap,   // angular velocity lower
        min_dist_sq, min_dist_sq, min_dist_sq, min_dist_sq, 
        min_dist_sq, min_dist_sq, min_dist_sq, min_dist_sq,
        min_dist_sq, min_dist_sq, min_dist_sq, min_dist_sq
    };
    double uh[14] = {
        v_cap,
        omega_cap,
        1.0e9, 1.0e9, 1.0e9, 1.0e9,
        1.0e9, 1.0e9, 1.0e9, 1.0e9,
        1.0e9, 1.0e9, 1.0e9, 1.0e9};

    // Build stage-0 bounds using physical limits (never the behavioral cap)
    double lh0[14] = {
        -v_linear_max_,
        -omega_max_,
        min_dist_sq, min_dist_sq, min_dist_sq, min_dist_sq,
        min_dist_sq, min_dist_sq, min_dist_sq, min_dist_sq,
        min_dist_sq, min_dist_sq, min_dist_sq, min_dist_sq
    };
    double uh0[14] = {
        v_linear_max_,
        omega_max_,
        1.0e9, 1.0e9, 1.0e9, 1.0e9,
        1.0e9, 1.0e9, 1.0e9, 1.0e9,
        1.0e9, 1.0e9, 1.0e9, 1.0e9
    };

    for (int i = 0; i < N_; ++i) {
        if (!check_set(
                ocp_nlp_constraints_model_set(nlp_config_, nlp_dims_, nlp_in_, nlp_out_, i, "lh", lh),
                "ocp_nlp_constraints_model_set", "lh", i)) {
            return false;
        }
        if (!check_set(
                ocp_nlp_constraints_model_set(nlp_config_, nlp_dims_, nlp_in_, nlp_out_, i, "uh", uh),
                "ocp_nlp_constraints_model_set", "uh", i)) {
            return false;
        }
    }
    t_after_constraints = ros::WallTime::now();

    // =========================================================================
    // 9. COST WEIGHTS
    // =========================================================================
    int ny = 4 + nu_;  // 6
    std::vector<double> W(ny * ny, 0.0);
    W[0*ny+0] = eff_pos_weight;
    W[1*ny+1] = eff_pos_weight;
    W[2*ny+2] = eff_heading_weight;
    W[3*ny+3] = eff_velocity_weight;
    W[4*ny+4] = effective_accel_weight;
    W[5*ny+5] = effective_accel_weight;

    for (int i = 0; i < N_; ++i) {
        if (!check_set(
                ocp_nlp_cost_model_set(nlp_config_, nlp_dims_, nlp_in_, i, "W", W.data()),
                "ocp_nlp_cost_model_set", "W", i)) {
            return false;
        }
    }

    // Terminal cost W_e: [x, y, theta] — ny_e = 3
    {
        int ny_e = 3;
        std::vector<double> W_e(ny_e * ny_e, 0.0);
        if (mode_ == ControlMode::RUSH_GOAL) {
            W_e[0*ny_e+0] = rush_weight_position_;
            W_e[1*ny_e+1] = rush_weight_position_;
            W_e[2*ny_e+2] = rush_weight_heading_;
        } else {
            W_e[0*ny_e+0] = eff_pos_weight;
            W_e[1*ny_e+1] = eff_pos_weight;
            W_e[2*ny_e+2] = eff_heading_weight;
        }
        if (!check_set(
                ocp_nlp_cost_model_set(nlp_config_, nlp_dims_, nlp_in_, N_, "W", W_e.data()),
                "ocp_nlp_cost_model_set", "W", N_)) {
            return false;
        }
    }
    t_after_cost = ros::WallTime::now();

    // =========================================================================
    // 10. PER-STAGE: REFERENCE & OBSTACLE PARAMETERS
    //     Build KD-tree ONCE outside the loop, then query per stage.
    // =========================================================================
    const bool clear_obstacles = (mode_ == ControlMode::RUSH_GOAL ||
                                  mode_ == ControlMode::ROTATION_SHIM);

    // Build static obstacle point cloud and KD-tree once before the stage loop
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
    const size_t obs_n = std::min(obs_x.size(), obs_y.size());
    for (size_t j = 0; j < obs_n; ++j) {
        if (!std::isfinite(obs_x[j]) || !std::isfinite(obs_y[j])) continue;
        cloud->points.emplace_back(static_cast<float>(obs_x[j]),
                                   static_cast<float>(obs_y[j]),
                                   0.0f);
    }

    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    const bool has_static_cloud = !cloud->points.empty();
    if (has_static_cloud) {
        kdtree.setInputCloud(cloud);
    }

    const double search_radius_sq = obs_search_radius_ * obs_search_radius_;

    for (int i = 0; i <= N_; ++i) {
        const int ref_idx   = std::min(i, static_cast<int>(x_ref.size()) - 1);
        const int theta_idx = std::min(i, static_cast<int>(effective_theta_ref.size()) - 1);

        const double sx = x_ref[ref_idx];
        const double sy = y_ref[ref_idx];
        const double st = effective_theta_ref[theta_idx];

        // ---- cost reference ----
        if (i < N_) {
            double v_ref;
            if (mode_ == ControlMode::RUSH_GOAL)         v_ref = rush_vref_;
            else if (mode_ == ControlMode::STATIC_OBS) {
                v_ref = std::max(0.0, std::min(v_ref_static_, v_linear_max_));
                if (apply_retry_profile) {
                    v_ref *= std::max(0.0, retry_v_ref_scale_);
                }
            }
            else if (mode_ == ControlMode::ROTATION_SHIM) v_ref = 0.0;
            else                                          v_ref = v_linear_max_;
            double yref[6] = { sx, sy, st, v_ref, 0.0, 0.0 };
            if (!check_set(
                    ocp_nlp_cost_model_set(nlp_config_, nlp_dims_, nlp_in_, i, "yref", yref),
                    "ocp_nlp_cost_model_set", "yref", i)) {
                return false;
            }
        } else {
            double yref_e[3] = { sx, sy, st };
            if (!check_set(
                    ocp_nlp_cost_model_set(nlp_config_, nlp_dims_, nlp_in_, N_, "yref", yref_e),
                    "ocp_nlp_cost_model_set", "yref", N_)) {
                return false;
            }
        }

        // ---- obstacle parameters — 24 values: 2 static + 10 dynamic ----
        double p_data[24];
        if (clear_obstacles) {
            for (int _k = 0; _k < 24; ++_k) p_data[_k] = 1000.0;
        } else {
            double pred_x, pred_y;
            if (i == 0) {
                pred_x = current_state[0];
                pred_y = current_state[1];
            } else {
                double xs[5];
                ocp_nlp_out_get(nlp_config_, nlp_dims_, nlp_out_, i, "x", xs);
                pred_x = xs[0];
                pred_y = xs[1];
            }

            std::vector<double> local_obs_x, local_obs_y;
            if (has_static_cloud) {
                pcl::PointXYZ searchPoint(
                    static_cast<float>(pred_x),
                    static_cast<float>(pred_y),
                    0.0f);
                std::vector<int> indices;
                std::vector<float> sqr_dists;
                kdtree.radiusSearch(searchPoint, obs_search_radius_, indices, sqr_dists);
                for (int idx : indices) {
                    local_obs_x.push_back(cloud->points[idx].x);
                    local_obs_y.push_back(cloud->points[idx].y);
                }
            }
            selectObstacles(local_obs_x, local_obs_y, predicted_obstacles_,
                            pred_x, pred_y, st,
                            i, search_radius_sq, p_data);
        }

        if (!check_set(
                jackal_diff_drive_acados_update_params(acados_ocp_capsule_, i, p_data, 24),
                "jackal_diff_drive_acados_update_params", "p", i)) {
            return false;
        }
    }
    t_after_params = ros::WallTime::now();

    // =========================================================================
    // 11. SOLVE
    // =========================================================================
    int status = jackal_diff_drive_acados_solve(acados_ocp_capsule_);
    // if (status != 0) {
    //     ROS_WARN("ACADOS solver failed with status %d", status);
    //     return false;
    // }
    
    if (status != 0) {
        if (bench_log_enabled_) {
            int qp_status = -1;
            int sqp_iter = -1;
            int nlp_iter = -1;
            int stat_n = 0;
            int stat_m = 0;
            int qp_stat_last = -1;
            int qp_iter_last = -1;
            double time_tot = -1.0;
            double kkt_norm_inf = std::numeric_limits<double>::quiet_NaN();
            ocp_nlp_get(nlp_solver_, "qp_status", &qp_status);
            ocp_nlp_get(nlp_solver_, "sqp_iter", &sqp_iter);
            ocp_nlp_get(nlp_solver_, "nlp_iter", &nlp_iter);
            ocp_nlp_get(nlp_solver_, "time_tot", &time_tot);
            ocp_nlp_get(nlp_solver_, "stat_n", &stat_n);
            ocp_nlp_get(nlp_solver_, "stat_m", &stat_m);
            if (stat_n >= 2 && stat_m > 0) {
                const int nrow = std::min(nlp_iter + 1, stat_m);
                if (nrow > 0) {
                    std::vector<double> statistics((stat_n + 1) * nrow, 0.0);
                    ocp_nlp_get(nlp_solver_, "statistics", statistics.data());
                    const int last_i = nrow - 1;
                    qp_stat_last = static_cast<int>(statistics[last_i + 1 * nrow]);
                    qp_iter_last = static_cast<int>(statistics[last_i + 2 * nrow]);
                }
            }
            ocp_nlp_out_get(nlp_config_, nlp_dims_, nlp_out_, 0, "kkt_norm_inf", &kkt_norm_inf);

            const double vr = (current_state.size() > 3) ? current_state[3] : 0.0;
            const double vl = (current_state.size() > 4) ? current_state[4] : 0.0;
            const double v_meas = 0.5 * (vr + vl);
            const double omega_meas = (vr - vl) / WHEELBASE;
            const double v_viol = std::max(0.0, std::fabs(v_meas) - v_cap);
            const double omega_viol = std::max(0.0, std::fabs(omega_meas) - omega_cap);
            const double vr_viol = std::max(0.0, std::fabs(vr) - kWheelSpeedLimit);
            const double vl_viol = std::max(0.0, std::fabs(vl) - kWheelSpeedLimit);
            const double heading_err_deg =
                effective_theta_ref.empty() ? 0.0 :
                diffAngle(current_state[2], effective_theta_ref.front()) * 180.0 / M_PI;

            ROS_WARN(
                "MPC_FAIL_DIAG status=%d mode=%s v=%.3f v_cap=%.3f w=%.3f w_cap=%.3f "
                "vr=%.3f vl=%.3f wheel_lim=%.3f viol(v,w,vr,vl)=(%.4f,%.4f,%.4f,%.4f) "
                "min_dist_sq=%.4f close_static=%.3f close_dynamic=%.3f has_static=%d has_dynamic=%d "
                "heading_err_deg=%.2f refs=%zu obs=%zu dyn_pred=%zu "
                "qp_status=%d sqp_iter=%d nlp_iter=%d qp_stat_last=%d qp_iter_last=%d "
                "kkt_norm_inf=%.3e time_tot_ms=%.3f",
                status, display_text_.c_str(),
                v_meas, v_cap, omega_meas, omega_cap,
                vr, vl, kWheelSpeedLimit,
                v_viol, omega_viol, vr_viol, vl_viol,
                min_dist_sq, closest_static_dist, closest_dynamic_dist,
                has_static_obs ? 1 : 0, has_dynamic_obs ? 1 : 0,
                heading_err_deg, x_ref.size(), obs_x.size(), predicted_obstacles_.size(),
                qp_status, sqp_iter, nlp_iter, qp_stat_last, qp_iter_last,
                kkt_norm_inf, time_tot * 1e3);
        }
        ROS_WARN("ACADOS solver failed with status %d", status);
        return false;
    }
    t_after_solve = ros::WallTime::now();

    // =========================================================================
    // 12. EXTRACT SOLUTION
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
    if (bench_log_enabled_) {
        const auto t_end = ros::WallTime::now();
        ROS_INFO_STREAM_THROTTLE(bench_log_period_s_,
            "[BENCH][solveOCP] total=" << (t_end - t_start).toSec() * 1e3 << " ms"
            << " | init_state="      << (t_after_init - t_start).toSec() * 1e3 << " ms"
            << " | predict_dyn="      << (t_after_predict - t_after_init).toSec() * 1e3 << " ms"
            << " | mode_detection="   << (t_after_mode - t_after_predict).toSec() * 1e3 << " ms"
            << " | warmstart="        << (t_after_warmstart - t_after_mode).toSec() * 1e3 << " ms"
            << " | constraints="      << (t_after_constraints - t_after_warmstart).toSec() * 1e3 << " ms"
            << " | costs="            << (t_after_cost - t_after_constraints).toSec() * 1e3 << " ms"
            << " | stage_params="     << (t_after_params - t_after_cost).toSec() * 1e3 << " ms"
            << " | acados_solve="     << (t_after_solve - t_after_params).toSec() * 1e3 << " ms"
            << " | extract_publish="  << (t_end - t_after_solve).toSec() * 1e3 << " ms");
    }
    return true;
}

// =============================================================================
// Startup Symmetrical Scan
//   Phase sequence:  IDLE → SCAN_LEFT (+45°) → SCAN_RIGHT (−90°) → SCAN_CENTER (+45°) → DONE
//   Fixed speed: STARTUP_SCAN_OMEGA rad/s.  All angles hard-coded.
// =============================================================================
bool MpcController::runStartupScan(geometry_msgs::Twist& cmd_vel)
{
    const double yaw = current_state_[2];
    auto signed_delta = [](double from, double to) -> double {
        double d = to - from;
        while (d >  M_PI) d -= 2.0 * M_PI;
        while (d < -M_PI) d += 2.0 * M_PI;
        return d;
    };

    // start yaw, begin left sweep 
    if (startup_scan_phase_ == StartupScanPhase::IDLE) {
        startup_scan_start_yaw_ = yaw;
        startup_scan_phase_     = StartupScanPhase::SCAN_LEFT;
        mode_         = ControlMode::ROTATION_SHIM;   // borrow shim colour
        display_text_ = "SCAN_L";
        ROS_INFO("[StartupScan] BEGIN — rotating LEFT 45°");
    }

    // rotate CCW until +45° accumulated 
    if (startup_scan_phase_ == StartupScanPhase::SCAN_LEFT) {
        if (signed_delta(startup_scan_start_yaw_, yaw) < STARTUP_SCAN_STEP) {
            writeVelocityCommand(0.0, STARTUP_SCAN_OMEGA, cmd_vel);
        } else {
            writeVelocityCommand(0.0, 0.0, cmd_vel);
            startup_scan_start_yaw_ = yaw;
            startup_scan_phase_     = StartupScanPhase::SCAN_RIGHT;
            mode_         = ControlMode::ROTATION_SHIM;
            display_text_ = "SCAN_R";
            ROS_INFO("[StartupScan] LEFT done — rotating RIGHT 90°");
        }
        return true;
    }

    // rotate CW until −90° accumulated 
    if (startup_scan_phase_ == StartupScanPhase::SCAN_RIGHT) {
        if (signed_delta(startup_scan_start_yaw_, yaw) > -2.0 * STARTUP_SCAN_STEP) {
            writeVelocityCommand(0.0, -STARTUP_SCAN_OMEGA, cmd_vel);
        } else {
            writeVelocityCommand(0.0, 0.0, cmd_vel);
            startup_scan_start_yaw_ = yaw;
            startup_scan_phase_     = StartupScanPhase::SCAN_CENTER;
            mode_         = ControlMode::ROTATION_SHIM;
            display_text_ = "SCAN_C";
            ROS_INFO("[StartupScan] RIGHT done — returning to center (+45°)");
        }
        return true;
    }

    // rotate CCW until +45° accumulated (back to origin) 
    if (startup_scan_phase_ == StartupScanPhase::SCAN_CENTER) {
        if (signed_delta(startup_scan_start_yaw_, yaw) < STARTUP_SCAN_STEP) {
            writeVelocityCommand(0.0, STARTUP_SCAN_OMEGA, cmd_vel);
        } else {
            writeVelocityCommand(0.0, 0.0, cmd_vel);
            startup_scan_phase_ = StartupScanPhase::DONE;
            startup_scan_done_  = true;
            mode_         = ControlMode::NORMAL;
            display_text_ = "NORMAL";
            ROS_INFO("[StartupScan] COMPLETE — resuming normal MPC operation");
        }
        return true;
    }

    // DONE guard (shouldn't normally be reached)
    startup_scan_done_ = true;
    return true;
}

// =============================================================================
// Publishers
// =============================================================================
void MpcController::writeVelocityCommand(double v, double w, geometry_msgs::Twist& cmd_vel) const {
    cmd_vel.linear.x  = v;
    cmd_vel.angular.z = w;
}

void MpcController::publishTrajectory(const std::vector<double>& x_traj,
                                const std::vector<double>& y_traj) {
    nav_msgs::Path path;
    path.header.stamp    = ros::Time::now();
    path.header.frame_id = odom_frame_;
    for (size_t i = 0; i < x_traj.size(); ++i) {
        geometry_msgs::PoseStamped ps;
        ps.pose.position.x = x_traj[i];
        ps.pose.position.y = y_traj[i];
        ps.pose.orientation.w = 1.0;
        path.poses.push_back(ps);
    }
    pub_mpc_plan_.publish(path);
}

void MpcController::publishMarker() {
    visualization_msgs::Marker m;
    m.header.frame_id = odom_frame_;
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
        case ControlMode::NORMAL:
            m.color.r=0.0; m.color.g=1.0; m.color.b=0.0; break;
        case ControlMode::STATIC_OBS:
            m.color.r=1.0; m.color.g=0.5; m.color.b=0.0; break;
        case ControlMode::DYNAMIC_OBS:
            m.color.r=1.0; m.color.g=0.0; m.color.b=0.0; break;
        case ControlMode::RUSH_GOAL:
            m.color.r=0.5; m.color.g=0.0; m.color.b=1.0; break;
        case ControlMode::ROTATION_SHIM:
            m.color.r=0.0; m.color.g=0.8; m.color.b=1.0; break;  // cyan
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
bool MpcController::runOnce(geometry_msgs::Twist& cmd_vel) {
    cmd_vel = geometry_msgs::Twist();
    if (!solver_ready_ || nlp_config_ == nullptr) {
        ROS_ERROR_THROTTLE(5.0, "ACADOS solver not initialized");
        return false;
    }
    if (og_x_ref_.empty() || og_y_ref_.empty()) {
        return false;
    }
    if (enable_startup_scan_ && !startup_scan_done_) {
        return runStartupScan(cmd_vel);
    }
    try {
        const auto t_run_start = ros::WallTime::now();
        ros::WallTime t_after_refs = t_run_start;
        ros::WallTime t_after_theta_obs = t_run_start;
        ros::WallTime t_after_solve = t_run_start;

        if (og_x_ref_.empty() || theta_ref_.empty()) return false;

        int min_idx = 0;
        findClosestPoint(og_x_ref_, og_y_ref_, current_state_[0], current_state_[1], min_idx);

        x_ref_.clear(); y_ref_.clear();
         // this mean the reference points will be at least 20cm apart, which helps the solver converge better by not fighting over closely spaced references
         // if reduced too much, the solver can struggle to find a feasible solution
        // gap between reference points
        const double min_spacing = std::max(0.0, min_spacing_global_plan_);
        const double min_spacing_sq = min_spacing * min_spacing;

        double last_x = og_x_ref_[min_idx];
        double last_y = og_y_ref_[min_idx];
        x_ref_.push_back(last_x);
        y_ref_.push_back(last_y);

        for (size_t i = min_idx + 1; i < og_x_ref_.size() && (int)x_ref_.size() <= N_ + 5; ++i) {
            double dx = og_x_ref_[i] - last_x;
            double dy = og_y_ref_[i] - last_y;
            if ((dx*dx + dy*dy) >= min_spacing_sq) {
                x_ref_.push_back(og_x_ref_[i]);
                y_ref_.push_back(og_y_ref_[i]);
                last_x = og_x_ref_[i];
                last_y = og_y_ref_[i];
            }
        }
        if (x_ref_.empty()) { writeVelocityCommand(0.0, 0.0, cmd_vel); return false; }

        // Ensure the terminal reference contains the exact global-plan goal.
        const double gx_goal = og_x_ref_.back();
        const double gy_goal = og_y_ref_.back();
        const double goal_append_eps_sq = 1e-8;
        const double dx_goal = x_ref_.back() - gx_goal;
        const double dy_goal = y_ref_.back() - gy_goal;
        if ((dx_goal * dx_goal + dy_goal * dy_goal) > goal_append_eps_sq) {
            x_ref_.push_back(gx_goal);
            y_ref_.push_back(gy_goal);
        }

        double gx = x_ref_.back(), gy = y_ref_.back();
        while (x_ref_.size() <= static_cast<size_t>(N_)) {
            x_ref_.push_back(gx); y_ref_.push_back(gy);
        }
        t_after_refs = ros::WallTime::now();

        constexpr double Tf = 2.0;
        const double dt = Tf / N_;
        const auto predicted_for_behavior = predictObstaclesTrajectory(dynamic_obstacles_, dt, N_);
        applyDynamicBehaviorPlanning(x_ref_, y_ref_, predicted_for_behavior, current_state_, dt);

        std::vector<double> theta_sub = buildHeadingRefFromPath(
            x_ref_, y_ref_, current_state_[2]);
        while (!theta_sub.empty() && theta_sub.size() < x_ref_.size())
            theta_sub.push_back(theta_sub.back());
        if (goal_pose_valid_ && !theta_sub.empty()) {
            theta_sub.back() = headingPreprocess(theta_sub.back(), goal_yaw_);
        }
        if (theta_sub.empty()) {
            writeVelocityCommand(0.0, 0.0, cmd_vel);
            return false;
        }

        std::vector<double> all_obs_x, all_obs_y;
        all_obs_x.insert(all_obs_x.end(), obs_x_.begin(), obs_x_.end());
        all_obs_x.insert(all_obs_x.end(), map_x_.begin(), map_x_.end());
        all_obs_y.insert(all_obs_y.end(), obs_y_.begin(), obs_y_.end());
        all_obs_y.insert(all_obs_y.end(), map_y_.begin(), map_y_.end());
        t_after_theta_obs = ros::WallTime::now();

        retry_profile_active_ = false;
        bool success = solveOCP(x_ref_, y_ref_, theta_sub,
                                current_state_, all_obs_x, all_obs_y);

        const int retry_attempts = std::max(0, retry_profile_max_attempts_);
        if (!success && retry_profile_enabled_ && retry_attempts > 0) {
            for (int attempt = 1; attempt <= retry_attempts && !success; ++attempt) {
                retry_profile_active_ = true;
                ROS_WARN("MPC retry profile attempt %d/%d (mode=%s)",
                         attempt, retry_attempts, display_text_.c_str());
                success = solveOCP(x_ref_, y_ref_, theta_sub,
                                   current_state_, all_obs_x, all_obs_y);
                if (success) {
                    ROS_INFO("MPC retry profile recovered on attempt %d/%d",
                             attempt, retry_attempts);
                } else {
                    ROS_WARN("MPC retry profile failed on attempt %d/%d",
                             attempt, retry_attempts);
                }
            }
        }
        retry_profile_active_ = false;
        t_after_solve = ros::WallTime::now();

        // if (success) {
        //     if (mode_ == ControlMode::ROTATION_SHIM) {
        //         // Bypass the MPC output entirely — command a direct, steady
        //         // in-place rotation.  The solver ran to keep the warm solution
        //         // warm but its output is not used.
        //         const double shim_w = shim_turn_left_ ? shim_omega_ : -shim_omega_;
        //         v_opt_ = 0.0;
        //         w_opt_ = shim_w;
        //         writeVelocityCommand(0.0, shim_w, cmd_vel);
        //         ROS_INFO_THROTTLE(0.5, "[ROT_SHIM] spinning %s at w=%.2f rad/s",
        //                           shim_turn_left_ ? "LEFT" : "RIGHT", shim_w);
        //     } else {
        //         writeVelocityCommand(v_opt_, w_opt_, cmd_vel);
        //         ROS_INFO_THROTTLE(0.5, "[%s] V=%.3f W=%.3f",
        //                           display_text_.c_str(), v_opt_, w_opt_);
        //     }
        // } else {
        //     v_opt_ = 0.0; w_opt_ = 0.0;
        //     writeVelocityCommand(0.0, 0.0, cmd_vel);
        //     ROS_WARN("MPC solve failed — stopping robot");
        //     return false;
        // }
        
        // Dun Yan: Bottom one is new code with recovery logic, above is old simpler version without recovery
        if (success) {
            consecutive_solve_failures_ = 0;
            recovery_active_            = false;
            if (mode_ == ControlMode::ROTATION_SHIM) {
                const double shim_w = shim_turn_left_ ? shim_omega_ : -shim_omega_;
                v_opt_ = 0.0;
                w_opt_ = shim_w;
                writeVelocityCommand(0.0, shim_w, cmd_vel);
                ROS_INFO_THROTTLE(0.5, "[ROT_SHIM] spinning %s at w=%.2f rad/s",
                                  shim_turn_left_ ? "LEFT" : "RIGHT", shim_w);
            } else {
                writeVelocityCommand(v_opt_, w_opt_, cmd_vel);
                ROS_INFO_THROTTLE(0.5, "[%s] V=%.3f W=%.3f",
                                  display_text_.c_str(), v_opt_, w_opt_);
            }
        } else {
            v_opt_ = 0.0; w_opt_ = 0.0;
            consecutive_solve_failures_++;
            ROS_WARN("MPC solve failed (%d consecutive)", consecutive_solve_failures_);

            if (consecutive_solve_failures_ >= RECOVERY_TRIGGER_COUNT && !recovery_active_) {
                recovery_active_ = true;
                ROS_WARN("[RECOVERY] ACTIVATED — backtracking along reference path");
                recovery_path_x_.clear();
                recovery_path_y_.clear();
                const int snap_end = std::min(path_progress_idx_,
                                              static_cast<int>(og_x_ref_.size()) - 1);
                for (int k = snap_end; k >= 0; --k) {
                    recovery_path_x_.push_back(og_x_ref_[k]);
                    recovery_path_y_.push_back(og_y_ref_[k]);
                }
                recovery_path_idx_ = 0;
                // Safety timeout: arc length / speed * 25 Hz * 2x buffer
                double arc_len = 0.0;
                for (int k = 1; k < (int)recovery_path_x_.size(); ++k)
                    arc_len += std::hypot(recovery_path_x_[k] - recovery_path_x_[k - 1],
                                         recovery_path_y_[k] - recovery_path_y_[k - 1]);
                recovery_ticks_remaining_ = static_cast<int>(arc_len / std::abs(RECOVERY_V) * 50.0) + 30;
                ROS_WARN("[RECOVERY] Backtracking along %zu waypoints (%.2f m, %d tick budget)",
                         recovery_path_x_.size(), arc_len, recovery_ticks_remaining_);
            }

            if (recovery_active_) {
                // Advance past waypoints already within threshold
                while (recovery_path_idx_ < (int)recovery_path_x_.size()) {
                    double dx = recovery_path_x_[recovery_path_idx_] - current_state_[0];
                    double dy = recovery_path_y_[recovery_path_idx_] - current_state_[1];
                    if (std::hypot(dx, dy) > RECOVERY_WAYPOINT_THRESH) break;
                    recovery_path_idx_++;
                }
                // This is to prevent overshooting the backtrack path if the solver is failing repeatedly while already on the recovery path.  If we exhaust the path or run out of time, stop and wait for the next MPC solution to hopefully be feasible again.
                if (recovery_path_idx_ >= (int)recovery_path_x_.size() || recovery_ticks_remaining_ <= 0) {
                    ROS_WARN("[RECOVERY] Finished (idx=%d/%zu ticks=%d)",
                             recovery_path_idx_, recovery_path_x_.size(), recovery_ticks_remaining_);
                    recovery_active_ = false;
                    writeVelocityCommand(0.0, 0.0, cmd_vel);
                    return false;
                }

                recovery_ticks_remaining_--;
                double dx          = recovery_path_x_[recovery_path_idx_] - current_state_[0];
                double dy          = recovery_path_y_[recovery_path_idx_] - current_state_[1];
                double desired_dir = std::atan2(dy, dx);
                double travel_dir  = current_state_[2] + M_PI;
                double heading_err = std::atan2(std::sin(desired_dir - travel_dir),
                                                std::cos(desired_dir - travel_dir));
                double w_recovery  = std::max(-omega_max_, std::min(2.0 * heading_err, omega_max_));
                display_text_      = "RECOVERY";
                writeVelocityCommand(RECOVERY_V, w_recovery, cmd_vel);
                return true;
            }
            writeVelocityCommand(0.0, 0.0, cmd_vel);
            return false;
        }     

        if (bench_log_enabled_) {
            const auto t_run_end = ros::WallTime::now();
            ROS_INFO_STREAM_THROTTLE(bench_log_period_s_,
                "[BENCH][run] total=" << (t_run_end - t_run_start).toSec() * 1e3 << " ms"
                << " | ref_build=" << (t_after_refs - t_run_start).toSec() * 1e3 << " ms"
                << " | theta_obs_merge=" << (t_after_theta_obs - t_after_refs).toSec() * 1e3 << " ms"
                << " | solveOCP_call=" << (t_after_solve - t_after_theta_obs).toSec() * 1e3 << " ms"
                << " | cmd_publish=" << (t_run_end - t_after_solve).toSec() * 1e3 << " ms"
                << " | obs_counts(laser,map,dyn)="
                << obs_x_.size() << "," << map_x_.size() << "," << dynamic_obstacles_.size());
        }
        return true;
    } catch (const std::exception& e) {
        ROS_ERROR("Exception in MPC runOnce: %s", e.what());
        writeVelocityCommand(0.0, 0.0, cmd_vel);
        return false;
    }
}

} // namespace mpc_controller