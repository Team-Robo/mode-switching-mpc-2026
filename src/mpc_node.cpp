#include "mpc_node.hpp"
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <algorithm>
#include <limits>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>

// mpc_node.cpp

namespace mpc_controller {

MPCNode::MPCNode(ros::NodeHandle& nh, ros::NodeHandle& nh_private)
    : nh_(nh), nh_private_(nh_private)
{
    nh_private_.param<double>("v_linear_max",      v_linear_max_,      2.0);
    nh_private_.param<double>("v_static_obs_max",  v_static_obs_max_,  0.5);
    nh_private_.param<double>("omega_max",          omega_max_,         1.8);
    nh_private_.param<double>("omega_static_obs_max", omega_static_obs_max_, 0.8);

    nh_private_.param<double>("weight_position_error",  weight_position_error_,  130.0);
    nh_private_.param<double>("weight_heading_error",   weight_heading_error_,   49.0);
    nh_private_.param<double>("weight_velocity",        weight_velocity_,        27.0);
    nh_private_.param<double>("weight_acceleration",    weight_acceleration_,    0.0011130334330777412);

    nh_private_.param<double>("accel_weight_mult_static",   accel_weight_mult_static_,  3.0);
    nh_private_.param<double>("accel_weight_mult_dynamic",  accel_weight_mult_dynamic_, 5.5);

    nh_private_.param<double>("static_obs_safe_dist",  static_obs_safe_dist_,  1.1);
    nh_private_.param<double>("dynamic_obs_safe_dist", dynamic_obs_safe_dist_, 4.0);

    nh_private_.param<double>("robot_radius",       robot_radius_,       0.37);
    nh_private_.param<double>("dynamic_obs_radius", dynamic_obs_radius_, 0.5);
    nh_private_.param<double>("safety_margin",      safety_margin_,      0.01);
    nh_private_.param<double>("obs_search_radius",  obs_search_radius_,  4.0);

    nh_private_.param<double>("reversal_threshold", reversal_threshold_, 0.85);
    nh_private_.param<double>("reversal_angle_deg", reversal_angle_deg_, 90.0);

    // ROTATION_SHIM params
    nh_private_.param<double>("lidar_blind_angle_deg", lidar_blind_angle_deg_, 45.0);
    nh_private_.param<double>("shim_exit_heading_deg", shim_exit_heading_deg_, 30.0);
    nh_private_.param<double>("shim_omega",             shim_omega_,            1.2);

    // RUSH_GOAL params — direct absolute weights, no multipliers
    nh_private_.param<double>("rush_goal_dist",       rush_goal_dist_,       4.3);
    nh_private_.param<double>("rush_goal_exit_dist",  rush_goal_exit_dist_,  0.0);
    nh_private_.param<double>("rush_weight_position", rush_weight_position_, 0.0);
    nh_private_.param<double>("rush_weight_heading",  rush_weight_heading_,  3030.0);
    nh_private_.param<double>("rush_weight_velocity", rush_weight_velocity_, 3050.0);
    nh_private_.param<double>("rush_weight_accel",    rush_weight_accel_,    0.0);
    nh_private_.param<double>("rush_vref",            rush_vref_,            2.0);

    // Dynamic obstacle hysteresis timeout
    nh_private_.param<double>("dynamic_obs_timeout", dynamic_obs_timeout_, 0.2);

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
    ROS_INFO("  ROTATION_SHIM: blind=%.1f deg  exit=%.1f deg  omega=%.2f rad/s",
             lidar_blind_angle_deg_, shim_exit_heading_deg_, shim_omega_);
    ROS_INFO("  RUSH_GOAL: dist=%.1f m  pos=%.0f  hdg=%.0f  vel=%.0f  accel=%.5f  vref=%.1f",
             rush_goal_dist_, rush_weight_position_, rush_weight_heading_,
             rush_weight_velocity_, rush_weight_accel_, rush_vref_);
    ROS_INFO("  Dynamic obs hysteresis timeout: %.2f s", dynamic_obs_timeout_);

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

    // Initialise hysteresis timestamp to a time far in the past
    last_dynamic_obs_time_ = ros::Time(0);

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
    
    // Reset progress index for new plan
    path_progress_idx_ = 0;

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

double MPCNode::diffAngle(double a1, double a2) const {
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

// =============================================================================
// ROTATION_SHIM helper
// =============================================================================
bool MPCNode::isGoalInBlindSpot(double robot_theta,
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

void MPCNode::findClosestPoint(const std::vector<double>& x_ref,
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
// RUSH_GOAL helpers
// =============================================================================
double MPCNode::distToGoal(double rx, double ry) const {
    if (og_x_ref_.empty()) return std::numeric_limits<double>::max();
    double dx = og_x_ref_.back() - rx;
    double dy = og_y_ref_.back() - ry;
    return std::sqrt(dx*dx + dy*dy);
}

void MPCNode::warmStartFromCurrentState(const std::vector<double>& current_state) {
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
// selectTwoObstacles
// =============================================================================
void MPCNode::selectTwoObstacles(
    const std::vector<double>& obs_x,
    const std::vector<double>& obs_y,
    const std::vector<PredictedObstacle>& predicted_obstacles,
    double rx, double ry,
    int stage,
    double search_radius_sq,
    double p_data[4]) const
{
    constexpr double collinearity_thresh_rad = 30.0 * M_PI / 180.0;

    struct Candidate {
        double x, y;
        double eff_dist;
        double angle;
    };

    std::vector<Candidate> candidates;
    candidates.reserve(obs_x.size() + predicted_obstacles.size());

    for (size_t j = 0; j < obs_x.size(); ++j) {
        double dx = obs_x[j] - rx, dy = obs_y[j] - ry;
        double d2 = dx*dx + dy*dy;
        if (d2 > search_radius_sq) continue;
        double d = std::sqrt(d2);
        candidates.push_back({obs_x[j], obs_y[j], d, std::atan2(dy, dx)});
    }

    for (const auto& pred : predicted_obstacles) {
        if (stage >= static_cast<int>(pred.x_predicted.size())) continue;
        double px = pred.x_predicted[stage];
        double py = pred.y_predicted[stage];
        double pr = pred.radius_predicted[stage];
        double dx = px - rx, dy = py - ry;
        double dist_center = std::sqrt(dx*dx + dy*dy);
        if (dist_center > obs_search_radius_) continue;
        double eff_dist = std::max(0.0, dist_center - pr);
        candidates.push_back({px, py, eff_dist, std::atan2(dy, dx)});
    }

    if (candidates.empty()) {
        p_data[0] = p_data[2] = 1000.0;
        p_data[1] = p_data[3] = 1000.0;
        return;
    }

    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate& a, const Candidate& b){
                  return a.eff_dist < b.eff_dist;
              });

    p_data[0] = candidates[0].x;
    p_data[1] = candidates[0].y;

    if (candidates.size() == 1) {
        p_data[2] = candidates[0].x;
        p_data[3] = candidates[0].y;
        return;
    }

    bool found_slot2 = false;
    for (size_t k = 1; k < candidates.size(); ++k) {
        double angle_diff = std::fabs(candidates[k].angle - candidates[0].angle);
        if (angle_diff > M_PI) angle_diff = 2.0*M_PI - angle_diff;
        if (angle_diff > collinearity_thresh_rad) {
            p_data[2] = candidates[k].x;
            p_data[3] = candidates[k].y;
            found_slot2 = true;
            break;
        }
    }

    if (!found_slot2) {
        p_data[2] = candidates[1].x;
        p_data[3] = candidates[1].y;
    }
}

// =============================================================================
// checkEmergencyStop
// =============================================================================
bool MPCNode::checkEmergencyStop(
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
            if (slot > 2) {
                ROS_WARN_THROTTLE(0.5,
                    "Emergency stop: 3rd+ dynamic obstacle at %.2fm (threshold %.2fm)",
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
bool MPCNode::solveOCP(const std::vector<double>& x_ref,
                       const std::vector<double>& y_ref,
                       const std::vector<double>& theta_ref,
                       const std::vector<double>& current_state,
                       const std::vector<double>& obs_x,
                       const std::vector<double>& obs_y
                    )
{
    auto t0 = ros::WallTime::now();

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
    // 3. EMERGENCY STOP CHECK
    // =========================================================================
    if (checkEmergencyStop(predicted_obstacles_, current_state)) {
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
    if (has_dynamic_obs) {
        last_dynamic_obs_time_ = now;
    }
    const bool dynamic_obs_active =
        has_dynamic_obs ||
        ((now - last_dynamic_obs_time_).toSec() < dynamic_obs_timeout_);

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
    bool facing_goal = false;
    if (!og_x_ref_.empty()) {
        double dx_goal = og_x_ref_.back() - current_state[0];
        double dy_goal = og_y_ref_.back() - current_state[1];
        double angle_to_goal = std::atan2(dy_goal, dx_goal);
        double heading_err = diffAngle(current_state[2], angle_to_goal);
        facing_goal = (heading_err < (45.0 * M_PI / 180.0));
    }
    const bool trigger_rush = near_goal && !dynamic_obs_active && !has_static_obs && facing_goal;

    if (dynamic_obs_active) {
        rush_goal_latched_ = false;
    } else if (trigger_rush) {
        rush_goal_latched_ = true;
    } else if (rush_goal_latched_ && goal_dist < rush_goal_exit_dist_) {
        rush_goal_latched_ = false;
        ROS_INFO("RUSH_GOAL: goal reached, releasing latch (dist=%.2fm)", goal_dist);
    }

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

    // =========================================================================
    // 4b. ROTATION_SHIM — evaluated after primary mode, overrides everything
    //     except DYNAMIC_OBS (a moving obstacle trumps a blind-spot spin).
    //
    //     Triggers when a reversal is needed AND the goal direction falls inside
    //     the lidar blind zone.  The robot spins in place until the goal clears
    //     the blind zone edge (plus shim_exit_heading_deg_ buffer), then this
    //     mode releases and normal reversal takes over.
    // =========================================================================
    if (mode_ != ControlMode::DYNAMIC_OBS) {
        bool reversal_pending = checkReversalNeeded(theta_ref, current_state[2]);

        if (reversal_pending && !og_x_ref_.empty()) {
            double gx = og_x_ref_.back();
            double gy = og_y_ref_.back();
            bool in_blind = isGoalInBlindSpot(current_state[2], gx, gy,
                                              current_state[0], current_state[1]);

            // Entry: latch the shim ON and choose spin direction once.
            if (!shim_active_ && in_blind) {
                shim_active_ = true;

                // Rotate toward whichever side brings the goal into the FOV
                // faster — use the cross product to find which side the goal is on.
                double dx = gx - current_state[0];
                double dy = gy - current_state[1];
                // Cross product z-component: positive → goal is left of heading
                double cross = std::cos(current_state[2]) * dy
                             - std::sin(current_state[2]) * dx;
                shim_turn_left_ = (cross > 0.0);
                ROS_INFO("ROTATION_SHIM engaged — turning %s (goal in blind spot)",
                         shim_turn_left_ ? "LEFT" : "RIGHT");
            }

            // Exit: use shim_exit_heading_deg_ as an angular buffer so the goal
            // must be clearly inside the FOV before we release (avoids chattering
            // at the blind-zone boundary).
            if (shim_active_) {
                double dx = gx - current_state[0];
                double dy = gy - current_state[1];
                double angle_to_goal = std::atan2(dy, dx);
                double rear_dir = current_state[2] + M_PI;
                double diff_from_rear = diffAngle(angle_to_goal, rear_dir);
                double clear_angle = lidar_blind_angle_deg_ * M_PI / 180.0
                                   + shim_exit_heading_deg_ * M_PI / 180.0;
                if (diff_from_rear >= clear_angle) {
                    shim_active_ = false;
                    ROS_INFO("ROTATION_SHIM released — goal now %.1f deg from rear (need >= %.1f deg)",
                             diff_from_rear * 180.0 / M_PI, clear_angle * 180.0 / M_PI);
                }
            }
        } else {
            // No reversal needed — always clear the shim.
            if (shim_active_) {
                shim_active_ = false;
                ROS_INFO("ROTATION_SHIM cleared (no reversal needed)");
            }
        }

        if (shim_active_) {
            mode_         = ControlMode::ROTATION_SHIM;
            display_text_ = "ROT_SHIM";
            ROS_INFO_THROTTLE(0.5, "Mode: ROTATION_SHIM (spinning %s)",
                              shim_turn_left_ ? "LEFT" : "RIGHT");
        }
    } else {
        // Dynamic obstacle active — disengage shim so it re-evaluates cleanly
        // once the dynamic obstacle clears.
        shim_active_ = false;
    }

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
    if (mode_ == ControlMode::ROTATION_SHIM) {
        v_opt_ = 0.0;
        w_opt_ = shim_turn_left_ ? shim_omega_ : -shim_omega_;
        const double ms_shim = (ros::WallTime::now() - t0).toSec() * 1e3;
        ROS_INFO_STREAM_THROTTLE(1.0, "[BENCH] ROTATION_SHIM early-exit = " << ms_shim << " ms");
        return true;
    }

    // =========================================================================
    // 5. RUSH_GOAL WARM-START
    // =========================================================================
    if (mode_ == ControlMode::RUSH_GOAL && !prev_was_rush_goal_) {
        ROS_INFO("RUSH_GOAL: warm-starting solver from current state");
        warmStartFromCurrentState(current_state);
    }
    prev_was_rush_goal_ = (mode_ == ControlMode::RUSH_GOAL);

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
            v_cap                  = v_static_obs_max_;
            omega_cap              = omega_static_obs_max_;
            effective_accel_weight = weight_acceleration_ * accel_weight_mult_static_;
            eff_pos_weight         = weight_position_error_;
            eff_heading_weight     = weight_heading_error_;
            eff_velocity_weight    = weight_velocity_;
            break;

        case ControlMode::DYNAMIC_OBS:
            v_cap                  = v_linear_max_;
            omega_cap              = omega_max_;
            effective_accel_weight = weight_acceleration_ * accel_weight_mult_dynamic_;
            eff_pos_weight         = weight_position_error_;
            eff_heading_weight     = weight_heading_error_;
            eff_velocity_weight    = weight_velocity_;
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

    // =========================================================================
    // 7. REVERSAL OVERLAY
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
    // 8. CONSTRAINT BOUNDS
    // =========================================================================
    double min_dist_sq;
    if (mode_ == ControlMode::DYNAMIC_OBS) {
        min_dist_sq = std::pow(robot_radius_ + dynamic_obs_radius_ + safety_margin_, 2.0);
    } else {
        min_dist_sq = std::pow(robot_radius_ + safety_margin_, 2.0);
    }
    double lh[4] = { -v_cap, -omega_cap, min_dist_sq, min_dist_sq };
    double uh[4] = {  v_cap,  omega_cap, 1.0e9,       1.0e9 };

    for (int i = 0; i < N_; ++i) {
        ocp_nlp_constraints_model_set(nlp_config_, nlp_dims_, nlp_in_, nlp_out_, i, "lh", lh);
        ocp_nlp_constraints_model_set(nlp_config_, nlp_dims_, nlp_in_, nlp_out_, i, "uh", uh);
    }

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

    for (int i = 0; i < N_; ++i)
        ocp_nlp_cost_model_set(nlp_config_, nlp_dims_, nlp_in_, i, "W", W.data());

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
        ocp_nlp_cost_model_set(nlp_config_, nlp_dims_, nlp_in_, N_, "W", W_e.data());
    }

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
            else if (mode_ == ControlMode::STATIC_OBS)   v_ref = v_static_obs_max_;
            else if (mode_ == ControlMode::ROTATION_SHIM) v_ref = 0.0;
            else                                          v_ref = v_linear_max_;
            double yref[6] = { sx, sy, st, v_ref, 0.0, 0.0 };
            ocp_nlp_cost_model_set(nlp_config_, nlp_dims_, nlp_in_, i, "yref", yref);
        } else {
            double yref_e[3] = { sx, sy, st };
            ocp_nlp_cost_model_set(nlp_config_, nlp_dims_, nlp_in_, N_, "yref", yref_e);
        }

        // ---- obstacle parameters ----
        double p_data[4];
        if (clear_obstacles) {
            p_data[0] = 1000.0; p_data[1] = 1000.0;
            p_data[2] = 1000.0; p_data[3] = 1000.0;
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
            selectTwoObstacles(local_obs_x, local_obs_y, predicted_obstacles_,
                            pred_x, pred_y,
                            i, search_radius_sq, p_data);
        }

        jackal_diff_drive_acados_update_params(acados_ocp_capsule_, i, p_data, 4);
    }

    // =========================================================================
    // 11. SOLVE
    // =========================================================================
    int status = jackal_diff_drive_acados_solve(acados_ocp_capsule_);
    if (status != 0) {
        ROS_WARN("ACADOS solver failed with status %d", status);
        return false;
    }

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
    const double ms = (ros::WallTime::now() - t0).toSec() * 1e3;
    ROS_INFO_STREAM_THROTTLE(1.0,
        "[BENCH] solveOCP wall-time = " << ms << " ms");
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
void MPCNode::run() {
    std::lock_guard<std::mutex> lock(solver_mutex_);
    try {
        if (og_x_ref_.empty() || theta_ref_.empty()) return;

        int min_idx = 0;
        findClosestPoint(og_x_ref_, og_y_ref_, current_state_[0], current_state_[1], min_idx);

        x_ref_.clear(); y_ref_.clear();
         // this mean the reference points will be at least 20cm apart, which helps the solver converge better by not fighting over closely spaced references
         // if reduced too much, the solver can struggle to find a feasible solution
        const double min_spacing_sq = 0.1 * 0.1;

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
            if (mode_ == ControlMode::ROTATION_SHIM) {
                // Bypass the MPC output entirely — command a direct, steady
                // in-place rotation.  The solver ran to keep the warm solution
                // warm but its output is not used.
                const double shim_w = shim_turn_left_ ? shim_omega_ : -shim_omega_;
                v_opt_ = 0.0;
                w_opt_ = shim_w;
                publishVelocity(0.0, shim_w);
                ROS_INFO_THROTTLE(0.5, "[ROT_SHIM] spinning %s at w=%.2f rad/s",
                                  shim_turn_left_ ? "LEFT" : "RIGHT", shim_w);
            } else {
                publishVelocity(v_opt_, w_opt_);
                ROS_INFO_THROTTLE(0.5, "[%s] V=%.3f W=%.3f",
                                  display_text_.c_str(), v_opt_, w_opt_);
            }
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
    nh_private.param<double>("mpc_rate", mpc_rate, 25.0);
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