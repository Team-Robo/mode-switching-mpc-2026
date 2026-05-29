#include <teamrobo2026/mpc_local_planner_ros.h>

#include <pluginlib/class_list_macros.h>
#include <tf2/utils.h>
#include <algorithm>
#include <cmath>

namespace mpc_controller {

MpcLocalPlannerROS::MpcLocalPlannerROS() = default;

MpcLocalPlannerROS::~MpcLocalPlannerROS() = default;

double MpcLocalPlannerROS::shortestAngularDistance(double from, double to) const {
    double diff = to - from;
    while (diff > M_PI) diff -= 2.0 * M_PI;
    while (diff < -M_PI) diff += 2.0 * M_PI;
    return diff;
}

bool MpcLocalPlannerROS::isRotateToGoalSafe(const geometry_msgs::PoseStamped& robot_pose, double w_cmd) const {
    if (!costmap_ros_ || !world_model_) {
        return false;
    }

    const std::vector<geometry_msgs::Point> footprint = costmap_ros_->getRobotFootprint();
    if (footprint.empty()) {
        ROS_WARN_THROTTLE(1.0, "Rotate-to-goal safety check: empty footprint");
        return false;
    }

    const double x = robot_pose.pose.position.x;
    const double y = robot_pose.pose.position.y;
    const double yaw0 = tf2::getYaw(robot_pose.pose.orientation);
    const double total_delta = w_cmd * std::max(0.0, rotate_to_goal_lookahead_s_);
    const double step = std::max(1.0, rotate_to_goal_yaw_step_deg_) * M_PI / 180.0;
    const int n_steps = std::max(1, static_cast<int>(std::ceil(std::fabs(total_delta) / step)));

    for (int i = 0; i <= n_steps; ++i) {
        const double alpha = static_cast<double>(i) / static_cast<double>(n_steps);
        const double yaw = yaw0 + alpha * total_delta;
        const double cost = world_model_->footprintCost(x, y, yaw, footprint);
        if (cost < 0.0) {
            if (!rotate_to_goal_block_unknown_ &&
                cost == static_cast<double>(-2)) {
                continue;
            }
            return false;
        }
    }
    return true;
}

void MpcLocalPlannerROS::initialize(std::string name, tf2_ros::Buffer* tf,
                                    costmap_2d::Costmap2DROS* costmap_ros) {
    if (initialized_) {
        ROS_WARN("MpcLocalPlannerROS already initialized");
        return;
    }

    name_ = name;
    tf_ = tf;
    costmap_ros_ = costmap_ros;

    ros::NodeHandle nh;
    ros::NodeHandle nh_private("~/" + name);

    nh_private.param("xy_goal_tolerance", xy_goal_tolerance_, 0.25);
    nh_private.param("yaw_goal_tolerance", yaw_goal_tolerance_, 0.157);
    nh_private.param("enable_rotate_to_goal", enable_rotate_to_goal_, true);
    nh_private.param("rotate_to_goal_kp", rotate_to_goal_kp_, 1.8);
    nh_private.param("rotate_to_goal_max_w", rotate_to_goal_max_w_, 0.6);
    nh_private.param("rotate_to_goal_min_w", rotate_to_goal_min_w_, 0.15);
    nh_private.param("rotate_to_goal_lookahead_s", rotate_to_goal_lookahead_s_, 0.8);
    nh_private.param("rotate_to_goal_yaw_step_deg", rotate_to_goal_yaw_step_deg_, 5.0);
    nh_private.param("rotate_to_goal_block_unknown", rotate_to_goal_block_unknown_, true);

    controller_ = std::make_unique<MpcController>(nh, nh_private);
    world_model_ = std::make_unique<base_local_planner::CostmapModel>(*costmap_ros_->getCostmap());

    if (!controller_->solverReady()) {
        ROS_ERROR("MpcLocalPlannerROS: ACADOS solver failed to initialize");
    } else {
        ROS_INFO("MpcLocalPlannerROS initialized as '%s'", name_.c_str());
    }

    initialized_ = true;
}

bool MpcLocalPlannerROS::setPlan(
    const std::vector<geometry_msgs::PoseStamped>& plan) {
    if (!initialized_ || !controller_ || !tf_) {
        ROS_ERROR("MpcLocalPlannerROS::setPlan called before initialize");
        return false;
    }
    return controller_->setPlan(plan, tf_);
}

bool MpcLocalPlannerROS::computeVelocityCommands(geometry_msgs::Twist& cmd_vel) {
    cmd_vel = geometry_msgs::Twist();
    if (!initialized_ || !controller_ || !costmap_ros_) {
        return false;
    }
    if (!controller_->solverReady()) {
        return false;
    }

    ros::spinOnce();

    geometry_msgs::PoseStamped robot_pose;
    if (!costmap_ros_->getRobotPose(robot_pose)) {
        ROS_WARN_THROTTLE(2.0, "MpcLocalPlannerROS: could not get robot pose");
        return false;
    }
    controller_->updateRobotPose(robot_pose);

    if (enable_rotate_to_goal_) {
        double gx, gy, gyaw;
        if (controller_->getGoalPose(gx, gy, gyaw)) {
            const double rx = robot_pose.pose.position.x;
            const double ry = robot_pose.pose.position.y;
            const double ryaw = tf2::getYaw(robot_pose.pose.orientation);
            const double dist_to_goal = std::hypot(gx - rx, gy - ry);
            const double yaw_err = shortestAngularDistance(ryaw, gyaw);

            if (dist_to_goal <= xy_goal_tolerance_ &&
                std::fabs(yaw_err) > yaw_goal_tolerance_) {
                double w_cmd = rotate_to_goal_kp_ * yaw_err;
                w_cmd = std::max(-rotate_to_goal_max_w_, std::min(rotate_to_goal_max_w_, w_cmd));
                if (std::fabs(w_cmd) < rotate_to_goal_min_w_) {
                    w_cmd = std::copysign(rotate_to_goal_min_w_, yaw_err);
                }

                if (!isRotateToGoalSafe(robot_pose, w_cmd)) {
                    ROS_WARN_THROTTLE(0.5,
                        "Rotate-to-goal blocked by footprint collision risk (dist=%.3f yaw_err_deg=%.1f)",
                        dist_to_goal, yaw_err * 180.0 / M_PI);
                    cmd_vel.linear.x = 0.0;
                    cmd_vel.angular.z = 0.0;
                    return false;
                }

                cmd_vel.linear.x = 0.0;
                cmd_vel.angular.z = w_cmd;
                ROS_INFO_THROTTLE(0.5,
                    "Rotate-to-goal active: dist=%.3f yaw_err_deg=%.1f cmd_w=%.3f",
                    dist_to_goal, yaw_err * 180.0 / M_PI, w_cmd);
                return true;
            }
        }
    }

    return controller_->runOnce(cmd_vel);
}

bool MpcLocalPlannerROS::isGoalReached() {
    if (!initialized_ || !controller_) {
        return false;
    }
    return controller_->isGoalReached(xy_goal_tolerance_, yaw_goal_tolerance_);
}

}  // namespace mpc_controller

PLUGINLIB_EXPORT_CLASS(mpc_controller::MpcLocalPlannerROS, nav_core::BaseLocalPlanner)
