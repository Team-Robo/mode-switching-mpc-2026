#include <teamrobo2026/mpc_local_planner_ros.h>

#include <pluginlib/class_list_macros.h>

namespace mpc_controller {

MpcLocalPlannerROS::MpcLocalPlannerROS() = default;

MpcLocalPlannerROS::~MpcLocalPlannerROS() = default;

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

    controller_ = std::make_unique<MpcController>(nh, nh_private);

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
