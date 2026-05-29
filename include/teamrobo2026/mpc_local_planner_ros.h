#ifndef TEAMROBO2026_MPC_LOCAL_PLANNER_ROS_H
#define TEAMROBO2026_MPC_LOCAL_PLANNER_ROS_H

#include <memory>
#include <string>
#include <vector>

#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Twist.h>
#include <nav_core/base_local_planner.h>
#include <costmap_2d/costmap_2d_ros.h>
#include <base_local_planner/costmap_model.h>
#include <tf2_ros/buffer.h>
#include <ros/ros.h>

#include <teamrobo2026/mpc_controller.hpp>

namespace mpc_controller {

class MpcLocalPlannerROS : public nav_core::BaseLocalPlanner {
public:
    MpcLocalPlannerROS();
    ~MpcLocalPlannerROS() override;

    void initialize(std::string name, tf2_ros::Buffer* tf,
                    costmap_2d::Costmap2DROS* costmap_ros) override;

    bool setPlan(const std::vector<geometry_msgs::PoseStamped>& plan) override;

    bool computeVelocityCommands(geometry_msgs::Twist& cmd_vel) override;

    bool isGoalReached() override;

private:
    double shortestAngularDistance(double from, double to) const;
    bool isRotateToGoalSafe(const geometry_msgs::PoseStamped& robot_pose, double w_cmd) const;

    bool initialized_ = false;
    std::string name_;
    tf2_ros::Buffer* tf_ = nullptr;
    costmap_2d::Costmap2DROS* costmap_ros_ = nullptr;
    std::unique_ptr<base_local_planner::CostmapModel> world_model_;

    std::unique_ptr<MpcController> controller_;

    double xy_goal_tolerance_ = 0.25;
    double yaw_goal_tolerance_ = 0.157;
    bool enable_rotate_to_goal_ = true;
    double rotate_to_goal_kp_ = 1.8;
    double rotate_to_goal_max_w_ = 0.6;
    double rotate_to_goal_min_w_ = 0.15;
    double rotate_to_goal_lookahead_s_ = 0.8;
    double rotate_to_goal_yaw_step_deg_ = 5.0;
    bool rotate_to_goal_block_unknown_ = true;
};

}  // namespace mpc_controller

#endif  // TEAMROBO2026_MPC_LOCAL_PLANNER_ROS_H
