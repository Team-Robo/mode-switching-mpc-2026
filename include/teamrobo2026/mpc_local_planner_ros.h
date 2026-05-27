#ifndef TEAMROBO2026_MPC_LOCAL_PLANNER_ROS_H
#define TEAMROBO2026_MPC_LOCAL_PLANNER_ROS_H

#include <memory>
#include <string>
#include <vector>

#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Twist.h>
#include <nav_core/base_local_planner.h>
#include <costmap_2d/costmap_2d_ros.h>
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
    bool initialized_ = false;
    std::string name_;
    tf2_ros::Buffer* tf_ = nullptr;
    costmap_2d::Costmap2DROS* costmap_ros_ = nullptr;

    std::unique_ptr<MpcController> controller_;

    double xy_goal_tolerance_ = 0.25;
    double yaw_goal_tolerance_ = 0.157;
};

}  // namespace mpc_controller

#endif  // TEAMROBO2026_MPC_LOCAL_PLANNER_ROS_H
