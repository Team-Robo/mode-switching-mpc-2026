#include <ros/ros.h>
#include <sensor_msgs/LaserScan.h>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PolygonStamped.h>
#include <nav_msgs/Path.h>
#include <nav_msgs/Odometry.h>
#include <tf/transform_datatypes.h>

class Inspection {
private:
    // Topic definitions
    const std::string TOPIC_CMD_VEL = "/cmd_vel";
    const std::string TOPIC_FRONT_SCAN = "/front/scan";
    const std::string TOPIC_LOCAL_FOOTPRINT = "/move_base/local_costmap/footprint";
    const std::string TOPIC_GLOBAL_PLAN = "/move_base/TrajectoryPlannerROS/global_plan";
    const std::string TOPIC_LOCAL_PLAN = "/move_base/TrajectoryPlannerROS/local_plan";
    const std::string TOPIC_ODOM = "/odometry/filtered";
    const std::string TOPIC_MPC = "/mpc_plan";
    
    // Stored objects
    sensor_msgs::LaserScan scan;
    geometry_msgs::Twist cmd_vel;
    nav_msgs::Path global_plan;
    nav_msgs::Path local_plan;
    nav_msgs::Odometry odometry;
    geometry_msgs::PolygonStamped footprint;
    
    // Subscribers and Publishers
    ros::Subscriber sub_front_scan;
    ros::Subscriber sub_odometry;
    ros::Subscriber sub_global_plan;
    ros::Subscriber sub_local_plan;
    ros::Subscriber sub_footprint;
    ros::Subscriber sub_cmd_vel;
    ros::Publisher pub_cmd_vel;
    
public:
    Inspection(ros::NodeHandle& nh) {
        // Initialize subscribers
        sub_front_scan = nh.subscribe(TOPIC_FRONT_SCAN, 10, &Inspection::callbackFrontScan, this);
        sub_odometry = nh.subscribe(TOPIC_ODOM, 10, &Inspection::callbackOdometry, this);
        sub_global_plan = nh.subscribe(TOPIC_GLOBAL_PLAN, 10, &Inspection::callbackGlobalPlan, this);
        sub_local_plan = nh.subscribe(TOPIC_MPC, 10, &Inspection::callbackLocalPlan, this);
        sub_footprint = nh.subscribe(TOPIC_LOCAL_FOOTPRINT, 10, &Inspection::callbackFootprint, this);
        sub_cmd_vel = nh.subscribe(TOPIC_CMD_VEL, 10, &Inspection::callbackCmdVel, this);
        
        // Initialize publisher
        pub_cmd_vel = nh.advertise<geometry_msgs::Twist>(TOPIC_CMD_VEL, 10);
    }
    
    void callbackFrontScan(const sensor_msgs::LaserScan::ConstPtr& msg) {
        scan = *msg;
        if (false) {
            ROS_INFO("Scan points: %lu From Max: %.2f | Min: %.2f", 
                     msg->ranges.size(), msg->range_max, msg->range_min);
            ROS_INFO("Angle from: %.2f to: %.2f increment: %.3f",
                     msg->angle_min * 180.0 / M_PI,
                     msg->angle_max * 180.0 / M_PI,
                     msg->angle_increment * 180.0 / M_PI);
        }
    }
    
    void callbackGlobalPlan(const nav_msgs::Path::ConstPtr& msg) {
        global_plan = *msg;
        if (true) {
            ROS_INFO("Global Path points: %lu", msg->poses.size());
            if (msg->poses.size() > 3) {
                ROS_INFO("Pose 3: x=%.2f, y=%.2f", 
                         msg->poses[3].pose.position.x,
                         msg->poses[3].pose.position.y);
            }
            ROS_INFO("Local Path points: %lu", local_plan.poses.size());
        }
    }
    
    void callbackLocalPlan(const nav_msgs::Path::ConstPtr& msg) {
        local_plan = *msg;
    }
    
    void callbackOdometry(const nav_msgs::Odometry::ConstPtr& msg) {
        odometry = *msg;
        if (false) {
            ROS_INFO("==========================");
            ROS_INFO("----------------------- pose.position");
            ROS_INFO("x=%.3f, y=%.3f, z=%.3f",
                     msg->pose.pose.position.x,
                     msg->pose.pose.position.y,
                     msg->pose.pose.position.z);
            
            ROS_INFO("----------------------- pose.orientation");
            ROS_INFO("x=%.3f, y=%.3f, z=%.3f, w=%.3f",
                     msg->pose.pose.orientation.x,
                     msg->pose.pose.orientation.y,
                     msg->pose.pose.orientation.z,
                     msg->pose.pose.orientation.w);
            
            ROS_INFO("----------------------- pose.heading");
            tf::Quaternion q(
                msg->pose.pose.orientation.x,
                msg->pose.pose.orientation.y,
                msg->pose.pose.orientation.z,
                msg->pose.pose.orientation.w
            );
            tf::Matrix3x3 m(q);
            double roll, pitch, yaw;
            m.getRPY(roll, pitch, yaw);
            
            ROS_INFO("Rad: %.3f", yaw);
            ROS_INFO("Degree: %.3f", yaw * 180.0 / M_PI);
            
            ROS_INFO("----------------------- twist.linear");
            ROS_INFO("x=%.3f, y=%.3f, z=%.3f",
                     msg->twist.twist.linear.x,
                     msg->twist.twist.linear.y,
                     msg->twist.twist.linear.z);
            
            ROS_INFO("----------------------- twist.angular");
            ROS_INFO("x=%.3f, y=%.3f, z=%.3f",
                     msg->twist.twist.angular.x,
                     msg->twist.twist.angular.y,
                     msg->twist.twist.angular.z);
        }
    }
    
    void callbackFootprint(const geometry_msgs::PolygonStamped::ConstPtr& msg) {
        footprint = *msg;
        if (false) {
            ROS_INFO("Number of points on the Polygon: %lu", msg->polygon.points.size());
            for (const auto& point : msg->polygon.points) {
                ROS_INFO("Point: x=%.3f, y=%.3f, z=%.3f", point.x, point.y, point.z);
            }
        }
    }
    
    void callbackCmdVel(const geometry_msgs::Twist::ConstPtr& msg) {
        if (false) {
            ROS_INFO("Linear: x=%.3f, y=%.3f, z=%.3f; Angular: x=%.3f, y=%.3f, z=%.3f",
                     msg->linear.x, msg->linear.y, msg->linear.z,
                     msg->angular.x, msg->angular.y, msg->angular.z);
        }
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "inspection_node");
    ros::NodeHandle nh;
    
    ROS_INFO("Inspection Node Started");
    
    Inspection inspect(nh);
    
    ros::spin();
    
    return 0;
}