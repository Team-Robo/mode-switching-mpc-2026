#include <ros/ros.h>
#include <sensor_msgs/LaserScan.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <laser_geometry/laser_geometry.h>
#include <tf/transform_listener.h>
#include <cmath>

class LaserScanToPointCloud {
private:
    const double DETECTED = 1.0;
    const int SCAN_SPACING = 15;
    const std::string TOPIC_LASER_SCAN = "/front/scan";
    const std::string TOPIC_POINT_CLOUD_LASER = "/front/laser/cloud";
    const std::string TOPIC_POINT_CLOUD_ODOM = "/front/odom/cloud";
    
    sensor_msgs::LaserScan laser_scan;
    sensor_msgs::PointCloud2 point_cloud;
    laser_geometry::LaserProjection laser_projector;
    
    ros::Subscriber sub_laser_scan;
    ros::Publisher pub_point_cloud_laser;
    ros::Publisher pub_point_cloud_odom;
    
    tf::TransformListener tf_listener;
    
public:
    LaserScanToPointCloud(ros::NodeHandle& nh) {
        sub_laser_scan = nh.subscribe(TOPIC_LASER_SCAN, 1, 
                                      &LaserScanToPointCloud::callbackLaserScan, this);
        pub_point_cloud_laser = nh.advertise<sensor_msgs::PointCloud2>(
                                      TOPIC_POINT_CLOUD_LASER, 1);
        pub_point_cloud_odom = nh.advertise<sensor_msgs::PointCloud2>(
                                      TOPIC_POINT_CLOUD_ODOM, 1);
    }
    
    void callbackLaserScan(const sensor_msgs::LaserScan::ConstPtr& msg) {
        laser_scan = *msg;
        laser_projector.projectLaser(*msg, point_cloud);
        pub_point_cloud_laser.publish(point_cloud);
    }
    
    void run() {
        if (point_cloud.width == 0) {
            return;  // No point cloud data yet
        }
        
        tf::StampedTransform transform;
        try {
            tf_listener.lookupTransform("/odom", "/front_laser", ros::Time(0), transform);
        } catch (tf::TransformException& ex) {
            ROS_ERROR("%s", ex.what());
            return;
        }
        
        // Get transform components
        tf::Vector3 trans = transform.getOrigin();
        tf::Quaternion rot = transform.getRotation();
        
        // Create rotation matrix from quaternion
        tf::Matrix3x3 rot_matrix(rot);
        
        // Parse input point cloud
        sensor_msgs::PointCloud2ConstIterator<float> iter_x(point_cloud, "x");
        sensor_msgs::PointCloud2ConstIterator<float> iter_y(point_cloud, "y");
        sensor_msgs::PointCloud2ConstIterator<float> iter_z(point_cloud, "z");
        
        // Create output point cloud
        sensor_msgs::PointCloud2 output_cloud;
        output_cloud.header.frame_id = "odom";
        output_cloud.header.stamp = ros::Time::now();
        output_cloud.height = 1;
        output_cloud.is_dense = false;
        output_cloud.is_bigendian = false;
        
        sensor_msgs::PointCloud2Modifier modifier(output_cloud);
        modifier.setPointCloud2Fields(4,
            "x", 1, sensor_msgs::PointField::FLOAT32,
            "y", 1, sensor_msgs::PointField::FLOAT32,
            "z", 1, sensor_msgs::PointField::FLOAT32,
            "intensity", 1, sensor_msgs::PointField::FLOAT32);
        
        std::vector<float> points_data;
        int count = 0;
        
        for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z) {
            if (count % SCAN_SPACING == 0) {
                // Point in laser frame
                tf::Vector3 p_laser(*iter_x, *iter_y, *iter_z);
                
                // Transform to odom frame
                tf::Vector3 p_odom = transform * p_laser;
                
                // Calculate distance from robot
                double dist_sq = std::pow(p_odom.x() - trans.x(), 2) + 
                                std::pow(p_odom.y() - trans.y(), 2);
                
                if (dist_sq < DETECTED * DETECTED) {
                    points_data.push_back(p_odom.x());
                    points_data.push_back(p_odom.y());
                    points_data.push_back(p_odom.z());
                    points_data.push_back(1.0); // intensity
                }
            }
            count++;
        }
        
        // Resize and populate output cloud
        modifier.resize(points_data.size() / 4);
        output_cloud.width = points_data.size() / 4;
        
        sensor_msgs::PointCloud2Iterator<float> out_x(output_cloud, "x");
        sensor_msgs::PointCloud2Iterator<float> out_y(output_cloud, "y");
        sensor_msgs::PointCloud2Iterator<float> out_z(output_cloud, "z");
        sensor_msgs::PointCloud2Iterator<float> out_i(output_cloud, "intensity");
        
        for (size_t i = 0; i < points_data.size(); i += 4) {
            *out_x = points_data[i];
            *out_y = points_data[i + 1];
            *out_z = points_data[i + 2];
            *out_i = points_data[i + 3];
            ++out_x; ++out_y; ++out_z; ++out_i;
        }
        
        pub_point_cloud_odom.publish(output_cloud);
        ROS_INFO("Published %lu points", points_data.size() / 4);
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "laser_scan_to_point_cloud");
    ros::NodeHandle nh;
    
    ROS_INFO("Laser Scan to Point Cloud Node");
    
    LaserScanToPointCloud converter(nh);
    ros::Rate rate(20);
    
    while (ros::ok()) {
        ros::spinOnce();
        converter.run();
        rate.sleep();
    }
    
    return 0;
}