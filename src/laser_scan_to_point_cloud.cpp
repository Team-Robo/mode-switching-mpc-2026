#include <ros/ros.h>
#include <sensor_msgs/LaserScan.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <laser_geometry/laser_geometry.h>
#include <tf/transform_listener.h>
#include <cmath>
#include <mutex>

class LaserScanToPointCloud {
private:
    const int SCAN_SPACING = 15;
    const std::string TOPIC_LASER_SCAN = "/front/scan";
    const std::string TOPIC_POINT_CLOUD_LASER = "/front/laser/cloud";
    const std::string TOPIC_POINT_CLOUD_ODOM = "/front/odom/cloud";
    const double TF_TIMEOUT = 0.1; // 100ms timeout for TF lookups
    
    sensor_msgs::LaserScan laser_scan;
    sensor_msgs::PointCloud2 point_cloud;
    laser_geometry::LaserProjection laser_projector;
    
    ros::Subscriber sub_laser_scan;
    ros::Publisher pub_point_cloud_odom;
    
    tf::TransformListener tf_listener;
    std::mutex cloud_mutex_;
    bool has_new_data_;

    bool bench_log_enabled_ = true;
    double bench_log_period_s_ = 1.0;
    
public:
    LaserScanToPointCloud(ros::NodeHandle& nh) : has_new_data_(false) {
        sub_laser_scan = nh.subscribe(TOPIC_LASER_SCAN, 1, 
                                      &LaserScanToPointCloud::callbackLaserScan, this);
        pub_point_cloud_odom = nh.advertise<sensor_msgs::PointCloud2>(
                                      TOPIC_POINT_CLOUD_ODOM, 1);

        ros::NodeHandle nh_private("~");
        nh_private.param<bool>("bench_log_enabled", bench_log_enabled_, true);
        nh_private.param<double>("bench_log_period_s", bench_log_period_s_, 1.0);
        ROS_INFO("[LaserScanToPointCloud] bench logging: enabled=%s period=%.2fs",
                 bench_log_enabled_ ? "true" : "false", bench_log_period_s_);
    }
    
    void callbackLaserScan(const sensor_msgs::LaserScan::ConstPtr& msg) {
        std::lock_guard<std::mutex> lock(cloud_mutex_);
        const auto t_cb_start = ros::WallTime::now();
        sensor_msgs::LaserScan filtered = *msg;
        const float max_range = filtered.range_max;
        for (float& r : filtered.ranges) {
            if (!std::isfinite(r) || r < filtered.range_min) {
                r = max_range;
            }
        }
        const auto t_after_filter = ros::WallTime::now();

        laser_scan = filtered;
        laser_projector.projectLaser(filtered, point_cloud);
        const auto t_after_project = ros::WallTime::now();
        has_new_data_ = true;

        if (bench_log_enabled_) {
            ROS_INFO_STREAM_THROTTLE(bench_log_period_s_,
                "[BENCH][laser_scan_to_point_cloud][callback] total="
                << (t_after_project - t_cb_start).toSec() * 1e3 << " ms"
                << " | sanitize_ranges=" << (t_after_filter - t_cb_start).toSec() * 1e3 << " ms"
                << " | project_laser=" << (t_after_project - t_after_filter).toSec() * 1e3 << " ms"
                << " | ranges=" << filtered.ranges.size());
        }
    }
    
    void run() {
        std::lock_guard<std::mutex> lock(cloud_mutex_);
        const auto t_start = ros::WallTime::now();
        
        // Only process if we have new data
        if (!has_new_data_ || point_cloud.width == 0) {
            return;
        }

        ros::WallTime t_after_tf = t_start;
        ros::WallTime t_after_transform_loop = t_start;
        ros::WallTime t_after_pack = t_start;
        
        tf::StampedTransform transform;
        
        try {
            // Strategy 1: Try the exact timestamp first
            tf_listener.lookupTransform("/odom", "/front_laser", 
                                       point_cloud.header.stamp, transform);
        } catch (tf::TransformException& ex) {
            try {
                // Strategy 2: Use the latest available transform (time 0)
                // This is more robust and prevents extrapolation errors
                tf_listener.lookupTransform("/odom", "/front_laser", 
                                           ros::Time(0), transform);
                
                // Update the point cloud timestamp to match the transform
                // This keeps things consistent
                point_cloud.header.stamp = transform.stamp_;
                
            } catch (tf::TransformException& ex2) {
                // Only log occasionally to avoid spam
                ROS_WARN_THROTTLE(2.0, "TF lookup failed: %s", ex2.what());
                return;
            }
        }
        t_after_tf = ros::WallTime::now();
        
        // Parse input point cloud
        sensor_msgs::PointCloud2ConstIterator<float> iter_x(point_cloud, "x");
        sensor_msgs::PointCloud2ConstIterator<float> iter_y(point_cloud, "y");
        sensor_msgs::PointCloud2ConstIterator<float> iter_z(point_cloud, "z");
        
        // Create output point cloud
        sensor_msgs::PointCloud2 output_cloud;
        output_cloud.header.frame_id = "odom";
        output_cloud.header.stamp = point_cloud.header.stamp;
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
                // Safety net: skip any residual NaN / inf after the scan filter
                if (!std::isfinite(*iter_x) || !std::isfinite(*iter_y) || !std::isfinite(*iter_z)) {
                    count++;
                    continue;
                }

                // Point in laser frame
                tf::Vector3 p_laser(*iter_x, *iter_y, *iter_z);
                
                // Transform to odom frame
                tf::Vector3 p_odom = transform * p_laser;
                
                points_data.push_back(p_odom.x());
                points_data.push_back(p_odom.y());
                points_data.push_back(p_odom.z());
                points_data.push_back(1.0); // intensity
            }
            count++;
        }
        t_after_transform_loop = ros::WallTime::now();
        
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
        t_after_pack = ros::WallTime::now();
        
        pub_point_cloud_odom.publish(output_cloud);
        has_new_data_ = false; // Mark as processed
        
        ROS_INFO_THROTTLE(2.0, "Published %lu points", points_data.size() / 4);
        if (bench_log_enabled_) {
            const auto t_end = ros::WallTime::now();
            ROS_INFO_STREAM_THROTTLE(bench_log_period_s_,
                "[BENCH][laser_scan_to_point_cloud][run] total=" << (t_end - t_start).toSec() * 1e3 << " ms"
                << " | tf_lookup=" << (t_after_tf - t_start).toSec() * 1e3 << " ms"
                << " | transform_and_sample=" << (t_after_transform_loop - t_after_tf).toSec() * 1e3 << " ms"
                << " | pack_cloud=" << (t_after_pack - t_after_transform_loop).toSec() * 1e3 << " ms"
                << " | publish=" << (t_end - t_after_pack).toSec() * 1e3 << " ms"
                << " | input_points=" << point_cloud.width
                << " | output_points=" << output_cloud.width);
        }
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "laser_scan_to_point_cloud");
    ros::NodeHandle nh;
    
    ROS_INFO("Laser Scan to Point Cloud Node");
    
    LaserScanToPointCloud converter(nh);
    
    // Use a lower rate or make it event-driven
    ros::Rate rate(20);
    
    while (ros::ok()) {
        ros::spinOnce();
        converter.run();
        rate.sleep();
    }
    
    return 0;
}