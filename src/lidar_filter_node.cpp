// Neighbor-based laser scan outlier filter (algorithm from jie_ware / Waterplus lidar_filter_node).

#include <ros/ros.h>
#include <sensor_msgs/LaserScan.h>

#include <cmath>
#include <limits>
#include <string>

class LidarFilter {
public:
    LidarFilter(ros::NodeHandle& nh, ros::NodeHandle& pnh);

private:
    ros::NodeHandle nh_;
    ros::Publisher scan_pub_;
    ros::Subscriber scan_sub_;

    std::string source_topic_;
    std::string pub_topic_;
    double outlier_threshold_;

    void lidarCallback(const sensor_msgs::LaserScan::ConstPtr& scan);
};

LidarFilter::LidarFilter(ros::NodeHandle& nh, ros::NodeHandle& pnh) : nh_(nh) {
    pnh.param<std::string>("source_topic", source_topic_, "/front/scan");
    pnh.param<std::string>("pub_topic", pub_topic_, "/front/scan_filtered");
    pnh.param<double>("outlier_threshold", outlier_threshold_, 0.1);

    scan_pub_ = nh_.advertise<sensor_msgs::LaserScan>(pub_topic_, 10);
    scan_sub_ = nh_.subscribe<sensor_msgs::LaserScan>(
        source_topic_, 10, &LidarFilter::lidarCallback, this);

    ROS_INFO("[lidar_filter] %s -> %s (outlier_threshold=%.3f m)",
             source_topic_.c_str(), pub_topic_.c_str(), outlier_threshold_);
}

void LidarFilter::lidarCallback(const sensor_msgs::LaserScan::ConstPtr& scan) {
    const int n_ranges = static_cast<int>(scan->ranges.size());

    if (n_ranges < 3) {
        scan_pub_.publish(scan);
        return;
    }

    sensor_msgs::LaserScan filtered;
    filtered.header = scan->header;
    filtered.angle_min = scan->angle_min;
    filtered.angle_max = scan->angle_max;
    filtered.angle_increment = scan->angle_increment;
    filtered.time_increment = scan->time_increment;
    filtered.scan_time = scan->scan_time;
    filtered.range_min = scan->range_min;
    filtered.range_max = scan->range_max;
    filtered.ranges = scan->ranges;
    if (!scan->intensities.empty()) {
        filtered.intensities = scan->intensities;
    }

    for (int i = 1; i < n_ranges - 1; ++i) {
        const float prev_range = filtered.ranges[i - 1];
        const float current_range = filtered.ranges[i];
        const float next_range = filtered.ranges[i + 1];

        const bool current_valid = std::isfinite(current_range) &&
                                   current_range >= filtered.range_min &&
                                   current_range <= filtered.range_max;

        if (!current_valid) {
            continue;
        }

        if (std::abs(current_range - prev_range) > outlier_threshold_ &&
            std::abs(current_range - next_range) > outlier_threshold_) {
            filtered.ranges[i] = std::numeric_limits<float>::infinity();
            if (!filtered.intensities.empty() &&
                i < static_cast<int>(filtered.intensities.size())) {
                filtered.intensities[i] = 0.0f;
            }
        }
    }

    scan_pub_.publish(filtered);
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "lidar_filter_node");

    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");

    LidarFilter lidar_filter(nh, pnh);

    ros::spin();
    return 0;
}
