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
    bool bench_log_enabled_ = true;
    double bench_log_period_s_ = 1.0;

    void lidarCallback(const sensor_msgs::LaserScan::ConstPtr& scan);
};

LidarFilter::LidarFilter(ros::NodeHandle& nh, ros::NodeHandle& pnh) : nh_(nh) {
    pnh.param<std::string>("source_topic", source_topic_, "/front/scan");
    pnh.param<std::string>("pub_topic", pub_topic_, "/front/scan_filtered");
    pnh.param<double>("outlier_threshold", outlier_threshold_, 0.1);
    pnh.param<bool>("bench_log_enabled", bench_log_enabled_, true);
    pnh.param<double>("bench_log_period_s", bench_log_period_s_, 1.0);

    scan_pub_ = nh_.advertise<sensor_msgs::LaserScan>(pub_topic_, 10);
    scan_sub_ = nh_.subscribe<sensor_msgs::LaserScan>(
        source_topic_, 10, &LidarFilter::lidarCallback, this);

    ROS_INFO("[lidar_filter] %s -> %s threshold=%.3f m bench=%s period=%.2fs",
             source_topic_.c_str(), pub_topic_.c_str(), outlier_threshold_,
             bench_log_enabled_ ? "on" : "off", bench_log_period_s_);
}

void LidarFilter::lidarCallback(const sensor_msgs::LaserScan::ConstPtr& scan) {
    const auto t_start = ros::WallTime::now();
    const int n_ranges = static_cast<int>(scan->ranges.size());

    if (n_ranges < 3) {
        scan_pub_.publish(scan);
        if (bench_log_enabled_) {
            ROS_INFO_STREAM_THROTTLE(bench_log_period_s_,
                "[BENCH][lidar_filter] ranges=" << n_ranges
                << " filtered=0 kept=" << n_ranges << " (passthrough, n<3)");
        }
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

    int n_filtered = 0;
    int n_valid_interior = 0;

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
        ++n_valid_interior;

        if (std::abs(current_range - prev_range) > outlier_threshold_ &&
            std::abs(current_range - next_range) > outlier_threshold_) {
            filtered.ranges[i] = std::numeric_limits<float>::infinity();
            if (!filtered.intensities.empty() &&
                i < static_cast<int>(filtered.intensities.size())) {
                filtered.intensities[i] = 0.0f;
            }
            ++n_filtered;
        }
    }

    scan_pub_.publish(filtered);

    if (bench_log_enabled_) {
        int n_output_valid = 0;
        for (float r : filtered.ranges) {
            if (std::isfinite(r) && r >= filtered.range_min && r <= filtered.range_max) {
                ++n_output_valid;
            }
        }
        const auto t_end = ros::WallTime::now();
        ROS_INFO_STREAM_THROTTLE(bench_log_period_s_,
            "[BENCH][lidar_filter] total=" << (t_end - t_start).toSec() * 1e3 << " ms"
            << " | ranges=" << n_ranges
            << " interior_valid=" << n_valid_interior
            << " filtered=" << n_filtered
            << " output_valid=" << n_output_valid
            << " | filter_rate="
            << (n_valid_interior > 0
                    ? 100.0 * n_filtered / n_valid_interior
                    : 0.0)
            << "%");
    }
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "lidar_filter_node");

    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");

    LidarFilter lidar_filter(nh, pnh);

    ros::spin();
    return 0;
}
