#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <nav_msgs/OccupancyGrid.h>
#include <visualization_msgs/Marker.h>
#include <geometry_msgs/Point.h>
#include <tf/transform_listener.h>
#include <tf/transform_datatypes.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

class OccupancyToCloud {
private:
    const double BOX_LENGTH = 1.0;  // full side of local search bbox around the robot [m]
    /** Half-FOV in rad: M_PI * (3/4) => 135 deg half-angle, 270 deg total (Jackal lidar). */
    const double PORTION_OF_PI = 3.0 / 4.0;
    const std::string TOPIC_LOCAL_MAP = "/move_base/local_costmap/costmap";
    const std::string TOPIC_MAP_CLOUD = "/map/cloud";

    std::vector<int8_t> map_data_;
    int map_width_ = 0;
    int map_height_ = 0;
    double map_res_ = 0.05;
    double map_origin_x_ = 0.0;
    double map_origin_y_ = 0.0;
    std::string map_frame_id_;
    ros::Time map_stamp_;

    uint32_t last_map_seq_ = 0;
    ros::Time last_map_stamp_;
    bool have_map_stamp_ = false;
    bool map_updated_ = false;

    bool bench_log_enabled_ = true;
    double bench_log_period_s_ = 1.0;
    bool debug_enabled_ = false;
    double debug_log_period_s_ = 1.0;
    bool publish_bbox_marker_ = true;

    std::string base_frame_ = "base_link";
    /** If non-empty, publish cloud in this frame (TF from costmap frame). Empty = costmap frame. */
    std::string output_frame_;

    ros::Subscriber sub_map;
    ros::Publisher pub_point_cloud;
    ros::Publisher pub_bbox_marker_;
    tf::TransformListener tf_listener_;

    static void addRectLineStrip(visualization_msgs::Marker& m,
                               double z,
                               double x0, double y0,
                               double x1, double y1) {
        geometry_msgs::Point pt;
        pt.z = z;
        pt.x = x0; pt.y = y0; m.points.push_back(pt);
        pt.x = x1; pt.y = y0; m.points.push_back(pt);
        pt.x = x1; pt.y = y1; m.points.push_back(pt);
        pt.x = x0; pt.y = y1; m.points.push_back(pt);
        pt.x = x0; pt.y = y0; m.points.push_back(pt);
    }

    void publishBboxMarkers(double tx, double ty,
                            int itlx, int itly, int ibrx, int ibry) const {
        if (!publish_bbox_marker_) return;

        const double z = 0.05;
        const double half = BOX_LENGTH / 2.0;

        visualization_msgs::Marker search_box;
        search_box.header.frame_id = map_frame_id_;
        search_box.header.stamp = map_stamp_;
        search_box.ns = "map_to_cloud";
        search_box.id = 0;
        search_box.type = visualization_msgs::Marker::LINE_STRIP;
        search_box.action = visualization_msgs::Marker::ADD;
        search_box.pose.orientation.w = 1.0;
        search_box.scale.x = 0.03;
        search_box.color.r = 0.0f;
        search_box.color.g = 1.0f;
        search_box.color.b = 0.0f;
        search_box.color.a = 1.0f;
        addRectLineStrip(search_box, z, tx - half, ty - half, tx + half, ty + half);
        pub_bbox_marker_.publish(search_box);

        visualization_msgs::Marker cell_box = search_box;
        cell_box.id = 1;
        cell_box.points.clear();
        cell_box.color.r = 1.0f;
        cell_box.color.g = 1.0f;
        cell_box.color.b = 0.0f;
        const double cx0 = map_origin_x_ + itlx * map_res_;
        const double cy0 = map_origin_y_ + itly * map_res_;
        const double cx1 = map_origin_x_ + ibrx * map_res_;
        const double cy1 = map_origin_y_ + ibry * map_res_;
        addRectLineStrip(cell_box, z, cx0, cy0, cx1, cy1);
        pub_bbox_marker_.publish(cell_box);
    }

    static int clampInt(int v, int lo, int hi) {
        return std::max(lo, std::min(v, hi));
    }

    bool lookupBaseInMapFrame(tf::StampedTransform& out) const {
        ros::Time query_time = map_stamp_;
        if (query_time.isZero()) {
            query_time = ros::Time(0);
        }
        try {
            tf_listener_.lookupTransform(
                map_frame_id_, base_frame_, query_time, out);
            return true;
        } catch (const tf::TransformException& ex_stamp) {
            if (query_time == ros::Time(0)) {
                ROS_WARN_THROTTLE(1.0,
                    "[map_to_cloud] TF failed (%s -> %s): %s",
                    base_frame_.c_str(), map_frame_id_.c_str(), ex_stamp.what());
                return false;
            }
            ROS_WARN_THROTTLE(1.0,
                "[map_to_cloud] TF at map stamp failed (%s), retrying latest: %s",
                map_frame_id_.c_str(), ex_stamp.what());
            try {
                tf_listener_.lookupTransform(
                    map_frame_id_, base_frame_, ros::Time(0), out);
                return true;
            } catch (const tf::TransformException& ex_latest) {
                ROS_WARN_THROTTLE(1.0,
                    "[map_to_cloud] TF failed (%s -> %s): %s",
                    base_frame_.c_str(), map_frame_id_.c_str(), ex_latest.what());
                return false;
            }
        }
    }

    bool lookupMapToOutputFrame(tf::StampedTransform& tf_map_to_out) const {
        try {
            tf_listener_.lookupTransform(
                output_frame_, map_frame_id_, map_stamp_, tf_map_to_out);
            return true;
        } catch (const tf::TransformException& ex_stamp) {
            ROS_WARN_THROTTLE(1.0,
                "[map_to_cloud] output TF at map stamp failed, retrying latest: %s",
                ex_stamp.what());
            try {
                tf_listener_.lookupTransform(
                    output_frame_, map_frame_id_, ros::Time(0), tf_map_to_out);
                return true;
            } catch (const tf::TransformException& ex_latest) {
                ROS_WARN_THROTTLE(1.0,
                    "[map_to_cloud] output TF failed (%s -> %s): %s",
                    map_frame_id_.c_str(), output_frame_.c_str(), ex_latest.what());
                return false;
            }
        }
    }

public:
    OccupancyToCloud(ros::NodeHandle& nh)
        : tf_listener_(ros::Duration(10.0)) {
        sub_map = nh.subscribe(TOPIC_LOCAL_MAP, 1, &OccupancyToCloud::callbackMap, this);
        pub_point_cloud = nh.advertise<sensor_msgs::PointCloud2>(TOPIC_MAP_CLOUD, 1);
        pub_bbox_marker_ = nh.advertise<visualization_msgs::Marker>("bbox", 1);

        ros::NodeHandle nh_private("~");
        nh_private.param<bool>("bench_log_enabled", bench_log_enabled_, true);
        nh_private.param<double>("bench_log_period_s", bench_log_period_s_, 1.0);
        nh_private.param<bool>("debug_enabled", debug_enabled_, false);
        nh_private.param<double>("debug_log_period_s", debug_log_period_s_, 1.0);
        nh_private.param<std::string>("base_frame", base_frame_, std::string("base_link"));
        nh_private.param<std::string>("output_frame", output_frame_, std::string());
        nh_private.param<bool>("publish_bbox_marker", publish_bbox_marker_, true);
        ROS_INFO("[map_to_cloud] bench=%s debug=%s base=%s bbox_marker=%s output_frame=%s",
                 bench_log_enabled_ ? "on" : "off",
                 debug_enabled_ ? "on" : "off",
                 base_frame_.c_str(),
                 publish_bbox_marker_ ? "on" : "off",
                 output_frame_.empty() ? "(costmap frame)" : output_frame_.c_str());
    }

    void callbackMap(const nav_msgs::OccupancyGrid::ConstPtr& msg) {
        const auto t_cb_start = ros::WallTime::now();

        // Skip only when both stamp and seq repeat (seq alone is often stuck at 0).
        if (have_map_stamp_ &&
            msg->header.stamp == last_map_stamp_ &&
            msg->header.seq == last_map_seq_) {
            if (debug_enabled_) {
                ROS_INFO_THROTTLE(debug_log_period_s_,
                    "[DEBUG][map_to_cloud] callbackMap: skip duplicate stamp=%.3f seq=%u",
                    msg->header.stamp.toSec(), msg->header.seq);
            }
            return;
        }

        have_map_stamp_ = true;
        last_map_stamp_ = msg->header.stamp;
        last_map_seq_   = msg->header.seq;
        map_updated_    = true;

        map_frame_id_ = msg->header.frame_id;
        map_stamp_    = msg->header.stamp;
        map_res_      = msg->info.resolution;
        map_width_    = static_cast<int>(msg->info.width);
        map_height_   = static_cast<int>(msg->info.height);
        map_origin_x_ = msg->info.origin.position.x;
        map_origin_y_ = msg->info.origin.position.y;
        map_data_     = msg->data;

        if (debug_enabled_) {
            ROS_INFO("[DEBUG][map_to_cloud] callbackMap: NEW %dx%d res=%.3f origin=(%.2f,%.2f) "
                     "frame=%s stamp=%.3f data=%zu",
                     map_width_, map_height_, map_res_,
                     map_origin_x_, map_origin_y_, map_frame_id_.c_str(),
                     map_stamp_.toSec(), map_data_.size());
        } else {
            ROS_INFO_THROTTLE(2.0, "Map updated: %dx%d frame=%s",
                              map_width_, map_height_, map_frame_id_.c_str());
        }

        if (bench_log_enabled_) {
            const auto t_cb_end = ros::WallTime::now();
            ROS_INFO_STREAM_THROTTLE(bench_log_period_s_,
                "[BENCH][map_to_cloud][callbackMap] total="
                << (t_cb_end - t_cb_start).toSec() * 1e3 << " ms"
                << " | flat_copy=" << map_data_.size() << " cells");
        }
    }

    static double boundedAngle(double angle) {
        angle = std::fmod(angle, 2.0 * M_PI);
        if (angle > M_PI) angle -= 2.0 * M_PI;
        if (angle < -M_PI) angle += 2.0 * M_PI;
        return angle;
    }

    void run() {
        const auto t_start = ros::WallTime::now();
        if (map_data_.empty() || map_width_ <= 0 || map_height_ <= 0) {
            if (debug_enabled_) {
                ROS_WARN_THROTTLE(debug_log_period_s_,
                    "[DEBUG][map_to_cloud] run: skip - no costmap yet (sub=%s)",
                    TOPIC_LOCAL_MAP.c_str());
            }
            return;
        }
        if (!map_updated_) {
            if (debug_enabled_) {
                ROS_INFO_THROTTLE(debug_log_period_s_,
                    "[DEBUG][map_to_cloud] run: skip - no new map (stamp=%.3f seq=%u)",
                    last_map_stamp_.toSec(), last_map_seq_);
            }
            return;
        }
        if (map_frame_id_.empty()) {
            ROS_WARN_THROTTLE(2.0, "[map_to_cloud] run: skip - costmap frame_id empty");
            return;
        }

        tf::StampedTransform tf_map_to_base;
        if (!lookupBaseInMapFrame(tf_map_to_base)) {
            if (debug_enabled_) {
                ROS_WARN_THROTTLE(debug_log_period_s_,
                    "[DEBUG][map_to_cloud] run: abort - TF %s->%s",
                    base_frame_.c_str(), map_frame_id_.c_str());
            }
            return;
        }

        const ros::WallTime t_after_tf = ros::WallTime::now();

        const double tx = tf_map_to_base.getOrigin().x();
        const double ty = tf_map_to_base.getOrigin().y();
        double roll = 0.0, pitch = 0.0, yaw = 0.0;
        tf::Matrix3x3(tf_map_to_base.getRotation()).getRPY(roll, pitch, yaw);

        // Grid indices: (world - map_origin) / resolution
        const double mx = tx - map_origin_x_;
        const double my = ty - map_origin_y_;
        const double half = BOX_LENGTH / 2.0;
        const double tl_x = mx - half;
        const double tl_y = my - half;
        const double br_x = mx + half;
        const double br_y = my + half;

        int itlx = static_cast<int>(std::floor(tl_x / map_res_));
        int itly = static_cast<int>(std::floor(tl_y / map_res_));
        int ibrx = static_cast<int>(std::ceil(br_x / map_res_));
        int ibry = static_cast<int>(std::ceil(br_y / map_res_));

        const int W = map_width_;
        const int H = map_height_;
        itlx = clampInt(itlx, 0, W - 1);
        itly = clampInt(itly, 0, H - 1);
        ibrx = clampInt(ibrx, itlx + 1, W);
        ibry = clampInt(ibry, itly + 1, H);

        publishBboxMarkers(tx, ty, itlx, itly, ibrx, ibry);

        const ros::WallTime t_after_bbox = ros::WallTime::now();

        constexpr double sparsity = 0.05;
        const int num_angles = static_cast<int>(2.0 * M_PI / sparsity);

        std::vector<double> points_dist(
            num_angles, std::numeric_limits<double>::infinity());
        std::vector<std::pair<float, float>> points_coord(num_angles, {0.0f, 0.0f});

        const double half_fov = M_PI * PORTION_OF_PI;

        int cells_in_bbox = 0;
        int cells_lethal = 0;
        int cells_in_fov = 0;

        for (int j = itly; j < ibry; ++j) {
            const int row_base = j * W;
            for (int i = itlx; i < ibrx; ++i) {
                ++cells_in_bbox;
                const int8_t cost = map_data_[row_base + i];
                if (cost != 100) continue;
                ++cells_lethal;

                const double wx = map_origin_x_ + (i + 0.5) * map_res_;
                const double wy = map_origin_y_ + (j + 0.5) * map_res_;
                const double dx = wx - tx;
                const double dy = wy - ty;

                const double heading = std::atan2(dy, dx);
                const double rel_heading = boundedAngle(heading - yaw);
                if (std::abs(rel_heading) > half_fov) continue;
                ++cells_in_fov;

                const double wrapped = rel_heading + M_PI;
                int idx = static_cast<int>(std::floor(wrapped / sparsity));
                idx = clampInt(idx, 0, num_angles - 1);

                const double dist_sq = dx * dx + dy * dy;
                if (dist_sq >= points_dist[idx]) continue;

                points_dist[idx] = dist_sq;
                points_coord[idx] = {
                    static_cast<float>(wx),
                    static_cast<float>(wy)
                };
            }
        }

        const ros::WallTime t_after_bin_loop = ros::WallTime::now();

        const std::string& cloud_frame =
            output_frame_.empty() ? map_frame_id_ : output_frame_;

        tf::StampedTransform tf_map_to_out;
        const bool transform_output =
            !output_frame_.empty() && output_frame_ != map_frame_id_;
        if (transform_output && !lookupMapToOutputFrame(tf_map_to_out)) {
            return;  // map_updated_ still true
        }

        sensor_msgs::PointCloud2 pointcloud;
        pointcloud.header.frame_id = cloud_frame;
        pointcloud.header.stamp = map_stamp_;
        pointcloud.height = 1;
        pointcloud.is_dense = false;

        int valid_count = 0;
        for (int k = 0; k < num_angles; ++k) {
            if (points_dist[k] != std::numeric_limits<double>::infinity()) {
                ++valid_count;
            }
        }

        sensor_msgs::PointCloud2Modifier modifier(pointcloud);
        modifier.setPointCloud2Fields(4,
            "x", 1, sensor_msgs::PointField::FLOAT32,
            "y", 1, sensor_msgs::PointField::FLOAT32,
            "z", 1, sensor_msgs::PointField::FLOAT32,
            "intensity", 1, sensor_msgs::PointField::FLOAT32);
        modifier.resize(valid_count);
        pointcloud.width = valid_count;

        sensor_msgs::PointCloud2Iterator<float> out_x(pointcloud, "x");
        sensor_msgs::PointCloud2Iterator<float> out_y(pointcloud, "y");
        sensor_msgs::PointCloud2Iterator<float> out_z(pointcloud, "z");
        sensor_msgs::PointCloud2Iterator<float> out_i(pointcloud, "intensity");

        for (int k = 0; k < num_angles; ++k) {
            if (points_dist[k] == std::numeric_limits<double>::infinity()) continue;

            float px = points_coord[k].first;
            float py = points_coord[k].second;
            if (transform_output) {
                tf::Vector3 p_map(px, py, 0.0);
                tf::Vector3 p_out = tf_map_to_out * p_map;
                px = static_cast<float>(p_out.x());
                py = static_cast<float>(p_out.y());
            }

            *out_x = px;
            *out_y = py;
            *out_z = 0.0f;
            *out_i = 1.0f;
            ++out_x;
            ++out_y;
            ++out_z;
            ++out_i;
        }

        const ros::WallTime t_after_pack = ros::WallTime::now();

        pub_point_cloud.publish(pointcloud);
        map_updated_ = false;

        if (debug_enabled_) {
            ROS_INFO_THROTTLE(debug_log_period_s_,
                "[DEBUG][map_to_cloud] run: published %d pts frame=%s | robot=(%.2f,%.2f) "
                "yaw=%.2f bbox [%d,%d)x[%d,%d) lethal=%d in_fov=%d",
                valid_count, cloud_frame.c_str(), tx, ty, yaw,
                itlx, ibrx, itly, ibry, cells_lethal, cells_in_fov);
        } else {
            ROS_INFO_THROTTLE(2.0, "Published %d map-cloud points (%s)",
                              valid_count, cloud_frame.c_str());
        }

        if (bench_log_enabled_) {
            const auto t_end = ros::WallTime::now();
            ROS_INFO_STREAM_THROTTLE(bench_log_period_s_,
                "[BENCH][map_to_cloud][run] total=" << (t_end - t_start).toSec() * 1e3 << " ms"
                << " | tf=" << (t_after_tf - t_start).toSec() * 1e3 << " ms"
                << " | bbox=" << (t_after_bbox - t_after_tf).toSec() * 1e3 << " ms"
                << " | loop=" << (t_after_bin_loop - t_after_bbox).toSec() * 1e3 << " ms"
                << " | pack=" << (t_after_pack - t_after_bin_loop).toSec() * 1e3 << " ms"
                << " | valid=" << valid_count);
        }
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "map_to_cloud");
    ros::NodeHandle nh;
    ROS_INFO("Map to Cloud Node");

    OccupancyToCloud converter(nh);
    ros::Rate rate(20);

    while (ros::ok()) {
        ros::spinOnce();
        converter.run();
        rate.sleep();
    }
    return 0;
}
