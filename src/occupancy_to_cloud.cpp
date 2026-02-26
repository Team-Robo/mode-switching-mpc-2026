#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <nav_msgs/OccupancyGrid.h>
#include <tf/transform_listener.h>
#include <tf/transform_datatypes.h>
#include <cmath>
#include <vector>

class OccupancyToCloud {
private:
    const double BOX_HALFLENGTH = 1.0;
    const double PORTION_OF_PI = 3.0 / 4.0;
    const std::string TOPIC_LOCAL_MAP = "/move_base/local_costmap/costmap";
    const std::string TOPIC_MAP_CLOUD = "/map/cloud";

    nav_msgs::OccupancyGrid map;
    std::vector<std::vector<int8_t>> map_grid;
    double map_res;
    std::vector<double> map_origin;
    uint32_t last_map_seq_ = 0;
    bool map_updated_ = false;

    ros::Subscriber sub_map;
    ros::Publisher pub_point_cloud;
    tf::TransformListener tf_listener;

public:
    OccupancyToCloud(ros::NodeHandle& nh) {
        sub_map = nh.subscribe(TOPIC_LOCAL_MAP, 1, &OccupancyToCloud::callbackMap, this);
        pub_point_cloud = nh.advertise<sensor_msgs::PointCloud2>(TOPIC_MAP_CLOUD, 1);
        map_origin.resize(2);
    }

    void callbackMap(const nav_msgs::OccupancyGrid::ConstPtr& msg) {
        if (msg->header.seq == last_map_seq_) return;
        last_map_seq_ = msg->header.seq;
        map_updated_  = true;

        map = *msg;

        double map_Ox = msg->info.origin.position.x;
        double map_Oy = msg->info.origin.position.y;

        map_res = msg->info.resolution;
        int map_width  = msg->info.width;
        int map_height = msg->info.height;

        // Reshape flat data into column-major 2-D grid
        map_grid.assign(map_width, std::vector<int8_t>(map_height));
        for (int i = 0; i < map_width; ++i)
            for (int j = 0; j < map_height; ++j)
                map_grid[i][j] = msg->data[j * map_width + i];

        map_origin[0] = map_Ox;
        map_origin[1] = map_Oy;

        ROS_INFO_THROTTLE(2.0, "Map updated: %d x %d, res=%.3f, origin=(%.2f, %.2f)",
                          map_width, map_height, map_res, map_Ox, map_Oy);
    }

    double boundedAngle(double angle) {
        angle = fmod(angle, 2.0 * M_PI);
        if (angle >  M_PI) angle -= 2.0 * M_PI;
        if (angle < -M_PI) angle += 2.0 * M_PI;
        return angle;
    }

    void run() {
        if (map_grid.empty()) return;
        if (!map_updated_) return;
        map_updated_ = false;

        tf::StampedTransform transform;
        try {
            tf_listener.lookupTransform("/odom", "/base_link", ros::Time(0), transform);
        } catch (tf::TransformException& ex) {
            ROS_WARN_THROTTLE(1.0, "TF lookup failed: %s", ex.what());
            return;
        }

        double tx  = transform.getOrigin().x();
        double ty  = transform.getOrigin().y();
        double roll, pitch, yaw;
        tf::Matrix3x3(transform.getRotation()).getRPY(roll, pitch, yaw);

        double x_map_to_chassis = tx - map_origin[0];
        double y_map_to_chassis = ty - map_origin[1];

        double half = BOX_HALFLENGTH / 2.0;
        double tl_x = x_map_to_chassis - half,  tl_y = y_map_to_chassis - half;
        double br_x = x_map_to_chassis + half,  br_y = y_map_to_chassis + half;

        // Convert to cell indices
        int itlx = static_cast<int>(tl_x / map_res);
        int itly = static_cast<int>(tl_y / map_res);
        int ibrx = static_cast<int>(br_x / map_res);
        int ibry = static_cast<int>(br_y / map_res);

        // Clamp to grid extents
        int W = static_cast<int>(map_grid.size());
        int H = static_cast<int>(map_grid[0].size());
        itlx = std::max(0, std::min(itlx, W - 1));
        itly = std::max(0, std::min(itly, H - 1));
        ibrx = std::max(0, std::min(ibrx, W - 1));
        ibry = std::max(0, std::min(ibry, H - 1));

        // -----------------------------------------------------------------------
        // Angle bins
        // -----------------------------------------------------------------------
        constexpr double sparsity  = 0.05;                         // [rad] bin width
        const int num_angles = static_cast<int>(2.0 * M_PI / sparsity);  // ~125

        std::vector<double> points_dist(num_angles, std::numeric_limits<double>::infinity());
        std::vector<std::pair<float, float>> points_coord(num_angles, {0.0f, 0.0f});

        // Field-of-view bounds
        double half_fov = M_PI * PORTION_OF_PI;
        double L_bound  = boundedAngle( half_fov + yaw);
        double R_bound  = boundedAngle(-half_fov + yaw);

        // Local-frame centre of the bounding box
        int mid_i = (ibrx - itlx) / 2;
        int mid_j = (ibry - itly) / 2;

        // -----------------------------------------------------------------------
        // Main loop — O(cells), O(1) per cell
        // -----------------------------------------------------------------------
        for (int i = itlx; i < ibrx; ++i) {
            for (int j = itly; j < ibry; ++j) {

                if (map_grid[i][j] != 100) continue;   // only occupied cells

                int local_i = i - itlx;
                int local_j = j - itly;

                double di = local_i - mid_i;
                double dj = local_j - mid_j;

                double heading = atan2(dj, di);

                // ---------------------------------------------------------------
                // FOV check
                // ---------------------------------------------------------------
                bool is_bounded = false;
                if (L_bound >= 0 && R_bound >= 0) {
                    is_bounded = heading > L_bound && heading < R_bound;
                } else if (L_bound < 0 && R_bound >= 0) {
                    is_bounded = (heading > L_bound && heading < 0.0) ||
                                 (heading >= 0.0   && heading < R_bound);
                } else if (L_bound >= 0 && R_bound < 0) {
                    is_bounded = (heading > L_bound && heading <= M_PI) ||
                                 (heading >= -M_PI  && heading < R_bound);
                } else {
                    is_bounded = heading > L_bound && heading < R_bound;
                }
                if (!is_bounded) continue;

                int idx = static_cast<int>((heading + M_PI) / sparsity + 0.5);
                idx = std::max(0, std::min(idx, num_angles - 1));

                double dist = std::sqrt(di * di + dj * dj);
                if (dist >= points_dist[idx]) continue;   // not closer → skip

                points_dist[idx]  = dist;
                points_coord[idx] = {
                    static_cast<float>(local_i * map_res + tl_x + map_origin[0]),
                    static_cast<float>(local_j * map_res + tl_y + map_origin[1])
                };
            }
        }

        // -----------------------------------------------------------------------
        // Build PointCloud2
        // -----------------------------------------------------------------------
        sensor_msgs::PointCloud2 pointcloud;
        pointcloud.header.frame_id = "odom";
        pointcloud.header.stamp    = ros::Time::now();
        pointcloud.height          = 1;
        pointcloud.is_dense        = false;

        // Count valid bins first to avoid repeated push_back reallocations
        int valid_count = 0;
        for (int i = 0; i < num_angles; ++i)
            if (points_dist[i] != std::numeric_limits<double>::infinity()) ++valid_count;

        sensor_msgs::PointCloud2Modifier modifier(pointcloud);
        modifier.setPointCloud2Fields(4,
            "x",         1, sensor_msgs::PointField::FLOAT32,
            "y",         1, sensor_msgs::PointField::FLOAT32,
            "z",         1, sensor_msgs::PointField::FLOAT32,
            "intensity", 1, sensor_msgs::PointField::FLOAT32);
        modifier.resize(valid_count);
        pointcloud.width = valid_count;

        sensor_msgs::PointCloud2Iterator<float> out_x(pointcloud, "x");
        sensor_msgs::PointCloud2Iterator<float> out_y(pointcloud, "y");
        sensor_msgs::PointCloud2Iterator<float> out_z(pointcloud, "z");
        sensor_msgs::PointCloud2Iterator<float> out_i(pointcloud, "intensity");

        for (int i = 0; i < num_angles; ++i) {
            if (points_dist[i] == std::numeric_limits<double>::infinity()) continue;
            *out_x = points_coord[i].first;
            *out_y = points_coord[i].second;
            *out_z = 0.0f;
            *out_i = 1.0f;
            ++out_x; ++out_y; ++out_z; ++out_i;
        }

        pub_point_cloud.publish(pointcloud);
        ROS_INFO_THROTTLE(2.0, "Published %d map-cloud points", valid_count);
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