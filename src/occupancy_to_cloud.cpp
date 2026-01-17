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
        map = *msg;
        
        double map_Ox = msg->info.origin.position.x;
        double map_Oy = msg->info.origin.position.y;
        
        int occupied = 0, free = 0;
        for (const auto& val : msg->data) {
            if (val >= 90) occupied++;
            else if (val >= 0) free++;
        }
        
        ROS_INFO("Occupied: %d, Free: %d", occupied, free);
        ROS_INFO("Map Origin: %.2f %.2f wrt %s", map_Ox, map_Oy, msg->header.frame_id.c_str());
        
        map_res = msg->info.resolution;
        int map_width = msg->info.width;
        int map_height = msg->info.height;
        
        ROS_INFO("Map: %d x %d, Resolution: %.3f", map_width, map_height, map_res);
        
        // Reshape data into 2D grid
        map_grid.resize(map_width);
        for (int i = 0; i < map_width; i++) {
            map_grid[i].resize(map_height);
            for (int j = 0; j < map_height; j++) {
                map_grid[i][j] = msg->data[j * map_width + i];
            }
        }
        
        map_origin[0] = map_Ox;
        map_origin[1] = map_Oy;
    }
    
    double boundedAngle(double angle) {
        angle = fmod(angle, 2 * M_PI);
        if (angle > M_PI) {
            angle = -M_PI + (angle - M_PI);
        }
        if (angle < -M_PI) {
            angle = M_PI + (angle + M_PI);
        }
        return angle;
    }
    
    void run() {
        if (map_grid.empty()) {
            return;
        }
        
        tf::StampedTransform transform;
        try {
            tf_listener.lookupTransform("/odom", "/base_link", ros::Time(0), transform);
        } catch (tf::TransformException& ex) {
            ROS_ERROR("%s", ex.what());
            return;
        }
        
        double tx = transform.getOrigin().x();
        double ty = transform.getOrigin().y();
        
        double roll, pitch, yaw;
        tf::Matrix3x3(transform.getRotation()).getRPY(roll, pitch, yaw);
        
        ROS_INFO("Yaw: %.3f", yaw);
        
        double x_map_to_chassis = tx - map_origin[0];
        double y_map_to_chassis = ty - map_origin[1];
        
        std::vector<double> top_left_map = {
            x_map_to_chassis - BOX_HALFLENGTH / 2,
            y_map_to_chassis - BOX_HALFLENGTH / 2
        };
        std::vector<double> bot_right_map = {
            x_map_to_chassis + BOX_HALFLENGTH / 2,
            y_map_to_chassis + BOX_HALFLENGTH / 2
        };
        
        int index_top_left_x = static_cast<int>(top_left_map[0] / map_res);
        int index_top_left_y = static_cast<int>(top_left_map[1] / map_res);
        int index_bot_right_x = static_cast<int>(bot_right_map[0] / map_res);
        int index_bot_right_y = static_cast<int>(bot_right_map[1] / map_res);
        
        // Bounds checking
        index_top_left_x = std::max(0, std::min(index_top_left_x, (int)map_grid.size() - 1));
        index_top_left_y = std::max(0, std::min(index_top_left_y, (int)map_grid[0].size() - 1));
        index_bot_right_x = std::max(0, std::min(index_bot_right_x, (int)map_grid.size() - 1));
        index_bot_right_y = std::max(0, std::min(index_bot_right_y, (int)map_grid[0].size() - 1));
        
        // Field of view bounds
        double half_fov = M_PI * PORTION_OF_PI;
        double L_bound = boundedAngle(half_fov + yaw);
        double R_bound = boundedAngle(-half_fov + yaw);
        
        double sparsity = 0.05;
        int num_angles = static_cast<int>(2 * M_PI / sparsity);
        std::vector<double> points_angle(num_angles);
        std::vector<double> points_dist(num_angles, std::numeric_limits<double>::infinity());
        std::vector<std::pair<double, double>> points_coord(num_angles);
        
        for (int i = 0; i < num_angles; i++) {
            points_angle[i] = -M_PI + i * sparsity;
        }
        
        ROS_INFO("Points: %d", num_angles);
        
        int mid_i = (index_bot_right_x - index_top_left_x) / 2;
        int mid_j = (index_bot_right_y - index_top_left_y) / 2;
        
        for (int i = index_top_left_x; i < index_bot_right_x; i++) {
            for (int j = index_top_left_y; j < index_bot_right_y; j++) {
                int local_i = i - index_top_left_x;
                int local_j = j - index_top_left_y;
                
                double heading = atan2(local_j - mid_j, local_i - mid_i);
                
                // Find closest angle index
                int idx = 0;
                double min_diff = std::abs(points_angle[0] - heading);
                for (int k = 1; k < num_angles; k++) {
                    double diff = std::abs(points_angle[k] - heading);
                    if (diff < min_diff) {
                        min_diff = diff;
                        idx = k;
                    }
                }
                
                double dist = sqrt(pow(local_j - mid_j, 2) + pow(local_i - mid_i, 2));
                bool is_smaller = dist < points_dist[idx];
                
                // Check bounds
                bool is_bounded = false;
                if (L_bound >= 0 && R_bound >= 0) {
                    is_bounded = heading > L_bound && heading < R_bound;
                } else if (L_bound < 0 && R_bound >= 0) {
                    is_bounded = (heading > L_bound && heading < 0) || 
                                 (heading >= 0 && heading < R_bound);
                } else if (L_bound >= 0 && R_bound < 0) {
                    is_bounded = (heading > L_bound && heading <= M_PI) || 
                                 (heading >= -M_PI && heading < R_bound);
                } else {
                    is_bounded = heading > L_bound && heading < R_bound;
                }
                
                if (map_grid[i][j] == 100 && is_bounded && is_smaller) {
                    points_coord[idx].first = local_i * map_res + top_left_map[0] + map_origin[0];
                    points_coord[idx].second = local_j * map_res + top_left_map[1] + map_origin[1];
                    points_dist[idx] = dist;
                }
            }
        }
        
        // Create point cloud
        sensor_msgs::PointCloud2 pointcloud;
        pointcloud.header.frame_id = "odom";
        pointcloud.header.stamp = ros::Time::now();
        pointcloud.height = 1;
        pointcloud.is_dense = false;
        
        std::vector<float> points_data;
        for (int i = 0; i < num_angles; i++) {
            if (points_dist[i] != std::numeric_limits<double>::infinity()) {
                points_data.push_back(points_coord[i].first);
                points_data.push_back(points_coord[i].second);
                points_data.push_back(0.0);
                points_data.push_back(1.0);
            }
        }
        
        sensor_msgs::PointCloud2Modifier modifier(pointcloud);
        modifier.setPointCloud2Fields(4,
            "x", 1, sensor_msgs::PointField::FLOAT32,
            "y", 1, sensor_msgs::PointField::FLOAT32,
            "z", 1, sensor_msgs::PointField::FLOAT32,
            "intensity", 1, sensor_msgs::PointField::FLOAT32);
        
        modifier.resize(points_data.size() / 4);
        pointcloud.width = points_data.size() / 4;
        
        sensor_msgs::PointCloud2Iterator<float> iter_x(pointcloud, "x");
        sensor_msgs::PointCloud2Iterator<float> iter_y(pointcloud, "y");
        sensor_msgs::PointCloud2Iterator<float> iter_z(pointcloud, "z");
        sensor_msgs::PointCloud2Iterator<float> iter_i(pointcloud, "intensity");
        
        for (size_t i = 0; i < points_data.size(); i += 4) {
            *iter_x = points_data[i];
            *iter_y = points_data[i + 1];
            *iter_z = points_data[i + 2];
            *iter_i = points_data[i + 3];
            ++iter_x; ++iter_y; ++iter_z; ++iter_i;
        }
        
        pub_point_cloud.publish(pointcloud);
        ROS_INFO("Published %lu points", points_data.size() / 4);
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