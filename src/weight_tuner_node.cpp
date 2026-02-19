/*
 * weight_tuner_node.cpp
 * =====================
 * C++ SAC Weight Tuner for MPC — processes sensor data into RL observations,
 * validates/clamps weight updates, and (optionally) runs ONNX inference.
 *
 * Two runtime modes (set via ~mode param):
 *   "bridge" (default) — receives weights from Python via /rl/weight_update,
 *                         validates, forwards to /mpc/weight_update.
 *   "onnx"             — loads an ONNX actor model and directly computes
 *                         weights from observations.  Requires building with
 *                         -DUSE_ONNX_RUNTIME=ON and linking onnxruntime.
 *
 * Topics published:
 *   /mpc/weight_update   (std_msgs/Float64MultiArray) — validated weights → MPC
 *   /rl/observations      (std_msgs/Float64MultiArray) — observation vector
 *   /rl/tuner_status      (std_msgs/String)            — JSON status
 *
 * Topics subscribed:
 *   /move_base/local_costmap/costmap  (nav_msgs/OccupancyGrid)
 *   /front/scan                        (sensor_msgs/LaserScan)
 *   /odometry/filtered                 (nav_msgs/Odometry)
 *   /rl/weight_update                  (std_msgs/Float64MultiArray) [bridge]
 */

#include <ros/ros.h>
#include <std_msgs/Float64MultiArray.h>
#include <std_msgs/String.h>
#include <nav_msgs/OccupancyGrid.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/LaserScan.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <mutex>
#include <sstream>
#include <vector>

#ifdef USE_ONNX_RUNTIME
#include <onnxruntime_cxx_api.h>
#endif

namespace sac_weight_tuner {

// ── Observation dimensions (must match Python barn_env.py) ───────────────
static constexpr int COSTMAP_SIZE  = 10;
static constexpr int N_LIDAR_RAYS  = 36;
static constexpr double LIDAR_MAX  = 10.0;
static constexpr double LIDAR_MIN  = 0.5;
static constexpr int N_ROBOT_STATE = 5;
static constexpr int N_WEIGHTS     = 4;
static constexpr int OBS_DIM       = COSTMAP_SIZE * COSTMAP_SIZE
                                    + N_LIDAR_RAYS + N_ROBOT_STATE + N_WEIGHTS;

// ── Weight bounds ────────────────────────────────────────────────────────
static constexpr double W_POS_MIN   = 1.0,    W_POS_MAX   = 100.0;
static constexpr double W_HEAD_MIN  = 1.0,    W_HEAD_MAX  = 100.0;
static constexpr double W_VEL_MIN   = 0.1,    W_VEL_MAX   = 50.0;
static constexpr double W_ACCEL_MIN = 0.0001, W_ACCEL_MAX = 1.0;

// =========================================================================
class WeightTunerNode {
public:
    WeightTunerNode(ros::NodeHandle& nh, ros::NodeHandle& nh_priv)
        : nh_(nh), nhp_(nh_priv)
    {
        // Parameters
        nhp_.param<std::string>("mode", mode_, "bridge");
        nhp_.param<double>("update_rate", update_rate_, 1.0);
        nhp_.param<double>("goal_x",  goal_x_,  0.0);
        nhp_.param<double>("goal_y",  goal_y_,  10.0);
        nhp_.param<double>("init_x",  init_x_, -2.25);
        nhp_.param<double>("init_y",  init_y_,  3.0);

        // Publishers
        pub_mpc_weights_  = nh_.advertise<std_msgs::Float64MultiArray>(
                                "/mpc/weight_update", 1, /*latch=*/true);
        pub_observations_ = nh_.advertise<std_msgs::Float64MultiArray>(
                                "/rl/observations", 1);
        pub_status_       = nh_.advertise<std_msgs::String>(
                                "/rl/tuner_status", 1);

        // Subscribers
        sub_costmap_ = nh_.subscribe("/move_base/local_costmap/costmap", 1,
                                     &WeightTunerNode::cbCostmap, this);
        sub_lidar_   = nh_.subscribe("/front/scan", 1,
                                     &WeightTunerNode::cbLidar, this);
        sub_odom_    = nh_.subscribe("/odometry/filtered", 1,
                                     &WeightTunerNode::cbOdom, this);
        sub_rl_      = nh_.subscribe("/rl/weight_update", 1,
                                     &WeightTunerNode::cbRLWeights, this);

        // Default weights
        weights_ = {{49.0, 37.0, 10.0, 0.0021}};

#ifdef USE_ONNX_RUNTIME
        if (mode_ == "onnx") {
            std::string model_path;
            nhp_.param<std::string>("onnx_model_path", model_path, "");
            if (!model_path.empty()) {
                loadOnnx(model_path);
            } else {
                ROS_ERROR("mode=onnx but ~onnx_model_path is empty; falling back to bridge");
                mode_ = "bridge";
            }
        }
#else
        if (mode_ == "onnx") {
            ROS_WARN("Compiled without USE_ONNX_RUNTIME — falling back to bridge mode");
            mode_ = "bridge";
        }
#endif

        ROS_INFO("WeightTunerNode: mode=%s  rate=%.1f Hz", mode_.c_str(), update_rate_);
    }

    /* Main loop */
    void spin() {
        ros::Rate rate(update_rate_);
        while (ros::ok()) {
            ros::spinOnce();

            // Compute & publish observations for monitoring / Python RL
            auto obs = computeObservations();
            publishObservations(obs);

            if (mode_ == "onnx") {
#ifdef USE_ONNX_RUNTIME
                auto w = runOnnxInference(obs);
                weights_ = clamp(w);
                publishWeights(weights_);
#endif
            }
            // In bridge mode, weights are forwarded inside cbRLWeights

            publishStatus();
            rate.sleep();
        }
    }

private:
    ros::NodeHandle nh_, nhp_;
    std::string mode_;
    double update_rate_;
    double goal_x_, goal_y_, init_x_, init_y_;

    // Pub / Sub
    ros::Publisher  pub_mpc_weights_, pub_observations_, pub_status_;
    ros::Subscriber sub_costmap_, sub_lidar_, sub_odom_, sub_rl_;

    // Sensor cache (guarded by mutex_)
    std::mutex mutex_;
    nav_msgs::OccupancyGrid::ConstPtr costmap_;
    sensor_msgs::LaserScan::ConstPtr  lidar_;
    nav_msgs::Odometry::ConstPtr      odom_;

    // Current weights
    std::array<double, 4> weights_;

    // ── Callbacks ────────────────────────────────────────────────────────
    void cbCostmap(const nav_msgs::OccupancyGrid::ConstPtr& m) {
        std::lock_guard<std::mutex> lk(mutex_); costmap_ = m;
    }
    void cbLidar(const sensor_msgs::LaserScan::ConstPtr& m) {
        std::lock_guard<std::mutex> lk(mutex_); lidar_ = m;
    }
    void cbOdom(const nav_msgs::Odometry::ConstPtr& m) {
        std::lock_guard<std::mutex> lk(mutex_); odom_ = m;
    }

    void cbRLWeights(const std_msgs::Float64MultiArray::ConstPtr& msg) {
        if (msg->data.size() < 4) return;
        std::array<double, 4> w = {{msg->data[0], msg->data[1],
                                     msg->data[2], msg->data[3]}};
        weights_ = clamp(w);
        publishWeights(weights_);
        ROS_DEBUG("Bridge fwd: [%.2f, %.2f, %.2f, %.6f]",
                  weights_[0], weights_[1], weights_[2], weights_[3]);
    }

    // ── Observation computation (mirrors Python barn_env._get_obs) ───────
    std::vector<double> computeObservations() {
        std::lock_guard<std::mutex> lk(mutex_);
        std::vector<double> obs(OBS_DIM, 0.0);
        int idx = 0;

        // 1. Costmap 10×10
        auto patch = getCostmapPatch();
        std::copy(patch.begin(), patch.end(), obs.begin() + idx);
        idx += COSTMAP_SIZE * COSTMAP_SIZE;

        // 2. Lidar 36 rays
        auto lidar = getLidarObs();
        std::copy(lidar.begin(), lidar.end(), obs.begin() + idx);
        idx += N_LIDAR_RAYS;

        // 3. Robot state 5
        auto robot = getRobotState();
        std::copy(robot.begin(), robot.end(), obs.begin() + idx);
        idx += N_ROBOT_STATE;

        // 4. Normalised weights 4
        auto wn = normWeights();
        std::copy(wn.begin(), wn.end(), obs.begin() + idx);

        return obs;
    }

    std::vector<double> getCostmapPatch() {
        const int N = COSTMAP_SIZE;
        std::vector<double> p(N * N, 0.5);
        if (!costmap_ || !odom_) return p;

        auto& info = costmap_->info;
        double rx = odom_->pose.pose.position.x;
        double ry = odom_->pose.pose.position.y;
        int cx = static_cast<int>((rx - info.origin.position.x) / info.resolution);
        int cy = static_cast<int>((ry - info.origin.position.y) / info.resolution);
        int h = info.height, w = info.width, half = N / 2;

        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                int mi = cy - half + i, mj = cx - half + j;
                if (mi >= 0 && mi < h && mj >= 0 && mj < w) {
                    int8_t v = costmap_->data[mi * w + mj];
                    p[i * N + j] = (v < 0) ? 0.5
                                            : std::min(1.0, std::max(0.0, v / 100.0));
                }
            }
        }
        return p;
    }

    std::vector<double> getLidarObs() {
        std::vector<double> out(N_LIDAR_RAYS, 1.0);
        if (!lidar_) return out;

        int n = static_cast<int>(lidar_->ranges.size());
        int sec = std::max(1, n / N_LIDAR_RAYS);

        for (int i = 0; i < N_LIDAR_RAYS; ++i) {
            double mn = LIDAR_MAX;
            int s = i * sec, e = std::min(s + sec, n);
            for (int j = s; j < e; ++j) {
                double r = lidar_->ranges[j];
                if (std::isfinite(r)) {
                    r = std::max(LIDAR_MIN, std::min(LIDAR_MAX, r));
                    mn = std::min(mn, r);
                }
            }
            out[i] = std::max(0.0, std::min(1.0,
                        (mn - LIDAR_MIN) / (LIDAR_MAX - LIDAR_MIN)));
        }
        return out;
    }

    std::vector<double> getRobotState() {
        std::vector<double> st(N_ROBOT_STATE, 0.0);
        if (!odom_) return st;

        double px = odom_->pose.pose.position.x;
        double py = odom_->pose.pose.position.y;
        double gx = init_x_ + goal_x_;
        double gy = init_y_ + goal_y_;

        double dx = gx - px, dy = gy - py;
        double dist = std::sqrt(dx * dx + dy * dy);
        double max_d = std::sqrt(goal_x_ * goal_x_ + goal_y_ * goal_y_) + 1e-6;
        st[0] = std::max(0.0, std::min(1.0, dist / max_d));

        // heading error
        double desired = std::atan2(dy, dx);
        auto& q = odom_->pose.pose.orientation;
        double yaw = std::atan2(2.0 * (q.z * q.w + q.x * q.y),
                                1.0 - 2.0 * (q.y * q.y + q.z * q.z));
        double he = desired - yaw;
        while (he >  M_PI) he -= 2.0 * M_PI;
        while (he < -M_PI) he += 2.0 * M_PI;
        st[1] = (std::sin(he) + 1.0) / 2.0;
        st[2] = (std::cos(he) + 1.0) / 2.0;

        // velocities
        double v = odom_->twist.twist.linear.x;
        double w = odom_->twist.twist.angular.z;
        st[3] = std::max(0.0, std::min(1.0, (v + 2.0) / 4.0));
        st[4] = std::max(0.0, std::min(1.0, (w + 1.8) / 3.6));
        return st;
    }

    std::array<double, 4> normWeights() {
        return {{
            (weights_[0] - W_POS_MIN)   / (W_POS_MAX   - W_POS_MIN),
            (weights_[1] - W_HEAD_MIN)  / (W_HEAD_MAX  - W_HEAD_MIN),
            (weights_[2] - W_VEL_MIN)   / (W_VEL_MAX   - W_VEL_MIN),
            (weights_[3] - W_ACCEL_MIN) / (W_ACCEL_MAX - W_ACCEL_MIN),
        }};
    }

    // ── Weight helpers ───────────────────────────────────────────────────
    static std::array<double, 4> clamp(const std::array<double, 4>& w) {
        return {{
            std::max(W_POS_MIN,   std::min(W_POS_MAX,   w[0])),
            std::max(W_HEAD_MIN,  std::min(W_HEAD_MAX,  w[1])),
            std::max(W_VEL_MIN,   std::min(W_VEL_MAX,   w[2])),
            std::max(W_ACCEL_MIN, std::min(W_ACCEL_MAX, w[3])),
        }};
    }

    void publishWeights(const std::array<double, 4>& w) {
        std_msgs::Float64MultiArray msg;
        msg.data = {w[0], w[1], w[2], w[3]};
        pub_mpc_weights_.publish(msg);
    }

    void publishObservations(const std::vector<double>& obs) {
        std_msgs::Float64MultiArray msg;
        msg.data = obs;
        pub_observations_.publish(msg);
    }

    void publishStatus() {
        std_msgs::String msg;
        std::ostringstream ss;
        ss << "{\"mode\":\"" << mode_ << "\""
           << ",\"w\":[" << weights_[0] << "," << weights_[1]
           << "," << weights_[2] << "," << weights_[3] << "]}";
        msg.data = ss.str();
        pub_status_.publish(msg);
    }

    // ── ONNX inference (optional) ────────────────────────────────────────
#ifdef USE_ONNX_RUNTIME
    Ort::Env ort_env_{ORT_LOGGING_LEVEL_WARNING, "wt"};
    std::unique_ptr<Ort::Session> ort_session_;
    Ort::MemoryInfo mem_info_{
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)};

    void loadOnnx(const std::string& path) {
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(1);
        ort_session_ = std::make_unique<Ort::Session>(
            ort_env_, path.c_str(), opts);
        ROS_INFO("ONNX model loaded: %s", path.c_str());
    }

    std::array<double, 4> runOnnxInference(const std::vector<double>& obs) {
        if (!ort_session_) return weights_;

        std::vector<float> in(obs.begin(), obs.end());
        std::array<int64_t, 2> shape = {{1, OBS_DIM}};

        auto tensor = Ort::Value::CreateTensor<float>(
            mem_info_, in.data(), in.size(), shape.data(), shape.size());

        const char* in_names[]  = {"observation"};
        const char* out_names[] = {"action"};

        auto out = ort_session_->Run(
            Ort::RunOptions{nullptr}, in_names, &tensor, 1, out_names, 1);
        float* a = out[0].GetTensorMutableData<float>();

        // Rescale [-1,1] → weight ranges
        return {{
            W_POS_MIN   + (a[0] + 1.0) / 2.0 * (W_POS_MAX   - W_POS_MIN),
            W_HEAD_MIN  + (a[1] + 1.0) / 2.0 * (W_HEAD_MAX  - W_HEAD_MIN),
            W_VEL_MIN   + (a[2] + 1.0) / 2.0 * (W_VEL_MAX   - W_VEL_MIN),
            W_ACCEL_MIN + (a[3] + 1.0) / 2.0 * (W_ACCEL_MAX - W_ACCEL_MIN),
        }};
    }
#endif
};

}  // namespace sac_weight_tuner

// =========================================================================
int main(int argc, char** argv) {
    ros::init(argc, argv, "weight_tuner");
    ros::NodeHandle nh;
    ros::NodeHandle nhp("~");

    sac_weight_tuner::WeightTunerNode node(nh, nhp);
    node.spin();
    return 0;
}
