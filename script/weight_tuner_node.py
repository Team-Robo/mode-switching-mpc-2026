#!/usr/bin/env python3
"""
Lightweight ROS node for deploying the trained SAC policy on i3 CPU.

Runs at 5 Hz (configurable). Reads compact observations from ROS topics,
runs a tiny [32,32] MLP forward pass (<0.3ms on i3), and publishes
the 3 MPC weights to /mpc_weights.

Inference budget breakdown on Intel i3:
  - ROS subscriber callbacks: ~0.1 ms
  - Observation assembly:     ~0.05 ms
  - Policy forward pass:      ~0.1-0.3 ms (PyTorch) / ~0.05 ms (ONNX)
  - Weight publish:           ~0.05 ms
  Total: < 1 ms per step — negligible vs MPC's ~5-20 ms solve time

Usage:
  rosrun teamrobo2026 weight_tuner_node.py --model checkpoints/sac_mpc_final
  rosrun teamrobo2026 weight_tuner_node.py --onnx checkpoints/sac_policy.onnx
"""

import argparse
import math
import time

import numpy as np
import rospy
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float64MultiArray, Bool
from geometry_msgs.msg import Quaternion

# Only import what we need — keep startup fast
LASER_MAX_RANGE = 10.0

# Weight bounds (must match training)
W_POS_LO,  W_POS_HI  = 5.0, 100.0
W_HEAD_LO, W_HEAD_HI = 5.0, 100.0
W_ACC_LO,  W_ACC_HI  = 0.0001, 0.1

def _quat_to_yaw(q):
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


def _angle_diff(a, b):
    d = a - b
    while d > math.pi:
        d -= 2 * math.pi
    while d < -math.pi:
        d += 2 * math.pi
    return d


def _rescale(v, lo, hi):
    return lo + (v + 1.0) * 0.5 * (hi - lo)


class WeightTunerNode:
    """
    Subscribes to sensor topics, computes 14-dim observation,
    runs SAC policy, publishes MPC weights.
    """

    def __init__(self, model_path=None, onnx_path=None, rate_hz=5.0,
                 goal_x=0.0, goal_y=10.0):
        rospy.init_node("weight_tuner", anonymous=True)

        self.rate = rospy.Rate(rate_hz)
        self.goal = (goal_x, goal_y)

        # --- Load model ---
        self.use_onnx = onnx_path is not None
        if self.use_onnx:
            self._load_onnx(onnx_path)
        else:
            self._load_sb3(model_path)

        # --- State ---
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0
        self.robot_v = 0.0
        self.robot_w = 0.0
        self.laser_ranges = np.full(720, LASER_MAX_RANGE, dtype=np.float32)
        self.plan_xs = []
        self.plan_ys = []
        self.mpc_solve_time = 0.0
        self.is_reversal = False
        self.n_dynamic_obs = 0
        self.last_dist_to_goal = None

        # --- Publishers ---
        self.pub_weights = rospy.Publisher(
            "/mpc_weights", Float64MultiArray, queue_size=1, latch=True
        )

        # --- Subscribers (same topics as training env) ---
        rospy.Subscriber("/odometry/filtered", Odometry, self._cb_odom, queue_size=1)
        rospy.Subscriber("/front/scan", LaserScan, self._cb_scan, queue_size=1)
        rospy.Subscriber(
            "/move_base/TrajectoryPlannerROS/global_plan",
            Path, self._cb_plan, queue_size=1,
        )
        rospy.Subscriber(
            "/mpc_diagnostics", Float64MultiArray, self._cb_diag, queue_size=1
        )

        # Publish default weights immediately
        self._publish_weights(49.0, 37.0, 0.0021)

        rospy.loginfo(
            f"WeightTuner ready: {'ONNX' if self.use_onnx else 'SB3'} model, "
            f"{rate_hz} Hz, goal=({goal_x}, {goal_y})"
        )

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    def _load_sb3(self, path):
        """Load SB3 SAC model for CPU inference."""
        import torch
        from stable_baselines3 import SAC

        # Force CPU
        self.model = SAC.load(path, device="cpu")
        self.model.policy.set_training_mode(False)
        rospy.loginfo(f"Loaded SB3 model from {path} (CPU)")

    def _load_onnx(self, path):
        """Load ONNX model for minimal-overhead CPU inference."""
        import onnxruntime as ort

        self.ort_session = ort.InferenceSession(
            path,
            providers=["CPUExecutionProvider"],
        )
        rospy.loginfo(f"Loaded ONNX model from {path}")

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    def _cb_odom(self, msg):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        self.robot_yaw = _quat_to_yaw(msg.pose.pose.orientation)
        self.robot_v = msg.twist.twist.linear.x
        self.robot_w = msg.twist.twist.angular.z

    def _cb_scan(self, msg):
        self.laser_ranges = np.array(msg.ranges, dtype=np.float32)
        self.laser_ranges[np.isinf(self.laser_ranges)] = LASER_MAX_RANGE

    def _cb_plan(self, msg):
        self.plan_xs = [p.pose.position.x for p in msg.poses]
        self.plan_ys = [p.pose.position.y for p in msg.poses]

    def _cb_diag(self, msg):
        if len(msg.data) >= 3:
            self.mpc_solve_time = msg.data[0] / 1000.0
            self.is_reversal = msg.data[1] > 0.5
            self.n_dynamic_obs = int(msg.data[2])

    # ------------------------------------------------------------------
    # Observation (identical to training env)
    # ------------------------------------------------------------------
    def _get_obs(self):
        n = len(self.laser_ranges)
        third = n // 3
        ranges = np.clip(self.laser_ranges, 0.0, LASER_MAX_RANGE)

        min_front = float(np.min(ranges[third:2*third]))
        min_left = float(np.min(ranges[2*third:]))
        min_right = float(np.min(ranges[:third]))

        heading_err = 0.0
        cross_track = 0.0
        path_curv = 0.0
        if len(self.plan_xs) >= 2:
            pts = np.array(list(zip(self.plan_xs, self.plan_ys)))
            rpt = np.array([self.robot_x, self.robot_y])
            dists = np.linalg.norm(pts - rpt, axis=1)
            ci = int(np.argmin(dists))
            cross_track = float(dists[ci])

            look = min(ci + 3, len(pts) - 1)
            dx = pts[look, 0] - self.robot_x
            dy = pts[look, 1] - self.robot_y
            heading_err = _angle_diff(math.atan2(dy, dx), self.robot_yaw)

            curvs = []
            end = min(ci + 15, len(pts) - 1)
            for i in range(ci, end - 1):
                dx1 = pts[i+1, 0] - pts[i, 0]
                dy1 = pts[i+1, 1] - pts[i, 1]
                if i + 2 <= end:
                    dx2 = pts[i+2, 0] - pts[i+1, 0]
                    dy2 = pts[i+2, 1] - pts[i+1, 1]
                    curvs.append(abs(_angle_diff(
                        math.atan2(dy2, dx2), math.atan2(dy1, dx1)
                    )))
            path_curv = float(np.mean(curvs)) if curvs else 0.0

        dist_goal = math.sqrt(
            (self.robot_x - self.goal[0])**2 + (self.robot_y - self.goal[1])**2
        )
        n_nearby = int(np.sum(ranges < 1.25))
        n_nearby_norm = min(n_nearby / 50.0, 1.0)

        if self.last_dist_to_goal is not None:
            progress = (self.last_dist_to_goal - dist_goal) / 0.2  # step_dt
        else:
            progress = 0.0
        self.last_dist_to_goal = dist_goal

        return np.array([
            self.robot_v,
            self.robot_w,
            heading_err,
            cross_track,
            dist_goal,
            min_front / LASER_MAX_RANGE,
            min_left / LASER_MAX_RANGE,
            min_right / LASER_MAX_RANGE,
            path_curv,
            n_nearby_norm,
            float(self.is_reversal),
            progress,
            min(self.n_dynamic_obs / 5.0, 1.0),
            min(self.mpc_solve_time / 0.05, 1.0),
        ], dtype=np.float32)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def _predict(self, obs):
        if self.use_onnx:
            inputs = {self.ort_session.get_inputs()[0].name: obs.reshape(1, -1)}
            outputs = self.ort_session.run(None, inputs)
            action = np.tanh(outputs[0][0])  # squash to [-1, 1]
        else:
            action, _ = self.model.predict(obs, deterministic=True)
        return np.clip(action, -1.0, 1.0)

    def _publish_weights(self, w_pos, w_head, w_acc):
        msg = Float64MultiArray()
        msg.data = [w_pos, w_head, w_acc]
        self.pub_weights.publish(msg)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self):
        while not rospy.is_shutdown():
            t0 = time.time()

            obs = self._get_obs()
            action = self._predict(obs)

            w_pos  = _rescale(float(action[0]), W_POS_LO, W_POS_HI)
            w_head = _rescale(float(action[1]), W_HEAD_LO, W_HEAD_HI)
            w_acc  = _rescale(float(action[2]), W_ACC_LO, W_ACC_HI)

            self._publish_weights(w_pos, w_head, w_acc)

            dt_ms = (time.time() - t0) * 1000
            rospy.loginfo_throttle(
                5.0,
                f"Weights: pos={w_pos:.1f} head={w_head:.1f} acc={w_acc:.5f} "
                f"({dt_ms:.1f}ms)"
            )

            self.rate.sleep()


def main():
    parser = argparse.ArgumentParser(description="MPC weight tuner (deployment)")
    parser.add_argument(
        "--model", type=str, default="checkpoints/sac_mpc_final",
        help="Path to SB3 model (without .zip)",
    )
    parser.add_argument(
        "--onnx", type=str, default=None,
        help="Path to ONNX model (overrides --model)",
    )
    parser.add_argument("--rate", type=float, default=5.0, help="Inference rate (Hz)")
    parser.add_argument("--goal-x", type=float, default=0.0)
    parser.add_argument("--goal-y", type=float, default=10.0)
    args = parser.parse_args()

    node = WeightTunerNode(
        model_path=args.model,
        onnx_path=args.onnx,
        rate_hz=args.rate,
        goal_x=args.goal_x,
        goal_y=args.goal_y,
    )
    node.run()


if __name__ == "__main__":
    main()
