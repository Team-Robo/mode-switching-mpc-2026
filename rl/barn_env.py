#!/usr/bin/env python3
"""
BARN Gazebo Gym Environment for SAC-based MPC weight tuning.

Observation (14-dim, compact for i3 CPU):
  [0]  v_linear          - current linear velocity
  [1]  w_angular         - current angular velocity
  [2]  heading_error     - angle diff between robot heading & path
  [3]  cross_track_error - lateral offset from reference path
  [4]  dist_to_goal      - Euclidean distance to navigation goal
  [5]  min_obs_front     - closest obstacle in front sector
  [6]  min_obs_left      - closest obstacle in left sector
  [7]  min_obs_right     - closest obstacle in right sector
  [8]  path_curvature    - mean curvature of upcoming path segment
  [9]  n_nearby_obs      - number of obstacles within SAFE_DISTANCE (normalised)
  [10] is_reversal       - 1.0 if reversal mode, else 0.0
  [11] progress_rate     - distance-to-goal change per step (progress signal)
  [12] n_dynamic_obs     - number of tracked dynamic obstacles (normalised)
  [13] mpc_solve_time    - last MPC solve wall-clock time (normalised)

Action (3-dim, continuous, normalised to [-1,1] then mapped):
  [0] -> weight_position_error  in [1, 100]
  [1] -> weight_heading_error   in [1, 100]
  [2] -> weight_acceleration    in [0.0001, 0.1]
"""

import os
import time
import math
import subprocess
import signal

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import rospy
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import Twist, Quaternion
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float64MultiArray, Bool
from gazebo_msgs.srv import SetModelState, GetModelState
from gazebo_msgs.msg import ModelState
from std_srvs.srv import Empty

import rospkg
from os.path import join


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
INIT_POSITION_STATIC  = [-2.25, 3.0, 1.57]
GOAL_OFFSET_STATIC    = [0.0, 10.0]

INIT_POSITION_DYNAMIC = [11.0, 0.0, 3.14]
GOAL_OFFSET_DYNAMIC   = [-20.0, 0.0]

MAX_EPISODE_TIME = 100.0   # seconds (sim time)
STEP_DT          = 0.2     # RL decision period (5 Hz) — balances reactivity & CPU cost
LASER_SECTORS    = 3       # front / left / right
LASER_MAX_RANGE  = 10.0

# Weight bounds (action space is mapped from [-1,1] into these)
W_POS_LO,  W_POS_HI  = 1.0, 100.0
W_HEAD_LO, W_HEAD_HI = 1.0, 100.0
W_ACC_LO,  W_ACC_HI  = 0.0001, 0.1


def _quat_to_yaw(q):
    """geometry_msgs.Quaternion -> yaw."""
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


def _compute_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def _angle_diff(a, b):
    """Signed shortest angle difference a-b, in [-pi, pi]."""
    d = a - b
    while d > math.pi:
        d -= 2.0 * math.pi
    while d < -math.pi:
        d += 2.0 * math.pi
    return d


def _rescale(action_val, lo, hi):
    """Map [-1, 1] -> [lo, hi]."""
    return lo + (action_val + 1.0) * 0.5 * (hi - lo)


def _create_model_state(x, y, z, angle):
    ms = ModelState()
    ms.model_name = "jackal"
    ms.pose.position.x = x
    ms.pose.position.y = y
    ms.pose.position.z = z
    ms.pose.orientation = Quaternion(0, 0, math.sin(angle / 2.0), math.cos(angle / 2.0))
    ms.reference_frame = "world"
    return ms


class BarnMpcEnv(gym.Env):
    """
    Gymnasium environment that:
      1. Boots a BARN Gazebo world + the MPC nav stack.
      2. At each step receives compact observations from ROS topics.
      3. Outputs 3 MPC weight actions published to /mpc_weights.
      4. Computes a dense reward: progress + safety + smoothness.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        world_indices=None,
        gui=False,
        launch_file="move_base_mlda_2026.launch",
        step_dt=STEP_DT,
        max_episode_time=MAX_EPISODE_TIME,
        randomize_worlds=True,
    ):
        super().__init__()

        # --- Config ---
        self.step_dt = step_dt
        self.max_episode_time = max_episode_time
        self.launch_file = launch_file
        self.gui = gui
        self.randomize_worlds = randomize_worlds

        # World indices for training curriculum
        if world_indices is None:
            self.world_indices = list(range(0, 50))  # easy-to-medium worlds
        else:
            self.world_indices = list(world_indices)

        self._current_world_idx = 0
        self._world_cursor = 0  # sequential fallback

        # --- Gym spaces ---
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32
        )
        # Actions normalised to [-1, 1]; mapped to weight ranges in step()
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )

        # --- Internal state ---
        self._robot_x = 0.0
        self._robot_y = 0.0
        self._robot_yaw = 0.0
        self._robot_v = 0.0
        self._robot_w = 0.0
        self._laser_ranges = np.full(720, LASER_MAX_RANGE, dtype=np.float32)
        self._global_plan_xs = []
        self._global_plan_ys = []
        self._collision = False
        self._collision_count = 0
        self._last_dist_to_goal = None
        self._mpc_solve_time = 0.0
        self._is_reversal = False
        self._n_dynamic_obs = 0

        self._goal_world = (0.0, 0.0)
        self._init_pos = INIT_POSITION_STATIC

        # ROS processes (managed per-episode)
        self._gazebo_proc = None
        self._nav_proc = None
        self._ros_initialized = False

        # ROS package paths
        rospack = rospkg.RosPack()
        self._helper_path = rospack.get_path("jackal_helper")

    # ======================================================================
    # Gym API
    # ======================================================================
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # Pick world
        if self.randomize_worlds:
            self._current_world_idx = self.np_random.choice(self.world_indices)
        else:
            self._current_world_idx = self.world_indices[
                self._world_cursor % len(self.world_indices)
            ]
            self._world_cursor += 1

        # Determine init / goal based on world type
        widx = self._current_world_idx
        if widx < 300:
            self._init_pos = INIT_POSITION_STATIC[:]
            goal_offset = GOAL_OFFSET_STATIC
            world_name = f"BARN/world_{widx}.world"
        else:
            self._init_pos = INIT_POSITION_DYNAMIC[:]
            goal_offset = GOAL_OFFSET_DYNAMIC
            world_name = f"DynaBARN/world_{widx - 300}.world"

        self._goal_world = (
            self._init_pos[0] + goal_offset[0],
            self._init_pos[1] + goal_offset[1],
        )

        # Restart Gazebo + nav stack
        self._shutdown_ros()
        self._launch_gazebo(world_name)
        self._wait_for_gazebo()
        self._reset_robot()
        self._launch_nav_stack()
        self._setup_subscribers()
        self._send_goal(goal_offset)

        # Wait for robot to start receiving data
        time.sleep(2.0)

        self._collision_count = 0
        self._collision = False
        self._episode_start_time = rospy.get_time()
        self._last_dist_to_goal = _compute_distance(
            (self._robot_x, self._robot_y), self._goal_world
        )

        obs = self._get_observation()
        return obs, {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)

        # Map normalised actions to actual weight values
        w_pos  = _rescale(action[0], W_POS_LO, W_POS_HI)
        w_head = _rescale(action[1], W_HEAD_LO, W_HEAD_HI)
        w_acc  = _rescale(action[2], W_ACC_LO, W_ACC_HI)

        # Publish weights to MPC node
        self._publish_weights(w_pos, w_head, w_acc)

        # Let the sim run for step_dt seconds (sim time)
        t0 = rospy.get_time()
        while rospy.get_time() - t0 < self.step_dt:
            time.sleep(0.01)

        # Compute reward & check termination
        obs = self._get_observation()
        reward, terminated, truncated, info = self._compute_reward()

        return obs, reward, terminated, truncated, info

    def close(self):
        self._shutdown_ros()

    # ======================================================================
    # Observation
    # ======================================================================
    def _get_observation(self):
        # --- Laser-derived obstacle distances in 3 sectors ---
        n_rays = len(self._laser_ranges)
        third = n_rays // 3
        ranges = np.clip(self._laser_ranges, 0.0, LASER_MAX_RANGE)

        min_front = float(np.min(ranges[third : 2 * third]))  # center third
        min_left  = float(np.min(ranges[2 * third :]))
        min_right = float(np.min(ranges[: third]))

        # --- Heading error & cross-track error ---
        heading_err = 0.0
        cross_track = 0.0
        path_curvature = 0.0
        if len(self._global_plan_xs) >= 2:
            # Find closest point on plan
            plan_pts = np.array(
                list(zip(self._global_plan_xs, self._global_plan_ys))
            )
            robot_pt = np.array([self._robot_x, self._robot_y])
            dists = np.linalg.norm(plan_pts - robot_pt, axis=1)
            closest_idx = int(np.argmin(dists))
            cross_track = float(dists[closest_idx])

            # Heading to next waypoint
            look = min(closest_idx + 3, len(plan_pts) - 1)
            dx = plan_pts[look, 0] - self._robot_x
            dy = plan_pts[look, 1] - self._robot_y
            desired_heading = math.atan2(dy, dx)
            heading_err = _angle_diff(desired_heading, self._robot_yaw)

            # Path curvature (average angular change over next ~N points)
            curvatures = []
            end = min(closest_idx + 15, len(plan_pts) - 1)
            for i in range(closest_idx, end - 1):
                dx1 = plan_pts[i + 1, 0] - plan_pts[i, 0]
                dy1 = plan_pts[i + 1, 1] - plan_pts[i, 1]
                if i + 2 <= end:
                    dx2 = plan_pts[i + 2, 0] - plan_pts[i + 1, 0]
                    dy2 = plan_pts[i + 2, 1] - plan_pts[i + 1, 1]
                    a1 = math.atan2(dy1, dx1)
                    a2 = math.atan2(dy2, dx2)
                    curvatures.append(abs(_angle_diff(a2, a1)))
            path_curvature = float(np.mean(curvatures)) if curvatures else 0.0

        # --- Distance to goal ---
        dist_to_goal = _compute_distance(
            (self._robot_x, self._robot_y), self._goal_world
        )

        # --- Number of nearby obstacles ---
        n_nearby = int(np.sum(ranges < 1.25))  # SAFE_DISTANCE
        n_nearby_norm = min(n_nearby / 50.0, 1.0)

        # --- Progress rate ---
        if self._last_dist_to_goal is not None:
            progress_rate = (self._last_dist_to_goal - dist_to_goal) / max(
                self.step_dt, 0.01
            )
        else:
            progress_rate = 0.0

        obs = np.array(
            [
                self._robot_v,                     # 0
                self._robot_w,                     # 1
                heading_err,                       # 2
                cross_track,                       # 3
                dist_to_goal,                      # 4
                min_front / LASER_MAX_RANGE,       # 5  normalised
                min_left / LASER_MAX_RANGE,        # 6
                min_right / LASER_MAX_RANGE,       # 7
                path_curvature,                    # 8
                n_nearby_norm,                     # 9
                float(self._is_reversal),          # 10
                progress_rate,                     # 11
                min(self._n_dynamic_obs / 5.0, 1.0),  # 12
                min(self._mpc_solve_time / 0.05, 1.0), # 13  norm by 50ms budget
            ],
            dtype=np.float32,
        )
        return obs

    # ======================================================================
    # Reward
    # ======================================================================
    def _compute_reward(self):
        dist_to_goal = _compute_distance(
            (self._robot_x, self._robot_y), self._goal_world
        )
        elapsed = rospy.get_time() - self._episode_start_time

        terminated = False
        truncated = False
        info = {}

        # --- Collision ---
        if self._collision_count > 0:
            terminated = True
            info["outcome"] = "collision"
            self._collision_count = 0
            reward = -50.0
            self._last_dist_to_goal = dist_to_goal
            return reward, terminated, truncated, info

        # --- Timeout ---
        if elapsed >= self.max_episode_time:
            truncated = True
            info["outcome"] = "timeout"
            reward = -20.0
            self._last_dist_to_goal = dist_to_goal
            return reward, terminated, truncated, info

        # --- Goal reached ---
        if dist_to_goal < 1.0:
            terminated = True
            info["outcome"] = "success"
            # Bonus inversely proportional to time taken
            time_bonus = max(0.0, 1.0 - elapsed / self.max_episode_time) * 50.0
            reward = 100.0 + time_bonus
            self._last_dist_to_goal = dist_to_goal
            return reward, terminated, truncated, info

        # --- Dense reward ---
        # 1. Progress toward goal
        if self._last_dist_to_goal is not None:
            progress = self._last_dist_to_goal - dist_to_goal
        else:
            progress = 0.0
        r_progress = 10.0 * progress  # ~10 reward per metre of progress

        # 2. Time penalty (encourages speed)
        r_time = -0.05

        # 3. Safety bonus / penalty
        min_range = float(np.min(np.clip(self._laser_ranges, 0.0, LASER_MAX_RANGE)))
        if min_range < 0.5:
            r_safety = -2.0  # dangerously close
        elif min_range < 1.0:
            r_safety = -0.5
        else:
            r_safety = 0.1   # comfortable margin

        # 4. Smoothness (penalise large angular velocity)
        r_smooth = -0.1 * abs(self._robot_w)

        reward = r_progress + r_time + r_safety + r_smooth

        self._last_dist_to_goal = dist_to_goal
        info["dist_to_goal"] = dist_to_goal
        info["elapsed"] = elapsed

        return reward, terminated, truncated, info

    # ======================================================================
    # ROS helpers
    # ======================================================================
    def _publish_weights(self, w_pos, w_head, w_acc):
        """Publish weight triplet on /mpc_weights (Float64MultiArray)."""
        if not hasattr(self, "_pub_weights") or self._pub_weights is None:
            self._pub_weights = rospy.Publisher(
                "/mpc_weights", Float64MultiArray, queue_size=1, latch=True
            )
        msg = Float64MultiArray()
        msg.data = [w_pos, w_head, w_acc]
        self._pub_weights.publish(msg)

    def _setup_subscribers(self):
        """Subscribe to necessary ROS topics for observations."""
        if hasattr(self, "_subs_created") and self._subs_created:
            return
        self._sub_odom = rospy.Subscriber(
            "/odometry/filtered", Odometry, self._cb_odom, queue_size=1
        )
        self._sub_scan = rospy.Subscriber(
            "/front/scan", LaserScan, self._cb_scan, queue_size=1
        )
        self._sub_plan = rospy.Subscriber(
            "/move_base/TrajectoryPlannerROS/global_plan",
            Path,
            self._cb_plan,
            queue_size=1,
        )
        self._sub_collision = rospy.Subscriber(
            "/collision", Bool, self._cb_collision, queue_size=1
        )
        self._sub_mpc_diag = rospy.Subscriber(
            "/mpc_diagnostics", Float64MultiArray, self._cb_mpc_diag, queue_size=1
        )
        self._subs_created = True

    def _cb_odom(self, msg):
        self._robot_x = msg.pose.pose.position.x
        self._robot_y = msg.pose.pose.position.y
        self._robot_yaw = _quat_to_yaw(msg.pose.pose.orientation)
        self._robot_v = msg.twist.twist.linear.x
        self._robot_w = msg.twist.twist.angular.z

    def _cb_scan(self, msg):
        self._laser_ranges = np.array(msg.ranges, dtype=np.float32)
        self._laser_ranges[np.isinf(self._laser_ranges)] = LASER_MAX_RANGE

    def _cb_plan(self, msg):
        self._global_plan_xs = [p.pose.position.x for p in msg.poses]
        self._global_plan_ys = [p.pose.position.y for p in msg.poses]

    def _cb_collision(self, msg):
        if msg.data:
            self._collision_count += 1

    def _cb_mpc_diag(self, msg):
        """Expect [solve_time_ms, is_reversal, n_dynamic_obs]."""
        if len(msg.data) >= 3:
            self._mpc_solve_time = msg.data[0] / 1000.0  # ms -> s
            self._is_reversal = msg.data[1] > 0.5
            self._n_dynamic_obs = int(msg.data[2])

    # ------------------------------------------------------------------
    # Gazebo / nav stack lifecycle
    # ------------------------------------------------------------------
    def _launch_gazebo(self, world_name):
        os.environ["JACKAL_LASER"] = "1"
        os.environ["JACKAL_LASER_MODEL"] = "ust10"
        os.environ["JACKAL_LASER_OFFSET"] = "-0.065 0 0.01"

        base_path = self._helper_path
        os.environ["GAZEBO_PLUGIN_PATH"] = join(base_path, "plugins")

        launch_file = join(base_path, "launch", "gazebo_launch.launch")
        world_path = join(base_path, "worlds", world_name)

        self._gazebo_proc = subprocess.Popen(
            [
                "roslaunch",
                launch_file,
                "world_name:=" + world_path,
                "gui:=" + ("true" if self.gui else "false"),
                "rviz:=false",
            ],
            preexec_fn=os.setsid,
        )
        time.sleep(6)

    def _wait_for_gazebo(self):
        rospy.wait_for_service("/gazebo/get_model_state", timeout=30)

    def _reset_robot(self):
        reset_srv = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
        ms = _create_model_state(
            self._init_pos[0], self._init_pos[1], 0.0, self._init_pos[2]
        )
        for _ in range(5):
            try:
                reset_srv(ms)
                time.sleep(0.5)
                get_state = rospy.ServiceProxy(
                    "/gazebo/get_model_state", GetModelState
                )
                state = get_state("jackal", "world")
                pos = state.pose.position
                if _compute_distance(
                    (pos.x, pos.y), (self._init_pos[0], self._init_pos[1])
                ) < 0.1:
                    break
            except rospy.ServiceException:
                time.sleep(1)

    def _launch_nav_stack(self):
        rospack = rospkg.RosPack()
        # Search for the launch file in ROS packages
        for pkg_name in rospack.list():
            pkg_path = rospack.get_path(pkg_name)
            candidate = join(pkg_path, "launch", self.launch_file)
            if os.path.isfile(candidate):
                launch_path = candidate
                break
        else:
            raise FileNotFoundError(
                f"launch/{self.launch_file} not found in any ROS package"
            )

        self._nav_proc = subprocess.Popen(
            ["roslaunch", launch_path],
            preexec_fn=os.setsid,
        )
        time.sleep(4)

    def _send_goal(self, goal_offset):
        """Send navigation goal via actionlib."""
        import actionlib
        from move_base_msgs.msg import MoveBaseGoal, MoveBaseAction

        nav_as = actionlib.SimpleActionClient("/move_base", MoveBaseAction)
        nav_as.wait_for_server(timeout=rospy.Duration(10))

        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "odom"
        goal.target_pose.pose.position.x = goal_offset[0]
        goal.target_pose.pose.position.y = goal_offset[1]
        goal.target_pose.pose.orientation = Quaternion(0, 0, 0, 1)
        nav_as.send_goal(goal)

    def _shutdown_ros(self):
        for proc in [self._nav_proc, self._gazebo_proc]:
            if proc is not None:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                    proc.wait(timeout=5)
                except Exception:
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    except Exception:
                        pass
        self._gazebo_proc = None
        self._nav_proc = None


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
gym.register(
    id="BarnMpc-v0",
    entry_point="rl.barn_env:BarnMpcEnv",
)
