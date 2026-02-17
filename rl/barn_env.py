#!/usr/bin/env python3
"""
barn_env.py
BARN Gazebo Gym Environment for SAC-based MPC weight tuning

Observation (15-dim):
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
  [11] progress_rate     - distance-to-goal change per step
  [12] n_dynamic_obs     - number of tracked dynamic obstacles (normalised)
  [13] mpc_solve_time    - last MPC solve wall-clock time (normalised)
  [14] solver_status     - 0 if success, 1 if failed

Action (3-dim, continuous, normalised to [-1,1] then mapped):
  [0] -> weight_position_error  in [5, 100]
  [1] -> weight_heading_error   in [5, 100]
  [2] -> weight_acceleration    in [0.0001, 0.1]
"""

import os
import time
import math
import subprocess
import signal
import gc

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import rospy
import actionlib
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import Quaternion
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float64MultiArray, Bool
from move_base_msgs.msg import MoveBaseGoal, MoveBaseAction
from gazebo_msgs.srv import SetModelState, GetModelState
from gazebo_msgs.msg import ModelState

import rospkg
from os.path import join


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
INIT_POSITION_STATIC  = [-2.25, 3.0, 1.57]
GOAL_OFFSET_STATIC    = [0.0, 10.0]

INIT_POSITION_DYNAMIC = [11.0, 0.0, 3.14]
GOAL_OFFSET_DYNAMIC   = [-20.0, 0.0]

STEP_DT         = 0.2
LASER_MAX_RANGE = 10.0

W_POS_LO,  W_POS_HI  = 5.0, 100.0
W_HEAD_LO, W_HEAD_HI = 5.0, 100.0
W_ACC_LO,  W_ACC_HI  = 0.0001, 0.1


def _quat_to_yaw(q):
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


def _compute_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def _angle_diff(a, b):
    d = a - b
    while d > math.pi:
        d -= 2.0 * math.pi
    while d < -math.pi:
        d += 2.0 * math.pi
    return d


def _rescale(action_val, lo, hi):
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
    metadata = {"render_modes": []}

    def __init__(
        self,
        world_indices=None,
        gui=False,
        launch_file="move_base_mlda_2026.launch",
        step_dt=STEP_DT,
        max_episode_time=150.0,
        randomize_worlds=True,
        restart_interval=2000,
        ros_master_port=11311,
    ):
        super().__init__()

        self.step_dt = step_dt
        self.max_episode_time = max_episode_time
        self.launch_file = launch_file
        self.gui = gui
        self.randomize_worlds = randomize_worlds
        self.restart_interval = restart_interval
        self.ros_master_port = ros_master_port

        os.environ["ROS_MASTER_URI"] = f"http://localhost:{ros_master_port}"
        os.environ["ROS_HOSTNAME"] = "localhost"
        os.environ["ROS_IP"] = "127.0.0.1"

        self.gazebo_master_port = ros_master_port + 1000
        os.environ["GAZEBO_MASTER_URI"] = f"http://localhost:{self.gazebo_master_port}"

        self.world_indices = list(range(0, 50)) if world_indices is None else list(world_indices)
        self._current_world_idx = 0
        self._world_cursor = 0

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        # Internal state
        self._robot_x = 0.0
        self._robot_y = 0.0
        self._robot_yaw = 0.0
        self._robot_v = 0.0
        self._robot_w = 0.0
        self._laser_ranges = np.full(720, LASER_MAX_RANGE, dtype=np.float32)
        self._global_plan_xs = []
        self._global_plan_ys = []
        self._collision_count = 0
        self._last_dist_to_goal = None
        self._mpc_solve_time = 0.0
        self._is_reversal = False
        self._n_dynamic_obs = 0
        self._solver_status = 0
        self._stuck_counter = 0
        self._last_progress_check_dist = None
        self._steps_since_progress = 0
        self._steps_in_reversal = 0
        self._goal_odom = (0.0, 0.0)
        self._init_pos = INIT_POSITION_STATIC

        self._gazebo_proc = None
        self._nav_proc = None
        self._rosmaster_proc = None          # FIX: persistent rosmaster handle
        self._ros_initialized = False
        self._shutdown_in_progress = False
        self._subs_created = False
        self._reset_counter = 0
        self._total_resets = 0
        self._pub_weights = None
        self._nav_client = None

        rospack = rospkg.RosPack()
        self._helper_path = rospack.get_path("jackal_helper")

        self._cleanup_stray_processes()
        self._ensure_rosmaster()             # FIX: start persistent rosmaster early

    # ======================================================================
    # Gym API
    # ======================================================================
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self._reset_counter += 1
        self._total_resets += 1

        if self._reset_counter >= self.restart_interval:
            rospy.logwarn(f"=== HARD RESTART: {self._total_resets} total resets ===")
            self._hard_restart_gazebo()
            self._reset_counter = 0

        # Pick world
        if self.randomize_worlds:
            self._current_world_idx = self.np_random.choice(self.world_indices)
        else:
            self._current_world_idx = self.world_indices[self._world_cursor % len(self.world_indices)]
            self._world_cursor += 1

        widx = self._current_world_idx
        if widx < 300:
            self._init_pos = INIT_POSITION_STATIC[:]
            goal_offset = GOAL_OFFSET_STATIC
            world_name = f"BARN/world_{widx}.world"
        else:
            self._init_pos = INIT_POSITION_DYNAMIC[:]
            goal_offset = GOAL_OFFSET_DYNAMIC
            world_name = f"DynaBARN/world_{widx - 300}.world"

        self._goal_odom = (goal_offset[0], goal_offset[1])

        if self._ros_initialized:
            rospy.loginfo(f"=== Episode: World {widx} (Reset #{self._total_resets}) ===")

        # Restart Gazebo + nav stack (rosmaster stays alive)
        self._shutdown_ros()
        time.sleep(0.5)
        self._ensure_rosmaster()             # FIX: make sure master is still up
        self._launch_gazebo(world_name)

        if not self._ros_initialized:
            time.sleep(2.0)
            try:
                rospy.init_node(f"barn_env_{self.ros_master_port}", anonymous=True, disable_signals=True)
                self._ros_initialized = True
                rospy.loginfo(f"ROS node initialized on port {self.ros_master_port}")
            except Exception as e:
                print(f"Failed to initialize ROS node: {e}")
                raise

        self._wait_for_gazebo()
        self._reset_robot()
        self._launch_nav_stack()
        self._setup_subscribers()
        self._send_goal(goal_offset)

        # FIX: Increased sleep + reset collision AFTER sleep to flush stale msgs
        time.sleep(2.0)
        self._collision_count = 0
        self._solver_status = 0

        # FIX: wait for sim clock to start ticking before returning
        self._wait_for_clock()

        self._episode_start_time = rospy.get_time()
        self._last_dist_to_goal = _compute_distance((self._robot_x, self._robot_y), self._goal_odom)
        self._stuck_counter = 0
        self._last_progress_check_dist = self._last_dist_to_goal
        self._steps_since_progress = 0
        self._steps_in_reversal = 0

        return self._get_observation(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        w_pos  = _rescale(action[0], W_POS_LO, W_POS_HI)
        w_head = _rescale(action[1], W_HEAD_LO, W_HEAD_HI)
        w_acc  = _rescale(action[2], W_ACC_LO, W_ACC_HI)
        self._publish_weights(w_pos, w_head, w_acc)

        t0_wall = time.time()
        timeout_wall = self.step_dt * 5.0  # 1 second hard wall timeout

        try:
            t0_sim = rospy.get_time()
        except Exception:
            return self._get_observation(), -2.0, True, True, {"outcome": "gazebo_error"}

        if rospy.is_shutdown():
            rospy.logwarn("STEP: ros_shutdown")
            return self._get_observation(), -2.0, True, True, {"outcome": "ros_shutdown"}

        last_clock_update = time.time()
        last_sim_time = t0_sim

        while True:
            if rospy.is_shutdown():
                return self._get_observation(), -2.0, True, True, {"outcome": "ros_shutdown"}

            if self._gazebo_proc is not None and self._gazebo_proc.poll() is not None:
                rospy.logwarn(f"STEP: gazebo_crashed (exit={self._gazebo_proc.returncode})")
                return self._get_observation(), -2.0, True, True, {"outcome": "gazebo_crashed"}

            if time.time() - t0_wall > timeout_wall:
                rospy.logwarn("STEP: sim_timeout (step took too long)")
                return self._get_observation(), -2.0, True, True, {"outcome": "sim_timeout"}

            try:
                curr_sim = rospy.get_time()
            except Exception:
                rospy.logwarn("STEP: time_read_error")
                return self._get_observation(), -2.0, True, True, {"outcome": "time_read_error"}

            if curr_sim != last_sim_time:
                last_clock_update = time.time()
                last_sim_time = curr_sim
            elif time.time() - last_clock_update > 2.0:
                rospy.logwarn("STEP: clock_frozen (sim not advancing)")
                return self._get_observation(), -2.0, True, True, {"outcome": "clock_frozen"}

            if curr_sim - t0_sim >= self.step_dt:
                break

            time.sleep(0.01)

        obs = self._get_observation()
        reward, terminated, truncated, info = self._compute_reward()
        return obs, reward, terminated, truncated, info

    def close(self):
        self._shutdown_ros()
        # FIX: also kill the persistent rosmaster on final close
        if self._rosmaster_proc is not None:
            try:
                os.killpg(os.getpgid(self._rosmaster_proc.pid), signal.SIGTERM)
                self._rosmaster_proc.wait(timeout=5)
            except Exception:
                pass
            self._rosmaster_proc = None

    # ======================================================================
    # Observation
    # ======================================================================
    def _get_observation(self):
        n_rays = len(self._laser_ranges)
        third = n_rays // 3
        ranges = np.clip(self._laser_ranges, 0.0, LASER_MAX_RANGE)

        min_front = float(np.min(ranges[third:2 * third]))
        min_left  = float(np.min(ranges[2 * third:]))
        min_right = float(np.min(ranges[:third]))

        heading_err = 0.0
        cross_track = 0.0
        path_curvature = 0.0

        if len(self._global_plan_xs) >= 2:
            plan_pts = np.array(list(zip(self._global_plan_xs, self._global_plan_ys)))
            robot_pt = np.array([self._robot_x, self._robot_y])
            dists = np.linalg.norm(plan_pts - robot_pt, axis=1)
            closest_idx = int(np.argmin(dists))
            cross_track = float(dists[closest_idx])

            look = min(closest_idx + 3, len(plan_pts) - 1)
            dx = plan_pts[look, 0] - self._robot_x
            dy = plan_pts[look, 1] - self._robot_y
            heading_err = _angle_diff(math.atan2(dy, dx), self._robot_yaw)

            curvatures = []
            end = min(closest_idx + 15, len(plan_pts) - 1)
            for i in range(closest_idx, end - 1):
                dx1 = plan_pts[i + 1, 0] - plan_pts[i, 0]
                dy1 = plan_pts[i + 1, 1] - plan_pts[i, 1]
                if i + 2 <= end:
                    dx2 = plan_pts[i + 2, 0] - plan_pts[i + 1, 0]
                    dy2 = plan_pts[i + 2, 1] - plan_pts[i + 1, 1]
                    curvatures.append(abs(_angle_diff(math.atan2(dy2, dx2), math.atan2(dy1, dx1))))
            path_curvature = float(np.mean(curvatures)) if curvatures else 0.0

        dist_to_goal = _compute_distance((self._robot_x, self._robot_y), self._goal_odom)
        n_nearby_norm = min(int(np.sum(ranges < 1.25)) / 50.0, 1.0)
        progress_rate = ((self._last_dist_to_goal - dist_to_goal) / max(self.step_dt, 0.01)
                         if self._last_dist_to_goal is not None else 0.0)

        return np.array([
            self._robot_v,
            self._robot_w,
            heading_err,
            cross_track,
            dist_to_goal,
            min_front / LASER_MAX_RANGE,
            min_left / LASER_MAX_RANGE,
            min_right / LASER_MAX_RANGE,
            path_curvature,
            n_nearby_norm,
            float(self._is_reversal),
            progress_rate,
            min(self._n_dynamic_obs / 5.0, 1.0),
            min(self._mpc_solve_time / 0.05, 1.0),
            float(self._solver_status),
        ], dtype=np.float32)

    # ======================================================================
    # Reward
    # ======================================================================
    def _compute_reward(self):
        dist_to_goal = _compute_distance((self._robot_x, self._robot_y), self._goal_odom)
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
            rospy.loginfo(f"COLLISION (dist: {dist_to_goal:.2f}m, elapsed: {elapsed:.1f}s)")
            return reward, terminated, truncated, info

        # --- Timeout ---
        if elapsed >= self.max_episode_time:
            terminated = True
            truncated = True
            info["outcome"] = "timeout"
            reward = -20.0
            self._last_dist_to_goal = dist_to_goal
            rospy.loginfo(f"TIMEOUT (dist: {dist_to_goal:.2f}m)")
            return reward, terminated, truncated, info

        # --- Goal reached ---
        if dist_to_goal < 1.0:
            terminated = True
            info["outcome"] = "success"
            time_bonus = max(0.0, 1.0 - elapsed / self.max_episode_time) * 50.0
            reward = 100.0 + time_bonus
            self._last_dist_to_goal = dist_to_goal
            rospy.loginfo(f"SUCCESS! (elapsed: {elapsed:.1f}s, reward: {reward:.1f})")
            return reward, terminated, truncated, info

        # --- Stuck detection (every 5 seconds) ---
        self._steps_since_progress += 1
        if self._steps_since_progress >= 25:
            if self._last_progress_check_dist - dist_to_goal < 0.3:
                self._stuck_counter += 1
                if self._stuck_counter >= 3:
                    terminated = True
                    truncated = True
                    info["outcome"] = "stuck"
                    reward = -30.0
                    self._last_dist_to_goal = dist_to_goal
                    rospy.loginfo(f"STUCK (dist: {dist_to_goal:.2f}m)")
                    return reward, terminated, truncated, info
            else:
                self._stuck_counter = 0
            self._last_progress_check_dist = dist_to_goal
            self._steps_since_progress = 0

        # --- Dense reward ---
        progress = (self._last_dist_to_goal - dist_to_goal) if self._last_dist_to_goal is not None else 0.0
        r_progress = 10.0 * progress
        r_time = -0.05

        min_range = float(np.min(np.clip(self._laser_ranges, 0.0, LASER_MAX_RANGE)))
        if min_range < 0.5:
            r_safety = -5.0
        elif min_range < 0.75:
            r_safety = -2.0
        elif min_range < 1.0:
            r_safety = -0.5
        else:
            r_safety = 0.1

        r_smooth = -0.1 * abs(self._robot_w)
        r_solver = -10.0 if self._solver_status != 0 else 0.0
        r_solve_time = -2.0 if self._mpc_solve_time > 0.05 else 0.0

        if self._is_reversal:
            self._steps_in_reversal += 1
            if self._steps_in_reversal > 50:
                r_reversal = -5.0
            elif self._steps_in_reversal > 25:
                r_reversal = -2.0
            else:
                r_reversal = -0.5
        else:
            self._steps_in_reversal = 0
            r_reversal = 0.0

        reward = r_progress + r_time + r_safety + r_smooth + r_solver + r_solve_time + r_reversal
        self._last_dist_to_goal = dist_to_goal
        info["dist_to_goal"] = dist_to_goal
        info["elapsed"] = elapsed
        info["solver_status"] = self._solver_status
        return reward, terminated, truncated, info

    def _wait_for_clock(self, timeout=30.0):
        """Block until sim time is actually advancing. Prevents clock_frozen
        on the very first step after a reset."""
        rospy.loginfo("Waiting for sim clock to tick...")
        t0 = time.time()
        t_last = rospy.get_time()
        while time.time() - t0 < timeout:
            time.sleep(0.1)
            t_now = rospy.get_time()
            if t_now > t_last:
                rospy.loginfo(f"Sim clock is ticking (sim_t={t_now:.2f}s)")
                return
            t_last = t_now
        rospy.logwarn("Sim clock never started ticking — proceeding anyway")

    # ======================================================================
    # ROS helpers
    # ======================================================================
    def _publish_weights(self, w_pos, w_head, w_acc):
        if self._pub_weights is None:
            self._pub_weights = rospy.Publisher("/mpc_weights", Float64MultiArray, queue_size=1, latch=True)
        msg = Float64MultiArray()
        msg.data = [w_pos, w_head, w_acc]
        self._pub_weights.publish(msg)

    def _send_goal(self, goal_offset):
        if self._nav_client is None:
            self._nav_client = actionlib.SimpleActionClient('/move_base', MoveBaseAction)
            rospy.loginfo("Waiting for move_base action server...")
            self._nav_client.wait_for_server(timeout=rospy.Duration(30.0))
            rospy.loginfo("Connected to move_base action server")

        mb_goal = MoveBaseGoal()
        mb_goal.target_pose.header.frame_id = 'odom'
        mb_goal.target_pose.header.stamp = rospy.Time.now()
        mb_goal.target_pose.pose.position.x = goal_offset[0]
        mb_goal.target_pose.pose.position.y = goal_offset[1]
        mb_goal.target_pose.pose.position.z = 0.0
        mb_goal.target_pose.pose.orientation = Quaternion(0, 0, 0, 1)
        self._nav_client.send_goal(mb_goal)
        rospy.loginfo(f"Goal sent: ({goal_offset[0]:.2f}, {goal_offset[1]:.2f})")

    def _setup_subscribers(self):
        if self._subs_created:
            return
        self._sub_odom = rospy.Subscriber("/odometry/filtered", Odometry, self._cb_odom, queue_size=1)
        self._sub_scan = rospy.Subscriber("/front/scan", LaserScan, self._cb_scan, queue_size=1)
        self._sub_plan = rospy.Subscriber("/move_base/TrajectoryPlannerROS/global_plan", Path, self._cb_plan, queue_size=1)
        self._sub_collision = rospy.Subscriber("/collision", Bool, self._cb_collision, queue_size=1)
        self._sub_mpc_diag = rospy.Subscriber("/mpc_diagnostics", Float64MultiArray, self._cb_mpc_diag, queue_size=1)
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
        if len(msg.data) >= 4:
            self._mpc_solve_time = msg.data[0] / 1000.0
            self._is_reversal = msg.data[1] > 0.5
            self._n_dynamic_obs = int(msg.data[2])
            self._solver_status = int(msg.data[3])

    # ======================================================================
    # Gazebo / nav stack lifecycle
    # ======================================================================

    def _ensure_rosmaster(self):
        """Start a persistent rosmaster on this port if not already running.
        This master survives episode resets — only Gazebo and the nav stack
        are torn down between episodes."""
        try:
            # Quick probe: if rostopic list succeeds the master is alive
            result = subprocess.run(
                ["rostopic", "list"],
                env={**os.environ,
                     "ROS_MASTER_URI": f"http://localhost:{self.ros_master_port}",
                     "ROS_HOSTNAME": "localhost"},
                timeout=2,
                capture_output=True,
            )
            if result.returncode == 0:
                return  # master already up
        except Exception:
            pass

        # Kill any stale master on this port first
        if self._rosmaster_proc is not None:
            try:
                self._rosmaster_proc.terminate()
                self._rosmaster_proc.wait(timeout=3)
            except Exception:
                pass
            self._rosmaster_proc = None

        env = os.environ.copy()
        env["ROS_MASTER_URI"] = f"http://localhost:{self.ros_master_port}"
        env["ROS_HOSTNAME"] = "localhost"
        env["ROS_IP"] = "127.0.0.1"

        log_dir = os.path.expanduser(f"~/.ros/sac_logs/env_{self.ros_master_port}")
        os.makedirs(log_dir, exist_ok=True)

        self._rosmaster_proc = subprocess.Popen(
            ["rosmaster", "--core", "-p", str(self.ros_master_port)],
            preexec_fn=os.setsid,
            env=env,
            stdout=open(f"{log_dir}/rosmaster.log", "w"),
            stderr=subprocess.STDOUT,
        )
        time.sleep(1.5)
        print(f"[barn_env] Started persistent rosmaster on port {self.ros_master_port}")

    def _launch_gazebo(self, world_name):
        os.environ["JACKAL_LASER"] = "1"
        os.environ["JACKAL_LASER_MODEL"] = "ust10"
        os.environ["JACKAL_LASER_OFFSET"] = "-0.065 0 0.01"
        os.environ["GAZEBO_PLUGIN_PATH"] = join(self._helper_path, "plugins")
        os.environ["GAZEBO_MASTER_URI"] = f"http://localhost:{self.gazebo_master_port}"

        launch_file = join(self._helper_path, "launch", "gazebo_launch.launch")
        world_path = join(self._helper_path, "worlds", world_name)

        env = os.environ.copy()
        env["ROS_MASTER_URI"] = f"http://localhost:{self.ros_master_port}"
        env["GAZEBO_MASTER_URI"] = f"http://localhost:{self.gazebo_master_port}"
        env["ROS_HOSTNAME"] = "localhost"
        env["ROS_IP"] = "127.0.0.1"

        log_dir = os.path.expanduser(f"~/.ros/sac_logs/env_{self.ros_master_port}")
        os.makedirs(log_dir, exist_ok=True)

        self._gazebo_proc = subprocess.Popen(
            ["roslaunch", "-p", str(self.ros_master_port), launch_file,
             "world_name:=" + world_path,
             "gui:=" + ("true" if self.gui else "false"),
             "rviz:=false"],
            preexec_fn=os.setsid,
            env=env,
            stdout=open(f"{log_dir}/gazebo.log", "w"),
            stderr=subprocess.STDOUT,
        )
        time.sleep(3)

        if self._gazebo_proc.poll() is not None and self._gazebo_proc.returncode != 0:
            raise RuntimeError(f"Gazebo failed to start (exit code: {self._gazebo_proc.returncode})")

    def _wait_for_gazebo(self):
        """Wait for Gazebo with a hard 60s timeout and crash detection."""
        max_wait = 60
        start = time.time()
        while time.time() - start < max_wait:
            if self._gazebo_proc is not None and self._gazebo_proc.poll() is not None:
                raise RuntimeError(f"Gazebo crashed during startup (exit: {self._gazebo_proc.returncode})")
            try:
                rospy.wait_for_service("/gazebo/get_model_state", timeout=5.0)
                return  # success
            except rospy.ROSException:
                rospy.logwarn("Still waiting for Gazebo...")
        raise RuntimeError("Gazebo failed to start within 60 seconds")

    def _reset_robot(self):
        reset_srv = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
        ms = _create_model_state(self._init_pos[0], self._init_pos[1], 0.0, self._init_pos[2])
        for _ in range(5):
            try:
                reset_srv(ms)
                time.sleep(0.5)
                get_state = rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)
                state = get_state("jackal", "world")
                pos = state.pose.position
                if _compute_distance((pos.x, pos.y), (self._init_pos[0], self._init_pos[1])) < 0.1:
                    break
            except rospy.ServiceException:
                time.sleep(1)

    def _launch_nav_stack(self):
        rospack = rospkg.RosPack()
        launch_path = None
        for pkg_name in rospack.list():
            candidate = join(rospack.get_path(pkg_name), "launch", self.launch_file)
            if os.path.isfile(candidate):
                launch_path = candidate
                break
        if launch_path is None:
            raise FileNotFoundError(f"launch/{self.launch_file} not found in any ROS package")

        env = os.environ.copy()
        env["ROS_MASTER_URI"] = f"http://localhost:{self.ros_master_port}"
        env["GAZEBO_MASTER_URI"] = f"http://localhost:{self.gazebo_master_port}"
        env["ROS_HOSTNAME"] = "localhost"
        env["ROS_IP"] = "127.0.0.1"

        log_dir = os.path.expanduser(f"~/.ros/sac_logs/env_{self.ros_master_port}")
        os.makedirs(log_dir, exist_ok=True)

        self._nav_proc = subprocess.Popen(
            ["roslaunch", "-p", str(self.ros_master_port), launch_path],
            preexec_fn=os.setsid,
            env=env,
            stdout=open(f"{log_dir}/nav.log", "w"),
            stderr=subprocess.STDOUT,
        )
        # DynaBARN worlds (index >= 300) have more nodes and need longer to start
        nav_sleep = 6.0 if self._current_world_idx >= 300 else 3.0
        rospy.loginfo(f"Waiting {nav_sleep}s for nav stack (world {self._current_world_idx})...")
        time.sleep(nav_sleep)

    def _shutdown_ros(self):
        """Shut down Gazebo and the nav stack, but leave the rosmaster running."""
        if self._shutdown_in_progress:
            return
        self._shutdown_in_progress = True
        try:
            # Unsubscribe all topics
            for attr in ['_sub_odom', '_sub_scan', '_sub_plan', '_sub_collision', '_sub_mpc_diag']:
                if hasattr(self, attr):
                    try:
                        getattr(self, attr).unregister()
                    except Exception:
                        pass
            self._subs_created = False

            self._cleanup_ros_logs()

            # --- Kill nav stack ---
            if self._nav_proc is not None:
                try:
                    if self._nav_proc.poll() is None:          # FIX: only kill if alive
                        os.killpg(os.getpgid(self._nav_proc.pid), signal.SIGTERM)
                        try:
                            self._nav_proc.wait(timeout=15)
                        except subprocess.TimeoutExpired:
                            os.killpg(os.getpgid(self._nav_proc.pid), signal.SIGKILL)
                            self._nav_proc.wait(timeout=5)
                except (ProcessLookupError, OSError):
                    pass  # already dead, that's fine
                except Exception as e:
                    print(f"Error killing nav_proc: {e}")

            # --- Kill Gazebo ---
            if self._gazebo_proc is not None:
                try:
                    if self._gazebo_proc.poll() is None:       # FIX: only kill if alive
                        os.killpg(os.getpgid(self._gazebo_proc.pid), signal.SIGTERM)
                        try:
                            self._gazebo_proc.wait(timeout=20)
                        except subprocess.TimeoutExpired:
                            os.killpg(os.getpgid(self._gazebo_proc.pid), signal.SIGKILL)
                            self._gazebo_proc.wait(timeout=5)
                except (ProcessLookupError, OSError):
                    pass  # already dead, that's fine
                except Exception as e:
                    print(f"Error killing gazebo_proc: {e}")

            self._gazebo_proc = None
            self._nav_proc = None
            self._nav_client = None  # Force reconnect on next episode

            # Kill any lingering ROS nodes (but NOT rosmaster)
            for node_name in ["move_base", "obstacle_extractor", "obstacle_tracker",
                               "teamrobo2026", "laser_scan_to_point_cloud", "map_to_cloud",
                               "robot_state_publisher", "ekf_localization", "twist_mux",
                               "controller_spawner", "collision_publisher_node"]:
                try:
                    subprocess.run(["rosnode", "kill", f"/{node_name}"],
                                   stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL, timeout=3)
                except Exception:
                    pass

            time.sleep(3.0)
            self._cleanup_gazebo_resources()

            # FIX: only kill gzserver/gzclient, NOT roslaunch (which would take rosmaster with it)
            for proc in ["gzserver", "gzclient"]:
                try:
                    subprocess.run(["killall", "-15", proc],
                                   stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL, timeout=3)
                except Exception:
                    pass
            time.sleep(1.5)
            for proc in ["gzserver", "gzclient"]:
                try:
                    subprocess.run(["killall", "-9", proc],
                                   stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL, timeout=2)
                except Exception:
                    pass

            time.sleep(2.0)
        finally:
            self._shutdown_in_progress = False

    def _cleanup_gazebo_resources(self):
        import glob
        try:
            # FIX: widen semaphore pattern to catch all Gazebo/boost shared memory
            for sem_file in glob.glob("/dev/shm/sem.*"):
                try:
                    os.remove(sem_file)
                except Exception:
                    pass
            for shm_file in glob.glob("/dev/shm/boost_interprocess*"):
                try:
                    os.remove(shm_file)
                except Exception:
                    pass
            gazebo_tmp = os.path.expanduser("~/.gazebo/tmp")
            if os.path.exists(gazebo_tmp):
                for f in glob.glob(os.path.join(gazebo_tmp, "*")):
                    try:
                        if os.path.isfile(f):
                            os.remove(f)
                    except Exception:
                        pass
        except Exception:
            pass

    def _cleanup_ros_logs(self):
        import glob, shutil
        try:
            ros_log_dir = os.path.expanduser("~/.ros/log")
            if not os.path.exists(ros_log_dir):
                return
            sessions = sorted(
                [d for d in glob.glob(os.path.join(ros_log_dir, "*"))
                 if os.path.isdir(d) and d != os.path.join(ros_log_dir, "latest")],
                key=os.path.getmtime
            )
            for old_session in sessions[:-3]:
                try:
                    shutil.rmtree(old_session, ignore_errors=True)
                except Exception:
                    pass
        except Exception:
            pass

    def _cleanup_stray_processes(self):
        for node_name in ["move_base", "obstacle_extractor", "obstacle_tracker",
                          "teamrobo2026", "laser_scan_to_point_cloud", "map_to_cloud",
                          "robot_state_publisher", "ekf_localization", "twist_mux",
                          "controller_spawner", "collision_publisher_node", "gazebo"]:
            try:
                subprocess.run(["rosnode", "kill", f"/{node_name}"],
                               stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL, timeout=1)
            except Exception:
                pass
        for proc_name in ["gzserver", "gzclient"]:      # FIX: removed roslaunch from stray cleanup
            try:
                subprocess.run(["killall", "-9", proc_name],
                               stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL, timeout=1)
            except Exception:
                pass
        time.sleep(1.0)

    def _hard_restart_gazebo(self):
        rospy.logwarn("Hard Gazebo restart...")
        self._shutdown_ros()
        time.sleep(5.0)
        self._cleanup_stray_processes()
        self._cleanup_gazebo_resources()
        self._cleanup_ros_logs()
        try:
            subprocess.run(["sh", "-c", "rm -f /dev/shm/sem.* /dev/shm/boost_interprocess* 2>/dev/null || true"],
                           stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL, timeout=5)
        except Exception:
            pass
        gc.collect()
        try:
            import psutil
            mem = psutil.virtual_memory()
            rospy.loginfo(f"Memory: {mem.percent:.1f}% used ({mem.available / (1024**3):.1f} GB free)")
        except ImportError:
            pass
        # FIX: make sure rosmaster is still alive after hard restart
        self._ensure_rosmaster()
        time.sleep(2.0)
        rospy.loginfo("Hard restart complete")

    def _check_memory_pressure(self):
        try:
            import psutil
            return psutil.virtual_memory().percent > 90
        except ImportError:
            return False


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
gym.register(
    id="BarnMpc-v0",
    entry_point="rl.barn_env:BarnMpcEnv",
)