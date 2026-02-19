#!/usr/bin/env python3
"""
BARN SAC MPC Weight Tuning Environment
=======================================
Gymnasium environment for training SAC to tune MPC weights online.
Follows the run.py pattern for Gazebo lifecycle management.

Observation (145-dim):
    - Local costmap patch  10x10 = 100  (normalised 0-1)
    - Lidar rays           36           (normalised 0-1, min-in-sector)
    - Robot state          5            (dist_goal, sin/cos heading err, v, w)
    - Current MPC weights  4            (normalised 0-1)

Action (4-dim, continuous [-1,1]):
    Rescaled to:
        weight_position     [1, 100]
        weight_heading      [1, 100]
        weight_velocity     [0.1, 50]
        weight_acceleration [0.0001, 1.0]

Episode terminates on: goal reached | collision | timeout (100 s).
"""

import os
import sys
import time
import signal
import subprocess

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import rospy
import rospkg
from std_msgs.msg import Float64MultiArray, String
from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Quaternion

# ---------------------------------------------------------------------------
# Import GazeboSimulation from the-barn-challenge-robo (same as run.py)
# ---------------------------------------------------------------------------
_rospack = rospkg.RosPack()
_barn_path = os.path.dirname(_rospack.get_path("jackal_helper"))
if _barn_path not in sys.path:
    sys.path.insert(0, _barn_path)
from gazebo_simulation import GazeboSimulation  # noqa: E402

# ===========================================================================
# Constants
# ===========================================================================
INIT_POSITION_STATIC  = [-2.25, 3, 1.57]
GOAL_OFFSET_STATIC    = [0, 10]
INIT_POSITION_DYNAMIC = [11, 0, 3.14]
GOAL_OFFSET_DYNAMIC   = [-20, 0]

# Weight bounds ── (min, max) per weight
WEIGHT_BOUNDS = np.array([
    [1.0,    100.0],   # position
    [1.0,    100.0],   # heading
    [0.1,     50.0],   # velocity
    [0.0001,   1.0],   # acceleration
], dtype=np.float32)

DEFAULT_WEIGHTS = np.array([49.0, 37.0, 10.0, 0.0021], dtype=np.float32)

# Lidar specs (Hokuyo UST-10LX, 270° FOV, 0.5-10 m)
LIDAR_MAX_RANGE = 10.0
LIDAR_MIN_RANGE = 0.5
LIDAR_FOV_DEG   = 270.0

# Observation dimensions
COSTMAP_SIZE  = 10         # 10x10 local costmap patch
N_LIDAR_RAYS  = 36         # downsampled from ~1081 rays
N_ROBOT_STATE = 5          # dist_goal, sin_h, cos_h, v, w
N_WEIGHTS     = 4          # current MPC weights
OBS_DIM       = COSTMAP_SIZE * COSTMAP_SIZE + N_LIDAR_RAYS + N_ROBOT_STATE + N_WEIGHTS  # 145


def _compute_distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def _find_ros_file(subdir, filename):
    """Locate *subdir/filename* inside any installed ROS package."""
    for pkg in _rospack.list():
        candidate = os.path.join(_rospack.get_path(pkg), subdir, filename)
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(f"{subdir}/{filename} not found in any ROS package")


# ===========================================================================
# Environment
# ===========================================================================
class BarnEnv(gym.Env):
    """
    Gymnasium wrapper around the BARN challenge for SAC MPC weight tuning.

    Lifecycle (mirrors run.py):
        __init__  → set up ROS, define spaces
        reset()   → (re)launch Gazebo & nav-stack, reset robot, send goal
        step()    → publish weights, let MPC act, observe, reward
        close()   → kill sub-processes
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        world_idx: int = 0,
        gui: bool = False,
        rviz: bool = False,
        rviz_config: str = "mpc.rviz",
        launch_file: str = "move_base_mlda_2026.launch",
        step_duration: float = 1.0,
        max_episode_time: float = 100.0,
        goal_threshold: float = 1.0,
        mpc_verbose: bool = False,
        random_worlds: bool = False,
        world_range: tuple = (0, 300),
        episodes_per_world: int = 5,
    ):
        super().__init__()

        # ── Config ─────────────────────────────────────────────────────────
        self.world_idx = world_idx
        self.gui = gui
        self.rviz = rviz
        self.rviz_config = rviz_config
        self.launch_file = launch_file
        self.step_duration = step_duration
        self.max_episode_time = max_episode_time
        self.goal_threshold = goal_threshold
        self.mpc_verbose = mpc_verbose
        self.random_worlds = random_worlds
        self.world_range = world_range
        self.episodes_per_world = episodes_per_world

        # ── Spaces ─────────────────────────────────────────────────────────
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(N_WEIGHTS,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
        )

        # ── Internal state ─────────────────────────────────────────────────
        self.current_weights = DEFAULT_WEIGHTS.copy()
        self.prev_dist_to_goal = None
        self.episode_start_time = None
        self._current_world_idx = None
        self._episode_count = 0

        # Sub-processes
        self.gazebo_process = None
        self.nav_process = None
        self.gazebo_sim = None

        # World geometry
        self.init_position = None
        self.goal_offset = None
        self.init_coor = None
        self.goal_coor = None

        # Cached sensor data (written by callbacks)
        self._costmap_data = None
        self._costmap_info = None
        self._lidar_msg = None
        self._odom_msg = None
        self._mpc_verbose_data = None

        # actionlib client (lazy-init)
        self._nav_client = None

        # ── ROS initialisation ─────────────────────────────────────────────
        self._init_ros()

    # ======================================================================
    # ROS setup
    # ======================================================================
    def _init_ros(self):
        try:
            rospy.init_node("sac_barn_env", anonymous=True, disable_signals=True)
        except rospy.exceptions.ROSException:
            pass  # node already initialised
        rospy.set_param("/use_sim_time", True)

        # Publisher → MPC weight update
        self.weight_pub = rospy.Publisher(
            "/mpc/weight_update", Float64MultiArray, queue_size=1, latch=True
        )

        # Subscribers
        rospy.Subscriber(
            "/move_base/local_costmap/costmap", OccupancyGrid, self._cb_costmap
        )
        rospy.Subscriber("/front/scan", LaserScan, self._cb_lidar)
        rospy.Subscriber("/odometry/filtered", Odometry, self._cb_odom)

        if self.mpc_verbose:
            rospy.Subscriber("/mpc/verbose", String, self._cb_verbose)

    # ── Sensor callbacks ──────────────────────────────────────────────────
    def _cb_costmap(self, msg):
        self._costmap_data = np.array(msg.data, dtype=np.float32).reshape(
            msg.info.height, msg.info.width
        )
        self._costmap_info = msg.info

    def _cb_lidar(self, msg):
        self._lidar_msg = msg

    def _cb_odom(self, msg):
        self._odom_msg = msg

    def _cb_verbose(self, msg):
        self._mpc_verbose_data = msg.data

    # ======================================================================
    # Process management  (mirrors run.py §0-§1)
    # ======================================================================
    def _kill_process(self, proc):
        if proc is None:
            return
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=5)
        except Exception:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                pass

    def _launch_gazebo(self, world_idx):
        """Launch Gazebo with the specified BARN world (run.py §0)."""
        self._kill_process(self.gazebo_process)
        self._kill_process(self.nav_process)
        self.nav_process = None
        time.sleep(2)

        os.environ["JACKAL_LASER"] = "1"
        os.environ["JACKAL_LASER_MODEL"] = "ust10"
        os.environ["JACKAL_LASER_OFFSET"] = "-0.065 0 0.01"

        base_path = _rospack.get_path("jackal_helper")
        os.environ["GAZEBO_PLUGIN_PATH"] = os.path.join(base_path, "plugins")

        if world_idx < 300:
            world_file = f"BARN/world_{world_idx}.world"
            self.init_position = list(INIT_POSITION_STATIC)
            self.goal_offset = list(GOAL_OFFSET_STATIC)
        elif world_idx < 360:
            world_file = f"DynaBARN/world_{world_idx - 300}.world"
            self.init_position = list(INIT_POSITION_DYNAMIC)
            self.goal_offset = list(GOAL_OFFSET_DYNAMIC)
        else:
            raise ValueError(f"World index {world_idx} is out of range")

        world_path = os.path.join(base_path, "worlds", world_file)
        launch_path = os.path.join(base_path, "launch", "gazebo_launch.launch")
        rviz_cfg = _find_ros_file("configs", self.rviz_config)

        self.gazebo_process = subprocess.Popen(
            [
                "roslaunch", launch_path,
                f"world_name:={world_path}",
                f"gui:={'true' if self.gui else 'false'}",
                f"rviz:={'true' if self.rviz else 'false'}",
                f"rviz_config:={rviz_cfg}",
            ],
            preexec_fn=os.setsid,
        )
        time.sleep(5)

        self.init_coor = (self.init_position[0], self.init_position[1])
        self.goal_coor = (
            self.init_position[0] + self.goal_offset[0],
            self.init_position[1] + self.goal_offset[1],
        )
        self.gazebo_sim = GazeboSimulation(init_position=self.init_position)

    def _launch_nav_stack(self):
        """Launch move_base + MPC navigation stack (run.py §1)."""
        self._kill_process(self.nav_process)
        time.sleep(1)

        rospy.set_param("/teamrobo2026/mpc_verbose", self.mpc_verbose)

        launch_path = _find_ros_file("launch", self.launch_file)
        self.nav_process = subprocess.Popen(
            ["roslaunch", launch_path],
            preexec_fn=os.setsid,
        )
        self._nav_client = None  # force re-creation
        time.sleep(3)

    def _send_goal(self):
        """Send navigation goal via actionlib (run.py §1)."""
        import actionlib
        from move_base_msgs.msg import MoveBaseGoal, MoveBaseAction

        if self._nav_client is None:
            self._nav_client = actionlib.SimpleActionClient(
                "/move_base", MoveBaseAction
            )
        if not self._nav_client.wait_for_server(timeout=rospy.Duration(10)):
            rospy.logwarn("move_base action server unavailable")
            return False

        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "odom"
        goal.target_pose.pose.position.x = self.goal_offset[0]
        goal.target_pose.pose.position.y = self.goal_offset[1]
        goal.target_pose.pose.orientation = Quaternion(0, 0, 0, 1)
        self._nav_client.send_goal(goal)
        return True

    def _wait_for_topics(self, timeout=10.0):
        """Block until essential sensor topics start publishing."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self._odom_msg is not None and self._lidar_msg is not None:
                return True
            time.sleep(0.1)
        rospy.logwarn("Timeout waiting for sensor topics")
        return False

    # ======================================================================
    # Observation helpers
    # ======================================================================
    def _get_obs(self):
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        idx = 0

        # 1. Costmap patch (10×10)
        patch = self._get_costmap_patch()
        obs[idx : idx + COSTMAP_SIZE * COSTMAP_SIZE] = patch.ravel()
        idx += COSTMAP_SIZE * COSTMAP_SIZE

        # 2. Lidar (36 rays)
        obs[idx : idx + N_LIDAR_RAYS] = self._get_lidar_obs()
        idx += N_LIDAR_RAYS

        # 3. Robot state (5)
        obs[idx : idx + N_ROBOT_STATE] = self._get_robot_state()
        idx += N_ROBOT_STATE

        # 4. Current weights normalised to [0,1]
        w_lo, w_hi = WEIGHT_BOUNDS[:, 0], WEIGHT_BOUNDS[:, 1]
        obs[idx : idx + N_WEIGHTS] = (self.current_weights - w_lo) / (w_hi - w_lo + 1e-8)
        idx += N_WEIGHTS

        return np.clip(obs, 0.0, 1.0).astype(np.float32)

    def _get_costmap_patch(self):
        """10×10 patch of the local costmap centred on the robot."""
        patch = np.full((COSTMAP_SIZE, COSTMAP_SIZE), 0.5, dtype=np.float32)
        if self._costmap_data is None or self._odom_msg is None:
            return patch

        info = self._costmap_info
        rx = self._odom_msg.pose.pose.position.x
        ry = self._odom_msg.pose.pose.position.y
        cx = int((rx - info.origin.position.x) / info.resolution)
        cy = int((ry - info.origin.position.y) / info.resolution)

        h, w = self._costmap_data.shape
        half = COSTMAP_SIZE // 2
        for i in range(COSTMAP_SIZE):
            for j in range(COSTMAP_SIZE):
                mi, mj = cy - half + i, cx - half + j
                if 0 <= mi < h and 0 <= mj < w:
                    v = self._costmap_data[mi, mj]
                    patch[i, j] = max(0.0, v) / 100.0  # OccupancyGrid 0-100
        return patch

    def _get_lidar_obs(self):
        """Downsample lidar to N_LIDAR_RAYS (min-in-sector), normalised."""
        if self._lidar_msg is None:
            return np.ones(N_LIDAR_RAYS, dtype=np.float32)

        ranges = np.array(self._lidar_msg.ranges, dtype=np.float32)
        ranges = np.where(np.isfinite(ranges), ranges, LIDAR_MAX_RANGE)
        ranges = np.clip(ranges, LIDAR_MIN_RANGE, LIDAR_MAX_RANGE)

        n = len(ranges)
        sector = max(1, n // N_LIDAR_RAYS)
        out = np.ones(N_LIDAR_RAYS, dtype=np.float32)
        for i in range(N_LIDAR_RAYS):
            s, e = i * sector, min((i + 1) * sector, n)
            if s < n:
                out[i] = np.min(ranges[s:e])

        return np.clip(
            (out - LIDAR_MIN_RANGE) / (LIDAR_MAX_RANGE - LIDAR_MIN_RANGE), 0.0, 1.0
        )

    def _get_robot_state(self):
        """[dist_to_goal, sin(h_err), cos(h_err), v, w] all in [0,1]."""
        state = np.zeros(N_ROBOT_STATE, dtype=np.float32)
        if self._odom_msg is None:
            return state

        px = self._odom_msg.pose.pose.position.x
        py = self._odom_msg.pose.pose.position.y

        # Distance to goal (normalised by initial distance)
        dist = _compute_distance((px, py), self.goal_coor)
        max_dist = _compute_distance(self.init_coor, self.goal_coor) + 1e-6
        state[0] = np.clip(dist / max_dist, 0.0, 1.0)

        # Heading error
        dx, dy = self.goal_coor[0] - px, self.goal_coor[1] - py
        desired = np.arctan2(dy, dx)
        q = self._odom_msg.pose.pose.orientation
        yaw = np.arctan2(
            2.0 * (q.z * q.w + q.x * q.y),
            1.0 - 2.0 * (q.y ** 2 + q.z ** 2),
        )
        h_err = (desired - yaw + np.pi) % (2 * np.pi) - np.pi
        state[1] = (np.sin(h_err) + 1.0) / 2.0
        state[2] = (np.cos(h_err) + 1.0) / 2.0

        # Velocities
        v = self._odom_msg.twist.twist.linear.x
        w = self._odom_msg.twist.twist.angular.z
        state[3] = np.clip((v + 2.0) / 4.0, 0.0, 1.0)   # v ∈ [-2, 2]
        state[4] = np.clip((w + 1.8) / 3.6, 0.0, 1.0)    # w ∈ [-1.8, 1.8]
        return state

    # ======================================================================
    # Action helpers
    # ======================================================================
    @staticmethod
    def _rescale_action(action: np.ndarray) -> np.ndarray:
        """Map [-1,1]^4 → actual MPC weight ranges."""
        lo, hi = WEIGHT_BOUNDS[:, 0], WEIGHT_BOUNDS[:, 1]
        return lo + (action + 1.0) / 2.0 * (hi - lo)

    def _publish_weights(self, weights: np.ndarray):
        msg = Float64MultiArray()
        msg.data = weights.tolist()
        self.weight_pub.publish(msg)
        self.current_weights = weights.copy()

    # ======================================================================
    # Reward
    # ======================================================================
    def _compute_reward(self, dist_to_goal: float) -> float:
        """Per-step reward (terminal bonuses are added in step())."""
        reward = 0.0

        # Progress toward goal
        if self.prev_dist_to_goal is not None:
            progress = self.prev_dist_to_goal - dist_to_goal
            reward += 5.0 * progress

        # Small time penalty
        reward -= 0.1

        # Proximity bonus
        max_dist = _compute_distance(self.init_coor, self.goal_coor) + 1e-6
        reward += 0.05 * (1.0 - dist_to_goal / max_dist)

        # Forward velocity bonus
        if self._odom_msg is not None:
            v = self._odom_msg.twist.twist.linear.x
            reward += 0.1 * max(0.0, v)

        return reward

    # ======================================================================
    # Gym interface
    # ======================================================================
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._episode_count += 1

        # ── Pick world ────────────────────────────────────────────────────
        new_world = self.world_idx
        if self.random_worlds and self._episode_count % self.episodes_per_world == 0:
            new_world = self.np_random.integers(
                self.world_range[0], self.world_range[1]
            )
        if options and "world_idx" in options:
            new_world = options["world_idx"]

        # ── (Re)launch Gazebo if world changed ───────────────────────────
        if self._current_world_idx != new_world or self.gazebo_sim is None:
            self._current_world_idx = new_world
            self._launch_gazebo(new_world)
            self._launch_nav_stack()

        # ── Reset robot to init pose (run.py §0) ─────────────────────────
        attempts = 0
        collided = True
        curr = (999, 999)
        while (_compute_distance(self.init_coor, curr) > 0.1 or collided) and attempts < 10:
            self.gazebo_sim.reset()
            time.sleep(1)
            pos = self.gazebo_sim.get_model_state().pose.position
            curr = (pos.x, pos.y)
            collided = self.gazebo_sim.get_hard_collision()
            attempts += 1

        # ── Launch nav stack if needed ────────────────────────────────────
        if self.nav_process is None:
            self._launch_nav_stack()

        # ── Send goal ─────────────────────────────────────────────────────
        self._send_goal()

        # ── Reset weights to defaults ─────────────────────────────────────
        self._publish_weights(DEFAULT_WEIGHTS.copy())

        # ── Wait for sensor data ──────────────────────────────────────────
        self._wait_for_topics(timeout=10.0)
        time.sleep(1)

        # ── Init episode bookkeeping ──────────────────────────────────────
        self.episode_start_time = rospy.get_time()
        pos = self.gazebo_sim.get_model_state().pose.position
        self.prev_dist_to_goal = _compute_distance(
            (pos.x, pos.y), self.goal_coor
        )

        obs = self._get_obs()
        info = {
            "world_idx": self._current_world_idx,
            "init_position": self.init_position,
            "goal_position": self.goal_offset,
        }
        return obs, info

    def step(self, action):
        # 1. Rescale action → MPC weights, publish
        weights = self._rescale_action(np.asarray(action, dtype=np.float32))
        self._publish_weights(weights)

        # 2. Let MPC act with new weights for step_duration
        t0 = rospy.get_time()
        while rospy.get_time() - t0 < self.step_duration:
            time.sleep(0.05)

        # 3. Observe
        pos = self.gazebo_sim.get_model_state().pose.position
        curr = (pos.x, pos.y)
        dist_to_goal = _compute_distance(curr, self.goal_coor)
        collided = self.gazebo_sim.get_hard_collision()
        elapsed = rospy.get_time() - self.episode_start_time

        # 4. Reward
        reward = self._compute_reward(dist_to_goal)

        # 5. Termination (mirrors run.py §2)
        terminated = False
        truncated = False

        if collided:
            terminated = True
            reward += -100.0
        elif dist_to_goal < self.goal_threshold:
            terminated = True
            reward += 100.0
        elif elapsed >= self.max_episode_time:
            truncated = True
            reward += -50.0

        self.prev_dist_to_goal = dist_to_goal

        obs = self._get_obs()
        info = {
            "dist_to_goal": dist_to_goal,
            "elapsed": elapsed,
            "collided": collided,
            "weights": weights.tolist(),
        }
        if self.mpc_verbose and self._mpc_verbose_data:
            info["mpc_verbose"] = self._mpc_verbose_data

        return obs, float(reward), terminated, truncated, info

    def close(self):
        self._kill_process(self.nav_process)
        self.nav_process = None
        self._kill_process(self.gazebo_process)
        self.gazebo_process = None

    def render(self):
        pass  # handled by Gazebo GUI / RViz
