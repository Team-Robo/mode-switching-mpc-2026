#!/usr/bin/env python3
"""
Evaluate trained SAC policy on BARN worlds and compare with fixed-weight baseline.

Usage:
  # Evaluate RL-tuned weights on worlds 0-49:
  python3 evaluate.py --model checkpoints/sac_mpc_final --worlds 0-49

  # Compare with fixed baseline:
  python3 evaluate.py --baseline --worlds 0-49

  # Full benchmark (RL + baseline):
  python3 evaluate.py --model checkpoints/sac_mpc_final --baseline --worlds 0-49
"""

import argparse
import os
import sys
import time
import subprocess
import signal
import math
from os.path import join

import numpy as np

# Will be imported after rospy init
rospy = None
GazeboSimulation = None


def parse_world_range(s):
    result = []
    for part in s.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            result.extend(range(int(lo), int(hi) + 1))
        else:
            result.append(int(part))
    return result


def compute_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def path_coord_to_gazebo_coord(x, y):
    RADIUS = 0.075
    r_shift = -RADIUS - (30 * RADIUS * 2)
    c_shift = RADIUS + 5
    return (x * (RADIUS * 2) + r_shift, y * (RADIUS * 2) + c_shift)


def run_single_world(world_idx, launch_file, gui=False, with_weight_tuner=False,
                     model_path=None, onnx_path=None):
    """
    Run a single BARN world and return metrics.
    Returns dict: {success, collided, timeout, time, nav_metric}
    """
    import rospy
    import rospkg
    from gazebo_simulation import GazeboSimulation

    rospack = rospkg.RosPack()
    base_path = rospack.get_path("jackal_helper")

    # Determine world type
    if world_idx < 300:
        init_pos = [-2.25, 3.0, 1.57]
        goal_offset = [0.0, 10.0]
        world_name = f"BARN/world_{world_idx}.world"
    else:
        init_pos = [11.0, 0.0, 3.14]
        goal_offset = [-20.0, 0.0]
        world_name = f"DynaBARN/world_{world_idx - 300}.world"

    goal_world = (init_pos[0] + goal_offset[0], init_pos[1] + goal_offset[1])

    # Environment setup
    os.environ["JACKAL_LASER"] = "1"
    os.environ["JACKAL_LASER_MODEL"] = "ust10"
    os.environ["JACKAL_LASER_OFFSET"] = "-0.065 0 0.01"
    os.environ["GAZEBO_PLUGIN_PATH"] = join(base_path, "plugins")

    # Launch Gazebo
    gazebo_launch = join(base_path, "launch", "gazebo_launch.launch")
    world_path = join(base_path, "worlds", world_name)

    gazebo_proc = subprocess.Popen([
        "roslaunch", gazebo_launch,
        "world_name:=" + world_path,
        "gui:=" + ("true" if gui else "false"),
        "rviz:=false",
    ], preexec_fn=os.setsid)
    time.sleep(5)

    gazebo_sim = GazeboSimulation(init_position=init_pos)

    # Reset robot
    pos = gazebo_sim.get_model_state().pose.position
    curr = (pos.x, pos.y)
    while compute_distance((init_pos[0], init_pos[1]), curr) > 0.1:
        gazebo_sim.reset()
        time.sleep(1)
        pos = gazebo_sim.get_model_state().pose.position
        curr = (pos.x, pos.y)

    # Launch nav stack
    if with_weight_tuner:
        # Use RL launch file
        launch_path = None
        for pkg in rospack.list():
            candidate = join(rospack.get_path(pkg), "launch", "move_base_mlda_2026_rl.launch")
            if os.path.isfile(candidate):
                launch_path = candidate
                break
    else:
        launch_path = None
        for pkg in rospack.list():
            candidate = join(rospack.get_path(pkg), "launch", launch_file)
            if os.path.isfile(candidate):
                launch_path = candidate
                break

    if launch_path is None:
        raise FileNotFoundError(f"Launch file {launch_file} not found")

    nav_proc = subprocess.Popen(
        ["roslaunch", launch_path],
        preexec_fn=os.setsid,
    )
    time.sleep(4)

    # Send goal
    import actionlib
    from move_base_msgs.msg import MoveBaseGoal, MoveBaseAction
    from geometry_msgs.msg import Quaternion

    nav_as = actionlib.SimpleActionClient("/move_base", MoveBaseAction)
    nav_as.wait_for_server(timeout=rospy.Duration(10))

    mb_goal = MoveBaseGoal()
    mb_goal.target_pose.header.frame_id = "odom"
    mb_goal.target_pose.pose.position.x = goal_offset[0]
    mb_goal.target_pose.pose.position.y = goal_offset[1]
    mb_goal.target_pose.pose.orientation = Quaternion(0, 0, 0, 1)
    nav_as.send_goal(mb_goal)

    # Wait for start
    curr_time = rospy.get_time()
    pos = gazebo_sim.get_model_state().pose.position
    curr = (pos.x, pos.y)
    while compute_distance((init_pos[0], init_pos[1]), curr) < 0.1:
        curr_time = rospy.get_time()
        pos = gazebo_sim.get_model_state().pose.position
        curr = (pos.x, pos.y)
        time.sleep(0.01)

    # Navigate
    start_time = curr_time
    collided = False

    while (compute_distance(goal_world, curr) > 1.0
           and not collided
           and curr_time - start_time < 100):
        curr_time = rospy.get_time()
        pos = gazebo_sim.get_model_state().pose.position
        curr = (pos.x, pos.y)
        collided = gazebo_sim.get_hard_collision()
        time.sleep(0.05)

    # Results
    elapsed = curr_time - start_time
    success = compute_distance(goal_world, curr) <= 1.0 and not collided
    timeout = elapsed >= 100

    # Compute BARN nav metric
    if world_idx < 300:
        path_file = join(base_path, "worlds/BARN/path_files", f"path_{world_idx}.npy")
        path_arr = np.load(path_file)
        path_arr = [path_coord_to_gazebo_coord(*p) for p in path_arr]
        path_arr = np.insert(path_arr, 0, (init_pos[0], init_pos[1]), axis=0)
        path_arr = np.insert(path_arr, len(path_arr), goal_world, axis=0)
        path_length = sum(
            compute_distance(p1, p2)
            for p1, p2 in zip(path_arr[:-1], path_arr[1:])
        )
    else:
        path_length = abs(goal_offset[0])

    optimal_time = path_length / 2.0
    nav_metric = int(success) * optimal_time / np.clip(elapsed, 2 * optimal_time, 8 * optimal_time)

    # Cleanup
    try:
        os.killpg(os.getpgid(nav_proc.pid), signal.SIGTERM)
        nav_proc.wait(timeout=5)
    except Exception:
        pass
    try:
        os.killpg(os.getpgid(gazebo_proc.pid), signal.SIGTERM)
        gazebo_proc.wait(timeout=5)
    except Exception:
        pass

    return {
        "world": world_idx,
        "success": success,
        "collided": collided,
        "timeout": timeout,
        "time": elapsed,
        "nav_metric": nav_metric,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate SAC weight tuner")
    parser.add_argument("--worlds", type=str, default="0-9")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--onnx", type=str, default=None)
    parser.add_argument("--baseline", action="store_true", help="Also run fixed-weight baseline")
    parser.add_argument("--launch", type=str, default="move_base_mlda_2026.launch")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--out", type=str, default="eval_results.txt")
    args = parser.parse_args()

    worlds = parse_world_range(args.worlds)

    import rospy
    if not rospy.core.is_initialized():
        rospy.init_node("sac_evaluator", anonymous=True)

    results = {"rl": [], "baseline": []}

    # Run RL-tuned evaluation
    if args.model or args.onnx:
        print("=" * 60)
        print("Evaluating RL-tuned weights")
        print("=" * 60)
        for widx in worlds:
            print(f"  World {widx}...", end=" ", flush=True)
            try:
                r = run_single_world(
                    widx, args.launch, gui=args.gui,
                    with_weight_tuner=True,
                    model_path=args.model, onnx_path=args.onnx,
                )
                results["rl"].append(r)
                status = "OK" if r["success"] else ("CRASH" if r["collided"] else "TIMEOUT")
                print(f"{status} t={r['time']:.1f}s metric={r['nav_metric']:.4f}")
            except Exception as e:
                print(f"ERROR: {e}")
            time.sleep(2)

    # Run baseline evaluation
    if args.baseline:
        print("=" * 60)
        print("Evaluating fixed-weight baseline")
        print("=" * 60)
        for widx in worlds:
            print(f"  World {widx}...", end=" ", flush=True)
            try:
                r = run_single_world(
                    widx, args.launch, gui=args.gui,
                    with_weight_tuner=False,
                )
                results["baseline"].append(r)
                status = "OK" if r["success"] else ("CRASH" if r["collided"] else "TIMEOUT")
                print(f"{status} t={r['time']:.1f}s metric={r['nav_metric']:.4f}")
            except Exception as e:
                print(f"ERROR: {e}")
            time.sleep(2)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for label, res in results.items():
        if not res:
            continue
        n = len(res)
        successes = sum(1 for r in res if r["success"])
        collisions = sum(1 for r in res if r["collided"])
        timeouts = sum(1 for r in res if r["timeout"])
        avg_metric = np.mean([r["nav_metric"] for r in res])
        avg_time = np.mean([r["time"] for r in res if r["success"]]) if successes else 0

        print(f"\n  [{label.upper()}] ({n} worlds)")
        print(f"    Success rate : {successes}/{n} ({100*successes/n:.1f}%)")
        print(f"    Collisions   : {collisions}")
        print(f"    Timeouts     : {timeouts}")
        print(f"    Avg metric   : {avg_metric:.4f}")
        print(f"    Avg time (ok): {avg_time:.2f}s")

    # Save results
    with open(args.out, "w") as f:
        f.write("method,world,success,collided,timeout,time,nav_metric\n")
        for label, res in results.items():
            for r in res:
                f.write(f"{label},{r['world']},{r['success']},{r['collided']},"
                        f"{r['timeout']},{r['time']:.4f},{r['nav_metric']:.4f}\n")
    print(f"\nResults saved to {args.out}")


if __name__ == "__main__":
    main()
