#!/usr/bin/env python3
"""
train.py
SAC training script for online MPC weight tuning.

Usage:
  python3 rl/train.py --total-timesteps 200000 --worlds 0-49 --seed 42
  python3 rl/train.py --resume checkpoints/sac_mpc_best.zip --total-timesteps 50000
"""

import argparse
import os
import time
import warnings

import numpy as np
import torch

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

import rl.barn_env  # registers BarnMpc-v0
import gymnasium as gym


# ============================================================================
# Hyperparameters
# ============================================================================
DEFAULT_HP = dict(
    learning_rate=3e-4,
    buffer_size=100_000,
    batch_size=256,
    gamma=0.99,
    tau=0.005,
    ent_coef="auto",
    target_entropy="auto",
    learning_starts=1000,
    train_freq=(1, "step"),
    gradient_steps=1,
    policy_kwargs=dict(
        net_arch=[32, 32],
        activation_fn=torch.nn.ReLU,
    ),
)


# ============================================================================
# Callbacks
# ============================================================================
class NavigationMetricCallback(BaseCallback):
    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "outcome" in info:
                outcome = info["outcome"]
                self.logger.record("barn/outcome_success",   int(outcome == "success"))
                self.logger.record("barn/outcome_collision", int(outcome == "collision"))
                self.logger.record("barn/outcome_timeout",   int(outcome == "timeout"))
                self.logger.record("barn/outcome_stuck",     int(outcome == "stuck"))
                self.logger.record("barn/outcome_sim_timeout",   int(outcome == "sim_timeout"))
                self.logger.record("barn/outcome_gazebo_crashed", int(outcome == "gazebo_crashed"))
                self.logger.record("barn/outcome_clock_frozen",   int(outcome == "clock_frozen"))
            if "dist_to_goal" in info:
                self.logger.record("barn/dist_to_goal", info["dist_to_goal"])
        return True


class WeightLogCallback(BaseCallback):
    def _on_step(self) -> bool:
        actions = self.locals.get("actions")
        if actions is not None and len(actions) > 0:
            from rl.barn_env import W_POS_LO, W_POS_HI, W_HEAD_LO, W_HEAD_HI, W_ACC_LO, W_ACC_HI, _rescale
            a = actions[0]
            self.logger.record("weights/position_error", _rescale(float(a[0]), W_POS_LO, W_POS_HI))
            self.logger.record("weights/heading_error",  _rescale(float(a[1]), W_HEAD_LO, W_HEAD_HI))
            self.logger.record("weights/acceleration",   _rescale(float(a[2]), W_ACC_LO, W_ACC_HI))
        return True


class HeartbeatCallback(BaseCallback):
    """Print a timestamped heartbeat every N steps so you can confirm training is alive."""
    def __init__(self, interval=200, verbose=0):
        super().__init__(verbose)
        self.interval = interval

    def _on_step(self) -> bool:
        if self.n_calls % self.interval == 0:
            print(f"[HEARTBEAT] step={self.num_timesteps}  time={time.strftime('%H:%M:%S')}", flush=True)
        return True


# ============================================================================
# Environment factory
# ============================================================================
def make_env(world_indices, rank, seed, gui=False, launch_file="move_base_mlda_2026.launch", ros_master_port=11311):
    def _init():
        env_port = ros_master_port + rank * 10
        for attempt in range(3):
            try:
                env = gym.make("BarnMpc-v0", world_indices=world_indices, gui=gui,
                               launch_file=launch_file, ros_master_port=env_port)
                env = Monitor(env)
                env.reset(seed=seed + rank)
                return env
            except Exception as e:
                print(f"[Env {rank}] Attempt {attempt+1}/3 failed: {e}")
                if attempt < 2:
                    time.sleep(5)
                else:
                    raise RuntimeError(f"Failed to create env {rank} after 3 attempts")
    return _init


# ============================================================================
# Export utilities
# ============================================================================
def export_onnx(model, save_path="checkpoints/sac_policy.onnx"):
    import torch.onnx
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    obs_dim = model.observation_space.shape[0]
    dummy_obs = torch.randn(1, obs_dim, device=model.policy.device)
    torch.onnx.export(
        model.policy.actor, dummy_obs, save_path,
        input_names=["observation"],
        output_names=["action_mean", "action_log_std"],
        opset_version=13,
        dynamic_axes={"observation": {0: "batch"}, "action_mean": {0: "batch"}},
    )
    print(f"ONNX policy exported to {save_path}")


def export_torchscript(model, save_path="checkpoints/sac_policy.pt"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    sb3_path = save_path.replace(".pt", "_sb3")
    model.save(sb3_path)
    print(f"SB3 model saved to {sb3_path}")

    obs_dim = model.observation_space.shape[0]
    actor_net = model.policy.actor
    actor_net.eval()
    dummy = torch.randn(1, obs_dim).to(model.policy.device)
    try:
        traced = torch.jit.trace(actor_net, dummy)
        traced.save(save_path)
        print(f"TorchScript policy exported to {save_path}")
    except Exception as e:
        print(f"TorchScript trace failed ({e}), SB3 native save only.")


# ============================================================================
# Main
# ============================================================================
def parse_world_range(s):
    result = []
    for part in s.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            result.extend(range(int(lo), int(hi) + 1))
        else:
            result.append(int(part))
    return result


def main():
    parser = argparse.ArgumentParser(description="SAC training for MPC weight tuning")
    parser.add_argument("--total-timesteps", type=int, default=200_000)
    parser.add_argument("--worlds",      type=str, default="0,30,50,100,150,200,250,299,300,310,315,320,330,340,349")
    parser.add_argument("--eval-worlds", type=str, default="1,99,199,296,301,309,318,328,345")
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--gui",         action="store_true")
    parser.add_argument("--n-envs",      type=int, default=1)
    parser.add_argument("--resume",      type=str, default=None)
    parser.add_argument("--launch",      type=str, default="move_base_mlda_2026.launch")
    parser.add_argument("--log-dir",     type=str, default="rl_logs")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--export-onnx", action="store_true")
    args = parser.parse_args()

    train_worlds = parse_world_range(args.worlds)
    eval_worlds  = parse_world_range(args.eval_worlds)

    print("=" * 60)
    print("SAC MPC Weight Tuning — BARN Navigation")
    print("=" * 60)
    print(f"  Train worlds : {train_worlds[:5]}... ({len(train_worlds)} total)")
    print(f"  Eval worlds  : {eval_worlds[:5]}... ({len(eval_worlds)} total)")
    print(f"  Timesteps    : {args.total_timesteps:,}")
    print(f"  Parallel envs: {args.n_envs}")
    print(f"  Seed         : {args.seed}")
    print(f"  Device       : {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"  Policy arch  : {DEFAULT_HP['policy_kwargs']['net_arch']}")
    print("=" * 60)

    # NOTE: do NOT call rospy.init_node here — the env starts the rosmaster
    # in __init__ and initializes the ROS node itself on first reset.

    # Create training environments
    if args.n_envs > 1:
        print(f"Creating {args.n_envs} parallel training environments...")
        train_env = SubprocVecEnv([
            make_env(train_worlds, i, args.seed, gui=(args.gui and i == 0),
                     launch_file=args.launch, ros_master_port=11311)
            for i in range(args.n_envs)
        ])
    else:
        train_env = DummyVecEnv([
            make_env(train_worlds, 0, args.seed, gui=args.gui,
                     launch_file=args.launch, ros_master_port=11311)
        ])

    eval_env = SubprocVecEnv([
        make_env(eval_worlds, 0, args.seed + 100, gui=False,
                 launch_file=args.launch, ros_master_port=11311 + args.n_envs * 10)
    ])

    # Create or resume SAC model
    if args.resume:
        print(f"Resuming from {args.resume}")
        model = SAC.load(args.resume, env=train_env, device="auto")
    else:
        model = SAC("MlpPolicy", train_env, verbose=1, seed=args.seed,
                    device="auto", tensorboard_log=args.log_dir, **DEFAULT_HP)

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    callbacks = [
        NavigationMetricCallback(),
        WeightLogCallback(),
        HeartbeatCallback(interval=200),
        CheckpointCallback(save_freq=5000, save_path=args.checkpoint_dir, name_prefix="sac_mpc"),
        EvalCallback(eval_env, best_model_save_path=args.checkpoint_dir,
                     log_path=args.log_dir, eval_freq=10000,
                     n_eval_episodes=3, deterministic=True),
    ]

    print("\nStarting training...")
    t0 = time.time()

    warnings.filterwarnings("ignore", message="Training and eval env are not of the same type")

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
        log_interval=10,
        tb_log_name="sac_mpc_weights",
        reset_num_timesteps=args.resume is None,
    )
    print(f"\nTraining finished in {(time.time() - t0)/3600:.1f} hours")

    final_path = os.path.join(args.checkpoint_dir, "sac_mpc_final")
    model.save(final_path)
    print(f"Final model saved to {final_path}")

    if args.export_onnx:
        export_onnx(model, os.path.join(args.checkpoint_dir, "sac_policy.onnx"))
    export_torchscript(model, os.path.join(args.checkpoint_dir, "sac_policy.pt"))

    train_env.close()
    eval_env.close()
    print(f"\nDone. Deploy with:\n  rosrun teamrobo2026 weight_tuner_node.py --model {final_path}")


if __name__ == "__main__":
    main()