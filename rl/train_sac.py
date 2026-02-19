#!/usr/bin/env python3
"""
SAC MPC Weight Tuner — Training Script
========================================
Uses Stable-Baselines3 SAC to learn MPC weight policies in the BARN challenge.

Usage (training on GPU):
    python3 train_sac.py --total_timesteps 200000 --random_worlds --mpc_verbose

Usage (resume):
    python train_sac.py --resume rl/models/sac_mpc_final --total_timesteps 50000

After training the script saves:
    rl/models/sac_mpc_final.zip   — SB3 model (load with SAC.load)
    rl/models/sac_mpc_actor.onnx  — ONNX actor  (for C++ weight_tuner_node)
"""

import argparse
import json
import os
import sys

import numpy as np

# Ensure the rl/ package is importable when invoked from any directory
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from barn_env import BarnEnv, OBS_DIM  # noqa: E402

from stable_baselines3 import SAC  # noqa: E402
from stable_baselines3.common.callbacks import (  # noqa: E402
    BaseCallback,
    CheckpointCallback,
)


# ===========================================================================
# Callbacks
# ===========================================================================
class MPCVerboseCallback(BaseCallback):
    """
    Log MPC verbose JSON data (published on /mpc/verbose) into TensorBoard
    so that weight evolution can be monitored in real time during training.
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._sub = None
        self._data = None

    def _on_training_start(self):
        import rospy
        from std_msgs.msg import String

        try:
            self._sub = rospy.Subscriber("/mpc/verbose", String, self._cb)
        except Exception:
            pass

    def _cb(self, msg):
        try:
            self._data = json.loads(msg.data)
        except Exception:
            pass

    def _on_step(self) -> bool:
        if self._data is not None:
            d = self._data
            self.logger.record("mpc/w_position", d.get("w_pos", 0))
            self.logger.record("mpc/w_heading", d.get("w_head", 0))
            self.logger.record("mpc/w_velocity", d.get("w_vel", 0))
            self.logger.record("mpc/w_acceleration", d.get("w_accel", 0))
            self.logger.record("mpc/v_opt", d.get("v", 0))
            self.logger.record("mpc/w_opt", d.get("w", 0))
            self.logger.record(
                "mpc/reversal", 1.0 if d.get("mode") == "REVERSAL" else 0.0
            )
        return True


class EpisodeLogCallback(BaseCallback):
    """Print a summary line at the end of every episode."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._ep_rewards = []
        self._ep_count = 0

    def _on_step(self) -> bool:
        # SB3 stores per-env info in self.locals
        infos = self.locals.get("infos", [])
        for info in infos:
            ep = info.get("episode")
            if ep is not None:
                self._ep_count += 1
                self.logger.record("episode/reward", ep["r"])
                self.logger.record("episode/length", ep["l"])
                print(
                    f"[Episode {self._ep_count}]  "
                    f"R={ep['r']:.2f}  L={ep['l']}  "
                    f"dist={info.get('dist_to_goal', '?'):.2f}  "
                    f"col={info.get('collided', '?')}"
                )
        return True


# ===========================================================================
# ONNX export
# ===========================================================================
def export_onnx(model, path: str, obs_dim: int):
    """Export the deterministic SAC actor to ONNX for C++ inference."""
    import torch
    import torch.nn as nn

    class DeterministicActor(nn.Module):
        """Wraps SB3 SAC actor → tanh(mu) for deterministic inference."""

        def __init__(self, sb3_actor):
            super().__init__()
            self.features_extractor = sb3_actor.features_extractor
            self.latent_pi = sb3_actor.latent_pi
            self.mu = sb3_actor.mu

        def forward(self, obs):
            features = self.features_extractor(obs)
            latent = self.latent_pi(features)
            return torch.tanh(self.mu(latent))

    actor = model.policy.actor
    det = DeterministicActor(actor)
    det.eval()

    dummy = torch.randn(1, obs_dim)
    torch.onnx.export(
        det,
        dummy,
        path,
        input_names=["observation"],
        output_names=["action"],
        dynamic_axes={"observation": {0: "batch"}, "action": {0: "batch"}},
        opset_version=11,
    )
    print(f"ONNX actor exported → {path}")


# ===========================================================================
# Main
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description="Train SAC for MPC weight tuning")

    # ── Training ───────────────────────────────────────────────────────────
    parser.add_argument("--total_timesteps", type=int, default=100_000)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--buffer_size", type=int, default=100_000)
    parser.add_argument("--ent_coef", type=str, default="auto")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)

    # ── Environment ────────────────────────────────────────────────────────
    parser.add_argument("--world_idx", type=int, default=0)
    parser.add_argument("--random_worlds", action="store_true")
    parser.add_argument("--world_range", type=int, nargs=2, default=[0, 300])
    parser.add_argument("--episodes_per_world", type=int, default=5)
    parser.add_argument("--step_duration", type=float, default=1.0)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--rviz", action="store_true")
    parser.add_argument("--mpc_verbose", action="store_true")

    # ── IO ─────────────────────────────────────────────────────────────────
    parser.add_argument("--save_dir", type=str, default="rl/models")
    parser.add_argument("--log_dir", type=str, default="rl/logs")
    parser.add_argument("--checkpoint_freq", type=int, default=5_000)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--no_onnx", action="store_true")

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # ── Environment ────────────────────────────────────────────────────────
    print("Creating BarnEnv …")
    env = BarnEnv(
        world_idx=args.world_idx,
        gui=args.gui,
        rviz=args.rviz,
        mpc_verbose=args.mpc_verbose,
        step_duration=args.step_duration,
        random_worlds=args.random_worlds,
        world_range=tuple(args.world_range),
        episodes_per_world=args.episodes_per_world,
    )

    # ── SAC model ──────────────────────────────────────────────────────────
    if args.resume:
        print(f"Resuming from {args.resume}")
        model = SAC.load(args.resume, env=env)
    else:
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            buffer_size=args.buffer_size,
            ent_coef=args.ent_coef,
            gamma=args.gamma,
            tau=args.tau,
            verbose=1,
            tensorboard_log=args.log_dir,
            device="auto",  # CUDA when available, else CPU
        )

    # ── Callbacks ──────────────────────────────────────────────────────────
    callbacks = [
        CheckpointCallback(
            save_freq=args.checkpoint_freq,
            save_path=args.save_dir,
            name_prefix="sac_mpc",
        ),
        EpisodeLogCallback(),
    ]
    if args.mpc_verbose:
        callbacks.append(MPCVerboseCallback())

    # ── Train ──────────────────────────────────────────────────────────────
    print(
        f"Starting SAC training — {args.total_timesteps} timesteps, "
        f"device={model.device}"
    )
    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callbacks,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")

    # ── Save ───────────────────────────────────────────────────────────────
    final_path = os.path.join(args.save_dir, "sac_mpc_final")
    model.save(final_path)
    print(f"Model saved → {final_path}.zip")

    # ── ONNX export ────────────────────────────────────────────────────────
    if not args.no_onnx:
        onnx_path = os.path.join(args.save_dir, "sac_mpc_actor.onnx")
        try:
            export_onnx(model, onnx_path, OBS_DIM)
        except Exception as e:
            print(f"ONNX export failed: {e}")

    env.close()
    print("Training complete!")


if __name__ == "__main__":
    main()
