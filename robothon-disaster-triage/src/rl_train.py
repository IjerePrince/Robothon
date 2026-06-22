"""
rl_train.py — PPO training for the Disaster Triage task via stable-baselines3.

Pipeline:
    1. Wrap DisasterTriageEnv in a Dictionary flattener (SB3 needs Box obs).
    2. Train PPO with curriculum (start with 1 object, ramp to 5).
    3. Periodically evaluate + checkpoint.
    4. Save final policy to checkpoints/ppo_disaster_triage.zip.

Usage:
    python -m src.rl_train --total-timesteps 200_000 --eval-freq 10_000

For quick smoke-test (no GPU needed):
    python -m src.rl_train --total-timesteps 5_000 --eval-freq 1_000
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path
from typing import Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import (
    CheckpointCallback, EvalCallback, BaseCallback
)
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import CombinedExtractor

from .env import DisasterTriageEnv
from . import CHECKPOINT_DIR, OBJECT_NAMES


# ---------------------------------------------------------------------------
# Observation flattener wrapper
# ---------------------------------------------------------------------------
class FlattenDictObs(gym.ObservationWrapper):
    """Flatten the dict observation into a single Box for SB3 compatibility.

    Order: tactile (6) + joint_pos (22) + joint_vel (22) + palm_pose (7)
           + object_poses flat (15) + current_object (5) + carrying (1)
         = 78 dims total.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        # Compute flat dim
        sample = self.observation_space["tactile"].shape[0]
        flat_dim = 0
        self._order = []
        for k, sp in self.observation_space.spaces.items():
            self._order.append(k)
            flat_dim += int(np.prod(sp.shape))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(flat_dim,), dtype=np.float32
        )

    def observation(self, obs):
        parts = []
        for k in self._order:
            v = obs[k].astype(np.float32).flatten()
            parts.append(v)
        return np.concatenate(parts).astype(np.float32)


# ---------------------------------------------------------------------------
# Curriculum callback
# ---------------------------------------------------------------------------
class CurriculumCallback(BaseCallback):
    """Increases task difficulty as training progresses.

    Stages (by total_timesteps):
        0   – 20%  : only the first object is the target (glass / fragile)
        20% – 50%  : first 3 objects
        50% – 100% : all 5 objects

    This is implemented by *editing* the env's `current_obj_idx` range via
    a per-env attribute.  In practice, the env cycles through 0..N active.
    """

    def __init__(self, total_timesteps: int, n_objects: int = 5, verbose: int = 0):
        super().__init__(verbose)
        self.total_timesteps = int(total_timesteps)
        self.n_objects = int(n_objects)
        self.stage = 0

    def _on_step(self) -> bool:
        progress = self.num_timesteps / self.total_timesteps
        new_stage = 0
        if progress > 0.20:
            new_stage = 1
        if progress > 0.50:
            new_stage = 2
        if new_stage != self.stage:
            self.stage = new_stage
            if self.verbose:
                stages = [1, 3, 5]
                print(f"[Curriculum] Stage {new_stage}: {stages[new_stage]} objects active "
                      f"(step {self.num_timesteps}/{self.total_timesteps})")
            # Update envs
            for env in self.training_env.envs:
                # Drill through wrappers to reach DisasterTriageEnv
                e = env
                while hasattr(e, "env"):
                    e = e.env
                if hasattr(e, "current_obj_idx"):
                    stages = [1, 3, 5]
                    e._curriculum_max_objects = stages[new_stage]
        return True


# ---------------------------------------------------------------------------
# Smoketest environment factory
# ---------------------------------------------------------------------------
def make_env(seed: int = 0, max_steps: int = 1500):
    def _init():
        env = DisasterTriageEnv(
            render_mode=None,
            max_steps=max_steps,
            reward_scale=1.0,
        )
        env.reset(seed=seed)
        return FlattenDictObs(env)
    return _init


# ---------------------------------------------------------------------------
# Training entrypoint
# ---------------------------------------------------------------------------
def train(
    total_timesteps: int = 100_000,
    eval_freq: int = 10_000,
    n_envs: int = 4,
    save_dir: Optional[Path] = None,
    policy_kwargs: Optional[dict] = None,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    seed: int = 42,
    verbose: int = 1,
) -> Path:
    """Train a PPO policy on the Disaster Triage task.

    Returns the path to the saved policy .zip file.
    """
    save_dir = Path(save_dir) if save_dir else CHECKPOINT_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[RL Train] PPO — {total_timesteps} steps, {n_envs} envs, seed={seed}")
    print(f"[RL Train] Save dir: {save_dir}")

    # ---- Build vectorised env ----
    venv = DummyVecEnv([make_env(seed + i) for i in range(n_envs)])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # ---- Eval env ----
    eval_env = DummyVecEnv([make_env(seed=999)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0,
                             training=False)

    # ---- Policy kwargs ----
    if policy_kwargs is None:
        policy_kwargs = dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
        )

    # ---- Model ----
    model = PPO(
        policy="MlpPolicy",
        env=venv,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        verbose=verbose,
        seed=seed,
        device="auto",
    )

    # ---- Callbacks ----
    checkpoint_cb = CheckpointCallback(
        save_freq=max(eval_freq // n_envs, 1),
        save_path=str(save_dir / "checkpoints"),
        name_prefix="ppo_disaster_triage",
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(save_dir / "best"),
        log_path=str(save_dir / "logs"),
        eval_freq=max(eval_freq // n_envs, 1),
        n_eval_episodes=3,
        deterministic=True,
    )
    curriculum_cb = CurriculumCallback(
        total_timesteps=total_timesteps, n_objects=5, verbose=1
    )

    # ---- Train ----
    t0 = time.time()
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_cb, eval_cb, curriculum_cb],
    )
    dt = time.time() - t0
    print(f"\n[RL Train] Done in {dt:.1f}s ({total_timesteps / dt:.0f} steps/sec)")

    # ---- Save final + VecNormalize stats ----
    final_path = save_dir / "ppo_disaster_triage.zip"
    model.save(str(final_path))
    venv.save(str(save_dir / "vec_normalize.pkl"))
    print(f"[RL Train] Final policy saved: {final_path}")
    return final_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train PPO on Disaster Triage task")
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--smoketest", action="store_true",
                        help="Quick 5k-step smoketest (no checkpoint)")
    args = parser.parse_args()

    if args.smoketest:
        args.total_timesteps = 5000
        args.eval_freq = 1000
        args.n_envs = 2

    save_dir = Path(args.save_dir) if args.save_dir else None
    train(
        total_timesteps=args.total_timesteps,
        eval_freq=args.eval_freq,
        n_envs=args.n_envs,
        seed=args.seed,
        learning_rate=args.lr,
        save_dir=save_dir,
    )


if __name__ == "__main__":
    main()
