"""
rl_eval.py — Load a trained PPO policy and evaluate it on the Disaster Triage task.

Usage:
    python -m src.rl_eval --policy checkpoints/ppo_disaster_triage.zip --episodes 5
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Optional

import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from .env import DisasterTriageEnv
from .rl_train import FlattenDictObs, make_env
from . import CHECKPOINT_DIR


def evaluate(
    policy_path: Optional[Path] = None,
    episodes: int = 5,
    deterministic: bool = True,
    render: bool = False,
    seed: int = 0,
) -> dict:
    """Evaluate a trained PPO policy.

    Returns dict with per-episode rewards, lengths, and success counts.
    """
    policy_path = Path(policy_path) if policy_path else (CHECKPOINT_DIR / "ppo_disaster_triage.zip")
    if not policy_path.exists():
        raise FileNotFoundError(
            f"Policy not found at {policy_path}.  Train one first:\n"
            f"  python -m src.rl_train"
        )
    vec_norm_path = policy_path.parent / "vec_normalize.pkl"

    # Build eval env (single env, not normalised)
    eval_env = DisasterTriageEnv(
        render_mode="human" if render else None,
        max_steps=1500,
    )
    flat_env = FlattenDictObs(eval_env)
    dummy = DummyVecEnv([lambda: flat_env])
    if vec_norm_path.exists():
        eval_venv = VecNormalize.load(str(vec_norm_path), dummy)
        eval_venv.training = False
        eval_venv.norm_reward = False
    else:
        eval_venv = dummy

    # Load policy
    model = PPO.load(str(policy_path), device="auto")

    rewards, lengths, successes = [], [], []
    completed_counts = []
    for ep in range(episodes):
        obs = eval_venv.reset()
        total_reward = 0.0
        steps = 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, r, dones, infos = eval_venv.step(action)
            total_reward += float(r[0])
            steps += 1
            done = bool(dones[0])
            if render:
                eval_env.render()
        info = infos[0]
        success = info.get("success", False)
        n_completed = info.get("n_completed", 0)
        rewards.append(total_reward)
        lengths.append(steps)
        successes.append(success)
        completed_counts.append(n_completed)
        print(f"  Episode {ep+1}: reward={total_reward:8.2f}  "
              f"steps={steps:4d}  completed={n_completed}/5  success={success}")

    print("\n[RL Eval] Summary:")
    print(f"  Mean reward: {np.mean(rewards):8.2f} ± {np.std(rewards):.2f}")
    print(f"  Mean length: {np.mean(lengths):8.1f}")
    print(f"  Mean completed: {np.mean(completed_counts):.2f} / 5")
    print(f"  Success rate:  {np.mean(successes) * 100:.1f}%")
    eval_env.close()
    return {
        "rewards": rewards, "lengths": lengths,
        "successes": successes, "completed_counts": completed_counts,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained PPO policy")
    parser.add_argument("--policy", type=str, default=None,
                        help="Path to policy .zip (default: checkpoints/ppo_disaster_triage.zip)")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--stochastic", action="store_true",
                        help="Use stochastic policy (default: deterministic)")
    args = parser.parse_args()
    policy_path = Path(args.policy) if args.policy else None
    evaluate(
        policy_path=policy_path,
        episodes=args.episodes,
        deterministic=not args.stochastic,
        render=args.render,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
