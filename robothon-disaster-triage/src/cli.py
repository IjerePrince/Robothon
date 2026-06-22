"""
cli.py — Unified CLI entrypoint for the Disaster Triage Robothon project.

Usage:
    python -m src.cli <mode> [options]

Modes:
    scripted   — run the scripted autonomous controller
    rl-train   — train a PPO policy
    rl-eval    — evaluate a trained policy
    demo       — render the demo MP4 video
    inspect    — load the scene and print stats (smoketest)
"""
from __future__ import annotations
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="src.cli",
        description="Disaster Triage Robothon — unified CLI",
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    # scripted
    p_scripted = sub.add_parser("scripted", help="Run scripted autonomous controller")
    p_scripted.add_argument("--max-steps", type=int, default=3000)
    p_scripted.add_argument("--render", action="store_true")
    p_scripted.add_argument("--no-verbose", action="store_true",
                            help="Suppress per-100-step logging")

    # rl-train
    p_train = sub.add_parser("rl-train", help="Train PPO policy")
    p_train.add_argument("--total-timesteps", type=int, default=100_000)
    p_train.add_argument("--eval-freq", type=int, default=10_000)
    p_train.add_argument("--n-envs", type=int, default=4)
    p_train.add_argument("--smoketest", action="store_true")

    # rl-eval
    p_eval = sub.add_parser("rl-eval", help="Evaluate trained policy")
    p_eval.add_argument("--policy", type=str, default=None)
    p_eval.add_argument("--episodes", type=int, default=5)
    p_eval.add_argument("--render", action="store_true")

    # demo
    p_demo = sub.add_parser("demo", help="Render demo MP4")
    p_demo.add_argument("--out", type=str, default=None)
    p_demo.add_argument("--duration", type=int, default=120)
    p_demo.add_argument("--scripted-steps", type=int, default=1200)

    # inspect
    p_inspect = sub.add_parser("inspect", help="Inspect scene + smoketest")
    p_inspect.add_argument("--steps", type=int, default=500)

    # view — interactive MuJoCo viewer with scripted controller
    p_view = sub.add_parser("view", help="Launch interactive MuJoCo viewer + run scripted controller")
    p_view.add_argument("--max-steps", type=int, default=6000)
    p_view.add_argument("--slow-mo", type=float, default=1.0,
                        help="Slow-mo factor (0.5 = half speed)")

    args = parser.parse_args()

    if args.mode == "scripted":
        from .scripted_controller import main as scripted_main
        sys.argv = ["scripted", "--max-steps", str(args.max_steps)]
        if args.render:
            sys.argv.append("--render")
        if args.no_verbose:
            sys.argv.append("--no-verbose")
        scripted_main()

    elif args.mode == "rl-train":
        from .rl_train import main as train_main
        sys.argv = ["rl-train",
                    "--total-timesteps", str(args.total_timesteps),
                    "--eval-freq", str(args.eval_freq),
                    "--n-envs", str(args.n_envs)]
        if args.smoketest:
            sys.argv.append("--smoketest")
        train_main()

    elif args.mode == "rl-eval":
        from .rl_eval import main as eval_main
        sys.argv = ["rl-eval", "--episodes", str(args.episodes)]
        if args.policy:
            sys.argv.extend(["--policy", args.policy])
        if args.render:
            sys.argv.append("--render")
        eval_main()

    elif args.mode == "demo":
        from .demo_render import main as demo_main
        sys.argv = ["demo", "--duration", str(args.duration),
                    "--scripted-steps", str(args.scripted_steps)]
        if args.out:
            sys.argv.extend(["--out", args.out])
        demo_main()

    elif args.mode == "inspect":
        from .env import DisasterTriageEnv
        import mujoco
        import numpy as np
        env = DisasterTriageEnv(render_mode=None)
        print(f"Scene:  {env.scene_xml}")
        print(f"nq={env.model.nq} nv={env.model.nv} nu={env.model.nu} "
              f"nbody={env.model.nbody} ngeom={env.model.ngeom} "
              f"njnt={env.model.njnt} nsite={env.model.nsite} "
              f"nsensor={env.model.nsensor} neq={env.model.neq}")
        print(f"Actuators: {env.model.nu}  (UR5e: 6, Shadow Hand: 16)")
        print(f"Sensors: {env.model.nsensor}  (6 tactile + 22 joint pos + 6 joint vel + 7 frame)")
        print(f"Equality constraints: {env.model.neq}")
        print(f"Timestep: {env.sim_dt*1000:.1f} ms,  frame_skip={env.frame_skip}, "
              f"env dt={env.dt*1000:.1f} ms")
        print(f"\nStepping for {args.steps} steps...")
        obs, _ = env.reset(seed=0)
        for i in range(args.steps):
            env.step(env.action_space.sample() * 0.1)
        print(f"  Done. Final time: {env.data.time:.2f}s")
        print(f"  Any NaN: {np.isnan(env.data.qpos).any()}")
        env.close()

    elif args.mode == "view":
        from .view_scripted import main as view_main
        view_main()


if __name__ == "__main__":
    main()
