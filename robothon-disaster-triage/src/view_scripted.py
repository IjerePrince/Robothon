"""
view_scripted.py — Launch the MuJoCo interactive viewer and run the scripted
controller in real-time.

Usage:
    python -m src.view_scripted
    python -m src.view_scripted --max-steps 6000
    python -m src.view_scripted --slow-mo 0.5

Controls (in the viewer window):
    Mouse drag   — rotate camera
    Scroll       — zoom
    Right-drag   — pan
    Space        — pause/resume
    Tab          — slow down
    Esc          — quit
"""
from __future__ import annotations
import argparse
import time
import sys
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.env import DisasterTriageEnv
from src.scripted_controller import ScriptedController, State, OBJECT_NAMES


def main():
    parser = argparse.ArgumentParser(
        description="Launch interactive MuJoCo viewer + run scripted controller"
    )
    parser.add_argument("--max-steps", type=int, default=6000,
                        help="Max episode steps (default: 6000)")
    parser.add_argument("--slow-mo", type=float, default=1.0,
                        help="Slow-mo factor (0.5 = half speed, 2.0 = double speed)")
    args = parser.parse_args()

    # Create env WITHOUT render_mode — we'll handle rendering manually
    env = DisasterTriageEnv(render_mode=None, max_steps=args.max_steps)
    controller = ScriptedController(env)

    print("\n" + "="*60)
    print("  Disaster Triage Robothon — Interactive Viewer")
    print("="*60)
    print()
    print("  Controls:")
    print("    Mouse drag   — rotate camera")
    print("    Scroll       — zoom")
    print("    Right-drag   — pan")
    print("    Space        — pause/resume")
    print("    Esc          — quit")
    print()
    print(f"  Running {args.max_steps} steps at {args.slow_mo}x speed")
    print(f"  (Episode will take ~{args.max_steps * 0.02 / args.slow_mo:.0f}s real-time)")
    print()
    print("  Press Enter to start...")
    input()

    obs, info = env.reset(seed=0)
    controller.reset()

    # Launch the passive viewer — this opens the window
    print("  Launching MuJoCo viewer...")
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        print("  Viewer launched! Watch the robot work.")
        print()
        print("  Episode progress:")
        print("  " + "-"*56)

        last_state = None
        step_count = 0
        t0 = time.time()

        while viewer.is_running() and step_count < args.max_steps:
            # Check if episode is done
            if env.success or env.failed or controller.state.fsm_state == State.DONE:
                break

            # Compute and apply action
            action = controller.compute_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            step_count += 1

            # Update the viewer with the new physics state
            viewer.sync()

            # Log state transitions
            if controller.state.fsm_state != last_state:
                obj_idx = controller.state.current_obj_idx
                obj_name = OBJECT_NAMES[obj_idx] if obj_idx < len(OBJECT_NAMES) else "DONE"
                print(f"  step {step_count:5d}  | state={controller.state.fsm_state.value:12s} "
                      f"| obj={obj_name:10s} "
                      f"| completed={len(controller.state.completed)}/5 "
                      f"| failed={len(controller.state.failed)}")
                last_state = controller.state.fsm_state

            # Real-time pacing: env.dt = 20ms per step
            # Sleep to match real-time (scaled by slow-mo factor)
            target_dt = env.dt / args.slow_mo
            elapsed = time.time() - t0
            expected = step_count * target_dt
            if elapsed < expected:
                time.sleep(expected - elapsed)

        # Final summary
        dt = time.time() - t0
        print()
        print("  " + "-"*56)
        print(f"  Episode finished: {step_count} steps in {dt:.1f}s")
        print(f"  Objects completed: {len(controller.state.completed)} / 5")
        print(f"  Objects failed:    {len(controller.state.failed)}")
        print(f"  Success:           {len(controller.state.completed) == 5}")
        if controller.state.completed:
            print(f"  Completed: {[OBJECT_NAMES[i] for i in controller.state.completed]}")
        if controller.state.failed:
            print(f"  Failed:    {[OBJECT_NAMES[i] for i in controller.state.failed]}")
        print()
        print("  Closing viewer... (press Esc if it doesn't close)")

    env.close()
    print("  Done.")


if __name__ == "__main__":
    main()
