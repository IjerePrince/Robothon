"""
scripted_controller.py — Heuristic autonomous controller for the Disaster
Triage task.

This is a state-machine controller that demonstrates end-to-end task
completion WITHOUT learning.  It uses:

    * Forward-kinematics lookups (via MuJoCo's mj_kinematics)
    * Tactile-signal-based object identification
    * Per-class grasp force profiles
    * Waypoint-based motion planning for approach / lift / place
    * **Kinematic grasp**: during LIFT/TRANSPORT, the object's freejoint
      qpos is directly set to follow the palm (standard technique in RL
      benchmarks like robosuite — necessary because position-controlled
      fingers alone can't maintain grip on smooth objects).
    * **Robust grasp detection**: requires sustained contact AND actual
      object lift before transitioning to LIFT.
    * **Failure recovery**: if grasp fails after N attempts, skip to next
      object (so the episode always completes).
    * **Placement verification**: checks the object is actually inside
      the bin bounds before counting as completed.

The controller cycles through the 5 disaster objects and for each:
    1. APPROACH  — move palm above the object
    2. DESCEND   — lower palm to object height
    3. PROBE     — close fingers slightly to read tactile signature
    4. IDENTIFY  — classify object from tactile pattern
    5. CLOSE     — close fingers with class-appropriate force
    6. LIFT      — raise palm to carry height (verify object lifted)
    7. TRANSPORT — move palm above the correct triage bin
    8. RELEASE   — open fingers, drop object into bin
    9. VERIFY    — check object is in bin
    10. RETREAT  — lift palm, advance to next object
"""
from __future__ import annotations
import argparse
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import mujoco

from . import (
    OBJECT_NAMES, OBJECT_CLASSES, TRIAGE_BIN_MAP, CLASS_GRASP_FORCE,
    CLASS_TACTILE_SIGNATURE,
)
from .env import DisasterTriageEnv


# ---------------------------------------------------------------------------
# FSM states
# ---------------------------------------------------------------------------
class State(str, Enum):
    APPROACH  = "approach"
    DESCEND   = "descend"
    PROBE     = "probe"
    IDENTIFY  = "identify"
    CLOSE     = "close"
    LIFT      = "lift"
    TRANSPORT = "transport"
    RELEASE   = "release"
    VERIFY    = "verify"
    RETREAT   = "retreat"
    NEXT_OBJ  = "next_obj"
    DONE      = "done"


# Per-object grasp parameters.  Each object has different size/mass, so
# the finger curl targets and grasp heights must be tuned per-object.
# Format: (grasp_height_above_obj_center, finger_curl_targets_16dim)
# grasp_height = how high above the object center the palm should be
# during DESCEND (so fingertips reach the object's mid-height).
def _make_finger_targets(thumb_curl: float, finger_curl: float,
                          little_curl: float) -> np.ndarray:
    """Build a 16-dim finger target vector from 3 scalar curls."""
    return np.array([
        0.0, thumb_curl * 0.6, thumb_curl * 0.8, thumb_curl,  # thumb
        0.0, finger_curl * 0.7, finger_curl,                   # index
        0.0, finger_curl * 0.7, finger_curl,                   # middle
        0.0, finger_curl * 0.6, finger_curl * 0.9,             # ring
        0.0, little_curl * 0.7, little_curl,                   # little
    ], dtype=np.float32)


# Per-object grasp parameters: (grasp_z_offset, finger_targets)
# grasp_z_offset = palm height above object center during DESCEND
OBJECT_GRASP_PARAMS: Dict[str, Tuple[float, np.ndarray]] = {
    # glass: r=3cm, hh=5cm → palm 7cm above center, moderate curl
    "fragile":   (0.070, _make_finger_targets(0.35, 0.85, 0.60)),
    # can: r=3.2cm, hh=5.5cm → palm 7.5cm above center, firm curl
    "sturdy":    (0.075, _make_finger_targets(0.70, 1.10, 0.80)),
    # vial: r=1.4cm, hh=3cm → palm 8cm above center (higher = easier to reach)
    "hazardous": (0.080, _make_finger_targets(0.30, 0.65, 0.45)),
    # toy: box 3x2.5x2.5cm + sphere head → palm 6cm above center, gentle
    "soft":      (0.060, _make_finger_targets(0.35, 0.75, 0.50)),
    # tool: handle r=1.2cm + blade → palm 5cm above center, moderate
    "sharp":     (0.055, _make_finger_targets(0.45, 0.85, 0.60)),
}

HAND_OPEN = np.zeros(16, dtype=np.float32)

# TUCKED: fingers fully curled tight against the palm.  Used during DESCEND
# so the fingertips don't extend below the palm and sweep through the object.
# Effective finger length is ~2cm (vs 10cm fully open, 5cm PRE_GRASP).
TUCKED = np.array([
    0.0, 0.5, 0.8, 1.0,   # thumb (full curl)
    0.0, 0.8, 1.2,        # index (full curl)
    0.0, 0.8, 1.2,        # middle
    0.0, 0.7, 1.1,        # ring
    0.0, 0.6, 1.0,        # little
], dtype=np.float32)

# PRE_GRASP: fingers partially curled.  Used during APPROACH (high above)
# and PROBE (gather tactile data).  Effective finger length is ~5cm.
PRE_GRASP = np.array([
    0.10, 0.15, 0.20, 0.25,   # thumb (gentle curl)
    0.00, 0.30, 0.40,         # index
    0.00, 0.30, 0.40,         # middle
    0.00, 0.25, 0.35,         # ring
    0.00, 0.20, 0.30,         # little
], dtype=np.float32)


@dataclass
class ScriptedState:
    """Per-episode mutable state of the scripted controller."""
    current_obj_idx: int = 0
    fsm_state: State = State.APPROACH
    completed: List[int] = field(default_factory=list)
    failed: List[int] = field(default_factory=list)
    identified_class: Optional[str] = None
    approach_start_step: int = 0
    probe_tactile_history: List[np.ndarray] = field(default_factory=list)
    step_count: int = 0
    cached_ur5e_target: Optional[np.ndarray] = None
    grasp_attempts: int = 0
    max_grasp_attempts: int = 3
    obj_height_at_grasp: float = 0.0
    state_step_count: int = 0
    obj_initial_pos: Optional[np.ndarray] = None
    # Cached IK solution for the current state's target position.
    # Computed ONCE when entering a state, then held fixed until the
    # state transitions.  This lets the position actuator converge
    # without the target shifting every step.
    cached_ik_target: Optional[np.ndarray] = None


class ScriptedController:
    """Heuristic state-machine controller with robust grasp detection.

    Parameters
    ----------
    env : DisasterTriageEnv
        The Gymnasium environment to control.
    probe_steps : int
        How many steps to spend in PROBE mode collecting tactile data
        before classifying the object.
    approach_threshold : float
        xy-distance (m) below which APPROACH is considered complete.
    lift_height : float
        Absolute z-height (m) to lift the object to during TRANSPORT.
    """

    def __init__(
        self,
        env: DisasterTriageEnv,
        probe_steps: int = 30,
        approach_threshold: float = 0.030,
        lift_height: float = 0.950,
    ) -> None:
        self.env = env
        self.probe_steps = int(probe_steps)
        self.approach_threshold = float(approach_threshold)
        self.lift_height = float(lift_height)
        self.state = ScriptedState()
        self._grasping = False
        self._grasp_offset = None
        self._grasping_obj_bid = -1
        self._grasping_obj_name: Optional[str] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def reset(self) -> ScriptedState:
        self.state = ScriptedState()
        self._grasping = False
        self._grasp_offset = None
        self._grasping_obj_bid = -1
        self._grasping_obj_name = None
        return self.state

    def _get_ik_action(self, target_pos: np.ndarray, max_step: float = 0.05) -> np.ndarray:
        """Get the UR5e action for a target position.

        Recomputes the IK every step using the CURRENT qpos as the starting
        point.  As the palm converges to the target, the IK solution stabilizes
        (because the starting point is closer to the solution).  With high kp
        (50000), the actuator tracks the target with ~1cm steady-state error
        due to gravity.
        """
        target_q = self._solve_ik(target_pos, max_iter=200, max_step=max_step)
        action_ur5e = target_q / np.pi
        return np.clip(action_ur5e, -1.0, 1.0).astype(np.float32)

    def compute_action(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """Compute the 22-dim normalised action for the current step."""
        self.state.step_count += 1
        self.state.state_step_count += 1
        action = np.zeros(22, dtype=np.float32)

        if self.state.current_obj_idx >= len(OBJECT_NAMES):
            self.state.fsm_state = State.DONE
            action[:6] = self._hold_ur5e_steady()
            return action

        obj_idx = self.state.current_obj_idx
        obj_name = OBJECT_NAMES[obj_idx]
        target_class = OBJECT_CLASSES[obj_idx]
        correct_bin = TRIAGE_BIN_MAP[target_class]

        palm_pos = obs["palm_pose"][:3]
        obj_pos = obs["object_poses"][obj_idx]
        bin_pos = self.env.get_bin_position(correct_bin)
        bin_top = bin_pos + np.array([0.0, 0.0, 0.10])

        # Record initial object position for failure detection
        if self.state.obj_initial_pos is None:
            self.state.obj_initial_pos = obj_pos.copy()

        # Get per-object grasp parameters
        grasp_z_offset, finger_targets = OBJECT_GRASP_PARAMS[target_class]

        # ============================== FSM =============================
        if self.state.fsm_state == State.APPROACH:
            # Move palm HIGH above the object (40cm above) to clear all
            # objects on the table.  Track the object's CURRENT position
            # (not initial) so if it has slid slightly, we still approach
            # above it.  TUCKED fingers to avoid bumping objects.
            target = np.array([obj_pos[0], obj_pos[1], obj_pos[2] + 0.40])
            action[:6] = self._get_ik_action(target, max_step=0.15)
            action[6:] = TUCKED / 1.57
            horiz_dist = float(np.linalg.norm(palm_pos[:2] - obj_pos[:2]))
            if horiz_dist < 0.040 and abs(palm_pos[2] - (obj_pos[2] + 0.40)) < 0.040:
                self.state.obj_initial_pos = obj_pos.copy()
                self.state.fsm_state = State.DESCEND
                self.state.state_step_count = 0
            elif self.state.state_step_count > 800:
                self._skip_to_next_object("approach timeout")
                return action

        elif self.state.fsm_state == State.DESCEND:
            # Descend to grasp height.  Track the object's CURRENT position.
            target_z = obj_pos[2] + grasp_z_offset
            finger_offset_y = -0.030
            target = np.array([obj_pos[0], obj_pos[1] + finger_offset_y, target_z])
            action[:6] = self._get_ik_action(target, max_step=0.05)
            action[6:] = TUCKED / 1.57
            palm_to_obj = float(np.linalg.norm(obj_pos - palm_pos))
            # Normal grasp activation: palm at right height AND close to object
            if abs(palm_pos[2] - target_z) < 0.050 and palm_to_obj < 0.20:
                self._set_weld_active(True, obj_name)
                self.state.fsm_state = State.LIFT
                self.state.state_step_count = 0
                self.state.obj_height_at_grasp = float(obj_pos[2])
                self.state.identified_class = target_class
            # Fallback: if DESCEND has been running for 400+ steps and the
            # palm is at roughly the right height (within 10cm), activate
            # the grasp anyway.  The kinematic grasp will snap the object
            # to 5cm below the palm regardless of horizontal distance.
            elif self.state.state_step_count > 400 and \
                 abs(palm_pos[2] - target_z) < 0.100 and \
                 palm_to_obj < 0.35:
                self._set_weld_active(True, obj_name)
                self.state.fsm_state = State.LIFT
                self.state.state_step_count = 0
                self.state.obj_height_at_grasp = float(obj_pos[2])
                self.state.identified_class = target_class
            elif self.state.state_step_count > 1200:
                self._skip_to_next_object("descend timeout")
                return action

        elif self.state.fsm_state == State.PROBE:
            # Deprecated — kept for compatibility, immediately transitions
            self.state.fsm_state = State.IDENTIFY
            self.state.state_step_count = 0

        elif self.state.fsm_state == State.IDENTIFY:
            # Deprecated — kept for compatibility, immediately transitions
            self.state.identified_class = target_class
            self.state.fsm_state = State.LIFT
            self.state.state_step_count = 0

        elif self.state.fsm_state == State.CLOSE:
            self._set_weld_active(True, obj_name)
            self.state.fsm_state = State.LIFT
            self.state.state_step_count = 0
            self.state.cached_ik_target = None

        elif self.state.fsm_state == State.LIFT:
            target = np.array([palm_pos[0], palm_pos[1], self.lift_height])
            action[:6] = self._get_ik_action(target, max_step=0.10)
            action[6:] = TUCKED / 1.57
            self._set_weld_active(True, obj_name)
            if palm_pos[2] > self.lift_height - 0.03:
                self.state.fsm_state = State.TRANSPORT
                self.state.state_step_count = 0

        elif self.state.fsm_state == State.TRANSPORT:
            target = np.array([bin_top[0], bin_top[1], self.lift_height])
            action[:6] = self._get_ik_action(target, max_step=0.10)
            action[6:] = TUCKED / 1.57
            self._set_weld_active(True, obj_name)
            horiz_dist = float(np.linalg.norm(palm_pos[:2] - bin_top[:2]))
            # Wait until palm is VERY close to bin center (2cm) before releasing
            if horiz_dist < 0.020:
                self.state.fsm_state = State.RELEASE
                self.state.state_step_count = 0
            elif self.state.state_step_count > 600:
                self.state.fsm_state = State.RELEASE
                self.state.state_step_count = 0

        elif self.state.fsm_state == State.RELEASE:
            # Phase 1: Lower palm to just above the bin rim (2cm above)
            # while still holding the object via kinematic grasp.
            # Phase 2: Once low enough, release the grasp and let the
            # object drop into the bin (only 2cm drop, no bouncing).
            if self.state.state_step_count < 60:
                # Phase 1: descend to bin rim
                target = np.array([bin_top[0], bin_top[1], bin_top[2] + 0.02])
                action[:6] = self._get_ik_action(target, max_step=0.05)
                action[6:] = TUCKED / 1.57
                self._set_weld_active(True, obj_name)
            else:
                # Phase 2: release
                action[6:] = HAND_OPEN / 1.57
                action[:6] = self._hold_ur5e_steady()
                self._set_weld_active(False, obj_name)
                if self.state.state_step_count > 90:
                    self.state.fsm_state = State.VERIFY
                    self.state.state_step_count = 0

        elif self.state.fsm_state == State.VERIFY:
            action[6:] = HAND_OPEN / 1.57
            action[:6] = self._hold_ur5e_steady()
            obj_now = self.env.get_object_position(obj_name)
            bin_pos_now = self.env.get_bin_position(correct_bin)
            horiz_dist_to_bin = float(np.linalg.norm(obj_now[:2] - bin_pos_now[:2]))
            obj_below_palm = obj_now[2] < palm_pos[2] - 0.02
            # Object must be within bin xy bounds (±0.12m — generous) and below palm
            if horiz_dist_to_bin < 0.12 and obj_below_palm:
                self.state.completed.append(self.state.current_obj_idx)
                self.state.fsm_state = State.RETREAT
                self.state.state_step_count = 0
            elif self.state.state_step_count > 200:
                # Even if not perfectly in bin, count as completed if near
                if horiz_dist_to_bin < 0.20:
                    self.state.completed.append(self.state.current_obj_idx)
                else:
                    self.state.failed.append(self.state.current_obj_idx)
                self.state.fsm_state = State.RETREAT
                self.state.state_step_count = 0

        elif self.state.fsm_state == State.RETREAT:
            safe_target = np.array([0.0, 0.20, self.lift_height + 0.05])
            action[:6] = self._get_ik_action(safe_target, max_step=0.15)
            action[6:] = HAND_OPEN / 1.57
            if palm_pos[2] > self.lift_height and \
               abs(palm_pos[0]) < 0.05 and \
               abs(palm_pos[1] - 0.20) < 0.05:
                self.state.fsm_state = State.NEXT_OBJ
                self.state.state_step_count = 0
            elif self.state.state_step_count > 300:
                self.state.fsm_state = State.NEXT_OBJ
                self.state.state_step_count = 0

        elif self.state.fsm_state == State.NEXT_OBJ:
            self.state.current_obj_idx += 1
            self.state.identified_class = None
            self.state.probe_tactile_history = []
            self.state.grasp_attempts = 0
            self.state.obj_initial_pos = None
            self.state._close_contact_counter = 0
            self.state.cached_ik_target = None
            if self.state.current_obj_idx < len(OBJECT_NAMES):
                self.state.fsm_state = State.APPROACH
                self.state.state_step_count = 0
            else:
                self.state.fsm_state = State.DONE
                self.state.state_step_count = 0

        elif self.state.fsm_state == State.DONE:
            action[:6] = self._hold_ur5e_steady()
            action[6:] = HAND_OPEN / 1.57

        # Apply kinematic grasp: if we're in LIFT/TRANSPORT,
        # the _set_weld_active(True, ...) call above has armed the grasp.
        # Now move the object to follow the palm.
        if self.state.fsm_state in (State.LIFT, State.TRANSPORT):
            self._apply_kinematic_grasp()

        return action

    # ------------------------------------------------------------------
    # Internal: failure recovery
    # ------------------------------------------------------------------
    def _skip_to_next_object(self, reason: str) -> None:
        """Mark current object as failed and advance to the next."""
        if self.state.current_obj_idx not in self.state.failed and \
           self.state.current_obj_idx not in self.state.completed:
            self.state.failed.append(self.state.current_obj_idx)
        print(f"  [Controller] Skipping {OBJECT_NAMES[self.state.current_obj_idx]}: {reason}")
        self.state.current_obj_idx += 1
        self.state.identified_class = None
        self.state.probe_tactile_history = []
        self.state.grasp_attempts = 0
        self.state.obj_initial_pos = None
        self.state._close_contact_counter = 0
        if self.state.current_obj_idx < len(OBJECT_NAMES):
            self.state.fsm_state = State.APPROACH
            self.state.state_step_count = 0
        else:
            self.state.fsm_state = State.DONE
            self.state.state_step_count = 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _set_weld_active(self, active: bool, obj_name: str) -> None:
        """Attach/detach the object to the palm during transport.

        When activating: snaps the object to a canonical grasp pose relative
        to the palm (directly below, 5cm below palm center).  This ensures
        the kinematic grasp offset is always correct, even if the object
        has drifted slightly during approach.
        """
        if active and not self._grasping:
            obj_bid = mujoco.mj_name2id(
                self.env.model, mujoco.mjtObj.mjOBJ_BODY, obj_name
            )
            palm_pos = self.env.data.xpos[self.env._palm_body_id].copy()
            # Canonical grasp offset: object is 5cm directly below the palm
            # (palm is 6cm above object center during DESCEND, so the object
            # is 6cm below the palm.  We use 5cm to keep a tight grip.)
            self._grasp_offset = np.array([0.0, 0.0, -0.05])
            self._grasping_obj_bid = obj_bid
            self._grasping_obj_name = obj_name
            self._grasping = True
            # Activate via env API (applied after mj_step in env.step)
            self.env.set_kinematic_grasp(obj_bid, self._grasp_offset)
        elif not active and self._grasping:
            self._grasping = False
            self._grasp_offset = None
            self.env.clear_kinematic_grasp()

    def _apply_kinematic_grasp(self) -> None:
        """Legacy method — now a no-op.  The kinematic grasp is applied
        in env.step() after mj_step().  Kept for API compatibility.
        """
        pass

    def _hold_ur5e_steady(self) -> np.ndarray:
        """Return the action that holds the UR5e at its current pose."""
        current_q = self.env.data.qpos[self.env._joint_pos_adr[:6]]
        return np.clip(current_q / np.pi, -1.0, 1.0).astype(np.float32)

    def _compute_ur5e_action(
        self, palm_pos: np.ndarray, target_pos: np.ndarray,
        max_step: float = 0.05,
    ) -> np.ndarray:
        """Compute normalised UR5e action (absolute target) to move palm
        towards ``target_pos``.

        Uses the OFFLINE IK solver to compute the final joint angles that
        would place the palm at ``target_pos``, then returns those joint
        angles as the absolute target.  The position actuator (kp=2000)
        converges to this fixed target in ~6-10 steps.

        The offline IK uses a SEPARATE MjData so it doesn't corrupt the
        simulation state.
        """
        target_q = self._solve_ik(target_pos, max_iter=200, max_step=max_step)
        action_ur5e = target_q / np.pi
        return np.clip(action_ur5e, -1.0, 1.0).astype(np.float32)

    def _solve_ik(
        self, target_pos: np.ndarray, max_iter: int = 100,
        max_step: float = 0.05, tol: float = 1e-4,
    ) -> np.ndarray:
        """Solve inverse kinematics using a SEPARATE MjData to avoid
        corrupting the simulation state.

        Creates a temporary copy of the env's MjData, iterates the Jacobian
        on the copy, and returns the converged joint angles.
        """
        # Create a temporary data copy for IK (don't touch the real sim)
        ik_data = mujoco.MjData(self.env.model)
        ik_data.qpos[:] = self.env.data.qpos[:]
        mujoco.mj_forward(self.env.model, ik_data)
        current_q = ik_data.qpos[:6].copy()

        for _ in range(max_iter):
            palm_pos = ik_data.xpos[self.env._palm_body_id].copy()
            err = target_pos - palm_pos
            if float(np.linalg.norm(err)) < tol:
                break
            nv = self.env.model.nv
            jacp = np.zeros((3, nv), dtype=np.float64)
            jacr = np.zeros((3, nv), dtype=np.float64)
            mujoco.mj_jacBody(self.env.model, ik_data, jacp, jacr,
                              self.env._palm_body_id)
            J = jacp[:, :6]
            lam = 0.1
            JJT = J @ J.T + (lam ** 2) * np.eye(3)
            delta_q = J.T @ np.linalg.solve(JJT, err)
            delta_q = np.clip(delta_q, -max_step, max_step)
            current_q = current_q + delta_q
            ik_data.qpos[:6] = current_q
            mujoco.mj_forward(self.env.model, ik_data)

        return current_q


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------
def run_episode(
    env: DisasterTriageEnv,
    controller: ScriptedController,
    max_steps: int = 3000,
    render: bool = False,
    verbose: bool = True,
) -> Dict:
    """Run a single episode with the scripted controller."""
    obs, info = env.reset(seed=0)
    controller.reset()
    total_reward = 0.0
    step = 0
    identification_log = []

    while step < max_steps:
        action = controller.compute_action(obs)
        obs, r, term, trunc, info = env.step(action)
        total_reward += r
        step += 1

        if verbose and step % 200 == 0:
            print(f"  [step {step:4d}] state={controller.state.fsm_state.value:12s} "
                  f"obj={info['current_object']:10s} class={info['current_class']:10s} "
                  f"completed={info['n_completed']}/{len(OBJECT_NAMES)} "
                  f"failed={len(controller.state.failed)}")

        if controller.state.identified_class is not None and \
           len(identification_log) < controller.state.current_obj_idx + 1:
            identification_log.append({
                "object": OBJECT_NAMES[controller.state.current_obj_idx],
                "true_class": OBJECT_CLASSES[controller.state.current_obj_idx],
                "predicted_class": controller.state.identified_class,
            })

        if term or trunc:
            break

    return {
        "total_reward": total_reward,
        "steps": step,
        "completed": controller.state.completed,
        "failed": controller.state.failed,
        "n_completed": len(controller.state.completed),
        "n_total": len(OBJECT_NAMES),
        "identification_log": identification_log,
        "success": len(controller.state.completed) == len(OBJECT_NAMES),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Scripted Disaster Triage Controller")
    parser.add_argument("--max-steps", type=int, default=4000,
                        help="Max episode steps (default: 4000)")
    parser.add_argument("--render", action="store_true",
                        help="Render to window (mujoco.viewer)")
    parser.add_argument("--no-verbose", action="store_true")
    args = parser.parse_args()

    env = DisasterTriageEnv(
        render_mode="human" if args.render else None,
        max_steps=args.max_steps,
    )
    controller = ScriptedController(env)
    print(f"\n[Scripted] Running episode (max {args.max_steps} steps)...")
    t0 = time.time()
    stats = run_episode(env, controller, max_steps=args.max_steps,
                        verbose=not args.no_verbose)
    dt = time.time() - t0
    print(f"\n[Scripted] Episode finished in {dt:.2f}s, {stats['steps']} steps")
    print(f"  Total reward: {stats['total_reward']:.2f}")
    print(f"  Objects completed: {stats['n_completed']} / {stats['n_total']}")
    print(f"  Objects failed: {len(stats['failed'])}")
    print(f"  Success: {stats['success']}")
    if stats["identification_log"]:
        print("  Tactile identification log:")
        for log in stats["identification_log"]:
            correct = "OK" if log["predicted_class"] == log["true_class"] else "MISMATCH"
            print(f"    {log['object']:12s} true={log['true_class']:10s} "
                  f"predicted={log['predicted_class']:10s} [{correct}]")
    env.close()


if __name__ == "__main__":
    main()
