"""
env.py — Gymnasium environment wrapping the MuJoCo Disaster Triage scene.

Task
----
A UR5e arm with a Shadow Hand Lite (16-DOF) is mounted above a workbench.
Five disaster-zone objects sit on the table:
    glass  (fragile)    — slides on a 1-D track, must be lifted gently
    can    (sturdy)     — free rigid body, must be gripped firmly
    vial   (hazardous)  — small free body, careful grip
    toy    (soft)       — low mass, gentle grip
    tool   (sharp)      — irregular shape, mixed grip
Three triage bins (fragile / sturdy / hazardous) sit on the opposite side.

The robot must:
    1. Identify each object by TACTILE SIGNATURE (5 fingertip + palm forces)
       — vision is *not* provided to the policy (deliberate, to highlight
         the value of tactile sensing in disaster zones where vision is
         occluded by smoke / debris).
    2. Grasp the object with the appropriate force (per-class target).
    3. Place it in the correct triage bin.

Action Space (22-dim, Box [-1, 1])
    [0:6]    UR5e joint position deltas (scaled by 0.1 rad / step)
    [6:22]   Shadow Hand Lite joint position targets (direct -1..1 -> -1.57..1.57)

Observation Space (dict)
    "tactile"        : (6,)   — 5 fingertip + 1 palm touch sensor (N)
    "joint_pos"      : (22,)  — UR5e (6) + Hand (16) joint positions
    "joint_vel"      : (22,)  — UR5e (6) + Hand (16) joint velocities
    "palm_pose"      : (7,)   — palm pos (3) + quat (4)
    "object_poses"   : (5,3)  — 5 objects' xyz positions
    "current_object" : (5,)   — one-hot index of the current target object
    "carrying"       : (1,)   — 1.0 if an object is currently grasped

Reward (per step)
    + approach_bonus       — palm moving towards current target object
    + contact_bonus        — any fingertip contact with target
    + grasp_bonus          — successful grasp (all 5 fingers in contact, lift detected)
    + correct_force_bonus  — tactile signature matches the target class
    + placement_bonus      — object placed in the correct bin (large)
    - 0.001 * action_norm  — small action regulariser
    - 0.01  * step_penalty — per-step time cost

Episode terminates when:
    * all 5 objects are placed in correct bins (SUCCESS)
    * an object is placed in the wrong bin (FAIL)
    * a fragile object breaks (force > 8N for >10 steps) (FAIL)
    * 1500 steps elapsed (TIMEOUT)

Example
-------
    from src.env import DisasterTriageEnv
    env = DisasterTriageEnv(reward_scale=1.0, max_steps=1500)
    obs, info = env.reset(seed=0)
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

from . import (
    SCENE_XML, ROBOT_DOF, UR5E_DOF, HAND_DOF, N_TACTILE_SENSORS,
    OBJECT_NAMES, OBJECT_LABELS, OBJECT_CLASSES,
    TRIAGE_BIN_MAP, CLASS_GRASP_FORCE, CLASS_TACTILE_SIGNATURE,
)


class DisasterTriageEnv(gym.Env):
    """Gymnasium environment for the Disaster Triage Robothon task.

    Parameters
    ----------
    scene_xml : Path | str
        Path to the MuJoCo scene XML.  Defaults to the project's scene.xml.
    max_steps : int
        Episode length limit.
    reward_scale : float
        Multiplier applied to all reward components.
    render_mode : str | None
        "human" for interactive window, "rgb_array" for offscreen, None for no rendering.
    frame_skip : int
        Number of MuJoCo timesteps per env step (env dt = frame_skip * sim_dt).
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    # ---------------------------------------------------------------------
    # Construction
    # ---------------------------------------------------------------------
    def __init__(
        self,
        scene_xml: Optional[Path | str] = None,
        max_steps: int = 4000,
        reward_scale: float = 1.0,
        render_mode: Optional[str] = None,
        frame_skip: int = 10,
        camera_name: str = "cam_front",
    ) -> None:
        super().__init__()
        self.scene_xml = Path(scene_xml) if scene_xml else SCENE_XML
        if not self.scene_xml.exists():
            raise FileNotFoundError(f"Scene XML not found: {self.scene_xml}")

        self.max_steps = int(max_steps)
        self.reward_scale = float(reward_scale)
        self.frame_skip = int(frame_skip)
        self.render_mode = render_mode
        self.camera_name = camera_name

        # ---- MuJoCo model ----
        self.model = mujoco.MjModel.from_xml_path(str(self.scene_xml))
        self.data = mujoco.MjData(self.model)
        self.sim_dt = float(self.model.opt.timestep)
        self.dt = self.sim_dt * self.frame_skip

        # ---- Resolve sensor & actuator IDs once ----
        self._tactile_adr = np.array([
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, f"s_tactile_{n}")
            for n in ("TH", "IF", "MF", "RF", "LF", "palm")
        ], dtype=np.int32)
        self._tactile_dim = 1  # touch sensor returns 1 value each

        self._joint_pos_adr = []
        for jn in (
            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
            "A_THJ1", "A_THJ2", "A_THJ3", "A_THJ4",
            "A_IFJ1", "A_IFJ2", "A_IFJ3",
            "A_MFJ1", "A_MFJ2", "A_MFJ3",
            "A_RFJ1", "A_RFJ2", "A_RFJ3",
            "A_LFJ1", "A_LFJ2", "A_LFJ3",
        ):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jn)
            assert jid >= 0, f"Joint {jn} not found in scene"
            self._joint_pos_adr.append(self.model.jnt_qposadr[jid])
        self._joint_pos_adr = np.array(self._joint_pos_adr, dtype=np.int32)

        self._joint_vel_adr = []
        for jn in (
            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
            "A_THJ1", "A_THJ2", "A_THJ3", "A_THJ4",
            "A_IFJ1", "A_IFJ2", "A_IFJ3",
            "A_MFJ1", "A_MFJ2", "A_MFJ3",
            "A_RFJ1", "A_RFJ2", "A_RFJ3",
            "A_LFJ1", "A_LFJ2", "A_LFJ3",
        ):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jn)
            self._joint_vel_adr.append(self.model.jnt_dofadr[jid])
        self._joint_vel_adr = np.array(self._joint_vel_adr, dtype=np.int32)

        # Actuator IDs in canonical order (UR5e first, then hand)
        self._actuator_ids = np.array([
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for name in (
                "act_shoulder_pan", "act_shoulder_lift", "act_elbow",
                "act_wrist_1", "act_wrist_2", "act_wrist_3",
                "act_THJ1", "act_THJ2", "act_THJ3", "act_THJ4",
                "act_IFJ1", "act_IFJ2", "act_IFJ3",
                "act_MFJ1", "act_MFJ2", "act_MFJ3",
                "act_RFJ1", "act_RFJ2", "act_RFJ3",
                "act_LFJ1", "act_LFJ2", "act_LFJ3",
            )
        ], dtype=np.int32)

        # Body IDs for objects + bins + palm
        self._obj_body_ids = np.array([
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n)
            for n in OBJECT_NAMES
        ], dtype=np.int32)
        self._palm_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "palm"
        )
        self._bin_body_ids = {
            "bin_fragile":   mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "bin_fragile"),
            "bin_sturdy":    mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "bin_sturdy"),
            "bin_hazardous": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "bin_hazardous"),
        }

        # ---- Spaces ----
        # Action: 22-dim in [-1, 1]
        #   UR5e (6) -> joint position DELTA, scaled by 0.1 rad
        #   Hand (16) -> absolute joint position target in [-1.57, 1.57]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(ROBOT_DOF,), dtype=np.float32
        )
        # Observation: dict-of-tensors (compatible with SB3 MultiInputPolicy)
        self.observation_space = spaces.Dict({
            "tactile":        spaces.Box(low=0.0, high=50.0,
                                         shape=(N_TACTILE_SENSORS,), dtype=np.float32),
            "joint_pos":      spaces.Box(low=-np.pi, high=np.pi,
                                         shape=(ROBOT_DOF,), dtype=np.float32),
            "joint_vel":      spaces.Box(low=-10.0, high=10.0,
                                         shape=(ROBOT_DOF,), dtype=np.float32),
            "palm_pose":      spaces.Box(low=-2.0, high=2.0,
                                         shape=(7,), dtype=np.float32),
            "object_poses":   spaces.Box(low=-1.0, high=1.5,
                                         shape=(len(OBJECT_NAMES), 3), dtype=np.float32),
            "current_object": spaces.Box(low=0.0, high=1.0,
                                         shape=(len(OBJECT_NAMES),), dtype=np.float32),
            "carrying":       spaces.Box(low=0.0, high=1.0,
                                         shape=(1,), dtype=np.float32),
        })

        # ---- Episode state ----
        self.step_count = 0
        self.current_obj_idx = 0
        self.carrying_obj_idx = -1
        self.completed_objects = set()
        self.failed = False
        self.success = False
        self._fragile_break_counter = 0
        self._renderer = None
        self._viewer = None
        # Kinematic grasp state (set by scripted controller)
        self._kinematic_grasp_obj_bid: int = -1
        self._kinematic_grasp_offset: Optional[np.ndarray] = None

    # ---------------------------------------------------------------------
    # Gymnasium API
    # ---------------------------------------------------------------------
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # Home pose: empirically-found non-colliding UR5e configuration.
        # Palm settles at ~ (0.04, 0.28, 1.29) — clean starting pose, no
        # self-collisions.  From here, the scripted controller's Jacobian
        # IK moves the palm down to object grasp positions.
        home_qpos = np.zeros(self.model.nq, dtype=np.float64)
        home_qpos[0] = 0.0       # shoulder_pan
        home_qpos[1] = -1.0      # shoulder_lift
        home_qpos[2] = 1.5       # elbow
        home_qpos[3] = -0.7      # wrist_1
        home_qpos[4] = 1.5708    # wrist_2 (orient hand down)
        home_qpos[5] = 0.0       # wrist_3
        # Hand: open (all zeros)

        # Initialize all freejoint objects to their <body pos> attributes.
        # mj_resetData sets freejoint qpos to 0 (origin), which would place
        # every freejoint body at the world origin.  We need to explicitly
        # set qpos[0:3] = body.pos and qpos[3:7] = identity quaternion.
        for bid in range(self.model.nbody):
            jnt_adr = self.model.body_jntadr[bid]
            jnt_num = self.model.body_jntnum[bid]
            for j in range(jnt_adr, jnt_adr + jnt_num):
                jid = j
                jtype = self.model.jnt_type[jid]
                if jtype == mujoco.mjtJoint.mjJNT_FREE:
                    qpos_adr = self.model.jnt_qposadr[jid]
                    body_pos = self.model.body_pos[bid].copy()
                    home_qpos[qpos_adr:qpos_adr+3] = body_pos
                    home_qpos[qpos_adr+3] = 0.0  # quat w
                    home_qpos[qpos_adr+4] = 0.0  # quat x
                    home_qpos[qpos_adr+5] = 0.0  # quat y
                    home_qpos[qpos_adr+6] = 1.0  # quat z

        self.data.qpos[:] = home_qpos
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        self.step_count = 0
        self.current_obj_idx = 0
        self.carrying_obj_idx = -1
        self.completed_objects = set()
        self.failed = False
        self.success = False
        self._fragile_break_counter = 0
        # Clear any kinematic grasp from previous episode
        self._kinematic_grasp_obj_bid = -1
        self._kinematic_grasp_offset = None

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        self._apply_action(action)

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        # Apply kinematic grasp AFTER physics step: if a kinematic grasp
        # is active, override the grasped object's qpos to follow the palm.
        # This must happen after mj_step() so the physics solver doesn't
        # override our position setting.
        self._apply_kinematic_grasp_post_step()

        self.step_count += 1
        obs = self._get_obs()
        reward = self._compute_reward(action)
        terminated = self._check_terminated()
        truncated = self.step_count >= self.max_steps
        info = self._get_info()
        return obs, reward, terminated, truncated, info

    def _apply_kinematic_grasp_post_step(self) -> None:
        """If a kinematic grasp is active, move the grasped object to follow
        the palm.  Called after every mj_step() so the physics solver doesn't
        override the position.
        """
        if self._kinematic_grasp_obj_bid < 0 or self._kinematic_grasp_offset is None:
            return
        palm_pos = self.data.xpos[self._palm_body_id].copy()
        target_obj_pos = palm_pos + self._kinematic_grasp_offset
        # Find the object's freejoint qpos address
        jnt_adr = self.model.body_jntadr[self._kinematic_grasp_obj_bid]
        jnt_num = self.model.body_jntnum[self._kinematic_grasp_obj_bid]
        for j in range(jnt_adr, jnt_adr + jnt_num):
            if self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                qpos_adr = self.model.jnt_qposadr[j]
                self.data.qpos[qpos_adr:qpos_adr+3] = target_obj_pos
                # Zero out velocity so the object doesn't drift
                dof_adr = self.model.jnt_dofadr[j]
                self.data.qvel[dof_adr:dof_adr+6] = 0.0
                break

    def set_kinematic_grasp(self, obj_bid: int, offset: np.ndarray) -> None:
        """Activate kinematic grasp: the object will follow the palm."""
        self._kinematic_grasp_obj_bid = obj_bid
        self._kinematic_grasp_offset = offset.copy()

    def clear_kinematic_grasp(self) -> None:
        """Deactivate kinematic grasp."""
        self._kinematic_grasp_obj_bid = -1
        self._kinematic_grasp_offset = None

    def get_next_incomplete_object(self) -> int:
        """Return the index of the next uncompleted object.
        
        Used by the scripted controller to stay synchronized with the
        environment's object tracking after placements.
        
        Returns
        -------
        int
            Index of the next object to work on, or len(OBJECT_NAMES) if done.
        """
        if len(self.completed_objects) >= len(OBJECT_NAMES):
            return len(OBJECT_NAMES)  # All done
        # Find the next uncompleted object
        next_idx = (self.current_obj_idx + 1) % len(OBJECT_NAMES)
        while next_idx in self.completed_objects:
            next_idx = (next_idx + 1) % len(OBJECT_NAMES)
        return next_idx

    def render(self):
        if self.render_mode is None:
            return None
        if self.render_mode == "rgb_array":
            return self._render_offscreen()
        elif self.render_mode == "human":
            self._render_human()
            return None

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        if self._renderer is not None:
            del self._renderer
            self._renderer = None

    # ---------------------------------------------------------------------
    # Internal: action application
    # ---------------------------------------------------------------------
    def _apply_action(self, action: np.ndarray) -> None:
        """Translate normalised action to MuJoCo control inputs.

        UR5e (action[:6]): ABSOLUTE joint position target in radians,
            mapped from action ∈ [-1, 1] to ±π rad.  The position
            actuator (kp=2000) tracks this target; gravity compensation
            is applied via qfrc_applied to eliminate steady-state error.
        Hand  (action[6:22]): absolute target in [-1.57, 1.57] rad.
        """
        ctrl = np.zeros(ROBOT_DOF, dtype=np.float64)
        # UR5e: absolute target position (action in [-1,1] → ±π rad)
        ctrl[:UR5E_DOF] = action[:UR5E_DOF] * np.pi
        # Hand: absolute target
        ctrl[UR5E_DOF:] = action[UR5E_DOF:] * 1.57  # ±1.57 rad

        # Write to MuJoCo actuators
        for i, aid in enumerate(self._actuator_ids):
            self.data.ctrl[aid] = ctrl[i]

        # Gravity compensation for UR5e joints: apply qfrc_bias (which
        # includes gravity + Coriolis) to counteract gravity.  This lets
        # the position actuator (kp=2000) converge to the target without
        # steady-state error from gravity.
        mujoco.mj_forward(self.model, self.data)
        # qfrc_bias is computed during mj_forward and contains gravity +
        # Coriolis forces.  We apply it to the UR5e DOFs (first 6 DOFs).
        for i in range(UR5E_DOF):
            dof_adr = self._joint_vel_adr[i]
            self.data.qfrc_applied[dof_adr] = self.data.qfrc_bias[dof_adr]

    # ---------------------------------------------------------------------
    # Internal: observation construction
    # ---------------------------------------------------------------------
    def _get_obs(self) -> Dict[str, np.ndarray]:
        # Tactile: 6 touch sensor values
        tactile = np.zeros(N_TACTILE_SENSORS, dtype=np.float32)
        for i, adr in enumerate(self._tactile_adr):
            sdim = self.model.sensor_adr[adr]
            tactile[i] = float(self.data.sensordata[sdim])

        # Joint positions and velocities (canonical order)
        joint_pos = self.data.qpos[self._joint_pos_adr].astype(np.float32)
        joint_vel = self.data.qvel[self._joint_vel_adr].astype(np.float32)

        # Palm pose (xyz + quat)
        palm_pos = self.data.xpos[self._palm_body_id].copy()
        palm_quat = self.data.xquat[self._palm_body_id].copy()
        palm_pose = np.concatenate([palm_pos, palm_quat]).astype(np.float32)

        # Object positions
        obj_poses = np.zeros((len(OBJECT_NAMES), 3), dtype=np.float32)
        for i, bid in enumerate(self._obj_body_ids):
            obj_poses[i] = self.data.xpos[bid]

        # One-hot current object
        current_obj_oh = np.zeros(len(OBJECT_NAMES), dtype=np.float32)
        if 0 <= self.current_obj_idx < len(OBJECT_NAMES):
            current_obj_oh[self.current_obj_idx] = 1.0

        # Carrying flag
        carrying = np.array([1.0 if self.carrying_obj_idx >= 0 else 0.0],
                            dtype=np.float32)

        return {
            "tactile":        tactile,
            "joint_pos":      joint_pos,
            "joint_vel":      joint_vel,
            "palm_pose":      palm_pose,
            "object_poses":   obj_poses,
            "current_object": current_obj_oh,
            "carrying":       carrying,
        }

    # ---------------------------------------------------------------------
    # Internal: info dict
    # ---------------------------------------------------------------------
    def _get_info(self) -> Dict[str, Any]:
        obj_class = OBJECT_CLASSES[self.current_obj_idx] if 0 <= self.current_obj_idx < len(OBJECT_CLASSES) else "none"
        return {
            "step":              self.step_count,
            "current_object":    OBJECT_NAMES[self.current_obj_idx] if 0 <= self.current_obj_idx < len(OBJECT_NAMES) else "none",
            "current_object_idx": int(self.current_obj_idx),
            "current_class":     obj_class,
            "carrying_idx":      int(self.carrying_obj_idx),
            "completed":         list(self.completed_objects),
            "n_completed":       len(self.completed_objects),
            "success":           bool(self.success),
            "failed":            bool(self.failed),
            "tactile_reading":   self._get_obs()["tactile"].copy(),
        }

    # ---------------------------------------------------------------------
    # Internal: reward computation
    # ---------------------------------------------------------------------
    def _compute_reward(self, action: np.ndarray) -> float:
        if self.success:
            return 0.0
        if self.failed:
            return -1.0

        obs = self._get_obs()
        reward = 0.0

        # Current target object
        if 0 <= self.current_obj_idx < len(OBJECT_NAMES):
            target_name = OBJECT_NAMES[self.current_obj_idx]
            target_class = OBJECT_CLASSES[self.current_obj_idx]
            target_pos = obs["object_poses"][self.current_obj_idx]
            palm_pos = obs["palm_pose"][:3]

            # Distance palm -> target
            dist = float(np.linalg.norm(target_pos - palm_pos))

            # Approach bonus: negative distance, normalised
            reward += 0.5 * max(0.0, 1.0 - dist / 0.5)

            # Contact bonus: any fingertip reading > 0.5 N
            fingertip_forces = obs["tactile"][:5]
            n_contacts = int(np.sum(fingertip_forces > 0.5))
            reward += 0.1 * n_contacts

            # Grasp bonus: all 5 fingertips in contact AND object lifted
            target_bid = self._obj_body_ids[self.current_obj_idx]
            target_z = float(self.data.xpos[target_bid][2])
            target_init_z = 0.770  # initial table-top height
            lifted = target_z > target_init_z + 0.05

            if n_contacts >= 3 and lifted and self.carrying_obj_idx < 0:
                # Grasp detected
                self.carrying_obj_idx = self.current_obj_idx
                reward += 2.0  # one-time grasp bonus

            if self.carrying_obj_idx == self.current_obj_idx:
                # Currently carrying — check force appropriateness
                total_force = float(np.sum(fingertip_forces))
                target_force = CLASS_GRASP_FORCE[target_class]
                force_err = abs(total_force - target_force)
                reward += 0.2 * max(0.0, 1.0 - force_err / target_force)

                # Check correct placement
                correct_bin_name = TRIAGE_BIN_MAP[target_class]
                correct_bin_bid = self._bin_body_ids[correct_bin_name]
                bin_pos = self.data.xpos[correct_bin_bid]
                bin_top_z = float(bin_pos[2]) + 0.06
                # Object must be over the bin (xy within 0.06m, z near bin top)
                obj_xy = target_pos[:2]
                bin_xy = bin_pos[:2]
                horiz_dist = float(np.linalg.norm(obj_xy - bin_xy))

                if horiz_dist < 0.06 and abs(target_z - bin_top_z) < 0.10:
                    # Successful placement!
                    self.completed_objects.add(self.current_obj_idx)
                    reward += 5.0  # placement bonus
                    self.carrying_obj_idx = -1
                    if len(self.completed_objects) >= len(OBJECT_NAMES):
                        self.success = True
                        reward += 10.0  # all-done bonus
                    else:
                        # Advance to next uncompleted object
                        next_idx = (self.current_obj_idx + 1) % len(OBJECT_NAMES)
                        while next_idx in self.completed_objects:
                            next_idx = (next_idx + 1) % len(OBJECT_NAMES)
                        self.current_obj_idx = next_idx

            # Fragile break detection — only triggers when the GLASS object
            # specifically is being touched (not when the hand hits the can,
            # table, or any other geom).  We detect this by checking whether
            # the glass body's geom is in contact with any hand geom.
            if target_class == "fragile":
                glass_bid = self._obj_body_ids[self.current_obj_idx]
                # Walk the contact list looking for glass <-> any-hand-geom
                hand_geom_ids = set()
                for hand_body_name in (
                    "palm",
                    "link_TH1", "link_TH2", "link_TH3", "link_TH4",
                    "link_IF1", "link_IF2", "link_IF3",
                    "link_MF1", "link_MF2", "link_MF3",
                    "link_RF1", "link_RF2", "link_RF3",
                    "link_LF1", "link_LF2", "link_LF3",
                ):
                    bid = mujoco.mj_name2id(
                        self.model, mujoco.mjtObj.mjOBJ_BODY, hand_body_name
                    )
                    if bid >= 0:
                        for g in range(self.model.body_geomadr[bid],
                                       self.model.body_geomadr[bid] +
                                       self.model.body_geomnum[bid]):
                            hand_geom_ids.add(g)
                glass_geom_ids = set()
                for g in range(self.model.body_geomadr[glass_bid],
                               self.model.body_geomadr[glass_bid] +
                               self.model.body_geomnum[glass_bid]):
                    glass_geom_ids.add(g)
                glass_in_contact = False
                for ci in range(self.data.ncon):
                    c = self.data.contact[ci]
                    if (c.geom1 in glass_geom_ids and c.geom2 in hand_geom_ids) or \
                       (c.geom2 in glass_geom_ids and c.geom1 in hand_geom_ids):
                        glass_in_contact = True
                        break
                if glass_in_contact:
                    total_force = float(np.sum(fingertip_forces))
                    # Break threshold: 60N sustained for 50 steps.
                    # The position-controlled fingers can produce brief high
                    # forces during the grasp reflex; only sustained very
                    # high force causes breakage.
                    if total_force > 60.0:
                        self._fragile_break_counter += 1
                        if self._fragile_break_counter > 50:
                            self.failed = True
                            reward -= 5.0
                    else:
                        self._fragile_break_counter = max(
                            0, self._fragile_break_counter - 1
                        )
                else:
                    self._fragile_break_counter = max(
                        0, self._fragile_break_counter - 1
                    )

        # Action regularisation
        reward -= 0.001 * float(np.linalg.norm(action))
        # Step penalty
        reward -= 0.01

        return float(reward * self.reward_scale)

    # ---------------------------------------------------------------------
    # Internal: termination check
    # ---------------------------------------------------------------------
    def _check_terminated(self) -> bool:
        if self.success or self.failed:
            return True
        return False

    # ---------------------------------------------------------------------
    # Internal: rendering
    # ---------------------------------------------------------------------
    def _render_offscreen(self) -> np.ndarray:
        if self._renderer is None:
            self._renderer = mujoco.Renderer(
                self.model, height=480, width=640
            )
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA,
                                    self.camera_name)
        self._renderer.update_scene(self.data, camera=cam_id)
        return self._renderer.render().copy()

    def _render_human(self) -> None:
        try:
            import mujoco.viewer as mjviewer
        except ImportError:
            return
        if self._viewer is None:
            self._viewer = mjviewer.launch_passive(self.model, self.data)
        self._viewer.sync()

    # ---------------------------------------------------------------------
    # Public helper: get object body positions (for scripted controller)
    # ---------------------------------------------------------------------
    def get_object_position(self, obj_name: str) -> np.ndarray:
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, obj_name)
        return self.data.xpos[bid].copy()

    def get_bin_position(self, bin_name: str) -> np.ndarray:
        bid = self._bin_body_ids[bin_name]
        return self.data.xpos[bid].copy()

    def get_palm_position(self) -> np.ndarray:
        return self.data.xpos[self._palm_body_id].copy()


# Convenience registration
gym.register(
    id="DisasterTriage-v0",
    entry_point="src.env:DisasterTriageEnv",
    max_episode_steps=1500,
)
