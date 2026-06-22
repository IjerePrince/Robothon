"""
Unit tests for the Disaster Triage Robothon environment.

Run with:  pytest tests/ -v
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.env import DisasterTriageEnv
from src.scripted_controller import ScriptedController, State, run_episode
from src import (
    ROBOT_DOF, UR5E_DOF, HAND_DOF, N_TACTILE_SENSORS,
    OBJECT_NAMES, OBJECT_CLASSES, TRIAGE_BIN_MAP,
)


# ---------------------------------------------------------------------------
# Scene model
# ---------------------------------------------------------------------------
class TestSceneModel:
    def test_scene_loads(self):
        env = DisasterTriageEnv()
        assert env.model.nq > 0
        assert env.model.nv > 0
        env.close()

    def test_actuator_count(self):
        env = DisasterTriageEnv()
        assert env.model.nu == ROBOT_DOF, \
            f"Expected {ROBOT_DOF} actuators, got {env.model.nu}"
        env.close()

    def test_sensor_count(self):
        env = DisasterTriageEnv()
        assert env.model.nsensor >= 41, \
            f"Expected >= 41 sensors, got {env.model.nsensor}"
        # 6 tactile + 6 UR5e pos + 6 UR5e vel + 16 hand pos + 7 frame
        assert env.model.nsensordata >= 56
        env.close()

    def test_equality_constraints(self):
        env = DisasterTriageEnv()
        assert env.model.neq == 2, \
            f"Expected 2 equality constraints, got {env.model.neq}"
        env.close()

    def test_tactile_sensors_present(self):
        env = DisasterTriageEnv()
        import mujoco
        sensor_names = [
            mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_SENSOR, n)
            for n in range(env.model.nsensor)
        ]
        for expected in ("s_tactile_TH", "s_tactile_IF", "s_tactile_MF",
                          "s_tactile_RF", "s_tactile_LF", "s_tactile_palm"):
            assert expected in sensor_names, f"Missing tactile sensor: {expected}"
        env.close()


# ---------------------------------------------------------------------------
# Environment API
# ---------------------------------------------------------------------------
class TestEnvironment:
    def test_reset_returns_valid_obs(self):
        env = DisasterTriageEnv()
        obs, info = env.reset(seed=42)
        assert "tactile" in obs
        assert "joint_pos" in obs
        assert "joint_vel" in obs
        assert "palm_pose" in obs
        assert "object_poses" in obs
        assert "current_object" in obs
        assert "carrying" in obs
        assert obs["tactile"].shape == (N_TACTILE_SENSORS,)
        assert obs["joint_pos"].shape == (ROBOT_DOF,)
        assert obs["joint_vel"].shape == (ROBOT_DOF,)
        assert obs["palm_pose"].shape == (7,)
        assert obs["object_poses"].shape == (len(OBJECT_NAMES), 3)
        assert obs["current_object"].shape == (len(OBJECT_NAMES),)
        env.close()

    def test_action_space_shape(self):
        env = DisasterTriageEnv()
        assert env.action_space.shape == (ROBOT_DOF,)
        env.close()

    def test_step_returns_5_tuple(self):
        env = DisasterTriageEnv()
        env.reset(seed=0)
        action = env.action_space.sample()
        result = env.step(action)
        assert len(result) == 5, "Gymnasium step() must return 5-tuple"
        obs, reward, terminated, truncated, info = result
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        env.close()

    def test_deterministic_reset(self):
        env = DisasterTriageEnv()
        obs1, _ = env.reset(seed=0)
        obs2, _ = env.reset(seed=0)
        np.testing.assert_array_equal(obs1["joint_pos"], obs2["joint_pos"])
        env.close()

    def test_physics_stable_over_500_steps(self):
        env = DisasterTriageEnv()
        env.reset(seed=0)
        for _ in range(500):
            action = env.action_space.sample() * 0.1
            obs, _, _, _, _ = env.step(action)
        assert not np.isnan(obs["joint_pos"]).any(), "NaN in joint_pos after 500 steps"
        assert not np.isnan(obs["tactile"]).any(), "NaN in tactile after 500 steps"
        env.close()


# ---------------------------------------------------------------------------
# Scripted controller
# ---------------------------------------------------------------------------
class TestScriptedController:
    def test_controller_initial_state(self):
        env = DisasterTriageEnv()
        ctrl = ScriptedController(env)
        assert ctrl.state.fsm_state == State.APPROACH
        assert ctrl.state.current_obj_idx == 0
        env.close()

    def test_controller_produces_valid_action(self):
        env = DisasterTriageEnv()
        ctrl = ScriptedController(env)
        obs, _ = env.reset(seed=0)
        action = ctrl.compute_action(obs)
        assert action.shape == (ROBOT_DOF,)
        assert (action >= -1.0).all() and (action <= 1.0).all()
        env.close()

    def test_short_episode_runs_without_crash(self):
        env = DisasterTriageEnv(max_steps=200)
        ctrl = ScriptedController(env)
        stats = run_episode(env, ctrl, max_steps=200, verbose=False)
        assert stats["steps"] <= 200
        assert stats["n_completed"] >= 0
        env.close()


# ---------------------------------------------------------------------------
# Object taxonomy
# ---------------------------------------------------------------------------
class TestObjectTaxonomy:
    def test_object_count(self):
        assert len(OBJECT_NAMES) == 5
        assert len(OBJECT_CLASSES) == 5

    def test_triage_bin_mapping(self):
        # Each class must have a valid bin
        for cls in OBJECT_CLASSES:
            assert cls in TRIAGE_BIN_MAP, f"Class {cls} missing from TRIAGE_BIN_MAP"

    def test_robot_dof(self):
        assert ROBOT_DOF == UR5E_DOF + HAND_DOF
        assert UR5E_DOF == 6
        assert HAND_DOF == 16
