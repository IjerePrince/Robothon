"""
Disaster Triage Robothon — source package.

Modules:
    env                   — MuJoCo + Gymnasium environment for the triage task
    robots                — Robot API (forward kinematics, joint control, named DOF groups)
    sensors               — Tactile sensor post-processing (force -> features)
    task                  — Disaster triage task state machine + reward
    scripted_controller   — Heuristic grasp + tactile-identify auto-controller
    rl_train              — PPO training entrypoint (stable-baselines3)
    rl_eval               — Load + evaluate a trained policy
    demo_render           — Offscreen renderer -> MP4 demo video
    utils                 — Path + config helpers
    cli                   — `python -m src.cli <mode>` entrypoint

Top-level package exports:
    DisasterTriageEnv  — the Gymnasium environment class
    ROBOT_DOF          — total actuated DOF (22)
    N_TACTILE_SENSORS  — number of tactile channels (6)
    OBJECT_NAMES       — tuple of object body names
    OBJECT_CLASSES     — tuple of class labels (fragile/sturdy/hazardous/soft/sharp)
"""

from __future__ import annotations
from pathlib import Path

# ---------------------------------------------------------------------------
# Package-level constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR   = PROJECT_ROOT / "models"
SCENE_XML    = MODELS_DIR / "scene.xml"
CONFIG_DIR   = PROJECT_ROOT / "config"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
DEMOS_DIR    = PROJECT_ROOT / "demos"

# Robot configuration
ROBOT_DOF = 22           # 6 UR5e + 16 Shadow Hand Lite
UR5E_DOF  = 6
HAND_DOF  = 16
N_TACTILE_SENSORS = 6    # 5 fingertips + 1 palm

# Object taxonomy — order matches sensor reading order
OBJECT_NAMES = ("obj_glass", "obj_can", "obj_vial", "obj_toy", "obj_tool")
OBJECT_LABELS = ("glass", "can", "vial", "toy", "tool")
OBJECT_CLASSES = ("fragile", "sturdy", "hazardous", "soft", "sharp")

# Mapping: object class -> correct triage bin
TRIAGE_BIN_MAP = {
    "fragile":   "bin_fragile",
    "sturdy":    "bin_sturdy",
    "hazardous": "bin_hazardous",
    # 'soft' and 'sharp' objects go into the 'sturdy' bin for simplicity
    "soft":      "bin_sturdy",
    "sharp":     "bin_hazardous",
}

# Per-class grasp force targets (Newtons) — used by the scripted controller
# and as reward signal for the RL policy
CLASS_GRASP_FORCE = {
    "fragile":   4.0,    # gentle
    "sturdy":    12.0,   # firm
    "hazardous": 6.0,    # careful (between fragile and sturdy)
    "soft":      3.0,    # very gentle (deformable)
    "sharp":     8.0,    # moderate (avoid cuts but firm grip)
}

# Per-class fingertip tactile signature templates (mean over 5 sensors)
# Used by the tactile-identification heuristic in scripted_controller.
# These represent the expected total fingertip force (N) during a grasp.
CLASS_TACTILE_SIGNATURE = {
    "fragile":   (2.5, 0.8),   # low force, low variance
    "sturdy":    (10.0, 2.0),  # high force, moderate variance
    "hazardous": (4.0, 1.2),   # moderate force, low variance (small object)
    "soft":      (1.5, 0.5),   # very low force (compliant)
    "sharp":     (6.0, 3.5),   # moderate force, HIGH variance (uneven contact)
}

__all__ = [
    "PROJECT_ROOT", "MODELS_DIR", "SCENE_XML", "CONFIG_DIR",
    "CHECKPOINT_DIR", "DEMOS_DIR",
    "ROBOT_DOF", "UR5E_DOF", "HAND_DOF", "N_TACTILE_SENSORS",
    "OBJECT_NAMES", "OBJECT_LABELS", "OBJECT_CLASSES",
    "TRIAGE_BIN_MAP", "CLASS_GRASP_FORCE", "CLASS_TACTILE_SIGNATURE",
]
