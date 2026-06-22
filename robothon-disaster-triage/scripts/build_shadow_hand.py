"""
build_shadow_hand.py — Generates the Shadow Hand Lite (16-DOF) MJCF model.

Shadow Hand Lite is a research-friendly 16-DOF dexterous hand derived from
the Shadow Robot Hand. Compared to the full 24-DOF Shadow Hand, it
collapses the DIP joint into the PIP joint (tendon-coupled in the real
hand) and removes the thumb's distal segment, yielding:

    Thumb  : 4 DOF  (CMC ab/ad, CMC flex, MCP flex, IP flex)
    Index  : 3 DOF  (MCP ab/ad, MCP flex, PIP flex)
    Middle : 3 DOF  (MCP ab/ad, MCP flex, PIP flex)
    Ring   : 3 DOF  (MCP ab/ad, MCP flex, PIP flex)
    Little : 3 DOF  (MCP ab/ad, MCP flex, PIP flex)
    ----------------------------------------
    TOTAL  : 16 DOF

Each fingertip carries a tactile sensor site (5 sites total), and the palm
carries a 6-axis force/torque sensor.  All links are built from MuJoCo
primitives (capsules/spheres) so the project remains fully self-contained.

Run:
    python3 build_shadow_hand.py
to (re)generate models/shadow_hand_lite.xml.
"""
from __future__ import annotations
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Finger specification table
# ---------------------------------------------------------------------------
# Each finger is described by:
#   name        — finger name
#   base_xyz    — mount position on the palm (metres, palm-local frame)
#   base_euler  — base orientation (degrees) so fingers splay naturally
#   segments    — list of (joint_name, joint_axis, segment_len, segment_r,
#                  joint_pos_in_parent, geom_pos, tactile_site_pos)
#
# Convention: joint axis "0 1 0" => finger flexes in the palm plane
#             joint axis "1 0 0" => finger abducts (spread) perpendicular
# All ranges are ±1.57 rad (≈ 90 deg), damping 0.3, armature 0.005.

FINGERS: List[Dict] = [
    # ----------------------------- THUMB ----------------------------------
    {
        "name": "TH",
        "base_xyz": "0.045 -0.040 0.018",
        "base_euler": "0 30 35",  # thumb opposes fingers
        "segments": [
            # CMC abduction
            {"joint": "A_THJ1", "axis": "1 0 0", "len": 0.040, "r": 0.009,
             "joint_pos": "0 0 0",     "geom_pos": "0 0 -0.020", "tactile": None},
            # CMC flexion
            {"joint": "A_THJ2", "axis": "0 1 0", "len": 0.038, "r": 0.0085,
             "joint_pos": "0 0 -0.040", "geom_pos": "0 0 -0.019", "tactile": None},
            # MCP flexion
            {"joint": "A_THJ3", "axis": "0 1 0", "len": 0.030, "r": 0.008,
             "joint_pos": "0 0 -0.038", "geom_pos": "0 0 -0.015", "tactile": None},
            # IP flexion (thumb has only one interphalangeal joint)
            {"joint": "A_THJ4", "axis": "0 1 0", "len": 0.028, "r": 0.0075,
             "joint_pos": "0 0 -0.030", "geom_pos": "0 0 -0.014", "tactile": "0 0 -0.026"},
        ],
    },
    # ----------------------------- INDEX ----------------------------------
    {
        "name": "IF",
        "base_xyz": "0.022 0.030 0.000",
        "base_euler": "0 0 -8",
        "segments": [
            {"joint": "A_IFJ1", "axis": "1 0 0", "len": 0.040, "r": 0.008,
             "joint_pos": "0 0 0",     "geom_pos": "0 0 0.020",  "tactile": None},
            {"joint": "A_IFJ2", "axis": "0 1 0", "len": 0.035, "r": 0.0075,
             "joint_pos": "0 0 0.040", "geom_pos": "0 0 0.0175", "tactile": None},
            {"joint": "A_IFJ3", "axis": "0 1 0", "len": 0.028, "r": 0.007,
             "joint_pos": "0 0 0.035", "geom_pos": "0 0 0.014",  "tactile": "0 0 0.026"},
        ],
    },
    # ----------------------------- MIDDLE ---------------------------------
    {
        "name": "MF",
        "base_xyz": "0.005 0.034 0.000",
        "base_euler": "0 0 0",
        "segments": [
            {"joint": "A_MFJ1", "axis": "1 0 0", "len": 0.042, "r": 0.008,
             "joint_pos": "0 0 0",     "geom_pos": "0 0 0.021",  "tactile": None},
            {"joint": "A_MFJ2", "axis": "0 1 0", "len": 0.036, "r": 0.0075,
             "joint_pos": "0 0 0.042", "geom_pos": "0 0 0.018",  "tactile": None},
            {"joint": "A_MFJ3", "axis": "0 1 0", "len": 0.028, "r": 0.007,
             "joint_pos": "0 0 0.036", "geom_pos": "0 0 0.014",  "tactile": "0 0 0.026"},
        ],
    },
    # ----------------------------- RING -----------------------------------
    {
        "name": "RF",
        "base_xyz": "-0.012 0.030 0.000",
        "base_euler": "0 0 8",
        "segments": [
            {"joint": "A_RFJ1", "axis": "1 0 0", "len": 0.038, "r": 0.0075,
             "joint_pos": "0 0 0",     "geom_pos": "0 0 0.019",  "tactile": None},
            {"joint": "A_RFJ2", "axis": "0 1 0", "len": 0.032, "r": 0.007,
             "joint_pos": "0 0 0.038", "geom_pos": "0 0 0.016",  "tactile": None},
            {"joint": "A_RFJ3", "axis": "0 1 0", "len": 0.026, "r": 0.0065,
             "joint_pos": "0 0 0.032", "geom_pos": "0 0 0.013",  "tactile": "0 0 0.024"},
        ],
    },
    # ----------------------------- LITTLE ---------------------------------
    {
        "name": "LF",
        "base_xyz": "-0.034 0.024 0.000",
        "base_euler": "0 0 18",
        "segments": [
            {"joint": "A_LFJ1", "axis": "1 0 0", "len": 0.034, "r": 0.007,
             "joint_pos": "0 0 0",     "geom_pos": "0 0 0.017",  "tactile": None},
            {"joint": "A_LFJ2", "axis": "0 1 0", "len": 0.028, "r": 0.0065,
             "joint_pos": "0 0 0.034", "geom_pos": "0 0 0.014",  "tactile": None},
            {"joint": "A_LFJ3", "axis": "0 1 0", "len": 0.024, "r": 0.006,
             "joint_pos": "0 0 0.028", "geom_pos": "0 0 0.012",  "tactile": "0 0 0.022"},
        ],
    },
]

TACTILE_SITE_NAMES = [
    "tactile_TH", "tactile_IF", "tactile_MF", "tactile_RF", "tactile_LF",
]


def _add(parent: ET.Element, tag: str, **attrs) -> ET.Element:
    """Helper: append a child element with cleaned attribute keys.

    Translates the trailing ``class_`` keyword (used because ``class`` is a
    Python reserved word) into the MJCF ``class`` attribute.
    """
    cleaned = {}
    for k, v in attrs.items():
        if v is None:
            continue
        key = "class" if k == "class_" else k
        cleaned[key] = str(v) if not isinstance(v, str) else v
    return ET.SubElement(parent, tag, cleaned)


def build_shadow_hand_xml() -> ET.Element:
    """Build the root <mujoco> element for the Shadow Hand Lite.

    The generated XML is self-contained (no <include>) so it loads standalone
    and also embeds cleanly into the parent scene.xml via <include>.
    """
    mujoco = ET.Element("mujoco", {"model": "shadow_hand_lite"})
    _add(mujoco, "compiler", angle="radian", autolimits="true",
         meshdir="meshes/shadow_hand", discardvisual="false")
    _add(mujoco, "option", timestep="0.002", integrator="implicitfast",
         iterations="50", solver="Newton")

    # Default class hierarchy (hand-local; does not collide with scene.xml)
    default_root = _add(mujoco, "default")
    _add(default_root, "joint", damping="0.3", armature="0.005", limited="true")
    _add(default_root, "geom", condim="6", friction="1.5 0.05 0.02",
         rgba="0.85 0.78 0.65 1")
    finger_default = _add(default_root, "default", **{"class": "shadow_finger"})
    _add(finger_default, "joint", range="-1.57 1.57", damping="0.3", armature="0.005")
    _add(finger_default, "position", kp="40", ctrlrange="-1.57 1.57")
    _add(finger_default, "geom", condim="6", friction="1.5 0.05 0.02",
         rgba="0.85 0.78 0.65 1")

    # Tactile sensor default
    tactile_default = _add(default_root, "default", **{"class": "tactile_sensor"})
    _add(tactile_default, "site", group="3", rgba="0.1 0.9 0.3 0.4",
         size="0.006 0.006 0.006")

    # ============================================================
    # PALM body — root of the hand; this is what gets welded to UR5e
    # ============================================================
    palm = _add(mujoco, "worldbody")
    palm_body = _add(palm, "body", name="palm", pos="0 0 0",
                     euler="0 0 0")
    # Palm visual & collision (slightly rounded box)
    _add(palm_body, "geom", name="palm_geom",
         type="box", size="0.045 0.030 0.012", pos="0 0 0",
         rgba="0.92 0.86 0.74 1", class_="shadow_finger")
    # Palm tactile sensor — broad contact patch in the centre
    _add(palm_body, "site", name="tactile_palm", class_="tactile_sensor",
         pos="0 0.025 0.012", size="0.020 0.020 0.002", type="box")
    # Inertial element (so the palm has sensible mass even before fingers)
    _add(palm_body, "inertial", pos="0 0 0", mass="0.35",
         diaginertia="0.001 0.002 0.0025")

    # ============================================================
    # FINGERS — build each one by chaining <body> elements
    # ============================================================
    finger_idx_to_tip_site: List[str] = []  # ordered for sensor retrieval

    for fdef in FINGERS:
        parent_body = palm_body
        for i, seg in enumerate(fdef["segments"]):
            body_name = f"link_{fdef['name']}{i+1}"
            body = _add(parent_body, "body",
                        name=body_name,
                        pos=seg["joint_pos"],
                        euler=("0 0 0" if i > 0 else "0 0 0"))
            # Joint at the base of this body
            _add(body, "joint", name=seg["joint"],
                 type="hinge", axis=seg["axis"],
                 class_="shadow_finger", range="-1.57 1.57",
                 damping="0.3", armature="0.005")
            # Capsule geom for the segment
            _add(body, "geom", name=f"geom_{body_name}",
                 type="capsule",
                 fromto=f"0 0 0 0 0 {seg['len']:.4f}" if fdef["name"] != "TH"
                        else f"0 0 0 0 0 -{seg['len']:.4f}",
                 size=f"{seg['r']:.4f}", class_="shadow_finger")
            # Small sphere at the joint for visual smoothness
            _add(body, "geom", name=f"knuckle_{body_name}",
                 type="sphere", size=f"{seg['r']*1.05:.4f}",
                 pos="0 0 0", class_="shadow_finger")
            # Tactile site on the last segment only
            if seg["tactile"] is not None:
                site_name = f"tactile_{fdef['name']}"
                _add(body, "site", name=site_name,
                     class_="tactile_sensor",
                     pos=seg["tactile"], size="0.008 0.008 0.004",
                     type="box")
                finger_idx_to_tip_site.append(site_name)
            parent_body = body

    # ============================================================
    # ACTUATORS — one <position> actuator per DOF
    # ============================================================
    actuators = _add(mujoco, "actuator")
    for fdef in FINGERS:
        for seg in fdef["segments"]:
            _add(actuators, "position", name=seg["joint"].replace("A_", "act_"),
                 joint=seg["joint"], class_="shadow_finger",
                 kp="40", ctrlrange="-1.57 1.57", forcerange="-3 3")

    # ============================================================
    # SENSORS — fingertip touch sensors + palm contact normal force
    # ============================================================
    sensors = _add(mujoco, "sensor")
    for site_name in finger_idx_to_tip_site:
        _add(sensors, "touch", name=f"s_{site_name}", site=site_name)
    _add(sensors, "touch", name="s_tactile_palm", site="tactile_palm")

    return mujoco


def main():
    out_dir = Path(__file__).resolve().parent.parent / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "shadow_hand_lite.xml"
    root = build_shadow_hand_xml()
    # Indent for readability
    try:
        ET.indent(root, space="  ")
    except AttributeError:
        pass  # Python < 3.9 fallback
    tree = ET.ElementTree(root)
    tree.write(out_path, encoding="utf-8", xml_declaration=True)
    print(f"[OK] Wrote {out_path}")
    # Quick DOF sanity check
    total_joints = sum(len(f["segments"]) for f in FINGERS)
    print(f"[OK] Shadow Hand Lite has {total_joints} DOF across {len(FINGERS)} fingers.")


if __name__ == "__main__":
    main()
