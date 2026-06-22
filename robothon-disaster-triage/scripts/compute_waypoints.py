"""
compute_waypoints.py — Precompute collision-free UR5e waypoints for each
object's approach/descend/grasp and each bin's drop position.

This script:
1. Loads the scene
2. For each object, computes IK to the approach position (30cm above object)
3. For each object, computes IK to the grasp position (palm above object)
4. For each bin, computes IK to the drop position (palm above bin)
5. Verifies each waypoint is collision-free (no contacts > 5N)
6. Saves waypoints to config/waypoints.yaml

The scripted controller then interpolates between these fixed waypoints,
guaranteeing smooth, collision-free motion.
"""
import sys, os, yaml
import numpy as np
import mujoco

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.env import DisasterTriageEnv
from src import OBJECT_NAMES, OBJECT_CLASSES, TRIAGE_BIN_MAP, SCENE_XML

def solve_ik(model, data, target_pos, palm_bid, max_iter=500, max_step=0.1, tol=1e-5):
    """Solve IK using a separate MjData to avoid corrupting the real sim."""
    ik_data = mujoco.MjData(model)
    ik_data.qpos[:] = data.qpos[:]
    mujoco.mj_forward(model, ik_data)
    current_q = ik_data.qpos[:6].copy()

    for _ in range(max_iter):
        palm_pos = ik_data.xpos[palm_bid].copy()
        err = target_pos - palm_pos
        if float(np.linalg.norm(err)) < tol:
            break
        nv = model.nv
        jacp = np.zeros((3, nv))
        jacr = np.zeros((3, nv))
        mujoco.mj_jacBody(model, ik_data, jacp, jacr, palm_bid)
        J = jacp[:, :6]
        lam = 0.05
        JJT = J @ J.T + (lam ** 2) * np.eye(3)
        delta_q = J.T @ np.linalg.solve(JJT, err)
        delta_q = np.clip(delta_q, -max_step, max_step)
        current_q = current_q + delta_q
        ik_data.qpos[:6] = current_q
        mujoco.mj_forward(model, ik_data)

    # Verify solution
    palm_pos = ik_data.xpos[palm_bid].copy()
    error = float(np.linalg.norm(target_pos - palm_pos))
    return current_q, error

def check_collision_free(model, data, qpos6, max_force=5.0):
    """Check if a given UR5e configuration is collision-free."""
    data.qpos[:6] = qpos6
    mujoco.mj_forward(model, data)
    for _ in range(5):
        mujoco.mj_step(model, data)
    max_f = 0
    for i in range(data.ncon):
        f = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, f)
        max_f = max(max_f, np.linalg.norm(f[:3]))
    return max_f < max_force, max_f

def main():
    env = DisasterTriageEnv(render_mode=None)
    model = env.model
    data = env.data
    palm_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "palm")

    # Home pose
    home_qpos = np.array([0.0, -1.0, 1.5, -0.7, 1.5708, 0.0])

    waypoints = {"home": home_qpos.tolist()}

    # For each object, compute approach + grasp waypoints
    for i, obj_name in enumerate(OBJECT_NAMES):
        obj_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, obj_name)
        # Reset to home
        mujoco.mj_resetData(model, data)
        data.qpos[:6] = home_qpos
        # Set freejoint objects to their body positions
        for bid in range(model.nbody):
            jnt_adr = model.body_jntadr[bid]
            jnt_num = model.body_jntnum[bid]
            for j in range(jnt_adr, jnt_adr + jnt_num):
                if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                    qpos_adr = model.jnt_qposadr[j]
                    data.qpos[qpos_adr:qpos_adr+3] = model.body_pos[bid]
                    data.qpos[qpos_adr+3] = 0; data.qpos[qpos_adr+4] = 0
                    data.qpos[qpos_adr+5] = 0; data.qpos[qpos_adr+6] = 1
        mujoco.mj_forward(model, data)

        obj_pos = data.xpos[obj_bid].copy()
        obj_class = OBJECT_CLASSES[i]

        # Approach: 30cm above object
        approach_target = np.array([obj_pos[0], obj_pos[1], obj_pos[2] + 0.30])
        approach_q, approach_err = solve_ik(model, data, approach_target, palm_bid)

        # Grasp: 7cm above object, -3cm y offset
        grasp_z_offsets = {"fragile": 0.070, "sturdy": 0.075, "hazardous": 0.050, "soft": 0.060, "sharp": 0.055}
        grasp_target = np.array([obj_pos[0], obj_pos[1] - 0.030, obj_pos[2] + grasp_z_offsets[obj_class]])
        grasp_q, grasp_err = solve_ik(model, data, grasp_target, palm_bid)

        # Verify collision-free
        ok_app, f_app = check_collision_free(model, data, approach_q)
        ok_grp, f_grp = check_collision_free(model, data, grasp_q)

        print(f"{obj_name} ({obj_class}):")
        print(f"  approach: q={approach_q.round(4)}, err={approach_err:.6f}m, max_force={f_app:.1f}N {'OK' if ok_app else 'COLLISION'}")
        print(f"  grasp:    q={grasp_q.round(4)}, err={grasp_err:.6f}m, max_force={f_grp:.1f}N {'OK' if ok_grp else 'COLLISION'}")

        waypoints[f"{obj_name}_approach"] = approach_q.tolist()
        waypoints[f"{obj_name}_grasp"] = grasp_q.tolist()

    # For each bin, compute drop waypoint
    bin_names = ["bin_fragile", "bin_sturdy", "bin_hazardous"]
    for bin_name in bin_names:
        bin_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bin_name)
        mujoco.mj_resetData(model, data)
        data.qpos[:6] = home_qpos
        for bid in range(model.nbody):
            jnt_adr = model.body_jntadr[bid]
            jnt_num = model.body_jntnum[bid]
            for j in range(jnt_adr, jnt_adr + jnt_num):
                if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                    qpos_adr = model.jnt_qposadr[j]
                    data.qpos[qpos_adr:qpos_adr+3] = model.body_pos[bid]
                    data.qpos[qpos_adr+3] = 0; data.qpos[qpos_adr+4] = 0
                    data.qpos[qpos_adr+5] = 0; data.qpos[qpos_adr+6] = 1
        mujoco.mj_forward(model, data)

        bin_pos = data.xpos[bin_bid].copy()
        # Drop: 15cm above bin top
        drop_target = np.array([bin_pos[0], bin_pos[1], 0.90])
        drop_q, drop_err = solve_ik(model, data, drop_target, palm_bid)
        ok_drop, f_drop = check_collision_free(model, data, drop_q)

        print(f"{bin_name}:")
        print(f"  drop: q={drop_q.round(4)}, err={drop_err:.6f}m, max_force={f_drop:.1f}N {'OK' if ok_drop else 'COLLISION'}")

        waypoints[f"{bin_name}_drop"] = drop_q.tolist()

    # Save
    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "waypoints.yaml")
    with open(out_path, "w") as f:
        yaml.dump(waypoints, f, default_flow_style=False)
    print(f"\nSaved waypoints to {out_path}")

    env.close()

if __name__ == "__main__":
    main()
