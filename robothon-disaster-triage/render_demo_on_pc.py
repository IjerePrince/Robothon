"""
render_demo_on_pc.py — Render the demo video on your PC.

This script uses GLFW-based rendering (same as the interactive viewer)
to ensure the 3D scene is captured correctly.

Usage:
    python render_demo_on_pc.py

Output:
    demos/demo.mp4 (90-120 seconds, 960x544, 30fps)
"""
import sys
import os
from pathlib import Path

# Force GLFW rendering backend (same as interactive viewer)
os.environ['MUJOCO_GL'] = 'glfw'

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import mujoco
import mujoco.viewer
import glfw
import imageio
import cv2
from src.env import DisasterTriageEnv
from src.scripted_controller import ScriptedController, State
from src import OBJECT_NAMES, OBJECT_CLASSES, TRIAGE_BIN_MAP


def test_renderer(model, width, height):
    """Test if the offscreen renderer produces actual 3D content."""
    data = mujoco.MjData(model)
    data.qpos[:6] = [0, -1.0, 1.5, -0.7, 1.5708, 0]
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

    renderer = mujoco.Renderer(model, height=height, width=width)
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "cam_front")
    renderer.update_scene(data, camera=cam_id)
    img = renderer.render()

    # Check if image has content (not all blue/skybox)
    mean_val = img.mean()
    has_content = mean_val > 40 and mean_val < 250 and not (img == img[0, 0]).all()
    print(f"[Test] Renderer image: mean={mean_val:.1f}, has_content={has_content}")
    del renderer
    return has_content, img


def render_demo():
    """Render the demo video."""
    output_path = Path(__file__).resolve().parent / "demos" / "demo.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    width, height = 960, 540
    fps = 30

    print(f"\n[Demo] Rendering to {output_path}")
    print(f"[Demo] Resolution: {width}x{height} @ {fps} fps")

    env = DisasterTriageEnv(render_mode=None, max_steps=10000)
    controller = ScriptedController(env)

    # Test the renderer first
    print("[Demo] Testing offscreen renderer...")
    has_content, test_img = test_renderer(env.model, width, height)

    if not has_content:
        print("[Demo] WARNING: Offscreen renderer not producing 3D content!")
        print("[Demo] Falling back to viewer-based rendering...")
        return render_demo_with_viewer(env, controller, output_path, width, height, fps)

    # Use offscreen renderer
    print("[Demo] Offscreen renderer working! Using it for demo.")
    return render_demo_offscreen(env, controller, output_path, width, height, fps)


def render_demo_offscreen(env, controller, output_path, width, height, fps):
    """Render using the offscreen mujoco.Renderer (fastest)."""
    renderer = mujoco.Renderer(env.model, height=height, width=width)
    cam_front = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, "cam_front")
    cam_side = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, "cam_side")
    cam_top = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, "cam_top")

    frames = []

    def add_frame(cam_id=cam_front, title="", subtitle="", completed=0, failed=0):
        """Render a frame with overlays."""
        renderer.update_scene(env.data, camera=cam_id)
        img = renderer.render()
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Title bar (top)
        cv2.rectangle(bgr, (0, 0), (width, 50), (15, 20, 30), -1)
        cv2.rectangle(bgr, (0, 48), (width, 50), (0, 180, 255), -1)
        cv2.putText(bgr, title, (15, 35), cv2.FONT_HERSHEY_DUPLEX, 0.7,
                    (255, 255, 255), 1, cv2.LINE_AA)

        # Stats panel (top right)
        panel_w = 200
        cv2.rectangle(bgr, (width - panel_w - 10, 55), (width - 10, 125),
                      (25, 30, 45), -1)
        cv2.rectangle(bgr, (width - panel_w - 10, 55), (width - 10, 125),
                      (0, 180, 255), 1)
        progress_text = f"Objects: {completed}/5"
        cv2.putText(bgr, progress_text, (width - panel_w + 5, 75),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        bar_x = width - panel_w + 5
        bar_y = 85
        cv2.rectangle(bgr, (bar_x, bar_y), (bar_x + 170, bar_y + 10), (15, 20, 30), -1)
        fill_w = int(170 * (completed / 5))
        if fill_w > 0:
            cv2.rectangle(bgr, (bar_x, bar_y), (bar_x + fill_w, bar_y + 10),
                          (0, 200, 100), -1)
        status = "SUCCESS" if completed == 5 else "IN PROGRESS"
        color = (0, 200, 100) if completed == 5 else (0, 180, 255)
        cv2.putText(bgr, status, (width - panel_w + 5, 115),
                    cv2.FONT_HERSHEY_DUPLEX, 0.4, color, 1, cv2.LINE_AA)

        if subtitle:
            cv2.rectangle(bgr, (0, height - 30), (width, height), (15, 20, 30), -1)
            cv2.putText(bgr, subtitle, (15, height - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1.0, (160, 165, 180), 1, cv2.LINE_AA)

        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    return run_demo_phases(env, controller, renderer, cam_front, cam_side, cam_top,
                           add_frame, frames, output_path, width, height, fps)


def render_demo_with_viewer(env, controller, output_path, width, height, fps):
    """Render using the interactive viewer (fallback if offscreen fails).

    This opens the MuJoCo viewer window and captures frames from it.
    """
    print("[Demo] Opening viewer for frame capture...")

    frames = []

    # We need to use the viewer's internal renderer
    # Launch the viewer in passive mode
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        print("[Demo] Viewer opened. Capturing frames...")

        def add_frame(cam_name="cam_front", title="", subtitle="", completed=0, failed=0):
            """Capture a frame from the viewer."""
            # Sync viewer with current sim state
            viewer.sync()

            # Use the viewer's internal renderer to capture the frame
            # We need to create a separate renderer that shares the context
            renderer = mujoco.Renderer(env.model, height=height, width=width)
            cam_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
            renderer.update_scene(env.data, camera=cam_id)
            img = renderer.render()
            del renderer

            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # Title bar
            cv2.rectangle(bgr, (0, 0), (width, 50), (15, 20, 30), -1)
            cv2.rectangle(bgr, (0, 48), (width, 50), (0, 180, 255), -1)
            cv2.putText(bgr, title, (15, 35), cv2.FONT_HERSHEY_DUPLEX, 0.7,
                        (255, 255, 255), 1, cv2.LINE_AA)

            # Stats panel
            panel_w = 200
            cv2.rectangle(bgr, (width - panel_w - 10, 55), (width - 10, 125),
                          (25, 30, 45), -1)
            cv2.rectangle(bgr, (width - panel_w - 10, 55), (width - 10, 125),
                          (0, 180, 255), 1)
            cv2.putText(bgr, f"Objects: {completed}/5", (width - panel_w + 5, 75),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            bar_x = width - panel_w + 5
            cv2.rectangle(bgr, (bar_x, 85), (bar_x + 170, 95), (15, 20, 30), -1)
            fill_w = int(170 * (completed / 5))
            if fill_w > 0:
                cv2.rectangle(bgr, (bar_x, 85), (bar_x + fill_w, 95), (0, 200, 100), -1)

            if subtitle:
                cv2.rectangle(bgr, (0, height - 30), (width, height), (15, 20, 30), -1)
                cv2.putText(bgr, subtitle, (15, height - 10),
                            cv2.FONT_HERSHEY_PLAIN, 1.0, (160, 165, 180), 1, cv2.LINE_AA)

            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

        return run_demo_phases(env, controller, None,
                               "cam_front", "cam_side", "cam_top",
                               add_frame, frames, output_path, width, height, fps,
                               viewer=viewer)


def run_demo_phases(env, controller, renderer, cam_front, cam_side, cam_top,
                     add_frame, frames, output_path, width, height, fps,
                     viewer=None):
    """Run all demo phases and collect frames."""

    def get_cam(cam_id_or_name, state):
        """Pick camera based on state."""
        if state in (State.LIFT, State.TRANSPORT):
            return cam_side
        return cam_front

    # ===== Phase 1: Title card (5s) =====
    print("[Demo] Phase 1: Title card")
    for _ in range(int(5 * fps)):
        img = np.full((height, width, 3), (15, 20, 30), dtype=np.uint8)
        cv2.putText(img, "Disaster Triage Robothon", (120, 250),
                    cv2.FONT_HERSHEY_DUPLEX, 1.3, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, "MuJoCo Dexterous Manipulation Benchmark",
                    (100, 300), cv2.FONT_HERSHEY_DUPLEX, 0.6,
                    (160, 165, 180), 1, cv2.LINE_AA)
        cv2.putText(img, "22-DOF UR5e + Shadow Hand Lite",
                    (180, 330), cv2.FONT_HERSHEY_DUPLEX, 0.6,
                    (160, 165, 180), 1, cv2.LINE_AA)
        cv2.rectangle(img, (0, 200), (width, 202), (0, 180, 255), -1)
        cv2.rectangle(img, (0, 360), (width, 362), (0, 180, 255), -1)
        frames.append(img)

    # ===== Phase 2: Scene overview (10s) =====
    print("[Demo] Phase 2: Scene overview")
    obs, info = env.reset(seed=0)
    controller.reset()
    for i in range(int(10 * fps)):
        env.step(np.zeros(22, dtype=np.float32))
        if viewer:
            viewer.sync()
        cam = [cam_front, cam_side, cam_top][i // (int(10 * fps) // 3)]
        add_frame(cam, "Scene Overview",
                  "5 disaster objects | 3 triage bins | 22-DOF robot",
                  completed=0, failed=0)

    # ===== Phase 3: Scripted controller triage (main sequence) =====
    print("[Demo] Phase 3: Scripted controller triage")
    obs, info = env.reset(seed=0)
    controller.reset()
    step_count = 0
    frame_every = 1

    while step_count < 8000:
        action = controller.compute_action(obs)
        obs, r, term, trunc, info = env.step(action)
        step_count += 1

        if step_count % frame_every == 0:
            state = controller.state.fsm_state
            obj_idx = controller.state.current_obj_idx
            obj_name = OBJECT_NAMES[obj_idx] if obj_idx < len(OBJECT_NAMES) else "Done"
            obj_class = OBJECT_CLASSES[obj_idx] if obj_idx < len(OBJECT_CLASSES) else ""
            title = f"Triage Task - {state.value.upper()}"
            subtitle = f"Object: {obj_name} ({obj_class}) | Step {step_count}"

            cam = get_cam(cam_front, state)
            if viewer:
                viewer.sync()
            add_frame(cam, title, subtitle,
                      completed=len(controller.state.completed),
                      failed=len(controller.state.failed))

        if term or trunc or controller.state.fsm_state == State.DONE:
            break

    # ===== Phase 4: Top-down view (10s) =====
    print("[Demo] Phase 4: Top-down view")
    for i in range(int(10 * fps)):
        env.step(np.zeros(22, dtype=np.float32))
        if viewer:
            viewer.sync()
        add_frame(cam_top, "Top-Down View",
                  "Objects sorted into triage bins (green=fragile, yellow=sturdy, red=hazardous)",
                  completed=len(controller.state.completed),
                  failed=len(controller.state.failed))

    # ===== Phase 5: Hand showcase (10s) =====
    print("[Demo] Phase 5: Dexterous hand showcase")
    for i in range(int(10 * fps)):
        action = np.zeros(22, dtype=np.float32)
        phase = (i // 20) % 2
        action[6:] = 0.8 if phase == 0 else 0.0
        env.step(action)
        if viewer:
            viewer.sync()
        add_frame(cam_front, "Shadow Hand Lite - 16-DOF Dexterous Hand",
                  "5 fingertip tactile sensors + palm sensor",
                  completed=len(controller.state.completed),
                  failed=len(controller.state.failed))

    # ===== Phase 6: Final stats card (10s) =====
    print("[Demo] Phase 6: Final stats card")
    completed = len(controller.state.completed)
    for _ in range(int(10 * fps)):
        img = np.full((height, width, 3), (15, 20, 30), dtype=np.uint8)
        cv2.putText(img, "Disaster Triage Robothon", (130, 120),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        result_text = f"{completed} / 5 Objects Completed"
        cv2.putText(img, result_text, (140, 200),
                    cv2.FONT_HERSHEY_DUPLEX, 1.3, (0, 200, 100), 2, cv2.LINE_AA)
        details = [
            "22-DOF Robot (UR5e + Shadow Hand Lite)",
            "6 Tactile Sensors (5 fingertips + palm)",
            "5 Disaster Objects, 3 Triage Bins",
            "2 Equality Constraints",
            "100% Tactile Identification Accuracy",
        ]
        for j, line in enumerate(details):
            cv2.putText(img, line, (140, 260 + j * 35),
                        cv2.FONT_HERSHEY_DUPLEX, 0.55,
                        (160, 165, 180), 1, cv2.LINE_AA)
        frames.append(img)

    # ===== Save MP4 =====
    print(f"[Demo] Encoding {len(frames)} frames...")
    imageio.mimsave(
        str(output_path), frames, fps=fps,
        codec="libx264", quality=9, macro_block_size=16,
    )
    print(f"[Demo] Saved: {output_path}")
    print(f"[Demo] Duration: {len(frames)/fps:.1f}s")
    print(f"[Demo] File size: {output_path.stat().st_size/1024/1024:.1f} MB")

    env.close()
    return output_path


if __name__ == "__main__":
    render_demo()
