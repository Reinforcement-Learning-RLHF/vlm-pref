"""
VARP-style trajectory collection pipeline.

Based on: "VARP: Reinforcement Learning from Vision-Language Model Feedback
with Agent Regularized Preferences"

For each collected episode this script saves to --output_dir/<timestamp>/:
  final_image.png       - last rendered camera frame
  trajectory_vis.png    - final frame with temporally color-coded EEF path overlay
  demo.hdf5             - states, actions, eef_positions, and metadata

Usage:
  python collect_trajectories_varp.py --environment PickPlace --robots Panda
"""

import argparse
import datetime
import json
import os
import time
from copy import deepcopy
from glob import glob

import cv2
import h5py
import numpy as np

import robosuite as suite
from robosuite.controllers import load_composite_controller_config
from robosuite.controllers.composite.composite_controller import WholeBody
from robosuite.utils.camera_utils import get_camera_transform_matrix
from robosuite.wrappers import DataCollectionWrapper, VisualizationWrapper

from trajectory_vis import (
    add_colorbar_legend,
    capture_clean_frame,
    draw_trajectory_on_image,
    project_eef_to_image,
)


# ---------------------------------------------------------------------------
# Trajectory collection
# ---------------------------------------------------------------------------

def collect_trajectory(env, device, arm, max_fr, goal_update_mode, camera_name, img_height, img_width):
    """
    Run one episode and return collected data.

    Returns:
        dict with keys:
            eef_positions  - (T, 3) end-effector world positions
            final_image    - (H, W, 3) uint8 RGB last camera frame (or None)
            success        - bool
    """
    obs = env.reset()
    env.render()

    device.start_control()
    for robot in env.robots:
        robot.print_action_info_dict()

    all_prev_gripper_actions = [
        {
            f"{robot_arm}_gripper": np.repeat([0], robot.gripper[robot_arm].dof)
            for robot_arm in robot.arms
            if robot.gripper[robot_arm].dof > 0
        }
        for robot in env.robots
    ]

    eef_positions = []
    last_camera_image = None
    task_completion_hold_count = -1

    while True:
        start = time.time()

        active_robot = env.robots[device.active_robot]
        input_ac_dict = device.input2action(goal_update_mode=goal_update_mode)

        if input_ac_dict is None:
            break

        action_dict = deepcopy(input_ac_dict)
        for robot_arm in active_robot.arms:
            if isinstance(active_robot.composite_controller, WholeBody):
                controller_input_type = active_robot.composite_controller.joint_action_policy.input_type
            else:
                controller_input_type = active_robot.part_controllers[robot_arm].input_type

            if controller_input_type == "delta":
                action_dict[robot_arm] = input_ac_dict[f"{robot_arm}_delta"]
            elif controller_input_type == "absolute":
                action_dict[robot_arm] = input_ac_dict[f"{robot_arm}_abs"]
            else:
                raise ValueError(f"Unknown input type: {controller_input_type}")

        env_action = [robot.create_action_vector(all_prev_gripper_actions[i]) for i, robot in enumerate(env.robots)]
        env_action[device.active_robot] = active_robot.create_action_vector(action_dict)
        env_action = np.concatenate(env_action)

        for gripper_ac in all_prev_gripper_actions[device.active_robot]:
            all_prev_gripper_actions[device.active_robot][gripper_ac] = action_dict[gripper_ac]

        obs, _reward, _done, _info = env.step(env_action)
        env.render()

        # Record end-effector position
        if "robot0_eef_pos" in obs:
            eef_positions.append(obs["robot0_eef_pos"].copy())

        # Capture camera frame from observation
        cam_key = f"{camera_name}_image"
        if cam_key in obs:
            last_camera_image = obs[cam_key].copy()  # (H, W, 3) RGB uint8

        if task_completion_hold_count == 0:
            break

        if env._check_success():
            if task_completion_hold_count > 0:
                task_completion_hold_count -= 1
            else:
                task_completion_hold_count = 10
        else:
            task_completion_hold_count = -1

        if max_fr is not None:
            elapsed = time.time() - start
            diff = 1 / max_fr - elapsed
            if diff > 0:
                time.sleep(diff)

    success = (task_completion_hold_count == 0)

    # Capture camera transform and clean frame NOW, while sim is still live.
    camera_transform = get_camera_transform_matrix(env.sim, camera_name, img_height, img_width)
    clean_frame = capture_clean_frame(env, camera_name, img_height, img_width)

    env.close()

    return {
        "eef_positions": np.array(eef_positions) if eef_positions else np.empty((0, 3)),
        "final_image": clean_frame,          # clean frame, no visualization overlays
        "camera_transform": camera_transform, # 4x4, valid because captured before close
        "success": success,
    }


# ---------------------------------------------------------------------------
# HDF5 + visualisation saving
# ---------------------------------------------------------------------------

def save_episode(tmp_dir, out_dir, env_info, episode_data, img_height, img_width):
    """
    Gather raw demos from tmp_dir, build HDF5, and write visualisation images.

    Args:
        tmp_dir (str): DataCollectionWrapper scratch directory
        out_dir (str): destination folder for this episode
        env_info (str): JSON-encoded environment config
        episode_data (dict): output of collect_trajectory()
        img_height (int): image height
        img_width (int): image width
    """
    os.makedirs(out_dir, exist_ok=True)

    eef_positions = episode_data["eef_positions"]       # (T, 3)
    final_image_rgb = episode_data["final_image"]       # (H, W, 3) OpenGL RGB, or None
    camera_transform = episode_data["camera_transform"] # 4x4, captured before close

    # ---- Write visualisation images ----------------------------------------
    if final_image_rgb is not None:
        # Flip vertically: robosuite render() uses OpenGL convention (row 0 = bottom).
        final_image_bgr = cv2.flip(cv2.cvtColor(final_image_rgb, cv2.COLOR_RGB2BGR), 0)
        cv2.imwrite(os.path.join(out_dir, "final_image.png"), final_image_bgr)

        if len(eef_positions) > 0:
            pixels = project_eef_to_image(eef_positions, camera_transform, img_height, img_width)
            vis = draw_trajectory_on_image(final_image_bgr, pixels)
            vis = add_colorbar_legend(vis)
            cv2.imwrite(os.path.join(out_dir, "trajectory_vis.png"), vis)
            print(f"  Saved trajectory_vis.png  ({sum(p is not None for p in pixels)}/{len(pixels)} points projected)")
    else:
        print("  Warning: no camera image captured – skipping visualisation.")

    # ---- Write HDF5 ----------------------------------------------------------
    hdf5_path = os.path.join(out_dir, "demo.hdf5")
    f = h5py.File(hdf5_path, "w")
    grp = f.create_group("data")

    num_eps = 0
    env_name = None

    for ep_directory in os.listdir(tmp_dir):
        state_paths = os.path.join(tmp_dir, ep_directory, "state_*.npz")
        states, actions = [], []
        success = False

        for state_file in sorted(glob(state_paths)):
            dic = np.load(state_file, allow_pickle=True)
            env_name = str(dic["env"])
            states.extend(dic["states"])
            for ai in dic["action_infos"]:
                actions.append(ai["actions"])
            success = success or dic["successful"]

        if len(states) == 0:
            continue

        if success:
            print("  Demonstration is successful and has been saved.")
            del states[-1]
            assert len(states) == len(actions)

            num_eps += 1
            ep_grp = grp.create_group(f"demo_{num_eps}")

            xml_path = os.path.join(tmp_dir, ep_directory, "model.xml")
            with open(xml_path, "r") as xml_f:
                ep_grp.attrs["model_file"] = xml_f.read()

            ep_grp.create_dataset("states", data=np.array(states))
            ep_grp.create_dataset("actions", data=np.array(actions))

            if len(eef_positions) > 0:
                # Trim/pad eef_positions to match states length
                T = len(states)
                eef = eef_positions[:T]
                if len(eef) < T:
                    pad = np.tile(eef[-1], (T - len(eef), 1))
                    eef = np.vstack([eef, pad])
                ep_grp.create_dataset("eef_positions", data=eef)
        else:
            print("  Demonstration is unsuccessful and has NOT been saved.")

    now = datetime.datetime.now()
    grp.attrs["date"] = f"{now.month}-{now.day}-{now.year}"
    grp.attrs["time"] = f"{now.hour}:{now.minute}:{now.second}"
    grp.attrs["repository_version"] = suite.__version__
    grp.attrs["env"] = env_name or ""
    grp.attrs["env_info"] = env_info

    f.close()
    print(f"  Saved demo.hdf5  ({num_eps} successful episode(s))")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VARP trajectory collection pipeline")
    parser.add_argument("--output_dir", type=str, default="collected_trajectories",
                        help="Root directory where per-episode folders are saved")
    parser.add_argument("--environment", type=str, default="PickPlace")
    parser.add_argument("--robots", nargs="+", type=str, default="Panda")
    parser.add_argument("--config", type=str, default="default")
    parser.add_argument("--arm", type=str, default="right")
    parser.add_argument("--camera", type=str, default="agentview",
                        help="Camera name for rendering and trajectory projection")
    parser.add_argument("--img_height", type=int, default=512)
    parser.add_argument("--img_width", type=int, default=512)
    parser.add_argument("--controller", type=str, default=None)
    parser.add_argument("--device", type=str, default="keyboard")
    parser.add_argument("--pos_sensitivity", type=float, default=1.0)
    parser.add_argument("--rot_sensitivity", type=float, default=1.0)
    parser.add_argument("--renderer", type=str, default="mjviewer")
    parser.add_argument("--max_fr", type=int, default=20)
    parser.add_argument("--goal_update_mode", type=str, default="target",
                        choices=["target", "achieved"])
    parser.add_argument("--colormap", type=str, default="plasma",
                        help="Matplotlib colormap for temporal color coding (e.g. plasma, viridis, RdYlGn)")
    args = parser.parse_args()

    # Build controller config
    controller_config = load_composite_controller_config(
        controller=args.controller,
        robot=args.robots[0] if isinstance(args.robots, list) else args.robots,
    )

    if controller_config["type"] == "WHOLE_BODY_MINK_IK":
        from robosuite.examples.third_party_controller.mink_controller import WholeBodyMinkIK

    if controller_config["type"] == "WHOLE_BODY_IK":
        assert len(args.robots) == 1, "Whole Body IK only supports one robot"

    env_config = {
        "env_name": args.environment,
        "robots": args.robots,
        "controller_configs": controller_config,
    }
    if "TwoArm" in args.environment:
        env_config["env_configuration"] = args.config

    # Create environment – offscreen renderer enabled to capture camera obs
    env = suite.make(
        **env_config,
        has_renderer=True,
        renderer=args.renderer,
        has_offscreen_renderer=True,
        render_camera=args.camera,
        ignore_done=True,
        use_camera_obs=True,
        camera_names=args.camera,
        camera_heights=args.img_height,
        camera_widths=args.img_width,
        reward_shaping=True,
        control_freq=20,
    )

    env = VisualizationWrapper(env)
    env_info = json.dumps(env_config)

    tmp_directory = "/tmp/varp_{}".format(str(time.time()).replace(".", "_"))
    env = DataCollectionWrapper(env, tmp_directory)

    # Initialise input device
    device_name = args.device
    if device_name == "keyboard":
        from robosuite.devices import Keyboard
        device = Keyboard(env=env, pos_sensitivity=args.pos_sensitivity,
                          rot_sensitivity=args.rot_sensitivity)
    elif device_name == "spacemouse":
        from robosuite.devices import SpaceMouse
        device = SpaceMouse(env=env, pos_sensitivity=args.pos_sensitivity,
                            rot_sensitivity=args.rot_sensitivity)
    elif device_name == "dualsense":
        from robosuite.devices import DualSense
        device = DualSense(env=env, pos_sensitivity=args.pos_sensitivity,
                           rot_sensitivity=args.rot_sensitivity)
    elif device_name == "mjgui":
        assert args.renderer == "mjviewer"
        from robosuite.devices.mjgui import MJGUI
        device = MJGUI(env=env)
    else:
        raise ValueError(f"Unknown device: {device_name}")

    os.makedirs(args.output_dir, exist_ok=True)
    episode_num = 0

    print("\n=== VARP Trajectory Collector ===")
    print(f"Saving to: {os.path.abspath(args.output_dir)}")
    print(f"Camera   : {args.camera}  ({args.img_height}x{args.img_width})")
    print(f"Colormap : {args.colormap}")
    print("Press the reset key to end an episode and start a new one.\n")

    while True:
        episode_num += 1
        t1, t2 = str(time.time()).split(".")
        ep_dir = os.path.join(args.output_dir, f"ep_{episode_num:04d}_{t1}_{t2}")

        print(f"--- Episode {episode_num} ---")
        episode_data = collect_trajectory(
            env, device, args.arm, args.max_fr, args.goal_update_mode,
            args.camera, args.img_height, args.img_width,
        )

        save_episode(
            tmp_directory, ep_dir, env_info, episode_data,
            args.img_height, args.img_width,
        )
        print(f"  Episode saved → {ep_dir}\n")
