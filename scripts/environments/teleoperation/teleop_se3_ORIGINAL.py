# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#  sudo ../docker/container.py start     --files docker-compose.cloudxr-runtime.patch.yaml     --env-file .env.cloudxr-runtime
#  sudo ../docker/container.py enter base
# sudo docker cp /home/shubham/summer/usd_extracted isaac-lab-base:/workspace/usd/usd_extracted    (in local machine)

#  ./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py     --task Isaac-PickPlace-Franka-custom --robot franka     --teleop_device handtracking     --enable_pinocchio --enable_cameras 



"""Script to run a keyboard teleoperation with Isaac Lab manipulation environments."""

"""Launch Isaac Sim Simulator first."""
import sys
sys.path.append("/workspace/isaaclab/source/droid/droid/controllers/")
from droid.controllers.oculus_controller import VRPolicy
import argparse
from collections.abc import Callable

from isaaclab.app import AppLauncher
from scipy.spatial.transform import Rotation as R
from isaaclab.utils import math
# add argparse arguments
parser = argparse.ArgumentParser(description="Keyboard teleoperation for Isaac Lab environments.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument(
    "--teleop_device",
    type=str,
    default="keyboard",
    help="Device for interacting with environment. Examples: keyboard, spacemouse, gamepad, handtracking, manusvive",
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--robot", type=str, default="franka", choices=["franka", "gr1t2"], 
                    help="Robot type to use. Supported: franka, gr1t2")
parser.add_argument("--sensitivity", type=float, default=1.0, help="Sensitivity factor.")
parser.add_argument(
    "--enable_pinocchio",
    action="store_true",
    default=False,
    help="Enable Pinocchio.",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

app_launcher_args = vars(args_cli)

if args_cli.enable_pinocchio:
    # Import pinocchio before AppLauncher to force the use of the version installed by IsaacLab and
    # not the one installed by Isaac Sim pinocchio is required by the Pink IK controllers and the
    # GR1T2 retargeter
    import pinocchio  # noqa: F401
if "handtracking" in args_cli.teleop_device.lower():
    app_launcher_args["xr"] = True

# launch omniverse app
app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

"""Rest everything follows."""


import gymnasium as gym
import torch
import numpy as np
import cv2
import os
import time

import omni.log

from isaaclab.devices import Se3Gamepad, Se3GamepadCfg, Se3Keyboard, Se3KeyboardCfg, Se3SpaceMouse, Se3SpaceMouseCfg
from isaaclab.devices.openxr import remove_camera_configs
from isaaclab.devices.teleop_device_factory import create_teleop_device
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.markers import FRAME_MARKER_CFG, VisualizationMarkers

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.utils import parse_env_cfg
if args_cli.enable_pinocchio:
    import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401


def print_joint_data(env, actions, frame_count):
    """Print detailed joint information for debugging."""
    joint_names = env.scene["robot"].data.joint_names
    joint_positions = env.scene["robot"].data.joint_pos[0]  # First environment
    joint_velocities = env.scene["robot"].data.joint_vel[0]  # First environment
    
    print(f"=== Joint Data (Step {frame_count}) ===")
    print(f"Total joints: {len(joint_names)}")
    for i, (name, pos, vel) in enumerate(zip(joint_names, joint_positions, joint_velocities)):
        print(f"Joint {i:2d}: {name:20s} | Pos: {pos:8.4f} | Vel: {vel:8.4f}")
    # print(f"Action sent: {actions[0][:8].cpu().numpy()}")  # Show first 8 action values
    print("=" * 60)


class OculusTeleop:
    """Simple random teleoperation device for SE(3) commands."""
    
    def __init__(self, env, pos_sensitivity: float = 1, rot_sensitivity: float = 1, robot_type: str = "franka"):
        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity
        self.robot_type = robot_type
        self.env = env
        self._callbacks = {}
        self._step_count = 0
        self._hold_steps = 30  # Hold each random command for 30 steps for smooth movement
        self._current_command = None
        self.action_dim = 7
        self.controller = VRPolicy()

    
    def reset(self) -> None:
        """Reset the device."""
        self._step_count = 0
        self._current_command = None
        self.controller.reset_state()

    
    def add_callback(self, key: str, callback) -> None:
        """Add callback (no-op for random device)."""
        self._callbacks[key] = callback
    
    '''def advance(self) -> torch.Tensor:
        self._current_command = self._generate_franka_command()
        self._step_count += 1 
        return torch.tensor(self._current_command, dtype=torch.float32)'''
    def advance(self) -> torch.Tensor:
            # Check if VR controller movement is enabled
            if hasattr(self.controller, '_state') and self.controller._state.get('movement_enabled', False):
                self._current_command = self._generate_franka_command()
            else:
                # Return zero action when grip button is not pressed
                self._current_command = np.zeros(7)
                
            self._step_count += 1 
            return torch.tensor(self._current_command, dtype=torch.float32)

    def _generate_franka_command(self) -> np.ndarray:
        eef_idx = self.env.scene["robot"].data.body_names.index("panda_hand")
        eef_pos = self.env.scene["robot"].data.body_pos_w[0, eef_idx].cpu().detach().numpy()
        #eef_quat = self.env.scene["robot"].data.body_quat_w[0, eef_idx].cpu().detach().numpy()  # [x,y,z,w] from Isaac Lab
        eef_quat = self.env.scene["robot"].data.body_quat_w[0, eef_idx]  # [x,y,z,w] from Isaac Lab

        joint_names = self.env.scene["robot"].data.joint_names
        joint_positions = self.env.scene["robot"].data.joint_pos[0]        
        gripper_indices = [
                joint_names.index("panda_finger_joint1"),
                joint_names.index("panda_finger_joint2"),
            ]
        gripper_state = joint_positions[gripper_indices]

        # Convert quaternion to Euler angles (what the original VR controller expects)
        from scipy.spatial.transform import Rotation as R
        # eef_euler = R.from_quat(eef_quat).as_euler('xyz', degrees=False)  # Convert to [roll, pitch, yaw]
        eef_euler = torch.cat(math.euler_xyz_from_quat(eef_quat.unsqueeze(0))).cpu().detach().numpy()
        cartesian_position = np.concatenate([eef_pos, eef_euler])  # [x, y, z, roll, pitch, yaw] - 6 elements
        gripper_position = gripper_state.mean().cpu().detach().numpy()
        robot_state = {
            "cartesian_position": cartesian_position,  # [x, y, z, roll, pitch, yaw] - 6 elements total
            "gripper_position": gripper_position       # scalar gripper state
        }
        action, controller_action_info = self.controller.forward({"robot_state": robot_state}, include_info=True)
        
        return action
    '''def _generate_franka_command(self) -> np.ndarray:
        eef_idx = self.env.scene["robot"].data.body_names.index("panda_hand")
        eef_pos = self.env.scene["robot"].data.body_pos_w[0, eef_idx]    # shape: (3,)
        eef_quat = self.env.scene["robot"].data.body_quat_w[0, eef_idx]  # shape: (4,)
        joint_names = self.env.scene["robot"].data.joint_names
        joint_positions = self.env.scene["robot"].data.joint_pos[0]        
        gripper_indices = [
                joint_names.index("panda_finger_joint1"),
                joint_names.index("panda_finger_joint2"),
            ]
        gripper_state = joint_positions[gripper_indices]
        cartesian_position = torch.cat([eef_pos, eef_quat]).cpu().detach().numpy()
        gripper_position = gripper_state.mean().cpu().detach().numpy()
        robot_state = {
            "cartesian_position": cartesian_position,  # [x, y, z, qx, qy, qz, qw]
            "gripper_position": gripper_position       # scalar gripper state
        }
        action, controller_action_info = self.controller.forward({"robot_state": robot_state}, include_info=True)
        
        # coordinate transformation tuning
        #pos_action = action[:3]
        #rot_action = action[3:6] 
        #gripper_action = action[6]

        # apply scaling
        #pos_action *= self.pos_sensitivity
        #rot_action *= self.rot_sensitivity
        
        #return np.concatenate([pos_action, rot_action, [gripper_action]])
        return action'''
    
    def __str__(self) -> str:
        return f"Se3RandomTeleop(robot={self.robot_type}, pos_sensitivity={self.pos_sensitivity}, rot_sensitivity={self.rot_sensitivity})"


def main() -> None:
    """
    Run keyboard teleoperation with Isaac Lab manipulation environment.

    Creates the environment, sets up teleoperation interfaces and callbacks,
    and runs the main simulation loop until the application is closed.

    Returns:
        None
    """
    # Set default task based on robot type if not specified
    if args_cli.task is None:
        if args_cli.robot == "franka":
            args_cli.task = "Isaac-PickPlace-Franka-custom"  # Use the custom Franka pick-place env
        elif args_cli.robot == "gr1t2":
            args_cli.task = "Isaac-PickPlace-GR1T2-Abs-v0"
        else:
            raise ValueError(f"No default task for robot type: {args_cli.robot}")
    
    print(f"Using robot: {args_cli.robot}")
    print(f"Using task: {args_cli.task}")
    
    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.env_name = args_cli.task
    env_cfg.sim.gravity = [0.0, 0.0, -9.81]
    print("PhysicsScene added with gravity (0, 0, -9.81)")
    

    
    # modify configuration
    env_cfg.terminations.time_out = None
    
    if "Lift" in args_cli.task:
        # set the resampling time range to large number to avoid resampling
        env_cfg.commands.object_pose.resampling_time_range = (1.0e9, 1.0e9)
        # add termination condition for reaching the goal otherwise the environment won't reset
        env_cfg.terminations.object_reached_goal = DoneTerm(func=mdp.object_reached_goal)

    if args_cli.xr:
        # External cameras are not supported with XR teleop
        # Check for any camera configs and disable them
        env_cfg = remove_camera_configs(env_cfg)
        env_cfg.sim.render.antialiasing_mode = "DLSS"

    try:
        # create environment
        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
        # check environment name (for reach , we don't allow the gripper)
        if "Reach" in args_cli.task:
            omni.log.warn(
                f"The environment '{args_cli.task}' does not support gripper control. The device command will be"
                " ignored."
            )
    except Exception as e:
        omni.log.error(f"Failed to create environment: {e}")
        simulation_app.close()
        return

    # Flags for controlling teleoperation flow
    should_reset_recording_instance = False
    teleoperation_active = True

    # Callback handlers
    def reset_recording_instance() -> None:
        """
        Reset the environment to its initial state.

        Sets a flag to reset the environment on the next simulation step.

        Returns:
            None
        """
        nonlocal should_reset_recording_instance
        should_reset_recording_instance = True
        print("Reset triggered - Environment will reset on next step")

    def start_teleoperation() -> None:
        """
        Activate teleoperation control of the robot.

        Enables the application of teleoperation commands to the environment.

        Returns:
            None
        """
        nonlocal teleoperation_active
        teleoperation_active = True
        print("Teleoperation activated")

    def stop_teleoperation() -> None:
        """
        Deactivate teleoperation control of the robot.

        Disables the application of teleoperation commands to the environment.

        Returns:
            None
        """
        nonlocal teleoperation_active
        teleoperation_active = False
        print("Teleoperation deactivated")

    # Create device config if not already in env_cfg
    teleoperation_callbacks: dict[str, Callable[[], None]] = {
        "R": reset_recording_instance,
        "START": start_teleoperation,
        "STOP": stop_teleoperation,
        "RESET": reset_recording_instance,
    }

    # For hand tracking devices, add additional callbacks
    if args_cli.xr:
        # Default to inactive for hand tracking
        teleoperation_active = False
    else:
        # Enable teleoperation by default for random device to see movement
        if args_cli.teleop_device.lower() == "oculus":
            teleoperation_active = True
        else:
            # Disable teleoperation by default - let physics run
            teleoperation_active = False

    # Create teleop device from config if present, otherwise create manually
    teleop_interface = None
    try:
        if hasattr(env_cfg, "teleop_devices") and args_cli.teleop_device in env_cfg.teleop_devices.devices:
            teleop_interface = create_teleop_device(
                args_cli.teleop_device, env_cfg.teleop_devices.devices, teleoperation_callbacks
            )
        else:
            omni.log.warn(f"No teleop device '{args_cli.teleop_device}' found in environment config. Creating default.")
            # Create fallback teleop device
            sensitivity = args_cli.sensitivity
            if args_cli.teleop_device.lower() == "keyboard":
                teleop_interface = Se3Keyboard(
                    Se3KeyboardCfg(pos_sensitivity=0.05 * sensitivity, rot_sensitivity=0.05 * sensitivity)
                )
            elif args_cli.teleop_device.lower() == "spacemouse":
                teleop_interface = Se3SpaceMouse(
                    Se3SpaceMouseCfg(pos_sensitivity=0.05 * sensitivity, rot_sensitivity=0.05 * sensitivity)
                )
            elif args_cli.teleop_device.lower() == "gamepad":
                teleop_interface = Se3Gamepad(
                    Se3GamepadCfg(pos_sensitivity=0.1 * sensitivity, rot_sensitivity=0.1 * sensitivity)
                )
            elif args_cli.teleop_device.lower() == "oculus":
                teleop_interface = OculusTeleop(
                    env, 
                    pos_sensitivity=0.5 * sensitivity, 
                    rot_sensitivity=0.5 * sensitivity,
                    robot_type=args_cli.robot
                )
            else:
                omni.log.error(f"Unsupported teleop device: {args_cli.teleop_device}")
                omni.log.error("Supported devices: keyboard, spacemouse, gamepad, handtracking, random")
                env.close()
                simulation_app.close()
                return

            # Add callbacks to fallback device
            for key, callback in teleoperation_callbacks.items():
                try:
                    teleop_interface.add_callback(key, callback)
                except (ValueError, TypeError) as e:
                    omni.log.warn(f"Failed to add callback for key {key}: {e}")
    except Exception as e:
        omni.log.error(f"Failed to create teleop device: {e}")
        env.close()
        simulation_app.close()
        return

    if teleop_interface is None:
        omni.log.error("Failed to create teleop interface")
        env.close()
        simulation_app.close()
        return

    print(f"Using teleop device: {teleop_interface}")

    # reset environment
    env.reset()
    teleop_interface.reset()

    # Create pose marker for end effector visualization
    frame_marker_cfg = FRAME_MARKER_CFG.copy()  # type: ignore
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    pose_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/debug_transform"))

    # Set camera position
    env.sim.set_camera_view(eye=(0.0, 2.0, 2.0), target=(0.0, 0.0, 1.0))
    
    # Video recording setup
    video_frames = []
    start_time = time.time()
    frame_count = 0
    video_duration = 5.0  # Record for 5 seconds
    fps = 30  # Target fps for video
    
    # Create output directory for videos
    output_dir = "./"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Letting physics settle...")
    print("Teleoperation started. Press 'R' to reset the environment.")
    print(f"Recording video for {video_duration} seconds...")

    # --- Trajectory points storage ---
    traj_points = []

    # simulate environment
    while simulation_app.is_running():
        try:
            # run everything in inference mode
            with torch.inference_mode():
                # get device command

                action = teleop_interface.advance()
                #action[3]=0.
                #action[4]=0.
                #action[5]=0.
                # print("Action:", [f"{x:.2f}" for x in action])
                # Only apply teleop commands when active
                if teleoperation_active:
                    # process actions
                    actions = action.repeat(env.num_envs, 1)
                    # apply actions - now using end-effector control (6D pose + 1D gripper)
                    if args_cli.robot == "gr1t2":
                        env.step(actions[:36])
                    else: 
                        env.step(actions[:, :7])  # 6D pose + gripper action
                    
                    # Print joint information for all joints
                    # print_joint_data(env, actions, frame_count)
                else:
                    # Still need to step physics even when teleop is inactive
                    env.sim.step()
                    env.sim.render()
                # Capture camera images if camera exists in scene
                current_time = time.time()
                rgb_data = env.scene["tiled_camera"].data.output["rgb"]  #this one
                img = rgb_data[0].cpu().numpy()  # Shape: (H, W, 3)
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                # --- Overlay EE trajectory ---
                # Get camera info
                K = env.scene["tiled_camera"].data.intrinsic_matrices[0].cpu().detach().numpy()
                q = env.scene["tiled_camera"].data.quat_w_world[0].cpu().detach().numpy()
                t = env.scene["tiled_camera"].data.pos_w[0].cpu().detach().numpy()  #camera coordinate in world frame
                R_cam = R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()                                   #need to chk this wxyz or xyzw
                # Get current EE position based on robot type
                if args_cli.robot == "franka":
                    # For Franka, get the end-effector (panda_hand)
                    eef_idx = env.scene["robot"].data.body_names.index("panda_hand")
                    eef_pos = env.scene["robot"].data.body_pos_w[0, eef_idx]     #wrt what? 
                    eef_quat = env.scene["robot"].data.body_quat_w[0, eef_idx]
                # elif args_cli.robot == "gr1t2":
                #     # For GR1T2, get the left hand
                #     eef_idx = env.scene["robot"].data.body_names.index("left_hand_roll_link")
                #     eef_pos = env.scene["robot"].data.body_pos_w[0, eef_idx]
                #     eef_quat = env.scene["robot"].data.body_quat_w[0, eef_idx]
                
                # Visualize end effector pose using markers
                pose_marker.visualize(
                    translations=eef_pos.unsqueeze(0),  # Shape: (1, 3)
                    orientations=eef_quat.unsqueeze(0),  # Shape: (1, 4)
                )
                
                # # Store trajectory points in world frame for projection
                traj_points.append(eef_pos.cpu().detach().numpy())
                
                def project(pt):
                    # Transform point from world to camera frame
                    pt_cam = R_cam.T @ (pt - t)  #ee_pos to pixel coordinates
                    # Project to image plane
                    if pt_cam[2] > 0:  # Check if point is in front of camera
                        px = K @ pt_cam
                        u = int(px[0] / px[2])
                        v = int(px[1] / px[2])
                        # Check if projection is within image bounds
                        # print(u, v)
                        if 0 <= u < img_bgr.shape[1] and 0 <= v < img_bgr.shape[0]:
                            return u, v
                    return None
                
                # Project and draw trajectory points
                for pt in traj_points:
                    projection = project(pt)
                    if projection is not None:
                        u, v = projection
                        print(f"Projected point: ({u}, {v})")
                        cv2.circle(img_bgr, (u, v), 5, (0, 0, 255), -1)
                    # else:
                    #     print("Point outside camera view or behind camera")
                # --- End overlay ---
                if frame_count == 1:
                    cv2.imwrite("current_frame.png", img_bgr)  
                video_frames.append(img_bgr)
                frame_count += 1
                if frame_count % 30 == 0:  # Print every 30 frames (~1 second)
                    print(f"Captured {frame_count} frames...")

                # Save video after 5 seconds                  this will save video after 5 seconds and see if there's point(maybe a circle) on ee
                if current_time - start_time >= video_duration and len(video_frames) > 0:
                    print(f"Saving video with {len(video_frames)} frames...")
                    height, width = video_frames[0].shape[:2]
                    timestamp = int(time.time())
                    video_filename = os.path.join(output_dir, f"teleop_recording.mp4")
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))
                    for frame in video_frames:
                        video_writer.write(frame)
                    video_writer.release()
                    print(f"Video saved as: {video_filename}")
                    # Reset for next recording
                    video_frames = []
                    start_time = time.time()
                    frame_count = 0

                if should_reset_recording_instance:
                    env.reset()
                    should_reset_recording_instance = False
                    print("Environment reset complete")
        except Exception as e:
            omni.log.error(f"Error during simulation step: {e}")
            break

    # close the simulator
    env.close()
    print("Environment closed")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
