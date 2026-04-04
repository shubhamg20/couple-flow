import threading
import queue
import sys
sys.path.append("/workspace/isaaclab/source/droid/droid/controllers/")
from droid.controllers.oculus_controller import VRPolicy
import argparse
from collections.abc import Callable
from pathlib import Path
import pickle

from isaaclab.app import AppLauncher
from scipy.spatial.transform import Rotation as R
from isaaclab.utils import math

parser = argparse.ArgumentParser(description="Teleoperation for Isaac Lab environments with per-episode recording.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument(
    "--teleop_device",
    type=str,
    default="keyboard",
    help="Device for interacting with environment. Examples: keyboard, spacemouse, gamepad, handtracking, oculus",
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--robot", type=str, default="franka", choices=["franka", "gr1t2"], 
                    help="Robot type to use. Supported: franka, gr1t2")
parser.add_argument("--sensitivity", type=float, default=1.0, help="Sensitivity factor.")
parser.add_argument("--record_trajectory", action="store_true", help="Enable trajectory recording.")
parser.add_argument("--task_name", type=str, default="task1", help="Task name for organizing recordings.")
parser.add_argument("--init_from_demo", type=str, default=None, help="Task name to load demo object positions from")
parser.add_argument(
    "--enable_pinocchio",
    action="store_true",
    default=False,
    help="Enable Pinocchio.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher_args = vars(args_cli)

if args_cli.enable_pinocchio:
    import pinocchio  # noqa: F401
if "handtracking" in args_cli.teleop_device.lower():
    app_launcher_args["xr"] = True

app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import numpy as np
import os
import time
import cv2
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


class OculusTeleop:
    """Oculus teleoperation device for SE(3) commands."""
    
    def __init__(self, env, pos_sensitivity: float = 1, rot_sensitivity: float = 1, robot_type: str = "franka"):
        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity
        self.robot_type = robot_type
        self.env = env
        self._callbacks = {}
        self._step_count = 0
        self._current_command = None
        self.action_dim = 7
        self.controller = VRPolicy()

    def reset(self) -> None:
        """Reset the device."""
        self._step_count = 0
        self._current_command = None
        self.controller.reset_state()

    def add_callback(self, key: str, callback) -> None:
        """Add callback."""
        self._callbacks[key] = callback
    
    def advance(self) -> torch.Tensor:
        if hasattr(self.controller, '_state') and self.controller._state.get('movement_enabled', False):
            self._current_command = self._generate_franka_command()
            self._current_command[5] *= 0.01
            self._current_command[3] *= 0.01


        else:
            self._current_command = np.zeros(7)
        self._step_count += 1 
        return torch.tensor(self._current_command, dtype=torch.float32)

    def _generate_franka_command(self) -> np.ndarray:
        eef_idx = self.env.scene["robot"].data.body_names.index("panda_hand")
        eef_pos = self.env.scene["robot"].data.body_pos_w[0, eef_idx].cpu().detach().numpy()
        eef_quat = self.env.scene["robot"].data.body_quat_w[0, eef_idx] 

        joint_names = self.env.scene["robot"].data.joint_names
        joint_positions = self.env.scene["robot"].data.joint_pos[0]        
        gripper_indices = [
                joint_names.index("panda_finger_joint1"),
                joint_names.index("panda_finger_joint2"),
            ]
        gripper_state = joint_positions[gripper_indices]
        robot_base_idx = self.env.scene["robot"].data.body_names.index("panda_link0")
        robot_base_quat = self.env.scene["robot"].data.body_quat_w[0, robot_base_idx][[1, 2, 3, 0]].cpu().detach().numpy()
        robot_base_rot = R.from_quat(robot_base_quat)
        robot_base_rot_inv = robot_base_rot.inv()

        eef_quat = eef_quat[[1, 2, 3, 0]].cpu().detach().numpy()
        eef_rot = R.from_quat(eef_quat)
        eef_rot_local = robot_base_rot_inv * eef_rot
        eef_euler = eef_rot_local.as_euler('xyz')

        robot_base_pos = self.env.scene["robot"].data.body_pos_w[0, robot_base_idx].cpu().detach().numpy()
        eef_pos_local = robot_base_rot_inv.apply(eef_pos - robot_base_pos)
        
        cartesian_position = np.concatenate([eef_pos_local, eef_euler])
        gripper_position = gripper_state.mean().cpu().detach().numpy()
        robot_state = {
            "cartesian_position": cartesian_position,
            "gripper_position": gripper_position
        }
        action, controller_action_info = self.controller.forward({"robot_state": robot_state}, include_info=True)
        return action

    def __str__(self) -> str:
        return f"OculusTeleop(robot={self.robot_type})"


def load_demo_objects(task_name, episode_num, max_episode):
    """Load demo object positions from pkl file."""
    if episode_num > max_episode:
        print(f"[DEMO] Episode {episode_num} exceeds demos (max: {max_episode}), using defaults")
        return None
    
    pkl_file = Path(f"source/recorded_runs/{task_name}/episode{episode_num}.pkl")
    if not pkl_file.exists():
        return None
    
    try:
        with open(pkl_file, 'rb') as f:
            demo_objects = pickle.load(f)['initial_objects']
        
        print(f"[DEMO] Loaded from PKL - Episode {episode_num}:")
        
        return demo_objects
    except Exception as e:
        print(f"[DEMO] Failed to load episode {episode_num}: {e}")
        return None


def apply_demo_objects(env, demo_objects, env_ids):
    """Apply demo object positions to scene."""
    if demo_objects is None:
        return
    
    for obj_name in ["bowl", "mug", "ball"]:
        if obj_name in env.scene.keys() and obj_name in demo_objects:
            asset = env.scene[obj_name]
            pos = torch.tensor(demo_objects[obj_name]["pos"], device=env.device).unsqueeze(0)
            quat = torch.tensor(demo_objects[obj_name]["quat"], device=env.device).unsqueeze(0)
            root_pose = torch.cat([pos, quat], dim=-1)
            velocities = torch.zeros((1, 6), device=env.device)
            asset.write_root_pose_to_sim(root_pose, env_ids=env_ids)
            asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)


def get_max_demo_episode(task_name):
    """Get highest demo episode number available."""
    task_dir = Path(f"source/recorded_runs/{task_name}")
    if not task_dir.exists():
        return -1
    
    episodes = [int(f.stem.replace("episode", "")) for f in task_dir.glob("episode*.pkl") 
                if f.stem.replace("episode", "").isdigit()]
    return max(episodes) if episodes else -1


def full_environment_reset(env, args_cli, current_joint_pos, demo_objects=None):
    """Performs a complete reset of the environment, robot, and objects."""
    env_ids = torch.arange(env.num_envs, device=env.device)
    
    # if args_cli.robot == "franka":
    #     root_state = env.scene["robot"].data.default_root_state.clone()
    #     env.scene["robot"].write_root_state_to_sim(root_state, env_ids=env_ids)
    #     default_joint_pos = env.scene["robot"].data.default_joint_pos.clone()
    #     env.scene["robot"].set_joint_position_target(default_joint_pos)
    #     current_joint_pos = default_joint_pos
    #     env.sim.step(render=False)
        
    # elif args_cli.robot == "gr1t2":
    #     root_state = env.scene["robot"].data.default_root_state.clone()
    #     env.scene["robot"].write_root_state_to_sim(root_state, env_ids=env_ids)
    #     default_joint_pos = env.scene["robot"].data.default_joint_pos.clone()
    #     env.scene["robot"].set_joint_position_target(default_joint_pos)
    #     current_joint_pos = default_joint_pos
    #     env.sim.step(render=False)
    
    if demo_objects is not None:
        apply_demo_objects(env, demo_objects, env_ids)
        env.sim.step(render=False)
    env.reset()
    ####################################################################zzzzz
    
    
    return current_joint_pos


def get_next_episode_number(task_name):
    """Find the next episode number by checking existing episodes in task folder."""
    task_dir = Path(f"source/recorded_runs/{task_name}")  # Added 'source/' prefix
    if not task_dir.exists():
        return 0
    
    episode_files = list(task_dir.glob("episode*.pkl"))
    if not episode_files:
        return 0
    
    episode_numbers = []
    for f in episode_files:
        try:
            num = int(f.stem.replace("episode", ""))
            episode_numbers.append(num)
        except ValueError:
            pass
    
    if not episode_numbers:
        return 0
    
    return max(episode_numbers) + 1

def get_image_async(env):
    """Return the tensor without CPU transfer - let the background thread handle it."""
    rgb_data = env.scene["tiled_camera"].data.output["rgb"]
    # Clone to avoid it being overwritten, but keep on GPU
    return rgb_data[0].clone()

def extract_trajectory_data(env, args_cli, timestep, action):
    """Extract current state for trajectory recording."""
    eef_idx = env.scene["robot"].data.body_names.index("panda_hand")
    eef_pos_w = env.scene["robot"].data.body_pos_w[0, eef_idx].cpu().detach().numpy()
    eef_quat_w = env.scene["robot"].data.body_quat_w[0, eef_idx][[1, 2, 3, 0]].cpu().detach().numpy()
    eef_rot = R.from_quat(eef_quat_w)
    eef_rpy = eef_rot.as_euler('xyz')

    # Store GPU tensor directly - conversion happens in background thread
    img_tensor = get_image_async(env)

    joint_names = env.scene["robot"].data.joint_names
    joint_positions = env.scene["robot"].data.joint_pos[0]
    gripper_indices = [joint_names.index("panda_finger_joint1"), joint_names.index("panda_finger_joint2")]
    gripper_state = joint_positions[gripper_indices].mean().cpu().detach().numpy()
    # Get camera parameters for projection
    K = env.scene["tiled_camera"].data.intrinsic_matrices[0].cpu().detach().numpy()
    q = env.scene["tiled_camera"].data.quat_w_world[0].cpu().detach().numpy()
    t = env.scene["tiled_camera"].data.pos_w[0].cpu().detach().numpy()
    R_cam = R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    # print(action[-1])
    step_data = {
        "timestep": timestep,
        "action": action.cpu().numpy().tolist() if isinstance(action, torch.Tensor) else action.tolist(),
        "franka_eef": {
            "pos": eef_pos_w.tolist(),
            "rpy": eef_rpy.tolist(),
            "gripper": float(gripper_state)
        },
        "image_tensor": img_tensor,
        "camera": {
            "intrinsics": K.tolist(),
            "rotation": R_cam.tolist(),
            "translation": t.tolist()
        },
        "objects": {}
    }
    
    # Try to capture object positions
    object_names = ["bowl", "mug"]
    for obj_name in object_names:
        if obj_name in env.scene.keys():
            obj_pos = env.scene[obj_name].data.body_pos_w[0][0].cpu().detach().numpy()
            obj_rot = env.scene[obj_name].data.body_quat_w[0][0].cpu().detach().numpy()
            step_data["objects"][obj_name] = {"pos": obj_pos.tolist(), "quat": obj_rot.tolist()}
    return step_data

def save_episode(episode_num, episode_trajectory, episode_initial_objects, task_name):
    """Save single episode to pkl file."""
    task_dir = Path(f"source/recorded_runs/{task_name}")
    task_dir.mkdir(parents=True, exist_ok=True)
    
    episode_data = {
        "episode": episode_num,
        "trajectory": episode_trajectory,
        "initial_objects": episode_initial_objects
    }
    
    episode_file = task_dir / f"episode{episode_num}.pkl"
    with open(episode_file, 'wb') as f:
        pickle.dump(episode_data, f)
    
    print(f"[RECORDING] Episode {episode_num} COMPLETED and SAVED to {episode_file} ({len(episode_trajectory)} timesteps)")

def save_worker(save_queue):
    """Background thread worker that saves episodes asynchronously."""
    while True:
        item = save_queue.get()
        if item is None:  # Poison pill to stop the thread
            save_queue.task_done()
            break

        episode_num, trajectory, initial_objects, task_name = item
        try:
            # Convert images in the background thread
            print(f"[RECORDING] Processing episode {episode_num} in background ({len(trajectory)} timesteps)...")

            ##############################################################################
            ##############################################################################
            ##  DEBUG: Collect BGR frames for video, then save .mp4 next to .pkl
            ##############################################################################
            ##############################################################################
            debug_frames = []

            for step_data in trajectory:
                if 'image_tensor' in step_data:
                    img_tensor = step_data['image_tensor']
                    # Move to CPU and convert to numpy
                    img = img_tensor.cpu().numpy()
                    # Convert RGB to BGR for OpenCV
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                    debug_frames.append(img_bgr)

                    # Compress to JPEG bytes (saves 80-95% space)
                    _, img_encoded = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    step_data['image'] = img_encoded.tobytes()

                    # Remove tensor to save memory
                    del step_data['image_tensor']

            # Save the processed episode
            save_episode(episode_num, trajectory, initial_objects, task_name)

            # Save debug video next to pkl
            if debug_frames:
                task_dir = Path(f"source/recorded_runs/{task_name}")
                video_file = task_dir / f"episode{episode_num}.mp4"
                h, w = debug_frames[0].shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(str(video_file), fourcc, 20.0, (w, h))
                for frame in debug_frames:
                    writer.write(frame)
                writer.release()
                print(f"[DEBUG] Saved video: {video_file} ({len(debug_frames)} frames)")
            del debug_frames
            ##############################################################################
            ##############################################################################

        except Exception as e:
            print(f"[ERROR] Failed to save episode {episode_num}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            save_queue.task_done()

def main() -> None:
    """Run teleoperation with per-episode trajectory recording."""
    
    recording_enabled = args_cli.record_trajectory
    current_episode = 0
    
    # Initialize save queue and worker thread
    save_queue = None
    save_thread = None
    
    if recording_enabled:
        current_episode = get_next_episode_number(args_cli.task_name)
        print("\n" + "="*60)
        print("TRAJECTORY RECORDING ENABLED")
        print(f"Task: {args_cli.task_name}")
        print(f"Starting Episode: {current_episode}")
        print("="*60 + "\n")
        
        # Start the save worker thread
        save_queue = queue.Queue()
        save_thread = threading.Thread(target=save_worker, args=(save_queue,), daemon=True)
        save_thread.start()
        print("[RECORDING] Background save thread started")
    
    # Demo mode setup
    demo_enabled = args_cli.init_from_demo is not None
    demo_task = args_cli.init_from_demo
    max_demo = get_max_demo_episode(demo_task) if demo_enabled else -1
    current_demo = load_demo_objects(demo_task, current_episode, max_demo) if demo_enabled else None
    
    if demo_enabled:
        print(f"[DEMO] Loading from '{demo_task}' - {max_demo + 1 if max_demo >= 0 else 0} episodes available")
    
    if args_cli.task is None:
        if args_cli.robot == "franka":
            args_cli.task = "Isaac-Kitchen-Franka-v0"
        elif args_cli.robot == "gr1t2":
            args_cli.task = "Isaac-PickPlace-GR1T2-Abs-v0"
        else:
            raise ValueError(f"No default task for robot type: {args_cli.robot}")
    
    print(f"Using robot: {args_cli.robot}")
    print(f"Using task: {args_cli.task}")
    
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.env_name = args_cli.task
    env_cfg.sim.gravity = [0.0, 0.0, -9.81]
    print("PhysicsScene added with gravity (0, 0, -9.81)")
    
    env_cfg.terminations.time_out = None
    
    if "Lift" in args_cli.task:
        env_cfg.commands.object_pose.resampling_time_range = (1.0e9, 1.0e9)
        env_cfg.terminations.object_reached_goal = DoneTerm(func=mdp.object_reached_goal)

    if args_cli.xr:
        env_cfg = remove_camera_configs(env_cfg)
        env_cfg.sim.render.antialiasing_mode = "DLSS"

    try:
        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
        if "Reach" in args_cli.task:
            omni.log.warn(
                f"The environment '{args_cli.task}' does not support gripper control. The device command will be"
                " ignored."
            )
    except Exception as e:
        omni.log.error(f"Failed to create environment: {e}")
        simulation_app.close()
        return

    should_reset_recording_instance = False
    teleoperation_active = True

    def reset_recording_instance() -> None:
        nonlocal should_reset_recording_instance
        should_reset_recording_instance = True
        print("Reset triggered - Environment will reset on next step")

    def start_teleoperation() -> None:
        nonlocal teleoperation_active
        teleoperation_active = True
        print("Teleoperation activated")

    def stop_teleoperation() -> None:
        nonlocal teleoperation_active
        teleoperation_active = False
        print("Teleoperation deactivated")

    teleoperation_callbacks: dict[str, Callable[[], None]] = {
        "R": reset_recording_instance,
        "START": start_teleoperation,
        "STOP": stop_teleoperation,
        "RESET": reset_recording_instance,
    }

    # if args_cli.xr:
    #     teleoperation_active = True
    # else:
    #     if args_cli.teleop_device.lower() == "oculus":
    #         teleoperation_active = True
    #     else:
    #         teleoperation_active = False

    teleop_interface = None
    try:
        if hasattr(env_cfg, "teleop_devices") and args_cli.teleop_device in env_cfg.teleop_devices.devices:
            teleop_interface = create_teleop_device(
                args_cli.teleop_device, env_cfg.teleop_devices.devices, teleoperation_callbacks
            )
        else:
            omni.log.warn(f"No teleop device '{args_cli.teleop_device}' found in environment config. Creating default.")
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
                    pos_sensitivity=1.0 * sensitivity, 
                    rot_sensitivity=1.0 * sensitivity,
                    robot_type=args_cli.robot
                )
            else:
                omni.log.error(f"Unsupported teleop device: {args_cli.teleop_device}")
                env.close()
                simulation_app.close()
                return

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
    env.reset()

    if args_cli.robot == "franka":
        print("Initialising Franka joint position targets to current positions")
        current_joint_pos = env.scene["robot"].data.joint_pos.clone()
        env_ids = torch.arange(env.num_envs, device=env.device)
        root_state = env.scene["robot"].data.default_root_state.clone()
        env.scene["robot"].write_root_state_to_sim(root_state, env_ids=env_ids)
        env.scene["robot"].set_joint_position_target(current_joint_pos)
        env.sim.step(render=False)
        

        

        if demo_enabled and current_demo is not None:
            apply_demo_objects(env, current_demo, env_ids)
            env.sim.step(render=False)

    teleop_interface.reset()

    if args_cli.robot == "franka":
        env.sim.set_camera_view(eye=(0.3, 0.0, 1.3), target=(-0.1, 1.0, 0.8))
    else:
        env.sim.set_camera_view(eye=(0.0, 2.0, 2.0), target=(0.0, 0.0, 1.0))
    
    # Print initial object positions
    if demo_enabled:
        print("\n[INITIAL] Scene object positions after first reset:")
        for obj_name in ["bowl", "mug", "ball"]:
            if obj_name in env.scene.keys():
                obj_pos = env.scene[obj_name].data.body_pos_w[0][0].cpu().detach().numpy()
                print(f"  {obj_name}: pos={obj_pos.tolist()}")
        print()
    
    print("Teleoperation started.")
    if recording_enabled:
        print(f"[RECORDING] Starting Episode {current_episode}")

    # print("Islfhp98Y9Po;o;o;;ouo;o;jlsdfl;shdfowh;dofhs;fouwh;owefwhfjhw;")
    # ##############################################################################################
    # for i in range(20):
    #     env.reset()
    #     env.sim.step(render=True)
    #     time.sleep(1)

    timestep = 0
    current_trajectory = []
    episode_initial_objects = {}
    
    # Add 20Hz recording control
    recording_freq = 20.0  # Hz
    recording_interval = 1.0 / recording_freq  # 0.05 seconds
    last_record_time = time.time()

    # Main simulation loop
    while simulation_app.is_running():
        try:
            with torch.inference_mode():
                action = teleop_interface.advance()
                # A button: save episode and reset

                if args_cli.teleop_device.lower() == "oculus":
                    if hasattr(teleop_interface.controller, "_state"):
                        teleoperation_active = teleop_interface.controller._state.get("movement_enabled", False)
                if teleop_interface.controller.get_info()["success"] and timestep > 10:
                    if recording_enabled:
                        trajectory_copy = [step.copy() for step in current_trajectory]
                        objects_copy = {k: v.copy() for k, v in episode_initial_objects.items()}
                        save_queue.put((current_episode, trajectory_copy, objects_copy, args_cli.task_name))
                        print(f"[RECORDING] Episode {current_episode} queued for saving ({len(current_trajectory)} timesteps)")
                        current_episode += 1
                        current_trajectory = []
                        timestep = 0
                        episode_initial_objects = {}
                        print(f"[RECORDING] Starting Episode {current_episode}")
                    
                    if demo_enabled:
                        current_demo = load_demo_objects(demo_task, current_episode, max_demo)
                    
                    current_joint_pos = full_environment_reset(env, args_cli, current_joint_pos, current_demo)
                    teleop_interface.reset()
                    continue
                
                # B button: discard episode and restart
                if teleop_interface.controller.get_info()["failure"]:
                    if recording_enabled:
                        print(f"[RECORDING] Episode {current_episode} DISCARDED")
                        current_trajectory = []
                        timestep = 0
                        episode_initial_objects = {}
                        teleoperation_active = False
                    
                    if demo_enabled:
                        current_demo = load_demo_objects(demo_task, current_episode, max_demo)
                    
                    current_joint_pos = full_environment_reset(env, args_cli, current_joint_pos, current_demo)
                    teleop_interface.reset()
                    continue
                
                # Print current object positions if demo mode enabled
                # if demo_enabled and timestep % 10 == 0:  # Print every 10 timesteps to avoid spam
                #     print(f"[TIMESTEP {timestep}] Current scene object positions:")
                #     for obj_name in ["sushi", "mug", "apple"]:
                #         if obj_name in env.scene.keys():
                #             obj_pos = env.scene[obj_name].data.body_pos_w[0][0].cpu().detach().numpy()
                #             print(f"  {obj_name}: pos={obj_pos.tolist()}")
                
                # Record trajectory if enabled
                current_time = time.time()
                if teleoperation_active and recording_enabled and (current_time - last_record_time >= recording_interval or timestep == 0):
                    step_data = extract_trajectory_data(env, args_cli, timestep, action)
                    current_trajectory.append(step_data)
                    last_record_time = current_time
                    if timestep == 0:
                        print(step_data["franka_eef"])

                    # Capture initial object positions on first timestep of episode
                    if timestep == 0:
                        object_names = ["bowl", "mug", "ball"]
                        for obj_name in object_names:
                            if obj_name in env.scene.keys():
                                obj_pos = env.scene[obj_name].data.body_pos_w[0][0].cpu().detach().numpy()
                                obj_quat = env.scene[obj_name].data.body_quat_w[0][0].cpu().detach().numpy()
                                episode_initial_objects[obj_name] = {"pos": obj_pos.tolist(), "quat": obj_quat.tolist()}
                    
                    timestep += 1
                if teleoperation_active:
                    actions = action.repeat(env.num_envs, 1)
                    if args_cli.robot == "gr1t2":
                        env.step(actions[:36])
                    else: 
                        env.step(actions[:, :7])
                else:
                    zero_action = torch.zeros((env.num_envs, action.shape[0]), device=env.device)
                    if args_cli.robot == "gr1t2":   
                        env.step(zero_action[:,:36])
                    else:
                        env.step(zero_action[:, :7])
                        # print(f"Action taken: {actions[0, 6].cpu().numpy().tolist()}")
                
                if should_reset_recording_instance:
                    if demo_enabled:
                        current_demo = load_demo_objects(demo_task, current_episode, max_demo)
                    current_joint_pos = full_environment_reset(env, args_cli, current_joint_pos, current_demo)
                    should_reset_recording_instance = False
                    print("Environment reset complete")

                

        except Exception as e:
            omni.log.error(f"Error during simulation step: {e}")
            import traceback
            traceback.print_exc()
            break

    # Cleanup: wait for all saves to complete
    if recording_enabled and save_queue is not None:
        print("\n[RECORDING] Waiting for pending saves to complete...")
        save_queue.join()  # Wait for all queued saves to finish
        print("[RECORDING] All queued episodes processed")
        save_queue.put(None)  # Send poison pill to stop thread
        if save_thread.is_alive():
            save_thread.join(timeout=10.0)  # Wait for thread to finish
        print("[RECORDING] Background thread stopped")
    
    env.close()
    print("Environment closed")


if __name__ == "__main__":
    main()
    simulation_app.close()