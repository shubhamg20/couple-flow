"""Script to replay recorded trajectories in Isaac Lab."""

import sys
sys.path.append("/workspace/isaaclab/source/droid/droid/controllers/")

import argparse
import pickle
import time
from pathlib import Path

from isaaclab.app import AppLauncher
from scipy.spatial.transform import Rotation as R

parser = argparse.ArgumentParser(description="Replay recorded trajectories in Isaac Lab.")
parser.add_argument("--trajectory_file", type=str, required=True, help="Trajectory pkl filename")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--robot", type=str, default="franka", choices=["franka", "gr1t2"], help="Robot type")
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

app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import numpy as np
import omni.log

from isaaclab.markers import FRAME_MARKER_CFG, VisualizationMarkers
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
if args_cli.enable_pinocchio:
    import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401


def load_trajectory(filepath):
    """Load trajectory data from pkl file.
    
    Supports both old format (multiple episodes) and new format (single episode).
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Trajectory file not found: {filepath}")
    
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    # New format: single episode file
    if "episode" in data and "trajectory" in data:
        episode_num = data["episode"]
        trajectory = data["trajectory"]
        initial_objects = data.get("initial_objects", {})
        
        return {episode_num: trajectory}, {episode_num: initial_objects}, {}
    
    # Old format: multiple episodes in one file
    else:
        return data.get("episodes", {}), data.get("episode_initial_objects", {}), data.get("metadata", {})


def main():
    """Replay recorded trajectories."""
    
    # Set default task
    if args_cli.task is None:
        if args_cli.robot == "franka":
            args_cli.task = "Isaac-PickPlace-Franka-custom"
        elif args_cli.robot == "gr1t2":
            args_cli.task = "Isaac-PickPlace-GR1T2-Abs-v0"
        else:
            raise ValueError(f"No default task for robot type: {args_cli.robot}")
    
    # Load trajectory
    trajectory_file = Path(args_cli.trajectory_file)
    
    if not trajectory_file.exists():
        omni.log.error(f"Trajectory file not found: {trajectory_file}")
        simulation_app.close()
        return
    
    try:
        trajectory_data, episode_initial_objects, metadata = load_trajectory(trajectory_file)
    except Exception as e:
        omni.log.error(f"Failed to load trajectory: {e}")
        simulation_app.close()
        return
    
    print(f"\n{'='*60}")
    print(f"TRAJECTORY REPLAY")
    print(f"File: {trajectory_file.name}")
    print(f"Total Episodes: {len(trajectory_data)}")
    if metadata:
        print(f"Robot: {metadata.get('robot', 'unknown')}")
        print(f"Task: {metadata.get('task', 'unknown')}")
    print(f"{'='*60}\n")
    
    # Create environment
    try:
        env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
        env.reset()
    except Exception as e:
        omni.log.error(f"Failed to create environment: {e}")
        simulation_app.close()
        return
    
    # Create pose marker for EEF visualization
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    pose_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/replay_eef"))
    
    # Set camera
    env.sim.set_camera_view(eye=(0.6, -0.3, 1.5), target=(-1.3, 2.3, 0.0))
    
    # Main replay loop
    while simulation_app.is_running():
        try:
            with torch.inference_mode():
                for episode_num in sorted(trajectory_data.keys()):
                    episode_trajectory = trajectory_data[episode_num]
                    
                    print(f"\n{'='*60}")
                    print(f"[REPLAY] Playing Episode {episode_num} ({len(episode_trajectory)} timesteps)")
                    print(f"{'='*60}\n")
                    
                    # Reset environment
                    env.reset()
                    env_ids = torch.arange(env.num_envs, device=env.device)
                    
                    # Set recorded initial object positions
                    if episode_num in episode_initial_objects:
                        for obj_name, obj_data in episode_initial_objects[episode_num].items():
                            try:
                                obj = env.scene[obj_name]
                                obj_root_state = obj.data.root_state_w.clone()
                                obj_root_state[:, :3] = torch.tensor(obj_data["pos"], dtype=torch.float32, device=env.device)
                                obj.write_root_state_to_sim(obj_root_state, env_ids=env_ids)
                            except KeyError:
                                pass
                    
                    env.sim.step(render=False)
                    
                    for step_idx, step_data in enumerate(episode_trajectory):
                        if not simulation_app.is_running():
                            break
                        
                        # Apply recorded action to robot
                        if "action" in step_data:
                            action = torch.tensor(step_data["action"], dtype=torch.float32, device=env.device)
                            actions = action.repeat(env.num_envs, 1)
                            if args_cli.robot == "gr1t2":
                                env.step(actions[:36])
                            else:
                                env.step(actions[:, :7])
                        
                        # Update object states during replay
                        if "objects" in step_data:
                            for obj_name, obj_data in step_data["objects"].items():
                                try:
                                    obj = env.scene[obj_name]
                                    obj_root_state = obj.data.root_state_w.clone()
                                    obj_root_state[:, :3] = torch.tensor(obj_data["pos"], dtype=torch.float32, device=env.device)
                                    if "quat" in obj_data:
                                        obj_root_state[:, 3:7] = torch.tensor(obj_data["quat"], dtype=torch.float32, device=env.device)
                                    obj.write_root_state_to_sim(obj_root_state, env_ids=env_ids)
                                except KeyError:
                                    pass
                        
                        # Extract and visualize EEF data
                        if "franka_eef" in step_data:
                            eef_data = step_data["franka_eef"]
                            eef_pos = torch.tensor(eef_data["pos"], dtype=torch.float32, device=env.device)
                            eef_rpy = torch.tensor(eef_data["rpy"], dtype=torch.float32, device=env.device)
                            eef_rot = R.from_euler('xyz', eef_rpy.cpu().numpy())
                            eef_quat = torch.tensor(eef_rot.as_quat(), dtype=torch.float32, device=env.device)  # [x,y,z,w]
                            
                            # Visualize EEF
                            pose_marker.visualize(
                                translations=eef_pos.unsqueeze(0),
                                orientations=torch.tensor([eef_quat[3], eef_quat[0], eef_quat[1], eef_quat[2]], 
                                                        device=env.device).unsqueeze(0),  # Convert to [w,x,y,z]
                            )
                            
                            # Print progress every 30 steps
                            if step_idx % 30 == 0:
                                print(f"[REPLAY] Episode {episode_num} | Step: {step_idx}/{len(episode_trajectory)} | " +
                                      f"EEF: [{eef_pos[0]:.3f}, {eef_pos[1]:.3f}, {eef_pos[2]:.3f}]")
                        
                        time.sleep(0.01)
                    
                    if not simulation_app.is_running():
                        break
                    
                    print(f"[REPLAY] Episode {episode_num} completed\n")
                
                print(f"\n[REPLAY] Full trajectory replay completed!\n")
                break
        
        except Exception as e:
            omni.log.error(f"Error during replay: {e}")
            import traceback
            traceback.print_exc()
            break
    
    env.close()
    print("Replay finished")


if __name__ == "__main__":
    main()
    simulation_app.close()