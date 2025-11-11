"""Script to replay recorded trajectories in Isaac Lab and save videos."""

import sys
sys.path.append("/workspace/isaaclab/source/droid/droid/controllers/")

import argparse
import pickle
import time
from pathlib import Path
import glob

from isaaclab.app import AppLauncher
from scipy.spatial.transform import Rotation as R

parser = argparse.ArgumentParser(description="Replay recorded trajectories in Isaac Lab.")
parser.add_argument("--trajectory_dir", type=str, required=True, help="Directory containing episode pkl files")
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

def get_image(env) -> torch.Tensor:
    """Get image from camera - returns torch tensor (defer .cpu() until saving)."""
    rgb_data = env.scene["tiled_camera"].data.output["rgb"]
    return rgb_data[0].clone()  # Return torch tensor directly

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


def find_episode_files(directory):
    """Find all episode*.pkl files in the directory and sort them by episode number."""
    episode_files = glob.glob(str(Path(directory) / "episode*.pkl"))
    
    # Extract episode numbers and sort
    episode_data = []
    for filepath in episode_files:
        filename = Path(filepath).stem  # e.g., 'episode0', 'episode123'
        try:
            # Extract number from 'episodeX'
            episode_num = int(filename.replace("episode", ""))
            episode_data.append((episode_num, filepath))
        except ValueError:
            print(f"Warning: Skipping file with invalid format: {filepath}")
            continue
    
    # Sort by episode number
    episode_data.sort(key=lambda x: x[0])
    
    return episode_data


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
    
    # Get trajectory directory
    trajectory_dir = Path(args_cli.trajectory_dir)
    
    if not trajectory_dir.exists():
        omni.log.error(f"Trajectory directory not found: {trajectory_dir}")
        simulation_app.close()
        return
    
    # Create videos directory
    videos_dir = trajectory_dir / "videos"
    videos_dir.mkdir(exist_ok=True)
    print(f"Videos will be saved to: {videos_dir}")
    
    # Find all episode files
    episode_files = find_episode_files(trajectory_dir)
    
    if not episode_files:
        omni.log.error(f"No episode*.pkl files found in {trajectory_dir}")
        simulation_app.close()
        return
    
    print(f"\n{'='*60}")
    print(f"TRAJECTORY REPLAY")
    print(f"Directory: {trajectory_dir}")
    print(f"Total Episodes Found: {len(episode_files)}")
    print(f"Episodes: {[ep_num for ep_num, _ in episode_files]}")
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
    
    # Main replay loop - iterate through each episode file
    successful_episodes = 0
    failed_episodes = []
    
    for episode_num, episode_filepath in episode_files:
        if episode_num not in [2 ,12 ,16 ,17 ,18, 19, 22 ,23, 25, 32 ,45 ,48]: continue
        
        if not simulation_app.is_running():
            break
        
        try:
            print(f"\n{'='*60}")
            print(f"[REPLAY] Loading Episode {episode_num}")
            print(f"File: {Path(episode_filepath).name}")
            print(f"{'='*60}\n")
            
            # Load trajectory for this episode
            trajectory_data, episode_initial_objects, metadata = load_trajectory(Path(episode_filepath))
            
            # Start video recording
            video_filepath = videos_dir / f"episode_{episode_num}.mp4"
            print(f"Recording video to: {video_filepath}")
            
            # Enable video recording
            env.sim.render_mode = "rgb_array"
            frames = []
            
            with torch.inference_mode():
                for ep_key in sorted(trajectory_data.keys()):
                    episode_trajectory = trajectory_data[ep_key]
                    
                    print(f"[REPLAY] Playing Episode {episode_num} ({len(episode_trajectory)} timesteps)")
                    
                    # Reset environment
                    env.reset()
                    env_ids = torch.arange(env.num_envs, device=env.device)
                    
                    # Set recorded initial object positions
                    if ep_key in episode_initial_objects:
                        for obj_name, obj_data in episode_initial_objects[ep_key].items():
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
                                env.step(actions[:, :36])
                            else:
                                if actions[:,-1] > .5: actions[:,-1] = -.01
                                else: actions[:,-1] = 1.0
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
                            
                            # # Visualize EEF
                            # pose_marker.visualize(
                            #     translations=eef_pos.unsqueeze(0),
                            #     orientations=torch.tensor([eef_quat[3], eef_quat[0], eef_quat[1], eef_quat[2]], 
                            #                             device=env.device).unsqueeze(0),  # Convert to [w,x,y,z]
                            # )
                            
                            # Print progress every 30 steps
                            if step_idx % 30 == 0:
                                print(f"[REPLAY] Episode {episode_num} | Step: {step_idx}/{len(episode_trajectory)} | " +
                                      f"EEF: [{eef_pos[0]:.3f}, {eef_pos[1]:.3f}, {eef_pos[2]:.3f}]")
                        
                        # Capture frame for video
                        try:
                            frame = get_image(env)
                            if frame is not None:
                                frames.append(frame)
                        except:
                            pass
                        
                        time.sleep(0.01)
                    
                    if not simulation_app.is_running():
                        break
            
            # Save video using opencv or imageio
            if frames:
                try:
                    import cv2
                    import numpy as np
                    
                    # Get frame dimensions
                    # import pdb; pdb.set_trace()
                    height, width = frames[0].shape[:2]
                    
                    # Create video writer
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    fps = 30  # Adjust as needed
                    out = cv2.VideoWriter(str(video_filepath), fourcc, fps, (width, height))
                    
                    for frame in frames:
                        # Convert RGB to BGR for OpenCV
                        frame_bgr = cv2.cvtColor(frame.detach().cpu().numpy(), cv2.COLOR_RGB2BGR)
                        out.write(frame_bgr)
                    
                    out.release()
                    print(f"✓ Video saved: {video_filepath}")
                    
                except ImportError:
                    # Fallback to imageio
                    try:
                        import imageio
                        imageio.mimsave(str(video_filepath), frames, fps=30)
                        print(f"✓ Video saved: {video_filepath}")
                    except Exception as e:
                        print(f"✗ Failed to save video: {e}")
            
            print(f"[REPLAY] Episode {episode_num} completed successfully\n")
            successful_episodes += 1
            
        except Exception as e:
            omni.log.error(f"Error replaying episode {episode_num}: {e}")
            import traceback
            traceback.print_exc()
            failed_episodes.append(episode_num)
            continue
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"REPLAY SUMMARY")
    print(f"Total Episodes Processed: {len(episode_files)}")
    print(f"Successful: {successful_episodes}")
    print(f"Failed: {len(failed_episodes)}")
    if failed_episodes:
        print(f"Failed Episodes: {failed_episodes}")
    print(f"Videos saved to: {videos_dir}")
    print(f"{'='*60}\n")
    
    env.close()
    print("Replay finished")


if __name__ == "__main__":
    main()
    simulation_app.close()