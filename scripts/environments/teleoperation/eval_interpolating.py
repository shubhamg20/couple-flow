import sys
sys.path.append("/workspace/isaaclab/source/droid/droid/controllers/")
sys.path.append('/home/shubham/summer/serl-flow/conditional-flow-matching')

import cv2
import argparse
import time
from pathlib import Path
from collections import deque
import pickle

from isaaclab.app import AppLauncher
from scipy.spatial.transform import Rotation as R
from time import sleep
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import imageio.v3 as iio
# Flow matching imports
from flow_policy.configs import FlowMatchingModelRunConfig
from flow_policy.make_networks import instantiate_flow_matching_artifacts
from flow_policy.dataset import IsaacLabDataset, unnormalize_data

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm

parser = argparse.ArgumentParser(description="Run diffusion policy inference in Isaac Lab.")
parser.add_argument("--robot", type=str, default="franka", choices=["franka", "gr1t2"], help="Robot type")
parser.add_argument("--num_steps", type=int, default=30, help="Number of integration steps for flow matching")
parser.add_argument("--max_episode_length", type=int, default=300, help="Maximum steps per episode")
parser.add_argument("--save_trajectories", action="store_true", default=True, help="Save trajectories to file")
parser.add_argument("--output_dir", type=str, default="source/serl-flow/outputs/", help="Output directory for saved trajectories")
parser.add_argument(
    "--enable_pinocchio",
    action="store_true",
    default=False,
    help="Enable Pinocchio.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.task = "Isaac-PickPlace-Franka-custom"
app_launcher_args = vars(args_cli)

#🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖
#🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖
#🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖
TRAIN_DATASET_PATH = "source/serl-flow/dataset/train_paired.pkl"  #required to extract stats
VAL_DATASET_PATH = "source/serl-flow/dataset/validation_paired.pkl"
# VAL_DATASET_PATH = "source/serl-flow/source/serl-flow/dataset/validation.pkl"
task_name = "dsrl-flow-uniform-mixed"
epoch_num = "600"
actual_action = True
bc_policy = True # true for gaussian, false for couple flow
action_replay = False 
state_replay = False
latent =  None 
# latent = "human_actions"
#🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖
#🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖

absolute_actions = True # does not matter dont chnage
args_cli.checkpoint = "source/serl-flow/chkpts/" + task_name + "/epoch_" + epoch_num +".pt"

if args_cli.enable_pinocchio:
    import pinocchio  # noqa: F401

app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

import gymnasium as gym
import omni.log

from isaaclab.markers import FRAME_MARKER_CFG, VisualizationMarkers
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
if args_cli.enable_pinocchio:
    import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401


def load_model_and_config(checkpoint_path, device='cuda'):
    """
    Load trained model and configuration from checkpoint.
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extract config from checkpoint
    cfg = checkpoint['config']
    print(f"Loaded config from checkpoint")
    
    # Initialize flow matching artifacts (model only)
    nets, device = instantiate_flow_matching_artifacts(cfg, model_only=True)
    
    # Load validation dataset for normalization stats
    
    train_dataset = IsaacLabDataset(
        dataset_path=TRAIN_DATASET_PATH,
        with_image=True,
        pred_horizon=cfg.pred_horizon,
        obs_horizon=cfg.obs_horizon,
        action_horizon=cfg.action_horizon,
        num_trajectories=cfg.dataset.num_traj,
    )
    val_dataset = IsaacLabDataset(
        dataset_path=VAL_DATASET_PATH,
        with_image=True,
        pred_horizon=cfg.pred_horizon,
        obs_horizon=cfg.obs_horizon,
        action_horizon=cfg.action_horizon,
        num_trajectories=cfg.dataset.num_traj,
        stats = train_dataset.stats
    )
    state_dict = checkpoint['state_dict']
    # Check if keys have 'flow_net.' prefix and remove it if necessary
    if any(key.startswith('flow_net.') for key in state_dict.keys()):
        print("Removing 'flow_net.' prefix from checkpoint keys")
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('flow_net.'):
                new_key = key.replace('flow_net.', '')
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        state_dict = new_state_dict
    nets['flow_net'].load_state_dict(state_dict)
    nets.eval()
    
    # Create normalization helper
    norm_stats = train_dataset.stats
    print(f"Model loaded successfully. Best validation loss: {checkpoint.get('best_val_loss', 'N/A')}")
    print(f"Training epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"Pred horizon: {cfg.pred_horizon}, Obs horizon: {cfg.obs_horizon}, Action horizon: {cfg.action_horizon}")
    return nets, val_dataset, cfg, device, norm_stats

def get_object_poses_from_env(env):
    object_names = ["apple", "mug", "sushi"]
    object_poses = {}

    for obj_name in object_names:
        if obj_name in env.scene.keys():
            obj_pos = env.scene[obj_name].data.body_pos_w[0, 0].to(env.device)
            object_poses[obj_name] = obj_pos

    return object_poses

def get_observation_from_env(env, episodes_ends, ep_latent_positions):
    """
    Extract observation from Isaac Lab environment.
    
    Returns observation matching the training data format.
    Expected format: [eef_pos (3), eef_rpy (3), gripper (1), object_pos (3), ...] 
    """
    # Get robot state (end-effector pose)
    eef_idx = env.scene["robot"].data.body_names.index("panda_hand")
    eef_pos_w = env.scene["robot"].data.body_pos_w[0, eef_idx]
    eef_quat_w = env.scene["robot"].data.body_quat_w[0, eef_idx][[1, 2, 3, 0]]
    eef_rpy = R.from_quat(eef_quat_w.detach().cpu().numpy()).as_euler('xyz')
    eef_rpy_tensor = torch.from_numpy(eef_rpy).float().to(env.device)

    joint_names = env.scene["robot"].data.joint_names
    joint_positions = env.scene["robot"].data.joint_pos[0]
    gripper_indices = [joint_names.index("panda_finger_joint1"), joint_names.index("panda_finger_joint2")]
    gripper_state = joint_positions[gripper_indices].mean()
    gripper_state = torch.tensor([gripper_state], dtype=torch.float32, device=env.device)
    # gripper_state = 1.0 if gripper_state < 0.03 else .0
    object_poses = get_object_poses_from_env(env)
    object_positions = torch.cat(list(object_poses.values()), dim=-1)
    # object_positions = torch.cat([object_positions[:3], object_positions[-3:]], dim=-1)
    obs = torch.cat([
        eef_pos_w,
        eef_rpy_tensor,
        gripper_state,
        object_positions[:]
        # object_positions
    ], dim=0)
    # if ep_latent_positions is not None: obs = add_latent_positions(obs, ep_latent_positions, env.device)
    return obs

def add_latent_positions(obs, ep_latent_positions, device, window_len=150):
    # import pdb; pdb.set_trace()
    flat = torch.tensor(ep_latent_positions.flatten(), dtype=torch.float32, device=device)
    dim = ep_latent_positions.shape[-1]
    padded = F.pad(flat, (0, max(0, window_len*dim - flat.shape[0])), mode='constant')
    padded = padded[:window_len*dim]
    obs = torch.cat([obs, padded], axis=-1)

    return obs

def project_pose_to_image(ee_pos, camera_data, img_shape):
    """Project 3D EE position to 2D image coordinates."""
    K = np.array(camera_data['intrinsics'])
    if "rotation" in camera_data:
        R_cam = np.array(camera_data['rotation'])
    elif "quat" in camera_data:
        q = camera_data['quat']
        R_cam = R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    t = np.array(camera_data['translation'])
    
    # Transform point from world to camera frame
    pt_cam = R_cam.T @ (ee_pos - t)
    
    # Project to image plane
    if pt_cam[0] > 0:  # Check if point is in front of camera (+X is forward)
        px = K @ np.array([-pt_cam[1], -pt_cam[2], pt_cam[0]])
        u = int(px[0] / px[2])
        v = int(px[1] / px[2])
        
        # Check if projection is within image bounds
        if 0 <= u < img_shape[1] and 0 <= v < img_shape[0]:
            return (u, v)
    
    return None


def draw_trajectory_on_frame(img, all_projections, current_idx, is_eef=False):
    """Draw all EE positions up to current frame as green dots."""
    # Draw all previous points
    for i in range(current_idx + 1):
        if all_projections[i] is not None:
            u, v = all_projections[i]
            # Draw green dot
            cv2.circle(img, (u, v), 1, (0, 255, 0), -1)
            
            # Optional: draw line connecting points
            if i > 0 and all_projections[i-1] is not None:
                u_prev, v_prev = all_projections[i-1]
                cv2.line(img, (u_prev, v_prev), (u, v), (0, 255, 0), 1)

            twenty_percent_idx = int(0. * len(all_projections))
            if is_eef and i >= twenty_percent_idx:
                # Draw red X at 20% timestep mark
                if twenty_percent_idx < len(all_projections) and all_projections[twenty_percent_idx] is not None:
                    u, v = all_projections[twenty_percent_idx]
                    # Draw red X with lines
                    line_length = 2
                    cv2.line(img, (u - line_length, v - line_length), (u + line_length, v + line_length), (0, 0, 255), 2)
                    cv2.line(img, (u - line_length, v + line_length), (u + line_length, v - line_length), (0, 0, 255), 2)
        
    # Highlight current position with larger circle
    if all_projections[current_idx] is not None:
        u, v = all_projections[current_idx]
        cv2.circle(img, (u, v), 1, (0, 255, 255), 2)  # Yellow outline
    
    return img

def get_camera_parameters(env, camera_name):
    K = env.scene[camera_name].data.intrinsic_matrices[0].detach().cpu().numpy()
    q = env.scene[camera_name].data.quat_w_world[0].detach().cpu().numpy()
    t = env.scene[camera_name].data.pos_w[0].detach().cpu().numpy()
    return {
            "intrinsics": K,
            "quat": q,
            "translation": t
        }

def get_image(env) -> torch.Tensor:
    """Get image from camera - returns torch tensor (defer .cpu() until saving)."""
    rgb_data = env.scene["tiled_camera"].data.output["rgb"]
    return rgb_data[0].clone()  # Return torch tensor directly

def apply_demo_objects(env, demo_objects, env_ids):
    """Apply demo object positions to scene."""
    ############################## FOR DEBUGGING MUG CONVEX DECOMPOSITION #####################################
    positions = {
        "tray": {
            "pos": [0.41, 0.42, 1.0],
            "quat": [0.707, 0.707, 0.0, 0.0]
        }
    }
    ############################################################################################################

    for obj_name in ["apple", "mug", "sushi"]:
        if obj_name in env.scene.keys() and obj_name in demo_objects:
            asset = env.scene[obj_name]
            pos = torch.tensor(demo_objects[obj_name]["pos"], device=env.device).unsqueeze(0)
            quat = torch.tensor(demo_objects[obj_name]["quat"], device=env.device).unsqueeze(0)
            ############################## FOR DEBUGGING MUG CONVEX DECOMPOSITION #####################################
            # pos = torch.tensor(debug_positions[obj_name]["pos"], device=env.device).unsqueeze(0)
            # quat = torch.tensor(debug_positions[obj_name]["quat"], device=env.device).unsqueeze(0)
            ############################################################################################################
            root_pose = torch.cat([pos, quat], dim=-1)
            velocities = torch.zeros((1, 6), device=env.device)
            asset.write_root_pose_to_sim(root_pose, env_ids=env_ids)
            asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)
    
    # Handle tray separately since it's an XFormPrim, not a RigidObject
    if "tray" in env.scene.keys():
        tray_asset = env.scene["tray"]
        if hasattr(tray_asset, 'set_world_poses'):
            pos = torch.tensor(positions["tray"]["pos"], device=env.device).unsqueeze(0)
            quat = torch.tensor(positions["tray"]["quat"], device=env.device).unsqueeze(0)
            tray_asset.set_world_poses(pos, quat)
        elif hasattr(tray_asset, 'write_root_pose_to_sim'):
            pos = torch.tensor(positions["tray"]["pos"], device=env.device).unsqueeze(0)
            quat = torch.tensor(positions["tray"]["quat"], device=env.device).unsqueeze(0)
            root_pose = torch.cat([pos, quat], dim=-1)
            velocities = torch.zeros((1, 6), device=env.device)
            tray_asset.write_root_pose_to_sim(root_pose, env_ids=env_ids)
            tray_asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)

def generate_latent_endpoints(pred_horizon, action_dim, device):
    """
    Generate two random points in Gaussian latent noise space.
    
    Args:
        pred_horizon: Prediction horizon length
        action_dim: Action dimension
        device: Device to create tensors on
        
    Returns:
        Tuple of two random Gaussian noise points
    """
    point1 = torch.randn(pred_horizon, action_dim, device=device)
    point2 = torch.randn(pred_horizon, action_dim, device=device)
    return point1, point2

def interpolate_latent_points(point1, point2, num_points=10):
    """
    Interpolate between two points in Gaussian latent noise space.
    
    Args:
        point1: First point in latent space [pred_horizon, action_dim]
        point2: Second point in latent space [pred_horizon, action_dim]  
        num_points: Number of interpolation points to generate
        
    Returns:
        List of interpolated points
    """
    interpolated_points = []
    for i in range(num_points):
        alpha = i / (num_points - 1)  # Alpha from 0 to 1
        interpolated_point = (1 - alpha) * point1 + alpha * point2
        interpolated_points.append(interpolated_point)
    return interpolated_points

def run_interpolation_rollout(env, nets, cfg, device, interpolation_point, max_episode_length=150, save_trajectory=True, obj_init_positions=None):
    """
    Run a single rollout using an interpolated latent point.
    """
    # Reset environment
    obs, _ = env.reset()
    env_ids = torch.arange(env.num_envs, device=env.device)

    print(f"Starting interpolation rollout...")
    # Get config parameters
    pred_horizon = cfg.pred_horizon
    obs_horizon = cfg.obs_horizon

    apply_demo_objects(env, obj_init_positions, env_ids)
    
    # Let physics settle
    sleep(2.0)
    
    print("🚀 Starting trajectory inference...")
    nets.eval()
    
    with torch.no_grad():
        # Initialize trajectory recording
        trajectory_data = {
            'observations': [],
            'images': [],
            'actions': [],
            'object_poses': [],
            'eef_poses': [],
            'camera_params': {},
            'moved_object': []
        }
        trajectory_data['camera_params'] = get_camera_parameters(env, "tiled_camera")
        
        # Get initial observation from environment
        current_obs = get_observation_from_env(env, None, None)
        current_obs[6] = 0.04  # Set initial gripper state

        obs_history = deque(maxlen=obs_horizon)
        for _ in range(obs_horizon):
            obs_history.append(current_obs)
        
        # Action ensembling buffer
        actions_queue = {}
        
        terminated = False
        truncated = False
        action_dim = 7
        
        # Get initial image
        image = get_image(env)
        prev_gripper = 1.0
        max_step = 50
        moved_object = -1
        for step_idx in tqdm(range(max_step)):  # Run complete episode
            if terminated or truncated:
                break

            obs_stack = torch.stack(list(obs_history), dim=0)  # [obs_horizon, state_dim]
            obs_cond = obs_stack.flatten().unsqueeze(0)  # [1, obs_horizon * state_dim]
            
            # Use interpolated latent point as starting noise
            x = interpolation_point.clone().unsqueeze(0) if interpolation_point.dim() == 2 else interpolation_point.clone()

            # Flow matching inference
            num_steps = 50
            dt = 1.0 / num_steps
            for fm_step in range(num_steps):
                t = torch.tensor(fm_step * dt, device=device)
                t_batch = t.unsqueeze(0)
                # Use observation conditioning as the model was trained
                vt = nets['flow_net'](x, t_batch, global_cond=obs_cond)
                x = x + dt * vt

            predicted_chunk = x.squeeze(0)[:pred_horizon]  # [pred_horizon, action_dim]
            
            # Store the entire action chunk for plotting (only on first step)
            if step_idx == 0:
                trajectory_data['predicted_action_chunk'] = predicted_chunk.detach().cpu().numpy()
            
            # Add predicted actions to the queue for their corresponding timesteps
            for act_t, act in enumerate(predicted_chunk):
                target_step = step_idx + act_t
                if target_step < max_episode_length:
                    if target_step not in actions_queue:
                        actions_queue[target_step] = []
                    actions_queue[target_step].append(act)
            
            # Get action for current step
            if step_idx in actions_queue:
                action = torch.stack(actions_queue[step_idx], dim=0).mean(dim=0)
                del actions_queue[step_idx]
            else:
                # Fallback to first action if queue is empty
                action = predicted_chunk[0]
                
            action = action.detach().cpu()
            
            # Process gripper action
            if action[-1] < 0.5:
                action[-1] = -0.01  # close
                prev_gripper = action[-1]
            elif action[-1] > 0.5:
                action[-1] = 1.0  # open
                prev_gripper = action[-1]
            
            # Apply action based on configuration
            if actual_action:
                final_action = action
            elif not absolute_actions:
                final_action = _delta_action(action)
            else: 
                final_action = absolute_to_relative_action(current_obs, action)
                
            obs, reward, terminated, truncated, info = env.step(final_action.unsqueeze(0)) 
            
            # Get updated state
            image = get_image(env)
            object_poses = get_object_poses_from_env(env)
            if step_idx == max_step-1:
                final_obj_poses = get_object_poses_from_env(env)
                # Infer moved object by comparing distances between initial and final positions
                if obj_init_positions is not None:
                    moved_object = max(final_obj_poses.keys(), key=lambda obj: torch.norm(final_obj_poses[obj] - torch.tensor(obj_init_positions[obj]["pos"], device=env.device)).item())
                

            current_obs = get_observation_from_env(env, None, None)
            
            if step_idx < 10:
                current_obs[6] = 0.04
            
            obs_history.append(current_obs)

            # Record trajectory
            if save_trajectory:
                trajectory_data['observations'].append(current_obs.cpu().numpy())
                trajectory_data['images'].append(image.cpu().numpy())
                trajectory_data['actions'].append(action.cpu().numpy())
                trajectory_data['object_poses'].append(object_poses)
                trajectory_data['moved_object'].append(moved_object)
                trajectory_data['eef_poses'].append({
                    'pos': current_obs[:3].cpu().numpy(),
                    'rpy': current_obs[3:6].cpu().numpy()
                })
    
    print(f"✅ Trajectory completed: {len(trajectory_data['actions'])} steps")
    return trajectory_data

def _delta_action(action):
    if action.shape[0] < 7:
        pad = torch.zeros(7 - action.shape[0], device=action.device)
        action = torch.cat([action, pad], dim=0)
    final_action = change_axis(action)
    final_action[:6] *= 5.0
    final_action[6] = 1.0
    return final_action

def change_axis(action):
    pos_action = torch.tensor([action[1], -action[0], action[2], action[4], -action[3], action[5], action[-1]], device=action.device)
    return pos_action

def absolute_to_relative_action(current_state, target_state):
    final_action = target_state[:6] - current_state[:6]
    final_action = torch.cat([final_action, target_state[6:7]], dim=0)
    final_action *= 6.5
    # final_action[3] = 0
    # final_action[4] = 0
    # final_action[5] = 0
    final_action = change_axis(final_action)
    return final_action

def main():
    """Main function to run interpolation-based rollouts."""
    
    # Load model
    try:
        nets, val_dataset, cfg, device, norm_stats = load_model_and_config(args_cli.checkpoint)
    except Exception as e:
        omni.log.error(f"Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        simulation_app.close()
        return
    
    # Create environment
    try:
        env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
        env.reset()
    except Exception as e:
        omni.log.error(f"Failed to create environment: {e}")
        import traceback
        traceback.print_exc()
        simulation_app.close()
        return
    
    # Set camera view
    env.sim.set_camera_view(eye=(0.6, -0.3, 1.5), target=(-1.3, 2.3, 0.0))
    
    print(f"\n{'='*60}")
    print(f"INTERPOLATION-BASED POLICY INFERENCE")
    print(f"Model: {args_cli.checkpoint}")
    print(f"Robot: {args_cli.robot}")
    print(f"Max episode length: {args_cli.max_episode_length}")
    if args_cli.save_trajectories:
        print(f"Saving trajectories to: {args_cli.output_dir}")
    print(f"{'='*60}\n")
    
    # Run interpolation experiments
    pred_horizon = cfg.pred_horizon
    action_dim = 7
    
    # Number of experiments to run
    num_experiments = 1
    obj_init_positions = val_dataset.normalized_train_data["blocks_init_dict"][-3]
    for exp_idx in range(num_experiments):
        if not simulation_app.is_running():
            break

        print(f"\n{'='*60}")
        print(f"Random Sampling Experiment {exp_idx + 1}/{num_experiments}")
        print(f"Running 20 random Gaussian latent points")
        print(f"{'='*60}\n")

        try:
            # Generate 20 random Gaussian latent points
            random_points = []
            for i in range(100):
                # random_point = torch.randn(pred_horizon, action_dim, device=device)
                action_min = torch.tensor([-0.334, -0.849, -0.250, -0.094, -0.102, -0.236, -0.080], dtype=torch.float32, device=device)
                action_max = torch.tensor([0.289, 0.379, 0.786, 0.064, 0.059, 0.071, 1.000], dtype=torch.float32, device=device)
                random_point = (action_min + (action_max - action_min) * torch.rand(pred_horizon, action_dim, device=device)).unsqueeze(0)
                random_points.append(random_point)
            
            print(f"Generated 20 random Gaussian points")
            
            # # Save starting XYZ coordinates to text file
            output_dir = Path(args_cli.output_dir) / task_name / epoch_num / "random_sampling"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            xyz_file_path = output_dir / f"random_gaussian_xyz_exp{exp_idx:02d}.txt"
            with open(xyz_file_path, 'w') as f:
                f.write("# Random Gaussian Latent Points - All 7 Dimensions\n")
                f.write("# Point_ID: numpy_array\n")
                for i, point in enumerate(random_points):
                    all_dims = point[:, :].cpu().numpy()  # First timestep, all dimensions
                    f.write(f"{i}: {all_dims}\n")
            
            # print(f"Saved starting XYZ coordinates to: {xyz_file_path}")
            
            # Run rollouts for each random point
            trajectories_data = []
            for point_idx, random_point in enumerate(random_points):
                print(f"\n--- Running random point {point_idx + 1}/20 ---")
                
                pred_data = run_interpolation_rollout(
                    env=env,
                    nets=nets,
                    cfg=cfg,
                    device=device,
                    interpolation_point=random_point,
                    max_episode_length=args_cli.max_episode_length,
                    save_trajectory=args_cli.save_trajectories,
                    obj_init_positions=obj_init_positions
                )

                trajectories_data.append(pred_data)

                # Save trajectory if requested
                if args_cli.save_trajectories and pred_data is not None:
                    # Create output directory with random sampling info
                    video_output_dir = output_dir / "videos"
                    video_output_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Save with experiment and point index in filename
                    output_path = video_output_dir / f"exp{exp_idx:02d}_random{point_idx:02d}.mp4"
                    
                    # Create simple visualization for random point trajectory
                    frames = []
                    for img in pred_data['images']:
                        if isinstance(img, (bytes, bytearray)):
                            frame = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)[:,:,::-1]
                        else:
                            frame = np.array(img, dtype=np.uint8)
                        frame = frame[..., ::].copy()
                        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        
                        frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
                    
                    iio.imwrite(output_path, frames, fps=30, codec="libx264", quality=8)
                    print(f"Saved random point video: {output_path}")

                print(f"Random point {point_idx + 1}/20 completed")

            print(f"\nExperiment {exp_idx + 1} completed with all 20 random points")


            # Plot analysis for this experiment (using random points)
            plot_path = plot_random_sampling_analysis(random_points, trajectories_data, exp_idx, output_dir)
            print(f"Saved analysis plot for experiment {exp_idx + 1}: {plot_path}")

        except Exception as e:
            omni.log.error(f"Error in experiment {exp_idx + 1}: {e}")
            import traceback
            traceback.print_exc()
            break
    
    env.close()
    print("Interpolation experiments finished")


def plot_interpolation_analysis(interpolated_points, trajectories_data, exp_idx, output_dir):
    """
    Plot simple side-by-side analysis: 3D latent points and corresponding action trajectories.
    
    Args:
        interpolated_points: List of interpolated latent points
        trajectories_data: List of trajectory data for each interpolated point
        exp_idx: Experiment index
        output_dir: Output directory for saving plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create figure with 2 subplots
    fig = plt.figure(figsize=(16, 6))
    
    # Generate colors for each interpolation point
    colors = cm.get_cmap('tab10')(np.linspace(0, 1, len(interpolated_points)))
    
    # Left plot: 3D visualization of interpolated latent points (first timestep, xyz)
    ax1 = fig.add_subplot(121, projection='3d')
    
    latent_points_array = torch.stack(interpolated_points).cpu().numpy()  # [num_points, pred_horizon, action_dim]
    
    # Use first timestep and first 3 dimensions as XYZ coordinates
    for i, point in enumerate(latent_points_array):
        x, y, z = point[0, :3]  # First timestep, first 3 action dimensions
        ax1.scatter(x, y, z, color=colors[i], s=100, alpha=0.8, label=f'Point {i}')
    
    # Draw line connecting the points to show interpolation path
    xyz_coords = latent_points_array[:, 0, :3]
    ax1.plot(xyz_coords[:, 0], xyz_coords[:, 1], xyz_coords[:, 2], 'k--', alpha=0.5, linewidth=1)
    
    ax1.set_xlabel('Latent X')
    ax1.set_ylabel('Latent Y')  
    ax1.set_zlabel('Latent Z')
    ax1.set_title('Interpolated Latent Points (3D)')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Right plot: 3D Action trajectories (XYZ position commands as 3D paths)
    ax2 = fig.add_subplot(122, projection='3d')
    
    for i, traj in enumerate(trajectories_data):
        if 'predicted_action_chunk' in traj and traj['predicted_action_chunk'] is not None:
            # Use the predicted action chunk (8 actions)
            actions = traj['predicted_action_chunk']  # Shape: [pred_horizon, action_dim]
            
            # Plot XYZ position commands as a 3D trajectory
            x_actions = actions[:, 0]
            y_actions = actions[:, 1] 
            z_actions = actions[:, 2]
            
            # Plot the 3D trajectory
            ax2.plot(x_actions, y_actions, z_actions, color=colors[i], linewidth=3, 
                    marker='o', markersize=4, alpha=0.8, label=f'Point {i}')
            
            # Mark start and end points
            ax2.scatter(x_actions[0], y_actions[0], z_actions[0], color=colors[i], 
                       s=100, marker='o', alpha=1.0)  # Start point
            ax2.scatter(x_actions[-1], y_actions[-1], z_actions[-1], color=colors[i], 
                       s=100, marker='s', alpha=1.0)  # End point
    
    ax2.set_xlabel('X Action')
    ax2.set_ylabel('Y Action')
    ax2.set_zlabel('Z Action')
    ax2.set_title('Predicted Action Trajectories (3D)')
    
    # Create custom legend for action plot
    from matplotlib.lines import Line2D
    legend_elements = []
    for i in range(len(trajectories_data)):
        legend_elements.append(Line2D([0], [0], color=colors[i], lw=3, label=f'Point {i}'))
    
    ax2.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.suptitle(f'Interpolation Analysis - Experiment {exp_idx + 1}', fontsize=14)
    plt.tight_layout()
    
    # Save the plot
    plot_path = output_dir / f"interpolation_analysis_exp{exp_idx:02d}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved interpolation analysis plot: {plot_path}")
    return plot_path

def plot_random_sampling_analysis(random_points, trajectories_data, exp_idx, output_dir):
    """
    Plot analysis of random Gaussian sampled points and their resulting action trajectories.
    
    Args:
        random_points: List of random Gaussian latent points
        trajectories_data: List of trajectory data for each random point
        exp_idx: Experiment index
        output_dir: Output director
        y for saving plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create figure with 2 subplots
    fig = plt.figure(figsize=(16, 6))
    
    # Generate colors based on moved objects
    object_to_color = {'apple': 'red', 'mug': 'blue', 'sushi': 'green'}
    colors = []
    moved_objects = []
    
    for i, traj in enumerate(trajectories_data):
        if 'moved_object' in traj and len(traj['moved_object']) > 0:
            moved_obj = traj['moved_object'][-1]  # Get the final moved object
            moved_objects.append(moved_obj)
            colors.append(object_to_color.get(moved_obj, 'gray'))
        else:
            moved_objects.append('unknown')
            colors.append('gray')
    
    # Left plot: 3D visualization of random latent points (first timestep, xyz)
    ax1 = fig.add_subplot(121, projection='3d')
    
    latent_points_array = torch.stack(random_points).cpu().numpy()  # [num_points, pred_horizon, action_dim]
    
    # Use first timestep and first 3 dimensions as XYZ coordinates
    for i, point in enumerate(latent_points_array):
        x, y, z = point[0, :3]  # First timestep, first 3 action dimensions
        color = colors[i] if i < len(colors) else 'gray'
        moved_obj = moved_objects[i] if i < len(moved_objects) else 'unknown'
        ax1.scatter(x, y, z, color=color, s=100, alpha=0.8, label=f'{moved_obj}' if moved_obj not in [mo for j, mo in enumerate(moved_objects[:i])] else "")
    
    ax1.set_xlabel('Latent X')
    ax1.set_ylabel('Latent Y')  
    ax1.set_zlabel('Latent Z')
    ax1.set_title('Random Gaussian Latent Points (3D)')
    if len(random_points) <= 10:
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Right plot: 3D Action trajectories (XYZ position commands as 3D paths)
    ax2 = fig.add_subplot(122, projection='3d')
    
    for i, traj in enumerate(trajectories_data):
        if 'predicted_action_chunk' in traj and traj['predicted_action_chunk'] is not None:
            # Use the predicted action chunk (8 actions)
            actions = traj['predicted_action_chunk']  # Shape: [pred_horizon, action_dim]
            
            # Plot XYZ position commands as a 3D trajectory
            x_actions = actions[:, 0]
            y_actions = actions[:, 1] 
            z_actions = actions[:, 2]
            
            # Plot the 3D trajectory
            ax2.plot(x_actions, y_actions, z_actions, color=colors[i], linewidth=2, 
                    marker='o', markersize=3, alpha=0.7, label=f'Point {i}' if i < 10 else "")
            
            # Mark start and end points
            ax2.scatter(x_actions[0], y_actions[0], z_actions[0], color=colors[i], 
                       s=80, marker='o', alpha=1.0)  # Start point
            ax2.scatter(x_actions[-1], y_actions[-1], z_actions[-1], color=colors[i], 
                       s=80, marker='s', alpha=1.0)  # End point
    
    ax2.set_xlabel('X Action')
    ax2.set_ylabel('Y Action')
    ax2.set_zlabel('Z Action')
    ax2.set_title('Predicted Action Trajectories (3D)')
    
    # Create custom legend for action plot (only show first 10 for readability)
    if len(trajectories_data) <= 10:
        from matplotlib.lines import Line2D
        legend_elements = []
        for i in range(min(10, len(trajectories_data))):
            legend_elements.append(Line2D([0], [0], color=colors[i], lw=2, label=f'Point {i}'))
        ax2.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.suptitle(f'Random Sampling Analysis - Experiment {exp_idx + 1} ({len(random_points)} points)', fontsize=14)
    plt.tight_layout()
    
    # Save the plot
    plot_path = output_dir / f"random_sampling_analysis_exp{exp_idx:02d}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved random sampling analysis plot: {plot_path}")
    return plot_path




if __name__ == "__main__":
    main()
    simulation_app.close()