import sys
sys.path.append("/workspace/isaaclab/source/droid/droid/controllers/")
sys.path.append('/home/shubham/summer/serl-flow/conditional-flow-matching')

import cv2
import argparse
import time
from pathlib import Path
from collections import deque
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
#🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖
#🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖
TRAIN_DATASET_PATH = "source/serl-flow/dataset/train_paired.pkl"  #required to extract stats
VAL_DATASET_PATH = "source/serl-flow/dataset/train_toy_xyz.pkl"
# VAL_DATASET_PATH = "source/serl-flow/source/serl-flow/dataset/validation.pkl"
task_name = "flow-toy-xyz-debug"
epoch_num = "500"
actual_action = True
bc_policy = True # true for gaussian, false for couple flow
action_replay = False 
state_replay = False
latent =  None 
# latent = "human_actions"
#🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖
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

def get_observation_from_env(env, episodes_ends, ep_latent_positions, class_label=None):
    """
    Extract observation from Isaac Lab environment.
    
    Returns observation matching the training data format.
    Expected format: [eef_pos (3), eef_rpy (3), gripper (1), object_pos (3), ...] 
    """
    # Get robot state (end-effector pose)
    eef_idx = env.scene["robot"].data.body_names.index("panda_hand")
    eef_pos_w = env.scene["robot"].data.body_pos_w[0, eef_idx]
    eef_quat_w = env.scene["robot"].data.body_quat_w[0, eef_idx][[1, 2, 3, 0]]
    eef_rpy = torch.tensor(R.from_quat(eef_quat_w.detach().cpu().numpy()).as_euler('xyz'), dtype=torch.float32, device=env.device)

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
        # torch.tensor(eef_rpy, dtype=torch.float32, device=env.device),
        # gripper_state,
        object_positions[:],
        # class_label.squeeze(0) if class_label is not None else torch.tensor([], device=env.device)
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

def inv_flow_simple(flow_net, x1, cond, steps=50):
    """Invert flow from x1 to x0 and return the full trajectory"""
    batch_size = x1.shape[0]
    curr_x = x1.clone()
    dt = 1.0 / steps
    
    # Store trajectory - list of states at each timestep
    trajectory = [curr_x.clone()]  # Start with x1 at t=1
    for i in range(steps, 0, -1):
        t_val = i / steps
        t = torch.full((batch_size,), t_val, device=x1.device)
        v = flow_net(curr_x, t, global_cond=cond)
        curr_x = curr_x - v * dt
        trajectory.append(curr_x.clone())

    return curr_x  # Shape: [batch, timesteps+1, action_dim]

def plot_3d_trajectories(stored_trajectories, save_path="trajectory_plot_3d.png"):
    """Plot stored trajectories in 3D space with different colors for different task types."""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Color mapping
    color_map = {
        'apple': 'red',
        'mug': 'blue', 
        'sushi': 'green',
        None: 'black'  # For random sampling
    }
    
    # Plot trajectories
    for trajectory, task_name in stored_trajectories:
        color = color_map.get(task_name, 'black')
        label = task_name if task_name is not None else 'random'
        
        # trajectory is [H, 3] where H is the horizon
        ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 
                color=color, alpha=0.7, linewidth=1.5, label=label)
        
        # Mark start and end points
        ax.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2], 
                  color=color, s=50, marker='o', alpha=0.8)
        ax.scatter(trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2], 
                  color=color, s=50, marker='s', alpha=0.8)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y') 
    ax.set_zlabel('Z')
    ax.set_title('3D Trajectories from Inverse Flow and Random Sampling')
    
    # Create legend with unique labels
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"3D trajectory plot saved: {save_path}")
    plt.show()

def run_diffusion_policy(env, dataset, episode_idx, nets, norm_stats, cfg, num_steps, max_episode_length, device, 
                         pose_marker=None, save_trajectory=False, bc_policy=bc_policy, stored_trajectories=None):

    # Reset environment
    obs, _ = env.reset()
    env_ids = torch.arange(env.num_envs, device=env.device)

##################################################################################################
    # # --- Prepare first observation and obs_cond ---
    # start_idx = 0
    # obs_horizon = cfg.obs_horizon
    # obs_dim = dataset.normalized_train_data['state'].shape[1]
    # obs_stack = torch.tensor(dataset.normalized_train_data['state'][start_idx:start_idx+obs_horizon])
    # obs_cond = obs_stack.flatten().unsqueeze(0).to(env.device)  

    # # --- Flow rollout sampling ---
    # pred_horizon = cfg.pred_horizon
    # action_dim = dataset.normalized_train_data['action'].shape[1]
    # num_steps = 50
    # dt = 1.0 / num_steps

    # trajectories = []
    # n = 100
    # for i in range(n):
    #     # Sample x0 ~ N(0,1)
    #     x = torch.randn(pred_horizon, action_dim).unsqueeze(0)
    #     x = x.to(obs_cond.device)
    #     # Integrate flow
    #     for fm_step in range(num_steps):
    #         t = torch.tensor(fm_step * dt, device=obs_cond.device)
    #         t_batch = t.unsqueeze(0)
    #         vt = nets['flow_net'](x, t_batch, global_cond=obs_cond)
    #         x = x + dt * vt
    #     predicted_chunk = x.squeeze(0).detach().cpu().numpy()  # [pred_horizon, action_dim]
    #     trajectories.append(predicted_chunk[:, :3])  # Only plot first 3 dims (position)

    # # --- Plot all sampled trajectories ---
    # fig = plt.figure(figsize=(10, 7))
    # ax = fig.add_subplot(111, projection='3d')
    # for traj in trajectories:
    #     ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], alpha=0.7)
    #     ax.scatter(traj[0, 0], traj[0, 1], traj[0, 2], color='green', s=30)  # Start
    #     ax.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2], color='red', s=30)  # End
    # ax.set_title(f"Sampled Flow Trajectories (n={n})")
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # plt.tight_layout()
    # plt.savefig("source/serl-flow/sampled_flow_trajectories.png", dpi=150, bbox_inches='tight')
    # import pdb; pdb.set_trace()
#####################################################################################################

    print(f"Starting episode...")
    # Get config parameters
    pred_horizon = cfg.pred_horizon
    action_horizon = cfg.action_horizon
    obs_horizon = cfg.obs_horizon

    end_idx = dataset.episode_ends[episode_idx]
    start_idx = 0
    if episode_idx > 0: start_idx = dataset.episode_ends[episode_idx - 1]

    apply_demo_objects(env, dataset.train_data["blocks_init_dict"][episode_idx], env_ids)
    env.reset()
    # Initialize gripper to open position
    # open_action = torch.zeros(7, device=device)
    # open_action[-1] = 1.0  # Open gripper command
    # for _ in range(5):
    #     _ = env.step(open_action.unsqueeze(0))
    
    # Let physics settle
    # sleep(2.0)
    #For visualization / evaluation
    gt_robot_data = {
        "images" : dataset.train_data['gt_robot_images'][start_idx:end_idx],
        "eef_poses" : dataset.train_data['state'][start_idx:end_idx][:,:3],
        "object_poses" : dataset.train_data['gt_robot_object_positions'][start_idx:end_idx]
    }
    human_data = {
        "images" : dataset.train_data['human_images'][start_idx:end_idx],
        "eef_poses" : dataset.train_data['human_poses'][start_idx:end_idx],
        "object_poses" : dataset.train_data['human_object_positions'][start_idx:end_idx]
    }
    episode_ends = dataset.episode_ends
    print("🚀 Starting trajectory inference...")
    nets.eval()
    prev_action = torch.zeros(7, device=device)
    with torch.no_grad():
        # Initialize trajectory recording
        if save_trajectory:
            trajectory_data = {
                'observations': [],
                'images': [],
                'actions': [],
                'object_poses': [],
                'eef_poses': [],
                'camera_params': {},
                'task_names': []
            }
            trajectory_data['camera_params'] = get_camera_parameters(env, "tiled_camera")
        
        gt_states = torch.tensor(dataset.normalized_train_data['state'][start_idx:end_idx]).to(device)
        max_steps = len(gt_states) 
        
        
        if latent == None:
            latent_variable =  None
        elif latent == "human_actions":
            # latent_variable =  human_data["object_poses"]
            latent_variable =  dataset.train_data['human_action'][start_idx:end_idx]
        
        
        # Action ensembling buffer: dictionary mapping timestep -> list of predicted actions
        actions_queue = {}
        
        terminated = False
        truncated = False
        action_dim = 7 
        # Initialize gripper to open position
        open_action = torch.zeros(7, device=device)
        open_action[-1] = 1.0  # Open gripper command
        for _ in range(20):
            _ = env.step(open_action.unsqueeze(0))
        # Get initial image and set up for main loop
        image = get_image(env)
        prev_gripper = 1.0
        # for step_idx in tqdm(range(int(max_steps))):
        task_name = dataset.normalized_train_data['task_name'][start_idx]
        class_label = []
        if task_name == 'apple':
            class_label = [1, 0, 0]
        elif task_name == 'mug':
            class_label = [0, 1, 0]
        elif task_name == 'sushi':
            class_label = [0, 0, 1]
        else:
            raise ValueError(f"Unknown task name: {task_name}")
        class_label = torch.tensor(class_label, dtype=torch.float32, device=device).unsqueeze(0)
        current_obs = get_observation_from_env(env, episode_ends, latent_variable, class_label=class_label)
        current_obs = gt_states[1]
        # print("initial data obs:", gt_states[0])
        print("env", get_observation_from_env(env, episode_ends, latent_variable, class_label=class_label)[:])
        current_obs[6] = .04
        # first_obs = current_obs.clone()
        task_name = None
        obs_history = deque(maxlen=obs_horizon)
        for _ in range(obs_horizon):
            obs_history.append(current_obs)
        # for step_idx in tqdm(range(70)):        
        for step_idx in tqdm(range(1,40)):
                                                                                                                     
            if terminated or truncated:
                break

            obs_stack = torch.stack(list(obs_history), dim=0)  # [obs_horizon, state_dim]
            obs_cond = obs_stack.flatten().unsqueeze(0)  # [1, obs_horizon * state_dim]
            
            # Use human action as initialization (optional)
            human_action_chunk = torch.from_numpy(
                dataset.normalized_train_data['human_action'][start_idx+step_idx:start_idx+step_idx + pred_horizon]
            ).to(device)

            gt_action_chunk = torch.from_numpy(
                dataset.normalized_train_data['action'][start_idx+step_idx:start_idx+step_idx + pred_horizon]
            ).to(device)
            # Pad to match full action dimension and pred_horizon
            if human_action_chunk.shape[0] < pred_horizon:
                padding = torch.zeros(pred_horizon - human_action_chunk.shape[0], 
                                    human_action_chunk.shape[1], device=device)
                human_action_chunk = torch.cat([human_action_chunk, padding], dim=0)

            if not bc_policy:
                if not torch.all(human_action_chunk == -1):
                    if human_action_chunk.shape[-1] < action_dim:
                        diff_dims = action_dim - human_action_chunk.shape[-1]
                        noise = torch.randn(*human_action_chunk.shape[:-1], diff_dims, device=device)
                        print("using human action with noise")
                        x = torch.cat([human_action_chunk[:, :3], noise, human_action_chunk[:, 3:]], dim=-1)
                        x = x.unsqueeze(0)
                    else:
                        x = human_action_chunk.unsqueeze(0)
                else:
                    x =  torch.randn(human_action_chunk.shape[0], action_dim, device=device).unsqueeze(0)   
                
            else:
                # Add more diversity to noise initialization
                noise_scale = 1.0  # Increase noise scale for more diversity
                x = noise_scale * torch.randn(human_action_chunk.shape[0], action_dim-4, device=device).unsqueeze(0)

            _t = 0 # 0.2-0.8
            # Probabilistic selection: 0.2 for inverse flow, 0.8 for random sampling
            task_name = dataset.normalized_train_data['task_name'][start_idx]
            if step_idx <= 4:
                # use_inverse = np.random.rand() < 2.1
                use_inverse = np.random.rand() < 0.0  # Set to 0.0 to force Random Sampling
                task_name = dataset.normalized_train_data['task_name'][start_idx]
                if task_name == "sushi": return None, None, None
                if use_inverse:
                    # Use inverse flow
                    x = inv_flow_simple(nets['flow_net'], x1=gt_action_chunk[:,:3].unsqueeze(0), cond=obs_cond, steps=50)
                    # Store trajectory (first 3 dimensions of the action trajectory)
                    if stored_trajectories is not None:
                        trajectory = x.squeeze(0)[:2, :3].detach().cpu().numpy()  # [H, 3]
                        stored_trajectories.append((trajectory, task_name))
                        print(f"Stored inverse flow trajectory for task: {task_name}")
                else:
                    # x = torch.randn(human_action_chunk.shape[0], action_dim-4, device=device).unsqueeze(0)
                    x = torch.randn(human_action_chunk.shape[0], action_dim-4, device=device).unsqueeze(0)
                    # Store trajectory (first 3 dimensions)
                    if stored_trajectories is not None:
                        trajectory = x.squeeze(0)[:2, :3].detach().cpu().numpy()  # [H, 3]  
                        stored_trajectories.append((trajectory, None))  # None for random sampling
                        print("Stored random sampling trajectory")
            else:
                x =  torch.randn(human_action_chunk.shape[0], action_dim-4, device=device).unsqueeze(0)                   
        
            num_steps = 50
            dt = 1.0 / num_steps
            # for fm_step in range(num_steps):
            for fm_step in range(int((.0)*num_steps), num_steps):
                t = torch.tensor(fm_step * dt, device=device)
                t_batch = t.unsqueeze(0)
                # Use observation conditioning as the model was trained
                vt = nets['flow_net'](x, t_batch, global_cond=obs_cond)
                x = x + dt * vt

            predicted_chunk = x.squeeze(0)[:]  # [pred_horizon, action_dim]
            
            # Add predicted actions to the queue for their corresponding timesteps
            for act_t, act in enumerate(predicted_chunk):
                target_step = step_idx + act_t
                if target_step < max_steps:
                    if target_step not in actions_queue:
                        actions_queue[target_step] = []
                    actions_queue[target_step].append(act)
            
            action = torch.stack(actions_queue[step_idx], dim=0).mean(dim=0)
            # Step through all actions in the predicted chunk (action horizon)
            if step_idx == 1:
                for act in predicted_chunk[:4]:
                    action = act.detach().cpu()
                    # # Track gripper state change and hold for 2 seconds (assuming 30Hz, ~60 steps)
                    # if not hasattr(run_diffusion_policy, "gripper_hold_counter"):
                    #     run_diffusion_policy.gripper_hold_counter = 0
                    #     run_diffusion_policy.last_gripper_value = prev_gripper
                    # gripper_changed = abs(action[-1] - run_diffusion_policy.last_gripper_value) > 1e-3
                    # if gripper_changed:
                    #     run_diffusion_policy.gripper_hold_counter = 60  # Hold for 2 seconds
                    #     run_diffusion_policy.last_gripper_value = action[-1]
                    # if action[-1] < .1:
                    #     action[-1] = -.01  # close
                    #     prev_gripper = action[-1]
                    #     run_diffusion_policy.last_gripper_value = action[-1]
                    # elif action[-1] > .1 :
                    #     action[-1] = 1.0  # open
                    #     prev_gripper = action[-1]
                    #     run_diffusion_policy.last_gripper_value = action[-1]
                    if actual_action:
                        final_action = action
                    elif not absolute_actions:
                        final_action = _delta_action(action)
                    else:
                        final_action = absolute_to_relative_action(current_obs, action)
                    final_action = torch.tensor([final_action[0], final_action[1], final_action[2],0,0,0,1.0], device=device)
                    obs, reward, terminated, truncated, info = env.step(final_action.unsqueeze(0))
                    image = get_image(env)
                    object_poses = get_object_poses_from_env(env)
                    # Get new observation
                    if state_replay: current_obs = gt_states[step_idx]
                    else: current_obs = get_observation_from_env(env, episode_ends, latent_variable)
                    # if step_idx < 10:
                    #     current_obs[6] = .04
                    obs_history.append(current_obs)
                    # Record trajectory
                    if save_trajectory:
                        trajectory_data['observations'].append(current_obs.cpu().numpy())
                        trajectory_data['task_names'].append(task_name)
                        trajectory_data['images'].append(image.cpu().numpy())
                        trajectory_data['actions'].append(action.cpu().numpy())
                        trajectory_data['object_poses'].append(object_poses)
                        trajectory_data['eef_poses'].append({
                            'pos': current_obs[:3].cpu().numpy(),
                            'rpy': current_obs[3:6].cpu().numpy()
                        })
                    if terminated or truncated:
                        break
            # Clean up used actions
            del actions_queue[step_idx]
            
            if action_replay:
                action = gt_action_chunk[0]  
            action = action.detach().cpu()
            # action = unnormalize_data(action, stats=norm_stats['action'])  # Actions are already normalized from flow network
            # print(action)
            # print("action:", action[-1])
            # Track gripper state change and hold for 2 seconds (assuming 30Hz, ~60 steps)
            if not hasattr(run_diffusion_policy, "gripper_hold_counter"):
                run_diffusion_policy.gripper_hold_counter = 0
                run_diffusion_policy.last_gripper_value = prev_gripper

            gripper_changed = abs(action[-1] - run_diffusion_policy.last_gripper_value) > 1e-3
            if gripper_changed:
                run_diffusion_policy.gripper_hold_counter = 60  # Hold for 2 seconds
                run_diffusion_policy.last_gripper_value = action[-1]

            # if action[-1] < .1:
            #     action[-1] = -.01  # close
            #     prev_gripper = action[-1]
            #     run_diffusion_policy.last_gripper_value = action[-1]
            # elif action[-1] > .1 :
            #     action[-1] = 1.0  # open
            #     prev_gripper = action[-1]
            #     run_diffusion_policy.last_gripper_value = action[-1]
            
            if step_idx < int(.0*max_steps): 
                if actual_action:
                    final_action = gt_action_chunk[0]
                elif not absolute_actions:
                    final_action = _delta_action(gt_action_chunk[0])
                else: 
                    final_action = absolute_to_relative_action(current_obs, gt_action_chunk[0])
                
                obs, reward, terminated, truncated, info = env.step(final_action.unsqueeze(0))
                actions_queue = {}
            else: 
                if actual_action:
                    final_action = action
                elif not absolute_actions:
                    # print("delta action:", action)
                    # print("human action chunk:", human<_action_chunk[0])
                    print("current obs:", current_obs[:7])
                    final_action = _delta_action(action)
                    # final_action = action
                else: 
                    final_action = absolute_to_relative_action(current_obs, action)
                    print("absolute action:", action)
                    print("current obs:", current_obs[:7])
                final_action = torch.tensor([final_action[0], final_action[1], final_action[2],0,0,0,1.0], device=device)
                obs, reward, terminated, truncated, info = env.step(final_action.unsqueeze(0)) 
            image = get_image(env)
            object_poses = get_object_poses_from_env(env)
            
            # Get new observation
            if state_replay: current_obs = gt_states[step_idx]
            # if step_idx < 8: current_obs = gt_states[step_idx]
            else: current_obs = get_observation_from_env(env, episode_ends, latent_variable)
            # if step_idx < 10:
            #     current_obs[6] = .04
            # current_obs[6] = 0.0
            obs_history.append(current_obs)

            # Record trajectory
            if save_trajectory:
                trajectory_data['observations'].append(current_obs.cpu().numpy())
                trajectory_data['task_names'].append(task_name)
                trajectory_data['images'].append(image.cpu().numpy())
                trajectory_data['actions'].append(action.cpu().numpy())
                trajectory_data['object_poses'].append(object_poses)
                trajectory_data['eef_poses'].append({
                    'pos': current_obs[:3].cpu().numpy(),
                    'rpy': current_obs[3:6].cpu().numpy()
                })
    
    print(f"✅ Trajectory completed: {len(trajectory_data['actions']) if save_trajectory else step_idx} steps")
    return trajectory_data, gt_robot_data, human_data if save_trajectory else None

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

def save_comparison_video(pred_data, gt_robot_data, human_data, episode_idx, output_dir, camera_data=None):
    output_dir = Path(output_dir) / task_name / epoch_num
    output_dir.mkdir(parents=True, exist_ok=True)
    task_gt = pred_data['task_names'][0]
    output_path = output_dir / f"{task_gt}_{episode_idx:03d}.mp4" if task_gt is not None else output_dir / f"comparison_{episode_idx:03d}.mp4"
    
    camera_data = pred_data['camera_params']

    # Unpack data
    pred_imgs, pred_eef, pred_objs = pred_data['images'], pred_data['eef_poses'], pred_data['object_poses']
    gt_imgs, gt_eef, gt_objs = gt_robot_data['images'], gt_robot_data['eef_poses'], gt_robot_data['object_poses']
    human_imgs, human_eef, human_objs = human_data['images'], human_data['eef_poses'], human_data['object_poses']
    human_imgs = np.array(human_imgs)
    gt_imgs = np.array(gt_imgs)
    
    # Check if human images are valid (not just -1 values)
    has_valid_human_imgs = not np.all(human_imgs == -1)
    # Check if gt images are valid (not just -1 values)
    has_valid_gt_imgs = not np.all(gt_imgs == -1)
    
    if not has_valid_human_imgs:
        print(f"[INFO] Human images are invalid (-1 values) for episode {episode_idx}")
        human_imgs, human_eef, human_objs = None, None, None
        
    if not has_valid_gt_imgs:
        print(f"[INFO] Ground truth images are invalid (-1 values) for episode {episode_idx}")
        gt_imgs, gt_eef, gt_objs = None, None, None
    
    # Count valid image sources
    valid_sources = sum([True, has_valid_gt_imgs, has_valid_human_imgs])  # pred is always valid
    print(f"[INFO] Creating {valid_sources}-panel comparison for episode {episode_idx}")

    n_frames = len(pred_imgs)
    frames = []
    
    # Project all trajectories once
    def project_all(eef_list, obj_poses, cam, img_shape):
        eef_proj = [project_pose_to_image(np.array(e['pos'] if isinstance(e, dict) else e), cam, img_shape) for e in eef_list]
        obj_names = list(obj_poses[0].keys()) if isinstance(obj_poses[0], dict) else ["apple", "mug", "sushi"]
        obj_proj = {name: [] for name in obj_names}
        for obj_data in obj_poses:
            if isinstance(obj_data, dict):
                for name in obj_names:
                    p = obj_data[name].detach().cpu().numpy() if hasattr(obj_data[name], 'detach') else np.array(obj_data[name])
                    obj_proj[name].append(project_pose_to_image(p, cam, img_shape))
            else:
                for name, pos in zip(obj_names, obj_data.reshape(-1, 3)):
                    p = pos.detach().cpu().numpy() if hasattr(pos, 'detach') else np.array(pos)
                    obj_proj[name].append(project_pose_to_image(p, cam, img_shape))
        return eef_proj, obj_proj
    
    if has_valid_gt_imgs and gt_imgs.shape[0] < n_frames:
        gt_pad = n_frames - gt_imgs.shape[0]
        last_gt = gt_imgs[-1:,...]
        gt_imgs = np.concatenate([gt_imgs, np.repeat(last_gt, gt_pad, axis=0)], axis=0)

    if has_valid_human_imgs and human_imgs.shape[0] < n_frames:
        human_pad = n_frames - human_imgs.shape[0]
        last_human = human_imgs[-1:,...]
        human_imgs = np.concatenate([human_imgs, np.repeat(last_human, human_pad, axis=0)], axis=0)
    
    if camera_data:
        img_shape = pred_imgs[0].shape
        pred_eef_proj, pred_obj_proj = project_all(pred_eef, pred_objs, camera_data, img_shape)
        if has_valid_gt_imgs:
            gt_eef_proj, gt_obj_proj = project_all(gt_eef, gt_objs, camera_data, img_shape)
        else:
            gt_eef_proj, gt_obj_proj = None, None
        if has_valid_human_imgs:
            human_eef_proj, human_obj_proj = project_all(human_eef, human_objs, camera_data, img_shape)
        else:
            human_eef_proj, human_obj_proj = None, None

    # Pad pred_eef_proj, pred_obj_proj, gt_eef_proj, gt_obj_proj, human_eef_proj, human_obj_proj if needed
    if len(pred_eef_proj) < n_frames:
        pad_len = n_frames - len(pred_eef_proj)
        last_pred_eef = pred_eef_proj[-1]
        pred_eef_proj += [last_pred_eef] * pad_len
        for name in pred_obj_proj:
            last_pred_obj = pred_obj_proj[name][-1]
            pred_obj_proj[name] += [last_pred_obj] * pad_len

    if has_valid_gt_imgs and len(gt_eef_proj) < n_frames:
        pad_len = n_frames - len(gt_eef_proj)
        last_gt_eef = gt_eef_proj[-1]
        gt_eef_proj += [last_gt_eef] * pad_len
        for name in gt_obj_proj:
            last_gt_obj = gt_obj_proj[name][-1]
            gt_obj_proj[name] += [last_gt_obj] * pad_len

    if has_valid_human_imgs and len(human_eef_proj) < n_frames:
        pad_len = n_frames - len(human_eef_proj)
        last_human_eef = human_eef_proj[-1]
        human_eef_proj += [last_human_eef] * pad_len
        for name in human_obj_proj:
            last_human_obj = human_obj_proj[name][-1]
            human_obj_proj[name] += [last_human_obj] * pad_len
    
    #pad human_obj_proj, gt_obj_proj

    for i in range(n_frames):
        # Process each image
        imgs = []
        # Define which data sources to include based on image validity
        img_sources = []
        labels = []
        
        # Always include predictions (assumed to be valid)
        img_sources.append((pred_imgs[i], pred_eef_proj, pred_obj_proj, (0, 255, 0)))
        labels.append('Predictions')
        
        # Add ground truth if valid
        if has_valid_gt_imgs:
            img_sources.append((gt_imgs[i], gt_eef_proj, gt_obj_proj, (255, 165, 0)))
            labels.append('Ground Truth')
            
        # Add human actions if valid
        if has_valid_human_imgs:
            img_sources.append((human_imgs[i], human_eef_proj, human_obj_proj, (0, 165, 255)))
            labels.append('Human Actions')
            
        for img_data, eef_proj, obj_proj, color in img_sources:

            if isinstance(img_data, (bytes, bytearray)):
                img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)[:,:,::-1]
            else:
                img = np.array(img_data, dtype=np.uint8)
            img = img[..., ::].copy()
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            if camera_data and eef_proj:
                img_bgr = draw_trajectory_on_frame(img_bgr, eef_proj, i, is_eef=True)
                for proj_list in obj_proj.values():
                    img_bgr = draw_trajectory_on_frame(img_bgr, proj_list, i)
            
            imgs.append(img_bgr)
        
        # Add labels
        for idx, (img, label) in enumerate(zip(imgs, labels)):
            color = (0, 255, 0) if idx == 0 else (255, 255, 255)
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            text_x = (img.shape[1] - text_size[0]) // 2
            # Black outline (increase thickness from 2 to 3 or 4)
            cv2.putText(img, label, (text_x, img.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 4)
            # Colored text on top
            cv2.putText(img, label, (text_x, img.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Stack horizontally
        combined = np.hstack(imgs)
        frames.append(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
    
    iio.imwrite(output_path, frames, fps=30, codec="libx264", quality=8)
    print(f"[INFO] Comparison video saved: {output_path} ({n_frames} frames)")
    return output_path

def main():
    """Main function to run diffusion policy inference."""
    
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
        obs = get_observation_from_env(env, None, None)
        print(obs)
    except Exception as e:
        omni.log.error(f"Failed to create environment: {e}")
        import traceback
        traceback.print_exc()
        simulation_app.close()
        return
    
    # Create pose marker for EEF visualization
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    pose_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/policy_eef"))
    
    # Set camera view
    env.sim.set_camera_view(eye=(0.6, -0.3, 1.5), target=(-1.3, 2.3, 0.0))
    
    print(f"\n{'='*60}")
    print(f"DIFFUSION POLICY INFERENCE")
    print(f"Model: {args_cli.checkpoint}")
    print(f"Robot: {args_cli.robot}")
    print(f"Integration steps: {args_cli.num_steps}")
    print(f"Max episode length: {args_cli.max_episode_length}")
    if args_cli.save_trajectories:
        print(f"Saving trajectories to: {args_cli.output_dir}")
    print(f"{'='*60}\n")
    
    # Run episodes
    import random
    # random.seed(42)  # Ensures reproducibility
    total_episodes = len(val_dataset.episode_ends)
    # total_episodes = 2

    # Initialize storage for trajectories
    stored_trajectories = []

    # episode_indices = np.array([i for i in range(30)] + [i for i in range(61, 90)])
    episode_indices = np.array([i for i in range(total_episodes)])
    # episode_indices = np.array([4,12,6,3,1,11,13,7,9,8])
    #################################################################################
    # This is the episode which the states I manually set above guarantee with state
    # replay the robot will successfully pick and place the mug.
    # episode_indices = np.array([20])
    ##################################################################################
    # episode_indices = np.array([0])
    # import pdb; pdb.set_trace()
    np.random.shuffle(episode_indices)
    for idx, episode_idx in enumerate(episode_indices):
        if episode_idx == 19: continue
        if not simulation_app.is_running():
            break

        print(f"\n{'='*60}")
        print(f"Episode {idx + 1}/{total_episodes} (dataset episode {episode_idx})")
        print(f"{'='*60}\n")

        try:
            pred_data, gt_robot_data, human_data  = run_diffusion_policy(
                env=env,
                dataset=val_dataset,
                episode_idx=episode_idx,
                nets=nets,
                norm_stats=norm_stats,
                cfg=cfg,
                num_steps=args_cli.num_steps,
                max_episode_length=args_cli.max_episode_length,
                device=device,
                pose_marker=pose_marker,
                save_trajectory=args_cli.save_trajectories,
                stored_trajectories=stored_trajectories
            )

            # Save trajectory if requested
            if args_cli.save_trajectories and pred_data is not None:
                save_comparison_video(pred_data, gt_robot_data, human_data, episode_idx, args_cli.output_dir)

            print(f"\nEpisode {idx + 1} (dataset episode {episode_idx}) completed:")

        except Exception as e:
            omni.log.error(f"Error in episode {idx + 1} (dataset episode {episode_idx}): {e}")
            import traceback
            traceback.print_exc()
            break
    
    # Plot all collected trajectories in 3D
    if stored_trajectories:
        print(f"\nPlotting {len(stored_trajectories)} collected trajectories...")
        plot_3d_trajectories(stored_trajectories, str(Path(args_cli.output_dir) / "trajectory_plot_3d.png"))
    
    env.close()
    print("Inference finished")


if __name__ == "__main__":
    main()
    simulation_app.close()