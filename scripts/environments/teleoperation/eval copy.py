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
from flow_policy.dataset import IsaacLabDataset

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

# Paths for validation dataset (used for normalization stats)
VAL_DATASET_PATH = "source/serl-flow/dataset/validation.pkl"
# VAL_DATASET_PATH = "source/serl-flow/dataset/train.pkl"
# args_cli.checkpoint = "source/serl-flow/chkpts/single_task.pt"
# args_cli.checkpoint = "source/serl-flow/chkpts/multitask_bc2.pt"
task_name = "couple_flow_human_conditioning"
epoch_num = "500"
args_cli.checkpoint = "source/serl-flow/chkpts/" + task_name + "/epoch_" + epoch_num +".pt"
explicit_conditioning = False   # Important variable 🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖

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


class NormalizationStats:
    """Helper class to store and apply normalization statistics."""
    
    def __init__(self, dataset):
        """Extract normalization stats from dataset."""
        # Get stats from dataset
        self.state_min = torch.tensor(dataset.stats['state']['min'], dtype=torch.float32)
        self.state_max = torch.tensor(dataset.stats['state']['max'], dtype=torch.float32)
    
    def normalize_state(self, state, device='cuda'):
        """Normalize state observation."""
        min = self.state_min.to(device)
        max = self.state_max.to(device)
        nstate = (state - min) / (max - min + 1e-8)
        # normalize to [-1, 1]
        nstate = nstate * 2 - 1
        return nstate

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
    val_dataset = IsaacLabDataset(
        dataset_path=VAL_DATASET_PATH,
        with_image=True,
        pred_horizon=cfg.pred_horizon,
        obs_horizon=cfg.obs_horizon,
        action_horizon=cfg.action_horizon,
        num_trajectories=cfg.dataset.num_traj,
    )
    
    # Load model weights
    nets.load_state_dict(checkpoint['state_dict'])
    nets.eval()
    
    # Create normalization helper
    norm_stats = NormalizationStats(val_dataset)
    
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
    eef_rpy = torch.tensor(R.from_quat(eef_quat_w.detach().cpu().numpy()).as_euler('xyz'), dtype=torch.float32, device=env.device)

    joint_names = env.scene["robot"].data.joint_names
    joint_positions = env.scene["robot"].data.joint_pos[0]
    gripper_indices = [joint_names.index("panda_finger_joint1"), joint_names.index("panda_finger_joint2")]
    gripper_state = joint_positions[gripper_indices].mean()
    gripper_state = torch.tensor([gripper_state], dtype=torch.float32, device=env.device)
    gripper_state = -(gripper_state - 0.03999)/0.03999       #0.03999->0  open              franka 0->close,   1->open
    print("real gripper state:", gripper_state.item())
    gripper_state = torch.tensor(                            #0->1 close
        [1.0 if float(gripper_state) > 0.3 else 0.0],
        dtype=torch.float32,
        device=env.device
    )

    object_poses = get_object_poses_from_env(env)
    object_positions = torch.cat(list(object_poses.values()), dim=-1)
    # object_positions = torch.cat([object_positions[:3], object_positions[-3:]], dim=-1)
    obs = torch.cat([
        eef_pos_w,
        torch.tensor(eef_rpy, dtype=torch.float32, device=env.device),
        gripper_state,
        # object_positions[:3]
        object_positions
    ], dim=0)
    if ep_latent_positions is not None: obs = add_latent_positions(obs, ep_latent_positions, env.device)
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


def draw_trajectory_on_frame(img, all_projections, current_idx):
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
    for obj_name in ["apple", "mug", "sushi"]:
        if obj_name in env.scene.keys() and obj_name in demo_objects:
            asset = env.scene[obj_name]
            pos = torch.tensor(demo_objects[obj_name]["pos"], device=env.device).unsqueeze(0)
            quat = torch.tensor(demo_objects[obj_name]["quat"], device=env.device).unsqueeze(0)
            root_pose = torch.cat([pos, quat], dim=-1)
            velocities = torch.zeros((1, 6), device=env.device)
            asset.write_root_pose_to_sim(root_pose, env_ids=env_ids)
            asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)

def run_diffusion_policy(env, dataset, episode_idx, nets, norm_stats, cfg, num_steps, max_episode_length, device, 
                         pose_marker=None, save_trajectory=False, bc_policy=False):

    # Reset environment
    obs, _ = env.reset()
    env_ids = torch.arange(env.num_envs, device=env.device)

    state_replay = True  # Whether to use ground-truth state replay for observations

    print(f"Starting episode...")
    # Get config parameters
    pred_horizon = cfg.pred_horizon
    action_horizon = cfg.action_horizon
    obs_horizon = cfg.obs_horizon

    end_idx = dataset.episode_ends[episode_idx]
    start_idx = 0
    if episode_idx > 0: start_idx = dataset.episode_ends[episode_idx - 1]

    apply_demo_objects(env, dataset.train_data["blocks_init_dict"][episode_idx], env_ids)
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
            }
            trajectory_data['camera_params'] = get_camera_parameters(env, "tiled_camera")
        
        gt_states = torch.tensor(dataset.normalized_train_data['state'][start_idx:end_idx]).to(device)
        # gt_states = torch.cat([gt_states[:,:7], gt_states[:,-3:]], dim=-1)  # Use only EEF pos, rpy and object pos
        max_steps = len(gt_states) if state_replay else max_episode_length
        
        # max_steps = max_episode_length

        #Important 🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖
        # latent_variable =  None
        # latent_variable =  human_data["object_poses"]
        latent_variable =  dataset.train_data['human_action'][start_idx:end_idx]
        if state_replay: 
            current_obs = gt_states[0]
        # else:
        # current_obs = get_observation_from_env(env, episode_ends, latent_variable)

        # import pdb; pdb.set_trace() 
        current_obs[6] = 0.0 
        # Initialize observation history buffer
        obs_history = deque(maxlen=obs_horizon)
        for _ in range(obs_horizon):
            obs_history.append(current_obs)
        
        # Action ensembling buffer: dictionary mapping timestep -> list of predicted actions
        actions_queue = {}
        
        terminated = False
        truncated = False
        action_dim = 7 
        _action = torch.zeros(action_dim, device=device)
        _action[-1] = 1.0  # Open gripper initially
        for _ in range(10): _ = env.step(_action.unsqueeze(0))
        sleep(3)
        image = get_image(env)
        prev_action = 1.0
        # for step_idx in tqdm(range(int(max_steps))):
        for step_idx in tqdm(range(int(len(gt_states)))):
        # for step_idx in tqdm(range(10)):
            # print(gt_states[step_idx][-3:])
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

            # if explicit_conditioning:
            #     obs_cond = torch.cat([obs_cond, human_action_chunk.flatten().unsqueeze(0)], dim=-1)

            
            if not explicit_conditioning:
                if human_action_chunk.shape[-1] < action_dim:
                    diff_dims = action_dim - human_action_chunk.shape[-1]
                    noise = torch.randn(*human_action_chunk.shape[:-1], diff_dims, device=device)
                    x = torch.cat([human_action_chunk[:, :3], noise, human_action_chunk[:, 3:]], dim=-1)
                    x = x.unsqueeze(0)
                else:
                    x = human_action_chunk.unsqueeze(0)

            else:
                x = torch.randn(human_action_chunk.shape[0], action_dim, device=device).unsqueeze(0)

            if bc_policy:
                x = torch.randn_like(gt_action_chunk).unsqueeze(0) 
            # Flow matching inference
            dt = 1.0 / num_steps
            for fm_step in range(num_steps):
                t = torch.tensor(fm_step * dt, device=device)
                t_batch = t.unsqueeze(0)  # [1]
                vt = nets['flow_net'](x, t_batch, global_cond=obs_cond)
                x = x + dt * vt
            action_horizon = 8
            predicted_chunk = x.squeeze(0)[:action_horizon]  # [pred_horizon, action_dim]
            
            # Add predicted actions to the queue for their corresponding timesteps
            for act_t, act in enumerate(predicted_chunk):
                target_step = step_idx + act_t
                if target_step < max_steps:
                    if target_step not in actions_queue:
                        actions_queue[target_step] = []
                    actions_queue[target_step].append(act)
            
            action = torch.stack(actions_queue[step_idx], dim=0).mean(dim=0)
            # Clean up used actions
            del actions_queue[step_idx]
            
            # Step environment
            # action = gt_action_chunk[0]  
            # action = predicted_chunk[0]
            # action = torch.cat([action[:3], torch.zeros(3, device=device), action[3:]], dim=0)
            print("action:", action[-1])
            if action[-1] > 0.2:
                action[-1] = -.01  # close
                prev_action = action[-1]
            elif action[-1] < 0.5: 
                action[-1] = 1.0  # open
                prev_action = action[-1]
            else: action[-1] = prev_action  # maintain
            # if step_idx < 40: action[-1] = 0.0  # keep open for first 20 steps
            if step_idx < int(.15*max_steps): 
                obs, reward, terminated, truncated, info = env.step(gt_action_chunk[0].unsqueeze(0))
            else: 
                obs, reward, terminated, truncated, info = env.step(action.unsqueeze(0))
            image = get_image(env)
            object_poses = get_object_poses_from_env(env)
            
            # Get new observation
            # if state_replay: current_obs = gt_states[step_idx]
            # print(get_observation_from_env(env, episode_ends, latent_variable))
            # else: current_obs = get_observation_from_env(env)
            
            
            current_obs = get_observation_from_env(env, episode_ends, latent_variable)
            # print("obs:", current_obs[6])
            # if step_idx < 20: current_obs[6] = 0.0
            # # current_obs[6] = 0.0
            # current_obs = torch.cat([current_obs, gt_states[0][-3:]], dim=-1) 
            # # current_obs = torch.cat([current_obs], dim=-1) 
            # obs_history.append(current_obs)


            # next_idx = min(step_idx + 1, gt_states.shape[0] - 1)
            # current_obs = gt_states[next_idx]            # shape == cfg.state_len used in training
            # import pdb; pdb.set_trace()
            obs_history.append(current_obs)

            # Record trajectory
            if save_trajectory:
                trajectory_data['observations'].append(current_obs.cpu().numpy())
                trajectory_data['images'].append(image.cpu().numpy())
                trajectory_data['actions'].append(action.cpu().numpy())
                trajectory_data['object_poses'].append(object_poses)
                trajectory_data['eef_poses'].append({
                    'pos': current_obs[:3].cpu().numpy(),
                    'rpy': current_obs[3:6].cpu().numpy()
                })
    
    print(f"✅ Trajectory completed: {len(trajectory_data['actions']) if save_trajectory else step_idx} steps")
    return trajectory_data, gt_robot_data, human_data if save_trajectory else None


def save_comparison_video(pred_data, gt_robot_data, human_data, episode_idx, output_dir, camera_data=None):
    output_dir = Path(output_dir) / task_name / epoch_num
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"comparison_{episode_idx:03d}.mp4"
    
    camera_data = pred_data['camera_params']

    # Unpack data
    pred_imgs, pred_eef, pred_objs = pred_data['images'], pred_data['eef_poses'], pred_data['object_poses']
    gt_imgs, gt_eef, gt_objs = gt_robot_data['images'], gt_robot_data['eef_poses'], gt_robot_data['object_poses']
    human_imgs, human_eef, human_objs = human_data['images'], human_data['eef_poses'], human_data['object_poses']
    
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
    
    if gt_imgs.shape[0] < n_frames:
        gt_pad = n_frames - gt_imgs.shape[0]
        last_gt = gt_imgs[-1:,...]
        gt_imgs = np.concatenate([gt_imgs, np.repeat(last_gt, gt_pad, axis=0)], axis=0)

    if human_imgs.shape[0] < n_frames:
        human_pad = n_frames - human_imgs.shape[0]
        last_human = human_imgs[-1:,...]
        human_imgs = np.concatenate([human_imgs, np.repeat(last_human, human_pad, axis=0)], axis=0)
    
    if camera_data:
        img_shape = pred_imgs[0].shape
        pred_eef_proj, pred_obj_proj = project_all(pred_eef, pred_objs, camera_data, img_shape)
        gt_eef_proj, gt_obj_proj = project_all(gt_eef, gt_objs, camera_data, img_shape)
        human_eef_proj, human_obj_proj = project_all(human_eef, human_objs, camera_data, img_shape)

    # Pad pred_eef_proj, pred_obj_proj, gt_eef_proj, gt_obj_proj, human_eef_proj, human_obj_proj if needed
    if len(pred_eef_proj) < n_frames:
        pad_len = n_frames - len(pred_eef_proj)
        last_pred_eef = pred_eef_proj[-1]
        pred_eef_proj += [last_pred_eef] * pad_len
        for name in pred_obj_proj:
            last_pred_obj = pred_obj_proj[name][-1]
            pred_obj_proj[name] += [last_pred_obj] * pad_len

    if len(gt_eef_proj) < n_frames:
        pad_len = n_frames - len(gt_eef_proj)
        last_gt_eef = gt_eef_proj[-1]
        gt_eef_proj += [last_gt_eef] * pad_len
        for name in gt_obj_proj:
            last_gt_obj = gt_obj_proj[name][-1]
            gt_obj_proj[name] += [last_gt_obj] * pad_len

    if len(human_eef_proj) < n_frames:
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
        for img_data, eef_proj, obj_proj, color in [
            (pred_imgs[i], pred_eef_proj , pred_obj_proj, (0, 255, 0)),
            (gt_imgs[i], gt_eef_proj, gt_obj_proj , (255, 165, 0)),
            (human_imgs[i], human_eef_proj, human_obj_proj , (0, 165, 255))
        ]:

            if isinstance(img_data, (bytes, bytearray)):
                img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)[:,:,::-1]
            else:
                img = np.array(img_data, dtype=np.uint8)
            img = img[..., ::].copy()
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            if camera_data and eef_proj:
                img_bgr = draw_trajectory_on_frame(img_bgr, eef_proj, i)
                for proj_list in obj_proj.values():
                    img_bgr = draw_trajectory_on_frame(img_bgr, proj_list, i)
            
            imgs.append(img_bgr)
        
        # Add labels
        for idx, (img, label) in enumerate(zip(imgs, ['Predictions', 'Ground Truth', 'Human Actions'])):
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

    # episode_indices = np.array([i for i in range(30)] + [i for i in range(61, 90)])
    episode_indices = np.array([i for i in range(total_episodes)] )
    # import pdb; pdb.set_trace()
    np.random.shuffle(episode_indices)
    for idx, episode_idx in enumerate(episode_indices):
        if episode_idx == 23: continue
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
                save_trajectory=args_cli.save_trajectories
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
    
    env.close()
    print("Inference finished")


if __name__ == "__main__":
    main()
    simulation_app.close()