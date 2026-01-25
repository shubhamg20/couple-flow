import sys
sys.path.append("/workspace/isaaclab/source/droid/droid/controllers/")
sys.path.append('/home/shubham/summer/serl-flow/conditional-flow-matching')

import argparse
from pathlib import Path
from collections import deque
import torch
import numpy as np
from time import sleep
from scipy.spatial.transform import Rotation as R
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import cv2
# Flow matching imports
from flow_policy.configs import FlowMatchingModelRunConfig
from flow_policy.make_networks import instantiate_flow_matching_artifacts
from flow_policy.dataset import IsaacLabDataset, unnormalize_data
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Run flow matching analysis.")
parser.add_argument("--output_dir", type=str, default="source/serl-flow/outputs/", help="Output directory for saved plots")
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
task_name = "dsrl-flow-mixed"
epoch_num = "600"
#🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖
#🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖

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
        torch.tensor(eef_rpy, dtype=torch.float32, device=env.device),
        gripper_state,
        object_positions[:]
        # object_positions
    ], dim=0)
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

def run_flow_matching_batch(current_obs, nets, cfg, device, interpolation_points_batch, obj_init_positions):
    """
    Run flow matching inference using real environment observations for conditioning on a batch of latent points.
    """
    print(f"Starting batched flow matching inference with {len(interpolation_points_batch)} points...")
    
    # Get config parameters
    pred_horizon = cfg.pred_horizon
    obs_horizon = cfg.obs_horizon
    batch_size = len(interpolation_points_batch)
    
    print("🚀 Starting batched flow matching inference...")
    nets.eval()
    
    with torch.no_grad():
        # Create observation history
        obs_history = deque(maxlen=obs_horizon)
        for _ in range(obs_horizon):
            obs_history.append(current_obs)
        
        # Stack observations for conditioning and expand for batch
        obs_stack = torch.stack(list(obs_history), dim=0)  # [obs_horizon, state_dim]
        obs_cond = obs_stack.flatten().unsqueeze(0)  # [1, obs_horizon * state_dim]
        obs_cond_batch = obs_cond.repeat(batch_size, 1)  # [batch_size, obs_horizon * state_dim]
        
        # Stack all interpolation points into a batch
        x_batch = torch.stack(interpolation_points_batch, dim=0)  # [batch_size, pred_horizon, action_dim]

        # Flow matching inference - flow from t=0 (noise) to t=1 (data)
        num_steps = 100
        dt = 1.0 / num_steps
        batch_flow_x_values = []  # Store x values at every timestep for all points
        
        # Store initial random points (t=0)
        print(f"  Flow step 0 (t=0.000): Initial noise distribution")
        batch_flow_x_values.append(x_batch.clone().detach().cpu().numpy())
        
        for fm_step in range(num_steps):
            # Current time for velocity field evaluation
            t = torch.tensor(fm_step * dt, device=device)  # t ∈ [0, 1-dt]
            t_batch = t.unsqueeze(0).repeat(batch_size)  # [batch_size]
            
            # Evaluate velocity field v_θ(x_t, t, context) at current state and time
            vt_batch = nets['flow_net'](x_batch, t_batch, global_cond=obs_cond_batch)
            
            # Forward Euler integration: x_{t+dt} = x_t + dt * v_θ(x_t, t, context)
            x_new = x_batch + dt * vt_batch
            
            # Calculate change magnitude for monitoring convergence
            if fm_step >= 0:
                change_norm = torch.mean(torch.norm(x_new - x_batch, dim=-1)).item()
                velocity_norm = torch.mean(torch.norm(vt_batch, dim=-1)).item()
                print(f"  Flow step {fm_step+1} (t={t.item():.3f}→{(t.item() + dt):.3f}): Change norm = {change_norm:.6f}, Velocity norm = {velocity_norm:.6f}")
            
            # Update state
            x_batch = x_new
            
            # Store evolved state
            batch_flow_x_values.append(x_batch.clone().detach().cpu().numpy())
        
        # Convert batch results to individual trajectory data
        trajectories_data = []
        for i in range(batch_size):
            # Extract trajectory for point i across all flow timesteps
            trajectory_flow_evolution = []
            for step_data in batch_flow_x_values:
                # step_data shape: [batch_size, pred_horizon, action_dim]
                # Extract the i-th point's trajectory at this flow timestep
                point_trajectory_at_timestep = step_data[i:i+1]  # [1, pred_horizon, action_dim]
                trajectory_flow_evolution.append(point_trajectory_at_timestep)
            
            trajectory_data = {
                'flow_x_values': trajectory_flow_evolution  # List of [1, pred_horizon, action_dim] for each flow step
            }
            trajectories_data.append(trajectory_data)
        
        # Debug: Print shapes to verify structure
        print(f"📊 Trajectory data structure:")
        print(f"  - Number of points: {len(trajectories_data)}")
        print(f"  - Flow steps per point: {len(trajectories_data[0]['flow_x_values'])}")
        print(f"  - Shape at each flow step: {trajectories_data[0]['flow_x_values'][0].shape}")
        print(f"  - Expected: [1, {pred_horizon}, {x_batch.shape[2]}]")
        print(f"  - Initial point shape: {x_batch[0].shape}")
        print(f"  - Final point shape: {x_batch[-1].shape}")
        print(f"  - Number of flow evolution steps stored: {len(batch_flow_x_values)}")
        print(f"  - Flow step indices available: 0 to {len(batch_flow_x_values)-1}")
        print(f"  - Time progression: t=0.000 (step 0) → t=0.980 (step 49) → t=1.000 (step 50)")
        
        # Calculate MSE and L1 (MAE) of each timestep compared to final timestep
        print(f"\n📊 Flow Evolution Analysis (MSE & L1 vs Final State):")
        print(f"{'Step':<4} {'Time':<8} {'MSE':<12} {'L1 (MAE)':<12} {'Distance to Final'}")
        print("-" * 60)
        
        final_state = batch_flow_x_values[-1]  # Final state at t=1.0
        for step_idx, step_data in enumerate(batch_flow_x_values):
            # Calculate time for this step
            if step_idx == 0:
                t_value = 0.000
            else:
                t_value = step_idx * dt  # Corrected time calculation
            
            # Calculate MSE and L1 between current step and final step
            diff = step_data - final_state
            mse = np.mean(diff ** 2)
            l1_mae = np.mean(np.abs(diff))
            
            # Calculate average distance per point
            distance_per_point = np.mean(np.sqrt(np.sum(diff ** 2, axis=(1, 2))))
            
            print(f"{step_idx:<4} {t_value:<8.3f} {mse:<12.6f} {l1_mae:<12.6f} {distance_per_point:<12.6f}")
        
        print("-" * 60)
    
    print(f"✅ Batched flow matching completed for {batch_size} points")
    return trajectories_data

def main():
    """Main function to run flow matching analysis."""
    
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
    
    print(f"\n{'='*60}")
    print(f"FLOW MATCHING ANALYSIS")
    print(f"Model: {args_cli.checkpoint}")
    print(f"{'='*60}\n")
    
    # Run flow matching experiments
    pred_horizon = cfg.pred_horizon
    action_dim = 7
    
    # Number of experiments to run
    num_experiments = 1
    for exp_idx in range(num_experiments):
        if not simulation_app.is_running():
            break

        print(f"\n{'='*60}")
        print(f"Flow Matching Analysis Experiment {exp_idx + 1}/{num_experiments}")
        print(f"Running flow matching on latent points")
        print(f"{'='*60}\n")

        try:
            # Use pre-generated random Gaussian latent points (these should be properly normalized noise)
            # Note: These points represent the starting noise distribution for flow matching
            random_points =   np.array([
    [[-0.38989195, 1.6479269, -1.6868241, -0.502044, -1.5895913, -0.34948483, 0.75461257],
     [-0.01024734, -1.8037754, -0.55633265, 0.6061539, -1.5756062, 1.0838592, -0.4077198],
     [0.04127409, -0.84073406, 1.3016742, 0.865241, -0.35286602, 0.5389369, 0.51996505],
     [-1.3392797, -0.13638057, 0.01511844, 0.0478447, -0.7916915, 0.30266392, 0.5290396],
     [-1.0951478, -0.51791626, -0.27885047, 1.8983481, -1.1587304, -0.9910183, 0.07613713],
     [-0.49580452, -0.14624809, 0.48577178, -0.99695575, 1.3671216, -0.50971454, -0.0327965],
     [1.303463, -0.7170623, -0.00787204, 0.44835484, -1.621658, 0.66963494, -0.9567803],
     [0.42077366, 1.5640441, -1.4628655, 0.47046477, 0.56831044, 0.2940973, -0.4324008]],
    
    [[-0.45508856, -1.1004981, 1.847526, 0.80526173, -0.14776112, -1.357065, 0.15640739],
     [-0.16309267, -0.30243716, -0.99835277, 0.6352795, 0.6761432, -2.010882, 1.0847526],
     [-0.3707075, -1.3903587, -0.22217363, -0.03258094, -0.9052818, -2.4312527, 0.88143456],
     [0.23166703, -1.40515, 0.8313639, -1.4677736, -0.03812024, -0.6324929, 1.4161838],
     [0.4931744, -0.31130695, -1.504412, -0.8988006, -1.369368, 0.48702663, 1.6333463],
     [1.5694609, -0.23810664, 0.9484445, -1.1636238, -0.15510392, -0.02546852, -0.7556621],
     [-0.29380044, -0.08989525, 1.2555647, 2.4673188, -0.44326657, 0.5828927, -0.8975973],
     [0.3958779, -0.71701705, 0.04836348, -0.19791824, 1.5071697, -1.4407305, 0.63348716]],
    
    [[-1.8808233, -0.64723575, 1.4860373, -0.23163794, -1.3638538, -1.5871761, 0.15930712],
     [0.859588, -0.21818307, 0.33159575, -0.03633296, 1.0486026, -3.3565235, -0.80031306],
     [0.5819937, 2.4868963, -0.4937778, -1.5051459, 1.1941328, -0.08546375, -0.13059935],
     [0.48953572, -1.3023788, 0.19789124, 1.2592572, -1.4537678, -0.15621026, -0.70110095],
     [-0.32643124, 0.85815036, 0.04801697, 0.39749336, 0.4463478, -0.05732839, 0.40837625],
     [1.002691, 0.173136, 1.4787315, -0.41826636, -0.16249862, -1.4579588, -0.04978189],
     [0.55439293, 1.6549749, -1.3090428, -1.6886866, 1.3760827, 0.4080722, 0.74297804],
     [0.29764205, -2.7746048, -1.2112926, 1.5920695, 2.6142538, 0.30346918, 1.8249276]],
    
    [[0.16955467, 1.6159602, -0.92111856, -2.0739303, 1.3564525, -0.92179537, 0.8654115],
     [1.2513622, 1.6443174, 0.38060793, 0.41939163, -0.9254255, 0.90805775, 0.7287451],
     [1.2946092, 0.89381725, 0.07911532, 0.4596975, -0.8364252, 1.1563183, 0.94063807],
     [0.6337829, 0.12220865, 1.4280154, -1.719535, -0.8909342, 0.15453967, -0.64482224],
     [-0.11524872, 1.0075535, 0.52634686, 0.04834121, -0.16195287, -0.55126196, -0.78183734],
     [1.121785, -0.21270443, 0.17430495, 2.0770173, 0.40018216, -0.34209102, 0.73173016],
     [0.4415913, 0.9595494, 2.5370479, 0.02700619, -1.0099987, 0.16339244, 0.9369535],
     [-0.0272971, 0.64755845, -1.1088401, -0.4795768, 0.9654328, 0.5804454, 0.12125956]],
    
    [[-0.4061372, 0.12119312, 0.78922236, -0.5836173, 0.55328447, -0.7780439, 1.5886112],
     [-0.3861799, 1.9793918, -0.6382304, -1.7306246, 0.9384983, 0.5935917, 0.8269965],
     [1.7209566, -1.3997941, 0.19372647, -0.15472315, 0.63544494, 1.2941552, 0.7985049],
     [-0.29336232, -0.15018414, -2.9894397, -0.96950614, -0.5566474, 0.7967227, -0.96747863],
     [-0.55564123, -0.81108034, -0.06730697, -0.24019866, -0.7704428, 1.8154451, -0.18028727],
     [0.9329026, 2.6350594, -0.71909183, -0.45489722, -0.27142203, -1.5485009, 0.5040253],
     [2.0182385, -0.6663202, -0.24780622, -0.49720022, 0.1831143, 0.38380557, 0.81138253],
     [-1.4450035, 0.02150679, -0.41389585, -0.8321159, 2.49778, 0.0683844, -0.936412]],
    
    [[-1.2701216, 0.22125599, 1.0851628, 0.9165417, 0.52231276, 0.29555959, 0.11083109],
     [-2.0700603, 0.35148564, 2.0577884, 0.5920369, 0.95827985, -0.08598845, -3.0692143],
     [-1.1883416, -0.3110147, -1.0800858, 0.2920892, -1.2583342, 0.39698225, -1.4649165],
     [-1.0272888, -0.83294666, -1.0472919, -0.42225212, 0.394178, -1.3993984, 0.88369894],
     [-0.07726429, 0.04167615, -0.30808333, 0.97631633, 0.22025946, 0.48326695, -1.3983828],
     [-0.36360613, -2.1278872, 0.50515056, 0.5381458, 0.67222494, -0.19896254, 1.2463855],
     [-1.6149436, 0.8019104, 0.0387237, -0.55268127, -0.6146168, -0.29096407, -1.3429735],
     [-0.05649597, -0.5638777, -0.3489486, -0.17447579, 1.16928, 1.7419766, -0.58783793]],
    
    [[1.2375692, -0.40315378, -1.3041376, 0.83019245, -0.17682439, 0.90503234, 1.4790137],
     [1.5790865, -1.1522766, 0.5270658, 0.22853924, -1.2804406, -0.20325825, 0.15154696],
     [0.53002495, 0.6794482, 2.6243098, -0.07362197, -0.19608082, -0.21178365, 0.7744898],
     [0.31041545, 1.3323419, -0.5297368, 0.60383266, -0.83045065, 1.33176, -0.48324484],
     [-1.0495691, 1.510399, 0.0255225, -0.89789546, -0.4261087, 1.4299806, 0.3393856],
     [2.9224334, 0.4385166, -0.5296013, 0.36516762, -1.374317, -1.2389214, 1.5622859],
     [-0.18973827, -0.57592124, 0.1314895, 0.26817954, -0.60686105, 0.6245452, 0.6836711],
     [0.46957007, 0.62289894, 1.1454498, 1.3535521, 1.2768515, 0.6681172, -0.65829736]],
    
    [[-0.1048833, 0.11347526, 0.7827899, 0.5217133, 0.31088495, -0.44367492, 1.6780629],
     [0.09281226, -1.9644439, -0.4798368, -0.8453653, -0.30712685, -1.2939347, 1.2754917],
     [0.23722118, -1.0971779, 0.16549495, -1.6104956, -0.05786026, -0.22307271, -0.20445816],
     [1.5258309, -1.0146605, 0.81507504, -2.336665, -1.9318514, 0.50568783, -0.43996033],
     [-0.55925393, 1.3770815, -0.5399181, -0.31850913, -1.3009061, -1.1821994, 0.6189266],
     [-0.8452378, -0.7535471, -0.48659822, 0.04413769, -0.60423934, -0.40931824, 0.18272376],
     [-1.0644214, 1.6181595, 0.5617999, 0.33103657, 2.3449616, -0.0168937, 0.29539675],
     [1.3151188, 0.04311512, 1.3772398, -0.40973577, 0.8228879, 0.41251144, -0.6897412]],
    
    [[-0.3447572, -0.04314372, 1.645654, 1.4038713, -2.0273666, 0.6137872, 1.8600782],
     [-1.5057384, 0.56408244, 1.1959848, 1.3019769, -0.78641415, 0.01989574, -0.23373713],
     [0.1541417, -1.5547397, -1.1710831, 0.12636024, -1.3431371, 1.6771253, -0.8412716],
     [-0.5520989, 2.244219, -0.70962036, -0.5381501, -1.1245553, -1.2631692, -2.012845],
     [0.5628162, -0.9421988, 0.38174272, 0.18808797, 0.7288989, -0.7598112, -0.3804104],
     [0.11076392, -0.8535618, -1.3044629, -1.1691165, -0.11513985, -0.42270496, 0.08970697],
     [-0.9653187, 0.08914414, -1.2742423, 0.51808023, 0.8606996, 0.84915257, -1.0107839],
     [-1.2041391, 1.4170868, 0.03594091, 0.21726866, -0.16431452, -0.41090125, 0.9108836]],
    
    [[0.25015277, -0.8722029, 0.28920814, -0.34332743, 0.6434769, 2.876616, -2.399836],
     [-0.94262105, -0.93441725, -0.92371887, -0.03373795, 0.5553146, -0.9696391, 0.798426],
     [1.2250375, -0.06308047, 0.5844362, -1.3240587, -0.20795473, -1.3510863, -0.4661377],
     [-0.7331246, -0.29771364, -0.22473156, 0.9163365, -0.8287796, -0.5013046, 0.21586034],
     [0.42393333, 0.6322504, -0.5967407, 0.4356903, -0.65660644, -0.75593495, -0.6700781],
     [-0.44554564, -0.45378312, -0.57114637, -0.6709412, -0.34854487, -0.6691836, -0.06332001],
     [0.61795753, -2.4656544, -0.55175614, 0.22545388, 0.21434446, 0.44501993, 1.2420382],
     [-0.2710099, 2.7324758, 0.31653696, -1.8896148, -0.42900348, -0.9581584, -0.3086231]],
    
    [[-0.6106842, -0.28294855, -1.60226, 0.02406977, 0.37098554, -0.41424975, -0.04549919],
     [-1.2195524, 0.8417409, 0.85132, 0.45923978, -0.74567115, -0.42516723, -1.7176069],
     [-0.4152461, 0.5109306, -0.55495554, 0.42266613, 1.2563238, 0.5804504, 0.9517849],
     [0.7188063, 2.2804332, 1.452365, 0.61003125, 0.14419647, 1.5900413, 0.03388804],
     [-2.0824044, 0.32838035, 0.40290743, 0.18840756, -0.03386361, -0.09263327, -0.06172727],
     [-1.1910706, 0.23769104, -1.1995114, 0.93960255, 0.785096, -2.2604914, 0.5474311],
     [-0.5818174, 1.8720226, -0.7462871, -0.8376934, -0.57613206, -0.09134845, 0.24387749],
     [0.69388735, -0.64370143, 1.551254, -0.11413109, -1.4799428, -1.0037228, -0.37002495]],
    
    [[0.5292349, 0.46446893, 1.8334911, -0.4671773, -0.8340525, -0.5943708, -0.28690886],
     [0.33286572, 0.01458689, 0.7288847, 1.1875386, 1.4715594, -0.5515223, 0.10461055],
     [1.5672683, 0.87914383, 0.9451967, 0.45369723, -0.22116674, -0.85101825, -0.606748],
     [-1.098828, -1.6736739, 0.8517996, 0.9332118, 0.49171373, -0.01777928, -1.0989718],
     [1.550437, -0.5890567, -0.63592726, -1.466971, 1.6952358, -2.1558316, -0.7330455],
     [0.232538, 1.8914583, 0.03849778, 0.19469735, 2.1779053, -1.5698267, -0.01312411],
     [-0.44935015, 0.36857757, -1.4530995, -0.26047045, 1.9135906, -0.17359939, 1.680769],
     [0.7173408, -1.7968314, -1.6384946, 0.22321177, -0.85403705, -0.48569694, -0.96604544]],
    
    [[-0.5383036, -1.2181481, 0.43942958, -1.6217642, 1.1609423, 0.9128398, -0.6143461],
     [0.23498246, -1.2830197, -0.2673631, 0.9614187, -0.6062083, -0.07630752, -1.0762516],
     [-0.5557005, 0.71114504, -0.98968494, -0.30177838, -0.6771689, -1.0834268, 0.7782952],
     [-2.4021847, 2.286812, -2.4619777, -0.7622783, 0.54112494, 0.06306198, -0.41746524],
     [-1.8251443, -0.8505423, -0.93556994, -1.1225209, 1.0838546, 1.6359063, -0.9432894],
     [-0.30703166, 0.5190709, -0.67853814, 0.22674443, -1.7195942, 1.242346, 0.00501916],
     [-0.11952709, 1.7049408, -1.698779, 0.5838112, -0.93911916, 0.908586, -0.7696324],
     [0.01215498, 0.6433142, -1.5250183, 0.48079076, 1.4110613, -1.2748215, 0.61217225]],
    
    [[-1.0807925, -0.43534496, -0.08897926, -0.32082382, -1.544414, 1.4963949, -0.13146159],
     [0.05900786, -0.25531465, 1.3622122, 0.06191755, -0.6176819, 1.1194632, 0.681271],
     [-1.9202822, -0.4883137, -1.2763888, -0.7997834, -0.24224102, -0.03934572, 0.37515953],
     [0.43664876, -0.2316653, 0.6945459, 1.112142, -0.4098489, -1.7564044, -0.05826207],
     [-1.1464971, -2.2257211, 2.0553615, -0.6931524, 1.526583, -0.28844503, -0.55779177],
     [0.2003354, 0.58538073, -2.4910686, 0.8724514, -2.5951664, 0.23172686, -1.7385006],
     [0.24632421, 1.5388603, 1.34111, -2.56013, 0.62269723, 2.643468, -1.3333707],
     [-0.02068938, -2.0053313, 1.0654128, -0.9857344, -0.6202292, 0.3360807, -1.0270989]],
    
    [[0.5558496, 1.4629824, -1.0405848, -0.38122696, 0.9446139, -0.12478203, 0.19951583],
     [-0.29489094, -0.2556953, -0.48063833, 1.9712552, 0.4283306, -0.12325668, -0.9437151],
     [-0.32003596, 0.72968096, -0.4887593, 1.0834821, 0.7607028, 0.5800177, -0.06016428],
     [0.555804, -1.529293, 0.04235904, 1.7858125, -0.17812562, -0.11530662, -0.70229787],
     [0.637077, 0.29281095, 0.35804588, 1.2473958, 1.5875527, 0.5713379, 0.4176134],
     [1.4690112, 0.48315066, 0.2525885, 0.2669747, 2.4502602, -1.8302035, -0.29263443],
     [0.90592796, 1.9429206, 0.5866877, -0.00681973, 0.7297934, -0.49149758, -1.2240387],
     [0.4477439, -0.7270145, 1.987223, -0.19008785, 0.16857043, -0.5148482, 1.5435493]],
    
    [[0.38939315, -0.00515546, 0.03699518, -0.8275869, -0.48677576, -0.11831954, 1.7251931],
     [1.6626531, 1.1392639, 0.14674467, -0.6495118, 0.01695779, -0.38217637, -0.17453371],
     [0.2439968, -0.76544976, -0.80289775, 0.14517449, 1.3819933, -0.7003789, 0.90764654],
     [1.6841573, -1.1500441, -1.0364236, 0.4223386, -0.00119748, 0.85031104, -0.47334537],
     [1.3159367, -0.8288161, -1.026375, 1.0718883, -0.66993856, 0.85025716, 0.22493158],
     [-0.20233853, 2.2397919, 0.1804943, 0.9119837, 0.14102054, 0.25930712, 0.5532141],
     [1.3118324, -0.956649, 0.6247271, -0.59422266, -0.20732905, 0.0453704, 0.9788129],
     [0.3746638, 0.7969622, -0.43771443, -1.1745095, 0.5736086, -0.01376265, -0.10891255]],
    
    [[-1.8323033, 0.00095476, 0.8176776, -0.00951422, 0.19259745, 1.7011656, -1.6011977],
     [-1.3736846, -0.05270012, 1.6349925, 0.6622166, -1.4344382, 0.8170847, -0.6837616],
     [-0.2021913, -0.60035783, 1.3238969, 0.23896773, -1.4861882, -1.9151381, 1.7508376],
     [1.3792002, 1.7790517, -1.9962976, 0.7549729, 1.8795842, 0.6008856, 0.60794365],
     [0.7581181, 0.14763072, -0.44215328, -0.94543684, -1.0400761, -1.354817, 0.06937849],
     [-0.2934103, 1.1159292, 1.3108478, -0.5243711, -0.9115515, -1.2642976, 0.7937618],
     [-0.77711695, 1.1768767, 0.74704456, 0.4878879, 0.14493002, 0.43941158, -0.69444126],
     [-0.9904829, -0.11627171, 0.7585781, -0.23919886, 1.9953986, -2.4211223, -0.5634401]],
    
    [[0.7000012, 0.42498913, -0.78991777, -0.8630677, -1.3453088, 0.24378765, -0.87042725],
     [-0.18795161, -0.328393, 0.7319952, 0.5423781, -1.4702685, 0.06889936, 0.53199875],
     [-1.0687748, -0.69720435, -1.0369712, 1.2287947, 0.69410664, -0.7052043, -0.35658932],
     [0.5737607, 1.2534512, -0.00825975, -0.04469847, 1.0476234, 0.54459, 0.5843612],
     [0.7784415, 0.96520674, 0.2977154, -0.27148575, -1.3247252, 0.37598827, 0.02054468],
     [0.6359961, -1.0087374, -0.6193635, 1.1095779, -0.61729586, -0.39257595, -0.6009164],
     [-0.530797, 0.1680676, 0.5644732, -3.214906, 0.5928093, 0.22202888, -1.1487986],
     [1.3273584, -1.435713, -1.613482, -0.74195254, -0.9102963, -0.51997924, 1.6943303]],
    
    [[0.32036367, 1.2597709, 0.7599852, 0.6917397, 0.19774504, -0.921017, -0.07286461],
     [1.6493709, 0.4161932, 1.9573574, -0.86331004, -0.0842692, -1.3146999, -0.67968917],
     [-0.398944, 1.1482074, -0.75652665, 0.7707247, 0.26195627, 0.35625157, 1.3570129],
     [-0.8179209, 0.74515176, -0.5367911, 0.46404308, 2.029531, -0.1392686, 0.75475127],
     [0.52806306, -0.71559846, -2.1363323, 0.7593254, 1.4661987, 0.2561413, -0.4415438],
     [1.085868, 1.2048948, 0.7968907, 1.5893614, 2.0814333, -2.2048466, 0.8616518],
     [1.3139585, 0.00597848, 0.593699, -0.44876486, -1.2910465, -1.7507269, 1.0824181],
     [-2.1522453, -0.6636486, 0.2722655, 0.6268792, -0.32347384, 1.6317489, 0.88937825]],
    
    [[0.65557784, -1.1949991, 1.1980816, -0.1034473, -1.495558, 0.68465674, -0.2172584],
     [-1.0098842, -0.04585415, 0.75231016, -0.05955217, 0.75348264, -0.37102, 1.287178],
     [-0.46614575, -1.717126, -1.4938262, 1.1750311, -0.39112666, -1.216449, 0.40856025],
     [1.2925428, 0.5500327, -0.09132461, -0.8606038, -0.5362373, 0.14619401, -2.5506945],
     [-0.82825947, 0.37084955, 1.0120975, -1.2994416, -0.48233178, 1.0383122, -1.0067374],
     [-0.7700197, -0.60825205, -0.82682854, -0.57228774, -1.2673756, -0.44849196, -0.91594124],
     [-0.40958175, 0.45663318, -1.0903181, -0.52847886, -1.6183631, 1.3184518, 0.7486764],
     [-1.0524719, -0.7424011, -0.90244734, 0.6409749, -1.132257, -0.8238338, -0.45362303]],
    
    [[1.1072688, 0.45329273, -0.89815044, -0.17191935, -1.1106447, -2.0521798, 0.871724],
     [-0.7198311, 1.374331, -0.22083929, 0.7422848, 0.30828437, 0.72219247, 0.6012866],
     [-0.90044457, 1.4401222, 0.5015449, -0.6199627, -1.3021199, 0.8592217, 0.45369968],
     [-0.5178422, 0.01721301, -0.95104194, -0.07160642, -0.62858117, 2.0045865, 1.2772427],
     [0.00694969, -0.12617114, -0.8123417, -1.2110318, -0.6032762, -1.6407814, 1.1768185],
     [0.22541516, 0.7397482, 0.5830305, -0.24244754, -1.1730375, -1.2506099, 0.47680438],
     [-1.3956234, -0.76684994, -0.43416852, 1.4180182, -0.07677416, -0.2428641, -0.11359168],
     [0.383234, -0.8642209, -0.9425437, -1.3412286, -0.6989347, -0.01301055, 0.6128642]],
    
    [[-0.62141603, 1.1828485, 0.985608, -0.5094219, 0.5729497, -0.26257864, 0.12860464],
     [-0.36026186, 0.6522161, -1.6890188, 0.81668115, -0.1801518, 0.04401552, 0.39882678],
     [-0.8399096, -1.5898036, 0.46654317, 1.5629288, 0.05808326, 0.8291455, -1.1419594],
     [1.1655104, 1.3923498, -1.463618, -0.5075199, -1.0207086, -0.97817665, -1.0257218],
     [0.2766025, -0.2004369, 0.10230491, 1.2204338, -0.04180702, 0.00267614, 0.64350826],
     [-0.17321976, 1.1776195, -0.40583912, -0.49729967, -1.0876151, -1.2934364, 0.5275144],
     [0.7049191, -0.4747732, -0.24704523, 1.2484525, -0.9186105, 0.85843253, -1.2259158],
     [-0.2160563, -0.78383726, -1.7245463, -0.3696409, -0.0541846, 0.36971474, -3.1758687]],
    
    [[0.6776681, 2.2343206, 0.68046373, 0.48306897, -1.3741995, -0.36270317, -0.3707571],
     [0.93257475, 0.8618964, 2.0132341, 0.47478446, 0.7453465, -3.21848, -0.41332263],
     [-0.01983665, -1.2199789, -1.0128732, 0.5538294, -0.08312495, 1.3258425, -1.8108329],
     [-1.7217551, 1.052843, -0.40144598, -0.18290858, 0.09337135, 0.56018937, -1.0194374],
     [0.24656832, -0.01822112, -0.03986784, -0.06193018, 1.0858461, 1.9409835, -0.23570661],
     [-1.2610487, 0.17418735, 1.4599768, 1.1742295, -0.06367689, 0.10172413, 0.03206641],
     [-0.3938347, 0.07263602, -1.0609095, -1.0890262, 1.1238538, -0.8436569, 0.59657574],
     [0.3151601, -1.8314595, 1.1831901, -0.5842025, -0.41292238, 0.32563868, -0.12766607]],
    
    [[1.291763, -0.21189067, -0.29515684, -0.22828268, 0.8546981, 0.4123078, -2.447037],
     [1.7993801, 0.5933927, -0.4632008, -0.860502, -1.1162013, 1.2005409, 0.8498304],
     [0.17450483, 0.59954196, -0.70376706, -1.4911886, 1.379094, 0.6834089, 0.3307764],
     [-0.24789526, 0.4262003, -0.48157004, 0.20411675, 0.10684918, 1.8547541, -0.5810634],
     [-0.95434225, 1.686636, 1.2027384, -0.8726799, 2.0415676, -0.11804277, 0.6406002],
     [-1.3176706, 1.6718439, -1.173361, -0.9236702, -0.5769516, -1.312375, 0.857841],
     [1.959868, -0.32198206, 1.2816248, -0.00570939, 0.76157963, -1.1243399, 1.2399303],
     [-0.5256073, -0.04363927, 0.4882556, -0.24521348, 1.085797, -1.8228104, 0.5725124]],
    
    [[-2.59474, -0.00181921, -0.37733456, 1.0998394, 1.589816, -0.92852134, 0.01063603],
     [-1.2029787, -1.0228024, -0.81819046, 0.8577473, 1.6567616, -0.90626574, -0.48193648],
     [-0.81387633, 0.24653342, 1.1152828, 1.147017, 0.7349518, 0.5399107, -0.8924435],
     [0.04815877, 1.3793545, -0.2213813, 1.7307563, 2.3363671, -0.5552263, -0.40480584],
     [0.7700645, 1.2104098, 0.97400683, 0.33235657, -1.3666846, -0.54420036, 1.9878403],
     [-0.12014081, 0.03926533, -1.5588958, -0.8802755, 0.8492624, 1.3105544, -0.72856396],
     [-0.9987547, 0.15701765, -0.6571221, 1.6755452, 0.34042102, -0.14126886, 0.15903114],
     [-0.1458615, 1.2558482, -1.0323582, -1.3888725, -1.0241685, -0.23424084, 0.500094]],
    
    [[-0.30163768, -0.1849424, 0.60753095, -0.71644557, 0.67822653, -0.14625926, -1.1640714],
     [-0.05253145, -0.6017475, -1.4522809, 1.2507204, -0.84382796, -0.65389717, 0.51715493],
     [0.37199852, -0.4547498, -0.72266865, -1.1521065, 0.6545937, 1.7323487, -0.70542186],
     [0.19164948, -0.6099772, -0.10928078, -0.28618968, -0.21820188, 1.6237596, -1.475583],
     [0.25106952, -0.22335806, -0.12261474, 0.9358752, 2.2433429, 0.46141705, 2.05498],
     [-0.35226226, 0.870752, -0.01091815, 0.1999152, -0.06073183, -0.48284027, -0.25184897],
     [-0.11648433, 0.912237, -0.06194019, 0.16942123, -1.3899, 0.34166393, 0.97756845],
     [-0.6842226, -1.1736702, 0.3762461, -1.6934628, -0.5110089, 1.2754049, -1.4404577]],
    
    [[-1.0608363, 0.89014876, -1.5791024, 0.74357873, 1.3088826, -1.7458462, -0.00101926],
     [1.0777059, 2.4398947, -1.075122, 1.2817347, -1.0457708, -0.0575823, 0.26823944],
     [-0.30824783, -0.39163467, 1.9040701, -1.2065144, -1.2719806, -0.47785115, 1.9137032],
     [-0.62619776, 0.15171216, 0.7074335, 0.2179752, 0.2695976, -0.51078844, 1.0533053],
     [1.6708642, -0.7983012, -0.73707086, 1.2570608, 0.8376691, 0.81455797, 0.96544117],
     [-2.2498147, 0.6785745, 2.1044676, -0.7904935, -0.03147621, 0.01628454, 1.0506629],
     [0.13615216, -1.2329048, -0.8277449, -0.64166594, -0.16468495, -1.3838128, 0.39345345],
     [0.87333214, 1.0460443, -1.385451, 0.01499553, -0.53248173, -2.0223582, -0.8001098]],
    
    [[-1.0760409, -0.9203657, -2.8689573, -1.3752652, 0.5311405, -0.634275, -1.2561296],
     [0.59831554, 0.89339656, 0.19349258, -0.26956186, 0.21125539, 0.87218064, 0.24916327],
     [0.3112421, 0.7543985, -1.5413432, 1.6631768, -0.97224444, -1.2167859, 1.8827425],
     [0.14262906, 1.5757157, -0.24100977, -1.9856219, 1.0706226, -0.95468074, 2.5968833],
     [-0.8204205, -0.7559574, -0.20047009, 0.38723275, -1.1692011, 0.9771397, -1.1447752],
     [1.3394694, 0.45433524, 2.144626, -0.72469556, -1.4892399, -0.02854938, -1.3155057],
     [-0.30164906, -0.2047148, 0.25258553, 0.54385686, -0.09807025, -0.69336486, -0.9638544],
     [0.29995304, 0.03704407, -0.3812418, 0.07943594, -0.6800071, 0.3042758, 0.7309296]],
    
    [[1.957356, -0.8688049, -0.478819, 1.7219703, 0.7667496, 1.4335315, 0.15738125],
     [0.8972799, -0.14302488, -0.96344626, -2.2751193, -0.01029627, 0.26221398, 0.62304413],
     [0.36077744, -1.1190478, -0.49001583, -0.35381666, -0.06878367, -0.46555066, -1.1312178],
     [-0.4365038, 1.0786947, 0.9898978, -1.1715653, 1.3520616, -1.377024, 0.7821855],
     [-0.5222794, 0.12157866, -2.2727368, 0.79923964, -0.8833105, 0.2689467, -0.24089247],
     [-0.9390301, 0.26974523, -0.5096051, -1.3423889, 0.671557, -0.29312912, -1.3745874],
     [0.676753, 0.02001736, 0.580792, 0.45866838, 0.73680997, -1.3242545, -1.44682],
     [1.0965801, -1.0571352, 0.8629156, -0.7979659, -1.0594661, 1.361402, 0.049418]],
    
    [[-0.7320414, -0.12415231, 0.39246544, -0.43017697, 0.84438896, 1.0383393, 0.4470866],
     [0.00599296, 0.96499246, 1.057814, -1.7107246, -0.29790705, -0.5500072, 0.9626524],
     [-0.8591317, 1.7196136, 0.8113776, 0.03830777, 0.37796554, 0.05873026, 0.21973135],
     [-0.21050932, -0.28574887, -0.53018296, 0.6395744, 0.26928946, -0.4737384, 1.0588343],
     [-0.3125468, -1.0695236, -0.72427356, 1.865214, -1.3113602, 0.43317696, 0.28688854],
     [-0.25749853, 0.61865366, 1.0902297, -0.7604784, -0.46240807, -0.9595564, 0.7637273],
     [1.3031865, -0.7231773, 1.2626712, -0.78928334, 0.18933122, 1.7001655, -0.6805369],
     [-0.43712208, -0.24309893, -0.05751095, 1.3146936, -0.05118451, -0.7748253, -0.10658897]],
    
    [[-0.8858597, -0.00717876, 0.44685164, 0.40706733, -1.0757563, -0.69406676, 0.9006848],
     [-0.42116398, -0.32476822, -1.9696139, 0.93381405, -0.347814, -0.59551173, 0.35409963],
     [0.03977225, 1.2829394, 0.1317779, -0.5868073, 0.68558824, 2.1993794, 0.38465983],
     [0.19223769, -0.5488948, 0.53848124, -1.9006051, -0.48802006, 0.79564226, 1.1061515],
     [-0.08698914, -0.7213156, 0.11400396, -0.25464672, -0.9789065, -0.608236, 0.4185338],
     [-2.2768536, -1.5680612, -0.4742592, 0.31941813, -0.47123176, 0.19932298, 1.2764031],
     [0.98729664, -1.8008803, 0.4840511, -0.78048736, -0.9791268, 0.7972983, 0.4310205],
     [1.0294197, 1.4321351, 0.09023922, -0.55651593, -0.8108699, 0.75015897, 1.1207545]],
    
    [[0.4502997, 0.28096187, -0.21081325, -0.7061103, 0.5000147, -0.11618412, 1.0984986],
     [-0.50642705, -1.395841, -1.9684322, -0.07476659, -1.847762, -0.97273546, 1.2062759],
     [-1.1832497, -0.04710979, -1.5368888, 0.09050962, -1.2835667, -1.1037765, -0.6352425],
     [0.03623419, 0.61532956, -1.3958316, -0.05764162, -0.3790958, -0.38869447, -1.0130723],
     [0.2981925, 0.69615734, 0.03989277, 0.10042699, -1.835501, -0.2659723, -0.3018257],
     [0.93354607, 0.78668576, -0.09502307, 0.39529875, -0.6265787, -0.8775214, -2.113317],
     [0.28395098, 0.9028449, 2.2263963, -1.4608351, -1.032141, -0.64069915, -0.9827405],
     [1.3760642, -0.54706883, 1.1421949, -1.381214, 0.79378784, 0.2762494, 0.09724076]],
    
    [[0.27444115, -0.6159855, -0.9077206, -0.3491557, 0.3912622, 0.43696415, 0.9827053],
     [-0.23421279, -0.5318441, 1.2086415, 1.3617507, 0.09451598, -2.741569, -0.13256383],
     [0.04287, -0.9252456, 0.44915748, -0.4996476, -2.0135736, -0.021699, 0.17079495],
     [0.37804303, 0.3290183, 0.753031, -0.9780002, 1.0816981, 0.7012037, 1.5465529],
     [2.3065367, 1.3708698, -1.7693636, 1.832203, -0.8964731, -0.19589011, 0.4869743],
     [0.94748706, -0.31524467, 0.551771, 0.10949931, 2.0978239, -0.63741785, -0.5251891],
     [0.7094176, -0.7455887, 0.9238634, 0.38761047, 0.5410211, 0.24324518, 0.60271525],
     [2.4113326, 0.07173862, 0.9099535, 0.65834486, 0.48412865, 0.42462233, 0.32429868]],
    
    [[2.3419528, 2.266094, 0.6104422, -1.24336, 1.347576, -0.92043144, -0.6444825],
     [0.5438298, 0.11353672, 1.3497149, -1.1640837, 0.18460801, 0.43374857, -0.39911154],
     [-1.3924711, -1.2088548, -0.28430766, -0.18688315, 1.1890152, -0.38412303, 0.30783403],
     [0.8300197, 0.34504363, -0.05160677, 0.06874485, -0.10720129, -1.0766201, 1.151205],
     [-0.80561477, 0.33879828, -1.1631224, 0.00342693, -1.1864235, -1.7714078, -0.55846256],
     [-0.2003743, 0.54651314, -1.2925158, 0.04360921, 2.4077578, -0.55318266, -1.2091233],
     [0.17129491, 1.0808557, -0.38210535, -0.5938776, -2.0481524, 2.130452, 1.4663231],
     [0.50382847, -0.71480703, 0.30012327, -1.8037304, 0.62807053, 0.01668003, 0.00010687]],
    
    [[-0.83296186, -0.67095333, 0.54271257, -0.4447467, -0.26850703, 1.439739, -0.6999207],
     [-0.79247934, 0.790536, -2.6073565, 0.0239189, 0.4505262, -0.7308653, 0.31396905],
     [-0.31025594, -0.5396589, 1.752556, -1.0533803, 0.9518338, 0.38116023, -0.14028986],
     [0.2973768, -0.9956112, 0.03112388, 1.9930489, 0.3941558, -0.7017968, -1.387223],
     [1.1082778, -1.0183873, 0.39777696, 0.5881747, 0.84491754, 0.55389935, 1.2504689],
     [0.23361401, 0.6445308, -1.6959119, 1.6489294, 1.4768391, 0.35961348, -0.3368996],
     [0.7805847, -0.90055877, 0.1002969, 0.13489738, -0.4817404, 0.31623995, -1.40849],
     [-0.5222229, 0.6778943, 0.93692833, 0.55800664, 1.739175, -2.0510468, -1.9011563]],
    
    [[-0.49966702, 0.37805218, 0.28774408, 0.80221033, 1.1355261, 1.46018, 1.8947716],
     [0.2793012, 2.378107, 0.52447706, -0.41605946, -0.18786949, 0.83420885, 1.4901091],
     [0.5096204, 1.3834916, 1.3099371, -1.1177974, 0.9233587, 0.84042466, 0.34625462],
     [1.5843941, 1.7831918, 0.7528205, -0.35285524, 0.81907463, -1.3265036, 0.14892593],
     [-1.1267295, 0.5507837, -0.57567954, -1.8048518, -0.01391057, 0.25332567, -1.9524001],
     [0.15302303, -1.2059829, -0.22030273, -0.8040847, 0.9350981, -0.23942019, 0.23318584],
     [-0.03883116, 0.7267606, 1.9843854, 0.5948334, 0.3378055, -1.7167985, 0.13213524],
     [-0.2766811, 1.0684518, 0.43116674, 0.6334991, 2.7785933, 0.76511705, 2.4957232]],
    
    [[0.48150745, -0.32040703, 1.4917707, -1.4131955, -0.7824414, -0.04473221, -0.09423822],
     [0.5826115, -1.7380102, 0.9087168, -0.3059203, 0.707993, 0.5302861, -0.01529173],
     [1.0285695, 0.6952356, -0.46711287, -0.34898242, 0.91925025, 0.58113545, -0.99423355],
     [2.441503, 2.3827846, 0.421873, -0.49484035, -0.36837888, 1.117872, -2.152353],
     [-1.0063254, 0.8873134, -0.7960882, 0.10826004, -0.36040914, 0.08283357, 3.318276],
     [-1.601249, -1.4762565, -0.25171548, 0.27028832, -1.2059499, -0.12670825, 0.93332976],
     [1.0788072, -1.0742241, 0.403653, 0.22936872, -0.24382341, 0.11088572, -0.12047929],
     [0.00556435, 0.13963354, 0.90099657, 1.1090373, 1.1861458, 1.44241, -0.7478856]],
    
    [[-1.1369542, 1.8587931, 1.0406264, 1.3246344, 1.1571591, -0.69498694, 0.3179156],
     [0.59690285, 0.30407992, -0.83015394, 1.9544983, -0.37200823, 0.9478438, -1.2121648],
     [-0.6873798, -0.9163224, -1.1665807, 0.45168707, -0.3018603, -0.6385228, -0.9144808],
     [-2.044211, -0.4935683, -1.8473966, -0.7452655, 0.2932037, -0.11863417, 0.43310302],
     [0.79446256, 0.707377, 2.5121398, -1.1041735, 0.48357362, -1.2674941, -1.120776],
     [0.38648066, 0.38056546, -0.45806012, 0.32948092, -0.21368699, -0.74591607, -0.48889023],
     [-0.47842902, 0.22854784, -0.24643315, -2.069844, 0.103049, 0.90802926, 0.19270198],
     [0.8506123, 0.6621215, 0.2042667, -1.4763446, 0.9636837, 0.457721, -0.6846171]],
    
    [[0.8853584, 0.5612104, 0.10812267, 0.09606382, 0.41751516, -0.19161734, 0.6409575],
     [-0.1837254, -0.31902626, 1.0911556, 2.0430825, -0.02935624, -1.2032852, 0.13378398],
     [0.6372215, 0.46786213, -2.2372222, -0.40551692, 1.032318, 3.4050224, -0.804289],
     [-0.00984093, -0.00413692, -2.043892, -0.5011179, -0.16392641, 0.18990083, 0.05671286],
     [-0.22184819, 1.4029222, 0.01216271, -0.84908867, 0.23220366, 0.56808877, 0.31192],
     [0.49168205, 0.0687003, -1.5647538, 3.9396455, -1.768812, 1.0510364, 0.2628251],
     [0.7880214, 1.5386645, -0.3204191, -0.28165218, 0.7230415, -1.1864082, -0.880725],
     [-0.6990944, -0.4720473, 0.38955915, -0.11739796, 1.4305097, -0.02257603, 0.84529865]],
    
    [[1.4389133, -0.6326712, 0.31587386, 0.43140432, 0.37104514, -0.14707251, 1.8016388],
     [-0.3274905, 0.7813513, 0.37797472, 0.28324443, 0.0233415, 1.1082617, 0.14696409],
     [1.5209819, 0.168412, 1.0286613, 0.31586602, 1.1291076, -0.10279872, 0.7335919],
     [-0.27348164, -1.3031857, 1.9324102, -0.5090103, -0.35740313, -0.29190928, 0.2168357],
     [0.751526, 0.33574829, 1.71823, -0.23006849, 0.4423312, 0.31041345, -0.12997021],
     [-0.9227169, 1.0012541, 0.64812833, -0.52484274, -0.6609661, -0.17653675, -1.3581274],
     [-1.3685123, -0.8297906, 0.8925549, -0.44799876, -0.8869055, 0.47454202, -1.0706333],
     [0.23923485, 0.596514, -0.55767435, 3.1489086, -0.2510702, -1.4637703, 0.15125409]],
    
    [[1.4432919, 0.40863574, 0.09064027, -0.26307395, 0.24845637, -0.17447886, 0.20500886],
     [-0.18372922, -0.20535418, 0.10924685, -0.60457385, 0.03482886, -1.4719927, -0.53050566],
     [-2.0633414, -1.2771286, 1.2865186, -0.95287895, -0.3649892, -0.0749408, 0.753496],
     [-0.7221364, -0.9600963, 0.74960315, -0.6883403, 0.23090087, 1.7119471, -1.8118477],
     [0.07661213, 0.47933754, 0.20504089, -0.13539097, 0.09145955, -0.70955807, -1.1093351],
     [0.10582258, 0.23483972, -0.9685352, 0.23103549, 1.1925538, -1.2734053, -0.845849],
     [1.2856016, 1.5843773, 0.1686462, -0.6534943, 0.06447883, -0.22050548, 0.29820564],
     [-0.47443986, -0.3561741, -2.1487334, 0.40114385, -0.26662236, -0.2225131, 0.13633096]],
    
    [[-0.50090706, 0.9836541, -0.72309834, 0.33581755, 0.27356178, -0.09816581, -0.14279993],
     [-1.4727582, -1.1402419, 0.71899605, -0.7360807, 1.1036323, -0.8169985, -0.559137],
     [0.03874158, -0.3500361, -0.36370602, 0.34682027, 0.07775027, 0.94501376, -1.6430002],
     [-1.0391974, -1.7201697, -0.70357835, 1.5002863, 0.13866246, -0.29421207, 0.91305405],
     [0.3438481, 0.74695814, 0.11036935, 0.78289086, -1.001515, 0.5687552, -0.6176505],
     [0.3134321, -1.0375038, -1.5941671, -0.18800193, 0.2042623, -1.3770927, -0.5164289],
     [0.7059186, -1.2775136, 0.5601877, -1.1655983, 0.58374906, 0.2867978, -1.475562],
     [-0.3790142, -0.20962788, 0.70582265, 1.3279319, 0.39593154, -0.7527053, 0.4824462]],
    
    [[1.945068, 0.22034678, -0.6519652, 1.2261134, -0.25767982, -0.16659155, -1.1739291],
     [1.7060181, -0.31624818, 0.11161104, 0.7077596, 0.6895517, 0.5082252, -1.0655246],
     [0.84506446, 1.6683276, 1.0804147, -1.2063113, 0.26119968, -0.08155034, 1.9534386],
     [-1.056925, -1.4889036, -2.8024523, 0.79172295, -0.9623386, 0.09310131, -0.12783727],
     [-0.06820646, -0.09216437, -1.7447118, -0.36374027, 0.05264863, 0.01848697, 0.815216],
     [0.03012928, -0.6252085, 0.58886236, 0.6792602, 0.17388524, -0.9417572, -1.2306292],
     [-0.06702902, 0.3223861, -0.4005192, -1.322837, 0.8170104, 0.37652487, -0.5388462],
     [-0.5566171, -0.6438817, 0.99396944, 0.40304014, -0.65557, 0.61457837, -0.19933325]],
    
    [[-0.02193078, 0.32158506, 0.09600195, -1.2329367, 0.7492863, 0.2461374, -2.1763668],
     [1.255319, -0.9510376, -2.11203, -0.32580185, -0.39476308, 1.9345438, 1.7843226],
     [0.71719515, 1.1960462, -0.56042063, -0.50346833, 1.0186776, -1.6359098, -0.2296646],
     [0.31189746, -1.1701431, 0.7263634, -0.85526955, -0.30548543, 0.83602846, -1.0508664],
     [-0.07857783, -2.2570658, -0.62261957, -0.74978536, -0.34662178, -0.7541806, -0.8288501],
     [-0.34755743, -1.1186867, 0.34952384, 1.0424857, -0.49294376, -0.8892171, 0.1367592],
     [0.59158826, 0.27250928, -2.3048978, -0.05882964, -1.7647171, 1.6594201, -0.8396546],
     [0.22806998, 1.971737, -0.03097878, -1.3620718, -0.5073694, 1.3326721, 0.3349245]],
    
    [[-0.87948155, 0.3567289, 1.6427234, -0.53016114, -1.2164687, -1.4128083, -1.6121237],
     [-1.3993251, -0.65766007, 0.69676995, -0.67160106, 1.3332534, -1.4450325, 1.2354064],
     [-0.77065426, -1.1397938, -0.730877, 1.5242308, -0.24497898, 1.3298101, -1.4687148],
     [0.4966562, 0.8515605, 1.4978958, 0.75685835, -0.53604275, 0.9230689, -0.96250415],
     [0.66376936, 0.7921751, 0.5504553, 0.49502042, -2.5128422, 0.8043288, -0.6290292],
     [-0.79981697, -0.7222679, -0.5948983, 0.18090709, -0.35378042, -0.15225285, -0.95420593],
     [-0.7765107, -0.16377874, -0.08900865, 1.3925242, 2.0550098, 1.3457111, -1.8553666],
     [0.62525195, 2.8079748, -2.3819366, -1.1619583, 0.40418074, 0.6996681, 0.42566642]],
    
    [[-0.5398392, 1.2721713, 1.5786334, 0.63013464, -0.19974774, -1.6466998, -0.34578508],
     [-0.6830255, -1.4371754, -0.5986936, -0.09373292, 0.2703549, -0.11325529, 0.6116078],
     [0.29516053, 0.8172026, -1.2201881, 0.9773082, 0.54224235, 0.14214271, 2.6923108],
     [-0.6422448, 0.12723026, -0.36536178, 0.00916145, -1.1405051, 0.55396324, -1.6698471],
     [-0.7165685, -1.3297731, -0.8552066, 0.5338791, -0.93755245, -0.89377075, 0.26319513],
     [-0.8264852, 2.3217936, 0.46491557, -0.0716849, -1.6087991, -0.23718823, -0.13611704],
     [0.15273927, -1.3435073, -0.2431849, 0.45200148, 0.6638381, -0.44217268, 1.1993718],
     [-0.4479535, 0.16927752, 1.4681038, 1.0840298, -0.06554849, 0.13661557, -0.69260675]],
    
    [[-1.1143047, -0.7388156, 0.04505922, 1.4535184, 0.7000627, -1.2535497, -0.26476642],
     [-0.62924176, -0.06895317, -0.47381333, 0.07168433, -0.24162085, 0.23609483, 0.4908991],
     [0.58802015, 1.8568169, -1.4664818, 0.13552691, 0.01602927, -0.87805414, -0.24751957],
     [0.18881929, -0.9898477, 1.7103597, -0.10698922, 0.14707616, -1.4915003, -0.04727007],
     [-1.1523079, 0.03901471, -0.7735915, 0.67490405, 0.5352779, -0.51085454, -2.0760963],
     [-1.1562266, 0.06087862, -0.3887601, -0.6050554, -0.7772089, -2.3824413, -2.02866],
     [0.43047142, -0.9680853, -1.7272409, -0.64066267, -0.03881302, 0.74227893, 1.9808877],
     [-0.2986719, 0.12569498, 0.02439793, 0.29822016, 1.6327963, 0.22618409, 1.1429639]],
    
    [[0.86813766, 0.31610218, 2.650461, -1.4169676, 0.3291654, -0.7091785, -0.5932599],
     [1.0239487, 0.8012353, 0.46223563, -0.6745125, -1.8194115, 0.11316124, -1.3973578],
     [0.86862177, 2.3761826, 0.65751606, -0.47656423, 0.47281617, 0.04652444, 0.11919155],
     [0.879981, 0.09854733, 0.8064252, 0.03313981, 0.4828424, 1.0518277, 0.7066174],
     [-0.14063661, 0.2862756, 0.5539883, -0.93106717, 0.5078787, 0.5414004, 0.30342114],
     [-0.18843299, -0.42547083, -1.3753343, 1.336718, -1.2634463, 0.06084029, 0.51768106],
     [-0.36843804, 1.2667968, -1.3314835, -0.15315276, 0.46387085, -0.2246355, -0.4940856],
     [0.8274832, 1.4540988, -0.03256404, 1.6916282, -2.2187455, 0.3446267, 0.5800504]],
    
    [[1.5497167, 0.00831517, 1.6788718, 1.709033, 0.74528617, 1.0227869, -1.0383476],
     [0.3281922, 0.8407094, 0.37353963, -1.2203051, -2.1893213, 0.31649718, 0.8480487],
     [0.20305903, 0.6467525, 0.5482997, 0.13529198, -0.52780306, 0.24640457, -1.5738835],
     [0.25813192, 0.34346747, 0.57144594, -1.324723, -0.6052619, -0.4575464, 0.8290694],
     [0.619632, 0.15880992, -0.25992784, -0.08625133, -1.3056115, 0.42005044, 1.144761],
     [-1.0581728, 0.40827137, -1.4265327, 0.03120423, -1.1449231, -0.95943683, 0.00862197],
     [1.3070747, 0.17301312, 0.21064064, 0.19074826, -0.3218065, 1.5555077, 0.7360219],
     [-0.14042455, -0.8780759, 0.31644362, 1.427395, -0.27087197, -0.36798885, 0.57504094]],
    
    [[0.21305053, -0.6266876, 1.0543494, -0.13894826, -0.0060407, 0.39617318, 2.1222916],
     [0.4569486, -0.9175309, -0.67141825, 0.2987879, -0.03972428, -0.3194652, -0.37916628],
     [0.24306826, 0.3920417, 0.02443402, 1.3712834, 1.1526486, 1.7127602, -1.1353463],
     [-3.0214503, 1.6188203, 0.03673983, 0.5827797, 1.14644, 1.1449307, -0.2879651],
     [0.16649067, -1.0432358, 0.67483383, 1.5845457, -1.3040129, -1.7409662, -0.18703537],
     [-0.9107379, 1.0143437, 0.12137226, 0.6234888, 1.5009955, -1.6970906, 0.531869],
     [-1.3988973, -0.6368376, 0.04056473, -0.33793095, 0.78766286, -0.12991959, 0.21551774],
     [-0.7105916, -1.1934172, -0.24457727, 0.24275109, 0.35098362, 0.7459229, 0.70132333]]
])
            labels = np.empty(len(random_points), dtype=int)

            # Original labels before removal
            apple = [3,10,12,18,21,22,24,26,35,37,38,44,47]
            mug   = [2,6,11,14,15,17,27,28,31,32,43,48]
            sushi = [0,1,4,5,7,8,9,13,16,19,20,23,25,29,30,33,34,36,39,40,41,42,45,46,49]

            # Create full labels array
            labels = np.empty(len(random_points), dtype=int)
            for i in apple:
                labels[i] = 0
            for i in mug:
                labels[i] = 1
            for i in sushi:
                labels[i] = 2

            # Remove specific indices from both points and labels together
            indices_to_remove = [12,48,4,8,16,20,23,33,36,42,27]
            indices_to_keep = [i for i in range(len(random_points)) if i not in indices_to_remove]

            # Now, for each class, sample 10 random indices
            np.random.seed(42)
            indices_apple = [i for i in indices_to_keep if labels[i] == 0]
            indices_mug = [i for i in indices_to_keep if labels[i] == 1]
            indices_sushi = [i for i in indices_to_keep if labels[i] == 2]

            sampled_apple = np.random.choice(indices_apple, size=min(10, len(indices_apple)), replace=False)
            sampled_mug = np.random.choice(indices_mug, size=min(10, len(indices_mug)), replace=False)
            sampled_sushi = np.random.choice(indices_sushi, size=min(10, len(indices_sushi)), replace=False)

            indices_to_keep = np.concatenate([sampled_apple, sampled_mug, sampled_sushi])

            # Filter both data and labels using the same indices
            random_points_viz = random_points[indices_to_keep]
            labels_viz = labels[indices_to_keep]  # This preserves the correct label-point correspondence
            label_names = {0: "apple", 1: "mug", 2: "sushi"}
            colors = {0: "red", 1: "blue", 2: "green"}
            # Run rollouts for each random point
            trajectories_data = []
            
            # Create output directory
            output_dir = Path(args_cli.output_dir) / task_name / epoch_num / "flow_analysis"
            output_dir.mkdir(parents=True, exist_ok=True)
            

            obj_init_positions = val_dataset.normalized_train_data["blocks_init_dict"][-1]
            # Get real observation from environment (only once for all latent points)
            env_ids = torch.arange(env.num_envs, device=env.device)
            apply_demo_objects(env, obj_init_positions, env_ids)
            sleep(1.0)  # Let physics settle
            current_obs = get_observation_from_env(env, None, None)
            current_obs[6] = 0.04  # Set initial gripper state
            
            # Convert filtered numpy arrays to torch tensors with proper shape
            interpolation_points_batch = []
            for random_point in random_points_viz:
                random_point_tensor = torch.tensor(random_point, device=device, dtype=torch.float32)
                # Expand to [pred_horizon, action_dim] shape if needed
                interpolation_points_batch.append(random_point_tensor)
            
            # Run batched flow matching for filtered latent points
            print(f"\nRunning batch flow matching for {len(random_points_viz)} filtered points...")
            trajectories_data = run_flow_matching_batch(
                current_obs=current_obs,
                nets=nets,
                cfg=cfg,
                device=device,
                interpolation_points_batch=interpolation_points_batch,
                obj_init_positions=obj_init_positions
            )

            print(f"\nExperiment {exp_idx + 1} completed with {len(random_points_viz)} filtered points")
               
            # Plot comprehensive flow matching analysis using filtered data
            plot_path = plot_flow_matching_analysis(random_points_viz, trajectories_data, labels_viz, label_names, colors, exp_idx, output_dir)
            print(f"Saved flow matching analysis plot: {plot_path}")

        except Exception as e:
            omni.log.error(f"Error in experiment {exp_idx + 1}: {e}")
            import traceback
            traceback.print_exc()
            break
    
    env.close()
    print("Flow matching analysis finished")

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# Select fewer key timesteps to avoid memory issues and speed up processing
key_timesteps = np.arange(100)
def create_2d_plot_for_action_step(args):
    """Helper function for parallel 2D plot generation."""
    action_step, trajectories_data, labels, label_names, colors, exp_idx, output_dir, num_flow_steps, pred_horizon = args
    
    print(f"[Process] Creating 2D plot for action timestep {action_step + 1}/{pred_horizon}")
    
    # Select key timesteps to visualize (not all 51 steps!)
    # We have 51 total steps: initial (t=0) + 50 flow steps (t=0.02 to t=1.0
    
    # Calculate grid size based on selected timesteps
    num_selected = len(key_timesteps)
    if num_selected <= 10:
        rows, cols = 1, 10
    elif num_selected <= 20:
        rows, cols = 2, 10
    elif num_selected <= 30:
        rows, cols = 3, 10
    elif num_selected <= 40:
        rows, cols = 4, 10
    elif num_selected <= 50:
        rows, cols = 5, 10
    elif num_selected <= 51:
        rows, cols = 6, 10  # 6 rows x 10 columns for 51 plots (with 9 empty spaces)
    else:
        rows = int(np.ceil(num_selected / 10))
        cols = 10
    
    # Create subplots: one for each selected flow timestep
    fig, axes_raw = plt.subplots(rows, cols, figsize=(cols*2.5, rows*2))
    
    # Convert to flat list for easier indexing - handle all cases properly
    if rows == 1 and cols == 1:
        axes_list = [axes_raw]
    elif rows == 1:
        axes_list = [axes_raw] if cols == 1 else list(axes_raw)
    elif cols == 1:
        axes_list = list(axes_raw)
    else:
        axes_list = list(axes_raw.flatten())
    
    for plot_idx, t_idx in enumerate(key_timesteps):
        if plot_idx >= len(axes_list):
            break
            
        # Safety check for valid timestep index
        max_available_steps = len(trajectories_data[0]['flow_x_values']) - 1
        if t_idx > max_available_steps:
            print(f"Warning: Timestep {t_idx} exceeds available steps (0-{max_available_steps}), skipping...")
            continue
            
        ax = axes_list[plot_idx]
        
        # Plot points colored by their label
        for point_idx, traj in enumerate(trajectories_data):
            if 'flow_x_values' in traj and t_idx < len(traj['flow_x_values']):
                # Get x values at this flow timestep
                x_values = traj['flow_x_values'][t_idx]  # Shape: [1, pred_horizon, action_dim]
                
                # Use current action timestep, first 3 dimensions for visualization
                if x_values.shape[0] == 1:  # Remove batch dimension
                    x_values = x_values[0]
                
                # Extract XYZ coordinates from current action timestep
                x, y, z = x_values[action_step, :3]  # Current action timestep, XYZ dimensions
                
                # Get label and color for this point
                label = labels[point_idx]
                color = colors[label]
                label_name = label_names[label]
                
                # Plot point (only show legend for first occurrence of each label)
                if plot_idx == 0 and point_idx == np.where(labels == label)[0][0]:
                    ax.scatter(x, y, c=color, s=50, alpha=0.7, label=label_name)
                else:
                    ax.scatter(x, y, c=color, s=50, alpha=0.7)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'Flow Step t={t_idx}')
        ax.grid(True, alpha=0.3)
        
        # Only show legend on first subplot, positioned at top
        if plot_idx == 0:
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.55), ncol=3)
    
    # Hide unused subplots
    for i in range(num_selected, len(axes_list)):
        axes_list[i].set_visible(False)
    
    plt.suptitle(f'Flow Matching Evolution - Exp {exp_idx + 1}, Action Step {action_step + 1}/{pred_horizon}\nHow Latent Points Evolve Through Flow Timesteps', fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for suptitle
    
    # Save the plot for this action timestep
    plot_path = output_dir / f"flow_matching_analysis_exp{exp_idx:02d}_action{action_step:02d}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return f"Saved 2D plot for action step {action_step}: {plot_path}"


def create_3d_plot_for_action_step(args):
    """Helper function for parallel 3D plot generation."""
    action_step, trajectories_data, labels, label_names, colors, exp_idx, output_dir, num_flow_steps, pred_horizon = args
    
    print(f"[Process] Creating 3D plot for action timestep {action_step + 1}/{pred_horizon}")
    
    # Select key timesteps for 3D visualization (same as 2D)
    step_indices = key_timesteps  # 7 key progression points
    max_3d_plots = len(step_indices)
    
    # Calculate 3D grid layout based on selected timesteps
    if max_3d_plots <= 10:
        grid_rows, grid_cols = 1, 10
    elif max_3d_plots <= 20:
        grid_rows, grid_cols = 2, 10
    elif max_3d_plots <= 30:
        grid_rows, grid_cols = 3, 10
    elif max_3d_plots <= 40:
        grid_rows, grid_cols = 4, 10
    elif max_3d_plots <= 50:
        grid_rows, grid_cols = 5, 10
    elif max_3d_plots <= 51:
        grid_rows, grid_cols = 6, 10  # 6 rows x 10 columns for 51 plots
    else:
        grid_rows = int(np.ceil(max_3d_plots / 10))
        grid_cols = 10
    
    # Generate timestep names based on actual step indices
    selected_timestep_names = [f"t={step_indices[i]}" for i in range(len(step_indices))]
    
    fig = plt.figure(figsize=(grid_cols*6, grid_rows*5))
    
    for plot_idx, t_idx in enumerate(step_indices):
        if plot_idx >= max_3d_plots:  # Safety check
            break
            
        ax = fig.add_subplot(grid_rows, grid_cols, plot_idx + 1, projection='3d')
        
        # Plot points colored by their label
        for point_idx, traj in enumerate(trajectories_data):
            if 'flow_x_values' in traj and t_idx < len(traj['flow_x_values']):
                # Get x values at this flow timestep
                x_values = traj['flow_x_values'][t_idx]  # Shape: [1, pred_horizon, action_dim]
                
                # Use current action timestep, first 3 dimensions for 3D visualization
                if x_values.shape[0] == 1:  # Remove batch dimension
                    x_values = x_values[0]
                
                # Extract XYZ coordinates from current action timestep
                x, y, z = x_values[action_step, :3]  # Current action timestep, XYZ dimensions
                
                # Get label and color for this point
                label = labels[point_idx]
                color = colors[label]
                label_name = label_names[label]
                
                # Plot point (only show legend for first occurrence of each label)
                if plot_idx == 0 and point_idx == np.where(labels == label)[0][0]:
                    ax.scatter(x, y, z, c=color, s=50, alpha=0.7, label=label_name)
                else:
                    ax.scatter(x, y, z, c=color, s=50, alpha=0.7)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Flow Step {selected_timestep_names[plot_idx]}')
        
        # Only show legend on first subplot, positioned at top
        if plot_idx == 0:
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3)
    
    plt.suptitle(f'Flow Matching Evolution 3D - Exp {exp_idx + 1}, Action Step {action_step + 1}/{pred_horizon}\nHow Latent Points Evolve Through Flow Timesteps', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for suptitle
    
    # Save the 3D plot for this action timestep
    plot_3d_path = output_dir / f"flow_matching_analysis_3d_exp{exp_idx:02d}_action{action_step:02d}.png"
    plt.savefig(plot_3d_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return f"Saved 3D plot for action step {action_step}: {plot_3d_path}"

def create_combined_3d_trajectory_plot(selected_flow_steps, trajectories_data, labels, label_names, colors, exp_idx, output_dir, pred_horizon):
    """Create multiple 3D trajectory subplots showing complete action trajectories at different flow timesteps."""
    
    print(f"Creating combined 3D trajectory plot for {len(selected_flow_steps)} flow timesteps")
    
    # Calculate subplot layout
    num_plots = len(selected_flow_steps)
    if num_plots <= 10:
        rows, cols = 1, 10
    elif num_plots <= 20:
        rows, cols = 2, 10
    elif num_plots <= 30:
        rows, cols = 3, 10
    elif num_plots <= 40:
        rows, cols = 4, 10
    elif num_plots <= 50:
        rows, cols = 5, 10
    elif num_plots <= 51:
        rows, cols = 6, 10  # 6 rows x 10 columns for 51 plots (with 9 empty spaces)
    else:
        # For more plots, use dynamic grid with 10 columns
        rows = int(np.ceil(num_plots / 10))
        cols = 10
    
    # Create figure with subplots
    fig = plt.figure(figsize=(cols * 5, rows * 4))
    
    for plot_idx, t_idx in enumerate(selected_flow_steps):
        ax = fig.add_subplot(rows, cols, plot_idx + 1, projection='3d')
        
        # Plot complete trajectory for each point at this flow timestep
        for point_idx, traj in enumerate(trajectories_data):
            if 'flow_x_values' in traj and t_idx < len(traj['flow_x_values']):
                # Get x values at this flow timestep
                x_values = traj['flow_x_values'][t_idx]  # Shape: [1, pred_horizon, action_dim]
                
                if x_values.shape[0] == 1:  # Remove batch dimension
                    x_values = x_values[0]  # Shape: [pred_horizon, action_dim]
                
                # Extract XYZ coordinates for all action timesteps (complete trajectory)
                trajectory_x = x_values[:, 0]  # X coordinates for all timesteps
                trajectory_y = x_values[:, 1]  # Y coordinates for all timesteps  
                trajectory_z = x_values[:, 2]  # Z coordinates for all timesteps
                
                # Get label and color for this point
                label = labels[point_idx]
                color = colors[label]
                label_name = label_names[label]
                
                # Plot trajectory line (only show legend for first subplot and first occurrence of each label)
                if plot_idx == 0 and point_idx == np.where(labels == label)[0][0]:
                    ax.plot(trajectory_x, trajectory_y, trajectory_z, 
                           color=color, alpha=0.7, linewidth=1, label=f'{label_name} trajectory')
                else:
                    ax.plot(trajectory_x, trajectory_y, trajectory_z, 
                           color=color, alpha=0.7, linewidth=1)
        
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position') 
        ax.set_zlabel('Z Position')
        ax.set_title(f'Flow Step t={t_idx}\n')
        ax.grid(True, alpha=0.3)
        
        # Set equal aspect ratio for better visualization
        ax.set_box_aspect([1,1,1])
        
        # Only show legend on first subplot, positioned at top
        if plot_idx == 0:
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3)
    
    plt.suptitle(f'Flow Matching 3D Trajectories Evolution - Exp {exp_idx + 1}\n'
                 f'Complete Action Chunks at Different Flow Timesteps', fontsize=16)
    plt.tight_layout()
    
    # Save the combined 3D trajectory plot
    plot_combined_path = output_dir / f"flow_matching_trajectories_3d_combined_exp{exp_idx:02d}.png"
    plt.savefig(plot_combined_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none', format='png')
    plt.close()
    
    return f"Saved combined 3D trajectory plot: {plot_combined_path}"


def create_dimensionality_reduction_plot_for_action_step(args):
    """
    Create PCA analysis plots for a single action step across ALL flow timesteps.
    Shows evolution of action space structure through flow matching process.
    Creates subplots for key timesteps showing PCA projections.
    """
    action_step, trajectories_data, labels, label_names, colors, exp_idx, output_dir, num_flow_steps, pred_horizon = args
    
    print(f"[Process] Creating PCA plot for action timestep {action_step + 1}/{pred_horizon} across ALL flow timesteps")
    
    # Use key timesteps for analysis (same as XYZ plots)
    selected_timesteps = key_timesteps
    
    # Calculate grid size based on selected timesteps  
    num_selected = len(selected_timesteps)
    if num_selected <= 10:
        rows, cols = 1, 10
    elif num_selected <= 20:
        rows, cols = 2, 10
    elif num_selected <= 30:
        rows, cols = 3, 10
    elif num_selected <= 40:
        rows, cols = 4, 10
    elif num_selected <= 50:
        rows, cols = 5, 10
    elif num_selected <= 51:
        rows, cols = 6, 10  # 6 rows x 10 columns for 51 plots (with 9 empty spaces)
    else:
        rows = int(np.ceil(num_selected / 10))
        cols = 10
    
    # Create TWO figures: one for PCA 2D, one for PCA 3D
    fig_2d, axes_2d = plt.subplots(rows, cols, figsize=(cols*3, rows*2.5))
    fig_3d = plt.figure(figsize=(cols*3, rows*2.5))
    
    # Convert axes to flat list for easier indexing
    if rows == 1 and cols == 1:
        axes_2d_list = [axes_2d]
    elif rows == 1:
        axes_2d_list = list(axes_2d) if cols > 1 else [axes_2d]
    elif cols == 1:
        axes_2d_list = list(axes_2d) if rows > 1 else [axes_2d]
    else:
        axes_2d_list = list(axes_2d.flatten())
    
    try:
        # Loop through each selected flow timestep
        for plot_idx, t_idx in enumerate(selected_timesteps):
            if plot_idx >= num_selected:
                break
                
            # Safety check for valid timestep index
            max_available_steps = len(trajectories_data[0]['flow_x_values']) - 1
            if t_idx > max_available_steps:
                print(f"Warning: Timestep {t_idx} exceeds available steps (0-{max_available_steps}), skipping...")
                continue
            
            # Collect all points at this specific flow timestep and action step
            all_points = []
            all_labels_list = []
            
            for point_idx, traj in enumerate(trajectories_data):
                if 'flow_x_values' in traj and t_idx < len(traj['flow_x_values']):
                    x_values = traj['flow_x_values'][t_idx]  # Shape: [1, pred_horizon, action_dim]
                    
                    if x_values.shape[0] == 1:
                        x_values = x_values[0]  # Shape: [pred_horizon, action_dim]
                    
                    # Extract COMPLETE action vector for this action timestep
                    action_vector = x_values[action_step, :]  # Shape: [action_dim]
                    all_points.append(action_vector)
                    all_labels_list.append(labels[point_idx])
            
            if len(all_points) == 0:
                print(f"  No data found for flow timestep {t_idx}, action step {action_step}")
                continue
            
            data = np.vstack(all_points)  # Shape: [num_points, action_dim]
            all_labels_array = np.array(all_labels_list)
            action_dim = data.shape[1]
            
            # PCA 2D analysis for this timestep
            pca_2d = PCA(n_components=2)
            data_pca_2d = pca_2d.fit_transform(data)
            
            # PCA 3D analysis for this timestep
            pca_3d = PCA(n_components=3)
            data_pca_3d = pca_3d.fit_transform(data)
            
            # ==================== PCA 2D SUBPLOT ====================
            ax_2d = axes_2d_list[plot_idx]
            
            # Plot points colored by their label
            for label_idx in np.unique(all_labels_array):
                mask = all_labels_array == label_idx
                ax_2d.scatter(
                    data_pca_2d[mask, 0], 
                    data_pca_2d[mask, 1],
                    c=colors[label_idx],
                    label=label_names[label_idx] if plot_idx == 0 else "",  # Only show legend on first plot
                    alpha=0.8,
                    s=40,
                    edgecolors='black',
                    linewidth=0.3
                )
            
            ax_2d.set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.1%})', fontsize=9)
            ax_2d.set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.1%})', fontsize=9)
            ax_2d.set_title(f'Flow Step t={t_idx}\nVar: {pca_2d.explained_variance_ratio_.sum():.1%}', 
                        fontsize=10, fontweight='bold')
            ax_2d.grid(True, alpha=0.3)
            
            # Only show legend on first subplot
            if plot_idx == 0:
                ax_2d.legend(loc='best', fontsize=8)
            
            # ==================== PCA 3D SUBPLOT ====================
            ax_3d = fig_3d.add_subplot(rows, cols, plot_idx + 1, projection='3d')
            
            # Plot points colored by their label
            for label_idx in np.unique(all_labels_array):
                mask = all_labels_array == label_idx
                ax_3d.scatter(
                    data_pca_3d[mask, 0], 
                    data_pca_3d[mask, 1],
                    data_pca_3d[mask, 2],
                    c=colors[label_idx],
                    label=label_names[label_idx] if plot_idx == 0 else "",  # Only show legend on first plot
                    alpha=0.8,
                    s=40,
                    edgecolors='black',
                    linewidth=0.3
                )
            
            ax_3d.set_xlabel(f'PC1 ({pca_3d.explained_variance_ratio_[0]:.1%})', fontsize=8)
            ax_3d.set_ylabel(f'PC2 ({pca_3d.explained_variance_ratio_[1]:.1%})', fontsize=8)
            ax_3d.set_zlabel(f'PC3 ({pca_3d.explained_variance_ratio_[2]:.1%})', fontsize=8)
            ax_3d.set_title(f'Flow Step t={t_idx}\nVar: {pca_3d.explained_variance_ratio_.sum():.1%}', 
                        fontsize=10, fontweight='bold')
            
            # Only show legend on first subplot
            if plot_idx == 0:
                ax_3d.legend(loc='best', fontsize=6)
        
        # Hide unused subplots for both figures
        for i in range(num_selected, len(axes_2d_list)):
            axes_2d_list[i].set_visible(False)
        
        # ==================== SAVE PCA 2D PLOT ====================
        fig_2d.suptitle(
            f'PCA 2D Evolution Analysis - Experiment {exp_idx + 1}\n'
            f'Action Step {action_step + 1}/{pred_horizon} - How Action Space Structure Evolves Through Flow',
            fontsize=12, fontweight='bold', y=0.98
        )
        plt.figure(fig_2d)
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for suptitle
        
        plot_path_2d = output_dir / f"pca_2d_evolution_exp{exp_idx:02d}_action{action_step:02d}.png"
        fig_2d.savefig(plot_path_2d, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig_2d)
        
        # ==================== SAVE PCA 3D PLOT ====================
        fig_3d.suptitle(
            f'PCA 3D Evolution Analysis - Experiment {exp_idx + 1}\n'
            f'Action Step {action_step + 1}/{pred_horizon} - How Action Space Structure Evolves Through Flow',
            fontsize=12, fontweight='bold', y=0.98
        )
        plt.figure(fig_3d)
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for suptitle
        
        plot_path_3d = output_dir / f"pca_3d_evolution_exp{exp_idx:02d}_action{action_step:02d}.png"
        fig_3d.savefig(plot_path_3d, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig_3d)
        
        return f"Saved PCA evolution plots for action step {action_step}: 2D={plot_path_2d}, 3D={plot_path_3d}"
        
    except Exception as e:
        plt.close(fig_2d)
        plt.close(fig_3d)
        print(f"  Error in PCA evolution analysis for action step {action_step}: {e}")
        return f"Failed action step {action_step}: {e}"


def plot_flow_matching_analysis_with_dimred(random_points, trajectories_data, labels, label_names, colors, exp_idx, output_dir):
    """
    Plot comprehensive flow matching analysis WITH dimensionality reduction (PCA + t-SNE + Variance).
    Creates separate comprehensive plots for each action timestep using parallel processing.
    
    This is in ADDITION to the original XYZ plots - it uses the COMPLETE action vector.
    
    Args:
        random_points: Initial random Gaussian latent points
        trajectories_data: List of trajectory data containing flow_x_values
        labels: Class labels for each point (0=apple, 1=mug, 2=sushi)
        label_names: Dictionary mapping label indices to names
        colors: Dictionary mapping label indices to colors
        exp_idx: Experiment index
        output_dir: Output directory for saving plots
    """
    output_dir = Path(output_dir)
    dimred_dir = output_dir / "dimensionality_reduction"
    dimred_dir.mkdir(parents=True, exist_ok=True)
    
    num_flow_steps = len(trajectories_data[0]['flow_x_values'])
    pred_horizon = trajectories_data[0]['flow_x_values'][0].shape[1]
    
    print(f"\n" + "="*80)
    print(f"Creating PCA ANALYSIS plots for {num_flow_steps} flow steps and {pred_horizon} action timesteps")
    print(f"Using COMPLETE action vector (not just XYZ)")
    print("="*80 + "\n")
    
    # Determine number of processes - reduced to prevent hanging
    num_cores = mp.cpu_count()
    max_workers = max(1, min(4, pred_horizon))  # Limit to max 4 workers to prevent hanging
    
    print(f"Using {max_workers} parallel workers out of {num_cores} available cores")
    
    # Prepare arguments for parallel processing
    dimred_args = [
        (action_step, trajectories_data, labels, label_names, colors, exp_idx, dimred_dir, num_flow_steps, pred_horizon)
        for action_step in range(pred_horizon)
    ]
    
    # Generate dimensionality reduction plots in parallel
    print("🚀 Starting parallel dimensionality reduction plot generation...")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_action = {
            executor.submit(create_dimensionality_reduction_plot_for_action_step, args): args[0] 
            for args in dimred_args
        }
        
        for future in as_completed(future_to_action, timeout=60):  # Add 60 second timeout per task
            action_step = future_to_action[future]
            try:
                result = future.result(timeout=60)  # Add timeout for result retrieval
                print(f"✅ {result}")
            except Exception as exc:
                print(f"❌ Dimensionality reduction plot for action step {action_step} generated an exception: {exc}")
                # Continue with other tasks even if one fails
    
    main_plot_path = dimred_dir / f"dimred_analysis_exp{exp_idx:02d}_action00.png"
    print(f"\n✅ Completed dimensionality reduction analysis for all {pred_horizon} action timesteps")
    print(f"📁 All dimensionality reduction plots saved in: {dimred_dir}")
    
    return main_plot_path


# Integrate with existing function
def plot_flow_matching_analysis(random_points, trajectories_data, labels, label_names, colors, exp_idx, output_dir):
    """
    Plot comprehensive flow matching analysis showing how latent points evolve through flow timesteps.
    Creates BOTH original XYZ plots AND dimensionality reduction plots.
    
    Args:
        random_points: Initial random Gaussian latent points
        trajectories_data: List of trajectory data containing flow_x_values
        labels: Class labels for each point (0=apple, 1=mug, 2=sushi)
        label_names: Dictionary mapping label indices to names
        colors: Dictionary mapping label indices to colors
        exp_idx: Experiment index
        output_dir: Output directory for saving plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    num_flow_steps = len(trajectories_data[0]['flow_x_values'])
    pred_horizon = trajectories_data[0]['flow_x_values'][0].shape[1]
    
    print(f"Creating plots for {num_flow_steps} flow steps and {pred_horizon} action timesteps using parallel processing")
    
    # Determine number of processes
    num_cores = mp.cpu_count()
    max_workers = max(1, min(num_cores - 2, pred_horizon))
    
    print(f"Using {max_workers} parallel workers out of {num_cores} available cores")
    
    # # ==================== ORIGINAL XYZ PLOTS ====================
    # # Prepare arguments for parallel processing
    # plot_2d_args = [
    #     (action_step, trajectories_data, labels, label_names, colors, exp_idx, output_dir, num_flow_steps, pred_horizon)
    #     for action_step in range(pred_horizon)
    # ]
    
    # plot_3d_args = [
    #     (action_step, trajectories_data, labels, label_names, colors, exp_idx, output_dir, num_flow_steps, pred_horizon)
    #     for action_step in range(pred_horizon)
    # ]
    
    # # Generate 2D plots in parallel
    # print("🚀 Starting parallel 2D plot generation...")
    # with ProcessPoolExecutor(max_workers=max_workers) as executor:
    #     future_to_action = {executor.submit(create_2d_plot_for_action_step, args): args[0] for args in plot_2d_args}
        
    #     for future in as_completed(future_to_action):
    #         action_step = future_to_action[future]
    #         try:
    #             result = future.result()
    #             print(f"✅ {result}")
    #         except Exception as exc:
    #             print(f"❌ 2D plot for action step {action_step} generated an exception: {exc}")
    
    # # Generate 3D plots in parallel
    # print("🚀 Starting parallel 3D plot generation...")
    # with ProcessPoolExecutor(max_workers=max_workers) as executor:
    #     future_to_action = {executor.submit(create_3d_plot_for_action_step, args): args[0] for args in plot_3d_args}
        
    #     for future in as_completed(future_to_action):
    #         action_step = future_to_action[future]
    #         try:
    #             result = future.result()
    #             print(f"✅ {result}")
    #         except Exception as exc:
    #             print(f"❌ 3D plot for action step {action_step} generated an exception: {exc}")
    
    # # Generate combined 3D trajectory plot
    # print("🚀 Creating single 3D trajectory plot with multiple flow timesteps...")
    # selected_flow_steps = key_timesteps
    
    # try:
    #     result = create_combined_3d_trajectory_plot(
    #         selected_flow_steps, trajectories_data, labels, label_names, colors, exp_idx, output_dir, pred_horizon
    #     )
    #     print(f"✅ {result}")
    # except Exception as exc:
    #     print(f"❌ Combined 3D trajectory plot generated an exception: {exc}")
    
    # # ==================== PCA ANALYSIS PLOTS ====================
    # print("\n" + "="*80)
    # print("NOW GENERATING PCA ANALYSIS PLOTS")
    # print("="*80)
    
    try:
        dimred_path = plot_flow_matching_analysis_with_dimred(
            random_points, trajectories_data, labels, label_names, colors, exp_idx, output_dir
        )
        print(f"✅ Dimensionality reduction analysis completed")
    except Exception as exc:
        print(f"❌ Dimensionality reduction analysis generated an exception: {exc}")
    
    # Return the first action step plot path as the main result
    main_plot_path = output_dir / f"flow_matching_analysis_exp{exp_idx:02d}_action00.png"
    print(f"\n✅ Completed ALL analyses:")
    print(f"   - XYZ 2D/3D plots for all {pred_horizon} action timesteps")
    print(f"   - Combined 3D trajectory plot")
    print(f"   - PCA analysis plots")
    print(f"📁 All plots saved in: {output_dir}")
    
    return main_plot_path


if __name__ == "__main__":
    main()
    simulation_app.close()