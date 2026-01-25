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
task_name = "dsrl-flow-uniform-mixed"
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
            random_points =  np.array([
[[  -0.20430586,    0.18881321,   -0.15020830,    0.03346069,   -0.03395078,   -0.10678385,   -0.07913022],
 [   0.06972742,   -0.74277450,    0.46539890,   -0.03252243,    0.05512556,    0.01777178,    0.17037477],
 [  -0.08467364,   -0.52409005,    0.25289524,   -0.03892860,   -0.06629990,   -0.19417424,    0.84849316],
 [  -0.09151687,   -0.36288800,    0.15770307,    0.00743554,    0.04148683,   -0.06831631,    0.45973676],
 [  -0.27175114,   -0.67692596,   -0.12191877,   -0.08463547,   -0.09790039,   -0.01836276,    0.57484870],
 [  -0.15173283,   -0.02695072,    0.25115275,    0.00329302,    0.00499536,    0.05064684,    0.73949546],
 [   0.02495143,   -0.62784970,    0.16489002,   -0.02384815,   -0.04120745,    0.01453772,    0.46577233],
 [  -0.08358225,   -0.62785020,    0.38143570,   -0.01910935,    0.04895116,   -0.19829407,    0.61456620]],
[[  -0.17830129,   -0.41278765,   -0.03971326,   -0.03154824,   -0.03601603,   -0.06321138,    0.42796570],
 [   0.17687428,   -0.09133524,   -0.18469670,    0.01793607,   -0.07887711,    0.05005953,    0.77286786],
 [   0.11874551,   -0.74997130,    0.34219295,   -0.02056660,    0.03764854,    0.02653983,   -0.07717563],
 [   0.20150185,    0.24411261,    0.20118389,   -0.00132952,    0.03426296,   -0.15469710,    0.31342490],
 [   0.22715270,   -0.14104390,    0.77075744,   -0.07448010,   -0.03845364,   -0.07117556,    0.93681580],
 [  -0.05896023,   -0.48504835,    0.72957486,   -0.01762562,    0.04881671,   -0.01675174,    0.69937086],
 [  -0.29516032,   -0.30670744,    0.70964370,   -0.09219506,   -0.03371727,    0.00308372,    0.75591385],
 [  -0.00752270,   -0.22405165,   -0.16207197,   -0.00351356,    0.05231863,    0.03374133,    0.71189720]],
[[  -0.07093480,   -0.76564753,   -0.24416159,   -0.04871570,   -0.06381401,   -0.15648389,    0.74101130],
 [  -0.31617093,   -0.38652690,    0.54957790,   -0.06245656,    0.03766477,   -0.14930070,   -0.07717441],
 [  -0.10606004,   -0.38593340,   -0.10620901,    0.02289417,    0.05027603,   -0.22067447,    0.28179073],
 [  -0.17523237,   -0.79569740,    0.34246200,   -0.06590830,   -0.02862363,   -0.08507152,    0.58267725],
 [   0.07029653,   -0.64680990,    0.74122890,   -0.01286358,    0.05352062,   -0.21647420,    0.29134960],
 [  -0.12411305,   -0.71257615,    0.67245550,    0.03232855,   -0.03254917,    0.06292811,    0.98965126],
 [   0.18921185,    0.23821247,    0.64219004,   -0.03846483,    0.01263861,   -0.12180211,    0.52609897],
 [   0.12598991,   -0.63363320,    0.50053170,   -0.01476071,   -0.02766184,   -0.05978736,    0.16058671]],
[[  -0.23086484,    0.23132944,    0.10364330,    0.06251064,    0.02140723,   -0.20990634,    0.03908543],
 [  -0.08221039,   -0.29067850,    0.49552453,   -0.03668814,    0.02829826,   -0.05907618,   -0.02771428],
 [   0.13649935,   -0.02998030,   -0.04697752,   -0.03568697,    0.00714345,   -0.03724734,    0.68117930],
 [  -0.26989985,   -0.75053060,   -0.22210205,   -0.04875030,    0.05081038,   -0.11751281,    0.28208053],
 [  -0.16082755,    0.26184332,    0.33700060,   -0.03401392,    0.01921114,   -0.10965633,    0.05374980],
 [  -0.24313325,   -0.76287010,    0.06655681,    0.00548960,   -0.00773678,    0.06136468,    0.57780105],
 [  -0.01338884,   -0.43696890,    0.10965088,   -0.01485574,    0.04243898,    0.03029755,    0.88029593],
 [   0.14388797,   -0.14541155,    0.52206653,    0.01152866,    0.05783775,   -0.08250259,    0.48120898]],
[[   0.01074484,   -0.61584973,   -0.08311392,   -0.03699339,    0.03837363,    0.05751362,    0.81593670],
 [  -0.30504563,   -0.09678179,    0.07714474,    0.00468257,   -0.05767180,   -0.06667683,    0.26331192],
 [  -0.05948284,   -0.77486060,   -0.04058473,    0.03775588,   -0.08359997,   -0.22732438,    0.61151576],
 [  -0.23653320,   -0.84385150,    0.60513896,   -0.08749668,    0.02126960,   -0.12816986,    0.88380010],
 [  -0.05458996,    0.22380460,    0.59565914,   -0.02343703,    0.01779578,   -0.10643449,    0.72463840],
 [  -0.08549395,    0.18617392,    0.22292945,    0.04762281,    0.03393264,   -0.13859475,    0.39023727],
 [   0.05180934,    0.17154491,    0.72932094,   -0.06354540,   -0.06552349,   -0.21741892,    0.20008124],
 [  -0.25811250,   -0.09738439,    0.36821693,    0.00196702,   -0.04469544,    0.06951353,    0.55212830]],
[[  -0.30188015,   -0.77667770,   -0.07906273,   -0.06197188,   -0.03132603,   -0.16072795,    0.93914300],
 [   0.19032460,   -0.64251660,    0.76786160,   -0.00527336,   -0.04736213,   -0.12984881,   -0.05268234],
 [   0.24277633,    0.17167080,    0.72400220,   -0.01817980,   -0.03593661,   -0.04091188,    0.70689684],
 [  -0.12244359,    0.25037408,   -0.00068903,   -0.05147896,   -0.09356318,   -0.10499673,    0.60328466],
 [  -0.29845488,    0.03127050,    0.56554820,   -0.08525977,   -0.07474425,    0.01650107,   -0.03026794],
 [   0.13382077,   -0.55959530,    0.39644045,   -0.04490709,    0.02711014,   -0.20992404,    0.50910190],
 [  -0.01937672,    0.01528382,    0.45616330,    0.05228826,   -0.04719206,   -0.18051350,    0.21817054],
 [   0.14044386,   -0.64237285,    0.13561279,    0.02437856,   -0.09871609,   -0.15346187,    0.76993775]],
[[   0.15012082,   -0.76304730,    0.35810792,   -0.02150676,    0.00855903,   -0.21007656,    0.32395053],
 [   0.28131747,   -0.14248526,    0.47760790,    0.03551279,    0.02255365,    0.03678387,    0.00764041],
 [  -0.02035096,   -0.39092904,    0.42135150,   -0.08987754,    0.00245334,   -0.05222812,    0.08934678],
 [   0.19527006,   -0.73334700,   -0.20631531,   -0.05204460,   -0.08452279,   -0.01731020,    0.63650500],
 [  -0.20927867,   -0.05744082,   -0.08980593,   -0.02954351,    0.01313989,   -0.04628749,    0.83827930],
 [  -0.16372491,   -0.80223070,   -0.12357847,    0.03289865,   -0.08980864,    0.01160675,    0.57750374],
 [  -0.07314077,    0.19036174,   -0.03770867,   -0.01401684,   -0.03298736,   -0.09502900,    0.38810910],
 [   0.13941252,   -0.14057880,    0.28540087,    0.06017995,    0.04333781,    0.01915568,    0.03523600]],
[[   0.14093366,   -0.03271979,    0.20775574,    0.00049262,   -0.00897307,   -0.01936159,    0.76858180],
 [  -0.07911777,   -0.41864350,    0.38310194,    0.05190343,   -0.01909504,   -0.15439859,    0.73711240],
 [  -0.11397696,   -0.08083731,   -0.00749350,   -0.04624745,   -0.06379329,   -0.00135253,    0.50564550],
 [  -0.20010500,   -0.42708560,    0.50696343,   -0.04224915,   -0.06438667,   -0.11139203,    0.13240102],
 [   0.08479142,    0.25018300,   -0.14872852,   -0.03658973,   -0.09378131,   -0.12031886,    0.59554166],
 [   0.17219311,   -0.64356405,   -0.09033325,   -0.08752032,   -0.00932720,   -0.14360291,    0.41868854],
 [  -0.19505477,   -0.61960700,    0.68182206,   -0.01574095,   -0.00903944,   -0.18348914,    0.22718693],
 [   0.25755906,   -0.44221708,    0.22076488,   -0.08689883,   -0.06976320,   -0.07519136,    0.97088640]],
[[  -0.16556655,    0.25858128,    0.23558813,   -0.02863634,   -0.04712275,   -0.11482791,    0.83360213],
 [  -0.05565059,   -0.47035646,    0.41252910,   -0.00342765,    0.03866853,   -0.10682131,    0.85976290],
 [  -0.27509004,   -0.26396507,    0.78341030,    0.04336359,    0.04377528,   -0.04909961,    0.85473010],
 [   0.03756404,    0.33315504,    0.14527917,   -0.06820690,   -0.07896376,   -0.18764314,    0.40850842],
 [  -0.18106356,   -0.66854200,    0.25427067,    0.00356194,   -0.00754588,   -0.21118450,    0.81996010],
 [  -0.08510219,   -0.60272765,    0.64809780,   -0.08421636,    0.02414660,   -0.14462003,    0.77787644],
 [  -0.30564234,   -0.04531866,    0.41939986,    0.04430872,   -0.08728611,   -0.16716835,    0.39005953],
 [  -0.27729034,    0.15296924,    0.76208720,   -0.05372937,   -0.09538925,   -0.14151329,    0.88986920]],
[[   0.06889126,   -0.32726884,    0.34871858,   -0.08547863,    0.00462414,   -0.06319647,    0.24897142],
 [   0.09354761,   -0.68276170,   -0.05358167,   -0.04709846,   -0.00843228,    0.00697643,    0.27129042],
 [   0.25333834,   -0.07633483,    0.58298860,   -0.06666768,   -0.02424288,   -0.07327688,    0.58100910],
 [   0.22586775,   -0.07750857,    0.30003726,   -0.04303271,    0.04436915,   -0.01765934,    0.88546880],
 [  -0.21715972,    0.01352906,    0.27834260,    0.01035122,   -0.05468008,   -0.17601517,    0.74136420],
 [   0.20063013,   -0.44492805,    0.07334793,   -0.05011061,   -0.04762164,   -0.01837470,    0.27766806],
 [   0.19005000,   -0.31978744,   -0.12602057,    0.04109089,   -0.02032295,   -0.03161117,    0.71318036],
 [  -0.06491381,   -0.42682790,   -0.06083852,    0.01996417,   -0.04402017,    0.05593571,    0.79509870]],
[[  -0.16343000,   -0.50339985,    0.73231965,    0.00184739,   -0.09559049,    0.01150921,    0.63440555],
 [   0.22727930,    0.00231034,   -0.23871204,   -0.03362161,    0.05454670,   -0.15945241,    0.45035344],
 [   0.16044447,   -0.04589820,   -0.17196722,    0.04101114,   -0.06660336,   -0.13358386,    0.09999399],
 [  -0.09459637,   -0.81389034,    0.44681500,    0.05049285,   -0.05914922,    0.07029092,    0.55777110],
 [   0.07570583,   -0.23120868,   -0.16296000,   -0.08596447,    0.02136132,   -0.16444233,    0.00586710],
 [  -0.28573602,   -0.41388756,    0.01930752,   -0.07065405,   -0.00230730,    0.03798637,    0.36468434],
 [  -0.17541844,   -0.06376678,    0.59001500,    0.01565496,    0.03207251,   -0.11996698,    0.22567300],
 [   0.15196732,   -0.04857475,    0.63152784,   -0.03588240,   -0.04794640,   -0.19625053,    0.97798880]],
[[   0.11982375,   -0.70399320,   -0.03807175,    0.02515202,    0.04871842,   -0.05538093,    0.40822600],
 [  -0.32910576,   -0.29221922,    0.06243333,   -0.06246828,    0.04882467,   -0.12108997,    0.86655340],
 [  -0.07919756,   -0.47565883,    0.29695195,   -0.00765168,   -0.06955257,   -0.09515248,    0.92804690],
 [  -0.01155296,   -0.71628240,   -0.23888992,   -0.05929218,   -0.03761228,   -0.18990720,    0.66517700],
 [  -0.09290899,   -0.55356884,    0.63653725,    0.05369951,   -0.04992613,   -0.04229972,    0.30695170],
 [  -0.19909468,   -0.20664614,    0.56305980,    0.00664017,   -0.01027131,   -0.06814727,    0.86850744],
 [   0.11271176,   -0.30881852,   -0.02414794,    0.04988033,   -0.03374861,   -0.09831440,    0.68394880],
 [   0.24604827,   -0.79803960,    0.20272654,   -0.05790465,   -0.06811009,    0.03724679,    0.77357110]],
[[   0.18722856,   -0.77976197,   -0.05206889,   -0.00562014,    0.00782117,   -0.09133403,    0.65266940],
 [  -0.18848634,   -0.06614339,    0.24902266,    0.02211064,   -0.05254119,    0.05580154,    0.98952120],
 [  -0.09466623,   -0.09190387,    0.09930530,   -0.07294041,   -0.03442172,   -0.07816961,    0.76290180],
 [  -0.14860280,    0.06725818,    0.55079675,   -0.05955645,   -0.01527759,   -0.20076117,    0.77348810],
 [  -0.29670566,    0.32450044,   -0.14163536,    0.01446849,   -0.01652844,   -0.15964547,    0.93558186],
 [   0.26320190,   -0.00561249,    0.08919922,    0.05967087,    0.04204930,   -0.17518260,    0.10482678],
 [  -0.06344721,   -0.51191914,    0.20415786,   -0.00335335,   -0.05687539,   -0.15380910,    0.67276260],
 [  -0.00118014,    0.02074617,    0.66779400,   -0.08736207,   -0.01793835,   -0.13804469,    0.42916363]],
[[   0.17324030,   -0.72593580,    0.58345210,   -0.01051868,    0.01819997,   -0.21735522,    0.73753816],
 [   0.26856673,   -0.09793454,    0.28958982,    0.05869463,   -0.00015268,   -0.15367441,    0.21688114],
 [   0.19840854,   -0.70592560,    0.34139508,   -0.09128194,   -0.04598855,    0.05545753,    0.71677070],
 [   0.27655750,   -0.51919746,    0.72371270,    0.05813392,    0.04268155,   -0.04034397,    0.85816056],
 [  -0.15526763,   -0.48103303,    0.49665630,    0.02050649,   -0.00935766,   -0.23044670,    0.72196424],
 [   0.16140103,   -0.40911418,   -0.00797933,    0.01742372,   -0.02735995,   -0.05295561,    0.61458670],
 [  -0.13309158,   -0.74470615,    0.29740816,    0.02254433,   -0.01410808,   -0.17044663,    0.58746190],
 [   0.20556330,   -0.25142598,   -0.10807914,    0.00824698,   -0.08849292,   -0.23337421,    0.45449036]],
[[  -0.11501041,   -0.44433826,   -0.04501916,    0.00102868,   -0.00510993,   -0.03056620,    0.52995646],
 [   0.00336078,   -0.44135353,   -0.22142743,   -0.01950889,   -0.02742115,   -0.00579385,    0.89979410],
 [  -0.09846209,   -0.71824860,    0.43892348,   -0.04977145,    0.02805580,   -0.10823201,    0.00694669],
 [   0.19508314,    0.34906697,    0.76426890,    0.04432793,   -0.09864007,   -0.14121127,    0.20440154],
 [   0.19525737,    0.01515806,    0.42423457,    0.06254525,   -0.06201826,   -0.17426176,   -0.02221942],
 [  -0.03717330,   -0.56019000,    0.53219910,    0.00323264,   -0.08941542,    0.06194928,   -0.00147844],
 [  -0.13916382,    0.07033002,    0.74971470,   -0.04278439,    0.03493637,   -0.07773313,    0.37788750],
 [   0.19249558,    0.32259655,    0.16862291,   -0.05135947,   -0.02467135,    0.02454597,    0.18027048]],
[[   0.01952031,   -0.35263065,    0.01275370,    0.02539018,   -0.06417309,   -0.03123710,    0.23122461],
 [  -0.30558705,    0.17547154,    0.70474730,    0.00510848,   -0.02506755,   -0.19519350,    0.06765115],
 [   0.21444029,   -0.39513840,   -0.02060767,   -0.07780029,    0.02652703,   -0.10785592,    0.88123846],
 [  -0.25004420,   -0.05723816,   -0.16212697,   -0.08387921,   -0.03560679,   -0.05618632,    0.72235250],
 [  -0.22818267,    0.00693095,    0.07859722,   -0.04767946,   -0.07285530,   -0.21754658,    0.80258470],
 [  -0.24432309,    0.09286356,    0.55514956,   -0.08341551,   -0.00842037,    0.00083449,    0.81909823],
 [   0.00080833,   -0.08350945,    0.74747914,   -0.08216050,   -0.08390536,   -0.18384904,    0.74902680],
 [  -0.30014467,    0.06454909,    0.43194443,   -0.05341148,    0.02319445,    0.04625639,    0.57058823]],
[[  -0.22500858,    0.09877270,    0.64038610,   -0.03746019,   -0.09307027,   -0.06152068,    0.45558375],
 [   0.24745160,    0.14437490,    0.55025077,   -0.08449349,   -0.01745591,   -0.05746071,    0.92105620],
 [   0.12129745,   -0.64239950,    0.54915850,    0.01470047,    0.00271151,   -0.15187311,    0.07787640],
 [  -0.13489956,   -0.00817192,    0.74770510,    0.01135787,   -0.04109503,   -0.06912361,    0.55419147],
 [  -0.25627315,   -0.50047980,    0.15743732,    0.02699492,   -0.07725984,   -0.15812168,    0.17588265],
 [   0.23997670,   -0.83929090,    0.10354975,   -0.01997681,   -0.05947630,   -0.21106549,   -0.00375499],
 [  -0.29013956,   -0.15988850,    0.00417277,    0.02472948,    0.05053550,    0.06181309,    0.55769706],
 [  -0.24569076,   -0.05868977,    0.50785834,   -0.05351311,   -0.06166555,    0.00881280,    0.04191408]],
[[   0.25684553,   -0.43301657,   -0.19781092,   -0.06479564,   -0.00819822,   -0.21727806,    0.79300654],
 [  -0.05336779,   -0.66573450,    0.16184041,   -0.01117194,    0.05428632,   -0.14777714,    0.51090840],
 [  -0.29374358,    0.12852120,    0.63322985,   -0.00872208,    0.04420428,   -0.03835948,    0.08609773],
 [   0.17141944,    0.14573490,   -0.18601874,    0.03088365,    0.03471912,    0.04146978,    0.70508970],
 [  -0.20534053,   -0.08840322,    0.77179134,   -0.00216901,   -0.00566226,   -0.15752344,    0.84305550],
 [  -0.03256807,   -0.68100870,    0.56076074,   -0.08476179,    0.05351953,   -0.21440856,    0.45718074],
 [   0.15370706,   -0.50146854,    0.14723998,    0.01613537,    0.01206664,   -0.00281978,    0.29446715],
 [   0.17920429,   -0.52781737,    0.12117201,    0.05952892,   -0.03741067,   -0.04907924,    0.04327065]],
[[  -0.22793815,   -0.15565866,    0.06469160,   -0.01413837,   -0.01789266,    0.00623806,    0.51846770],
 [   0.07226628,    0.30972517,    0.53468570,   -0.01913233,   -0.02323420,   -0.10774565,    0.29076332],
 [   0.18152934,   -0.32912660,    0.36338400,   -0.06792620,   -0.05667727,    0.05785540,    0.16212367],
 [  -0.32877308,    0.28889716,    0.21597674,   -0.03657425,   -0.04933162,    0.01377313,    0.17986302],
 [   0.14636973,    0.10795134,    0.75532436,    0.04065999,   -0.08827571,   -0.05005856,    0.87130445],
 [  -0.19680487,   -0.20197958,    0.73625900,    0.03675753,    0.01122375,   -0.11584164,    0.17410960],
 [  -0.25505188,   -0.13426036,    0.17840388,   -0.06578945,   -0.00014443,   -0.14842048,    0.98776380],
 [  -0.01462194,    0.04717046,    0.66454420,   -0.07026562,   -0.06405893,    0.01990816,    0.44042212]],
[[   0.26319290,   -0.07425225,   -0.16610889,   -0.04795305,    0.05500160,   -0.19523224,    0.41103740],
 [   0.17726165,   -0.75265414,   -0.02299969,    0.01630329,    0.03494829,   -0.21625390,    0.41251850],
 [   0.20719397,    0.13953954,    0.69963920,   -0.05485198,    0.02888133,   -0.03068481,    0.55816936],
 [  -0.10059014,    0.25380838,    0.68940747,    0.04191579,    0.03681193,   -0.17805715,    0.20787467],
 [  -0.22508904,   -0.15932834,    0.10814810,    0.04285105,    0.03746940,   -0.18257944,    0.31286645],
 [  -0.05857027,   -0.51928806,    0.12784955,   -0.06113245,    0.03167614,    0.02889225,    0.17805837],
 [   0.12423444,   -0.58615040,   -0.23026922,    0.04821563,   -0.04732575,    0.00160131,    0.94414730],
 [   0.04622558,   -0.47902320,   -0.07966287,    0.05649275,   -0.08608200,   -0.19633986,    0.95715550]],
[[  -0.19756730,    0.26039636,   -0.02976564,    0.00422491,   -0.08606663,    0.06601098,    0.40379786],
 [  -0.13496240,   -0.83419940,    0.45976037,   -0.06489849,   -0.04511813,   -0.06053297,    0.96009000],
 [   0.16391781,    0.16628039,    0.33893216,    0.01994175,   -0.05406594,   -0.08614361,    0.00062937],
 [  -0.14680001,   -0.18111813,    0.47757173,   -0.04692153,    0.01924479,   -0.21485837,    0.13914432],
 [  -0.17895680,   -0.70079260,    0.02963173,   -0.03882325,    0.05507854,   -0.09285115,    0.16739203],
 [  -0.17380560,   -0.04244280,    0.71938370,   -0.00540585,   -0.06370211,   -0.05392563,    0.55566466],
 [  -0.30520958,   -0.10276937,    0.71293896,   -0.01258969,   -0.00716005,   -0.16960514,    0.24536638],
 [  -0.04467756,   -0.80717045,    0.74012590,    0.05385856,   -0.06930247,   -0.07648937,    0.77602840]],
[[   0.15487379,   -0.34827614,   -0.03028958,   -0.03288319,   -0.07727811,   -0.18935606,    0.33140486],
 [   0.22655260,   -0.24764192,    0.63611910,   -0.00625422,   -0.03332914,   -0.04206365,    0.31641918],
 [  -0.26858154,   -0.37521186,    0.05718458,   -0.04594526,   -0.06738329,    0.07063472,    0.10863838],
 [   0.12948778,    0.01566291,    0.72051090,   -0.02389012,   -0.04155592,    0.02044961,    0.16544509],
 [  -0.05826700,    0.23014080,    0.33636850,    0.05819291,   -0.08085962,    0.05493715,    0.46996260],
 [   0.20198971,   -0.35475734,    0.60221344,    0.02532978,    0.04317030,   -0.00513417,    0.35992068],
 [  -0.08674742,   -0.30295837,    0.66675880,   -0.07571767,   -0.02210494,   -0.02623358,    0.31845087],
 [   0.25612790,   -0.39169675,    0.45505673,   -0.00992426,   -0.07924454,    0.00876687,    0.45818436]],
[[  -0.17263580,   -0.08569026,    0.67173386,    0.06075746,   -0.05866537,    0.00024311,    0.83035580],
 [   0.19414765,   -0.43027710,    0.51663850,   -0.08828879,   -0.05464153,    0.00139146,    0.67042816],
 [  -0.09654102,   -0.40186614,   -0.03344750,   -0.03878156,   -0.07314311,   -0.15868425,    0.29211128],
 [   0.06207231,   -0.32945222,    0.31876117,   -0.04509022,   -0.00066507,   -0.11754050,    0.79356885],
 [   0.13449308,    0.19886374,    0.61940950,   -0.06620245,    0.05083565,    0.06658792,    0.98959140],
 [   0.10064021,    0.15763354,    0.58547070,    0.01320879,   -0.02155293,   -0.09035856,    0.65856640],
 [  -0.18998304,   -0.39389540,    0.37702173,   -0.04045094,   -0.05454488,   -0.20132978,    0.02786016],
 [  -0.02556613,   -0.00934219,    0.42053170,    0.02302237,   -0.04760791,   -0.13246864,    0.81039274]],
[[   0.12545237,   -0.19700927,    0.55102456,   -0.00447651,   -0.01244614,    0.06513670,    0.67004645],
 [   0.09883794,    0.06059343,   -0.04467022,    0.04671967,    0.04311207,   -0.07814284,   -0.04067506],
 [   0.26245892,   -0.03921837,    0.21749255,   -0.00107718,    0.04162487,   -0.02722362,    0.39331317],
 [   0.05667850,   -0.49434164,    0.69395745,   -0.08475414,   -0.06696422,   -0.18028231,    0.12200192],
 [   0.05597398,   -0.51723010,    0.35311258,   -0.01458658,   -0.01715473,   -0.17788520,    0.34268874],
 [  -0.26810664,   -0.02716261,    0.69611675,    0.02573280,    0.05351774,   -0.15599361,    0.63643090],
 [  -0.31223688,   -0.80086434,   -0.08730201,   -0.08986357,   -0.08633576,    0.02211520,    0.33396614],
 [  -0.01342651,   -0.72479780,    0.72392520,    0.06095366,   -0.02042613,   -0.12910321,    0.38559760]],
[[  -0.20561485,   -0.12330526,    0.28459102,    0.05919543,   -0.04249140,   -0.04209778,    0.32668066],
 [   0.08147576,    0.17767310,    0.41661180,    0.02752893,   -0.04758754,   -0.17568964,    0.48251820],
 [  -0.02534935,   -0.31861680,    0.09243774,   -0.03101699,   -0.03767671,   -0.01948132,    0.21695180],
 [  -0.04003203,   -0.56032467,    0.69420004,   -0.03540349,   -0.03633078,   -0.16556288,    0.19380300],
 [   0.03328195,   -0.05153561,   -0.22113551,    0.00612578,    0.03971456,   -0.13268077,    0.04263229],
 [  -0.32113603,    0.22479618,    0.50287306,    0.01022785,    0.00052465,   -0.03242640,    0.77145790],
 [  -0.13735937,   -0.11572838,    0.56085930,   -0.07443443,   -0.03019813,   -0.13435437,    0.81140846],
 [  -0.05334315,   -0.40121958,    0.51011820,   -0.07477328,   -0.09412806,   -0.18632174,   -0.04608236]],
[[   0.24578750,   -0.01589251,   -0.20066763,   -0.01999297,   -0.04988169,   -0.15240672,   -0.01511788],
 [   0.19776475,    0.24550533,    0.63687146,   -0.08427143,   -0.03499009,   -0.15400152,    0.34196615],
 [  -0.02461135,   -0.04728740,   -0.01088594,    0.00822376,    0.01284716,   -0.04432194,    0.40462077],
 [  -0.12038088,    0.17960024,    0.20149139,   -0.09089339,    0.03326541,   -0.11917099,    0.61890780],
 [   0.24547774,   -0.06109571,    0.34780854,    0.04500872,    0.00830323,   -0.19082037,    0.72379010],
 [   0.01537189,   -0.53892946,    0.76371574,   -0.07060462,   -0.05952241,   -0.09621547,   -0.01201138],
 [  -0.04249895,   -0.28827488,   -0.03743750,   -0.03904352,   -0.00052313,   -0.14544490,    0.76772550],
 [   0.04057744,    0.34234655,    0.62546110,   -0.06310298,    0.00779261,   -0.09137936,    0.01264028]],
[[   0.04502729,   -0.68447727,    0.75314856,   -0.03356045,    0.01045117,   -0.01586688,    0.30944680],
 [  -0.16151306,    0.22958767,    0.14504299,   -0.05706061,    0.03525151,   -0.03362113,    0.38270170],
 [   0.03784078,    0.13369322,    0.00723487,   -0.00494866,   -0.07876495,    0.04187062,    0.18187653],
 [  -0.33305654,    0.27421570,    0.05956912,    0.01369944,    0.02383073,   -0.16981271,    0.47845490],
 [  -0.21860974,   -0.76833844,   -0.18788627,    0.00689846,    0.05293299,   -0.06353372,    0.52879390],
 [   0.16199318,   -0.37749810,    0.47150190,    0.00908792,    0.04247396,   -0.03148480,    0.00800237],
 [  -0.30190536,   -0.67206850,    0.40330243,   -0.08282189,    0.00655197,    0.04195997,    0.32840210],
 [   0.18130368,    0.00386226,    0.64029443,   -0.06956390,   -0.02711484,   -0.00090307,    0.83804700]],
[[  -0.30314470,    0.02405328,   -0.14480706,   -0.02501795,   -0.03567327,   -0.05881141,    0.07032652],
 [  -0.05502042,    0.29187346,    0.23982564,    0.02770220,    0.05463493,   -0.15812942,    0.82373290],
 [   0.01899078,    0.33124554,    0.70255345,   -0.09338806,   -0.04394927,   -0.08845890,    0.26389897],
 [  -0.21102956,   -0.72783140,   -0.04931913,   -0.05724558,   -0.04442206,   -0.15268332,    0.82717390],
 [   0.20522922,   -0.46793136,    0.35340422,   -0.01209668,   -0.05906872,   -0.03923869,    0.57930790],
 [  -0.11884595,   -0.82255745,    0.35286033,   -0.07922197,   -0.02447531,    0.04447213,    0.71396900],
 [   0.01823902,    0.36893713,   -0.08194348,   -0.02688441,   -0.07886185,    0.00159237,    0.44612336],
 [  -0.32922438,   -0.75521690,    0.13291934,   -0.05915911,   -0.05619175,   -0.12078124,    0.13822988]],
[[   0.09578681,    0.27248847,    0.17954543,    0.01084952,    0.03873017,   -0.23120627,    0.60989916],
 [   0.16589758,   -0.74820435,    0.42682356,   -0.04834256,    0.02884220,   -0.19591191,    0.06334402],
 [   0.26399356,   -0.64452386,    0.38829505,   -0.05546289,   -0.05952719,    0.04482526,    0.15765716],
 [  -0.01506928,   -0.24712878,    0.40545052,   -0.04631812,   -0.07412763,   -0.13995700,    0.13843545],
 [   0.04887038,   -0.10913682,    0.62012583,    0.04098304,   -0.05024412,   -0.20903192,    0.21687533],
 [  -0.14160377,    0.04287910,    0.32400770,   -0.03716553,   -0.07696708,   -0.11495104,    0.35806662],
 [  -0.11265546,   -0.66720295,    0.64707094,   -0.07217936,    0.00477280,   -0.14638308,    0.54493034],
 [   0.13446432,   -0.20021856,    0.72182450,   -0.07852716,    0.01042276,    0.05756494,    0.01466054]],
[[   0.06279767,   -0.14742172,    0.75893915,    0.06076816,    0.04637924,    0.02376214,   -0.02819247],
 [  -0.03366616,    0.36117960,    0.10525632,    0.00517975,    0.03683735,    0.02783504,    0.68864346],
 [  -0.03956062,    0.12261742,    0.02253497,   -0.04584673,   -0.01670189,   -0.13704452,    0.72029227],
 [  -0.22738788,    0.32947123,   -0.16937852,   -0.00835335,   -0.05174292,   -0.22357269,    0.48229933],
 [  -0.17788490,   -0.65899830,   -0.09029315,   -0.03519173,   -0.07390741,   -0.04581107,    0.60675330],
 [  -0.32008988,   -0.73866030,    0.17327640,   -0.03313826,    0.04945858,   -0.21602683,    0.97311120],
 [   0.12769881,   -0.83470580,   -0.06430775,   -0.09383234,   -0.01582901,    0.02837792,    0.60529650],
 [  -0.12337865,    0.09653038,    0.05339104,    0.03254662,   -0.05750151,   -0.16546726,    0.17148621]],
[[  -0.08900256,   -0.20465457,    0.67274790,   -0.08505397,    0.04268825,   -0.23548132,    0.42331600],
 [   0.07182005,   -0.79128224,    0.05895746,    0.02642436,   -0.02330074,   -0.04712112,    0.49506880],
 [   0.07867110,    0.27800775,    0.24630901,    0.02583069,   -0.08628683,   -0.17964864,    0.18409939],
 [   0.24492067,    0.04461277,    0.25570285,   -0.03972053,    0.03555237,   -0.14809650,    0.34548485],
 [  -0.25340718,    0.06079870,    0.32072120,   -0.05929492,    0.01281728,   -0.12552074,    0.75623435],
 [  -0.14427002,    0.23299325,    0.01834840,   -0.02151200,   -0.09439169,   -0.19411516,    0.03829402],
 [   0.06002131,   -0.63449120,    0.40610135,   -0.02977990,    0.04875642,   -0.16240406,    0.08945557],
 [  -0.07758841,   -0.84243080,   -0.04302497,    0.03340662,   -0.08824814,   -0.13332275,   -0.06291007]],
[[   0.01692560,   -0.35995835,    0.73920320,    0.02695647,   -0.10161604,    0.02861851,    0.53712450],
 [   0.25731438,   -0.29973954,    0.32918787,   -0.08340483,   -0.08213585,   -0.18587540,    0.94114120],
 [  -0.08686468,    0.12342048,    0.19647685,   -0.03871683,   -0.00485054,   -0.13129050,    0.85595226],
 [  -0.25468838,   -0.70152080,    0.18462792,    0.02605195,   -0.03861473,    0.04087681,    0.91061280],
 [  -0.31998140,   -0.06109130,    0.33421373,   -0.03430183,    0.00946295,   -0.16486749,    0.00349297],
 [  -0.30073756,    0.15112996,   -0.12517968,   -0.07876409,   -0.06831068,   -0.20967849,    0.61414010],
 [   0.17589527,   -0.75721870,    0.19461134,   -0.08566615,   -0.05596755,   -0.03216915,    0.63236490],
 [  -0.15560961,   -0.61077356,    0.39412612,   -0.04232323,   -0.02399646,   -0.05476752,    0.01975486]],
[[   0.27829987,   -0.34927470,   -0.00589480,   -0.07184092,   -0.06616452,   -0.23215158,    0.34983766],
 [  -0.20095733,    0.35511494,    0.01673046,   -0.02073109,   -0.01849147,   -0.02081740,    0.00655133],
 [  -0.12112767,    0.24897349,   -0.10882212,   -0.08494944,   -0.06361717,   -0.18581760,    0.60694270],
 [  -0.18322636,   -0.18784225,    0.31941575,   -0.06483142,    0.00787986,    0.05341879,    0.83543050],
 [   0.22916019,   -0.01632845,    0.71204710,   -0.02500146,   -0.05655181,   -0.13593740,    0.13368106],
 [   0.13169384,    0.00174284,    0.74670650,    0.05716813,    0.05135763,   -0.16008171,    0.00691166],
 [   0.16199124,   -0.12044436,   -0.04930143,   -0.05758234,   -0.05735121,   -0.14002260,    0.89998280],
 [   0.26171840,   -0.81654050,    0.75551355,   -0.03330462,    0.05627757,   -0.17687770,    0.29588760]],
[[  -0.16824186,   -0.01769423,    0.02218398,   -0.00598519,   -0.05506811,   -0.22908966,    0.43577492],
 [  -0.18385197,   -0.72056484,   -0.03679825,    0.00879534,   -0.08282177,   -0.03243013,    0.80036706],
 [  -0.10233493,   -0.06789398,   -0.24207300,    0.01275420,    0.02668145,   -0.00634280,    0.20962851],
 [  -0.08425419,   -0.21702349,    0.50685227,   -0.05696731,    0.00735435,    0.04227710,   -0.01400217],
 [  -0.30252898,    0.30979670,    0.60575557,    0.00170813,    0.03722467,    0.01022026,    0.72429070],
 [  -0.05291376,   -0.48800860,    0.67126090,   -0.09072372,   -0.00689488,   -0.08895509,    0.35394913],
 [  -0.10224669,   -0.57144517,    0.54350950,   -0.00812151,   -0.00682126,   -0.18485558,    0.47763997],
 [   0.11576891,   -0.61819930,    0.66690516,   -0.01879258,    0.05089507,   -0.16819766,    0.79117080]],
[[   0.12442121,    0.37085760,    0.26034427,   -0.06440301,   -0.03902816,   -0.05710621,    0.73660170],
 [   0.20556945,   -0.27364695,    0.31060952,    0.03643952,    0.05845873,   -0.04644825,    0.60940367],
 [   0.21376204,   -0.02322775,    0.59710620,    0.00995565,   -0.08437768,   -0.06964141,    0.93923540],
 [   0.27259946,   -0.22971850,    0.20081094,   -0.01941825,   -0.08100640,   -0.09252577,    0.04644129],
 [   0.21822840,   -0.64544830,    0.28555995,    0.00619809,    0.02266680,   -0.00045730,    0.10171296],
 [  -0.23394340,   -0.46219260,    0.56061750,    0.00692499,   -0.00698136,    0.01003698,    0.41476554],
 [  -0.21601586,   -0.19314599,    0.08656743,   -0.00836275,   -0.06529573,   -0.20534012,    0.67091060],
 [   0.19314569,    0.28824460,    0.65327550,    0.03423606,   -0.08332551,   -0.21334746,    0.58005625]],
[[   0.20975077,   -0.40631405,   -0.22261067,    0.04649249,   -0.07358809,   -0.12423065,    0.59386340],
 [  -0.25645798,    0.34996283,   -0.01424289,    0.01108694,   -0.01791309,   -0.22616996,    0.63425130],
 [   0.21448940,   -0.56281040,    0.51927346,    0.05687398,   -0.02800063,   -0.01178530,    0.68077976],
 [  -0.30733836,    0.16499949,   -0.12601903,    0.04643120,   -0.00954153,   -0.18743291,    0.11492519],
 [   0.06556633,   -0.04692894,   -0.10126914,   -0.05114345,    0.00243063,    0.06596595,    0.11485483],
 [  -0.30697660,   -0.32918638,    0.26347023,    0.02667359,   -0.09026027,   -0.15797003,    0.97057396],
 [   0.28898560,   -0.73500340,   -0.00333166,   -0.06078722,   -0.00518164,   -0.02526313,    0.11630949],
 [  -0.04858941,    0.32318747,    0.10607582,    0.01968119,   -0.04008247,   -0.12375601,    0.98424690]],
[[  -0.18963224,    0.18704402,    0.21018583,   -0.05928166,    0.01172265,   -0.08689082,    0.85215030],
 [  -0.04287335,   -0.28011894,    0.40729558,   -0.01150934,   -0.02533408,   -0.17809945,    0.06262764],
 [  -0.05083862,    0.11592740,    0.71220756,   -0.00764109,   -0.05684960,    0.05715451,    0.10009830],
 [   0.00497293,   -0.27318550,    0.59908040,    0.02719260,   -0.02969850,   -0.05581397,    0.41459697],
 [   0.00956804,    0.25218344,   -0.20245817,    0.01017047,    0.02408247,    0.05593747,    0.57559955],
 [  -0.30996484,    0.36617923,    0.37257487,   -0.04265229,    0.03256376,   -0.20809042,    0.51727074],
 [   0.25400424,   -0.49935790,    0.62759080,    0.02568050,    0.01600643,   -0.02246751,   -0.03893177],
 [   0.14134940,    0.10247833,   -0.01131983,    0.01437847,   -0.03392295,   -0.04047343,    0.31104332]],
[[  -0.14697225,   -0.63612650,    0.76948080,    0.06043656,    0.04995286,   -0.00208817,    0.09738079],
 [   0.00992894,   -0.45115570,   -0.18654941,   -0.00815588,   -0.07647155,    0.06444520,    0.34120655],
 [  -0.26334460,    0.10175622,    0.46671999,    0.00270999,    0.04255739,    0.04429215,    0.70129484],
 [  -0.19133778,   -0.80873130,    0.26384622,   -0.06845187,   -0.07956070,   -0.08272408,    0.37121236],
 [   0.24277270,    0.32833445,    0.10697734,   -0.01256001,   -0.02379392,   -0.01573876,    0.16845982],
 [  -0.24626950,    0.28768206,    0.33173817,   -0.04464294,   -0.06940362,   -0.18232733,    0.12255064],
 [   0.25791162,    0.29357470,    0.64526610,   -0.04804179,   -0.04949877,   -0.09782630,    0.12929185],
 [   0.24054056,   -0.71023590,    0.00185272,    0.02256744,   -0.01349660,   -0.21293294,   -0.04590993]],
[[  -0.03185809,    0.32759273,    0.06434500,   -0.01852359,    0.00350861,   -0.06186913,    0.79502860],
 [  -0.21422845,   -0.65107703,    0.72795310,    0.02904773,    0.04667540,   -0.19600977,    0.59948520],
 [   0.04251143,   -0.04977959,    0.65622956,   -0.06636852,   -0.03357615,    0.06728551,    0.26136482],
 [  -0.14162068,    0.06863695,    0.35912532,   -0.02534411,    0.01485007,   -0.21973008,    0.42120040],
 [  -0.12597287,   -0.24124146,   -0.15285820,    0.03438033,    0.04844585,   -0.04325964,    0.44390422],
 [   0.26065832,   -0.62556434,    0.57798624,   -0.07227769,   -0.02661206,   -0.16559088,    0.97155875],
 [  -0.25192398,    0.07550442,    0.39532120,    0.03911553,    0.04785363,    0.04702827,    0.72701347],
 [   0.07677856,    0.08576500,    0.38903016,    0.01744214,   -0.09040069,    0.07036990,    0.88532710]],
[[   0.03537345,    0.07124776,    0.66476375,   -0.03665640,   -0.04465161,    0.06727043,    0.18908839],
 [  -0.16674925,   -0.81404686,    0.06822833,   -0.00696462,    0.02834672,    0.05160847,    0.08683419],
 [  -0.26817822,    0.17262423,    0.71697100,   -0.05220979,   -0.08031856,   -0.07535656,    0.06090026],
 [  -0.01495850,    0.31616986,    0.65079165,   -0.08625862,    0.02595167,    0.06234652,    0.55288080],
 [  -0.29140538,   -0.74383260,    0.27157688,   -0.05516398,    0.02889121,    0.02798209,    0.35436106],
 [  -0.02956736,   -0.37011838,    0.32225782,    0.00415923,   -0.07142623,   -0.14322874,    0.24185549],
 [  -0.01925877,   -0.54460670,   -0.02512421,   -0.05827679,   -0.09387235,   -0.17756772,    0.23311497],
 [   0.22504807,   -0.19617802,    0.35569274,    0.05009212,    0.05089408,   -0.00209370,   -0.07286017]],
[[  -0.19896653,   -0.16961831,    0.12775117,   -0.06209022,   -0.04949997,   -0.12609321,    0.38238883],
 [   0.21406043,   -0.27700377,    0.31794697,    0.00257020,   -0.04394899,    0.02370736,    0.36899440],
 [   0.03365174,    0.11884296,    0.74764410,   -0.04643322,    0.02226024,   -0.03383231,    0.90102774],
 [  -0.25449023,   -0.12543255,    0.56837590,   -0.04414108,    0.01575356,   -0.03293103,    0.46176870],
 [  -0.32686317,   -0.02175063,   -0.06221682,    0.04551795,   -0.03747696,    0.05519783,    0.04051643],
 [   0.20071805,   -0.69299650,    0.46265596,   -0.09358420,   -0.06886242,   -0.20203933,    0.87985030],
 [  -0.31490552,   -0.56150320,    0.76180375,    0.01509276,    0.03965087,   -0.01633747,    0.28561807],
 [  -0.10249193,   -0.33260328,    0.33640790,    0.05329420,   -0.03646082,   -0.13538560,    0.95538810]],
[[  -0.31805390,    0.33535814,    0.15607959,    0.03991474,    0.00985367,   -0.05126505,    0.21014680],
 [   0.27561766,   -0.01823598,    0.07673582,   -0.07207054,   -0.03191121,    0.04428267,    0.74717080],
 [  -0.32655895,   -0.70509590,    0.09093812,   -0.05686228,    0.01407947,    0.00547141,    0.63186926],
 [   0.21760458,   -0.70001507,    0.59823230,   -0.05178375,   -0.06510716,   -0.16406977,    0.24455737],
 [   0.00612205,   -0.15473408,    0.71941510,   -0.09307641,   -0.04665929,   -0.01531245,    0.99421626],
 [  -0.17217952,   -0.66365314,    0.46711642,   -0.07832068,   -0.09206662,    0.04990476,    0.94060640],
 [   0.11052638,   -0.76739705,    0.77290356,    0.05821288,   -0.03139047,   -0.15143427,    0.60822270],
 [   0.01406094,   -0.30535042,    0.66796464,   -0.01582121,    0.01040550,   -0.02612349,    0.67804146]],
[[  -0.08945890,   -0.23876113,    0.69201475,   -0.04304264,    0.01525395,   -0.11204735,    0.19141276],
 [  -0.01296791,    0.36765707,    0.76816640,   -0.07661001,   -0.07378753,   -0.04321563,    0.31771028],
 [  -0.14588700,    0.36816955,    0.26874554,   -0.06822491,    0.05357139,   -0.03358077,    0.66671610],
 [  -0.31552944,   -0.55449100,   -0.23796913,   -0.05738727,   -0.01497386,   -0.23331602,    0.50585920],
 [   0.00368866,   -0.48207954,    0.72456974,    0.02457344,    0.01887690,    0.05123973,    0.71874714],
 [  -0.28267170,   -0.58507170,    0.15987411,   -0.05217699,   -0.09521329,   -0.19475980,    0.90821743],
 [   0.27257782,    0.03250068,    0.37680024,    0.02621397,    0.04990479,   -0.23459415,    0.02596509],
 [  -0.01548907,   -0.40920880,    0.46738595,    0.05747290,    0.00942056,   -0.21484381,    0.36061710]],
[[   0.04350072,    0.37605214,    0.28981918,   -0.07298791,    0.00926226,   -0.15585709,    0.48357230],
 [  -0.03923693,   -0.70982957,    0.70076954,   -0.05921869,   -0.09734815,   -0.17137520,    0.35350543],
 [  -0.03687960,   -0.06727982,    0.49678130,   -0.02662121,   -0.05961905,   -0.17284934,    0.12504855],
 [  -0.07420853,   -0.63133230,    0.77388716,   -0.07412487,    0.05104111,    0.01373510,    0.49466240],
 [  -0.19323008,   -0.55272330,   -0.09511863,    0.04276018,   -0.08033471,   -0.02860142,    0.08711414],
 [   0.18809938,    0.17800832,    0.37749243,   -0.02427132,   -0.08390169,   -0.04687810,    0.91251050],
 [   0.28337830,   -0.04337025,    0.05784589,   -0.02306691,   -0.08917480,   -0.08358920,    0.28363610],
 [  -0.24189445,   -0.79113954,   -0.02723590,   -0.08561972,   -0.01890656,   -0.04983325,    0.35568643]],
[[   0.03072071,   -0.04573411,    0.15431875,    0.04737294,    0.03715338,    0.02328101,    0.38874060],
 [  -0.23595494,   -0.65731347,    0.35602498,    0.02870151,   -0.00348311,   -0.23567516,    0.23392860],
 [   0.28537280,    0.12000531,    0.32029850,   -0.07600349,    0.00469178,   -0.21629445,    0.29256725],
 [  -0.03484017,    0.16691613,    0.57512057,    0.02795731,    0.01551504,   -0.13429141,    0.32981002],
 [  -0.12136368,    0.01231682,    0.64285710,    0.02673713,   -0.01286154,    0.00359794,    0.71440804],
 [   0.05341679,   -0.61950547,    0.21460053,   -0.01838489,    0.05210477,   -0.08938986,    0.04016294],
 [   0.20788026,    0.23532534,    0.24510238,   -0.03106125,   -0.08523212,   -0.19663362,    0.88753610],
 [   0.16247067,   -0.27027340,    0.18088397,   -0.08647472,   -0.06530654,   -0.00569572,    0.08898339]],
[[   0.21633238,   -0.35974842,   -0.22157902,    0.02195960,   -0.05854145,    0.01285969,    0.05667591],
 [  -0.22684011,   -0.29523838,    0.29937857,   -0.09385383,   -0.07622029,   -0.18826142,    0.84419197],
 [   0.07114592,   -0.25450230,   -0.06752007,   -0.04240722,   -0.02723426,    0.04519907,    0.24629153],
 [  -0.16510819,   -0.20488435,    0.55865170,    0.04027575,   -0.04156501,   -0.05511428,    0.15265262],
 [   0.02679425,   -0.10596353,    0.44114184,    0.01646231,   -0.10178022,    0.04834393,    0.33591664],
 [  -0.06176823,    0.05363458,    0.12507728,    0.02485041,   -0.02471089,    0.00780644,    0.49342060],
 [  -0.13096105,   -0.03411257,   -0.01220722,   -0.02461744,   -0.00770935,   -0.01307379,    0.93339140],
 [  -0.19109966,   -0.35678380,    0.65447970,    0.01538204,    0.04908232,   -0.07793315,   -0.03822013]],
[[  -0.12900913,   -0.28879630,   -0.16665778,   -0.02468368,   -0.01836409,    0.05591273,    0.73718680],
 [  -0.15434305,    0.37455845,    0.01763618,   -0.04498103,    0.04035902,    0.05908987,   -0.01026195],
 [   0.20411098,   -0.73192465,   -0.05044995,    0.00099657,   -0.08132853,   -0.22049655,    0.46463156],
 [   0.24762595,   -0.55827140,    0.32347990,   -0.04159432,    0.03631382,    0.02894041,    0.85238010],
 [   0.18684846,   -0.09547973,    0.62200390,   -0.01211172,    0.03801225,   -0.06629921,    0.80537510],
 [  -0.29229417,    0.04952925,   -0.13670477,   -0.08486554,    0.03066927,   -0.15968439,    0.24691407],
 [  -0.16423863,   -0.12618989,    0.64422446,   -0.00731698,   -0.06917253,    0.05738795,    0.71250160],
 [  -0.21515042,    0.24278641,   -0.05585961,    0.03517325,    0.02422249,    0.00930268,    0.11717427]],
[[  -0.09606606,   -0.73361060,    0.69141114,   -0.02274814,    0.03479579,   -0.13537070,    0.09946939],
 [  -0.14282744,   -0.69450074,    0.21218944,   -0.02287530,   -0.00995012,   -0.23461811,    0.67095720],
 [  -0.12749064,   -0.11743641,    0.77421020,   -0.04334551,   -0.00746994,    0.02509561,    0.13692561],
 [  -0.27345413,    0.11814100,    0.22231820,   -0.05013249,   -0.03029734,   -0.12208512,    0.08484252],
 [  -0.06837282,   -0.44948077,    0.00890303,    0.03068008,   -0.07728165,   -0.04849553,    0.25943750],
 [  -0.21127840,    0.00218344,    0.65655650,   -0.03048380,   -0.07716230,   -0.04270384,    0.93006164],
 [   0.02947766,   -0.01945210,    0.47652847,    0.03995866,   -0.07259602,   -0.09021777,    0.88684550],
 [  -0.10083988,    0.32158375,    0.41975194,    0.01584097,   -0.03746329,   -0.10060786,    0.56240845]],
[[   0.04412168,   -0.36654934,    0.63562310,    0.05303861,   -0.09098101,   -0.10552628,    0.79334956],
 [  -0.03243339,    0.12942624,   -0.23926920,    0.00047318,   -0.03576562,   -0.09046224,    0.00243987],
 [  -0.09851526,   -0.04455394,   -0.02016300,    0.00439738,    0.05010816,   -0.15916249,    0.41382515],
 [   0.10714832,   -0.31561017,    0.74649630,   -0.06257819,   -0.01966684,   -0.15926000,    0.20599543],
 [  -0.11893825,   -0.03561562,    0.22804576,   -0.09369666,   -0.04556563,   -0.05966821,    0.22763170],
 [  -0.15986979,   -0.75834656,    0.41418678,   -0.00619614,   -0.00488098,   -0.08958986,    0.37550038],
 [   0.02133495,   -0.04421747,    0.13470453,   -0.03624533,   -0.04412382,   -0.11208555,    0.19006033],
 [  -0.19443920,   -0.03312212,    0.16933692,    0.02464730,   -0.07810515,   -0.16609850,    0.25216544]],
[[  -0.30234566,   -0.60836416,    0.68619186,    0.01342187,    0.04284799,    0.02890897,    0.17975114],
 [   0.14457199,    0.15805161,    0.65384120,    0.05047403,    0.01053554,   -0.08530548,    0.39857018],
 [  -0.28333345,    0.30436670,    0.42409010,   -0.08450791,   -0.00298569,   -0.06205942,    0.95156030],
 [  -0.12032998,   -0.72779363,   -0.13755071,   -0.03788181,   -0.02277788,   -0.17064556,    0.45270807],
 [   0.23482293,   -0.57249844,   -0.08711654,    0.04434724,   -0.03573204,   -0.10331310,    0.01772496],
 [   0.26640230,   -0.27684534,    0.52666650,   -0.09109246,   -0.06134875,   -0.17667526,    0.60456200],
 [  -0.05110231,    0.21148288,   -0.03598897,    0.00392368,   -0.06807841,   -0.15219718,    0.42132163],
 [   0.13567364,   -0.46680140,    0.38266444,    0.04672669,   -0.02057101,   -0.21502768,    0.65497935]],
[[  -0.23704103,   -0.63705050,   -0.09724915,   -0.02663264,   -0.05587148,   -0.08222330,    0.19113763],
 [   0.06652451,    0.28813958,   -0.02550070,   -0.05907772,   -0.04449964,   -0.22695270,    0.97704260],
 [   0.27427465,   -0.53982500,    0.21960550,   -0.06622639,    0.03177173,   -0.01092573,    0.17279746],
 [   0.16485697,   -0.26874828,   -0.09388271,   -0.07611404,    0.01025853,    0.06263444,    0.71599380],
 [   0.13065717,   -0.58958590,    0.60846347,   -0.06872855,    0.05274963,   -0.10837282,    0.18845434],
 [   0.01521015,    0.12255979,    0.26843083,    0.02760121,   -0.06848890,    0.01989225,   -0.05000143],
 [  -0.31203333,    0.22697389,   -0.01352100,   -0.06166117,   -0.05950620,   -0.01579198,    0.23931871],
 [  -0.09080377,    0.26190890,    0.46078850,   -0.08757637,   -0.02396530,    0.02764705,    0.68643610]],
[[   0.01878145,   -0.74282897,    0.76982665,    0.02298741,    0.00554822,   -0.19782212,    0.01432636],
 [   0.22037739,   -0.64468450,    0.39575964,   -0.05289542,   -0.08604030,   -0.14663154,    0.21922572],
 [   0.11623463,   -0.05472863,    0.02725148,   -0.08814757,   -0.06364252,   -0.20786513,    0.11640967],
 [   0.20753068,    0.07861978,    0.10850272,    0.04894851,   -0.01926224,   -0.13869270,    0.76731310],
 [  -0.12229107,   -0.35463074,   -0.19738360,   -0.09046512,   -0.08690614,    0.06307918,    0.02730542],
 [   0.26282150,   -0.63761175,    0.03307471,   -0.07081242,    0.02021720,   -0.08490531,    0.93516060],
 [  -0.10306014,    0.27127743,    0.37780112,   -0.03988151,    0.00483717,    0.05343297,    0.52072394],
 [  -0.27293570,    0.00041199,    0.62821054,    0.03315686,    0.04764591,    0.05425924,    0.13739851]],
[[   0.18772626,    0.30245268,    0.12157190,   -0.04234615,    0.01266523,   -0.12222023,    0.30923456],
 [   0.07487631,    0.12463677,    0.57852660,    0.00710646,   -0.07590681,    0.03463441,    0.25406283],
 [  -0.02970597,    0.02079296,    0.77379200,   -0.03978712,   -0.07019264,   -0.15517956,    0.73867960],
 [  -0.05869275,   -0.12568212,   -0.24929829,   -0.00778767,   -0.02447739,   -0.21499471,    0.49452156],
 [   0.24253368,    0.04722029,   -0.07305843,   -0.06721303,    0.03577603,   -0.11466605,    0.87334895],
 [   0.02514479,    0.24029875,    0.75346530,   -0.04075446,    0.03029965,   -0.19821993,    0.52582544],
 [  -0.30982605,    0.26085210,    0.69914174,    0.03024651,   -0.02835463,    0.04046461,   -0.06200989],
 [  -0.25867876,   -0.72718376,    0.48149710,    0.00041040,   -0.02964551,   -0.07451059,    0.74958780]],
[[  -0.05578920,   -0.55499494,    0.46431780,   -0.07525720,    0.04349403,   -0.02344152,    0.98969010],
 [  -0.03434324,   -0.11390841,    0.10637474,   -0.09009424,   -0.03387570,   -0.18313843,    0.84574276],
 [   0.04676637,   -0.22560811,   -0.20344606,   -0.02205260,   -0.00740946,   -0.08925439,    0.07750025],
 [   0.16439891,   -0.50870967,    0.03210449,   -0.09055290,   -0.03834955,   -0.09357499,   -0.00438160],
 [  -0.21608472,   -0.15001464,   -0.08699161,   -0.02631445,    0.01401741,    0.01081915,    0.09317482],
 [  -0.00798979,   -0.81903815,    0.32130890,    0.03157033,   -0.02339166,   -0.17078787,    0.62388736],
 [  -0.30699015,    0.13976389,    0.69397430,   -0.01138049,   -0.03945519,   -0.17543766,    0.78450200],
 [   0.11784834,    0.25159085,   -0.23579443,    0.02837256,    0.03313912,   -0.06433918,    0.40363175]],
[[  -0.28692040,   -0.76823160,   -0.17859788,   -0.03370273,   -0.09474506,   -0.01650016,    0.37813950],
 [   0.25868505,   -0.23981291,    0.33828956,   -0.01739253,   -0.03187507,   -0.08833808,    0.24910380],
 [  -0.25410813,   -0.60295415,    0.26573998,   -0.04072867,    0.00863297,   -0.00733244,    0.78719370],
 [   0.26372945,   -0.43185830,    0.64863974,   -0.02860958,    0.01887345,   -0.01876736,    0.54120785],
 [  -0.20772552,   -0.02231145,    0.27916790,    0.00545698,    0.00200868,   -0.17540544,    0.12015720],
 [   0.28104508,    0.02742720,    0.46716088,   -0.02411210,    0.02031846,    0.03943750,    0.23828177],
 [  -0.05754766,   -0.51889110,   -0.08108820,   -0.06016610,   -0.00837257,   -0.02603786,    0.21639793],
 [  -0.10480456,   -0.67250960,    0.00842735,   -0.05612397,   -0.07405864,    0.04122233,    0.80779284]],
[[   0.25910234,   -0.29170210,    0.73529667,   -0.09126367,   -0.04089598,   -0.15090278,    0.01968996],
 [  -0.19920567,   -0.65434920,    0.08938596,    0.04763715,    0.01504823,   -0.08448833,    0.46825350],
 [  -0.06715405,    0.16165972,    0.48525983,   -0.06030430,   -0.08154288,    0.05852047,    0.86015695],
 [   0.18829352,   -0.54677580,    0.42326695,   -0.07495362,    0.02714363,   -0.01086147,    0.73667437],
 [   0.03303704,    0.37407875,    0.71124184,   -0.08145270,    0.01020767,   -0.05111738,    0.22561218],
 [  -0.00616807,    0.13585126,    0.48380548,   -0.03779193,    0.00810864,   -0.20055884,    0.82597510],
 [   0.24179524,   -0.58752525,    0.58783776,    0.06327276,   -0.09962382,   -0.21574959,    0.53734480],
 [   0.03540868,    0.00862187,   -0.01300025,   -0.07075557,   -0.05700088,   -0.09635513,    0.52225600]],
[[  -0.16016412,   -0.79755450,    0.01863936,   -0.06467061,   -0.00637240,    0.01801485,    0.41586370],
 [  -0.14559601,   -0.21758288,    0.51876450,    0.05504569,    0.03757840,   -0.10873300,    0.17579944],
 [  -0.07078201,    0.21491277,   -0.00195335,   -0.04101679,   -0.05219974,   -0.19182874,    0.43080467],
 [   0.14240715,    0.20113933,    0.73264503,   -0.02042825,   -0.09996722,   -0.15840430,    0.14301372],
 [  -0.16160788,    0.34230757,    0.75141190,   -0.07547225,    0.04971208,    0.00956143,    0.65719600],
 [  -0.08440183,   -0.06779259,   -0.18012163,    0.01485118,   -0.01861344,   -0.05331817,    0.47032028],
 [   0.21928489,   -0.18325108,    0.31081450,   -0.03509896,    0.00216009,   -0.00739448,    0.29993212],
 [  -0.24384153,   -0.16150308,   -0.17976673,    0.06109180,   -0.01834124,   -0.07101065,    0.09325229]],
[[   0.21743584,    0.21289194,   -0.13414730,    0.02662615,   -0.09890322,    0.03794953,    0.93261060],
 [   0.17430550,   -0.38672610,    0.23985827,   -0.06240682,   -0.05403045,   -0.04621607,    0.37842268],
 [   0.02840665,    0.32405245,    0.24759388,   -0.00422455,   -0.07988263,   -0.15213469,    0.51294020],
 [   0.04747108,   -0.69225246,    0.62520295,   -0.02770039,   -0.03559010,    0.03406048,   -0.02737029],
 [  -0.13650544,   -0.23127759,    0.24277437,   -0.06601049,   -0.00124247,   -0.09534456,    0.78185713],
 [  -0.20143053,    0.27003598,    0.43489170,   -0.00319643,    0.05116612,   -0.08924457,    0.94849700],
 [  -0.25639790,   -0.61969830,    0.70860136,   -0.04433447,    0.00379275,   -0.06663075,    0.72820723],
 [   0.27058572,   -0.03745914,    0.45539343,    0.00962895,    0.03585191,    0.01667804,    0.69519400]],
[[   0.08487529,   -0.80662090,   -0.03215837,    0.00931283,   -0.05650801,   -0.16963620,    0.62949497],
 [   0.21720845,   -0.45083070,    0.72092766,   -0.03419663,   -0.02991053,   -0.20481831,    0.00700083],
 [   0.28518070,   -0.29303450,    0.47004390,   -0.07658933,   -0.08586302,   -0.21522260,   -0.00852513],
 [  -0.05188897,   -0.60134890,    0.13837415,    0.01046469,    0.03843015,   -0.10585816,    0.29833430],
 [  -0.31599534,   -0.26233876,    0.28011340,    0.00531362,   -0.08091363,   -0.01113686,    0.57740486],
 [  -0.25537420,    0.32248163,    0.66774553,    0.02679592,   -0.07070620,   -0.11885699,    0.89991890],
 [  -0.05858067,   -0.36773697,    0.15179124,   -0.05594938,    0.03281955,   -0.10931751,    0.55045110],
 [   0.15029886,    0.30429935,    0.36742157,   -0.05779378,   -0.03442234,   -0.02564503,   -0.04427650]],
[[   0.13775367,   -0.72708976,   -0.18795069,   -0.05948263,    0.03532353,   -0.08796781,    0.45562738],
 [  -0.15227404,   -0.02390921,   -0.05049430,   -0.00476271,   -0.02809993,   -0.15049341,   -0.03051175],
 [  -0.26563463,   -0.63677030,    0.52765256,    0.04736815,    0.05489978,   -0.00943457,   -0.05032403],
 [  -0.12810305,   -0.00595319,   -0.02819470,   -0.05749580,    0.04570381,   -0.01045734,    0.57845570],
 [  -0.20675199,   -0.30871780,    0.15792173,    0.01689820,    0.05054331,   -0.14180052,    0.57344085],
 [  -0.26296910,   -0.16598141,    0.24791971,   -0.06415954,   -0.09924059,   -0.20553912,    0.65979320],
 [   0.07005578,   -0.47836956,    0.28907454,    0.04436947,   -0.04554260,   -0.08777529,    0.43785720],
 [   0.18117428,    0.10355383,    0.56930700,   -0.00538020,   -0.09855333,   -0.09820281,    0.88730836]],
[[  -0.28451955,    0.11167073,   -0.14079177,   -0.05863419,   -0.01359868,   -0.22730711,    0.55369990],
 [  -0.12201948,   -0.44220617,    0.57414280,   -0.05922401,   -0.01206227,   -0.10578667,    0.35796535],
 [  -0.01704416,   -0.39478946,    0.05834955,   -0.04878768,    0.03564797,   -0.10608570,    0.70344960],
 [  -0.04991308,    0.22996330,    0.50581694,   -0.03585217,   -0.03280266,   -0.10451034,    0.40353888],
 [  -0.15044707,   -0.43900037,   -0.23676531,    0.06348282,    0.03791361,   -0.05609533,    0.60188705],
 [   0.20844972,   -0.21364260,    0.73842130,   -0.03220626,   -0.04008486,    0.06278697,    0.19892250],
 [   0.24083966,   -0.19052690,    0.29732817,    0.01150702,    0.04109585,    0.01678306,    0.95435290],
 [  -0.19852278,   -0.61204050,    0.32918584,   -0.04946086,   -0.09871437,   -0.02607965,    0.49365878]],
[[  -0.06731156,   -0.41531456,    0.56838350,   -0.00512153,    0.04470073,   -0.01884311,    0.22824268],
 [  -0.31026512,   -0.21102369,    0.26727020,   -0.05018321,    0.00801820,   -0.19451803,    0.52779950],
 [   0.03384674,   -0.06319124,    0.30298984,    0.03435364,    0.01327553,   -0.22785345,    0.54227120],
 [  -0.00252736,   -0.46945520,    0.59816800,   -0.01981042,    0.04879954,   -0.04142956,   -0.06637414],
 [  -0.25413990,   -0.08884960,   -0.24266203,   -0.00632157,   -0.05823973,   -0.04135381,    0.25582236],
 [  -0.30515683,   -0.54949427,    0.27139914,   -0.06360862,   -0.07368818,   -0.17566672,    0.61898524],
 [   0.27798027,   -0.69898444,    0.00081560,   -0.00515351,    0.04736622,   -0.21467008,    0.74348910],
 [   0.15823677,   -0.13159210,    0.69455340,   -0.05424843,   -0.04303854,   -0.23027414,    0.36374820]],
[[  -0.32025313,   -0.46235374,    0.47630012,    0.05313347,    0.05506055,    0.02298912,    0.98592940],
 [   0.25555676,   -0.21288884,    0.38691050,    0.05847448,   -0.06215488,   -0.20576687,    0.27154678],
 [  -0.00901094,   -0.26397407,    0.78461900,   -0.09148905,   -0.08247527,   -0.05483025,    0.39478510],
 [   0.03958741,    0.08904433,   -0.08150822,    0.03467007,    0.01885641,   -0.07953206,    0.48097700],
 [   0.05725610,   -0.30818033,    0.13307280,    0.00521941,   -0.03982174,   -0.10800040,    0.68197626],
 [  -0.22066134,    0.24470365,   -0.06448221,    0.01285680,   -0.01408160,   -0.04217365,    0.28820133],
 [  -0.19941114,   -0.29681880,    0.30506580,    0.03883950,   -0.08450481,   -0.03712423,    0.44362820],
 [  -0.32989950,   -0.01013875,   -0.19259898,    0.01353625,   -0.04553871,    0.02985954,    0.81341700]],
[[   0.16824901,   -0.06759119,    0.64535636,   -0.01554741,    0.01598504,   -0.18840745,    0.25176746],
 [  -0.23871150,    0.26242208,    0.23960900,   -0.08809388,    0.01858453,    0.01698998,    0.83145420],
 [   0.03818834,   -0.47622970,    0.10403720,    0.02664774,   -0.06201859,   -0.01275326,    0.60714560],
 [  -0.28564656,   -0.30604600,    0.71945030,    0.03135814,    0.05046800,   -0.15991522,    0.28055114],
 [   0.15092897,    0.10517454,    0.26634037,   -0.04603133,    0.05053255,    0.06686425,   -0.02100073],
 [  -0.04791582,   -0.28399628,    0.61007166,   -0.08981729,   -0.09702632,   -0.02741675,    0.55692196],
 [  -0.18079375,    0.13771075,    0.42672688,   -0.08216712,   -0.07924020,   -0.00094201,   -0.00990947],
 [   0.04093337,   -0.14200377,    0.39959693,    0.02050290,   -0.07227129,   -0.16140515,    0.99490280]],
[[   0.09443304,    0.00513071,    0.30851287,    0.02720144,   -0.05406939,   -0.16067944,    0.40171504],
 [   0.20005828,   -0.40320328,    0.72668370,   -0.01721237,   -0.01546440,   -0.09367965,    0.68025530],
 [  -0.25221440,   -0.57475410,    0.47873520,    0.00112528,   -0.04878590,    0.04632589,    0.81815195],
 [   0.12208748,   -0.75843436,    0.20317814,   -0.02772825,    0.02317345,   -0.20231147,    0.32003593],
 [  -0.03479883,    0.34018743,    0.00143811,   -0.02928905,    0.03582737,   -0.23103390,    0.49326550],
 [  -0.16500965,   -0.81522310,    0.60840577,    0.03513999,   -0.00266660,    0.00309686,    0.75031500],
 [  -0.28827223,   -0.73026097,    0.53500180,   -0.03424729,   -0.01052304,    0.04364079,    0.51550180],
 [   0.27176374,   -0.62686265,    0.12747526,   -0.02134956,    0.05699183,   -0.12842691,    0.36361367]],
[[   0.01876357,    0.18591893,   -0.13033189,    0.02929588,    0.05684401,   -0.19804668,    0.31427157],
 [  -0.17487219,    0.08209789,    0.68737670,    0.06373253,    0.03674915,   -0.01596385,    0.18526067],
 [  -0.22908828,    0.01781023,    0.57358336,    0.01867236,   -0.02092052,   -0.11085935,    0.81544566],
 [   0.23933566,   -0.77829266,    0.34128118,   -0.07670926,    0.00112942,   -0.05252616,    0.21633099],
 [   0.08104473,   -0.23289382,   -0.24069747,   -0.06896996,    0.01490885,   -0.14824453,   -0.02966775],
 [  -0.06374979,   -0.75082550,    0.42345798,    0.01194496,   -0.07344551,   -0.22226167,    0.55005836],
 [  -0.03659201,    0.15345168,    0.31661218,   -0.06371596,   -0.09324388,    0.04101583,    0.61048180],
 [  -0.20940168,   -0.40282220,    0.67397220,    0.00558121,   -0.01024838,   -0.01145156,    0.43732244]],
[[   0.09049416,   -0.83160996,    0.07070613,    0.03718266,    0.01599693,   -0.00325996,    0.12808340],
 [   0.13777262,   -0.49556500,    0.28188497,    0.03654829,   -0.08424918,   -0.01904793,    0.67712814],
 [  -0.10111947,    0.23475230,    0.39004207,   -0.08869049,    0.03074250,   -0.10074906,    0.97575530],
 [  -0.29533842,   -0.35521716,    0.11323082,   -0.04949711,    0.05163582,   -0.14873907,    0.39381720],
 [  -0.21381901,   -0.76992553,    0.06109342,   -0.02785295,    0.04854190,   -0.03430498,    0.45638000],
 [   0.01939830,    0.16654205,    0.28042555,    0.01165432,   -0.06544098,   -0.22977510,    0.95300406],
 [  -0.19105384,    0.23444450,    0.02227220,   -0.02682490,   -0.09036263,   -0.04698724,    0.94202965],
 [   0.10507223,   -0.46425542,    0.74156320,    0.04100659,   -0.01235957,   -0.01666558,    0.38109136]],
[[  -0.24061948,   -0.00144345,    0.61878820,    0.06318031,   -0.09841283,    0.02556521,    0.85240070],
 [   0.22828215,   -0.67826784,   -0.05512588,    0.01398019,   -0.07995796,    0.01069701,    0.90596160],
 [   0.23486840,   -0.30807042,    0.18407345,   -0.03716873,    0.03319496,   -0.21321914,    0.57538910],
 [   0.08588430,    0.20343494,    0.26216150,   -0.00120355,   -0.04987375,   -0.12719440,    0.76661205],
 [   0.07824361,    0.32342290,   -0.07044281,    0.02375426,   -0.03390097,    0.04893279,    0.93670636],
 [   0.14404556,   -0.37149972,    0.55924875,   -0.08060627,   -0.03125580,   -0.22004930,    0.29799968],
 [  -0.21013522,   -0.24315846,   -0.07074937,   -0.04368749,    0.01814085,    0.05882835,    0.48103620],
 [  -0.10080217,   -0.02485299,    0.09035054,   -0.07833923,   -0.02601721,   -0.15406565,    0.36333710]],
[[  -0.01598832,    0.25116038,    0.66401130,    0.02700252,   -0.03098138,   -0.05736691,    0.57005304],
 [  -0.07779470,   -0.53815200,    0.53006625,   -0.09021386,   -0.00993537,   -0.10033731,    0.75679330],
 [   0.15171969,   -0.76694860,    0.20016488,   -0.00120126,   -0.03720304,   -0.02060266,    0.68451273],
 [  -0.16375454,   -0.35303295,    0.30380166,    0.01000360,   -0.04912429,    0.04385376,    0.97409266],
 [   0.15301758,    0.00621241,    0.46017760,   -0.02621374,   -0.01252335,   -0.20871322,    0.48322845],
 [   0.22883427,   -0.50847740,    0.40958697,    0.02603356,   -0.04364364,   -0.02314112,    0.02334952],
 [  -0.02818704,   -0.69266670,    0.07167795,   -0.04467630,    0.02191836,   -0.16505610,    0.35753673],
 [  -0.20387398,    0.14860213,   -0.00441343,   -0.06971148,   -0.07740660,    0.04146826,    0.20205815]],
[[  -0.22366910,    0.05818856,    0.53158190,   -0.07102174,    0.04200241,   -0.21101192,    0.76336030],
 [  -0.33103913,    0.15640235,    0.43140960,    0.06370682,    0.00990565,   -0.07737164,    0.43710590],
 [   0.21301013,   -0.35933423,   -0.21332227,   -0.02536097,   -0.03498182,    0.01727057,    0.16318415],
 [  -0.10794036,    0.11006552,    0.68160564,    0.02546571,   -0.06680894,   -0.10436743,    0.20332108],
 [  -0.04736617,   -0.83366720,    0.11430734,    0.05785435,   -0.00081720,    0.00435264,    0.76606670],
 [  -0.15864384,   -0.26590890,    0.72138160,    0.01287026,    0.04548377,   -0.02498811,    0.57266587],
 [  -0.07087514,    0.15159643,    0.36377007,    0.00814974,    0.03170601,   -0.18366918,    0.96940225],
 [   0.02554381,   -0.28009045,    0.18418741,   -0.07093504,   -0.00675473,    0.00166313,    0.09089200]],
[[  -0.27793232,   -0.52618450,    0.29791510,   -0.06273876,    0.04218788,    0.00985926,    0.91499360],
 [  -0.15040997,   -0.76851540,   -0.12366284,    0.03408839,   -0.09146001,   -0.04790772,    0.35916590],
 [  -0.03898430,   -0.79070840,    0.09234163,    0.01415736,    0.04340827,    0.05437922,    0.78338680],
 [   0.23228228,    0.32089794,   -0.01721732,   -0.04926115,    0.01295376,   -0.10206717,    0.94757550],
 [  -0.26518777,   -0.75433180,    0.35740560,    0.01467561,    0.00928570,   -0.00319849,    0.43735080],
 [  -0.23291785,    0.10149550,    0.68879443,   -0.03270873,    0.00079188,   -0.19381756,    0.54651020],
 [   0.20572180,    0.36111534,    0.05866045,   -0.07157207,   -0.09856401,    0.01224491,    0.10962066],
 [  -0.30179968,   -0.73878443,    0.45385778,    0.05638415,   -0.03024182,   -0.12539887,    0.78178880]],
[[  -0.04306456,    0.17972577,    0.54150320,   -0.01295915,    0.01148243,   -0.10316603,    0.99875680],
 [   0.04988340,   -0.05915183,   -0.02201694,   -0.05265381,    0.04151598,   -0.17315851,    0.46290463],
 [   0.08327717,    0.05750841,    0.43170154,    0.01999836,   -0.05625197,    0.04786482,    0.87476355],
 [  -0.27088252,    0.16152048,    0.04142910,    0.05369444,   -0.07063334,    0.00184564,    0.99432950],
 [  -0.07757294,   -0.01993912,    0.35172720,   -0.04462336,    0.02085021,   -0.17696838,    0.00661102],
 [  -0.01040450,   -0.20312613,    0.36494428,    0.04091389,    0.05315420,   -0.19586079,    0.11271083],
 [   0.04200533,   -0.30446500,    0.05503145,   -0.03560511,   -0.04437194,    0.06317219,    0.34964418],
 [  -0.13139503,   -0.83329165,    0.47057062,   -0.01428324,   -0.04628728,   -0.18343252,    0.28972150]],
[[  -0.06991655,   -0.81925136,    0.14580545,    0.01408694,   -0.06281100,   -0.13840136,    0.53924410],
 [  -0.27863258,   -0.04065210,    0.75556517,   -0.03434328,   -0.05572477,   -0.19703364,    0.35604054],
 [   0.23055619,   -0.57162714,    0.35746545,   -0.01486355,   -0.06973756,   -0.18248690,    0.10628143],
 [  -0.01260003,   -0.71814156,    0.09737179,    0.00903865,   -0.08761345,   -0.12227295,    0.62795550],
 [   0.18629938,   -0.65430766,   -0.16549966,    0.01031850,   -0.05066497,   -0.04779209,    0.94166225],
 [  -0.21657836,    0.27806616,    0.18883070,    0.02224181,    0.00723559,    0.06717214,    0.66908800],
 [   0.17119229,   -0.67276170,   -0.05239469,   -0.03712991,    0.02647647,   -0.23116998,    0.61992640],
 [  -0.26971602,    0.11746907,    0.41316520,   -0.09101438,   -0.04160201,   -0.18245308,    0.44175446]],
[[   0.19641334,   -0.21787632,   -0.22775005,    0.05061312,   -0.05472076,    0.05432928,    0.54460454],
 [  -0.24079916,   -0.65908986,   -0.14028633,   -0.09274085,    0.04108527,   -0.01540491,    0.03560676],
 [  -0.03918445,   -0.58157180,    0.33480144,   -0.06772672,   -0.09701510,   -0.03395681,    0.29516720],
 [  -0.25242406,   -0.78871220,    0.03232872,    0.00862730,    0.00933123,   -0.12170844,    0.20213182],
 [   0.04202917,   -0.06145203,    0.18190932,   -0.08579116,    0.01372192,   -0.18440194,    0.40529000],
 [   0.11619121,    0.32736754,    0.19990360,   -0.02586932,   -0.09789454,   -0.11652858,    0.78809255],
 [  -0.28088945,    0.30760694,    0.07835367,   -0.00412726,    0.02189159,   -0.05305977,    0.89804740],
 [   0.00401607,   -0.76061887,    0.28741366,    0.00390127,    0.04189935,   -0.07208286,    0.35007077]],
[[  -0.17415217,   -0.46955702,    0.30619950,    0.00550757,    0.02058650,    0.01301093,    0.21801989],
 [   0.10272542,   -0.20786798,    0.69107180,   -0.00017115,   -0.09918629,   -0.16978809,    0.43958127],
 [  -0.26590514,   -0.57669246,   -0.21519690,   -0.09038463,   -0.03523687,    0.00919707,   -0.06470812],
 [  -0.28993730,   -0.11793685,    0.15856963,    0.01543854,    0.02733716,   -0.19287588,   -0.07889655],
 [   0.11237708,   -0.66806510,    0.53845700,   -0.06091537,   -0.10075738,   -0.01930100,    0.98194880],
 [   0.05407116,   -0.06827027,    0.28190643,   -0.03163913,   -0.02899941,   -0.22478914,    0.68392515],
 [   0.27488393,   -0.45433623,   -0.07632840,   -0.03055695,   -0.03470761,    0.05717105,    0.21348439],
 [   0.14096451,   -0.40050203,    0.51388880,    0.01685481,   -0.10186043,   -0.21891965,    0.76941590]],
[[  -0.21225610,   -0.81328450,    0.50982380,    0.01048721,    0.03551391,   -0.00459339,    0.89808446],
 [  -0.17141770,   -0.65465660,    0.76271190,    0.06130719,   -0.02280613,   -0.18509302,    0.71282730],
 [   0.20037341,    0.13175547,    0.62456760,    0.04909678,   -0.05991634,   -0.05648641,   -0.00783364],
 [  -0.00168836,    0.33897722,    0.18606052,   -0.02565747,    0.00252116,   -0.15827121,   -0.04779080],
 [  -0.26016653,   -0.28409850,    0.59827170,   -0.08029485,   -0.01467800,   -0.14005522,   -0.04898653],
 [  -0.04956692,   -0.10885334,    0.44765282,    0.03650708,   -0.01613611,   -0.07775685,    0.41242743],
 [   0.17490834,    0.07071084,    0.58645030,   -0.09227990,    0.04308690,   -0.19801393,    0.84960130],
 [  -0.23923638,   -0.31057662,    0.56936090,    0.01806764,   -0.09565869,   -0.01881728,    0.25549460]],
[[   0.01542526,    0.23217690,    0.62121870,   -0.08647813,    0.03060962,   -0.03501508,    0.52726310],
 [  -0.15039366,   -0.73203380,    0.10542396,   -0.04869958,    0.03010954,   -0.08469713,    0.87165046],
 [  -0.13274775,    0.22744930,   -0.08666067,   -0.03215338,   -0.06808826,   -0.10191540,    0.27382433],
 [   0.11510733,   -0.26077682,   -0.11670995,   -0.04728846,   -0.08814954,   -0.07477060,    0.65788980],
 [   0.24859715,    0.30620778,    0.41296720,   -0.07461506,   -0.06049709,    0.03421789,    0.48370677],
 [   0.15815052,   -0.10547948,    0.19029474,   -0.04509520,   -0.02485951,    0.04142991,    0.38973470],
 [   0.00533372,    0.27507400,    0.34303862,   -0.00928227,   -0.01767411,   -0.05742645,    0.09292868],
 [   0.18196225,   -0.60934390,    0.12765366,    0.04570684,    0.03505673,   -0.03000431,    0.18714331]],
[[  -0.05243337,   -0.78338635,    0.70260150,   -0.04070377,   -0.09786121,   -0.00823967,   -0.03728494],
 [  -0.21564227,    0.26636970,    0.24695346,   -0.00410202,   -0.05482559,   -0.07071321,    0.20722191],
 [  -0.19872715,    0.20371044,    0.14825750,   -0.08543611,   -0.06042979,   -0.17512450,    0.98900480],
 [  -0.11746563,   -0.82406280,    0.22219113,    0.05227728,    0.01649940,   -0.21079215,    0.59782887],
 [  -0.18098375,   -0.33855397,    0.47849750,   -0.02875491,   -0.07060942,    0.04109198,    0.55875254],
 [   0.15299484,    0.32705295,    0.69660294,   -0.03007488,   -0.01147385,   -0.18457209,    0.45494652],
 [   0.23341948,    0.14751232,    0.76722884,    0.01664948,    0.00087319,   -0.22717445,    0.11611372],
 [  -0.12787022,    0.25352287,    0.17443570,   -0.02496354,   -0.06442763,   -0.09436588,    0.79564660]],
[[  -0.27910578,   -0.43497200,    0.27881290,    0.05710039,   -0.04455363,    0.00550723,    0.87153180],
 [   0.24664390,   -0.73460130,   -0.04335791,   -0.05929262,   -0.06794272,   -0.15352705,    0.72154915],
 [  -0.03155217,   -0.75315666,    0.61067260,   -0.01535845,    0.00213544,   -0.00764072,    0.11884925],
 [  -0.11995561,    0.08069432,    0.15724560,   -0.05562264,   -0.06548216,   -0.17377627,    0.19971381],
 [   0.24671793,   -0.09348923,    0.46689640,   -0.04054707,    0.04927675,    0.05141774,    0.29456222],
 [   0.02097288,    0.25618505,    0.62306690,    0.03744818,   -0.04782221,   -0.18045464,    0.65220880],
 [  -0.25239658,   -0.21342600,   -0.03132759,    0.03055532,   -0.05339546,   -0.20182118,    0.47099817],
 [   0.16006479,   -0.36663336,    0.66201985,    0.04062441,    0.01506695,   -0.00200644,    0.23394953]],
[[  -0.00380442,    0.18726039,    0.74013140,    0.04237148,    0.01884148,    0.03427055,    0.95699340],
 [  -0.29713730,   -0.60399340,    0.05900767,   -0.02827919,   -0.01924736,   -0.10764213,   -0.00566664],
 [   0.25738984,    0.33218813,    0.29409504,    0.00294329,   -0.07650361,    0.06937718,    0.74002963],
 [   0.24087697,   -0.79888370,    0.71905340,   -0.03861644,   -0.10191799,    0.04681820,    0.92322200],
 [   0.08344349,   -0.82067305,   -0.10303941,    0.03630513,   -0.00751872,    0.03756166,    0.34528005],
 [  -0.00311106,   -0.71571280,    0.02710348,   -0.02200353,   -0.08344207,    0.01622805,    0.76894860],
 [  -0.03337440,   -0.07380706,    0.00662276,   -0.08568233,   -0.07075746,    0.02755493,    0.63665706],
 [  -0.14092572,   -0.65996320,    0.48831410,   -0.01952706,   -0.05955187,   -0.06319688,   -0.00141174]],
[[  -0.12170848,   -0.45203027,    0.32379055,   -0.06762222,   -0.03369788,    0.02728704,    0.71742540],
 [   0.17108900,   -0.02535784,    0.66591614,   -0.03047127,   -0.04075242,   -0.20902894,    0.63543970],
 [   0.16124645,    0.15609431,    0.71457880,    0.01291576,    0.03796963,   -0.22040730,    0.71316427],
 [   0.24145931,   -0.12873644,   -0.00766961,    0.05008206,   -0.10052137,    0.02836871,    0.98527914],
 [   0.03470474,   -0.82208830,   -0.03874891,    0.03659832,   -0.08722740,   -0.12418919,    0.65004710],
 [   0.13808289,   -0.19836956,   -0.24298910,   -0.08509451,   -0.04214050,   -0.18381314,    0.95625490],
 [   0.25172937,   -0.48974750,    0.04339269,    0.04053009,   -0.09763492,   -0.20382589,    0.21682392],
 [  -0.16460884,   -0.41743836,    0.43184626,    0.06094947,    0.04547746,    0.05822024,    0.40040510]],
[[   0.11372542,    0.09326130,   -0.10537720,    0.05691627,   -0.02945521,    0.03987989,    0.54791397],
 [  -0.06940287,   -0.03586662,    0.74837170,    0.02573486,   -0.05554055,    0.03459156,    0.25718010],
 [   0.10656387,    0.02190590,    0.25535816,   -0.07745045,   -0.00562140,   -0.00481656,    0.62523156],
 [  -0.09695582,   -0.25112045,    0.68720740,    0.03993429,    0.03977238,   -0.15732380,    0.71278540],
 [  -0.13359259,   -0.72345710,    0.66857300,    0.05225389,    0.02377786,   -0.22211716,    0.04079512],
 [  -0.20025319,    0.01398754,    0.11110550,    0.03427070,   -0.04377854,   -0.04029036,    0.15148792],
 [   0.26521690,   -0.18627661,    0.32753700,    0.03323925,   -0.07946000,   -0.10898037,    0.52398150],
 [  -0.33011276,   -0.27746934,    0.15964010,   -0.01039164,    0.04694819,    0.00464895,    0.23400013]],
[[  -0.20387454,   -0.26113462,    0.51272553,    0.04948334,   -0.09151934,   -0.14157261,    0.22455467],
 [   0.27635294,   -0.83032435,    0.02866691,    0.06254528,   -0.05859579,   -0.03848460,    0.23374377],
 [  -0.12382825,   -0.82421935,    0.61971074,   -0.05622156,   -0.05489788,   -0.13211821,    0.25212473],
 [  -0.28517820,    0.05013114,    0.64515120,   -0.03226869,    0.01119269,   -0.17513548,   -0.05862122],
 [   0.12009242,   -0.30834710,    0.01463014,   -0.00530651,    0.02932627,   -0.18446880,    0.28855962],
 [  -0.18527691,   -0.39136780,   -0.17914335,   -0.01393606,    0.01244105,    0.04820797,    0.41371626],
 [  -0.22984898,    0.00961822,    0.24563357,   -0.03961810,   -0.02532998,   -0.14877370,    0.80411740],
 [  -0.20444646,   -0.82935417,   -0.11822639,    0.05011129,   -0.02745012,   -0.19478543,    0.03192250]],
[[   0.19671589,   -0.73811483,    0.19173351,   -0.02931815,    0.03981414,   -0.09746531,    0.87824905],
 [  -0.31695700,   -0.36782240,    0.68429950,    0.00678798,   -0.07795429,   -0.10299146,    0.94204515],
 [  -0.27921720,   -0.61490620,    0.49349016,   -0.02030660,   -0.04363508,   -0.12112176,    0.21273814],
 [  -0.04932186,   -0.23977679,   -0.11502974,   -0.04561277,   -0.07062092,   -0.20900372,    0.14339572],
 [  -0.31438828,   -0.62879884,   -0.23800643,   -0.01615085,    0.05860655,   -0.19663414,    0.87887320],
 [  -0.08508888,   -0.33308834,    0.27131414,    0.00361999,    0.02811842,   -0.00967003,    0.45059390],
 [  -0.02954835,   -0.71750677,   -0.23964794,   -0.00748315,   -0.06594446,    0.05254164,    0.73426250],
 [  -0.30113646,   -0.01247001,   -0.05610995,   -0.05883598,   -0.03414166,    0.05016515,    0.00759123]],
[[   0.11381915,    0.04137492,    0.38152933,    0.02829898,   -0.02631669,    0.03537384,    0.91101160],
 [  -0.30237830,   -0.43320128,   -0.21949129,    0.03550848,   -0.05092560,   -0.00516212,    0.47244155],
 [  -0.25645512,    0.36907935,    0.54290770,   -0.01984460,   -0.08301856,   -0.01902683,    0.03415959],
 [   0.06356460,   -0.64048880,    0.72327130,   -0.08437841,    0.00014429,   -0.00089972,    0.81619320],
 [   0.24149358,    0.29952312,    0.43764060,   -0.07680282,   -0.07436878,   -0.17250304,    0.42170918],
 [  -0.11884181,   -0.71033645,   -0.21036963,    0.04498619,    0.01066153,   -0.14260805,    0.77050150],
 [  -0.25544682,   -0.77131134,    0.09824762,    0.03725277,   -0.01900026,   -0.17518085,    0.19781987],
 [   0.08961496,   -0.43930740,    0.02605349,   -0.07734546,   -0.05449818,   -0.05956061,    0.02775729]],
[[  -0.18328388,   -0.08424872,    0.36894650,    0.05090208,   -0.03598269,   -0.14188570,    0.25475800],
 [  -0.32139817,   -0.60950196,    0.65839870,    0.05405392,    0.02934520,    0.01485905,    0.91842330],
 [   0.02517852,   -0.25895423,    0.70790810,   -0.04330471,   -0.02187616,    0.06794184,    0.90855193],
 [  -0.03846490,   -0.78978634,    0.09970292,    0.00590947,    0.05622746,   -0.04575816,    0.59687877],
 [   0.05820185,   -0.07647377,    0.12298247,   -0.01218793,   -0.06382927,   -0.23598006,    0.27823430],
 [  -0.18932047,   -0.81394637,   -0.09247136,   -0.00448679,   -0.08681298,   -0.03555256,    0.72954730],
 [   0.28454220,   -0.47155768,    0.38009918,   -0.06480382,   -0.09157664,   -0.12390063,    0.88782823],
 [  -0.29095513,   -0.81316280,    0.55652180,   -0.07916620,   -0.04889646,   -0.00497413,    0.59475356]],
[[  -0.01423621,    0.22690511,   -0.16231877,   -0.08919455,    0.01909237,    0.01592675,    0.80485240],
 [   0.14121789,   -0.51700320,    0.69938743,   -0.08398525,   -0.05039765,   -0.22561544,    0.33190423],
 [  -0.03211093,   -0.06018382,    0.70887610,   -0.05535643,   -0.02658354,   -0.14933452,    0.50529130],
 [   0.11048925,   -0.21046263,    0.17101705,   -0.02387796,    0.02220145,   -0.21987796,   -0.05961222],
 [   0.20817757,   -0.61272230,   -0.13698918,    0.03611638,   -0.07406700,   -0.10408540,    0.89817290],
 [   0.09534305,   -0.41312566,   -0.07203433,    0.02231567,    0.03530423,    0.04727703,    0.62640590],
 [  -0.16526806,   -0.13260996,    0.46118290,   -0.06209633,    0.04266892,    0.00691596,    0.93402463],
 [  -0.21900263,   -0.64324623,    0.01864368,   -0.05786404,   -0.03802192,   -0.10101809,    0.50406380]],
[[   0.24262238,   -0.32314730,    0.76833355,    0.03193358,   -0.09400973,    0.01623493,    0.78225046],
 [  -0.19120444,   -0.16513681,   -0.17008491,   -0.02054340,   -0.01854029,   -0.16629344,    0.93087550],
 [  -0.10736741,   -0.46194738,    0.09423983,   -0.01788402,    0.00239911,    0.03706470,    0.13693051],
 [   0.07193235,   -0.82892080,    0.21191618,    0.01419903,   -0.03962151,   -0.03946297,    0.57919880],
 [   0.26385403,    0.26698650,    0.76837397,    0.04002494,   -0.08579183,    0.06743425,    0.50043480],
 [  -0.00594047,    0.00633377,    0.68950310,    0.02766208,   -0.01217059,   -0.08980645,    0.91581040],
 [   0.13894662,   -0.39016150,   -0.16721939,    0.04395929,   -0.06484466,   -0.05704926,    0.51860374],
 [  -0.13394505,   -0.71924880,    0.48821450,   -0.07076398,   -0.08434260,   -0.01927234,    0.44204634]],
[[  -0.26685882,   -0.83382020,    0.67991334,    0.05818109,   -0.00920904,   -0.15050533,    0.69410770],
 [   0.19145322,   -0.02771926,   -0.10424177,    0.05122240,    0.01205986,   -0.10157636,    0.25777400],
 [  -0.08893049,   -0.49759325,   -0.20770788,   -0.03257968,   -0.09831297,   -0.12385990,    0.86106980],
 [   0.27988994,   -0.04180884,   -0.06772058,    0.01913941,    0.05859074,   -0.02919194,    0.36722863],
 [   0.03157359,   -0.40866990,    0.61481464,   -0.09117340,   -0.09937295,   -0.08348650,    0.70042130],
 [  -0.28925294,   -0.62677306,    0.37811643,   -0.06772463,    0.01260573,   -0.13167429,    0.28380650],
 [  -0.00236955,   -0.28091848,   -0.11962137,   -0.07635091,   -0.05675795,   -0.15142110,    0.47319870],
 [  -0.32417350,   -0.01879406,   -0.13142973,   -0.08921819,   -0.09703661,   -0.08327541,   -0.02794056]],
[[   0.25792617,   -0.00300229,    0.37660800,   -0.04685238,    0.05348377,   -0.05520193,    0.74412490],
 [   0.04590249,   -0.07399899,   -0.04751955,   -0.07565484,   -0.01447763,   -0.04199156,   -0.04603508],
 [   0.26214945,    0.19627476,   -0.12977520,    0.02613395,   -0.00520366,   -0.17107353,   -0.01105031],
 [  -0.07785514,   -0.32899780,   -0.09025405,   -0.03373562,    0.01363293,    0.00499007,    0.08398855],
 [   0.18739325,    0.02625650,    0.28484958,   -0.00613664,   -0.01123306,   -0.08220898,    0.80074440],
 [  -0.06141251,   -0.27245694,    0.76384130,   -0.07312378,   -0.03457505,   -0.15526298,    0.49149078],
 [  -0.22725716,   -0.46449152,    0.25699437,    0.04446377,   -0.02906028,    0.01640603,    0.46354347],
 [  -0.05252513,   -0.65908170,    0.35301614,   -0.05915632,    0.01763520,   -0.10705960,    0.38707918]],
[[   0.07596695,    0.08780742,    0.77582407,    0.04364777,   -0.08465850,   -0.14520863,    0.73990840],
 [  -0.19287841,   -0.00025690,   -0.13493803,   -0.00804287,    0.04104747,    0.01578918,    0.74981190],
 [  -0.20107412,    0.02339065,    0.40493375,   -0.06585702,   -0.04126096,   -0.10331783,    0.73602730],
 [   0.18697750,   -0.74249680,   -0.00281046,    0.00704502,   -0.05225958,   -0.16008450,    0.01609085],
 [  -0.18997267,   -0.36326346,   -0.19530822,   -0.07816781,   -0.06192324,   -0.11454390,    0.80489820],
 [  -0.10693823,   -0.30052190,    0.08292675,   -0.01513033,   -0.07352376,   -0.15348418,    0.36695850],
 [   0.07014069,    0.37816405,    0.57810235,   -0.05362078,   -0.03045313,    0.03538099,    0.58465326],
 [   0.15150067,    0.10214746,    0.62867660,   -0.02370854,   -0.06726541,    0.02666017,    0.29507834]],
[[  -0.07717261,   -0.62732345,    0.08694285,   -0.08895740,   -0.08076153,   -0.18785015,   -0.02784339],
 [  -0.30907574,   -0.21976691,    0.69083460,   -0.04674264,   -0.05997643,   -0.16414010,    0.46281487],
 [   0.11835733,   -0.21101665,   -0.09473623,    0.01806735,   -0.04631552,   -0.00126162,    0.78556806],
 [   0.23886281,   -0.39676300,    0.17222825,   -0.03533455,   -0.00154112,   -0.16128956,    0.72544420],
 [   0.10909680,   -0.67489636,    0.75000200,    0.02914919,   -0.10095438,   -0.02858464,    0.54743080],
 [  -0.22466311,   -0.15008569,    0.52847220,    0.00085474,   -0.09327558,   -0.04546437,    0.03064486],
 [  -0.07599142,   -0.11741298,   -0.04069279,   -0.05350723,    0.02377979,   -0.10169008,    0.59603960],
 [  -0.23285499,    0.23294806,    0.09861496,    0.04864231,   -0.01114944,   -0.04699415,    0.49257034]],
[[  -0.33240030,   -0.76324195,    0.64213340,    0.03880452,   -0.01083275,   -0.01040770,    0.76809270],
 [   0.25255853,   -0.12444985,    0.65329844,   -0.08985522,   -0.06801133,   -0.13610303,    0.59562814],
 [   0.16814959,   -0.52863336,    0.38227242,   -0.01866668,    0.02894893,   -0.14173960,    0.92976254],
 [  -0.12113878,    0.37853825,    0.00480723,    0.01740935,   -0.03057948,   -0.16317528,    0.84369500],
 [  -0.27844974,   -0.28675288,   -0.04198644,   -0.03323535,    0.00969539,   -0.16502434,    0.99257916],
 [   0.13313577,    0.09878427,    0.07791057,   -0.03236932,   -0.00833951,   -0.14767289,    0.05707462],
 [   0.00928545,   -0.06110680,    0.71975434,    0.01020374,    0.03897345,   -0.06566674,    0.14514005],
 [  -0.31147707,    0.00889885,    0.58021104,    0.03936787,    0.02792272,   -0.09702562,   -0.03723207]],
[[   0.12659582,    0.02664340,    0.22695690,    0.05965218,    0.00351200,   -0.04981254,    0.35454696],
 [   0.17702794,    0.22574651,    0.48085743,    0.00443751,   -0.01442036,   -0.22561908,    0.52644080],
 [  -0.10040346,    0.10764611,   -0.23721880,   -0.09292942,   -0.02617657,   -0.11895921,    0.67083734],
 [  -0.15951435,   -0.30274278,    0.61214580,    0.01454529,   -0.00600825,   -0.11648524,    0.86474220],
 [  -0.22011685,   -0.06859094,    0.50722310,   -0.09339935,   -0.08812235,    0.05277404,    0.13309637],
 [  -0.04018420,   -0.13978225,    0.19441300,   -0.03380864,   -0.05522747,   -0.21103044,    0.26496792],
 [  -0.15642382,    0.10201776,   -0.14452755,    0.02473946,   -0.09753170,   -0.21097600,    0.74942094],
 [  -0.10686794,    0.31106830,    0.05068091,   -0.04876743,   -0.01941990,   -0.00676365,    0.65040920]],
[[  -0.08620028,   -0.71192390,   -0.01948388,   -0.07955055,   -0.06751666,   -0.07599321,    0.99194676],
 [  -0.01790637,   -0.61046666,    0.03483909,    0.05252053,   -0.06659617,   -0.14580733,    0.53626720],
 [  -0.07914501,    0.14951134,    0.48343480,   -0.05161333,    0.03714792,   -0.08195387,    0.05255207],
 [   0.18137753,    0.14042050,    0.28869104,   -0.03853970,   -0.08776364,   -0.21539302,    0.09715706],
 [  -0.06046829,   -0.10201293,    0.69472400,    0.06275953,   -0.02653021,    0.04366979,    0.72259027],
 [  -0.10178171,   -0.77422070,    0.21015674,   -0.06462009,   -0.01448946,   -0.07073075,    0.92209214],
 [   0.17141885,   -0.54175070,    0.23139599,    0.02735215,   -0.06275660,   -0.13814276,    0.34766072],
 [  -0.24920705,   -0.16250187,    0.74448460,   -0.02088559,   -0.09695192,    0.05987886,    0.36577750]],
[[   0.15486783,   -0.61685896,    0.56500685,    0.06252737,    0.03973491,    0.02852967,    0.21677531],
 [  -0.10926621,    0.21780622,    0.51966100,    0.05607168,   -0.07711536,   -0.04898329,    0.88584750],
 [   0.00507143,   -0.09957755,    0.04558688,   -0.04093509,   -0.04353308,   -0.20470210,    0.59843844],
 [   0.20115370,   -0.81399536,    0.24356696,   -0.05042077,   -0.02765485,   -0.20503900,    0.59050910],
 [  -0.07983971,   -0.23730910,    0.19327274,   -0.07850616,   -0.07499804,    0.00129595,    0.83450430],
 [   0.23602134,   -0.27324128,    0.33625007,    0.02928636,   -0.02214392,   -0.22720003,    0.40264815],
 [   0.26132178,    0.11151057,    0.18105197,   -0.01893316,    0.04683761,    0.04989141,   -0.06755669],
 [   0.10339397,   -0.41961694,    0.07813004,    0.05783278,    0.02239966,   -0.15768620,    0.40838300]],
[[   0.15059716,   -0.55408190,   -0.13910490,    0.03766195,   -0.08071688,   -0.06961231,    0.31095874],
 [  -0.18509406,   -0.01738656,   -0.18912816,   -0.05619378,    0.02603902,    0.02927279,    0.36485404],
 [   0.28648698,    0.31654680,    0.33888435,    0.00858814,    0.05897890,   -0.07391271,   -0.07924467],
 [   0.04358196,   -0.41711680,    0.55769970,    0.00894062,   -0.01761635,   -0.02917290,    0.34811294],
 [   0.02747747,   -0.16709197,    0.25612520,    0.05033064,    0.02453490,   -0.18022782,    0.05333485],
 [   0.13997859,   -0.28649730,    0.50346470,   -0.09289309,   -0.00243175,    0.02648064,    0.26100970],
 [  -0.29470600,   -0.24689084,    0.03165212,    0.02560306,   -0.00801039,    0.02062258,    0.28826225],
 [   0.24652815,   -0.62495410,    0.69870930,   -0.02360176,   -0.06533158,    0.00431150,    0.64842410]],
[[  -0.18358886,   -0.74959790,    0.69632995,   -0.05836391,    0.05509730,    0.05619198,    0.18353264],
 [   0.21296704,   -0.62343740,   -0.09174278,    0.03620157,   -0.07798733,   -0.22388613,    0.16242010],
 [  -0.29462364,   -0.56051886,    0.03408676,   -0.08029686,   -0.07378080,    0.06536761,    0.49434440],
 [  -0.17085540,   -0.39060065,    0.13707477,    0.05647135,   -0.02571609,   -0.05506489,    0.33544123],
 [   0.02183202,   -0.57727120,    0.19715664,   -0.08775686,    0.00952055,   -0.21426652,    0.05389386],
 [   0.15747166,   -0.25282532,    0.17490676,   -0.01571370,   -0.09053193,   -0.05973589,    0.84155580],
 [  -0.22982916,    0.11012888,   -0.04272592,    0.00724915,   -0.08803584,    0.06884760,    0.38326204],
 [   0.00970846,   -0.78555370,    0.38132375,   -0.04822778,   -0.06982891,   -0.05261187,    0.50923395]],
[[   0.20620900,    0.13296098,    0.09872636,   -0.00701104,   -0.08416009,   -0.00776465,    0.23071818],
 [  -0.08640334,   -0.33965945,    0.77966404,   -0.08194447,   -0.02577635,    0.02896231,    0.82161635],
 [  -0.23283267,    0.36122274,   -0.03622688,    0.05423404,   -0.04647377,   -0.11086978,    0.57521254],
 [  -0.15263335,   -0.68944510,    0.09773642,    0.05403699,    0.05732895,    0.05365276,    0.17939012],
 [   0.20663965,    0.02559954,   -0.04665107,   -0.02110620,   -0.09153587,   -0.17779098,    0.05427837],
 [  -0.07355130,   -0.79358090,    0.31445414,   -0.00065914,   -0.09263497,    0.03358066,    0.68224060],
 [  -0.25473392,   -0.79749350,    0.32606375,    0.02962644,    0.04193377,   -0.17193120,    0.45268798],
 [  -0.21792123,   -0.66330594,   -0.16991800,    0.04017854,   -0.09255441,   -0.21944822,    0.45495075]],
[[  -0.17652340,   -0.64675903,    0.39525372,   -0.06089334,    0.02141328,   -0.20404476,    0.31062007],
 [  -0.02087623,   -0.28995252,   -0.20332354,   -0.06188162,   -0.02765785,    0.05376875,    0.20849048],
 [  -0.20742630,    0.22718465,    0.47661126,   -0.08533260,   -0.07188030,   -0.12630545,    0.98854550],
 [  -0.13408037,    0.33448315,    0.58145380,   -0.02473510,   -0.00465405,   -0.14300533,    0.70700760],
 [   0.20510710,   -0.64313847,    0.22486180,   -0.03706741,   -0.02831851,   -0.00909027,   -0.01232858],
 [  -0.27173543,   -0.69006750,    0.07790279,   -0.04123763,    0.03674646,    0.07007000,    0.16070493],
 [  -0.13056852,   -0.61667150,    0.58630020,   -0.01577155,   -0.08119453,   -0.23284897,    0.38561332],
 [  -0.09692258,   -0.30364560,    0.46026950,   -0.00022953,   -0.05903238,   -0.21096796,    0.40301007]]])

            # Original labels before removal
            mug   = [2,4,5,6,11,13,15,16,17,20,23,30,31,40,47,53,59,61,65,66,69]
            sushi = [7,28,32,44,45,51,52,58,64,67]
            apple = [1,3,8,46,18,22,24,27,29,33]

            # Create full labels array
            labels = np.empty(len(random_points), dtype=int)
            for i in apple:
                labels[i] = 0
            for i in mug:
                labels[i] = 1
            for i in sushi:
                labels[i] = 2
                
            # Reshape random_points from (70, 1, 8, 7) to (70, 8, 7)
            # random_points = np.squeeze(random_points, axis=1)
            print(random_points.shape, labels.shape)
            # Remove specific indices from both points and labels together
            indices_to_remove = []
            indices_to_keep = [i for i in range(len(random_points)) if i not in indices_to_remove]
            # indices_to_keep = [3,8,9,19,21,22,26,27,29,43,    ]


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
            

            obj_init_positions = val_dataset.normalized_train_data["blocks_init_dict"][-3]
            # Get real observation from environment (only once for all latent points)
            env_ids = torch.arange(env.num_envs, device=env.device)
            apply_demo_objects(env, obj_init_positions, env_ids)
            
            # Force physics to settle deterministically
            for _ in range(20):  # Run exactly 10 simulation steps
                env.step(torch.zeros((env.num_envs, 7), device=env.device))
            
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
    
    # Set consistent axis limits for all subplots
    if len(axes_list) > 0:
        # Collect all x and y data points across all subplots
        all_x_data, all_y_data = [], []
        for plot_idx, t_idx in enumerate(key_timesteps):
            if plot_idx >= len(axes_list) or t_idx >= len(trajectories_data[0]['flow_x_values']):
                continue
            for point_idx, traj in enumerate(trajectories_data):
                if 'flow_x_values' in traj and t_idx < len(traj['flow_x_values']):
                    x_values = traj['flow_x_values'][t_idx]
                    if x_values.shape[0] == 1:
                        x_values = x_values[0]
                    x, y = x_values[action_step, :2]
                    all_x_data.append(float(x))
                    all_y_data.append(float(y))
        
        if all_x_data and all_y_data:
            x_margin = (max(all_x_data) - min(all_x_data)) * 0.1
            y_margin = (max(all_y_data) - min(all_y_data)) * 0.1
            xlims = [min(all_x_data) - x_margin, max(all_x_data) + x_margin]
            ylims = [min(all_y_data) - y_margin, max(all_y_data) + y_margin]
            
            # Apply same limits to all visible subplots
            for i in range(min(len(key_timesteps), len(axes_list))):
                axes_list[i].set_xlim(xlims[0], xlims[1])
                axes_list[i].set_ylim(ylims[0], ylims[1])
    
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
    
    # Set consistent axis limits for all 3D subplots
    if len(step_indices) > 0:
        # Collect all x, y, z data points across all subplots
        all_x_data, all_y_data, all_z_data = [], [], []
        for plot_idx, t_idx in enumerate(step_indices):
            if plot_idx >= max_3d_plots:
                break
            for point_idx, traj in enumerate(trajectories_data):
                if 'flow_x_values' in traj and t_idx < len(traj['flow_x_values']):
                    x_values = traj['flow_x_values'][t_idx]
                    if x_values.shape[0] == 1:
                        x_values = x_values[0]
                    x, y, z = x_values[action_step, :3]
                    all_x_data.append(float(x))
                    all_y_data.append(float(y))
                    all_z_data.append(float(z))
        
        if all_x_data and all_y_data and all_z_data:
            x_margin = (max(all_x_data) - min(all_x_data)) * 0.1
            y_margin = (max(all_y_data) - min(all_y_data)) * 0.1
            z_margin = (max(all_z_data) - min(all_z_data)) * 0.1
            xlims = [min(all_x_data) - x_margin, max(all_x_data) + x_margin]
            ylims = [min(all_y_data) - y_margin, max(all_y_data) + y_margin]
            zlims = [min(all_z_data) - z_margin, max(all_z_data) + z_margin]
            
            # Apply same limits to all visible 3D subplots
            for plot_idx in range(min(len(step_indices), max_3d_plots)):
                ax = fig.axes[plot_idx]
                ax.set_xlim(xlims[0], xlims[1])
                ax.set_ylim(ylims[0], ylims[1])
                if hasattr(ax, 'set_zlim'):  # Check if it's a 3D axis
                    ax.set_zlim(zlims[0], zlims[1])
    
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
    
    # Set consistent axis limits for all 3D subplots
    if len(selected_flow_steps) > 0:
        # Collect all x, y, z data points across all trajectories and flow steps
        all_x_data, all_y_data, all_z_data = [], [], []
        for plot_idx, t_idx in enumerate(selected_flow_steps):
            for point_idx, traj in enumerate(trajectories_data):
                if 'flow_x_values' in traj and t_idx < len(traj['flow_x_values']):
                    x_values = traj['flow_x_values'][t_idx]
                    if x_values.shape[0] == 1:
                        x_values = x_values[0]
                    
                    # Extract XYZ coordinates for complete trajectory
                    trajectory_x = x_values[:, 0]
                    trajectory_y = x_values[:, 1]
                    trajectory_z = x_values[:, 2]
                    
                    all_x_data.extend([float(x) for x in trajectory_x])
                    all_y_data.extend([float(y) for y in trajectory_y])
                    all_z_data.extend([float(z) for z in trajectory_z])
        
        if all_x_data and all_y_data and all_z_data:
            x_margin = (max(all_x_data) - min(all_x_data)) * 0.1
            y_margin = (max(all_y_data) - min(all_y_data)) * 0.1
            z_margin = (max(all_z_data) - min(all_z_data)) * 0.1
            xlims = [min(all_x_data) - x_margin, max(all_x_data) + x_margin]
            ylims = [min(all_y_data) - y_margin, max(all_y_data) + y_margin]
            zlims = [min(all_z_data) - z_margin, max(all_z_data) + z_margin]
            
            # Apply same limits to all visible 3D subplots
            for plot_idx in range(len(selected_flow_steps)):
                ax = fig.axes[plot_idx]
                ax.set_xlim(xlims[0], xlims[1])
                ax.set_ylim(ylims[0], ylims[1])
                if hasattr(ax, 'set_zlim'):
                    ax.set_zlim(zlims[0], zlims[1])
    
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
        
        # Set consistent axis limits for all 2D PCA subplots
        if num_selected > 0:
            # Collect all PCA data points across all flow timesteps for this action step
            all_pc1_data, all_pc2_data = [], []
            for plot_idx, t_idx in enumerate(selected_timesteps):
                if plot_idx >= num_selected:
                    break
                max_available_steps = len(trajectories_data[0]['flow_x_values']) - 1
                if t_idx > max_available_steps:
                    continue
                
                # Collect data for this timestep
                all_points = []
                for point_idx, traj in enumerate(trajectories_data):
                    if 'flow_x_values' in traj and t_idx < len(traj['flow_x_values']):
                        x_values = traj['flow_x_values'][t_idx]
                        if x_values.shape[0] == 1:
                            x_values = x_values[0]
                        action_vector = x_values[action_step, :]
                        all_points.append(action_vector)
                
                if len(all_points) > 0:
                    data = np.vstack(all_points)
                    pca_2d = PCA(n_components=2)
                    data_pca_2d = pca_2d.fit_transform(data)
                    all_pc1_data.extend(data_pca_2d[:, 0])
                    all_pc2_data.extend(data_pca_2d[:, 1])
            
            if all_pc1_data and all_pc2_data:
                pc1_margin = (max(all_pc1_data) - min(all_pc1_data)) * 0.1
                pc2_margin = (max(all_pc2_data) - min(all_pc2_data)) * 0.1
                pc1_limits = [min(all_pc1_data) - pc1_margin, max(all_pc1_data) + pc1_margin]
                pc2_limits = [min(all_pc2_data) - pc2_margin, max(all_pc2_data) + pc2_margin]
                
                # Apply same limits to all visible 2D PCA subplots
                for i in range(min(num_selected, len(axes_2d_list))):
                    axes_2d_list[i].set_xlim(pc1_limits[0], pc1_limits[1])
                    axes_2d_list[i].set_ylim(pc2_limits[0], pc2_limits[1])
            
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
        
        # Set consistent axis limits for all 3D PCA subplots
        if num_selected > 0:
            # Collect all PCA 3D data points across all flow timesteps for this action step
            all_pc1_3d_data, all_pc2_3d_data, all_pc3_3d_data = [], [], []
            for plot_idx_3d, t_idx_3d in enumerate(selected_timesteps):
                if plot_idx_3d >= num_selected:
                    break
                max_available_steps = len(trajectories_data[0]['flow_x_values']) - 1
                if t_idx_3d > max_available_steps:
                    continue
                
                # Collect data for this timestep
                all_points_3d = []
                for point_idx, traj in enumerate(trajectories_data):
                    if 'flow_x_values' in traj and t_idx_3d < len(traj['flow_x_values']):
                        x_values = traj['flow_x_values'][t_idx_3d]
                        if x_values.shape[0] == 1:
                            x_values = x_values[0]
                        action_vector = x_values[action_step, :]
                        all_points_3d.append(action_vector)
                
                if len(all_points_3d) > 0:
                    data_3d = np.vstack(all_points_3d)
                    pca_3d_temp = PCA(n_components=3)
                    data_pca_3d_temp = pca_3d_temp.fit_transform(data_3d)
                    all_pc1_3d_data.extend(data_pca_3d_temp[:, 0])
                    all_pc2_3d_data.extend(data_pca_3d_temp[:, 1])
                    all_pc3_3d_data.extend(data_pca_3d_temp[:, 2])
            
            if all_pc1_3d_data and all_pc2_3d_data and all_pc3_3d_data:
                pc1_3d_margin = (max(all_pc1_3d_data) - min(all_pc1_3d_data)) * 0.1
                pc2_3d_margin = (max(all_pc2_3d_data) - min(all_pc2_3d_data)) * 0.1
                pc3_3d_margin = (max(all_pc3_3d_data) - min(all_pc3_3d_data)) * 0.1
                pc1_3d_limits = [min(all_pc1_3d_data) - pc1_3d_margin, max(all_pc1_3d_data) + pc1_3d_margin]
                pc2_3d_limits = [min(all_pc2_3d_data) - pc2_3d_margin, max(all_pc2_3d_data) + pc2_3d_margin]
                pc3_3d_limits = [min(all_pc3_3d_data) - pc3_3d_margin, max(all_pc3_3d_data) + pc3_3d_margin]
                
                # Apply same limits to all visible 3D PCA subplots
                for i in range(min(num_selected, len(fig_3d.axes))):
                    ax = fig_3d.axes[i]
                    ax.set_xlim(pc1_3d_limits[0], pc1_3d_limits[1])
                    ax.set_ylim(pc2_3d_limits[0], pc2_3d_limits[1])
                    if hasattr(ax, 'set_zlim'):
                        ax.set_zlim(pc3_3d_limits[0], pc3_3d_limits[1])
        
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
    
    # ==================== ORIGINAL XYZ PLOTS ====================
    # Prepare arguments for parallel processing
    plot_2d_args = [
        (action_step, trajectories_data, labels, label_names, colors, exp_idx, output_dir, num_flow_steps, pred_horizon)
        for action_step in range(pred_horizon)
    ]
    
    plot_3d_args = [
        (action_step, trajectories_data, labels, label_names, colors, exp_idx, output_dir, num_flow_steps, pred_horizon)
        for action_step in range(pred_horizon)
    ]
    
    # Generate 2D plots in parallel
    print("🚀 Starting parallel 2D plot generation...")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_action = {executor.submit(create_2d_plot_for_action_step, args): args[0] for args in plot_2d_args}
        
        for future in as_completed(future_to_action):
            action_step = future_to_action[future]
            try:
                result = future.result()
                print(f"✅ {result}")
            except Exception as exc:
                print(f"❌ 2D plot for action step {action_step} generated an exception: {exc}")
    
    # Generate 3D plots in parallel
    print("🚀 Starting parallel 3D plot generation...")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_action = {executor.submit(create_3d_plot_for_action_step, args): args[0] for args in plot_3d_args}
        
        for future in as_completed(future_to_action):
            action_step = future_to_action[future]
            try:
                result = future.result()
                print(f"✅ {result}")
            except Exception as exc:
                print(f"❌ 3D plot for action step {action_step} generated an exception: {exc}")
    
    # Generate combined 3D trajectory plot
    print("🚀 Creating single 3D trajectory plot with multiple flow timesteps...")
    selected_flow_steps = key_timesteps
    
    try:
        result = create_combined_3d_trajectory_plot(
            selected_flow_steps, trajectories_data, labels, label_names, colors, exp_idx, output_dir, pred_horizon
        )
        print(f"✅ {result}")
    except Exception as exc:
        print(f"❌ Combined 3D trajectory plot generated an exception: {exc}")
    
    # ==================== PCA ANALYSIS PLOTS ====================
    print("\n" + "="*80)
    print("NOW GENERATING PCA ANALYSIS PLOTS")
    print("="*80)
    
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