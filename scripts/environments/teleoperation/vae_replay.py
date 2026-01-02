import sys
sys.path.append("/workspace/isaaclab/source/droid/droid/controllers/")
sys.path.append('/home/shubham/summer/serl-flow/conditional-flow-matching')

import cv2
import argparse
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm
import imageio.v3 as iio
from scipy.spatial.transform import Rotation as R

from isaaclab.app import AppLauncher
from flow_policy.make_networks import instantiate_vae_artifacts

parser = argparse.ArgumentParser(description="Run VAE inference in Isaac Lab.")
parser.add_argument("--robot", type=str, default="franka", choices=["franka", "gr1t2"], help="Robot type")
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

# Configuration
TRAIN_DATASET_PATH = "source/serl-flow/dataset/train_sushi.pkl"
VAL_DATASET_PATH = "source/serl-flow/dataset/train_paired.pkl"
task_name = "mixed_robot_vae_16dim_0.01kl"
epoch_num = "2000"
args_cli.checkpoint = "source/serl-flow/chkpts/" + task_name + "/epoch_" + epoch_num +".pt"

if args_cli.enable_pinocchio:
    import pinocchio  # noqa: F401

app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

import gymnasium as gym
import omni.log

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
if args_cli.enable_pinocchio:
    import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401

def load_vae_model(checkpoint_path, device='cuda'):
    """Load trained VAE model from checkpoint."""
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    cfg = checkpoint['config']
    cfg.dataset.val_dataset_path = VAL_DATASET_PATH
    cfg.dataset.dataset_path = TRAIN_DATASET_PATH
    cfg.with_image = True
    
    nets, optimizers, dataloader, val_dataloader, dataset, val_dataset, device = instantiate_vae_artifacts(cfg)
    nets['vae'].load_state_dict(checkpoint['state_dict'])
    nets['vae'].eval()
    
    print(f"VAE model loaded successfully. Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"Pred horizon: {cfg.pred_horizon}, Obs horizon: {cfg.obs_horizon}, Action horizon: {cfg.action_horizon}")
    
    return nets['vae'], val_dataset, cfg

def get_observation_from_env(env):
    """Extract observation from Isaac Lab environment."""
    # Get robot state (end-effector pose)
    eef_idx = env.scene["robot"].data.body_names.index("panda_hand")
    eef_pos_w = env.scene["robot"].data.body_pos_w[0, eef_idx]
    eef_quat_w = env.scene["robot"].data.body_quat_w[0, eef_idx][[1, 2, 3, 0]]
    eef_rpy = torch.tensor(R.from_quat(eef_quat_w.detach().cpu().numpy()).as_euler('xyz'), dtype=torch.float32, device=env.device)

    # Get gripper state
    joint_names = env.scene["robot"].data.joint_names
    joint_positions = env.scene["robot"].data.joint_pos[0]
    gripper_indices = [joint_names.index("panda_finger_joint1"), joint_names.index("panda_finger_joint2")]
    gripper_state = joint_positions[gripper_indices].mean()
    gripper_state = torch.tensor([gripper_state], dtype=torch.float32, device=env.device)
    
    # Get object positions
    object_names = ["apple", "mug", "sushi"]
    object_positions = []
    for obj_name in object_names:
        if obj_name in env.scene.keys():
            obj_pos = env.scene[obj_name].data.body_pos_w[0, 0].to(env.device)
            object_positions.append(obj_pos)
    object_positions = torch.cat(object_positions, dim=-1) if object_positions else torch.zeros(9, device=env.device)
    
    obs = torch.cat([
        eef_pos_w,
        eef_rpy,
        gripper_state,
        object_positions
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

def run_vae_inference(env, dataset, episode_idx, vae_model, cfg, device, save_trajectory=False):
    """Run VAE inference and replay actions."""
    # Reset environment
    obs, _ = env.reset()
    env_ids = torch.arange(env.num_envs, device=env.device)

    print(f"Starting episode...")
    
    end_idx = dataset.episode_ends[episode_idx]
    start_idx = 0 if episode_idx == 0 else dataset.episode_ends[episode_idx - 1]

    # Apply demo objects
    apply_demo_objects(env, dataset.train_data["blocks_init_dict"][episode_idx], env_ids)
    
    # Get dataset data for this episode
    gt_states = torch.tensor(dataset.normalized_train_data['state'][start_idx:end_idx]).to(device)
    gt_robot_images = dataset.train_data['gt_robot_images'][start_idx:end_idx]
    gt_actions = torch.tensor(dataset.normalized_train_data['action'][start_idx:end_idx]).to(device)
    max_steps = len(gt_states)
    
    print("🚀 Starting VAE trajectory inference...")
    vae_model.eval()
    
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
            
            # Prepare ground truth data
            gt_data = {
                'images': gt_robot_images
            }
        
        # Get initial observation
        current_obs = gt_states[0]
        current_obs[6] = 0.04  # Set initial gripper position
        
        # Initialize gripper
        action_dim = 7 
        _action = torch.zeros(action_dim, device=device)
        _action[-1] = 1.0  # Open gripper initially
        for _ in range(10): 
            _ = env.step(_action.unsqueeze(0))
        
        image = get_image(env)
        
        for step_idx in tqdm(range(max_steps)):
            # Sample random latent and generate action using VAE
            predicted_actions, mu, logvar = vae_model(gt_actions[step_idx:step_idx+8].unsqueeze(0), gt_states[step_idx:step_idx+1])
            # Take the first action
            action = predicted_actions[0][0]
            
            # Process gripper action
            if action[-1] < 0.5:
                action[-1] = -0.01  # Close gripper
            else:
                action[-1] = 1.0    # Open gripper
            
            print(f"Step {step_idx}: Action gripper = {action[-1]}")
            
            # Apply action to environment
            obs, reward, terminated, truncated, info = env.step(action.unsqueeze(0))
            
            # Get new observation and image
            current_obs = get_observation_from_env(env)
            image = get_image(env)
            
            # Record trajectory
            if save_trajectory:
                object_poses = {"apple": current_obs[7:10], "mug": current_obs[10:13], "sushi": current_obs[13:16]}
                trajectory_data['observations'].append(current_obs.cpu().numpy())
                trajectory_data['images'].append(image.cpu().numpy())
                trajectory_data['actions'].append(action.cpu().numpy())
                trajectory_data['object_poses'].append(object_poses)
                trajectory_data['eef_poses'].append({
                    'pos': current_obs[:3].cpu().numpy(),
                    'rpy': current_obs[3:6].cpu().numpy()
                })
            
            if terminated or truncated:
                print(f"Episode terminated at step {step_idx}")
                break
    
    print(f"✅ VAE trajectory completed: {len(trajectory_data['actions']) if save_trajectory else step_idx} steps")
    if save_trajectory:
        return trajectory_data, gt_data
    return None

def save_vae_video(pred_data, gt_data, episode_idx, output_dir):
    """Save side-by-side VAE inference and ground truth video."""
    output_dir = Path(output_dir) / task_name / epoch_num
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"vae_vs_gt_{episode_idx:03d}.mp4"
    
    camera_data = pred_data['camera_params']
    pred_imgs, pred_eef, pred_objs = pred_data['images'], pred_data['eef_poses'], pred_data['object_poses']
    gt_imgs = gt_data['images']
    
    # Ensure both have same number of frames (use minimum)
    n_frames = min(len(pred_imgs), len(gt_imgs))
    frames = []
    
    # Project all trajectories once for predicted data
    def project_all(eef_list, obj_poses, cam, img_shape):
        eef_proj = [project_pose_to_image(np.array(e['pos']), cam, img_shape) for e in eef_list]
        obj_names = ["apple", "mug", "sushi"]
        obj_proj = {name: [] for name in obj_names}
        for obj_data in obj_poses:
            for name in obj_names:
                pos = obj_data[name].detach().cpu().numpy() if hasattr(obj_data[name], 'detach') else np.array(obj_data[name])
                obj_proj[name].append(project_pose_to_image(pos, cam, img_shape))
        return eef_proj, obj_proj
    
    if camera_data:
        img_shape = pred_imgs[0].shape
        pred_eef_proj, pred_obj_proj = project_all(pred_eef, pred_objs, camera_data, img_shape)

    for i in range(n_frames):
        # Process predicted image
        pred_img = np.array(pred_imgs[i], dtype=np.uint8)
        pred_img_bgr = cv2.cvtColor(pred_img, cv2.COLOR_RGB2BGR)
        
        if camera_data and pred_eef_proj:
            pred_img_bgr = draw_trajectory_on_frame(pred_img_bgr, pred_eef_proj, i, is_eef=True)
            for proj_list in pred_obj_proj.values():
                pred_img_bgr = draw_trajectory_on_frame(pred_img_bgr, proj_list, i)
        
        # Process ground truth image
        gt_img = np.array(gt_imgs[i], dtype=np.uint8)
        gt_img_bgr = cv2.cvtColor(gt_img, cv2.COLOR_RGB2BGR)
        
        # Add labels
        pred_label = 'VAE Inference'
        gt_label = 'Ground Truth'
        
        # Add label to predicted image
        text_size = cv2.getTextSize(pred_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        text_x = (pred_img_bgr.shape[1] - text_size[0]) // 2
        cv2.putText(pred_img_bgr, pred_label, (text_x, pred_img_bgr.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 4)
        cv2.putText(pred_img_bgr, pred_label, (text_x, pred_img_bgr.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Add label to ground truth image
        text_size = cv2.getTextSize(gt_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        text_x = (gt_img_bgr.shape[1] - text_size[0]) // 2
        cv2.putText(gt_img_bgr, gt_label, (text_x, gt_img_bgr.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 4)
        cv2.putText(gt_img_bgr, gt_label, (text_x, gt_img_bgr.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # Concatenate images side by side
        combined_img = np.concatenate([pred_img_bgr, gt_img_bgr], axis=1)
        frames.append(cv2.cvtColor(combined_img, cv2.COLOR_BGR2RGB))
    
    iio.imwrite(output_path, frames, fps=30, codec="libx264", quality=8)
    print(f"[INFO] Side-by-side video saved: {output_path} ({n_frames} frames)")
    return output_path

def main():
    """Main function to run VAE inference."""
    
    # Load VAE model
    try:
        vae_model, val_dataset, cfg = load_vae_model(args_cli.checkpoint)
    except Exception as e:
        omni.log.error(f"Failed to load VAE model: {e}")
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
    print(f"VAE INFERENCE")
    print(f"Model: {args_cli.checkpoint}")
    print(f"Robot: {args_cli.robot}")
    print(f"Max episode length: {args_cli.max_episode_length}")
    if args_cli.save_trajectories:
        print(f"Saving trajectories to: {args_cli.output_dir}")
    print(f"{'='*60}\n")
    
    # Run episodes
    total_episodes = len(val_dataset.episode_ends)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Randomize episode selection
    np.random.seed(42)  # For reproducibility
    num_episodes_to_run = total_episodes
    episode_indices = np.random.choice(total_episodes, size=num_episodes_to_run, replace=False)
    
    for i, episode_idx in enumerate(episode_indices):
        if not simulation_app.is_running():
            break
            
        print(f"\nRunning episode {episode_idx} ({i + 1}/{num_episodes_to_run})")
        
        try:
            result = run_vae_inference(
                env=env,
                dataset=val_dataset,
                episode_idx=episode_idx,
                vae_model=vae_model,
                cfg=cfg,
                device=device,
                save_trajectory=args_cli.save_trajectories
            )
            
            # Save video if requested
            if args_cli.save_trajectories and result is not None:
                pred_data, gt_data = result
                save_vae_video(pred_data, gt_data, episode_idx, args_cli.output_dir)

        except Exception as e:
            omni.log.error(f"Error in episode {episode_idx + 1}: {e}")
            import traceback
            traceback.print_exc()
            break
    
    env.close()
    print("VAE inference finished")

if __name__ == "__main__":
    main()
    simulation_app.close()