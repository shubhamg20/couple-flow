import numpy as np
import torch
import sys
sys.path.append("source/p-diff-exp/")
from isaaclab.app import AppLauncher
headless = True
enable_cameras = True
app_launcher = AppLauncher({"headless": headless, "enable_cameras": enable_cameras})
simulation_app = app_launcher.app
import gymnasium as gym
import time
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from omegaconf import OmegaConf
import wandb
from models.nets import make_flow_nets
import traceback
import os
import glob
import random
import pickle
import gc
import pathlib
import cv2
import imageio
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
torch.cuda.set_device(0)
from stable_baselines3 import PPO as SB3_PPO
from stable_baselines3.common.vec_env import VecEnv
from isaaclab.envs import DirectMARLEnv, ManagerBasedRLEnv, DirectRLEnv
from isaaclab_rl.sb3 import Sb3VecEnvWrapper
from stable_baselines3.common.callbacks import BaseCallback
from collections import deque

def load_pkl(path):
    with open(path, 'rb') as f:
        d = pickle.load(f)
    return d


def load_model_checkpoint(model, checkpoint_path, device, flow_policy_path=None):
    """Load model checkpoint. If flow_policy_path is set, add it to sys.path so pickle can find 'flow_policy'."""
    if flow_policy_path is not None:
        flow_policy_path = os.path.abspath(flow_policy_path)
        if flow_policy_path not in sys.path:
            sys.path.insert(0, flow_policy_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    flow_state_dict = {}
    state_dict = checkpoint['state_dict']
    for key, value in state_dict.items():
        if key.startswith('flow_net.'):
            flow_state_dict[key.replace('flow_net.', '')] = value
    model.load_state_dict(flow_state_dict)
    print(f"Loaded model checkpoint from {checkpoint_path}")
    return model


def apply_demo_objects(env, demo_objects, env_ids):
    """Apply demo object positions to scene."""
    positions = {
        "tray": {"pos": [0.41, 0.42, 1.0], "quat": [0.707, 0.707, 0.0, 0.0]}
    }
    env_origin = env.scene.env_origins[env_ids].to(env.device)
    for obj_name in ["apple", "mug", "sushi"]:
        if obj_name in env.scene.keys() and obj_name in demo_objects:
            asset = env.scene[obj_name]

            pos = torch.tensor(demo_objects[obj_name]["pos"], device=env.device).unsqueeze(0)
            pos = pos + env_origin
            quat = torch.tensor(demo_objects[obj_name]["quat"], device=env.device).unsqueeze(0)
            root_pose = torch.cat([pos, quat], dim=-1)
            velocities = torch.zeros((1, 6), device=env.device)
            asset.write_root_pose_to_sim(root_pose, env_ids=env_ids)
            asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)
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


def compute_human_flattened_actions(traj):
    actions = []
    for i, s in enumerate(traj):
        s = traj[i]
        abs_pos = np.array(s['franka_eef']['pos'])
        gripper = s['franka_eef']['gripper']
        abs_action = np.concatenate([abs_pos, [gripper]])
        actions.append(abs_action)
    if len(actions) < 250:
        last_action = actions[-1]
        for _ in range(250 - len(actions)):
            actions.append(last_action)
    actions = np.array(actions).flatten()
    return actions


def quat_to_euler_torch(quat_wxyz):
    """Vectorized quaternion to euler conversion (batched).
    Input: (n_envs, 4) in wxyz format
    Output: (n_envs, 3) roll, pitch, yaw in radians
    """
    w, x, y, z = quat_wxyz[:, 0], quat_wxyz[:, 1], quat_wxyz[:, 2], quat_wxyz[:, 3]
    
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    pitch = torch.where(
        torch.abs(sinp) >= 1,
        torch.sign(sinp) * torch.pi / 2,
        torch.asin(sinp)
    )
    
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)
    
    return torch.stack([roll, pitch, yaw], dim=-1)


def get_observations_batch(env, device):
    """Batched obs (n_envs, flow_obs_dim). Fully vectorized, no loops."""
    env_origins = env.scene.env_origins.to(device)  # (n_envs, 3)
    eef_idx = env.scene["robot"].data.body_names.index("panda_hand")
    eef_pos_w = env.scene["robot"].data.body_pos_w[:, eef_idx, :]  # (n_envs, 3)
    eef_pos_local = eef_pos_w - env_origins
    
    eef_quat_w = env.scene["robot"].data.body_quat_w[:, eef_idx, :]  # (n_envs, 4) wxyz
    eef_rpy = quat_to_euler_torch(eef_quat_w)  # Vectorized conversion
    
    joint_names = env.scene["robot"].data.joint_names
    gripper_indices = [joint_names.index("panda_finger_joint1"), joint_names.index("panda_finger_joint2")]
    gripper_state = env.scene["robot"].data.joint_pos[:, gripper_indices].mean(dim=1, keepdim=True)  # (n_envs, 1)
    
    object_names = ["apple", "mug", "sushi"]
    obj_positions = []
    for obj_name in object_names:
        if obj_name in env.scene.keys():
            obj_pos_w = env.scene[obj_name].data.body_pos_w[:, 0, :]
            obj_positions.append(obj_pos_w - env_origins)
    object_positions = torch.cat(obj_positions, dim=-1)  # (n_envs, 9)
    
    obs = torch.cat([eef_pos_local, eef_rpy, gripper_state, object_positions], dim=-1)
    return obs


def forward_flow(obs_batch, model, device, action_horizon=16, noise=None):
    """Forward flow model to generate actions from noise"""
    num_steps = 10
    dt = 1.0 / num_steps
    obs_tensor = obs_batch if isinstance(obs_batch, torch.Tensor) else torch.stack(obs_batch).to(device)
    n_batch = obs_tensor.shape[0]
    if noise is not None:
        x = noise
    else:
        x = torch.randn(n_batch, action_horizon, 7, device=device)
    with torch.no_grad():
        for fm_step in range(int(0 * num_steps), num_steps):
            t = torch.tensor(fm_step * dt, device=device)
            t_batch = t.repeat(n_batch)
            vt = model(x, t_batch, global_cond=obs_tensor)
            x = x + dt * vt
    return x[:, :]


def compute_object_state_reward_batch(initial_sim, final_sim, initial_demo, final_demo, truncated_t, terminated_t):
    """Batched reward computation on GPU.
    Args: all (batch, 9) tensors
    Returns: rewards (batch,), successes (batch,)
    """
    batch_size = initial_sim.shape[0]
    # sim_obj_pos0 = initial_sim.view(batch_size, 3, 3)
    # sim_obj_pos1 = final_sim.view(batch_size, 3, 3)
    # demo_obj_pos0 = initial_demo.view(batch_size, 3, 3)
    # demo_obj_pos1 = final_demo.view(batch_size, 3, 3)
    
    # sim_displacements = torch.norm(sim_obj_pos1 - sim_obj_pos0, dim=2)  # (batch, 3)
    # demo_displacements = torch.norm(demo_obj_pos1 - demo_obj_pos0, dim=2)  # (batch, 3)
    
    # sim_max_idx = torch.argmax(sim_displacements, dim=1)
    # demo_max_idx = torch.argmax(demo_displacements, dim=1)
    
    # is_success = (sim_max_idx == demo_max_idx)
    #####################################################
    # is_success = is_success & ~truncated_t
    is_success = terminated_t
    #####################################################
    rewards = torch.where(is_success, 
                          torch.zeros(batch_size, dtype=torch.float32, device=initial_sim.device), 
                          torch.full((batch_size,), -1.0, dtype=torch.float32, device=initial_sim.device))
    
    return rewards, is_success

class IsaacLabSB3Wrapper(ManagerBasedRLEnv):
    """
    Wrapper that makes IsaacLab's vectorized environment compatible with SB3.
    Similar to IsaacLab's PettingZoo wrapper but for SB3.
    """
    def __init__(self, isaaclab_env, flow_net, cfg_rl, human_datas_cache, device, mode="train"):
        """
        Args:
            isaaclab_env: The unwrapped IsaacLab environment (env.unwrapped)
            flow_net: Your flow matching model
            cfg: Config
            human_datas_cache: Preloaded demonstrations
            device: torch device
        """
        self.env = isaaclab_env
        self.sim = self.env.unwrapped.sim
        self.scene = self.env.unwrapped.scene
        self.cfg = self.env.unwrapped.cfg

        self.flow_net = flow_net
        self.cfg_rl = cfg_rl
        self.human_datas_cache = human_datas_cache
        self._device = device
        self.n_envs = cfg_rl.n_envs
        # Tracking variables
        self.step_counts = torch.zeros(self.n_envs, dtype=torch.int32, device=self._device)
        self.initial_gt_obj_positions = torch.zeros((self.n_envs, 9), dtype=torch.float32, device=self._device)
        self.episode_rewards = torch.zeros(self.n_envs, dtype=torch.float32, device=self._device)
        self.episode_lengths = torch.zeros(self.n_envs, dtype=torch.int32, device=self._device)
        
        # Human demonstration tracking
        self.human_datas = []
        self.initial_demo_objects_list = []
        self.final_demo_objects_list = []
        
        self.human_actions_buf_tensor = torch.zeros(
            (self.n_envs, cfg_rl.human_action_traj_len),
            dtype=torch.float32,
            device=self._device
        )
        
        # Initialize demos for all envs
        for i in range(self.n_envs):
            self._sample_new_demo_for_env(i)
            env_id_tensor = torch.tensor([i], device=device, dtype=torch.int32)
            apply_demo_objects(self.env.unwrapped, self.initial_demo_objects_list[i], env_ids=env_id_tensor)
        
        # Define spaces
        flow_obs_dim = cfg_rl.flow_obs_dim
        human_action_dim = cfg_rl.human_action_traj_len
        total_obs_dim = flow_obs_dim #+ human_action_dim
        self.single_observation_space = {"policy": gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(total_obs_dim,), dtype=np.float32
        )}
        
        # Action space: noise for flow model (flattened)
        self.single_action_space = {"policy":gym.spaces.Box(
            low=-3.0, high=3.0,
            shape=(cfg_rl.pred_horizon * cfg_rl.action_dim,),
            dtype=np.float32
        )}
    
    @property
    def unwrapped(self):    
        return self
              
    def _sample_new_demo_for_env(self, env_idx):
        """Sample new demonstration for an environment"""
        idx = np.random.randint(len(self.human_datas_cache))
        hd = self.human_datas_cache[idx]
        
        if env_idx >= len(self.human_datas):
            self.human_datas.append(hd)
            self.initial_demo_objects_list.append(hd['initial_objects'])
            self.final_demo_objects_list.append(hd['final_objects'])
        else:
            self.human_datas[env_idx] = hd
            self.initial_demo_objects_list[env_idx] = hd['initial_objects']
            self.final_demo_objects_list[env_idx] = hd['final_objects']
        
        self.human_actions_buf_tensor[env_idx] = torch.from_numpy(hd['actions']).to(self._device)
    
    def reset(self):
        """Reset all environments - returns numpy array for SB3"""
        _ = self.env.reset()
        
        # Reset tracking
        self.step_counts.zero_()
        self.episode_rewards.zero_()
        self.episode_lengths.zero_()
        
        # Get observations
        obs_batch = get_observations_batch(self.env.unwrapped, self.device)
        self.initial_gt_obj_positions = obs_batch[:, 7:].clone()
        
        # Concatenate with human actions
        # combined_obs = torch.cat([obs_batch, self.human_actions_buf_tensor], dim=-1)        
        combined_obs = torch.cat([obs_batch], dim=-1)

        
        return {"policy" : combined_obs}
    
    def step(self, actions):
        """Main step function - does all the work directly"""
        # Convert numpy actions to torch
        if isinstance(actions, np.ndarray):
            actions_tensor = torch.from_numpy(actions).to(self.device).float()
        else:
            actions_tensor = actions
        extras = {}
        # Get current observations
        obs_batch = get_observations_batch(self.env.unwrapped, self.device)
        
        # Reshape noise: (n_envs, pred_horizon * action_dim) -> (n_envs, pred_horizon, action_dim)
        noise = actions_tensor.reshape(self.num_envs, self.cfg_rl.pred_horizon, self.cfg_rl.action_dim)
        
        # Generate actions via flow model
        with torch.no_grad():
            action_batch = forward_flow(obs_batch, self.flow_net, self.device, noise=noise)
        
        # Execute action sequence in IsaacLab
        for i in range(self.cfg_rl.replay_horizon):
            action_step = action_batch[:, i, :]
            _, _, terminated, truncated, _ = self.env.step(action_step)
        
        # Process termination
        terminated_t = terminated if torch.is_tensor(terminated) else torch.tensor(
            terminated, device=self.device, dtype=torch.bool
        )
        
        self.step_counts += self.cfg_rl.replay_horizon
        self.episode_lengths += self.cfg_rl.replay_horizon
        truncated_t = self.step_counts >= self.cfg_rl.max_episode_length
        
        next_obs_batch = get_observations_batch(self.env.unwrapped, self.device)
        
        # Set initial positions at episode start
        mask = (self.step_counts == self.cfg_rl.replay_horizon)
        if mask.any():
            self.initial_gt_obj_positions[mask] = obs_batch[mask, 7:]
        
        done_t = terminated_t | truncated_t
        done_indices = torch.where(done_t)[0]
        
        # Compute rewards
        step_rewards = -torch.ones(self.num_envs, dtype=torch.float32, device=self.device)
        infos = [{} for _ in range(self.num_envs)]
            
        self.episode_rewards += step_rewards

        if done_indices.numel() > 0:
            # Batched reward computation
            n_done = done_indices.numel()
            initial_sim_batch = self.initial_gt_obj_positions[done_indices]
            final_sim_batch = next_obs_batch[done_indices, 7:]
            
            initial_demo_batch = torch.zeros((n_done, 9), dtype=torch.float32, device=self.device)
            final_demo_batch = torch.zeros((n_done, 9), dtype=torch.float32, device=self.device)
            
            done_indices_cpu = done_indices.cpu().numpy()
            for idx_pos, env_idx in enumerate(done_indices_cpu):
                initial_demo_state = np.array([
                    self.human_datas[env_idx]['initial_objects'][obj]['pos']
                    for obj in sorted(self.human_datas[env_idx]['initial_objects'].keys())
                ]).flatten()
                final_demo_state = np.array(self.human_datas[env_idx]['final_objects']).flatten()
                
                initial_demo_batch[idx_pos] = torch.from_numpy(initial_demo_state).to(self.device)
                final_demo_batch[idx_pos] = torch.from_numpy(final_demo_state).to(self.device)
            
            rewards_batch, success_batch = compute_object_state_reward_batch(
                initial_sim_batch, final_sim_batch, initial_demo_batch, final_demo_batch, truncated_t[done_indices], terminated_t[done_indices]
            )
            step_rewards[done_indices] = rewards_batch
            for k in range(len(done_indices)):
                if done_indices[k] == 1 and rewards_batch[k] > 0:
                    print(rewards_batch[k], self.step_counts[k], terminated_t[k], truncated_t[k])
            # Add episode info
            success_tensor = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            for idx_pos, env_idx in enumerate(done_indices_cpu):
                is_success = success_batch[idx_pos].item()
                reward_val = rewards_batch[idx_pos].item()
                success_tensor[env_idx] = is_success
                
                extras['is_success'] = success_tensor.cpu().numpy() 
                
                # if is_success:
                #     print(f"👏 Env {env_idx} success! Reward: {reward_val}")
                
                # Sample new demo
                self._sample_new_demo_for_env(env_idx)
                
                env_id_tensor = torch.tensor([env_idx], device=self.device, dtype=torch.int32)
                apply_demo_objects(
                    self.env.unwrapped,
                    self.human_datas[env_idx]['initial_objects'],
                    env_ids=env_id_tensor
                )
            
            # Reset done environments
            self.env.unwrapped.reset(env_ids=done_indices)
            self.step_counts[done_indices] = 0
            self.episode_rewards[done_indices] = 0
            self.episode_lengths[done_indices] = 0
        
        
        # Get next observations
        next_obs_batch = get_observations_batch(self.env.unwrapped, self.device)
        # combined_obs = torch.cat([next_obs_batch, self.human_actions_buf_tensor], dim=-1)        
        combined_obs = torch.cat([next_obs_batch], dim=-1)

        
        # Return numpy arrays for SB3
        return (
            {"policy":combined_obs},
            step_rewards,
            terminated_t,
            truncated_t,
            extras
        )
    
    def close(self):
        """Close the environment"""
        self.env.close()
    
    def env_is_wrapped(self, wrapper_class, indices=None):
        """SB3 VecEnv method"""
        return [False] * self.num_envs
    
    def get_attr(self, attr_name, indices=None):
        """SB3 VecEnv method"""
        return [getattr(self.env, attr_name)] * self.num_envs
    
    def set_attr(self, attr_name, value, indices=None):
        """SB3 VecEnv method"""
        setattr(self.env, attr_name, value)
    
    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        """SB3 VecEnv method"""
        return [None] * self.num_envs
    
    def seed(self, seed=None):
        """Set random seed"""
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        return [seed] * self.num_envs


class PPOCallback(BaseCallback):
    """Custom callback for logging"""
    def __init__(self, cfg, verbose=0):
        super().__init__(verbose)
        self.cfg = cfg
        self.recent_successes = deque(maxlen=100)        
        self.recent_rewards = deque(maxlen=100)

        self.save_freq = 500
    
    def _on_step(self) -> bool:
        # Collect success info from episodes
        for info in self.locals.get('infos', []):
            if 'episode' in info and info['episode'] is not None:
                if 'r' in info['episode']:
                    self.recent_rewards.append(float(info['episode']['r']))
            if 'is_success' in info:
                self.recent_successes.append(float(info['is_success']))
    
        
        return True
    
    def _on_rollout_end(self) -> None:
        """Log metrics after rollout"""
        mean_rewards = np.mean(self.recent_rewards) if len(self.recent_rewards) > 0 else 0.0
        if len(self.recent_successes) > 0:
            success_rate = sum(self.recent_successes) / len(self.recent_successes)
        else:
            success_rate = 0.0
        
        # Prepare wandb logs
        wandb_logs = {
            'train/success_rate': success_rate,
            'train/mean_rewards': mean_rewards,
            'train/timesteps': self.num_timesteps,
        }
        
        # Extract PPO training metrics from the model's logger
        if hasattr(self.model, 'logger') and hasattr(self.model.logger, 'name_to_value'):
            logger_dict = self.model.logger.name_to_value
            
            # Map of metric names to log
            metric_names = [
                'train/approx_kl',
                'train/clip_fraction',
                'train/clip_range',
                'train/entropy_loss',
                'train/explained_variance',
                'train/learning_rate',
                'train/loss',
                'train/n_updates',
                'train/policy_gradient_loss',
                'train/std',
                'train/value_loss',
            ]
            
            # Extract available metrics from logger
            for metric_name in metric_names:
                # Try both with and without 'train/' prefix
                short_name = metric_name.replace('train/', '')
                if metric_name in logger_dict:
                    wandb_logs[metric_name] = logger_dict[metric_name]
                elif short_name in logger_dict:
                    wandb_logs[metric_name] = logger_dict[short_name]
        
        if self.cfg.use_wandb:
            wandb.log(wandb_logs)
        
        print(f"Success rate: {success_rate:.2%} (last {len(self.recent_successes)} episodes)")
        
        # Save checkpoint
        if self.num_timesteps % (self.save_freq * self.training_env.num_envs) < self.training_env.num_envs:
            ckpt_dir = pathlib.Path(self.cfg.save_dir)
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = ckpt_dir / f"ppo_model_step_{self.num_timesteps}.zip"
            self.model.save(ckpt_path)
            print(f'Saved checkpoint at timestep {self.num_timesteps}')

def main(cfg):
    try:
        if cfg.use_wandb:
            wandb.init(
                project="isaac-franka-ppo",
                name="sb3-ppo-run",
                config=OmegaConf.to_container(cfg, resolve=True)
            )
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create IsaacLab environment (the normal way)
        # base_env = IsaacLabGymEnv(cfg=cfg, num_envs=cfg.n_envs)
        
        # # Get the unwrapped IsaacLab environment
        # isaaclab_env = base_env.env
        
        # Load flow model
        nets = make_flow_nets(cfg)
        flow_net = nets['flow_net'].to(device)
        flow_policy_path = getattr(cfg, 'flow_policy_path', None)
        flow_net = load_model_checkpoint(
            flow_net, cfg.flow_model_checkpoint, device, flow_policy_path=flow_policy_path
        )
        flow_net.eval()
        
        # Pre-load human demonstrations
        print("Pre-loading human demonstrations...")
        human_data_path = cfg.human_data_path
        tasks = ["sushi", "mug", "apple"]
        human_episode_files = []
        for task in tasks:
            pattern = os.path.join(human_data_path, f"{task}_org", "episode*.pkl")
            files = sorted(glob.glob(pattern))[:40]
            human_episode_files.extend(files)
        random.shuffle(human_episode_files)
        
        human_datas_cache = []
        for h_file in tqdm(human_episode_files, desc="Loading demos"):
            hd = load_pkl(h_file)
            human_datas_cache.append({
                'initial_objects': hd['initial_objects'],
                'final_objects': [hd['trajectory'][-1]['objects'][obj]['pos']
                                  for obj in sorted(hd['trajectory'][-1]['objects'].keys())],
                'actions': compute_human_flattened_actions(hd['trajectory'])[:cfg.human_action_traj_len]
            })
        print(f"Loaded {len(human_datas_cache)} demonstrations")
        
        # Wrap IsaacLab environment for SB3 (similar to their MARL wrappers)
        from isaaclab_tasks.utils import parse_env_cfg
        env_cfg = parse_env_cfg('Isaac-PickPlace-Franka-custom', device="cuda", num_envs=cfg.n_envs)
        import gymnasium as gym
        env = gym.make('Isaac-PickPlace-Franka-custom', cfg=env_cfg, render_mode="rgb_array")
        video_kwargs = {
            "video_folder": os.path.join(cfg.video_output_path, "videos", "train"),
            "step_trigger": lambda step: step % cfg.video_interval == 0,
            "video_length": cfg.video_length,
            "disable_logger": True,
        }
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
        isaaclab_env = IsaacLabSB3Wrapper(env, flow_net, cfg, human_datas_cache, device, mode="train")
        sb3_vecenv = Sb3VecEnvWrapper(isaaclab_env, fast_variant=False)
        sb3_vecenv.action_space = isaaclab_env.single_action_space["policy"]
        # Initialize SB3 PPO
        # Convert OmegaConf to regular dict to allow Python object assignment
        ppo_cfg_dict = OmegaConf.to_container(cfg.ppo, resolve=True)
        
        # Extract and convert policy_kwargs activation_fn string to callable
        policy_kwargs = ppo_cfg_dict.get('policy_kwargs', {})
        if policy_kwargs and 'activation_fn' in policy_kwargs:
            activation_str = policy_kwargs['activation_fn']
            # Convert string like 'nn.ELU' to actual callable
            if activation_str == 'nn.ELU':
                policy_kwargs['activation_fn'] = nn.ELU
            elif activation_str == 'nn.ReLU':
                policy_kwargs['activation_fn'] = nn.ReLU
            elif activation_str == 'nn.Tanh':
                policy_kwargs['activation_fn'] = nn.Tanh
        
        # Build PPO args from config, excluding policy and policy_kwargs
        ppo_args = {k: v for k, v in ppo_cfg_dict.items() if k not in ['policy', 'policy_kwargs']}
        
        ppo_agent = SB3_PPO(
            "MlpPolicy",
            env=sb3_vecenv,
            verbose=1,
            device=device,
            policy_kwargs=policy_kwargs if policy_kwargs else None,
            **ppo_args
        )
        
        # Create callback for logging
        callback = PPOCallback(cfg)
        
        # Train
        ppo_agent.learn(
            total_timesteps=cfg.total_training_steps,
            callback=[callback],
            progress_bar=True
        )
        
        # Save final model
        ckpt_dir = pathlib.Path(cfg.save_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ppo_agent.save(ckpt_dir / "final_model")
        print('Saved final model')
        
        sb3_vecenv.close()
        isaaclab_env.close()
        env.close()
        
    except Exception as e:
        print(f"Exception occurred: {e}", file=sys.stderr)
        traceback.print_exc()
    finally:
        try:
            # Close environments
            if 'sb3_vecenv' in locals():
                sb3_vecenv.close()
            if 'isaaclab_env' in locals():
                isaaclab_env.close()

            # Explicitly delete large objects
            if 'human_datas_cache' in locals():
                del human_datas_cache
            if 'ppo_agent' in locals():
                del ppo_agent

            # Force garbage collection
            gc.collect()

            # Release GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as cleanup_error:
            print(f"Error during environment cleanup: {cleanup_error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    import hydra
    @hydra.main(version_base=None, config_path='cfg', config_name='ppo_isaac.yaml')
    def hydra_main(cfg):
        main(cfg)
    # Hydra automatically handles calling the function with config
    hydra_main()