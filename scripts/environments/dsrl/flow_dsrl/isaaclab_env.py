"""
IsaacLab environment wrapper for online DSRL training.
Wraps the PickPlace-Franka environment for RL training.
"""

import sys
sys.path.append("/workspace/isaaclab/source/droid/droid/controllers/")

import argparse
import pickle
import random
from pathlib import Path
import torch
import numpy as np
from typing import Dict, Tuple, Optional, Any, List
from scipy.spatial.transform import Rotation as R


class IsaacLabDSRLEnv:
    """
    Wrapper for IsaacLab PickPlace-Franka environment for online DSRL training.
    
    This wrapper:
    - Manages the IsaacLab environment lifecycle
    - Extracts observations in the format expected by the DSRL agent
    - Handles action chunking and execution
    - Computes rewards for the pick-place task
    """
    
    def __init__(
        self,
        task: str = "Isaac-PickPlace-Franka-custom",
        num_envs: int = 1,
        device: str = "cuda:0",
        headless: bool = True,
        target_object: str = "sushi",  # Which object to pick
        max_episodes: int = 49,  # Maximum number of human demo episodes (0 to max_episodes)
    ):
        self.task = task
        self.num_envs = num_envs
        self.device_str = device
        self.headless = headless
        self.target_object = target_object
        self.max_episodes = max_episodes
        
        # Dimensions
        self.obs_dim = 16  # 7 (eef) + 9 (3 objects * 3 pos)
        self.action_dim = 7  # 6 (delta pose) + 1 (gripper)
        
        # Will be set after environment creation
        self.env = None
        self.sim_app = None
        self.device = None  # Will be set in initialize()
        
        # Store initial objects for each episode
        self.human_initial_objects = None
        
        # Track chosen episodes for each environment
        self.chosen_episodes = None  # Will be initialized after env creation
        
    def initialize(self):
        """Initialize the IsaacLab environment."""
        from isaaclab.app import AppLauncher
        
        # Create parser and launcher
        parser = argparse.ArgumentParser()
        AppLauncher.add_app_launcher_args(parser)
        args = parser.parse_args([
            "--headless" if self.headless else "",
            f"--device={self.device_str}",
        ])
        
        app_launcher = AppLauncher(vars(args))
        self.sim_app = app_launcher.app
        
        # Import after launcher
        import gymnasium as gym
        import isaaclab_tasks
        from isaaclab_tasks.utils import parse_env_cfg
        
        # Setup environment config
        env_cfg = parse_env_cfg(self.task, device=self.device_str, num_envs=self.num_envs)
        env_cfg.env_name = self.task
        env_cfg.sim.gravity = [0.0, 0.0, -9.81]
        env_cfg.terminations.time_out = None
        
        # Create environment
        self.env = gym.make(self.task, cfg=env_cfg).unwrapped
        self.env.reset()
        
        # Initialize robot position
        current_joint_pos = self.env.scene["robot"].data.joint_pos.clone()
        env_ids = torch.arange(self.num_envs, device=self.env.device)
        root_state = self.env.scene["robot"].data.default_root_state.clone()
        self.env.scene["robot"].write_root_state_to_sim(root_state, env_ids=env_ids)
        self.env.scene["robot"].set_joint_position_target(current_joint_pos)
        self.env.sim.step(render=False)
        
        self.device = self.env.device
        
        # Initialize chosen_episodes tracking
        self.chosen_episodes = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
    
    def set_human_initial_objects(self, initial_objects_list: List[Dict]):
        """Set the initial objects list loaded from PKL files."""
        self.human_initial_objects = initial_objects_list
        print(f"[ENV] Set {len(initial_objects_list)} initial object configurations")
    
    def _apply_demo_objects(self, demo_objects: Optional[Dict], env_ids: torch.Tensor):
        """Apply the loaded object positions to the Isaac Sim scene."""
        if demo_objects is None:
            return

        for obj_name in ["apple", "mug", "sushi"]:
            if obj_name in self.env.scene.keys() and obj_name in demo_objects:
                asset = self.env.scene[obj_name]
                
                # Convert list to tensor and move to device
                pos = torch.tensor(demo_objects[obj_name]["pos"], device=self.device)
                quat = torch.tensor(demo_objects[obj_name]["quat"], device=self.device)
                
                # Ensure correct shape: (num_envs, 3) for pos, (num_envs, 4) for quat
                if pos.ndim == 1:
                    pos = pos.unsqueeze(0)
                if quat.ndim == 1:
                    quat = quat.unsqueeze(0)
                
                # Broadcast to all env_ids if needed
                num_envs_to_set = len(env_ids)
                if pos.shape[0] == 1 and num_envs_to_set > 1:
                    pos = pos.repeat(num_envs_to_set, 1)
                if quat.shape[0] == 1 and num_envs_to_set > 1:
                    quat = quat.repeat(num_envs_to_set, 1)
                
                # Concatenate pos and quat for root_pose: (num_envs, 7)
                root_pose = torch.cat([pos, quat], dim=-1)
                velocities = torch.zeros((num_envs_to_set, 6), device=self.device)
                
                # Write to simulation
                asset.write_root_pose_to_sim(root_pose, env_ids=env_ids)
                asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)

        # Handle Tray separately if needed (XFormPrim)
        # For now, tray is typically static, so we skip it
        
    def reset(self, env_ids: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Reset environment and initialize with a random human demo start state."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
            self.env.reset()
        else:
            # Partial reset - reset only specified environments
            # Note: IsaacLab's reset might reset all envs, but we'll apply objects only to specified ones
            self.env.reset()
            # Convert to tensor if needed
            if not isinstance(env_ids, torch.Tensor):
                env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.int32)

        # Apply specific human demo starts to each resetting environment
        for i in range(len(env_ids)):
            env_id = env_ids[i].item() if isinstance(env_ids, torch.Tensor) else env_ids[i]
            
            # 1. Randomly pick a human episode
            ep_num = random.randint(0, self.max_episodes)
            self.chosen_episodes[env_id] = ep_num
            
            # 2. Load the object positions for that episode (if available)
            if self.human_initial_objects is not None and ep_num < len(self.human_initial_objects):
                demo_objects = self.human_initial_objects[ep_num]
                
                # 3. Teleport Isaac Sim objects to match the human demo
                env_id_tensor = torch.tensor([env_id], device=self.device, dtype=torch.int32)
                self._apply_demo_objects(demo_objects, env_ids=env_id_tensor)

        # Get the new observation after teleporting objects (for all environments)
        obs_dict = self._get_obs()
        
        # Return both the observation AND the chosen episodes so the RL loop knows which context to fetch
        # This includes episodes for all environments (updated for reset ones, unchanged for others)
        obs_dict['chosen_episodes'] = self.chosen_episodes.clone()
        
        return obs_dict
    
    def step(
        self, 
        actions: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        Take a step in the environment.
        
        Args:
            actions: (num_envs, action_dim) tensor of actions
            
        Returns:
            obs: Dictionary of observations
            rewards: (num_envs,) tensor of rewards
            terminated: (num_envs,) tensor of termination flags
            truncated: (num_envs,) tensor of truncation flags
            info: Dictionary of additional info
        """
        # Execute action
        self.env.step(actions[:, :7])
        
        # Get observations
        obs = self._get_obs()
        
        # Compute rewards
        rewards = self._compute_reward()
        
        # Check termination
        terminated = self._check_termination()
        truncated = torch.zeros_like(terminated)
        
        info = {
            'success': self._check_success(),
        }
        
        return obs, rewards, terminated, truncated, info
    
    def step_chunk(
        self,
        action_chunk: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        Execute an action chunk (sequence of actions).
        
        Args:
            action_chunk: (num_envs, action_horizon, action_dim) tensor
            
        Returns:
            Same as step(), but accumulated over the chunk
        """
        chunk_rewards = []
        chunk_terminated = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        
        for t in range(action_chunk.shape[1]):
            actions = action_chunk[:, t]
            
            # Execute action
            self.env.step(actions[:, :7])
            
            # Accumulate rewards
            rewards = self._compute_reward()
            chunk_rewards.append(rewards)
            
            # Check termination
            terminated = self._check_termination()
            chunk_terminated = chunk_terminated | terminated
            
            # Break if all environments terminated
            if chunk_terminated.all():
                break
        
        # Get final observation
        obs = self._get_obs()
        
        # Aggregate rewards (sum over chunk)
        total_rewards = torch.stack(chunk_rewards).sum(dim=0)
        
        info = {
            'success': self._check_success(),
        }
        
        return obs, total_rewards, chunk_terminated, torch.zeros_like(chunk_terminated), info
    
    def _get_obs(self) -> Dict[str, torch.Tensor]:
        """Extract observations from the environment."""
        # End-effector pose
        eef_idx = self.env.scene["robot"].data.body_names.index("panda_hand")
        eef_pos = self.env.scene["robot"].data.body_pos_w[:, eef_idx]
        eef_quat = self.env.scene["robot"].data.body_quat_w[:, eef_idx]
        
        # Convert quaternion to euler
        eef_rpy = self._quat_to_euler(eef_quat)
        
        # Gripper state
        joint_names = self.env.scene["robot"].data.joint_names
        joint_positions = self.env.scene["robot"].data.joint_pos
        gripper_indices = [
            joint_names.index("panda_finger_joint1"),
            joint_names.index("panda_finger_joint2"),
        ]
        gripper_state = joint_positions[:, gripper_indices].mean(dim=1, keepdim=True)
        
        # Object positions
        object_names = ["sushi", "apple", "mug"]
        object_positions = []
        for obj_name in object_names:
            if obj_name in self.env.scene.keys():
                obj_pos = self.env.scene[obj_name].data.body_pos_w[:, 0]
                object_positions.append(obj_pos)
            else:
                object_positions.append(torch.zeros(self.num_envs, 3, device=self.device))
        
        # Concatenate into state
        state = torch.cat([
            eef_pos,           # 3
            eef_rpy,           # 3
            gripper_state,     # 1
            *object_positions  # 3 * 3 = 9
        ], dim=-1)
        
        return {
            'state': state,  # (num_envs, 16)
            'eef_pos': eef_pos,
            'eef_rpy': eef_rpy,
            'gripper': gripper_state,
            'object_positions': torch.stack(object_positions, dim=1),  # (num_envs, 3, 3)
        }
    
    def _quat_to_euler(self, quat: torch.Tensor) -> torch.Tensor:
        """Convert quaternion (w, x, y, z) to euler angles."""
        # Convert to scipy format (x, y, z, w)
        quat_np = quat[:, [1, 2, 3, 0]].cpu().numpy()
        euler = np.array([R.from_quat(q).as_euler('xyz') for q in quat_np])
        return torch.tensor(euler, device=self.device, dtype=torch.float32)
    
    def _compute_reward(self) -> torch.Tensor:
        """Compute reward for the pick-place task."""
        # Get current object and target positions
        target_obj = self.env.scene[self.target_object]
        obj_pos = target_obj.data.body_pos_w[:, 0]
        
        # Get tray position (target)
        tray_pos = self.env.scene["tray"].data.body_pos_w[:, 0] if "tray" in self.env.scene.keys() else torch.tensor([[0.41, 0.42, 1.0]], device=self.device)
        
        # Get end-effector position
        eef_idx = self.env.scene["robot"].data.body_names.index("panda_hand")
        eef_pos = self.env.scene["robot"].data.body_pos_w[:, eef_idx]
        
        # Distance-based reward components
        dist_to_obj = torch.norm(eef_pos - obj_pos, dim=-1)
        dist_obj_to_tray = torch.norm(obj_pos - tray_pos, dim=-1)
        
        # Reward shaping
        reward_reaching = -dist_to_obj  # Encourage reaching object
        reward_placing = -dist_obj_to_tray  # Encourage placing on tray
        
        # Success bonus
        success = self._check_success()
        reward_success = success.float() * 10.0
        
        # Combine rewards
        reward = reward_reaching * 0.1 + reward_placing * 0.1 + reward_success
        
        return reward
    
    def _check_termination(self) -> torch.Tensor:
        """Check if episode should terminate."""
        # Check if any object dropped below table
        terminated = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        
        for obj_name in ["sushi", "apple", "mug"]:
            if obj_name in self.env.scene.keys():
                obj_pos = self.env.scene[obj_name].data.body_pos_w[:, 0]
                dropped = obj_pos[:, 2] < 0.5
                terminated = terminated | dropped
        
        return terminated
    
    def _check_success(self) -> torch.Tensor:
        """Check if task is successful (object on tray)."""
        target_obj = self.env.scene[self.target_object]
        obj_pos = target_obj.data.body_pos_w[:, 0]
        
        # Tray position and bounds
        tray_center = torch.tensor([0.41, 0.42, 1.05], device=self.device)
        tray_radius = 0.1
        
        dist_to_tray = torch.norm(obj_pos[:, :2] - tray_center[:2], dim=-1)
        on_tray = (dist_to_tray < tray_radius) & (obj_pos[:, 2] > 1.0) & (obj_pos[:, 2] < 1.15)
        
        return on_tray
    
    def close(self):
        """Close the environment."""
        if self.env is not None:
            self.env.close()
        if self.sim_app is not None:
            self.sim_app.close()
    
    @property
    def observation_space(self) -> Dict:
        return {
            'state': (self.obs_dim,),
        }
    
    @property
    def action_space(self) -> Dict:
        return {
            'shape': (self.action_dim,),
            'low': -1.0,
            'high': 1.0,
        }


def make_isaaclab_env(
    task: str = "Isaac-PickPlace-Franka-custom",
    num_envs: int = 1,
    device: str = "cuda:0",
    headless: bool = True,
    target_object: str = "sushi",
    max_episodes: int = 49,
) -> IsaacLabDSRLEnv:
    """Factory function to create IsaacLab environment."""
    env = IsaacLabDSRLEnv(
        task=task,
        num_envs=num_envs,
        device=device,
        headless=headless,
        target_object=target_object,
        max_episodes=max_episodes,
    )
    env.initialize()
    return env
