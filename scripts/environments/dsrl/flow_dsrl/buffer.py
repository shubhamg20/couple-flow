"""
Replay buffer for online DSRL training in IsaacLab.
Stores transitions with robot state, actions, and human context.
"""

import torch
import numpy as np
from typing import Dict, Optional


class IsaacLabReplayBuffer:
    """
    Replay buffer for IsaacLab online RL.
    
    Stores:
    - Robot observations (eef pose + object positions)
    - Actions (7D: delta pose + gripper)
    - Human context (flattened human action trajectory for conditioning)
    - Rewards and terminals
    """
    
    def __init__(
        self,
        capacity: int,
        n_envs: int,
        obs_dim: int,  # Robot state dimension (16 = 7 eef + 9 object pos)
        action_dim: int,  # Action dimension (7)
        human_context_dim: int,  # Human context dimension (280 * 4 = 1120)
        action_horizon: int = 8,  # Action chunk length
        discount: float = 0.99,
    ):
        self.capacity = capacity
        self.n_envs = n_envs
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.human_context_dim = human_context_dim
        self.action_horizon = action_horizon
        self.discount = discount
        self.device = torch.device('cpu')
        
        # Observations
        self.observations = torch.zeros(
            (capacity, n_envs, obs_dim),
            dtype=torch.float32,
            device=self.device,
        )
        self.next_observations = torch.zeros_like(self.observations)
        
        # Actions (chunked)
        self.actions = torch.zeros(
            (capacity, n_envs, action_dim),
            dtype=torch.float32,
            device=self.device,
        )
        
        # Human context for DSRL conditioning
        self.human_contexts = torch.zeros(
            (capacity, n_envs, human_context_dim),
            dtype=torch.float32,
            device=self.device,
        )
        
        # Rewards and terminals
        self.rewards = torch.zeros(
            (capacity, n_envs),
            dtype=torch.float32,
            device=self.device,
        )
        self.terminals = torch.zeros(
            (capacity, n_envs),
            dtype=torch.float32,
            device=self.device,
        )
        
        self.idx = 0
        self.full = False
        self.size = 0
    
    def add(
        self,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        action: torch.Tensor,
        human_context: torch.Tensor,
        reward: torch.Tensor,
        terminal: torch.Tensor,
    ) -> None:
        """Add a batch of transitions to the buffer."""
        obs = torch.as_tensor(obs, device=self.device)
        next_obs = torch.as_tensor(next_obs, device=self.device)
        action = torch.as_tensor(action, device=self.device)
        human_context = torch.as_tensor(human_context, device=self.device)
        reward = torch.as_tensor(reward, device=self.device)
        terminal = torch.as_tensor(terminal, device=self.device)
        
        self.observations[self.idx].copy_(obs)
        self.next_observations[self.idx].copy_(next_obs)
        self.actions[self.idx].copy_(action)
        self.human_contexts[self.idx].copy_(human_context)
        self.rewards[self.idx].copy_(reward)
        self.terminals[self.idx].copy_(terminal)
        
        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample a batch of transitions."""
        if self.size == 0:
            raise RuntimeError("Cannot sample from empty buffer")
        
        # Sample indices
        max_idx = self.capacity if self.full else self.idx
        indices = np.random.randint(0, max_idx, size=batch_size)
        env_indices = np.random.randint(0, self.n_envs, size=batch_size)
        
        batch = {
            'obs': self.observations[indices, env_indices],
            'next_obs': self.next_observations[indices, env_indices],
            'actions': self.actions[indices, env_indices],
            'human_context': self.human_contexts[indices, env_indices],
            'rewards': self.rewards[indices, env_indices],
            'terminals': self.terminals[indices, env_indices],
        }
        
        return batch
    
    def __len__(self) -> int:
        return self.size * self.n_envs


class ChunkingReplayBuffer(IsaacLabReplayBuffer):
    """
    Replay buffer that stores action chunks for flow policy training.
    """
    
    def __init__(
        self,
        capacity: int,
        n_envs: int,
        obs_dim: int,
        action_dim: int,
        human_context_dim: int,
        action_horizon: int = 8,
        obs_horizon: int = 1,
        discount: float = 0.99,
    ):
        super().__init__(
            capacity=capacity,
            n_envs=n_envs,
            obs_dim=obs_dim,
            action_dim=action_dim,
            human_context_dim=human_context_dim,
            action_horizon=action_horizon,
            discount=discount,
        )
        
        self.obs_horizon = obs_horizon
        
        # Override actions to store chunks
        self.actions = torch.zeros(
            (capacity, n_envs, action_horizon, action_dim),
            dtype=torch.float32,
            device=self.device,
        )
        
        # Track episode boundaries for valid chunk sampling
        self.episode_starts = [[] for _ in range(n_envs)]
    
    def add_chunk(
        self,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        action_chunk: torch.Tensor,
        human_context: torch.Tensor,
        reward: torch.Tensor,
        terminal: torch.Tensor,
    ) -> None:
        """Add a transition with action chunk."""
        obs = torch.as_tensor(obs, device=self.device)
        next_obs = torch.as_tensor(next_obs, device=self.device)
        action_chunk = torch.as_tensor(action_chunk, device=self.device)
        human_context = torch.as_tensor(human_context, device=self.device)
        reward = torch.as_tensor(reward, device=self.device)
        terminal = torch.as_tensor(terminal, device=self.device)
        
        self.observations[self.idx].copy_(obs)
        self.next_observations[self.idx].copy_(next_obs)
        self.actions[self.idx].copy_(action_chunk)
        self.human_contexts[self.idx].copy_(human_context)
        self.rewards[self.idx].copy_(reward)
        self.terminals[self.idx].copy_(terminal)
        
        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample a batch of transitions with action chunks."""
        if self.size == 0:
            raise RuntimeError("Cannot sample from empty buffer")
        
        max_idx = self.capacity if self.full else self.idx
        indices = np.random.randint(0, max_idx, size=batch_size)
        env_indices = np.random.randint(0, self.n_envs, size=batch_size)
        
        batch = {
            'obs': self.observations[indices, env_indices],
            'next_obs': self.next_observations[indices, env_indices],
            'actions': self.actions[indices, env_indices],  # (B, action_horizon, action_dim)
            'human_context': self.human_contexts[indices, env_indices],
            'rewards': self.rewards[indices, env_indices],
            'terminals': self.terminals[indices, env_indices],
        }
        
        return batch
