"""
Utility functions for online DSRL training.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import pickle


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_device(data: Any, device: torch.device) -> Any:
    """Recursively move data to device."""
    if torch.is_tensor(data):
        return data.to(device, non_blocking=True)
    elif isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)(to_device(v, device) for v in data)
    return data


def to_numpy(x: Any) -> np.ndarray:
    """Convert tensor to numpy array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, np.ndarray):
        return x
    else:
        return np.array(x)


def to_tensor(
    data: Any,
    device: torch.device = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Convert data to tensor."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if isinstance(data, torch.Tensor):
        return data.to(device=device, dtype=dtype)
    elif isinstance(data, np.ndarray):
        return torch.tensor(data, device=device, dtype=dtype)
    else:
        return torch.tensor(data, device=device, dtype=dtype)


def save_checkpoint(
    path: str,
    agent: torch.nn.Module,
    optimizers: Dict[str, torch.optim.Optimizer],
    step: int,
    stats: Optional[Dict] = None,
):
    """Save training checkpoint."""
    checkpoint = {
        'step': step,
        'agent_state_dict': agent.state_dict(),
        'optimizers': {k: v.state_dict() for k, v in optimizers.items()},
    }
    if stats is not None:
        checkpoint['stats'] = stats
    
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    agent: torch.nn.Module,
    optimizers: Optional[Dict[str, torch.optim.Optimizer]] = None,
    device: str = 'cuda',
) -> int:
    """Load training checkpoint and return step."""
    checkpoint = torch.load(path, map_location=device)
    
    agent.load_state_dict(checkpoint['agent_state_dict'])
    
    if optimizers is not None and 'optimizers' in checkpoint:
        for k, v in checkpoint['optimizers'].items():
            if k in optimizers:
                optimizers[k].load_state_dict(v)
    
    return checkpoint.get('step', 0)


def load_human_context_from_episodes(
    episode_dir: str,
    window_len: int = 280,
    action_dim: int = 4,
) -> np.ndarray:
    """
    Load human context from episode pkl files.
    
    Args:
        episode_dir: Directory containing episode pkl files
        window_len: Length of the sliding window for human context
        action_dim: Dimension of human actions
        
    Returns:
        human_contexts: Array of shape (num_episodes, window_len * action_dim)
    """
    episode_dir = Path(episode_dir)
    episode_files = sorted(episode_dir.glob("episode*.pkl"))
    
    if not episode_files:
        raise ValueError(f"No episode files found in {episode_dir}")
    
    human_contexts = []
    
    for ep_file in episode_files:
        with open(ep_file, 'rb') as f:
            data = pickle.load(f)
        
        # Extract human actions from trajectory
        trajectory = data.get('trajectory', [])
        
        # Get human action sequence (assuming it's stored in trajectory)
        # Adapt this based on your actual data structure
        human_actions = []
        for step in trajectory:
            if 'human_action' in step:
                human_actions.append(step['human_action'])
        
        if human_actions:
            human_actions = np.array(human_actions)
            
            # Pad or truncate to window_len
            if len(human_actions) < window_len:
                pad = np.zeros((window_len - len(human_actions), action_dim))
                human_actions = np.vstack([human_actions, pad])
            else:
                human_actions = human_actions[:window_len]
            
            # Flatten
            context = human_actions.flatten()
            human_contexts.append(context)
    
    return np.array(human_contexts, dtype=np.float32)


def normalize_human_context(
    context: np.ndarray,
    stats: Optional[Dict[str, np.ndarray]] = None,
) -> tuple:
    """
    Normalize human context to [-1, 1] range.
    
    Returns:
        normalized_context: Normalized context
        stats: Dictionary with 'min' and 'max' arrays
    """
    if stats is None:
        stats = {
            'min': context.min(axis=0),
            'max': context.max(axis=0),
        }
    
    # Normalize to [0, 1] then scale to [-1, 1]
    range_val = stats['max'] - stats['min']
    range_val = np.where(range_val > 0, range_val, 1.0)  # Avoid division by zero
    
    normalized = (context - stats['min']) / range_val
    normalized = normalized * 2 - 1
    
    return normalized, stats


class RunningMeanStd:
    """Running mean and standard deviation tracker."""
    
    def __init__(self, shape: tuple = (), epsilon: float = 1e-8):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon
    
    def update(self, x: np.ndarray):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)
    
    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = m2 / tot_count
        
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count
    
    @property
    def std(self):
        return np.sqrt(self.var)
    
    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / (self.std + 1e-8)


def print_model_summary(model: torch.nn.Module, name: str = "Model"):
    """Print model parameter summary."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n{name} Summary:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Non-trainable parameters: {total_params - trainable_params:,}")
