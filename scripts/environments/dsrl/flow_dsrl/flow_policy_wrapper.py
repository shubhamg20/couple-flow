"""
Wrapper for the pre-trained flow policy from couple-flow-policy.
Loads the flow policy and provides a simple interface for DSRL.
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys

# Add couple-flow-policy to path
sys.path.insert(0, "/home/weirdlab/Documents/summers/IsaacLab/source/couple-flow-policy")


class FlowPolicyWrapper(nn.Module):
    """
    Wrapper for the pre-trained flow policy.
    
    The flow policy maps:
        noise (x0) + obs_cond -> actions (x1)
        
    This is the forward flow that we use during inference.
    The noise is predicted by the SAC actor based on human context.
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        obs_dim: int = 16,
        action_dim: int = 7,
        pred_horizon: int = 8,
        obs_horizon: int = 1,
        device: str = "cuda",
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.device = torch.device(device)
        
        # Load the flow policy
        self.flow_net = self._load_flow_policy(checkpoint_path)
        self.flow_net.eval()
        
        # Freeze all parameters
        for param in self.flow_net.parameters():
            param.requires_grad = False
    
    def _load_flow_policy(self, checkpoint_path: str) -> nn.Module:
        """Load the pre-trained flow network from checkpoint."""
        from flow_policy.networks import Transformer
        
        checkpoint_path = Path(checkpoint_path)
        
        if checkpoint_path.is_dir():
            # Load from directory with config
            checkpoint = torch.load(checkpoint_path / "last.pt", map_location=self.device)
        else:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Create flow network
        flow_net = Transformer(
            action_dim=self.action_dim,
            action_len=self.pred_horizon,
            global_cond_dim=self.obs_dim * self.obs_horizon,
        )
        
        # Load weights
        state_dict = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
        
        # Handle prefix in state dict
        if any(k.startswith('flow_net.') for k in state_dict.keys()):
            flow_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('flow_net.'):
                    flow_state_dict[k.replace('flow_net.', '')] = v
            flow_net.load_state_dict(flow_state_dict)
        else:
            flow_net.load_state_dict(state_dict)
        
        flow_net.to(self.device)
        return flow_net
    
    @torch.no_grad()
    def sample(
        self,
        obs: torch.Tensor,
        action_noise: torch.Tensor,
        num_steps: int = 50,
    ) -> torch.Tensor:
        """
        Sample actions by running forward flow from noise to actions.
        
        Args:
            obs: Observation tensor (B, obs_dim) or (B, obs_horizon, obs_dim)
            action_noise: Noise tensor (B, pred_horizon, action_dim)
            num_steps: Number of integration steps
            
        Returns:
            actions: Action tensor (B, pred_horizon, action_dim)
        """
        # Prepare observation conditioning
        if obs.ndim == 2:
            obs_cond = obs.unsqueeze(1)  # (B, 1, obs_dim)
        else:
            obs_cond = obs[:, :self.obs_horizon]
        obs_cond = obs_cond.flatten(start_dim=1)  # (B, obs_horizon * obs_dim)
        
        # Ensure action_noise has correct shape
        if action_noise.ndim == 2:
            action_noise = action_noise.view(-1, self.pred_horizon, self.action_dim)
        
        # Forward flow: integrate from x0 (noise) to x1 (actions)
        curr_x = action_noise.clone()
        batch_size = curr_x.shape[0]
        dt = 1.0 / num_steps
        
        for i in range(num_steps):
            t_val = i / num_steps
            t = torch.full((batch_size,), t_val, device=self.device)
            
            # Get velocity from flow network
            v = self.flow_net(curr_x, t, global_cond=obs_cond)
            
            # Euler step
            curr_x = curr_x + v * dt
        
        return curr_x
    
    @torch.no_grad()
    def inverse_flow(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        num_steps: int = 100,
    ) -> torch.Tensor:
        """
        Invert flow from actions back to noise (for debugging/analysis).
        
        Args:
            obs: Observation tensor (B, obs_dim)
            actions: Action tensor (B, pred_horizon, action_dim)
            num_steps: Number of integration steps
            
        Returns:
            noise: Noise tensor (B, pred_horizon, action_dim)
        """
        # Prepare observation conditioning
        if obs.ndim == 2:
            obs_cond = obs.unsqueeze(1)
        else:
            obs_cond = obs[:, :self.obs_horizon]
        obs_cond = obs_cond.flatten(start_dim=1)
        
        # Inverse flow using fixed point iteration
        curr_x = actions.clone()
        batch_size = curr_x.shape[0]
        dt = 1.0 / num_steps
        
        for i in range(num_steps, 0, -1):
            t_target_val = (i - 1) / num_steps
            t_target = torch.full((batch_size,), t_target_val, device=self.device)
            
            # Fixed point iteration
            x_prev = curr_x.clone()
            for _ in range(15):  # Fixed point iterations
                v = self.flow_net(x_prev, t_target, global_cond=obs_cond)
                x_new = curr_x - dt * v
                
                if torch.norm(x_new - x_prev) < 1e-6:
                    break
                x_prev = x_new
            
            curr_x = x_prev
        
        return curr_x
    
    def forward(
        self,
        obs: torch.Tensor,
        action_noise: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass (alias for sample)."""
        return self.sample(obs, action_noise)


def load_flow_policy(
    checkpoint_path: str,
    obs_dim: int = 16,
    action_dim: int = 7,
    pred_horizon: int = 8,
    obs_horizon: int = 1,
    device: str = "cuda",
) -> FlowPolicyWrapper:
    """Factory function to load flow policy."""
    return FlowPolicyWrapper(
        checkpoint_path=checkpoint_path,
        obs_dim=obs_dim,
        action_dim=action_dim,
        pred_horizon=pred_horizon,
        obs_horizon=obs_horizon,
        device=device,
    )
