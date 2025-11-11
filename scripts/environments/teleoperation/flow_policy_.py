"""Flow Matching Policy for inference."""
from flow_policy.dataset import OGBenchDataset, IsaacLabDataset
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Union
from flow_policy.configs import FlowMatchingModelRunConfig
from flow_policy.make_networks import instantiate_flow_matching_artifacts
from flow_policy.dataset import normalize_data, unnormalize_data

# Import the conditional flow matching from the submodule
import sys
import os
sys.path.append('/home/shubham/summer/serl-flow/conditional-flow-matching')
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher, ExactOptimalTransportConditionalFlowMatcher

# For ODE integration during inference
try:
    from torchdyn.core import NeuralODE
except ImportError:
    print("Warning: torchdyn not available. Install it for ODE-based inference.")
    NeuralODE = None

class FlowMatchingPolicy:
    """Flow Matching Policy for robotic control."""
    
    def __init__(self, cfg: FlowMatchingModelRunConfig, checkpoint_path: Optional[str] = None):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        
        # Load the model architecture
        self.nets, self.device = instantiate_flow_matching_artifacts(cfg, model_only=True)
        
        # Initialize the flow matcher (for consistency, though not needed for inference)
        if cfg.flow_matching_type == "ConditionalFlowMatcher":
            self.flow_matcher = ConditionalFlowMatcher(sigma=cfg.sigma)
        elif cfg.flow_matching_type == "ExactOptimalTransportConditionalFlowMatcher":
            self.flow_matcher = ExactOptimalTransportConditionalFlowMatcher(sigma=cfg.sigma)
        else:
            raise ValueError(f"Unknown flow matching type: {cfg.flow_matching_type}")
        
        # Load checkpoint if provided
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
        
        self.nets.eval()
        
    def load_checkpoint(self, checkpoint_path: str):
        """Load model weights from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.nets.load_state_dict(checkpoint['state_dict'])
        self.stats = checkpoint['stats']
        print(f"Loaded checkpoint from {checkpoint_path}")
        
    def create_flow_field(self, obs_cond: torch.Tensor) -> callable:
        """Create a flow field function for ODE integration."""
        def flow_field(t, x):
            # Ensure t is a tensor
            if not torch.is_tensor(t):
                t = torch.tensor(t, dtype=torch.float32, device=x.device)
            
            # Expand t to match batch size
            if len(t.shape) == 0:
                t = t.expand(x.shape[0])
            
            # Predict velocity field
            with torch.no_grad():
                vt = self.nets['flow_net'](x, t, global_cond=obs_cond)
            return vt
        
        return flow_field
    
    def predict_action_ode(self, 
                          obs_dict: Dict[str, np.ndarray], 
                          num_samples: int = 1,
                          num_steps: int = 50) -> np.ndarray:
        """
        Predict actions using ODE integration.
        
        Args:
            obs_dict: Dictionary containing observations
            num_samples: Number of action sequences to sample
            num_steps: Number of integration steps
            
        Returns:
            Predicted actions of shape (num_samples, pred_horizon, action_dim)
        """
        if NeuralODE is None:
            raise ImportError("torchdyn is required for ODE-based inference")
            
        with torch.no_grad():
            # Process observations
            obs_features = self._process_observations(obs_dict)
            obs_cond = obs_features.flatten(start_dim=1)
            
            # Repeat observation conditioning for num_samples
            obs_cond = obs_cond.repeat(num_samples, 1)
            
            # Sample initial noise (Gaussian source distribution)
            x0 = torch.randn(
                (num_samples, self.cfg.pred_horizon, self.cfg.action_dim), 
                device=self.device
            )
            
            # Create flow field
            flow_field = self.create_flow_field(obs_cond)
            
            # Create Neural ODE
            node = NeuralODE(flow_field, solver="dopri5", atol=1e-4, rtol=1e-4)
            
            # Integrate from t=0 to t=1
            t_span = torch.linspace(0, 1, num_steps, device=self.device)
            trajectory = node.trajectory(x0, t_span=t_span)
            
            # Get final actions (at t=1)
            actions = trajectory[-1]  # Shape: (num_samples, pred_horizon, action_dim)
            
            # Unnormalize actions
            actions_np = actions.cpu().numpy()
            for i in range(num_samples):
                actions_np[i] = unnormalize_data(actions_np[i], self.stats['action'])
            
            return actions_np
    
    def predict_action_euler(self, 
                            obs_dict: Dict[str, np.ndarray], 
                            num_samples: int = 1,
                            num_steps: int = 50) -> np.ndarray:
        """
        Predict actions using Euler integration (simpler alternative to ODE).
        
        Args:
            obs_dict: Dictionary containing observations
            num_samples: Number of action sequences to sample
            num_steps: Number of integration steps
            
        Returns:
            Predicted actions of shape (num_samples, pred_horizon, action_dim)
        """
        with torch.no_grad():
            # Process observations
            obs_features = self._process_observations(obs_dict)
            obs_cond = obs_features.flatten(start_dim=1)
            
            # Repeat observation conditioning for num_samples
            obs_cond = obs_cond.repeat(num_samples, 1)
            
            # Sample initial noise (Gaussian source distribution)
            x = torch.randn(
                (num_samples, self.cfg.pred_horizon, self.cfg.action_dim), 
                device=self.device
            )
            
            # Euler integration
            dt = 1.0 / num_steps
            for step in range(num_steps):
                t = torch.tensor(step * dt, device=self.device)
                t_batch = t.expand(num_samples)
                
                # Predict velocity field
                vt = self.nets['flow_net'](x, t_batch, global_cond=obs_cond)
                
                # Euler step
                x = x + dt * vt
            
            # Unnormalize actions
            actions_np = x.cpu().numpy()
            for i in range(num_samples):
                actions_np[i] = unnormalize_data(actions_np[i], self.stats['action'])
            
            return actions_np
    
    def _process_observations(self, obs_dict: Dict[str, np.ndarray]) -> torch.Tensor:
        """Process raw observations into the format expected by the network."""
        obs_features = None
        
        if self.cfg.with_image and 'image' in obs_dict:
            images = torch.from_numpy(obs_dict['image']).to(self.device).float()
            if len(images.shape) == 4:  # Add obs_horizon dimension if missing
                images = images.unsqueeze(0)
            
            # Take only obs_horizon frames
            images = images[:, :self.cfg.obs_horizon]
            B = images.shape[0]
            
            image_features = self.nets['vision_encoder'](
                images.flatten(end_dim=2)
            )
            
            vision_feature_dim = 512 * self.cfg.num_cameras
            image_features = image_features.reshape(
                B, self.cfg.obs_horizon, vision_feature_dim
            )
            obs_features = image_features
        
        if self.cfg.with_state and 'state' in obs_dict:
            states = torch.from_numpy(obs_dict['state']).to(self.device).float()
            if len(states.shape) == 1:  # Add batch and obs_horizon dimensions if missing
                states = states.unsqueeze(0).unsqueeze(0)
            elif len(states.shape) == 2:  # Add obs_horizon dimension if missing
                states = states.unsqueeze(1)
            
            # Take only obs_horizon frames
            states = states[:, :self.cfg.obs_horizon]
            
            if obs_features is not None:
                obs_features = torch.cat([obs_features, states], dim=-1)
            else:
                obs_features = states
        
        if obs_features is None:
            raise ValueError("No valid observations provided")
            
        return obs_features
    
    def predict_action(self, 
                      obs_dict: Dict[str, np.ndarray], 
                      num_samples: int = 1,
                      use_ode: bool = False,
                      num_steps: int = 50) -> np.ndarray:
        """
        Main prediction function.
        
        Args:
            obs_dict: Dictionary containing observations
            num_samples: Number of action sequences to sample
            use_ode: Whether to use ODE integration (True) or Euler integration (False)
            num_steps: Number of integration steps
            
        Returns:
            Predicted actions of shape (num_samples, pred_horizon, action_dim)
        """
        if use_ode and NeuralODE is not None:
            return self.predict_action_ode(obs_dict, num_samples, num_steps)
        else:
            return self.predict_action_euler(obs_dict, num_samples, num_steps)
