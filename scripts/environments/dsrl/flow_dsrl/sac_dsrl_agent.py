"""
SAC Agent for online DSRL in IsaacLab.
Learns to predict noise for a pre-trained flow policy conditioned on human actions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Dict, Any, Iterable
from copy import deepcopy


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for timestep conditioning."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, x: Tensor) -> Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class Actor(nn.Module):
    """
    Actor network that outputs noise for DSRL.
    Given observation + human context, outputs noise to steer the flow policy.
    """
    
    def __init__(
        self,
        obs_dim: int,
        human_context_dim: int,
        noise_dim: int,  # Output dimension (action_dim * pred_horizon for flat noise)
        hidden_dim: int = 256,
        n_layers: int = 3,
        noise_bound: float = 2.0,
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.human_context_dim = human_context_dim
        self.noise_dim = noise_dim
        self.noise_bound = noise_bound
        
        input_dim = obs_dim + human_context_dim
        
        # Build MLP
        layers = []
        for i in range(n_layers):
            in_features = input_dim if i == 0 else hidden_dim
            layers.extend([
                nn.Linear(in_features, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Mish(),
            ])
        self.net = nn.Sequential(*layers)
        
        # Output heads
        self.mu = nn.Linear(hidden_dim, noise_dim)
        self.log_std = nn.Linear(hidden_dim, noise_dim)
        
        # Scaling
        self.register_buffer('noise_scale', torch.tensor(noise_bound, dtype=torch.float32))
    
    def forward(self, obs: Tensor, human_context: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass returning distribution parameters."""
        x = torch.cat([obs, human_context], dim=-1)
        h = self.net(x)
        
        mu = self.mu(h)
        log_std = torch.tanh(self.log_std(h))
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        
        return mu, log_std
    
    def get_action(
        self, 
        obs: Tensor, 
        human_context: Tensor,
        deterministic: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Sample action and compute log probability."""
        mu, log_std = self(obs, human_context)
        std = log_std.exp()
        
        if deterministic:
            noise = torch.tanh(mu) * self.noise_scale
            log_prob = torch.zeros(obs.shape[0], 1, device=obs.device)
        else:
            normal = torch.distributions.Normal(mu, std)
            x = normal.rsample()
            y = torch.tanh(x)
            noise = y * self.noise_scale
            
            # Log probability with tanh squashing correction
            log_prob = normal.log_prob(x)
            log_prob -= torch.log(self.noise_scale * (1 - y.pow(2)) + 1e-6)
            log_prob = log_prob.sum(-1, keepdim=True)
        
        mu_scaled = torch.tanh(mu) * self.noise_scale
        
        return noise, log_prob, mu_scaled


class SoftQNetwork(nn.Module):
    """
    Soft Q-Network for DSRL.
    Q(s, z | human_context) where z is the noise vector.
    """
    
    def __init__(
        self,
        obs_dim: int,
        noise_dim: int,
        human_context_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 3,
    ):
        super().__init__()
        
        input_dim = obs_dim + noise_dim + human_context_dim
        
        layers = []
        for i in range(n_layers):
            in_features = input_dim if i == 0 else hidden_dim
            layers.extend([
                nn.Linear(in_features, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Mish(),
            ])
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.net = nn.Sequential(*layers)
    
    def forward(
        self,
        obs: Tensor,
        noise: Tensor,
        human_context: Tensor,
    ) -> Tensor:
        x = torch.cat([obs, noise, human_context], dim=-1)
        return self.net(x)


class ActionQNetwork(nn.Module):
    """
    Q-Network for actual actions (after flow policy).
    Q(s, a | human_context) where a is the actual robot action.
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        action_horizon: int,
        human_context_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 3,
    ):
        super().__init__()
        
        # Flatten action chunk
        input_dim = obs_dim + (action_dim * action_horizon) + human_context_dim
        
        layers = []
        for i in range(n_layers):
            in_features = input_dim if i == 0 else hidden_dim
            layers.extend([
                nn.Linear(in_features, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Mish(),
            ])
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.net = nn.Sequential(*layers)
    
    def forward(
        self,
        obs: Tensor,
        actions: Tensor,
        human_context: Tensor,
    ) -> Tensor:
        # Flatten action chunk if needed
        if actions.ndim == 3:
            actions = actions.view(actions.shape[0], -1)
        x = torch.cat([obs, actions, human_context], dim=-1)
        return self.net(x)


class SACDSRLAgent(nn.Module):
    """
    SAC Agent for online DSRL in IsaacLab.
    
    Architecture:
    - Actor: Outputs noise z given (obs, human_context)
    - Noise Critic (Q_z): Evaluates Q(s, z | human_context)
    - Action Critic (Q_a): Evaluates Q(s, a | human_context) for the flow-generated actions
    - Pre-trained Flow Policy: Maps noise -> actions (frozen)
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        action_horizon: int,
        human_context_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 3,
        num_critics: int = 2,
        noise_bound: float = 2.0,
        discount: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        use_autotune: bool = True,
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.human_context_dim = human_context_dim
        self.noise_dim = action_dim * action_horizon  # Flat noise dimension
        
        self.discount = discount
        self.tau = tau
        self.use_autotune = use_autotune
        
        # Actor
        self.actor = Actor(
            obs_dim=obs_dim,
            human_context_dim=human_context_dim,
            noise_dim=self.noise_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            noise_bound=noise_bound,
        )
        
        # Noise critic ensemble Q_z(s, z, human_context)
        self.noise_critic_ensemble = nn.ModuleList([
            SoftQNetwork(
                obs_dim=obs_dim,
                noise_dim=self.noise_dim,
                human_context_dim=human_context_dim,
                hidden_dim=hidden_dim,
                n_layers=n_layers,
            )
            for _ in range(num_critics)
        ])
        
        # Action critic ensemble Q_a(s, a, human_context)
        self.action_critic_ensemble = nn.ModuleList([
            ActionQNetwork(
                obs_dim=obs_dim,
                action_dim=action_dim,
                action_horizon=action_horizon,
                human_context_dim=human_context_dim,
                hidden_dim=hidden_dim,
                n_layers=n_layers,
            )
            for _ in range(num_critics)
        ])
        
        # Target networks
        self.target_action_critic_ensemble = nn.ModuleList([
            deepcopy(critic) for critic in self.action_critic_ensemble
        ])
        for critic in self.target_action_critic_ensemble:
            for param in critic.parameters():
                param.requires_grad = False
        
        # Entropy tuning
        if use_autotune:
            self.target_entropy = -self.noise_dim
            self.log_alpha = nn.Parameter(torch.zeros(1))
        else:
            self.alpha = alpha
        
        # Flow policy (will be set externally)
        self.flow_policy = None
    
    def set_flow_policy(self, flow_policy: nn.Module):
        """Set the pre-trained flow policy."""
        self.flow_policy = flow_policy
        self.flow_policy.eval()
        for param in self.flow_policy.parameters():
            param.requires_grad = False
    
    @property
    def alpha(self) -> float:
        if self.use_autotune:
            return self.log_alpha.exp().item()
        return self._alpha
    
    @alpha.setter
    def alpha(self, value: float):
        self._alpha = value
    
    def get_action(
        self,
        obs: Tensor,
        human_context: Tensor,
        deterministic: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Get action from the policy.
        
        Returns:
            noise: The noise vector z
            actions: The actual robot actions from flow policy
            log_prob: Log probability of the noise
        """
        noise, log_prob, mu = self.actor.get_action(obs, human_context, deterministic)
        
        # Generate actions through flow policy
        if self.flow_policy is not None:
            with torch.no_grad():
                # Reshape noise for flow policy: (B, noise_dim) -> (B, action_horizon, action_dim)
                noise_reshaped = noise.view(-1, self.action_horizon, self.action_dim)
                actions = self.flow_policy.sample(
                    obs=obs,
                    action_noise=noise_reshaped,
                )
        else:
            # If no flow policy, just use noise directly
            actions = noise.view(-1, self.action_horizon, self.action_dim)
        
        return noise, actions, log_prob
    
    def get_noise_q(
        self,
        obs: Tensor,
        noise: Tensor,
        human_context: Tensor,
    ) -> Tensor:
        """Get Q-values from noise critics."""
        qs = torch.stack([
            q(obs, noise, human_context) for q in self.noise_critic_ensemble
        ])
        return qs
    
    def get_action_q(
        self,
        obs: Tensor,
        actions: Tensor,
        human_context: Tensor,
    ) -> Tensor:
        """Get Q-values from action critics."""
        qs = torch.stack([
            q(obs, actions, human_context) for q in self.action_critic_ensemble
        ])
        return qs
    
    def get_target_action_q(
        self,
        obs: Tensor,
        actions: Tensor,
        human_context: Tensor,
    ) -> Tensor:
        """Get Q-values from target action critics."""
        qs = torch.stack([
            q(obs, actions, human_context) for q in self.target_action_critic_ensemble
        ])
        return qs
    
    def compute_critic_loss(
        self,
        obs: Tensor,
        actions: Tensor,
        next_obs: Tensor,
        rewards: Tensor,
        terminals: Tensor,
        human_context: Tensor,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        """Compute action critic loss."""
        log_dict = {}
        
        with torch.no_grad():
            # Get next actions
            next_noise, next_actions, next_log_prob = self.get_action(
                next_obs, human_context, deterministic=False
            )
            
            # Target Q-values
            next_qs = self.get_target_action_q(next_obs, next_actions, human_context)
            next_q = next_qs.min(dim=0).values
            
            # Entropy bonus
            alpha = self.log_alpha.exp() if self.use_autotune else self.alpha
            target_q = rewards.unsqueeze(-1) + (1 - terminals.unsqueeze(-1)) * self.discount * (
                next_q - alpha * next_log_prob
            )
        
        # Current Q-values
        current_qs = self.get_action_q(obs, actions, human_context)
        
        # MSE loss
        critic_loss = F.mse_loss(current_qs, target_q.unsqueeze(0).expand_as(current_qs))
        
        log_dict.update({
            'critic/loss': critic_loss.item(),
            'critic/q_mean': current_qs.mean().item(),
            'critic/q_min': current_qs.min().item(),
            'critic/q_max': current_qs.max().item(),
            'critic/target_q': target_q.mean().item(),
        })
        
        return critic_loss, log_dict
    
    def compute_noise_critic_loss(
        self,
        obs: Tensor,
        human_context: Tensor,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        """
        Compute noise critic loss.
        Q_z should match Q_a for the same (obs, noise -> action) mapping.
        """
        log_dict = {}
        
        with torch.no_grad():
            # Sample noise and get corresponding actions
            noise, actions, _ = self.get_action(obs, human_context)
            
            # Get target Q-values from action critics
            target_qs = self.get_action_q(obs, actions, human_context)
        
        # Get noise critic Q-values
        noise_qs = self.get_noise_q(obs, noise, human_context)
        
        # MSE loss to match action critic
        noise_critic_loss = F.mse_loss(noise_qs, target_qs)
        
        log_dict.update({
            'noise_critic/loss': noise_critic_loss.item(),
            'noise_critic/q_mean': noise_qs.mean().item(),
        })
        
        return noise_critic_loss, log_dict
    
    def compute_actor_loss(
        self,
        obs: Tensor,
        human_context: Tensor,
    ) -> Tuple[Tensor, Tensor, Dict[str, Any]]:
        """Compute actor loss using noise critic."""
        log_dict = {}
        
        # Get action and log prob
        noise, log_prob, mu = self.actor.get_action(obs, human_context)
        
        # Get Q-values from noise critic
        qs = self.get_noise_q(obs, noise, human_context)
        q = qs.min(dim=0).values
        
        # Entropy-regularized actor loss
        alpha = self.log_alpha.exp() if self.use_autotune else self.alpha
        actor_loss = (alpha * log_prob - q).mean()
        
        # Alpha loss for entropy tuning
        if self.use_autotune:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        else:
            alpha_loss = torch.tensor(0.0)
        
        log_dict.update({
            'actor/loss': actor_loss.item(),
            'actor/log_prob': log_prob.mean().item(),
            'actor/noise_mean': noise.mean().item(),
            'actor/noise_std': noise.std().item(),
            'actor/alpha': alpha,
        })
        
        return actor_loss, alpha_loss, log_dict
    
    def update(
        self,
        obs: Tensor,
        actions: Tensor,
        next_obs: Tensor,
        rewards: Tensor,
        terminals: Tensor,
        human_context: Tensor,
        optimizers: Dict[str, torch.optim.Optimizer],
    ) -> Dict[str, Any]:
        """Perform one update step."""
        log_dict = {}
        
        # Update action critics
        critic_loss, critic_log = self.compute_critic_loss(
            obs, actions, next_obs, rewards, terminals, human_context
        )
        optimizers['critic'].zero_grad()
        critic_loss.backward()
        optimizers['critic'].step()
        log_dict.update(critic_log)
        
        # Update noise critics
        noise_critic_loss, noise_critic_log = self.compute_noise_critic_loss(
            obs, human_context
        )
        optimizers['noise_critic'].zero_grad()
        noise_critic_loss.backward()
        optimizers['noise_critic'].step()
        log_dict.update(noise_critic_log)
        
        # Update actor
        actor_loss, alpha_loss, actor_log = self.compute_actor_loss(
            obs, human_context
        )
        optimizers['actor'].zero_grad()
        actor_loss.backward()
        optimizers['actor'].step()
        log_dict.update(actor_log)
        
        # Update alpha
        if self.use_autotune:
            optimizers['alpha'].zero_grad()
            alpha_loss.backward()
            optimizers['alpha'].step()
            log_dict['actor/alpha_loss'] = alpha_loss.item()
        
        # Update target networks
        self._update_target_networks()
        
        return log_dict
    
    def _update_target_networks(self):
        """Soft update of target networks."""
        for critic, target in zip(self.action_critic_ensemble, self.target_action_critic_ensemble):
            for param, target_param in zip(critic.parameters(), target.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )


def create_optimizers(agent: SACDSRLAgent, lr: float = 3e-4) -> Dict[str, torch.optim.Optimizer]:
    """Create optimizers for the DSRL agent."""
    optimizers = {
        'actor': torch.optim.Adam(agent.actor.parameters(), lr=lr),
        'critic': torch.optim.Adam(
            [p for critic in agent.action_critic_ensemble for p in critic.parameters()],
            lr=lr
        ),
        'noise_critic': torch.optim.Adam(
            [p for critic in agent.noise_critic_ensemble for p in critic.parameters()],
            lr=lr
        ),
    }
    
    if agent.use_autotune:
        optimizers['alpha'] = torch.optim.Adam([agent.log_alpha], lr=lr)
    
    return optimizers
