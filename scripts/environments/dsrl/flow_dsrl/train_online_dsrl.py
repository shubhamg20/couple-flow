"""
Online DSRL Training Script for IsaacLab.

This script implements online Diffusion Steering via Reinforcement Learning (DSRL)
in IsaacLab environments. It trains an actor to predict noise vectors that steer
a pre-trained flow policy towards actions conditioned on human demonstrations.

Usage:
    python train_online_dsrl.py --flow_checkpoint <path> --human_context_file <path>
"""

import argparse
import os
import sys
import time
import glob
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, "/home/weirdlab/Documents/summers/IsaacLab/source/couple-flow-policy")

from buffer import ChunkingReplayBuffer
from sac_dsrl_agent import SACDSRLAgent, create_optimizers
from flow_policy_wrapper import load_flow_policy


def parse_args():
    parser = argparse.ArgumentParser(description="Online DSRL Training for IsaacLab")
    
    # Environment
    parser.add_argument("--task", type=str, default="Isaac-PickPlace-Franka-custom")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--target_object", type=str, default="sushi", 
                       choices=["sushi", "apple", "mug"])
    parser.add_argument("--headless", action="store_true", default=True)
    
    # Flow policy
    parser.add_argument("--flow_checkpoint", type=str, required=True,
                       help="Path to pre-trained flow policy checkpoint")
    
    # Human context
    parser.add_argument("--human_context_file", type=str, default=None,
                       help="Path to human context data (npz file with human trajectories) - DEPRECATED: use --data_path instead")
    parser.add_argument("--data_path", type=str, default="source/recorded_runs",
                       help="Path to directory containing episode PKL files")
    parser.add_argument("--human_context_dim", type=int, default=1120,
                       help="Dimension of human context (280 * 4)")
    parser.add_argument("--window_size", type=int, default=280,
                       help="Size of human context window")
    
    # Model dimensions
    parser.add_argument("--obs_dim", type=int, default=16)
    parser.add_argument("--action_dim", type=int, default=7)
    parser.add_argument("--action_horizon", type=int, default=8)
    parser.add_argument("--obs_horizon", type=int, default=1)
    
    # Training
    parser.add_argument("--train_steps", type=int, default=100000)
    parser.add_argument("--learning_starts", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--noise_bound", type=float, default=2.0)
    
    # Buffer
    parser.add_argument("--buffer_capacity", type=int, default=100000)
    
    # Logging
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--eval_interval", type=int, default=1000)
    parser.add_argument("--save_interval", type=int, default=5000)
    parser.add_argument("--save_dir", type=str, default="./checkpoints/online_dsrl")
    
    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--use_wandb", action="store_true", default=False)
    parser.add_argument("--wandb_project", type=str, default="online_dsrl_isaaclab")
    
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_human_trajectories_from_pkls(
    data_path: str, 
    task: str, 
    device: str
) -> Tuple[List[torch.Tensor], List[Dict]]:
    """
    Loads human actions directly from individual episode PKL files.
    
    Returns:
        human_trajectories: List of tensors, where each tensor is the action trajectory for one episode
        initial_objects_list: List of dictionaries containing initial object positions for each episode
    """
    # E.g., "source/recorded_runs/sushi_org/episode*.pkl"
    pattern = os.path.join(data_path, f"{task}_org", "episode*.pkl")
    episode_files = sorted(glob.glob(pattern))
    
    if not episode_files:
        # Try alternative path structure
        alt_pattern = os.path.join(data_path, task, "episode*.pkl")
        episode_files = sorted(glob.glob(alt_pattern))
        if not episode_files:
            raise ValueError(f"No PKL files found for task {task} at {pattern} or {alt_pattern}")

    print(f"Loading {len(episode_files)} human trajectories from PKL files...")
    
    human_trajectories = []
    initial_objects_list = []
    
    for file_path in episode_files:
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                
            # 1. Extract the human actions from the 'trajectory' list
            # (Assuming the key is 'human_action' based on standard SERL/Franka formats)
            if 'trajectory' in data:
                actions = []
                for step in data['trajectory']:
                    if 'human_action' in step:
                        actions.append(step['human_action'])
                    elif 'action' in step:
                        actions.append(step['action'])
                
                if actions:
                    actions = np.array(actions, dtype=np.float32)
                    human_trajectories.append(torch.tensor(actions, device=device))
                else:
                    print(f"Warning: No actions found in {file_path}, skipping...")
                    continue
            else:
                print(f"Warning: No 'trajectory' key in {file_path}, skipping...")
                continue
                
            # 2. Extract the initial object coordinates for environment resetting
            if 'initial_objects' in data:
                initial_objects_list.append(data['initial_objects'])
            else:
                initial_objects_list.append(None)
                
        except Exception as e:
            print(f"Error loading {file_path}: {e}, skipping...")
            continue
    
    if not human_trajectories:
        raise ValueError(f"No valid trajectories loaded from {pattern}")
        
    print(f"  - Successfully loaded {len(human_trajectories)} trajectories")
    print(f"  - Trajectory lengths: {[len(t) for t in human_trajectories[:5]]}...")
    
    return human_trajectories, initial_objects_list


def get_human_context_window(
    human_trajectories: List[torch.Tensor],
    episodes: torch.Tensor,
    current_steps: torch.Tensor,
    window_size: int = 280,
) -> torch.Tensor:
    """
    Fetches the [t : t+window_size] window of human actions for the chosen episodes.
    
    Args:
        human_trajectories: List of trajectory tensors, one per episode
        episodes: (batch_size,) tensor of episode indices
        current_steps: (batch_size,) tensor of current step indices for each episode
        window_size: Size of context window
        
    Returns:
        contexts: (batch_size, window_size * action_dim) tensor of flattened context windows
    """
    batch_size = len(episodes)
    if batch_size == 0:
        return torch.zeros(0, window_size * 4, device=human_trajectories[0].device)
    
    action_dim = human_trajectories[0].shape[-1]
    device = human_trajectories[0].device
    contexts = torch.zeros((batch_size, window_size * action_dim), device=device)
    
    for i in range(batch_size):
        ep_idx = episodes[i].item()
        current_step = current_steps[i].item()
        
        if ep_idx >= len(human_trajectories):
            # Episode index out of range, use zeros
            continue
            
        traj = human_trajectories[ep_idx]
        traj_len = len(traj)
        
        # Get the window
        start = min(current_step, max(0, traj_len - 1))
        end = min(current_step + window_size, traj_len)
        window = traj[start:end]
        
        # Pad with the final action if we hit the end of the trajectory
        if len(window) < window_size:
            pad_len = window_size - len(window)
            if len(window) > 0:
                pad = window[-1:].repeat(pad_len, 1)
            else:
                # Empty window, pad with zeros
                pad = torch.zeros(pad_len, action_dim, device=device)
            window = torch.cat([window, pad], dim=0)
            
        contexts[i] = window.flatten()
        
    return contexts


def load_human_context(file_path: Optional[str], device: str) -> Optional[torch.Tensor]:
    """Load human context data from file (legacy support for npz files)."""
    if file_path is None:
        return None
    
    data = np.load(file_path, allow_pickle=True)
    
    # Expect human_context key with shape (N, context_dim)
    if 'human_context' in data:
        human_context = torch.tensor(data['human_context'], dtype=torch.float32, device=device)
    elif 'gr1t2_absolute_actions' in data:
        # Process from raw human actions
        human_actions = data['gr1t2_absolute_actions']
        # Flatten and normalize as done in dataset_dsrl.py
        # This is a simplified version - adapt as needed
        human_context = torch.tensor(human_actions, dtype=torch.float32, device=device)
    else:
        raise ValueError(f"Could not find human context in {file_path}")
    
    return human_context


def main():
    args = parse_args()
    
    # Setup
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # Create save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(args.save_dir) / f"{args.target_object}_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(save_dir / "config.txt", "w") as f:
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")
    
    print("=" * 60)
    print("Online DSRL Training for IsaacLab")
    print("=" * 60)
    print(f"Task: {args.task}")
    print(f"Target Object: {args.target_object}")
    print(f"Flow Checkpoint: {args.flow_checkpoint}")
    print(f"Save Directory: {save_dir}")
    print("=" * 60)
    
    # Initialize wandb
    if args.use_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            config=vars(args),
            name=f"dsrl_{args.target_object}_{timestamp}",
        )
    
    # Load human context from PKL files (before creating env to know max_episodes)
    print("\n[1/5] Loading human context data...")
    human_trajectories = None
    initial_objects_list = None
    human_context_data = None
    
    if args.human_context_file:
        # Legacy: load from npz file
        print("  - Using legacy npz file loading...")
        human_context_data = load_human_context(args.human_context_file, device)
        if human_context_data is not None:
            print(f"  - Human context shape: {human_context_data.shape}")
        else:
            print("  - No human context file provided, using dummy context")
    else:
        # New: load directly from PKL files
        try:
            human_trajectories, initial_objects_list = load_human_trajectories_from_pkls(
                data_path=args.data_path,
                task=args.target_object,
                device=device
            )
            print(f"  - Loaded {len(human_trajectories)} human trajectories from PKL files")
        except Exception as e:
            print(f"  - Failed to load from PKL files: {e}")
            print("  - Falling back to dummy context")
            human_trajectories = None
            initial_objects_list = None
    
    # Create environment
    print("\n[2/5] Creating IsaacLab environment...")
    from isaaclab_env import make_isaaclab_env
    env = make_isaaclab_env(
        task=args.task,
        num_envs=args.num_envs,
        device=args.device,
        headless=args.headless,
        target_object=args.target_object,
        max_episodes=len(human_trajectories) - 1 if human_trajectories else 49,
    )
    print(f"  - Observation dim: {args.obs_dim}")
    print(f"  - Action dim: {args.action_dim}")
    print(f"  - Num envs: {args.num_envs}")
    
    # Set initial objects if loaded from PKL
    if initial_objects_list is not None:
        env.set_human_initial_objects(initial_objects_list)
    
    # Load flow policy
    print("\n[3/5] Loading pre-trained flow policy...")
    flow_policy = load_flow_policy(
        checkpoint_path=args.flow_checkpoint,
        obs_dim=args.obs_dim,
        action_dim=args.action_dim,
        pred_horizon=args.action_horizon,
        obs_horizon=args.obs_horizon,
        device=args.device,
    )
    print(f"  - Flow policy loaded from: {args.flow_checkpoint}")
    
    # Create agent
    print("\n[4/5] Creating DSRL agent...")
    agent = SACDSRLAgent(
        obs_dim=args.obs_dim,
        action_dim=args.action_dim,
        action_horizon=args.action_horizon,
        human_context_dim=args.human_context_dim,
        hidden_dim=256,
        n_layers=3,
        num_critics=2,
        noise_bound=args.noise_bound,
        discount=args.discount,
        tau=args.tau,
        use_autotune=True,
    )
    agent.set_flow_policy(flow_policy)
    agent = agent.to(device)
    
    optimizers = create_optimizers(agent, lr=args.lr)
    print(f"  - Actor parameters: {sum(p.numel() for p in agent.actor.parameters()):,}")
    print(f"  - Critic parameters: {sum(p.numel() for c in agent.action_critic_ensemble for p in c.parameters()):,}")
    
    # Create replay buffer
    print("\n[5/5] Creating replay buffer...")
    buffer = ChunkingReplayBuffer(
        capacity=args.buffer_capacity,
        n_envs=args.num_envs,
        obs_dim=args.obs_dim,
        action_dim=args.action_dim,
        human_context_dim=args.human_context_dim,
        action_horizon=args.action_horizon,
        discount=args.discount,
    )
    print(f"  - Buffer capacity: {args.buffer_capacity}")
    
    # Training loop
    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60)
    
    obs_dict = env.reset()
    obs = obs_dict['state']
    
    # Get chosen episodes from reset (if using PKL loading)
    if human_trajectories is not None:
        current_episodes = obs_dict['chosen_episodes']  # (num_envs,)
        current_steps = torch.zeros(args.num_envs, dtype=torch.int32, device=device)
        episode_human_context = get_human_context_window(
            human_trajectories, current_episodes, current_steps, window_size=args.window_size
        )
    else:
        # Legacy: use random sampling from npz data
        if human_context_data is not None:
            indices = torch.randint(0, len(human_context_data), (args.num_envs,))
            episode_human_context = human_context_data[indices]
        else:
            episode_human_context = torch.zeros(args.num_envs, args.human_context_dim, device=device)
        current_episodes = None
        current_steps = None
    
    cumulative_rewards = torch.zeros(args.num_envs, device=device)
    episode_lengths = torch.zeros(args.num_envs, device=device)
    total_successes = 0
    
    pbar = tqdm(range(args.train_steps), desc="Training", dynamic_ncols=True)
    
    for step in pbar:
        log_dict = {}
        step_start = time.time()
        
        # Update human context for current step (before getting action)
        if human_trajectories is not None and current_episodes is not None:
            episode_human_context = get_human_context_window(
                human_trajectories, current_episodes, current_steps, window_size=args.window_size
            )
        
        # Get action from agent
        with torch.no_grad():
            if step < args.learning_starts:
                # Random exploration
                noise = torch.randn(args.num_envs, args.action_dim * args.action_horizon, device=device)
                noise = torch.tanh(noise) * args.noise_bound
                noise_reshaped = noise.view(args.num_envs, args.action_horizon, args.action_dim)
                actions = flow_policy.sample(obs, noise_reshaped)
            else:
                # Policy action
                noise, actions, log_prob = agent.get_action(obs, episode_human_context)
        
        # Step environment with action chunk
        next_obs_dict, rewards, terminated, truncated, info = env.step_chunk(actions)
        next_obs = next_obs_dict['state']
        done = terminated | truncated
        
        # Store transition
        buffer.add_chunk(
            obs=obs,
            next_obs=next_obs,
            action_chunk=actions,
            human_context=episode_human_context,
            reward=rewards,
            terminal=done.float(),
        )
        
        # Update tracking
        cumulative_rewards += rewards
        episode_lengths += args.action_horizon
        
        # Advance steps after executing action chunk (for next iteration's context)
        if human_trajectories is not None and current_episodes is not None:
            current_steps += args.action_horizon
        
        # Handle episode ends
        done_indices = torch.where(done)[0]
        if len(done_indices) > 0:
            for idx in done_indices:
                if info['success'][idx]:
                    total_successes += 1
                
                log_dict[f'episode/reward'] = cumulative_rewards[idx].item()
                log_dict[f'episode/length'] = episode_lengths[idx].item()
                log_dict[f'episode/success'] = float(info['success'][idx])
            
            # Reset completed environments
            cumulative_rewards[done_indices] = 0
            episode_lengths[done_indices] = 0
            
            if human_trajectories is not None:
                # Reset steps for done environments
                current_steps[done_indices] = 0
                
                # Reset environment (this will set new chosen_episodes)
                reset_obs = env.reset(env_ids=done_indices)
                obs[done_indices] = reset_obs['state'][done_indices]
                
                # Update current_episodes for reset environments
                current_episodes[done_indices] = reset_obs['chosen_episodes'][done_indices]
                
                # Get new human context for reset episodes (at step 0)
                reset_steps = torch.zeros(len(done_indices), dtype=torch.int32, device=device)
                reset_context = get_human_context_window(
                    human_trajectories, current_episodes[done_indices], reset_steps, window_size=args.window_size
                )
                episode_human_context[done_indices] = reset_context
            else:
                # Legacy: use random sampling
                if human_context_data is not None:
                    indices = torch.randint(0, len(human_context_data), (len(done_indices),))
                    episode_human_context[done_indices] = human_context_data[indices]
                
                # Reset environment
                reset_obs = env.reset()
                obs = reset_obs['state']
        else:
            obs = next_obs
        
        # Training updates
        if step >= args.learning_starts and len(buffer) >= args.batch_size:
            batch = buffer.sample(args.batch_size)
            batch = {k: v.to(device) for k, v in batch.items()}
            
            update_log = agent.update(
                obs=batch['obs'],
                actions=batch['actions'],
                next_obs=batch['next_obs'],
                rewards=batch['rewards'],
                terminals=batch['terminals'],
                human_context=batch['human_context'],
                optimizers=optimizers,
            )
            log_dict.update(update_log)
        
        # Logging
        if step % args.log_interval == 0 and log_dict:
            log_dict['train/step'] = step
            log_dict['train/total_successes'] = total_successes
            log_dict['train/buffer_size'] = len(buffer)
            log_dict['train/sps'] = 1.0 / (time.time() - step_start)
            
            pbar.set_postfix({
                'success': total_successes,
                'buffer': len(buffer),
            })
            
            if args.use_wandb:
                import wandb
                wandb.log(log_dict)
        
        # Save checkpoint
        if step > 0 and step % args.save_interval == 0:
            checkpoint = {
                'step': step,
                'agent_state_dict': agent.state_dict(),
                'optimizers': {k: v.state_dict() for k, v in optimizers.items()},
                'total_successes': total_successes,
            }
            torch.save(checkpoint, save_dir / f"checkpoint_{step}.pt")
            torch.save(checkpoint, save_dir / "last.pt")
            print(f"\n[Checkpoint] Saved at step {step}")
    
    # Final save
    checkpoint = {
        'step': args.train_steps,
        'agent_state_dict': agent.state_dict(),
        'optimizers': {k: v.state_dict() for k, v in optimizers.items()},
        'total_successes': total_successes,
    }
    torch.save(checkpoint, save_dir / "final.pt")
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Total Successes: {total_successes}")
    print(f"Checkpoints saved to: {save_dir}")
    print("=" * 60)
    
    # Cleanup
    env.close()
    
    if args.use_wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
