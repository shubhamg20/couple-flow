"""
Evaluation script for trained online DSRL agent.

Usage:
    python eval_dsrl.py --checkpoint <path> --flow_checkpoint <path>
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, "/home/weirdlab/Documents/summers/IsaacLab/source/couple-flow-policy")

from sac_dsrl_agent import SACDSRLAgent
from flow_policy_wrapper import load_flow_policy
from utils import set_seed, to_device


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate DSRL Agent")
    
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to agent checkpoint")
    parser.add_argument("--flow_checkpoint", type=str, required=True,
                       help="Path to flow policy checkpoint")
    
    # Environment
    parser.add_argument("--task", type=str, default="Isaac-PickPlace-Franka-custom")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--target_object", type=str, default="sushi")
    
    # Evaluation
    parser.add_argument("--num_episodes", type=int, default=50)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--render", action="store_true", default=False)
    
    # Model dimensions
    parser.add_argument("--obs_dim", type=int, default=16)
    parser.add_argument("--action_dim", type=int, default=7)
    parser.add_argument("--action_horizon", type=int, default=8)
    parser.add_argument("--human_context_dim", type=int, default=1120)
    
    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    
    return parser.parse_args()


def evaluate(
    agent: SACDSRLAgent,
    env,
    num_episodes: int,
    max_steps: int,
    human_context: torch.Tensor,
    deterministic: bool = True,
    device: str = "cuda",
):
    """Run evaluation episodes."""
    
    results = {
        'successes': [],
        'rewards': [],
        'lengths': [],
    }
    
    agent.eval()
    
    for episode in tqdm(range(num_episodes), desc="Evaluating"):
        obs_dict = env.reset()
        obs = obs_dict['state']
        
        episode_reward = 0.0
        episode_length = 0
        
        for step in range(max_steps // agent.action_horizon):
            with torch.no_grad():
                noise, actions, _ = agent.get_action(
                    obs, human_context, deterministic=deterministic
                )
            
            next_obs_dict, rewards, terminated, truncated, info = env.step_chunk(actions)
            
            episode_reward += rewards.sum().item()
            episode_length += agent.action_horizon
            
            if terminated.any() or truncated.any():
                break
            
            obs = next_obs_dict['state']
        
        success = info['success'].any().item()
        
        results['successes'].append(success)
        results['rewards'].append(episode_reward)
        results['lengths'].append(episode_length)
    
    # Compute statistics
    success_rate = np.mean(results['successes'])
    mean_reward = np.mean(results['rewards'])
    std_reward = np.std(results['rewards'])
    mean_length = np.mean(results['lengths'])
    
    print("\n" + "=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    print(f"Episodes: {num_episodes}")
    print(f"Success Rate: {success_rate:.2%}")
    print(f"Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"Mean Episode Length: {mean_length:.1f}")
    print("=" * 50)
    
    return results


def main():
    args = parse_args()
    
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    print("=" * 60)
    print("DSRL Agent Evaluation")
    print("=" * 60)
    
    # Load flow policy
    print("\nLoading flow policy...")
    flow_policy = load_flow_policy(
        checkpoint_path=args.flow_checkpoint,
        obs_dim=args.obs_dim,
        action_dim=args.action_dim,
        pred_horizon=args.action_horizon,
        device=args.device,
    )
    
    # Create agent
    print("Creating agent...")
    agent = SACDSRLAgent(
        obs_dim=args.obs_dim,
        action_dim=args.action_dim,
        action_horizon=args.action_horizon,
        human_context_dim=args.human_context_dim,
        hidden_dim=256,
        n_layers=3,
        num_critics=2,
    )
    agent.set_flow_policy(flow_policy)
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    agent.load_state_dict(checkpoint['agent_state_dict'])
    agent = agent.to(device)
    agent.eval()
    
    print(f"Loaded from step {checkpoint.get('step', 'unknown')}")
    
    # Create environment
    print("\nCreating environment...")
    from isaaclab_env import make_isaaclab_env
    env = make_isaaclab_env(
        task=args.task,
        num_envs=args.num_envs,
        device=args.device,
        headless=not args.render,
        target_object=args.target_object,
    )
    
    # Dummy human context for evaluation
    human_context = torch.zeros(
        args.num_envs, args.human_context_dim, device=device
    )
    
    # Run evaluation
    results = evaluate(
        agent=agent,
        env=env,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        human_context=human_context,
        deterministic=args.deterministic,
        device=args.device,
    )
    
    # Cleanup
    env.close()
    
    return results


if __name__ == "__main__":
    main()
