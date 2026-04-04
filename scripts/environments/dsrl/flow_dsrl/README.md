# Online DSRL for IsaacLab

This module implements **Online Diffusion Steering via Reinforcement Learning (DSRL)** for IsaacLab environments, specifically the Franka pick-place task.

## Overview

DSRL learns to steer a pre-trained flow policy by predicting noise vectors conditioned on human demonstrations. The architecture consists of:

1. **Pre-trained Flow Policy**: Maps noise (x0) → robot actions (x1), frozen during training
2. **Actor Network**: Predicts noise z given (observation, human_context)
3. **Noise Critic (Q_z)**: Evaluates Q(s, z | human_context)
4. **Action Critic (Q_a)**: Evaluates Q(s, a | human_context) for flow-generated actions

## Installation

Make sure you have IsaacLab and the couple-flow-policy repository set up:

```bash
# IsaacLab should be installed at /home/weirdlab/Documents/summers/IsaacLab
# couple-flow-policy should be at /home/weirdlab/Documents/summers/IsaacLab/source/couple-flow-policy
```

## Usage

### Training

```bash
cd /home/weirdlab/Documents/summers/IsaacLab/scripts/environments/online_dsrl

# Basic training
python train_online_dsrl.py \
    --flow_checkpoint /path/to/flow_policy_checkpoint \
    --target_object sushi \
    --train_steps 100000

# With human context data
python train_online_dsrl.py \
    --flow_checkpoint /path/to/flow_policy_checkpoint \
    --human_context_file /path/to/human_demos.npz \
    --target_object apple \
    --train_steps 200000 \
    --use_wandb
```

### Evaluation

```bash
python eval_dsrl.py \
    --checkpoint ./checkpoints/online_dsrl/sushi_XXXXXXXX/last.pt \
    --flow_checkpoint /path/to/flow_policy_checkpoint \
    --num_episodes 50 \
    --target_object sushi
```

## Arguments

### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--task` | Isaac-PickPlace-Franka-custom | IsaacLab task name |
| `--num_envs` | 1 | Number of parallel environments |
| `--target_object` | sushi | Object to pick (sushi/apple/mug) |
| `--flow_checkpoint` | required | Path to pre-trained flow policy |
| `--human_context_file` | None | Path to human context data |
| `--train_steps` | 100000 | Total training steps |
| `--learning_starts` | 1000 | Steps before training starts |
| `--batch_size` | 256 | Training batch size |
| `--lr` | 3e-4 | Learning rate |
| `--noise_bound` | 2.0 | Bound for noise output |

### Human Context

The human context should be stored in a `.npz` file with key `human_context` of shape `(N, 1120)` where:
- N is the number of demonstrations
- 1120 = 280 timesteps × 4 action dimensions

Alternatively, provide raw human action data with key `gr1t2_absolute_actions`.

## File Structure

```
online_dsrl/
├── __init__.py
├── buffer.py              # Replay buffer for DSRL
├── sac_dsrl_agent.py     # SAC agent with noise/action critics
├── isaaclab_env.py       # IsaacLab environment wrapper
├── flow_policy_wrapper.py # Wrapper for pre-trained flow policy
├── train_online_dsrl.py  # Main training script
├── eval_dsrl.py          # Evaluation script
├── utils.py              # Utility functions
└── README.md
```

## Architecture Details

### Actor
- Input: obs (16D) + human_context (1120D)
- Output: noise (56D = 7 action_dim × 8 pred_horizon)
- Architecture: 3-layer MLP with LayerNorm and Mish activation

### Critics
- **Action Critic**: Q(obs, actions | human_context)
- **Noise Critic**: Q(obs, noise | human_context)
- Both: 2-network ensemble with 3-layer MLP

### Training Loop
1. Actor predicts noise z from (obs, human_context)
2. Flow policy generates actions from noise: a = flow(z | obs)
3. Execute action chunk in environment
4. Update action critic with TD-error
5. Update noise critic to match action critic: Q_z → Q_a
6. Update actor to maximize noise critic: π → argmax Q_z

## Integration with Offline DSRL

This online implementation complements the offline training at:
`/home/weirdlab/Documents/summers/IsaacLab/source/couple-flow-policy/scripts_train/train_dsrl_cached.py`

The offline version:
1. Pre-computes inverse flow: actions → noise
2. Trains noise_net to predict noise from human_context

The online version:
1. Uses the same flow policy
2. Learns noise prediction through RL interaction
3. Can incorporate reward shaping for the specific task

## Troubleshooting

### Common Issues

1. **IsaacLab import errors**: Make sure to run from the IsaacLab environment
2. **CUDA out of memory**: Reduce batch_size or num_envs
3. **Flow policy loading fails**: Check checkpoint path and format

### Debugging

```python
# Test environment
from isaaclab_env import make_isaaclab_env
env = make_isaaclab_env(num_envs=1, headless=False)
obs = env.reset()
print(f"Obs shape: {obs['state'].shape}")

# Test flow policy
from flow_policy_wrapper import load_flow_policy
flow = load_flow_policy("path/to/checkpoint")
noise = torch.randn(1, 8, 7)
actions = flow.sample(obs['state'], noise)
print(f"Actions shape: {actions.shape}")
```
