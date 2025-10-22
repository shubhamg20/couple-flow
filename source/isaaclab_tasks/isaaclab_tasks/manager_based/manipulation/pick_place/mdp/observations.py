# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause



from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_obs(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """
    Object observations (in world frame):
        sushi pos,
        sushi quat,
        apple pos,
        apple quat,
        mug pos,
        mug quat,
        left_eef to apple,
        right_eef_to apple,
        left_eef to mug,
        right_eef_to mug,
    """

    body_pos_w = env.scene["robot"].data.body_pos_w
    left_eef_idx = env.scene["robot"].data.body_names.index("left_hand_roll_link")
    right_eef_idx = env.scene["robot"].data.body_names.index("right_hand_roll_link")
    left_eef_pos = body_pos_w[:, left_eef_idx] - env.scene.env_origins
    right_eef_pos = body_pos_w[:, right_eef_idx] - env.scene.env_origins

    sushi_pos = env.scene["sushi"].data.root_pos_w - env.scene.env_origins
    sushi_quat = env.scene["sushi"].data.root_quat_w
    apple_pos = env.scene["apple"].data.root_pos_w - env.scene.env_origins
    apple_quat = env.scene["apple"].data.root_quat_w
    mug_pos = env.scene["mug"].data.root_pos_w - env.scene.env_origins
    mug_quat = env.scene["mug"].data.root_quat_w

    left_eef_to_apple = apple_pos - left_eef_pos
    right_eef_to_apple = apple_pos - right_eef_pos
    left_eef_to_mug = mug_pos - left_eef_pos
    right_eef_to_mug = mug_pos - right_eef_pos

    return torch.cat(
        (      
            sushi_pos,
            sushi_quat,
            apple_pos,
            apple_quat,
            mug_pos,
            mug_quat,
            left_eef_to_apple,
            right_eef_to_apple,
            left_eef_to_mug,
            right_eef_to_mug,
        ),
        dim=1,
    )


def get_left_eef_pos(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    body_pos_w = env.scene["robot"].data.body_pos_w
    left_eef_idx = env.scene["robot"].data.body_names.index("left_hand_roll_link")
    left_eef_pos = body_pos_w[:, left_eef_idx] - env.scene.env_origins

    return left_eef_pos


def get_left_eef_quat(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    body_quat_w = env.scene["robot"].data.body_quat_w
    left_eef_idx = env.scene["robot"].data.body_names.index("left_hand_roll_link")
    left_eef_quat = body_quat_w[:, left_eef_idx]

    return left_eef_quat


def get_right_eef_pos(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    body_pos_w = env.scene["robot"].data.body_pos_w
    right_eef_idx = env.scene["robot"].data.body_names.index("right_hand_roll_link")
    right_eef_pos = body_pos_w[:, right_eef_idx]

    return right_eef_pos[0]


def get_right_eef_quat(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    body_quat_w = env.scene["robot"].data.body_quat_w
    right_eef_idx = env.scene["robot"].data.body_names.index("right_hand_roll_link")
    right_eef_quat = body_quat_w[:, right_eef_idx]

    return right_eef_quat


def get_hand_state(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    hand_joint_states = env.scene["robot"].data.joint_pos[:, -22:]  # Hand joints are last 22 entries of joint state

    return hand_joint_states


def get_head_state(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    robot_joint_names = env.scene["robot"].data.joint_names
    head_joint_names = ["head_pitch_joint", "head_roll_joint", "head_yaw_joint"]
    indexes = torch.tensor([robot_joint_names.index(name) for name in head_joint_names], dtype=torch.long)
    head_joint_states = env.scene["robot"].data.joint_pos[:, indexes]

    return head_joint_states


def get_all_robot_link_state(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    body_pos_w = env.scene["robot"].data.body_link_state_w[:, :, :]
    all_robot_link_pos = body_pos_w

    return all_robot_link_pos


def get_franka_eef_pos(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """Get Franka end-effector position."""
    body_pos_w = env.scene["robot"].data.body_pos_w
    eef_idx = env.scene["robot"].data.body_names.index("panda_hand")
    eef_pos = body_pos_w[:, eef_idx] - env.scene.env_origins

    return eef_pos


def get_franka_eef_quat(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """Get Franka end-effector quaternion."""
    body_quat_w = env.scene["robot"].data.body_quat_w
    eef_idx = env.scene["robot"].data.body_names.index("panda_hand")
    eef_quat = body_quat_w[:, eef_idx]

    return eef_quat
