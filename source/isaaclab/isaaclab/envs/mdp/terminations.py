# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers.command_manager import CommandTerm

"""
MDP terminations.
"""


def time_out(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Terminate the episode when the episode length exceeds the maximum episode length."""
    return env.episode_length_buf >= env.max_episode_length


def command_resample(env: ManagerBasedRLEnv, command_name: str, num_resamples: int = 1) -> torch.Tensor:
    """Terminate the episode based on the total number of times commands have been re-sampled.

    This makes the maximum episode length fluid in nature as it depends on how the commands are
    sampled. It is useful in situations where delayed rewards are used :cite:`rudin2022advanced`.
    """
    command: CommandTerm = env.command_manager.get_term(command_name)
    return torch.logical_and((command.time_left <= env.step_dt), (command.command_counter == num_resamples))


"""
Root terminations.
"""


def bad_orientation(
    env: ManagerBasedRLEnv, limit_angle: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the asset's orientation is too far from the desired orientation limits.

    This is computed by checking the angle between the projected gravity vector and the z-axis.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.acos(-asset.data.projected_gravity_b[:, 2]).abs() > limit_angle


def root_height_below_minimum(
    env: ManagerBasedRLEnv, minimum_height: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the asset's root height is below the minimum height.

    Note:
        This is currently only supported for flat terrains, i.e. the minimum height is in the world frame.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 2] < minimum_height


"""
Joint terminations.
"""


def joint_pos_out_of_limit(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Terminate when the asset's joint positions are outside of the soft joint limits."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if asset_cfg.joint_ids is None:
        asset_cfg.joint_ids = slice(None)

    limits = asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids]
    out_of_upper_limits = torch.any(asset.data.joint_pos[:, asset_cfg.joint_ids] > limits[..., 1], dim=1)
    out_of_lower_limits = torch.any(asset.data.joint_pos[:, asset_cfg.joint_ids] < limits[..., 0], dim=1)
    return torch.logical_or(out_of_upper_limits, out_of_lower_limits)


def joint_pos_out_of_manual_limit(
    env: ManagerBasedRLEnv, bounds: tuple[float, float], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the asset's joint positions are outside of the configured bounds.

    Note:
        This function is similar to :func:`joint_pos_out_of_limit` but allows the user to specify the bounds manually.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if asset_cfg.joint_ids is None:
        asset_cfg.joint_ids = slice(None)
    # compute any violations
    out_of_upper_limits = torch.any(asset.data.joint_pos[:, asset_cfg.joint_ids] > bounds[1], dim=1)
    out_of_lower_limits = torch.any(asset.data.joint_pos[:, asset_cfg.joint_ids] < bounds[0], dim=1)
    return torch.logical_or(out_of_upper_limits, out_of_lower_limits)


def joint_vel_out_of_limit(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Terminate when the asset's joint velocities are outside of the soft joint limits."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute any violations
    limits = asset.data.soft_joint_vel_limits
    return torch.any(torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids]) > limits[:, asset_cfg.joint_ids], dim=1)


def joint_vel_out_of_manual_limit(
    env: ManagerBasedRLEnv, max_velocity: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the asset's joint velocities are outside the provided limits."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute any violations
    return torch.any(torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids]) > max_velocity, dim=1)


def joint_effort_out_of_limit(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when effort applied on the asset's joints are outside of the soft joint limits.

    In the actuators, the applied torque are the efforts applied on the joints. These are computed by clipping
    the computed torques to the joint limits. Hence, we check if the computed torques are equal to the applied
    torques. If they are not, it means that clipping has occurred.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # check if any joint effort is out of limit
    out_of_limits = ~torch.isclose(
        asset.data.computed_torque[:, asset_cfg.joint_ids], asset.data.applied_torque[:, asset_cfg.joint_ids]
    )
    return torch.any(out_of_limits, dim=1)


"""
Contact sensor.
"""


def illegal_contact(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Terminate when the contact force on the sensor exceeds the force threshold."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    # check if any contact force exceeds the threshold
    return torch.any(
        torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold, dim=1
    )


def object_touching_tray(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("sushi"),
    tray_cfg: SceneEntityCfg = SceneEntityCfg("tray"),
    max_distance_x: float = 0.10,
    max_distance_y: float = 0.10,
    max_distance_z: float = 0.05,
    min_distance_z: float = -0.02,
    max_velocity: float = 0.10,
) -> torch.Tensor:
    """Determine if an object is touching the tray.
    
    This function checks whether an object has dropped onto and is touching the tray:
    1. Object is within horizontal distance threshold (x, y) from tray center
    2. Object is within vertical distance threshold (z) from tray surface
    3. Object velocity is below threshold (indicating it has settled)
    
    Args:
        env: The RL environment instance.
        object_cfg: Configuration for the object entity (sushi, apple, or mug).
        tray_cfg: Configuration for the tray entity.
        max_distance_x: Maximum horizontal distance in x direction for touching (default: 0.10m).
        max_distance_y: Maximum horizontal distance in y direction for touching (default: 0.10m).
        max_distance_z: Maximum vertical distance above tray surface for touching (default: 0.05m).
        min_distance_z: Minimum vertical distance (allows slight penetration, default: -0.02m).
        max_velocity: Maximum velocity magnitude for object to be considered settled (default: 0.10 m/s).
    
    Returns:
        Boolean tensor indicating which environments have the object touching the tray.
    """
    # Get object entity from the scene (RigidObject)
    object: RigidObject = env.scene[object_cfg.name]
    
    # Get tray from extras (XFormPrim, not RigidObject)
    tray = env.scene.extras[tray_cfg.name]
    
    # Get positions relative to environment origin
    object_pos = object.data.root_pos_w - env.scene.env_origins
    
    # Get tray position using get_world_poses for XFormPrim
    env_ids = torch.arange(env.scene.num_envs, device=env.device)
    tray_pos_w, _ = tray.get_world_poses(env_ids)
    tray_pos = tray_pos_w - env.scene.env_origins
    
    # Compute relative positions (CENTER TO CENTER comparison)
    # Note: This compares object center to tray center, not object bottom to tray top
    # For small objects, the center-to-center z distance should be close to zero when touching
    object_to_tray_x = torch.abs(object_pos[:, 0] - tray_pos[:, 0])
    object_to_tray_y = torch.abs(object_pos[:, 1] - tray_pos[:, 1])
    object_to_tray_z = object_pos[:, 2] - tray_pos[:, 2]
    
    # Get object velocity magnitude
    object_vel = torch.norm(object.data.root_vel_w, dim=1)
    
    # Check all conditions for touching
    # 1. Object is within horizontal distance from tray center
    done = object_to_tray_x < max_distance_x
    done = torch.logical_and(done, object_to_tray_y < max_distance_y)
    
    # 2. Object center is within vertical distance from tray center (center-to-center)
    # For touching, object center should be very close to tray center in z (accounting for object height)
    done = torch.logical_and(done, object_to_tray_z < max_distance_z)
    done = torch.logical_and(done, object_to_tray_z > min_distance_z)
    
    # 3. Object velocity is low (has settled)
    done = torch.logical_and(done, object_vel < max_velocity)
    return done
