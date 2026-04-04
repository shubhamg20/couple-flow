# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym
import os

from . import (
    agents,
    # pickplace_gr1t2_env_cfg,
    # pickplace_gr1t2_waist_enabled_env_cfg,
    pickplace_franka_env_cfg,
    kitchen_franka_env_cfg,
    kitchen_gr1t2_env_cfg,
)

# gym.register(
#     id="Isaac-PickPlace-GR1T2-Abs-v0",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     kwargs={
#         "env_cfg_entry_point": pickplace_gr1t2_env_cfg.PickPlaceGR1T2EnvCfg,
#         "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bc_rnn_low_dim.json"),
#     },
#     disable_env_checker=True,
# )

# gym.register(
#     id="Isaac-PickPlace-GR1T2-WaistEnabled-Abs-v0",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     kwargs={
#         "env_cfg_entry_point": pickplace_gr1t2_waist_enabled_env_cfg.PickPlaceGR1T2WaistEnabledEnvCfg,
#         "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bc_rnn_low_dim.json"),
#     },
#     disable_env_checker=True,
# )

gym.register(
    id="Isaac-PickPlace-Franka-custom",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": pickplace_franka_env_cfg.PickPlaceFrankaEnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Kitchen-GR1T2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": kitchen_gr1t2_env_cfg.KitchenGR1T2EnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Kitchen-Franka-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": kitchen_franka_env_cfg.KitchenFrankaEnvCfg,
    },
    disable_env_checker=True,
)