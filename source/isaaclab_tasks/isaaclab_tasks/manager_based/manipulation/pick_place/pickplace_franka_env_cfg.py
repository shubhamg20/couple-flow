# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import tempfile
import torch

import carb
from scipy.spatial.transform import Rotation as R

import isaaclab.controllers.utils as ControllerUtils
import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg, JointPositionActionCfg, BinaryJointPositionActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.devices.device_base import DevicesCfg
from isaaclab.devices.openxr import ManusViveCfg, OpenXRDeviceCfg, XrCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.devices import OpenXRDevice, OpenXRDeviceCfg
from isaaclab.devices.openxr.retargeters import Se3RelRetargeter, GripperRetargeter
from isaaclab.devices.openxr.retargeters import Se3RelRetargeterCfg, GripperRetargeterCfg
from . import mdp

from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG  # isort: skip

############################## FOR MUG CONVEX DECOMPOSITION #####################################
import omni.usd
from pxr import UsdPhysics

def add_collision(usd_path_load, usd_path_save):
    # Use omni.usd instead of get_context()
    usd_context = omni.usd.get_context()
    stage = usd_context.open_stage(usd_path_load)
    stage = usd_context.get_stage()
    rigidPrim = stage.GetPrimAtPath("/World/model_normalized/mesh")
    collisionAPI = UsdPhysics.CollisionAPI.Apply(rigidPrim)
    collisionAPI.GetPhysicsApproximationAttr().Set(UsdPhysics.Tokens.convexDecomposition)
    stage.Export(usd_path_save)
#################################################################################################

##
# Scene definition
##
@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):

    # tiled_camera: TiledCameraCfg = TiledCameraCfg(
    #     prim_path="/World/envs/env_.*/Camera2",
    #     offset=TiledCameraCfg.OffsetCfg(
    #         pos=(0.0, 1.4, 2.0), 
    #         rot=(.65, .27, .27, -.65),
    #         convention="world"
    #     ),
    #     update_latest_camera_pose=True,
    #     data_types=["rgb"],
    #     spawn=sim_utils.PinholeCameraCfg(
    #         focal_length=35.0, 
    #         focus_distance=400.0, 
    #         horizontal_aperture=34.32365, 
    #         clipping_range=(0.01, 10000000.0)
    #     ),
    #     width=256,
    #     height=256,
    # )
    
    packing_table = AssetBaseCfg(                              #takes huge memory
        prim_path="/World/envs/env_.*/PackingTable2",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.55, 0.0], rot=[1.0, 0.0, 0.0, 0.0]),
        spawn=UsdFileCfg(
            usd_path=f"/workspace/isaaclab/source/gr1t2/exhaust_pipe_task/exhaust_pipe_assets/table.usd",
            scale=(1.0, 1.0, 1.315),
            # usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/PackingTable/packing_table.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=False,  # Ensure object is dynamic, not kinematic
                disable_gravity=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
            ),
        ),
    )
    # table = AssetBaseCfg(
    #     prim_path="/World/envs/env_.*/Table",
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.55, 1.0], rot=[1.0, 0.0, 0.0, 0.0]),
    #     spawn=UsdFileCfg(
    #         usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
    #         scale=(1.5, 1.0, 1.0),
    #     )
    # )

    # tray = AssetBaseCfg(
    #     prim_path="/World/envs/env_.*/Tray",
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=[0.41, 0.42, 1], rot=[0.707, 0.707, 0.0, 0.0]),
    #     spawn=UsdFileCfg(
    #         usd_path=f"/workspace/isaaclab/usd_extracted/pink_plate_csm/pink_plate.usd",
    #         scale=(0.15, 0.15, 0.15),
    #         # rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #         #     rigid_body_enabled=True,
    #         #     kinematic_enabled=False,  # Ensure object is dynamic, not kinematic
    #         #     disable_gravity=False,
    #         #     solver_position_iteration_count=16,
    #         #     solver_velocity_iteration_count=1,
    #         #     max_angular_velocity=0.0,
    #         #     max_linear_velocity=0.0,
    #         #     max_depenetration_velocity=5.0,
    #         # ),
    #     ),
    # )

    sushi = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Sushi",
        # init_state=RigidObjectCfg.InitialStateCfg(pos=[-0.1, 0.36, 1.02], rot=[-.028, -.486, -.867, .102]),
        init_state=RigidObjectCfg.InitialStateCfg(pos=[-0.1, 0.5, 1.02], rot=[-.028, -.486, -.867, .102]),
        spawn=UsdFileCfg(
            # Use the physics-enabled sushi USD created by to_usd.py
            usd_path="/workspace/isaaclab/usd_extracted/sushi_csm/sushi2.usd",
            # usd_path="/workspace/isaaclab/source/gr1t2/YCB/Axis_Aligned/011_sushi.usd",
            scale=(.08, .08, .08),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=False,  # Ensure object is dynamic, not kinematic
                disable_gravity=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
            ),
            # Note: Mass is already defined in the USD file (0.12kg), but we can override it
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),  # Use realistic sushi mass
        ),
    )

    apple = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Apple",
        # init_state=RigidObjectCfg.InitialStateCfg(pos=[0.00, 0.5, 1], rot=[.707, .707, 0, 0]),
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.00, 0.36, 1], rot=[.707, .707, 0, 0]),
        spawn=UsdFileCfg(
            # Use the physics-enabled apple USD
            usd_path="/workspace/isaaclab/usd_extracted/apple_csm/apple2.usd",
            # usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            scale=(.06, .06, .06),
            # scale=(.8, .8, .8),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=False,  # Ensure object is dynamic, not kinematic
                disable_gravity=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
            ),
            # Apple mass (typical apple is around 0.18kg)
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
        ),
    )

    mug = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Mug",
        # init_state=RigidObjectCfg.InitialStateCfg(pos=[0.07, 0.35, 1.0], rot=[0, 0, 0, 1]),
        # init_state=RigidObjectCfg.InitialStateCfg(pos=[.1, 0.37, 1.0], rot=[.0, 0.0, -.707, -.707]),
        init_state=RigidObjectCfg.InitialStateCfg(pos=[.1, 0.5, 1.0], rot=[.0, 0.0, -.707, -.707]),
        spawn=UsdFileCfg(
            # Use the physics-enabled mug USD
            # usd_path="/workspace/isaaclab/source/gr1t2/Mugs/SM_Mug_C1.usd",
            usd_path="/workspace/isaaclab/usd_extracted/mug_csm/mug2.usd",
            # scale=(.009, .009, .009),
            scale=(.1, .1, .1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=False,  # Ensure object is dynamic, not kinematic
                disable_gravity=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
            ),
            ############################# FOR MUG CONVEX DECOMPOSITION #####################################
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.02,
                rest_offset=0.01,
            ),
            ################################################################################################

            # Mug mass (typical mug is around 0.25kg)
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
        ),
    )
    
    # Franka robot configured for pick-place manipulation tasks
    robot: ArticulationCfg = FRANKA_PANDA_HIGH_PD_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            # Convert Euler degrees (0, 0, 0) to quaternion [w, x, y, z]
            # rot=R.from_euler("xyz", [90, 0, 0], degrees=True).as_quat()[[3, 0, 1, 2]].tolist(),
            joint_pos={
                "panda_joint1": -.0,
                "panda_joint2": -.7,
                "panda_joint3": 0.0,
                "panda_joint4": -2.4,  # This is within the valid range [-3.072, -0.070]
                "panda_joint5": 0.0,     
                "panda_joint6": 1.7,
                "panda_joint7": 1.57/2,
                "panda_finger_joint.*": 0.04,
            },   
            pos=(0, 0, .7),
            rot=(0.707, 0.0, 0.0, 0.707),
            # rot=(1.0, 0.0, 0.0, 0.0),
            # joint_vel={".*": 0.0},
        ),
    )

    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=GroundPlaneCfg(),
    )

    # Lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# MDP settings
##
@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # Franka arm action: differential inverse kinematics with relative mode
    arm_action = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        body_name="panda_hand",
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
        scale=0.5,
        # body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.0]),
    )
    
    # Franka gripper action: binary control
    gripper_action = BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_finger.*"],
        open_command_expr={"panda_finger_.*": 0.04},
        close_command_expr={"panda_finger_.*": 0.0},
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values."""

        actions = ObsTerm(func=mdp.last_action)
        robot_joint_pos = ObsTerm(
            func=base_mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        robot_root_pos = ObsTerm(func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("robot")})
        robot_root_rot = ObsTerm(func=base_mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("robot")})
        # sushi_pos = ObsTerm(func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("sushi")})
        # sushi_rot = ObsTerm(func=base_mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("sushi")})
        apple_pos = ObsTerm(func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("apple")})
        apple_rot = ObsTerm(func=base_mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("apple")})
        # mug_pos = ObsTerm(func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("mug")})
        # mug_rot = ObsTerm(func=base_mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("mug")})

        # Franka end-effector observations
        eef_pos = ObsTerm(func=mdp.get_franka_eef_pos)
        eef_quat = ObsTerm(func=mdp.get_franka_eef_quat)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # sushi_dropping = DoneTerm(
    #     func=mdp.root_height_below_minimum, params={"minimum_height": 0.5, "asset_cfg": SceneEntityCfg("sushi")}
    # )

    # apple_dropping = DoneTerm(
    #     func=mdp.root_height_below_minimum, params={"minimum_height": 0.5, "asset_cfg": SceneEntityCfg("apple")}
    # )

    # mug_dropping = DoneTerm(
    #     func=mdp.root_height_below_minimum, params={"minimum_height": 0.5, "asset_cfg": SceneEntityCfg("mug")}
    # )

    # sushi_on_tray = DoneTerm(
    #     func=mdp.object_touching_tray,
    #     params={
    #         "object_cfg": SceneEntityCfg("sushi"),
    #         # "tray_cfg": SceneEntityCfg("tray"),
    #         "max_distance_x": 0.10,
    #         "max_distance_y": 0.10,
    #         "max_distance_z": 0.05,
    #         "min_distance_z": -0.02,
    #         "max_velocity": 1.0,
    #     },
    # )

    apple_on_tray = DoneTerm(
        func=mdp.object_touching_tray,
        params={
            "object_cfg": SceneEntityCfg("apple"),
            # "tray_cfg": SceneEntityCfg("tray"),
            "max_distance_x": 0.10,
            "max_distance_y": 0.10,
            "max_distance_z": 0.05,
            "min_distance_z": -0.02,
            "max_velocity": 1.0,
        },
    )

    # mug_on_tray = DoneTerm(
    #     func=mdp.object_touching_tray,
    #     params={
    #         "object_cfg": SceneEntityCfg("mug"),
    #         # "tray_cfg": SceneEntityCfg("tray"),
    #         "max_distance_x": 0.10,
    #         "max_distance_y": 0.10,
    #         "max_distance_z": 0.05,
    #         "min_distance_z": -0.02,
    #         "max_velocity": 1.0,
    #     },
    # )

@configclass
class EventCfg:
    """Configuration for events."""

    # Reset joint 1 to zero position
    # reset_joint1 = EventTerm(
    #     func=mdp.reset_joints_by_scale,
    #     mode="reset",
    #     params={
    #         "position_range": (0.0, 0.0),  # Set joint 1 to zero
    #         "velocity_range": (0.0, 0.0),  # Zero velocities
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint1"]),
    #     },
    # )

    # # Reset other joints to their configured initial positions
    # reset_other_joints = EventTerm(
    #     func=mdp.reset_joints_by_scale,
    #     mode="reset",
    #     params={
    #         "position_range": (1, 1),  # Keep exact initial positions
    #         "velocity_range": (0.0, 0.0),  # Zero velocities
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint[2-7]", "panda_finger_joint.*"]),
    #     },
    # )

    reset_all_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),  # Keep exact initial positions
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*"]),
        },
    )
    reset_gripper = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),  # Keep exact initial positions
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot", joint_names=["panda_finger_joint.*"]),
        },
    )

    # reset_objects = EventTerm(func=mdp.swap_objects, mode="reset",
    #                           params={
    #                               "pose_ranges": {
    #                                   "x": [-0.03, 0.03],
    #                                   "y": [-0.04, 0.01],
    #                               },
    #                               "asset_cfgs": [
    #                                   SceneEntityCfg("sushi"),
    #                                   SceneEntityCfg("apple"),
    #                                   SceneEntityCfg("mug"),
    #                               ]
    #                           })
    # # reset_sushi = EventTerm(
    #     func=mdp.reset_root_state_uniform,
    #     mode="reset",
    #     params={
    #         "pose_range": {
    #             "x": [-0.03, 0.03],
    #             "y": [-0.03, 0.03],
    #         },
    #         "velocity_range": {},
    #         "asset_cfg": SceneEntityCfg("sushi"),
    #     },
    # )

    # reset_apple = EventTerm(
    #     func=mdp.reset_root_state_uniform,
    #     mode="reset",
    #     params={
    #         "pose_range": {
    #             "x": [-0.03, 0.03],
    #             "y": [-0.03, 0.03],
    #         },
    #         "velocity_range": {},
    #         "asset_cfg": SceneEntityCfg("apple"),
    #     },
    # )

    # reset_mug = EventTerm(
    #     func=mdp.reset_root_state_uniform,
    #     mode="reset",
    #     params={
    #         "pose_range": {
    #             "x": [-0.03, 0.03],
    #             "y": [-0.03, 0.03],
    #         },
    #         "velocity_range": {},
    #         "asset_cfg": SceneEntityCfg("mug"),
    #     },
    # )


##
# Environment configuration
##
@configclass
class PickPlaceFrankaEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the Franka pick-place environment."""

    # Scene settings
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # MDP settings
    terminations: TerminationsCfg = TerminationsCfg()
    events = EventCfg()

    # Unused managers
    commands = None
    rewards = None
    curriculum = None

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 5.0
        # simulation settings
        self.sim.dt = 1.0 / 120.0
        self.sim.substeps = 1
        self.sim.render_mode = "partial_rendering"
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        # if self.scene.tiled_camera is not None:
        #     self.scene.tiled_camera.update_period = self.decimation * self.sim.dt
        
        # Add teleoperation devices for hand tracking support
        self.teleop_devices = DevicesCfg(
            devices={
                "handtracking": OpenXRDeviceCfg(
                    retargeters=[
                        Se3RelRetargeterCfg(
                            bound_hand=OpenXRDevice.TrackingTarget.HAND_RIGHT,
                            zero_out_xy_rotation=True,
                            use_wrist_rotation=False,
                            use_wrist_position=True,
                            delta_pos_scale_factor=10.0,
                            delta_rot_scale_factor=10.0,
                            sim_device=self.sim.device,
                        ),
                        GripperRetargeterCfg(
                            bound_hand=OpenXRDevice.TrackingTarget.HAND_RIGHT, 
                            sim_device=self.sim.device
                        ),
                    ],
                    sim_device=self.sim.device,
                    xr_cfg=self.xr,
                ),
            }
        )


