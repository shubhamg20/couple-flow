# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
'''                                                           
                                                                                         
  Get microwave out of docker on this machine:                                           
  docker cp isaac-lab-base:/workspace/isaaclab/usd_extracted/microwave_csm/.
  /home/weirdlab/Documents/summers/IsaacLab/usd_extracted/microwave_csm                  
                  
  Files to copy to the new machine:

  ┌─────────┬────────────────────────────────────────────────────────────────────────┐
  │  File   │                          Path on this machine                          │
  ├─────────┼────────────────────────────────────────────────────────────────────────┤
  │ Kitchen │ /home/weirdlab/Documents/summers/IsaacLab/source/kitchen_assets/kitche │
  │         │ n_background.usd                                                       │
  ├─────────┼────────────────────────────────────────────────────────────────────────┤
  │ Bowl    │ /home/weirdlab/Documents/summers/IsaacLab/usd_extracted/bowl_csm/white │
  │         │ _bowl.usd                                                              │
  ├─────────┼────────────────────────────────────────────────────────────────────────┤
  │ Microwa │ /home/weirdlab/Documents/summers/IsaacLab/usd_extracted/microwave_csm/ │
  │ ve      │ Microwave039.usd                                                       │
  └─────────┴────────────────────────────────────────────────────────────────────────┘

  Where they need to end up inside docker on the new machine:

  ┌───────────┬──────────────────────────────────────────────────────────────────┐
  │   File    │                           Docker path                            │
  ├───────────┼──────────────────────────────────────────────────────────────────┤
  │ Kitchen   │ /workspace/isaaclab/source/kitchen_assets/kitchen_background.usd │
  ├───────────┼──────────────────────────────────────────────────────────────────┤
  │ Bowl      │ /workspace/isaaclab/usd_extracted/bowl_csm/white_bowl.usd        │
  ├───────────┼──────────────────────────────────────────────────────────────────┤
  │ Microwave │ /workspace/isaaclab/usd_extracted/microwave_csm/Microwave039.usd │
  └───────────┴──────────────────────────────────────────────────────────────────┘

  Kitchen is auto-available via the source/ bind mount. Bowl and microwave need to go
  into whatever host path is bind-mounted to usd_extracted in docker-compose (on this
  machine that's /home/shubham/summer/usd_extracted).

'''
import tempfile
import torch

import carb
from pink.tasks import DampingTask, FrameTask

import isaaclab.controllers.utils as ControllerUtils
import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers.pink_ik import NullSpacePostureTask, PinkIKControllerCfg
from isaaclab.devices.device_base import DevicesCfg
from isaaclab.devices.openxr import ManusViveCfg, OpenXRDeviceCfg, XrCfg
from isaaclab.devices.openxr.retargeters.humanoid.fourier.gr1t2_retargeter import GR1T2RetargeterCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions.pink_actions_cfg import PinkInverseKinematicsActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

from . import mdp

from isaaclab_assets.robots.fourier import GR1T2_HIGH_PD_CFG  # isort: skip


##
# Scene definition
##
@configclass
class KitchenTableSceneCfg(InteractiveSceneCfg):

    # Camera rotation axes (quaternion w,x,y,z convention="world"):
    #   Roll (rotate image CW/CCW):  local X axis. CW = negative X rotation.
    #   Pan right/left:              local Z axis. Right = negative Z rotation (CW).
    #   Pan down/up (tilt):          local Y axis. Down = positive Y rotation (CCW).
    # To apply a rotation, use scipy:
    #   r_current = R.from_quat([x, y, z, w])  # scipy uses xyzw
    #   r_rot = R.from_euler('<axis>', <degrees>, degrees=True)
    #   r_result = r_current * r_rot  # local frame rotation
    #   Then convert back to (w, x, y, z) for IsaacLab.
    # Base orientation was (1,1,1,1) normalized = (0.5, 0.5, 0.5, 0.5),
    # then rolled CW 90° (local X -90°) -> (0.7071, 0, 0, 0.7071),
    # then panned right 20° (local Z -20°) -> (0.8192, 0, 0, 0.5736),
    # then panned down 10° (local Y +10°) -> current value.
    # tiled_camera: TiledCameraCfg = TiledCameraCfg(
    #     prim_path="/World/envs/env_.*/Camera2",
    #     offset=TiledCameraCfg.OffsetCfg(
    #         pos=(-0.4, -0.40, 1.4),
    #         rot=(0.8160, -0.0500, 0.0714, 0.5714),
    #         convention="world"
    #     ),
    #     update_latest_camera_pose=True,
    #     data_types=["rgb"],
    #     spawn=sim_utils.PinholeCameraCfg(
    #         focal_length=30.0,
    #         focus_distance=400.0,
    #         horizontal_aperture=34.32365,
    #         clipping_range=(0.01, 10000000.0)
    #     ),
    #     width=256,
    #     height=256,
    # )
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Camera1",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0, 1.4, 2.0), 
            rot=(.65, .27, .27, -.65),
            convention="world"
        ),
        update_latest_camera_pose=True,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=35.0, 
            focus_distance=400.0, 
            horizontal_aperture=34.32365, 
            clipping_range=(0.01, 10000000.0)
        ),
        width=256,
        height=256,
    )
    eval_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/EvalOnlyCamera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.15, 0.20, 2.2),
            rot=(0.482, -0.544, 0.445, 0.523),
            convention="world"
        ),
        update_latest_camera_pose=True,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=40.0,
            focus_distance=400.0,
            horizontal_aperture=34.32365,
            clipping_range=(0.01, 10000000.0)
        ),
        width=1024,
        height=1024,
    )

    # Kitchen background scene from IsaacLab Arena
    kitchen = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Kitchen",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(-2.6, 0.75, 0.1),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        spawn=UsdFileCfg(
            usd_path="/workspace/isaaclab/source/kitchen_assets/kitchen_background.usd",
        ),
    )

    bowl = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Bowl",
        # init_state=RigidObjectCfg.InitialStateCfg(pos=[-0.15, 0.35, 1.05], rot=[0, 0, 0, 1]),
        init_state=RigidObjectCfg.InitialStateCfg(pos=[-0.15, 0.35, 1.027], rot=[0, 0, 0, 1]),
        spawn=UsdFileCfg(
            usd_path="/workspace/isaaclab/usd_extracted/bowl_csm/white_bowl.usd",
            scale=(1.0, 1.0, 1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=False,
                disable_gravity=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.00,
                rest_offset=0.01,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
        ),
    )

    # microwave = ArticulationCfg(
    #     prim_path="/World/envs/env_.*/Microwave",
    #     init_state=ArticulationCfg.InitialStateCfg(pos=[0.05, 0.6, 1.05], rot=[1, 0, 0, 0]),
    #     spawn=UsdFileCfg(
    #         usd_path="/workspace/isaaclab/usd_extracted/microwave_csm/Microwave039.usd",
    #         scale=(0.7, 0.75, 0.7),
    #     ),
    #     actuators={},
    # )

    ball = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Ball",
        # init_state=RigidObjectCfg.InitialStateCfg(pos=[0.00, 0.5, 1.16], rot=[1, 0, 0, 0]),
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.00, 0.5, 1.037], rot=[1, 0, 0, 0]),
        spawn=sim_utils.SphereCfg(
            radius=0.01,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.2, 0.2)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=False,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
            ),
        ),
    )
    mug = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Mug",
        # init_state=RigidObjectCfg.InitialStateCfg(pos=[0.00, 0.5, 1.15], rot=[0.5, 0.5, 0.5, 0.5]),
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.00, 0.5, 1.027], rot=[0.5, 0.5, 0.5, 0.5]),
        spawn=UsdFileCfg(
            # Use the physics-enabled mug USD
            # usd_path="/workspace/isaaclab/source/gr1t2/Mugs/SM_Mug_C1.usd",
            usd_path="/workspace/isaaclab/usd_extracted/mug_csm/mug.usd",
            # scale=(.009, .009, .009),
            scale=(.08, .09, .08),
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
                contact_offset=0.00,
                rest_offset=0.01,
            ),
            ################################################################################################

            # Mug mass (typical mug is around 0.25kg)
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
        ),
    )

    rack = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Rack",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.2, 0.35, 1.15], rot=[0, 0, 0, 1]),
        spawn=UsdFileCfg(
            # Use the physics-enabled rack USD
            usd_path="/workspace/isaaclab/usd_extracted/rack_csm/rack.usd",
        ),
    )

    # Humanoid robot configured for pick-place manipulation tasks
    robot: ArticulationCfg = GR1T2_HIGH_PD_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(-0.1, 0.1, 1.09),
            rot=(0.7071, 0, 0, 0.7071),
            joint_pos={
                # right-arm
                "right_shoulder_pitch_joint": 0.0,
                "right_shoulder_roll_joint": 0.0,
                "right_shoulder_yaw_joint": 0.0,
                "right_elbow_pitch_joint": -1.5708,
                "right_wrist_yaw_joint": 0.0,
                "right_wrist_roll_joint": 0.0,
                "right_wrist_pitch_joint": 0.0,
                # left-arm
                "left_shoulder_pitch_joint": 0.0,
                "left_shoulder_roll_joint": 0.0,
                "left_shoulder_yaw_joint": 0.0,
                "left_elbow_pitch_joint": -1.5708,
                "left_wrist_yaw_joint": 0.0,
                "left_wrist_roll_joint": 0.0,
                "left_wrist_pitch_joint": 0.0,
                # --
                "head_.*": 0.0,
                "waist_.*": 0.0,
                ".*_hip_.*": 0.0,
                ".*_knee_.*": 0.0,
                ".*_ankle_.*": 0.0,
                "R_.*": 0.0,
                "L_.*": 0.0,
            },
            joint_vel={".*": 0.0},
        ),
    )

    # Box under robot
    robot_box = AssetBaseCfg(
        prim_path="/World/envs/env_.*/RobotBox",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0, -0.2, 0.1), rot=(1, 0, 0, 0)),
        spawn=sim_utils.CuboidCfg(
            size=(0.5, 0.5, 0.16),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5)),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
            ),
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

    pink_ik_cfg = PinkInverseKinematicsActionCfg(
        pink_controlled_joint_names=[
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_pitch_joint",
            "left_wrist_yaw_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_pitch_joint",
            "right_wrist_yaw_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
        ],
        # Joints to be locked in URDF
        ik_urdf_fixed_joint_names=[
            "left_hip_roll_joint",
            "right_hip_roll_joint",
            "left_hip_yaw_joint",
            "right_hip_yaw_joint",
            "left_hip_pitch_joint",
            "right_hip_pitch_joint",
            "left_knee_pitch_joint",
            "right_knee_pitch_joint",
            "left_ankle_pitch_joint",
            "right_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "right_ankle_roll_joint",
            "L_index_proximal_joint",
            "L_middle_proximal_joint",
            "L_pinky_proximal_joint",
            "L_ring_proximal_joint",
            "L_thumb_proximal_yaw_joint",
            "R_index_proximal_joint",
            "R_middle_proximal_joint",
            "R_pinky_proximal_joint",
            "R_ring_proximal_joint",
            "R_thumb_proximal_yaw_joint",
            "L_index_intermediate_joint",
            "L_middle_intermediate_joint",
            "L_pinky_intermediate_joint",
            "L_ring_intermediate_joint",
            "L_thumb_proximal_pitch_joint",
            "R_index_intermediate_joint",
            "R_middle_intermediate_joint",
            "R_pinky_intermediate_joint",
            "R_ring_intermediate_joint",
            "R_thumb_proximal_pitch_joint",
            "L_thumb_distal_joint",
            "R_thumb_distal_joint",
            "head_roll_joint",
            "head_pitch_joint",
            "head_yaw_joint",
            "waist_yaw_joint",
            "waist_pitch_joint",
            "waist_roll_joint",
        ],
        hand_joint_names=[
            "L_index_proximal_joint",
            "L_middle_proximal_joint",
            "L_pinky_proximal_joint",
            "L_ring_proximal_joint",
            "L_thumb_proximal_yaw_joint",
            "R_index_proximal_joint",
            "R_middle_proximal_joint",
            "R_pinky_proximal_joint",
            "R_ring_proximal_joint",
            "R_thumb_proximal_yaw_joint",
            "L_index_intermediate_joint",
            "L_middle_intermediate_joint",
            "L_pinky_intermediate_joint",
            "L_ring_intermediate_joint",
            "L_thumb_proximal_pitch_joint",
            "R_index_intermediate_joint",
            "R_middle_intermediate_joint",
            "R_pinky_intermediate_joint",
            "R_ring_intermediate_joint",
            "R_thumb_proximal_pitch_joint",
            "L_thumb_distal_joint",
            "R_thumb_distal_joint",
        ],
        target_eef_link_names={
            "left_wrist": "left_hand_pitch_link",
            "right_wrist": "right_hand_pitch_link",
        },
        # the robot in the sim scene we are controlling
        asset_name="robot",
        # Configuration for the IK controller
        controller=PinkIKControllerCfg(
            articulation_name="robot",
            base_link_name="base_link",
            num_hand_joints=22,
            show_ik_warnings=True,
            fail_on_joint_limit_violation=False,
            variable_input_tasks=[
                FrameTask(
                    "GR1T2_fourier_hand_6dof_left_hand_pitch_link",
                    position_cost=8.0,
                    orientation_cost=1.0,
                    lm_damping=10,
                    gain=0.5,
                ),
                FrameTask(
                    "GR1T2_fourier_hand_6dof_right_hand_pitch_link",
                    position_cost=8.0,
                    orientation_cost=1.0,
                    lm_damping=10,
                    gain=0.5,
                ),
                DampingTask(
                    cost=0.5,
                ),
                NullSpacePostureTask(
                    cost=0.5,
                    lm_damping=1,
                    controlled_frames=[
                        "GR1T2_fourier_hand_6dof_left_hand_pitch_link",
                        "GR1T2_fourier_hand_6dof_right_hand_pitch_link",
                    ],
                    controlled_joints=[
                        "left_shoulder_pitch_joint",
                        "left_shoulder_roll_joint",
                        "left_shoulder_yaw_joint",
                        "left_elbow_pitch_joint",
                        "right_shoulder_pitch_joint",
                        "right_shoulder_roll_joint",
                        "right_shoulder_yaw_joint",
                        "right_elbow_pitch_joint",
                        "waist_yaw_joint",
                        "waist_pitch_joint",
                        "waist_roll_joint",
                    ],
                ),
            ],
            fixed_input_tasks=[],
            xr_enabled=bool(carb.settings.get_settings().get("/app/xr/enabled")),
        ),
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):

        actions = ObsTerm(func=mdp.last_action)
        robot_joint_pos = ObsTerm(
            func=base_mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        robot_root_pos = ObsTerm(func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("robot")})
        robot_root_rot = ObsTerm(func=base_mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("robot")})
        # bowl_pos = ObsTerm(func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("bowl")})
        # bowl_rot = ObsTerm(func=base_mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("bowl")})
        # mug_pos = ObsTerm(func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("mug")})
        # mug_rot = ObsTerm(func=base_mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("mug")})
        robot_links_state = ObsTerm(func=mdp.get_all_robot_link_state)

        left_eef_pos = ObsTerm(func=mdp.get_left_eef_pos)
        left_eef_quat = ObsTerm(func=mdp.get_left_eef_quat)
        right_eef_pos = ObsTerm(func=mdp.get_right_eef_pos)
        right_eef_quat = ObsTerm(func=mdp.get_right_eef_quat)

        hand_joint_state = ObsTerm(func=mdp.get_hand_state)
        head_joint_state = ObsTerm(func=mdp.get_head_state)

        # obs = ObsTerm(func=mdp.object_obs)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    bowl_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": 0.5, "asset_cfg": SceneEntityCfg("bowl")}
    )


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_bowl = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.02, 0.02), "y": (-0.02, 0.02)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("bowl"),
        },
    )
    # reset_microwave = EventTerm(
    #     func=mdp.reset_joints_by_scale,
    #     mode="reset",
    #     params={
    #         "position_range": (1.0, 1.0),
    #         "velocity_range": (0.0, 0.0),
    #         "asset_cfg": SceneEntityCfg("microwave"),
    #     },
    # )
    reset_mug_and_ball = EventTerm(
        func=mdp.reset_mug_and_ball,
        mode="reset",
        params={
            "pose_range": {"x": (-0.02, 0.02), "y": (-0.02, 0.02)},
            "mug_cfg": SceneEntityCfg("mug"),
            "ball_cfg": SceneEntityCfg("ball"),
        },
    )
    reset_rack = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("rack"),
        },
    )


@configclass
class KitchenGR1T2EnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the GR1T2 kitchen environment."""

    # Scene settings
    scene: KitchenTableSceneCfg = KitchenTableSceneCfg(num_envs=1, env_spacing=5.0, replicate_physics=True)
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

    # Position of the XR anchor in the world frame
    xr: XrCfg = XrCfg(
        anchor_pos=(0.0, 0.0, 0.0),
        anchor_rot=(1.0, 0.0, 0.0, 0.0),
    )

    # OpenXR hand tracking has 26 joints per hand
    NUM_OPENXR_HAND_JOINTS = 26

    # Temporary directory for URDF files
    temp_urdf_dir = tempfile.gettempdir()

    # Idle action to hold robot in default pose
    idle_action = torch.tensor([
        -0.22878,
        0.2536,
        1.0953,
        0.5,
        0.5,
        -0.5,
        0.5,
        0.22878,
        0.2536,
        1.0953,
        0.5,
        0.5,
        -0.5,
        0.5,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ])

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 6
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 1 / 120  # 120Hz
        self.sim.render_interval = 2

        # Convert USD to URDF and change revolute joints to fixed
        temp_urdf_output_path, temp_urdf_meshes_output_path = ControllerUtils.convert_usd_to_urdf(
            self.scene.robot.spawn.usd_path, self.temp_urdf_dir, force_conversion=True
        )
        ControllerUtils.change_revolute_to_fixed(
            temp_urdf_output_path, self.actions.pink_ik_cfg.ik_urdf_fixed_joint_names
        )

        # Set the URDF and mesh paths for the IK controller
        self.actions.pink_ik_cfg.controller.urdf_path = temp_urdf_output_path
        self.actions.pink_ik_cfg.controller.mesh_path = temp_urdf_meshes_output_path

        self.teleop_devices = DevicesCfg(
            devices={
                "handtracking": OpenXRDeviceCfg(
                    retargeters=[
                        GR1T2RetargeterCfg(
                            enable_visualization=True,
                            num_open_xr_hand_joints=2 * self.NUM_OPENXR_HAND_JOINTS,
                            sim_device=self.sim.device,
                            hand_joint_names=self.actions.pink_ik_cfg.hand_joint_names,
                        ),
                    ],
                    sim_device=self.sim.device,
                    xr_cfg=self.xr,
                ),
                "manusvive": ManusViveCfg(
                    retargeters=[
                        GR1T2RetargeterCfg(
                            enable_visualization=True,
                            num_open_xr_hand_joints=2 * 26,
                            sim_device=self.sim.device,
                            hand_joint_names=self.actions.pink_ik_cfg.hand_joint_names,
                        ),
                    ],
                    sim_device=self.sim.device,
                    xr_cfg=self.xr,
                ),
            }
        )
