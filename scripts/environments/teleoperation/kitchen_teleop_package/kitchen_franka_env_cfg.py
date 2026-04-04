# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg, BinaryJointPositionActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab_tasks.manager_based.manipulation.pick_place import mdp
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG  # isort: skip

##
# Scene definition
##
@configclass
class KitchenTableSceneCfg(InteractiveSceneCfg):

    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Camera1",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(-0.4, -0.40, 1.4),
            rot=(0.8160, -0.0500, 0.0714, 0.5714),
            convention="world"
        ),
        update_latest_camera_pose=True,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=30.0,
            focus_distance=400.0,
            horizontal_aperture=34.32365,
            clipping_range=(0.01, 10000000.0)
        ),
        width=256,
        height=256,
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
        init_state=RigidObjectCfg.InitialStateCfg(pos=[-0.15, 0.35, 1.05], rot=[0, 0, 0, 1]),
        spawn=UsdFileCfg(
            # Use the physics-enabled bowl USD
            usd_path="/workspace/isaaclab/usd_extracted/bowl_csm/white_bowl.usd",
            # scale=(.009, .009, .009),
            scale=(1.0, 1.0, 1.0),
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

    # ball = RigidObjectCfg(
    #     prim_path="/World/envs/env_.*/ball3",
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=[0.00, 0.5, 1.06], rot=[1, 0, 0, 0]),
    #     spawn=UsdFileCfg(
    #         usd_path="/workspace/isaaclab/usd_extracted/ball_csm/ball3.usd",
    #         scale=(3, 3, 3),
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #             rigid_body_enabled=True,
    #             kinematic_enabled=False,
    #             disable_gravity=False,
    #             solver_position_iteration_count=16,
    #             solver_velocity_iteration_count=1,
    #             max_angular_velocity=1000.0,
    #             max_linear_velocity=1000.0,
    #             max_depenetration_velocity=5.0,
    #         ),
    #         collision_props=sim_utils.CollisionPropertiesCfg(
    #             collision_enabled=True,
    #             contact_offset=0.00,
    #             rest_offset=0.00,
    #         ),
    #         mass_props=sim_utils.MassPropertiesCfg(mass=0.4),
    #     ),
    # )
    ball = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Ball",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.00, 0.5, 1.16], rot=[1, 0, 0, 0]),
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
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.00, 0.5, 1.15], rot=[0.5, 0.5, 0.5, 0.5]),
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

    # Franka robot configured for manipulation tasks
    robot: ArticulationCfg = FRANKA_PANDA_HIGH_PD_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": 0.0,
                "panda_joint2": -0.7,
                "panda_joint3": 0.0,
                "panda_joint4": -2.4,
                "panda_joint5": 0.0,
                "panda_joint6": 1.7,
                "panda_joint7": 0.785,
                "panda_finger_joint.*": 0.04,
            },
            pos=(0.0, 0.0, 0.7),
            rot=(0.707, 0.0, 0.0, 0.707),
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


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*"]),
        },
    )
    reset_gripper = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot", joint_names=["panda_finger_joint.*"]),
        },
    )
  
    # reset_bowl = EventTerm(
    #     func=mdp.reset_root_state_uniform,
    #     mode="reset",
    #     params={
    #         "pose_range": {"x": (-0.02, 0.02), "y": (-0.02, 0.02)},
    #         "velocity_range": {},
    #         "asset_cfg": SceneEntityCfg("bowl"),
    #     },
    # )
    # reset_microwave = EventTerm(
    #     func=mdp.reset_joints_by_scale,
    #     mode="reset",
    #     params={
    #         "position_range": (1.0, 1.0),
    #         "velocity_range": (0.0, 0.0),
    #         "asset_cfg": SceneEntityCfg("microwave"),
    #     },
    # )  
    # reset_mug_and_ball = EventTerm(
    #     func=mdp.reset_mug_and_ball,
    #     mode="reset",
    #     params={
    #         "pose_range": {"x": (-0.02, 0.02), "y": (-0.02, 0.02)},
    #         "mug_cfg": SceneEntityCfg("mug"),
    #         "ball_cfg": SceneEntityCfg("ball"),
    #     },
    # )
   
    # reset_rack = EventTerm(
    #     func=mdp.reset_root_state_uniform,
    #     mode="reset",
    #     params={
    #         "pose_range": {},
    #         "velocity_range": {},
    #         "asset_cfg": SceneEntityCfg("rack"),
    #     },
    # ) 


##
# Environment configuration
##
@configclass
class KitchenFrankaEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the Franka kitchen environment."""

    # Scene settings
    scene: KitchenTableSceneCfg = KitchenTableSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)
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
