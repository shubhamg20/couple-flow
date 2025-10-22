"""Script to run keyboard teleoperation with YCB objects in Isaac Lab."""

import argparse
from collections.abc import Callable

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Keyboard teleoperation with YCB objects.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--teleop_device", type=str, default="keyboard", help="Teleoperation device.")
parser.add_argument("--task", type=str, default="Isaac-Lift-Cube-Franka-v0", help="Name of the task.")
parser.add_argument("--sensitivity", type=float, default=1.0, help="Sensitivity factor.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# IMPORTANT: Import everything AFTER AppLauncher initialization
import gymnasium as gym
import torch
import random
import omni.log

from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.assets import RigidObjectCfg
import isaaclab.sim.spawners as sim_utils
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg, MassPropertiesCfg, CollisionPropertiesCfg

import isaaclab_tasks
from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.utils import parse_env_cfg

def modify_env_for_ycb(env_cfg):
    """Replace the default cube with a YCB object."""
    
    # Create YCB mug configuration
    mug_cfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0, 0.055], rot=[1, 0, 0, 0]),
        spawn=sim_utils.UsdFileCfg(
            usd_path="http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.2/Isaac/Props/YCB/Axis_Aligned/025_mug.usd",
            rigid_props=RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=False,
                disable_gravity=False,
            ),
            mass_props=MassPropertiesCfg(mass=0.118),
            collision_props=CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.02,
                rest_offset=0.0,
            ),
            activate_contact_sensors=True,
        ),
    )
    
    # Replace the object in the scene
    env_cfg.scene.object = mug_cfg
    
    return env_cfg

def main():
    """Main function."""
    try:
        # Parse environment configuration
        env_cfg = parse_env_cfg(args_cli.task, device="cuda", num_envs=args_cli.num_envs)
        
        # Modify environment to use YCB objects
        env_cfg = modify_env_for_ycb(env_cfg)
        
        # Modify configuration for manual control
        env_cfg.terminations.time_out = None
        if "Lift" in args_cli.task:
            env_cfg.commands.object_pose.resampling_time_range = (1.0e9, 1.0e9)
            env_cfg.terminations.object_reached_goal = DoneTerm(func=mdp.object_reached_goal)

        # Create environment
        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
        print(f"✓ Environment created successfully with YCB mug!")
        
        # Create teleoperation interface
        teleop_interface = Se3Keyboard(
            Se3KeyboardCfg(pos_sensitivity=0.05 * args_cli.sensitivity, 
                          rot_sensitivity=0.05 * args_cli.sensitivity)
        )
        
        # Reset environment and teleop interface
        env.reset()
        teleop_interface.reset()
        
        print("✓ Teleoperation started. Use arrow keys to control the robot.")
        print("  Press 'R' to reset the environment.")
        print("  Close the window to exit.")
        
        # Simulation loop
        while simulation_app.is_running():
            with torch.inference_mode():
                # Get teleoperation command
                action = teleop_interface.advance()
                actions = action.repeat(env.num_envs, 1)
                
                # Step environment
                env.step(actions)
        
        # Clean up
        env.close()
        print("Environment closed successfully.")
        
    except Exception as e:
        omni.log.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()

if __name__ == "__main__":
    main()
