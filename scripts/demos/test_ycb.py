"""Minimal test script for YCB objects in Isaac Lab."""

import argparse
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Test YCB objects in Isaac Lab.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app - this initializes all the omni modules
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Now we can import Isaac Lab modules safely (AFTER AppLauncher)
print("AppLauncher initialized successfully!")

try:
    import isaaclab.sim.spawners as sim_utils
    print("Isaac Lab sim modules imported successfully!")
    
    from isaaclab.assets import RigidObjectCfg
    print("Isaac Lab assets imported successfully!")
    
    from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg, MassPropertiesCfg, CollisionPropertiesCfg
    print("Isaac Lab schemas imported successfully!")
    
    import isaaclab_tasks
    print("Isaac Lab tasks imported successfully!")
    
    print("✓ All Isaac Lab modules loaded successfully!")
    
except Exception as e:
    print(f"✗ Error importing Isaac Lab modules: {e}")

# Close the app
simulation_app.close()
print("Test completed.")
