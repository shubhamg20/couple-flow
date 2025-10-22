#!/usr/bin/env python3
"""
Isaac Lab Configuration Checker
Verifies that your Isaac Lab task is correctly configured to load the banana USD
"""

import os
import sys

def check_isaac_lab_config():
    print("Isaac Lab Configuration Check")
    print("="*50)
    
    # Check common Isaac Lab configuration locations
    config_paths = [
        "omni/isaac/lab_tasks/manager_based/manipulation/",
        "source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/manipulation/",
        "./",
    ]
    
    # Look for configuration files
    config_files = []
    for path in config_paths:
        if os.path.exists(path):
            for file in os.listdir(path):
                if file.endswith('.py') and ('config' in file.lower() or 'task' in file.lower()):
                    config_files.append(os.path.join(path, file))
    
    print(f"Found potential config files:")
    for config_file in config_files:
        print(f"  {config_file}")
    
    print(f"\nWhat to check in your Isaac Lab task configuration:")
    print(f"="*50)
    
    print("""
1. **Asset Configuration**: Look for something like this in your task config:

```python
from omni.isaac.lab.assets import RigidObjectCfg
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR

# Object configuration
object_cfg = RigidObjectCfg(
    prim_path="/World/envs/env_.*/Object",  # This path must match!
    spawn=sim_utils.UsdFileCfg(
        usd_path="{BANANA_USD_PATH}/banana.usd",  # Your USD file path
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0,
            disable_gravity=False,
        ),
    ),
    init_state=RigidObjectInitialStateCfg(pos=[0.5, 0, 0.05]),
)
```

2. **Scene Configuration**: Make sure your scene includes the object:

```python
# Scene configuration
scene_cfg = InteractiveSceneCfg(
    num_envs=args_cli.num_envs, 
    env_spacing=2.5,
    replicate_physics=True
)
scene_cfg.object = object_cfg  # Add your object to the scene
```

3. **USD File Path**: Verify the path to your banana.usd file is correct:
   - Absolute path: "/home/shubham/summer/usd_extracted/banana_csm/banana.usd"
   - Or relative to your Isaac Lab installation

4. **Prim Path Pattern**: The key is this pattern:
   `/World/envs/env_.*/Object`
   
   This means Isaac Lab expects:
   - Your USD file has a root prim called "Object" 
   - It will be instantiated at /World/envs/env_0/Object, /World/envs/env_1/Object, etc.

5. **Common Issues**:
   - USD file path is wrong
   - USD file doesn't have RigidBodyAPI applied to /Object
   - Prim path pattern doesn't match
   - USD file is corrupted or has wrong structure

**Next Steps:**
1. Run the USD debug script: `python debug_usd.py /path/to/your/banana.usd`
2. Check that your task config uses the correct prim_path pattern
3. Verify the USD file path in your configuration
4. Make sure your scene configuration includes the object

**Example Isaac Lab Task Structure:**
```python
@configclass
class ObjectManipulationEnvCfg(ManagerBasedEnvCfg):
    # Scene settings
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=2.5, replicate_physics=True
    )
    
    # Add your banana object
    scene.object = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/shubham/summer/usd_extracted/banana_csm/banana.usd"
        ),
        init_state=RigidObjectInitialStateCfg(pos=[0.5, 0, 0.05]),
    )
```
""")

if __name__ == "__main__":
    check_isaac_lab_config()