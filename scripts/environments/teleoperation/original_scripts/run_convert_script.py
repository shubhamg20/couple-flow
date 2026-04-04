#!/usr/bin/env python
"""
Launcher script to run convert2instantiatedassets.py through Isaac Sim.
This script initializes the Omniverse context and then executes the conversion.

Usage:
    ./isaaclab.sh -p scripts/environments/teleoperation/run_convert_script.py
"""

import sys
import os

# Add the script directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

# Initialize Omniverse context
try:
    from omni.isaac.kit import SimulationApp
    
    # Create a headless simulation app (no rendering)
    simulation_app = SimulationApp({"headless": True})
    
    # Now import and run the conversion script
    from convert2instantiatedassets import ASSET_USD_PATH, SAVE_AS_PATH, convert_asset_instanceable
    
    print("[INFO] Running USD asset conversion...")
    
    # Convert the asset
    convert_asset_instanceable(
        asset_usd_path=ASSET_USD_PATH,
        source_prim_path="/mug",
        save_as_path=SAVE_AS_PATH,
        create_xforms=True
    )
    
    print("[INFO] Conversion complete!")
    
    # Cleanup
    simulation_app.close()
    
except ImportError as e:
    print(f"[ERROR] Failed to initialize Omniverse context: {e}")
    print("[INFO] Make sure you're running this with the Isaac Sim Python environment:")
    print("       ./isaaclab.sh -p scripts/environments/teleoperation/run_convert_script.py")
    sys.exit(1)
except Exception as e:
    print(f"[ERROR] Conversion failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
