# read_usd_contents.py
from pxr import Usd, UsdGeom

# Path to your USD file
usd_path = "source/gr1t2/GR1T2_fourier_hand_6dof/configuration/GR1T2_fourier_hand_6dof_physics.usd"

# Open the USD stage in read-only mode
stage = Usd.Stage.Open(usd_path)
if not stage:
    print(f"Failed to open USD file: {usd_path}")
    exit(1)

print(f"Listing all prims in {usd_path}:\n")

# Traverse all prims
for prim in stage.Traverse():
    print(f"Prim: {prim.GetPath()} | Type: {prim.GetTypeName()}")
    # List all attributes for this prim
    # for attr in prim.GetAttributes():
    #     print(f"    Attr: {attr.GetName()} | Value: {attr.Get()}")
    print("-" * 50)