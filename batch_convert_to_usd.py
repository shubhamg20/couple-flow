#!/usr/bin/env python3
"""
Batch USD Converter for Isaac Lab
Converts multiple CSM objects to Isaac Lab-compatible USD files
"""

from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Gf, Sdf
import os
import sys

# Object definitions with appropriate colors and properties
OBJECTS = {
    "apple_csm": {
        "mass": 0.18,  # 180g apple
        "color": (0.8, 0.2, 0.2),  # Red apple
        "friction_static": 0.4,
        "friction_dynamic": 0.3,
        "restitution": 0.1
    },
    "banana_csm": {
        "mass": 0.12,  # 120g banana
        "color": (0.9, 0.8, 0.2),  # Yellow banana
        "friction_static": 0.3,
        "friction_dynamic": 0.2,
        "restitution": 0.1
    },
    "battery_csm": {
        "mass": 0.05,  # 50g battery
        "color": (0.2, 0.2, 0.2),  # Dark gray battery
        "friction_static": 0.6,
        "friction_dynamic": 0.5,
        "restitution": 0.05
    },
    "blue_block_csm": {
        "mass": 0.1,  # 100g block
        "color": (0.2, 0.4, 0.8),  # Blue
        "friction_static": 0.7,
        "friction_dynamic": 0.6,
        "restitution": 0.3
    },
    "blue_cup_csm": {
        "mass": 0.08,  # 80g cup
        "color": (0.2, 0.4, 0.8),  # Blue
        "friction_static": 0.5,
        "friction_dynamic": 0.4,
        "restitution": 0.2
    },
    "blue_plate_csm": {
        "mass": 0.15,  # 150g plate
        "color": (0.2, 0.4, 0.8),  # Blue
        "friction_static": 0.6,
        "friction_dynamic": 0.5,
        "restitution": 0.1
    },
    "carrot_csm": {
        "mass": 0.06,  # 60g carrot
        "color": (0.9, 0.5, 0.1),  # Orange carrot
        "friction_static": 0.4,
        "friction_dynamic": 0.3,
        "restitution": 0.1
    },
    "corn_csm": {
        "mass": 0.15,  # 150g corn
        "color": (0.9, 0.8, 0.3),  # Yellow corn
        "friction_static": 0.4,
        "friction_dynamic": 0.3,
        "restitution": 0.15
    },
    "grapes_csm": {
        "mass": 0.2,   # 200g grapes
        "color": (0.5, 0.2, 0.7),  # Purple grapes
        "friction_static": 0.3,
        "friction_dynamic": 0.2,
        "restitution": 0.2
    },
    "green_block_csm": {
        "mass": 0.1,   # 100g block
        "color": (0.2, 0.7, 0.3),  # Green
        "friction_static": 0.7,
        "friction_dynamic": 0.6,
        "restitution": 0.3
    },
    "green_cup_csm": {
        "mass": 0.08,  # 80g cup
        "color": (0.2, 0.7, 0.3),  # Green
        "friction_static": 0.5,
        "friction_dynamic": 0.4,
        "restitution": 0.2
    },
    "green_tray_csm": {
        "mass": 0.12,  # 120g tray
        "color": (0.2, 0.7, 0.3),  # Green
        "friction_static": 0.6,
        "friction_dynamic": 0.5,
        "restitution": 0.15
    },
    "ice_cream_csm": {
        "mass": 0.05,  # 50g ice cream
        "color": (0.9, 0.9, 0.8),  # Cream color
        "friction_static": 0.2,
        "friction_dynamic": 0.1,
        "restitution": 0.05
    },
    "mug_csm": {
        "mass": 0.25,  # 250g mug
        "color": (0.8, 0.7, 0.6),  # Beige/brown mug
        "friction_static": 0.6,
        "friction_dynamic": 0.5,
        "restitution": 0.1
    },
    "pen_csm": {
        "mass": 0.01,  # 10g pen
        "color": (0.1, 0.1, 0.8),  # Blue pen
        "friction_static": 0.5,
        "friction_dynamic": 0.4,
        "restitution": 0.3
    },
    "pink_plate_csm": {
        "mass": 0.15,  # 150g plate
        "color": (0.9, 0.4, 0.7),  # Pink
        "friction_static": 0.6,
        "friction_dynamic": 0.5,
        "restitution": 0.1
    },
    "plastic_cup_csm": {
        "mass": 0.03,  # 30g plastic cup
        "color": (0.9, 0.9, 0.9),  # White plastic
        "friction_static": 0.4,
        "friction_dynamic": 0.3,
        "restitution": 0.4
    },
    "red_block_csm": {
        "mass": 0.1,   # 100g block
        "color": (0.8, 0.2, 0.2),  # Red
        "friction_static": 0.7,
        "friction_dynamic": 0.6,
        "restitution": 0.3
    },
    "red_bowl_csm": {
        "mass": 0.18,  # 180g bowl
        "color": (0.8, 0.2, 0.2),  # Red
        "friction_static": 0.5,
        "friction_dynamic": 0.4,
        "restitution": 0.15
    },
    "wood_block_csm": {
        "mass": 0.15,  # 150g wood block
        "color": (0.6, 0.4, 0.2),  # Brown wood
        "friction_static": 0.8,
        "friction_dynamic": 0.7,
        "restitution": 0.2
    },
    "yellow_bowl_csm": {
        "mass": 0.18,  # 180g bowl
        "color": (0.9, 0.8, 0.2),  # Yellow
        "friction_static": 0.5,
        "friction_dynamic": 0.4,
        "restitution": 0.15
    }
}

def convert_object_to_usd(object_name, base_path="/home/shubham/summer/usd_extracted"):
    """Convert a single CSM object to Isaac Lab-compatible USD"""
    
    if object_name not in OBJECTS:
        print(f"ERROR: Unknown object {object_name}")
        return False
    
    object_props = OBJECTS[object_name]
    object_path = os.path.join(base_path, object_name)
    mesh_file = os.path.join(object_path, "mesh.usdc")
    texture_path = os.path.join(object_path, "textures")
    output_file = os.path.join(object_path, f"{object_name.replace('_csm', '')}.usd")
    
    print(f"\n{'='*60}")
    print(f"Converting {object_name}...")
    print(f"{'='*60}")
    print(f"Input mesh: {mesh_file}")
    print(f"Output USD: {output_file}")
    
    if not os.path.exists(mesh_file):
        print(f"ERROR: {mesh_file} not found")
        return False
    
    try:
        # Create new USD stage
        stage = Usd.Stage.CreateNew(output_file)
        
        # Set up stage metadata (Isaac Lab requirements)
        stage.SetMetadata("metersPerUnit", 1.0)
        stage.SetMetadata("upAxis", "Z")
        
        # Create physics scene (REQUIRED for Isaac Lab)
        physics_scene_prim = stage.DefinePrim("/PhysicsScene", "PhysicsScene")
        physics_scene = UsdPhysics.Scene(physics_scene_prim)
        physics_scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0, 0, -1))
        physics_scene.CreateGravityMagnitudeAttr().Set(9.81)
        
        # Open source mesh file
        mesh_stage = Usd.Stage.Open(mesh_file)
        
        # Find the mesh prim
        source_mesh_prim = None
        for prim in mesh_stage.Traverse():
            if prim.IsA(UsdGeom.Mesh):
                source_mesh_prim = prim
                break
        
        if not source_mesh_prim:
            print("ERROR: No mesh found in source file")
            return False
        
        # Create root object prim - use the object name for the prim path
        # This creates prims like "/Apple", "/Banana", "/BlueBlock", etc. to match Isaac Lab expectations
        # Isaac Lab will instantiate these at paths like /World/envs/env_.*/Apple
        object_name_clean = object_name.replace('_csm', '')  # "banana_csm" -> "banana"
        
        # Convert to proper case for prim names
        if 'block' in object_name_clean:
            # Handle block naming: "blue_block" -> "BlueBlock"
            parts = object_name_clean.split('_')
            object_name_clean = ''.join(word.capitalize() for word in parts)
        elif 'cup' in object_name_clean:
            # Handle cup naming: "blue_cup" -> "BlueCup"
            parts = object_name_clean.split('_')
            object_name_clean = ''.join(word.capitalize() for word in parts)
        elif 'plate' in object_name_clean:
            # Handle plate naming: "blue_plate" -> "BluePlate"
            parts = object_name_clean.split('_')
            object_name_clean = ''.join(word.capitalize() for word in parts)
        elif 'bowl' in object_name_clean:
            # Handle bowl naming: "red_bowl" -> "RedBowl"
            parts = object_name_clean.split('_')
            object_name_clean = ''.join(word.capitalize() for word in parts)
        elif 'tray' in object_name_clean:
            # Handle tray naming: "green_tray" -> "GreenTray"
            parts = object_name_clean.split('_')
            object_name_clean = ''.join(word.capitalize() for word in parts)
        elif '_' in object_name_clean:
            # Handle other compound names: "ice_cream" -> "IceCream", "plastic_cup" -> "PlasticCup"
            parts = object_name_clean.split('_')
            object_name_clean = ''.join(word.capitalize() for word in parts)
        else:
            # Single word names: "apple" -> "Apple", "banana" -> "Banana"
            object_name_clean = object_name_clean.capitalize()
        
        object_prim_path = f"/{object_name_clean}"
        object_prim = stage.DefinePrim(object_prim_path, "Xform")
        object_xform = UsdGeom.Xform(object_prim)
        
        print(f"Creating USD prim: {object_prim_path}")
        print(f"This will be accessible in Isaac Lab as: /World/envs/env_.*/{object_name_clean}")
        
        # Create mesh under object
        target_mesh_path = f"{object_prim_path}/mesh"
        target_mesh_prim = stage.DefinePrim(target_mesh_path, "Mesh")
        target_mesh = UsdGeom.Mesh(target_mesh_prim)
        source_mesh = UsdGeom.Mesh(source_mesh_prim)
        
        # Copy mesh data
        print("Copying mesh geometry...")
        if source_mesh.GetPointsAttr().HasValue():
            points = source_mesh.GetPointsAttr().Get()
            target_mesh.CreatePointsAttr().Set(points)
            print(f"  Points: {len(points)}")
        
        if source_mesh.GetFaceVertexIndicesAttr().HasValue():
            indices = source_mesh.GetFaceVertexIndicesAttr().Get()
            target_mesh.CreateFaceVertexIndicesAttr().Set(indices)
            print(f"  Indices: {len(indices)}")
        
        if source_mesh.GetFaceVertexCountsAttr().HasValue():
            counts = source_mesh.GetFaceVertexCountsAttr().Get()
            target_mesh.CreateFaceVertexCountsAttr().Set(counts)
            print(f"  Face counts: {len(counts)}")
        
        # Copy normals
        if source_mesh.GetNormalsAttr().HasValue():
            normals = source_mesh.GetNormalsAttr().Get()
            target_mesh.CreateNormalsAttr().Set(normals)
            target_mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
            print("  Normals copied")
        
        # Copy primvars (texture coordinates)
        print("Copying primvars...")
        source_primvars_api = None
        try:
            source_primvars_api = UsdGeom.PrimvarsAPI(source_mesh_prim)
            target_primvars_api = UsdGeom.PrimvarsAPI(target_mesh_prim)
            
            all_primvars = source_primvars_api.GetPrimvarsWithValues()
            for primvar in all_primvars:
                try:
                    primvar_name = primvar.GetPrimvarName()
                    if primvar.HasValue():
                        value = primvar.Get()
                        interpolation = primvar.GetInterpolation()
                        type_name = primvar.GetTypeName()
                        target_primvar = target_primvars_api.CreatePrimvar(
                            primvar_name, type_name, interpolation
                        )
                        target_primvar.Set(value)
                        print(f"  Copied primvar: {primvar_name}")
                except Exception as e:
                    print(f"  Failed to copy primvar {primvar_name}: {e}")
                    continue
        except Exception as e:
            print(f"  Primvar copy failed: {e}")
        
        # Apply physics APIs
        print("Applying physics APIs...")
        
        # Apply RigidBodyAPI
        rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(object_prim)
        rigid_body_api.CreateRigidBodyEnabledAttr().Set(True)
        print("  RigidBodyAPI applied and enabled")
        
        # Apply MassAPI 
        mass_api = UsdPhysics.MassAPI.Apply(object_prim)
        mass_api.CreateMassAttr().Set(object_props["mass"])
        print(f"  MassAPI applied ({object_props['mass']}kg)")
        
        # Apply CollisionAPI
        collision_api = UsdPhysics.CollisionAPI.Apply(object_prim)
        collision_api.CreateCollisionEnabledAttr().Set(True)
        print("  CollisionAPI applied and enabled")
        
        # Apply MeshCollisionAPI
        mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(object_prim)
        mesh_collision_api.CreateApproximationAttr().Set("convexHull")
        print("  MeshCollisionAPI applied (convexHull)")
        
        # Create materials
        print("Creating materials...")
        material_path = f"/Materials/{object_name_clean}Material"
        material_prim = stage.DefinePrim(material_path, "Material")
        material = UsdShade.Material(material_prim)
        
        # Physics material properties
        physics_material = UsdPhysics.MaterialAPI.Apply(material_prim)
        physics_material.CreateStaticFrictionAttr().Set(object_props["friction_static"])
        physics_material.CreateDynamicFrictionAttr().Set(object_props["friction_dynamic"])
        physics_material.CreateRestitutionAttr().Set(object_props["restitution"])
        print("  Physics material created")
        
        # Visual material with object-specific color and texture support
        shader = UsdShade.Shader.Define(stage, f"{material_path}/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        material_binding_api = UsdShade.MaterialBindingAPI.Apply(object_prim)
        material_binding_api.Bind(material)
        
        # Also bind to mesh directly for better compatibility
        mesh_binding_api = UsdShade.MaterialBindingAPI.Apply(target_mesh_prim)
        mesh_binding_api.Bind(material)
        # Check for textures and apply them if available
        texture_applied = False
        if os.path.exists(texture_path):
            texture_files = [f for f in os.listdir(texture_path) 
                           if f.lower().endswith(('.hdr', '.jpg', '.png', '.exr', '.tiff', '.tif'))]
            
            if texture_files:
                # Try to find common texture types
                diffuse_texture = None
                for tex_file in texture_files:
                    tex_lower = tex_file.lower()
                    if any(keyword in tex_lower for keyword in ['diffuse', 'albedo', 'color', 'base']):
                        diffuse_texture = tex_file
                        break
                
                # If no specifically named diffuse texture, use the first one
                if not diffuse_texture and texture_files:
                    diffuse_texture = texture_files[0]
                
                if diffuse_texture:
                    # Create texture reader
                    texture_reader = UsdShade.Shader.Define(stage, f"{material_path}/DiffuseTexture")
                    texture_reader.CreateIdAttr("UsdUVTexture")
                    
                    # Set texture file path (relative to USD file)
                    texture_file_path = f"textures/{diffuse_texture}"
                    texture_reader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(texture_file_path)
                    texture_reader.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
                    texture_reader.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
                    
                    # Connect texture to shader
                    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
                        texture_reader.ConnectableAPI(), "rgb"
                    )
                    
                    # Try to connect UV coordinates
                    if source_primvars_api:
                        # Look for UV primvars
                        for primvar in source_primvars_api.GetPrimvarsWithValues():
                            primvar_name = primvar.GetPrimvarName()
                            if 'st' in primvar_name.lower() or 'uv' in primvar_name.lower():
                                # Create primvar reader for UV coordinates
                                primvar_reader = UsdShade.Shader.Define(stage, f"{material_path}/PrimvarReader")
                                primvar_reader.CreateIdAttr("UsdPrimvarReader_float2")
                                primvar_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set(primvar_name)
                                
                                # Connect to texture reader
                                texture_reader.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
                                    primvar_reader.ConnectableAPI(), "result"
                                )
                                break
                    
                    texture_applied = True
                    print(f"  Texture applied: {diffuse_texture}")
                    print(f"  Available textures: {texture_files}")
        
        if not texture_applied:
            # Set object-specific fallback color
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(object_props["color"])
            print(f"  Using fallback color: {object_props['color']}")
        
        # Set material properties
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.4)
        
        # Connect shader to material
        shader_output = shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        material_surface_output = material.CreateSurfaceOutput()
        material_surface_output.ConnectToSource(shader.ConnectableAPI(), "surface")
        if texture_applied:
            print(f"  Visual material created with texture")
        else:
            print(f"  Visual material created with color: {object_props['color']}")
        
        # Bind material to object
        material_binding_api = UsdShade.MaterialBindingAPI.Apply(object_prim)
        material_binding_api.Bind(material)
        print("  Material bound to object")
        
        # Set default prim
        stage.SetDefaultPrim(object_prim)
        
        # Set stage metadata
        stage.SetStartTimeCode(0)
        stage.SetEndTimeCode(1)
        stage.SetTimeCodesPerSecond(24)
        stage.GetRootLayer().customLayerData = {
            "metersPerUnit": 1.0,
            "upAxis": "Z"
        }
        
        # Clean up source stage
        del mesh_stage
        
        # Save the USD file
        print("Saving USD file...")
        stage.Save()
        
        # Quick verification
        verification_stage = Usd.Stage.Open(output_file)
        if not verification_stage:
            print("ERROR: Could not re-open USD file")
            return False
        
        default_prim = verification_stage.GetDefaultPrim()
        if not default_prim or not default_prim.IsValid():
            print("ERROR: No valid default prim found")
            return False
        
        # Check physics APIs
        has_rigid_body = default_prim.HasAPI(UsdPhysics.RigidBodyAPI)
        has_mass = default_prim.HasAPI(UsdPhysics.MassAPI)
        has_collision = default_prim.HasAPI(UsdPhysics.CollisionAPI)
        
        if not all([has_rigid_body, has_mass, has_collision]):
            print("ERROR: Missing required physics APIs")
            return False
        
        print(f"SUCCESS: {object_name} converted successfully!")
        print(f"  Output: {output_file}")
        print(f"  Root prim: {object_prim_path}")
        print(f"  Mass: {object_props['mass']}kg")
        print(f"  Color: {object_props['color']}")
        print(f"  Physics APIs: ✓ RigidBody ✓ Mass ✓ Collision")
        print(f"  Compatible with Isaac Lab prim path: /World/envs/env_.*/{object_name_clean}")
        
        return True
        
    except Exception as e:
        print(f"ERROR converting {object_name}: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Convert all objects to USD"""
    
    print("Isaac Lab Batch USD Converter")
    print("="*60)
    print(f"Processing {len(OBJECTS)} objects...")
    
    successes = []
    failures = []
    
    for object_name in OBJECTS.keys():
        if convert_object_to_usd(object_name):
            successes.append(object_name)
        else:
            failures.append(object_name)
    
    print("\n" + "="*60)
    print("BATCH CONVERSION SUMMARY")
    print("="*60)
    
    print(f"✓ Successfully converted: {len(successes)}")
    for success in successes:
        print(f"  - {success}")
    
    if failures:
        print(f"\n✗ Failed to convert: {len(failures)}")
        for failure in failures:
            print(f"  - {failure}")
    
    print(f"\nTotal: {len(successes)}/{len(OBJECTS)} objects converted")
    
    if len(successes) == len(OBJECTS):
        print("\n🎉 ALL OBJECTS CONVERTED SUCCESSFULLY!")
        print("\nAll USD files are now ready for Isaac Lab!")
        print("You can use them in your pick-place tasks by updating the usd_path in your configuration files.")
    else:
        print(f"\n⚠️  {len(failures)} objects failed to convert. Check the errors above.")
    
    return len(failures) == 0

if __name__ == "__main__":
    success = main()
    if not success:
        print("\nSome conversions failed. Check the output above for details.")
        sys.exit(1)
    else:
        print("\nBatch conversion completed successfully!")
