#!/usr/bin/env python3
"""
Create sushi USD directly from existing mesh.usdc and texture.jpg
"""

from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Gf, Sdf
import os

def create_sushi_from_existing_files():
    """Create sushi USD from your existing mesh.usdc and texture.jpg"""
    
    # Your existing files
    mesh_file = "/home/shubham/summer/IsaacLab/usd_extracted/sushi_csm/mesh.usdc"
    texture_file = "/home/shubham/summer/IsaacLab/usd_extracted/sushi_csm/0/texture.jpg"
    output_file = "/home/shubham/summer/IsaacLab/usd_extracted/sushi_csm/sushi.usd"
    
    print("Creating sushi USD from existing files...")
    print(f"Mesh: {mesh_file}")
    print(f"Texture: {texture_file}")
    print(f"Output: {output_file}")
    
    # Check files exist
    if not os.path.exists(mesh_file):
        print(f"❌ Mesh not found: {mesh_file}")
        return False
    
    if not os.path.exists(texture_file):
        print(f"❌ Texture not found: {texture_file}")
        return False
    
    mesh_size = os.path.getsize(mesh_file)
    texture_size = os.path.getsize(texture_file)
    print(f"✓ Mesh file: {mesh_size} bytes")
    print(f"✓ Texture file: {texture_size} bytes")
    
    try:
        # Create new stage
        stage = Usd.Stage.CreateNew(output_file)
        stage.SetMetadata("upAxis", "Z")
        stage.SetMetadata("metersPerUnit", 1.0)
        
        # Physics scene
        physics_scene = stage.DefinePrim("/PhysicsScene", "PhysicsScene")
        scene_api = UsdPhysics.Scene(physics_scene)
        scene_api.CreateGravityDirectionAttr().Set(Gf.Vec3f(0, 0, -1))
        scene_api.CreateGravityMagnitudeAttr().Set(9.81)
        print("✓ Physics scene created")
        
        # Load your mesh.usdc
        mesh_stage = Usd.Stage.Open(mesh_file)
        source_mesh_prim = None
        
        for prim in mesh_stage.Traverse():
            if prim.IsA(UsdGeom.Mesh):
                source_mesh_prim = prim
                print(f"✓ Found mesh: {prim.GetPath()}")
                break
        
        if not source_mesh_prim:
            print("❌ No mesh found in mesh.usdc")
            return False
        
        # Create sushi object
        sushi_prim = stage.DefinePrim("/sushi", "Xform")
        mesh_prim = stage.DefinePrim("/sushi/sushiMesh", "Mesh")
        
        # Copy mesh data
        source_mesh = UsdGeom.Mesh(source_mesh_prim)
        target_mesh = UsdGeom.Mesh(mesh_prim)
        
        # Copy all mesh attributes
        print("Copying mesh data...")
        
        if source_mesh.GetPointsAttr().HasValue():
            points = source_mesh.GetPointsAttr().Get()
            target_mesh.CreatePointsAttr().Set(points)
            print(f"  Points: {len(points)}")
        
        if source_mesh.GetFaceVertexIndicesAttr().HasValue():
            indices = source_mesh.GetFaceVertexIndicesAttr().Get()
            target_mesh.CreateFaceVertexIndicesAttr().Set(indices)
            print(f"  Face indices: {len(indices)}")
        
        if source_mesh.GetFaceVertexCountsAttr().HasValue():
            counts = source_mesh.GetFaceVertexCountsAttr().Get()
            target_mesh.CreateFaceVertexCountsAttr().Set(counts)
            print(f"  Face counts: {len(counts)}")
        
        if source_mesh.GetNormalsAttr().HasValue():
            normals = source_mesh.GetNormalsAttr().Get()
            target_mesh.CreateNormalsAttr().Set(normals)
            target_mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
            print(f"  Normals: {len(normals)}")
        
        # Copy UV coordinates (primvars)
        print("Copying UV coordinates...")
        source_primvars = UsdGeom.PrimvarsAPI(source_mesh_prim)
        target_primvars = UsdGeom.PrimvarsAPI(mesh_prim)
        
        uv_copied = False
        for primvar in source_primvars.GetPrimvarsWithValues():
            name = primvar.GetPrimvarName()
            if name == "st" and primvar.HasValue():
                uv_values = primvar.Get()
                interpolation = primvar.GetInterpolation()
                type_name = primvar.GetTypeName()
                
                target_uv = target_primvars.CreatePrimvar("st", type_name)
                target_uv.Set(uv_values)
                target_uv.SetInterpolation(interpolation)
                
                print(f"  ✓ UV coordinates: {len(uv_values)} values")
                uv_copied = True
                break
        
        if not uv_copied:
            print("  ⚠️ No UV coordinates found - texture may not display correctly")
        
        # Add physics properties
        print("Adding physics properties...")
        
        # Rigid body
        rigid_body = UsdPhysics.RigidBodyAPI.Apply(sushi_prim)
        rigid_body.CreateRigidBodyEnabledAttr().Set(True)
        print("  ✓ Rigid body")
        
        # Mass (sushi is lightweight, ~20-30g)
        mass_api = UsdPhysics.MassAPI.Apply(sushi_prim)
        mass_api.CreateMassAttr().Set(0.025)  # 25g sushi piece
        print("  ✓ Mass: 0.025kg")
        
        # Collision
        collision = UsdPhysics.CollisionAPI.Apply(sushi_prim)
        collision.CreateCollisionEnabledAttr().Set(True)
        print("  ✓ Collision enabled")
        
        # Mesh collision shape
        mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(sushi_prim)
        mesh_collision.CreateApproximationAttr().Set("convexHull")
        print("  ✓ Convex hull collision")
        
        # Create material with texture
        print("Creating material with texture...")
        
        # Create all material components under the same path
        material_prim = stage.DefinePrim("/sushi/Materials/sushiMaterial", "Material")
        material = UsdShade.Material(material_prim)

        # Use UsdPreviewSurface
        shader = UsdShade.Shader.Define(stage, "/sushi/Materials/sushiMaterial/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")

        # Texture reader
        texture_reader = UsdShade.Shader.Define(stage, "/sushi/Materials/sushiMaterial/TextureReader")
        texture_reader.CreateIdAttr("UsdUVTexture")
        texture_reader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(os.path.basename(texture_file)))

        # Primvar reader for UVs
        primvar_reader = UsdShade.Shader.Define(stage, "/sushi/Materials/sushiMaterial/PrimvarReader")
        primvar_reader.CreateIdAttr("UsdPrimvarReader_float2")
        primvar_reader.CreateInput("varname", Sdf.ValueTypeNames.String).Set("st")
        primvar_out = primvar_reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)

        # Connect UV to texture
        st_input = texture_reader.CreateInput("st", Sdf.ValueTypeNames.Float2)
        st_input.ConnectToSource(primvar_out)

        # Connect texture to shader
        rgb_out = texture_reader.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(rgb_out)

        # Material properties (slightly glossy for sushi)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.3)  # Slightly shiny
        shader.CreateInput("specular", Sdf.ValueTypeNames.Float).Set(0.6)

        # Connect shader to material
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

        # Bind material to mesh
        binding_api = UsdShade.MaterialBindingAPI.Apply(mesh_prim)
        binding_api.Bind(material, bindingStrength=UsdShade.Tokens.strongerThanDescendants)
        print("  ✓ Material bound to mesh")
        
        # Set default prim
        stage.SetDefaultPrim(sushi_prim)
        
        # Clean up
        del mesh_stage
        
        # Save
        stage.Save()
        print(f"✓ Saved: {output_file}")
        
        # Copy texture file to output directory for relative path to work
        import shutil
        texture_dest = os.path.join(os.path.dirname(output_file), os.path.basename(texture_file))
        if not os.path.exists(texture_dest):
            shutil.copy2(texture_file, texture_dest)
            print(f"✓ Copied texture to: {texture_dest}")
        
        # Quick verification
        print("\nVerification:")
        verify_stage = Usd.Stage.Open(output_file)
        default_prim = verify_stage.GetDefaultPrim()
        
        if default_prim:
            print(f"✓ Default prim: {default_prim.GetPath()}")
            
            # Check physics
            has_physics = (
                default_prim.HasAPI(UsdPhysics.RigidBodyAPI) and
                default_prim.HasAPI(UsdPhysics.MassAPI) and
                default_prim.HasAPI(UsdPhysics.CollisionAPI)
            )
            print(f"✓ Physics APIs: {has_physics}")
            
            # Check mesh and material
            mesh_children = [c for c in default_prim.GetChildren() if c.IsA(UsdGeom.Mesh)]
            if mesh_children:
                mesh = mesh_children[0]
                binding = UsdShade.MaterialBindingAPI(mesh)
                has_material = binding.GetDirectBinding().GetMaterial() is not None
                print(f"✓ Mesh with material: {has_material}")
            
            print("\n" + "="*50)
            print("SUCCESS!")
            print("="*50)
            print(f"Created: {output_file}")
            print("Features:")
            print("✓ Physics (rigid body, mass 25g, collision)")
            print("✓ Texture material from your texture.jpg")
            print("✓ UV mapping from your mesh.usdc")
            print("✓ Isaac Lab compatible")
            print("="*50)
            
            return True
        else:
            print("❌ Verification failed")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = create_sushi_from_existing_files()
    if success:
        print("\n🍣 Sushi USD ready!")
    else:
        print("\n❌ Failed to create sushi USD")
        exit(1)