import omni
from pxr import UsdGeom, Gf, Usd

def setup_demo_scene():
    """Create a simple scene with a cube for the demo"""
    print("Setting up Isaac Sim scene for demo...")
    
    # Get the stage
    stage = omni.usd.get_context().get_stage()
    
    # Create a cube if it doesn't exist
    cube_path = "/World/Cube"
    cube_prim = stage.GetPrimAtPath(cube_path)
    
    if not cube_prim or not cube_prim.IsValid():
        print("Creating cube...")
        # Create a cube primitive
        cube_geom = UsdGeom.Cube.Define(stage, cube_path)
        cube_geom.CreateSizeAttr(1.0)  # 1x1x1 cube
        
        # Position it at origin
        xform = UsdGeom.Xformable(cube_geom)
        xform.AddTranslateOp().Set(Gf.Vec3f(0, 0, 0))
        
        print("✓ Cube created at /World/Cube")
    else:
        print("✓ Cube already exists")
    
    print("✓ Scene setup complete!")
    print("\nNow run standalone_pkl_demo.py to start the navigation demo!")

if __name__ == "__main__":
    setup_demo_scene()
