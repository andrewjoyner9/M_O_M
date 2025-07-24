

start_coordinates = (0, 0, 0)
end_coordinates = (5, 5, 5)

def find_path(start, end):
    path = []
    x, y, z = start
    end_x, end_y, end_z = end

    while (x, y, z) != (end_x, end_y, end_z):
        if x < end_x:
            x += 1
            print(f"incrementing x by 1: now x = {x}")
        elif x > end_x:
            x -= 1
            print(f"decrementing x by 1: now x = {x}")

        if y < end_y:
            y += 1
            print(f"incrementing y by 1: now y = {y}")
        elif y > end_y:
            y -= 1
            print(f"decrementing y by 1: now y = {y}")

        if z < end_z:
            z += 1
            print(f"incrementing z by 1: now z = {z}")
        elif z > end_z:
            z -= 1
            print(f"decrementing z by 1: now z = {z}")

        path.append((x, y, z))
    
    return path

def move_cube_isaac_sim(path, movement_delay=0.5):
    """
    Move the cube in Isaac Sim along the calculated path
    This function should be called from within Isaac Sim
    """
    try:
        from pxr import UsdGeom, Gf
        import omni
        import time
        
        # Get the stage and cube
        stage = omni.usd.get_context().get_stage()
        cube_prim = stage.GetPrimAtPath("/World/Cube")
        
        if not cube_prim:
            print("Error: Cube not found at /World/Cube")
            return False
        
        print(f"Moving cube along path with {len(path)} steps...")
        
        for i, (x, y, z) in enumerate(path):
            print(f"Step {i+1}/{len(path)}: Moving to ({x}, {y}, {z})")
            
            # Set cube position
            xform = UsdGeom.Xformable(cube_prim)
            translate_ops = [op for op in xform.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeTranslate]
            
            if translate_ops:
                translate_ops[0].Set(Gf.Vec3f(x, y, z))
            else:
                xform.AddTranslateOp().Set(Gf.Vec3f(x, y, z))
            
            # Wait before next movement
            time.sleep(movement_delay)
        
        print("Path movement completed!")
        return True
        
    except ImportError:
        print("Isaac Sim imports not available. Run this from within Isaac Sim.")
        return False
    except Exception as e:
        print(f"Error moving cube: {e}")
        return False

def execute_path_in_isaac_sim(start_coords, end_coords, movement_delay=0.5):
    """
    Complete function to calculate path and move cube in Isaac Sim
    """
    # Calculate the path
    path = find_path(start_coords, end_coords)
    print("Path found:", path)
    
    # Move the cube along the path
    success = move_cube_isaac_sim(path, movement_delay)
    
    if success:
        print("Cube successfully moved to target!")
    else:
        print("Failed to move cube in Isaac Sim")
    
    return path

if __name__ == "__main__":
    path = find_path(start_coordinates, end_coordinates)
    print("Path found:", path)
    print("Final coordinates:", path[-1] if path else start_coordinates)
    
    # To use in Isaac Sim, call:
    # execute_path_in_isaac_sim(start_coordinates, end_coordinates)
