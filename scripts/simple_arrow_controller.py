from pxr import UsdGeom, Gf
import omni
import carb

def get_cube_position():
    """Get the current position of the cube"""
    stage = omni.usd.get_context().get_stage() # Get the current Stage and the cube prim
    cube_prim = stage.GetPrimAtPath("/World/Cube")
    
    if not cube_prim:
        return Gf.Vec3f(0, 0, 0)
    
    xform = UsdGeom.Xformable(cube_prim) # Get the transformable object for the cube so we can manipulate its position
    translate_ops = [op for op in xform.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeTranslate] # This retrieves the translation operations applied to the cube
    if translate_ops:
        return translate_ops[0].Get() 
    return Gf.Vec3f(0, 0, 0)

def set_cube_position(x, y, z):
    """Set the position of the cube"""
    stage = omni.usd.get_context().get_stage()
    cube_prim = stage.GetPrimAtPath("/World/Cube")
    
    if not cube_prim:
        print("Error: Cube not found at /World/Cube")
        return False
    
    xform = UsdGeom.Xformable(cube_prim)
    translate_ops = [op for op in xform.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeTranslate] 

    if translate_ops: # If a translation operation exists, set its value
        translate_ops[0].Set(Gf.Vec3f(x, y, z)) 
    else:
        xform.AddTranslateOp().Set(Gf.Vec3f(x, y, z))
    
    print(f"Cube moved to position: ({x}, {y}, {z})")
    return True

def increment_cube_x():
    """Increment the cube's X position by 1"""
    current_pos = get_cube_position()
    new_x = current_pos[0] + 1.0
    set_cube_position(new_x, current_pos[1], current_pos[2])

def decrement_cube_x():
    """Decrement the cube's X position by 1"""
    current_pos = get_cube_position()
    new_x = current_pos[0] - 1.0
    set_cube_position(new_x, current_pos[1], current_pos[2])

def increment_cube_y():
    """Increment the cube's Y position by 1"""
    current_pos = get_cube_position()
    new_y = current_pos[1] + 1.0
    set_cube_position(current_pos[0], new_y, current_pos[2])

def decrement_cube_y():
    """Decrement the cube's Y position by 1"""
    current_pos = get_cube_position()
    new_y = current_pos[1] - 1.0
    set_cube_position(current_pos[0], new_y, current_pos[2])

def increment_cube_z():
    """Increment the cube's Z position by 1"""
    current_pos = get_cube_position()
    new_z = current_pos[2] + 1.0
    set_cube_position(current_pos[0], current_pos[1], new_z)

def decrement_cube_z():
    """Decrement the cube's Z position by 1"""
    current_pos = get_cube_position()
    new_z = current_pos[2] - 1.0
    set_cube_position(current_pos[0], current_pos[1], new_z)

def on_keyboard_event(event):
    """Handle keyboard events for directional movement"""
    if event.type == carb.input.KeyboardEventType.KEY_PRESS: # Check if a key was pressed
        if event.input == carb.input.KeyboardInput.UP: 
            decrement_cube_x()
            return True
        elif event.input == carb.input.KeyboardInput.DOWN:
            increment_cube_x()
            return True
        elif event.input == carb.input.KeyboardInput.LEFT:
            decrement_cube_y()
            return True
        elif event.input == carb.input.KeyboardInput.RIGHT:
            increment_cube_y()
            return True
        elif event.input == carb.input.KeyboardInput.PAGE_UP:
            increment_cube_z()
            return True
        elif event.input == carb.input.KeyboardInput.PAGE_DOWN:
            decrement_cube_z()
            return True
    return False

def setup_arrow_key_controls():
    """Set up arrow key controls for cube movement"""
    import omni.appwindow
    
    try:
        # Get the app window and keyboard interface
        app_window = omni.appwindow.get_default_app_window()
        keyboard = app_window.get_keyboard()
        
        if keyboard:
            # Get input interface
            input_interface = carb.input.acquire_input_interface()
            
            # Subscribe to keyboard events
            keyboard_sub = input_interface.subscribe_to_keyboard_events(keyboard, on_keyboard_event)
            print("Full directional controls activated!")
            print("Arrow Keys: UP/DOWN = X-axis, LEFT/RIGHT = Y-axis")
            print("Page Up/Down: Z-axis movement")
            return keyboard_sub
        else:
            print("Failed to get keyboard from app window")
            return None
    except Exception as e:
        print(f"Failed to setup keyboard controls: {e}")
        return None

# Global variable to keep subscription alive
_keyboard_subscription = None

def start_arrow_controls():
    """Start listening for arrow key input"""
    global _keyboard_subscription # Make sure to use the global variable
    try:
        _keyboard_subscription = setup_arrow_key_controls() 
        return True
    except Exception as e:
        print(f"Error setting up arrow controls: {e}")
        return False

def stop_arrow_controls():
    """Stop listening for arrow key input"""
    global _keyboard_subscription
    if _keyboard_subscription:
        try:
            _keyboard_subscription.unsubscribe()
            _keyboard_subscription = None
            print("Arrow key controls deactivated")
        except Exception as e:
            print(f"Error stopping arrow controls: {e}")

def test_manual_increment():
    """Test function to manually test all movement directions"""
    print("Testing manual movement in all directions...")
    current_pos = get_cube_position()
    print(f"Current position: {current_pos}")
    
    print("Testing X+ movement...")
    increment_cube_x()
    print(f"Position after X+: {get_cube_position()}")
    
    print("Testing X- movement...")
    decrement_cube_x()
    print(f"Position after X-: {get_cube_position()}")
    
    print("Testing Y+ movement...")
    increment_cube_y()
    print(f"Position after Y+: {get_cube_position()}")
    
    print("Testing Y- movement...")
    decrement_cube_y()
    print(f"Position after Y-: {get_cube_position()}")
    
    print("Testing Z+ movement...")
    increment_cube_z()
    print(f"Position after Z+: {get_cube_position()}")
    
    print("Testing Z- movement...")
    decrement_cube_z()
    print(f"Position after Z-: {get_cube_position()}")
    
    print("Movement tests complete!")

# Example usage:
if __name__ == "__main__":
    print("=== Full Directional Cube Control ===")
    print("This script allows you to control cube movement with keyboard input")
    print("Arrow Keys:")
    print("  UP ARROW    = Move backward (-X)")
    print("  DOWN ARROW  = Move forward (+X)")
    print("  LEFT ARROW  = Move left (-Y)")
    print("  RIGHT ARROW = Move right (+Y)")
    print("Page Keys:")
    print("  PAGE UP     = Move up (+Z)")
    print("  PAGE DOWN   = Move down (-Z)")
    print()
    
    # Test that we can find and move the cube
    current_pos = get_cube_position()
    print(f"Current cube position: {current_pos}")
    
    # Start arrow key controls
    success = start_arrow_controls()
    if success:
        print("\nFull directional controls are now active!")
        print("Use arrow keys and Page Up/Down to move the cube in 3D space")
    else:
        print("\nFailed to start directional controls")
        print("You can still test with: test_manual_increment()")