from pxr import UsdGeom, Gf
import omni
import carb

def get_cube_position():
    """Get the current position of the cube"""
    stage = omni.usd.get_context().get_stage()
    cube_prim = stage.GetPrimAtPath("/World/Cube")
    
    if not cube_prim:
        return Gf.Vec3f(0, 0, 0)
    
    xform = UsdGeom.Xformable(cube_prim)
    translate_ops = [op for op in xform.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeTranslate]
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
    if translate_ops:
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

def on_keyboard_event(event):
    """Handle keyboard events - specifically up arrow key"""
    if event.type == carb.input.KeyboardEventType.KEY_PRESS:
        if event.input == carb.input.KeyboardInput.UP:
            increment_cube_x()
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
            print("Arrow key controls activated! Press UP ARROW to increment cube X position by +1")
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
    global _keyboard_subscription
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
    """Test function to manually increment X position"""
    print("Testing manual X increment...")
    current_pos = get_cube_position()
    print(f"Current position: {current_pos}")
    increment_cube_x()
    new_pos = get_cube_position()
    print(f"New position: {new_pos}")

# Example usage:
if __name__ == "__main__":
    print("=== Arrow Key Cube Control ===")
    print("This script allows you to control cube movement with arrow keys")
    print("UP ARROW = Increment X position by +1")
    print()
    
    # Test that we can find and move the cube
    current_pos = get_cube_position()
    print(f"Current cube position: {current_pos}")
    
    # Start arrow key controls
    success = start_arrow_controls()
    if success:
        print("\nArrow key controls are now active!")
        print("Press UP ARROW to move the cube in the +X direction")
    else:
        print("\nFailed to start arrow key controls")
        print("You can still test with: test_manual_increment()")