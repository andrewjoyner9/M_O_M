from pxr import UsdGeom, Gf
import omni
import asyncio
import time
import carb

class IsaacSimPathfinder:
    def __init__(self, cube_path="/World/Cube"):
        """Initialize the Isaac Sim pathfinder with cube path"""
        self.cube_path = cube_path
        self.stage = omni.usd.get_context().get_stage()
        self.cube_prim = self.stage.GetPrimAtPath(cube_path)
        self.movement_delay = 1  # Delay between movements in seconds
        
        # Path movement state
        self.current_path = []
        self.current_step = 0
        self.is_moving = False
        self.keyboard_subscription = None
        
    def get_cube_position(self):
        """Get the current position of the cube"""
        if not self.cube_prim:
            print(f"Error: Cube not found at {self.cube_path}")
            return Gf.Vec3f(0, 0, 0)
        
        xform = UsdGeom.Xformable(self.cube_prim)
        translate_ops = [op for op in xform.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeTranslate]
        if translate_ops:
            return translate_ops[0].Get()
        return Gf.Vec3f(0, 0, 0)

    def set_cube_position(self, x, y, z):
        """Set the position of the cube"""
        if not self.cube_prim:
            print(f"Error: Cube not found at {self.cube_path}")
            return False
        
        xform = UsdGeom.Xformable(self.cube_prim)
        translate_ops = [op for op in xform.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeTranslate]
        
        if translate_ops:
            translate_ops[0].Set(Gf.Vec3f(x, y, z))
        else:
            xform.AddTranslateOp().Set(Gf.Vec3f(x, y, z))
        
        print(f"Cube moved to position: ({x}, {y}, {z})")
        return True

    def find_path(self, start, end):
        """
        Find a path from start to end coordinates using simple line algorithm
        Returns a list of 3D coordinate tuples representing the path
        """
        path = []
        x, y, z = start
        end_x, end_y, end_z = end

        print(f"Planning path from {start} to {end}")
        
        while (x, y, z) != (end_x, end_y, end_z):
            # Move in X direction first
            if x < end_x:
                x += 1
                print(f"Incrementing x by 1: now x = {x}")
            elif x > end_x:
                x -= 1
                print(f"Decrementing x by 1: now x = {x}")

            # Then move in Y direction
            if y < end_y:
                y += 1
                print(f"Incrementing y by 1: now y = {y}")
            elif y > end_y:
                y -= 1
                print(f"Decrementing y by 1: now y = {y}")

            # Finally move in Z direction
            if z < end_z:
                z += 1
                print(f"Incrementing z by 1: now z = {z}")
            elif z > end_z:
                z -= 1
                print(f"Decrementing z by 1: now z = {z}")

            path.append((x, y, z))
        
        print(f"Path calculated with {len(path)} steps")
        return path

    def on_keyboard_event(self, event):
        """Handle keyboard events for step-by-step movement"""
        try:
            import carb
            if event.type == carb.input.KeyboardEventType.KEY_PRESS:
                if event.input == carb.input.KeyboardInput.ENTER and self.is_moving:
                    self.advance_to_next_step()
                    return True
        except ImportError:
            # Fallback for when carb is not available
            pass
        return False

    def setup_keyboard_controls(self):
        """Set up keyboard controls for step-by-step movement"""
        try:
            import omni.appwindow
            import carb
            
            app_window = omni.appwindow.get_default_app_window()
            keyboard = app_window.get_keyboard()
            
            if keyboard:
                input_interface = carb.input.acquire_input_interface()
                self.keyboard_subscription = input_interface.subscribe_to_keyboard_events(keyboard, self.on_keyboard_event)
                print("Keyboard controls activated! Press ENTER to advance to next step.")
                return True
            else:
                print("Failed to get keyboard from app window")
                return False
        except Exception as e:
            print(f"Failed to setup keyboard controls: {e}")
            return False

    def cleanup_keyboard_controls(self):
        """Clean up keyboard controls"""
        if self.keyboard_subscription:
            try:
                self.keyboard_subscription.unsubscribe()
                self.keyboard_subscription = None
                print("Keyboard controls deactivated")
            except Exception as e:
                print(f"Error cleaning up keyboard controls: {e}")

    def advance_to_next_step(self):
        """Move to the next step in the path"""
        if not self.is_moving or self.current_step >= len(self.current_path):
            return
        
        x, y, z = self.current_path[self.current_step]
        print(f"Step {self.current_step + 1}/{len(self.current_path)}: Moving to ({x}, {y}, {z})")
        self.set_cube_position(x, y, z)
        
        self.current_step += 1
        
        if self.current_step >= len(self.current_path):
            print("Path movement completed!")
            self.is_moving = False
            self.cleanup_keyboard_controls()
        else:
            print(f"Press ENTER to continue to step {self.current_step + 1}...")

    def move_along_path_interactive(self, path):
        """
        Move the cube along the calculated path with keyboard input for each step
        """
        if not path:
            print("No path to follow!")
            return
        
        print(f"Starting interactive movement along path with {len(path)} steps...")
        print("Press ENTER to advance to each step...")
        
        self.current_path = path
        self.current_step = 0
        self.is_moving = True
        
        # Set up keyboard controls
        if self.setup_keyboard_controls():
            # Move to first step
            self.advance_to_next_step()
        else:
            print("Failed to set up keyboard controls. Falling back to manual mode.")
            print("Call pathfinder.advance_to_next_step() manually to advance.")
            # Still set up the state for manual advancement
            self.advance_to_next_step()

    async def move_along_path_async(self, path):
        """
        Move the cube along the calculated path with keyboard input for each step (non-blocking)
        """
        print("Note: Using interactive keyboard-based movement instead of async delays")
        self.move_along_path_interactive(path)

    def move_along_path_blocking(self, path):
        """
        Move the cube along the calculated path with keyboard input for each step
        """
        print("Note: Using interactive keyboard-based movement instead of blocking delays")
        self.move_along_path_interactive(path)

    def move_to_target(self, target_coordinates, use_async=True):
        """
        Move the cube from its current position to target coordinates
        
        Args:
            target_coordinates: (x, y, z) tuple of target position
            use_async: Whether to use async movement (default: True)
        """
        # Get current position and convert to integer coordinates for pathfinding
        current_pos = self.get_cube_position()
        start_coordinates = (int(current_pos[0]), int(current_pos[1]), int(current_pos[2]))
        
        print(f"Current cube position: {current_pos}")
        print(f"Start coordinates (rounded): {start_coordinates}")
        print(f"Target coordinates: {target_coordinates}")
        
        # Calculate path
        path = self.find_path(start_coordinates, target_coordinates)
        
        if not path:
            print("No movement needed - already at target!")
            return
        
        # Move along path
        if use_async:
            # Both async and blocking now use the same interactive method
            self.move_along_path_interactive(path)
        else:
            self.move_along_path_interactive(path)

    def set_movement_speed(self, delay_seconds):
        """Set the delay between movements (lower = faster movement) - now deprecated as we use user input"""
        self.movement_delay = max(0.1, delay_seconds)  # Minimum 0.1 seconds
        print(f"Movement delay set to {self.movement_delay} seconds (Note: Now using keyboard input instead of time delays)")

    def stop_movement(self):
        """Stop the current movement and clean up"""
        self.is_moving = False
        self.cleanup_keyboard_controls()
        print("Movement stopped")

    def get_movement_status(self):
        """Get the current movement status"""
        if self.is_moving:
            return f"Moving: Step {self.current_step}/{len(self.current_path)}"
        else:
            return "Not moving"


# Convenience functions for easy usage
def move_cube_to_target(target_coordinates, movement_delay=0.5, use_async=True):
    """
    Convenience function to move cube to target coordinates
    
    Args:
        target_coordinates: (x, y, z) tuple of target position
        movement_delay: Delay between movements in seconds (deprecated - now uses user input)
        use_async: Whether to use async movement
    """
    pathfinder = IsaacSimPathfinder()
    pathfinder.set_movement_speed(movement_delay)
    pathfinder.move_to_target(target_coordinates, use_async)


# Example usage and test functions
def test_basic_movement():
    """Test basic movement functionality"""
    print("=== Testing Basic Movement ===")
    pathfinder = IsaacSimPathfinder()
    
    # Test getting current position
    current_pos = pathfinder.get_cube_position()
    print(f"Current position: {current_pos}")
    
    # Test setting position
    pathfinder.set_cube_position(1, 1, 1)
    
    # Test getting position again
    new_pos = pathfinder.get_cube_position()
    print(f"New position: {new_pos}")


def test_pathfinding():
    """Test pathfinding from (0,0,0) to (5,5,5)"""
    print("=== Testing Pathfinding ===")
    pathfinder = IsaacSimPathfinder()
    
    # Set cube to starting position
    start_pos = (0, 0, 0)
    pathfinder.set_cube_position(*start_pos)
    
    # Move to target
    target_pos = (5, 5, 5)
    pathfinder.move_to_target(target_pos, use_async=False)


def test_custom_path():
    """Test with custom start and end coordinates"""
    print("=== Testing Custom Path ===")
    pathfinder = IsaacSimPathfinder()
    pathfinder.set_movement_speed(0.3)  # Faster movement
    
    # Set custom starting position
    start_pos = (2, -1, 3)
    pathfinder.set_cube_position(*start_pos)
    
    # Move to custom target
    target_pos = (-2, 4, 0)
    pathfinder.move_to_target(target_pos, use_async=False)


if __name__ == "__main__":
    print("=== Isaac Sim Pathfinder ===")
    print("This script moves a cube along a calculated path in Isaac Sim 4.5.0")
    print()
    
    # Example usage:
    print("Example 1: Move from current position to (5, 5, 5)")
    move_cube_to_target((5, 5, 5), movement_delay=0.5)
    
    print("\nExample 2: Test basic functionality")
    # Uncomment the line below to test basic movement
    # test_basic_movement()
    
    print("\nExample 3: Test pathfinding algorithm")
    # Uncomment the line below to test pathfinding
    # test_pathfinding()
    
    print("\nExample 4: Test custom path")
    # Uncomment the line below to test custom path
    # test_custom_path()
