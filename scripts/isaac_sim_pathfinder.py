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
        
        # Obstacle detection
        self.obstacles = set()  # Set of (x, y, z) tuples representing obstacle positions
        self.obstacle_prims = []  # List of obstacle primitive paths to check
        self.detection_radius = 0.5  # Radius for obstacle detection
        
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

    def add_obstacle(self, x, y, z):
        """Add an obstacle at the specified coordinates"""
        self.obstacles.add((int(x), int(y), int(z)))
        print(f"Obstacle added at ({x}, {y}, {z})")

    def remove_obstacle(self, x, y, z):
        """Remove an obstacle at the specified coordinates"""
        coord = (int(x), int(y), int(z))
        if coord in self.obstacles:
            self.obstacles.remove(coord)
            print(f"Obstacle removed from ({x}, {y}, {z})")
        else:
            print(f"No obstacle found at ({x}, {y}, {z})")

    def clear_obstacles(self):
        """Clear all manually added obstacles"""
        self.obstacles.clear()
        print("All manual obstacles cleared")

    def add_obstacle_prim_path(self, prim_path):
        """Add a primitive path to check for obstacles"""
        if prim_path not in self.obstacle_prims:
            self.obstacle_prims.append(prim_path)
            print(f"Added obstacle primitive: {prim_path}")

    def detect_obstacles_from_scene(self):
        """Detect obstacles from the Isaac Sim scene"""
        detected_obstacles = set()
        
        for prim_path in self.obstacle_prims:
            prim = self.stage.GetPrimAtPath(prim_path)
            if prim and prim.IsValid():
                try:
                    xform = UsdGeom.Xformable(prim)
                    translate_ops = [op for op in xform.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeTranslate]
                    if translate_ops:
                        pos = translate_ops[0].Get()
                        obstacle_coord = (int(pos[0]), int(pos[1]), int(pos[2]))
                        detected_obstacles.add(obstacle_coord)
                except Exception as e:
                    print(f"Error detecting obstacle at {prim_path}: {e}")
        
        # Update obstacles with detected ones
        self.obstacles.update(detected_obstacles)
        return detected_obstacles

    def is_position_blocked(self, x, y, z):
        """Check if a position is blocked by an obstacle"""
        coord = (int(x), int(y), int(z))
        return coord in self.obstacles

    def get_neighbors(self, x, y, z):
        """Get valid neighboring positions (not blocked by obstacles)"""
        neighbors = []
        # 6-directional movement (up, down, left, right, forward, backward)
        directions = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
        
        for dx, dy, dz in directions:
            new_x, new_y, new_z = x + dx, y + dy, z + dz
            if not self.is_position_blocked(new_x, new_y, new_z):
                neighbors.append((new_x, new_y, new_z))
        
        return neighbors

    def find_path(self, start, end):
        """
        Find a path from start to end coordinates using A* algorithm with obstacle avoidance
        Returns a list of 3D coordinate tuples representing the path
        """
        import heapq
        
        # Detect obstacles from scene first
        self.detect_obstacles_from_scene()
        
        print(f"Planning path from {start} to {end}")
        print(f"Known obstacles: {len(self.obstacles)} positions")
        
        # Check if start or end positions are blocked
        if self.is_position_blocked(*start):
            print(f"Error: Start position {start} is blocked by an obstacle!")
            return []
        
        if self.is_position_blocked(*end):
            print(f"Error: End position {end} is blocked by an obstacle!")
            return []
        
        # A* algorithm implementation
        def heuristic(pos1, pos2):
            """Manhattan distance heuristic"""
            return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]) + abs(pos1[2] - pos2[2])
        
        # Priority queue: (f_score, g_score, position)
        open_set = [(0, 0, start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, end)}
        closed_set = set()
        
        while open_set:
            # Get position with lowest f_score
            current_f, current_g, current = heapq.heappop(open_set)
            
            if current in closed_set:
                continue
                
            if current == end:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                print(f"Path found with {len(path)} steps")
                return path
            
            closed_set.add(current)
            
            # Check all neighbors
            for neighbor in self.get_neighbors(*current):
                if neighbor in closed_set:
                    continue
                
                tentative_g = g_score[current] + 1
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + heuristic(neighbor, end)
                    heapq.heappush(open_set, (f_score[neighbor], tentative_g, neighbor))
        
        print("No path found! Target may be unreachable due to obstacles.")
        return []

    def find_path_simple(self, start, end):
        """
        Find a path using simple line algorithm (ignores obstacles) - kept for fallback
        Returns a list of 3D coordinate tuples representing the path
        """
        path = []
        x, y, z = start
        end_x, end_y, end_z = end

        print(f"Planning simple path from {start} to {end}")
        
        while (x, y, z) != (end_x, end_y, end_z):
            # Move in X direction first
            if x < end_x:
                x += 1
            elif x > end_x:
                x -= 1

            # Then move in Y direction
            if y < end_y:
                y += 1
            elif y > end_y:
                y -= 1

            # Finally move in Z direction
            if z < end_z:
                z += 1
            elif z > end_z:
                z -= 1

            path.append((x, y, z))
        
        print(f"Simple path calculated with {len(path)} steps")
        return path

    def validate_path(self, path):
        """Check if a path is clear of obstacles"""
        blocked_positions = []
        for step, (x, y, z) in enumerate(path):
            if self.is_position_blocked(x, y, z):
                blocked_positions.append((step, (x, y, z)))
        
        if blocked_positions:
            print(f"Warning: Path has {len(blocked_positions)} blocked positions:")
            for step, pos in blocked_positions:
                print(f"  Step {step + 1}: {pos}")
            return False
        return True

    def replan_if_blocked(self, start, end):
        """Try to find an alternative path if the current path is blocked"""
        print("Attempting to replan path to avoid obstacles...")
        
        # First try A* algorithm
        path = self.find_path(start, end)
        if path and self.validate_path(path):
            return path
        
        # If A* fails, try simple path as fallback
        print("A* pathfinding failed, trying simple pathfinding...")
        simple_path = self.find_path_simple(start, end)
        if self.validate_path(simple_path):
            return simple_path
        
        print("Both pathfinding methods failed due to obstacles.")
        return []

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
        
        # Check if the next position is blocked by an obstacle
        if self.is_position_blocked(x, y, z):
            print(f"Warning: Next position ({x}, {y}, {z}) is blocked by an obstacle!")
            print("Attempting to replan path...")
            
            # Get current position and remaining target
            current_pos = self.get_cube_position()
            current_coords = (int(current_pos[0]), int(current_pos[1]), int(current_pos[2]))
            
            # Find the final destination from the remaining path
            if self.current_path:
                final_destination = self.current_path[-1]
                
                # Try to find a new path
                new_path = self.replan_if_blocked(current_coords, final_destination)
                if new_path:
                    print("Successfully replanned path!")
                    self.current_path = new_path
                    self.current_step = 0
                    # Continue with the new path
                    x, y, z = self.current_path[self.current_step]
                else:
                    print("Failed to find alternative path. Stopping movement.")
                    self.is_moving = False
                    self.cleanup_keyboard_controls()
                    return
            else:
                print("No remaining path to replan. Stopping movement.")
                self.is_moving = False
                self.cleanup_keyboard_controls()
                return
        
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

    def create_wall_obstacle(self, start_pos, end_pos, axis='x'):
        """Create a wall of obstacles between two points along a specified axis"""
        start_x, start_y, start_z = start_pos
        end_x, end_y, end_z = end_pos
        
        obstacles_added = 0
        
        if axis == 'x':
            for x in range(min(start_x, end_x), max(start_x, end_x) + 1):
                self.add_obstacle(x, start_y, start_z)
                obstacles_added += 1
        elif axis == 'y':
            for y in range(min(start_y, end_y), max(start_y, end_y) + 1):
                self.add_obstacle(start_x, y, start_z)
                obstacles_added += 1
        elif axis == 'z':
            for z in range(min(start_z, end_z), max(start_z, end_z) + 1):
                self.add_obstacle(start_x, start_y, z)
                obstacles_added += 1
        
        print(f"Created wall obstacle with {obstacles_added} blocks along {axis}-axis")

    def get_obstacle_info(self):
        """Get information about current obstacles"""
        manual_obstacles = len(self.obstacles)
        scene_prims = len(self.obstacle_prims)
        
        print(f"Obstacle Status:")
        print(f"  Manual obstacles: {manual_obstacles}")
        print(f"  Scene primitive paths: {scene_prims}")
        print(f"  Detection radius: {self.detection_radius}")
        
        if self.obstacles:
            print(f"  Obstacle positions: {sorted(list(self.obstacles))}")
        
        return {
            'manual_obstacles': manual_obstacles,
            'scene_prims': scene_prims,
            'positions': list(self.obstacles),
            'detection_radius': self.detection_radius
        }


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


def test_obstacle_avoidance():
    """Test pathfinding with obstacles"""
    print("=== Testing Obstacle Avoidance ===")
    pathfinder = IsaacSimPathfinder()
    
    # Add some obstacles
    pathfinder.add_obstacle(1, 0, 1)
    pathfinder.add_obstacle(2, 0, 0)
    pathfinder.add_obstacle(3, 0, 0)
    pathfinder.add_obstacle(1, 1, 0)
    pathfinder.add_obstacle(2, 1, 0)
    
    print(f"Added {len(pathfinder.obstacles)} obstacles")
    
    # Set starting position
    start_pos = (0, 0, 1)
    pathfinder.set_cube_position(*start_pos)
    
    # Try to move to target (should go around obstacles)
    target_pos = (4, 0, 1)
    pathfinder.move_to_target(target_pos, use_async=False)


def test_scene_obstacle_detection():
    """Test detecting obstacles from scene primitives"""
    print("=== Testing Scene Obstacle Detection ===")
    pathfinder = IsaacSimPathfinder()
    
    # Add some primitive paths that might contain obstacles
    # Note: These paths should exist in your Isaac Sim scene
    pathfinder.add_obstacle_prim_path("/World/Obstacle1")
    pathfinder.add_obstacle_prim_path("/World/Obstacle2")
    pathfinder.add_obstacle_prim_path("/World/Wall")
    
    # Detect obstacles from scene
    detected = pathfinder.detect_obstacles_from_scene()
    print(f"Detected {len(detected)} obstacles from scene")
    
    # Set starting position
    start_pos = (0, 0, 0)
    pathfinder.set_cube_position(*start_pos)
    
    # Move to target
    target_pos = (5, 5, 0)
    pathfinder.move_to_target(target_pos, use_async=False)


def test_dynamic_obstacles():
    """Test adding obstacles dynamically during movement"""
    print("=== Testing Dynamic Obstacles ===")
    pathfinder = IsaacSimPathfinder()
    
    # Set starting position
    start_pos = (0, 0, 0)
    pathfinder.set_cube_position(*start_pos)
    
    # Start movement to target
    target_pos = (5, 0, 0)
    print("Starting movement...")
    
    # Calculate initial path
    path = pathfinder.find_path(start_pos, target_pos)
    if path:
        pathfinder.current_path = path
        pathfinder.current_step = 0
        pathfinder.is_moving = True
        
        print(f"Initial path: {path}")
        print("You can now add obstacles using:")
        print("  pathfinder.add_obstacle(x, y, z)")
        print("Then continue movement with:")
        print("  pathfinder.advance_to_next_step()")
    else:
        print("Failed to find initial path")


if __name__ == "__main__":
    print("=== Isaac Sim Pathfinder with Obstacle Avoidance ===")
    print("This script moves a cube along a calculated path in Isaac Sim 4.5.0")
    print("Now includes obstacle detection and avoidance capabilities!")
    print()
    
    # Example usage:
    # print("Example 1: Move from current position to (5, 5, 5)")
    # move_cube_to_target((5, 5, 5), movement_delay=0.5)
    
    # print("\nExample 2: Test basic functionality")
    # Uncomment the line below to test basic movement
    # test_basic_movement()

    # print("\nExample 3: Test pathfinding algorithm")
    # Uncomment the line below to test pathfinding
    # test_pathfinding()

    # print("\nExample 4: Test custom path")
    # Uncomment the line below to test custom path
    # test_custom_path()
    
    print("\nExample 5: Test obstacle avoidance")
    # Uncomment the line below to test obstacle avoidance
    test_obstacle_avoidance()
    
    # print("\nExample 6: Test scene obstacle detection")
    # Uncomment the line below to test scene-based obstacle detection
    # test_scene_obstacle_detection()
    
    # print("\nExample 7: Test dynamic obstacles")
    # Uncomment the line below to test dynamic obstacle handling
    # test_dynamic_obstacles()
    
    print("\n" + "="*50)
    print("OBSTACLE DETECTION FEATURES:")
    print("="*50)
    print("• add_obstacle(x, y, z) - Add manual obstacle")
    print("• remove_obstacle(x, y, z) - Remove obstacle")
    print("• clear_obstacles() - Clear all manual obstacles")
    print("• add_obstacle_prim_path(path) - Track scene objects as obstacles")
    print("• detect_obstacles_from_scene() - Auto-detect from scene")
    print("• is_position_blocked(x, y, z) - Check if position has obstacle")
    print("• validate_path(path) - Check if path is clear")
    print("• replan_if_blocked(start, end) - Find alternative path")
    print("="*50)
