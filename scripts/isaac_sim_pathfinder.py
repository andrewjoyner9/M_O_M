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

    def add_obstacle(self, x, y, z, create_sphere=True):
        """Add an obstacle at the specified coordinates"""
        coord = (int(x), int(y), int(z))
        self.obstacles.add(coord)
        print(f"Obstacle added at ({x}, {y}, {z})")
        
        # Optionally create a visual sphere for the obstacle
        if create_sphere:
            self._create_single_obstacle_sphere(coord)

    def remove_obstacle(self, x, y, z, remove_sphere=True):
        """Remove an obstacle at the specified coordinates"""
        coord = (int(x), int(y), int(z))
        if coord in self.obstacles:
            self.obstacles.remove(coord)
            print(f"Obstacle removed from ({x}, {y}, {z})")
            
            # Optionally remove the visualization sphere
            if remove_sphere:
                self._remove_single_obstacle_sphere(coord)
        else:
            print(f"No obstacle found at ({x}, {y}, {z})")

    def clear_obstacles(self):
        """Clear all manually added obstacles and their spheres"""
        self.obstacles.clear()
        self.remove_obstacle_spheres()
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
        
        print(f"Planning path from {start} to {end} with {len(self.obstacles)} obstacles")
        
        # Check if start or end positions are blocked
        if self.is_position_blocked(*start):
            print(f"Error: Start position {start} is blocked!")
            return []
        
        if self.is_position_blocked(*end):
            print(f"Error: End position {end} is blocked!")
            return []
        
        # A* algorithm implementation
        def heuristic(pos1, pos2):
            return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]) + abs(pos1[2] - pos2[2])
        
        open_set = [(0, 0, start)]
        came_from = {}
        g_score = {start: 0}
        closed_set = set()
        
        while open_set:
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
            
            for neighbor in self.get_neighbors(*current):
                if neighbor in closed_set:
                    continue
                
                tentative_g = g_score[current] + 1
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor, end)
                    heapq.heappush(open_set, (f_score, tentative_g, neighbor))
        
        print("No path found! Target may be unreachable.")
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
            print(f"Warning: Next position ({x}, {y}, {z}) is blocked!")
            print("Attempting to replan path...")
            
            # Get current position and remaining target
            current_pos = self.get_cube_position()
            current_coords = (int(current_pos[0]), int(current_pos[1]), int(current_pos[2]))
            
            # Find the final destination from the remaining path
            if self.current_path:
                final_destination = self.current_path[-1]
                
                # Try to find a new path
                new_path = self.find_path(current_coords, final_destination)
                if new_path:
                    print("Successfully replanned path!")
                    self.current_path = new_path
                    self.current_step = 0
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

    def move_to_target(self, target_coordinates):
        """
        Move the cube from its current position to target coordinates
        
        Args:
            target_coordinates: (x, y, z) tuple of target position
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
        self.move_along_path_interactive(path)

    def create_wall_obstacle(self, start_pos, end_pos, axis='x', create_spheres=True):
        """Create a wall of obstacles between two points along a specified axis"""
        start_x, start_y, start_z = start_pos
        end_x, end_y, end_z = end_pos
        
        obstacles_added = 0
        
        if axis == 'x':
            for x in range(min(start_x, end_x), max(start_x, end_x) + 1):
                self.add_obstacle(x, start_y, start_z, create_sphere=create_spheres)
                obstacles_added += 1
        elif axis == 'y':
            for y in range(min(start_y, end_y), max(start_y, end_y) + 1):
                self.add_obstacle(start_x, y, start_z, create_sphere=create_spheres)
                obstacles_added += 1
        elif axis == 'z':
            for z in range(min(start_z, end_z), max(start_z, end_z) + 1):
                self.add_obstacle(start_x, start_y, z, create_sphere=create_spheres)
                obstacles_added += 1
        
        print(f"Created wall obstacle with {obstacles_added} blocks along {axis}-axis")
        return obstacles_added

    def _remove_single_obstacle_sphere(self, coord):
        """Remove a single obstacle visualization sphere"""
        try:
            x, y, z = coord
            sphere_path = f"/World/ObstacleSphere_{x}_{y}_{z}"
            material_path = f"/World/Materials/ObstacleMaterial_{x}_{y}_{z}"
            
            # Remove sphere
            sphere_prim = self.stage.GetPrimAtPath(sphere_path)
            if sphere_prim and sphere_prim.IsValid():
                self.stage.RemovePrim(sphere_prim.GetPath())
                print(f"Removed obstacle sphere at ({x}, {y}, {z})")
            
            # Remove material
            material_prim = self.stage.GetPrimAtPath(material_path)
            if material_prim and material_prim.IsValid():
                self.stage.RemovePrim(material_prim.GetPath())
                
        except Exception as e:
            print(f"Error removing visualization sphere at {coord}: {e}")

    def _create_single_obstacle_sphere(self, coord, sphere_radius=0.3, color=(1.0, 0.0, 0.0)):
        """Create a single sphere primitive to visualize an obstacle"""
        try:
            from pxr import UsdGeom, UsdShade, Sdf
            
            x, y, z = coord
            sphere_path = f"/World/ObstacleSphere_{x}_{y}_{z}"
            
            # Check if sphere already exists
            existing_prim = self.stage.GetPrimAtPath(sphere_path)
            if existing_prim and existing_prim.IsValid():
                return sphere_path
            
            # Create sphere primitive
            sphere_geom = UsdGeom.Sphere.Define(self.stage, sphere_path)
            
            # Set sphere properties
            sphere_geom.CreateRadiusAttr(sphere_radius)
            
            # Position the sphere
            xform = UsdGeom.Xformable(sphere_geom)
            xform.AddTranslateOp().Set((float(x), float(y), float(z)))
            
            # Create material for coloring
            material_path = f"/World/Materials/ObstacleMaterial_{x}_{y}_{z}"
            material = UsdShade.Material.Define(self.stage, material_path)
            
            # Create shader
            shader = UsdShade.Shader.Define(self.stage, f"{material_path}/PBRShader")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(color)
            shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
            
            # Connect shader to material
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            
            # Bind material to sphere
            UsdShade.MaterialBindingAPI(sphere_geom).Bind(material)
            
            print(f"Created visualization sphere at ({x}, {y}, {z})")
            return sphere_path
            
        except Exception as e:
            print(f"Error creating visualization sphere at {coord}: {e}")
            return None

    def create_obstacle_spheres(self, sphere_radius=0.3, color=(1.0, 0.0, 0.0)):
        """Create sphere primitives to visualize obstacles in the scene"""
        from pxr import UsdGeom, UsdShade, Sdf
        
        created_spheres = []
        
        for i, (x, y, z) in enumerate(self.obstacles):
            # Create unique sphere path
            sphere_path = f"/World/ObstacleSphere_{i}_{x}_{y}_{z}"
            
            # Check if sphere already exists
            existing_prim = self.stage.GetPrimAtPath(sphere_path)
            if existing_prim and existing_prim.IsValid():
                print(f"Sphere already exists at {sphere_path}")
                continue
            
            try:
                # Create sphere primitive
                sphere_geom = UsdGeom.Sphere.Define(self.stage, sphere_path)
                
                # Set sphere properties
                sphere_geom.CreateRadiusAttr(sphere_radius)
                
                # Position the sphere
                xform = UsdGeom.Xformable(sphere_geom)
                xform.AddTranslateOp().Set((float(x), float(y), float(z)))
                
                # Create material for coloring
                material_path = f"/World/Materials/ObstacleMaterial_{i}"
                material = UsdShade.Material.Define(self.stage, material_path)
                
                # Create shader
                shader = UsdShade.Shader.Define(self.stage, f"{material_path}/PBRShader")
                shader.CreateIdAttr("UsdPreviewSurface")
                shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(color)
                shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
                shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
                
                # Connect shader to material
                material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
                
                # Bind material to sphere
                UsdShade.MaterialBindingAPI(sphere_geom).Bind(material)
                
                created_spheres.append(sphere_path)
                print(f"Created obstacle sphere at ({x}, {y}, {z}) -> {sphere_path}")
                
            except Exception as e:
                print(f"Error creating sphere at ({x}, {y}, {z}): {e}")
        
        print(f"Created {len(created_spheres)} obstacle visualization spheres")
        return created_spheres

    def remove_obstacle_spheres(self):
        """Remove all obstacle visualization spheres from the scene"""
        removed_count = 0
        
        # Find all obstacle spheres
        for prim in self.stage.Traverse():
            prim_path = str(prim.GetPath())
            if "ObstacleSphere_" in prim_path or "ObstacleMaterial_" in prim_path:
                try:
                    self.stage.RemovePrim(prim.GetPath())
                    removed_count += 1
                except Exception as e:
                    print(f"Error removing {prim_path}: {e}")
        
        print(f"Removed {removed_count} obstacle visualization elements")
        return removed_count

    def get_obstacle_info(self):
        """Get basic information about current obstacles"""
        manual_obstacles = len(self.obstacles)
        scene_prims = len(self.obstacle_prims)
        
        print(f"Obstacles: {manual_obstacles} manual, {scene_prims} scene prims")
        if self.obstacles:
            print(f"Positions: {sorted(list(self.obstacles))}")
        
        return list(self.obstacles)


# Simple example usage
def run_example():
    """Simple example of how to use the pathfinder"""
    print("=== Isaac Sim Pathfinder Example ===")
    pathfinder = IsaacSimPathfinder()
    
    # Add some obstacles with visualization
    pathfinder.add_obstacle(1, 0, 1)
    pathfinder.add_obstacle(2, 0, 0)
    pathfinder.add_obstacle(3, 0, 0)
    
    print(f"Added {len(pathfinder.obstacles)} obstacles")
    
    # Set starting position
    start_pos = (0, 0, 1)
    pathfinder.set_cube_position(*start_pos)
    
    # Move to target (will find path around obstacles)
    target_pos = (4, 0, 1)
    pathfinder.move_to_target(target_pos)


if __name__ == "__main__":
    print("=== Isaac Sim Pathfinder with Obstacle Avoidance ===")
    print("This script moves a cube along a calculated path in Isaac Sim 4.5.0")
    print("Includes obstacle detection and avoidance with sphere visualization!")
    print()
    
    # Run the simple example
    run_example()
    
    print("\n" + "="*50)
    print("CORE FEATURES:")
    print("="*50)
    print("• add_obstacle(x, y, z) - Add obstacle with red sphere visualization")
    print("• move_to_target(target) - Move cube to target, avoiding obstacles")
    print("• create_wall_obstacle(start, end, axis) - Create wall of obstacles")
    print("• remove_obstacle_spheres() - Clear all obstacle visualizations")
    print("• Press ENTER during movement to advance step-by-step")
    print("="*50)
