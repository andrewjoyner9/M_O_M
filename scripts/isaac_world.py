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
        
        # Path movement state
        self.current_path = []
        self.current_step = 0
        self.is_moving = False
        self.keyboard_subscription = None
        
        # Obstacle detection
        self.obstacles = set()  # Set of (x, y, z) tuples representing obstacle positions
        self.sphericalObstacles = []
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

    def add_obstacle(self, x, y, z, radius= 0.0):
        """Add an obstacle at the specified coordinates"""
        edge_x, edge_y, edge_z = map(float, (x, y, z))
        if radius <= 0:
            self.obstacles.add((edge_x, edge_y, edge_z))
            print(f"Singularity Obstacle added at ({edge_x}, {edge_y}, {edge_z})")
            return
        else:
            self.sphericalObstacles.append((edge_x, edge_y, edge_z, float(radius)))
            print(f"Spherical Obstacle added at ({edge_x}, {edge_y}, {edge_z}) with radius={radius}")

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
        if (x, y, z) in self.obstacles:
            return True
        for edge_x, edge_y, edge_z, r in self.sphericalObstacles:
            if (x - edge_x) ** 2 + (y - edge_y) ** 2 + (z - edge_z) ** 2 <= r ** 2:
                return True
        return False    
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
        for (x , y, z) in path:
            self.set_cube_position(float(x), float(y), float(z))

if __name__ == "__main__":
    print("Sample Smokefinding PathFinding")
    ipf = IsaacSimPathfinder()

    ipf.add_obstacle(2, 0, 1)
    ipf.add_obstacle(3, 0, 1)
    ipf.set_cube_position(0,0,1)
    target = (5,0, 1)
    path = ipf.move_to_target(target)
    print(f"Path Length = {len(path)} / Final Position = {ipf.get_cube_position()}")
