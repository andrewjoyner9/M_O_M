#!/usr/bin/env python3
"""
Simplified Pathfinder for DreamerV3 Training

A standalone pathfinding system that doesn't require Isaac Sim USD libraries.
Uses A* algorithm for optimal path generation with obstacle avoidance.
"""

import numpy as np
import heapq
from typing import List, Tuple, Set, Optional


class SimplifiedPathfinder:
    """
    Enhanced pathfinding system for DreamerV3 training guidance with high-resolution A*
    """
    
    def __init__(self, resolution: float = 0.5):
        """Initialize the enhanced pathfinder with configurable resolution"""
        self.obstacles = set()  # Set of (x, y, z) tuples representing obstacle positions
        self.spherical_obstacles = []  # List of (x, y, z, radius) tuples
        self.resolution = resolution  # Grid resolution in meters (0.5m = 2 cells per meter)
        self.safety_margin = 0.3  # Additional safety margin around obstacles
        
        # Arena bounds (will be set by environment)
        self.arena_min = None
        self.arena_max = None
        
        print(f"🗺️ Enhanced pathfinder initialized with {self.resolution}m resolution")
    
    def add_obstacle(self, x: float, y: float, z: float, radius: float = 0.0):
        """Add an obstacle at the specified coordinates"""
        x, y, z = float(x), float(y), float(z)
        
        if radius <= 0:
            # Point obstacle - discretize to grid
            grid_x, grid_y, grid_z = self._world_to_grid(x, y, z)
            self.obstacles.add((grid_x, grid_y, grid_z))
            print(f"Point obstacle added at ({x:.1f}, {y:.1f}, {z:.1f}) -> grid ({grid_x}, {grid_y}, {grid_z})")
        else:
            # Spherical obstacle with enhanced radius for safety
            enhanced_radius = float(radius) + self.safety_margin
            self.spherical_obstacles.append((x, y, z, enhanced_radius))
            print(f"Spherical obstacle added at ({x:.1f}, {y:.1f}, {z:.1f}) with radius={radius:.1f}+{self.safety_margin:.1f}={enhanced_radius:.1f}")
    
    def clear_obstacles(self):
        """Clear all manually added obstacles"""
        self.obstacles.clear()
        self.spherical_obstacles.clear()
        print("All obstacles cleared")
    
    def set_arena_bounds(self, arena_min: np.ndarray, arena_max: np.ndarray):
        """Set arena boundaries for path planning"""
        self.arena_min = arena_min.copy()
        self.arena_max = arena_max.copy()
        print(f"Arena bounds set: {self.arena_min} to {self.arena_max}")
    
    def _is_within_arena_bounds(self, x: float, y: float, z: float) -> bool:
        """Check if position is within arena bounds"""
        if self.arena_min is None or self.arena_max is None:
            return True  # No bounds set, assume valid
        
        position = np.array([x, y, z])
        # Add small safety margin to arena bounds
        safety_buffer = 0.5  # 0.5m safety buffer from arena walls
        safe_min = self.arena_min + safety_buffer
        safe_max = self.arena_max - safety_buffer
        
        return np.all((position >= safe_min) & (position <= safe_max))
    
    def _world_to_grid(self, x: float, y: float, z: float) -> Tuple[int, int, int]:
        """Convert world coordinates to grid coordinates"""
        return (
            int(round(x / self.resolution)),
            int(round(y / self.resolution)),
            int(round(z / self.resolution))
        )
    
    def _grid_to_world(self, gx: int, gy: int, gz: int) -> Tuple[float, float, float]:
        """Convert grid coordinates to world coordinates"""
        return (
            gx * self.resolution,
            gy * self.resolution,
            gz * self.resolution
        )
    
    def is_position_blocked(self, x: float, y: float, z: float) -> bool:
        """Check if a position is blocked by an obstacle"""
        # Check point obstacles in grid space
        grid_x, grid_y, grid_z = self._world_to_grid(x, y, z)
        if (grid_x, grid_y, grid_z) in self.obstacles:
            return True
        
        # Check spherical obstacles in world space
        for obs_x, obs_y, obs_z, radius in self.spherical_obstacles:
            distance_sq = (x - obs_x) ** 2 + (y - obs_y) ** 2 + (z - obs_z) ** 2
            if distance_sq <= radius ** 2:
                return True
        
        return False
    
    def get_neighbors(self, gx: int, gy: int, gz: int) -> List[Tuple[int, int, int]]:
        """Get valid neighboring positions with enhanced movement options"""
        neighbors = []
        
        # Enhanced movement patterns: 6-directional + diagonals + 3D diagonals
        directions = [
            # Cardinal directions (6-directional)
            (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
            # Horizontal diagonals
            (1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0),
            # Vertical diagonals
            (1, 0, 1), (1, 0, -1), (-1, 0, 1), (-1, 0, -1),
            (0, 1, 1), (0, 1, -1), (0, -1, 1), (0, -1, -1),
            # 3D diagonals (corners of cube)
            (1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1),
            (-1, 1, 1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1)
        ]
        
        for dx, dy, dz in directions:
            new_gx, new_gy, new_gz = gx + dx, gy + dy, gz + dz
            # Convert to world coordinates to check collision AND bounds
            world_x, world_y, world_z = self._grid_to_world(new_gx, new_gy, new_gz)
            
            # Check both obstacle collision and arena bounds
            if (not self.is_position_blocked(world_x, world_y, world_z) and 
                self._is_within_arena_bounds(world_x, world_y, world_z)):
                neighbors.append((new_gx, new_gy, new_gz))
        
        return neighbors
    
    def heuristic(self, pos1: Tuple[int, int, int], pos2: Tuple[int, int, int]) -> float:
        """Enhanced Euclidean distance heuristic with better long-distance handling"""
        dx = pos1[0] - pos2[0]
        dy = pos1[1] - pos2[1] 
        dz = pos1[2] - pos2[2]
        
        # Use Euclidean distance scaled by resolution
        euclidean_dist = np.sqrt(dx*dx + dy*dy + dz*dz) * self.resolution
        
        # For long distances, slightly increase heuristic weight to guide search more aggressively
        distance_factor = euclidean_dist / 10.0  # Normalize by 10m
        if distance_factor > 2.0:  # For distances > 20m
            euclidean_dist *= 1.1  # 10% increase in heuristic weight
        
        return euclidean_dist
    
    def movement_cost(self, from_pos: Tuple[int, int, int], to_pos: Tuple[int, int, int]) -> float:
        """Calculate movement cost between adjacent positions"""
        dx = abs(to_pos[0] - from_pos[0])
        dy = abs(to_pos[1] - from_pos[1])
        dz = abs(to_pos[2] - from_pos[2])
        
        # Different costs for different movement types
        if dx + dy + dz == 1:
            # Cardinal movement
            return self.resolution
        elif dx + dy + dz == 2:
            # 2D diagonal movement
            return self.resolution * np.sqrt(2)
        elif dx + dy + dz == 3:
            # 3D diagonal movement  
            return self.resolution * np.sqrt(3)
        else:
            # Fallback
            return self.resolution * np.sqrt(dx*dx + dy*dy + dz*dz)
    
    def find_path(self, start: Tuple[int, int, int], goal: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
        """
        Enhanced A* pathfinding with higher resolution and better movement
        
        Args:
            start: Starting position in grid coordinates
            goal: Goal position in grid coordinates
            
        Returns:
            List of grid coordinate tuples representing the path
        """
        print(f"🗺️ Enhanced A* planning from {start} to {goal}")
        print(f"   • Resolution: {self.resolution}m per cell")
        print(f"   • Point obstacles: {len(self.obstacles)}")
        print(f"   • Spherical obstacles: {len(self.spherical_obstacles)}")
        
        # Check if start or goal positions are blocked
        start_world = self._grid_to_world(*start)
        goal_world = self._grid_to_world(*goal)
        
        if self.is_position_blocked(*start_world):
            print(f"❌ Start position {start} (world: {start_world}) is blocked!")
            return []
        
        if self.is_position_blocked(*goal_world):
            print(f"❌ Goal position {goal} (world: {goal_world}) is blocked!")
            return []
        
        # Enhanced A* algorithm implementation
        open_set = [(0, 0, start)]  # (f_score, g_score, position)
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}
        closed_set = set()
        
        nodes_explored = 0
        max_nodes = min(50000, abs(start[0] - goal[0]) * abs(start[1] - goal[1]) * 10)  # Dynamic limit based on distance
        
        while open_set and nodes_explored < max_nodes:
            # Get position with lowest f_score
            current_f, current_g, current = heapq.heappop(open_set)
            nodes_explored += 1
            
            if current in closed_set:
                continue
                
            if current == goal:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                print(f"✅ Enhanced path found with {len(path)} waypoints ({nodes_explored} nodes explored)")
                return path
            
            closed_set.add(current)
            
            # Check all neighbors with enhanced movement
            for neighbor in self.get_neighbors(*current):
                if neighbor in closed_set:
                    continue
                
                # Calculate movement cost
                tentative_g = g_score[current] + self.movement_cost(current, neighbor)
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], tentative_g, neighbor))
        
        print(f"❌ No path found after exploring {nodes_explored} nodes!")
        return []
    
    def find_path_simple(self, start: Tuple[int, int, int], goal: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
        """
        Find a path using simple line algorithm (ignores obstacles) - fallback method
        
        Args:
            start: Starting position (x, y, z)
            goal: Goal position (x, y, z)
            
        Returns:
            List of 3D coordinate tuples representing the path
        """
        path = []
        x, y, z = start
        goal_x, goal_y, goal_z = goal

        print(f"🔄 Using simple pathfinding from {start} to {goal}")
        
        while (x, y, z) != (goal_x, goal_y, goal_z):
            # Move in X direction first
            if x < goal_x:
                x += 1
            elif x > goal_x:
                x -= 1
            # Then move in Y direction
            elif y < goal_y:
                y += 1
            elif y > goal_y:
                y -= 1
            # Finally move in Z direction
            elif z < goal_z:
                z += 1
            elif z > goal_z:
                z -= 1
            
            path.append((x, y, z))
        
        print(f"✅ Simple path calculated with {len(path)} steps")
        return path
    
    def validate_path(self, path: List[Tuple[int, int, int]]) -> bool:
        """Check if a path is clear of obstacles"""
        blocked_positions = []
        for step, (x, y, z) in enumerate(path):
            if self.is_position_blocked(x, y, z):
                blocked_positions.append((step, (x, y, z)))
        
        if blocked_positions:
            print(f"⚠️ Path has {len(blocked_positions)} blocked positions")
            return False
        return True
    
    def get_optimal_path(self, start: Tuple[float, float, float], 
                        goal: Tuple[float, float, float]) -> List[Tuple[float, float, float]]:
        """
        Get optimal path with high-resolution floating point coordinates
        
        Args:
            start: Starting position (x, y, z) in world coordinates
            goal: Goal position (x, y, z) in world coordinates
            
        Returns:
            List of floating point coordinate tuples in world coordinates
        """
        print(f"🎯 High-resolution pathfinding from {start} to {goal}")
        
        # Convert to grid coordinates for pathfinding
        start_grid = self._world_to_grid(*start)
        goal_grid = self._world_to_grid(*goal)
        
        print(f"   • Grid start: {start_grid}, Grid goal: {goal_grid}")
        
        # Find path in grid space
        grid_path = self.find_path(start_grid, goal_grid)
        
        if not grid_path:
            print("   • A* failed, trying simple fallback...")
            # Try simple path as fallback
            grid_path = self.find_path_simple(start_grid, goal_grid)
        
        if not grid_path:
            print("   • All pathfinding failed, returning direct line")
            return [start, goal]
        
        # Convert back to world coordinates with smoothing
        world_path = [start]  # Start with actual start position
        
        for grid_pos in grid_path:
            world_pos = self._grid_to_world(*grid_pos)
            world_path.append(world_pos)
        
        # Add actual goal position
        if world_path[-1] != goal:
            world_path.append(goal)
        
        # Apply path smoothing for better guidance
        smoothed_path = self._smooth_path(world_path)
        
        print(f"   ✅ Path generated: {len(grid_path)} grid waypoints → {len(smoothed_path)} smooth waypoints")
        return smoothed_path
    
    def _smooth_path(self, path: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
        """Apply simple path smoothing to reduce waypoints and create smoother trajectories"""
        if len(path) <= 2:
            return path
            
        smoothed = [path[0]]  # Always keep start
        
        i = 0
        while i < len(path) - 1:
            # Look ahead to find the farthest point we can reach in a straight line
            farthest = i + 1
            
            for j in range(i + 2, min(i + 6, len(path))):  # Look ahead up to 5 points
                if self._is_line_clear(path[i], path[j]):
                    farthest = j
                else:
                    break
            
            smoothed.append(path[farthest])
            i = farthest
        
        return smoothed
    
    def _is_line_clear(self, start: Tuple[float, float, float], end: Tuple[float, float, float]) -> bool:
        """Check if a straight line between two points is clear of obstacles"""
        # Sample points along the line
        samples = max(3, int(np.linalg.norm(np.array(end) - np.array(start)) / self.resolution))
        
        for i in range(samples + 1):
            t = i / samples
            point = (
                start[0] + t * (end[0] - start[0]),
                start[1] + t * (end[1] - start[1]),
                start[2] + t * (end[2] - start[2])
            )
            
            if self.is_position_blocked(*point):
                return False
        
        return True
    
    def min_distance_to_spheres(self, x: float, y: float, z: float) -> float:
        """Get minimum distance to any spherical obstacle"""
        if not self.spherical_obstacles:
            return np.inf
        
        distances = []
        for obs_x, obs_y, obs_z, radius in self.spherical_obstacles:
            distance_to_center = np.sqrt((x - obs_x)**2 + (y - obs_y)**2 + (z - obs_z)**2)
            distance_to_surface = max(0.0, distance_to_center - radius)
            distances.append(distance_to_surface)
        
        return min(distances)
    
    def local_occupancy_grid_3x3(self, x: float, y: float, z: float) -> List[float]:
        """
        3x3x3 occupancy grid centered around position
        
        Args:
            x, y, z: Center position
            
        Returns:
            List of 27 values (1.0 for blocked, 0.0 for free)
        """
        occupancy = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    check_x = x + dx
                    check_y = y + dy
                    check_z = z + dz
                    blocked = 1.0 if self.is_position_blocked(check_x, check_y, check_z) else 0.0
                    occupancy.append(blocked)
        return occupancy


def test_pathfinder():
    """Test the enhanced pathfinder with high resolution"""
    print("🧪 Testing Enhanced High-Resolution Pathfinder")
    print("=" * 50)
    
    # Test different resolutions
    for resolution in [0.5, 0.25]:
        print(f"\n📏 Testing with {resolution}m resolution:")
        pathfinder = SimplifiedPathfinder(resolution=resolution)
        
        # Add some obstacles
        pathfinder.add_obstacle(2.0, 0.0, 1.0, radius=1.0)
        pathfinder.add_obstacle(3.0, 0.0, 1.0, radius=1.0)
        pathfinder.add_obstacle(1.0, 2.0, 1.0, radius=0.5)
        
        # Test pathfinding
        start = (0.0, 0.0, 1.0)
        goal = (5.0, 0.0, 1.0)
        
        path = pathfinder.get_optimal_path(start, goal)
        
        print(f"\n📍 Path from {start} to {goal}:")
        for i, pos in enumerate(path):
            print(f"  {i+1}: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
        
        print(f"\n📊 Path statistics:")
        print(f"  • Length: {len(path)} waypoints")
        print(f"  • Total distance: {sum(np.linalg.norm(np.array(path[i+1]) - np.array(path[i])) for i in range(len(path)-1)):.2f}m")
        
        # Test line-of-sight checking
        print(f"\n🔍 Line-of-sight test:")
        clear = pathfinder._is_line_clear(start, goal)
        print(f"  • Direct line clear: {clear}")
        
        pathfinder.clear_obstacles()
    
    return pathfinder, path


if __name__ == "__main__":
    test_pathfinder()
