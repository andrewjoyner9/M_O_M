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
    Simplified pathfinding system for DreamerV3 training guidance
    """
    
    def __init__(self):
        """Initialize the simplified pathfinder"""
        self.obstacles = set()  # Set of (x, y, z) tuples representing obstacle positions
        self.spherical_obstacles = []  # List of (x, y, z, radius) tuples
        self.detection_radius = 0.5  # Radius for obstacle detection
        
    def add_obstacle(self, x: float, y: float, z: float, radius: float = 0.0):
        """Add an obstacle at the specified coordinates"""
        x, y, z = float(x), float(y), float(z)
        
        if radius <= 0:
            # Point obstacle
            self.obstacles.add((int(x), int(y), int(z)))
            print(f"Point obstacle added at ({x:.1f}, {y:.1f}, {z:.1f})")
        else:
            # Spherical obstacle
            self.spherical_obstacles.append((x, y, z, float(radius)))
            print(f"Spherical obstacle added at ({x:.1f}, {y:.1f}, {z:.1f}) with radius={radius:.1f}")
    
    def clear_obstacles(self):
        """Clear all manually added obstacles"""
        self.obstacles.clear()
        self.spherical_obstacles.clear()
        print("All obstacles cleared")
    
    def is_position_blocked(self, x: float, y: float, z: float) -> bool:
        """Check if a position is blocked by an obstacle"""
        # Check point obstacles
        if (int(x), int(y), int(z)) in self.obstacles:
            return True
        
        # Check spherical obstacles
        for obs_x, obs_y, obs_z, radius in self.spherical_obstacles:
            distance_sq = (x - obs_x) ** 2 + (y - obs_y) ** 2 + (z - obs_z) ** 2
            if distance_sq <= radius ** 2:
                return True
        
        return False
    
    def get_neighbors(self, x: int, y: int, z: int) -> List[Tuple[int, int, int]]:
        """Get valid neighboring positions (not blocked by obstacles)"""
        neighbors = []
        # 6-directional movement (up, down, left, right, forward, backward)
        directions = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
        
        for dx, dy, dz in directions:
            new_x, new_y, new_z = x + dx, y + dy, z + dz
            if not self.is_position_blocked(new_x, new_y, new_z):
                neighbors.append((new_x, new_y, new_z))
        
        return neighbors
    
    def heuristic(self, pos1: Tuple[int, int, int], pos2: Tuple[int, int, int]) -> float:
        """Manhattan distance heuristic for A* algorithm"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]) + abs(pos1[2] - pos2[2])
    
    def find_path(self, start: Tuple[int, int, int], goal: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
        """
        Find a path from start to goal using A* algorithm with obstacle avoidance
        
        Args:
            start: Starting position (x, y, z)
            goal: Goal position (x, y, z)
            
        Returns:
            List of 3D coordinate tuples representing the path
        """
        print(f"🗺️ Planning path from {start} to {goal}")
        print(f"   • Point obstacles: {len(self.obstacles)}")
        print(f"   • Spherical obstacles: {len(self.spherical_obstacles)}")
        
        # Check if start or goal positions are blocked
        if self.is_position_blocked(*start):
            print(f"❌ Start position {start} is blocked!")
            return []
        
        if self.is_position_blocked(*goal):
            print(f"❌ Goal position {goal} is blocked!")
            return []
        
        # A* algorithm implementation
        open_set = [(0, 0, start)]  # (f_score, g_score, position)
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}
        closed_set = set()
        
        while open_set:
            # Get position with lowest f_score
            current_f, current_g, current = heapq.heappop(open_set)
            
            if current in closed_set:
                continue
                
            if current == goal:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                print(f"✅ Path found with {len(path)} steps")
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
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], tentative_g, neighbor))
        
        print("❌ No path found!")
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
        Get optimal path with floating point coordinates
        
        Args:
            start: Starting position (x, y, z)
            goal: Goal position (x, y, z)
            
        Returns:
            List of floating point coordinate tuples
        """
        # Convert to integer coordinates for pathfinding
        start_int = (int(start[0]), int(start[1]), int(start[2]))
        goal_int = (int(goal[0]), int(goal[1]), int(goal[2]))
        
        # Find integer path
        int_path = self.find_path(start_int, goal_int)
        
        if not int_path:
            # Try simple path as fallback
            int_path = self.find_path_simple(start_int, goal_int)
        
        # Convert back to floating point and add start position
        float_path = [start]  # Start with actual start position
        
        for int_pos in int_path:
            float_pos = (float(int_pos[0]), float(int_pos[1]), float(int_pos[2]))
            float_path.append(float_pos)
        
        return float_path
    
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
    """Test the simplified pathfinder"""
    print("🧪 Testing Simplified Pathfinder")
    print("=" * 40)
    
    pathfinder = SimplifiedPathfinder()
    
    # Add some obstacles
    pathfinder.add_obstacle(2, 0, 1, radius=1.0)
    pathfinder.add_obstacle(3, 0, 1, radius=1.0)
    pathfinder.add_obstacle(1, 2, 1, radius=0.5)
    
    # Test pathfinding
    start = (0.0, 0.0, 1.0)
    goal = (5.0, 0.0, 1.0)
    
    path = pathfinder.get_optimal_path(start, goal)
    
    print(f"\n📍 Path from {start} to {goal}:")
    for i, pos in enumerate(path):
        print(f"  {i+1}: {pos}")
    
    print(f"\n📊 Path statistics:")
    print(f"  • Length: {len(path)} waypoints")
    print(f"  • Valid: {pathfinder.validate_path([(int(p[0]), int(p[1]), int(p[2])) for p in path])}")
    
    return pathfinder, path


if __name__ == "__main__":
    test_pathfinder()
