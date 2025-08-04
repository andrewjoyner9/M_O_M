"""
Core pathfinding algorithms for Isaac Sim navigation
"""
import heapq
from typing import List, Tuple, Set


class PathfindingCore:
    """Core pathfinding algorithms with obstacle avoidance"""
    
    def __init__(self):
        pass
    
    def manhattan_distance(self, pos1: Tuple[int, int, int], pos2: Tuple[int, int, int]) -> int:
        """Calculate Manhattan distance between two 3D positions"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]) + abs(pos1[2] - pos2[2])
    
    def get_neighbors(self, x: int, y: int, z: int, obstacles: Set[Tuple[int, int, int]]) -> List[Tuple[int, int, int]]:
        """Get valid neighboring positions (not blocked by obstacles)"""
        neighbors = []
        # 6-directional movement (up, down, left, right, forward, backward)
        directions = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
        
        for dx, dy, dz in directions:
            new_x, new_y, new_z = x + dx, y + dy, z + dz
            coord = (new_x, new_y, new_z)
            if coord not in obstacles:
                neighbors.append(coord)
        
        return neighbors
    
    def find_path_astar(self, start: Tuple[int, int, int], end: Tuple[int, int, int], 
                        obstacles: Set[Tuple[int, int, int]]) -> List[Tuple[int, int, int]]:
        """
        Find a path from start to end coordinates using A* algorithm with obstacle avoidance
        Returns a list of 3D coordinate tuples representing the path
        """
        print(f"Planning A* path from {start} to {end}")
        print(f"Known obstacles: {len(obstacles)} positions")
        
        # Check if start or end positions are blocked
        if start in obstacles:
            print(f"Error: Start position {start} is blocked by an obstacle!")
            return []
        
        if end in obstacles:
            print(f"Error: End position {end} is blocked by an obstacle!")
            return []
        
        # Priority queue: (f_score, g_score, position)
        open_set = [(0, 0, start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.manhattan_distance(start, end)}
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
                print(f"A* path found with {len(path)} steps")
                return path
            
            closed_set.add(current)
            
            # Check all neighbors
            for neighbor in self.get_neighbors(*current, obstacles):
                if neighbor in closed_set:
                    continue
                
                tentative_g = g_score[current] + 1
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.manhattan_distance(neighbor, end)
                    heapq.heappush(open_set, (f_score[neighbor], tentative_g, neighbor))
        
        print("No A* path found! Target may be unreachable due to obstacles.")
        return []
    
    def find_path_simple(self, start: Tuple[int, int, int], end: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
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
    
    def validate_path(self, path: List[Tuple[int, int, int]], obstacles: Set[Tuple[int, int, int]]) -> bool:
        """Check if a path is clear of obstacles"""
        blocked_positions = []
        for step, (x, y, z) in enumerate(path):
            coord = (x, y, z)
            if coord in obstacles:
                blocked_positions.append((step, coord))
        
        if blocked_positions:
            print(f"Warning: Path has {len(blocked_positions)} blocked positions:")
            for step, pos in blocked_positions:
                print(f"  Step {step + 1}: {pos}")
            return False
        return True
    
    def find_best_path(self, start: Tuple[int, int, int], end: Tuple[int, int, int], 
                       obstacles: Set[Tuple[int, int, int]]) -> List[Tuple[int, int, int]]:
        """Try to find the best available path, starting with A* and falling back to simple"""
        # First try A* algorithm
        path = self.find_path_astar(start, end, obstacles)
        if path and self.validate_path(path, obstacles):
            return path
        
        # If A* fails, try simple path as fallback
        print("A* pathfinding failed, trying simple pathfinding...")
        simple_path = self.find_path_simple(start, end)
        if self.validate_path(simple_path, obstacles):
            return simple_path
        
        print("Both pathfinding methods failed due to obstacles.")
        return []
