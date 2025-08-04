"""
Obstacle detection and management for Isaac Sim pathfinding
"""
from typing import Set, Tuple, List, Dict, Any
from pxr import UsdGeom


class ObstacleManager:
    """Manages obstacle detection and avoidance in Isaac Sim"""
    
    def __init__(self, stage=None):
        self.stage = stage
        self.obstacles: Set[Tuple[int, int, int]] = set()  # Manual obstacles
        self.obstacle_prims: List[str] = []  # Scene primitive paths to check
        self.detection_radius = 0.5  # Radius for obstacle detection
    
    def add_obstacle(self, x: int, y: int, z: int) -> None:
        """Add an obstacle at the specified coordinates"""
        self.obstacles.add((int(x), int(y), int(z)))
        print(f"Obstacle added at ({x}, {y}, {z})")

    def remove_obstacle(self, x: int, y: int, z: int) -> None:
        """Remove an obstacle at the specified coordinates"""
        coord = (int(x), int(y), int(z))
        if coord in self.obstacles:
            self.obstacles.remove(coord)
            print(f"Obstacle removed from ({x}, {y}, {z})")
        else:
            print(f"No obstacle found at ({x}, {y}, {z})")

    def clear_obstacles(self) -> None:
        """Clear all manually added obstacles"""
        self.obstacles.clear()
        print("All manual obstacles cleared")

    def add_obstacle_prim_path(self, prim_path: str) -> None:
        """Add a primitive path to check for obstacles"""
        if prim_path not in self.obstacle_prims:
            self.obstacle_prims.append(prim_path)
            print(f"Added obstacle primitive: {prim_path}")

    def detect_obstacles_from_scene(self) -> Set[Tuple[int, int, int]]:
        """Detect obstacles from the Isaac Sim scene"""
        if not self.stage:
            print("Warning: No stage available for scene obstacle detection")
            return set()
            
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

    def is_position_blocked(self, x: int, y: int, z: int) -> bool:
        """Check if a position is blocked by an obstacle"""
        coord = (int(x), int(y), int(z))
        return coord in self.obstacles

    def get_all_obstacles(self) -> Set[Tuple[int, int, int]]:
        """Get all current obstacles (manual + scene detected)"""
        return self.obstacles.copy()

    def create_wall_obstacle(self, start_pos: Tuple[int, int, int], end_pos: Tuple[int, int, int], axis: str = 'x') -> int:
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
        return obstacles_added

    def get_obstacle_info(self) -> Dict[str, Any]:
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

    def update_stage(self, stage) -> None:
        """Update the USD stage reference"""
        self.stage = stage
