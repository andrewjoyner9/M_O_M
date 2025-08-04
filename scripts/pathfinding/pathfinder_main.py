"""
Main orchestrator class that ties all pathfinding components together
"""
from typing import Tuple, List, Optional, Callable
from .pathfinding_core import PathfindingCore
from .obstacle_manager import ObstacleManager
from .movement_controller import MovementController
from .isaac_sim_interface import IsaacSimInterface


class IsaacSimPathfinder:
    """Main pathfinder class that orchestrates all components"""
    
    def __init__(self, cube_path: str = "/World/Cube"):
        """Initialize the Isaac Sim pathfinder with cube path"""
        self.cube_path = cube_path
        
        # Initialize components
        self.isaac_sim = IsaacSimInterface(cube_path)
        self.pathfinder = PathfindingCore()
        self.obstacles = ObstacleManager(self.isaac_sim.get_stage())
        self.movement = MovementController(self.isaac_sim, self.pathfinder, self.obstacles)
        
        # Legacy compatibility properties
        self.movement_delay = 1
        
    # === Position Management ===
    def get_cube_position(self) -> Tuple[float, float, float]:
        """Get the current position of the cube"""
        return self.isaac_sim.get_cube_position()

    def set_cube_position(self, x: float, y: float, z: float) -> bool:
        """Set the position of the cube"""
        return self.isaac_sim.set_cube_position(x, y, z)

    # === Obstacle Management ===
    def add_obstacle(self, x: int, y: int, z: int) -> None:
        """Add an obstacle at the specified coordinates"""
        self.obstacles.add_obstacle(x, y, z)

    def remove_obstacle(self, x: int, y: int, z: int) -> None:
        """Remove an obstacle at the specified coordinates"""
        self.obstacles.remove_obstacle(x, y, z)

    def clear_obstacles(self) -> None:
        """Clear all manually added obstacles"""
        self.obstacles.clear_obstacles()

    def add_obstacle_prim_path(self, prim_path: str) -> None:
        """Add a primitive path to check for obstacles"""
        self.obstacles.add_obstacle_prim_path(prim_path)

    def detect_obstacles_from_scene(self):
        """Detect obstacles from the Isaac Sim scene"""
        return self.obstacles.detect_obstacles_from_scene()

    def is_position_blocked(self, x: int, y: int, z: int) -> bool:
        """Check if a position is blocked by an obstacle"""
        return self.obstacles.is_position_blocked(x, y, z)

    def create_wall_obstacle(self, start_pos: Tuple[int, int, int], end_pos: Tuple[int, int, int], axis: str = 'x') -> int:
        """Create a wall of obstacles between two points along a specified axis"""
        return self.obstacles.create_wall_obstacle(start_pos, end_pos, axis)

    def get_obstacle_info(self):
        """Get information about current obstacles"""
        return self.obstacles.get_obstacle_info()

    # === Pathfinding ===
    def get_neighbors(self, x: int, y: int, z: int) -> List[Tuple[int, int, int]]:
        """Get valid neighboring positions (not blocked by obstacles)"""
        obstacles = self.obstacles.get_all_obstacles()
        return self.pathfinder.get_neighbors(x, y, z, obstacles)

    def find_path(self, start: Tuple[int, int, int], end: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
        """Find a path from start to end coordinates using A* algorithm with obstacle avoidance"""
        # Detect obstacles from scene first
        self.obstacles.detect_obstacles_from_scene()
        obstacles = self.obstacles.get_all_obstacles()
        return self.pathfinder.find_path_astar(start, end, obstacles)

    def find_path_simple(self, start: Tuple[int, int, int], end: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
        """Find a path using simple line algorithm (ignores obstacles) - kept for fallback"""
        return self.pathfinder.find_path_simple(start, end)

    def validate_path(self, path: List[Tuple[int, int, int]]) -> bool:
        """Check if a path is clear of obstacles"""
        obstacles = self.obstacles.get_all_obstacles()
        return self.pathfinder.validate_path(path, obstacles)

    def replan_if_blocked(self, start: Tuple[int, int, int], end: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
        """Try to find an alternative path if the current path is blocked"""
        print("Attempting to replan path to avoid obstacles...")
        self.obstacles.detect_obstacles_from_scene()
        obstacles = self.obstacles.get_all_obstacles()
        return self.pathfinder.find_best_path(start, end, obstacles)

    # === Movement Control ===
    def move_along_path_interactive(self, path: List[Tuple[int, int, int]]) -> None:
        """Move the cube along the calculated path with keyboard input for each step"""
        self.movement.move_along_path_interactive(path)

    async def move_along_path_async(self, path: List[Tuple[int, int, int]]) -> None:
        """Move the cube along the calculated path with keyboard input for each step (non-blocking)"""
        print("Note: Using interactive keyboard-based movement instead of async delays")
        self.movement.move_along_path_interactive(path)

    def move_along_path_blocking(self, path: List[Tuple[int, int, int]]) -> None:
        """Move the cube along the calculated path with keyboard input for each step"""
        print("Note: Using interactive keyboard-based movement instead of blocking delays")
        self.movement.move_along_path_interactive(path)

    def move_to_target(self, target_coordinates: Tuple[int, int, int], use_async: bool = True) -> None:
        """Move the cube from its current position to target coordinates"""
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
        
        # Move along path - both async and blocking now use the same interactive method
        self.move_along_path_interactive(path)

    def advance_to_next_step(self) -> None:
        """Move to the next step in the path"""
        self.movement.advance_to_next_step()

    def stop_movement(self) -> None:
        """Stop the current movement and clean up"""
        self.movement.stop_movement()

    def get_movement_status(self) -> str:
        """Get the current movement status"""
        return self.movement.get_movement_status()

    def set_movement_speed(self, delay_seconds: float) -> None:
        """Set the delay between movements (legacy - now uses keyboard input)"""
        self.movement_delay = delay_seconds
        self.movement.set_movement_speed(delay_seconds)

    # === Keyboard Controls ===
    def setup_keyboard_controls(self) -> bool:
        """Set up keyboard controls for step-by-step movement"""
        return self.movement.setup_keyboard_controls()

    def cleanup_keyboard_controls(self) -> None:
        """Clean up keyboard controls"""
        self.movement.cleanup_keyboard_controls()

    # === Callbacks ===
    def set_callbacks(self, on_movement_complete: Optional[Callable] = None, 
                     on_step_complete: Optional[Callable] = None) -> None:
        """Set callback functions for movement events"""
        self.movement.set_callbacks(on_movement_complete, on_step_complete)

    # === Legacy Properties for Backward Compatibility ===
    @property
    def current_path(self) -> List[Tuple[int, int, int]]:
        """Get the current path"""
        return self.movement.current_path

    @property
    def current_step(self) -> int:
        """Get the current step"""
        return self.movement.current_step

    @property
    def is_moving(self) -> bool:
        """Check if currently moving"""
        return self.movement.is_moving

    @property
    def obstacles_set(self):
        """Get obstacles (legacy compatibility)"""
        return self.obstacles.obstacles

    @property
    def obstacle_prims(self):
        """Get obstacle primitive paths (legacy compatibility)"""
        return self.obstacles.obstacle_prims

    @property
    def detection_radius(self):
        """Get detection radius (legacy compatibility)"""
        return self.obstacles.detection_radius

    @detection_radius.setter
    def detection_radius(self, value: float):
        """Set detection radius (legacy compatibility)"""
        self.obstacles.detection_radius = value
