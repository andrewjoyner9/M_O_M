"""
Test functions and examples for Isaac Sim pathfinding
"""
from .pathfinder_main import IsaacSimPathfinder


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
    
    print(f"Added {len(pathfinder.obstacles_set)} obstacles")
    
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
        pathfinder.movement.current_path = path
        pathfinder.movement.current_step = 0
        pathfinder.movement.is_moving = True
        
        print(f"Initial path: {path}")
        print("You can now add obstacles using:")
        print("  pathfinder.add_obstacle(x, y, z)")
        print("Then continue movement with:")
        print("  pathfinder.advance_to_next_step()")
    else:
        print("Failed to find initial path")


def run_all_tests():
    """Run all test functions"""
    print("Running all pathfinding tests...")
    
    tests = [
        test_basic_movement,
        test_pathfinding,
        test_custom_path,
        test_obstacle_avoidance,
        test_scene_obstacle_detection,
        test_dynamic_obstacles
    ]
    
    for i, test_func in enumerate(tests, 1):
        print(f"\n{'='*60}")
        print(f"Test {i}/{len(tests)}: {test_func.__name__}")
        print('='*60)
        try:
            test_func()
            print(f"✓ Test {test_func.__name__} completed")
        except Exception as e:
            print(f"✗ Test {test_func.__name__} failed: {e}")
        print("Press Enter to continue to next test...")
        input()


if __name__ == "__main__":
    print("=== Isaac Sim Pathfinder Test Suite ===")
    print("This module contains test functions for the pathfinding system")
    print("\nAvailable test functions:")
    print("- test_basic_movement()")
    print("- test_pathfinding()")
    print("- test_custom_path()")
    print("- test_obstacle_avoidance()")
    print("- test_scene_obstacle_detection()")
    print("- test_dynamic_obstacles()")
    print("- run_all_tests()")
    print("\nTo run a specific test, import this module and call the function.")
    print("Example: from pathfinding.test_examples import test_obstacle_avoidance; test_obstacle_avoidance()")
