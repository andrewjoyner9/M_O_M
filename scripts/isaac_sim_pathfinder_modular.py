"""
Simplified Isaac Sim Pathfinder - now using modular architecture
This is the main entry point that imports from the pathfinding package
"""
from pathfinding import IsaacSimPathfinder, test_examples


def main():
    """Main function demonstrating the modular pathfinder"""
    print("=== Isaac Sim Pathfinder with Obstacle Avoidance (Modular Version) ===")
    print("This script moves a cube along a calculated path in Isaac Sim 4.5.0")
    print("Now includes obstacle detection and avoidance capabilities!")
    print("Code is now organized into separate modules for better maintainability.")
    print()
    
    # Example usage with the new modular system:
    print("Example: Move from current position to (5, 5, 5)")
    # pathfinder = IsaacSimPathfinder()
    # pathfinder.move_to_target((5, 5, 5))
    
    print("\nAvailable test functions:")
    print("- test_examples.test_basic_movement()")
    print("- test_examples.test_pathfinding()")
    print("- test_examples.test_custom_path()")
    print("- test_examples.test_obstacle_avoidance()")
    print("- test_examples.test_scene_obstacle_detection()")
    print("- test_examples.test_dynamic_obstacles()")
    print("- test_examples.run_all_tests()")
    
    print("\nExample 5: Test obstacle avoidance")
    # Uncomment the line below to test obstacle avoidance
    test_examples.test_obstacle_avoidance()
    
    print("\n" + "="*50)
    print("MODULAR ARCHITECTURE:")
    print("="*50)
    print("• pathfinding_core.py - A* and simple pathfinding algorithms")
    print("• obstacle_manager.py - Obstacle detection and management")
    print("• movement_controller.py - Movement execution and keyboard controls")
    print("• isaac_sim_interface.py - Isaac Sim USD scene interaction")
    print("• pathfinder_main.py - Main orchestrator class")
    print("• test_examples.py - Test functions and examples")
    print("="*50)
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


# Convenience functions for backward compatibility
def move_cube_to_target(target_coordinates, movement_delay=0.5, use_async=True):
    """
    Convenience function to move cube to target coordinates
    
    Args:
        target_coordinates: (x, y, z) tuple of target position
        movement_delay: Delay between movements in seconds (deprecated - now uses user input)
        use_async: Whether to use async movement
    """
    return test_examples.move_cube_to_target(target_coordinates, movement_delay, use_async)


if __name__ == "__main__":
    main()
