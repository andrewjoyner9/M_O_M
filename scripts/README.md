# Isaac Sim 4.5.0 Cube Pathfinding

This project translates basic pathfinding algorithms into Isaac Sim 4.5.0 code that can move a cube along calculated paths.

## Files Overview

1. **`find_path.py`** - Original pathfinding algorithm with Isaac Sim integration
2. **`isaac_sim_pathfinder.py`** - Full-featured pathfinding class for Isaac Sim
3. **`setup_env.py`** - Sets up the physics environment and cube in Isaac Sim
4. **`simple_arrow_controller.py`** - Manual keyboard control for cube movement
5. **`simple_x_movement.py`** - Basic X-axis movement example

## How to Use in Isaac Sim 4.5.0

### Method 1: Using the Enhanced find_path.py

1. **First, set up your environment:**
   ```python
   # In Isaac Sim Script Editor, run:
   exec(open(r"c:\Users\andre\Documents\BAH_PROJECT_2025\M_O_M\scripts\setup_env.py").read())
   ```

2. **Load and use the pathfinding functions:**
   ```python
   # Load the pathfinding script
   exec(open(r"c:\Users\andre\Documents\BAH_PROJECT_2025\M_O_M\scripts\find_path.py").read())
   
   # Move cube from (0,0,0) to (5,5,5)
   execute_path_in_isaac_sim((0, 0, 0), (5, 5, 5))
   
   # Move with custom speed (0.2 seconds between moves)
   execute_path_in_isaac_sim((0, 0, 0), (3, 3, 3), movement_delay=0.2)
   ```

### Method 2: Using the IsaacSimPathfinder Class

1. **Set up environment and load the pathfinder:**
   ```python
   # In Isaac Sim Script Editor:
   exec(open(r"c:\Users\andre\Documents\BAH_PROJECT_2025\M_O_M\scripts\setup_env.py").read())
   exec(open(r"c:\Users\andre\Documents\BAH_PROJECT_2025\M_O_M\scripts\isaac_sim_pathfinder.py").read())
   ```

2. **Use the pathfinder class:**
   ```python
   # Create pathfinder instance
   pathfinder = IsaacSimPathfinder()
   
   # Move to target coordinates
   pathfinder.move_to_target((5, 5, 5))
   
   # Adjust movement speed
   pathfinder.set_movement_speed(0.3)  # 0.3 seconds between moves
   pathfinder.move_to_target((0, 0, 0))
   ```

3. **Or use the convenience function:**
   ```python
   # Quick movement with custom settings
   move_cube_to_target((3, -2, 4), movement_delay=0.4)
   ```

## Key Features

### Pathfinding Algorithm
- Moves step-by-step from start to end coordinates
- Prioritizes X-axis, then Y-axis, then Z-axis movement
- Each step moves by 1 unit in the appropriate direction
- Provides detailed console output showing each movement

### Isaac Sim Integration
- **Cube Position Management**: Get and set cube positions using USD transforms
- **Visual Movement**: Actual cube movement in the 3D viewport
- **Configurable Speed**: Adjustable delay between movements
- **Error Handling**: Checks for cube existence and proper setup
- **Both Sync and Async**: Support for blocking and non-blocking movement

### Movement Control Options
1. **Automatic Path Following**: Calculate and follow a path automatically
2. **Manual Control**: Use arrow keys for manual cube movement
3. **Programmatic Control**: Direct position setting via functions

## Example Usage Scenarios

### Scenario 1: Basic Pathfinding
```python
# Set up environment
exec(open(r"path_to_setup_env.py").read())

# Load pathfinder
exec(open(r"path_to_isaac_sim_pathfinder.py").read())

# Move cube from current position to (5, 5, 5)
move_cube_to_target((5, 5, 5))
```

### Scenario 2: Multiple Waypoints
```python
# Load the pathfinder
exec(open(r"path_to_isaac_sim_pathfinder.py").read())

pathfinder = IsaacSimPathfinder()
pathfinder.set_movement_speed(0.2)  # Fast movement

# Move through multiple waypoints
waypoints = [(2, 2, 2), (5, 2, 2), (5, 5, 2), (5, 5, 5)]

for waypoint in waypoints:
    print(f"Moving to waypoint: {waypoint}")
    pathfinder.move_to_target(waypoint, use_async=False)
    print("Waypoint reached!")
```

### Scenario 3: Interactive Testing
```python
# Load the pathfinder
exec(open(r"path_to_isaac_sim_pathfinder.py").read())

# Run tests
test_basic_movement()      # Test position get/set
test_pathfinding()         # Test (0,0,0) to (5,5,5)
test_custom_path()         # Test custom coordinates
```

## Important Notes

1. **Run from Isaac Sim**: These scripts must be executed from within Isaac Sim 4.5.0's Script Editor or through the Python console.

2. **Cube Setup**: Make sure to run `setup_env.py` first to create the cube at `/World/Cube`.

3. **Import Errors**: Import errors for `pxr` and `omni` are expected when viewing the files outside Isaac Sim.

4. **Movement Speed**: Adjust `movement_delay` to control how fast the cube moves (lower values = faster movement).

5. **Coordinate System**: Uses Isaac Sim's coordinate system where:
   - X: Forward/Backward
   - Y: Left/Right  
   - Z: Up/Down

## Troubleshooting

- **"Cube not found" error**: Ensure you've run `setup_env.py` to create the cube
- **No movement**: Check that the cube exists at `/World/Cube` path
- **Import errors**: These are normal when viewing outside Isaac Sim
- **Slow movement**: Reduce the `movement_delay` parameter

## Next Steps

You can extend this system by:
1. Adding obstacle avoidance
2. Implementing more sophisticated pathfinding algorithms (A*, Dijkstra)
3. Adding smooth interpolated movement instead of discrete steps
4. Creating a visual path preview before movement
5. Adding multiple cube support
