# Running Drone RL in Isaac Sim

This guide shows you how to integrate your drone RL environment with Isaac Sim for realistic physics simulation.

## Prerequisites

1. **Isaac Sim Installation**: You need Isaac Sim 4.5.0 or later installed via Omniverse Launcher
2. **Python Environment**: Your `isaac-rl-env` virtual environment with the required packages

## Files Overview

- `isaac_sim_drone_env.py` - Isaac Sim integrated drone environment
- `train_isaac_sim.py` - Training script with Isaac Sim support
- `setup_isaac_sim.py` - Isaac Sim environment setup
- `run_in_isaac_sim.py` - Complete Isaac Sim integration script

## Method 1: Quick Start (Fallback Mode)

If you don't have Isaac Sim installed, you can still run the new training script:

```bash
# Test the Isaac Sim training script in fallback mode
python train_isaac_sim.py --mode demo --no-isaac
```

This will use the simple simulation but with the new Isaac Sim-ready codebase.

## Method 2: Full Isaac Sim Integration

### Step 1: Launch Isaac Sim

1. Open Omniverse Launcher
2. Launch Isaac Sim
3. Create a new scene or use an empty scene

### Step 2: Set Up Environment

In Isaac Sim's Script Editor, run:

```python
# Navigate to your project directory
import os
os.chdir(r"/home/omniverse/isaac_projects/M_O_M/isaac-rl")

# Load setup script
exec(open("setup_isaac_sim.py").read())

# Run setup
run_isaac_sim_setup()
```

### Step 3: Test the Environment

```python
# Quick test
quick_test()
```

### Step 4: Train the Agent

```python
# Load training script
exec(open("train_isaac_sim.py").read())

# Run quick demo
run_isaac_sim_demo()

# Or train with specific parameters
train_isaac_sim_drone(use_isaac_sim=True, total_timesteps=20000)
```

## Method 3: Manual Integration

### Step 1: Prepare Isaac Sim Scene

1. In Isaac Sim, create a new scene
2. Add a ground plane: `Create > Physics > Ground Plane`
3. Set up lighting: `Create > Light > Dome Light`

### Step 2: Load Python Environment

```python
import sys
sys.path.append("/home/omniverse/isaac_projects/M_O_M/isaac-rl")

# Import the environment
from isaac_sim_drone_env import IsaacSimDroneEnv

# Create environment
env = IsaacSimDroneEnv(use_isaac_sim=True)
```

### Step 3: Test Manual Control

```python
# Reset environment
obs, info = env.reset()
print(f"Initial observation: {obs}")

# Take some random actions
for i in range(10):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step {i}: Action={action}, Reward={reward:.3f}")
    
    if terminated or truncated:
        print("Episode finished!")
        break
```

### Step 4: Train with Stable-Baselines3

```python
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

# Wrap environment
env = Monitor(env)

# Create and train model
model = PPO("MlpPolicy", env, verbose=1, device="cpu")
model.learn(total_timesteps=10000)

# Save model
model.save("isaac_sim_drone_model")
```

## Key Features

### Isaac Sim Integration

- **Realistic Physics**: Uses Isaac Sim's PhysX engine
- **3D Visualization**: See the drone moving in real-time
- **Collision Detection**: Proper collision handling
- **Configurable Environment**: Easily modify scene elements

### Fallback Support

- **Automatic Detection**: Detects if Isaac Sim is available
- **Graceful Degradation**: Falls back to simple simulation
- **Same Interface**: Consistent API regardless of backend

### Training Features

- **GPU/CPU Support**: Optimized for both CPU and GPU training
- **Tensorboard Logging**: Monitor training progress
- **Evaluation Callbacks**: Automatic model evaluation
- **Model Saving**: Automatic model checkpointing

## Environment Details

### Drone (Blue Cube)
- Starting position: (0, 0, 1)
- Size: 0.2 x 0.2 x 0.1 meters
- Physics: Dynamic body with mass

### Goal (Green Cube)
- Position: (5, 5, 1)
- Size: 0.3 x 0.3 x 0.3 meters
- Physics: Static visual marker

### Threat (Red Cube)
- Position: (2.5, 2.5, 1)
- Size: 0.4 x 0.4 x 0.4 meters
- Physics: Static obstacle

### Actions
- 3D velocity commands: [vx, vy, vz]
- Range: [-1, 1] (scaled internally)

### Observations
- Drone position (3D)
- Goal position (3D)
- Threat position (3D)
- Distance to goal (1D)
- Total: 10-dimensional vector

### Rewards
- Distance penalty: -0.1 × distance_to_goal
- Threat penalty: -10.0 if within 1.0 units
- Goal reward: +100.0 if within 0.5 units
- Step penalty: -0.01 per step

## Troubleshooting

### Common Issues

1. **"Import omni could not be resolved"**
   - These errors are expected outside Isaac Sim
   - The environment automatically falls back to simple simulation

2. **"Isaac Sim not available"**
   - Install Isaac Sim via Omniverse Launcher
   - Or run with `--no-isaac` flag for fallback mode

3. **Slow training in Isaac Sim**
   - Use `device="cpu"` in PPO settings
   - Reduce render frequency
   - Lower timesteps for testing

4. **Environment doesn't reset properly**
   - Make sure Isaac Sim scene is properly set up
   - Check that all objects exist in the scene

### Performance Tips

- **Use CPU for training**: Isaac Sim + GPU RL can conflict
- **Reduce render frequency**: Set `render=False` in step function for faster training
- **Monitor memory usage**: Isaac Sim can be memory intensive
- **Save frequently**: Use callbacks to save model during training

## Advanced Usage

### Custom Scenes

You can modify the scene by editing `isaac_sim_drone_env.py`:

```python
# Add more obstacles
self.obstacle = DynamicCuboid(
    prim_path="/World/Obstacle",
    name="obstacle",
    position=np.array([3.0, 3.0, 1.0]),
    size=np.array([0.5, 0.5, 2.0]),
    color=np.array([0.8, 0.8, 0.0])  # Yellow
)
```

### Different Drone Models

Replace the simple cube with a proper drone model:

```python
# Load drone USD file
from omni.isaac.core.utils.stage import add_reference_to_stage
drone_asset_path = "/path/to/drone.usd"
add_reference_to_stage(usd_path=drone_asset_path, prim_path="/World/Drone")
```

### Multiple Agents

Extend the environment for multi-agent training:

```python
# Create multiple drones
for i in range(num_drones):
    drone = DynamicCuboid(
        prim_path=f"/World/Drone_{i}",
        name=f"drone_{i}",
        position=start_positions[i],
        # ... other parameters
    )
```

## Next Steps

1. **Experiment with hyperparameters**: Adjust PPO settings for better performance
2. **Add complexity**: Include moving obstacles, wind effects, or battery constraints
3. **Multi-agent setup**: Train multiple drones simultaneously
4. **Real drone integration**: Use trained models on real hardware
5. **Advanced sensors**: Add camera, lidar, or other sensor data to observations

## Command Reference

```bash
# Demo mode (works without Isaac Sim)
python train_isaac_sim.py --mode demo --no-isaac

# Training mode
python train_isaac_sim.py --mode train --timesteps 50000

# Testing mode
python train_isaac_sim.py --mode test --model isaac_sim_drone_model

# Force simple simulation
python train_isaac_sim.py --mode demo --no-isaac
```
