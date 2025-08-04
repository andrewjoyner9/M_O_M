# Original Drone RL Environment

This folder contains the original, simplified drone RL environment using basic Gymnasium simulation.

## Files

- `drone_env.py` - Original DroneEnv class with simple physics
- `train.py` - Basic training script with PPO
- `demo.py` - Complete demonstration of environment and training
- `test_env.py` - Environment testing script

## Usage

```bash
# Test the environment
python test_env.py

# Run complete demo
python demo.py

# Train the agent
python train.py
```

## Features

- **Simple Physics**: Basic 3D movement simulation
- **Fast Training**: Lightweight environment for quick iterations
- **Educational**: Easy to understand and modify
- **No Dependencies**: Only requires gymnasium and stable-baselines3

## Environment Details

- **Action Space**: 3D velocity commands (vx, vy, vz)
- **Observation Space**: 10D vector (drone pos, goal pos, threat pos, distance)
- **Goal**: Navigate from (0,0,1) to (5,5,1) while avoiding threat at (2.5,2.5,1)
- **Reward**: Distance penalty + goal reward + threat avoidance

This is perfect for:
- Learning RL concepts
- Quick prototyping
- Testing new algorithms
- Running on systems without Isaac Sim
