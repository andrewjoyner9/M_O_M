# Drone RL Environment with Gymnasium and Stable-Baselines3

This project implements a reinforcement learning environment for drone navigation using Gymnasium (formerly OpenAI Gym) and Stable-Baselines3.

## Project Structure

```
isaac-rl/
├── drone_env.py          # Main RL environment
├── train.py             # Training script with advanced features
├── demo.py              # Complete demonstration
├── test_env.py          # Environment testing
├── requirements.txt     # Python dependencies
└── isaac-rl-env/        # Virtual environment
```

## Environment Description

The `DroneEnv` simulates a 3D drone navigation task:

- **Objective**: Navigate from start position (0,0,1) to goal position (5,5,1)
- **Obstacles**: Single threat at position (2.5,2.5,1) 
- **Action Space**: Continuous 3D velocity commands (vx, vy, vz) in range [-1,1]
- **Observation Space**: 10D vector containing:
  - Drone position (x,y,z)
  - Goal position (x,y,z) 
  - Threat position (x,y,z)
  - Distance to goal

## Reward Function

- **Distance penalty**: -0.1 × distance_to_goal
- **Threat penalty**: -10.0 if within 1.0 units of threat
- **Goal reward**: +100.0 if within 0.5 units of goal
- **Step penalty**: -0.01 per step (encourages efficiency)

## Getting Started

### 1. Environment Setup

The virtual environment is already configured. To activate it manually:

```bash
source isaac-rl-env/bin/activate
```

### 2. Test the Environment

```bash
python test_env.py
```

This will verify the environment works correctly and show sample interactions.

### 3. Run the Complete Demo

```bash
python demo.py
```

This demonstrates:
- Environment basics
- Hand-coded smart agent (for comparison)
- RL training with PPO
- Testing the trained agent

### 4. Full Training

```bash
python train.py
```

Features:
- Advanced PPO configuration
- Evaluation callbacks
- Early stopping on reward threshold
- Tensorboard logging
- Model testing after training

## Key Components

### DroneEnv Class (`drone_env.py`)

```python
class DroneEnv(gym.Env):
    def __init__(self):
        # Define action and observation spaces
        # Initialize simulation state
        
    def reset(self, seed=None, options=None):
        # Reset environment to initial state
        # Returns: observation, info
        
    def step(self, action):
        # Apply action and advance simulation
        # Returns: observation, reward, terminated, truncated, info
        
    def _get_observation(self):
        # Build state vector
        
    def _get_reward(self):
        # Calculate reward based on current state
        
    def _check_done(self):
        # Check termination conditions
```

### Training Script (`train.py`)

- Uses PPO (Proximal Policy Optimization)
- Includes evaluation callbacks
- Supports Tensorboard logging
- Automatically stops when reward threshold is reached
- Tests the trained model

## Customization

### Modify the Environment

1. **Change goal/threat positions**: Edit `__init__` and `reset` methods
2. **Adjust reward function**: Modify `_get_reward` method
3. **Add more obstacles**: Extend observation space and reward calculation
4. **Change dynamics**: Modify `step` method physics

### Try Different Algorithms

Replace PPO with other Stable-Baselines3 algorithms:

```python
from stable_baselines3 import SAC, TD3, A2C

# For continuous control
model = SAC("MlpPolicy", env, verbose=1)
model = TD3("MlpPolicy", env, verbose=1)
```

### Hyperparameter Tuning

Modify training parameters in `train.py`:

```python
model = PPO(
    "MlpPolicy", 
    env,
    learning_rate=3e-4,    # Learning rate
    n_steps=2048,          # Steps per rollout
    batch_size=64,         # Batch size
    n_epochs=10,           # Training epochs per rollout
    gamma=0.99,            # Discount factor
    # ... other parameters
)
```

## Next Steps

1. **Isaac Sim Integration**: Replace simple physics with Isaac Sim
2. **Complex Environments**: Add multiple threats, moving obstacles
3. **Sensor Simulation**: Add camera, lidar, or other sensors
4. **Multi-Agent**: Multiple drones with cooperation/competition
5. **Curriculum Learning**: Gradually increase difficulty
6. **Domain Randomization**: Vary environment parameters

## Troubleshooting

### Common Issues

1. **Import errors**: Make sure virtual environment is activated
2. **CUDA warnings**: Normal for PPO, can use `device='cpu'` to force CPU
3. **Training slow**: Reduce `total_timesteps` for faster testing

### Performance Tips

- Use CPU for MLP policies (they're not optimized for GPU)
- Monitor training with Tensorboard: `tensorboard --logdir ./drone_tensorboard/`
- Adjust reward function if agent doesn't learn effectively

## Dependencies

Key packages installed:
- `gymnasium` - RL environment interface
- `stable-baselines3[extra]` - RL algorithms and utilities
- `numpy` - Numerical computing
- `torch` - Deep learning backend

The full environment includes many other packages for potential Isaac Sim integration.
