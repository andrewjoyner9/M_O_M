# Isaac Sim Drone RL Integration

This folder contains the Isaac Sim integrated version of the drone RL environment with realistic physics simulation.

## Files

- `isaac_sim_drone_env.py` - Isaac Sim integrated environment with fallback
- `train_isaac_sim.py` - Advanced training script with Isaac Sim support
- `setup_isaac_sim.py` - Isaac Sim environment setup script
- `isaac_sim_script_editor_rl.py` - Complete script for Isaac Sim Script Editor
- `ISAAC_SIM_GUIDE.md` - Detailed integration guide

## Quick Start

### Without Isaac Sim (Fallback Mode)
```bash
python train_isaac_sim.py --mode demo --no-isaac
```

### With Isaac Sim
1. Launch Isaac Sim via Omniverse Launcher
2. In Isaac Sim Script Editor:
```python
exec(open("isaac_sim_script_editor_rl.py").read())
```

## Features

- **Realistic Physics**: PhysX engine when Isaac Sim is available
- **Automatic Fallback**: Works with or without Isaac Sim
- **3D Visualization**: Real-time drone movement in 3D environment
- **Advanced Training**: PPO with callbacks, evaluation, early stopping
- **Same API**: Consistent interface regardless of backend

## Usage Options

### Command Line Training
```bash
# Demo mode (works without Isaac Sim)
python train_isaac_sim.py --mode demo --no-isaac

# Training mode (requires Isaac Sim)
python train_isaac_sim.py --mode train --timesteps 50000

# Testing mode
python train_isaac_sim.py --mode test --model isaac_sim_drone_model
```

### Isaac Sim Script Editor
```python
# Copy isaac_sim_script_editor_rl.py content into Isaac Sim Script Editor
# Or run: exec(open("isaac_sim_script_editor_rl.py").read())
```

## Environment Details

### Isaac Sim Mode
- **Physics**: PhysX engine with realistic dynamics
- **Visualization**: 3D scene with lighting and textures
- **Objects**: Dynamic drone cube, static goal and threat markers
- **Collision**: Proper collision detection and response

### Fallback Mode
- **Physics**: Simple mathematical simulation
- **Performance**: Fast execution for development/testing
- **Compatibility**: Works on any system with Python

## Requirements

- **Optional**: Isaac Sim 4.5.0+ via Omniverse Launcher
- **Required**: gymnasium, stable-baselines3, numpy, torch
- **Fallback**: Automatically uses simple simulation if Isaac Sim unavailable

## File Descriptions

- **isaac_sim_drone_env.py**: Main environment class with Isaac Sim integration
- **train_isaac_sim.py**: Complete training pipeline with argument parsing
- **setup_isaac_sim.py**: Isaac Sim scene setup (lights, physics, ground plane)
- **isaac_sim_script_editor_rl.py**: One-file solution for Isaac Sim Script Editor
- **ISAAC_SIM_GUIDE.md**: Step-by-step integration instructions
