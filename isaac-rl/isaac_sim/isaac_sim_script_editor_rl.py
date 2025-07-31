# Fix Python path for Isaac Sim to find packages
import sys
import os

# Add virtual environment to Python path
venv_path = "/home/omniverse/isaac_projects/M_O_M/isaac-rl/isaac-rl-env/lib/python3.10/site-packages"
if venv_path not in sys.path:
    sys.path.insert(0, venv_path)

# Test imports
try:
    import stable_baselines3
    import gymnasium
    print("✅ Successfully loaded RL packages!")
    print(f"Stable Baselines3 version: {stable_baselines3.__version__}")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please check the venv_path above")

# Change to isaac_sim directory
os.chdir(r"/home/omniverse/isaac_projects/M_O_M/isaac-rl/isaac_sim")

# Set up environment
exec(open("setup_isaac_sim.py").read())
run_isaac_sim_setup()

# Run training
exec(open("train_isaac_sim.py").read())
run_isaac_sim_demo()