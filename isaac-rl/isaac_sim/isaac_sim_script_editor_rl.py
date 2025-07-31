import os
os.chdir(r"/home/omniverse/isaac_projects/M_O_M/isaac-rl/isaac_sim")

# Set up environment
exec(open("setup_isaac_sim.py").read())
run_isaac_sim_setup()

# Run training
exec(open("train_isaac_sim.py").read())
run_isaac_sim_demo()