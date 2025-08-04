#!/usr/bin/env python3
"""
Training script for Isaac Sim integrated drone environment.
This script can run both with Isaac Sim (if available) and without (fallback mode).
"""

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
import sys
import os

# Try to import Isaac Sim environment
try:
    from isaac_sim_drone_env import IsaacSimDroneEnv, create_drone_env
    ISAAC_ENV_AVAILABLE = True
except ImportError:
    # Fallback to original environment
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'original'))
    from drone_env import DroneEnv
    ISAAC_ENV_AVAILABLE = False
    print("Isaac Sim environment not available, using fallback environment")

def train_isaac_sim_drone(use_isaac_sim=True, total_timesteps=50000):
    """
    Train drone agent with Isaac Sim integration
    
    Args:
        use_isaac_sim: Whether to use Isaac Sim physics (if available)
        total_timesteps: Number of training timesteps
    """
    
    print("=== Isaac Sim Drone RL Training ===")
    
    # Create environment
    if ISAAC_ENV_AVAILABLE:
        print(f"Creating Isaac Sim environment (use_isaac_sim={use_isaac_sim})")
        env = create_drone_env(use_isaac_sim=use_isaac_sim)
        env = Monitor(env)
        
        # Create evaluation environment
        eval_env = create_drone_env(use_isaac_sim=use_isaac_sim)
        eval_env = Monitor(eval_env)
    else:
        print("Using fallback DroneEnv")
        env = DroneEnv()
        env = Monitor(env)
        eval_env = DroneEnv()
        eval_env = Monitor(eval_env)
    
    # Set up callbacks
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=50, verbose=1)
    eval_callback = EvalCallback(
        eval_env, 
        callback_on_new_best=callback_on_best,
        eval_freq=5000,
        deterministic=True,
        render=False,
        verbose=1
    )
    
    # Create PPO model with Isaac Sim optimized settings
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,  # Encourage exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log="./isaac_drone_tensorboard/",
        device="cpu"  # Use CPU for better stability with Isaac Sim
    )
    
    print(f"Starting training for {total_timesteps} timesteps...")
    
    # Train the model
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        progress_bar=True
    )
    
    # Save the model
    model_name = "isaac_sim_drone_model" if (ISAAC_ENV_AVAILABLE and use_isaac_sim) else "sim_drone_model"
    model.save(model_name)
    print(f"Model saved as {model_name}.zip")
    
    return model, env

def test_isaac_sim_model(model_path=None, use_isaac_sim=True, num_episodes=5):
    """
    Test the trained model in Isaac Sim
    
    Args:
        model_path: Path to saved model (if None, uses default)
        use_isaac_sim: Whether to use Isaac Sim for testing
        num_episodes: Number of test episodes
    """
    
    print("=== Testing Isaac Sim Drone Model ===")
    
    # Create test environment
    if ISAAC_ENV_AVAILABLE:
        env = create_drone_env(use_isaac_sim=use_isaac_sim)
    else:
        env = DroneEnv()
    
    # Load model
    if model_path is None:
        model_path = "isaac_sim_drone_model" if (ISAAC_ENV_AVAILABLE and use_isaac_sim) else "sim_drone_model"
    
    try:
        model = PPO.load(model_path)
        print(f"Loaded model from {model_path}")
    except FileNotFoundError:
        print(f"Model {model_path} not found. Please train first.")
        return
    
    # Test episodes
    total_rewards = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        steps = 0
        
        print(f"\nEpisode {episode + 1}/{num_episodes}")
        
        for step in range(1000):  # Max steps per episode
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            
            # Print progress every 50 steps
            if step % 50 == 0:
                print(f"  Step {step}: Reward={reward:.3f}, Total={episode_reward:.3f}")
                if ISAAC_ENV_AVAILABLE and use_isaac_sim:
                    try:
                        print(f"    Drone pos: {obs[:3]}")
                    except:
                        pass
            
            if terminated or truncated:
                print(f"  Episode finished at step {steps}")
                if terminated:
                    if obs[9] < 0.5:  # Distance to goal
                        print("  SUCCESS: Goal reached!")
                    else:
                        print("  FAILED: Crashed into threat")
                break
        
        total_rewards.append(episode_reward)
        print(f"  Episode {episode + 1} reward: {episode_reward:.3f}")
    
    # Summary
    avg_reward = sum(total_rewards) / len(total_rewards)
    print(f"\n=== Test Results ===")
    print(f"Average reward: {avg_reward:.3f}")
    print(f"Best episode: {max(total_rewards):.3f}")
    print(f"Worst episode: {min(total_rewards):.3f}")
    
    # Clean up
    env.close()
    
    return total_rewards

def run_isaac_sim_demo():
    """
    Run a complete demo showing Isaac Sim integration
    """
    print("=== Isaac Sim Drone RL Demo ===")
    
    if not ISAAC_ENV_AVAILABLE:
        print("Isaac Sim environment not available. Running with fallback.")
        use_isaac_sim = False
    else:
        # Try Isaac Sim first, fallback if needed
        print("Attempting to use Isaac Sim physics...")
        use_isaac_sim = True
    
    # Quick training
    print("\n1. Quick training (10k timesteps)")
    model, env = train_isaac_sim_drone(use_isaac_sim=use_isaac_sim, total_timesteps=10000)
    
    # Test the model
    print("\n2. Testing trained model")
    test_isaac_sim_model(use_isaac_sim=use_isaac_sim, num_episodes=3)
    
    # Clean up
    env.close()
    
    print("\n=== Demo completed! ===")
    print("\nNext steps:")
    print("1. Install Isaac Sim for full physics simulation")
    print("2. Run longer training with: train_isaac_sim_drone(total_timesteps=100000)")
    print("3. Experiment with different reward functions")
    print("4. Add more complex environments and obstacles")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Isaac Sim Drone RL Training")
    parser.add_argument("--mode", choices=["train", "test", "demo"], default="demo",
                        help="Mode to run: train, test, or demo")
    parser.add_argument("--timesteps", type=int, default=50000,
                        help="Number of training timesteps")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to model for testing")
    parser.add_argument("--no-isaac", action="store_true",
                        help="Force use of simple simulation instead of Isaac Sim")
    
    args = parser.parse_args()
    
    use_isaac_sim = not args.no_isaac
    
    if args.mode == "train":
        print("Starting training...")
        train_isaac_sim_drone(use_isaac_sim=use_isaac_sim, total_timesteps=args.timesteps)
        
    elif args.mode == "test":
        print("Starting testing...")
        test_isaac_sim_model(model_path=args.model, use_isaac_sim=use_isaac_sim)
        
    elif args.mode == "demo":
        run_isaac_sim_demo()
    
    print("Done!")
