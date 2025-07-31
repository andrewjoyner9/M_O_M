#!/usr/bin/env python3
"""
Demo script showing how to use the drone environment and train an agent.
This demonstrates the complete workflow from environment creation to training.
"""

from stable_baselines3 import PPO
from drone_env import DroneEnv
import numpy as np

def demo_environment():
    """Demonstrate the basic environment functionality"""
    print("=== ENVIRONMENT DEMO ===")
    env = DroneEnv()
    
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Reset environment
    obs, info = env.reset()
    print(f"\nInitial state:")
    print(f"  Drone position: {obs[:3]}")
    print(f"  Goal position: {obs[3:6]}")
    print(f"  Threat position: {obs[6:9]}")
    print(f"  Distance to goal: {obs[9]:.2f}")
    
    # Take some random actions
    print(f"\nTaking 10 random actions...")
    total_reward = 0
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        print(f"Step {i+1}: Action={action}, Reward={reward:.3f}, Total={total_reward:.3f}")
        
        if terminated or truncated:
            print("Episode finished early!")
            break
    
    print(f"Final drone position: {obs[:3]}")
    print(f"Total reward: {total_reward:.3f}")

def demo_training():
    """Demonstrate training a simple agent"""
    print("\n=== TRAINING DEMO ===")
    
    # Create environment
    env = DroneEnv()
    
    # Create and train a simple PPO agent
    print("Creating PPO agent...")
    model = PPO("MlpPolicy", env, verbose=1)
    
    print("Training for 5000 timesteps...")
    model.learn(total_timesteps=5000)
    
    print("Training completed!")
    
    # Test the trained agent
    print("\nTesting trained agent...")
    obs, info = env.reset()
    total_reward = 0
    
    for i in range(100):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if i % 10 == 0:  # Print every 10 steps
            print(f"Step {i}: Drone={obs[:3]}, Reward={reward:.3f}, Total={total_reward:.3f}")
        
        if terminated or truncated:
            print(f"Episode finished at step {i}!")
            break
    
    print(f"Final total reward: {total_reward:.3f}")
    
    # Save the model
    model.save("demo_drone_model")
    print("Model saved as demo_drone_model.zip")

def demo_smart_agent():
    """Demonstrate a hand-coded smart agent for comparison"""
    print("\n=== SMART AGENT DEMO ===")
    
    env = DroneEnv()
    obs, info = env.reset()
    total_reward = 0
    
    print("Running hand-coded smart agent...")
    
    for i in range(100):
        # Smart policy: move towards goal, avoid threats
        drone_pos = obs[:3]
        goal_pos = obs[3:6]
        threat_pos = obs[6:9]
        
        # Direction to goal
        to_goal = goal_pos - drone_pos
        to_goal_norm = np.linalg.norm(to_goal)
        
        if to_goal_norm > 0:
            to_goal = to_goal / to_goal_norm
        
        # Direction away from threat
        to_threat = threat_pos - drone_pos
        threat_dist = np.linalg.norm(to_threat)
        
        if threat_dist < 2.0 and threat_dist > 0:  # If close to threat
            away_from_threat = -to_threat / threat_dist
            # Combine goal seeking with threat avoidance
            action = 0.7 * to_goal + 0.3 * away_from_threat
        else:
            action = to_goal
        
        # Clip to action space
        action = np.clip(action, -1, 1)
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if i % 10 == 0:
            print(f"Step {i}: Drone={obs[:3]}, Reward={reward:.3f}, Total={total_reward:.3f}")
        
        if terminated or truncated:
            print(f"Episode finished at step {i}!")
            break
    
    print(f"Smart agent total reward: {total_reward:.3f}")

if __name__ == "__main__":
    print("Drone Environment and Training Demo")
    print("===================================")
    
    try:
        demo_environment()
        demo_smart_agent()
        demo_training()
        
        print("\n=== DEMO COMPLETED SUCCESSFULLY ===")
        print("\nNext steps:")
        print("1. Run 'python train.py' for full training with callbacks")
        print("2. Integrate with Isaac Sim for realistic physics")
        print("3. Add more complex obstacles and dynamics")
        print("4. Experiment with different RL algorithms")
        
    except Exception as e:
        print(f"Error during demo: {e}")
        import traceback
        traceback.print_exc()
