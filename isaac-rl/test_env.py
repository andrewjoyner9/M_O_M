#!/usr/bin/env python3
"""Test script to verify the drone environment works correctly."""

from drone_env import DroneEnv
import numpy as np

def test_environment():
    print("Testing DroneEnv...")
    
    # Create environment
    env = DroneEnv()
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Test reset
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial observation: {obs}")
    
    # Test a few steps
    for i in range(5):
        action = env.action_space.sample()  # Random action
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: Action={action}, Reward={reward:.3f}, Done={terminated or truncated}")
        print(f"  Drone position: {obs[:3]}")
        
        if terminated or truncated:
            print("Episode finished!")
            break
    
    print("Environment test completed successfully!")

if __name__ == "__main__":
    test_environment()
