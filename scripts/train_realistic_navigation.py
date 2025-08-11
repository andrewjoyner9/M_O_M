#!/usr/bin/env python3
"""
Enhanced Training Script for Realistic Navigation Visualization

This script trains the drone with realistic configurations and saves detailed
path data that can be visualized with visualize_paths.py
"""

import numpy as np
import torch
import yaml
import time
import os
import pickle
from pathlib import Path
from collections import deque
import matplotlib.pyplot as plt

# Import our enhanced environment
from simple_drone_env import SimpleDroneEnv

def create_enhanced_gymnasium_wrapper():
    """Create an enhanced Gymnasium wrapper with path tracking"""
    
    try:
        import gymnasium as gym
        from gymnasium.spaces import Box
        base_class = gym.Env
        gym_type = "Gymnasium"
    except ImportError:
        try:
            import gym
            from gym.spaces import Box
            base_class = gym.Env  
            gym_type = "OpenAI Gym"
        except ImportError:
            print("❌ No gym library found! Please install gymnasium or gym:")
            print("   pip install gymnasium")
            return None
    
    print(f"✓ Using {gym_type}")
    
    class EnhancedGymnasiumDroneEnv(base_class):
        """Enhanced drone environment with path tracking for visualization"""
        
        metadata = {"render_modes": ["human"], "render_fps": 4}
        
        def __init__(self, config=None):
            super().__init__()
            
            # Create the underlying environment
            self.env = SimpleDroneEnv(config=config)
            
            # Set up spaces properly
            self.observation_space = Box(
                low=-np.inf, high=np.inf, 
                shape=(self.env.obs_dim,), dtype=np.float32
            )
            self.action_space = Box(
                low=-1.0, high=1.0, 
                shape=(self.env.act_dim,), dtype=np.float32
            )
            
            # Path tracking for visualization
            self.episode_paths = []
            self.current_episode_path = []
            self.episode_count = 0
            
        def reset(self, seed=None, options=None):
            if seed is not None:
                np.random.seed(seed)
            
            # Save previous episode if it exists
            if self.current_episode_path:
                self._save_episode_path()
            
            # Reset environment
            obs = self.env.reset()
            
            # Start new episode tracking
            self.current_episode_path = [self.env.drone_position.copy()]
            self.episode_start_pos = self.env.drone_position.copy()
            self.episode_goal_pos = self.env.goal_position.copy()
            self.episode_obstacles = []
            
            # Store obstacle information
            for i, pos in enumerate(self.env.obstacle_positions):
                radius = self.env.obstacle_radii[i] if i < len(self.env.obstacle_radii) else 1.0
                self.episode_obstacles.append((pos.copy(), radius))
            
            return obs, {}
            
        def step(self, action):
            obs, reward, terminated, info = self.env.step(action)
            
            # Track path
            self.current_episode_path.append(self.env.drone_position.copy())
            
            truncated = False
            return obs, reward, terminated, truncated, info
            
        def _save_episode_path(self):
            """Save current episode path data"""
            if len(self.current_episode_path) < 2:
                return
            
            final_pos = self.current_episode_path[-1]
            final_distance = np.linalg.norm(final_pos - self.episode_goal_pos)
            success = final_distance <= self.env.config.get('goal_threshold', 1.0)
            
            episode_data = {
                'path': self.current_episode_path.copy(),
                'start': self.episode_start_pos.copy(),
                'goal': self.episode_goal_pos.copy(),
                'obstacles': self.episode_obstacles.copy(),
                'success': success,
                'steps': len(self.current_episode_path),
                'final_distance': final_distance,
                'final_reward': getattr(self, '_last_episode_reward', 0)
            }
            
            self.episode_paths.append(episode_data)
            self.episode_count += 1
            
            # Save to file periodically for visualization
            if self.episode_count % 50 == 0:
                self.save_paths_to_file()
            
        def save_paths_to_file(self, save_dir="./logs_realistic_navigation"):
            """Save episode paths for visualization"""
            save_dir = Path(save_dir)
            save_dir.mkdir(exist_ok=True, parents=True)
            
            if self.episode_paths:
                save_file = save_dir / "episode_paths.pkl"
                with open(save_file, 'wb') as f:
                    pickle.dump(self.episode_paths, f)
                print(f"📊 Saved {len(self.episode_paths)} episode paths to {save_file}")
            
        def get_episode_paths(self):
            """Get all episode paths for analysis"""
            return self.episode_paths.copy()
            
        def render(self, mode="human"):
            return self.env.render(mode)
            
        def close(self):
            # Save final paths before closing
            if self.current_episode_path:
                self._save_episode_path()
            self.save_paths_to_file()
            self.env.close()
    
    return EnhancedGymnasiumDroneEnv

def train_realistic_navigation(config_file="config_realistic_navigation.yaml"):
    """Train with realistic navigation that generates good visualization data"""
    
    print("🚁 Realistic Navigation Training")
    print("=" * 50)
    
    # Load configuration
    if Path(config_file).exists():
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        print(f"✓ Loaded config from {config_file}")
    else:
        print(f"⚠️ Config file {config_file} not found, using defaults")
        config = {
            'environment': {
                'arena_size': 15.0, 
                'num_obstacles': 6, 
                'min_start_goal_distance': 8.0
            },
            'training': {'timesteps': 200000}
        }
    
    # Create enhanced environment
    EnvClass = create_enhanced_gymnasium_wrapper()
    if EnvClass is None:
        print("❌ Cannot create Gymnasium wrapper")
        return
    
    # Create environment with realistic config
    env_config = config.get('environment', {})
    env = EnvClass(config=env_config)
    
    print(f"✓ Enhanced environment created:")
    print(f"  Arena size: {env_config.get('arena_size', 15.0)}")
    print(f"  Obstacles: {env_config.get('num_obstacles', 6)}")
    print(f"  Min start-goal distance: {env_config.get('min_start_goal_distance', 8.0)}")
    print(f"  Goal threshold: {env_config.get('goal_threshold', 1.0)}")
    print(f"  Max steps: {env_config.get('max_steps', 800)}")
    
    # Import and setup PPO
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import BaseCallback
        
        class PathTrackingCallback(BaseCallback):
            """Callback to track episode rewards for path data"""
            def __init__(self, verbose=0):
                super().__init__(verbose)
                self.episode_rewards = []
            
            def _on_step(self) -> bool:
                # Track episode rewards
                if self.locals.get('dones', [False])[0]:
                    reward = self.locals.get('rewards', [0])[0]
                    if hasattr(self.training_env.envs[0], '_last_episode_reward'):
                        self.training_env.envs[0]._last_episode_reward += reward
                    else:
                        self.training_env.envs[0]._last_episode_reward = reward
                return True
        
        print("✓ Using PPO for realistic navigation training")
        
        # Create PPO model with good hyperparameters for navigation
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=2048,         # Longer episodes
            batch_size=256,       # Larger batches
            n_epochs=10,          # More training per batch
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,        # Some exploration
            vf_coef=0.5,
            verbose=1,
            device="cpu",
            tensorboard_log="./logs_realistic_navigation"
        )
        
        # Setup callback
        callback = PathTrackingCallback()
        
        # Train
        total_timesteps = config.get('training', {}).get('timesteps', 200000)
        print(f"🚀 Training for {total_timesteps:,} timesteps...")
        print(f"📊 Path data will be saved every 50 episodes")
        print(f"🎥 Use 'python3 visualize_paths.py' to view paths during training")
        
        start_time = time.time()
        model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=True)
        training_time = time.time() - start_time
        
        # Save final model
        os.makedirs("./models_realistic", exist_ok=True)
        model.save("./models_realistic/ppo_realistic_navigation")
        print(f"💾 Model saved to ./models_realistic/ppo_realistic_navigation.zip")
        
        # Save final paths
        env.save_paths_to_file()
        
        # Training summary
        total_episodes = len(env.episode_paths)
        successful_episodes = sum(1 for ep in env.episode_paths if ep['success'])
        success_rate = (successful_episodes / total_episodes * 100) if total_episodes > 0 else 0
        
        print(f"\n🎉 Training Complete!")
        print(f"⏱️  Training time: {training_time/60:.1f} minutes")
        print(f"📈 Total episodes: {total_episodes}")
        print(f"✅ Success rate: {success_rate:.1f}% ({successful_episodes}/{total_episodes})")
        
        if total_episodes > 0:
            avg_steps = np.mean([ep['steps'] for ep in env.episode_paths])
            avg_distance = np.mean([np.linalg.norm(np.array(ep['goal']) - np.array(ep['start'])) 
                                   for ep in env.episode_paths])
            print(f"📊 Average episode length: {avg_steps:.1f} steps")
            print(f"📏 Average start-goal distance: {avg_distance:.1f}")
        
        # Quick evaluation
        print(f"\n🧪 Quick evaluation...")
        evaluate_realistic_model(model, env, episodes=3)
        
        return model, env.episode_paths
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None, []

def evaluate_realistic_model(model, env, episodes=5):
    """Evaluate the model with realistic scenarios"""
    
    print(f"\nRunning {episodes} evaluation episodes...")
    
    for episode in range(episodes):
        obs, _ = env.reset()
        episode_reward = 0
        steps = 0
        done = False
        
        start_pos = env.env.drone_position.copy()
        goal_pos = env.env.goal_position.copy()
        initial_distance = np.linalg.norm(goal_pos - start_pos)
        
        print(f"\nEpisode {episode + 1}:")
        print(f"  Start: {start_pos}")
        print(f"  Goal: {goal_pos}")
        print(f"  Distance: {initial_distance:.2f}")
        
        while not done and steps < 1000:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            done = terminated or truncated
        
        final_pos = env.env.drone_position.copy()
        final_distance = np.linalg.norm(goal_pos - final_pos)
        success = final_distance <= env.env.config.get('goal_threshold', 1.0)
        
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"  Result: {status}")
        print(f"  Steps: {steps}")
        print(f"  Final distance: {final_distance:.2f}")
        print(f"  Reward: {episode_reward:.1f}")

def create_realistic_training_config():
    """Create a configuration that ensures longer, more realistic navigation"""
    
    config = {
        'environment': {
            'arena_size': 15.0,           # Medium-sized arena
            'arena_height': 8.0,          # Good vertical space
            'max_steps': 800,             # Plenty of time for navigation
            'num_obstacles': 6,           # Moderate challenge
            'obstacle_radius_range': [0.8, 1.2],  # Reasonable obstacle sizes
            'goal_threshold': 1.0,        # Must get fairly close to goal
            'obstacle_clearance': 2.5,    # Safe clearance around start/goal
            'min_start_goal_distance': 8.0,  # NEW: Force minimum distance
            'difficulty_level': 'realistic'
        },
        'reward_shaping': {
            'success_reward': 100.0,
            'collision_penalty': -30.0,
            'step_penalty': -0.05,
            'distance_reward_scale': 2.0,
            'progress_reward_scale': 1.0,
            'altitude_penalty_scale': 0.5
        },
        'training': {
            'timesteps': 300000,          # More training for complex navigation
            'save_path_frequency': 50,    # Save paths every 50 episodes for visualization
            'eval_frequency': 10000       # Evaluate every 10k steps
        }
    }
    
    config_path = Path('./config_realistic_navigation.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print(f"✅ Realistic navigation config saved to: {config_path}")
    return config_path

def main():
    """Main training function"""
    
    print("🎯 Setting up Realistic Navigation Training")
    print("=" * 50)
    
    # Create configuration
    config_file = create_realistic_training_config()
    
    # Run training
    model, episode_paths = train_realistic_navigation(config_file)
    
    if episode_paths:
        print(f"\n🎨 Visualization Instructions:")
        print("=" * 40)
        print("1. To view paths in real-time during training:")
        print("   python3 visualize_paths.py")
        print()
        print("2. To view paths with custom update interval:")
        print("   python3 visualize_paths.py --interval 5.0")
        print()
        print("3. Path data is saved in:")
        print("   ./logs_realistic_navigation/episode_paths.pkl")
        print()
        print("4. The visualization will show:")
        print("   • 3D flight paths with actual navigation")
        print("   • Top-down view with start/goal/obstacles")
        print("   • Success rate over time")
        print("   • Path efficiency metrics")

if __name__ == "__main__":
    main()
