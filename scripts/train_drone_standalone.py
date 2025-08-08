#!/usr/bin/env python3
"""
Standalone PPO Training Script for Drone Navigation (No Isaac Lab Required)

This script provides a standalone training environment that doesn't require Isaac Lab
for development and testing purposes. It includes a simplified drone environment
that mimics the behavior of your Isaac Lab task.
"""

import os
import sys
import time
import argparse
import torch
import numpy as np
from typing import Optional, Tuple, Dict, Any
import gymnasium as gym
from gymnasium.spaces import Box

# RL training imports  
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.utils import set_random_seed
    print("✓ Stable Baselines3 found")
except ImportError as e:
    print(f"❌ Error importing Stable Baselines3: {e}")
    print("Please install with: pip install stable-baselines3[extra]")
    sys.exit(1)


class SimplePathfinder:
    """Simplified pathfinder for standalone training"""
    
    def __init__(self, env_id: int = 0):
        self.env_id = env_id
        self.position = np.array([0.0, 0.0, 1.0])  # Start at center
        self.obstacles = set()
        self.spherical_obstacles = []
        
    def get_cube_position(self):
        """Get current position"""
        return self.position.copy()
        
    def set_cube_position(self, x, y, z):
        """Set position"""
        self.position = np.array([float(x), float(y), float(z)])
        return True
        
    def add_obstacle(self, x, y, z, radius=0.0):
        """Add obstacle"""
        if radius <= 0:
            self.obstacles.add((int(x), int(y), int(z)))
        else:
            self.spherical_obstacles.append((float(x), float(y), float(z), float(radius)))
            
    def clear_obstacles(self):
        """Clear all obstacles"""
        self.obstacles.clear()
        self.spherical_obstacles.clear()
        
    def is_position_blocked(self, x, y, z):
        """Check if position is blocked"""
        # Check point obstacles
        if (int(x), int(y), int(z)) in self.obstacles:
            return True
            
        # Check spherical obstacles  
        for ox, oy, oz, r in self.spherical_obstacles:
            dist = np.sqrt((x - ox)**2 + (y - oy)**2 + (z - oz)**2)
            if dist <= r:
                return True
                
        return False
        
    def find_path(self, start, end):
        """Simple pathfinding - move directly towards goal avoiding obstacles"""
        current = np.array(start, dtype=float)
        target = np.array(end, dtype=float)
        
        # Simple direct movement with obstacle avoidance
        direction = target - current
        distance = np.linalg.norm(direction)
        
        if distance < 0.1:
            return []
            
        # Normalize direction
        if distance > 0:
            direction = direction / distance
            
        # Take a step towards target
        step_size = min(1.0, distance)
        next_pos = current + direction * step_size
        
        # Check if next position is blocked
        if self.is_position_blocked(*next_pos.astype(int)):
            # Try to go around obstacle
            alternatives = [
                current + np.array([1, 0, 0]),
                current + np.array([-1, 0, 0]),
                current + np.array([0, 1, 0]),
                current + np.array([0, -1, 0]),
                current + np.array([0, 0, 1]),
                current + np.array([0, 0, -1])
            ]
            
            for alt in alternatives:
                if not self.is_position_blocked(*alt.astype(int)):
                    return [tuple(alt.astype(int))]
                    
            return []  # No valid path
        
        return [tuple(next_pos.astype(int))]
        
    def local_occupancy_grid(self, x, y, z):
        """3x3x3 occupancy grid around position"""
        grid = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    blocked = self.is_position_blocked(x + dx, y + dy, z + dz)
                    grid.append(1.0 if blocked else 0.0)
        return grid
        
    def min_distance_to_spheres(self, x, y, z):
        """Minimum distance to spherical obstacles"""
        if not self.spherical_obstacles:
            return np.inf
            
        min_dist = np.inf
        for ox, oy, oz, r in self.spherical_obstacles:
            dist = np.sqrt((x - ox)**2 + (y - oy)**2 + (z - oz)**2)
            surface_dist = max(0.0, dist - r)
            min_dist = min(min_dist, surface_dist)
            
        return min_dist


class TrainingProgressCallback(BaseCallback):
    """Simple callback to track and display training progress"""
    
    def __init__(self, verbose=1):
        super().__init__(verbose)
        self.start_time = time.time()
        self.last_log_time = time.time()
        
    def _on_step(self) -> bool:
        current_time = time.time()
        
        # Log progress every 10 seconds
        if current_time - self.last_log_time > 10:
            elapsed = current_time - self.start_time
            progress = self.num_timesteps / self.locals.get('total_timesteps', 1)
            
            print(f"Progress: {progress*100:.1f}% | "
                  f"Timesteps: {self.num_timesteps:,} | "
                  f"Time: {elapsed/60:.1f}m")
                  
            self.last_log_time = current_time
            
        return True


class SingleDroneEnv(gym.Env):
    """Single drone environment compatible with SB3"""
    
    def __init__(self, env_id: int = 0):
        super().__init__()
        
        # Environment configuration
        self.env_id = env_id
        self.arena_bounds = ((-5, -5, 0), (5, 5, 5))
        self.num_obstacles = 10
        self.obstacle_radius = 1.0
        self.min_goal_clearance = 1.5
        self.max_steps = 100
        
        # Observation and action spaces
        self.obs_dim = 33  # pos(3) + goal(3) + occupancy_grid(27)
        self.act_dim = 3   # x, y, z movement
        
        self.observation_space = Box(
            low=-np.inf, high=np.inf, 
            shape=(self.obs_dim,), dtype=np.float32
        )
        self.action_space = Box(
            low=-1.0, high=1.0, 
            shape=(self.act_dim,), dtype=np.float32
        )
        
        # Initialize pathfinder and state
        self.pathfinder = SimplePathfinder(env_id)
        self.goal = np.zeros(3)
        self.episode_length = 0
        
    def reset(self, seed=None, options=None):
        """Reset environment"""
        if seed is not None:
            np.random.seed(seed)
            
        self._reset_env()
        obs = self._compute_obs()
        info = {}
        
        return obs, info
        
    def _reset_env(self):
        """Reset the environment state"""
        low = np.array(self.arena_bounds[0])
        high = np.array(self.arena_bounds[1])
        
        # Clear obstacles
        self.pathfinder.clear_obstacles()
        
        # Start at arena center
        center = (low + high) / 2
        self.pathfinder.set_cube_position(*center)
        
        # Random goal (ensure minimum distance from center)
        while True:
            goal = np.random.uniform(low, high)
            if np.linalg.norm(goal - center) > 2.0:
                break
        self.goal = goal
        
        # Add random spherical obstacles
        added, tries = 0, 0
        while added < self.num_obstacles and tries < self.num_obstacles * 10:
            tries += 1
            obs_pos = np.random.uniform(low, high)
            
            # Ensure obstacles aren't too close to start or goal
            if (np.linalg.norm(obs_pos - center) < self.min_goal_clearance or
                np.linalg.norm(obs_pos - goal) < self.min_goal_clearance):
                continue
                
            self.pathfinder.add_obstacle(*obs_pos, radius=self.obstacle_radius)
            added += 1
            
        # Reset episode length
        self.episode_length = 0
        
    def step(self, action):
        """Step the environment"""
        action = np.array(action, dtype=np.float32)
        
        # Apply action to pathfinder
        current_pos = self.pathfinder.get_cube_position()
        target_pos = current_pos + action.clip(-1, 1)
        
        # Find path and move
        path = self.pathfinder.find_path(
            tuple(current_pos.astype(int)), 
            tuple(target_pos.astype(int))
        )
        
        if path:
            self.pathfinder.set_cube_position(*[float(p) for p in path[-1]])
            
        # Update episode length
        self.episode_length += 1
        
        # Compute reward and done
        pos = self.pathfinder.get_cube_position()
        
        # Calculate distances
        dist_goal = np.linalg.norm(pos - self.goal)
        dist_safe = self.pathfinder.min_distance_to_spheres(*pos)
        
        # Reward function
        reward = -0.1 - 0.05 * dist_goal + 0.05 * min(dist_safe, 5.0)
        
        # Done conditions
        terminated = (
            dist_goal < 0.3 or  # Reached goal
            dist_safe <= 0.0    # Collision
        )
        
        truncated = self.episode_length >= self.max_steps  # Timeout
        
        obs = self._compute_obs()
        info = {
            "dist_goal": dist_goal,
            "dist_safe": dist_safe,
            "episode_length": self.episode_length
        }
        
        return obs, reward, terminated, truncated, info
        
    def _compute_obs(self):
        """Compute observation for this environment"""
        pos = self.pathfinder.get_cube_position().astype(np.float32)
        goal = self.goal.astype(np.float32)
        occupancy = np.array(
            self.pathfinder.local_occupancy_grid(*pos.astype(int)), 
            dtype=np.float32
        )
        
        obs = np.concatenate([pos, goal, occupancy])
        return obs
        
    def render(self, mode="human"):
        """Render the environment (no-op for now)"""
        pass
        
    def close(self):
        """Close the environment"""
        pass


def train_standalone_drone(
    total_timesteps: int = 1_000_000,
    learning_rate: float = 3e-4,
    batch_size: int = 2048,
    num_envs: int = 64,
    log_dir: str = "./logs",
    model_dir: str = "./models",
    seed: Optional[int] = 42,
    save_freq: int = 100_000
):
    """Train PPO agent on standalone drone navigation task"""
    
    print("🚁 Starting Standalone Drone PPO Training")
    print("=" * 60)
    
    # Set random seed
    if seed is not None:
        set_random_seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        print(f"🎲 Random seed set to: {seed}")
    
    # Create directories
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # Create vectorized environment using SB3's vectorization
    print("🌍 Creating standalone drone environment...")
    
    # Import SB3's vectorization utilities
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    
    # Create environment factory
    def make_env(env_id: int):
        def _init():
            return SingleDroneEnv(env_id)
        return _init
    
    # Create vectorized environment
    if num_envs == 1:
        env = SingleDroneEnv(0)
    else:
        # Use DummyVecEnv for better compatibility
        env = DummyVecEnv([make_env(i) for i in range(num_envs)])
    
    print(f"✓ Environment created:")
    print(f"  • {num_envs} parallel environments")
    print(f"  • Observation dim: 33")
    print(f"  • Action dim: 3")
    print(f"  • Device: CPU (optimal for MLP policies)")
    
    # Configure logging
    logger = configure(log_dir, ["stdout", "csv", "tensorboard"])
    
    # Create PPO model
    print("🤖 Creating PPO model...")
    n_steps = batch_size // num_envs
    
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=log_dir,
        device="cpu"  # Use CPU for better MLP performance
    )
    
    model.set_logger(logger)
    
    print(f"✓ PPO model created:")
    print(f"  • Learning rate: {learning_rate}")
    print(f"  • Batch size: {batch_size}")
    print(f"  • Steps per env: {n_steps}")
    
    # Create callbacks
    callbacks = []
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=model_dir,
        name_prefix="ppo_drone_standalone"
    )
    callbacks.append(checkpoint_callback)
    
    # Progress callback
    progress_callback = TrainingProgressCallback()
    callbacks.append(progress_callback)
    
    # Start training
    print("\n🚀 Starting training...")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Estimated episodes: ~{total_timesteps // (100 * num_envs):,}")
    print(f"Logs: {log_dir}")
    print(f"Models: {model_dir}")
    print("-" * 60)
    
    start_time = time.time()
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True
        )
        
        print("\n" + "=" * 60)
        print("🎉 Training completed successfully!")
        
        # Save final model
        final_model_path = os.path.join(model_dir, "ppo_drone_standalone_final.zip")
        model.save(final_model_path)
        print(f"💾 Final model saved: {final_model_path}")
        
        # Training summary
        training_time = time.time() - start_time
        print(f"⏱️  Total training time: {training_time/3600:.2f} hours")
        
        return final_model_path
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        interrupt_model_path = os.path.join(model_dir, "ppo_drone_standalone_interrupted.zip")
        model.save(interrupt_model_path)
        print(f"💾 Model saved: {interrupt_model_path}")
        return interrupt_model_path
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        raise
        
    finally:
        env.close()


def evaluate_standalone_model(
    model_path: str,
    num_episodes: int = 10,
    num_envs: int = 1  # Use single env for evaluation
):
    """Evaluate a trained PPO model on standalone environment"""
    
    print(f"🧪 Evaluating model: {model_path}")
    print("=" * 60)
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return
    
    # Load model
    print("📦 Loading model...")
    model = PPO.load(model_path)
    
    # Create environment
    print("🌍 Creating environment...")
    env = SingleDroneEnv(0)
    
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    
    print(f"🏃 Running {num_episodes} evaluation episodes...")
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            
            # Check for success (reached goal)
            if terminated and info.get("dist_goal", float('inf')) < 0.3:
                success_count += 1
                
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Show episode result
        status = "✅ SUCCESS" if terminated and info.get("dist_goal", float('inf')) < 0.3 else "❌ FAILED"
        print(f"Episode {episode + 1:2d}: {status} | Reward = {episode_reward:7.2f} | Length = {episode_length:3d}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 EVALUATION RESULTS")
    print("=" * 60)
    print(f"Episodes evaluated: {num_episodes}")
    print(f"Mean reward:       {np.mean(episode_rewards):7.2f} ± {np.std(episode_rewards):5.2f}")
    print(f"Mean length:       {np.mean(episode_lengths):7.1f} ± {np.std(episode_lengths):5.1f}")
    print(f"Best reward:       {np.max(episode_rewards):7.2f}")
    print(f"Worst reward:      {np.min(episode_rewards):7.2f}")
    print(f"Success rate:      {success_count/num_episodes*100:5.1f}%")
    
    env.close()
    return episode_rewards, episode_lengths


def main():
    """Main function with command line argument parsing"""
    
    parser = argparse.ArgumentParser(description="Train or evaluate PPO agent for standalone drone navigation")
    
    # Training arguments
    parser.add_argument("--timesteps", type=int, default=500_000,
                       help="Total training timesteps (default: 500,000)")
    parser.add_argument("--lr", type=float, default=3e-4,
                       help="Learning rate (default: 3e-4)")
    parser.add_argument("--batch-size", type=int, default=2048,
                       help="Batch size (default: 2048)")
    parser.add_argument("--num-envs", type=int, default=64,
                       help="Number of parallel environments (default: 64)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed (default: 42)")
    parser.add_argument("--log-dir", type=str, default="./logs_standalone",
                       help="Log directory (default: ./logs_standalone)")
    parser.add_argument("--model-dir", type=str, default="./models_standalone",
                       help="Model save directory (default: ./models_standalone)")
    parser.add_argument("--save-freq", type=int, default=100_000,
                       help="Model checkpoint frequency (default: 100,000)")
    
    # Evaluation arguments
    parser.add_argument("--eval", type=str, default=None,
                       help="Path to model for evaluation only")
    parser.add_argument("--eval-episodes", type=int, default=10,
                       help="Number of evaluation episodes (default: 10)")
    
    args = parser.parse_args()
    
    if args.eval:
        # Evaluation mode
        evaluate_standalone_model(args.eval, args.eval_episodes, args.num_envs)
    else:
        # Training mode
        train_standalone_drone(
            total_timesteps=args.timesteps,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            num_envs=args.num_envs,
            log_dir=args.log_dir,
            model_dir=args.model_dir,
            seed=args.seed,
            save_freq=args.save_freq
        )


if __name__ == "__main__":
    main()
