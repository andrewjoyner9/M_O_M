from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from drone_env_diverse import DiverseDroneEnv
import numpy as np
import os
import matplotlib.pyplot as plt
from collections import deque

class PathDiversityCallback(BaseCallback):
    """
    Custom callback to track and encourage path diversity during training
    """
    def __init__(self, eval_env, verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.successful_paths = []
        self.path_diversity_scores = []
        self.episode_rewards = deque(maxlen=100)
        
    def _on_step(self) -> bool:
        # Track episode rewards
        if len(self.locals.get('rewards', [])) > 0:
            self.episode_rewards.extend(self.locals['rewards'])
            
        # Every 1000 steps, evaluate path diversity
        if self.n_calls % 1000 == 0:
            self._evaluate_path_diversity()
            
        return True
    
    def _evaluate_path_diversity(self):
        """Evaluate current model's path diversity"""
        if self.verbose > 0:
            print(f"\nEvaluating path diversity at step {self.n_calls}")
        
        paths = []
        rewards = []
        
        # Run multiple episodes to collect paths
        for i in range(5):
            obs, _ = self.eval_env.reset()
            episode_reward = 0
            trajectory = []
            
            for step in range(1500):
                action, _ = self.model.predict(obs, deterministic=False)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                episode_reward += reward
                
                # Get current position from observation
                drone_pos = obs[:3]
                trajectory.append(drone_pos.copy())
                
                if terminated or truncated:
                    break
            
            paths.append(trajectory)
            rewards.append(episode_reward)
            
            if self.verbose > 0:
                goal_distance = np.linalg.norm(obs[:3] - obs[3:6])
                print(f"  Episode {i+1}: Reward={episode_reward:.1f}, Final distance={goal_distance:.2f}")
        
        # Calculate path diversity score
        diversity_score = self._calculate_paths_diversity(paths)
        self.path_diversity_scores.append(diversity_score)
        
        if self.verbose > 0:
            avg_reward = np.mean(rewards)
            print(f"  Average reward: {avg_reward:.1f}")
            print(f"  Path diversity score: {diversity_score:.3f}")
            print(f"  Recent average reward: {np.mean(list(self.episode_rewards)) if self.episode_rewards else 0:.1f}")
    
    def _calculate_paths_diversity(self, paths):
        """Calculate diversity score between multiple paths"""
        if len(paths) < 2:
            return 0.0
        
        total_diversity = 0.0
        comparisons = 0
        
        for i in range(len(paths)):
            for j in range(i + 1, len(paths)):
                path1, path2 = paths[i], paths[j]
                if len(path1) > 5 and len(path2) > 5:
                    # Sample points from both paths
                    samples1 = np.array(path1)[::max(1, len(path1)//20)]
                    samples2 = np.array(path2)[::max(1, len(path2)//20)]
                    
                    # Calculate average minimum distance between paths
                    total_distance = 0
                    for p1 in samples1:
                        min_dist = min([np.linalg.norm(p1 - p2) for p2 in samples2])
                        total_distance += min_dist
                    
                    if len(samples1) > 0:
                        avg_diversity = total_distance / len(samples1)
                        total_diversity += avg_diversity
                        comparisons += 1
        
        return total_diversity / comparisons if comparisons > 0 else 0.0

def create_env():
    """Create a single environment instance"""
    env = DiverseDroneEnv(encourage_exploration=True, num_threats=3)
    return Monitor(env)

def train_diverse_drone_agent():
    """Train agent with emphasis on path diversity and exploration"""
    print("Creating diverse drone training environment...")
    
    # Create vectorized training environment for parallel training
    print("Setting up parallel training environments...")
    train_env = DummyVecEnv([create_env for _ in range(4)])  # 4 parallel environments
    
    # Create evaluation environment
    eval_env = DiverseDroneEnv(encourage_exploration=True, num_threats=3)
    eval_env = Monitor(eval_env)
    
    # Set up callbacks
    path_diversity_callback = PathDiversityCallback(eval_env, verbose=1)
    
    # Stop training when model reaches good performance but ensure diversity
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=120, verbose=1)
    eval_callback = EvalCallback(
        eval_env, 
        callback_on_new_best=callback_on_best, 
        eval_freq=2000,
        n_eval_episodes=5,
        verbose=1
    )
    
    # Create PPO model with parameters encouraging exploration
    print("Creating PPO model with exploration-friendly parameters...")
    model = PPO(
        "MlpPolicy", 
        train_env, 
        learning_rate=3e-4,
        n_steps=2048,           # Steps per environment per update
        batch_size=64,
        n_epochs=10,
        gamma=0.99,             # Discount factor
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,          # Increased entropy for more exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log="./diverse_drone_tensorboard/",
        policy_kwargs=dict(
            net_arch=[dict(pi=[256, 256], vf=[256, 256])]  # Larger networks
        )
    )
    
    # Train the model
    print("Starting diverse path training...")
    print("The agent will learn to:")
    print("- Explore different routes to the goal")
    print("- Avoid multiple moving threats")
    print("- Find optimal paths through trial and error")
    print("- Receive bonuses for path diversity\n")
    
    model.learn(
        total_timesteps=200_000,  # Increased training time
        callback=[eval_callback, path_diversity_callback],
        progress_bar=True
    )
    
    # Save the model
    model.save("diverse_drone_rl_model")
    print("Model saved as diverse_drone_rl_model.zip")
    
    return model, path_diversity_callback

def test_path_diversity(model_path="diverse_drone_rl_model", num_tests=10):
    """Test the trained model and visualize path diversity"""
    print(f"\nTesting path diversity with {num_tests} episodes...")
    
    env = DiverseDroneEnv(encourage_exploration=True, num_threats=3)
    model = PPO.load(model_path)
    
    all_paths = []
    all_rewards = []
    success_count = 0
    
    for episode in range(num_tests):
        print(f"\nEpisode {episode + 1}:")
        obs, info = env.reset()
        total_reward = 0
        trajectory = []
        
        print(f"  Start: ({obs[0]:.2f}, {obs[1]:.2f}, {obs[2]:.2f})")
        print(f"  Goal:  ({obs[3]:.2f}, {obs[4]:.2f}, {obs[5]:.2f})")
        
        for step in range(1500):
            action, _states = model.predict(obs, deterministic=False)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            trajectory.append(obs[:3].copy())
            
            if terminated or truncated:
                distance_to_goal = np.linalg.norm(obs[:3] - obs[3:6])
                print(f"  Finished at step {step + 1}")
                print(f"  Final distance to goal: {distance_to_goal:.2f}")
                print(f"  Total reward: {total_reward:.1f}")
                print(f"  Exploration coverage: {info.get('exploration_coverage', 0) * 100:.1f}%")
                
                if distance_to_goal < 0.5:
                    print("  ✅ SUCCESS - Goal reached!")
                    success_count += 1
                else:
                    print("  ❌ Failed to reach goal")
                break
        
        all_paths.append(trajectory)
        all_rewards.append(total_reward)
    
    # Calculate overall statistics
    print(f"\n=== RESULTS SUMMARY ===")
    print(f"Success rate: {success_count}/{num_tests} ({success_count/num_tests*100:.1f}%)")
    print(f"Average reward: {np.mean(all_rewards):.1f} ± {np.std(all_rewards):.1f}")
    
    # Calculate path diversity
    if len(all_paths) > 1:
        successful_paths = [path for i, path in enumerate(all_paths) 
                          if len(path) > 0 and np.linalg.norm(np.array(path[-1]) - np.array([5, 5, 1])) < 1.0]
        
        if len(successful_paths) > 1:
            diversity_score = calculate_overall_path_diversity(successful_paths)
            print(f"Path diversity score: {diversity_score:.3f}")
            print(f"Successful paths found: {len(successful_paths)}")
            
            # Create simple visualization
            try:
                visualize_paths(successful_paths)
            except ImportError:
                print("Matplotlib not available for visualization")
        else:
            print("Not enough successful paths for diversity analysis")
    
    return all_paths, all_rewards

def calculate_overall_path_diversity(paths):
    """Calculate overall diversity between successful paths"""
    if len(paths) < 2:
        return 0.0
    
    total_diversity = 0.0
    comparisons = 0
    
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            path1, path2 = paths[i], paths[j]
            if len(path1) > 10 and len(path2) > 10:
                # Sample points from both paths
                samples1 = np.array(path1)[::max(1, len(path1)//30)]
                samples2 = np.array(path2)[::max(1, len(path2)//30)]
                
                # Calculate average minimum distance between paths
                distances = []
                for p1 in samples1:
                    min_dist = min([np.linalg.norm(p1 - p2) for p2 in samples2])
                    distances.append(min_dist)
                
                if distances:
                    avg_diversity = np.mean(distances)
                    total_diversity += avg_diversity
                    comparisons += 1
    
    return total_diversity / comparisons if comparisons > 0 else 0.0

def visualize_paths(paths, max_paths=5):
    """Create a simple 2D visualization of different paths"""
    plt.figure(figsize=(10, 8))
    
    # Plot up to max_paths different successful paths
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, path in enumerate(paths[:max_paths]):
        if len(path) > 0:
            path_array = np.array(path)
            plt.plot(path_array[:, 0], path_array[:, 1], 
                    color=colors[i % len(colors)], 
                    alpha=0.7, linewidth=2, 
                    label=f'Path {i+1}')
            
            # Mark start and end
            plt.plot(path_array[0, 0], path_array[0, 1], 'go', markersize=8, label='Start' if i == 0 else "")
            plt.plot(path_array[-1, 0], path_array[-1, 1], 'ro', markersize=8, label='End' if i == 0 else "")
    
    # Add typical threat zone (approximate)
    circle = plt.Circle((2.5, 2.5), 1.0, color='red', alpha=0.3, label='Threat Zone')
    plt.gca().add_patch(circle)
    
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Diverse Drone Paths to Goal')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # Save the plot
    plt.savefig('diverse_drone_paths.png', dpi=150, bbox_inches='tight')
    print("Path visualization saved as 'diverse_drone_paths.png'")
    plt.show()

if __name__ == "__main__":
    print("🚁 Enhanced Drone RL Training with Path Diversity")
    print("=" * 50)
    
    # Train the model
    model, diversity_callback = train_diverse_drone_agent()
    
    # Test path diversity
    print("\n" + "=" * 50)
    print("Testing trained model for path diversity...")
    paths, rewards = test_path_diversity("diverse_drone_rl_model", num_tests=8)
    
    print("\n🎯 Training completed!")
    print("The agent should now:")
    print("- Take different paths in different episodes")
    print("- Explore alternative routes when threats block direct paths")
    print("- Balance efficiency with safety")
    print("- Show creative problem-solving in navigation")
