from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from isaac_sim_drone_env_diverse import DiverseIsaacSimDroneEnv
import numpy as np
import argparse
import os

def create_diverse_isaac_env(use_isaac_sim=True):
    """Create diverse Isaac Sim environment"""
    env = DiverseIsaacSimDroneEnv(
        use_isaac_sim=use_isaac_sim, 
        encourage_exploration=True,
        num_threats=4
    )
    return Monitor(env)

def train_diverse_isaac_agent(use_isaac_sim=True, timesteps=150000):
    """Train agent with diverse Isaac Sim environment"""
    print(f"Training diverse drone agent...")
    print(f"Isaac Sim: {'Enabled' if use_isaac_sim else 'Disabled (fallback mode)'}")
    print(f"Target timesteps: {timesteps:,}")
    
    # Create training environment
    train_env = DummyVecEnv([lambda: create_diverse_isaac_env(use_isaac_sim) for _ in range(2)])
    
    # Create evaluation environment
    eval_env = create_diverse_isaac_env(use_isaac_sim)
    
    # Setup callbacks
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=140, verbose=1)
    eval_callback = EvalCallback(
        eval_env,
        callback_on_new_best=callback_on_best,
        eval_freq=3000,
        n_eval_episodes=3,
        verbose=1
    )
    
    # Create enhanced PPO model
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=2e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,  # Higher entropy for exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log="./isaac_diverse_tensorboard/",
        policy_kwargs=dict(
            net_arch=[dict(pi=[256, 256, 128], vf=[256, 256, 128])]
        )
    )
    
    print("Starting diverse path training...")
    print("Enhanced features:")
    print("- Randomized start/goal positions each episode")
    print("- Multiple moving threats")
    print("- Path diversity rewards")
    print("- Exploration bonuses")
    print("- Environmental randomization")
    
    # Train the model
    model.learn(
        total_timesteps=timesteps,
        callback=[eval_callback],
        progress_bar=True
    )
    
    # Save the model
    model_name = "isaac_sim_diverse_drone_model" if use_isaac_sim else "diverse_fallback_drone_model"
    model.save(model_name)
    print(f"Model saved as {model_name}.zip")
    
    return model

def test_diverse_paths(model_name, use_isaac_sim=True, num_episodes=8):
    """Test trained model for path diversity"""
    print(f"\nTesting path diversity with {num_episodes} episodes...")
    
    env = create_diverse_isaac_env(use_isaac_sim)
    
    try:
        model = PPO.load(model_name)
    except FileNotFoundError:
        print(f"Model {model_name} not found. Please train first.")
        return
    
    successful_paths = []
    all_rewards = []
    episode_stats = []
    
    for episode in range(num_episodes):
        print(f"\n--- Episode {episode + 1} ---")
        obs, info = env.reset()
        total_reward = 0
        steps = 0
        trajectory = []
        
        print(f"Start position: ({obs[0]:.2f}, {obs[1]:.2f}, {obs[2]:.2f})")
        print(f"Goal position:  ({obs[3]:.2f}, {obs[4]:.2f}, {obs[5]:.2f})")
        print(f"Threats: {info.get('num_threats', 'Unknown')}")
        
        for step in range(2000):
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            
            # Store position for trajectory analysis
            trajectory.append(obs[:3].copy())
            
            if terminated or truncated:
                break
        
        # Calculate final statistics
        final_distance = np.linalg.norm(obs[:3] - obs[3:6])
        exploration_areas = info.get('exploration_areas', 0)
        success = final_distance < 0.5
        
        episode_stats.append({
            'success': success,
            'reward': total_reward,
            'steps': steps,
            'final_distance': final_distance,
            'exploration_areas': exploration_areas,
            'trajectory': trajectory
        })
        
        print(f"Result: {'SUCCESS' if success else 'FAILED'}")
        print(f"Final distance to goal: {final_distance:.3f}")
        print(f"Total reward: {total_reward:.1f}")
        print(f"Steps taken: {steps}")
        print(f"Areas explored: {exploration_areas}")
        
        if success:
            successful_paths.append(trajectory)
        
        all_rewards.append(total_reward)
    
    # Overall statistics
    print(f"\n{'='*50}")
    print("OVERALL RESULTS")
    print(f"{'='*50}")
    
    success_count = sum(1 for stat in episode_stats if stat['success'])
    print(f"Success rate: {success_count}/{num_episodes} ({success_count/num_episodes*100:.1f}%)")
    print(f"Average reward: {np.mean(all_rewards):.1f} ± {np.std(all_rewards):.1f}")
    print(f"Average steps (successful): {np.mean([s['steps'] for s in episode_stats if s['success']]):.1f}" if success_count > 0 else "No successful episodes")
    print(f"Average exploration (successful): {np.mean([s['exploration_areas'] for s in episode_stats if s['success']]):.1f}" if success_count > 0 else "No successful episodes")
    
    # Path diversity analysis
    if len(successful_paths) > 1:
        diversity_score = calculate_path_diversity(successful_paths)
        print(f"Path diversity score: {diversity_score:.3f}")
        print(f"Unique successful paths: {len(successful_paths)}")
        
        # Analyze path characteristics
        path_lengths = [len(path) for path in successful_paths]
        print(f"Path length variation: {np.std(path_lengths):.1f} steps")
        
        # Check if paths use different strategies
        strategies = analyze_path_strategies(successful_paths)
        print(f"Different strategies observed: {len(strategies)}")
        for i, strategy in enumerate(strategies):
            print(f"  Strategy {i+1}: {strategy}")
    
    elif len(successful_paths) == 1:
        print("Only one successful path found - need more successes to analyze diversity")
    else:
        print("No successful paths found - model needs more training")
    
    return episode_stats

def calculate_path_diversity(paths):
    """Calculate overall diversity between successful paths"""
    if len(paths) < 2:
        return 0.0
    
    total_diversity = 0.0
    comparisons = 0
    
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            path1, path2 = paths[i], paths[j]
            
            if len(path1) > 10 and len(path2) > 10:
                # Sample points from both paths for comparison
                sample_rate1 = max(1, len(path1) // 25)
                sample_rate2 = max(1, len(path2) // 25)
                
                samples1 = np.array(path1)[::sample_rate1]
                samples2 = np.array(path2)[::sample_rate2]
                
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

def analyze_path_strategies(paths):
    """Analyze different strategies used in successful paths"""
    strategies = []
    
    for i, path in enumerate(paths):
        if len(path) < 20:
            continue
            
        path_array = np.array(path)
        
        # Analyze path characteristics
        strategy_desc = f"Path {i+1}: "
        
        # Check if path goes around obstacles (indirect)
        start_to_goal = path_array[-1] - path_array[0]
        direct_distance = np.linalg.norm(start_to_goal)
        actual_distance = sum([np.linalg.norm(path_array[j+1] - path_array[j]) 
                              for j in range(len(path_array)-1)])
        
        efficiency = direct_distance / actual_distance if actual_distance > 0 else 0
        
        if efficiency > 0.8:
            strategy_desc += "Direct route"
        elif efficiency > 0.6:
            strategy_desc += "Moderately indirect"
        else:
            strategy_desc += "Highly indirect/exploratory"
        
        # Check movement patterns
        movements = [path_array[j+1] - path_array[j] for j in range(len(path_array)-1)]
        movement_variance = np.var(movements, axis=0)
        
        if np.mean(movement_variance) > 0.01:
            strategy_desc += ", varied movement"
        else:
            strategy_desc += ", steady movement"
        
        # Check altitude strategy
        altitudes = path_array[:, 2]
        if np.std(altitudes) > 0.2:
            strategy_desc += ", altitude changes"
        else:
            strategy_desc += ", constant altitude"
        
        strategies.append(strategy_desc)
    
    return strategies

def main():
    parser = argparse.ArgumentParser(description='Train and test diverse drone RL agent')
    parser.add_argument('--mode', choices=['train', 'test', 'demo'], default='demo',
                       help='Mode: train new model, test existing, or demo')
    parser.add_argument('--timesteps', type=int, default=150000,
                       help='Training timesteps')
    parser.add_argument('--episodes', type=int, default=8,
                       help='Test episodes')
    parser.add_argument('--no-isaac', action='store_true',
                       help='Use fallback simulation instead of Isaac Sim')
    parser.add_argument('--model', type=str, default=None,
                       help='Model name for testing')
    
    args = parser.parse_args()
    
    use_isaac_sim = not args.no_isaac
    
    if args.mode == 'train':
        print("🚁 Training Enhanced Diverse Drone RL Agent")
        print("=" * 50)
        model = train_diverse_isaac_agent(use_isaac_sim, args.timesteps)
        
    elif args.mode == 'test':
        model_name = args.model or ("isaac_sim_diverse_drone_model" if use_isaac_sim else "diverse_fallback_drone_model")
        print(f"🎯 Testing Path Diversity: {model_name}")
        print("=" * 50)
        test_diverse_paths(model_name, use_isaac_sim, args.episodes)
        
    elif args.mode == 'demo':
        print("🚁 Diverse Drone RL Demo")
        print("=" * 50)
        print("Training diverse agent...")
        model = train_diverse_isaac_agent(use_isaac_sim, args.timesteps)
        
        print("\nTesting path diversity...")
        model_name = "isaac_sim_diverse_drone_model" if use_isaac_sim else "diverse_fallback_drone_model"
        test_diverse_paths(model_name, use_isaac_sim, args.episodes)
        
        print("\n🎉 Demo completed!")
        print("The agent should now:")
        print("- Take different paths in different episodes")
        print("- Explore alternative routes around threats") 
        print("- Show creative navigation strategies")
        print("- Balance efficiency with safety and exploration")

if __name__ == "__main__":
    main()
