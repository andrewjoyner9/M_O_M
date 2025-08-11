#!/usr/bin/env python3
"""
Generate Longer Navigation Paths

Creates training scenarios that force the drone to actually navigate
longer distances, generating better visualization data.
"""

import numpy as np
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
from simple_drone_env import SimpleDroneEnv

def create_challenging_episode(seed=None, min_distance=10.0):
    """Create a single challenging navigation episode"""
    
    if seed is not None:
        np.random.seed(seed)
    
    # Configuration for longer paths
    config = {
        'arena_size': 20.0,           # Larger arena
        'arena_height': 8.0,
        'num_obstacles': 8,           # More obstacles
        'obstacle_radius_range': [1.0, 2.0],  # Bigger obstacles
        'goal_threshold': 1.5,        # Must get close to goal
        'obstacle_clearance': 3.0,    # Safe clearance
        'min_start_goal_distance': min_distance,  # Force long distance
        'max_steps': 1000,            # Allow more time
        'difficulty_level': 'hard'
    }
    
    # Create environment
    env = SimpleDroneEnv(config=config)
    
    # Reset and verify distance
    max_attempts = 20
    for attempt in range(max_attempts):
        obs = env.reset()
        actual_distance = np.linalg.norm(env.goal_position - env.drone_position)
        
        if actual_distance >= min_distance:
            print(f"✓ Created scenario with distance {actual_distance:.2f}m (attempt {attempt + 1})")
            break
    else:
        print(f"⚠️ Using best distance found: {actual_distance:.2f}m")
    
    return env

def run_smart_navigation(env, strategy='direct_with_avoidance'):
    """Run a smart navigation algorithm that creates realistic paths"""
    
    path = [env.drone_position.copy()]
    
    for step in range(env.config['max_steps']):
        current_pos = env.drone_position
        goal_pos = env.goal_position
        
        # Calculate direction to goal
        direction_to_goal = goal_pos - current_pos
        distance_to_goal = np.linalg.norm(direction_to_goal)
        
        if distance_to_goal < env.config['goal_threshold']:
            # Success!
            break
        
        # Normalize direction
        if distance_to_goal > 0:
            direction_to_goal /= distance_to_goal
        
        # Check for obstacles and avoid them
        avoidance_force = np.zeros(3)
        
        for obs_pos, obs_radius in zip(env.obstacle_positions, env.obstacle_radii):
            obs_vector = current_pos - obs_pos
            obs_distance = np.linalg.norm(obs_vector)
            
            # If too close to obstacle, add avoidance force
            safe_distance = obs_radius + 2.0  # 2m safety margin
            if obs_distance < safe_distance and obs_distance > 0:
                # Avoidance force inversely proportional to distance
                avoidance_strength = (safe_distance - obs_distance) / safe_distance
                avoidance_force += (obs_vector / obs_distance) * avoidance_strength * 2.0
        
        # Combine goal-seeking and avoidance
        if strategy == 'direct_with_avoidance':
            action = direction_to_goal * 0.8 + avoidance_force
        elif strategy == 'cautious':
            action = direction_to_goal * 0.5 + avoidance_force * 1.5
        elif strategy == 'aggressive':
            action = direction_to_goal * 1.2 + avoidance_force * 0.3
        else:  # 'random_walk'
            action = direction_to_goal * 0.3 + np.random.normal(0, 0.5, 3) + avoidance_force
        
        # Add some exploration noise
        action += np.random.normal(0, 0.1, 3)
        
        # Clip action to valid range
        action = np.clip(action, -1.0, 1.0)
        
        # Take step
        obs, reward, done, info = env.step(action)
        path.append(env.drone_position.copy())
        
        if done:
            break
    
    # Check if successful
    final_distance = np.linalg.norm(env.drone_position - env.goal_position)
    success = final_distance <= env.config['goal_threshold']
    
    return {
        'path': path,
        'start': path[0] if path else None,
        'goal': env.goal_position.copy(),
        'obstacles': [(pos.copy(), radius) for pos, radius in zip(env.obstacle_positions, env.obstacle_radii)],
        'success': success,
        'steps': len(path),
        'final_distance': final_distance,
        'strategy': strategy
    }

def generate_diverse_navigation_episodes(num_episodes=10):
    """Generate multiple episodes with different strategies"""
    
    print(f"🎯 Generating {num_episodes} Diverse Navigation Episodes")
    print("=" * 60)
    
    episodes = []
    strategies = ['direct_with_avoidance', 'cautious', 'aggressive', 'random_walk']
    
    for i in range(num_episodes):
        print(f"\n📍 Episode {i + 1}/{num_episodes}")
        
        # Create challenging scenario
        env = create_challenging_episode(seed=i * 42, min_distance=12.0)
        
        # Choose strategy
        strategy = strategies[i % len(strategies)]
        print(f"🤖 Using strategy: {strategy}")
        
        # Run navigation
        episode_data = run_smart_navigation(env, strategy)
        
        # Report results
        status = "✅ SUCCESS" if episode_data['success'] else "❌ FAILED"
        print(f"   {status}")
        print(f"   Steps: {episode_data['steps']}")
        print(f"   Final distance: {episode_data['final_distance']:.2f}m")
        print(f"   Start-goal distance: {np.linalg.norm(episode_data['goal'] - episode_data['start']):.2f}m")
        
        episodes.append(episode_data)
    
    return episodes

def save_and_visualize_episodes(episodes, save_dir="./logs_long_navigation"):
    """Save episodes and create a sample visualization"""
    
    # Save episodes
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    save_file = save_dir / "episode_paths.pkl"
    with open(save_file, 'wb') as f:
        pickle.dump(episodes, f)
    
    print(f"\n💾 Saved {len(episodes)} episodes to {save_file}")
    
    # Quick statistics
    successful_episodes = [ep for ep in episodes if ep['success']]
    success_rate = len(successful_episodes) / len(episodes) * 100
    
    avg_steps = np.mean([ep['steps'] for ep in episodes])
    avg_distance = np.mean([np.linalg.norm(np.array(ep['goal']) - np.array(ep['start'])) for ep in episodes])
    
    print(f"\n📊 Generation Summary:")
    print(f"   Success rate: {success_rate:.1f}% ({len(successful_episodes)}/{len(episodes)})")
    print(f"   Average steps: {avg_steps:.1f}")
    print(f"   Average start-goal distance: {avg_distance:.1f}m")
    
    # Create a quick preview visualization
    if episodes:
        best_episode = max(episodes, key=lambda x: x['steps'] if x['success'] else 0)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot best episode
        if best_episode['path']:
            path = np.array(best_episode['path'])
            
            # 3D view
            ax1 = fig.add_subplot(121, projection='3d')
            ax1.plot(path[:, 0], path[:, 1], path[:, 2], 'g-', linewidth=3, label='Navigation Path')
            ax1.scatter(*best_episode['start'], color='blue', s=200, marker='o', label='Start')
            ax1.scatter(*best_episode['goal'], color='gold', s=300, marker='*', label='Goal')
            
            # Plot obstacles
            for obs_pos, obs_radius in best_episode['obstacles']:
                u = np.linspace(0, 2 * np.pi, 10)
                v = np.linspace(0, np.pi, 10)
                x = obs_radius * np.outer(np.cos(u), np.sin(v)) + obs_pos[0]
                y = obs_radius * np.outer(np.sin(u), np.sin(v)) + obs_pos[1]
                z = obs_radius * np.outer(np.ones(np.size(u)), np.cos(v)) + obs_pos[2]
                ax1.plot_surface(x, y, z, alpha=0.3, color='red')
            
            ax1.set_title(f'Best Episode - {best_episode["steps"]} steps')
            ax1.legend()
            
            # Top-down view
            ax2.plot(path[:, 0], path[:, 1], 'g-', linewidth=3, label='Navigation Path')
            ax2.scatter(best_episode['start'][0], best_episode['start'][1], color='blue', s=200, marker='o', label='Start')
            ax2.scatter(best_episode['goal'][0], best_episode['goal'][1], color='gold', s=300, marker='*', label='Goal')
            
            for obs_pos, obs_radius in best_episode['obstacles']:
                circle = plt.Circle((obs_pos[0], obs_pos[1]), obs_radius, color='red', alpha=0.3)
                ax2.add_patch(circle)
            
            ax2.set_title('Top-Down View')
            ax2.set_aspect('equal')
            ax2.grid(True)
            ax2.legend()
        
        plt.tight_layout()
        preview_file = save_dir / "navigation_preview.png"
        plt.savefig(preview_file, dpi=150, bbox_inches='tight')
        print(f"📊 Preview saved as {preview_file}")
        
        try:
            plt.show()
        except:
            print("⚠️ Display not available, preview saved to file")

def main():
    """Main function"""
    
    print("🚁 Long Navigation Path Generator")
    print("=" * 50)
    
    # Generate diverse episodes
    episodes = generate_diverse_navigation_episodes(num_episodes=15)
    
    # Save and visualize
    save_and_visualize_episodes(episodes)
    
    print(f"\n🎨 Visualization Instructions:")
    print("=" * 40)
    print("1. View single clean episode:")
    print("   python3 single_episode_visualizer.py")
    print()
    print("2. Compare algorithms:")
    print("   python3 compare_algorithms_visualization.py")
    print()
    print("3. View original multi-episode visualization:")
    print("   python3 visualize_paths.py")
    print()
    print("4. Data saved in: ./logs_long_navigation/episode_paths.pkl")

if __name__ == "__main__":
    main()
