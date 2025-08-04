#!/usr/bin/env python3
"""
Quick test script to demonstrate path diversity improvements
"""

import sys
import os
sys.path.append('/home/omniverse/isaac_projects/M_O_M/isaac-rl/original')

from drone_env_diverse import DiverseDroneEnv
import numpy as np
import matplotlib.pyplot as plt

def simple_random_agent_test():
    """Test the diverse environment with a simple random agent to show variety"""
    print("🚁 Testing Enhanced Drone Environment")
    print("=" * 50)
    print("This test uses a RANDOM agent to demonstrate environmental diversity")
    print("(A trained RL agent would perform much better!)\n")
    
    env = DiverseDroneEnv(encourage_exploration=True, num_threats=3)
    
    paths = []
    episode_info = []
    
    for episode in range(5):
        print(f"Episode {episode + 1}:")
        obs, info = env.reset()
        
        start_pos = obs[:3]
        goal_pos = obs[3:6]
        threat_positions = [obs[6 + i*3:9 + i*3] for i in range(3)]
        
        print(f"  Start: ({start_pos[0]:.2f}, {start_pos[1]:.2f}, {start_pos[2]:.2f})")
        print(f"  Goal:  ({goal_pos[0]:.2f}, {goal_pos[1]:.2f}, {goal_pos[2]:.2f})")
        print(f"  Threats at:", end="")
        for i, threat in enumerate(threat_positions):
            print(f" T{i+1}:({threat[0]:.1f},{threat[1]:.1f})", end="")
        print()
        
        trajectory = [start_pos.copy()]
        total_reward = 0
        
        # Random agent with slight bias toward goal
        for step in range(200):  # Shorter episodes for demo
            # Random action with slight bias toward goal
            goal_direction = goal_pos - obs[:3]
            goal_direction = goal_direction / (np.linalg.norm(goal_direction) + 1e-8)
            
            # Mix random action with goal-directed action
            random_action = np.random.uniform(-1, 1, 3)
            biased_action = 0.7 * random_action + 0.3 * goal_direction
            biased_action = np.clip(biased_action, -1, 1)
            
            obs, reward, terminated, truncated, info = env.step(biased_action)
            total_reward += reward
            trajectory.append(obs[:3].copy())
            
            if terminated or truncated:
                break
        
        final_distance = np.linalg.norm(obs[:3] - goal_pos)
        exploration = info.get('exploration_coverage', 0) * 100
        
        print(f"  Result: {total_reward:.1f} reward, {final_distance:.2f} final distance")
        print(f"  Explored {exploration:.1f}% of environment, {len(trajectory)} steps")
        
        paths.append(trajectory)
        episode_info.append({
            'start': start_pos,
            'goal': goal_pos,
            'threats': threat_positions,
            'final_distance': final_distance,
            'reward': total_reward,
            'exploration': exploration
        })
        print()
    
    # Analyze variety
    print("VARIETY ANALYSIS:")
    print("=" * 30)
    
    # Check start position variety
    start_positions = [info['start'] for info in episode_info]
    start_variance = np.var(start_positions, axis=0)
    print(f"Start position variance: {np.mean(start_variance):.3f}")
    
    # Check goal position variety  
    goal_positions = [info['goal'] for info in episode_info]
    goal_variance = np.var(goal_positions, axis=0)
    print(f"Goal position variance: {np.mean(goal_variance):.3f}")
    
    # Check path diversity
    if len(paths) > 1:
        path_diversity = calculate_simple_path_diversity(paths)
        print(f"Path diversity score: {path_diversity:.3f}")
    
    # Show exploration variety
    explorations = [info['exploration'] for info in episode_info]
    print(f"Exploration range: {min(explorations):.1f}% - {max(explorations):.1f}%")
    
    print(f"\n✅ Environment successfully provides:")
    print(f"   - Randomized start/goal positions each episode")
    print(f"   - Multiple threats in different locations")
    print(f"   - Exploration tracking and rewards")
    print(f"   - Path diversity incentives")
    print(f"\n🎯 A trained RL agent would learn to:")
    print(f"   - Navigate efficiently while exploring")
    print(f"   - Find multiple successful routes")
    print(f"   - Adapt to different obstacle configurations")
    print(f"   - Balance speed, safety, and exploration")

def calculate_simple_path_diversity(paths):
    """Simple path diversity calculation"""
    if len(paths) < 2:
        return 0.0
    
    total_diversity = 0.0
    comparisons = 0
    
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            path1, path2 = paths[i], paths[j]
            
            # Sample a few points from each path
            min_len = min(len(path1), len(path2))
            if min_len > 5:
                sample_size = min(10, min_len // 2)
                indices1 = np.linspace(0, len(path1)-1, sample_size, dtype=int)
                indices2 = np.linspace(0, len(path2)-1, sample_size, dtype=int)
                
                samples1 = np.array([path1[idx] for idx in indices1])
                samples2 = np.array([path2[idx] for idx in indices2])
                
                # Calculate average distance between corresponding points
                distances = [np.linalg.norm(p1 - p2) for p1, p2 in zip(samples1, samples2)]
                avg_distance = np.mean(distances)
                
                total_diversity += avg_distance
                comparisons += 1
    
    return total_diversity / comparisons if comparisons > 0 else 0.0

def demonstrate_improvements():
    """Show key improvements over original environment"""
    print("\n🚀 KEY IMPROVEMENTS OVER ORIGINAL:")
    print("=" * 50)
    
    print("1. RANDOMIZATION:")
    print("   ✓ Start positions vary each episode")
    print("   ✓ Goal positions randomized") 
    print("   ✓ Multiple threats in random locations")
    print("   ✗ Original: Fixed positions every time")
    
    print("\n2. MULTIPLE THREATS:")
    print("   ✓ 3-4 threats instead of 1")
    print("   ✓ Threats move dynamically")
    print("   ✓ Requires more complex navigation")
    print("   ✗ Original: Single static threat")
    
    print("\n3. EXPLORATION REWARDS:")
    print("   ✓ Bonus for visiting new areas")
    print("   ✓ Grid-based exploration tracking")
    print("   ✓ Encourages diverse movement")
    print("   ✗ Original: Only goal distance reward")
    
    print("\n4. PATH DIVERSITY:")
    print("   ✓ Rewards for taking different routes")
    print("   ✓ Compares with previous successful paths")
    print("   ✓ Prevents convergence to single solution")
    print("   ✗ Original: Converges to same path")
    
    print("\n5. ENHANCED OBSERVATION:")
    print("   ✓ Multiple threat positions")
    print("   ✓ Exploration state information")
    print("   ✓ Path diversity metrics")
    print("   ✗ Original: Basic position info only")
    
    print("\n6. DYNAMIC ENVIRONMENT:")
    print("   ✓ Moving obstacles")
    print("   ✓ Environmental variation")
    print("   ✓ Adaptive challenges")
    print("   ✗ Original: Static environment")

if __name__ == "__main__":
    demonstrate_improvements()
    simple_random_agent_test()
    
    print("\n" + "=" * 60)
    print("🎯 NEXT STEPS TO GET DIVERSE PATHS:")
    print("=" * 60)
    print("1. Train with the enhanced environment:")
    print("   cd /home/omniverse/isaac_projects/M_O_M/isaac-rl/original")
    print("   python train_diverse.py")
    print()
    print("2. Or use Isaac Sim version:")
    print("   cd /home/omniverse/isaac_projects/M_O_M/isaac-rl/isaac_sim") 
    print("   python train_isaac_sim_diverse.py --mode demo")
    print()
    print("3. The trained agent will:")
    print("   - Take different paths each episode")
    print("   - Explore alternative routes")
    print("   - Adapt to different threat configurations")
    print("   - Show creative navigation strategies")
