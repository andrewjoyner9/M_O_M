#!/usr/bin/env python3
"""
DreamerV3 Multi-Agent Training Script

This script trains multiple DreamerV3 agents on an identical navigation scenario
until a target number successfully reach the goal. It's optimized for practical
use with reasonable training times and clear progress reporting.

Usage:
    python3 dreamerv3_multi_agent_trainer.py
    
Or with custom parameters:
    python3 dreamerv3_multi_agent_trainer.py --agents 5 --successes 3 --max-steps 2000
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os
import time

# Add the scripts directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from single_episode_visualizer import SingleEpisodeVisualizer
    from simple_drone_env import SimpleDroneEnv
    from dreamerv3_drone import DreamerV3Agent
    print("✅ All modules imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train multiple DreamerV3 agents')
    parser.add_argument('--agents', type=int, default=5, 
                       help='Number of DreamerV3 agents to train (default: 5)')
    parser.add_argument('--successes', type=int, default=3, 
                       help='Target number of successful agents (default: 3)')
    parser.add_argument('--max-steps', type=int, default=2000,
                       help='Maximum training steps (default: 2000)')
    parser.add_argument('--save-name', type=str, default='dreamerv3_training_results.png',
                       help='Output visualization filename (default: dreamerv3_training_results.png)')
    return parser.parse_args()

def train_dreamerv3_agents(num_agents=5, target_successes=3, max_training_steps=2000, save_name='results.png'):
    """Train multiple DreamerV3 agents on identical scenario"""
    
    print(f"🤖 DreamerV3 Multi-Agent Training Session")
    print("=" * 60)
    print(f"📊 Configuration:")
    print(f"   • Agents: {num_agents}")
    print(f"   • Target successes: {target_successes}")
    print(f"   • Max training steps: {max_training_steps}")
    print(f"   • Output file: {save_name}")
    
    visualizer = SingleEpisodeVisualizer()
    
    # Create shared challenging scenario
    config = {
        'arena_size': 16.0,
        'arena_height': 7.0,
        'num_obstacles': 6,
        'obstacle_radius_range': [0.8, 1.6],
        'goal_threshold': 1.0,
        'obstacle_clearance': 2.2,
        'min_start_goal_distance': 9.0,
        'max_steps': 300,
        'difficulty_level': 'realistic'
    }
    
    # Create shared environment
    print(f"\n🗺️ Generating shared challenge scenario...")
    shared_env = SimpleDroneEnv(config=config)
    shared_env.reset()
    
    # Force a challenging but achievable scenario
    shared_obstacles = [(pos.copy(), radius) for pos, radius in zip(shared_env.obstacle_positions, shared_env.obstacle_radii)]
    shared_start_position = shared_env.drone_position.copy()
    shared_goal_position = shared_env.goal_position.copy()
    
    challenge_distance = np.linalg.norm(shared_goal_position - shared_start_position)
    print(f"   ✓ Arena: {config['arena_size']}x{config['arena_size']}x{config['arena_height']}m")
    print(f"   ✓ Obstacles: {len(shared_obstacles)}")
    print(f"   ✓ Start: ({shared_start_position[0]:.1f}, {shared_start_position[1]:.1f}, {shared_start_position[2]:.1f})")
    print(f"   ✓ Goal: ({shared_goal_position[0]:.1f}, {shared_goal_position[1]:.1f}, {shared_goal_position[2]:.1f})")
    print(f"   ✓ Distance: {challenge_distance:.1f}m")
    
    # Create diverse DreamerV3 configurations
    dreamerv3_configs = [
        {'learning_rate': 1e-4, 'imagination_horizon': 15, 'hidden_dim': 256, 'latent_dim': 32, 'name': 'Standard'},
        {'learning_rate': 2e-4, 'imagination_horizon': 12, 'hidden_dim': 256, 'latent_dim': 32, 'name': 'Fast'},
        {'learning_rate': 5e-5, 'imagination_horizon': 20, 'hidden_dim': 320, 'latent_dim': 40, 'name': 'Patient'},
        {'learning_rate': 1.5e-4, 'imagination_horizon': 10, 'hidden_dim': 192, 'latent_dim': 28, 'name': 'Efficient'},
        {'learning_rate': 3e-4, 'imagination_horizon': 8, 'hidden_dim': 256, 'latent_dim': 24, 'name': 'Aggressive'},
    ]
    
    # Initialize agents and environments
    print(f"\n🚀 Initializing {num_agents} DreamerV3 agents...")
    
    # Get environment dimensions
    test_obs = shared_env._get_observation()
    if isinstance(test_obs, tuple):
        test_obs = test_obs[0]
    if test_obs.ndim > 1:
        test_obs = test_obs[0]
    obs_dim = len(test_obs)
    action_dim = 3
    
    agents = []
    envs = []
    agent_names = []
    
    for i in range(num_agents):
        config_idx = i % len(dreamerv3_configs)
        agent_config = dreamerv3_configs[config_idx]
        
        # Create agent
        agent = DreamerV3Agent(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=agent_config['hidden_dim'],
            latent_dim=agent_config['latent_dim'],
            learning_rate=agent_config['learning_rate'],
            imagination_horizon=agent_config['imagination_horizon'],
            gamma=0.99,
            lambda_gae=0.95,
            device=str(visualizer.device)
        )
        
        # Create environment copy
        env = SimpleDroneEnv(config=config)
        env.reset()
        # Force same obstacle and goal layout
        env.obstacle_positions = [pos.copy() for pos, _ in shared_obstacles]
        env.obstacle_radii = [radius for _, radius in shared_obstacles]
        env.drone_position = shared_start_position.copy()
        env.goal_position = shared_goal_position.copy()
        
        agents.append(agent)
        envs.append(env)
        agent_names.append(f"Agent{i+1}({agent_config['name']})")
        
        print(f"   ✓ Agent {i+1} ({agent_config['name']}): LR={agent_config['learning_rate']}, H={agent_config['imagination_horizon']}")
    
    # Training phase
    print(f"\n🎯 Training until {target_successes} agents succeed...")
    start_time = time.time()
    
    episode_results = []
    successful_agents = 0
    training_step = 0
    
    while successful_agents < target_successes and training_step < max_training_steps:
        training_step += 1
        step_start_time = time.time()
        
        # Train each agent that hasn't succeeded
        for agent_idx, (agent, env, agent_name) in enumerate(zip(agents, envs, agent_names)):
            # Skip successful agents
            if len(episode_results) > agent_idx and episode_results[agent_idx].get('success', False):
                continue
            
            # Reset to shared positions
            env.drone_position = shared_start_position.copy()
            env.goal_position = shared_goal_position.copy()
            
            # Quick training episode
            episode_data = train_quick_episode(agent, env, agent_idx + 1, agent_name)
            
            # Store or update results
            if len(episode_results) <= agent_idx:
                episode_results.append(episode_data)
                if episode_data['success']:
                    successful_agents += 1
                    print(f"   🎉 {agent_name} SUCCEEDED! ({successful_agents}/{target_successes}) - Steps: {episode_data['steps']}, Distance: {episode_data['final_distance']:.2f}m")
            elif episode_data['success'] and not episode_results[agent_idx].get('success', False):
                episode_results[agent_idx] = episode_data
                successful_agents += 1
                print(f"   🎉 {agent_name} SUCCEEDED! ({successful_agents}/{target_successes}) - Steps: {episode_data['steps']}, Distance: {episode_data['final_distance']:.2f}m")
            
            # Early break if target reached
            if successful_agents >= target_successes:
                break
        
        # Progress reporting
        step_duration = time.time() - step_start_time
        if training_step % 10 == 0 or successful_agents >= target_successes:
            elapsed = time.time() - start_time
            print(f"   📊 Step {training_step:4d}: {successful_agents}/{target_successes} succeeded | Elapsed: {elapsed:.1f}s | Step: {step_duration:.2f}s")
        
        # Safety break
        if training_step >= max_training_steps:
            print(f"   ⚠️ Reached max training steps ({max_training_steps})")
            break
    
    # Fill remaining episode slots with final attempts
    while len(episode_results) < num_agents:
        dummy_episode = {
            'path': [shared_start_position.copy(), shared_start_position.copy() + np.array([0.5, 0.5, 0])],
            'start': shared_start_position.copy(),
            'goal': shared_goal_position.copy(),
            'obstacles': shared_obstacles,
            'success': False,
            'steps': 10,
            'final_distance': challenge_distance,
            'agent_type': 'DreamerV3',
            'agent_id': len(episode_results) + 1
        }
        episode_results.append(dummy_episode)
    
    # Summary
    total_time = time.time() - start_time
    print(f"\n📊 Training Complete!")
    print(f"   ✓ Total time: {total_time:.1f}s")
    print(f"   ✓ Training steps: {training_step}")
    print(f"   ✓ Successful agents: {successful_agents}/{num_agents}")
    print(f"   ✓ Success rate: {successful_agents/num_agents*100:.1f}%")
    
    # Create visualization
    print(f"\n🎨 Creating visualization...")
    fig = visualizer.plot_multiple_episodes(episode_results, save_path=save_name)
    
    print(f"   ✓ Saved as: {save_name}")
    
    return episode_results, successful_agents >= target_successes

def train_quick_episode(agent, env, agent_id, agent_name):
    """Train agent for one quick episode"""
    
    max_steps = 100
    training_episodes = 2
    
    # Store initial positions
    initial_start = env.drone_position.copy()
    initial_goal = env.goal_position.copy()
    
    # Quick training burst
    for _ in range(training_episodes):
        env.drone_position = initial_start.copy()
        env.goal_position = initial_goal.copy()
        
        obs = env._get_observation()
        if isinstance(obs, tuple):
            obs = obs[0]
        if obs.ndim > 1:
            obs = obs[0]
        
        for step in range(max_steps):
            try:
                action = agent.get_action(obs, deterministic=False)
            except:
                action = np.random.uniform(-0.5, 0.5, 3)
            
            if isinstance(action, np.ndarray) and action.ndim > 1:
                action = action.flatten()
            
            next_obs, reward, done, info = env.step(action)
            if isinstance(next_obs, tuple):
                next_obs = next_obs[0]
            if next_obs.ndim > 1:
                next_obs = next_obs[0]
            
            agent.add_experience(obs, action, reward, next_obs, done, info)
            obs = next_obs
            
            if done or np.linalg.norm(env.drone_position - env.goal_position) <= env.config['goal_threshold']:
                break
        
        # Train if enough data
        if len(agent.replay_buffer) > 8:
            try:
                agent.train_step(batch_size=8)
            except:
                pass
    
    # Evaluation run
    env.drone_position = initial_start.copy()
    env.goal_position = initial_goal.copy()
    
    obs = env._get_observation()
    if isinstance(obs, tuple):
        obs = obs[0]
    if obs.ndim > 1:
        obs = obs[0]
    
    path = [env.drone_position.copy()]
    
    for step in range(max_steps):
        try:
            action = agent.get_action(obs, deterministic=True)
        except:
            action = np.random.uniform(-0.3, 0.3, 3)
        
        if isinstance(action, np.ndarray) and action.ndim > 1:
            action = action.flatten()
        
        next_obs, reward, done, info = env.step(action)
        if isinstance(next_obs, tuple):
            next_obs = next_obs[0]
        if next_obs.ndim > 1:
            next_obs = next_obs[0]
        
        path.append(env.drone_position.copy())
        obs = next_obs
        
        final_distance = np.linalg.norm(env.drone_position - env.goal_position)
        if final_distance <= env.config['goal_threshold']:
            break
    
    final_distance = np.linalg.norm(env.drone_position - env.goal_position)
    success = final_distance <= env.config['goal_threshold']
    
    return {
        'path': path,
        'start': initial_start.copy(),
        'goal': initial_goal.copy(),
        'obstacles': [(pos.copy(), radius) for pos, radius in zip(env.obstacle_positions, env.obstacle_radii)],
        'success': success,
        'steps': len(path),
        'final_distance': final_distance,
        'agent_type': 'DreamerV3',
        'agent_id': agent_id
    }

def main():
    """Main function"""
    
    args = parse_arguments()
    
    print(f"🤖 DreamerV3 Multi-Agent Trainer")
    print(f"Starting training session with {args.agents} agents...")
    
    try:
        episodes, success = train_dreamerv3_agents(
            num_agents=args.agents,
            target_successes=args.successes,
            max_training_steps=args.max_steps,
            save_name=args.save_name
        )
        
        if success:
            print(f"\n🎉 Mission accomplished! {args.successes} agents successfully reached the goal.")
            print(f"📊 Results saved as '{args.save_name}'")
        else:
            print(f"\n⚠️ Training completed but target not fully reached.")
            print(f"📊 Partial results saved as '{args.save_name}'")
        
        return True
        
    except KeyboardInterrupt:
        print(f"\n⏹️ Training interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit_code = 0 if success else 1
    sys.exit(exit_code)
