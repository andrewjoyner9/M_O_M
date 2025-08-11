#!/usr/bin/env python3
"""
Self-Improving DreamerV3 Multi-Agent Training System

This script implements an iterative training system where:
1. Multiple DreamerV3 agents train on a shared scenario
2. The best performing agent from each iteration becomes the "parent" for the next iteration
3. New agents are initialized with the parent's weights plus variations
4. This continues until we achieve successful navigation for 5 consecutive iterations

This creates a self-improving, evolutionary training approach.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os
import time
import torch
import copy

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

class SelfImprovingDreamerV3Trainer:
    """Self-improving DreamerV3 training system with iterative evolution"""
    
    def __init__(self, num_agents=7, target_successes=3, max_steps_per_iteration=800):
        self.num_agents = num_agents
        self.target_successes = target_successes
        self.max_steps_per_iteration = max_steps_per_iteration
        self.visualizer = SingleEpisodeVisualizer()
        
        # Training history
        self.iteration_results = []
        self.best_agents_history = []
        self.success_iterations = 0
        
        # Shared scenario
        self.shared_env = None
        self.shared_obstacles = None
        self.shared_start_position = None
        self.shared_goal_position = None
        self.challenge_distance = 0
        
        print(f"🤖 Self-Improving DreamerV3 Trainer Initialized")
        print(f"   • Agents per iteration: {num_agents}")
        print(f"   • Target successes per iteration: {target_successes}")
        print(f"   • Max steps per iteration: {max_steps_per_iteration}")
    
    def create_shared_scenario(self):
        """Create the shared challenge scenario that all iterations will use"""
        
        config = {
            'arena_size': 18.0,
            'arena_height': 8.0,
            'num_obstacles': 7,
            'obstacle_radius_range': [0.8, 1.8],
            'goal_threshold': 1.0,
            'obstacle_clearance': 2.2,
            'min_start_goal_distance': 12.0,  # Make it challenging
            'max_steps': 400,
            'difficulty_level': 'challenging'
        }
        
        print(f"\n🗺️ Creating persistent challenge scenario...")
        self.shared_env = SimpleDroneEnv(config=config)
        self.shared_env.reset()
        
        # Store the permanent challenge
        self.shared_obstacles = [(pos.copy(), radius) for pos, radius in zip(self.shared_env.obstacle_positions, self.shared_env.obstacle_radii)]
        self.shared_start_position = self.shared_env.drone_position.copy()
        self.shared_goal_position = self.shared_env.goal_position.copy()
        self.challenge_distance = np.linalg.norm(self.shared_goal_position - self.shared_start_position)
        
        print(f"   ✓ Arena: {config['arena_size']}x{config['arena_size']}x{config['arena_height']}m")
        print(f"   ✓ Obstacles: {len(self.shared_obstacles)}")
        print(f"   ✓ Start: ({self.shared_start_position[0]:.1f}, {self.shared_start_position[1]:.1f}, {self.shared_start_position[2]:.1f})")
        print(f"   ✓ Goal: ({self.shared_goal_position[0]:.1f}, {self.shared_goal_position[1]:.1f}, {self.shared_goal_position[2]:.1f})")
        print(f"   ✓ Challenge distance: {self.challenge_distance:.1f}m")
        
        return True
    
    def create_agent_variations(self, parent_agent=None):
        """Create diverse agent configurations, optionally based on a parent agent"""
        
        base_configs = [
            {'learning_rate': 1e-4, 'imagination_horizon': 15, 'hidden_dim': 256, 'latent_dim': 32, 'name': 'Standard'},
            {'learning_rate': 2e-4, 'imagination_horizon': 12, 'hidden_dim': 256, 'latent_dim': 32, 'name': 'Fast'},
            {'learning_rate': 5e-5, 'imagination_horizon': 20, 'hidden_dim': 320, 'latent_dim': 40, 'name': 'Patient'},
            {'learning_rate': 1.5e-4, 'imagination_horizon': 10, 'hidden_dim': 192, 'latent_dim': 28, 'name': 'Efficient'},
            {'learning_rate': 3e-4, 'imagination_horizon': 8, 'hidden_dim': 256, 'latent_dim': 24, 'name': 'Aggressive'},
            {'learning_rate': 8e-5, 'imagination_horizon': 18, 'hidden_dim': 288, 'latent_dim': 36, 'name': 'Balanced'},
            {'learning_rate': 2.5e-4, 'imagination_horizon': 14, 'hidden_dim': 224, 'latent_dim': 30, 'name': 'Adaptive'},
        ]
        
        # Get environment dimensions
        test_obs = self.shared_env._get_observation()
        if isinstance(test_obs, tuple):
            test_obs = test_obs[0]
        if test_obs.ndim > 1:
            test_obs = test_obs[0]
        obs_dim = len(test_obs)
        action_dim = 3
        
        agents = []
        envs = []
        agent_names = []
        
        for i in range(self.num_agents):
            config_idx = i % len(base_configs)
            agent_config = base_configs[config_idx].copy()
            
            # If we have a parent agent, modify the config to be similar but varied
            if parent_agent is not None:
                # Create variations around the parent's successful configuration
                variation_factor = 0.3  # 30% variation
                agent_config['learning_rate'] *= np.random.uniform(1-variation_factor, 1+variation_factor)
                agent_config['imagination_horizon'] = max(5, int(agent_config['imagination_horizon'] * np.random.uniform(0.8, 1.2)))
                agent_config['hidden_dim'] = max(128, int(agent_config['hidden_dim'] * np.random.uniform(0.9, 1.1)))
                agent_config['latent_dim'] = max(16, int(agent_config['latent_dim'] * np.random.uniform(0.9, 1.1)))
                agent_config['name'] += f"-Evolved"
            
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
                device=str(self.visualizer.device)
            )
            
            # If we have a parent, transfer some knowledge
            if parent_agent is not None:
                self.transfer_knowledge(parent_agent, agent)
            
            # Create environment copy
            env = SimpleDroneEnv(config={'arena_size': 18.0, 'arena_height': 8.0, 'num_obstacles': 7, 
                                       'obstacle_radius_range': [0.8, 1.8], 'goal_threshold': 1.0,
                                       'obstacle_clearance': 2.2, 'min_start_goal_distance': 12.0,
                                       'max_steps': 400, 'difficulty_level': 'challenging'})
            env.reset()
            # Force same obstacle and goal layout
            env.obstacle_positions = [pos.copy() for pos, _ in self.shared_obstacles]
            env.obstacle_radii = [radius for _, radius in self.shared_obstacles]
            env.drone_position = self.shared_start_position.copy()
            env.goal_position = self.shared_goal_position.copy()
            
            agents.append(agent)
            envs.append(env)
            agent_names.append(f"Agent{i+1}({agent_config['name']})")
        
        return agents, envs, agent_names
    
    def transfer_knowledge(self, parent_agent, child_agent):
        """Transfer knowledge from parent to child agent with some mutation"""
        
        try:
            # Copy parent's network weights with small random mutations
            mutation_strength = 0.1
            
            # Transfer world model
            parent_world_state = parent_agent.world_model.state_dict()
            child_world_state = child_agent.world_model.state_dict()
            
            for key in parent_world_state:
                if key in child_world_state:
                    # Copy with mutation
                    parent_weights = parent_world_state[key]
                    mutation = torch.randn_like(parent_weights) * mutation_strength
                    child_world_state[key] = parent_weights + mutation
            
            child_agent.world_model.load_state_dict(child_world_state)
            
            # Transfer actor (with more mutation for exploration)
            parent_actor_state = parent_agent.actor.state_dict()
            child_actor_state = child_agent.actor.state_dict()
            
            for key in parent_actor_state:
                if key in child_actor_state:
                    parent_weights = parent_actor_state[key]
                    mutation = torch.randn_like(parent_weights) * mutation_strength * 1.5  # More mutation for policy
                    child_actor_state[key] = parent_weights + mutation
            
            child_agent.actor.load_state_dict(child_actor_state)
            
            # Transfer some replay buffer experience
            if len(parent_agent.replay_buffer) > 0:
                # Copy a portion of parent's experience
                parent_experiences = parent_agent.replay_buffer.buffer[:min(100, len(parent_agent.replay_buffer))]
                for exp in parent_experiences:
                    child_agent.replay_buffer.add(exp)
            
            print(f"   ✓ Knowledge transferred to child agent")
            
        except Exception as e:
            print(f"   ⚠️ Knowledge transfer failed: {e}")
    
    def train_iteration(self, iteration_num, parent_agent=None):
        """Train one iteration of agents"""
        
        print(f"\n🚀 ITERATION {iteration_num}")
        print("=" * 50)
        
        if parent_agent is not None:
            print(f"   👨‍👩‍👧‍👦 Evolving from best agent of previous iteration")
        else:
            print(f"   🌱 Starting from scratch (first iteration)")
        
        # Create agents for this iteration
        agents, envs, agent_names = self.create_agent_variations(parent_agent)
        
        print(f"   🎯 Training {len(agents)} agents until {self.target_successes} succeed...")
        
        # Training loop
        episode_results = []
        successful_agents = 0
        training_step = 0
        start_time = time.time()
        
        while successful_agents < self.target_successes and training_step < self.max_steps_per_iteration:
            training_step += 1
            
            # Train each agent that hasn't succeeded
            for agent_idx, (agent, env, agent_name) in enumerate(zip(agents, envs, agent_names)):
                # Skip successful agents
                if len(episode_results) > agent_idx and episode_results[agent_idx].get('success', False):
                    continue
                
                # Reset to shared positions
                env.drone_position = self.shared_start_position.copy()
                env.goal_position = self.shared_goal_position.copy()
                
                # Train episode
                episode_data = self.train_quick_episode(agent, env, agent_idx + 1, agent_name)
                
                # Store or update results
                if len(episode_results) <= agent_idx:
                    episode_results.append(episode_data)
                    if episode_data['success']:
                        successful_agents += 1
                        print(f"   🎉 {agent_name} SUCCEEDED! ({successful_agents}/{self.target_successes}) - Distance: {episode_data['final_distance']:.2f}m")
                elif episode_data['success'] and not episode_results[agent_idx].get('success', False):
                    episode_results[agent_idx] = episode_data
                    successful_agents += 1
                    print(f"   🎉 {agent_name} SUCCEEDED! ({successful_agents}/{self.target_successes}) - Distance: {episode_data['final_distance']:.2f}m")
                
                # Early break if target reached
                if successful_agents >= self.target_successes:
                    break
            
            # Progress reporting
            if training_step % 20 == 0 or successful_agents >= self.target_successes:
                elapsed = time.time() - start_time
                print(f"   📊 Step {training_step:4d}: {successful_agents}/{self.target_successes} succeeded | Elapsed: {elapsed:.1f}s")
            
            # Safety break
            if training_step >= self.max_steps_per_iteration:
                print(f"   ⚠️ Reached max steps ({self.max_steps_per_iteration}) for iteration {iteration_num}")
                break
        
        # Fill remaining slots
        while len(episode_results) < self.num_agents:
            dummy_episode = {
                'path': [self.shared_start_position.copy(), self.shared_start_position.copy() + np.array([0.5, 0.5, 0])],
                'start': self.shared_start_position.copy(),
                'goal': self.shared_goal_position.copy(),
                'obstacles': self.shared_obstacles,
                'success': False,
                'steps': 10,
                'final_distance': self.challenge_distance,
                'agent_type': 'DreamerV3',
                'agent_id': len(episode_results) + 1,
                'iteration': iteration_num
            }
            episode_results.append(dummy_episode)
        
        # Add iteration info to all episodes
        for ep in episode_results:
            ep['iteration'] = iteration_num
        
        # Find best agent for next iteration
        best_agent = None
        best_score = -float('inf')
        
        for agent_idx, (agent, episode) in enumerate(zip(agents, episode_results)):
            # Score: prioritize success, then minimize distance, then minimize steps
            if episode['success']:
                score = 1000 - episode['final_distance'] - episode['steps'] * 0.1
            else:
                score = -episode['final_distance']
            
            if score > best_score:
                best_score = score
                best_agent = agent
        
        # Summary
        total_time = time.time() - start_time
        success_rate = successful_agents / self.num_agents
        
        print(f"\n   📊 Iteration {iteration_num} Results:")
        print(f"      • Successes: {successful_agents}/{self.num_agents} ({success_rate*100:.1f}%)")
        print(f"      • Training time: {total_time:.1f}s")
        print(f"      • Training steps: {training_step}")
        
        # Store iteration results
        iteration_result = {
            'iteration': iteration_num,
            'episodes': episode_results,
            'successful_agents': successful_agents,
            'success_rate': success_rate,
            'training_time': total_time,
            'training_steps': training_step,
            'best_agent': best_agent
        }
        
        self.iteration_results.append(iteration_result)
        self.best_agents_history.append(best_agent)
        
        # Check if this iteration was successful
        if successful_agents >= self.target_successes:
            self.success_iterations += 1
            print(f"      ✅ ITERATION {iteration_num} SUCCESSFUL! ({self.success_iterations}/5 needed)")
        else:
            self.success_iterations = 0  # Reset counter
            print(f"      ❌ Iteration {iteration_num} failed - resetting success counter")
        
        return iteration_result, best_agent
    
    def train_quick_episode(self, agent, env, agent_id, agent_name):
        """Train agent for one quick episode"""
        
        max_steps = 150
        training_episodes = 3
        
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
            if len(agent.replay_buffer) > 12:
                try:
                    agent.train_step(batch_size=12)
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
            'obstacles': self.shared_obstacles,
            'success': success,
            'steps': len(path),
            'final_distance': final_distance,
            'agent_type': 'DreamerV3',
            'agent_id': agent_id
        }
    
    def run_evolution(self, max_iterations=20):
        """Run the full evolutionary training process"""
        
        print(f"🧬 Starting Self-Improving DreamerV3 Evolution")
        print("=" * 60)
        print(f"Goal: Achieve {self.target_successes} successes for 5 consecutive iterations")
        
        # Create the persistent challenge
        if not self.create_shared_scenario():
            print("❌ Failed to create shared scenario")
            return False
        
        parent_agent = None
        iteration = 1
        
        while self.success_iterations < 5 and iteration <= max_iterations:
            iteration_result, best_agent = self.train_iteration(iteration, parent_agent)
            
            # The best agent becomes the parent for the next iteration
            parent_agent = best_agent
            
            print(f"\n📈 Evolution Progress: {self.success_iterations}/5 successful iterations")
            
            # Create iteration visualization
            save_name = f'dreamerv3_evolution_iteration_{iteration}.png'
            self.visualizer.plot_multiple_episodes(iteration_result['episodes'], save_path=save_name)
            print(f"   💾 Iteration {iteration} visualization saved as '{save_name}'")
            
            iteration += 1
        
        # Final results
        if self.success_iterations >= 5:
            print(f"\n🎉 EVOLUTION COMPLETE! Achieved 5 consecutive successful iterations!")
            print(f"   Total iterations: {iteration - 1}")
        else:
            print(f"\n⏰ Evolution stopped after {max_iterations} iterations")
            print(f"   Achieved {self.success_iterations}/5 consecutive successes")
        
        # Create final comprehensive visualization
        self.create_evolution_summary()
        
        return self.success_iterations >= 5
    
    def create_evolution_summary(self):
        """Create a comprehensive visualization of the entire evolution process"""
        
        print(f"\n🎨 Creating evolution summary visualization...")
        
        # Collect all episodes from all iterations
        all_episodes = []
        iteration_colors = plt.cm.viridis(np.linspace(0, 1, len(self.iteration_results)))
        
        for iter_result in self.iteration_results:
            for episode in iter_result['episodes']:
                episode['iteration_color'] = iteration_colors[iter_result['iteration'] - 1]
                all_episodes.append(episode)
        
        # Create the summary visualization
        fig = plt.figure(figsize=(24, 16))
        
        # Evolution progress plot
        ax1 = fig.add_subplot(231)
        iterations = [r['iteration'] for r in self.iteration_results]
        success_rates = [r['success_rate'] * 100 for r in self.iteration_results]
        ax1.plot(iterations, success_rates, 'bo-', linewidth=2, markersize=8)
        ax1.set_title('Evolution Progress: Success Rate by Iteration', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Success Rate (%)')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 100)
        
        # Training time evolution
        ax2 = fig.add_subplot(232)
        training_times = [r['training_time'] for r in self.iteration_results]
        ax2.plot(iterations, training_times, 'ro-', linewidth=2, markersize=8)
        ax2.set_title('Training Efficiency: Time per Iteration', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Training Time (seconds)')
        ax2.grid(True, alpha=0.3)
        
        # Path visualization for latest iteration
        ax3 = fig.add_subplot(233, projection='3d')
        latest_episodes = self.iteration_results[-1]['episodes']
        
        for i, episode in enumerate(latest_episodes):
            if episode.get('path') and len(episode['path']) > 1:
                path = np.array(episode['path'])
                color = 'green' if episode['success'] else 'red'
                alpha = 0.8 if episode['success'] else 0.5
                ax3.plot(path[:, 0], path[:, 1], path[:, 2], color=color, alpha=alpha, linewidth=2)
        
        # Plot shared scenario
        start = self.shared_start_position
        goal = self.shared_goal_position
        ax3.scatter(*start, color='blue', s=200, marker='o', label='Start')
        ax3.scatter(*goal, color='gold', s=200, marker='*', label='Goal')
        
        ax3.set_title(f'Latest Iteration Paths (Iteration {len(self.iteration_results)})', fontsize=14, fontweight='bold')
        ax3.set_xlabel('X Position')
        ax3.set_ylabel('Y Position')
        ax3.set_zlabel('Z Position')
        ax3.legend()
        
        # Success timeline
        ax4 = fig.add_subplot(234)
        successful_counts = [r['successful_agents'] for r in self.iteration_results]
        colors = ['green' if count >= self.target_successes else 'red' for count in successful_counts]
        bars = ax4.bar(iterations, successful_counts, color=colors, alpha=0.7)
        ax4.axhline(y=self.target_successes, color='blue', linestyle='--', linewidth=2, label=f'Target ({self.target_successes})')
        ax4.set_title('Successful Agents per Iteration', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Number of Successful Agents')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Training steps evolution
        ax5 = fig.add_subplot(235)
        training_steps = [r['training_steps'] for r in self.iteration_results]
        ax5.plot(iterations, training_steps, 'mo-', linewidth=2, markersize=8)
        ax5.set_title('Training Steps per Iteration', fontsize=14, fontweight='bold')
        ax5.set_xlabel('Iteration')
        ax5.set_ylabel('Training Steps')
        ax5.grid(True, alpha=0.3)
        
        # Summary statistics
        ax6 = fig.add_subplot(236)
        ax6.axis('off')
        
        total_time = sum(r['training_time'] for r in self.iteration_results)
        total_steps = sum(r['training_steps'] for r in self.iteration_results)
        best_success_rate = max(r['success_rate'] for r in self.iteration_results) * 100
        
        summary_text = f"""
🧬 DREAMERV3 EVOLUTION SUMMARY
{'='*40}

🎯 Mission Status:
• Consecutive Successes: {self.success_iterations}/5
• Total Iterations: {len(self.iteration_results)}
• Best Success Rate: {best_success_rate:.1f}%

⏱️ Performance Metrics:
• Total Training Time: {total_time:.1f}s
• Total Training Steps: {total_steps:,}
• Average Time/Iteration: {total_time/len(self.iteration_results):.1f}s

🗺️ Challenge Parameters:
• Distance: {self.challenge_distance:.1f}m
• Obstacles: {len(self.shared_obstacles)}
• Goal Threshold: 1.0m

🤖 Evolution Strategy:
• Agents per iteration: {self.num_agents}
• Knowledge transfer: ✓
• Weight mutations: ✓
• Experience inheritance: ✓
"""
        
        ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
        
        # Overall title
        status = "COMPLETED" if self.success_iterations >= 5 else "IN PROGRESS"
        fig.suptitle(f'DreamerV3 Self-Improving Evolution - {status}', fontsize=18, fontweight='bold')
        
        plt.tight_layout()
        
        # Save
        save_name = 'dreamerv3_evolution_complete_summary.png'
        plt.savefig(save_name, dpi=150, bbox_inches='tight')
        print(f"   💾 Complete evolution summary saved as '{save_name}'")
        
        return fig

def main():
    """Main function"""
    
    print(f"🧬 DreamerV3 Self-Improving Evolution System")
    print("=" * 60)
    
    # Create trainer
    trainer = SelfImprovingDreamerV3Trainer(
        num_agents=5,           # Fewer agents for faster iterations
        target_successes=3,     # Need 3 successes per iteration
        max_steps_per_iteration=600  # Reasonable time limit
    )
    
    try:
        # Run the evolution
        success = trainer.run_evolution(max_iterations=15)
        
        if success:
            print(f"\n🎉 MISSION ACCOMPLISHED!")
            print(f"Successfully achieved 5 consecutive iterations with target successes!")
        else:
            print(f"\n⚠️ Evolution incomplete but progress made")
            print(f"Check the iteration visualizations for detailed analysis")
        
        return success
        
    except KeyboardInterrupt:
        print(f"\n⏹️ Evolution interrupted by user")
        print(f"Partial results have been saved")
        return False
    except Exception as e:
        print(f"\n❌ Error during evolution: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit_code = 0 if success else 1
    sys.exit(exit_code)
