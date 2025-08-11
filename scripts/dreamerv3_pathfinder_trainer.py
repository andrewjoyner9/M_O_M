#!/usr/bin/env python3
"""
DreamerV3 Self-Improving Trainer with Isaac Sim Pathfinding Bootstrap

This system uses the IsaacSimPathfinder to generate optimal initial trajectories,
then trains DreamerV3 agents using these paths as starting guidance to reduce training time.
"""

import os
import sys
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Optional, Any
from collections import deque, defaultdict

# Import the simplified pathfinding system
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from simplified_pathfinder import SimplifiedPathfinder

# Import our existing modules
try:
    from simple_drone_env import SimpleDroneEnv
    from dreamerv3_drone import DreamerV3Agent
    print("✅ All modules imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

class PathfinderGuidedDreamerV3Trainer:
    """
    Enhanced DreamerV3 trainer that uses Isaac Sim pathfinding for initial guidance
    """
    
    def __init__(self, 
                 agents_per_iteration: int = 5, 
                 target_successes: int = 3,
                 max_steps_per_iteration: int = 400,
                 pathfinder_guidance_ratio: float = 0.3,
                 device: str = None):
        """
        Initialize the pathfinder-guided trainer
        
        Args:
            agents_per_iteration: Number of agents to train per iteration
            target_successes: Target successes needed per iteration
            max_steps_per_iteration: Maximum training steps per iteration
            pathfinder_guidance_ratio: Fraction of training steps to use pathfinder guidance
            device: Device to use for training ('cuda' or 'cpu')
        """
        self.agents_per_iteration = agents_per_iteration
        self.target_successes = target_successes
        self.max_steps_per_iteration = max_steps_per_iteration
        self.pathfinder_guidance_ratio = pathfinder_guidance_ratio
        
        # Setup device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        print(f"🤖 Using device: {self.device}")
        
        # Initialize pathfinder
        self.pathfinder = SimplifiedPathfinder()
        print("🗺️ Simplified Pathfinder initialized")
        
        # Training state
        self.current_iteration = 0
        self.consecutive_successes = 0
        self.target_consecutive = 5
        self.best_agent_state = None
        self.training_history = []
        
        # Environment template
        self.env_template = None
        self.shared_scenario = None
        
        print(f"🤖 Pathfinder-Guided DreamerV3 Trainer Initialized")
        print(f"   • Agents per iteration: {agents_per_iteration}")
        print(f"   • Target successes per iteration: {target_successes}")
        print(f"   • Max steps per iteration: {max_steps_per_iteration}")
        print(f"   • Pathfinder guidance ratio: {pathfinder_guidance_ratio:.1%}")
    
    def setup_pathfinder_for_env(self, env: SimpleDroneEnv) -> List[Tuple[float, float, float]]:
        """
        Setup pathfinder with environment obstacles and generate optimal path
        
        Args:
            env: The drone environment
            
        Returns:
            List of (x, y, z) waypoints from start to goal
        """
        # Clear existing obstacles
        self.pathfinder.clear_obstacles()
        
        # Add environment obstacles to pathfinder
        for obs_pos, obs_radius in zip(env.obstacle_positions, env.obstacle_radii):
            # Add as spherical obstacle with radius
            self.pathfinder.add_obstacle(obs_pos[0], obs_pos[1], obs_pos[2], radius=float(obs_radius))
        
        # Get start and goal positions
        start_pos = env.drone_position  # Current drone position is the start
        goal_pos = env.goal_position
        
        print(f"🗺️ Pathfinding from {tuple(start_pos)} to {tuple(goal_pos)}")
        print(f"   • Obstacles: {len(env.obstacle_positions)} spherical")
        
        # Generate optimal path
        path = self.pathfinder.get_optimal_path(start_pos, goal_pos)
        
        print(f"✅ Generated optimal path with {len(path)} waypoints")
        return path
    
    def create_guided_training_sequence(self, env: SimpleDroneEnv, optimal_path: List[Tuple[float, float, float]]) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create a training sequence that follows the optimal path initially
        
        Args:
            env: The drone environment
            optimal_path: Optimal path from pathfinder
            
        Returns:
            List of (observation, action) pairs for initial training
        """
        guidance_sequences = []
        
        if len(optimal_path) < 2:
            return guidance_sequences
        
        # Reset environment
        obs = env.reset()  # SimpleDroneEnv returns only observation
        
        for i in range(len(optimal_path) - 1):
            current_pos = np.array(optimal_path[i])
            next_pos = np.array(optimal_path[i + 1])
            
            # Calculate action to move toward next waypoint
            direction = next_pos - current_pos
            direction_norm = np.linalg.norm(direction)
            
            if direction_norm > 0:
                # Normalize and scale action
                action = direction / direction_norm
                # Ensure action is within bounds [-1, 1]
                action = np.clip(action, -1.0, 1.0)
            else:
                action = np.zeros(3)
            
            guidance_sequences.append((obs.copy(), action.copy()))
            
            # Take action in environment to get next observation
            obs, reward, done, info = env.step(action)  # SimpleDroneEnv returns 4 values
            
            if done:
                break
        
        print(f"🧭 Generated {len(guidance_sequences)} guided training steps")
        return guidance_sequences
    
    def train_agent_with_guidance(self, agent: DreamerV3Agent, env: SimpleDroneEnv, 
                                optimal_path: List[Tuple[float, float, float]], 
                                max_steps: int) -> Dict[str, Any]:
        """
        Train a DreamerV3 agent using pathfinder guidance initially
        
        Args:
            agent: The DreamerV3 agent to train
            env: The environment
            optimal_path: Optimal path from pathfinder
            max_steps: Maximum training steps
            
        Returns:
            Training results dictionary
        """
        start_time = time.time()
        
        # Generate guided training sequence
        guidance_sequence = self.create_guided_training_sequence(env, optimal_path)
        guidance_steps = int(max_steps * self.pathfinder_guidance_ratio)
        
        print(f"🧭 Using pathfinder guidance for first {guidance_steps} steps")
        
        # Phase 1: Guided training with pathfinder
        for step in range(min(guidance_steps, len(guidance_sequence))):
            obs, action = guidance_sequence[step]
            
            # Train agent on guided action
            agent.add_experience(obs, action, 0.1, obs, False)  # Small positive reward for following path
            
            if step % 10 == 0:  # Update networks every 10 steps
                agent.train_step()
        
        # Phase 2: Standard DreamerV3 training
        obs = env.reset()  # SimpleDroneEnv returns only observation
        total_reward = 0
        trajectory = [env.drone_position.copy()]
        success = False
        
        for step in range(guidance_steps, max_steps):
            # Get action from agent
            action = agent.get_action(obs)
            
            # Take step in environment
            next_obs, reward, done, info = env.step(action)  # SimpleDroneEnv returns 4 values
            total_reward += reward
            trajectory.append(env.drone_position.copy())
            
            # Update agent
            agent.add_experience(obs, action, reward, next_obs, done)
            
            if step % 5 == 0:  # Update networks every 5 steps
                agent.train_step()
            
            obs = next_obs
            
            if done:
                success = True
                break
            
            # Check for timeout (since no truncated in this env)
            if step >= max_steps - 1:
                break
        
        training_time = time.time() - start_time
        final_distance = np.linalg.norm(env.drone_position - env.goal_position)
        
        return {
            'success': success,
            'total_reward': total_reward,
            'final_distance': final_distance,
            'trajectory': trajectory,
            'training_time': training_time,
            'steps_taken': step + 1,
            'guidance_steps': min(guidance_steps, len(guidance_sequence))
        }
    
    def create_persistent_scenario(self) -> Dict[str, Any]:
        """Create a persistent challenging scenario for all iterations"""
        print("🗺️ Creating persistent challenge scenario...")
        
        # Create challenging environment configuration
        config = {
            'arena_size': 18.0,
            'arena_height': 8.0,
            'max_steps': 400,
            'num_obstacles': 7,
            'obstacle_radius_range': [0.8, 1.5],
            'difficulty_level': 'challenging',
            'max_velocity': 2.0,
            'goal_threshold': 1.0,
            'obstacle_clearance': 2.0
        }
        
        # Create environment
        env = SimpleDroneEnv(env_id=0, config=config)
        
        # Reset to generate scenario
        env.reset()
        
        # Get scenario parameters
        scenario = {
            'arena_size': (config['arena_size'], config['arena_size'], config['arena_height']),
            'start_position': env.drone_position.copy(),  # Use drone_position as start
            'goal_position': env.goal_position.copy(),
            'obstacle_positions': [pos.copy() for pos in env.obstacle_positions],
            'obstacle_radii': env.obstacle_radii.copy(),
            'max_episode_steps': config['max_steps'],
            'config': config
        }
        
        # Calculate challenge metrics
        start_goal_distance = np.linalg.norm(scenario['goal_position'] - scenario['start_position'])
        
        print(f"   ✓ Arena: {scenario['arena_size'][0]}x{scenario['arena_size'][1]}x{scenario['arena_size'][2]}m")
        print(f"   ✓ Obstacles: {len(scenario['obstacle_positions'])}")
        print(f"   ✓ Start: {tuple(scenario['start_position'])}")
        print(f"   ✓ Goal: {tuple(scenario['goal_position'])}")
        print(f"   ✓ Challenge distance: {start_goal_distance:.1f}m")
        
        return scenario
    
    def create_env_from_scenario(self, scenario: Dict[str, Any]) -> SimpleDroneEnv:
        """Create environment from saved scenario"""
        # Create environment with same config
        env = SimpleDroneEnv(env_id=0, config=scenario['config'])
        
        # Reset environment first to initialize all components
        env.reset()
        
        # Set fixed scenario parameters
        env.drone_position = np.array(scenario['start_position'])  # Set drone to start position
        env.goal_position = np.array(scenario['goal_position'])
        env.obstacle_positions = [np.array(pos) for pos in scenario['obstacle_positions']]
        env.obstacle_radii = np.array(scenario['obstacle_radii'])
        
        return env
    
    def run_iteration(self, iteration_num: int) -> Dict[str, Any]:
        """
        Run a single training iteration with pathfinder guidance
        
        Args:
            iteration_num: Current iteration number
            
        Returns:
            Iteration results
        """
        print(f"\n🚀 ITERATION {iteration_num}")
        print("=" * 50)
        
        if iteration_num == 1:
            print("   🌱 Starting from scratch with pathfinder guidance")
            self.shared_scenario = self.create_persistent_scenario()
        else:
            print("   👨‍👩‍👧‍👦 Evolving from best agent with pathfinder guidance")
        
        # Create environment from shared scenario
        env = self.create_env_from_scenario(self.shared_scenario)
        
        # Generate optimal path for this scenario
        optimal_path = self.setup_pathfinder_for_env(env)
        
        # Create and train agents
        agents = []
        results = []
        
        for i in range(self.agents_per_iteration):
            # Create agent with variation
            if iteration_num == 1 or self.best_agent_state is None:
                agent = DreamerV3Agent(
                    obs_dim=env.observation_space.shape[0],
                    action_dim=env.action_space.shape[0],
                    device=self.device
                )
                agent_type = "Standard"
            else:
                agent = self.create_evolved_agent(env, i)
                agent_type = f"Evolved-{i+1}"
            
            agents.append((agent, agent_type))
        
        print(f"   🎯 Training {self.agents_per_iteration} agents until {self.target_successes} succeed...")
        
        # Train agents in parallel-like fashion
        successful_agents = []
        
        for step in range(0, self.max_steps_per_iteration, 20):
            step_results = []
            
            for agent_idx, (agent, agent_type) in enumerate(agents):
                if len([r for r in results if r['agent_idx'] == agent_idx and r['success']]) > 0:
                    continue  # Agent already succeeded
                
                # Create fresh environment for this agent
                agent_env = self.create_env_from_scenario(self.shared_scenario)
                
                # Train for this step batch
                result = self.train_agent_with_guidance(
                    agent, agent_env, optimal_path, 
                    min(20, self.max_steps_per_iteration - step)
                )
                
                result.update({
                    'agent_idx': agent_idx,
                    'agent_type': agent_type,
                    'step': step + result['steps_taken']
                })
                
                step_results.append(result)
                
                if result['success']:
                    if result not in successful_agents:
                        successful_agents.append(result)
                        print(f"   🎉 Agent{agent_idx+1}({agent_type}) SUCCEEDED! ({len(successful_agents)}/{self.target_successes}) - Distance: {result['final_distance']:.2f}m")
            
            results.extend(step_results)
            
            # Progress update
            if (step + 20) % 20 == 0:
                elapsed = sum(r['training_time'] for r in results) / len(agents)
                print(f"   📊 Step {step + 20:4d}: {len(successful_agents)}/{self.target_successes} succeeded | Elapsed: {elapsed:.1f}s")
            
            # Check if we have enough successes
            if len(successful_agents) >= self.target_successes:
                break
        
        # Calculate iteration results
        iteration_result = {
            'iteration': iteration_num,
            'successes': len(successful_agents),
            'success_rate': len(successful_agents) / self.agents_per_iteration,
            'total_agents': self.agents_per_iteration,
            'training_time': sum(r['training_time'] for r in results) / len(agents),
            'training_steps': max(r['step'] for r in results) if results else 0,
            'results': results,
            'successful_agents': successful_agents,
            'optimal_path': optimal_path,
            'pathfinder_guidance_used': True
        }
        
        # Update best agent if we have successes
        if successful_agents:
            best_result = min(successful_agents, key=lambda x: x['final_distance'])
            best_agent_idx = best_result['agent_idx']
            
            # Save best agent state dictionary
            best_agent = agents[best_agent_idx][0]
            self.best_agent_state = {
                'world_model': best_agent.world_model.state_dict(),
                'actor': best_agent.actor.state_dict(),
                'critic': best_agent.critic.state_dict(),
                'config': {
                    'obs_dim': best_agent.obs_dim,
                    'action_dim': best_agent.action_dim,
                    'hidden_dim': best_agent.hidden_dim,
                    'latent_dim': best_agent.latent_dim
                }
            }
        
        # Print iteration summary
        print(f"\n   📊 Iteration {iteration_num} Results:")
        print(f"      • Successes: {len(successful_agents)}/{self.agents_per_iteration} ({len(successful_agents)/self.agents_per_iteration:.1%})")
        print(f"      • Training time: {iteration_result['training_time']:.1f}s")
        print(f"      • Training steps: {iteration_result['training_steps']}")
        print(f"      • Pathfinder guidance: {len(optimal_path)} waypoints")
        
        # Check if iteration was successful
        if len(successful_agents) >= self.target_successes:
            self.consecutive_successes += 1
            print(f"   ✅ Iteration {iteration_num} succeeded")
        else:
            self.consecutive_successes = 0
            print(f"   ❌ Iteration {iteration_num} failed - resetting success counter")
        
        return iteration_result
    
    def create_evolved_agent(self, env: SimpleDroneEnv, agent_idx: int) -> DreamerV3Agent:
        """Create an evolved agent based on the best agent from previous iteration"""
        if self.best_agent_state is None:
            return DreamerV3Agent(
                obs_dim=env.observation_space.shape[0],
                action_dim=env.action_space.shape[0],
                device=self.device
            )
        
        # Create new agent with mutations
        agent = DreamerV3Agent(
            obs_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            device=self.device
        )
        
        try:
            # Try to load best agent state
            if 'world_model' in self.best_agent_state:
                agent.world_model.load_state_dict(self.best_agent_state['world_model'])
            if 'actor' in self.best_agent_state:
                agent.actor.load_state_dict(self.best_agent_state['actor'])
            if 'critic' in self.best_agent_state:
                agent.critic.load_state_dict(self.best_agent_state['critic'])
            
            # TODO: Apply mutations for diversity (implement later)
            # mutation_rate = 0.1 + agent_idx * 0.05
            # agent.mutate_weights(mutation_rate)
            
            print(f"   ✅ Agent {agent_idx+1} evolved from best parent")
            
        except Exception as e:
            print(f"   ⚠️ Evolution failed for agent {agent_idx+1}: {e}")
            # Fallback to new agent
        
        return agent
    
    def visualize_iteration(self, iteration_result: Dict[str, Any]) -> str:
        """Create visualization for an iteration"""
        from single_episode_visualizer import plot_multiple_episodes
        
        # Prepare episodes for visualization
        episodes = []
        
        for result in iteration_result['results']:
            if 'trajectory' in result:
                episode = {
                    'trajectory': result['trajectory'],
                    'success': result['success'],
                    'agent_type': result['agent_type'],
                    'final_distance': result['final_distance'],
                    'total_reward': result['total_reward']
                }
                episodes.append(episode)
        
        # Add optimal path as reference
        if 'optimal_path' in iteration_result:
            optimal_episode = {
                'trajectory': iteration_result['optimal_path'],
                'success': True,
                'agent_type': 'Pathfinder-Optimal',
                'final_distance': 0.0,
                'total_reward': 100.0
            }
            episodes.insert(0, optimal_episode)  # Add at beginning for visibility
        
        # Create visualization
        save_path = f"dreamerv3_pathfinder_iteration_{iteration_result['iteration']}.png"
        
        plot_multiple_episodes(
            episodes, 
            self.shared_scenario,
            save_path,
            title=f"🧭 DreamerV3 + Pathfinder Evolution - Iteration {iteration_result['iteration']}"
        )
        
        print(f"   💾 Iteration {iteration_result['iteration']} visualization saved as '{save_path}'")
        return save_path
    
    def run_evolution(self) -> List[Dict[str, Any]]:
        """
        Run the complete evolution process with pathfinder guidance
        
        Returns:
            List of iteration results
        """
        print("🧬 DreamerV3 + Pathfinder Self-Improving Evolution System")
        print("=" * 60)
        print(f"Goal: Achieve {self.target_successes} successes for {self.target_consecutive} consecutive iterations")
        print(f"Enhancement: Using Isaac Sim pathfinder for {self.pathfinder_guidance_ratio:.1%} initial guidance")
        
        iteration_results = []
        
        try:
            while self.consecutive_successes < self.target_consecutive:
                self.current_iteration += 1
                
                # Run iteration
                result = self.run_iteration(self.current_iteration)
                iteration_results.append(result)
                
                # Create visualization
                self.visualize_iteration(result)
                
                # Update training history
                self.training_history.append(result)
                
                # Progress update
                print(f"\n📈 Evolution Progress: {self.consecutive_successes}/{self.target_consecutive} successful iterations")
                
                # Save checkpoint
                self.save_checkpoint(iteration_results)
                
        except KeyboardInterrupt:
            print("\n⏹️ Evolution interrupted by user")
            print("Partial results have been saved")
        
        # Create final summary
        self.create_evolution_summary(iteration_results)
        
        return iteration_results
    
    def save_checkpoint(self, results: List[Dict[str, Any]]):
        """Save training checkpoint"""
        checkpoint = {
            'results': results,
            'current_iteration': self.current_iteration,
            'consecutive_successes': self.consecutive_successes,
            'best_agent_state': self.best_agent_state,
            'shared_scenario': self.shared_scenario,
            'training_history': self.training_history
        }
        
        checkpoint_path = f"dreamerv3_pathfinder_checkpoint_iter_{self.current_iteration}.pkl"
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint, f)
    
    def create_evolution_summary(self, results: List[Dict[str, Any]]):
        """Create comprehensive evolution summary"""
        if not results:
            return
        
        print(f"\n🏆 Evolution Summary")
        print("=" * 50)
        
        for result in results:
            success_rate = result['success_rate']
            print(f"Iteration {result['iteration']}: {result['successes']}/{result['total_agents']} ({success_rate:.1%}) - {result['training_time']:.1f}s")
        
        if self.consecutive_successes >= self.target_consecutive:
            print(f"\n🎉 EVOLUTION COMPLETED! Achieved {self.target_consecutive} consecutive successful iterations!")
        else:
            print(f"\n📊 Evolution stopped at {self.consecutive_successes}/{self.target_consecutive} consecutive successes")
        
        print(f"\nTotal iterations: {len(results)}")
        avg_success_rate = np.mean([r['success_rate'] for r in results])
        print(f"Average success rate: {avg_success_rate:.1%}")
        print(f"Pathfinder guidance: {self.pathfinder_guidance_ratio:.1%} of training steps")

def main():
    """Main execution function"""
    print("🧬 DreamerV3 + Isaac Sim Pathfinder Training System")
    print("=" * 60)
    
    # Create trainer with pathfinder guidance
    trainer = PathfinderGuidedDreamerV3Trainer(
        agents_per_iteration=5,
        target_successes=3,
        max_steps_per_iteration=400,
        pathfinder_guidance_ratio=0.4,  # Use 40% of training for pathfinder guidance
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Run evolution
    results = trainer.run_evolution()
    
    print("\n✅ Training completed!")
    return results

if __name__ == "__main__":
    results = main()
