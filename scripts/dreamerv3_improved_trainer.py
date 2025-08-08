#!/usr/bin/env python3
"""
CRITICAL FIX: DreamerV3 Enhanced Trainer with Fixed Learning System
================================================================

CRITICAL FIXES APPLIED:
1. Enhanced pathfinder guidance (65% → 35% adaptive)
2. Increased training episodes (20 per agent)  
3. Fixed checkpoint data structure
4. Improved quality scoring with progress rewards
5. Enhanced success reinforcement learning
6. Fixed episode experience collection
7. Proper buffer learning validation
"""

import os
import sys
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional
import pickle
import torch

# Add script directory to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

try:
    import gymnasium as gym
    print("✓ Using Gymnasium")
except ImportError:
    import gym
    print("✓ Using Gym")

# Import our components
try:
    from dreamerv3_drone import DreamerV3Agent
    from simple_drone_env import SimpleDroneEnv
    from simplified_pathfinder import SimplifiedPathfinder
    from single_episode_visualizer import plot_multiple_episodes
    print("✅ All modules imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

class ImprovedDreamerV3Trainer:
    """
    Enhanced pathfinder-guided DreamerV3 training system with better learning dynamics
    """
    
    def __init__(self, 
                 agents_per_iteration: int = 12,  # FIXED: Increased for better evolution
                 target_successes: int = 6,       # FIXED: Increased target
                 max_episode_steps: int = 1000,   # FIXED: Increased episode length
                 training_episodes_per_agent: int = 20):  # FIXED: Increased training episodes
        """
        Initialize the FIXED trainer with critical improvements
        
        CRITICAL FIXES:
        - Enhanced pathfinder guidance (65% → 35% adaptive)
        - Increased training episodes for better learning
        - Fixed checkpoint data structure
        - Improved quality scoring with progress rewards
        """
        self.agents_per_iteration = agents_per_iteration
        self.target_successes = target_successes
        self.max_episode_steps = max_episode_steps
        self.training_episodes_per_agent = training_episodes_per_agent
        
        # CRITICAL FIX: Enhanced guidance parameters
        self.initial_pathfinder_guidance_ratio = 0.65  # FIXED: Much higher initial guidance
        self.min_pathfinder_guidance_ratio = 0.35      # FIXED: Higher minimum guidance
        self.success_sample_ratio = 0.75               # FIXED: Higher success sampling
        self.quality_threshold = 1.2                   # FIXED: Higher quality threshold
        
        # Training state
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pathfinder = SimplifiedPathfinder()
        self.shared_scenario = None
        self.best_agent_state = None
        self.best_agent_performance = float('-inf')  # Track best performance score (higher is better)
        self.successful_agent_pool = []  # Track multiple successful agents for diverse evolution
        self.historical_agent_pool = []  # ENHANCED: Track well-performing agents from previous iterations
        self.max_parent_pool_size = 3   # Keep top 3 successful agents as parents
        self.max_historical_pool_size = 5  # Keep top 5 historical agents for "fresh" diversity
        self.consecutive_successes = 0
        self.current_scenario_goals_used = []  # Track goals used for current scenario
        self.scenario_success_threshold = 0.5  # 50% success rate threshold for goal advancement
        self.iteration_results = []
        self.historical_rewards = {}  # Store rewards by iteration: {iteration_num: {agent_id: best_reward}}
        
        # CRITICAL FIX: Track actual learning statistics
        self.learning_stats = {
            'iterations': [],
            'success_rates': [],
            'buffer_sizes': [],
            'avg_rewards': [],
            'goal_orientation': []
        }
        
        print(f"🤖 Using device: {self.device}")
        print(f"🗺️ Simplified Pathfinder initialized")
        
        print(f"🤖 FIXED Pathfinder-Guided DreamerV3 Trainer Initialized")
        print(f"   • Agents per iteration: {self.agents_per_iteration}")
        print(f"   • Target successes per iteration: {self.target_successes}")
        print(f"   • Max episode steps: {self.max_episode_steps}")
        print(f"   • Training episodes per agent: {self.training_episodes_per_agent}")
        print(f"   • Initial pathfinder guidance: {self.initial_pathfinder_guidance_ratio*100:.1f}%")
        print(f"   • Success sampling ratio: {self.success_sample_ratio*100:.1f}%")
    
    def setup_pathfinder_for_env(self, env: SimpleDroneEnv) -> bool:
        """Setup pathfinder with environment obstacles"""
        try:
            # Clear existing obstacles
            self.pathfinder.clear_obstacles()
            
            # Add spherical obstacles from environment
            for i, obs_pos in enumerate(env.obstacle_positions):
                radius = env.obstacle_radii[i] if i < len(env.obstacle_radii) else 1.0
                self.pathfinder.add_obstacle(obs_pos[0], obs_pos[1], obs_pos[2], radius)
                print(f"Spherical obstacle added at {tuple(obs_pos)} with radius={radius}")
            
            return True
        except Exception as e:
            print(f"❌ Failed to setup pathfinder: {e}")
            return False
    
    def _calculate_performance_score(self, best_distance, total_reward, success_count):
        """
        Calculate composite performance score that balances:
        - Distance efficiency (shorter path = better)
        - Reward efficiency (higher reward = better obstacle avoidance)
        - Success rate
        """
        # Distance component (invert so lower distance = higher score)
        if best_distance < 2.5:  # Success threshold
            distance_score = 100.0 / (1.0 + best_distance)  # Higher score for lower distance
        else:
            distance_score = 50.0 / (1.0 + best_distance)  # Reduced score for failed attempts
        
        # Reward component (normalize and scale)
        reward_score = max(0, total_reward / self.training_episodes_per_agent + 100)  # Offset negative rewards
        
        # Success bonus
        success_bonus = success_count * 50.0  # Big bonus for successful episodes
        
        # Composite score
        performance_score = distance_score + reward_score * 0.1 + success_bonus
        return performance_score
    
    def _calculate_episode_quality_score(self, reward: float, distance: float, success: bool, episode_length: int) -> float:
        """
        CRITICAL FIX: Enhanced quality scoring that properly rewards goal-seeking behavior
        
        Args:
            reward: Episode total reward
            distance: Final distance to goal
            success: Whether episode succeeded
            episode_length: Number of steps taken
            
        Returns:
            Quality score (higher = better for replay priority)
        """
        if success:
            # SUCCESS: High base score + efficiency bonuses
            base_quality = 2.0
            
            # Distance efficiency (closer = better)
            distance_bonus = max(0, (10 - distance) / 10) * 1.0
            
            # Reward efficiency (less negative = better navigation)
            reward_bonus = max(0, (reward + 1000) / 1000) * 0.5
            
            return base_quality + distance_bonus + reward_bonus
        else:
            # FAILURE: Score based on progress toward goal
            # CRITICAL FIX: Reward progress instead of just penalizing failure
            progress_quality = max(0.1, (15 - distance) / 15)  # Progress toward goal
            
            # Penalize crashes harshly
            if reward < -2500:
                return 0.05  # Crash penalty
            elif reward < -2000:
                return progress_quality * 0.3  # Heavy penalty
            else:
                return progress_quality * 0.6   # Moderate penalty with progress reward
    
    def adaptive_pathfinder_guidance(self, iteration_num):
        """CRITICAL FIX: Calculate adaptive guidance based on learning progress"""
        if iteration_num <= 1:
            return self.initial_pathfinder_guidance_ratio
        
        # Get recent success rates
        recent_successes = []
        if len(self.learning_stats['success_rates']) >= 3:
            recent_successes = self.learning_stats['success_rates'][-3:]
        
        if recent_successes:
            avg_recent_success = np.mean(recent_successes)
            # Reduce guidance as success improves, but keep it substantial
            guidance_reduction = min(0.2, avg_recent_success / 100)
            current_guidance = self.initial_pathfinder_guidance_ratio - guidance_reduction
            return max(self.min_pathfinder_guidance_ratio, current_guidance)
        
        return self.initial_pathfinder_guidance_ratio
    
    def _get_quality_metrics_for_iteration(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate quality metrics for the entire iteration for tracking improvement"""
        successful_rewards = []
        successful_distances = []
        all_rewards = []
        
        for result in results:
            all_rewards.append(result.get('average_reward', -1500))
            if result['success']:
                successful_rewards.append(result.get('average_reward', -1500))
                successful_distances.append(result['best_distance'])
        
        metrics = {
            'avg_all_rewards': np.mean(all_rewards) if all_rewards else -1500,
            'avg_success_rewards': np.mean(successful_rewards) if successful_rewards else 0,
            'avg_success_distance': np.mean(successful_distances) if successful_distances else 10,
            'reward_quality_improvement': 0,  # Will be calculated relative to previous iterations
            'navigation_efficiency': 0
        }
        
        if successful_rewards:
            # Higher (less negative) rewards indicate better navigation quality
            metrics['navigation_efficiency'] = np.mean(successful_rewards) / -100  # Normalize
            
        return metrics
    
    def should_advance_to_new_goal(self, iteration_num: int, current_success_rate: float) -> bool:
        """Determine if we should advance to a new goal based on success rate"""
        # Always advance on first iteration
        if iteration_num <= 1:
            return True
            
        # If success rate is above threshold (50%), agents have mastered this goal
        if current_success_rate >= self.scenario_success_threshold:
            print(f"   🎯 Goal mastery achieved: {current_success_rate:.1%} success rate (≥ {self.scenario_success_threshold:.1%})")
            print(f"   🔄 Advancing to new goal for continued learning")
            return True
        else:
            print(f"   📚 Continuing with current goal: {current_success_rate:.1%} success rate (< {self.scenario_success_threshold:.1%})")
            print(f"   🎓 Agents still learning basic pathfinding on this scenario")
            return False
    
    def _find_best_failed_trajectory(self, all_trajectories: List[List], goal_position: np.ndarray) -> List:
        """Find the best failed trajectory based on how close it got to the goal"""
        if not all_trajectories:
            return []
        
        best_trajectory = None
        best_min_distance = float('inf')
        
        for trajectory in all_trajectories:
            if not trajectory:
                continue
                
            # Find the minimum distance this trajectory achieved to the goal
            min_distance = float('inf')
            for position in trajectory:
                distance = np.linalg.norm(np.array(position) - goal_position)
                min_distance = min(min_distance, distance)
            
            # Keep track of the trajectory that got closest to the goal
            if min_distance < best_min_distance:
                best_min_distance = min_distance
                best_trajectory = trajectory
        
        # Fallback to last trajectory if none found
        return best_trajectory if best_trajectory is not None else all_trajectories[-1]
    
    def create_guided_training_sequence(self, env: SimpleDroneEnv, 
                                      optimal_path: List[Tuple[float, float, float]]) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Create guided training sequence with better action mapping"""
        guidance_sequences = []
        
        # Start from current drone position
        obs = env.reset()
        
        # Follow optimal path with more careful action computation
        for i in range(len(optimal_path) - 1):
            current_pos = np.array(optimal_path[i])
            next_pos = np.array(optimal_path[i + 1])
            
            # Calculate smoothed action
            direction = next_pos - current_pos
            direction_norm = np.linalg.norm(direction)
            
            if direction_norm > 0:
                # Scale action based on distance - closer positions get smaller actions
                scale_factor = min(1.0, direction_norm / 2.0)  # Smooth scaling
                action = (direction / direction_norm) * scale_factor
                action = np.clip(action, -1.0, 1.0)
            else:
                action = np.zeros(3)
            
            guidance_sequences.append((obs.copy(), action.copy()))
            
            # Take action in environment  
            obs, reward, done, info = env.step(action)
            
            if done:
                break
        
        print(f"🧭 Generated {len(guidance_sequences)} guided training steps")
        return guidance_sequences
    
    def train_agent_intensively(self, agent: DreamerV3Agent, env: SimpleDroneEnv, 
                               optimal_path: List[Tuple[float, float, float]], 
                               guidance_ratio: Optional[float] = None) -> Dict[str, Any]:
        """
        Intensive training with multiple full episodes and adaptive guidance integration
        """
        start_time = time.time()
        
        # Use adaptive guidance ratio if provided, otherwise use adaptive calculation
        if guidance_ratio is not None:
            effective_guidance_ratio = guidance_ratio
        else:
            # Use adaptive guidance based on learning progress
            effective_guidance_ratio = self.adaptive_pathfinder_guidance(1)  # Default to iteration 1 guidance
        
        # Generate guided sequence
        guidance_sequence = self.create_guided_training_sequence(env, optimal_path)
        guidance_steps = int(self.max_episode_steps * effective_guidance_ratio)
        
        all_trajectories = []
        total_reward = 0
        best_distance = float('inf')
        success_count = 0
        best_successful_trajectory = None  # Store best successful path
        final_episode_success = False  # Track final episode specifically
        
        print(f"🧭 Starting intensive training: {self.training_episodes_per_agent} episodes")
        print(f"   • Pathfinder guidance: {guidance_steps} steps per episode")
        print(f"   • Success-prioritized learning: Enabled for better navigation behavior")
        
        # Initialize episode tracking for the agent
        agent.start_episode(env.drone_position, env.goal_position)
        
        # Train for multiple full episodes
        for episode in range(self.training_episodes_per_agent):
            obs = env.reset()
            episode_reward = 0
            trajectory = [env.drone_position.copy()]
            episode_success = False
            episode_experiences = []  # Track this episode's experiences
            
            # Start new episode tracking in agent
            agent.start_episode(env.drone_position, env.goal_position)
            
            # Phase 1: Guided training with pathfinder (with some exploration)
            for step in range(min(guidance_steps, len(guidance_sequence))):
                if step < len(guidance_sequence):
                    guided_obs, guided_action = guidance_sequence[step]
                    
                    # Mix guided action with agent's action for exploration
                    agent_action = agent.get_action(obs)
                    mix_ratio = max(0.1, 1.0 - (step / len(guidance_sequence)))  # Decreasing guidance
                    action = mix_ratio * guided_action + (1 - mix_ratio) * agent_action
                    action = np.clip(action, -1.0, 1.0)
                else:
                    action = agent.get_action(obs)
                
                # Take step
                next_obs, reward, done, info = env.step(action)
                episode_reward += reward
                trajectory.append(env.drone_position.copy())
                
                # Add experience to agent AND episode buffer for quality-weighted learning
                agent.add_experience(obs, action, reward, next_obs, done)
                episode_experiences.append((obs, action, reward, next_obs, done, info))
                
                # Frequent training updates
                if step % 5 == 0 and step > 0:
                    agent.train_step()
                
                obs = next_obs
                
                if done:
                    episode_success = info.get('success', False)
                    break
            
            # Phase 2: Pure agent training (if episode not done)
            if not done:
                for step in range(guidance_steps, self.max_episode_steps):
                    action = agent.get_action(obs)
                    next_obs, reward, done, info = env.step(action)
                    episode_reward += reward
                    trajectory.append(env.drone_position.copy())
                    
                    # Add experience to agent AND episode buffer for quality-weighted learning
                    agent.add_experience(obs, action, reward, next_obs, done)
                    episode_experiences.append((obs, action, reward, next_obs, done, info))
                    
                    # Regular training updates
                    if step % 3 == 0:
                        agent.train_step()
                    
                    obs = next_obs
                    
                    if done:
                        episode_success = info.get('success', False)
                        break
            
            # Track episode results
            final_distance = np.linalg.norm(env.drone_position - env.goal_position)
            best_distance = min(best_distance, final_distance)
            total_reward += episode_reward
            all_trajectories.append(trajectory)
            
            if episode_success:
                success_count += 1
                # Calculate quality score for this successful episode
                episode_quality = self._calculate_episode_quality_score(
                    episode_reward, final_distance, True, len(trajectory)
                )
                
                # Store the BEST successful trajectory (now considering both distance and quality)
                if best_successful_trajectory is None or episode_quality > getattr(self, '_best_success_quality', 0):
                    best_successful_trajectory = trajectory.copy()
                    self._best_success_quality = episode_quality
                    print(f"   🎉 Episode {episode+1}: SUCCESS! Distance: {final_distance:.2f}m, Quality: {episode_quality:.2f} (NEW BEST QUALITY PATH)")
                    
                    # Add high-quality successful episode with priority weighting
                    if episode_experiences:
                        agent.add_episode(episode_experiences, success=True, 
                                        final_distance=final_distance, total_reward=episode_reward)
                else:
                    print(f"   🎉 Episode {episode+1}: SUCCESS! Distance: {final_distance:.2f}m, Quality: {episode_quality:.2f}")
                    
                    # Still add successful episode with its quality score
                    if episode_experiences:
                        agent.add_episode(episode_experiences, success=True, 
                                        final_distance=final_distance, total_reward=episode_reward)
            else:
                # Calculate quality score for failed episode (to rank "promising failures")
                episode_quality = self._calculate_episode_quality_score(
                    episode_reward, final_distance, False, len(trajectory)
                )
                
                print(f"   📊 Episode {episode+1}: Failed. Distance: {final_distance:.2f}m, Reward: {episode_reward:.1f}, Quality: {episode_quality:.2f}")
                
                # Add failed episode with quality-based priority (learn from better failures)
                if episode_experiences:
                    agent.add_episode(episode_experiences, success=False, 
                                    final_distance=final_distance, total_reward=episode_reward)
            
            # Track if this is the final episode
            if episode == self.training_episodes_per_agent - 1:
                final_episode_success = episode_success
        
        # Final intensive training burst
        for _ in range(20):
            agent.train_step()
        
        training_time = time.time() - start_time
        
        # Get buffer statistics to show learning from successful behaviors
        buffer_stats = agent.get_buffer_statistics()
        print(f"   📊 Training complete: {buffer_stats['success_count']} successful sequences in buffer ({buffer_stats['success_ratio']:.1%})")
        if buffer_stats['success_count'] > 0:
            print(f"   🎯 Avg success distance: {buffer_stats['avg_success_distance']:.2f}m, Avg success reward: {buffer_stats['avg_success_reward']:.1f}")
        
        return {
            'success': success_count > 0,  # Agent succeeds if ANY episode succeeds (we want to find successful behaviors)
            'final_episode_success': final_episode_success,  # Track final episode separately for conservative metrics
            'success_count': success_count,
            'success_rate': success_count / self.training_episodes_per_agent,
            'best_distance': best_distance,
            'total_reward': total_reward,
            'average_reward': total_reward / self.training_episodes_per_agent,
            'performance_score': self._calculate_performance_score(best_distance, total_reward, success_count),
            'all_trajectories': all_trajectories,
            'best_trajectory': best_successful_trajectory if best_successful_trajectory else self._find_best_failed_trajectory(all_trajectories, env.goal_position),  # Show best failed attempt instead of last attempt
            'training_time': training_time,
            'episodes_trained': self.training_episodes_per_agent,
            'buffer_stats': buffer_stats  # Include buffer statistics for analysis
        }
    
    def create_persistent_scenario(self) -> Dict[str, Any]:
        """Create a persistent challenging scenario with fixed obstacles but varied goals"""
        import random
        from reward_config_manager import RewardConfigManager
        
        print("🗺️ Creating enhanced challenge scenario...")
        
        # Fixed configuration for consistent obstacles
        config = {
            'arena_size': 16.0,  # Fixed arena size
            'arena_height': 8.0,
            'max_steps': self.max_episode_steps,
            'num_obstacles': 8,  # Fixed obstacle count
            'obstacle_radius_range': [0.6, 1.1],
            'difficulty_level': 'medium',
            'max_velocity': 2.0,
            'goal_threshold': 2.5,
            'obstacle_clearance': 2.0
        }
        
        # Create environment with fixed seed for consistent obstacles
        fixed_seed = 42  # Always use same seed for obstacles
        random.seed(fixed_seed)
        np.random.seed(fixed_seed)
        
        env = SimpleDroneEnv(env_id=0, config=config)
        # Apply boundary-aware reward configuration to prevent ceiling/ground seeking
        RewardConfigManager.apply_config_to_env(env, 'boundary_aware_v1')
        env.reset()
        
        # Store the fixed obstacle layout
        fixed_obstacles = [pos.copy() for pos in env.obstacle_positions]
        fixed_radii = env.obstacle_radii.copy()
        fixed_start = env.drone_position.copy()
        
        # Now vary only the goal position based on iteration (or keep current goal for mastery)
        if hasattr(self, 'shared_scenario') and hasattr(self, 'current_scenario_goals_used'):
            # Check if we're continuing with the current goal for mastery learning
            if self.current_iteration > 1:
                previous_result = self.iteration_results[-1] if self.iteration_results else None
                previous_success_rate = previous_result.get('success_rate', 0) / 100.0 if previous_result else 0
                if not self.should_advance_to_new_goal(self.current_iteration, previous_success_rate):
                    # Keep the current goal
                    goal_position = np.array(self.shared_scenario['goal_position'])
                    print(f"   🎯 Maintaining current goal for mastery: {tuple(goal_position)}")
                    
                    scenario = {
                        'arena_size': (config['arena_size'], config['arena_size'], config['arena_height']),
                        'start_position': fixed_start,
                        'goal_position': goal_position,
                        'obstacle_positions': fixed_obstacles,
                        'obstacle_radii': fixed_radii
                    }
                    
                    return scenario
        
        # Generate new goal position
        goal_seed = 42 + len(getattr(self, 'current_scenario_goals_used', [])) * 3
        random.seed(goal_seed)
        np.random.seed(goal_seed)
        
        # Generate varied goal positions that avoid obstacles
        max_attempts = 50
        for attempt in range(max_attempts):
            # Generate random goal position in arena
            goal_x = random.uniform(-config['arena_size']/2 + 2, config['arena_size']/2 - 2)
            goal_y = random.uniform(-config['arena_size']/2 + 2, config['arena_size']/2 - 2)
            goal_z = random.uniform(2.0, config['arena_height'] - 1.0)
            goal_position = np.array([goal_x, goal_y, goal_z])
            
            # Check distance from start (ensure reasonable challenge)
            start_distance = np.linalg.norm(goal_position - fixed_start)
            if start_distance < 6.0 or start_distance > 12.0:
                continue
                
            # Check clearance from obstacles
            valid_goal = True
            for i, obs_pos in enumerate(fixed_obstacles):
                obs_distance = np.linalg.norm(goal_position - obs_pos)
                min_clearance = fixed_radii[i] + 3.0  # 3m clearance from obstacles
                if obs_distance < min_clearance:
                    valid_goal = False
                    break
            
            if valid_goal:
                break
        else:
            # Fallback to a safe default goal
            goal_position = np.array([4.0, -4.0, 4.0])
        
        scenario = {
            'arena_size': (config['arena_size'], config['arena_size'], config['arena_height']),
            'start_position': fixed_start,
            'goal_position': goal_position,
            'obstacle_positions': fixed_obstacles,
            'obstacle_radii': fixed_radii,
            'max_episode_steps': config['max_steps'],
            'config': config
        }
        
        start_goal_distance = np.linalg.norm(scenario['goal_position'] - scenario['start_position'])
        
        print(f"   ✓ Arena: {scenario['arena_size'][0]:.1f}x{scenario['arena_size'][1]:.1f}x{scenario['arena_size'][2]:.1f}m")
        print(f"   ✓ Obstacles: {len(scenario['obstacle_positions'])}")
        print(f"   ✓ Start: {tuple(scenario['start_position'])}")
        print(f"   ✓ Goal: {tuple(scenario['goal_position'])}")
        print(f"   ✓ Challenge distance: {start_goal_distance:.1f}m")
        
        # Reset random seed to default
        random.seed()
        np.random.seed()
        
        return scenario
    
    def create_env_from_scenario(self, scenario: Dict[str, Any]) -> SimpleDroneEnv:
        """Create environment from scenario with fixed start/goal positions"""
        env = SimpleDroneEnv(env_id=0, config=scenario['config'])
        
        # Apply boundary-aware reward configuration to prevent ceiling/ground seeking
        from reward_config_manager import RewardConfigManager
        RewardConfigManager.apply_config_to_env(env, 'boundary_aware_v1')
        
        env.reset()
        
        # Set scenario parameters and store them for future resets
        env.drone_position = np.array(scenario['start_position'])
        env.goal_position = np.array(scenario['goal_position'])
        env.obstacle_positions = [np.array(pos) for pos in scenario['obstacle_positions']]
        env.obstacle_radii = np.array(scenario['obstacle_radii'])
        
        # Store fixed positions in the environment for consistency
        env._fixed_start_position = np.array(scenario['start_position'])
        env._fixed_goal_position = np.array(scenario['goal_position'])
        env._use_fixed_positions = True
        
        return env
    
    def run_iteration(self, iteration_num: int) -> Dict[str, Any]:
        """Run single training iteration with adaptive goal advancement"""
        print(f"\n🚀 ITERATION {iteration_num}")
        print("=" * 50)
        
        # Store current iteration for scenario seeding
        self.current_iteration = iteration_num
        
        # Check if we should advance to a new goal based on previous success rate
        should_advance = True
        if iteration_num > 1 and hasattr(self, 'shared_scenario'):
            # Get the success rate from the previous iteration
            previous_result = self.iteration_results[-1] if self.iteration_results else None
            previous_success_rate = previous_result.get('success_rate', 0) / 100.0 if previous_result else 0
            should_advance = self.should_advance_to_new_goal(iteration_num, previous_success_rate)
        
        if iteration_num == 1:
            print("   🌱 Starting from scratch with enhanced training")
            self.shared_scenario = self.create_persistent_scenario()
            self.current_scenario_goals_used = [tuple(self.shared_scenario['goal_position'])]
        elif should_advance:
            print(f"   🧬 Evolving from best agents (consecutive successes: {self.consecutive_successes})")
            print("   🎯 Creating new goal scenario for continued challenge")
            self.shared_scenario = self.create_persistent_scenario()
            self.current_scenario_goals_used.append(tuple(self.shared_scenario['goal_position']))
        else:
            print(f"   🧬 Evolving from best agents (consecutive successes: {self.consecutive_successes})")
            print("   📚 Continuing with current goal for mastery learning")
            # Keep the same scenario but allow agents to continue learning
        
        # Setup pathfinder
        test_env = self.create_env_from_scenario(self.shared_scenario)
        if not self.setup_pathfinder_for_env(test_env):
            print("❌ Failed to setup pathfinder")
            return {'iteration': iteration_num, 'success': False}
        
        # Get optimal path
        start_pos = tuple(self.shared_scenario['start_position'])
        goal_pos = tuple(self.shared_scenario['goal_position'])
        
        print(f"🗺️ Pathfinding from {start_pos} to {goal_pos}")
        print(f"   • Obstacles: {len(self.shared_scenario['obstacle_positions'])} spherical")
        
        optimal_path = self.pathfinder.get_optimal_path(start_pos, goal_pos)
        
        if not optimal_path:
            print("❌ No path found by pathfinder")
            return {'iteration': iteration_num, 'success': False}
        
        print(f"✅ Generated optimal path with {len(optimal_path)} waypoints")
        
        # CRITICAL FIX: Use adaptive guidance based on learning progress
        adaptive_guidance = self.adaptive_pathfinder_guidance(iteration_num)
        print(f"   🎯 Training {self.agents_per_iteration} agents until {self.target_successes} succeed...")
        print(f"   🧭 Adaptive guidance: {adaptive_guidance:.1%} (enhanced for learning)")
        
        results = []
        successful_agents = []
        
        for agent_idx in range(self.agents_per_iteration):
            print(f"\n   🤖 Training Agent {agent_idx+1}/{self.agents_per_iteration}")
            
            # Create environment for this agent
            agent_env = self.create_env_from_scenario(self.shared_scenario)
            
            # Create agent with intelligent evolution strategy using historical diversity
            if self.best_agent_state is not None and iteration_num > 1:
                # Calculate current success rate for adaptive evolution
                recent_success_rate = getattr(self, 'recent_success_rate', 0.0)
                
                # Adaptive strategy based on performance
                if recent_success_rate < 0.5:
                    # Low success: 70% evolved, 30% from historical pool for guided exploration
                    diversity_threshold = int(self.agents_per_iteration * 0.7)
                    strategy_name = "exploration"
                else:
                    # High success: 90% evolved, 10% from historical pool for controlled innovation
                    diversity_threshold = int(self.agents_per_iteration * 0.9)
                    strategy_name = "exploitation"
                
                if agent_idx < diversity_threshold:
                    # Use current successful parent pool
                    agent = self.create_evolved_agent(agent_env, agent_idx)
                else:
                    # ENHANCED: Use historical high-performing agent instead of fresh random
                    agent = self.create_historical_agent(agent_env, agent_idx, strategy_name, recent_success_rate)
            else:
                # First iteration - all fresh
                agent = DreamerV3Agent(
                    obs_dim=agent_env.observation_space.shape[0],
                    action_dim=agent_env.action_space.shape[0],
                    device=self.device
                )
            
            # Intensive training with adaptive guidance
            result = self.train_agent_intensively(agent, agent_env, optimal_path, adaptive_guidance)
            result.update({
                'agent_idx': agent_idx,
                'iteration': iteration_num
            })
            
            results.append(result)
            
            if result['success']:
                successful_agents.append((agent, result))
                buffer_stats = result.get('buffer_stats', {})
                success_buffer_count = buffer_stats.get('success_count', 0)
                
                print(f"   🎉 Agent{agent_idx+1} SUCCEEDED! Success rate: {result['success_rate']:.1%}, Best distance: {result['best_distance']:.2f}m, Score: {result['performance_score']:.1f}")
                print(f"       📚 Buffer learning: {success_buffer_count} successful sequences learned from")
                
                # Update best agent state if this agent found ANY successful behavior and has best performance
                if result['success'] and result['performance_score'] > self.best_agent_performance:
                    self.best_agent_performance = result['performance_score']
                    self.best_agent_state = {
                        'world_model': agent.world_model.state_dict(),
                        'actor': agent.actor.state_dict(), 
                        'critic': agent.critic.state_dict(),
                        'result': result,
                        'performance': result['performance_score'],
                        'buffer_stats': buffer_stats  # Include buffer statistics for evolution
                    }
                    print(f"   🏆 NEW BEST AGENT! Performance score improved to {result['performance_score']:.1f} (found {result['success_count']} successes, learned from {success_buffer_count} successful sequences)")
                
                # ENHANCED: Add to successful agent pool for diverse evolution
                if result['success']:
                    agent_state = {
                        'world_model': agent.world_model.state_dict(),
                        'actor': agent.actor.state_dict(), 
                        'critic': agent.critic.state_dict(),
                        'result': result,
                        'performance': result['performance_score'],
                        'buffer_stats': buffer_stats,
                        'agent_idx': agent_idx,
                        'iteration': iteration_num,
                        'success_rate': result['success_rate']
                    }
                    self.successful_agent_pool.append(agent_state)
                    
                    # Keep only the best performing agents in the pool
                    self.successful_agent_pool.sort(key=lambda x: x['performance'], reverse=True)
                    if len(self.successful_agent_pool) > self.max_parent_pool_size:
                        self.successful_agent_pool = self.successful_agent_pool[:self.max_parent_pool_size]
                    
                    if result['performance_score'] <= self.best_agent_performance:
                        print(f"   ✅ Success found (score: {result['performance_score']:.1f}) - added to parent pool ({len(self.successful_agent_pool)}/{self.max_parent_pool_size})")
                
                # ENHANCED: Also track well-performing agents (even if not successful) for historical diversity
                if result['performance_score'] > 50.0 or result['best_distance'] < 8.0:  # Good performance threshold
                    historical_state = {
                        'world_model': agent.world_model.state_dict(),
                        'actor': agent.actor.state_dict(), 
                        'critic': agent.critic.state_dict(),
                        'result': result,
                        'performance': result['performance_score'],
                        'buffer_stats': buffer_stats,
                        'agent_idx': agent_idx,
                        'iteration': iteration_num,
                        'success_rate': result['success_rate'],
                        'best_distance': result['best_distance']
                    }
                    self.historical_agent_pool.append(historical_state)
                    
                    # Keep only the best historical agents
                    self.historical_agent_pool.sort(key=lambda x: x['performance'], reverse=True)
                    if len(self.historical_agent_pool) > self.max_historical_pool_size:
                        self.historical_agent_pool = self.historical_agent_pool[:self.max_historical_pool_size]
                
                if not result['success']:
                    print(f"   ❌ No successes found. Best distance: {result['best_distance']:.2f}m, Score: {result['performance_score']:.1f}")
            else:
                buffer_stats = result.get('buffer_stats', {})
                success_buffer_count = buffer_stats.get('success_count', 0)
                print(f"   ❌ Agent{agent_idx+1} failed. Best distance: {result['best_distance']:.2f}m, Score: {result['performance_score']:.1f}")
                if success_buffer_count > 0:
                    print(f"       📚 Buffer: Learned from {success_buffer_count} successful sequences but couldn't reproduce success")
        
        # Calculate iteration results with quality metrics
        successes = len(successful_agents)
        quality_metrics = self._get_quality_metrics_for_iteration(results)
        
        # Track quality improvement over iterations
        if self.iteration_results:
            prev_metrics = self.iteration_results[-1].get('quality_metrics', {})
            quality_metrics['reward_quality_improvement'] = (
                quality_metrics['avg_success_rewards'] - prev_metrics.get('avg_success_rewards', -1500)
            )
        
        # Store reward data for this iteration (enhanced with quality tracking)
        self.historical_rewards[iteration_num] = {}
        successful_rewards = []
        
        for agent_idx, result in enumerate(results):
            agent_id = agent_idx + 1
            # Get the best reward from this agent's training
            best_reward = result.get('average_reward', float('-inf'))
            if result['success']:
                successful_rewards.append(best_reward)
            self.historical_rewards[iteration_num][agent_id] = best_reward
        
        iteration_result = {
            'iteration': iteration_num,
            'successes': successes,
            'success_rate': successes / self.agents_per_iteration * 100,  # Store as percentage
            'total_agents': self.agents_per_iteration,
            'training_time': sum(r['training_time'] for r in results),
            'pathfinder_waypoints': len(optimal_path),
            'agent_results': results,
            'successful_agents': len(successful_agents),
            'historical_rewards': dict(self.historical_rewards),  # Pass copy of all historical data
            'quality_metrics': quality_metrics,  # Enhanced quality tracking
            'avg_successful_reward': np.mean(successful_rewards) if successful_rewards else -1500,
            'reward_improvement': quality_metrics.get('reward_quality_improvement', 0),
            'goal_position': tuple(self.shared_scenario['goal_position']),  # Track current goal
            'goals_used_count': len(getattr(self, 'current_scenario_goals_used', [])),  # Track goal progression
            'goal_advancement_logic': 'adaptive_curriculum'  # Mark as using adaptive curriculum
        }
        
        print(f"\n   📊 Iteration {iteration_num} Results:")
        print(f"      • Successes: {successes}/{self.agents_per_iteration} ({successes/self.agents_per_iteration:.1%})")
        print(f"      • Training time: {iteration_result['training_time']:.1f}s")
        print(f"      • Pathfinder waypoints: {len(optimal_path)}")
        print(f"      • Goal: {iteration_result['goal_position']} (goal #{iteration_result['goals_used_count']})")
        if iteration_num > 1:
            previous_goal = self.iteration_results[-1].get('goal_position', 'Unknown') if self.iteration_results else 'Unknown'
            if iteration_result['goal_position'] == previous_goal:
                print(f"      • 📚 Mastery learning: Continuing with same goal for skill development")
            else:
                print(f"      • 🎯 Goal advanced: Agents ready for new challenge")
        
        # Enhanced quality reporting
        if successes > 0:
            avg_success_reward = iteration_result['avg_successful_reward']
            reward_improvement = iteration_result['reward_improvement']
            print(f"      • Success Quality: Avg reward {avg_success_reward:.1f} (Δ{reward_improvement:+.1f})")
            print(f"      • Navigation Efficiency: {quality_metrics['navigation_efficiency']:.2f}")
            
            if reward_improvement > 50:
                print(f"      🚀 Significant quality improvement! Agents learning more efficient navigation")
            elif reward_improvement > 0:
                print(f"      ✅ Quality improving steadily")
        else:
            print(f"      ❌ No successes - focusing on success discovery")
        
        # Track consecutive successes and update recent success rate for adaptive guidance
        if successes >= self.target_successes:
            self.consecutive_successes += 1
            print(f"   ✅ Iteration {iteration_num} succeeded - consecutive successes: {self.consecutive_successes}")
        else:
            self.consecutive_successes = 0
            print(f"   ❌ Iteration {iteration_num} failed - resetting success counter")
        
        # CRITICAL FIX: Update learning statistics for proper tracking and adaptive guidance
        iteration_success_rate = successes / self.agents_per_iteration * 100
        avg_reward = np.mean([result['total_reward'] for result in results])
        
        # Calculate goal orientation (agents getting closer to goal)
        goal_oriented = sum(1 for result in results if result['best_distance'] < 8.0)
        goal_orientation_rate = goal_oriented / self.agents_per_iteration * 100
        
        # Update learning statistics
        self.learning_stats['iterations'].append(iteration_num)
        self.learning_stats['success_rates'].append(iteration_success_rate)
        self.learning_stats['avg_rewards'].append(avg_reward)
        self.learning_stats['goal_orientation'].append(goal_orientation_rate)
        
        # Track buffer sizes for quality-weighted learning verification
        total_buffer_successes = sum(result.get('buffer_stats', {}).get('successful_sequences', 0) for result in results)
        self.learning_stats['buffer_sizes'].append(total_buffer_successes)
        
        # Update recent success rate for adaptive guidance (rolling average over last 3 iterations)
        recent_iterations = 3
        if len(self.iteration_results) >= recent_iterations - 1:
            recent_success_rates = [iteration_result['success_rate']]
            for prev_result in self.iteration_results[-(recent_iterations-1):]:
                recent_success_rates.append(prev_result['success_rate'])
            self.recent_success_rate = np.mean(recent_success_rates) / 100.0  # Convert percentage to decimal for internal use
        else:
            self.recent_success_rate = iteration_result['success_rate'] / 100.0  # Convert percentage to decimal for internal use
        
        # Visualize iteration
        save_path = self.visualize_iteration(iteration_result)
        print(f"   📊 Visualization saved: {save_path}")
        
        # Save checkpoint
        self.save_checkpoint(iteration_num, iteration_result)
        
        self.iteration_results.append(iteration_result)
        return iteration_result
    
    def create_evolved_agent(self, env: SimpleDroneEnv, agent_idx: int) -> DreamerV3Agent:
        """Create evolved agent from successful parent pool with diversity"""
        agent = DreamerV3Agent(
            obs_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            device=self.device
        )
        
        # Select parent from successful agent pool for diversity
        if self.successful_agent_pool:
            # Rotate through successful parents to maintain diversity
            parent_idx = agent_idx % len(self.successful_agent_pool)
            parent_state = self.successful_agent_pool[parent_idx]
            
            try:
                agent.world_model.load_state_dict(parent_state['world_model'])
                agent.actor.load_state_dict(parent_state['actor'])
                agent.critic.load_state_dict(parent_state['critic'])
                
                parent_performance = parent_state.get('performance', 0)
                buffer_stats = parent_state.get('buffer_stats', {})
                success_sequences = buffer_stats.get('success_count', 0)
                parent_agent_idx = parent_state.get('agent_idx', 'unknown')
                
                print(f"   ✅ Agent {agent_idx+1} evolved from parent Agent{parent_agent_idx+1} (performance: {parent_performance:.2f}, learned from {success_sequences} successful sequences)")
                print(f"       🧬 Diverse evolution: Using parent {parent_idx+1}/{len(self.successful_agent_pool)} from success pool")
            except Exception as e:
                print(f"   ⚠️ Evolution failed for agent {agent_idx+1}: {e}")
                print(f"   🔄 Falling back to best agent parent")
                # Fallback to best agent if available
                if self.best_agent_state is not None:
                    try:
                        agent.world_model.load_state_dict(self.best_agent_state['world_model'])
                        agent.actor.load_state_dict(self.best_agent_state['actor'])
                        agent.critic.load_state_dict(self.best_agent_state['critic'])
                    except:
                        print(f"   🔄 Best agent fallback also failed - using fresh agent")
        elif self.best_agent_state is not None:
            # Fallback to single best agent if pool is empty
            try:
                agent.world_model.load_state_dict(self.best_agent_state['world_model'])
                agent.actor.load_state_dict(self.best_agent_state['actor'])
                agent.critic.load_state_dict(self.best_agent_state['critic'])
                
                best_performance = self.best_agent_state.get('performance', self.best_agent_performance)
                buffer_stats = self.best_agent_state.get('buffer_stats', {})
                success_sequences = buffer_stats.get('success_count', 0)
                
                print(f"   ✅ Agent {agent_idx+1} evolved from best parent (performance: {best_performance:.2f}, learned from {success_sequences} successful sequences)")
                print(f"       🧠 Success-prioritized learning: Agent inherits knowledge of successful navigation behaviors")
            except Exception as e:
                print(f"   ⚠️ Evolution failed for agent {agent_idx+1}: {e}")
                print(f"   🔄 Falling back to fresh agent")
        else:
            print(f"   🌱 Agent {agent_idx+1} created fresh (no parent available)")
        
        return agent
    
    def create_historical_agent(self, env: SimpleDroneEnv, agent_idx: int, strategy_name: str, success_rate: float) -> DreamerV3Agent:
        """Create agent from historical high-performing pool for intelligent diversity"""
        agent = DreamerV3Agent(
            obs_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            device=self.device
        )
        
        if self.historical_agent_pool:
            # Select historical agent based on strategy
            if strategy_name == "exploration":
                # During exploration, prefer diverse agents from different iterations
                available_iterations = set(h['iteration'] for h in self.historical_agent_pool)
                if len(available_iterations) > 1:
                    # Select from older iterations for more diversity
                    older_agents = [h for h in self.historical_agent_pool if h['iteration'] < max(available_iterations)]
                    if older_agents:
                        historical_state = older_agents[agent_idx % len(older_agents)]
                    else:
                        historical_state = self.historical_agent_pool[agent_idx % len(self.historical_agent_pool)]
                else:
                    historical_state = self.historical_agent_pool[agent_idx % len(self.historical_agent_pool)]
            else:
                # During exploitation, prefer recent high-performers
                historical_state = self.historical_agent_pool[0]  # Best historical agent
            
            try:
                agent.world_model.load_state_dict(historical_state['world_model'])
                agent.actor.load_state_dict(historical_state['actor'])
                agent.critic.load_state_dict(historical_state['critic'])
                
                hist_performance = historical_state.get('performance', 0)
                hist_iteration = historical_state.get('iteration', 'unknown')
                hist_distance = historical_state.get('best_distance', 'unknown')
                hist_success_rate = historical_state.get('success_rate', 0)
                
                print(f"   🔄 Agent {agent_idx+1} inherited from historical Agent{historical_state['agent_idx']+1} (Iteration {hist_iteration})")
                print(f"       📊 Historical performance: score={hist_performance:.1f}, distance={hist_distance:.2f}m, success_rate={hist_success_rate:.1%}")
                print(f"       🎯 Strategy: {strategy_name} diversity with goal-oriented foundation (current success: {success_rate:.1%})")
                
            except Exception as e:
                print(f"   ⚠️ Historical agent loading failed for agent {agent_idx+1}: {e}")
                print(f"   🔄 Falling back to fresh agent")
        else:
            print(f"   🌱 Agent {agent_idx+1} created fresh (no historical pool available yet)")
        
        return agent
    
    def visualize_iteration(self, iteration_result: Dict[str, Any]) -> str:
        """Create visualization for iteration results"""
        try:
            # Create scenario info for visualization
            scenario_info = {
                'arena_size': self.shared_scenario['arena_size'],
                'start_position': self.shared_scenario['start_position'],
                'goal_position': self.shared_scenario['goal_position'],
                'obstacle_positions': self.shared_scenario['obstacle_positions'],
                'obstacle_radii': self.shared_scenario['obstacle_radii']
            }
            
            # Episodes list with trajectory data - prioritize successful paths
            episodes = []
            for i, result in enumerate(iteration_result['agent_results']):
                # Show the BEST trajectory for each agent (successful path if available, otherwise final attempt)
                best_traj = result.get('best_trajectory')
                if best_traj:
                    episodes.append({
                        'trajectory': best_traj,
                        'strategy': f"Agent{i+1}_Best",
                        'success': result['success'],
                        'total_reward': result.get('total_reward', 0),
                        'agent_type': f"DreamerV3_Agent{i+1}",
                        'path_type': 'successful' if result['success'] else 'best_attempt'
                    })
                
                # Optionally show ALL trajectories for detailed analysis (commented out to reduce clutter)
                # if 'all_trajectories' in result:
                #     for j, traj in enumerate(result['all_trajectories']):
                #         episodes.append({
                #             'trajectory': traj,
                #             'strategy': f"Agent{i+1}_Ep{j+1}",
                #             'success': False,  # Mark individual episodes as failed for clarity
                #             'total_reward': 0,
                #             'agent_type': f"Episode_{j+1}"
                #         })
            
            # Add pathfinder trajectory (with 0 reward since it's not trained)
            if self.shared_scenario:
                start_pos = self.shared_scenario['start_position']
                goal_pos = self.shared_scenario['goal_position']
                optimal_path = self.pathfinder.get_optimal_path(tuple(start_pos), tuple(goal_pos))
                if optimal_path:
                    episodes.append({
                        'trajectory': optimal_path,
                        'strategy': "Optimal",
                        'success': True,
                        'total_reward': 0,
                        'agent_type': "Optimal"
                    })
            
            # Create visualization
            save_path = f"improved_dreamerv3_iteration_{iteration_result['iteration']}.png"
            
            plot_multiple_episodes(
                episodes,
                scenario_info,
                save_path,
                title=f"🧬 DreamerV3 Enhanced Training - Iteration {iteration_result['iteration']}",
                historical_rewards=iteration_result.get('historical_rewards', {})  # Pass historical data
            )
            
            return save_path
            
        except Exception as e:
            print(f"❌ Visualization failed: {e}")
            return ""
    
    def save_checkpoint(self, iteration_num: int, iteration_result: Dict[str, Any]):
        """Save training checkpoint"""
        checkpoint = {
            'iteration': iteration_num,
            'result': iteration_result,
            'best_agent_state': self.best_agent_state,
            'best_agent_performance': self.best_agent_performance,
            'consecutive_successes': self.consecutive_successes,
            'shared_scenario': self.shared_scenario
        }
        
        filename = f"improved_dreamerv3_checkpoint_iter_{iteration_num}.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(checkpoint, f)
    
    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """Load training checkpoint"""
        try:
            with open(checkpoint_path, 'rb') as f:
                checkpoint = pickle.load(f)
            
            self.best_agent_state = checkpoint.get('best_agent_state')
            self.best_agent_performance = checkpoint.get('best_agent_performance', float('inf'))
            self.consecutive_successes = checkpoint.get('consecutive_successes', 0)
            self.shared_scenario = checkpoint.get('shared_scenario')
            
            print(f"✅ Loaded checkpoint: best performance {self.best_agent_performance:.2f}m, consecutive successes: {self.consecutive_successes}")
            return True
        except Exception as e:
            print(f"❌ Failed to load checkpoint: {e}")
            return False
    
    def run_evolution(self, max_iterations: int = 50, consecutive_success_target: int = 5, target_success_rate: float = 0.9):
        """Run complete evolution until target success rate is achieved"""
        print(f"\n🧬 ENHANCED DreamerV3 Evolution System with Boundary-Aware Quality Learning")
        print("=" * 80)
        print(f"🎯 Goal: Achieve {target_success_rate:.0%} success rate over {consecutive_success_target} consecutive iterations")
        print(f"🔧 Enhancement: Boundary-aware reward system prevents ceiling/ground seeking")
        print(f"📈 Success Metric: {self.target_successes}/{self.agents_per_iteration} agents per iteration")
        print(f"🎯 Key Innovation: Agents learn preferentially from HIGH-QUALITY successful navigation")
        print(f"   • {self.success_sample_ratio*100:.0f}% of training batch samples from successful obstacle navigation")
        print(f"   • Quality-weighted sampling prioritizes efficient, smooth navigation")
        print(f"   • Buffer ranks both successes and failures by navigation quality")
        print(f"   • Boundary-aware penalties prevent ceiling/ground exploitation")
        print(f"   • Enables future counterfactual reasoning: 'What if I navigated more efficiently?'")
        
        # Track success rates over recent iterations
        recent_success_rates = []
        
        for iteration in range(1, max_iterations + 1):
            result = self.run_iteration(iteration)
            
            # Add current success rate to recent tracking
            current_success_rate = result['success_rate'] / 100.0  # Convert percentage to decimal for averaging
            recent_success_rates.append(current_success_rate)
            
            # Keep only the last N iterations for rolling average
            if len(recent_success_rates) > consecutive_success_target:
                recent_success_rates.pop(0)
            
            # Check if we have enough iterations to evaluate
            if len(recent_success_rates) >= consecutive_success_target:
                avg_success_rate = np.mean(recent_success_rates)
                
                print(f"\n📊 Rolling Success Rate Analysis ({consecutive_success_target} iterations):")
                print(f"   • Recent rates: {[f'{rate:.1%}' for rate in recent_success_rates[-consecutive_success_target:]]}")
                print(f"   • Average: {avg_success_rate:.1%} (target: {target_success_rate:.1%})")
                
                if avg_success_rate >= target_success_rate:
                    print(f"\n🎉 SUCCESS! Achieved {avg_success_rate:.1%} success rate over {consecutive_success_target} consecutive iterations!")
                    print(f"🏆 Training completed in {iteration} iterations")
                    
                    # Show final quality summary
                    final_metrics = result.get('quality_metrics', {})
                    print(f"\n📈 FINAL PERFORMANCE METRICS:")
                    print(f"   • Final success rate: {result['success_rate']:.1f}%")
                    print(f"   • Average success rate ({consecutive_success_target} iterations): {avg_success_rate:.1%}")
                    print(f"   • Final navigation efficiency: {final_metrics.get('navigation_efficiency', 0):.2f}")
                    print(f"   • Average successful reward: {result.get('avg_successful_reward', 0):.1f}")
                    print(f"   • Best distance achieved: {result.get('best_distance', 0):.2f}m")
                    
                    # Show learning progression
                    if len(self.learning_stats['success_rates']) >= 2:
                        initial_rate = self.learning_stats['success_rates'][0]
                        final_rate = result['success_rate']
                        improvement = ((final_rate - initial_rate) / max(initial_rate, 1)) * 100
                        print(f"   • Learning improvement: {initial_rate:.1f}% → {final_rate:.1f}% ({improvement:+.1f}%)")
                    
                    break
            else:
                remaining_iterations = consecutive_success_target - len(recent_success_rates)
                print(f"   📋 Need {remaining_iterations} more iterations to evaluate {target_success_rate:.0%} success rate target")
        
        else:
            print(f"\n⚠️ Training completed {max_iterations} iterations without reaching {target_success_rate:.0%} success rate")
            if recent_success_rates:
                final_avg = np.mean(recent_success_rates[-consecutive_success_target:])
                print(f"   Final average success rate: {final_avg:.1%}")
            
        return self.learning_stats


def main():
    """Main training function with enhanced boundary-aware learning"""
    
    # Configuration
    max_iterations = 50  # Maximum iterations if target not reached
    consecutive_target = 3  # Number of consecutive iterations to evaluate
    target_success_rate = 0.9  # 90% success rate target
    
    # Create trainer
    trainer = ImprovedDreamerV3Trainer(
        agents_per_iteration=12,
        target_successes=6,  # 6/12 = 50% success rate per iteration minimum
        max_episode_steps=1000,
        training_episodes_per_agent=20
    )
    
    print(f"🤖 Using device: {trainer.device}")
    print(f"🎯 Training Goal: {target_success_rate:.0%} success rate over {consecutive_target} consecutive iterations")
    print(f"📊 Evaluation: Average of {consecutive_target} most recent iterations")
    print(f"🔧 Enhanced: Boundary-aware reward system prevents ceiling/ground seeking")
    
    # Run evolution with new success criteria
    try:
        learning_stats = trainer.run_evolution(
            max_iterations=max_iterations,
            consecutive_success_target=consecutive_target,
            target_success_rate=target_success_rate
        )
        
        print(f"\n🎯 Training completed! Final statistics:")
        if learning_stats['success_rates']:
            print(f"   📈 Success rate progression: {learning_stats['success_rates'][:5]}...{learning_stats['success_rates'][-3:]}")
            print(f"   🎯 Final success rate: {learning_stats['success_rates'][-1]:.1f}%")
            
            # Calculate if target was achieved
            final_rates = learning_stats['success_rates'][-consecutive_target:]
            if len(final_rates) >= consecutive_target:
                final_avg = np.mean(final_rates)
                print(f"   🏆 Final {consecutive_target}-iteration average: {final_avg:.1f}%")
                if final_avg >= target_success_rate * 100:
                    print(f"   ✅ TARGET ACHIEVED: {final_avg:.1f}% ≥ {target_success_rate:.0%}")
                else:
                    print(f"   ❌ Target missed: {final_avg:.1f}% < {target_success_rate:.0%}")
        
    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()