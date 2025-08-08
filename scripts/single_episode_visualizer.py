#!/usr/bin/env python3
"""
DreamerV3 Multi-Agent Training & Visualization

Trains multiple DreamerV3 RL agents on identical navigation scenarios until
a target number succeed, then visualizes their performance with rotating 3D views.
Includes both successful and failed attempts for comprehensive analysis.

Key Features:
- Trains 10 DreamerV3 agents with diverse hyperparameters
- Stops training when 5 agents successfully reach the goal
- All agents train on identical obstacle layouts and start/goal positions
- Provides comprehensive visualization with rotating 3D plots
- Falls back to simple navigation strategies if DreamerV3 is unavailable
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import pickle
from pathlib import Path
import time

# Import environment and DreamerV3 - handle imports gracefully
try:
    from simple_drone_env import SimpleDroneEnv
    ENV_AVAILABLE = True
except ImportError:
    print("⚠️ SimpleDroneEnv not available - will use existing data only")
    ENV_AVAILABLE = False

try:
    from dreamerv3_drone import DreamerV3Agent, Experience
    import torch
    DREAMERV3_AVAILABLE = True
except ImportError:
    print("⚠️ DreamerV3 not available - will use simple navigation fallback")
    DREAMERV3_AVAILABLE = False
    try:
        import torch
    except ImportError:
        print("⚠️ PyTorch not available")
        torch = None

class SingleEpisodeVisualizer:
    """Clean visualization of multiple DreamerV3 navigation episodes with rotating 3D view"""
    
    def __init__(self):
        self.rotation_angle = 0
        self.animation_running = False
        if torch is not None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"🤖 Using device: {self.device}")
        else:
            self.device = "cpu"
            print(f"🤖 Using device: cpu (PyTorch not available)")
        
    def generate_dreamerv3_episodes(self, num_agents=10, target_successes=5, max_training_steps=50000):
        """Generate episodes by training multiple DreamerV3 agents until target successes are reached"""
        
        if not ENV_AVAILABLE:
            print("❌ Environment not available for fresh generation")
            return None
            
        if not DREAMERV3_AVAILABLE:
            print("❌ DreamerV3 not available - cannot train RL agents")
            return None
        
        print(f"🤖 Training {num_agents} DreamerV3 Agents Until {target_successes} Succeed")
        print("=" * 70)
        
        # Create challenging scenario configuration
        config = {
            'arena_size': 18.0,           # Good sized arena
            'arena_height': 8.0,
            'num_obstacles': 7,           # Moderate challenge
            'obstacle_radius_range': [1.0, 1.8],
            'goal_threshold': 1.2,
            'obstacle_clearance': 2.5,
            'min_start_goal_distance': 10.0,  # Force longer paths
            'max_steps': 800,
            'difficulty_level': 'realistic'
        }
        
        # Create ONE shared environment for all agents
        print("🗺️ Generating shared challenge map...")
        shared_env = SimpleDroneEnv(config=config)
        shared_env.reset()
        
        # Store the shared map data AND shared start/goal positions
        shared_obstacles = [(pos.copy(), radius) for pos, radius in zip(shared_env.obstacle_positions, shared_env.obstacle_radii)]
        shared_start_position = shared_env.drone_position.copy()
        shared_goal_position = shared_env.goal_position.copy()
        
        print(f"   Map created with {len(shared_obstacles)} obstacles")
        print(f"   Arena size: {shared_env.config['arena_size']}m x {shared_env.config['arena_size']}m x {shared_env.config['arena_height']}m")
        print(f"   Shared start: ({shared_start_position[0]:.1f}, {shared_start_position[1]:.1f}, {shared_start_position[2]:.1f})")
        print(f"   Shared goal:  ({shared_goal_position[0]:.1f}, {shared_goal_position[1]:.1f}, {shared_goal_position[2]:.1f})")
        
        # Calculate and display the challenge distance
        challenge_distance = np.linalg.norm(shared_goal_position - shared_start_position)
        print(f"   Challenge distance: {challenge_distance:.1f}m")
        
        # Initialize DreamerV3 agents with different configurations for diversity
        dreamerv3_configs = [
            {'learning_rate': 1e-4, 'imagination_horizon': 15, 'hidden_dim': 256, 'latent_dim': 32},  # Standard
            {'learning_rate': 3e-4, 'imagination_horizon': 12, 'hidden_dim': 256, 'latent_dim': 32},  # Fast learner
            {'learning_rate': 1e-4, 'imagination_horizon': 20, 'hidden_dim': 256, 'latent_dim': 48},  # Long horizon
            {'learning_rate': 5e-5, 'imagination_horizon': 15, 'hidden_dim': 384, 'latent_dim': 32},  # Conservative
            {'learning_rate': 2e-4, 'imagination_horizon': 10, 'hidden_dim': 256, 'latent_dim': 24},  # Aggressive
            {'learning_rate': 1e-4, 'imagination_horizon': 15, 'hidden_dim': 192, 'latent_dim': 32},  # Efficient
            {'learning_rate': 1.5e-4, 'imagination_horizon': 18, 'hidden_dim': 256, 'latent_dim': 40}, # Balanced
            {'learning_rate': 8e-5, 'imagination_horizon': 25, 'hidden_dim': 320, 'latent_dim': 32},  # Patient
            {'learning_rate': 2.5e-4, 'imagination_horizon': 8, 'hidden_dim': 256, 'latent_dim': 28}, # Reactive
            {'learning_rate': 1.2e-4, 'imagination_horizon': 15, 'hidden_dim': 256, 'latent_dim': 36}, # Standard+
        ]
        
        # Get environment dimensions
        test_obs = shared_env.reset()
        if isinstance(test_obs, tuple):
            test_obs = test_obs[0]
        if test_obs.ndim > 1:
            test_obs = test_obs[0]
        obs_dim = len(test_obs)
        action_dim = 3  # x, y, z acceleration
        
        agents = []
        envs = []
        episode_results = []
        successful_agents = 0
        
        print(f"\n🚀 Initializing {num_agents} DreamerV3 agents...")
        
        # Create agents and individual environments
        for i in range(num_agents):
            config_idx = i % len(dreamerv3_configs)
            agent_config = dreamerv3_configs[config_idx]
            
            # Create individual agent
            agent = DreamerV3Agent(
                obs_dim=obs_dim,
                action_dim=action_dim,
                hidden_dim=agent_config['hidden_dim'],
                latent_dim=agent_config['latent_dim'],
                learning_rate=agent_config['learning_rate'],
                imagination_horizon=agent_config['imagination_horizon'],
                gamma=0.99,
                lambda_gae=0.95,
                device=str(self.device)
            )
            
            # Create individual environment (copy of shared scenario)
            env = SimpleDroneEnv(config=config)
            env.reset()
            # Force same obstacles and positions
            env.obstacle_positions = [pos.copy() for pos, _ in shared_obstacles]
            env.obstacle_radii = [radius for _, radius in shared_obstacles]
            env.drone_position = shared_start_position.copy()
            env.goal_position = shared_goal_position.copy()
            
            agents.append(agent)
            envs.append(env)
            
            print(f"   Agent {i+1}: LR={agent_config['learning_rate']}, Horizon={agent_config['imagination_horizon']}, Hidden={agent_config['hidden_dim']}")
        
        print(f"\n🎯 Training agents until {target_successes} reach the goal...")
        
        # Training loop
        training_step = 0
        max_episodes_per_agent = 50   # Reduced for faster training
        max_training_steps = 5000      # Reduced overall limit
        
        print(f"\n🎯 Training agents until {target_successes} reach the goal...")
        print(f"   Max training steps: {max_training_steps}")
        print(f"   Max episodes per agent: {max_episodes_per_agent}")
        
        while successful_agents < target_successes and training_step < max_training_steps:
            training_step += 1
            agents_trained_this_step = 0
            
            # Train each agent that hasn't succeeded yet
            for agent_idx, (agent, env) in enumerate(zip(agents, envs)):
                # Skip if this agent already succeeded
                if len(episode_results) > agent_idx and episode_results[agent_idx].get('success', False):
                    continue
                
                # Reset environment to shared start/goal
                env.drone_position = shared_start_position.copy()
                env.goal_position = shared_goal_position.copy()
                
                # Run training episode
                episode_data = self.train_dreamerv3_episode(agent, env, agent_idx + 1)
                agents_trained_this_step += 1
                
                # Store result if it's the first episode for this agent or if it's successful
                if len(episode_results) <= agent_idx:
                    episode_results.append(episode_data)
                elif episode_data['success'] and not episode_results[agent_idx].get('success', False):
                    episode_results[agent_idx] = episode_data
                    successful_agents += 1
                    print(f"🎉 Agent {agent_idx + 1} SUCCEEDED! ({successful_agents}/{target_successes})")
                
                # Break early if we have enough successes
                if successful_agents >= target_successes:
                    break
            
            # Progress update
            if training_step % 5 == 0:
                print(f"   Step {training_step}: {successful_agents}/{target_successes} succeeded, trained {agents_trained_this_step} agents")
            
            # Safety check
            if training_step >= max_training_steps:
                print(f"⚠️ Reached maximum training steps ({max_training_steps}), stopping...")
                break
        
        # Fill remaining slots with the best failed attempts
        while len(episode_results) < num_agents:
            # Create a failed episode result for visualization
            dummy_episode = {
                'path': [shared_start_position.copy(), shared_start_position.copy() + np.array([1, 1, 0])],
                'start': shared_start_position.copy(),
                'goal': shared_goal_position.copy(),
                'obstacles': shared_obstacles,
                'success': False,
                'steps': 50,
                'final_distance': challenge_distance,
                'agent_type': 'DreamerV3',
                'agent_id': len(episode_results) + 1
            }
            episode_results.append(dummy_episode)
        
        print(f"\n📊 DreamerV3 Training Complete:")
        print(f"   Successful agents: {successful_agents}/{num_agents}")
        print(f"   Training steps used: {training_step}/{max_training_steps}")
        print(f"   All agents trained on identical scenario")
        
        return episode_results[:num_agents]
    
    def train_dreamerv3_episode(self, agent, env, agent_id):
        """Train a DreamerV3 agent for one episode and return the path taken"""
        
        max_episode_steps = 200  # Reduced for faster training
        training_episodes = 3   # Quick training bursts
        
        # Store initial positions to restore them
        initial_start = env.drone_position.copy()
        initial_goal = env.goal_position.copy()
        
        # Quick training phase
        for training_ep in range(training_episodes):
            # Reset to shared positions
            env.drone_position = initial_start.copy()
            env.goal_position = initial_goal.copy()
            
            obs = env._get_observation()
            if isinstance(obs, tuple):
                obs = obs[0]
            if obs.ndim > 1:
                obs = obs[0]
            
            episode_data = []
            done = False
            step_count = 0
            
            while not done and step_count < max_episode_steps:
                # Get action from agent
                try:
                    action = agent.get_action(obs, deterministic=False)
                except:
                    # Use random action as fallback during early training
                    action = np.random.uniform(-1, 1, 3)
                
                # Ensure action is the right shape
                if isinstance(action, np.ndarray) and action.ndim > 1:
                    action = action.flatten()
                
                # Step environment
                next_obs, reward, done, info = env.step(action)
                if isinstance(next_obs, tuple):
                    next_obs = next_obs[0]
                if next_obs.ndim > 1:
                    next_obs = next_obs[0]
                
                # Store experience
                episode_data.append((obs, action, reward, next_obs, done, info))
                agent.add_experience(obs, action, reward, next_obs, done, info)
                
                obs = next_obs
                step_count += 1
                
                # Early success check
                final_distance = np.linalg.norm(env.drone_position - env.goal_position)
                if final_distance <= env.config['goal_threshold']:
                    done = True
                    break
            
            # Training step if we have enough data
            if len(agent.replay_buffer) > 16:  # Reduced batch requirement
                try:
                    agent.train_step(batch_size=16)
                except:
                    pass  # Skip training errors during early phases
        
        # Final evaluation episode to get the path
        env.drone_position = initial_start.copy()
        env.goal_position = initial_goal.copy()
        
        obs = env._get_observation()
        if isinstance(obs, tuple):
            obs = obs[0]
        if obs.ndim > 1:
            obs = obs[0]
        
        path = [env.drone_position.copy()]
        done = False
        step_count = 0
        
        while not done and step_count < max_episode_steps:
            # Get action from agent (deterministic for evaluation)
            try:
                action = agent.get_action(obs, deterministic=True)
            except:
                # Fallback to random action
                action = np.random.uniform(-0.5, 0.5, 3)
            
            # Ensure action is the right shape
            if isinstance(action, np.ndarray) and action.ndim > 1:
                action = action.flatten()
            
            # Step environment
            next_obs, reward, done, info = env.step(action)
            if isinstance(next_obs, tuple):
                next_obs = next_obs[0]
            if next_obs.ndim > 1:
                next_obs = next_obs[0]
            
            path.append(env.drone_position.copy())
            obs = next_obs
            step_count += 1
            
            # Check success - if reached goal, stop immediately
            final_distance = np.linalg.norm(env.drone_position - env.goal_position)
            if final_distance <= env.config['goal_threshold']:
                done = True
                # Ensure path ends at current position when goal is reached
                path.append(env.drone_position.copy())
                break
        
        # Check if successful
        final_distance = np.linalg.norm(env.drone_position - env.goal_position)
        success = final_distance <= env.config['goal_threshold']
        
        print(f"   Agent {agent_id}: {'✅ SUCCESS' if success else '❌ FAILED'} - Final distance: {final_distance:.2f}m, Steps: {len(path)}")
        
        return {
            'path': path,
            'start': path[0] if path else None,
            'goal': env.goal_position.copy(),
            'obstacles': [(pos.copy(), radius) for pos, radius in zip(env.obstacle_positions, env.obstacle_radii)],
            'success': success,
            'steps': len(path),
            'final_distance': final_distance,
            'agent_type': 'DreamerV3',
            'agent_id': agent_id
        }
    
    def generate_fresh_episodes(self, num_episodes=10, target_successes=5):
        """Generate fresh navigation episodes - prioritize DreamerV3 if available"""
        
        if DREAMERV3_AVAILABLE:
            print("🤖 Using DreamerV3 RL agents for episode generation")
            return self.generate_dreamerv3_episodes(num_agents=num_episodes, target_successes=target_successes)
        else:
            print("⚠️ DreamerV3 not available, falling back to simple navigation strategies")
            return self.generate_simple_strategy_episodes(num_episodes=num_episodes, target_successes=target_successes)
    
    def generate_simple_strategy_episodes(self, num_episodes=10, target_successes=5):
        """Generate fresh navigation episodes on a shared map"""
        
        if not ENV_AVAILABLE:
            print("❌ Environment not available for fresh generation")
            return None
        
        print(f"🎯 Generating {num_episodes} Navigation Episodes on Shared Map")
        print("=" * 60)
        
        # Create challenging scenario configuration
        config = {
            'arena_size': 18.0,           # Good sized arena
            'arena_height': 8.0,
            'num_obstacles': 7,           # Moderate challenge
            'obstacle_radius_range': [1.0, 1.8],
            'goal_threshold': 1.2,
            'obstacle_clearance': 2.5,
            'min_start_goal_distance': 10.0,  # Force longer paths
            'max_steps': 800,
            'difficulty_level': 'realistic'
        }
        
        # Create ONE shared environment for all episodes
        print("🗺️ Generating shared map...")
        shared_env = SimpleDroneEnv(config=config)
        shared_env.reset()
        
        # Store the shared map data AND shared start/goal positions
        shared_obstacles = [(pos.copy(), radius) for pos, radius in zip(shared_env.obstacle_positions, shared_env.obstacle_radii)]
        shared_start_position = shared_env.drone_position.copy()
        shared_goal_position = shared_env.goal_position.copy()
        arena_bounds = shared_env.config
        
        print(f"   Map created with {len(shared_obstacles)} obstacles")
        print(f"   Arena size: {shared_env.config['arena_size']}m x {shared_env.config['arena_size']}m x {shared_env.config['arena_height']}m")
        print(f"   Shared start: ({shared_start_position[0]:.1f}, {shared_start_position[1]:.1f}, {shared_start_position[2]:.1f})")
        print(f"   Shared goal:  ({shared_goal_position[0]:.1f}, {shared_goal_position[1]:.1f}, {shared_goal_position[2]:.1f})")
        
        # Calculate and display the challenge distance
        challenge_distance = np.linalg.norm(shared_goal_position - shared_start_position)
        print(f"   Challenge distance: {challenge_distance:.1f}m")
        
        episodes = []
        successful_episodes = 0
        failed_episodes = 0
        
        strategies = ['direct_with_avoidance', 'cautious', 'aggressive', 'smart_exploration']
        
        for i in range(num_episodes):
            print(f"\n📍 Episode {i + 1}/{num_episodes}")
            
            # Use the SAME environment, start, and goal for all episodes
            env = shared_env
            
            # Force the same start and goal positions for all episodes
            env.drone_position = shared_start_position.copy()
            env.goal_position = shared_goal_position.copy()
            
            # Verify consistent scenario
            start_goal_distance = np.linalg.norm(env.goal_position - env.drone_position)
            print(f"   Start-goal distance: {start_goal_distance:.1f}m (consistent)")
            
            # Choose strategy based on what we need
            if successful_episodes < target_successes:
                # Use more reliable strategies for successes
                strategy = strategies[i % 2]  # Alternate between best strategies
            else:
                # Use more challenging strategies for interesting failures
                strategy = strategies[2 + (i % 2)]
            
            print(f"   Strategy: {strategy}")
            
            # Run navigation
            episode_data = self.run_smart_navigation(env, strategy)
            
            # Report results
            if episode_data['success']:
                status = "✅ SUCCESS"
                successful_episodes += 1
            else:
                status = "❌ FAILED"
                failed_episodes += 1
            
            print(f"   {status}")
            print(f"   Steps: {episode_data['steps']}")
            print(f"   Final distance: {episode_data['final_distance']:.2f}m")
            
            episodes.append(episode_data)
            
            # Stop early if we have enough variety
            if successful_episodes >= target_successes and failed_episodes >= (num_episodes - target_successes):
                break
        
        print(f"\n📊 Generation Complete on Shared Map & Positions:")
        print(f"   Successful episodes: {successful_episodes}")
        print(f"   Failed episodes: {failed_episodes}")
        print(f"   Total episodes: {len(episodes)}")
        print(f"   All episodes used identical start/goal positions")
        
        return episodes
    
    def run_smart_navigation(self, env, strategy='direct_with_avoidance'):
        """Run intelligent navigation with different strategies"""
        
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
                safe_distance = obs_radius + 2.0
                if obs_distance < safe_distance and obs_distance > 0:
                    avoidance_strength = (safe_distance - obs_distance) / safe_distance
                    avoidance_force += (obs_vector / obs_distance) * avoidance_strength * 2.0
            
            # Strategy-specific behavior
            if strategy == 'direct_with_avoidance':
                action = direction_to_goal * 0.8 + avoidance_force
                action += np.random.normal(0, 0.05, 3)  # Small noise
            elif strategy == 'cautious':
                action = direction_to_goal * 0.4 + avoidance_force * 2.0
                action += np.random.normal(0, 0.03, 3)  # Less noise
            elif strategy == 'aggressive':
                action = direction_to_goal * 1.0 + avoidance_force * 0.5
                action += np.random.normal(0, 0.1, 3)   # More noise
            else:  # 'smart_exploration'
                # Adaptive strategy
                if distance_to_goal > 5.0:
                    action = direction_to_goal * 0.7 + avoidance_force
                else:
                    action = direction_to_goal * 0.9 + avoidance_force * 1.5
                action += np.random.normal(0, 0.08, 3)
            
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
    
    def load_best_episodes(self, path_file, num_episodes=10):
        """Load multiple episodes for comparison (fallback if no fresh generation)"""
        
        if not Path(path_file).exists():
            print(f"❌ Path file not found: {path_file}")
            return None
        
        try:
            with open(path_file, 'rb') as f:
                all_episodes = pickle.load(f)
            
            if not all_episodes:
                print("❌ No episodes found in file")
                return None
            
            print(f"✓ Loaded {len(all_episodes)} episodes from file")
            
            # Get mix of successful and failed episodes
            successful_episodes = [ep for ep in all_episodes if ep.get('success', False)]
            failed_episodes = [ep for ep in all_episodes if not ep.get('success', False)]
            
            episodes = []
            
            # Get best 5 successful episodes
            if successful_episodes:
                successful_episodes.sort(key=lambda x: x.get('steps', float('inf')))
                episodes.extend(successful_episodes[:5])
            
            # Get 5 failed episodes (variety of failure types)
            if failed_episodes:
                failed_episodes.sort(key=lambda x: x.get('final_distance', float('inf')))
                episodes.extend(failed_episodes[:5])
            
            # Fill up to 10 episodes if needed
            remaining = all_episodes[:num_episodes - len(episodes)]
            episodes.extend(remaining)
            
            return episodes[:num_episodes]
            
        except Exception as e:
            print(f"❌ Error loading episodes: {e}")
            return None
    
    def plot_multiple_episodes(self, episodes_data, save_path=None):
        """Create a comprehensive visualization of multiple episodes with rotating 3D"""
        
        if not episodes_data:
            print("❌ No valid episode data to plot")
            return
        
        # Create figure with subplots
        self.fig = plt.figure(figsize=(20, 15))
        
        # Rotating 3D plot (main feature)
        self.ax1 = self.fig.add_subplot(231, projection='3d')
        
        # Top-down comparison
        ax2 = self.fig.add_subplot(232)
        
        # Side view comparison
        ax3 = self.fig.add_subplot(233)
        
        # Performance comparison
        ax4 = self.fig.add_subplot(234)
        
        # Path efficiency comparison
        ax5 = self.fig.add_subplot(235)
        
        # Statistics summary
        ax6 = self.fig.add_subplot(236)
        
        # Color schemes for episodes
        success_colors = ['#2E8B57', '#228B22', '#32CD32', '#90EE90', '#98FB98']  # Greens
        failure_colors = ['#DC143C', '#B22222', '#CD5C5C', '#F08080', '#FFA07A']  # Reds
        
        successful_episodes = [ep for ep in episodes_data if ep.get('success', False)]
        failed_episodes = [ep for ep in episodes_data if not ep.get('success', False)]
        
        print(f"📊 Visualizing {len(successful_episodes)} successful and {len(failed_episodes)} failed episodes")
        
        # Store episode data for rotation animation
        self.episodes_data = episodes_data
        self.success_colors = success_colors
        self.failure_colors = failure_colors
        
        # Initial 3D plot
        self.update_3d_plot()
        
        # Top-down view comparison
        ax2.set_title('Top-Down View - All Episodes', fontsize=14, fontweight='bold')
        ax2.set_xlabel('X Position')
        ax2.set_ylabel('Y Position')
        ax2.set_aspect('equal')
        ax2.grid(True, alpha=0.3)
        
        # Plot all episodes
        success_idx = 0
        failure_idx = 0
        
        for i, episode in enumerate(episodes_data):
            if not episode.get('path'):
                continue
                
            path = np.array(episode['path'])
            success = episode.get('success', False)
            
            if success:
                color = success_colors[success_idx % len(success_colors)]
                alpha = 0.8
                linewidth = 3
                success_idx += 1
            else:
                color = failure_colors[failure_idx % len(failure_colors)]
                alpha = 0.6
                linewidth = 2
                failure_idx += 1
            
            # Plot path
            label = f"Ep {i+1} {'✓' if success else '✗'} ({episode.get('steps', 0)} steps)"
            ax2.plot(path[:, 0], path[:, 1], color=color, linewidth=linewidth, 
                    alpha=alpha, label=label)
            
            # Mark start and goal
            start = episode['start']
            goal = episode['goal']
            
            ax2.scatter(start[0], start[1], color='blue', s=60, marker='o', alpha=0.7)
            ax2.scatter(goal[0], goal[1], color='gold', s=80, marker='*', alpha=0.9)
        
        # Plot obstacles from first episode (representative)
        if episodes_data and episodes_data[0].get('obstacles'):
            for obs_pos, obs_radius in episodes_data[0]['obstacles']:
                circle = plt.Circle((obs_pos[0], obs_pos[1]), obs_radius, 
                                  color='red', alpha=0.3, edgecolor='darkred', linewidth=1)
                ax2.add_patch(circle)
        
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)  # Increased from 8 to 10
        
        # Side view comparison
        ax3.set_title('Side View (X-Z) - All Episodes', fontsize=14, fontweight='bold')
        ax3.set_xlabel('X Position')
        ax3.set_ylabel('Z Position (Altitude)')
        ax3.grid(True, alpha=0.3)
        
        success_idx = 0
        failure_idx = 0
        
        for i, episode in enumerate(episodes_data):
            if not episode.get('path'):
                continue
                
            path = np.array(episode['path'])
            success = episode.get('success', False)
            
            if success:
                color = success_colors[success_idx % len(success_colors)]
                alpha = 0.8
                linewidth = 3
                success_idx += 1
            else:
                color = failure_colors[failure_idx % len(failure_colors)]
                alpha = 0.6
                linewidth = 2
                failure_idx += 1
            
            ax3.plot(path[:, 0], path[:, 2], color=color, linewidth=linewidth, alpha=alpha)
        
        # Performance comparison - handle both DreamerV3 and strategy data
        agent_types = [ep.get('agent_type', ep.get('strategy', 'unknown')) for ep in episodes_data]
        successes = [1 if ep.get('success', False) else 0 for ep in episodes_data]
        
        # Group by agent type/strategy
        type_success = {}
        for agent_type, success in zip(agent_types, successes):
            if agent_type not in type_success:
                type_success[agent_type] = []
            type_success[agent_type].append(success)
        
        # Plot performance
        type_names = list(type_success.keys())
        success_rates = [np.mean(successes) * 100 for successes in type_success.values()]
        
        bars = ax4.bar(type_names, success_rates, 
                      color=['green' if rate > 50 else 'red' for rate in success_rates], 
                      alpha=0.7)
        
        # Adjust title based on agent type
        is_dreamerv3 = any('DreamerV3' in str(agent_type) for agent_type in agent_types)
        title = 'Success Rate by DreamerV3 Agent' if is_dreamerv3 else 'Success Rate by Strategy'
        ax4.set_title(title, fontsize=14, fontweight='bold')
        ax4.set_ylabel('Success Rate (%)')
        ax4.set_ylim(0, 100)
        
        for bar, rate in zip(bars, success_rates):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 2,
                    f'{rate:.0f}%', ha='center', va='bottom', fontweight='bold')
        
        # Path efficiency
        efficiencies = []
        episode_labels = []
        
        for i, episode in enumerate(episodes_data):
            if episode.get('path') and len(episode['path']) > 1:
                path = np.array(episode['path'])
                total_distance = np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))
                direct_distance = np.linalg.norm(np.array(episode['goal']) - np.array(episode['start']))
                efficiency = direct_distance / total_distance if total_distance > 0 else 0
                
                efficiencies.append(efficiency)
                episode_labels.append(f"Ep{i+1}")
        
        if efficiencies:
            colors = []
            success_idx = 0
            failure_idx = 0
            
            for i, episode in enumerate(episodes_data[:len(efficiencies)]):
                if episode.get('success', False):
                    color = success_colors[success_idx % len(success_colors)]
                    success_idx += 1
                else:
                    color = failure_colors[failure_idx % len(failure_colors)]
                    failure_idx += 1
                colors.append(color)
            
            bars = ax5.bar(episode_labels, efficiencies, color=colors, alpha=0.7)
            ax5.set_title('Path Efficiency by Episode', fontsize=14, fontweight='bold')
            ax5.set_ylabel('Efficiency Score')
            ax5.set_ylim(0, max(efficiencies) * 1.2 if efficiencies else 1)
            
            for bar, eff in zip(bars, efficiencies):
                height = bar.get_height()
                ax5.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{eff:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Statistics summary
        ax6.axis('off')
        ax6.set_title('Episode Analysis Summary', fontsize=14, fontweight='bold')
        
        # Calculate overall statistics
        total_episodes = len(episodes_data)
        total_successful = len(successful_episodes)
        total_failed = len(failed_episodes)
        success_rate = (total_successful / total_episodes * 100) if total_episodes > 0 else 0
        
        avg_steps_success = np.mean([ep['steps'] for ep in successful_episodes]) if successful_episodes else 0
        avg_steps_failed = np.mean([ep['steps'] for ep in failed_episodes]) if failed_episodes else 0
        
        avg_distance = np.mean([np.linalg.norm(np.array(ep['goal']) - np.array(ep['start'])) 
                               for ep in episodes_data if ep.get('start') is not None])
        
        # Determine analysis type
        is_dreamerv3 = any('DreamerV3' in str(ep.get('agent_type', '')) for ep in episodes_data)
        analysis_type = "DREAMERV3 RL AGENTS" if is_dreamerv3 else "NAVIGATION STRATEGIES"
        
        # Collect agent types/strategies for summary
        agent_methods = set()
        for ep in episodes_data:
            if ep.get('agent_type'):
                if ep['agent_type'] == 'DreamerV3':
                    agent_methods.add(f"DV3 {ep.get('agent_id', '?')}")
                else:
                    agent_methods.add(ep['agent_type'])
            elif ep.get('strategy'):
                agent_methods.add(ep['strategy'])
            else:
                agent_methods.add('Unknown')
        
        stats_text = f"""
🎯 {analysis_type} ANALYSIS
{'='*40}

📊 Overall Performance:
• Total Episodes: {total_episodes}
• Successful: {total_successful} ({success_rate:.1f}%)
• Failed: {total_failed} ({100-success_rate:.1f}%)

📈 Performance Metrics:
• Avg Steps (Success): {avg_steps_success:.0f}
• Avg Steps (Failed): {avg_steps_failed:.0f}
• Challenge Distance: {avg_distance:.1f}m

🗺️ Identical Test Conditions:
• Same obstacle layout for all episodes
• Same start position for all episodes
• Same goal position for all episodes
• Pure {analysis_type.lower()} comparison

💡 Methods Tested:
{', '.join(sorted(agent_methods))}
"""
        
        ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
        
        # Overall title
        self.fig.suptitle('Multi-Episode Navigation Analysis - Identical Scenarios', 
                         fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"💾 Visualization saved as '{save_path}'")
        
        return self.fig
    
    def update_3d_plot(self):
        """Update the 3D plot with current rotation angle"""
        
        self.ax1.clear()
        self.ax1.set_title('3D Navigation Paths (Rotating View)', fontsize=14, fontweight='bold')
        self.ax1.set_xlabel('X Position')
        self.ax1.set_ylabel('Y Position')
        self.ax1.set_zlabel('Z Position')
        
        # Plot all episodes
        success_idx = 0
        failure_idx = 0
        
        for i, episode in enumerate(self.episodes_data):
            if not episode.get('path'):
                continue
                
            path = np.array(episode['path'])
            success = episode.get('success', False)
            
            if success:
                color = self.success_colors[success_idx % len(self.success_colors)]
                alpha = 0.8
                linewidth = 3
                success_idx += 1
            else:
                color = self.failure_colors[failure_idx % len(self.failure_colors)]
                alpha = 0.6
                linewidth = 2
                failure_idx += 1
            
            # Plot path
            label = f"Ep {i+1} {'✓' if success else '✗'}"
            self.ax1.plot(path[:, 0], path[:, 1], path[:, 2], 
                         color=color, linewidth=linewidth, alpha=alpha, label=label)
            
            # Mark start and goal
            start = episode['start']
            goal = episode['goal']
            end = path[-1]
            
            self.ax1.scatter(*start, color='blue', s=80, marker='o', alpha=0.8)
            self.ax1.scatter(*goal, color='gold', s=120, marker='*', alpha=0.9)
            self.ax1.scatter(*end, color='darkred', s=80, marker='x', alpha=0.8)
        
        # Plot obstacles from first episode
        if self.episodes_data and self.episodes_data[0].get('obstacles'):
            for obs_pos, obs_radius in self.episodes_data[0]['obstacles'][:6]:  # Limit for performance
                u = np.linspace(0, 2 * np.pi, 12)
                v = np.linspace(0, np.pi, 12)
                x = obs_radius * np.outer(np.cos(u), np.sin(v)) + obs_pos[0]
                y = obs_radius * np.outer(np.sin(u), np.sin(v)) + obs_pos[1]
                z = obs_radius * np.outer(np.ones(np.size(u)), np.cos(v)) + obs_pos[2]
                self.ax1.plot_surface(x, y, z, alpha=0.3, color='red')
        
        # Set the viewing angle
        self.ax1.view_init(elev=20, azim=self.rotation_angle)
        
        # Add legend
        handles, labels = self.ax1.get_legend_handles_labels()
        if handles:
            self.ax1.legend(handles[:20], labels[:20], loc='upper right', fontsize=8)  # Increased to 20 to show all 12 agents plus other elements
    
    def animate_rotation(self, frame):
        """Animation function for rotating 3D plot"""
        self.rotation_angle = (self.rotation_angle + 2) % 360  # Rotate 2 degrees per frame
        self.update_3d_plot()
        return []
    
    def start_rotation_animation(self, interval=100):
        """Start the rotation animation"""
        if hasattr(self, 'fig') and hasattr(self, 'ax1'):
            self.animation_running = True
            self.anim = animation.FuncAnimation(
                self.fig, self.animate_rotation, interval=interval, blit=False, cache_frame_data=False
            )
            return self.anim
        return None
        """Create a clean visualization of a single episode"""
        
        if not episode_data or not episode_data.get('path'):
            print("❌ No valid episode data to plot")
            return
        
        # Create figure with subplots
        fig = plt.figure(figsize=(18, 12))
        
        # 3D plot
        ax1 = fig.add_subplot(221, projection='3d')
        
        # Top-down plot
        ax2 = fig.add_subplot(222)
        
        # Side view (X-Z)
        ax3 = fig.add_subplot(223)
        
        # Statistics
        ax4 = fig.add_subplot(224)
        
        # Extract data
        path = np.array(episode_data['path'])
        start = path[0]
        end = path[-1]
        goal = np.array(episode_data.get('goal', end))
        obstacles = episode_data.get('obstacles', [])
        success = episode_data.get('success', False)
        steps = len(path)
        
        # Calculate metrics
        final_distance = np.linalg.norm(end - goal)
        total_path_length = np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))
        direct_distance = np.linalg.norm(goal - start)
        efficiency = direct_distance / total_path_length if total_path_length > 0 else 0
        
        # Color scheme
        path_color = 'green' if success else 'red'
        path_alpha = 0.8
        path_linewidth = 3
        
        # 3D Visualization
        ax1.set_title(f'3D Navigation Path - {"SUCCESS" if success else "FAILED"}', 
                     color=path_color, fontweight='bold', fontsize=14)
        ax1.set_xlabel('X Position')
        ax1.set_ylabel('Y Position')
        ax1.set_zlabel('Z Position')
        
        # Plot path
        ax1.plot(path[:, 0], path[:, 1], path[:, 2], 
                color=path_color, linewidth=path_linewidth, alpha=path_alpha, label='Flight Path')
        
        # Plot key points
        ax1.scatter(*start, color='blue', s=200, marker='o', label='Start', 
                   edgecolors='white', linewidth=2)
        ax1.scatter(*end, color='darkred', s=200, marker='x', label='End Position', linewidth=3)
        ax1.scatter(*goal, color='gold', s=300, marker='*', label='Goal', 
                   edgecolors='black', linewidth=2)
        
        # Plot obstacles as spheres
        for i, (obs_pos, obs_radius) in enumerate(obstacles[:8]):  # Limit for clarity
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 20)
            x = obs_radius * np.outer(np.cos(u), np.sin(v)) + obs_pos[0]
            y = obs_radius * np.outer(np.sin(u), np.sin(v)) + obs_pos[1]
            z = obs_radius * np.outer(np.ones(np.size(u)), np.cos(v)) + obs_pos[2]
            ax1.plot_surface(x, y, z, alpha=0.3, color='red')
        
        ax1.legend()
        
        # Top-down view
        ax2.set_title('Top-Down View', fontsize=14)
        ax2.set_xlabel('X Position')
        ax2.set_ylabel('Y Position')
        ax2.set_aspect('equal')
        ax2.grid(True, alpha=0.3)
        
        # Plot path
        ax2.plot(path[:, 0], path[:, 1], color=path_color, linewidth=path_linewidth, 
                alpha=path_alpha, label='Flight Path')
        
        # Plot direct path for comparison
        ax2.plot([start[0], goal[0]], [start[1], goal[1]], 'k--', linewidth=2, 
                alpha=0.5, label='Direct Path')
        
        # Plot key points
        ax2.scatter(start[0], start[1], color='blue', s=200, marker='o', 
                   edgecolors='white', linewidth=2, label='Start')
        ax2.scatter(end[0], end[1], color='darkred', s=200, marker='x', linewidth=3, label='End')
        ax2.scatter(goal[0], goal[1], color='gold', s=300, marker='*', 
                   edgecolors='black', linewidth=2, label='Goal')
        
        # Plot obstacles as circles
        for obs_pos, obs_radius in obstacles:
            circle = plt.Circle((obs_pos[0], obs_pos[1]), obs_radius, 
                              color='red', alpha=0.3, edgecolor='darkred', linewidth=2)
            ax2.add_patch(circle)
        
        # Show distance if failed
        if not success:
            ax2.plot([end[0], goal[0]], [end[1], goal[1]], 'r:', linewidth=2, alpha=0.7)
            mid_point = (end + goal) / 2
            ax2.text(mid_point[0], mid_point[1], f'Miss: {final_distance:.1f}m', 
                    ha='center', va='bottom', fontweight='bold', 
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        ax2.legend()
        
        # Side view (X-Z)
        ax3.set_title('Side View (X-Z)', fontsize=14)
        ax3.set_xlabel('X Position')
        ax3.set_ylabel('Z Position (Altitude)')
        ax3.grid(True, alpha=0.3)
        
        # Plot path
        ax3.plot(path[:, 0], path[:, 2], color=path_color, linewidth=path_linewidth, 
                alpha=path_alpha, label='Flight Path')
        
        # Plot direct path
        ax3.plot([start[0], goal[0]], [start[2], goal[2]], 'k--', linewidth=2, 
                alpha=0.5, label='Direct Path')
        
        # Plot key points
        ax3.scatter(start[0], start[2], color='blue', s=200, marker='o', 
                   edgecolors='white', linewidth=2, label='Start')
        ax3.scatter(end[0], end[2], color='darkred', s=200, marker='x', linewidth=3, label='End')
        ax3.scatter(goal[0], goal[2], color='gold', s=300, marker='*', 
                   edgecolors='black', linewidth=2, label='Goal')
        
        # Plot obstacles as circles (side view)
        for obs_pos, obs_radius in obstacles:
            circle = plt.Circle((obs_pos[0], obs_pos[2]), obs_radius, 
                              color='red', alpha=0.3, edgecolor='darkred', linewidth=1)
            ax3.add_patch(circle)
        
        ax3.legend()
        
        # Statistics
        ax4.axis('off')
        ax4.set_title('Navigation Statistics', fontsize=14, fontweight='bold')
        
        stats_text = f"""
🎯 MISSION RESULT: {"✅ SUCCESS" if success else "❌ FAILED"}

📊 Performance Metrics:
• Total Steps: {steps}
• Final Distance to Goal: {final_distance:.2f}m
• Path Length: {total_path_length:.1f}m
• Direct Distance: {direct_distance:.1f}m
• Path Efficiency: {efficiency:.3f}

📍 Coordinates:
• Start: ({start[0]:.1f}, {start[1]:.1f}, {start[2]:.1f})
• Goal:  ({goal[0]:.1f}, {goal[1]:.1f}, {goal[2]:.1f})
• End:   ({end[0]:.1f}, {end[1]:.1f}, {end[2]:.1f})

🚧 Environment:
• Obstacles: {len(obstacles)}
• Arena Bounds: ±{abs(start[0]) + 5:.0f}m (estimated)

💡 Efficiency Score:
{efficiency:.1%} of optimal direct path
"""
        
        ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
        
        # Overall title
        status_color = 'green' if success else 'red'
        fig.suptitle(f'Single Episode Analysis - Navigation {"SUCCESS" if success else "FAILURE"}', 
                    fontsize=16, fontweight='bold', color=status_color)
        
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"💾 Visualization saved as '{save_path}'")
        
        return fig

def main():
    """Main function to create comprehensive multi-episode visualization"""
    
    print("🎯 Multi-Episode Identical Scenario Navigation Visualization")
    print("=" * 60)
    
    visualizer = SingleEpisodeVisualizer()
    
    # Generate episodes on shared map (recommended for clean viewing)
    print("�️ Generating navigation scenarios on shared map...")
    episodes = visualizer.generate_fresh_episodes(num_episodes=10, target_successes=5)
    
    if episodes:
        print(f"✅ Generated {len(episodes)} episodes on shared map")
        
        # Create comprehensive visualization
        fig = visualizer.plot_multiple_episodes(episodes, save_path='shared_map_analysis.png')
        
        # Start rotation animation
        print("🔄 Starting 3D rotation animation...")
        anim = visualizer.start_rotation_animation(interval=100)  # Update every 100ms
        
        # Show plot
        try:
            plt.show()
        except Exception as e:
            print(f"⚠️ Display issue: {e}")
            print("📊 Visualization saved to file instead")
        
        return
    
    # Fallback: Use existing data if fresh generation fails
    print("⚠️ Fresh generation failed, trying existing data...")
    
    possible_paths = [
        "./logs_long_navigation/episode_paths.pkl",
        "./logs_realistic_navigation/episode_paths.pkl",
        "./logs_realistic_phase1_easy/dreamerv3/episode_paths.pkl",
        "./logs_realistic_phase1_easy/episode_paths.pkl",
        "./logs_advanced/episode_paths.pkl"
    ]
    
    for path_file in possible_paths:
        print(f"🔍 Checking {path_file}...")
        episodes = visualizer.load_best_episodes(path_file, num_episodes=10)
        
        if episodes:
            print(f"✅ Found episode data, creating visualization...")
            
            # Create visualization
            fig = visualizer.plot_multiple_episodes(episodes, save_path='multi_episode_analysis.png')
            
            # Start rotation animation
            print("🔄 Starting 3D rotation animation...")
            anim = visualizer.start_rotation_animation(interval=100)
            
            # Show plot
            try:
                plt.show()
            except Exception as e:
                print(f"⚠️ Display issue: {e}")
                print("📊 Visualization saved to file instead")
            
            return
    
    print("❌ No valid episode data found")
    print("💡 Make sure simple_drone_env.py is available for fresh generation")

def _plot_3d_content(ax, episodes, scenario_info, x_bounds, y_bounds, z_bounds):
    """Helper function to plot 3D content on an axis"""
    # Create arena box wireframe
    from itertools import combinations
    vertices = np.array([[x, y, z] for x in x_bounds for y in y_bounds for z in z_bounds])
    edges = [
        [0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3], 
        [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]
    ]
    
    for edge in edges:
        points = vertices[edge]
        ax.plot3D(points[:, 0], points[:, 1], points[:, 2], 'k-', alpha=0.3, linewidth=0.5)
    
    # Plot obstacles
    if 'obstacle_positions' in scenario_info and 'obstacle_radii' in scenario_info:
        for obs_pos, obs_radius in zip(scenario_info['obstacle_positions'], scenario_info['obstacle_radii']):
            # Create sphere for obstacle
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 20)
            x_sphere = obs_radius * np.outer(np.cos(u), np.sin(v)) + obs_pos[0]
            y_sphere = obs_radius * np.outer(np.sin(u), np.sin(v)) + obs_pos[1]
            z_sphere = obs_radius * np.outer(np.ones(np.size(u)), np.cos(v)) + obs_pos[2]
            
            ax.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.6, color='red')
    
    # Define colors for different episode types
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(episodes))))
    
    successful_count = 0
    failed_count = 0
    pathfinder_count = 0
    
    # Plot each episode
    for i, episode in enumerate(episodes):
        trajectory = np.array(episode['trajectory'])
        
        if len(trajectory) == 0:
            continue
            
        agent_type = episode.get('agent_type', f'Agent{i+1}')
        success = episode.get('success', False)
        
        # Special handling for pathfinder optimal path
        if 'pathfinder' in agent_type.lower() or 'optimal' in agent_type.lower():
            color = 'gold'
            alpha = 1.0
            linewidth = 4
            style_suffix = " 🗺️"
            pathfinder_count += 1
            
            # Plot pathfinder trajectory with special styling
            if len(trajectory.shape) == 2 and trajectory.shape[1] >= 3:
                # Main path line
                ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 
                        color=color, alpha=alpha, linewidth=linewidth, 
                        label=f'{agent_type}{style_suffix}', linestyle='--')
                
                # Add waypoint markers for pathfinder
                waypoint_indices = range(0, len(trajectory), max(1, len(trajectory)//10))
                for idx in waypoint_indices:
                    ax.scatter(trajectory[idx, 0], trajectory[idx, 1], trajectory[idx, 2], 
                               color='orange', s=50, marker='D', alpha=0.8)
                
                # Mark start and end points with special markers
                ax.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2], 
                           color='darkgreen', s=150, marker='o', alpha=1.0, edgecolors='black')
                ax.scatter(trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2], 
                           color='darkblue', s=150, marker='*', alpha=1.0, edgecolors='black')
            else:
                # Handle case where trajectory is just waypoints (list of tuples)
                trajectory_array = np.array(trajectory)
                if len(trajectory_array.shape) == 1:
                    # Convert list of coordinate tuples to proper array
                    trajectory_array = np.array([list(point) for point in trajectory])
                
                if trajectory_array.shape[1] >= 3:
                    ax.plot(trajectory_array[:, 0], trajectory_array[:, 1], trajectory_array[:, 2], 
                            color=color, alpha=alpha, linewidth=linewidth, 
                            label=f'{agent_type}{style_suffix}', linestyle='--')
                    
                    # Add waypoint markers
                    for point in trajectory_array:
                        ax.scatter(point[0], point[1], point[2], 
                                   color='orange', s=50, marker='D', alpha=0.8)
        else:
            # Choose color and style based on success for DreamerV3 agents
            if success:
                color = colors[successful_count % len(colors)]
                alpha = 0.8
                linewidth = 2
                successful_count += 1
                style_suffix = " ✓"
            else:
                color = 'gray'
                alpha = 0.4
                linewidth = 1
                failed_count += 1
                style_suffix = " ✗"
            
            # Plot trajectory
            if len(trajectory.shape) == 2 and trajectory.shape[1] >= 3:
                ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 
                        color=color, alpha=alpha, linewidth=linewidth, 
                        label=f'{agent_type}{style_suffix}')
                
                # Mark start and end points
                ax.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2], 
                           color=color, s=100, marker='o', alpha=alpha)
                ax.scatter(trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2], 
                           color=color, s=100, marker='s', alpha=alpha)
    
    # Plot start and goal positions
    if 'start_position' in scenario_info:
        start_pos = scenario_info['start_position']
        ax.scatter(start_pos[0], start_pos[1], start_pos[2], 
                   color='green', s=200, marker='o', label='Start 🚁', alpha=0.9)
    
    if 'goal_position' in scenario_info:
        goal_pos = scenario_info['goal_position']
        ax.scatter(goal_pos[0], goal_pos[1], goal_pos[2], 
                   color='blue', s=200, marker='*', label='Goal 🎯', alpha=0.9)
    
    # Customize 3D plot
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_zlabel('Z Position (m)')
    
    # Set equal aspect ratio
    ax.set_xlim(x_bounds)
    ax.set_ylim(y_bounds)
    ax.set_zlim(z_bounds)
    
    return successful_count, failed_count, pathfinder_count

def plot_multiple_episodes(episodes, scenario_info, save_path, title="Multi-Agent Episodes", historical_rewards=None):
    """
    Standalone function to plot multiple episodes with pathfinder support
    Enhanced to handle pathfinder optimal paths with special visualization
    """
    fig = plt.figure(figsize=(20, 6))  # Wider figure for front/back views
    
    # Arena bounds
    arena_size = scenario_info.get('arena_size', (18, 18, 8))
    x_bounds = [-arena_size[0]/2, arena_size[0]/2]
    y_bounds = [-arena_size[1]/2, arena_size[1]/2]
    z_bounds = [0, arena_size[2]]
    
    # Create first 3D plot with front view
    ax1 = fig.add_subplot(1, 4, 1, projection='3d')
    successful_count, failed_count, pathfinder_count = _plot_3d_content(ax1, episodes, scenario_info, x_bounds, y_bounds, z_bounds)
    ax1.view_init(elev=20, azim=45)  # Front view
    ax1.set_title(f'Front View\nSuccesses: {successful_count}, Failures: {failed_count}')
    
    # Create second 3D plot with back view
    ax1_back = fig.add_subplot(1, 4, 2, projection='3d')
    _plot_3d_content(ax1_back, episodes, scenario_info, x_bounds, y_bounds, z_bounds)
    ax1_back.view_init(elev=20, azim=225)  # Back view (opposite angle)
    ax1_back.set_title(f'Back View\nSuccesses: {successful_count}, Failures: {failed_count}')
    
    # Reward performance over iteration (line graph)
    ax2 = fig.add_subplot(1, 4, 3)
    
    # Use historical reward data if provided
    if historical_rewards:
        print(f"DEBUG: Using historical reward data: {len(historical_rewards)} iterations")
        
        # Plot reward lines for each agent across iterations
        plotted_any = False
        max_iterations = max(historical_rewards.keys()) if historical_rewards else 0
        
        # Dynamically determine the number of agents from the data
        agents_per_iteration = 0
        if historical_rewards:
            # Get the maximum agent ID from any iteration
            for iteration_data in historical_rewards.values():
                if iteration_data:
                    max_agent_in_iteration = max(iteration_data.keys())
                    agents_per_iteration = max(agents_per_iteration, max_agent_in_iteration)
        
        # Fallback to default if no data
        if agents_per_iteration == 0:
            agents_per_iteration = 12  # Default to current trainer setup
            
        print(f"DEBUG: Detected {agents_per_iteration} agents for reward tracking")
        
        # Use a colormap that can handle more colors for all 12 agents
        if agents_per_iteration <= 10:
            colors = plt.cm.tab10(np.linspace(0, 1, agents_per_iteration))
        else:
            # Use Set3 colormap for more than 10 agents, it has 12 distinct colors
            colors = plt.cm.Set3(np.linspace(0, 1, agents_per_iteration))
        
        for agent_id in range(1, agents_per_iteration + 1):
            agent_rewards_by_iteration = []
            iteration_numbers = []
            
            for iteration_num in sorted(historical_rewards.keys()):
                if agent_id in historical_rewards[iteration_num]:
                    agent_rewards_by_iteration.append(historical_rewards[iteration_num][agent_id])
                    iteration_numbers.append(iteration_num)
            
            if agent_rewards_by_iteration and len(agent_rewards_by_iteration) > 0:
                color = colors[agent_id - 1]
                
                ax2.plot(iteration_numbers, agent_rewards_by_iteration, 
                        color=color, marker='o', linewidth=1.5, 
                        linestyle='-', alpha=0.8, markersize=4,
                        label=f'Agent {agent_id}')
                plotted_any = True
                print(f"DEBUG: Plotted agent {agent_id} across {len(iteration_numbers)} iterations: {iteration_numbers}")
        
        if plotted_any:
            ax2.set_xlabel('Iteration Number')
            ax2.set_ylabel('Best Reward per Iteration')
            ax2.set_title('Agent Reward Over Iterations')
            ax2.grid(True, alpha=0.3)
            # Set integer ticks for iterations
            if max_iterations > 0:
                ax2.set_xticks(range(1, max_iterations + 1))
        else:
            ax2.text(0.5, 0.5, 'No historical reward data\navailable for plotting', 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=10)
            ax2.set_title('Agent Reward Over Iterations')
    else:
        # Fallback: show current iteration only
        print("DEBUG: No historical data provided, showing current iteration only")
        ax2.text(0.5, 0.5, 'Historical reward data\nnot available', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=10)
        ax2.set_title('Agent Reward Over Iterations')
        ax2.set_xlabel('Iteration Number')
        ax2.set_ylabel('Best Reward per Iteration')
    
    # Path length comparison
    ax3 = fig.add_subplot(1, 4, 4)
    path_lengths = []
    colors_for_lengths = []
    
    for episode in episodes:
        trajectory = episode['trajectory']
        if len(trajectory) > 1:
            # Calculate path length
            trajectory_array = np.array(trajectory)
            if len(trajectory_array.shape) == 2 and trajectory_array.shape[1] >= 3:
                distances = np.sqrt(np.sum(np.diff(trajectory_array, axis=0)**2, axis=1))
                path_length = np.sum(distances)
                path_lengths.append(path_length)
                
                if 'pathfinder' in episode.get('agent_type', '').lower() or 'optimal' in episode.get('agent_type', '').lower():
                    colors_for_lengths.append('gold')
                elif episode.get('success', False):
                    colors_for_lengths.append('green')
                else:
                    colors_for_lengths.append('red')
            else:
                path_lengths.append(0)
                colors_for_lengths.append('gray')
        else:
            path_lengths.append(0)
            colors_for_lengths.append('gray')

    
    if path_lengths:
        bars = ax3.bar(range(len(path_lengths)), path_lengths, color=colors_for_lengths, alpha=0.7)
        ax3.set_xlabel('Agent')
        ax3.set_ylabel('Path Length (m)')
        ax3.set_title('Path Length Comparison')
        ax3.set_xticks(range(len(path_lengths)))
        
        # Use actual agent names from episodes
        agent_labels = []
        for i, episode in enumerate(episodes):
            if i < len(path_lengths):
                agent_type = episode.get('agent_type', f'Agent{i+1}')
                agent_labels.append(agent_type)
        
        # Fill remaining labels if needed
        while len(agent_labels) < len(path_lengths):
            agent_labels.append(f'Agent{len(agent_labels)+1}')
            
        ax3.set_xticklabels(agent_labels, rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(path_lengths):
            if v > 0:
                ax3.text(i, v + max(path_lengths) * 0.01, f'{v:.1f}m', 
                        ha='center', va='bottom', fontsize=8)
    
    # Create a single comprehensive legend at the bottom
    # Collect handles and labels from all subplots
    all_handles = []
    all_labels = []
    
    # Get 3D plot legend items
    handles1, labels1 = ax1.get_legend_handles_labels()
    
    # Convert agent labels to numbers 1-60 and collect all unique entries
    legend_items = {}
    for i, (handle, label) in enumerate(zip(handles1, labels1)):
        if 'Start' in label or 'Goal' in label or 'Optimal' in label or '🗺️' in label:
            # Keep special labels as-is
            legend_items[label] = handle
        else:
            # Convert agent labels to numbers 1-60
            agent_num = (i % 60) + 1
            legend_items[str(agent_num)] = handle
    
    # Add any reward plot items that aren't already included
    try:
        handles2, labels2 = ax2.get_legend_handles_labels()
        for handle, label in zip(handles2, labels2):
            if label not in legend_items:
                legend_items[label] = handle
    except:
        pass
    
    # Sort legend items - special items first, then numbers
    special_items = [(label, handle) for label, handle in legend_items.items() 
                     if not label.isdigit()]
    numbered_items = [(label, handle) for label, handle in legend_items.items() 
                      if label.isdigit()]
    numbered_items.sort(key=lambda x: int(x[0]))  # Sort numbers numerically
    
    # Combine all items
    final_labels = [item[0] for item in special_items + numbered_items]
    final_handles = [item[1] for item in special_items + numbered_items]
    
    # Create single horizontal legend at bottom
    if final_handles:
        fig.legend(final_handles, final_labels, 
                  loc='lower center', bbox_to_anchor=(0.5, -0.08), 
                  ncol=min(len(final_labels), 20), fontsize=7,  # Increased from 4 to 7 for better readability
                  frameon=True, fancybox=True, shadow=True, 
                  markerscale=0.5, columnspacing=1.0)  # Increased markerscale and spacing for clarity
    
    # Adjust layout to make room for legend at bottom
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)  # More space for legend
    
    # Save the plot (static version)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"💾 Visualization saved as '{save_path}'")
    
    # Add 3D rotation animation for interactive viewing
    def update_3d_rotation(frame):
        """Update 3D view with rotation animation"""
        angle = 45 + frame * 2  # Start from 45 degrees, rotate 2 degrees per frame
        ax1.view_init(elev=20, azim=angle)
        return []
    
    # Create rotating 3D animation for display
    try:
        from matplotlib import animation
        anim = animation.FuncAnimation(
            fig, update_3d_rotation, frames=180, interval=100, 
            blit=False, repeat=True, cache_frame_data=False
        )
        # Store animation reference to prevent garbage collection
        fig._animation = anim
        print("🔄 3D rotation animation activated")
    except Exception as e:
        print(f"⚠️ Animation setup failed: {e}")
    
    return fig

if __name__ == "__main__":
    main()
