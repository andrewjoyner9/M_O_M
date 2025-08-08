#!/usr/bin/env python3
"""
Reward Configuration Manager

This module provides utilities to easily modify and test different reward configurations
for the SimpleDroneEnv without changing the core environment code.
"""

import copy
from typing import Dict, Any
import numpy as np

class RewardConfigManager:
    """Manager for different reward weight configurations"""
    
    @staticmethod
    def get_current_weights() -> Dict[str, float]:
        """Get the current default reward weights from SimpleDroneEnv"""
        return {
            'success_reward': 100.0,
            'collision_penalty': -50.0,
            'step_penalty': -0.02,
            'distance_weight': 10.0,
            'velocity_weight': 0.01,
            'obstacle_proximity_weight': 5.0,
            'safety_bonus_weight': 0.5,
            'out_of_bounds_penalty': 50.0,
            'efficiency_bonus_scale': 50.0
        }
    
    @staticmethod
    def get_preset_configs() -> Dict[str, Dict]:
        """Get predefined reward weight presets optimized for different training goals"""
        
        return {
            'default': {
                'name': 'Default Weights',
                'description': 'Current default configuration',
                'success_reward': 100.0,
                'collision_penalty': -50.0,
                'step_penalty': -0.02,
                'distance_weight': 10.0,
                'velocity_weight': 0.01,
                'obstacle_proximity_weight': 5.0,
                'safety_bonus_weight': 0.5,
                'out_of_bounds_penalty': 50.0,
                'efficiency_bonus_scale': 50.0
            },
            
            'balanced_v1': {
                'name': 'Balanced Learning v1',
                'description': 'Balanced weights for stable learning progression',
                'success_reward': 50.0,
                'collision_penalty': -25.0,
                'step_penalty': -0.1,
                'distance_weight': 5.0,
                'velocity_weight': 0.005,
                'obstacle_proximity_weight': 10.0,
                'safety_bonus_weight': 1.0,
                'out_of_bounds_penalty': 30.0,
                'efficiency_bonus_scale': 30.0
            },
            
            'exploration_focused': {
                'name': 'Exploration Focused',
                'description': 'Encourages exploration with reduced penalties',
                'success_reward': 200.0,
                'collision_penalty': -10.0,
                'step_penalty': -0.01,
                'distance_weight': 3.0,
                'velocity_weight': 0.001,
                'obstacle_proximity_weight': 15.0,
                'safety_bonus_weight': 2.0,
                'out_of_bounds_penalty': 20.0,
                'efficiency_bonus_scale': 80.0
            },
            
            'safety_prioritized': {
                'name': 'Safety Prioritized',
                'description': 'Strong penalties for unsafe behavior',
                'success_reward': 75.0,
                'collision_penalty': -100.0,
                'step_penalty': -0.05,
                'distance_weight': 8.0,
                'velocity_weight': 0.02,
                'obstacle_proximity_weight': 25.0,
                'safety_bonus_weight': 5.0,
                'out_of_bounds_penalty': 80.0,
                'efficiency_bonus_scale': 40.0
            },
            
            'efficiency_focused': {
                'name': 'Efficiency Focused',
                'description': 'Rewards fast, direct paths to goal',
                'success_reward': 150.0,
                'collision_penalty': -30.0,
                'step_penalty': -0.5,
                'distance_weight': 15.0,
                'velocity_weight': 0.001,
                'obstacle_proximity_weight': 8.0,
                'safety_bonus_weight': 0.2,
                'out_of_bounds_penalty': 40.0,
                'efficiency_bonus_scale': 100.0
            },
            
            'gentle_learning': {
                'name': 'Gentle Learning',
                'description': 'Soft penalties for gradual skill development',
                'success_reward': 25.0,
                'collision_penalty': -5.0,
                'step_penalty': -0.001,
                'distance_weight': 2.0,
                'velocity_weight': 0.0001,
                'obstacle_proximity_weight': 3.0,
                'safety_bonus_weight': 0.1,
                'out_of_bounds_penalty': 10.0,
                'efficiency_bonus_scale': 15.0
            },
            
            'aggressive_training': {
                'name': 'Aggressive Training',
                'description': 'High rewards and penalties for rapid learning',
                'success_reward': 300.0,
                'collision_penalty': -150.0,
                'step_penalty': -0.8,
                'distance_weight': 20.0,
                'velocity_weight': 0.05,
                'obstacle_proximity_weight': 30.0,
                'safety_bonus_weight': 8.0,
                'out_of_bounds_penalty': 120.0,
                'efficiency_bonus_scale': 150.0
            },
            
            'boundary_aware_v1': {
                'name': 'Boundary-Aware Training v1',
                'description': 'Balanced boundary penalties that prevent ceiling/ground seeking while allowing learning',
                'success_reward': 150.0,
                'collision_penalty': -100.0,
                'step_penalty': -0.03,  # Reduced from -0.1 to allow longer exploration
                'distance_weight': 8.0,
                'velocity_weight': 0.01,
                'obstacle_proximity_weight': 15.0,
                'safety_bonus_weight': 2.0,
                'out_of_bounds_penalty': 150.0,  # Reduced from 300 to 150 (still 3x standard)
                'efficiency_bonus_scale': 60.0,
                # Balanced vertical-specific penalties
                'vertical_boundary_penalty': 75.0,  # Reduced from 150 to 75 (still 1.5x standard)
                'goal_direction_bonus': 5.0,
                'vertical_exploration_penalty': 15.0, # Reduced from 20 to 15 (still 1.5x standard)
                'goal_progress_bonus': 3.0,
                'ground_crash_penalty': 100.0       # Reduced from 200 to 100 (still strong deterrent)
            }
        }
    
    @staticmethod
    def create_custom_reward_function(config: Dict[str, float]):
        """Create a custom reward calculation function with specified weights"""
        
        def calculate_custom_reward(env) -> float:
            """Custom reward calculation with configurable weights"""
            reward = 0.0
            
            # 1. Distance to goal reward (continuous shaping)
            goal_distance = np.linalg.norm(env.drone_position - env.goal_position)
            max_possible_distance = np.linalg.norm(env.arena_max - env.arena_min)
            normalized_goal_distance = goal_distance / max_possible_distance
            distance_weight = config.get('distance_weight', 10.0)
            reward -= normalized_goal_distance * distance_weight
            
            # 2. Success bonus with efficiency bonus
            if env._is_success():
                reward += config.get('success_reward', 100.0)
                # Efficiency bonus for fast completion
                efficiency_scale = config.get('efficiency_bonus_scale', 50.0)
                efficiency_bonus = (1.0 - env.current_step / env.max_steps) * efficiency_scale
                reward += efficiency_bonus
            
            # 3. Collision penalty
            if env._is_collision():
                reward += config.get('collision_penalty', -50.0)  # Note: already negative
            
            # 4. Out of bounds penalty
            if env._is_out_of_bounds():
                penalty = config.get('out_of_bounds_penalty', 50.0)
                reward -= penalty
            
            # 5. Velocity penalty (encourage smooth movement)
            velocity_magnitude = np.linalg.norm(env.drone_velocity)
            velocity_weight = config.get('velocity_weight', 0.01)
            reward -= velocity_magnitude * velocity_weight
            
            # 6. Step penalty (encourage efficiency)
            step_penalty = config.get('step_penalty', -0.02)
            reward += step_penalty  # Note: step_penalty is already negative
            
            # 7. Progressive obstacle avoidance
            min_obstacle_distance = float('inf')
            obstacle_proximity_weight = config.get('obstacle_proximity_weight', 5.0)
            safety_bonus_weight = config.get('safety_bonus_weight', 0.5)
            
            for i, obs_pos in enumerate(env.obstacle_positions):
                obs_radius = env.obstacle_radii[i] if i < len(env.obstacle_radii) else 1.0
                dist = np.linalg.norm(env.drone_position - obs_pos)
                min_obstacle_distance = min(min_obstacle_distance, dist - obs_radius)
                
                # Safety zone penalty
                safety_zone = obs_radius + 2.0
                if dist < safety_zone:
                    danger_factor = (safety_zone - dist) / safety_zone
                    reward -= danger_factor * obstacle_proximity_weight
            
            # 8. Bonus for maintaining safe distance from all obstacles
            if min_obstacle_distance > 2.0:
                reward += safety_bonus_weight
            
            # 9. CRITICAL FIX: Vertical boundary penalties (ceiling/ground seeking prevention)
            vertical_boundary_penalty = config.get('vertical_boundary_penalty', 0.0)
            ground_crash_penalty = config.get('ground_crash_penalty', 0.0)
            
            if vertical_boundary_penalty > 0 or ground_crash_penalty > 0:
                arena_height = env.arena_max[2]
                drone_z = env.drone_position[2]
                
                # CRITICAL FIX: Severe penalty for ground crashes (z <= 0.5m)
                if ground_crash_penalty > 0 and drone_z <= 0.5:
                    ground_factor = (0.5 - drone_z) / 0.5  # 0 to 1 scale
                    reward -= ground_factor * ground_crash_penalty
                
                # Strong penalty for being too close to ceiling or ground (beyond ground crash zone)
                if vertical_boundary_penalty > 0:
                    ceiling_distance = arena_height - drone_z
                    ground_distance = drone_z
                    
                    # Exponential penalty when within 2.0m of ceiling/ground (increased from 1.5m)
                    if ceiling_distance < 2.0:
                        ceiling_penalty = (2.0 - ceiling_distance) / 2.0
                        reward -= ceiling_penalty * vertical_boundary_penalty
                    
                    if ground_distance < 2.0 and drone_z > 0.5:  # Only if not in crash zone
                        ground_penalty = (2.0 - ground_distance) / 2.0
                        reward -= ground_penalty * vertical_boundary_penalty
            
            # 10. CRITICAL FIX: Goal direction bonus (encourage movement toward goal)
            goal_direction_bonus = config.get('goal_direction_bonus', 0.0)
            if goal_direction_bonus > 0 and hasattr(env, '_previous_goal_distance'):
                current_distance = np.linalg.norm(env.drone_position - env.goal_position)
                if current_distance < env._previous_goal_distance:
                    # Reward for getting closer to goal
                    progress = env._previous_goal_distance - current_distance
                    reward += progress * goal_direction_bonus
                env._previous_goal_distance = current_distance
            elif not hasattr(env, '_previous_goal_distance'):
                env._previous_goal_distance = np.linalg.norm(env.drone_position - env.goal_position)
            
            # 11. CRITICAL FIX: Vertical exploration penalty (discourage excessive vertical movement)
            vertical_exploration_penalty = config.get('vertical_exploration_penalty', 0.0)
            if vertical_exploration_penalty > 0:
                vertical_velocity = abs(env.drone_velocity[2])
                if vertical_velocity > 0.5:  # Threshold for "excessive" vertical movement
                    reward -= vertical_velocity * vertical_exploration_penalty
            
            # 12. CRITICAL FIX: Goal progress bonus (reward getting closer to goal overall)
            goal_progress_bonus = config.get('goal_progress_bonus', 0.0)
            if goal_progress_bonus > 0:
                # Bonus based on how close to goal (inverted distance)
                max_distance = np.linalg.norm(env.arena_max - env.arena_min)
                progress_factor = 1.0 - (goal_distance / max_distance)
                reward += progress_factor * goal_progress_bonus
            
            return reward
        
        return calculate_custom_reward
    
    @staticmethod
    def apply_config_to_env(env, config_name: str = None, custom_config: Dict = None):
        """Apply a reward configuration to an existing environment instance"""
        
        if custom_config:
            config = custom_config
        elif config_name:
            configs = RewardConfigManager.get_preset_configs()
            if config_name not in configs:
                raise ValueError(f"Unknown config '{config_name}'. Available: {list(configs.keys())}")
            config = configs[config_name]
        else:
            raise ValueError("Must provide either config_name or custom_config")
        
        # Create custom reward function
        custom_reward_fn = RewardConfigManager.create_custom_reward_function(config)
        
        # Store original method in case we want to restore it
        if not hasattr(env, '_original_calculate_reward'):
            env._original_calculate_reward = env._calculate_reward
        
        # Replace the reward calculation method
        env._calculate_reward = lambda: custom_reward_fn(env)
        
        # Update relevant config values that are used elsewhere
        if 'success_reward' in config:
            env.config['success_reward'] = config['success_reward']
        if 'collision_penalty' in config:
            env.config['collision_penalty'] = config['collision_penalty']  
        if 'step_penalty' in config:
            env.config['step_penalty'] = config['step_penalty']
        
        # Store the applied config for reference
        env._applied_reward_config = config
        env._applied_config_name = config_name or 'custom'
        
        return env
    
    @staticmethod
    def restore_original_rewards(env):
        """Restore the original reward calculation to an environment"""
        
        if hasattr(env, '_original_calculate_reward'):
            env._calculate_reward = env._original_calculate_reward
            delattr(env, '_original_calculate_reward')
        
        if hasattr(env, '_applied_reward_config'):
            delattr(env, '_applied_reward_config')
        
        if hasattr(env, '_applied_config_name'):
            delattr(env, '_applied_config_name')
        
        return env
    
    @staticmethod
    def print_config_comparison(config_names: list = None):
        """Print a comparison table of different reward configurations"""
        
        configs = RewardConfigManager.get_preset_configs()
        
        if config_names is None:
            config_names = list(configs.keys())
        
        print("\n📊 Reward Configuration Comparison")
        print("=" * 100)
        
        # Header
        header = f"{'Parameter':<25}"
        for name in config_names:
            header += f"{name:<15}"
        print(header)
        print("-" * 100)
        
        # Get all parameter names
        all_params = set()
        for config_name in config_names:
            if config_name in configs:
                all_params.update(configs[config_name].keys())
        
        all_params = sorted([p for p in all_params if p not in ['name', 'description']])
        
        # Print each parameter
        for param in all_params:
            row = f"{param:<25}"
            for config_name in config_names:
                if config_name in configs:
                    value = configs[config_name].get(param, 'N/A')
                    if isinstance(value, float):
                        row += f"{value:<15.3f}"
                    else:
                        row += f"{str(value):<15}"
                else:
                    row += f"{'N/A':<15}"
            print(row)
        
        # Print descriptions
        print("\n📝 Configuration Descriptions:")
        print("-" * 50)
        for config_name in config_names:
            if config_name in configs:
                config = configs[config_name]
                print(f"{config_name:15}: {config.get('description', 'No description')}")

# Quick usage examples and testing
if __name__ == "__main__":
    # Example usage
    print("🎯 Reward Configuration Manager")
    print("=" * 40)
    
    # Print configuration comparison
    RewardConfigManager.print_config_comparison(['default', 'balanced_v1', 'exploration_focused', 'safety_prioritized'])
    
    # Example of creating a custom config
    custom_weights = {
        'success_reward': 80.0,
        'collision_penalty': -40.0,
        'step_penalty': -0.08,
        'distance_weight': 6.0,
        'velocity_weight': 0.008,
        'obstacle_proximity_weight': 12.0,
        'safety_bonus_weight': 1.5
    }
    
    print(f"\n🔧 Example Custom Configuration:")
    for param, value in custom_weights.items():
        print(f"   {param:25}: {value:8.3f}")
