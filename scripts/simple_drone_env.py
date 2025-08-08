#!/usr/bin/env python3
"""
Simple Drone Environment for DreamerV3 Training

A lightweight 3D navigation environment compatible with Gymnasium interface.
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional
import math

try:
    import gymnasium as gym
    from gymnasium.spaces import Box
    GYM_AVAILABLE = True
    print("✓ Using Gymnasium")
except ImportError:
    try:
        import gym
        from gym.spaces import Box
        GYM_AVAILABLE = True
        print("✓ Using OpenAI Gym")
    except ImportError:
        GYM_AVAILABLE = False
        print("⚠️ No Gym/Gymnasium available, using custom Box")
        
        class Box:
            """Simple Box space replacement for gym.spaces.Box"""
            def __init__(self, low, high, shape, dtype=np.float32):
                self.low = np.full(shape, low, dtype=dtype)
                self.high = np.full(shape, high, dtype=dtype)
                self.shape = shape
                self.dtype = dtype
            
            def sample(self):
                return np.random.uniform(self.low, self.high).astype(self.dtype)


# Base class selection
if GYM_AVAILABLE:
    if 'gymnasium' in str(gym.__file__):
        BaseEnv = gym.Env
    else:
        BaseEnv = gym.Env
else:
    class BaseEnv:
        """Fallback base environment class"""
        pass


class SimpleDroneEnv(BaseEnv):
    """
    Simple 3D drone navigation environment
    
    The drone needs to navigate from start to goal while avoiding obstacles.
    Configurable for realistic training scenarios.
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(self, env_id: int = 0, config: dict = None):
        super().__init__()
        self.env_id = env_id
        
        # Default configuration for realistic scenarios
        default_config = {
            'arena_size': 20.0,           # Reasonable arena for learning
            'arena_height': 8.0,          # Reasonable height
            'max_steps': 800,             # Long enough for complex navigation
            'num_obstacles': 8,           # Moderate obstacles for learning
            'obstacle_radius_range': [0.8, 1.5],  # Reasonable obstacle sizes
            'max_velocity': 2.0,          # Moderate drone speed
            'max_acceleration': 4.0,      # Moderate acceleration
            'goal_threshold': 1.2,        # Achievable goal threshold
            'success_reward': 100.0,      # Good success reward
            'collision_penalty': -50.0,   # Moderate collision penalty
            'step_penalty': -0.02,        # Small step penalty
            'obstacle_clearance': 2.5,    # Safe clearance from start/goal
            'difficulty_level': 'easy'    # Start with easy difficulty
        }
        
        # Merge with provided config
        if config:
            default_config.update(config)
        self.config = default_config
        
        # Set environment bounds based on arena size
        arena_half = self.config['arena_size'] / 2
        # CRITICAL FIX: Set minimum Z to 0.5m to make ground level (z=0) out-of-bounds
        self.arena_min = np.array([-arena_half, -arena_half, 0.5])
        self.arena_max = np.array([arena_half, arena_half, self.config['arena_height']])
        
        # Drone parameters
        self.max_velocity = self.config['max_velocity']
        self.max_acceleration = self.config['max_acceleration']
        
        # Episode parameters
        self.max_steps = self.config['max_steps']
        self.current_step = 0
        
        # Obstacle parameters (dynamic based on difficulty)
        self.num_obstacles = self._get_difficulty_obstacles()
        self.obstacle_positions = []
        self.obstacle_radii = []  # Variable sizes
        
        # Observation space: position(3) + velocity(3) + goal(3) + obstacle_distances(num_obstacles) + goal_distance(1) + step_normalized(1)
        obs_dim = 3 + 3 + 3 + self.num_obstacles + 1 + 1
        self.obs_dim = obs_dim  # Store for external access
        self.act_dim = 3        # Store for external access
        
        self.observation_space = Box(
            low=-100.0, high=100.0, shape=(obs_dim,), dtype=np.float32
        )
        
        # Action space: acceleration in x, y, z
        self.action_space = Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )
        
        # State variables
        self.drone_position = np.zeros(3)
        self.drone_velocity = np.zeros(3)
        self.goal_position = np.zeros(3)
        
        # Path tracking for visualization
        self.path_history = []
        self.episode_paths = []  # Store completed episode paths
        self.max_stored_paths = 50  # Keep last 50 episodes
        
        print(f"✓ Enhanced SimpleDroneEnv {env_id} created")
        print(f"  Arena: {self.config['arena_size']}x{self.config['arena_size']}x{self.config['arena_height']}")
        print(f"  Max steps: {self.config['max_steps']}")
        print(f"  Obstacles: {self.num_obstacles}")
        print(f"  Difficulty: {self.config['difficulty_level']}")
    
    def _get_difficulty_obstacles(self) -> int:
        """Get number of obstacles based on difficulty level"""
        base_obstacles = self.config['num_obstacles']
        difficulty = self.config['difficulty_level']
        
        if difficulty == 'easy':
            return max(3, base_obstacles // 3)
        elif difficulty == 'medium':
            return max(8, base_obstacles // 2)
        elif difficulty == 'hard':
            return base_obstacles
        elif difficulty == 'expert':
            return int(base_obstacles * 1.5)
        else:
            return base_obstacles
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state"""
        self.current_step = 0
        
        # Store previous episode path if it exists
        if len(self.path_history) > 0:
            episode_info = {
                'path': self.path_history.copy(),
                'start': self.path_history[0] if self.path_history else None,
                'goal': self.goal_position.copy(),
                'obstacles': [(pos.copy(), radius) for pos, radius in zip(self.obstacle_positions, self.obstacle_radii)],
                'success': self._is_success(),
                'steps': len(self.path_history),
                'final_reward': getattr(self, '_last_reward', 0)
            }
            self.episode_paths.append(episode_info)
            
            # Keep only recent episodes
            if len(self.episode_paths) > self.max_stored_paths:
                self.episode_paths.pop(0)
        
        # Store previous episode path if it exists
        if len(self.path_history) > 0:
            episode_info = {
                'path': self.path_history.copy(),
                'start': self.path_history[0] if self.path_history else None,
                'goal': self.goal_position.copy(),
                'obstacles': [(pos.copy(), radius) for pos, radius in zip(self.obstacle_positions, self.obstacle_radii)],
                'success': self._is_success(),
                'steps': len(self.path_history),
                'final_reward': getattr(self, '_last_reward', 0)
            }
            self.episode_paths.append(episode_info)
            
            # Keep only recent episodes
            if len(self.episode_paths) > self.max_stored_paths:
                self.episode_paths.pop(0)
        
        # Reset path tracking
        self.path_history = []
        
        # Random start position (avoid walls with safety margin) - use fixed position if set
        safety_margin = 3.0
        if hasattr(self, '_use_fixed_positions') and self._use_fixed_positions:
            # Use fixed start position from scenario
            self.drone_position = np.array(self._fixed_start_position)
        else:
            # Original randomization behavior
            self.drone_position = np.random.uniform(
                self.arena_min + safety_margin, 
                self.arena_max - safety_margin
            )
        
        # Zero velocity
        self.drone_velocity = np.zeros(3)
        
        # Random goal position (ensure minimum distance from start) - use fixed position if set
        # Use config value if provided, otherwise default to 40% of arena width
        min_goal_distance = self.config.get('min_start_goal_distance', self.config['arena_size'] * 0.4)
        max_attempts = 100  # More attempts for better positioning
        
        if hasattr(self, '_use_fixed_positions') and self._use_fixed_positions:
            # Use fixed goal position from scenario
            self.goal_position = np.array(self._fixed_goal_position)
        else:
            # Original randomization behavior
            for attempt in range(max_attempts):
                self.goal_position = np.random.uniform(
                    self.arena_min + safety_margin, 
                    self.arena_max - safety_margin
                )
                distance = np.linalg.norm(self.goal_position - self.drone_position)
                if distance >= min_goal_distance:
                    break
        
        # Verify we achieved minimum distance (only for randomized positions)
        if not (hasattr(self, '_use_fixed_positions') and self._use_fixed_positions):
            actual_distance = np.linalg.norm(self.goal_position - self.drone_position)
            if actual_distance < min_goal_distance:
                print(f"⚠️ Warning: Start-goal distance {actual_distance:.2f} < minimum {min_goal_distance:.2f}")
        
        # Generate obstacles with variable sizes and strategic placement
        self.obstacle_positions = []
        self.obstacle_radii = []
        clearance = self.config['obstacle_clearance']
        
        for _ in range(self.num_obstacles):
            max_placement_attempts = 100
            placed = False
            
            for _ in range(max_placement_attempts):
                # Random position with margins
                obs_pos = np.random.uniform(
                    self.arena_min + 2.0, 
                    self.arena_max - 2.0
                )
                
                # Random radius within specified range
                obs_radius = np.random.uniform(
                    self.config['obstacle_radius_range'][0],
                    self.config['obstacle_radius_range'][1]
                )
                
                # Check clearance from start, goal, and other obstacles
                start_clear = np.linalg.norm(obs_pos - self.drone_position) > (obs_radius + clearance)
                goal_clear = np.linalg.norm(obs_pos - self.goal_position) > (obs_radius + clearance)
                
                # Check clearance from existing obstacles
                obstacle_clear = True
                for i, existing_pos in enumerate(self.obstacle_positions):
                    existing_radius = self.obstacle_radii[i]
                    min_distance = obs_radius + existing_radius + 1.0  # 1m minimum separation
                    if np.linalg.norm(obs_pos - existing_pos) < min_distance:
                        obstacle_clear = False
                        break
                
                if start_clear and goal_clear and obstacle_clear:
                    self.obstacle_positions.append(obs_pos)
                    self.obstacle_radii.append(obs_radius)
                    placed = True
                    break
            
            if not placed:
                # If we can't place obstacle with all constraints, place with reduced constraints
                obs_pos = np.random.uniform(
                    self.arena_min + 2.0, 
                    self.arena_max - 2.0
                )
                obs_radius = self.config['obstacle_radius_range'][0]  # Use minimum radius
                
                # Only check that it's not too close to start/goal
                if (np.linalg.norm(obs_pos - self.drone_position) > obs_radius + 2.0 and 
                    np.linalg.norm(obs_pos - self.goal_position) > obs_radius + 2.0):
                    self.obstacle_positions.append(obs_pos)
                    self.obstacle_radii.append(obs_radius)
        
        return self._get_observation()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Step the environment"""
        self.current_step += 1
        
        # Clip and scale action
        action = np.clip(action, -1.0, 1.0)
        acceleration = action * self.max_acceleration
        
        # Update velocity and position
        dt = 0.02  # 50 Hz
        self.drone_velocity += acceleration * dt
        self.drone_velocity = np.clip(self.drone_velocity, -self.max_velocity, self.max_velocity)
        self.drone_position += self.drone_velocity * dt
        
        # Track path for visualization
        self.path_history.append(self.drone_position.copy())
        
        # Calculate reward
        reward = self._calculate_reward()
        self._last_reward = reward  # Store for episode tracking
        
        # Check termination
        done = self._is_done()
        
        # Info
        info = {
            'success': self._is_success(),
            'collision': self._is_collision(),
            'out_of_bounds': self._is_out_of_bounds(),
            'timeout': self.current_step >= self.max_steps
        }
        
        return self._get_observation(), reward, done, info
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation"""
        # Distances to obstacles (considering variable radii)
        obstacle_distances = []
        for i, obs_pos in enumerate(self.obstacle_positions):
            obs_radius = self.obstacle_radii[i] if i < len(self.obstacle_radii) else 1.0
            dist = np.linalg.norm(self.drone_position - obs_pos) - obs_radius
            obstacle_distances.append(max(0.0, dist))  # 0 if inside obstacle
        
        # Pad obstacle distances if we have fewer obstacles than expected
        while len(obstacle_distances) < self.num_obstacles:
            obstacle_distances.append(100.0)  # Large distance for non-existent obstacles
        
        # Distance to goal
        goal_distance = np.linalg.norm(self.drone_position - self.goal_position)
        
        # Normalized step
        step_normalized = self.current_step / self.max_steps
        
        obs = np.concatenate([
            self.drone_position,
            self.drone_velocity,
            self.goal_position,
            obstacle_distances[:self.num_obstacles],  # Ensure we only take expected number
            [goal_distance],
            [step_normalized]
        ]).astype(np.float32)
        
        return obs
    
    def _calculate_reward(self) -> float:
        """Calculate reward for current state"""
        reward = 0.0
        
        # Distance to goal reward (shaped to guide towards goal)
        goal_distance = np.linalg.norm(self.drone_position - self.goal_position)
        max_possible_distance = np.linalg.norm(self.arena_max - self.arena_min)
        normalized_goal_distance = goal_distance / max_possible_distance
        reward -= normalized_goal_distance * 10.0  # Scale down distance penalty
        
        # Success bonus
        if self._is_success():
            reward += self.config['success_reward']
            # Bonus for efficient completion
            efficiency_bonus = (1.0 - self.current_step / self.max_steps) * 50.0
            reward += efficiency_bonus
        
        # Collision penalty
        if self._is_collision():
            reward += self.config['collision_penalty']
        
        # Out of bounds penalty
        if self._is_out_of_bounds():
            reward -= 50.0
        
        # Velocity penalty (encourage smooth movement)
        velocity_magnitude = np.linalg.norm(self.drone_velocity)
        reward -= velocity_magnitude * 0.01
        
        # Step penalty (encourage efficiency)
        reward += self.config['step_penalty']
        
        # Progressive obstacle avoidance reward
        min_obstacle_distance = float('inf')
        for i, obs_pos in enumerate(self.obstacle_positions):
            obs_radius = self.obstacle_radii[i] if i < len(self.obstacle_radii) else 1.0
            dist = np.linalg.norm(self.drone_position - obs_pos)
            min_obstacle_distance = min(min_obstacle_distance, dist - obs_radius)
            
            # Safety zone reward/penalty
            safety_zone = obs_radius + 2.0
            if dist < safety_zone:
                danger_factor = (safety_zone - dist) / safety_zone
                reward -= danger_factor * 5.0  # Penalty for being too close
            
        # Bonus for maintaining safe distance from all obstacles
        if min_obstacle_distance > 2.0:
            reward += 0.5
        
        return reward
    
    def _is_success(self) -> bool:
        """Check if goal is reached"""
        goal_distance = np.linalg.norm(self.drone_position - self.goal_position)
        return goal_distance < self.config['goal_threshold']
    
    def _is_collision(self) -> bool:
        """Check if drone collided with obstacle"""
        for i, obs_pos in enumerate(self.obstacle_positions):
            obs_radius = self.obstacle_radii[i] if i < len(self.obstacle_radii) else 1.0
            dist = np.linalg.norm(self.drone_position - obs_pos)
            if dist < obs_radius:
                return True
        return False
    
    def _is_out_of_bounds(self) -> bool:
        """Check if drone is out of arena bounds"""
        return not np.all((self.drone_position >= self.arena_min) & 
                         (self.drone_position <= self.arena_max))
    
    def _is_done(self) -> bool:
        """Check if episode is done"""
        return (self._is_success() or 
                self._is_collision() or 
                self._is_out_of_bounds() or 
                self.current_step >= self.max_steps)
    
    def render(self, mode: str = 'human'):
        """Render the environment (placeholder)"""
        if mode == 'human':
            print(f"Step {self.current_step}: Drone at {self.drone_position}, "
                  f"Goal at {self.goal_position}, "
                  f"Distance: {np.linalg.norm(self.drone_position - self.goal_position):.2f}")
    
    def close(self):
        """Close the environment"""
        pass
    
    def seed(self, seed: Optional[int] = None):
        """Set random seed"""
        if seed is not None:
            np.random.seed(seed)
    
    def get_episode_paths(self, num_episodes: int = None):
        """Get stored episode paths for visualization"""
        if num_episodes is None:
            return self.episode_paths
        else:
            return self.episode_paths[-num_episodes:] if len(self.episode_paths) >= num_episodes else self.episode_paths
    
    def get_current_path(self):
        """Get current episode path"""
        return {
            'path': self.path_history.copy(),
            'start': self.path_history[0] if self.path_history else self.drone_position.copy(),
            'goal': self.goal_position.copy(),
            'obstacles': [(pos.copy(), radius) for pos, radius in zip(self.obstacle_positions, self.obstacle_radii)],
            'current_position': self.drone_position.copy(),
            'steps': len(self.path_history)
        }
