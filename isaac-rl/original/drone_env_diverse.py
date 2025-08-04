import gymnasium as gym
import numpy as np

class DiverseDroneEnv(gym.Env):
    """
    Enhanced Drone Environment that encourages path exploration and diversity.
    
    Key improvements:
    1. Randomized start/goal positions each episode
    2. Multiple threats with random positions
    3. Path diversity rewards
    4. Exploration bonuses
    5. Dynamic obstacle positions
    """
    
    def __init__(self, encourage_exploration=True, num_threats=3):
        super().__init__()
        
        # Expanded action space for better control
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)  # vx, vy, vz
        
        # Expanded observation space to include multiple threats and exploration info
        # [drone_pos(3), goal_pos(3), threats(3*num_threats), visited_areas(4), exploration_bonus(1)]
        obs_size = 3 + 3 + (3 * num_threats) + 4 + 1
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)
        
        # Environment parameters
        self.num_threats = num_threats
        self.encourage_exploration = encourage_exploration
        self.max_steps = 1500  # Increased to allow for exploration
        
        # Exploration tracking
        self.visited_grid = None
        self.grid_size = 10  # 10x10 grid for tracking visited areas
        self.exploration_bonus_scale = 1.0
        
        # Path diversity tracking
        self.trajectory = []
        self.previous_paths = []  # Store previous successful paths
        self.path_diversity_bonus = 5.0
        
        # Dynamic environment
        self.threat_movement_enabled = True
        self.threat_velocities = None
        
        # Initialize positions (will be randomized in reset)
        self.drone_pos = None
        self.goal_pos = None
        self.threat_positions = None
        self.current_step = 0
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        
        # Randomize starting position within a region
        self.drone_pos = np.array([
            np.random.uniform(-1.0, 1.0),  # x: start near origin but with variation
            np.random.uniform(-1.0, 1.0),  # y: start near origin but with variation  
            np.random.uniform(0.5, 1.5)    # z: vary height slightly
        ])
        
        # Randomize goal position within target region
        self.goal_pos = np.array([
            np.random.uniform(4.0, 6.0),   # x: goal region
            np.random.uniform(4.0, 6.0),   # y: goal region
            np.random.uniform(0.5, 1.5)    # z: vary height
        ])
        
        # Randomize threat positions
        self.threat_positions = []
        for i in range(self.num_threats):
            # Place threats in the path between start and goal with some randomness
            threat_x = np.random.uniform(1.0, 4.0)
            threat_y = np.random.uniform(1.0, 4.0) 
            threat_z = np.random.uniform(0.5, 1.5)
            self.threat_positions.append(np.array([threat_x, threat_y, threat_z]))
        
        # Initialize threat movement
        if self.threat_movement_enabled:
            self.threat_velocities = []
            for _ in range(self.num_threats):
                vel = np.random.uniform(-0.02, 0.02, 3)  # Slow random movement
                self.threat_velocities.append(vel)
        
        # Reset exploration tracking
        self.visited_grid = np.zeros((self.grid_size, self.grid_size))
        self.trajectory = [self.drone_pos.copy()]
        self.current_step = 0
        
        obs = self._get_observation()
        info = {
            'start_pos': self.drone_pos.copy(),
            'goal_pos': self.goal_pos.copy(),
            'threat_positions': [t.copy() for t in self.threat_positions]
        }
        return obs, info

    def step(self, action):
        # Apply action with some noise to encourage exploration during training
        if self.encourage_exploration and np.random.random() < 0.1:
            # 10% chance to add small exploration noise
            exploration_noise = np.random.normal(0, 0.05, 3)
            action = np.clip(action + exploration_noise, -1, 1)
        
        # Apply action to drone
        self.drone_pos += action * 0.15  # Slightly increased movement for faster exploration
        self.current_step += 1
        
        # Update trajectory
        self.trajectory.append(self.drone_pos.copy())
        
        # Move threats if enabled
        if self.threat_movement_enabled and self.threat_velocities:
            for i, (threat_pos, vel) in enumerate(zip(self.threat_positions, self.threat_velocities)):
                # Update threat position
                new_pos = threat_pos + vel
                
                # Keep threats within bounds and occasionally change direction
                if np.random.random() < 0.01:  # 1% chance to change direction
                    self.threat_velocities[i] = np.random.uniform(-0.02, 0.02, 3)
                
                # Boundary checking
                new_pos = np.clip(new_pos, [0, 0, 0.2], [6, 6, 2.0])
                self.threat_positions[i] = new_pos
        
        # Update exploration grid
        self._update_exploration_grid()
        
        obs = self._get_observation()
        reward = self._get_reward()
        terminated = self._check_done()
        truncated = self.current_step >= self.max_steps
        
        info = {
            'exploration_coverage': np.sum(self.visited_grid > 0) / (self.grid_size * self.grid_size),
            'path_length': len(self.trajectory),
            'current_threats': len([t for t in self.threat_positions if np.linalg.norm(self.drone_pos - t) < 2.0])
        }
        
        return obs, reward, terminated, truncated, info

    def _update_exploration_grid(self):
        """Update the grid tracking visited areas"""
        # Convert drone position to grid coordinates
        grid_x = int(np.clip((self.drone_pos[0] + 1) / 7 * self.grid_size, 0, self.grid_size - 1))
        grid_y = int(np.clip((self.drone_pos[1] + 1) / 7 * self.grid_size, 0, self.grid_size - 1))
        
        # Mark as visited
        self.visited_grid[grid_x, grid_y] += 1

    def _get_observation(self):
        """Enhanced observation including exploration info"""
        # Basic positions
        obs_components = [
            self.drone_pos,      # 3 elements
            self.goal_pos,       # 3 elements
        ]
        
        # All threat positions
        for threat_pos in self.threat_positions:
            obs_components.append(threat_pos)  # 3 elements each
        
        # Exploration information
        total_visited = np.sum(self.visited_grid > 0)
        exploration_coverage = total_visited / (self.grid_size * self.grid_size)
        
        # Nearby area information (4 quadrants around drone)
        nearby_info = self._get_nearby_area_info()
        
        obs_components.extend([
            nearby_info,                    # 4 elements
            [exploration_coverage]          # 1 element
        ])
        
        # Flatten and return
        obs = np.concatenate(obs_components).astype(np.float32)
        return obs

    def _get_nearby_area_info(self):
        """Get information about nearby areas (obstacles, visited status)"""
        drone_x, drone_y = self.drone_pos[:2]
        
        # Check 4 directions from drone position
        directions = [(0.5, 0), (-0.5, 0), (0, 0.5), (0, -0.5)]  # right, left, up, down
        area_info = []
        
        for dx, dy in directions:
            check_pos = np.array([drone_x + dx, drone_y + dy, self.drone_pos[2]])
            
            # Check if area has threats nearby
            threat_nearby = 0
            for threat_pos in self.threat_positions:
                if np.linalg.norm(check_pos - threat_pos) < 1.0:
                    threat_nearby = 1
                    break
            
            area_info.append(threat_nearby)
        
        return area_info

    def _get_reward(self):
        """Enhanced reward function encouraging exploration and path diversity"""
        distance_to_goal = np.linalg.norm(self.drone_pos - self.goal_pos)
        
        reward = 0.0
        
        # 1. Basic distance reward (reduced weight)
        reward -= distance_to_goal * 0.05
        
        # 2. Threat penalties (multiple threats)
        threat_penalty = 0.0
        min_threat_distance = float('inf')
        
        for threat_pos in self.threat_positions:
            threat_distance = np.linalg.norm(self.drone_pos - threat_pos)
            min_threat_distance = min(min_threat_distance, threat_distance)
            
            if threat_distance < 1.0:
                threat_penalty += 15.0 * (1.0 - threat_distance)  # Scaled penalty
            elif threat_distance < 1.5:
                threat_penalty += 2.0 * (1.5 - threat_distance)   # Warning zone
        
        reward -= threat_penalty
        
        # 3. Exploration bonus
        if self.encourage_exploration:
            # Reward for visiting new areas
            current_coverage = np.sum(self.visited_grid > 0) / (self.grid_size * self.grid_size)
            exploration_reward = current_coverage * self.exploration_bonus_scale
            reward += exploration_reward
            
            # Bonus for maintaining distance from all threats
            if min_threat_distance > 1.5:
                reward += 0.5  # Safe navigation bonus
        
        # 4. Path diversity bonus
        if len(self.trajectory) > 10:
            path_diversity = self._calculate_path_diversity()
            reward += path_diversity * 0.1
        
        # 5. Goal achievement
        if distance_to_goal < 0.5:
            # Big reward for reaching goal
            base_goal_reward = 150.0
            
            # Bonus for efficient exploration
            efficiency_bonus = (1.0 - min(1.0, len(self.trajectory) / 500)) * 25.0
            
            # Bonus for path diversity
            diversity_bonus = self._calculate_final_path_diversity() * 10.0
            
            reward += base_goal_reward + efficiency_bonus + diversity_bonus
        
        # 6. Small step penalty (reduced)
        reward -= 0.005
        
        return reward

    def _calculate_path_diversity(self):
        """Calculate how diverse the current path is"""
        if len(self.trajectory) < 5:
            return 0.0
        
        # Calculate path variation using direction changes
        directions = []
        for i in range(len(self.trajectory) - 1):
            direction = self.trajectory[i + 1] - self.trajectory[i]
            if np.linalg.norm(direction) > 0:
                directions.append(direction / np.linalg.norm(direction))
        
        if len(directions) < 2:
            return 0.0
        
        # Reward direction changes (non-straight-line paths)
        direction_changes = 0
        for i in range(len(directions) - 1):
            dot_product = np.dot(directions[i], directions[i + 1])
            if dot_product < 0.8:  # Significant direction change
                direction_changes += 1
        
        return direction_changes / len(directions)

    def _calculate_final_path_diversity(self):
        """Calculate final path diversity bonus when goal is reached"""
        if len(self.previous_paths) == 0:
            # First successful path
            self.previous_paths.append(self.trajectory.copy())
            return 0.0
        
        # Compare with previous successful paths
        current_path_array = np.array(self.trajectory)
        max_diversity = 0.0
        
        for prev_path in self.previous_paths[-3:]:  # Compare with last 3 paths
            if len(prev_path) == 0:
                continue
                
            prev_path_array = np.array(prev_path)
            
            # Calculate average distance between path points
            min_len = min(len(current_path_array), len(prev_path_array))
            if min_len > 5:
                # Sample points along both paths
                current_samples = current_path_array[::max(1, len(current_path_array)//10)]
                prev_samples = prev_path_array[::max(1, len(prev_path_array)//10)]
                
                # Calculate average distance between sampled points
                total_distance = 0
                comparisons = 0
                for cp in current_samples:
                    min_dist = min([np.linalg.norm(cp - pp) for pp in prev_samples])
                    total_distance += min_dist
                    comparisons += 1
                
                if comparisons > 0:
                    avg_distance = total_distance / comparisons
                    diversity = min(1.0, avg_distance / 2.0)  # Normalize
                    max_diversity = max(max_diversity, diversity)
        
        # Store current path for future comparisons
        self.previous_paths.append(self.trajectory.copy())
        if len(self.previous_paths) > 5:  # Keep only recent paths
            self.previous_paths.pop(0)
        
        return max_diversity

    def _check_done(self):
        """Check termination conditions"""
        distance_to_goal = np.linalg.norm(self.drone_pos - self.goal_pos)
        
        # Goal reached
        if distance_to_goal < 0.5:
            return True
        
        # Crashed into any threat
        for threat_pos in self.threat_positions:
            if np.linalg.norm(self.drone_pos - threat_pos) < 0.3:
                return True
        
        # Out of bounds
        if (np.any(self.drone_pos[:2] < -2) or 
            np.any(self.drone_pos[:2] > 8) or 
            self.drone_pos[2] < 0 or 
            self.drone_pos[2] > 3):
            return True
        
        return False

    def render(self, mode='human'):
        """Simple text rendering of current state"""
        if mode == 'human':
            print(f"Step {self.current_step}: Drone at {self.drone_pos}")
            print(f"  Goal: {self.goal_pos}, Distance: {np.linalg.norm(self.drone_pos - self.goal_pos):.2f}")
            print(f"  Threats: {len(self.threat_positions)}, Closest: {min([np.linalg.norm(self.drone_pos - t) for t in self.threat_positions]):.2f}")
            print(f"  Exploration: {np.sum(self.visited_grid > 0) / (self.grid_size * self.grid_size) * 100:.1f}%")
            print()
