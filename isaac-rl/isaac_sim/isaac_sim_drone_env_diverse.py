import gymnasium as gym
import numpy as np

# Isaac Sim Integration
try:
    # Isaac Sim imports
    from pxr import UsdGeom, Gf
    import omni
    from omni.isaac.core import World
    from omni.isaac.core.objects import DynamicCuboid
    from omni.isaac.core.utils.prims import create_prim
    from omni.isaac.core.utils.stage import add_reference_to_stage
    ISAAC_SIM_AVAILABLE = True
    print("Isaac Sim detected - Using full physics simulation with path diversity")
except ImportError:
    ISAAC_SIM_AVAILABLE = False
    print("Isaac Sim not available - Using enhanced diverse simulation")

class DiverseIsaacSimDroneEnv(gym.Env):
    """
    Enhanced Isaac Sim Drone Environment with path diversity and exploration incentives.
    
    Improvements over original:
    1. Randomized start/goal positions
    2. Multiple dynamic threats
    3. Path diversity rewards
    4. Exploration bonuses
    5. Environmental randomization
    """
    
    def __init__(self, use_isaac_sim=True, encourage_exploration=True, num_threats=4):
        super().__init__()
        
        # Enhanced observation and action spaces
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        
        # Expanded observation space: [drone_pos(3), goal_pos(3), threats(3*num_threats), 
        #                             exploration_info(5), velocity(3)]
        obs_size = 3 + 3 + (3 * num_threats) + 5 + 3
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)
        
        # Environment settings
        self.num_threats = num_threats
        self.encourage_exploration = encourage_exploration
        self.max_steps = 2000
        self.current_step = 0
        self.use_isaac_sim = use_isaac_sim and ISAAC_SIM_AVAILABLE
        
        # Path tracking and diversity
        self.trajectory = []
        self.previous_successful_paths = []
        self.visited_areas = set()
        self.exploration_radius = 0.5
        
        # Dynamic environment parameters
        self.threat_movement_speed = 0.03
        self.environmental_randomization = True
        self.wind_effects = True
        
        # Initialize environment
        if self.use_isaac_sim:
            self._init_isaac_sim()
        else:
            self._init_enhanced_sim()
    
    def _init_isaac_sim(self):
        """Initialize enhanced Isaac Sim environment"""
        try:
            # Create or get existing world
            self.world = World.instance()
            if self.world is None:
                self.world = World(stage_units_in_meters=1.0)
            
            # Clear existing objects
            objects_to_clear = ["drone", "goal"] + [f"threat_{i}" for i in range(10)]
            for obj_name in objects_to_clear:
                if self.world.scene.object_exists(obj_name):
                    self.world.scene.remove_object(obj_name)
            
            # Create enhanced drone with better physics properties
            self.drone = DynamicCuboid(
                prim_path="/World/Drone",
                name="drone",
                position=np.array([0.0, 0.0, 1.0]),
                size=np.array([0.15, 0.15, 0.08]),  # Smaller, more agile
                color=np.array([0.2, 0.2, 1.0]),    # Blue
                linear_velocity=np.array([0.0, 0.0, 0.0])
            )
            
            # Create goal with visual enhancement
            self.goal = DynamicCuboid(
                prim_path="/World/Goal",
                name="goal",
                position=np.array([5.0, 5.0, 1.0]),
                size=np.array([0.4, 0.4, 0.4]),
                color=np.array([0.0, 1.0, 0.0])     # Green
            )
            
            # Create multiple threats with different properties
            self.threats = []
            self.threat_velocities = []
            
            for i in range(self.num_threats):
                threat = DynamicCuboid(
                    prim_path=f"/World/Threat_{i}",
                    name=f"threat_{i}",
                    position=np.array([2.0 + i * 0.5, 2.0 + i * 0.5, 1.0]),
                    size=np.array([0.3, 0.3, 0.3]),
                    color=np.array([1.0, 0.2, 0.2])  # Red
                )
                self.threats.append(threat)
                self.threat_velocities.append(np.zeros(3))
            
            # Add all objects to scene
            self.world.scene.add(self.drone)
            self.world.scene.add(self.goal)
            for threat in self.threats:
                self.world.scene.add(threat)
            
            # Reset world
            self.world.reset()
            print("Enhanced Isaac Sim environment initialized successfully")
            
        except Exception as e:
            print(f"Failed to initialize Isaac Sim: {e}")
            print("Falling back to enhanced simulation")
            self.use_isaac_sim = False
            self._init_enhanced_sim()
    
    def _init_enhanced_sim(self):
        """Initialize enhanced simple simulation"""
        # Enhanced simulation state
        self.drone_pos = np.array([0.0, 0.0, 1.0])
        self.drone_velocity = np.array([0.0, 0.0, 0.0])
        self.goal_pos = np.array([5.0, 5.0, 1.0])
        
        # Multiple threats with individual properties
        self.threat_positions = []
        self.threat_velocities = []
        for i in range(self.num_threats):
            pos = np.array([2.0 + i * 0.8, 2.0 + i * 0.8, 1.0])
            vel = np.zeros(3)
            self.threat_positions.append(pos)
            self.threat_velocities.append(vel)
        
        print("Enhanced simulation initialized")
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        
        self.current_step = 0
        self.trajectory = []
        self.visited_areas = set()
        
        if self.use_isaac_sim:
            return self._reset_isaac_sim()
        else:
            return self._reset_enhanced_sim()
    
    def _reset_isaac_sim(self):
        """Reset Isaac Sim with randomization"""
        try:
            # Randomize drone start position
            start_x = np.random.uniform(-1.0, 1.0)
            start_y = np.random.uniform(-1.0, 1.0)
            start_z = np.random.uniform(0.8, 1.2)
            start_pos = np.array([start_x, start_y, start_z])
            
            # Randomize goal position
            goal_x = np.random.uniform(4.0, 6.0)
            goal_y = np.random.uniform(4.0, 6.0)
            goal_z = np.random.uniform(0.8, 1.2)
            goal_pos = np.array([goal_x, goal_y, goal_z])
            
            # Reset world first
            self.world.reset()
            
            # Set positions
            self.drone.set_world_pose(position=start_pos)
            self.goal.set_world_pose(position=goal_pos)
            
            # Reset drone dynamics
            self.drone.set_linear_velocity(np.array([0.0, 0.0, 0.0]))
            self.drone.set_angular_velocity(np.array([0.0, 0.0, 0.0]))
            
            # Randomize threat positions and initialize movement
            for i, threat in enumerate(self.threats):
                # Random threat position in middle area
                threat_x = np.random.uniform(1.5, 4.5)
                threat_y = np.random.uniform(1.5, 4.5)
                threat_z = np.random.uniform(0.8, 1.2)
                threat_pos = np.array([threat_x, threat_y, threat_z])
                
                threat.set_world_pose(position=threat_pos)
                threat.set_linear_velocity(np.array([0.0, 0.0, 0.0]))
                
                # Random initial movement direction
                vel = np.random.uniform(-self.threat_movement_speed, self.threat_movement_speed, 3)
                vel[2] *= 0.3  # Reduce vertical movement
                self.threat_velocities[i] = vel
            
            obs = self._get_observation_isaac_sim()
            info = {'randomized_start': True, 'num_threats': len(self.threats)}
            return obs, info
            
        except Exception as e:
            print(f"Error in Isaac Sim reset: {e}")
            return self._reset_enhanced_sim()
    
    def _reset_enhanced_sim(self):
        """Reset enhanced simulation with full randomization"""
        # Randomize drone start
        self.drone_pos = np.array([
            np.random.uniform(-1.0, 1.0),
            np.random.uniform(-1.0, 1.0), 
            np.random.uniform(0.8, 1.2)
        ])
        self.drone_velocity = np.zeros(3)
        
        # Randomize goal
        self.goal_pos = np.array([
            np.random.uniform(4.0, 6.0),
            np.random.uniform(4.0, 6.0),
            np.random.uniform(0.8, 1.2)
        ])
        
        # Randomize threats
        for i in range(len(self.threat_positions)):
            self.threat_positions[i] = np.array([
                np.random.uniform(1.5, 4.5),
                np.random.uniform(1.5, 4.5),
                np.random.uniform(0.8, 1.2)
            ])
            
            # Random movement pattern
            vel = np.random.uniform(-self.threat_movement_speed, self.threat_movement_speed, 3)
            vel[2] *= 0.3  # Reduce vertical movement
            self.threat_velocities[i] = vel
        
        obs = self._get_observation_enhanced_sim()
        info = {'randomized_start': True, 'num_threats': len(self.threat_positions)}
        return obs, info
    
    def step(self, action):
        self.current_step += 1
        
        # Add exploration noise during training
        if self.encourage_exploration and np.random.random() < 0.05:
            action = action + np.random.normal(0, 0.03, 3)
            action = np.clip(action, -1, 1)
        
        if self.use_isaac_sim:
            return self._step_isaac_sim(action)
        else:
            return self._step_enhanced_sim(action)
    
    def _step_isaac_sim(self, action):
        """Enhanced Isaac Sim step with environmental effects"""
        try:
            # Apply action with environmental effects
            velocity = action * 2.5  # Increased responsiveness
            
            # Add wind effects if enabled
            if self.wind_effects:
                wind = np.random.normal(0, 0.05, 3)  # Small random wind
                velocity += wind
            
            self.drone.set_linear_velocity(velocity)
            
            # Update threat movements
            for i, (threat, vel) in enumerate(zip(self.threats, self.threat_velocities)):
                # Apply threat movement
                current_pos, _ = threat.get_world_pose()
                new_velocity = vel.copy()
                
                # Occasionally change direction
                if np.random.random() < 0.02:
                    new_velocity = np.random.uniform(-self.threat_movement_speed, 
                                                   self.threat_movement_speed, 3)
                    new_velocity[2] *= 0.3
                    self.threat_velocities[i] = new_velocity
                
                # Boundary avoidance
                if current_pos[0] < 0.5 or current_pos[0] > 6.5:
                    new_velocity[0] *= -1
                if current_pos[1] < 0.5 or current_pos[1] > 6.5:
                    new_velocity[1] *= -1
                if current_pos[2] < 0.3 or current_pos[2] > 2.0:
                    new_velocity[2] *= -1
                
                threat.set_linear_velocity(new_velocity)
                self.threat_velocities[i] = new_velocity
            
            # Step physics
            self.world.step(render=True)
            
            # Update trajectory
            drone_pos, _ = self.drone.get_world_pose()
            self.trajectory.append(drone_pos.copy())
            self._update_visited_areas(drone_pos)
            
            obs = self._get_observation_isaac_sim()
            reward = self._get_enhanced_reward_isaac_sim()
            terminated = self._check_done_isaac_sim()
            truncated = self.current_step >= self.max_steps
            
            info = {
                'exploration_areas': len(self.visited_areas),
                'path_length': len(self.trajectory),
                'threat_distances': [np.linalg.norm(drone_pos - t.get_world_pose()[0]) for t in self.threats]
            }
            
            return obs, reward, terminated, truncated, info
            
        except Exception as e:
            print(f"Error in Isaac Sim step: {e}")
            return self._step_enhanced_sim(action)
    
    def _step_enhanced_sim(self, action):
        """Enhanced simulation step"""
        # Physics with momentum
        acceleration = action * 0.2
        self.drone_velocity = self.drone_velocity * 0.9 + acceleration  # Add momentum
        self.drone_pos += self.drone_velocity
        
        # Update threat movements
        for i, (pos, vel) in enumerate(zip(self.threat_positions, self.threat_velocities)):
            # Update threat position
            new_pos = pos + vel
            
            # Boundary checking and direction changes
            if new_pos[0] < 0.5 or new_pos[0] > 6.5:
                vel[0] *= -1
            if new_pos[1] < 0.5 or new_pos[1] > 6.5:
                vel[1] *= -1
            if new_pos[2] < 0.3 or new_pos[2] > 2.0:
                vel[2] *= -1
            
            # Occasional random direction change
            if np.random.random() < 0.02:
                vel = np.random.uniform(-self.threat_movement_speed, self.threat_movement_speed, 3)
                vel[2] *= 0.3
            
            self.threat_positions[i] = np.clip(new_pos, [0.2, 0.2, 0.2], [6.8, 6.8, 2.0])
            self.threat_velocities[i] = vel
        
        # Update trajectory and exploration
        self.trajectory.append(self.drone_pos.copy())
        self._update_visited_areas(self.drone_pos)
        
        obs = self._get_observation_enhanced_sim()
        reward = self._get_enhanced_reward_enhanced_sim()
        terminated = self._check_done_enhanced_sim()
        truncated = self.current_step >= self.max_steps
        
        info = {
            'exploration_areas': len(self.visited_areas),
            'path_length': len(self.trajectory),
            'threat_distances': [np.linalg.norm(self.drone_pos - t) for t in self.threat_positions]
        }
        
        return obs, reward, terminated, truncated, info
    
    def _update_visited_areas(self, position):
        """Track visited areas for exploration bonus"""
        # Discretize position into grid cells
        grid_x = int(position[0] / self.exploration_radius)
        grid_y = int(position[1] / self.exploration_radius)
        grid_z = int(position[2] / self.exploration_radius)
        self.visited_areas.add((grid_x, grid_y, grid_z))
    
    def _get_observation_isaac_sim(self):
        """Enhanced observation for Isaac Sim"""
        try:
            drone_pos, _ = self.drone.get_world_pose()
            drone_vel, _ = self.drone.get_linear_velocity()
            goal_pos, _ = self.goal.get_world_pose()
            
            # Get all threat positions
            threat_positions = []
            for threat in self.threats:
                threat_pos, _ = threat.get_world_pose()
                threat_positions.extend(threat_pos)
            
            # Exploration info
            exploration_info = self._get_exploration_info(drone_pos)
            
            # Build observation
            obs = np.concatenate([
                drone_pos,           # 3
                goal_pos,           # 3
                threat_positions,   # 3 * num_threats
                exploration_info,   # 5
                drone_vel          # 3
            ]).astype(np.float32)
            
            return obs
            
        except Exception as e:
            print(f"Error getting Isaac Sim observation: {e}")
            return self._get_observation_enhanced_sim()
    
    def _get_observation_enhanced_sim(self):
        """Enhanced observation for simulation"""
        # All threat positions flattened
        threat_positions = []
        for threat_pos in self.threat_positions:
            threat_positions.extend(threat_pos)
        
        # Exploration info
        exploration_info = self._get_exploration_info(self.drone_pos)
        
        # Build observation
        obs = np.concatenate([
            self.drone_pos,        # 3
            self.goal_pos,         # 3
            threat_positions,      # 3 * num_threats
            exploration_info,      # 5
            self.drone_velocity    # 3
        ]).astype(np.float32)
        
        return obs
    
    def _get_exploration_info(self, drone_pos):
        """Get exploration-related information"""
        # Distance to goal
        dist_to_goal = np.linalg.norm(drone_pos - self.goal_pos)
        
        # Exploration coverage (normalized)
        exploration_coverage = min(1.0, len(self.visited_areas) / 100.0)
        
        # Path diversity metric (simplified)
        path_diversity = self._calculate_current_path_diversity()
        
        # Safety metric (average distance to threats)
        if self.use_isaac_sim:
            threat_distances = [np.linalg.norm(drone_pos - threat.get_world_pose()[0]) 
                              for threat in self.threats]
        else:
            threat_distances = [np.linalg.norm(drone_pos - threat_pos) 
                              for threat_pos in self.threat_positions]
        
        avg_threat_distance = np.mean(threat_distances) if threat_distances else 5.0
        safety_metric = min(1.0, avg_threat_distance / 3.0)
        
        # Recent movement diversity
        movement_diversity = self._calculate_recent_movement_diversity()
        
        return [dist_to_goal, exploration_coverage, path_diversity, safety_metric, movement_diversity]
    
    def _calculate_current_path_diversity(self):
        """Calculate diversity of current path"""
        if len(self.trajectory) < 10:
            return 0.0
        
        # Calculate direction changes in recent path
        recent_path = self.trajectory[-20:]  # Last 20 steps
        directions = []
        
        for i in range(len(recent_path) - 1):
            direction = recent_path[i + 1] - recent_path[i]
            if np.linalg.norm(direction) > 0:
                directions.append(direction / np.linalg.norm(direction))
        
        if len(directions) < 2:
            return 0.0
        
        # Count significant direction changes
        direction_changes = 0
        for i in range(len(directions) - 1):
            dot_product = np.dot(directions[i], directions[i + 1])
            if dot_product < 0.8:  # Significant change
                direction_changes += 1
        
        return direction_changes / len(directions)
    
    def _calculate_recent_movement_diversity(self):
        """Calculate diversity in recent movement patterns"""
        if len(self.trajectory) < 5:
            return 0.0
        
        recent_positions = self.trajectory[-10:]
        if len(recent_positions) < 3:
            return 0.0
        
        # Calculate variance in movement
        movements = [recent_positions[i+1] - recent_positions[i] 
                    for i in range(len(recent_positions)-1)]
        
        if len(movements) < 2:
            return 0.0
        
        movement_variance = np.var(movements, axis=0)
        return min(1.0, np.mean(movement_variance) * 10)
    
    def _get_enhanced_reward_isaac_sim(self):
        """Enhanced reward function for Isaac Sim"""
        try:
            drone_pos, _ = self.drone.get_world_pose()
            goal_pos, _ = self.goal.get_world_pose()
            threat_positions = [threat.get_world_pose()[0] for threat in self.threats]
            
            return self._calculate_enhanced_reward(drone_pos, goal_pos, threat_positions)
            
        except Exception as e:
            print(f"Error calculating Isaac Sim reward: {e}")
            return self._get_enhanced_reward_enhanced_sim()
    
    def _get_enhanced_reward_enhanced_sim(self):
        """Enhanced reward function for simulation"""
        return self._calculate_enhanced_reward(self.drone_pos, self.goal_pos, self.threat_positions)
    
    def _calculate_enhanced_reward(self, drone_pos, goal_pos, threat_positions):
        """Enhanced reward calculation encouraging diversity and exploration"""
        reward = 0.0
        
        # 1. Distance to goal (reduced weight)
        distance_to_goal = np.linalg.norm(drone_pos - goal_pos)
        reward -= distance_to_goal * 0.03
        
        # 2. Threat avoidance (enhanced for multiple threats)
        min_threat_distance = float('inf')
        total_threat_penalty = 0.0
        
        for threat_pos in threat_positions:
            threat_distance = np.linalg.norm(drone_pos - threat_pos)
            min_threat_distance = min(min_threat_distance, threat_distance)
            
            if threat_distance < 0.8:
                total_threat_penalty += 20.0 * (0.8 - threat_distance)
            elif threat_distance < 1.2:
                total_threat_penalty += 5.0 * (1.2 - threat_distance)
        
        reward -= total_threat_penalty
        
        # 3. Exploration bonus
        exploration_bonus = len(self.visited_areas) * 0.1
        reward += exploration_bonus
        
        # 4. Path diversity bonus
        if len(self.trajectory) > 15:
            diversity_bonus = self._calculate_current_path_diversity() * 2.0
            reward += diversity_bonus
        
        # 5. Movement efficiency vs exploration balance
        if len(self.trajectory) > 5:
            recent_exploration = len(set((int(pos[0]), int(pos[1])) for pos in self.trajectory[-10:]))
            if recent_exploration > 3:  # Reward diverse recent movement
                reward += 1.0
        
        # 6. Safety bonus for maintaining good threat distance
        if min_threat_distance > 1.5:
            reward += 0.8
        elif min_threat_distance > 1.0:
            reward += 0.3
        
        # 7. Goal achievement with bonuses
        if distance_to_goal < 0.5:
            base_reward = 200.0
            
            # Efficiency bonus
            efficiency_bonus = max(0, (2000 - len(self.trajectory)) / 20)
            
            # Exploration bonus  
            exploration_bonus = len(self.visited_areas) * 0.5
            
            # Path uniqueness bonus
            uniqueness_bonus = self._calculate_path_uniqueness_bonus()
            
            total_goal_reward = base_reward + efficiency_bonus + exploration_bonus + uniqueness_bonus
            reward += total_goal_reward
        
        # 8. Small step penalty
        reward -= 0.003
        
        return reward
    
    def _calculate_path_uniqueness_bonus(self):
        """Calculate bonus for taking a unique path"""
        if len(self.previous_successful_paths) == 0:
            return 10.0  # First successful path
        
        current_path_sample = self.trajectory[::max(1, len(self.trajectory)//20)]
        
        max_uniqueness = 0.0
        for prev_path in self.previous_successful_paths[-3:]:  # Compare with recent paths
            if len(prev_path) < 10:
                continue
                
            prev_path_sample = prev_path[::max(1, len(prev_path)//20)]
            
            # Calculate average distance between path samples
            total_distance = 0
            comparisons = 0
            
            for curr_point in current_path_sample:
                min_dist = min([np.linalg.norm(curr_point - prev_point) 
                               for prev_point in prev_path_sample])
                total_distance += min_dist
                comparisons += 1
            
            if comparisons > 0:
                avg_distance = total_distance / comparisons
                uniqueness = min(1.0, avg_distance / 2.0)
                max_uniqueness = max(max_uniqueness, uniqueness)
        
        return max_uniqueness * 15.0
    
    def _check_done_isaac_sim(self):
        """Enhanced termination checking for Isaac Sim"""
        try:
            drone_pos, _ = self.drone.get_world_pose()
            goal_pos, _ = self.goal.get_world_pose()
            
            # Goal reached
            if np.linalg.norm(drone_pos - goal_pos) < 0.5:
                # Store successful path
                self.previous_successful_paths.append(self.trajectory.copy())
                if len(self.previous_successful_paths) > 5:
                    self.previous_successful_paths.pop(0)
                return True
            
            # Collision with threats
            for threat in self.threats:
                threat_pos, _ = threat.get_world_pose()
                if np.linalg.norm(drone_pos - threat_pos) < 0.25:
                    return True
            
            # Out of bounds
            if (np.any(drone_pos[:2] < -1) or np.any(drone_pos[:2] > 8) or 
                drone_pos[2] < 0 or drone_pos[2] > 3):
                return True
            
            return False
            
        except Exception as e:
            print(f"Error checking Isaac Sim termination: {e}")
            return self._check_done_enhanced_sim()
    
    def _check_done_enhanced_sim(self):
        """Enhanced termination checking for simulation"""
        # Goal reached
        if np.linalg.norm(self.drone_pos - self.goal_pos) < 0.5:
            # Store successful path
            self.previous_successful_paths.append(self.trajectory.copy())
            if len(self.previous_successful_paths) > 5:
                self.previous_successful_paths.pop(0)
            return True
        
        # Collision with threats
        for threat_pos in self.threat_positions:
            if np.linalg.norm(self.drone_pos - threat_pos) < 0.25:
                return True
        
        # Out of bounds
        if (np.any(self.drone_pos[:2] < -1) or np.any(self.drone_pos[:2] > 8) or 
            self.drone_pos[2] < 0 or self.drone_pos[2] > 3):
            return True
        
        return False
    
    def close(self):
        """Clean up environment"""
        if self.use_isaac_sim and hasattr(self, 'world'):
            try:
                self.world.stop()
            except:
                pass

# Convenience function
def create_diverse_isaac_env(use_isaac_sim=True, num_threats=4):
    """Create enhanced drone environment with Isaac Sim integration"""
    return DiverseIsaacSimDroneEnv(
        use_isaac_sim=use_isaac_sim, 
        encourage_exploration=True,
        num_threats=num_threats
    )
