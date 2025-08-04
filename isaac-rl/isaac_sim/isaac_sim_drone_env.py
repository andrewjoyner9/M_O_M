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
    print("Isaac Sim detected - Using full physics simulation")
except ImportError:
    ISAAC_SIM_AVAILABLE = False
    print("Isaac Sim not available - Using simple simulation")

class IsaacSimDroneEnv(gym.Env):
    """
    Drone RL Environment integrated with Isaac Sim for realistic physics.
    Falls back to simple simulation if Isaac Sim is not available.
    """
    
    def __init__(self, use_isaac_sim=True):
        super().__init__()
        
        # Action and observation spaces
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)  # vx, vy, vz
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
        
        # Environment settings
        self.max_steps = 1000
        self.current_step = 0
        self.use_isaac_sim = use_isaac_sim and ISAAC_SIM_AVAILABLE
        
        # Initialize Isaac Sim or fallback
        if self.use_isaac_sim:
            self._init_isaac_sim()
        else:
            self._init_simple_sim()
    
    def _init_isaac_sim(self):
        """Initialize Isaac Sim world and objects"""
        try:
            # Create or get existing world
            self.world = World.instance()
            if self.world is None:
                self.world = World(stage_units_in_meters=1.0)
            
            # Clear existing objects
            if self.world.scene.object_exists("drone"):
                self.world.scene.remove_object("drone")
            if self.world.scene.object_exists("goal"):
                self.world.scene.remove_object("goal")
            if self.world.scene.object_exists("threat"):
                self.world.scene.remove_object("threat")
            
            # Create drone (dynamic cube that can be controlled)
            self.drone = DynamicCuboid(
                prim_path="/World/Drone",
                name="drone",
                position=np.array([0.0, 0.0, 1.0]),
                size=np.array([0.2, 0.2, 0.1]),  # Small drone-like dimensions
                color=np.array([0.0, 0.0, 1.0])  # Blue
            )
            
            # Create goal (static visual marker)
            self.goal = DynamicCuboid(
                prim_path="/World/Goal",
                name="goal", 
                position=np.array([5.0, 5.0, 1.0]),
                size=np.array([0.3, 0.3, 0.3]),
                color=np.array([0.0, 1.0, 0.0])  # Green
            )
            
            # Create threat (static visual marker)  
            self.threat = DynamicCuboid(
                prim_path="/World/Threat",
                name="threat",
                position=np.array([2.5, 2.5, 1.0]),
                size=np.array([0.4, 0.4, 0.4]),
                color=np.array([1.0, 0.0, 0.0])  # Red
            )
            
            # Add objects to scene
            self.world.scene.add(self.drone)
            self.world.scene.add(self.goal)
            self.world.scene.add(self.threat)
            
            # Reset world
            self.world.reset()
            
            print("Isaac Sim environment initialized successfully")
            
        except Exception as e:
            print(f"Failed to initialize Isaac Sim: {e}")
            print("Falling back to simple simulation")
            self.use_isaac_sim = False
            self._init_simple_sim()
    
    def _init_simple_sim(self):
        """Initialize simple simulation (fallback)"""
        # Simple simulation state
        self.drone_pos = np.array([0.0, 0.0, 1.0])
        self.goal_pos = np.array([5.0, 5.0, 1.0])
        self.threat_pos = np.array([2.5, 2.5, 1.0])
        print("Simple simulation initialized")
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        
        if self.use_isaac_sim:
            return self._reset_isaac_sim()
        else:
            return self._reset_simple_sim()
    
    def _reset_isaac_sim(self):
        """Reset Isaac Sim environment"""
        try:
            # Reset world
            self.world.reset()
            
            # Reset positions
            self.drone.set_world_pose(position=np.array([0.0, 0.0, 1.0]))
            self.goal.set_world_pose(position=np.array([5.0, 5.0, 1.0]))  
            self.threat.set_world_pose(position=np.array([2.5, 2.5, 1.0]))
            
            # Reset velocities
            self.drone.set_linear_velocity(np.array([0.0, 0.0, 0.0]))
            self.drone.set_angular_velocity(np.array([0.0, 0.0, 0.0]))
            
            obs = self._get_observation_isaac_sim()
            return obs, {}
            
        except Exception as e:
            print(f"Error in Isaac Sim reset: {e}")
            return self._reset_simple_sim()
    
    def _reset_simple_sim(self):
        """Reset simple simulation"""
        self.drone_pos = np.array([0.0, 0.0, 1.0])
        self.goal_pos = np.array([5.0, 5.0, 1.0])
        self.threat_pos = np.array([2.5, 2.5, 1.0])
        
        obs = self._get_observation_simple_sim()
        return obs, {}
    
    def step(self, action):
        self.current_step += 1
        
        if self.use_isaac_sim:
            return self._step_isaac_sim(action)
        else:
            return self._step_simple_sim(action)
    
    def _step_isaac_sim(self, action):
        """Step function for Isaac Sim"""
        try:
            # Apply velocity action to drone
            velocity = action * 2.0  # Scale action
            self.drone.set_linear_velocity(velocity)
            
            # Step physics simulation
            self.world.step(render=True)
            
            # Get new observation
            obs = self._get_observation_isaac_sim()
            reward = self._get_reward_isaac_sim()
            terminated = self._check_done_isaac_sim()
            truncated = self.current_step >= self.max_steps
            
            return obs, reward, terminated, truncated, {}
            
        except Exception as e:
            print(f"Error in Isaac Sim step: {e}")
            return self._step_simple_sim(action)
    
    def _step_simple_sim(self, action):
        """Step function for simple simulation"""
        # Apply action to drone (simple physics simulation)
        self.drone_pos += action * 0.1  # Scale the action
        
        obs = self._get_observation_simple_sim()
        reward = self._get_reward_simple_sim()
        terminated = self._check_done_simple_sim()
        truncated = self.current_step >= self.max_steps
        
        return obs, reward, terminated, truncated, {}
    
    def _get_observation_isaac_sim(self):
        """Get observation from Isaac Sim"""
        try:
            # Get positions from Isaac Sim
            drone_pos, _ = self.drone.get_world_pose()
            goal_pos, _ = self.goal.get_world_pose()
            threat_pos, _ = self.threat.get_world_pose()
            
            # Calculate distance to goal
            distance_to_goal = np.linalg.norm(drone_pos - goal_pos)
            
            # Build observation vector
            obs = np.concatenate([
                drone_pos,         # 3 elements
                goal_pos,          # 3 elements
                threat_pos,        # 3 elements
                [distance_to_goal] # 1 element
            ]).astype(np.float32)
            
            return obs
            
        except Exception as e:
            print(f"Error getting Isaac Sim observation: {e}")
            return self._get_observation_simple_sim()
    
    def _get_observation_simple_sim(self):
        """Get observation from simple simulation"""
        distance_to_goal = np.linalg.norm(self.drone_pos - self.goal_pos)
        obs = np.concatenate([
            self.drone_pos,      # 3 elements
            self.goal_pos,       # 3 elements  
            self.threat_pos,     # 3 elements
            [distance_to_goal]   # 1 element
        ]).astype(np.float32)
        return obs
    
    def _get_reward_isaac_sim(self):
        """Calculate reward for Isaac Sim"""
        try:
            drone_pos, _ = self.drone.get_world_pose()
            goal_pos, _ = self.goal.get_world_pose()
            threat_pos, _ = self.threat.get_world_pose()
            
            return self._calculate_reward(drone_pos, goal_pos, threat_pos)
            
        except Exception as e:
            print(f"Error calculating Isaac Sim reward: {e}")
            return self._get_reward_simple_sim()
    
    def _get_reward_simple_sim(self):
        """Calculate reward for simple simulation"""
        return self._calculate_reward(self.drone_pos, self.goal_pos, self.threat_pos)
    
    def _calculate_reward(self, drone_pos, goal_pos, threat_pos):
        """Common reward calculation"""
        distance_to_goal = np.linalg.norm(drone_pos - goal_pos)
        distance_to_threat = np.linalg.norm(drone_pos - threat_pos)
        
        reward = 0.0
        
        # Reward for getting closer to goal
        reward -= distance_to_goal * 0.1
        
        # Penalty for being close to threat
        if distance_to_threat < 1.0:
            reward -= 10.0
            
        # Big reward for reaching goal
        if distance_to_goal < 0.5:
            reward += 100.0
            
        # Small penalty per step to encourage efficiency
        reward -= 0.01
        
        return reward
    
    def _check_done_isaac_sim(self):
        """Check termination for Isaac Sim"""
        try:
            drone_pos, _ = self.drone.get_world_pose()
            goal_pos, _ = self.goal.get_world_pose()
            threat_pos, _ = self.threat.get_world_pose()
            
            return self._check_termination(drone_pos, goal_pos, threat_pos)
            
        except Exception as e:
            print(f"Error checking Isaac Sim termination: {e}")
            return self._check_done_simple_sim()
    
    def _check_done_simple_sim(self):
        """Check termination for simple simulation"""
        return self._check_termination(self.drone_pos, self.goal_pos, self.threat_pos)
    
    def _check_termination(self, drone_pos, goal_pos, threat_pos):
        """Common termination logic"""
        distance_to_goal = np.linalg.norm(drone_pos - goal_pos)
        distance_to_threat = np.linalg.norm(drone_pos - threat_pos)
        
        # Goal reached
        if distance_to_goal < 0.5:
            return True
            
        # Crashed into threat
        if distance_to_threat < 0.5:
            return True
            
        # Check if drone is out of bounds (optional)
        if np.any(np.abs(drone_pos) > 10):
            return True
            
        return False
    
    def close(self):
        """Clean up environment"""
        if self.use_isaac_sim and hasattr(self, 'world'):
            try:
                self.world.stop()
            except:
                pass

# Convenience function to create environment
def create_drone_env(use_isaac_sim=True):
    """Create drone environment with optional Isaac Sim integration"""
    return IsaacSimDroneEnv(use_isaac_sim=use_isaac_sim)
