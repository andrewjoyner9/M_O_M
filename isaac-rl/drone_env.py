import gymnasium as gym
import numpy as np

class DroneEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)  # vx, vy, vz
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)  # customize

        # Simple simulation state
        self.drone_pos = np.array([0.0, 0.0, 1.0])  # x, y, z
        self.goal_pos = np.array([5.0, 5.0, 1.0])   # goal position
        self.threat_pos = np.array([2.5, 2.5, 1.0]) # threat position
        self.max_steps = 1000
        self.current_step = 0
        
        # TODO: connect to Isaac Sim, load scene, get drone, threats, goal

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Reset drone position and environment
        self.drone_pos = np.array([0.0, 0.0, 1.0])
        self.goal_pos = np.array([5.0, 5.0, 1.0])
        self.threat_pos = np.array([2.5, 2.5, 1.0])
        self.current_step = 0
        
        obs = self._get_observation()
        info = {}
        return obs, info

    def step(self, action):
        # Apply action to drone (simple physics simulation)
        self.drone_pos += action * 0.1  # scale the action
        self.current_step += 1
        
        obs = self._get_observation()
        reward = self._get_reward()
        terminated = self._check_done()
        truncated = self.current_step >= self.max_steps
        info = {}
        return obs, reward, terminated, truncated, info

    def _get_observation(self):
        # Return state vector: [drone_x, drone_y, drone_z, goal_x, goal_y, goal_z, threat_x, threat_y, threat_z, distance_to_goal]
        distance_to_goal = np.linalg.norm(self.drone_pos - self.goal_pos)
        obs = np.concatenate([
            self.drone_pos,      # 3 elements
            self.goal_pos,       # 3 elements  
            self.threat_pos,     # 3 elements
            [distance_to_goal]   # 1 element
        ]).astype(np.float32)
        return obs

    def _get_reward(self):
        # Calculate reward based on distance to goal and threats
        distance_to_goal = np.linalg.norm(self.drone_pos - self.goal_pos)
        distance_to_threat = np.linalg.norm(self.drone_pos - self.threat_pos)
        
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

    def _check_done(self):
        # Check if goal reached or crashed into threat
        distance_to_goal = np.linalg.norm(self.drone_pos - self.goal_pos)
        distance_to_threat = np.linalg.norm(self.drone_pos - self.threat_pos)
        
        # Goal reached
        if distance_to_goal < 0.5:
            return True
            
        # Crashed into threat
        if distance_to_threat < 0.5:
            return True
            
        return False
