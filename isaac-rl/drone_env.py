import gym
import numpy as np

class DroneEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)  # vx, vy, vz
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)  # customize

        # TODO: connect to Isaac Sim, load scene, get drone, threats, goal

    def reset(self):
        # TODO: Reset drone + threats + goal
        return self._get_observation()

    def step(self, action):
        # TODO: Apply action to drone in Isaac Sim
        # e.g. move drone in x/y/z
        # Get new state, calculate reward, check if done
        obs = self._get_observation()
        reward = self._get_reward()
        done = self._check_done()
        return obs, reward, done, {}

    def _get_observation(self):
        # Return state vector like [drone_x, drone_y, goal_x, goal_y, threat1_x, threat1_y, ...]
        return np.array([...], dtype=np.float32)

    def _get_reward(self):
        # +1 if close to goal, -1 if hit threat, small penalty per step
        return reward

    def _check_done(self):
        # True if goal reached or crash
        return done
