from stable_baselines3 import PPO
from drone_env import DroneEnv

env = DroneEnv()
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100_000)
model.save("drone_rl_model")
