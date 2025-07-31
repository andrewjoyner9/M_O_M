from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
from drone_env import DroneEnv
import os

def train_drone_agent():
    # Create environment
    env = DroneEnv()
    env = Monitor(env)
    
    # Create evaluation environment
    eval_env = DroneEnv()
    eval_env = Monitor(eval_env)
    
    # Set up callbacks
    # Stop training when the model reaches the reward threshold
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=50, verbose=1)
    eval_callback = EvalCallback(eval_env, callback_on_new_best=callback_on_best, verbose=1)
    
    # Create PPO model
    model = PPO(
        "MlpPolicy", 
        env, 
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        tensorboard_log="./drone_tensorboard/"
    )
    
    # Train the model
    print("Starting training...")
    model.learn(total_timesteps=100_000, callback=eval_callback)
    
    # Save the model
    model.save("drone_rl_model")
    print("Model saved as drone_rl_model.zip")
    
    return model

def test_trained_model():
    """Test the trained model"""
    env = DroneEnv()
    model = PPO.load("drone_rl_model")
    
    obs, info = env.reset()
    total_reward = 0
    
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        print(f"Step {i+1}: Reward={reward:.3f}, Total={total_reward:.3f}")
        print(f"  Drone: {obs[:3]}, Goal: {obs[3:6]}")
        
        if terminated or truncated:
            print(f"Episode finished! Total reward: {total_reward:.3f}")
            break
    
    return total_reward

if __name__ == "__main__":
    # Train the model
    model = train_drone_agent()
    
    # Test the trained model
    print("\nTesting trained model...")
    test_trained_model()
