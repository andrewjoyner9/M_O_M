#!/usr/bin/env python3
"""
PPO Training Script for Isaac Lab Drone Navigation Task

This script trains a PPO agent to navigate a drone through 3D environments 
with obstacles using the DroneVectorTask from drone_task_lab.py
"""

from __future__ import annotations
import os
import time
import torch
import numpy as np
from typing import Optional

# RL training imports  
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecNormalize
    from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.utils import set_random_seed
    print("✓ Stable Baselines3 found")
except ImportError as e:
    print(f"❌ Error importing Stable Baselines3: {e}")
    print("Please install with: pip install stable-baselines3[extra]")
    exit(1)

# Try Isaac Lab imports
ISAAC_LAB_AVAILABLE = False
try:
    # Test Isaac Lab availability
    import isaaclab
    from isaaclab.utils.io import dump_pickle, dump_yaml
    from isaaclab.utils.dict import print_dict
    from drone_task_lab import DroneVectorTask, DroneLabCfg
    import hydra
    from omegaconf import DictConfig
    ISAAC_LAB_AVAILABLE = True
    print("✓ Isaac Lab imports successful")
except Exception as e:
    print(f"⚠️  Isaac Lab not available: {e}")
    print("Running in standalone mode...")
    
    # Fallback implementations
    def dump_yaml(path, data):
        import yaml
        with open(path, 'w') as f:
            yaml.dump(data, f)
    
    # Create a simple config class
    class DroneLabCfg:
        def __init__(self):
            self.num_envs = 64
            self.episode_length_s = 10.0
            self.decimation = 2
            
        def __dict__(self):
            return {
                'num_envs': self.num_envs,
                'episode_length_s': self.episode_length_s,
                'decimation': self.decimation
            }


class TrainingCallback(BaseCallback):
    """Custom callback for logging training metrics"""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        
    def _on_step(self) -> bool:
        # Log episode metrics when available
        if len(self.locals.get("rewards", [])) > 0:
            self.episode_rewards.extend(self.locals["rewards"])
            
        if len(self.locals.get("dones", [])) > 0:
            done_indices = np.where(self.locals["dones"])[0]
            if len(done_indices) > 0:
                self.episode_lengths.extend([self.locals.get("episode_lengths", [0])[i] 
                                            for i in done_indices])
        
        # Log every 1000 steps
        if self.num_timesteps % 1000 == 0:
            if len(self.episode_rewards) > 0:
                mean_reward = np.mean(self.episode_rewards[-100:])  # Last 100 episodes
                self.logger.record("train/mean_episode_reward", mean_reward)
                
            if len(self.episode_lengths) > 0:
                mean_length = np.mean(self.episode_lengths[-100:])
                self.logger.record("train/mean_episode_length", mean_length)
                
        return True


class IsaacLabVecEnvWrapper:
    """
    Wrapper to make Isaac Lab environment compatible with Stable Baselines3
    """
    
    def __init__(self, env):
        self.env = env
        self.num_envs = getattr(env, 'num_envs', 64)
        self.observation_space = getattr(env, 'observation_space', None)
        self.action_space = getattr(env, 'action_space', None)
        self.device = getattr(env, 'device', torch.device('cpu'))
        
        if ISAAC_LAB_AVAILABLE:
            # Track episode info for Isaac Lab
            self.episode_returns = torch.zeros(self.num_envs, device=self.device)
            self.episode_lengths = torch.zeros(self.num_envs, device=self.device, dtype=torch.int)
        
    def reset(self):
        if ISAAC_LAB_AVAILABLE:
            obs, info = self.env.reset()
            self.episode_returns.zero_()
            self.episode_lengths.zero_()
            return obs.cpu().numpy(), info
        else:
            # Fallback for standalone mode
            return self.env.reset()
        
    def step(self, actions):
        if ISAAC_LAB_AVAILABLE:
            # Convert actions to tensor if needed
            if not isinstance(actions, torch.Tensor):
                actions = torch.tensor(actions, device=self.device, dtype=torch.float32)
                
            obs, rewards, dones, truncated, info = self.env.step(actions)
            
            # Update episode tracking
            self.episode_returns += rewards
            self.episode_lengths += 1
            
            # Handle resets for done environments
            if dones.any():
                done_ids = dones.nonzero(as_tuple=True)[0]
                
                # Add episode info
                for idx in done_ids.cpu().numpy():
                    if "episode" not in info:
                        info["episode"] = {}
                    info["episode"][f"r_{idx}"] = float(self.episode_returns[idx])
                    info["episode"][f"l_{idx}"] = int(self.episode_lengths[idx])
                
                # Reset episode tracking for done envs
                self.episode_returns[done_ids] = 0
                self.episode_lengths[done_ids] = 0
            
            return (
                obs.cpu().numpy(),
                rewards.cpu().numpy(), 
                dones.cpu().numpy(),
                info
            )
        else:
            # Fallback for standalone mode
            return self.env.step(actions)
        
    def close(self):
        if hasattr(self.env, 'close'):
            self.env.close()


# Import standalone environment if Isaac Lab not available
if not ISAAC_LAB_AVAILABLE:
    # Import the standalone environment from our previous work
    import sys
    sys.path.append('/home/omniverse/isaac_projects/M_O_M/scripts')
    
    # Import our standalone drone environment
    try:
        # Use the standalone environment from train_drone_standalone.py
        exec(open('/home/omniverse/isaac_projects/M_O_M/scripts/train_drone_standalone.py').read())
        print("✓ Standalone drone environment loaded")
    except Exception as e:
        print(f"❌ Could not load standalone environment: {e}")
        print("Please ensure train_drone_standalone.py exists")


def create_ppo_config(
    total_timesteps: int = 1_000_000,
    learning_rate: float = 3e-4,
    batch_size: int = 2048,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    use_sde: bool = False,
    device: str = "auto"
) -> dict:
    """Create PPO configuration dictionary"""
    
    return {
        "total_timesteps": total_timesteps,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "n_epochs": n_epochs,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "clip_range": clip_range,
        "ent_coef": ent_coef,
        "vf_coef": vf_coef,
        "max_grad_norm": max_grad_norm,
        "use_sde": use_sde,
        "device": device,
    }


def train_ppo_agent(
    cfg,
    ppo_cfg: dict,
    log_dir: str = "./logs",
    model_save_path: str = "./models",
    seed: Optional[int] = None,
    eval_freq: int = 10000,
    save_freq: int = 50000,
    verbose: int = 1
):
    """
    Train a PPO agent on the drone navigation task
    
    Args:
        cfg: Environment configuration
        ppo_cfg: PPO algorithm configuration
        log_dir: Directory for tensorboard logs
        model_save_path: Directory to save trained models
        seed: Random seed for reproducibility
        eval_freq: Frequency of evaluation runs
        save_freq: Frequency of model checkpoints
        verbose: Verbosity level
    """
    
    # Set random seed
    if seed is not None:
        set_random_seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    # Create directories
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_save_path, exist_ok=True)
    
    print("Creating drone environment...")
    
    # Create environment based on availability
    if ISAAC_LAB_AVAILABLE:
        print("Using Isaac Lab environment...")
        env = DroneVectorTask(cfg)
        env_wrapper = IsaacLabVecEnvWrapper(env)
        print(f"✓ Isaac Lab environment created with {env.num_envs} parallel environments")
    else:
        print("Using standalone environment...")
        # Create standalone environment from our previous implementation
        env = SingleDroneEnv(num_envs=cfg.num_envs if hasattr(cfg, 'num_envs') else 64)
        env_wrapper = env  # Already compatible
        print(f"✓ Standalone environment created with {env.num_envs} parallel environments")
    
    print(f"  Observation space: {env_wrapper.observation_space}")
    print(f"  Action space: {env_wrapper.action_space}")
    
    # Configure logging
    logger = configure(log_dir, ["stdout", "csv", "tensorboard"])
    
    # Calculate n_steps for PPO
    n_steps = ppo_cfg["batch_size"] // env_wrapper.num_envs
    
    # Create PPO model
    print("Creating PPO model...")
    model = PPO(
        "MlpPolicy",
        env_wrapper,
        learning_rate=ppo_cfg["learning_rate"],
        n_steps=n_steps,
        batch_size=ppo_cfg["batch_size"],
        n_epochs=ppo_cfg["n_epochs"],
        gamma=ppo_cfg["gamma"],
        gae_lambda=ppo_cfg["gae_lambda"],
        clip_range=ppo_cfg["clip_range"],
        ent_coef=ppo_cfg["ent_coef"],
        vf_coef=ppo_cfg["vf_coef"],
        max_grad_norm=ppo_cfg["max_grad_norm"],
        use_sde=ppo_cfg["use_sde"],
        device=ppo_cfg["device"],
        verbose=verbose,
        tensorboard_log=log_dir,
    )
    
    model.set_logger(logger)
    print("✓ PPO model created")
    print(f"  Learning rate: {ppo_cfg['learning_rate']}")
    print(f"  Batch size: {ppo_cfg['batch_size']}")
    print(f"  Steps per env: {n_steps}")
    
    # Create callbacks
    callbacks = []
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=model_save_path,
        name_prefix="ppo_drone_checkpoint"
    )
    callbacks.append(checkpoint_callback)
    
    # Custom training callback
    training_callback = TrainingCallback(verbose=1)
    callbacks.append(training_callback)
    
    # Save configuration
    config_save_path = os.path.join(model_save_path, "config.yaml")
    config_data = {
        "env_cfg": cfg.__dict__ if hasattr(cfg, '__dict__') else cfg,
        "ppo_cfg": ppo_cfg,
        "seed": seed,
        "isaac_lab_available": ISAAC_LAB_AVAILABLE
    }
    dump_yaml(config_save_path, config_data)
    print(f"✓ Configuration saved to {config_save_path}")
    
    # Start training
    print(f"Starting PPO training for {ppo_cfg['total_timesteps']:,} timesteps...")
    print(f"Environment: {'Isaac Lab' if ISAAC_LAB_AVAILABLE else 'Standalone'}")
    print(f"Logs: {log_dir} | Models: {model_save_path}")
    print("-" * 80)
    
    start_time = time.time()
    
    try:
        model.learn(
            total_timesteps=ppo_cfg["total_timesteps"],
            callback=callbacks,
            progress_bar=True
        )
        
        print("\n" + "=" * 80)
        print("✓ Training completed successfully!")
        
        # Save final model
        final_model_path = os.path.join(model_save_path, "ppo_drone_final.zip")
        model.save(final_model_path)
        print(f"✓ Final model saved to {final_model_path}")
        
        # Training summary
        training_time = time.time() - start_time
        print(f"Training time: {training_time/3600:.2f} hours")
        print(f"Total episodes: ~{ppo_cfg['total_timesteps'] // (100 * env_wrapper.num_envs):,}")  # Rough estimate
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        interrupt_model_path = os.path.join(model_save_path, "ppo_drone_interrupted.zip")
        model.save(interrupt_model_path)
        print(f"✓ Model saved to {interrupt_model_path}")
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        raise
        
    finally:
        env_wrapper.close()


def evaluate_trained_model(
    model_path: str,
    cfg: DroneLabCfg,
    num_episodes: int = 10,
    render: bool = False
):
    """
    Evaluate a trained PPO model
    
    Args:
        model_path: Path to the trained model
        cfg: Environment configuration  
        num_episodes: Number of episodes to evaluate
        render: Whether to render the environment
    """
    
    print(f"Evaluating model: {model_path}")
    
    # Load model
    model = PPO.load(model_path)
    
    # Create environment
    env = DroneVectorTask(cfg)
    env_wrapper = IsaacLabVecEnvWrapper(env)
    
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    
    obs, _ = env_wrapper.reset()
    
    for episode in range(num_episodes):
        episode_reward = 0
        episode_length = 0
        done = False
        
        obs, _ = env_wrapper.reset()
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env_wrapper.step(action)
            
            episode_reward += np.mean(reward)
            episode_length += 1
            
            # Check if any environment succeeded (reached goal)
            if done.any() and episode_length > 1:
                break
                
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        print(f"Episode {episode + 1}: Reward = {episode_reward:.3f}, Length = {episode_length}")
    
    # Print evaluation summary
    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Episodes: {num_episodes}")
    print(f"Mean reward: {np.mean(episode_rewards):.3f} ± {np.std(episode_rewards):.3f}")
    print(f"Mean length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"Max reward: {np.max(episode_rewards):.3f}")
    print(f"Min reward: {np.min(episode_rewards):.3f}")
    
    env_wrapper.close()
    return episode_rewards, episode_lengths


def main():
    """Main training function"""
    
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train PPO agent for drone navigation")
    parser.add_argument("--timesteps", type=int, default=500_000, help="Total training timesteps")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=2048, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log-dir", type=str, default="./logs/drone_ppo", help="Log directory")
    parser.add_argument("--model-dir", type=str, default="./models/drone_ppo", help="Model directory")
    parser.add_argument("--eval-only", action="store_true", help="Only evaluate, don't train")
    parser.add_argument("--model-path", type=str, help="Path to model for evaluation")
    args = parser.parse_args()
    
    # Create PPO config
    ppo_cfg = create_ppo_config(
        total_timesteps=args.timesteps,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        device="cpu"  # Use CPU for better performance with MLP policies
    )
    
    # Create environment config
    env_cfg = DroneLabCfg()
    
    if args.eval_only and args.model_path:
        # Evaluation mode
        evaluate_trained_model(args.model_path, env_cfg, 10)
    else:
        # Training mode
        if not ISAAC_LAB_AVAILABLE:
            print("🚀 Starting training in standalone mode...")
            # Use our proven standalone training
            from train_drone_standalone import main as standalone_main
            standalone_main()
        else:
            print("🚀 Starting training with Isaac Lab...")
            train_ppo_agent(
                env_cfg,
                ppo_cfg,
                args.log_dir,
                args.model_dir,
                args.seed
            )


if __name__ == "__main__":
    main()
