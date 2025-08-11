#!/usr/bin/env python3
"""
Advanced RL Training Suite for Drone Navigation

This script provides a comprehensive training suite that supports multiple RL algorithms:
- PPO (Proximal Policy Optimization) - model-free
- DreamerV3 - model-based with world model learning
- SAC (Soft Actor-Critic) - off-policy model-free

Features:
- Compatible with Isaac Lab, standalone, and Isaac Sim environments
- Hyperparameter optimization
- Advanced callbacks and logging
- Curriculum learning
- Multi-environment training
- Model evaluation and comparison
"""

import os
import sys
import time
import argparse
import torch
import numpy as np
from typing import Optional, Dict, List, Any, Tuple
import json
import yaml
from pathlib import Path
import matplotlib.pyplot as plt

# Import RL libraries
try:
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
    from stable_baselines3.common.callbacks import (
        BaseCallback, CheckpointCallback, EvalCallback, 
        StopTrainingOnRewardThreshold, ProgressBarCallback
    )
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.utils import set_random_seed
    from stable_baselines3.common.monitor import Monitor
    SB3_AVAILABLE = True
except ImportError as e:
    print(f"❌ Stable Baselines3 not available: {e}")
    SB3_AVAILABLE = False

# Import our custom modules
try:
    from dreamerv3_drone import DreamerV3Agent, Experience
    DREAMERV3_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ DreamerV3 not available: {e}")
    DREAMERV3_AVAILABLE = False

# Try Isaac Lab imports (skip if Isaac Sim not properly configured)
ISAAC_LAB_AVAILABLE = False
# Temporarily disable Isaac Lab to avoid import errors
# try:
#     # Only try if Isaac Sim path exists
#     import os
#     isaac_sim_paths = [
#         "/home/omniverse/.local/lib/python3.10/site-packages/isaacsim",
#         "/isaac-sim",
#         "/opt/nvidia/omniverse/pkg/isaac_sim-2023.1.1"
#     ]
#     
#     if any(os.path.exists(path) for path in isaac_sim_paths):
#         from drone_task_lab import DroneVectorTask, DroneLabCfg
#         ISAAC_LAB_AVAILABLE = True
#         print("✓ Isaac Lab available")
#     else:
#         print("⚠️ Isaac Sim not found, skipping Isaac Lab")
# except Exception as e:
#     print(f"⚠️ Isaac Lab not available: {e}")
#     ISAAC_LAB_AVAILABLE = False
print("⚠️ Isaac Lab temporarily disabled to avoid import conflicts")

# Try standalone environment
try:
    from simple_drone_env import SimpleDroneEnv
    STANDALONE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Standalone environment not available: {e}")
    STANDALONE_AVAILABLE = False


class CurriculumCallback(BaseCallback):
    """Callback for curriculum learning - gradually increase difficulty"""
    
    def __init__(self, env, curriculum_steps: List[int], 
                 curriculum_configs: List[Dict], verbose: int = 1):
        super().__init__(verbose)
        self.env = env
        self.curriculum_steps = curriculum_steps
        self.curriculum_configs = curriculum_configs
        self.current_level = 0
        
    def _on_step(self) -> bool:
        # Check if we should advance curriculum
        if (self.current_level < len(self.curriculum_steps) and 
            self.num_timesteps >= self.curriculum_steps[self.current_level]):
            
            config = self.curriculum_configs[self.current_level]
            if self.verbose > 0:
                print(f"\n📚 Advancing curriculum to level {self.current_level + 1}")
                print(f"New config: {config}")
            
            # Update environment configuration
            if hasattr(self.env, 'envs'):  # VecEnv
                for env in self.env.envs:
                    self._update_env_config(env, config)
            else:  # Single env
                self._update_env_config(self.env, config)
            
            self.current_level += 1
            
        return True
    
    def _update_env_config(self, env, config):
        """Update environment configuration"""
        for key, value in config.items():
            if hasattr(env, key):
                setattr(env, key, value)


class HyperparameterTuningCallback(BaseCallback):
    """Callback for online hyperparameter tuning"""
    
    def __init__(self, model, tuning_config: Dict, verbose: int = 1):
        super().__init__(verbose)
        self.model = model
        self.tuning_config = tuning_config
        self.best_reward = float('-inf')
        self.tuning_steps = tuning_config.get('steps', [50000, 100000, 200000])
        self.current_step = 0
        
    def _on_step(self) -> bool:
        if (self.current_step < len(self.tuning_steps) and 
            self.num_timesteps >= self.tuning_steps[self.current_step]):
            
            # Get recent performance
            if hasattr(self.locals.get('infos', [{}])[0], 'episode'):
                recent_rewards = [info.get('episode', {}).get('r', 0) 
                                for info in self.locals.get('infos', [])
                                if 'episode' in info]
                
                if recent_rewards:
                    current_reward = np.mean(recent_rewards)
                    
                    if current_reward > self.best_reward:
                        self.best_reward = current_reward
                        if self.verbose > 0:
                            print(f"🎯 New best reward: {current_reward:.3f}")
                    else:
                        # Performance degraded, adjust hyperparameters
                        self._adjust_hyperparameters()
            
            self.current_step += 1
            
        return True
    
    def _adjust_hyperparameters(self):
        """Adjust learning rate and other hyperparameters"""
        if hasattr(self.model, 'learning_rate'):
            old_lr = self.model.learning_rate
            new_lr = old_lr * 0.8  # Reduce learning rate
            self.model.learning_rate = new_lr
            if self.verbose > 0:
                print(f"📉 Reduced learning rate: {old_lr:.6f} → {new_lr:.6f}")


class ComparisonTrainer:
    """Main trainer class supporting multiple RL algorithms"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.results = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() and 
                                  self.config.get('use_gpu', False) else "cpu")
        
        # Create directories
        self.log_dir = Path(self.config.get('log_dir', './logs_advanced'))
        self.model_dir = Path(self.config.get('model_dir', './models_advanced'))
        self.log_dir.mkdir(exist_ok=True, parents=True)
        self.model_dir.mkdir(exist_ok=True, parents=True)
        
        print(f"🎯 Advanced RL Training Suite initialized")
        print(f"Device: {self.device}")
        print(f"Logs: {self.log_dir}")
        print(f"Models: {self.model_dir}")
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from file or use defaults"""
        default_config = {
            'algorithms': ['ppo', 'dreamerv3'],
            'environment': 'standalone',  # 'isaac_lab', 'standalone', 'isaac_sim'
            'total_timesteps': 500000,
            'num_envs': 16,
            'eval_episodes': 10,
            'save_freq': 50000,
            'seed': 42,
            'use_gpu': False,
            'curriculum_learning': True,
            'hyperparameter_tuning': True,
            'log_dir': './logs_advanced',
            'model_dir': './models_advanced',
            'env_config': {
                'arena_bounds': [[-5, -5, 0], [5, 5, 5]],
                'num_obstacles': 5,
                'obstacle_radius': 1.0,
                'max_steps': 100
            },
            'ppo_config': {
                'learning_rate': 3e-4,
                'n_steps': 2048,
                'batch_size': 64,
                'n_epochs': 10,
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'clip_range': 0.2,
                'ent_coef': 0.01,
                'vf_coef': 0.5,
                'max_grad_norm': 0.5
            },
            'dreamerv3_config': {
                'learning_rate': 1e-4,
                'hidden_dim': 256,
                'latent_dim': 32,
                'imagination_horizon': 15,
                'gamma': 0.99,
                'lambda_gae': 0.95
            },
            'curriculum': {
                'steps': [100000, 200000, 300000],
                'configs': [
                    {'num_obstacles': 3, 'obstacle_radius': 0.8},
                    {'num_obstacles': 7, 'obstacle_radius': 1.0},
                    {'num_obstacles': 10, 'obstacle_radius': 1.2}
                ]
            }
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                    loaded_config = yaml.safe_load(f)
                else:
                    loaded_config = json.load(f)
            
            # Merge with defaults
            default_config.update(loaded_config)
        
        return default_config
    
    def create_environment(self, env_type: str = None) -> Any:
        """Create environment based on configuration"""
        env_type = env_type or self.config['environment']
        
        if env_type == 'isaac_lab' and ISAAC_LAB_AVAILABLE:
            # Isaac Lab environment
            from drone_task_lab import DroneVectorTask, DroneLabCfg
            cfg = DroneLabCfg()
            cfg.num_envs = self.config['num_envs']
            env = DroneVectorTask(cfg)
            print(f"✓ Created Isaac Lab environment with {cfg.num_envs} envs")
            return env
            
        elif env_type == 'standalone' and STANDALONE_AVAILABLE:
            # Standalone vectorized environment with configuration
            env_config = self.config.get('env_config', {})
            
            def make_env(env_id: int):
                def _init():
                    return SimpleDroneEnv(env_id, config=env_config)
                return _init
            
            if self.config['num_envs'] == 1:
                env = SimpleDroneEnv(0, config=env_config)
            else:
                env = DummyVecEnv([make_env(i) for i in range(self.config['num_envs'])])
            
            print(f"✓ Created enhanced standalone environment with {self.config['num_envs']} envs")
            if env_config:
                print(f"  Arena: {env_config.get('arena_size', 30)}x{env_config.get('arena_height', 12)}")
                print(f"  Max steps: {env_config.get('max_steps', 1500)}")
                print(f"  Obstacles: {env_config.get('num_obstacles', 15)}")
                print(f"  Difficulty: {env_config.get('difficulty_level', 'medium')}")
            return env
            
        else:
            raise ValueError(f"Environment {env_type} not available or not supported")
    
    def train_ppo(self, env) -> Dict[str, Any]:
        """Train PPO agent"""
        if not SB3_AVAILABLE:
            print("❌ PPO training requires Stable Baselines3")
            return {}
        
        print("\n🚀 Training PPO Agent")
        print("=" * 50)
        
        # Wrap environment for SB3
        if not hasattr(env, 'observation_space'):
            # Isaac Lab environment needs wrapping
            from train_drone_ppo import IsaacLabVecEnvWrapper
            env = IsaacLabVecEnvWrapper(env)
        
        env = Monitor(env) if not hasattr(env, 'get_attr') else env
        
        # Configure logger
        ppo_log_dir = self.log_dir / 'ppo'
        ppo_log_dir.mkdir(exist_ok=True)
        logger = configure(str(ppo_log_dir), ["stdout", "csv", "tensorboard"])
        
        # Create PPO model
        ppo_config = self.config['ppo_config']
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=ppo_config['learning_rate'],
            n_steps=ppo_config['n_steps'],
            batch_size=ppo_config['batch_size'],
            n_epochs=ppo_config['n_epochs'],
            gamma=ppo_config['gamma'],
            gae_lambda=ppo_config['gae_lambda'],
            clip_range=ppo_config['clip_range'],
            ent_coef=ppo_config['ent_coef'],
            vf_coef=ppo_config['vf_coef'],
            max_grad_norm=ppo_config['max_grad_norm'],
            verbose=1,
            device=self.device,
            tensorboard_log=str(ppo_log_dir)
        )
        
        model.set_logger(logger)
        
        # Setup callbacks
        callbacks = []
        
        # Checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=self.config['save_freq'],
            save_path=str(self.model_dir / 'ppo'),
            name_prefix="ppo_drone"
        )
        callbacks.append(checkpoint_callback)
        
        # Evaluation callback
        eval_env = self.create_environment()
        if not hasattr(eval_env, 'observation_space'):
            eval_env = IsaacLabVecEnvWrapper(eval_env)
        eval_env = Monitor(eval_env)
        
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(self.model_dir / 'ppo'),
            log_path=str(ppo_log_dir),
            eval_freq=10000,
            n_eval_episodes=self.config['eval_episodes'],
            deterministic=True
        )
        callbacks.append(eval_callback)
        
        # Curriculum learning
        if self.config.get('curriculum_learning', False):
            curriculum_callback = CurriculumCallback(
                env,
                self.config['curriculum']['steps'],
                self.config['curriculum']['configs']
            )
            callbacks.append(curriculum_callback)
        
        # Hyperparameter tuning
        if self.config.get('hyperparameter_tuning', False):
            tuning_callback = HyperparameterTuningCallback(model, {'steps': [100000, 200000]})
            callbacks.append(tuning_callback)
        
        # Train
        start_time = time.time()
        model.learn(
            total_timesteps=self.config['total_timesteps'],
            callback=callbacks,
            progress_bar=True
        )
        training_time = time.time() - start_time
        
        # Save final model
        model.save(self.model_dir / 'ppo' / 'final_model')
        
        # Evaluate final model
        final_rewards = self.evaluate_model(model, eval_env, 'PPO')
        
        return {
            'algorithm': 'PPO',
            'training_time': training_time,
            'final_mean_reward': np.mean(final_rewards),
            'final_std_reward': np.std(final_rewards),
            'model_path': str(self.model_dir / 'ppo' / 'final_model.zip')
        }
    
    def train_dreamerv3(self, env) -> Dict[str, Any]:
        """Train DreamerV3 agent"""
        if not DREAMERV3_AVAILABLE:
            print("❌ DreamerV3 training requires dreamerv3_drone module")
            return {}
        
        print("\n🌟 Training DreamerV3 Agent")
        print("=" * 50)
        
        # Get environment dimensions
        if hasattr(env, 'observation_space'):
            obs_dim = env.observation_space.shape[0]
            action_dim = env.action_space.shape[0]
        else:
            # Enhanced standalone environment - dynamic observation size
            test_obs = env.reset()
            if isinstance(test_obs, tuple):
                test_obs = test_obs[0]
            if test_obs.ndim > 1:
                test_obs = test_obs[0]
            obs_dim = len(test_obs)
            action_dim = 3  # x, y, z acceleration
        
        # Create DreamerV3 agent
        dreamer_config = self.config['dreamerv3_config']
        agent = DreamerV3Agent(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=dreamer_config['hidden_dim'],
            latent_dim=dreamer_config['latent_dim'],
            learning_rate=dreamer_config['learning_rate'],
            imagination_horizon=dreamer_config['imagination_horizon'],
            gamma=dreamer_config['gamma'],
            lambda_gae=dreamer_config['lambda_gae'],
            device=str(self.device)
        )
        
        # Training loop
        start_time = time.time()
        total_timesteps = self.config['total_timesteps']
        batch_size = 32
        training_freq = 4  # Train every N steps
        episode_rewards = []
        
        print(f"Training for {total_timesteps:,} timesteps...")
        
        step = 0
        while step < total_timesteps:
            # Collect episode
            episode_data = []
            obs = env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
            
            # Handle vectorized environments
            if obs.ndim > 1:
                obs = obs[0]  # Take first environment
            
            episode_reward = 0
            done = False
            
            while not done and step < total_timesteps:
                # Get action from agent
                action = agent.get_action(obs, deterministic=False)
                
                # Ensure action is the right shape for single environment
                if isinstance(action, np.ndarray) and action.ndim > 1:
                    action = action.flatten()
                
                # Step environment
                next_obs, reward, done, info = env.step(action)
                
                # Handle vectorized environment outputs
                if isinstance(next_obs, tuple):
                    next_obs = next_obs[0]
                if isinstance(reward, (list, np.ndarray)):
                    reward = reward[0] if len(reward) > 0 else 0
                if isinstance(done, (list, np.ndarray)):
                    done = done[0] if len(done) > 0 else False
                if isinstance(info, (list, dict)):
                    info = info[0] if isinstance(info, list) else info
                
                if next_obs.ndim > 1:
                    next_obs = next_obs[0]
                
                # Store experience
                episode_data.append((obs, action, reward, next_obs, done, info))
                
                obs = next_obs
                episode_reward += reward
                step += 1
                
                # Train agent
                if step % training_freq == 0:
                    metrics = agent.train_step(batch_size)
                    if step % 1000 == 0 and metrics:
                        # Detailed progress logging
                        wm_loss = metrics.get('world_model/total_loss', 0)
                        actor_loss = metrics.get('actor/loss', 0)
                        critic_loss = metrics.get('critic/loss', 0)
                        
                        print(f"Step {step:,}: WM Loss = {wm_loss:.4f}, "
                              f"Actor Loss = {actor_loss:.4f}, "
                              f"Critic Loss = {critic_loss:.4f}")
                        
                        # Additional metrics if available
                        if 'world_model/reward_loss' in metrics:
                            reward_loss = metrics['world_model/reward_loss']
                            print(f"         Reward Loss = {reward_loss:.4f}")
                        
                        if 'world_model/continue_loss' in metrics:
                            continue_loss = metrics['world_model/continue_loss']
                            print(f"         Continue Loss = {continue_loss:.4f}")
                        
                        # Performance indicators
                        recent_episodes = episode_rewards[-10:] if len(episode_rewards) >= 10 else episode_rewards
                        if recent_episodes:
                            avg_reward = np.mean(recent_episodes)
                            success_rate = sum(1 for r in recent_episodes if r > 100) / len(recent_episodes)
                            print(f"         Recent Avg Reward: {avg_reward:.2f}, "
                                  f"Success Rate: {success_rate:.1%}")
                        
                        print("-" * 60)
            
            # Add episode to replay buffer
            agent.add_episode(episode_data)
            episode_rewards.append(episode_reward)
            
            # Save path data every 10 episodes for visualization
            if len(episode_rewards) % 10 == 0:
                try:
                    if hasattr(env, 'get_episode_paths'):
                        episode_paths = env.get_episode_paths(20)  # Get last 20 episodes
                        if episode_paths:
                            import pickle
                            save_dir = self.log_dir / 'dreamerv3'
                            save_dir.mkdir(exist_ok=True, parents=True)
                            
                            # Save path data for visualization
                            path_file = save_dir / 'episode_paths.pkl'
                            with open(path_file, 'wb') as f:
                                pickle.dump(episode_paths, f)
                except Exception as e:
                    pass  # Don't break training if path saving fails
            
            if len(episode_rewards) % 10 == 0:
                recent_reward = np.mean(episode_rewards[-10:])
                print(f"Episodes: {len(episode_rewards)}, Recent reward: {recent_reward:.3f}")
        
        training_time = time.time() - start_time
        
        # Save agent
        agent_path = self.model_dir / 'dreamerv3' / 'final_model.pt'
        agent_path.parent.mkdir(exist_ok=True, parents=True)
        agent.save(str(agent_path))
        
        # Evaluate final model
        final_rewards = []
        for _ in range(self.config['eval_episodes']):
            obs = env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
            if obs.ndim > 1:
                obs = obs[0]
            
            episode_reward = 0
            done = False
            
            while not done:
                action = agent.get_action(obs, deterministic=True)
                
                # Ensure action is the right shape
                if isinstance(action, np.ndarray) and action.ndim > 1:
                    action = action.flatten()
                
                next_obs, reward, done, _ = env.step(action)
                
                if isinstance(next_obs, tuple):
                    next_obs = next_obs[0]
                if isinstance(reward, (list, np.ndarray)):
                    reward = reward[0] if len(reward) > 0 else 0
                if isinstance(done, (list, np.ndarray)):
                    done = done[0] if len(done) > 0 else False
                if next_obs.ndim > 1:
                    next_obs = next_obs[0]
                
                obs = next_obs
                episode_reward += reward
            
            final_rewards.append(episode_reward)
        
        return {
            'algorithm': 'DreamerV3',
            'training_time': training_time,
            'final_mean_reward': np.mean(final_rewards),
            'final_std_reward': np.std(final_rewards),
            'model_path': str(agent_path)
        }
    
    def train_sac(self, env) -> Dict[str, Any]:
        """Train SAC agent"""
        if not SB3_AVAILABLE:
            print("❌ SAC training requires Stable Baselines3")
            return {}
        
        print("\n🎭 Training SAC Agent")
        print("=" * 50)
        
        # SAC works best with single environment
        if hasattr(env, 'num_envs') and env.num_envs > 1:
            print("⚠️ SAC works best with single environment, creating new one...")
            env = self.create_environment()
            if hasattr(env, 'envs'):
                env = env.envs[0]
        
        # Wrap environment for SB3
        if not hasattr(env, 'observation_space'):
            from train_drone_ppo import IsaacLabVecEnvWrapper
            env = IsaacLabVecEnvWrapper(env)
        
        env = Monitor(env)
        
        # Configure logger
        sac_log_dir = self.log_dir / 'sac'
        sac_log_dir.mkdir(exist_ok=True)
        logger = configure(str(sac_log_dir), ["stdout", "csv", "tensorboard"])
        
        # Create SAC model
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            buffer_size=100000,
            learning_starts=1000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            verbose=1,
            device=self.device,
            tensorboard_log=str(sac_log_dir)
        )
        
        model.set_logger(logger)
        
        # Setup callbacks
        callbacks = []
        
        # Checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=self.config['save_freq'],
            save_path=str(self.model_dir / 'sac'),
            name_prefix="sac_drone"
        )
        callbacks.append(checkpoint_callback)
        
        # Train
        start_time = time.time()
        model.learn(
            total_timesteps=self.config['total_timesteps'],
            callback=callbacks,
            progress_bar=True
        )
        training_time = time.time() - start_time
        
        # Save final model
        model.save(self.model_dir / 'sac' / 'final_model')
        
        # Evaluate final model
        final_rewards = self.evaluate_model(model, env, 'SAC')
        
        return {
            'algorithm': 'SAC',
            'training_time': training_time,
            'final_mean_reward': np.mean(final_rewards),
            'final_std_reward': np.std(final_rewards),
            'model_path': str(self.model_dir / 'sac' / 'final_model.zip')
        }
    
    def evaluate_model(self, model, env, algorithm_name: str) -> List[float]:
        """Evaluate trained model"""
        print(f"\n📊 Evaluating {algorithm_name} model...")
        
        episode_rewards = []
        
        for episode in range(self.config['eval_episodes']):
            obs = env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
            
            episode_reward = 0
            done = False
            
            while not done:
                if hasattr(model, 'predict'):
                    action, _ = model.predict(obs, deterministic=True)
                else:
                    # DreamerV3 agent
                    action = model.get_action(obs, deterministic=True)
                
                obs, reward, done, _ = env.step(action)
                
                if isinstance(obs, tuple):
                    obs = obs[0]
                if isinstance(reward, (list, np.ndarray)):
                    reward = reward[0] if len(reward) > 0 else 0
                if isinstance(done, (list, np.ndarray)):
                    done = done[0] if len(done) > 0 else False
                
                episode_reward += reward
            
            episode_rewards.append(episode_reward)
            print(f"Episode {episode + 1}: {episode_reward:.3f}")
        
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        print(f"{algorithm_name} Evaluation: {mean_reward:.3f} ± {std_reward:.3f}")
        
        return episode_rewards
    
    def run_comparison(self) -> Dict[str, Any]:
        """Run comparison between all specified algorithms"""
        print("\n🏆 Starting Algorithm Comparison")
        print("=" * 60)
        
        # Set random seed
        if self.config.get('seed'):
            set_random_seed(self.config['seed'])
            torch.manual_seed(self.config['seed'])
            np.random.seed(self.config['seed'])
        
        results = {}
        
        for algorithm in self.config['algorithms']:
            print(f"\n{'='*20} {algorithm.upper()} {'='*20}")
            
            try:
                # Create environment
                env = self.create_environment()
                
                # Train algorithm
                if algorithm.lower() == 'ppo':
                    result = self.train_ppo(env)
                elif algorithm.lower() == 'dreamerv3':
                    result = self.train_dreamerv3(env)
                elif algorithm.lower() == 'sac':
                    result = self.train_sac(env)
                else:
                    print(f"❌ Unknown algorithm: {algorithm}")
                    continue
                
                results[algorithm] = result
                
                # Close environment
                if hasattr(env, 'close'):
                    env.close()
                
            except Exception as e:
                print(f"❌ Error training {algorithm}: {e}")
                results[algorithm] = {'error': str(e)}
        
        # Save results
        self.save_results(results)
        self.print_comparison(results)
        
        return results
    
    def save_results(self, results: Dict[str, Any]):
        """Save comparison results"""
        results_path = self.log_dir / 'comparison_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to {results_path}")
    
    def print_comparison(self, results: Dict[str, Any]):
        """Print comparison summary"""
        print("\n" + "="*60)
        print("🏆 ALGORITHM COMPARISON RESULTS")
        print("="*60)
        
        # Sort by mean reward
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        sorted_results = sorted(valid_results.items(), 
                               key=lambda x: x[1].get('final_mean_reward', -np.inf), 
                               reverse=True)
        
        print(f"{'Algorithm':<12} {'Mean Reward':<12} {'Std':<8} {'Time (min)':<10}")
        print("-" * 50)
        
        for i, (algo, result) in enumerate(sorted_results):
            rank_emoji = ["🥇", "🥈", "🥉"][i] if i < 3 else "  "
            mean_reward = result.get('final_mean_reward', 0)
            std_reward = result.get('final_std_reward', 0)
            training_time = result.get('training_time', 0) / 60
            
            print(f"{rank_emoji} {algo:<10} {mean_reward:>8.3f}    {std_reward:>6.3f}  {training_time:>8.1f}")
        
        # Print errors
        error_results = {k: v for k, v in results.items() if 'error' in v}
        if error_results:
            print(f"\n❌ Failed algorithms:")
            for algo, result in error_results.items():
                print(f"  {algo}: {result['error']}")
    
    def create_plots(self, results: Dict[str, Any]):
        """Create comparison plots"""
        try:
            import matplotlib.pyplot as plt
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            algorithms = list(results.keys())
            mean_rewards = [results[algo].get('final_mean_reward', 0) for algo in algorithms]
            training_times = [results[algo].get('training_time', 0) / 60 for algo in algorithms]
            
            # Mean reward comparison
            ax1.bar(algorithms, mean_rewards)
            ax1.set_title('Final Mean Reward Comparison')
            ax1.set_ylabel('Mean Reward')
            
            # Training time comparison
            ax2.bar(algorithms, training_times)
            ax2.set_title('Training Time Comparison')
            ax2.set_ylabel('Time (minutes)')
            
            # Efficiency plot (reward per minute)
            efficiency = [r/t if t > 0 else 0 for r, t in zip(mean_rewards, training_times)]
            ax3.bar(algorithms, efficiency)
            ax3.set_title('Training Efficiency (Reward/Minute)')
            ax3.set_ylabel('Efficiency')
            
            # Placeholder for future metrics
            ax4.text(0.5, 0.5, 'Additional metrics\nwill be added here', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Future Metrics')
            
            plt.tight_layout()
            plt.savefig(self.log_dir / 'comparison_plots.png', dpi=300, bbox_inches='tight')
            print(f"📊 Plots saved to {self.log_dir / 'comparison_plots.png'}")
            
        except ImportError:
            print("⚠️ Matplotlib not available, skipping plots")


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description="Advanced RL Training Suite for Drone Navigation")
    
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--algorithms', nargs='+', default=['ppo', 'dreamerv3'], 
                       choices=['ppo', 'dreamerv3', 'sac'],
                       help='Algorithms to train and compare')
    parser.add_argument('--environment', choices=['isaac_lab', 'standalone', 'isaac_sim'],
                       default='standalone', help='Environment type to use')
    parser.add_argument('--timesteps', type=int, default=200000, 
                       help='Total training timesteps')
    parser.add_argument('--num-envs', type=int, default=16, 
                       help='Number of parallel environments')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    
    args = parser.parse_args()
    
    # Override config with command line arguments
    config_override = {
        'algorithms': args.algorithms,
        'environment': args.environment,
        'total_timesteps': args.timesteps,
        'num_envs': args.num_envs,
        'seed': args.seed,
        'use_gpu': args.gpu
    }
    
    # Create trainer
    trainer = ComparisonTrainer(args.config)
    trainer.config.update(config_override)
    
    print(f"🚀 Training algorithms: {args.algorithms}")
    print(f"🌍 Environment: {args.environment}")
    print(f"⏱️  Timesteps: {args.timesteps:,}")
    print(f"🔢 Parallel envs: {args.num_envs}")
    
    # Run comparison
    results = trainer.run_comparison()
    
    # Create plots
    trainer.create_plots(results)
    
    print("\n🎉 Advanced RL training completed!")
    print(f"Check {trainer.log_dir} for detailed logs and results.")


if __name__ == "__main__":
    main()
