#!/usr/bin/env python3
"""
DreamerV3 Implementation for Drone Navigation

This module implements DreamerV3, a model-based reinforcement learning algorithm,
specifically adapted for drone navigation tasks. DreamerV3 learns a world model
of the environment and uses it for planning and policy learning.

Key Features:
- World model learning with RSSM (Recurrent State Space Model)
- Actor-critic learning in the latent space
- Imagination rollouts for sample efficiency
- Compatible with both Isaac Lab and standalone environments
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from collections import deque, namedtuple
import math

try:
    import jax
    import jax.numpy as jnp
    import flax.linen as nn_flax
    import optax
    JAX_AVAILABLE = True
    print("✓ JAX available for DreamerV3")
except ImportError:
    JAX_AVAILABLE = False
    print("⚠️ JAX not available, using PyTorch fallback")


# Experience replay buffer
Experience = namedtuple('Experience', 
                       ['obs', 'action', 'reward', 'next_obs', 'done', 'info'])


class SuccessPrioritizedReplayBuffer:
    """
    Success-prioritized experience replay buffer for DreamerV3
    Prioritizes learning from successful navigation sequences
    """
    
    def __init__(self, capacity: int = 100000, sequence_length: int = 50, 
                 success_priority_ratio: float = 0.7):
        self.capacity = capacity
        self.sequence_length = sequence_length
        self.success_priority_ratio = success_priority_ratio  # Ratio of batch to sample from successes
        
        # Separate buffers for successful and unsuccessful sequences
        self.success_sequences = deque(maxlen=capacity // 4)  # Smaller buffer for successes (they're rarer)
        self.regular_sequences = deque(maxlen=capacity)
        
        # Track sequence metadata
        self.success_metadata = deque(maxlen=capacity // 4)  # Store final distance, total reward, etc.
        self.regular_metadata = deque(maxlen=capacity)
        
    def add(self, experience: Experience):
        """Add single experience to buffer (for backward compatibility)"""
        # Convert single experience to sequence of length 1
        self.add_sequence([experience], success=False)
        
    def add_sequence(self, sequence: List[Experience], success: bool = False, 
                    final_distance: float = float('inf'), total_reward: float = 0.0):
        """
        Add a sequence of experiences with success information
        
        Args:
            sequence: List of experiences
            success: Whether this sequence represents successful navigation
            final_distance: Final distance to goal (for ranking successful sequences)
            total_reward: Total reward accumulated (for ranking sequences)
        """
        if len(sequence) < self.sequence_length:
            # Pad short sequences
            padded = sequence + [sequence[-1]] * (self.sequence_length - len(sequence))
            sequence = padded
        
        # Create metadata for this sequence
        metadata = {
            'success': success,
            'final_distance': final_distance,
            'total_reward': total_reward,
            'sequence_length': len(sequence),
            'timestamp': time.time()
        }
        
        if success:
            # Store successful sequences for prioritized sampling
            if len(sequence) >= self.sequence_length:
                # Split long sequences into chunks
                for i in range(0, len(sequence) - self.sequence_length + 1, self.sequence_length):
                    chunk = sequence[i:i + self.sequence_length]
                    self.success_sequences.append(chunk)
                    self.success_metadata.append(metadata.copy())
            else:
                self.success_sequences.append(sequence)
                self.success_metadata.append(metadata)
            
            print(f"   🎯 Added SUCCESS sequence (distance: {final_distance:.2f}m, reward: {total_reward:.1f}) - Total successes: {len(self.success_sequences)}")
        else:
            # Store regular sequences
            if len(sequence) >= self.sequence_length:
                # Split long sequences into chunks
                for i in range(0, len(sequence) - self.sequence_length + 1, self.sequence_length):
                    chunk = sequence[i:i + self.sequence_length]
                    self.regular_sequences.append(chunk)
                    self.regular_metadata.append(metadata.copy())
            else:
                self.regular_sequences.append(sequence)
                self.regular_metadata.append(metadata)
    
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample batch with prioritization of successful sequences"""
        if len(self.regular_sequences) < batch_size // 2:
            return None
            
        batch_sequences = []
        
        # Determine how many samples to draw from each buffer
        if len(self.success_sequences) > 0:
            success_samples = min(int(batch_size * self.success_priority_ratio), 
                                len(self.success_sequences), 
                                batch_size)
            regular_samples = batch_size - success_samples
            
            # Sample from successful sequences (prioritize better performances)
            if success_samples > 0:
                success_indices = self._sample_success_indices(success_samples)
                for idx in success_indices:
                    batch_sequences.append(self.success_sequences[idx])
            
            # Sample from regular sequences
            if regular_samples > 0 and len(self.regular_sequences) >= regular_samples:
                regular_indices = np.random.choice(len(self.regular_sequences), 
                                                 regular_samples, replace=False)
                for idx in regular_indices:
                    batch_sequences.append(self.regular_sequences[idx])
        else:
            # No successful sequences yet, sample only from regular buffer
            if len(self.regular_sequences) < batch_size:
                return None
            regular_indices = np.random.choice(len(self.regular_sequences), 
                                             batch_size, replace=False)
            batch_sequences = [self.regular_sequences[i] for i in regular_indices]
    
    def _sample_success_indices(self, num_samples: int) -> List[int]:
        """Sample indices from successful sequences with performance-based weighting"""
        if not self.success_metadata:
            return []
        
        # Create weights based on performance (better performance = higher weight)
        weights = []
        for metadata in self.success_metadata:
            # Weight by inverse of final distance (closer to goal = higher weight)
            distance_weight = 1.0 / (1.0 + metadata['final_distance'])
            # Weight by total reward (higher reward = higher weight)
            reward_weight = max(0.1, metadata['total_reward'] / 100.0 + 1.0)
            # Combined weight
            weight = distance_weight * reward_weight
            weights.append(weight)
        
        # Normalize weights
        weights = np.array(weights)
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = np.ones(len(weights)) / len(weights)
        
        # Sample with replacement based on weights
        indices = np.random.choice(len(self.success_sequences), 
                                 size=min(num_samples, len(self.success_sequences)),
                                 replace=num_samples > len(self.success_sequences),
                                 p=weights)
        return indices.tolist()
        
        # Continue with remaining logic for converting sequences to tensors
        if not batch_sequences:
            return None
            
        # Convert to tensors
        obs = torch.stack([torch.stack([torch.tensor(exp.obs, dtype=torch.float32) 
                                       for exp in seq]) for seq in batch_sequences])
        actions = torch.stack([torch.stack([torch.tensor(exp.action, dtype=torch.float32) 
                                           for exp in seq]) for seq in batch_sequences])
        rewards = torch.stack([torch.stack([torch.tensor(exp.reward, dtype=torch.float32) 
                                           for exp in seq]) for seq in batch_sequences])
        next_obs = torch.stack([torch.stack([torch.tensor(exp.next_obs, dtype=torch.float32) 
                                            for exp in seq]) for seq in batch_sequences])
        dones = torch.stack([torch.stack([torch.tensor(exp.done, dtype=torch.float32) 
                                         for exp in seq]) for seq in batch_sequences])
        
        return {
            'observations': obs,     # [batch, sequence, obs_dim]
            'actions': actions,      # [batch, sequence, action_dim]
            'rewards': rewards,      # [batch, sequence]
            'next_observations': next_obs,
            'dones': dones
        }
    
    def get_success_statistics(self) -> Dict[str, Any]:
        """Get statistics about successful vs regular sequences"""
        total_sequences = len(self.success_sequences) + len(self.regular_sequences)
        if total_sequences == 0:
            return {'total': 0, 'success_ratio': 0.0, 'success_count': 0}
            
        return {
            'total': total_sequences,
            'success_count': len(self.success_sequences),
            'regular_count': len(self.regular_sequences),
            'success_ratio': len(self.success_sequences) / total_sequences,
            'avg_success_distance': np.mean([m['final_distance'] for m in self.success_metadata]) if self.success_metadata else float('inf'),
            'avg_success_reward': np.mean([m['total_reward'] for m in self.success_metadata]) if self.success_metadata else 0.0
        }
    
    def __len__(self):
        return len(self.success_sequences) + len(self.regular_sequences)


# Keep old ReplayBuffer for backward compatibility
class ReplayBuffer(SuccessPrioritizedReplayBuffer):
    """Backward compatibility wrapper"""
    def __init__(self, capacity: int = 100000, sequence_length: int = 50):
        super().__init__(capacity, sequence_length, success_priority_ratio=0.3)  # Moderate prioritization


class RSSM(nn.Module):
    """Recurrent State Space Model - the world model component of DreamerV3"""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256, 
                 latent_dim: int = 32, categorical_dim: int = 32):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.categorical_dim = categorical_dim
        
        # Observation encoder
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # Recurrent model (GRU)
        self.rnn = nn.GRU(latent_dim + action_dim, hidden_dim, batch_first=True)
        
        # Prior network (predicts next state from current state and action)
        self.prior_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * latent_dim)  # mean and logvar
        )
        
        # Posterior network (predicts state from observation)
        self.posterior_net = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * latent_dim)  # mean and logvar
        )
        
        # Observation decoder
        self.obs_decoder = nn.Sequential(
            nn.Linear(latent_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim)
        )
        
        # Reward predictor
        self.reward_predictor = nn.Sequential(
            nn.Linear(latent_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Continuation predictor (predicts if episode continues)
        self.continue_predictor = nn.Sequential(
            nn.Linear(latent_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def encode_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode observation to latent representation"""
        return self.obs_encoder(obs)
    
    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for VAE"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def forward(self, obs_seq: torch.Tensor, action_seq: torch.Tensor, 
                initial_hidden: Optional[torch.Tensor] = None):
        """
        Forward pass through RSSM
        
        Args:
            obs_seq: [batch, sequence, obs_dim]
            action_seq: [batch, sequence, action_dim]
            initial_hidden: [batch, hidden_dim]
        """
        batch_size, seq_len = obs_seq.shape[:2]
        device = obs_seq.device
        
        if initial_hidden is None:
            initial_hidden = torch.zeros(batch_size, self.hidden_dim, device=device)
        
        # Encode all observations
        encoded_obs = self.encode_obs(obs_seq.view(-1, self.obs_dim))
        encoded_obs = encoded_obs.view(batch_size, seq_len, -1)
        
        # Lists to store outputs
        states = []
        hidden_states = []
        prior_means = []
        prior_logvars = []
        posterior_means = []
        posterior_logvars = []
        pred_obs = []
        pred_rewards = []
        pred_continues = []
        
        hidden = initial_hidden
        
        for t in range(seq_len):
            # Prior: predict next state from current hidden state
            prior_params = self.prior_net(hidden)
            prior_mean, prior_logvar = torch.chunk(prior_params, 2, dim=-1)
            
            # Posterior: infer state from observation and hidden state
            posterior_input = torch.cat([hidden, encoded_obs[:, t]], dim=-1)
            posterior_params = self.posterior_net(posterior_input)
            posterior_mean, posterior_logvar = torch.chunk(posterior_params, 2, dim=-1)
            
            # Sample state from posterior during training, prior during imagination
            state = self.reparameterize(posterior_mean, posterior_logvar)
            
            # Predict observations, rewards, and continuation
            state_hidden = torch.cat([state, hidden], dim=-1)
            pred_ob = self.obs_decoder(state_hidden)
            pred_reward = self.reward_predictor(state_hidden)
            pred_continue = self.continue_predictor(state_hidden)
            
            # Update hidden state using RNN
            if t < seq_len - 1:  # Don't need to update on last timestep
                rnn_input = torch.cat([state, action_seq[:, t]], dim=-1).unsqueeze(1)
                _, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
                hidden = hidden.squeeze(0)
            
            # Store outputs
            states.append(state)
            hidden_states.append(hidden)
            prior_means.append(prior_mean)
            prior_logvars.append(prior_logvar)
            posterior_means.append(posterior_mean)
            posterior_logvars.append(posterior_logvar)
            pred_obs.append(pred_ob)
            pred_rewards.append(pred_reward.squeeze(-1))
            pred_continues.append(torch.sigmoid(pred_continue.squeeze(-1)))
        
        return {
            'states': torch.stack(states, dim=1),
            'hidden_states': torch.stack(hidden_states, dim=1),
            'prior_means': torch.stack(prior_means, dim=1),
            'prior_logvars': torch.stack(prior_logvars, dim=1),
            'posterior_means': torch.stack(posterior_means, dim=1),
            'posterior_logvars': torch.stack(posterior_logvars, dim=1),
            'pred_observations': torch.stack(pred_obs, dim=1),
            'pred_rewards': torch.stack(pred_rewards, dim=1),
            'pred_continues': torch.stack(pred_continues, dim=1)
        }
    
    def imagine(self, initial_state: torch.Tensor, initial_hidden: torch.Tensor,
                actions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Imagination rollout using learned world model
        
        Args:
            initial_state: [batch, latent_dim]
            initial_hidden: [batch, hidden_dim]
            actions: [batch, horizon, action_dim]
        """
        batch_size, horizon = actions.shape[:2]
        
        states = [initial_state]
        hidden_states = [initial_hidden]
        pred_obs = []
        pred_rewards = []
        pred_continues = []
        
        state = initial_state
        hidden = initial_hidden
        
        for t in range(horizon):
            # Predict next state from current state and action
            prior_params = self.prior_net(hidden)
            prior_mean, prior_logvar = torch.chunk(prior_params, 2, dim=-1)
            state = self.reparameterize(prior_mean, prior_logvar)
            
            # Predict observations, rewards, and continuation
            state_hidden = torch.cat([state, hidden], dim=-1)
            pred_ob = self.obs_decoder(state_hidden)
            pred_reward = self.reward_predictor(state_hidden)
            pred_continue = self.continue_predictor(state_hidden)
            
            # Update hidden state
            rnn_input = torch.cat([state, actions[:, t]], dim=-1).unsqueeze(1)
            _, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
            hidden = hidden.squeeze(0)
            
            states.append(state)
            hidden_states.append(hidden)
            pred_obs.append(pred_ob)
            pred_rewards.append(pred_reward.squeeze(-1))
            pred_continues.append(torch.sigmoid(pred_continue.squeeze(-1)))
        
        return {
            'states': torch.stack(states[1:], dim=1),  # Don't include initial state
            'hidden_states': torch.stack(hidden_states[1:], dim=1),
            'pred_observations': torch.stack(pred_obs, dim=1),
            'pred_rewards': torch.stack(pred_rewards, dim=1),
            'pred_continues': torch.stack(pred_continues, dim=1)
        }


class Actor(nn.Module):
    """Actor network for policy learning"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.logstd_head = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Returns:
            mean: [batch, action_dim]
            logstd: [batch, action_dim]
        """
        features = self.net(state)
        mean = self.mean_head(features)
        logstd = self.logstd_head(features)
        logstd = torch.clamp(logstd, -5, 2)  # Clamp for stability
        return mean, logstd
    
    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action from policy"""
        mean, logstd = self.forward(state)
        std = torch.exp(logstd)
        normal = torch.distributions.Normal(mean, std)
        action = normal.rsample()
        log_prob = normal.log_prob(action).sum(-1)
        action = torch.tanh(action)  # Squash to [-1, 1]
        return action, log_prob


class Critic(nn.Module):
    """Critic network for value estimation"""
    
    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass returning value estimate"""
        return self.net(state).squeeze(-1)


class DreamerV3Agent:
    """
    DreamerV3 Agent for drone navigation
    
    This implements the core DreamerV3 algorithm:
    1. Learn world model from experience
    2. Use world model for imagination rollouts
    3. Learn policy and value function in imagination
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        latent_dim: int = 32,
        learning_rate: float = 1e-4,
        imagination_horizon: int = 15,
        gamma: float = 0.99,
        lambda_gae: float = 0.95,
        device: str = "cpu"
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.imagination_horizon = imagination_horizon
        self.gamma = gamma
        self.lambda_gae = lambda_gae
        self.device = device
        
        # Initialize networks
        self.world_model = RSSM(obs_dim, action_dim, hidden_dim, latent_dim).to(device)
        self.actor = Actor(latent_dim + hidden_dim, action_dim, hidden_dim).to(device)
        self.critic = Critic(latent_dim + hidden_dim, hidden_dim).to(device)
        
        # Initialize optimizers
        self.world_model_optimizer = torch.optim.Adam(self.world_model.parameters(), lr=learning_rate)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=learning_rate)
        
        # Replay buffer with success prioritization
        self.replay_buffer = SuccessPrioritizedReplayBuffer(
            capacity=100000, 
            sequence_length=50,
            success_priority_ratio=0.6  # 60% of batch from successful sequences when available
        )
        
        # Episode tracking for success evaluation
        self.current_episode_experiences = []
        self.episode_start_position = None
        self.episode_goal_position = None
        
        # Training metrics
        self.training_step = 0
        self.world_model_loss_history = []
        self.actor_loss_history = []
        self.critic_loss_history = []
    
    def add_experience(self, obs: np.ndarray, action: np.ndarray, reward: float, 
                      next_obs: np.ndarray, done: bool, info: dict = None):
        """Add single experience to replay buffer and episode tracking"""
        experience = Experience(obs, action, reward, next_obs, done, info)
        
        # Add to current episode tracking
        self.current_episode_experiences.append(experience)
        
        # If episode is done, evaluate success and add to appropriate buffer
        if done:
            self._finalize_episode(info or {})
        else:
            # For non-terminal experiences, add to regular buffer
            self.replay_buffer.add(experience)
    
    def start_episode(self, start_position: np.ndarray = None, goal_position: np.ndarray = None):
        """Start new episode tracking"""
        self.current_episode_experiences = []
        self.episode_start_position = start_position.copy() if start_position is not None else None
        self.episode_goal_position = goal_position.copy() if goal_position is not None else None
    
    def _finalize_episode(self, final_info: dict):
        """Finalize episode and add to appropriate buffer based on success"""
        if not self.current_episode_experiences:
            return
            
        # Determine if episode was successful
        episode_success = final_info.get('success', False)
        final_distance = final_info.get('distance_to_goal', float('inf'))
        
        # Calculate total reward for this episode
        total_reward = sum(exp.reward for exp in self.current_episode_experiences)
        
        # Add sequence to prioritized buffer
        self.replay_buffer.add_sequence(
            self.current_episode_experiences.copy(),
            success=episode_success,
            final_distance=final_distance,
            total_reward=total_reward
        )
        
        # Clear episode tracking
        self.current_episode_experiences = []
    
    def add_episode(self, episode_data: List[Tuple], success: bool = False, 
                   final_distance: float = float('inf'), total_reward: float = None):
        """Add complete episode to replay buffer with success information"""
        sequence = []
        episode_reward = 0.0
        
        for obs, action, reward, next_obs, done, info in episode_data:
            experience = Experience(obs, action, reward, next_obs, done, info or {})
            sequence.append(experience)
            episode_reward += reward
        
        # Use provided total_reward or calculate from sequence
        if total_reward is None:
            total_reward = episode_reward
            
        self.replay_buffer.add_sequence(sequence, success, final_distance, total_reward)
    
    def compute_world_model_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute world model loss (reconstruction + KL divergence)"""
        obs = batch['observations']
        actions = batch['actions']
        rewards = batch['rewards']
        next_obs = batch['next_observations']
        dones = batch['dones']
        
        # Forward pass through world model
        outputs = self.world_model(obs, actions)
        
        # Reconstruction losses
        obs_recon_loss = F.mse_loss(outputs['pred_observations'], next_obs)
        reward_recon_loss = F.mse_loss(outputs['pred_rewards'], rewards)
        continue_recon_loss = F.binary_cross_entropy(outputs['pred_continues'], 1.0 - dones)
        
        # KL divergence between posterior and prior
        posterior_mean = outputs['posterior_means']
        posterior_logvar = outputs['posterior_logvars']
        prior_mean = outputs['prior_means']
        prior_logvar = outputs['prior_logvars']
        
        kl_loss = 0.5 * torch.mean(
            prior_logvar - posterior_logvar + 
            (torch.exp(posterior_logvar) + (posterior_mean - prior_mean) ** 2) / torch.exp(prior_logvar) - 1
        )
        
        # Total world model loss
        total_loss = obs_recon_loss + reward_recon_loss + continue_recon_loss + 0.1 * kl_loss
        
        return total_loss, {
            'obs_recon': obs_recon_loss.item(),
            'reward_recon': reward_recon_loss.item(),
            'continue_recon': continue_recon_loss.item(),
            'kl_divergence': kl_loss.item()
        }
    
    def compute_actor_critic_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute actor and critic losses using imagination rollouts"""
        obs = batch['observations']
        actions = batch['actions']
        
        # Get initial states from world model (detached to avoid gradients)
        with torch.no_grad():
            initial_outputs = self.world_model(obs[:, :1], actions[:, :1])
            initial_states = initial_outputs['states'][:, 0].detach()
            initial_hidden = initial_outputs['hidden_states'][:, 0].detach()
        
        batch_size = initial_states.shape[0]
        
        # Generate imagination rollouts
        imagination_states = []
        imagination_actions = []
        imagination_rewards = []
        imagination_values = []
        imagination_continues = []
        
        states = initial_states
        hidden = initial_hidden
        
        for t in range(self.imagination_horizon):
            # Current state representation
            state_hidden = torch.cat([states, hidden], dim=-1)
            imagination_states.append(state_hidden)
            
            # Sample action from policy
            action, _ = self.actor.sample(state_hidden)
            imagination_actions.append(action)
            
            # Predict value
            value = self.critic(state_hidden)
            imagination_values.append(value)
            
            # Imagine next state (detached to avoid gradients through world model)
            with torch.no_grad():
                action_expanded = action.unsqueeze(1)  # [batch, 1, action_dim]
                imagination_output = self.world_model.imagine(states, hidden, action_expanded)
                
                # Extract predictions
                imagination_rewards.append(imagination_output['pred_rewards'][:, 0])
                imagination_continues.append(imagination_output['pred_continues'][:, 0])
                
                # Update state for next iteration
                states = imagination_output['states'][:, 0]
                hidden = imagination_output['hidden_states'][:, 0]
        
        # Final value estimate
        with torch.no_grad():
            final_state_hidden = torch.cat([states, hidden], dim=-1)
        final_value = self.critic(final_state_hidden)
        
        # Compute advantages using GAE
        advantages = []
        values = torch.stack(imagination_values)  # [horizon, batch]
        rewards = torch.stack(imagination_rewards)  # [horizon, batch]
        continues = torch.stack(imagination_continues)  # [horizon, batch]
        
        # Bootstrap from final value
        next_value = final_value.detach()
        advantage = 0
        
        for t in reversed(range(self.imagination_horizon)):
            delta = rewards[t] + self.gamma * continues[t] * next_value - values[t]
            advantage = delta + self.gamma * self.lambda_gae * continues[t] * advantage
            advantages.insert(0, advantage)
            next_value = values[t]
        
        advantages = torch.stack(advantages)  # [horizon, batch]
        returns = advantages.detach() + values
        
        # Actor loss (policy gradient)
        actor_actions = torch.stack(imagination_actions)  # [horizon, batch, action_dim]
        state_features = torch.stack(imagination_states)  # [horizon, batch, state_dim]
        
        # Compute log probabilities
        log_probs = []
        for t in range(self.imagination_horizon):
            _, log_prob = self.actor.sample(state_features[t])
            log_probs.append(log_prob)
        log_probs = torch.stack(log_probs)  # [horizon, batch]
        
        # Policy gradient loss
        actor_loss = -(log_probs * advantages.detach()).mean()
        
        # Critic loss
        critic_loss = F.mse_loss(values, returns.detach())
        
        return actor_loss, critic_loss
    
    def train_step(self, batch_size: int = 32) -> Dict[str, float]:
        """Perform one training step"""
        if len(self.replay_buffer) < batch_size:
            return {}
        
        # Sample batch
        batch = self.replay_buffer.sample(batch_size)
        if batch is None:
            return {}
        
        # Move to device
        for key in batch:
            batch[key] = batch[key].to(self.device)
        
        metrics = {}
        
        # Train world model
        self.world_model_optimizer.zero_grad()
        world_model_loss, wm_metrics = self.compute_world_model_loss(batch)
        world_model_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), 1.0)
        self.world_model_optimizer.step()
        
        metrics.update({f'world_model/{k}': v for k, v in wm_metrics.items()})
        metrics['world_model/total_loss'] = world_model_loss.item()
        
        # Train actor and critic (separate forward passes to avoid graph issues)
        with torch.no_grad():
            # Get detached batch for actor-critic training
            detached_batch = {k: v.detach() for k, v in batch.items()}
        
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        
        actor_loss, critic_loss = self.compute_actor_critic_loss(detached_batch)
        
        actor_loss.backward()
        critic_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        
        self.actor_optimizer.step()
        self.critic_optimizer.step()
        
        metrics['actor/loss'] = actor_loss.item()
        metrics['critic/loss'] = critic_loss.item()
        
        # Update training step
        self.training_step += 1
        
        # Store loss history
        self.world_model_loss_history.append(world_model_loss.item())
        self.actor_loss_history.append(actor_loss.item())
        self.critic_loss_history.append(critic_loss.item())
        
        # Add buffer statistics to metrics for monitoring success-prioritized learning
        buffer_stats = self.get_buffer_statistics()
        metrics.update({
            'buffer/total_sequences': buffer_stats['total'],
            'buffer/success_count': buffer_stats['success_count'],
            'buffer/success_ratio': buffer_stats['success_ratio'],
            'buffer/avg_success_distance': buffer_stats.get('avg_success_distance', float('inf')),
            'buffer/avg_success_reward': buffer_stats.get('avg_success_reward', 0.0)
        })
        
        return metrics
    
    def get_buffer_statistics(self) -> Dict[str, Any]:
        """Get statistics about the replay buffer"""
        return self.replay_buffer.get_success_statistics()
    
    def should_prioritize_successful_learning(self) -> bool:
        """Check if we have enough successful experiences to prioritize them"""
        stats = self.get_buffer_statistics()
        return stats['success_count'] > 5  # Prioritize once we have some successful experiences
    
    def get_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Get action from current policy"""
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            
            # Encode observation
            encoded_obs = self.world_model.encode_obs(obs_tensor)
            
            # Use zero hidden state for single-step inference
            hidden = torch.zeros(1, self.hidden_dim, device=self.device)
            state_hidden = torch.cat([encoded_obs, hidden], dim=-1)
            
            if deterministic:
                mean, _ = self.actor(state_hidden)
                action = torch.tanh(mean)
            else:
                action, _ = self.actor.sample(state_hidden)
            
            return action.cpu().numpy()[0]
    
    def save(self, path: str):
        """Save agent state"""
        torch.save({
            'world_model': self.world_model.state_dict(),
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'world_model_optimizer': self.world_model_optimizer.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'training_step': self.training_step,
            'config': {
                'obs_dim': self.obs_dim,
                'action_dim': self.action_dim,
                'hidden_dim': self.hidden_dim,
                'latent_dim': self.latent_dim,
                'imagination_horizon': self.imagination_horizon,
                'gamma': self.gamma,
                'lambda_gae': self.lambda_gae
            }
        }, path)
    
    def load(self, path: str):
        """Load agent state"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.world_model.load_state_dict(checkpoint['world_model'])
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.world_model_optimizer.load_state_dict(checkpoint['world_model_optimizer'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.training_step = checkpoint['training_step']


if __name__ == "__main__":
    # Test DreamerV3 components
    print("🤖 Testing DreamerV3 Components")
    print("=" * 50)
    
    # Test RSSM
    obs_dim = 33  # Drone observation dimension
    action_dim = 3  # Drone action dimension
    batch_size = 4
    sequence_length = 10
    
    rssm = RSSM(obs_dim, action_dim)
    
    # Create dummy data
    obs_seq = torch.randn(batch_size, sequence_length, obs_dim)
    action_seq = torch.randn(batch_size, sequence_length, action_dim)
    
    print(f"✓ RSSM initialized with obs_dim={obs_dim}, action_dim={action_dim}")
    
    # Test forward pass
    output = rssm(obs_seq, action_seq)
    print(f"✓ RSSM forward pass successful")
    print(f"  States shape: {output['states'].shape}")
    print(f"  Predicted observations shape: {output['pred_observations'].shape}")
    print(f"  Predicted rewards shape: {output['pred_rewards'].shape}")
    
    # Test imagination
    initial_state = torch.randn(batch_size, rssm.latent_dim)
    initial_hidden = torch.randn(batch_size, rssm.hidden_dim)
    action_sequence = torch.randn(batch_size, sequence_length, action_dim)
    
    imagination_output = rssm.imagine(initial_state, initial_hidden, action_sequence)
    print(f"✓ RSSM imagination successful")
    print(f"  Imagined states shape: {imagination_output['states'].shape}")
    
    # Test Actor and Critic
    state_dim = rssm.latent_dim + rssm.hidden_dim
    actor = Actor(state_dim, action_dim)
    critic = Critic(state_dim)
    
    state = torch.randn(batch_size, state_dim)
    action, log_prob = actor.sample(state)
    value = critic(state)
    
    print(f"✓ Actor-Critic networks successful")
    print(f"  Action shape: {action.shape}")
    print(f"  Value shape: {value.shape}")
    
    # Test DreamerV3 Agent
    agent = DreamerV3Agent(obs_dim, action_dim)
    print(f"✓ DreamerV3 Agent initialized")
    
    # Test action selection
    dummy_obs = np.random.randn(obs_dim)
    action = agent.get_action(dummy_obs)
    print(f"✓ Action selection successful: {action}")
    
    print("\n🎉 All DreamerV3 components working correctly!")
