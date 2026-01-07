# ==============================================================================
# PROMPT FOR COPILOT:
# I need to implement a PPO (Proximal Policy Optimization) Agent using PyTorch.
# Key Requirement: The Actor Network must use a Multi-Head Attention (MHA) layer 
# to process the input state before making decisions, as per Ahmmed et al. (2025).
# ==============================================================================

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Normal

# 1. Define the MHA_Actor class (The Brain)
# Input: state_dim (Wind_x, Wind_y, Battery, Goal_Dist, Velocity)
# Architecture: Input -> Multi-Head Attention -> Linear Layer -> Tanh Activation -> Action (Thrust, Pitch, Roll)
# Use 4 Attention Heads.
class MHA_Actor(nn.Module):
    def __init__(self, state_dim, action_dim, num_heads=4, embed_dim=64):
        super(MHA_Actor, self).__init__()
        
        # Input embedding layer to project state to embed_dim
        self.input_projection = nn.Linear(state_dim, embed_dim)
        
        # Multi-Head Attention layer
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        
        # Fully connected layers after attention
        self.fc1 = nn.Linear(embed_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        
        # Output layers for mean and log_std of action distribution
        self.mean_layer = nn.Linear(64, action_dim)
        self.log_std_layer = nn.Linear(64, action_dim)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
    
    def forward(self, state):
        # Project input state to embedding dimension
        x = self.input_projection(state)
        
        # Multi-Head Attention expects (batch, seq_len, embed_dim)
        # Since we have a single state vector, we add a sequence dimension
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, embed_dim)
        
        # Apply Multi-Head Attention (self-attention)
        attn_output, _ = self.mha(x, x, x)
        
        # Remove sequence dimension
        x = attn_output.squeeze(1)  # (batch, embed_dim)
        
        # Pass through fully connected layers
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        
        # Get mean and log_std for action distribution
        mean = self.tanh(self.mean_layer(x))  # Bounded actions [-1, 1]
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, min=-20, max=2)  # Stability
        
        return mean, log_std

# 2. Define the Critic class (The Evaluator)
# Input: state_dim
# Output: Single value (V) estimating how "good" the current state is.
class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)  # Single value output
        
        self.relu = nn.ReLU()
    
    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        value = self.fc3(x)
        return value

# 3. Define the PPO Agent class
# This class handles memory (replay buffer), selecting actions, and updating the networks.
# It needs an 'update' function that calculates the PPO Loss and updates weights.
class PPO_Agent:
    def __init__(self, state_dim, action_dim, lr=0.0003, gamma=0.99, eps_clip=0.2, K_epochs=4):
        # Initialize Actor and Critic networks
        self.actor = MHA_Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        # Old policy for PPO (used to calculate ratio)
        self.old_actor = MHA_Actor(state_dim, action_dim)
        self.old_actor.load_state_dict(self.actor.state_dict())
        
        # Hyperparameters
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        # Memory buffers
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.dones = []
        
        # Loss function
        self.MseLoss = nn.MSELoss()
    
    def select_action(self, state):
        # Convert state to tensor
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state)
        
        # Add batch dimension if needed
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        # Get action distribution from actor
        with torch.no_grad():
            mean, log_std = self.old_actor(state)
            std = log_std.exp()
            
            # Create normal distribution and sample action
            dist = Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
        
        # Store in memory
        self.states.append(state.squeeze(0))
        self.actions.append(action.squeeze(0))
        self.log_probs.append(log_prob)
        
        return action.squeeze(0).numpy()
    
    def store_reward(self, reward, done):
        # Store reward and done flag
        self.rewards.append(reward)
        self.dones.append(done)
    
    def update(self):
        # Convert lists to tensors
        states = torch.stack(self.states).detach()
        actions = torch.stack(self.actions).detach()
        old_log_probs = torch.stack(self.log_probs).detach()
        
        # Calculate discounted rewards (Monte Carlo Returns)
        rewards = []
        discounted_reward = 0
        for reward, done in zip(reversed(self.rewards), reversed(self.dones)):
            if done:
                discounted_reward = 0
            discounted_reward = reward + self.gamma * discounted_reward
            rewards.insert(0, discounted_reward)
        
        # Normalize rewards
        rewards = torch.FloatTensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        # PPO update for K epochs
        for _ in range(self.K_epochs):
            # Get current policy predictions
            mean, log_std = self.actor(states)
            std = log_std.exp()
            dist = Normal(mean, std)
            
            # Calculate log probabilities of actions under current policy
            log_probs = dist.log_prob(actions).sum(dim=-1)
            
            # Calculate entropy for exploration
            entropy = dist.entropy().sum(dim=-1).mean()
            
            # Get state values from critic
            state_values = self.critic(states).squeeze()
            
            # Calculate advantages
            advantages = rewards - state_values.detach()
            
            # Calculate ratio (pi_theta / pi_theta_old)
            ratios = torch.exp(log_probs - old_log_probs)
            
            # Calculate surrogate losses
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            # Actor loss (PPO clipped objective + entropy bonus)
            actor_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy
            
            # Critic loss (MSE between predicted value and actual return)
            critic_loss = self.MseLoss(state_values, rewards)
            
            # Update Actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor_optimizer.step()
            
            # Update Critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.critic_optimizer.step()
        
        # Copy new weights to old policy
        self.old_actor.load_state_dict(self.actor.state_dict())
        
        # Clear memory
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.dones.clear()
    
    def save(self, filepath):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, filepath)
    
    def load(self, filepath):
        checkpoint = torch.load(filepath)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.old_actor.load_state_dict(self.actor.state_dict())
