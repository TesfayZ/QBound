"""
Twin Delayed DDPG (TD3) Agent

Implementation of TD3 from Fujimoto et al. (2018):
"Addressing Function Approximation Error in Actor-Critic Methods"

Key features:
1. Clipped Double-Q Learning: Two critics, takes minimum (pessimistic)
2. Delayed Policy Updates: Updates actor less frequently than critics
3. Target Policy Smoothing: Adds noise to target actions

This is compared against QBound to test if simple environment-aware bounds
can replace TD3's complex stabilization mechanisms.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random


class ReplayBuffer:
    """Experience replay buffer"""

    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Add a transition to the buffer"""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Sample a batch of transitions"""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32)
        )

    def __len__(self):
        return len(self.buffer)


class Actor(nn.Module):
    """Actor network - outputs deterministic actions"""

    def __init__(self, state_dim, action_dim, max_action, hidden_dims=[400, 300]):
        super(Actor, self).__init__()

        self.max_action = max_action

        # Build network
        layers = []
        input_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim

        self.network = nn.Sequential(*layers)
        self.output = nn.Linear(input_dim, action_dim)

    def forward(self, state):
        x = self.network(state)
        action = torch.tanh(self.output(x)) * self.max_action
        return action


class Critic(nn.Module):
    """Critic network - outputs Q-value for state-action pair"""

    def __init__(self, state_dim, action_dim, hidden_dims=[400, 300]):
        super(Critic, self).__init__()

        # Concatenate state and action
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value


class OUNoise:
    """Ornstein-Uhlenbeck process for exploration noise"""

    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dim) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.action_dim)
        self.state = self.state + dx
        return self.state


class TD3Agent:
    """
    Twin Delayed DDPG (TD3) Agent

    Standard implementation with all TD3 features:
    - Two critic networks (clipped double-Q)
    - Target networks for actor and both critics
    - Delayed policy updates
    - Target policy smoothing

    Args:
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        max_action: Maximum absolute value of action
        lr_actor: Learning rate for actor
        lr_critic: Learning rate for critic
        gamma: Discount factor
        tau: Soft update parameter for target networks
        policy_noise: Noise added to target actions
        noise_clip: Maximum absolute value of target noise
        policy_freq: Frequency of delayed policy updates
        use_qbound: Whether to apply QBound to critics
        qbound_min: Minimum Q-value bound (if using QBound)
        qbound_max: Maximum Q-value bound (if using QBound)
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        lr_actor=0.001,
        lr_critic=0.001,
        gamma=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        use_qbound=False,
        qbound_min=None,
        qbound_max=None,
        device='cpu'
    ):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.max_action = max_action
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.total_it = 0
        self.use_qbound = use_qbound
        self.qbound_min = qbound_min
        self.qbound_max = qbound_max

        # Actor networks
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)

        # Twin critic networks (critic 1)
        self.critic_1 = Critic(state_dim, action_dim).to(device)
        self.critic_1_target = Critic(state_dim, action_dim).to(device)
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=lr_critic)

        # Twin critic networks (critic 2)
        self.critic_2 = Critic(state_dim, action_dim).to(device)
        self.critic_2_target = Critic(state_dim, action_dim).to(device)
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=lr_critic)

        # Replay buffer
        self.replay_buffer = ReplayBuffer()

        # Exploration noise
        self.noise = OUNoise(action_dim)

    def select_action(self, state, add_noise=True):
        """Select action using actor network with optional exploration noise"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action = self.actor(state).cpu().numpy()[0]

        if add_noise:
            action = action + self.noise.sample()
            action = np.clip(action, -self.max_action, self.max_action)

        return action

    def train(self, batch_size=256):
        """Train the agent on a batch from replay buffer"""
        if len(self.replay_buffer) < batch_size:
            return None, None

        self.total_it += 1

        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # ===== Critic Update =====
        with torch.no_grad():
            # Target policy smoothing: Add noise to target actions
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_actions = (self.actor_target(next_states) + noise).clamp(
                -self.max_action, self.max_action
            )

            # Clipped Double-Q: Compute Q-values from both critics, take minimum
            target_q1 = self.critic_1_target(next_states, next_actions)
            target_q2 = self.critic_2_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)

            # Apply QBound if enabled
            if self.use_qbound and self.qbound_min is not None and self.qbound_max is not None:
                target_q = torch.clamp(target_q, self.qbound_min, self.qbound_max)

            target_q = rewards + (1 - dones) * self.gamma * target_q

            # Safety clip targets if using QBound
            if self.use_qbound and self.qbound_min is not None and self.qbound_max is not None:
                target_q = torch.clamp(target_q, self.qbound_min, self.qbound_max)

        # Current Q-values from both critics
        current_q1 = self.critic_1(states, actions)
        current_q2 = self.critic_2(states, actions)

        # Critic losses
        critic_1_loss = F.mse_loss(current_q1, target_q)
        critic_2_loss = F.mse_loss(current_q2, target_q)

        # Update critic 1
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()

        # Update critic 2
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        actor_loss = None

        # ===== Delayed Actor Update =====
        if self.total_it % self.policy_freq == 0:
            # Maximize Q1(s, μ(s))
            actor_loss = -self.critic_1(states, self.actor(states)).mean()

            # Update actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # ===== Soft Update Target Networks =====
            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic_1, self.critic_1_target)
            self._soft_update(self.critic_2, self.critic_2_target)

            actor_loss = actor_loss.item()

        critic_loss = (critic_1_loss.item() + critic_2_loss.item()) / 2
        return critic_loss, actor_loss

    def _soft_update(self, source, target):
        """Soft update of target network parameters"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def reset_noise(self):
        """Reset exploration noise"""
        self.noise.reset()

    def save(self, filepath):
        """Save model parameters"""
        torch.save({
            'actor': self.actor.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_1': self.critic_1.state_dict(),
            'critic_1_target': self.critic_1_target.state_dict(),
            'critic_2': self.critic_2.state_dict(),
            'critic_2_target': self.critic_2_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_1_optimizer': self.critic_1_optimizer.state_dict(),
            'critic_2_optimizer': self.critic_2_optimizer.state_dict(),
        }, filepath)

    def load(self, filepath):
        """Load model parameters"""
        checkpoint = torch.load(filepath)
        self.actor.load_state_dict(checkpoint['actor'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic_1.load_state_dict(checkpoint['critic_1'])
        self.critic_1_target.load_state_dict(checkpoint['critic_1_target'])
        self.critic_2.load_state_dict(checkpoint['critic_2'])
        self.critic_2_target.load_state_dict(checkpoint['critic_2_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_1_optimizer.load_state_dict(checkpoint['critic_1_optimizer'])
        self.critic_2_optimizer.load_state_dict(checkpoint['critic_2_optimizer'])
