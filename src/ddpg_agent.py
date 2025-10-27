"""
Deep Deterministic Policy Gradient (DDPG) Agent

Implementation of DDPG for continuous control environments.
Based on Lillicrap et al. (2015) "Continuous Control with Deep Reinforcement Learning"

This implementation will be used to compare against QBound on continuous control tasks.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random


class ReplayBuffer:
    """Experience replay buffer for DDPG"""

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
    """Actor network for DDPG - outputs deterministic actions"""

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
    """Critic network for DDPG - outputs Q-value for state-action pair"""

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


class DDPGAgent:
    """
    Deep Deterministic Policy Gradient Agent

    Args:
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        max_action: Maximum absolute value of action
        lr_actor: Learning rate for actor
        lr_critic: Learning rate for critic
        gamma: Discount factor
        tau: Soft update parameter for target networks
        use_qbound: Whether to apply QBound to critic
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
        use_qbound=False,
        qbound_min=None,
        qbound_max=None,
        device='cpu'
    ):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.max_action = max_action
        self.use_qbound = use_qbound
        self.qbound_min = qbound_min
        self.qbound_max = qbound_max

        # Actor networks
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)

        # Critic networks
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

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

        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # ===== Critic Update =====
        with torch.no_grad():
            # Compute target Q-value
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)

            # Apply QBound if enabled
            if self.use_qbound and self.qbound_min is not None and self.qbound_max is not None:
                target_q = torch.clamp(target_q, self.qbound_min, self.qbound_max)

            target_q = rewards + (1 - dones) * self.gamma * target_q

            # Safety clip targets if using QBound
            if self.use_qbound and self.qbound_min is not None and self.qbound_max is not None:
                target_q = torch.clamp(target_q, self.qbound_min, self.qbound_max)

        # Current Q-value
        current_q = self.critic(states, actions)

        # Critic loss
        critic_loss = F.mse_loss(current_q, target_q)

        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ===== Actor Update =====
        # Maximize Q(s, μ(s))
        actor_loss = -self.critic(states, self.actor(states)).mean()

        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ===== Soft Update Target Networks =====
        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)

        return critic_loss.item(), actor_loss.item()

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
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }, filepath)

    def load(self, filepath):
        """Load model parameters"""
        checkpoint = torch.load(filepath)
        self.actor.load_state_dict(checkpoint['actor'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
