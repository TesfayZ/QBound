"""
Simplified DDPG Agents for Comparison

Implements:
1. Simple DDPG: 1 actor + 1 critic, NO target networks
2. QBound-only: Simple DDPG + QBound clipping (no target networks)

These are used to test if QBound can replace target network stabilization.

QBound Support:
- Hard clipping on TD targets (prevents bootstrapping errors)
- Soft clipping on actor Q-values (preserves gradients for policy improvement)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random

try:
    from soft_qbound_penalty import SoftQBoundPenalty
    SOFT_QBOUND_AVAILABLE = True
except ImportError:
    SOFT_QBOUND_AVAILABLE = False


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


class SimpleDDPGAgent:
    """
    Simplified DDPG Agent WITHOUT target networks

    This tests if simpler architecture works or if target networks are necessary.

    Args:
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        max_action: Maximum absolute value of action
        lr_actor: Learning rate for actor
        lr_critic: Learning rate for critic
        gamma: Discount factor
        use_qbound: Whether to apply QBound (hard clipping on TD targets)
        qbound_min: Minimum Q-value bound (if using QBound)
        qbound_max: Maximum Q-value bound (if using QBound)
        use_soft_clip: Whether to apply soft clipping on actor Q-values
        soft_clip_beta: Steepness parameter for soft clipping (higher = closer to hard clip)
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        lr_actor=0.001,
        lr_critic=0.001,
        gamma=0.99,
        use_qbound=False,
        qbound_min=None,
        qbound_max=None,
        use_soft_clip=False,
        soft_clip_beta=0.1,
        use_step_aware_qbound=False,
        max_episode_steps=None,
        step_reward=None,
        device='cpu'
    ):
        self.device = device
        self.gamma = gamma
        self.max_action = max_action
        self.use_qbound = use_qbound
        self.qbound_min = qbound_min if qbound_min is not None else -np.inf
        self.qbound_max = qbound_max if qbound_max is not None else np.inf
        self.use_soft_clip = use_soft_clip
        self.soft_clip_beta = soft_clip_beta

        # Step-aware QBound parameters (for time-step dependent rewards)
        self.use_step_aware_qbound = use_step_aware_qbound
        self.max_episode_steps = max_episode_steps
        self.step_reward = step_reward

        # Initialize soft clipping function if needed
        if self.use_soft_clip and SOFT_QBOUND_AVAILABLE:
            self.penalty_fn = SoftQBoundPenalty()
        else:
            self.penalty_fn = None

        # Single actor network (NO target)
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)

        # Single critic network (NO target)
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Replay buffer
        self.replay_buffer = ReplayBuffer()

        # Exploration noise
        self.noise = OUNoise(action_dim)

    def compute_qbound(self, current_step=None):
        """
        Compute Q-value bounds.

        Static bounds (sparse rewards):
            Q_min, Q_max = known reward bounds with discount

        Dynamic bounds (dense rewards):
            Q_max(t) = sum_{k=0}^{H-t-1} γ^k * r
            For Pendulum: Q_max(t) = r * (1 - γ^(H-t)) / (1 - γ)
            where H = max_episode_steps, t = current_step, r = step_reward
        """
        if self.use_step_aware_qbound and current_step is not None:
            # Dynamic bound for dense reward survival tasks
            remaining_steps = self.max_episode_steps - current_step
            if remaining_steps > 0:
                # Geometric series for discounted sum of future rewards
                if abs(self.gamma - 1.0) < 1e-6:
                    # γ ≈ 1: undiscounted case
                    Q_max_dynamic = self.step_reward * remaining_steps
                else:
                    # Standard geometric series
                    Q_max_dynamic = self.step_reward * (1 - self.gamma ** remaining_steps) / (1 - self.gamma)
                return self.qbound_min, Q_max_dynamic
            else:
                # No remaining steps
                return self.qbound_min, 0.0
        else:
            # Static bounds for sparse reward tasks
            return self.qbound_min, self.qbound_max

    def select_action(self, state, add_noise=True):
        """Select action using actor network with optional exploration noise"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action = self.actor(state).cpu().numpy()[0]

        if add_noise:
            action = action + self.noise.sample()
            action = np.clip(action, -self.max_action, self.max_action)

        return action

    def train(self, batch_size=256, current_step=None):
        """
        Train the agent on a batch from replay buffer

        Args:
            batch_size: Number of transitions to sample
            current_step: Current time step (for dynamic QBound)
        """
        if len(self.replay_buffer) < batch_size:
            return None, None, None

        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # ===== Critic Update =====
        violation_stats = None

        # Compute bounds (static or dynamic)
        qbound_min, qbound_max = self.compute_qbound(current_step)

        # Compute target Q-value using CURRENT networks (no target networks)
        with torch.no_grad():
            next_actions = self.actor(next_states)
            next_q_raw = self.critic(next_states, next_actions)

            # Two-stage hard clipping (like DQN)
            if self.use_qbound:
                # STAGE 1: Clip next-state Q-values
                next_q_clipped = torch.clamp(next_q_raw, qbound_min, qbound_max)

                # Compute TD target
                target_q_raw = rewards + (1 - dones) * self.gamma * next_q_clipped

                # STAGE 2: Clip final TD target after adding reward
                target_q = torch.clamp(target_q_raw, qbound_min, qbound_max)

                # Track violations (for analysis)
                next_q_violate_max = (next_q_raw > qbound_max).float()
                next_q_violate_min = (next_q_raw < qbound_min).float()
                target_violate_max = (target_q_raw > qbound_max).float()
                target_violate_min = (target_q_raw < qbound_min).float()

                violation_stats = {
                    'next_q_violate_max_rate': next_q_violate_max.mean().item(),
                    'next_q_violate_min_rate': next_q_violate_min.mean().item(),
                    'target_violate_max_rate': target_violate_max.mean().item(),
                    'target_violate_min_rate': target_violate_min.mean().item(),
                    'total_violation_rate': ((next_q_violate_max + next_q_violate_min +
                                             target_violate_max + target_violate_min) > 0).float().mean().item(),
                    'violation_magnitude_max_next': torch.relu(next_q_raw - qbound_max).mean().item(),
                    'violation_magnitude_min_next': torch.relu(qbound_min - next_q_raw).mean().item(),
                    'violation_magnitude_max_target': torch.relu(target_q_raw - qbound_max).mean().item(),
                    'violation_magnitude_min_target': torch.relu(qbound_min - target_q_raw).mean().item(),
                    'qbound_max': qbound_max,
                    'qbound_min': qbound_min,
                }
            else:
                target_q = rewards + (1 - dones) * self.gamma * next_q_raw

        # Current Q-value
        current_q = self.critic(states, actions)

        # Critic loss - NO PENALTY! Bootstrapping handles bounds
        critic_loss = F.mse_loss(current_q, target_q)

        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ===== Actor Update =====
        # Compute Q-values for actor's actions
        q_for_actor = self.critic(states, self.actor(states))

        # Apply soft clipping to preserve gradients during policy improvement
        if self.use_qbound and self.use_soft_clip:
            q_for_actor = self.penalty_fn.softplus_clip(
                q_for_actor,
                torch.tensor(qbound_min, device=self.device),
                torch.tensor(qbound_max, device=self.device),
                beta=self.soft_clip_beta
            )

        # Maximize Q(s, μ(s))
        actor_loss = -q_for_actor.mean()

        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Return violation stats for tracking
        return critic_loss.item(), actor_loss.item(), violation_stats

    def reset_noise(self):
        """Reset exploration noise"""
        self.noise.reset()

    def save(self, filepath):
        """Save model parameters"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }, filepath)

    def load(self, filepath):
        """Load model parameters"""
        checkpoint = torch.load(filepath)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
