"""
Deep Deterministic Policy Gradient (DDPG) Agent

Implementation of DDPG for continuous control environments.
Based on Lillicrap et al. (2015) "Continuous Control with Deep Reinforcement Learning"

This implementation will be used to compare against QBound on continuous control tasks.

QBound Support:
- Hard clipping on TD targets (prevents bootstrapping errors)
- Soft clipping on actor Q-values (preserves gradients for policy improvement)
"""

import sys
sys.path.insert(0, '/root/projects/QBound/src')

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
    """Experience replay buffer for DDPG"""

    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, time_step=None):
        """
        Add a transition to the buffer

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
            time_step: Time step within episode (for dynamic QBound)
        """
        self.buffer.append((state, action, reward, next_state, done, time_step))

    def sample(self, batch_size):
        """Sample a batch of transitions with time steps"""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, time_steps = zip(*batch)

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
            np.array(time_steps, dtype=np.int32) if time_steps[0] is not None else None
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

    def __init__(self, state_dim, action_dim, hidden_dims=[400, 300],
                 use_negative_activation=False):
        super(Critic, self).__init__()
        self.use_negative_activation = use_negative_activation

        # Concatenate state and action
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)

        if self.use_negative_activation:
            # Architectural bound for negative rewards: Q ≤ 0
            q_value = -F.softplus(logits)
        else:
            q_value = logits

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
        tau=0.005,
        use_qbound=False,
        qbound_min=None,
        qbound_max=None,
        use_soft_clip=False,
        soft_clip_beta=0.1,
        use_step_aware_qbound=False,
        max_episode_steps=None,
        step_reward=None,
        use_architectural_qbound=False,
        device='cpu'
    ):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.max_action = max_action
        self.use_qbound = use_qbound
        self.qbound_min = qbound_min if qbound_min is not None else -np.inf
        self.qbound_max = qbound_max if qbound_max is not None else np.inf
        self.use_soft_clip = use_soft_clip
        self.soft_clip_beta = soft_clip_beta
        self.use_architectural_qbound = use_architectural_qbound

        # Step-aware QBound parameters (for time-step dependent rewards)
        self.use_step_aware_qbound = use_step_aware_qbound
        self.max_episode_steps = max_episode_steps
        self.step_reward = step_reward

        # Initialize soft clipping function if needed
        if self.use_soft_clip and SOFT_QBOUND_AVAILABLE:
            self.penalty_fn = SoftQBoundPenalty()
        else:
            self.penalty_fn = None

        # Actor networks
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)

        # Critic networks with optional architectural bounds
        self.critic = Critic(state_dim, action_dim,
                            use_negative_activation=use_architectural_qbound).to(device)
        self.critic_target = Critic(state_dim, action_dim,
                                    use_negative_activation=use_architectural_qbound).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
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
                    bound_dynamic = self.step_reward * remaining_steps
                else:
                    # Standard geometric series
                    bound_dynamic = self.step_reward * (1 - self.gamma ** remaining_steps) / (1 - self.gamma)

                # For positive rewards: Q ∈ [0, Q_max(t)] where Q_max(t) decreases
                # For negative rewards: Q ∈ [Q_min(t), 0] where Q_min(t) increases (becomes less negative)
                if self.step_reward >= 0:
                    return 0.0, bound_dynamic  # Positive: Q_max decreases, Q_min stays at 0
                else:
                    return bound_dynamic, 0.0  # Negative: Q_min becomes less negative, Q_max stays at 0
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
            current_step: Current time step (DEPRECATED - now uses per-transition steps from buffer)
        """
        if len(self.replay_buffer) < batch_size:
            return None, None, None

        # Sample batch WITH TIME STEPS
        states, actions, rewards, next_states, dones, time_steps = self.replay_buffer.sample(batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # ===== Critic Update =====
        violation_stats = None

        # Compute bounds per-transition (FIXED: was using single current_step for whole batch)
        if self.use_step_aware_qbound and time_steps is not None:
            # Compute bounds for EACH transition based on its time step
            qbound_mins = []
            qbound_maxs = []
            for t in time_steps:
                q_min, q_max = self.compute_qbound(current_step=t)
                qbound_mins.append(q_min)
                qbound_maxs.append(q_max)

            qbound_mins = torch.tensor(qbound_mins, device=self.device, dtype=torch.float32).unsqueeze(1)
            qbound_maxs = torch.tensor(qbound_maxs, device=self.device, dtype=torch.float32).unsqueeze(1)
        else:
            # Static bounds (same for all transitions)
            qbound_min, qbound_max = self.compute_qbound(None)
            qbound_mins = torch.full((batch_size, 1), qbound_min, device=self.device, dtype=torch.float32)
            qbound_maxs = torch.full((batch_size, 1), qbound_max, device=self.device, dtype=torch.float32)

        with torch.no_grad():
            # Compute target Q-value
            next_actions = self.actor_target(next_states)
            next_q_raw = self.critic_target(next_states, next_actions)

            # Two-stage hard clipping (like DQN) with PER-TRANSITION bounds
            if self.use_qbound:
                # STAGE 1: Clip next-state Q-values with per-transition bounds
                next_q_clipped = torch.clamp(next_q_raw, qbound_mins, qbound_maxs)

                # Compute TD target
                target_q_raw = rewards + (1 - dones) * self.gamma * next_q_clipped

                # STAGE 2: Clip final TD target after adding reward
                target_q = torch.clamp(target_q_raw, qbound_mins, qbound_maxs)

                # Track violations in TD error (for analysis)
                next_q_violate_max = (next_q_raw > qbound_maxs).float()
                next_q_violate_min = (next_q_raw < qbound_mins).float()
                target_violate_max = (target_q_raw > qbound_maxs).float()
                target_violate_min = (target_q_raw < qbound_mins).float()

                violation_stats = {
                    'next_q_violate_max_rate': next_q_violate_max.mean().item(),
                    'next_q_violate_min_rate': next_q_violate_min.mean().item(),
                    'target_violate_max_rate': target_violate_max.mean().item(),
                    'target_violate_min_rate': target_violate_min.mean().item(),
                    'total_violation_rate': ((next_q_violate_max + next_q_violate_min +
                                             target_violate_max + target_violate_min) > 0).float().mean().item(),
                    'violation_magnitude_max_next': torch.relu(next_q_raw - qbound_maxs).mean().item(),
                    'violation_magnitude_min_next': torch.relu(qbound_mins - next_q_raw).mean().item(),
                    'violation_magnitude_max_target': torch.relu(target_q_raw - qbound_maxs).mean().item(),
                    'violation_magnitude_min_target': torch.relu(qbound_mins - target_q_raw).mean().item(),
                    'qbound_max': qbound_maxs.mean().item(),  # Average for logging
                    'qbound_min': qbound_mins.mean().item(),  # Average for logging
                }
            else:
                target_q = rewards + (1 - dones) * self.gamma * next_q_raw

        # Current Q-value (from replay buffer actions)
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
            # Use per-transition bounds for soft clipping too
            q_for_actor = self.penalty_fn.softplus_clip(
                q_for_actor,
                qbound_mins,
                qbound_maxs,
                beta=self.soft_clip_beta
            )

        # Maximize Q(s, μ(s))
        actor_loss = -q_for_actor.mean()

        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ===== Soft Update Target Networks =====
        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)

        # Return violation stats
        return critic_loss.item(), actor_loss.item(), violation_stats

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
