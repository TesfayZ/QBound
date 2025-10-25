"""
DQN Agent with optional QClip
Implements Deep Q-Network with neural network function approximator.
Supports QClip for improved learning in sparse reward environments.

QClip Implementation based on the paper:
- Clips both primary (current) and auxiliary (target) Q-values
- Applies proportional scaling to next state Q-values
- Prevents Q-value overestimation in sparse reward settings
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from typing import List, Tuple


class QNetwork(nn.Module):
    """Neural network for Q-value approximation."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(QNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return self.network(x)


class ReplayBuffer:
    """Experience replay buffer for DQN."""

    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, current_step=None):
        """
        Store transition with optional current_step for step-aware Q-bounds.

        Args:
            current_step: Current timestep in episode (for dynamic Q_max calculation)
        """
        self.buffer.append((state, action, reward, next_state, done, current_step))

    def sample(self, batch_size: int) -> List[Tuple]:
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    DQN Agent with optional QBound.

    QBound uses dual-loss training to enforce environment-aware value constraints:

    PRIMARY LOSS (Standard TD):
    - Clips next-state Q-values using per-sample proportional scaling
    - Computes Bellman targets: target = r + γ * clip(max_a' Q(s',a'))
    - Clips final targets to bounds
    - Updates network via TD error

    AUXILIARY LOSS (Bound Enforcement):
    - Only penalizes Q-values that violate bounds [Q_min, Q_max]
    - Clips violating Q-values to bounds without affecting others
    - Avoids degrading well-behaved actions within bounds
    - Teaches network to naturally output bounded Q-values

    Combined: total_loss = primary_loss + aux_weight * auxiliary_loss

    NOTE: Current Q-values are NEVER clipped during forward pass (preserves gradients)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 10000,
        batch_size: int = 64,
        target_update_freq: int = 100,
        use_qclip: bool = False,
        qclip_max: float = 1.0,
        qclip_min: float = 0.0,
        aux_weight: float = 0.5,
        device: str = "cpu",
        use_step_aware_qbound: bool = False,
        max_episode_steps: int = 500,
        step_reward: float = 1.0
    ):
        """
        Args:
            use_step_aware_qbound: Enable step-aware dynamic Q-bounds (for dense rewards)
            max_episode_steps: Maximum episode length (for computing dynamic Q_max)
            step_reward: Reward per step (for computing dynamic Q_max)
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.use_qclip = use_qclip
        self.qclip_max = qclip_max
        self.qclip_min = qclip_min
        self.aux_weight = aux_weight
        self.device = torch.device(device)

        # Step-aware Q-bounds for dense rewards
        self.use_step_aware_qbound = use_step_aware_qbound
        self.max_episode_steps = max_episode_steps
        self.step_reward = step_reward

        # Q-networks
        self.q_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer(buffer_size)

        self.steps = 0
        self.losses = []

    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> int:
        """Select action using epsilon-greedy policy."""
        if not eval_mode and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state, done, current_step=None):
        """
        Store transition in replay buffer.

        Args:
            current_step: Current timestep in episode (for step-aware Q-bounds)
        """
        self.replay_buffer.push(state, action, reward, next_state, done, current_step)

    def train_step(self) -> float:
        """
        Perform one training step with optional QBound.

        QBound uses TWO losses:
        1. PRIMARY LOSS: Standard TD loss for the taken action
        2. AUXILIARY LOSS: Penalizes only Q-values that violate [Q_min, Q_max]

        The auxiliary loss teaches the network to output bounded Q-values by
        clipping only the violating actions, leaving well-behaved actions unchanged.
        This avoids degrading good learners due to one bad action.
        """
        if len(self.replay_buffer) < self.batch_size:
            return 0.0

        # Sample batch
        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones, current_steps = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Compute dynamic Q_max for step-aware bounds (dense rewards)
        if self.use_step_aware_qbound and current_steps[0] is not None:
            # Q_max = (max_steps - current_step) * reward_per_step
            current_steps_tensor = torch.FloatTensor([s if s is not None else 0 for s in current_steps]).to(self.device)
            dynamic_qmax = (self.max_episode_steps - current_steps_tensor) * self.step_reward
            dynamic_qmin = torch.zeros_like(dynamic_qmax)
        else:
            # Use static Q bounds
            dynamic_qmax = torch.full((self.batch_size,), self.qclip_max, device=self.device)
            dynamic_qmin = torch.full((self.batch_size,), self.qclip_min, device=self.device)

        # Current Q-values - NEVER CLIP THESE (breaks gradient flow)
        current_q_all = self.q_network(states)
        current_q_values = current_q_all.gather(1, actions.unsqueeze(1)).squeeze()

        # ============================================================
        # PRIMARY LOSS: Standard TD loss
        # ============================================================
        with torch.no_grad():
            # Get next state Q-values from target network
            next_q_values_all = self.target_network(next_states)
            next_q_values = next_q_values_all.max(1)[0]

            if self.use_qclip:
                # Use dynamic Q-bounds for clipping (per-sample bounds)
                # For step-aware: Q_max decreases as episode progresses
                # For static: Q_max is constant across all samples
                qmax_for_clipping = dynamic_qmax
                qmin_for_clipping = dynamic_qmin

                # Clip next state Q-values to dynamic bounds (bootstrapping only, no auxiliary)
                next_q_values = torch.clamp(next_q_values,
                                           min=qmin_for_clipping,
                                           max=qmax_for_clipping)

            # Compute target Q-values
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

            if self.use_qclip:
                # Clip target Q-values to dynamic bounds
                target_q_values = torch.clamp(target_q_values,
                                             min=dynamic_qmin,
                                             max=dynamic_qmax)

        # Primary TD loss (only loss used - bootstrapping handles Q-bounds)
        total_loss = nn.MSELoss()(current_q_values, target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # Update target network
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        self.losses.append(total_loss.item())
        return total_loss.item()

    def save(self, path: str):
        """Save model weights."""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, path)

    def load(self, path: str):
        """Load model weights."""
        checkpoint = torch.load(path)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
