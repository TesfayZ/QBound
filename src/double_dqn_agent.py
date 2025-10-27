"""
Double DQN Agent - Industry Standard Solution to Overestimation

This implementation uses Double Q-Learning to reduce overestimation bias
WITHOUT hard Q-value clipping. This is the approach used in modern RL:
- TD3 (Twin Delayed DDPG)
- SAC (Soft Actor-Critic)
- Modern DQN implementations

Key Difference from QBound:
- QBound: Clips Q-values to arbitrary bounds (causes underestimation)
- Double DQN: Uses two networks for soft pessimism (no hard bounds)

Reference: "Deep Reinforcement Learning with Double Q-learning" (van Hasselt et al., 2015)
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

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> List[Tuple]:
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class DoubleDQNAgent:
    """
    Double DQN Agent - Modern approach to overestimation bias.

    Double Q-Learning Mechanism:
    1. Use online network to SELECT the best action:
       a* = argmax_a Q_online(s', a)

    2. Use target network to EVALUATE that action:
       Q_target(s', a*)

    3. Compute TD target:
       target = r + γ * Q_target(s', a*)

    This decouples action selection from value estimation, reducing
    overestimation bias WITHOUT hard clipping.

    Benefits over QBound:
    - No arbitrary bounds needed
    - No underestimation bias
    - Proven effective in practice
    - Industry standard
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
        device: str = "cpu",
        use_huber_loss: bool = True,
        gradient_clip: float = 1.0
    ):
        """
        Args:
            use_huber_loss: Use Huber loss instead of MSE (more robust to outliers)
            gradient_clip: Maximum gradient norm (prevents exploding gradients)
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = torch.device(device)
        self.use_huber_loss = use_huber_loss
        self.gradient_clip = gradient_clip

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

    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)

    def train_step(self) -> float:
        """
        Perform one training step with Double Q-Learning.

        Double DQN Update:
        1. Select best action using ONLINE network: a* = argmax Q_online(s', a)
        2. Evaluate action using TARGET network: Q_target(s', a*)
        3. Compute TD target: r + γ * Q_target(s', a*)
        4. Update online network to minimize TD error

        No Q-value clipping needed!
        """
        if len(self.replay_buffer) < self.batch_size:
            return 0.0

        # Sample batch
        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Current Q-values
        current_q_all = self.q_network(states)
        current_q_values = current_q_all.gather(1, actions.unsqueeze(1)).squeeze()

        # Double Q-Learning: Decouple action selection from evaluation
        with torch.no_grad():
            # Use ONLINE network to SELECT best action
            next_actions = self.q_network(next_states).argmax(1, keepdim=True)

            # Use TARGET network to EVALUATE that action
            next_q_values = self.target_network(next_states).gather(1, next_actions).squeeze()

            # Compute target Q-values
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute loss
        if self.use_huber_loss:
            # Huber loss is more robust to outliers than MSE
            loss = nn.SmoothL1Loss()(current_q_values, target_q_values)
        else:
            loss = nn.MSELoss()(current_q_values, target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping (prevents exploding gradients)
        if self.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.gradient_clip)

        self.optimizer.step()

        # Update target network
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        self.losses.append(loss.item())
        return loss.item()

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
