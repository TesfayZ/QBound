"""
Transformed DQN Agent - Shifts negative Q-values to positive range

This agent transforms negative reward environments to positive Q-value spaces by:
1. Computing abs_Q_min = |Q_min|
2. Transforming bounds: Q_min_new = 0, Q_max_new = abs_Q_min
3. Adding abs_Q_min to target Q-values BEFORE computing MSE loss
4. Applying QBound clipping with transformed bounds

The transformation tests whether QBound's failure on negative rewards is due to
the negative value range itself, or due to other factors like violation rates.

Key Insight:
- Pendulum (negative rewards): Q ∈ [-1409, 0] → Transform to Q ∈ [0, 1409]
- MountainCar (negative rewards): Q ∈ [-86.6, 0] → Transform to Q ∈ [0, 86.6]
- Acrobot (negative rewards): Q ∈ [-99.3, 0] → Transform to Q ∈ [0, 99.3]

This mirrors CartPole's positive reward structure: Q ∈ [0, 99.34]
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
        """Store transition."""
        self.buffer.append((state, action, reward, next_state, done, current_step))

    def sample(self, batch_size: int) -> List[Tuple]:
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class TransformedDQNAgent:
    """
    Transformed DQN Agent - Shifts negative Q-values to positive range.

    Transformation:
    1. Original bounds: Q ∈ [Q_min, Q_max] where Q_min < 0, Q_max = 0
    2. Compute shift: abs_Q_min = |Q_min|
    3. New bounds: Q ∈ [0, abs_Q_min]
    4. Apply transformation: Q_transformed = Q_original + abs_Q_min

    Implementation:
    - Target Q-values are shifted by adding abs_Q_min before computing MSE
    - QBound clipping uses transformed bounds [0, abs_Q_min]
    - Network outputs are interpreted as transformed Q-values
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
        qclip_max_original: float = 0.0,  # Original Q_max (e.g., 0.0)
        qclip_min_original: float = -100.0,  # Original Q_min (e.g., -86.6)
        device: str = "cpu"
    ):
        """
        Args:
            qclip_max_original: Original Q_max (typically 0 for negative rewards)
            qclip_min_original: Original Q_min (typically negative)
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
        self.device = torch.device(device)

        # Compute transformation shift
        self.abs_qmin = abs(qclip_min_original)

        # Transformed bounds (shifted to positive range)
        self.qclip_min_transformed = 0.0  # Q_min + abs_Q_min
        self.qclip_max_transformed = self.abs_qmin  # Q_max + abs_Q_min

        print(f"\n🔄 Q-Value Transformation:")
        print(f"   Original bounds: Q ∈ [{qclip_min_original:.2f}, {qclip_max_original:.2f}]")
        print(f"   Shift amount: +{self.abs_qmin:.2f}")
        print(f"   Transformed bounds: Q ∈ [{self.qclip_min_transformed:.2f}, {self.qclip_max_transformed:.2f}]")

        # Q-networks
        self.q_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer(buffer_size)

        self.steps = 0
        self.losses = []
        self.violation_stats_history = []

    def select_action(self, state, eval_mode=False):
        """Select action using epsilon-greedy policy."""
        if not eval_mode and np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return q_values.argmax(1).item()

    def store_transition(self, state, action, reward, next_state, done, current_step=None):
        """Store transition in replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done, current_step)

    def train_step(self):
        """
        Train the agent for one step with Q-value transformation.

        The transformation is applied to the target Q-values BEFORE computing MSE:
        1. Compute raw next-state Q-values from target network
        2. Clip next-state Q-values to transformed bounds (if QBound enabled)
        3. Compute raw TD target: target_raw = r + γ * max Q(s',a')
        4. TRANSFORM: target_transformed = target_raw + abs_Q_min
        5. Clip transformed target to transformed bounds
        6. Compute MSE loss between current Q and transformed target

        Returns:
            total_loss (float): TD loss value
            violation_stats (dict or None): QBound violation statistics if use_qclip=True
        """
        if len(self.replay_buffer) < self.batch_size:
            return 0.0, None

        # Sample batch
        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones, current_steps = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Use static transformed Q bounds
        qmax_transformed = torch.full((self.batch_size,), self.qclip_max_transformed, device=self.device)
        qmin_transformed = torch.full((self.batch_size,), self.qclip_min_transformed, device=self.device)

        # Current Q-values (interpreted as transformed Q-values)
        current_q_all = self.q_network(states)
        current_q_values = current_q_all.gather(1, actions.unsqueeze(1)).squeeze()

        # ============================================================
        # TRANSFORMED Q-VALUE LEARNING
        # ============================================================
        violation_stats = None

        with torch.no_grad():
            # Get next state Q-values from target network (transformed Q-values)
            next_q_values_all = self.target_network(next_states)
            next_q_values_raw = next_q_values_all.max(1)[0]

            # Track violations BEFORE clipping (for analysis)
            if self.use_qclip:
                # Violations in next-state transformed Q-values
                next_q_violate_max = (next_q_values_raw > qmax_transformed).float()
                next_q_violate_min = (next_q_values_raw < qmin_transformed).float()

                violation_magnitude_max_next = torch.relu(next_q_values_raw - qmax_transformed)
                violation_magnitude_min_next = torch.relu(qmin_transformed - next_q_values_raw)

                # STEP 1: Clip next-state transformed Q-values to bounds
                next_q_values = torch.clamp(next_q_values_raw,
                                           min=qmin_transformed,
                                           max=qmax_transformed)
            else:
                next_q_values = next_q_values_raw

            # Compute raw TD target (in ORIGINAL Q-space, not transformed yet)
            # This is the standard Bellman equation: Q(s,a) = r + γ * max Q(s',a')
            # BUT next_q_values are already transformed, so we need to un-transform first
            if self.use_qclip:
                # Un-transform next Q-values to original space
                next_q_original = next_q_values - self.abs_qmin
                # Compute TD target in original space
                target_q_original = rewards + (1 - dones) * self.gamma * next_q_original
                # Transform to positive space
                target_q_transformed = target_q_original + self.abs_qmin
            else:
                # Without QBound, network learns in whatever space it wants
                # We still interpret rewards as original space, so transform
                target_q_original = rewards + (1 - dones) * self.gamma * (next_q_values - self.abs_qmin)
                target_q_transformed = target_q_original + self.abs_qmin

            if self.use_qclip:
                # Track violations in transformed TD targets BEFORE final clipping
                target_violate_max = (target_q_transformed > qmax_transformed).float()
                target_violate_min = (target_q_transformed < qmin_transformed).float()

                violation_magnitude_max_target = torch.relu(target_q_transformed - qmax_transformed)
                violation_magnitude_min_target = torch.relu(qmin_transformed - target_q_transformed)

                # STEP 2: Clip transformed target to transformed bounds
                target_q_values = torch.clamp(target_q_transformed,
                                             min=qmin_transformed,
                                             max=qmax_transformed)

                # Compute violation statistics
                violation_stats = {
                    'next_q_violate_max_rate': next_q_violate_max.mean().item(),
                    'next_q_violate_min_rate': next_q_violate_min.mean().item(),
                    'target_violate_max_rate': target_violate_max.mean().item(),
                    'target_violate_min_rate': target_violate_min.mean().item(),
                    'total_violation_rate': ((next_q_violate_max + next_q_violate_min +
                                             target_violate_max + target_violate_min) > 0).float().mean().item(),
                    'violation_magnitude_max_next': violation_magnitude_max_next.mean().item(),
                    'violation_magnitude_min_next': violation_magnitude_min_next.mean().item(),
                    'violation_magnitude_max_target': violation_magnitude_max_target.mean().item(),
                    'violation_magnitude_min_target': violation_magnitude_min_target.mean().item(),
                    'qbound_max': qmax_transformed.mean().item(),
                    'qbound_min': qmin_transformed.mean().item(),
                }

                self.violation_stats_history.append(violation_stats)
            else:
                target_q_values = target_q_transformed

        # TD loss (current Q-values vs transformed targets)
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
        return total_loss.item(), violation_stats

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
