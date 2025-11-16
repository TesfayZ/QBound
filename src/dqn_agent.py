"""
DQN Agent with optional QBound
Implements Deep Q-Network with neural network function approximator.
Supports QBound for improved learning in sparse and dense reward environments.

QBound Implementation:
- Clips next-state Q-values to environment-aware bounds during bootstrapping
- Prevents Q-value overestimation through bounded Bellman targets
- Supports static bounds (sparse rewards) and dynamic bounds (dense rewards)
- No auxiliary loss needed - bootstrapping naturally enforces bounds
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

    QBound enforces environment-aware Q-value bounds through bootstrapping:

    MECHANISM:
    - Clips next-state Q-values to [Q_min, Q_max] bounds
    - Computes Bellman targets: target = r + γ * clip(max_a' Q(s',a'))
    - Clips final targets to bounds for safety
    - Standard TD loss updates network

    KEY INSIGHT:
    Since RL agents select actions based on CURRENT Q-values (not next-state Q-values),
    the bootstrapping process naturally propagates the bounds through the network.
    No auxiliary loss is needed - clipping during target computation is sufficient.

    BOUNDS:
    - Static bounds for sparse rewards (e.g., Q_max = 1.0 for binary rewards)
    - Dynamic step-aware bounds for dense rewards (e.g., Q_max(t) = H - t for survival tasks)
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
        device: str = "cpu",
        use_step_aware_qbound: bool = False,
        max_episode_steps: int = 500,
        step_reward: float = 1.0,
        reward_is_negative: bool = False
    ):
        """
        Args:
            use_step_aware_qbound: Enable step-aware dynamic Q-bounds
            max_episode_steps: Maximum episode length (for computing dynamic Q-bounds)
            step_reward: Reward magnitude per step (for computing dynamic Q-bounds)
            reward_is_negative: True for negative rewards (sparse), False for positive (dense)
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
        self.device = torch.device(device)

        # Step-aware Q-bounds
        self.use_step_aware_qbound = use_step_aware_qbound
        self.max_episode_steps = max_episode_steps
        self.step_reward = step_reward
        self.reward_is_negative = reward_is_negative

        # Q-networks
        self.q_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer(buffer_size)

        self.steps = 0
        self.losses = []

        # Violation tracking for QBound analysis
        self.violation_stats_history = []

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

    def train_step(self):
        """
        Perform one training step with optional QBound.

        QBound simply clips next-state Q-values during bootstrapping:
        - Clipped targets: target = r + γ * clip(max_a' Q(s',a'), Q_min, Q_max)
        - Standard TD loss: MSE(Q(s,a), target)

        Since agents select actions using current Q-values, bootstrapping naturally
        propagates bounds through the network without needing auxiliary losses.

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

        # Compute dynamic Q bounds for step-aware bounds
        if self.use_step_aware_qbound and current_steps[0] is not None:
            # CORRECT FORMULA: geometric sum = (1 - γ^(H-t)) / (1 - γ)
            current_steps_tensor = torch.FloatTensor([s if s is not None else 0 for s in current_steps]).to(self.device)
            remaining_steps_current = self.max_episode_steps - current_steps_tensor
            remaining_steps_next = torch.clamp(remaining_steps_current - 1, min=0)

            # Dynamic bounds for CURRENT state (time t) - for clipping final target
            geometric_sum_current = (1 - torch.pow(self.gamma, remaining_steps_current)) / (1 - self.gamma)

            # Dynamic bounds for NEXT state (time t+1) - for clipping next-state Q-values
            geometric_sum_next = (1 - torch.pow(self.gamma, remaining_steps_next)) / (1 - self.gamma)

            if self.reward_is_negative:
                # Negative rewards: Q ∈ [Q_min(t), 0]
                # step_reward is already negative (e.g., -16.27), geometric_sum is positive
                # Q_min(t) = step_reward * geometric_sum = (-16.27) * 86.60 = -1409.33 ✓
                dynamic_qmin_current = geometric_sum_current * self.step_reward
                dynamic_qmax_current = torch.zeros_like(dynamic_qmin_current)
                dynamic_qmin_next = geometric_sum_next * self.step_reward
                dynamic_qmax_next = torch.zeros_like(dynamic_qmin_next)
            else:
                # Positive rewards: Q ∈ [0, Q_max(t)]
                dynamic_qmin_current = torch.zeros_like(geometric_sum_current)
                dynamic_qmax_current = geometric_sum_current * self.step_reward
                dynamic_qmin_next = torch.zeros_like(geometric_sum_next)
                dynamic_qmax_next = geometric_sum_next * self.step_reward
        else:
            # Use static Q bounds (same for current and next state)
            dynamic_qmax_current = torch.full((self.batch_size,), self.qclip_max, device=self.device)
            dynamic_qmin_current = torch.full((self.batch_size,), self.qclip_min, device=self.device)
            dynamic_qmax_next = torch.full((self.batch_size,), self.qclip_max, device=self.device)
            dynamic_qmin_next = torch.full((self.batch_size,), self.qclip_min, device=self.device)

        # Current Q-values - NEVER CLIP THESE (breaks gradient flow)
        current_q_all = self.q_network(states)
        current_q_values = current_q_all.gather(1, actions.unsqueeze(1)).squeeze()

        # ============================================================
        # PRIMARY LOSS: Standard TD loss with QBound
        # ============================================================
        violation_stats = None

        with torch.no_grad():
            # Get next state Q-values from target network
            next_q_values_all = self.target_network(next_states)
            next_q_values_raw = next_q_values_all.max(1)[0]

            # Track violations BEFORE clipping (for analysis)
            if self.use_qclip:
                # Violations in next-state Q-values
                next_q_violate_max = (next_q_values_raw > dynamic_qmax_next).float()
                next_q_violate_min = (next_q_values_raw < dynamic_qmin_next).float()

                violation_magnitude_max_next = torch.relu(next_q_values_raw - dynamic_qmax_next)
                violation_magnitude_min_next = torch.relu(dynamic_qmin_next - next_q_values_raw)

                # STEP 1: Clip next-state Q-values to bounds at time t+1
                next_q_values = torch.clamp(next_q_values_raw,
                                           min=dynamic_qmin_next,
                                           max=dynamic_qmax_next)
            else:
                next_q_values = next_q_values_raw

            # Compute target Q-values (before final clipping)
            target_q_values_raw = rewards + (1 - dones) * self.gamma * next_q_values

            if self.use_qclip:
                # Track violations in TD targets BEFORE final clipping
                target_violate_max = (target_q_values_raw > dynamic_qmax_current).float()
                target_violate_min = (target_q_values_raw < dynamic_qmin_current).float()

                violation_magnitude_max_target = torch.relu(target_q_values_raw - dynamic_qmax_current)
                violation_magnitude_min_target = torch.relu(dynamic_qmin_current - target_q_values_raw)

                # STEP 2: Clip final target to bounds at time t (IMPORTANT!)
                # This ensures target respects the bounds for the current state
                target_q_values = torch.clamp(target_q_values_raw,
                                             min=dynamic_qmin_current,
                                             max=dynamic_qmax_current)

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
                    'qbound_max': dynamic_qmax_current.mean().item(),
                    'qbound_min': dynamic_qmin_current.mean().item(),
                }

                self.violation_stats_history.append(violation_stats)
            else:
                target_q_values = target_q_values_raw

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
