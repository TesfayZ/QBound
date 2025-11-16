"""
Dueling DQN Agent with optional QBound
Implements Dueling Deep Q-Network with value and advantage stream decomposition.
Supports QBound for improved learning in sparse and dense reward environments.

Architecture:
- Shared feature layers
- Split into value stream V(s) and advantage stream A(s,a)
- Combine: Q(s,a) = V(s) + (A(s,a) - mean_a A(s,a))

QBound Integration:
- Clips next-state Q-values (after combining V and A streams)
- Same bootstrapping-based mechanism as standard DQN
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from typing import List, Tuple


class DuelingQNetwork(nn.Module):
    """Dueling network architecture with separate value and advantage streams."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(DuelingQNetwork, self).__init__()

        # Shared feature layers
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Value stream V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Advantage stream A(s,a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        """
        Compute Q-values using dueling architecture:
        Q(s,a) = V(s) + (A(s,a) - mean_a A(s,a))

        The mean-centering ensures identifiability and improves optimization.
        """
        features = self.feature(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)

        # Combine with mean-centering for identifiability
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values


class ReplayBuffer:
    """Experience replay buffer for DQN."""

    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, current_step=None):
        """Store transition with optional current_step for step-aware Q-bounds."""
        self.buffer.append((state, action, reward, next_state, done, current_step))

    def sample(self, batch_size: int) -> List[Tuple]:
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class DuelingDQNAgent:
    """
    Dueling DQN Agent with optional QBound.

    Dueling Architecture Benefits:
    - Separates state value V(s) from action advantages A(s,a)
    - Improves learning when many actions have similar Q-values
    - More stable value estimation, especially beneficial with QBound

    QBound Integration:
    - Clips next-state Q-values (after dueling combination) to [Q_min, Q_max]
    - Same bootstrapping-based enforcement as standard DQN
    - No auxiliary loss needed - bounds propagate naturally
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
        reward_is_negative: bool = False,
        use_double_dqn: bool = False
    ):
        """
        Args:
            use_qclip: Enable QBound
            use_double_dqn: Enable Double DQN (action selection from online, evaluation from target)
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
        self.use_double_dqn = use_double_dqn

        # Step-aware Q-bounds
        self.use_step_aware_qbound = use_step_aware_qbound
        self.max_episode_steps = max_episode_steps
        self.step_reward = step_reward
        self.reward_is_negative = reward_is_negative

        # Dueling Q-networks
        self.q_network = DuelingQNetwork(state_dim, action_dim).to(self.device)
        self.target_network = DuelingQNetwork(state_dim, action_dim).to(self.device)
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
        """Store transition in replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done, current_step)

    def train_step(self):
        """
        Perform one training step with optional QBound and Double DQN.

        Dueling DQN: Q(s,a) = V(s) + (A(s,a) - mean_a A(s,a))
        Double DQN: Uses online network for action selection, target for evaluation
        QBound: Clips next-state Q-values during bootstrapping

        Returns:
            loss (float): TD loss value
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

        # Compute current Q-values (from online network)
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Initialize violation tracking
        violation_stats = None

        # Compute next-state Q-values
        with torch.no_grad():
            if self.use_double_dqn:
                # Double DQN: action selection from online network
                next_actions = self.q_network(next_states).argmax(1)
                # Q-value evaluation from target network
                next_q_values_raw = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            else:
                # Standard DQN: max Q-value from target network
                next_q_values_raw = self.target_network(next_states).max(1)[0]

            # QBound: Clip next-state Q-values to environment-aware bounds
            if self.use_qclip:
                if self.use_step_aware_qbound and current_steps[0] is not None:
                    # Dynamic step-aware bounds
                    current_steps_tensor = torch.FloatTensor([s if s is not None else 0 for s in current_steps]).to(self.device)
                    remaining_steps_next = torch.clamp(self.max_episode_steps - current_steps_tensor - 1, min=0)
                    remaining_steps_current = torch.clamp(self.max_episode_steps - current_steps_tensor, min=0)

                    # Geometric sum formula: (1 - γ^H) / (1 - γ)
                    geometric_sum_next = (1 - torch.pow(self.gamma, remaining_steps_next)) / (1 - self.gamma)
                    geometric_sum_current = (1 - torch.pow(self.gamma, remaining_steps_current)) / (1 - self.gamma)

                    if self.reward_is_negative:
                        # Negative rewards: Q ∈ [Q_min(t), 0]
                        # step_reward is already negative (e.g., -16.27), geometric_sum is positive
                        # Q_min(t) = step_reward * geometric_sum = (-16.27) * 86.60 = -1409.33 ✓
                        dynamic_qmin_next = geometric_sum_next * self.step_reward
                        dynamic_qmax_next = torch.zeros_like(dynamic_qmin_next)
                        dynamic_qmin_current = geometric_sum_current * self.step_reward
                        dynamic_qmax_current = torch.zeros_like(dynamic_qmin_current)
                    else:
                        # Positive rewards: Q ∈ [0, Q_max(t)]
                        dynamic_qmin_next = torch.zeros_like(geometric_sum_next)
                        dynamic_qmax_next = geometric_sum_next * self.step_reward
                        dynamic_qmin_current = torch.zeros_like(geometric_sum_current)
                        dynamic_qmax_current = geometric_sum_current * self.step_reward
                else:
                    # Static bounds
                    dynamic_qmin_next = torch.full_like(next_q_values_raw, self.qclip_min)
                    dynamic_qmax_next = torch.full_like(next_q_values_raw, self.qclip_max)
                    dynamic_qmin_current = torch.full_like(next_q_values_raw, self.qclip_min)
                    dynamic_qmax_current = torch.full_like(next_q_values_raw, self.qclip_max)

                # Track violations BEFORE clipping (for analysis)
                next_q_violate_max = (next_q_values_raw > dynamic_qmax_next).float()
                next_q_violate_min = (next_q_values_raw < dynamic_qmin_next).float()

                violation_magnitude_max_next = torch.relu(next_q_values_raw - dynamic_qmax_next)
                violation_magnitude_min_next = torch.relu(dynamic_qmin_next - next_q_values_raw)

                # Clip next-state Q-values
                next_q_values = torch.clamp(next_q_values_raw, dynamic_qmin_next, dynamic_qmax_next)
            else:
                next_q_values = next_q_values_raw

            # Compute Bellman targets (before final clipping)
            targets_raw = rewards + self.gamma * next_q_values * (1 - dones)

            # QBound: Clip final targets for safety
            if self.use_qclip:
                # Track violations in TD targets BEFORE final clipping
                target_violate_max = (targets_raw > dynamic_qmax_current).float()
                target_violate_min = (targets_raw < dynamic_qmin_current).float()

                violation_magnitude_max_target = torch.relu(targets_raw - dynamic_qmax_current)
                violation_magnitude_min_target = torch.relu(dynamic_qmin_current - targets_raw)

                # Clip final targets
                targets = torch.clamp(targets_raw, dynamic_qmin_current, dynamic_qmax_current)

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
                targets = targets_raw

        # Standard TD loss
        loss = nn.MSELoss()(current_q_values, targets)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network periodically
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        self.losses.append(loss.item())
        return loss.item(), violation_stats

    def save(self, path: str):
        """Save agent checkpoint."""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'steps': self.steps,
            'epsilon': self.epsilon
        }, path)

    def load(self, path: str):
        """Load agent checkpoint."""
        checkpoint = torch.load(path)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.steps = checkpoint['steps']
        self.epsilon = checkpoint['epsilon']
