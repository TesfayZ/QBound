"""
Transformed DDPG Agent - Shifts negative Q-values to positive range

This agent transforms negative reward environments to positive Q-value spaces by:
1. Computing abs_Q_min = |Q_min|
2. Transforming bounds: Q_min_new = 0, Q_max_new = abs_Q_min
3. Adding abs_Q_min to target Q-values BEFORE computing MSE loss
4. Applying QBound clipping with transformed bounds

The transformation tests whether QBound's failure on negative rewards is due to
the negative value range itself, or due to other factors like violation rates.

Key Insight for Continuous Control (DDPG/TD3):
- Critic Q-values are transformed: Q ∈ [Q_min, 0] → Q ∈ [0, |Q_min|]
- Actor policy gradients computed on transformed Q-values
- Same transformation as DQN, adapted for actor-critic
- CRITICAL: Uses SOFT clipping to preserve gradients for actor learning

Example: Pendulum
- Original: Q ∈ [-1409, 0]
- Transformed: Q ∈ [0, 1409]
- Similar to CartPole's positive reward structure
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

from soft_qbound_penalty import SoftQBoundPenalty


class ReplayBuffer:
    """Experience replay buffer for DDPG"""

    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, time_step=None):
        """Add a transition to the buffer"""
        self.buffer.append((state, action, reward, next_state, done, time_step))

    def sample(self, batch_size):
        """Sample a batch of transitions"""
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

    def __init__(self, state_dim, action_dim, hidden_dims=[400, 300]):
        super(Critic, self).__init__()

        # Build state processing layers
        self.state_layer = nn.Linear(state_dim, hidden_dims[0])

        # Combine state and action
        self.combined_layer = nn.Linear(hidden_dims[0] + action_dim, hidden_dims[1])

        # Output Q-value
        self.output = nn.Linear(hidden_dims[1], 1)

    def forward(self, state, action):
        # Process state
        x = F.relu(self.state_layer(state))

        # Combine with action
        x = torch.cat([x, action], dim=1)
        x = F.relu(self.combined_layer(x))

        # Output Q-value (transformed space)
        q = self.output(x)
        return q.squeeze()


class TransformedDDPGAgent:
    """
    Transformed DDPG Agent - Shifts negative Q-values to positive range.

    Transformation:
    1. Original bounds: Q ∈ [Q_min, Q_max] where Q_min < 0, Q_max = 0
    2. Compute shift: abs_Q_min = |Q_min|
    3. New bounds: Q ∈ [0, abs_Q_min]
    4. Apply transformation: Q_transformed = Q_original + abs_Q_min

    Implementation for DDPG:
    - Critic learns transformed Q-values
    - Actor policy gradients computed on transformed Q
    - Target Q-values shifted before computing critic loss
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        learning_rate_actor=1e-4,
        learning_rate_critic=1e-3,
        gamma=0.99,
        tau=0.005,
        buffer_size=100000,
        batch_size=64,
        use_qbound=False,
        qbound_min_original=-1409.0,  # Original Q_min (e.g., -1409 for Pendulum)
        qbound_max_original=0.0,       # Original Q_max (typically 0 for negative rewards)
        use_soft_clip=True,            # CRITICAL: Use soft clipping for DDPG
        soft_clip_beta=0.1,            # Steepness parameter for soft clipping
        hidden_dims=[400, 300],
        device='cpu'
    ):
        """
        Args:
            qbound_min_original: Original Q_min (typically negative)
            qbound_max_original: Original Q_max (typically 0 for negative rewards)
            use_soft_clip: Whether to use soft clipping (True) or hard clipping (False)
            soft_clip_beta: Steepness of soft clipping (higher = closer to hard clip)
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.use_qbound = use_qbound
        self.use_soft_clip = use_soft_clip
        self.soft_clip_beta = soft_clip_beta
        self.device = torch.device(device)

        # Initialize soft clipping function
        self.penalty_fn = SoftQBoundPenalty()

        # Compute transformation shift
        self.abs_qmin = abs(qbound_min_original)

        # Transformed bounds (shifted to positive range)
        self.qbound_min_transformed = 0.0
        self.qbound_max_transformed = self.abs_qmin

        print(f"\n🔄 Q-Value Transformation (DDPG):")
        print(f"   Original bounds: Q ∈ [{qbound_min_original:.2f}, {qbound_max_original:.2f}]")
        print(f"   Shift amount: +{self.abs_qmin:.2f}")
        print(f"   Transformed bounds: Q ∈ [{self.qbound_min_transformed:.2f}, {self.qbound_max_transformed:.2f}]")
        print(f"   Clipping type: {'SOFT (preserves gradients)' if use_soft_clip else 'HARD'}")

        # Networks
        self.actor = Actor(state_dim, action_dim, max_action, hidden_dims).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, max_action, hidden_dims).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(state_dim, action_dim, hidden_dims).to(self.device)
        self.critic_target = Critic(state_dim, action_dim, hidden_dims).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate_critic)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)

        # Training stats
        self.critic_losses = []
        self.actor_losses = []
        self.violation_stats_history = []

    def select_action(self, state, noise_scale=0.1, eval_mode=False):
        """Select action using actor network with optional exploration noise"""
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)

        with torch.no_grad():
            action = self.actor(state).cpu().data.numpy().flatten()

        if not eval_mode:
            # Add exploration noise
            noise = np.random.normal(0, noise_scale, size=action.shape)
            action = action + noise
            action = np.clip(action, -self.max_action, self.max_action)

        return action

    def store_transition(self, state, action, reward, next_state, done, time_step=None):
        """Store transition in replay buffer"""
        self.replay_buffer.push(state, action, reward, next_state, done, time_step)

    def train_step(self):
        """
        Train the agent for one step with Q-value transformation.

        The transformation is applied to the target Q-values BEFORE computing critic loss:
        1. Compute raw next-state Q-values from target networks
        2. Clip next-state Q-values to transformed bounds (if QBound enabled)
        3. Compute raw TD target: target_raw = r + γ * Q_target(s', π_target(s'))
        4. TRANSFORM: target_transformed = target_raw + abs_Q_min
        5. Clip transformed target to transformed bounds
        6. Compute critic MSE loss and update
        7. Compute actor loss using transformed Q-values

        Returns:
            critic_loss, actor_loss, violation_stats
        """
        if len(self.replay_buffer) < self.batch_size:
            return None, None, None

        # Sample batch
        states, actions, rewards, next_states, dones, _ = self.replay_buffer.sample(self.batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        violation_stats = None

        # ========== Update Critic ==========
        with torch.no_grad():
            # Get next actions from target actor
            next_actions = self.actor_target(next_states)

            # Get next Q-values from target critic (these are transformed Q-values)
            next_q_transformed = self.critic_target(next_states, next_actions)

            if self.use_qbound:
                # Track violations BEFORE clipping
                qbound_max = torch.tensor(self.qbound_max_transformed, device=self.device)
                qbound_min = torch.tensor(self.qbound_min_transformed, device=self.device)

                next_q_violate_max = (next_q_transformed > qbound_max).float()
                next_q_violate_min = (next_q_transformed < qbound_min).float()
                violation_magnitude_max_next = torch.relu(next_q_transformed - qbound_max)
                violation_magnitude_min_next = torch.relu(qbound_min - next_q_transformed)

                # Clip next Q-values in transformed space
                # Use SOFT clipping for DDPG to preserve gradients!
                if self.use_soft_clip:
                    next_q_clipped = self.penalty_fn.softplus_clip(
                        next_q_transformed, qbound_min, qbound_max, beta=self.soft_clip_beta
                    )
                else:
                    next_q_clipped = torch.clamp(next_q_transformed, qbound_min, qbound_max)
            else:
                next_q_clipped = next_q_transformed

            # Un-transform to original space for Bellman equation
            next_q_original = next_q_clipped - self.abs_qmin

            # Compute TD target in original space
            target_q_original = rewards + (1 - dones) * self.gamma * next_q_original

            # Transform back to positive space
            target_q_transformed = target_q_original + self.abs_qmin

            if self.use_qbound:
                # Track violations in transformed targets BEFORE final clipping
                target_violate_max = (target_q_transformed > qbound_max).float()
                target_violate_min = (target_q_transformed < qbound_min).float()
                violation_magnitude_max_target = torch.relu(target_q_transformed - qbound_max)
                violation_magnitude_min_target = torch.relu(qbound_min - target_q_transformed)

                # Final clip in transformed space
                # Use SOFT clipping for DDPG to preserve gradients!
                if self.use_soft_clip:
                    target_q = self.penalty_fn.softplus_clip(
                        target_q_transformed, qbound_min, qbound_max, beta=self.soft_clip_beta
                    )
                else:
                    target_q = torch.clamp(target_q_transformed, qbound_min, qbound_max)

                # Violation statistics
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
                    'qbound_max': qbound_max.item(),
                    'qbound_min': qbound_min.item(),
                }
                self.violation_stats_history.append(violation_stats)
            else:
                target_q = target_q_transformed

        # Current Q-value (in transformed space)
        current_q = self.critic(states, actions)

        # Critic loss
        critic_loss = F.mse_loss(current_q, target_q)

        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ========== Update Actor ==========
        # Actor loss: maximize Q-value (transformed space)
        actor_actions = self.actor(states)
        q_for_actor = self.critic(states, actor_actions)

        # Apply soft clipping to preserve gradients during policy improvement
        if self.use_qbound and self.use_soft_clip:
            qbound_max = torch.tensor(self.qbound_max_transformed, device=self.device)
            qbound_min = torch.tensor(self.qbound_min_transformed, device=self.device)
            q_for_actor = self.penalty_fn.softplus_clip(
                q_for_actor, qbound_min, qbound_max, beta=self.soft_clip_beta
            )

        actor_loss = -q_for_actor.mean()

        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ========== Update Target Networks ==========
        self._update_target_networks()

        # Store losses
        self.critic_losses.append(critic_loss.item())
        self.actor_losses.append(actor_loss.item())

        return critic_loss.item(), actor_loss.item(), violation_stats

    def _update_target_networks(self):
        """Soft update of target networks"""
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename):
        """Save model weights"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }, filename)

    def load(self, filename):
        """Load model weights"""
        checkpoint = torch.load(filename)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
