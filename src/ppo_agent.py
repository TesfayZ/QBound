"""
PPO (Proximal Policy Optimization) Agent Implementation

Standard PPO with Generalized Advantage Estimation (GAE).
Uses:
- Actor network: π(a|s) for policy
- Critic network: V(s) for value function
- Clipped surrogate objective
- GAE for advantage estimation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal
import numpy as np


class ActorNetwork(nn.Module):
    """
    Policy network π(a|s).

    For discrete actions: outputs action probabilities
    For continuous actions: outputs mean and log_std
    """

    def __init__(self, state_dim, action_dim, continuous_action=False, hidden_sizes=[64, 64]):
        super(ActorNetwork, self).__init__()
        self.continuous_action = continuous_action
        self.action_dim = action_dim

        # Shared layers
        layers = []
        prev_size = state_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size

        self.shared = nn.Sequential(*layers)

        if continuous_action:
            # For continuous actions: output mean
            self.mean_head = nn.Linear(prev_size, action_dim)
            # Learnable log std (independent of state)
            self.log_std = nn.Parameter(torch.zeros(action_dim))
        else:
            # For discrete actions: output action logits
            self.action_head = nn.Linear(prev_size, action_dim)

    def forward(self, state):
        """Forward pass to get action distribution."""
        x = self.shared(state)

        if self.continuous_action:
            mean = self.mean_head(x)
            std = torch.exp(self.log_std)
            return mean, std
        else:
            logits = self.action_head(x)
            return F.softmax(logits, dim=-1)

    def get_action(self, state, deterministic=False):
        """Sample action from policy."""
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).unsqueeze(0)

        if self.continuous_action:
            mean, std = self.forward(state)
            if deterministic:
                action = mean
            else:
                dist = Normal(mean, std)
                action = dist.sample()
            log_prob = Normal(mean, std).log_prob(action).sum(dim=-1)
            return action.detach().numpy()[0], log_prob.detach()
        else:
            probs = self.forward(state)
            if deterministic:
                action = torch.argmax(probs, dim=-1)
            else:
                dist = Categorical(probs)
                action = dist.sample()
            log_prob = torch.log(probs.squeeze()[action])
            return action.item(), log_prob.detach()

    def evaluate(self, states, actions):
        """Evaluate actions under current policy (for PPO update)."""
        if self.continuous_action:
            mean, std = self.forward(states)
            dist = Normal(mean, std)
            log_probs = dist.log_prob(actions).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
            return log_probs, entropy
        else:
            probs = self.forward(states)
            dist = Categorical(probs)
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy()
            return log_probs, entropy


class CriticNetwork(nn.Module):
    """
    Value network V(s).

    Estimates the expected return from state s.
    """

    def __init__(self, state_dim, hidden_sizes=[64, 64],
                 use_negative_activation=False):
        super(CriticNetwork, self).__init__()
        self.use_negative_activation = use_negative_activation

        layers = []
        prev_size = state_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, state):
        """Forward pass to get V(s)."""
        logits = self.network(state).squeeze(-1)

        if self.use_negative_activation:
            # Architectural bound for negative rewards: V ≤ 0
            value = -torch.nn.functional.softplus(logits)
        else:
            value = logits

        return value


class PPOAgent:
    """
    Standard PPO implementation with GAE.

    Components:
    - Actor: π(a|s) - policy network
    - Critic: V(s) - value network

    Loss functions:
    - Actor loss: -E[min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)] - β*H(π)
    - Critic loss: MSE(V(s), G_t) where G_t = r + γV(s')

    Training:
    - Collect trajectories using current policy
    - Compute advantages using GAE
    - Update policy and value function using mini-batch SGD
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        continuous_action=False,
        hidden_sizes=[64, 64],
        lr_actor=3e-4,
        lr_critic=1e-3,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        entropy_coef=0.01,
        ppo_epochs=10,
        minibatch_size=64,
        max_grad_norm=0.5,
        use_architectural_qbound=False,
        device='cpu'
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.continuous_action = continuous_action
        self.device = device

        # Networks
        self.actor = ActorNetwork(state_dim, action_dim, continuous_action, hidden_sizes).to(device)
        self.critic = CriticNetwork(state_dim, hidden_sizes,
                                    use_negative_activation=use_architectural_qbound).to(device)

        # Optimizers
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # PPO hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.ppo_epochs = ppo_epochs
        self.minibatch_size = minibatch_size
        self.max_grad_norm = max_grad_norm

        # For tracking
        self.training_info = {}

    def get_action(self, state, deterministic=False):
        """Get action from current policy."""
        self.actor.eval()
        with torch.no_grad():
            action, log_prob = self.actor.get_action(state, deterministic)
        self.actor.train()
        return action, log_prob

    def compute_gae(self, rewards, values, next_values, dones):
        """
        Compute Generalized Advantage Estimation (GAE).

        GAE(λ): A^GAE(s_t) = Σ(γλ)^l δ_{t+l}
        where δ_t = r_t + γV(s_{t+1}) - V(s_t)

        Returns:
            advantages: List of advantage estimates
            returns: List of return estimates (for critic training)
        """
        advantages = []
        gae = 0

        for t in reversed(range(len(rewards))):
            if dones[t]:
                # Terminal state: no future value
                delta = rewards[t] - values[t]
                gae = delta
            else:
                # Non-terminal: bootstrap from next value
                delta = rewards[t] + self.gamma * next_values[t] - values[t]
                gae = delta + self.gamma * self.gae_lambda * gae

            advantages.insert(0, gae)

        # Returns = advantages + values (for critic targets)
        returns = [adv + val for adv, val in zip(advantages, values)]

        return advantages, returns

    def update(self, trajectories):
        """
        Update policy and value function using collected trajectories.

        Args:
            trajectories: List of (state, action, reward, next_state, done, log_prob) tuples

        Returns:
            Dictionary with training statistics
        """
        # Unpack trajectories
        states = torch.FloatTensor([t[0] for t in trajectories]).to(self.device)
        actions = torch.FloatTensor([t[1] for t in trajectories]).to(self.device)
        if not self.continuous_action:
            actions = actions.long()
        rewards = [t[2] for t in trajectories]
        next_states = torch.FloatTensor([t[3] for t in trajectories]).to(self.device)
        dones = [t[4] for t in trajectories]
        old_log_probs = torch.FloatTensor([t[5] for t in trajectories]).to(self.device)

        # Get value predictions
        with torch.no_grad():
            values = self.critic(states).cpu().numpy()
            next_values = self.critic(next_states).cpu().numpy()

        # Compute advantages and returns using GAE
        advantages, returns = self.compute_gae(rewards, values, next_values, dones)

        # Convert to tensors
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)

        # Normalize advantages (standard PPO trick)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Tracking metrics
        actor_losses = []
        critic_losses = []
        entropies = []

        # Multiple epochs of SGD
        for epoch in range(self.ppo_epochs):
            # Create mini-batches
            indices = np.arange(len(trajectories))
            np.random.shuffle(indices)

            for start in range(0, len(trajectories), self.minibatch_size):
                end = start + self.minibatch_size
                mb_indices = indices[start:end]

                mb_states = states[mb_indices]
                mb_actions = actions[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_returns = returns[mb_indices]

                # ===== Actor Update =====
                new_log_probs, entropy = self.actor.evaluate(mb_states, mb_actions)

                # Importance sampling ratio
                ratio = torch.exp(new_log_probs - mb_old_log_probs)

                # Clipped surrogate objective
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * mb_advantages
                actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy.mean()

                self.optimizer_actor.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.optimizer_actor.step()

                # ===== Critic Update =====
                value_pred = self.critic(mb_states)
                critic_loss = F.mse_loss(value_pred, mb_returns)

                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.optimizer_critic.step()

                # Track metrics
                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())
                entropies.append(entropy.mean().item())

        # Store training info
        self.training_info = {
            'actor_loss': np.mean(actor_losses),
            'critic_loss': np.mean(critic_losses),
            'entropy': np.mean(entropies),
            'advantage_mean': advantages.mean().item(),
            'advantage_std': advantages.std().item(),
            'value_mean': torch.FloatTensor(values).mean().item(),
            'value_std': torch.FloatTensor(values).std().item(),
        }

        return self.training_info

    def save(self, path):
        """Save model."""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'optimizer_actor_state_dict': self.optimizer_actor.state_dict(),
            'optimizer_critic_state_dict': self.optimizer_critic.state_dict(),
        }, path)

    def load(self, path):
        """Load model."""
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.optimizer_actor.load_state_dict(checkpoint['optimizer_actor_state_dict'])
        self.optimizer_critic.load_state_dict(checkpoint['optimizer_critic_state_dict'])
