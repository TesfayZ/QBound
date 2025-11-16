"""
PPO + QBound Agent Implementation

PPO with QBound applied to the critic V(s).

Key modifications:
1. Bound V(s) predictions: V_min ≤ V(s) ≤ V_max
2. Bound bootstrapped targets: V_min ≤ r + γV(s') ≤ V_max
3. Use bounded values for advantage computation

Hypothesis: Bounding V(s) stabilizes advantage estimation,
improving sample efficiency without disrupting policy gradients.

Critical difference from DDPG/TD3:
- DDPG/TD3 clip Q(s,a), disrupting ∇_a Q needed for deterministic policy
- PPO clips V(s), which doesn't affect policy gradient ∇_θ log π(a|s)
- This should work even for continuous action spaces!

QBound Support:
- Hard clipping on V(s) targets during GAE computation
- Bounds applied to bootstrapped values and returns
"""

import sys
sys.path.insert(0, '/root/projects/QBound/src')

import torch
import torch.nn.functional as F
import numpy as np
from ppo_agent import PPOAgent


class PPOQBoundAgent(PPOAgent):
    """
    PPO with QBound applied to the critic V(s).

    Inherits from PPOAgent and overrides:
    - compute_gae(): Apply bounds during bootstrapping
    - update(): Apply bounds to value predictions
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        continuous_action=False,
        V_min=0.0,
        V_max=100.0,
        use_step_aware_bounds=False,
        max_episode_steps=None,
        step_reward=None,
        **kwargs  # Pass through to PPOAgent
    ):
        super().__init__(state_dim, action_dim, continuous_action, **kwargs)

        # QBound parameters
        self.V_min = V_min
        self.V_max = V_max
        self.use_step_aware_bounds = use_step_aware_bounds
        self.max_episode_steps = max_episode_steps
        self.step_reward = step_reward

        # Tracking QBound statistics
        self.bound_violations = 0
        self.total_predictions = 0

    def compute_bounds(self, current_step=None):
        """
        Compute V(s) bounds.

        Static bounds (sparse rewards):
            V_min, V_max = known reward bounds with discount

        Dynamic bounds (dense rewards):
            V_max(t) = step_reward * (1 - γ^(H-t)) / (1 - γ)
            Uses geometric series for discounted sum
        """
        if self.use_step_aware_bounds and current_step is not None:
            # Dynamic bound for dense reward survival tasks
            remaining_steps = self.max_episode_steps - current_step

            if remaining_steps > 0:
                # Geometric series for discounted sum of future rewards
                if abs(self.gamma - 1.0) < 1e-6:
                    # γ ≈ 1: undiscounted case
                    bound_dynamic = self.step_reward * remaining_steps
                else:
                    # Standard geometric series: Σ(γ^k * r) = r * (1 - γ^H) / (1 - γ)
                    bound_dynamic = self.step_reward * (1 - self.gamma ** remaining_steps) / (1 - self.gamma)

                # For positive rewards: V ∈ [0, V_max(t)] where V_max(t) decreases
                # For negative rewards: V ∈ [V_min(t), 0] where V_min(t) increases (becomes less negative)
                if self.step_reward >= 0:
                    return 0.0, bound_dynamic  # Positive: V_max decreases, V_min stays at 0
                else:
                    return bound_dynamic, 0.0  # Negative: V_min becomes less negative, V_max stays at 0
            else:
                # No remaining steps
                return (0.0, 0.0) if self.step_reward >= 0 else (0.0, 0.0)
        else:
            # Static bounds for sparse reward tasks
            return self.V_min, self.V_max

    def compute_gae_with_bounds(self, rewards, values, next_values, dones, steps=None):
        """
        GAE with QBound applied to critic targets (Option 1).

        CORRECTED IMPLEMENTATION:
        - DON'T clip next_values during GAE (compute unbiased advantages)
        - ONLY clip final returns (targets for critic training)

        This matches DQN/DDQN philosophy: clip what the value network learns from,
        not what the policy sees. Prevents overestimated values from being learned
        without distorting advantages used for policy updates.
        """
        advantages = []
        gae = 0

        # Compute GAE WITHOUT clipping (unbiased advantages for policy)
        for t in reversed(range(len(rewards))):
            if dones[t]:
                # Terminal state: no future value
                delta = rewards[t] - values[t]
                gae = delta
            else:
                # Use RAW next value (no clipping!) for unbiased TD error
                delta = rewards[t] + self.gamma * next_values[t] - values[t]
                gae = delta + self.gamma * self.gae_lambda * gae

            advantages.insert(0, gae)

        # Compute returns (for critic targets)
        returns = [adv + val for adv, val in zip(advantages, values)]

        # CLIP RETURNS ONLY (targets for critic, not used for policy)
        return_violations = 0
        total = len(returns)

        if steps is not None and self.use_step_aware_bounds:
            # Dynamic bounds: clip each return according to its step
            returns_clipped = []
            for ret, step in zip(returns, steps):
                V_min_s, V_max_s = self.compute_bounds(step)
                ret_clipped = np.clip(ret, V_min_s, V_max_s)
                if ret < V_min_s or ret > V_max_s:
                    return_violations += 1
                returns_clipped.append(ret_clipped)
            returns = returns_clipped
        else:
            # Static bounds: clip all returns to same bounds
            returns_original = returns.copy()
            returns = [np.clip(ret, self.V_min, self.V_max) for ret in returns]
            return_violations = sum(1 for orig, clipped in zip(returns_original, returns)
                                   if orig != clipped)

        # Track violations (for analysis)
        self.bound_violations += return_violations
        self.total_predictions += total

        return advantages, returns

    def update(self, trajectories):
        """
        PPO update with bounded value targets.

        Modified from base PPO to:
        1. Use compute_gae_with_bounds() instead of compute_gae()
        2. Track bound violation statistics
        3. Optionally apply bounds to value predictions during training
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

        # Extract steps if using step-aware bounds
        steps = None
        if self.use_step_aware_bounds and len(trajectories[0]) > 6:
            steps = [t[6] for t in trajectories]

        # Get value predictions
        with torch.no_grad():
            values = self.critic(states).cpu().numpy()
            next_values = self.critic(next_states).cpu().numpy()

        # Compute advantages and returns using GAE WITH BOUNDS
        advantages, returns = self.compute_gae_with_bounds(
            rewards, values, next_values, dones, steps
        )

        # Convert to tensors
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)

        # Normalize advantages (standard PPO trick)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Tracking metrics
        actor_losses = []
        critic_losses = []
        entropies = []
        value_preds_before_clip = []
        value_preds_after_clip = []

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

                # ===== Actor Update (UNCHANGED from base PPO) =====
                new_log_probs, entropy = self.actor.evaluate(mb_states, mb_actions)

                # Importance sampling ratio
                ratio = torch.exp(new_log_probs - mb_old_log_probs)

                # Clipped surrogate objective
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * mb_advantages
                actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy.mean()

                self.optimizer_actor.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.optimizer_actor.step()

                # ===== Critic Update =====
                value_pred = self.critic(mb_states)

                # Track predictions for statistics
                value_preds_before_clip.extend(value_pred.detach().cpu().numpy())
                value_preds_after_clip.extend(value_pred.detach().cpu().numpy())

                # Critic loss - NO PENALTY! Bounds are enforced in returns (targets)
                # Returns were already clipped in compute_gae_with_bounds()
                critic_loss = F.mse_loss(value_pred, mb_returns)

                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.optimizer_critic.step()

                # Track metrics
                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())
                entropies.append(entropy.mean().item())

        # Compute fraction of predictions that violated bounds
        value_preds_before = np.array(value_preds_before_clip)
        value_preds_after = np.array(value_preds_after_clip)
        clipped_fraction = np.mean(value_preds_before != value_preds_after)

        # Store training info with QBound statistics
        self.training_info = {
            'actor_loss': np.mean(actor_losses),
            'critic_loss': np.mean(critic_losses),
            'entropy': np.mean(entropies),
            'advantage_mean': advantages.mean().item(),
            'advantage_std': advantages.std().item(),
            'value_mean': torch.FloatTensor(values).mean().item(),
            'value_std': torch.FloatTensor(values).std().item(),
            # QBound-specific metrics (enhanced)
            'v_violation_rate': self.bound_violations / max(self.total_predictions, 1),  # Renamed for clarity
            'v_clipped_fraction': clipped_fraction,  # Renamed for clarity
            'penalty_activation_rate': self.bound_violations / max(self.total_predictions, 1),  # Alias for consistency
            'qbound_max': self.V_max,
            'qbound_min': self.V_min,
        }

        # Reset statistics for next update
        self.bound_violations = 0
        self.total_predictions = 0

        return self.training_info
