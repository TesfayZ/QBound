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
- Hard QBound: use_soft_qbound=False (default, hard clipping)
- Soft QBound: use_soft_qbound=True (smooth penalties, preserves gradients)
"""

import sys
sys.path.insert(0, '/root/projects/QBound/src')

import torch
import torch.nn.functional as F
import numpy as np
from ppo_agent import PPOAgent

try:
    from soft_qbound_penalty import SoftQBoundPenalty
    SOFT_QBOUND_AVAILABLE = True
except ImportError:
    SOFT_QBOUND_AVAILABLE = False


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
        use_soft_qbound=False,
        qbound_penalty_weight=0.1,
        qbound_penalty_type='quadratic',
        soft_clip_beta=0.1,
        **kwargs  # Pass through to PPOAgent
    ):
        super().__init__(state_dim, action_dim, continuous_action, **kwargs)

        # QBound parameters
        self.V_min = V_min
        self.V_max = V_max
        self.use_step_aware_bounds = use_step_aware_bounds
        self.max_episode_steps = max_episode_steps
        self.step_reward = step_reward
        self.use_soft_qbound = use_soft_qbound and SOFT_QBOUND_AVAILABLE
        self.qbound_penalty_weight = qbound_penalty_weight
        self.qbound_penalty_type = qbound_penalty_type
        self.soft_clip_beta = soft_clip_beta

        # Initialize soft QBound penalty function if needed
        if self.use_soft_qbound:
            self.penalty_fn = SoftQBoundPenalty()
            self.recent_penalties = []
        else:
            self.penalty_fn = None
            self.recent_penalties = None

        # Tracking QBound statistics
        self.bound_violations = 0
        self.total_predictions = 0

    def compute_bounds(self, current_step=None):
        """
        Compute V(s) bounds.

        Static bounds (sparse rewards):
            V_min, V_max = known reward bounds with discount

        Dynamic bounds (dense rewards):
            V_max(t) = (H - t) * step_reward
            Adapts to remaining episode potential
        """
        if self.use_step_aware_bounds and current_step is not None:
            # Dynamic bound for dense reward survival tasks
            remaining_steps = self.max_episode_steps - current_step
            V_max_dynamic = remaining_steps * self.step_reward
            return self.V_min, V_max_dynamic
        else:
            # Static bounds for sparse reward tasks
            return self.V_min, self.V_max

    def compute_gae_with_bounds(self, rewards, values, next_values, dones, steps=None):
        """
        GAE with bounded value estimates.

        Critical: Apply bounds to next_values during bootstrapping,
        preventing value overestimation from propagating through advantages.

        This is the key innovation: bounding V(s') in the TD target
        r + γV(s') stabilizes the advantage estimation without affecting
        the policy gradient computation.
        """
        advantages = []
        gae = 0

        # Track bound violations
        violations = 0
        total = 0

        for t in reversed(range(len(rewards))):
            # Get bounds (static or dynamic based on step)
            if steps is not None and self.use_step_aware_bounds:
                V_min, V_max = self.compute_bounds(steps[t])
            else:
                V_min, V_max = self.V_min, self.V_max

            # Apply bounds to next value during bootstrapping
            if self.use_soft_qbound:
                # SOFT CLIPPING: Smooth bounds (preserves gradient-like behavior)
                next_value_tensor = torch.FloatTensor([next_values[t]])
                next_value_bounded = self.penalty_fn.softplus_clip(
                    next_value_tensor,
                    torch.tensor(V_min, dtype=torch.float32),
                    torch.tensor(V_max, dtype=torch.float32),
                    beta=self.soft_clip_beta
                ).item()
            else:
                # HARD CLIPPING: Standard clipping
                next_value_bounded = np.clip(next_values[t], V_min, V_max)

            # Track if bound was violated
            if next_values[t] < V_min or next_values[t] > V_max:
                violations += 1
            total += 1

            if dones[t]:
                # Terminal state: no future value
                delta = rewards[t] - values[t]
                gae = delta
            else:
                # Use bounded next value in TD error
                delta = rewards[t] + self.gamma * next_value_bounded - values[t]
                gae = delta + self.gamma * self.gae_lambda * gae

            advantages.insert(0, gae)

        # Compute returns with bounded advantages
        returns = [adv + val for adv, val in zip(advantages, values)]

        # Apply bounds to returns as well (prevent target from being unbounded)
        if self.use_soft_qbound:
            # SOFT CLIPPING for returns
            if steps is not None and self.use_step_aware_bounds:
                returns_clipped = []
                for ret, step in zip(returns, steps):
                    V_min_s, V_max_s = self.compute_bounds(step)
                    ret_tensor = torch.FloatTensor([ret])
                    ret_clipped = self.penalty_fn.softplus_clip(
                        ret_tensor,
                        torch.tensor(V_min_s, dtype=torch.float32),
                        torch.tensor(V_max_s, dtype=torch.float32),
                        beta=self.soft_clip_beta
                    ).item()
                    returns_clipped.append(ret_clipped)
                returns = returns_clipped
            else:
                returns_tensor = torch.FloatTensor(returns)
                returns_clipped = self.penalty_fn.softplus_clip(
                    returns_tensor,
                    torch.tensor(self.V_min, dtype=torch.float32),
                    torch.tensor(self.V_max, dtype=torch.float32),
                    beta=self.soft_clip_beta
                )
                returns = returns_clipped.tolist()
        else:
            # HARD CLIPPING for returns
            if steps is not None and self.use_step_aware_bounds:
                returns = [np.clip(ret, *self.compute_bounds(step))
                          for ret, step in zip(returns, steps)]
            else:
                returns = [np.clip(ret, self.V_min, self.V_max) for ret in returns]

        # Update statistics
        self.bound_violations += violations
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

                # ===== Critic Update WITH BOUNDS =====
                value_pred = self.critic(mb_states)

                # Track predictions before any modifications
                value_preds_before_clip.extend(value_pred.detach().cpu().numpy())

                # Compute critic loss and penalties
                if self.use_soft_qbound:
                    # SOFT QBOUND: Add smooth penalty instead of hard clipping
                    critic_loss = F.mse_loss(value_pred, mb_returns)

                    # Compute penalty
                    if steps is not None and self.use_step_aware_bounds:
                        mb_steps = [steps[i] for i in mb_indices]
                        penalties = []
                        for v, step in zip(value_pred, mb_steps):
                            V_min_s, V_max_s = self.compute_bounds(step)
                            if self.qbound_penalty_type == 'quadratic':
                                p = self.penalty_fn.quadratic_penalty(
                                    v.unsqueeze(0),
                                    torch.tensor(V_min_s, device=self.device),
                                    torch.tensor(V_max_s, device=self.device)
                                )
                            elif self.qbound_penalty_type == 'huber':
                                p = self.penalty_fn.huber_penalty(
                                    v.unsqueeze(0),
                                    torch.tensor(V_min_s, device=self.device),
                                    torch.tensor(V_max_s, device=self.device),
                                    delta=10.0
                                )
                            elif self.qbound_penalty_type == 'exponential':
                                p = self.penalty_fn.exponential_penalty(
                                    v.unsqueeze(0),
                                    torch.tensor(V_min_s, device=self.device),
                                    torch.tensor(V_max_s, device=self.device),
                                    alpha=0.1
                                )
                            else:
                                p = torch.tensor(0.0, device=self.device)
                            penalties.append(p)
                        value_penalty = torch.stack(penalties).mean()
                    else:
                        # Static bounds
                        if self.qbound_penalty_type == 'quadratic':
                            value_penalty = self.penalty_fn.quadratic_penalty(
                                value_pred,
                                torch.tensor(self.V_min, device=self.device),
                                torch.tensor(self.V_max, device=self.device)
                            )
                        elif self.qbound_penalty_type == 'huber':
                            value_penalty = self.penalty_fn.huber_penalty(
                                value_pred,
                                torch.tensor(self.V_min, device=self.device),
                                torch.tensor(self.V_max, device=self.device),
                                delta=10.0
                            )
                        elif self.qbound_penalty_type == 'exponential':
                            value_penalty = self.penalty_fn.exponential_penalty(
                                value_pred,
                                torch.tensor(self.V_min, device=self.device),
                                torch.tensor(self.V_max, device=self.device),
                                alpha=0.1
                            )
                        else:
                            value_penalty = torch.tensor(0.0, device=self.device)

                    critic_loss = critic_loss + self.qbound_penalty_weight * value_penalty

                    # Track penalty
                    if self.recent_penalties is not None:
                        self.recent_penalties.append(value_penalty.item())
                        if len(self.recent_penalties) > 100:
                            self.recent_penalties.pop(0)

                    # For tracking, use original predictions
                    value_preds_after_clip.extend(value_pred.detach().cpu().numpy())
                else:
                    # HARD QBOUND: Apply bounds to predictions (standard clipping)
                    if steps is not None and self.use_step_aware_bounds:
                        # Dynamic bounds per step
                        mb_steps = [steps[i] for i in mb_indices]
                        value_pred_bounded = torch.stack([
                            torch.clamp(v, *self.compute_bounds(step))
                            for v, step in zip(value_pred, mb_steps)
                        ])
                    else:
                        # Static bounds
                        value_pred_bounded = torch.clamp(value_pred, self.V_min, self.V_max)

                    # Track predictions after clipping
                    value_preds_after_clip.extend(value_pred_bounded.detach().cpu().numpy())

                    # Compute loss using bounded predictions
                    critic_loss = F.mse_loss(value_pred_bounded, mb_returns)

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
            # QBound-specific metrics
            'qbound_violations_bootstrap': self.bound_violations / max(self.total_predictions, 1),
            'qbound_clipped_fraction': clipped_fraction,
            'qbound_bounds': (self.V_min, self.V_max),
        }

        # Reset statistics for next update
        self.bound_violations = 0
        self.total_predictions = 0

        return self.training_info
