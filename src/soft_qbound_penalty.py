"""
Soft QBound: Smooth Penalty Functions for Continuous RL

Hard clipping (torch.clamp) sets gradients to ZERO when Q-values violate bounds.
This breaks gradient-based methods like DDPG and PPO.

Soft QBound uses smooth, differentiable penalty functions that:
1. Allow gradients to flow even when Q-values are outside bounds
2. Apply increasing penalties as Q-values move further from valid range
3. Behave like "double-sided soft ReLU" with smooth transitions

Three penalty variants:
1. Quadratic Penalty: Smooth, stronger penalty for large violations
2. Huber Penalty: L1-like penalty with smooth transition (robust to outliers)
3. Exponential Penalty: Very smooth, exponentially increasing penalty
"""

import torch
import torch.nn as nn
import numpy as np


class SoftQBoundPenalty:
    """
    Smooth QBound penalty functions for continuous action spaces.

    Instead of hard clipping: Q_clipped = clamp(Q, Q_min, Q_max)
    We use soft penalties: loss = primary_loss + λ * penalty(Q, Q_min, Q_max)

    This allows gradients to flow while still encouraging Q-values to stay in bounds.
    """

    @staticmethod
    def quadratic_penalty(q_values, q_min, q_max, margin=0.0):
        """
        Quadratic penalty: penalty = (violation)^2

        Smooth everywhere, stronger penalty for larger violations.
        Similar to L2 regularization.

        Args:
            q_values: Q-values to penalize (any shape)
            q_min: Lower bound (scalar or tensor matching q_values shape)
            q_max: Upper bound (scalar or tensor matching q_values shape)
            margin: Safety margin (start penalizing before hitting bounds)

        Returns:
            penalty: Smooth penalty term (scalar)
        """
        # Adjust bounds with margin
        q_min_safe = q_min + margin
        q_max_safe = q_max - margin

        # Penalty for violating lower bound
        lower_violation = torch.relu(q_min_safe - q_values)  # > 0 when Q < Q_min
        lower_penalty = lower_violation ** 2

        # Penalty for violating upper bound
        upper_violation = torch.relu(q_values - q_max_safe)  # > 0 when Q > Q_max
        upper_penalty = upper_violation ** 2

        # Total penalty (mean over batch)
        total_penalty = (lower_penalty + upper_penalty).mean()

        return total_penalty

    @staticmethod
    def huber_penalty(q_values, q_min, q_max, margin=0.0, delta=1.0):
        """
        Huber penalty: Smooth L1-like penalty with quadratic transition.

        Combines benefits of L1 (robust to outliers) and L2 (smooth).
        Penalty is quadratic near bounds, linear far from bounds.

        Args:
            q_values: Q-values to penalize
            q_min: Lower bound
            q_max: Upper bound
            margin: Safety margin
            delta: Huber loss threshold (transition point)

        Returns:
            penalty: Smooth penalty term
        """
        q_min_safe = q_min + margin
        q_max_safe = q_max - margin

        # Lower bound violations
        lower_violation = torch.relu(q_min_safe - q_values)
        lower_penalty = torch.where(
            lower_violation <= delta,
            0.5 * lower_violation ** 2,
            delta * (lower_violation - 0.5 * delta)
        )

        # Upper bound violations
        upper_violation = torch.relu(q_values - q_max_safe)
        upper_penalty = torch.where(
            upper_violation <= delta,
            0.5 * upper_violation ** 2,
            delta * (upper_violation - 0.5 * delta)
        )

        total_penalty = (lower_penalty + upper_penalty).mean()
        return total_penalty

    @staticmethod
    def exponential_penalty(q_values, q_min, q_max, margin=0.0, alpha=1.0):
        """
        Exponential penalty: Very smooth, exponentially increasing penalty.

        Penalty grows exponentially as Q-values move away from bounds.
        Most gradual of all penalties - useful when bounds are approximate.

        Args:
            q_values: Q-values to penalize
            q_min: Lower bound
            q_max: Upper bound
            margin: Safety margin
            alpha: Exponential growth rate (higher = steeper penalty)

        Returns:
            penalty: Smooth penalty term
        """
        q_min_safe = q_min + margin
        q_max_safe = q_max - margin

        # Lower bound violations
        lower_violation = torch.relu(q_min_safe - q_values)
        lower_penalty = (torch.exp(alpha * lower_violation) - 1.0) / alpha

        # Upper bound violations
        upper_violation = torch.relu(q_values - q_max_safe)
        upper_penalty = (torch.exp(alpha * upper_violation) - 1.0) / alpha

        total_penalty = (lower_penalty + upper_penalty).mean()
        return total_penalty

    @staticmethod
    def log_barrier_penalty(q_values, q_min, q_max, margin=0.1, epsilon=1e-6):
        """
        Log barrier penalty: Interior point method-inspired penalty.

        Penalty approaches infinity as Q-values approach bounds.
        Keeps Q-values strictly inside bounds with soft enforcement.

        Args:
            q_values: Q-values to penalize
            q_min: Lower bound
            q_max: Upper bound
            margin: Minimum distance from bounds
            epsilon: Numerical stability term

        Returns:
            penalty: Smooth penalty term
        """
        # Distance from bounds (clamped for stability)
        lower_dist = torch.clamp(q_values - q_min, min=epsilon)
        upper_dist = torch.clamp(q_max - q_values, min=epsilon)

        # Log barrier (penalty increases as we approach bounds)
        lower_penalty = -torch.log(lower_dist / margin)
        upper_penalty = -torch.log(upper_dist / margin)

        # Only penalize when within margin of bounds
        lower_penalty = torch.relu(lower_penalty)
        upper_penalty = torch.relu(upper_penalty)

        total_penalty = (lower_penalty + upper_penalty).mean()
        return total_penalty

    @staticmethod
    def softplus_clip(q_values, q_min, q_max, beta=1.0):
        """
        Soft clipping using softplus (smooth approximation of ReLU).

        Instead of penalty, this directly clips Q-values smoothly:
        - Approaches Q_min/Q_max asymptotically
        - Gradients always non-zero (unlike hard clipping)

        This is an alternative to penalties - use for Q-value clipping in targets.

        Args:
            q_values: Q-values to clip
            q_min: Lower bound
            q_max: Upper bound
            beta: Steepness (higher = closer to hard clipping)

        Returns:
            clipped_q: Smoothly clipped Q-values
        """
        # Soft lower bound: q_clipped >= q_min (smooth ReLU)
        q_shifted = q_values - q_min
        q_lower = q_min + torch.nn.functional.softplus(q_shifted, beta=beta)

        # Soft upper bound: q_clipped <= q_max (inverted smooth ReLU)
        q_shifted = q_max - q_lower
        q_clipped = q_max - torch.nn.functional.softplus(q_shifted, beta=beta)

        return q_clipped

    @staticmethod
    def tanh_squash(q_values, q_min, q_max):
        """
        Squash Q-values to bounded range using tanh.

        Maps any Q-value to [Q_min, Q_max] smoothly.
        Good for ensuring Q-values NEVER violate bounds.

        Args:
            q_values: Unbounded Q-values
            q_min: Lower bound
            q_max: Upper bound

        Returns:
            squashed_q: Q-values mapped to [Q_min, Q_max]
        """
        q_center = (q_max + q_min) / 2
        q_range = (q_max - q_min) / 2

        # tanh maps to [-1, 1], then scale to [Q_min, Q_max]
        squashed = q_center + q_range * torch.tanh(q_values / q_range)

        return squashed


def visualize_penalties():
    """Visualize different penalty functions for comparison."""
    import matplotlib.pyplot as plt

    # Test Q-values ranging from -150 to +150
    q_min, q_max = -100.0, 100.0
    q_values = torch.linspace(-150, 150, 300)

    penalty_fn = SoftQBoundPenalty()

    # Compute penalties
    quad_penalties = [penalty_fn.quadratic_penalty(q.unsqueeze(0), q_min, q_max).item()
                      for q in q_values]
    huber_penalties = [penalty_fn.huber_penalty(q.unsqueeze(0), q_min, q_max, delta=10.0).item()
                       for q in q_values]
    exp_penalties = [penalty_fn.exponential_penalty(q.unsqueeze(0), q_min, q_max, alpha=0.1).item()
                     for q in q_values]

    # Soft clipping
    soft_clipped = [penalty_fn.softplus_clip(q.unsqueeze(0), q_min, q_max, beta=0.1).item()
                    for q in q_values]
    hard_clipped = [torch.clamp(q, q_min, q_max).item() for q in q_values]

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Penalties
    axes[0].plot(q_values.numpy(), quad_penalties, label='Quadratic', linewidth=2)
    axes[0].plot(q_values.numpy(), huber_penalties, label='Huber', linewidth=2)
    axes[0].plot(q_values.numpy(), exp_penalties, label='Exponential', linewidth=2)
    axes[0].axvline(q_min, color='red', linestyle='--', alpha=0.5, label='Bounds')
    axes[0].axvline(q_max, color='red', linestyle='--', alpha=0.5)
    axes[0].set_xlabel('Q-value')
    axes[0].set_ylabel('Penalty')
    axes[0].set_title('Soft QBound Penalty Functions')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Clipping
    axes[1].plot(q_values.numpy(), q_values.numpy(), 'k--', alpha=0.3, label='No clipping')
    axes[1].plot(q_values.numpy(), hard_clipped, label='Hard clipping (gradient=0)', linewidth=2)
    axes[1].plot(q_values.numpy(), soft_clipped, label='Soft clipping (gradient>0)', linewidth=2)
    axes[1].axvline(q_min, color='red', linestyle='--', alpha=0.5)
    axes[1].axvline(q_max, color='red', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Q-value (input)')
    axes[1].set_ylabel('Q-value (output)')
    axes[1].set_title('Hard vs Soft Clipping')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/root/projects/QBound/docs/soft_qbound_visualization.png', dpi=300)
    print("✓ Visualization saved to docs/soft_qbound_visualization.png")


if __name__ == '__main__':
    # Test gradients
    print("=" * 80)
    print("Soft QBound: Gradient Flow Test")
    print("=" * 80)

    q_min, q_max = -100.0, 100.0
    penalty_fn = SoftQBoundPenalty()

    # Test Q-value that violates bounds
    q = torch.tensor([120.0], requires_grad=True)  # Violates Q_max

    print(f"\nTest Q-value: {q.item():.2f} (violates Q_max={q_max})")
    print(f"Violation amount: {q.item() - q_max:.2f}")

    # Hard clipping (ZERO gradient)
    q_hard = torch.clamp(q, q_min, q_max)
    loss_hard = q_hard.mean()
    loss_hard.backward()
    print(f"\nHard clipping gradient: {q.grad.item():.6f} ❌ (ZERO!)")

    # Soft penalty (NON-ZERO gradient)
    q.grad = None  # Reset gradient
    penalty = penalty_fn.quadratic_penalty(q, q_min, q_max)
    penalty.backward()
    print(f"Soft penalty gradient: {q.grad.item():.6f} ✓ (non-zero!)")

    print("\n" + "=" * 80)
    print("✓ Soft QBound allows gradient flow!")
    print("=" * 80)

    # Visualize
    visualize_penalties()
