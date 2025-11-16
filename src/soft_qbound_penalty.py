"""
Soft Clipping for Continuous RL

Hard clipping (torch.clamp) sets gradients to ZERO when Q-values violate bounds.
This breaks gradient-based methods like DDPG and TD3.

Soft clipping uses a smooth, differentiable function based on softplus that:
1. Allows gradients to flow even when Q-values are outside bounds
2. Approaches the bounds asymptotically (never exceeds them)
3. Behaves like a smooth approximation of hard clipping

Usage in QBound:
- Hard clipping on TD targets (prevents bootstrapping errors)
- Soft clipping on actor Q-values (preserves gradients for policy improvement)
"""

import torch
import torch.nn.functional as F


class SoftQBoundPenalty:
    """
    Soft clipping function for continuous action spaces.

    This class provides the softplus_clip method used in DDPG/TD3
    to preserve gradients during policy improvement.
    """

    @staticmethod
    def softplus_clip(q_values, q_min, q_max, beta=1.0):
        """
        Soft clipping using softplus (smooth approximation of ReLU).

        Instead of hard clipping which zeros gradients, this smoothly approaches
        the bounds asymptotically while maintaining non-zero gradients.

        Args:
            q_values: Q-values to clip
            q_min: Lower bound
            q_max: Upper bound
            beta: Steepness (higher = closer to hard clipping)

        Returns:
            clipped_q: Smoothly clipped Q-values

        Example:
            >>> penalty_fn = SoftQBoundPenalty()
            >>> q = torch.tensor([120.0], requires_grad=True)
            >>> q_clipped = penalty_fn.softplus_clip(q, -100.0, 100.0, beta=0.1)
            >>> # q_clipped ≈ 100.0 but gradients still flow!
        """
        # Soft lower bound: q_clipped >= q_min (smooth ReLU)
        q_shifted = q_values - q_min
        q_lower = q_min + F.softplus(q_shifted, beta=beta)

        # Soft upper bound: q_clipped <= q_max (inverted smooth ReLU)
        q_shifted = q_max - q_lower
        q_clipped = q_max - F.softplus(q_shifted, beta=beta)

        return q_clipped


if __name__ == '__main__':
    # Test gradients
    print("=" * 80)
    print("Soft Clipping: Gradient Flow Test")
    print("=" * 80)

    q_min, q_max = -100.0, 100.0
    penalty_fn = SoftQBoundPenalty()

    # Test Q-value that violates bounds
    q = torch.tensor([120.0], requires_grad=True)

    print(f"\nTest Q-value: {q.item():.2f} (violates Q_max={q_max})")
    print(f"Violation amount: {q.item() - q_max:.2f}")

    # Hard clipping (ZERO gradient)
    q_hard = torch.clamp(q, q_min, q_max)
    loss_hard = q_hard.mean()
    loss_hard.backward()
    print(f"\nHard clipping gradient: {q.grad.item():.6f} ❌ (ZERO!)")

    # Soft clipping (NON-ZERO gradient)
    q.grad = None  # Reset gradient
    q_soft = penalty_fn.softplus_clip(q, q_min, q_max, beta=0.1)
    loss_soft = q_soft.mean()
    loss_soft.backward()
    print(f"Soft clipping gradient: {q.grad.item():.6f} ✓ (non-zero!)")
    print(f"Soft clipped value: {q_soft.item():.6f} (approaches {q_max})")

    print("\n" + "=" * 80)
    print("✓ Soft clipping preserves gradient flow!")
    print("=" * 80)
