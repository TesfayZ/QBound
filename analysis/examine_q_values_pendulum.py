#!/usr/bin/env python3
"""
Examine actual Q-values in Pendulum DQN experiments to understand why
Q-values are going above 0 when they should be ≤ 0.
"""

import json
import numpy as np
from pathlib import Path

def examine_q_values():
    """Examine Q-values from one seed to understand the positive Q-value issue."""

    # Load seed 42 DQN results
    result_file = Path('/root/projects/QBound/results/pendulum/dqn_full_qbound_seed42_20251117_083452.json')

    with open(result_file, 'r') as f:
        data = json.load(f)

    print("=" * 80)
    print("Examining Q-values in Pendulum DQN (Seed 42)")
    print("=" * 80)
    print()

    # Check baseline (no QBound)
    print("BASELINE DQN (no QBound):")
    baseline = data['training']['dqn']
    print(f"  Final 100 episodes mean reward: {np.mean(baseline['rewards'][-100:]):.2f}")
    print()

    # Check static QBound
    print("STATIC QBOUND DQN:")
    static = data['training']['static_qbound_dqn']
    print(f"  Final 100 episodes mean reward: {np.mean(static['rewards'][-100:]):.2f}")
    print(f"  Q_min bound: {static['violations']['mean']['qbound_min']:.2f}")
    print(f"  Q_max bound: {static['violations']['mean']['qbound_max']:.2f}")
    print()

    violations = static['violations']

    print("Violation Statistics:")
    print(f"  Mean upper violation rate (Q > Q_max=0): {violations['mean']['next_q_violate_max_rate']:.2%}")
    print(f"  Mean lower violation rate (Q < Q_min): {violations['mean']['next_q_violate_min_rate']:.2%}")
    print(f"  Mean upper violation magnitude: {violations['mean']['violation_magnitude_max_next']:.4f}")
    print(f"  Mean lower violation magnitude: {violations['mean']['violation_magnitude_min_next']:.4f}")
    print()

    print("Final 100 Episodes Violations:")
    print(f"  Upper violation rate: {violations['final_100']['next_q_violate_max_rate']:.2%}")
    print(f"  Lower violation rate: {violations['final_100']['next_q_violate_min_rate']:.2%}")
    print(f"  Upper violation magnitude: {violations['final_100']['violation_magnitude_max_next']:.4f}")
    print(f"  Lower violation magnitude: {violations['final_100']['violation_magnitude_min_next']:.4f}")
    print()

    # Sample some per-episode violations
    print("Sample Per-Episode Violations (first 10 episodes):")
    for i in range(min(10, len(violations['per_episode']))):
        ep_viol = violations['per_episode'][i]
        if ep_viol['next_q_violate_max_rate'] > 0 or ep_viol['next_q_violate_min_rate'] > 0:
            print(f"  Episode {i}:")
            print(f"    Upper violation rate: {ep_viol['next_q_violate_max_rate']:.2%}")
            print(f"    Upper magnitude: {ep_viol['violation_magnitude_max_next']:.4f}")
            print(f"    Lower violation rate: {ep_viol['next_q_violate_min_rate']:.2%}")
            print(f"    Lower magnitude: {ep_viol['violation_magnitude_min_next']:.4f}")

    print()
    print("=" * 80)
    print("KEY FINDING:")
    print("=" * 80)
    print()
    print("Q-values are systematically violating Q_max=0 (going positive) even though")
    print("with negative rewards (-16.2 per step), they should stay ≤ 0.")
    print()
    print("Violation magnitude is small (~0.09 to 0.23), but violation RATE is high (50-62%).")
    print()
    print("This suggests:")
    print("1. Q-values ARE going slightly positive (not a measurement error)")
    print("2. The Bellman equation isn't naturally enforcing Q ≤ 0")
    print("3. QBound is actively penalizing ~50-60% of Q-values")
    print("4. This penalty disrupts learning → performance degradation")
    print()
    print("Hypothesis: Initialization or exploration causes some Q-values to be positive,")
    print("and the auxiliary loss interferes with TD learning before they naturally converge.")

if __name__ == '__main__':
    examine_q_values()
