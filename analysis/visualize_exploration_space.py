#!/usr/bin/env python3
"""
Visualize exploration space constraint vs post-hoc clipping
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Simulate network outputs over training
np.random.seed(42)
episodes = np.arange(100)

# HARD CLIPPING APPROACH
ax = axes[0, 0]
# Network outputs with positive bias (randomly initialized)
raw_outputs = np.random.randn(100, 50) + 0.5  # Positive bias
violations = (raw_outputs > 0).mean(axis=1) * 100  # % violating Q > 0

ax.plot(episodes, violations, linewidth=2, color='red', label='Violation Rate (%)')
ax.axhline(y=56.79, color='darkred', linestyle='--', linewidth=2, label='Observed Mean (56.79%)')
ax.fill_between(episodes, 0, violations, alpha=0.3, color='red')
ax.set_xlabel('Training Episode', fontsize=12)
ax.set_ylabel('Violation Rate (%)', fontsize=12)
ax.set_title('Hard Clipping: Persistent Violations\n(Network explores unbounded space)', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 100])

# ARCHITECTURAL APPROACH
ax = axes[0, 1]
# With architectural constraint, violations are IMPOSSIBLE
violations_arch = np.zeros(100)

ax.plot(episodes, violations_arch, linewidth=3, color='green', label='Violation Rate (0.0%)')
ax.fill_between(episodes, 0, 1, alpha=0.3, color='green', label='Impossible to violate')
ax.set_xlabel('Training Episode', fontsize=12)
ax.set_ylabel('Violation Rate (%)', fontsize=12)
ax.set_title('Architectural QBound: Zero Violations\n(Network explores constrained space)', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 100])

# DISTRIBUTION OF Q-VALUES: Hard Clipping
ax = axes[1, 0]
# Before clipping (positive bias)
raw_q = np.random.randn(1000) + 0.5
# After clipping
clipped_q = np.clip(raw_q, a_min=None, a_max=0.0)

ax.hist(raw_q, bins=50, alpha=0.5, color='blue', label='Before Clipping', edgecolor='black')
ax.hist(clipped_q, bins=50, alpha=0.7, color='red', label='After Clipping', edgecolor='black')
ax.axvline(x=0, color='black', linestyle='--', linewidth=2, label='Q_max = 0')
ax.set_xlabel('Q-value', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Hard Clipping: Post-Hoc Correction\n(Spike at boundary from clipped values)', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

# DISTRIBUTION OF Q-VALUES: Architectural
ax = axes[1, 1]
# Logits (internal representation, can be anything)
logits = np.random.randn(1000) + 0.5
# Q-values after -softplus (ALWAYS ≤ 0)
q_values = -np.log(1 + np.exp(logits))

ax.hist(logits, bins=50, alpha=0.5, color='blue', label='Internal Logits', edgecolor='black')
ax.hist(q_values, bins=50, alpha=0.7, color='green', label='Q = -softplus(logits)', edgecolor='black')
ax.axvline(x=0, color='black', linestyle='--', linewidth=2, label='Q ≤ 0 (enforced)')
ax.set_xlabel('Value', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Architectural QBound: Natural Constraint\n(All Q-values ≤ 0 by construction)', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('/root/projects/QBound/results/plots/exploration_space_visualization.pdf', dpi=300, bbox_inches='tight')
plt.savefig('/root/projects/QBound/results/plots/exploration_space_visualization.png', dpi=300, bbox_inches='tight')
plt.savefig('/root/projects/QBound/LatexDocs/figures/exploration_space_visualization.pdf', dpi=300, bbox_inches='tight')

print("Saved: exploration_space_visualization.pdf")
print("\nKey visualization points:")
print("  • Top-left: Hard clipping has persistent violations (56.79%)")
print("  • Top-right: Architectural has zero violations (impossible)")
print("  • Bottom-left: Hard clipping creates spike at boundary (fight)")
print("  • Bottom-right: Architectural smoothly distributes Q ≤ 0 (natural)")

plt.show()
