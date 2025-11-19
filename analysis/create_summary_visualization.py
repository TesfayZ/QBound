#!/usr/bin/env python3
"""
Create a comprehensive summary visualization showing where QBound works and where it doesn't
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# Use non-interactive backend
matplotlib.use('Agg')

# Data from multiseed analysis
results = {
    'CartPole DQN': {'baseline': 351.07, 'qbound': 393.24, 'improvement': 12.0},
    'CartPole DDQN': {'baseline': 147.83, 'qbound': 197.50, 'improvement': 33.6},
    'CartPole Dueling': {'baseline': 289.30, 'qbound': 354.45, 'improvement': 22.5},
    'CartPole Double-Dueling': {'baseline': 321.80, 'qbound': 371.79, 'improvement': 15.5},
    'Pendulum DDPG': {'baseline': -213.10, 'qbound': -159.79, 'improvement': 25.0},
    'Pendulum TD3': {'baseline': -202.39, 'qbound': -171.52, 'improvement': 15.3},
    'Pendulum DQN': {'baseline': -156.25, 'qbound': -167.19, 'improvement': -7.0},
    'Pendulum Double-DQN': {'baseline': -171.35, 'qbound': -177.08, 'improvement': -3.3},
    'Pendulum PPO': {'baseline': -784.96, 'qbound': -945.09, 'improvement': -20.4},
    'GridWorld DQN': {'baseline': 0.99, 'qbound': 0.98, 'improvement': -1.0},
    'FrozenLake DQN': {'baseline': 0.60, 'qbound': 0.59, 'improvement': -1.7},
    'MountainCar DQN': {'baseline': -124.14, 'qbound': -134.31, 'improvement': -8.2},
    'MountainCar DDQN': {'baseline': -122.72, 'qbound': -180.93, 'improvement': -47.4},
    'Acrobot DQN': {'baseline': -88.74, 'qbound': -93.07, 'improvement': -4.9},
    'Acrobot DDQN': {'baseline': -83.99, 'qbound': -87.04, 'improvement': -3.6},
}

# Sort by improvement percentage
sorted_results = sorted(results.items(), key=lambda x: x[1]['improvement'], reverse=True)

# Create figure
fig, ax = plt.subplots(figsize=(14, 10))

methods = [item[0] for item in sorted_results]
improvements = [item[1]['improvement'] for item in sorted_results]

# Color coding
colors = []
for imp in improvements:
    if imp >= 10:
        colors.append('#2ca02c')  # Strong green for significant positive
    elif imp >= 5:
        colors.append('#8bc34a')  # Light green for moderate positive
    elif imp >= -5:
        colors.append('#ffc107')  # Yellow for neutral/small change
    elif imp >= -10:
        colors.append('#ff9800')  # Orange for moderate negative
    else:
        colors.append('#f44336')  # Red for significant negative

# Create horizontal bar chart
y_pos = np.arange(len(methods))
bars = ax.barh(y_pos, improvements, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

# Add percentage labels on bars
for i, (bar, imp) in enumerate(zip(bars, improvements)):
    x_pos = bar.get_width()
    label_x = x_pos + (2 if x_pos > 0 else -2)
    ha = 'left' if x_pos > 0 else 'right'
    ax.text(label_x, bar.get_y() + bar.get_height()/2, f'{imp:+.1f}%',
            va='center', ha=ha, fontsize=11, fontweight='bold')

# Vertical line at 0
ax.axvline(x=0, color='black', linewidth=2, linestyle='-')

# Formatting
ax.set_yticks(y_pos)
ax.set_yticklabels(methods, fontsize=11)
ax.set_xlabel('Performance Improvement (%)', fontsize=14, fontweight='bold')
ax.set_title('QBound Performance Impact Across All Experiments\n(Positive = QBound Better, Negative = Baseline Better)',
             fontsize=16, fontweight='bold', pad=20)

# Grid
ax.grid(True, axis='x', alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#2ca02c', label='Strong Positive (≥10%)', alpha=0.8, edgecolor='black'),
    Patch(facecolor='#8bc34a', label='Moderate Positive (5-10%)', alpha=0.8, edgecolor='black'),
    Patch(facecolor='#ffc107', label='Neutral (±5%)', alpha=0.8, edgecolor='black'),
    Patch(facecolor='#ff9800', label='Moderate Negative (-5 to -10%)', alpha=0.8, edgecolor='black'),
    Patch(facecolor='#f44336', label='Strong Negative (<-10%)', alpha=0.8, edgecolor='black'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

plt.tight_layout()

# Save
plt.savefig('results/plots/qbound_summary_all_experiments.pdf', dpi=300, bbox_inches='tight')
plt.savefig('results/plots/qbound_summary_all_experiments.png', dpi=300, bbox_inches='tight')
print("Saved: results/plots/qbound_summary_all_experiments.pdf")
print("Saved: results/plots/qbound_summary_all_experiments.png")

# Copy to paper directory
import shutil
shutil.copy2('results/plots/qbound_summary_all_experiments.pdf', 'QBound/figures/qbound_summary_all_experiments.pdf')
print("Copied to: QBound/figures/qbound_summary_all_experiments.pdf")

plt.close()

# Create category summary
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

categories = {
    'Dense Positive\nRewards\n(CartPole)': [12.0, 33.6, 22.5, 15.5],
    'Continuous\nControl\n(DDPG/TD3)': [25.0, 15.3],
    'Sparse/State\nRewards': [-7.0, -3.3, -20.4, -1.0, -1.7, -8.2, -47.4, -4.9, -3.6]
}

# Plot 1: Dense Positive Rewards
ax1.bar(range(len(categories['Dense Positive\nRewards\n(CartPole)'])),
        categories['Dense Positive\nRewards\n(CartPole)'],
        color='#2ca02c', alpha=0.8, edgecolor='black', linewidth=1.5)
ax1.axhline(y=0, color='black', linewidth=1)
ax1.set_ylabel('Improvement (%)', fontsize=12, fontweight='bold')
ax1.set_title('✅ Dense Positive Rewards\n(CartPole)', fontsize=13, fontweight='bold')
ax1.set_xticks(range(4))
ax1.set_xticklabels(['DQN', 'DDQN', 'Dueling', 'Double-\nDueling'], fontsize=10)
ax1.grid(True, axis='y', alpha=0.3)
ax1.set_ylim([-10, 40])
for i, v in enumerate(categories['Dense Positive\nRewards\n(CartPole)']):
    ax1.text(i, v + 1, f'+{v:.1f}%', ha='center', va='bottom', fontweight='bold')

# Plot 2: Continuous Control
ax2.bar(range(len(categories['Continuous\nControl\n(DDPG/TD3)'])),
        categories['Continuous\nControl\n(DDPG/TD3)'],
        color='#2ca02c', alpha=0.8, edgecolor='black', linewidth=1.5)
ax2.axhline(y=0, color='black', linewidth=1)
ax2.set_ylabel('Improvement (%)', fontsize=12, fontweight='bold')
ax2.set_title('✅ Continuous Control\n(Pendulum DDPG/TD3)', fontsize=13, fontweight='bold')
ax2.set_xticks(range(2))
ax2.set_xticklabels(['DDPG', 'TD3'], fontsize=10)
ax2.grid(True, axis='y', alpha=0.3)
ax2.set_ylim([-10, 40])
for i, v in enumerate(categories['Continuous\nControl\n(DDPG/TD3)']):
    ax2.text(i, v + 1, f'+{v:.1f}%', ha='center', va='bottom', fontweight='bold')

# Plot 3: Sparse/State Rewards
vals = categories['Sparse/State\nRewards']
colors_sparse = ['#ff9800' if v > -10 else '#f44336' for v in vals]
ax3.bar(range(len(vals)), vals, color=colors_sparse, alpha=0.8, edgecolor='black', linewidth=1.5)
ax3.axhline(y=0, color='black', linewidth=1)
ax3.set_ylabel('Improvement (%)', fontsize=12, fontweight='bold')
ax3.set_title('❌ Sparse/State-Dependent\n(All Other Envs)', fontsize=13, fontweight='bold')
ax3.set_xticks(range(9))
ax3.set_xticklabels(['Pend\nDQN', 'Pend\nDDQN', 'Pend\nPPO', 'Grid\nDQN', 'Frozen\nDQN',
                     'Mtn\nDQN', 'Mtn\nDDQN', 'Acro\nDQN', 'Acro\nDDQN'], fontsize=9)
ax3.grid(True, axis='y', alpha=0.3)
ax3.set_ylim([-55, 10])
for i, v in enumerate(vals):
    y_pos = v - 2 if v < -10 else v + 1
    va = 'top' if v < -10 else 'bottom'
    ax3.text(i, y_pos, f'{v:.1f}%', ha='center', va=va, fontweight='bold', fontsize=9)

plt.tight_layout()
plt.savefig('results/plots/qbound_category_summary.pdf', dpi=300, bbox_inches='tight')
plt.savefig('results/plots/qbound_category_summary.png', dpi=300, bbox_inches='tight')
print("Saved: results/plots/qbound_category_summary.pdf")

shutil.copy2('results/plots/qbound_category_summary.pdf', 'QBound/figures/qbound_category_summary.pdf')
print("Copied to: QBound/figures/qbound_category_summary.pdf")

print("\n✅ Summary visualizations complete!")
