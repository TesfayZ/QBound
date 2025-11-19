#!/usr/bin/env python3
"""
Visualize how clipping affects learning in Pendulum.

Shows:
1. Learning curves: Baseline vs Static QBound
2. The mechanism: How clipping biases targets
3. State-value granularity loss
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_learning_curves():
    """Compare learning curves with and without clipping."""
    # Load seed 42
    result_file = Path('/root/projects/QBound/results/pendulum/dqn_full_qbound_seed42_20251117_083452.json')

    with open(result_file, 'r') as f:
        data = json.load(f)

    baseline_rewards = np.array(data['training']['dqn']['rewards'])
    static_rewards = np.array(data['training']['static_qbound_dqn']['rewards'])

    # Smooth with moving average
    window = 20
    baseline_smooth = np.convolve(baseline_rewards, np.ones(window)/window, mode='valid')
    static_smooth = np.convolve(static_rewards, np.ones(window)/window, mode='valid')

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Plot 1: Learning curves
    ax = axes[0]
    episodes = np.arange(len(baseline_smooth))
    ax.plot(episodes, baseline_smooth, label='Baseline (no clipping)', linewidth=2, alpha=0.8)
    ax.plot(episodes, static_smooth, label='Static QBound (clipping)', linewidth=2, alpha=0.8)
    ax.axhline(y=baseline_rewards[-100:].mean(), color='C0', linestyle='--', alpha=0.5, label=f'Baseline final: {baseline_rewards[-100:].mean():.1f}')
    ax.axhline(y=static_rewards[-100:].mean(), color='C1', linestyle='--', alpha=0.5, label=f'QBound final: {static_rewards[-100:].mean():.1f}')

    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Episode Return (smoothed)', fontsize=12)
    ax.set_title('Pendulum DQN: Clipping Hurts Performance\n(Seed 42, 20-episode moving average)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Add annotation
    ax.annotate('Clipping causes 16.7% degradation',
                xy=(450, static_rewards[-100:].mean()),
                xytext=(350, static_rewards[-100:].mean() - 100),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=11, color='red', fontweight='bold')

    # Plot 2: Violation rates over time
    ax = axes[1]
    violations = data['training']['static_qbound_dqn']['violations']['per_episode']
    violation_rates = [v['next_q_violate_max_rate'] * 100 for v in violations]

    ax.plot(violation_rates, label='Q > Q_max=0 violation rate', linewidth=2, color='red', alpha=0.7)
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% threshold')
    ax.fill_between(range(len(violation_rates)), 0, violation_rates, alpha=0.3, color='red')

    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Violation Rate (%)', fontsize=12)
    ax.set_title('Q-value Violations Persist Throughout Training\n(50-62% of Q-values exceed Q_max=0)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 100])

    # Add annotation
    mean_viol = np.mean(violation_rates[-100:])
    ax.annotate(f'Final 100 episodes:\n{mean_viol:.1f}% violation rate',
                xy=(400, mean_viol),
                xytext=(300, 75),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=11, color='red', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig('/root/projects/QBound/results/plots/clipping_effect_analysis.png', dpi=150, bbox_inches='tight')
    plt.savefig('/root/projects/QBound/results/plots/clipping_effect_analysis.pdf', bbox_inches='tight')
    print("Saved: results/plots/clipping_effect_analysis.{png,pdf}")

def plot_mechanism_diagram():
    """Illustrate the clipping mechanism and its effect."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Mechanism diagram
    ax = axes[0]
    ax.axis('off')

    # Title
    ax.text(0.5, 0.95, 'Why Clipping Hurts: Loss of Granularity',
            ha='center', va='top', fontsize=14, fontweight='bold')

    # WITHOUT clipping
    y_pos = 0.75
    ax.text(0.25, y_pos, 'WITHOUT Clipping', ha='center', fontsize=12, fontweight='bold', color='green')

    y_pos = 0.65
    states = [
        ('Near terminal (1 step)', '+0.1', '-16.1', 'green'),
        ('Medium distance (10 steps)', '-50', '-155', 'green'),
        ('Far from terminal (50 steps)', '-300', '-640', 'green'),
    ]

    for i, (state, q_pred, q_true, color) in enumerate(states):
        y = y_pos - i * 0.12
        ax.text(0.05, y, state, ha='left', fontsize=9)
        ax.text(0.25, y, f'Q={q_pred}', ha='center', fontsize=9, color=color, fontweight='bold')
        ax.text(0.35, y, f'→ target ≈ {q_true}', ha='left', fontsize=9, color=color)

    ax.text(0.25, y_pos - 3*0.12 - 0.05, '✓ Relative ordering preserved\n✓ Fine distinctions learned',
            ha='center', fontsize=10, color='green', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

    # WITH clipping
    y_pos = 0.75
    ax.text(0.75, y_pos, 'WITH Clipping', ha='center', fontsize=12, fontweight='bold', color='red')

    y_pos = 0.65
    states_clipped = [
        ('Near terminal (1 step)', '+0.1 → 0', '-16.2', 'red'),
        ('Medium distance (10 steps)', '-50', '-66', 'orange'),
        ('Far from terminal (50 steps)', '-300', '-640', 'green'),
    ]

    for i, (state, q_pred, q_true, color) in enumerate(states_clipped):
        y = y_pos - i * 0.12
        ax.text(0.55, y, state, ha='left', fontsize=9)
        ax.text(0.75, y, f'Q={q_pred}', ha='center', fontsize=9, color=color, fontweight='bold')
        ax.text(0.85, y, f'→ target ≈ {q_true}', ha='left', fontsize=9, color=color)

    ax.text(0.75, y_pos - 3*0.12 - 0.05, '✗ Clipping biases near-terminal states\n✗ Can\'t distinguish 1 vs 2 steps',
            ha='center', fontsize=10, color='red', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3))

    # Right: Theoretical Q-values
    ax = axes[1]
    steps = np.array([1, 5, 10, 20, 50, 100, 200])
    gamma = 0.99
    reward = -16.2
    q_values = [reward * (1 - gamma**s) / (1 - gamma) for s in steps]

    ax.barh(range(len(steps)), q_values, color='steelblue', alpha=0.7)
    ax.axvline(x=0, color='red', linewidth=3, linestyle='--', label='Q_max=0 (clipping boundary)')

    # Annotate near-terminal states
    ax.fill_betweenx([0, 2], 0, -50, alpha=0.2, color='red',
                     label='Near-terminal states\n(most affected by clipping)')

    ax.set_yticks(range(len(steps)))
    ax.set_yticklabels([f'{s} steps' for s in steps])
    ax.set_xlabel('Theoretical Q-value', fontsize=12)
    ax.set_ylabel('Steps Remaining', fontsize=12)
    ax.set_title('Theoretical Q-values in Pendulum\n(All negative, but vary widely)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, alpha=0.3, axis='x')

    # Add value labels
    for i, (s, q) in enumerate(zip(steps, q_values)):
        ax.text(q - 50, i, f'{q:.0f}', ha='right', va='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig('/root/projects/QBound/results/plots/clipping_mechanism.png', dpi=150, bbox_inches='tight')
    plt.savefig('/root/projects/QBound/results/plots/clipping_mechanism.pdf', bbox_inches='tight')
    print("Saved: results/plots/clipping_mechanism.{png,pdf}")

if __name__ == '__main__':
    plot_learning_curves()
    plot_mechanism_diagram()
    print("\nVisualization complete!")
