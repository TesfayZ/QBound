#!/usr/bin/env python3
"""
Visualize degradation comparison across all algorithms.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_all_results():
    """Load results for all algorithms."""
    results = {
        'DQN': {'baseline': [], 'qbound': [], 'seeds': []},
        'DDQN': {'baseline': [], 'qbound': [], 'seeds': []},
        'DDPG': {'baseline': [], 'qbound': [], 'seeds': []},
        'TD3': {'baseline': [], 'qbound': [], 'seeds': []},
        'PPO': {'baseline': [], 'qbound': [], 'seeds': []},
    }

    results_dir = Path('/root/projects/QBound/results/pendulum')

    # DQN results
    for seed in [42, 43, 44, 45, 46]:
        pattern = f"dqn_full_qbound_seed{seed}_*.json"
        files = list(results_dir.glob(pattern))
        if files:
            file = [f for f in files if 'in_progress' not in str(f)]
            if file:
                with open(file[0], 'r') as f:
                    data = json.load(f)
                    baseline = np.mean(data['training']['dqn']['rewards'][-100:])
                    qbound = np.mean(data['training']['static_qbound_dqn']['rewards'][-100:])
                    results['DQN']['baseline'].append(baseline)
                    results['DQN']['qbound'].append(qbound)
                    results['DQN']['seeds'].append(seed)

                    # DDQN
                    baseline_ddqn = np.mean(data['training']['double_dqn']['rewards'][-100:])
                    qbound_ddqn = np.mean(data['training']['static_qbound_double_dqn']['rewards'][-100:])
                    results['DDQN']['baseline'].append(baseline_ddqn)
                    results['DDQN']['qbound'].append(qbound_ddqn)
                    results['DDQN']['seeds'].append(seed)

    # DDPG results
    for seed in [42, 43, 44, 45, 46]:
        pattern = f"ddpg_full_qbound_seed{seed}_*.json"
        files = list(results_dir.glob(pattern))
        if files:
            file = [f for f in files if 'in_progress' not in str(f)]
            if file:
                with open(file[0], 'r') as f:
                    data = json.load(f)
                    baseline = np.mean(data['training']['baseline']['rewards'][-100:])
                    qbound = np.mean(data['training']['static_soft_qbound']['rewards'][-100:])
                    results['DDPG']['baseline'].append(baseline)
                    results['DDPG']['qbound'].append(qbound)
                    results['DDPG']['seeds'].append(seed)

    # TD3 results
    for seed in [42, 43, 44, 45, 46]:
        pattern = f"td3_full_qbound_seed{seed}_*.json"
        files = list(results_dir.glob(pattern))
        if files:
            file = [f for f in files if 'in_progress' not in str(f)]
            if file:
                with open(file[0], 'r') as f:
                    data = json.load(f)
                    baseline = np.mean(data['training']['baseline']['rewards'][-100:])
                    qbound = np.mean(data['training']['static_soft_qbound']['rewards'][-100:])
                    results['TD3']['baseline'].append(baseline)
                    results['TD3']['qbound'].append(qbound)
                    results['TD3']['seeds'].append(seed)

    # PPO results
    for seed in [42, 43, 44, 45, 46]:
        pattern = f"ppo_full_qbound_seed{seed}_*.json"
        files = list(results_dir.glob(pattern))
        if files:
            file = [f for f in files if 'in_progress' not in str(f)]
            if file:
                with open(file[0], 'r') as f:
                    data = json.load(f)
                    baseline = np.mean(data['training']['baseline']['rewards'][-100:])
                    qbound = np.mean(data['training']['static_soft_qbound']['rewards'][-100:])
                    results['PPO']['baseline'].append(baseline)
                    results['PPO']['qbound'].append(qbound)
                    results['PPO']['seeds'].append(seed)

    return results

def plot_comparison():
    """Create comprehensive comparison plot."""
    results = load_all_results()

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    # Plot 1: Mean degradation by algorithm
    ax = axes[0]
    algorithms = ['DQN\n(hard)', 'DDQN\n(hard)', 'DDPG\n(soft)', 'TD3\n(soft)', 'PPO\n(soft V)']
    degradations = []
    errors = []
    colors = []

    for algo in ['DQN', 'DDQN', 'DDPG', 'TD3', 'PPO']:
        if results[algo]['baseline']:
            baseline = np.array(results[algo]['baseline'])
            qbound = np.array(results[algo]['qbound'])
            deg = ((qbound / baseline) - 1) * 100
            degradations.append(np.mean(deg))
            errors.append(np.std(deg))

            # Color based on performance
            if np.mean(deg) < -5:
                colors.append('green')
            elif np.mean(deg) < 0:
                colors.append('lightgreen')
            elif np.mean(deg) < 5:
                colors.append('orange')
            else:
                colors.append('red')

    x_pos = np.arange(len(algorithms))
    bars = ax.bar(x_pos, degradations, yerr=errors, capsize=5, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

    ax.axhline(y=0, color='black', linestyle='--', linewidth=2, label='No change')
    ax.set_xlabel('Algorithm (Clipping Type)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Mean Degradation (%)', fontsize=13, fontweight='bold')
    ax.set_title('QBound Impact on Negative Rewards (Pendulum)\nBy Algorithm and Clipping Type', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(algorithms, fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(fontsize=11)

    # Add value labels
    for i, (bar, val, err) in enumerate(zip(bars, degradations, errors)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + err + 2,
                f'{val:.1f}%\n±{err:.1f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Plot 2: Performance comparison (baseline vs QBound)
    ax = axes[1]
    x_offset = np.array([-0.2, -0.1, 0, 0.1, 0.2])

    for i, algo in enumerate(['DQN', 'DDQN', 'DDPG', 'TD3', 'PPO']):
        if results[algo]['baseline']:
            baseline = np.array(results[algo]['baseline'])
            qbound = np.array(results[algo]['qbound'])

            ax.scatter(np.ones(len(baseline)) * i + x_offset[:len(baseline)] - 0.15,
                      baseline, color='blue', alpha=0.6, s=100, marker='o', label='Baseline' if i == 0 else '')
            ax.scatter(np.ones(len(qbound)) * i + x_offset[:len(qbound)] + 0.15,
                      qbound, color='red', alpha=0.6, s=100, marker='s', label='QBound' if i == 0 else '')

            # Connect with lines
            for j in range(len(baseline)):
                color = 'green' if qbound[j] > baseline[j] else 'red'
                ax.plot([i + x_offset[j] - 0.15, i + x_offset[j] + 0.15],
                       [baseline[j], qbound[j]], color=color, alpha=0.3, linewidth=2)

    ax.set_xlabel('Algorithm', fontsize=13, fontweight='bold')
    ax.set_ylabel('Final Performance (Mean Reward)', fontsize=13, fontweight='bold')
    ax.set_title('Baseline vs QBound Performance by Seed', fontsize=14, fontweight='bold')
    ax.set_xticks(range(5))
    ax.set_xticklabels(['DQN', 'DDQN', 'DDPG', 'TD3', 'PPO'], fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 3: Variance comparison
    ax = axes[2]
    variances = []
    for algo in ['DQN', 'DDQN', 'DDPG', 'TD3', 'PPO']:
        if results[algo]['baseline']:
            baseline = np.array(results[algo]['baseline'])
            qbound = np.array(results[algo]['qbound'])
            deg = ((qbound / baseline) - 1) * 100
            variances.append(np.std(deg))

    bars = ax.bar(range(len(algorithms)), variances, color=['steelblue', 'steelblue', 'orange', 'orange', 'darkred'], alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.set_xlabel('Algorithm', fontsize=13, fontweight='bold')
    ax.set_ylabel('Standard Deviation of Degradation', fontsize=13, fontweight='bold')
    ax.set_title('Consistency of QBound Effect\n(Lower = More Consistent)', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(algorithms)))
    ax.set_xticklabels(algorithms, fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    # Add annotations
    for i, (bar, val) in enumerate(zip(bars, variances)):
        height = bar.get_height()
        consistency = 'Consistent' if val < 10 else 'Variable' if val < 30 else 'Highly Variable'
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{val:.1f}%\n({consistency})',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Plot 4: Hard vs Soft clipping comparison
    ax = axes[3]
    hard_clip = ['DQN', 'DDQN']
    soft_clip = ['DDPG', 'TD3']

    hard_degs = []
    soft_degs = []

    for algo in hard_clip:
        if results[algo]['baseline']:
            baseline = np.array(results[algo]['baseline'])
            qbound = np.array(results[algo]['qbound'])
            deg = ((qbound / baseline) - 1) * 100
            hard_degs.extend(deg)

    for algo in soft_clip:
        if results[algo]['baseline']:
            baseline = np.array(results[algo]['baseline'])
            qbound = np.array(results[algo]['qbound'])
            deg = ((qbound / baseline) - 1) * 100
            soft_degs.extend(deg)

    bp = ax.boxplot([hard_degs, soft_degs], labels=['Hard Clipping\n(DQN, DDQN)', 'Soft Clipping\n(DDPG, TD3)'],
                     patch_artist=True, widths=0.6)

    bp['boxes'][0].set_facecolor('lightcoral')
    bp['boxes'][1].set_facecolor('lightgreen')

    ax.axhline(y=0, color='black', linestyle='--', linewidth=2)
    ax.set_ylabel('Degradation (%)', fontsize=13, fontweight='bold')
    ax.set_title('Hard vs Soft Clipping Comparison\n(Excluding PPO)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Add statistics
    hard_mean = np.mean(hard_degs)
    soft_mean = np.mean(soft_degs)
    ax.text(1, max(hard_degs) + 5, f'Mean: {hard_mean:.1f}%', ha='center', fontsize=11, fontweight='bold', color='red')
    ax.text(2, max(soft_degs) + 5, f'Mean: {soft_mean:.1f}%', ha='center', fontsize=11, fontweight='bold', color='green')

    plt.tight_layout()
    plt.savefig('/root/projects/QBound/results/plots/all_algorithms_comparison.png', dpi=150, bbox_inches='tight')
    plt.savefig('/root/projects/QBound/results/plots/all_algorithms_comparison.pdf', bbox_inches='tight')
    print("Saved: results/plots/all_algorithms_comparison.{png,pdf}")

if __name__ == '__main__':
    plot_comparison()
