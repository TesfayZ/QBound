#!/usr/bin/env python3
"""
Comprehensive analysis of CartPole Corrected 6-way comparison experiment
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def moving_average(data, window=20):
    """Calculate moving average with given window size"""
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window)/window, mode='valid')

def analyze_agent(name, data):
    """Analyze performance metrics for a single agent"""
    rewards = np.array(data['rewards'])

    # Overall statistics
    stats = {
        'name': name,
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'max_reward': np.max(rewards),
        'min_reward': np.min(rewards),
        'final_100_mean': np.mean(rewards[-100:]),
        'final_100_std': np.std(rewards[-100:]),
    }

    # Success rate (reward >= 475 is considered success for CartPole, ~95% of max)
    stats['success_rate'] = np.sum(rewards >= 475) / len(rewards) * 100
    stats['final_100_success_rate'] = np.sum(rewards[-100:] >= 475) / 100 * 100

    return stats

def main():
    # Find the latest CartPole corrected results
    results_dir = Path('/root/projects/QBound/results/cartpole_corrected')
    result_files = sorted(results_dir.glob('6way_comparison_*.json'))

    if not result_files:
        print("No CartPole corrected results found!")
        return

    latest_file = result_files[-1]
    print(f"Analyzing: {latest_file.name}\n")

    # Load results
    with open(latest_file, 'r') as f:
        results = json.load(f)

    # Print configuration
    print("=" * 80)
    print("CONFIGURATION")
    print("=" * 80)
    for key, value in results['config'].items():
        print(f"{key:20s}: {value}")
    print()

    # Analyze each agent
    print("=" * 80)
    print("TRAINING PERFORMANCE")
    print("=" * 80)

    agents = ['dqn', 'qbound_static_dqn', 'qbound_dynamic_dqn', 'ddqn', 'qbound_static_ddqn', 'qbound_dynamic_ddqn']
    agent_names = {
        'dqn': 'Baseline DQN',
        'qbound_static_dqn': 'QBound DQN',
        'qbound_dynamic_dqn': 'QBound Dynamic DQN',
        'ddqn': 'Double DQN',
        'qbound_static_ddqn': 'QBound Double DQN',
        'qbound_dynamic_ddqn': 'QBound Dynamic Double DQN'
    }

    stats_list = []
    for agent in agents:
        if agent in results['training']:
            stats = analyze_agent(agent_names[agent], results['training'][agent])
            stats_list.append(stats)

    # Print statistics table
    print(f"\n{'Agent':<30} {'Mean±Std':<20} {'Max':<10} {'Final 100':<20} {'Success %':<12} {'Final Success %':<15}")
    print("-" * 125)
    for stats in stats_list:
        print(f"{stats['name']:<30} "
              f"{stats['mean_reward']:>7.1f}±{stats['std_reward']:<7.1f} "
              f"{stats['max_reward']:>9.1f} "
              f"{stats['final_100_mean']:>7.1f}±{stats['final_100_std']:<7.1f} "
              f"{stats['success_rate']:>10.1f}% "
              f"{stats['final_100_success_rate']:>13.1f}%")

    # Evaluation performance
    if 'evaluation' in results:
        print("\n" + "=" * 80)
        print("EVALUATION PERFORMANCE (100 episodes)")
        print("=" * 80)

        eval_stats = []
        for agent in agents:
            if agent in results['evaluation']:
                eval_data = results['evaluation'][agent]
                rewards = np.array(eval_data['rewards'])
                eval_stats.append({
                    'name': agent_names[agent],
                    'mean': np.mean(rewards),
                    'std': np.std(rewards),
                    'max': np.max(rewards),
                    'success_rate': np.sum(rewards >= 475) / len(rewards) * 100
                })

        print(f"\n{'Agent':<30} {'Mean±Std':<20} {'Max':<10} {'Success %':<12}")
        print("-" * 80)
        for stats in eval_stats:
            print(f"{stats['name']:<30} "
                  f"{stats['mean']:>7.1f}±{stats['std']:<7.1f} "
                  f"{stats['max']:>9.1f} "
                  f"{stats['success_rate']:>10.1f}%")

    # Generate plots
    print("\n" + "=" * 80)
    print("GENERATING PLOTS")
    print("=" * 80)

    plots_dir = Path('/root/projects/QBound/results/plots')
    plots_dir.mkdir(exist_ok=True)

    # Learning curves (2x3 grid)
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()

    colors = {
        'dqn': 'blue',
        'qbound_static_dqn': 'green',
        'qbound_dynamic_dqn': 'red',
        'ddqn': 'purple',
        'qbound_static_ddqn': 'orange',
        'qbound_dynamic_ddqn': 'brown'
    }

    for idx, agent in enumerate(agents):
        if agent in results['training']:
            rewards = results['training'][agent]['rewards']
            episodes = range(1, len(rewards) + 1)

            ax = axes[idx]
            ax.plot(episodes, rewards, alpha=0.3, color=colors[agent], linewidth=0.5)

            # Moving average
            smoothed = moving_average(rewards, window=20)
            smooth_episodes = range(20, 20 + len(smoothed))
            ax.plot(smooth_episodes, smoothed, color=colors[agent], linewidth=2,
                   label=f'{agent_names[agent]} (20-ep MA)')

            # Success threshold
            ax.axhline(y=475, color='black', linestyle='--', alpha=0.5, label='Success (475)')
            ax.axhline(y=500, color='gray', linestyle=':', alpha=0.5, label='Max (500)')

            ax.set_xlabel('Episode', fontsize=12)
            ax.set_ylabel('Total Reward', fontsize=12)
            ax.set_title(f'{agent_names[agent]}', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)
            ax.set_ylim([0, 550])

    plt.tight_layout()
    plot_file = plots_dir / f'cartpole_corrected_learning_curves_{results["timestamp"]}.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved learning curves: {plot_file.name}")

    # PDF version for paper
    plot_file_pdf = plots_dir / f'cartpole_corrected_learning_curves_{results["timestamp"]}.pdf'
    plt.savefig(plot_file_pdf, bbox_inches='tight')
    print(f"✓ Saved PDF version: {plot_file_pdf.name}")
    plt.close()

    # Comparison plot - all agents together
    fig, ax = plt.subplots(figsize=(14, 8))

    for agent in agents:
        if agent in results['training']:
            rewards = results['training'][agent]['rewards']
            smoothed = moving_average(rewards, window=20)
            smooth_episodes = range(20, 20 + len(smoothed))
            ax.plot(smooth_episodes, smoothed, color=colors[agent], linewidth=2,
                   label=agent_names[agent])

    ax.axhline(y=475, color='black', linestyle='--', alpha=0.5, label='Success Threshold (475)')
    ax.axhline(y=500, color='gray', linestyle=':', alpha=0.5, label='Maximum (500)')
    ax.set_xlabel('Episode', fontsize=14)
    ax.set_ylabel('Total Reward (20-episode MA)', fontsize=14)
    ax.set_title('CartPole-v1 Corrected: 6-Way QBound Comparison', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='lower right')
    ax.set_ylim([0, 550])

    plt.tight_layout()
    plot_file = plots_dir / f'cartpole_corrected_comparison_{results["timestamp"]}.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved comparison plot: {plot_file.name}")

    plot_file_pdf = plots_dir / f'cartpole_corrected_comparison_{results["timestamp"]}.pdf'
    plt.savefig(plot_file_pdf, bbox_inches='tight')
    print(f"✓ Saved PDF version: {plot_file_pdf.name}")
    plt.close()

    # Bar chart comparison
    fig, ax = plt.subplots(figsize=(14, 8))

    x_pos = np.arange(len(stats_list))
    means = [s['final_100_mean'] for s in stats_list]
    stds = [s['final_100_std'] for s in stats_list]
    names = [s['name'] for s in stats_list]
    bar_colors = [colors[agent] for agent in agents if agent in results['training']]

    bars = ax.bar(x_pos, means, yerr=stds, color=bar_colors, alpha=0.7,
                   capsize=10, edgecolor='black', linewidth=1.5)

    ax.axhline(y=475, color='red', linestyle='--', linewidth=2, label='Success Threshold (475)')
    ax.axhline(y=500, color='gray', linestyle=':', linewidth=2, label='Maximum (500)')
    ax.set_xlabel('Agent', fontsize=14)
    ax.set_ylabel('Mean Reward (Final 100 Episodes)', fontsize=14)
    ax.set_title('CartPole-v1 Corrected: Final Performance Comparison', fontsize=16, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, rotation=25, ha='right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(fontsize=12)
    ax.set_ylim([0, 550])

    # Add value labels on bars
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std + 10,
                f'{mean:.1f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plot_file = plots_dir / f'cartpole_corrected_bar_comparison_{results["timestamp"]}.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved bar chart: {plot_file.name}")

    plot_file_pdf = plots_dir / f'cartpole_corrected_bar_comparison_{results["timestamp"]}.pdf'
    plt.savefig(plot_file_pdf, bbox_inches='tight')
    print(f"✓ Saved PDF version: {plot_file_pdf.name}")
    plt.close()

    # Comparison: DQN vs Double DQN for each QBound variant
    print("\n" + "=" * 80)
    print("COMPARISON: DQN VS DOUBLE DQN")
    print("=" * 80)

    comparison = []
    for variant in [('dqn', 'ddqn', 'Baseline'),
                    ('qbound_static_dqn', 'qbound_static_ddqn', 'QBound Static'),
                    ('qbound_dynamic_dqn', 'qbound_dynamic_ddqn', 'QBound Dynamic')]:
        dqn_key, ddqn_key, name = variant
        if dqn_key in results['training'] and ddqn_key in results['training']:
            dqn_rewards = np.array(results['training'][dqn_key]['rewards'])
            ddqn_rewards = np.array(results['training'][ddqn_key]['rewards'])

            comparison.append({
                'type': name,
                'dqn_mean': np.mean(dqn_rewards[-100:]),
                'ddqn_mean': np.mean(ddqn_rewards[-100:]),
                'improvement': np.mean(ddqn_rewards[-100:]) - np.mean(dqn_rewards[-100:])
            })

    print(f"\n{'Variant':<20} {'DQN':<15} {'Double DQN':<15} {'Improvement':<15}")
    print("-" * 70)
    for comp in comparison:
        print(f"{comp['type']:<20} "
              f"{comp['dqn_mean']:>13.1f} "
              f"{comp['ddqn_mean']:>13.1f} "
              f"{comp['improvement']:>13.1f}")

    print("\n✅ Analysis complete!")

if __name__ == '__main__':
    main()
