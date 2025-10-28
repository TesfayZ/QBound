"""
Analysis Script: Dueling DQN vs Standard DQN Comparison
Compares QBound effectiveness across different DQN architectures
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
RESULTS_DIR = Path("/root/projects/QBound/results/lunarlander")
PLOTS_DIR = Path("/root/projects/QBound/results/plots")
PLOTS_DIR.mkdir(exist_ok=True)


def load_latest_results(pattern):
    """Load the most recent results file matching pattern"""
    files = sorted(RESULTS_DIR.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No results found for pattern: {pattern}")

    latest = files[-1]
    print(f"Loading: {latest.name}")

    with open(latest, 'r') as f:
        return json.load(f)


def moving_average(data, window=20):
    """Compute moving average for smoothing"""
    return np.convolve(data, np.ones(window)/window, mode='valid')


def plot_comparison(standard_results, dueling_results, save_path):
    """Plot comparison between standard DQN and Dueling DQN"""

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('QBound Effectiveness: Standard DQN vs Dueling DQN on LunarLander-v3',
                 fontsize=16, fontweight='bold')

    # Method configurations
    methods_standard = {
        'dqn': ('Baseline DQN', 'red', '--'),
        'qbound_static_dqn': ('QBound DQN', 'blue', '-'),
        'ddqn': ('Double DQN', 'orange', '--'),
        'qbound_static_ddqn': ('QBound+Double DQN', 'green', '-')
    }

    methods_dueling = {
        'dueling_dqn': ('Baseline Dueling', 'red', '--'),
        'qbound_dueling_dqn': ('QBound Dueling', 'blue', '-'),
        'double_dueling_dqn': ('Double Dueling', 'orange', '--'),
        'qbound_double_dueling_dqn': ('QBound+Double Dueling', 'green', '-')
    }

    # Plot 1: Standard DQN Learning Curves
    ax = axes[0, 0]
    for method, (label, color, style) in methods_standard.items():
        if method in standard_results['training']:
            rewards = standard_results['training'][method]['rewards']
            smoothed = moving_average(rewards, window=20)
            ax.plot(range(len(smoothed)), smoothed, label=label, color=color,
                   linestyle=style, linewidth=2)

    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Reward (20-ep MA)', fontsize=12)
    ax.set_title('Standard DQN Architecture', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=200, color='black', linestyle=':', linewidth=1, label='Success threshold')

    # Plot 2: Dueling DQN Learning Curves
    ax = axes[0, 1]
    for method, (label, color, style) in methods_dueling.items():
        if method in dueling_results['training']:
            rewards = dueling_results['training'][method]['rewards']
            smoothed = moving_average(rewards, window=20)
            ax.plot(range(len(smoothed)), smoothed, label=label, color=color,
                   linestyle=style, linewidth=2)

    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Reward (20-ep MA)', fontsize=12)
    ax.set_title('Dueling DQN Architecture', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=200, color='black', linestyle=':', linewidth=1, label='Success threshold')

    # Plot 3: Final Performance Comparison
    ax = axes[1, 0]

    methods_all = ['baseline', 'qbound', 'double', 'qbound_double']
    labels = ['Baseline', 'QBound', 'Double', 'QBound+Double']

    # Get final 100 episode statistics
    standard_means = []
    standard_stds = []
    dueling_means = []
    dueling_stds = []

    method_map_standard = ['dqn', 'qbound_static_dqn', 'ddqn', 'qbound_static_ddqn']
    method_map_dueling = ['dueling_dqn', 'qbound_dueling_dqn', 'double_dueling_dqn', 'qbound_double_dueling_dqn']

    for std_m, duel_m in zip(method_map_standard, method_map_dueling):
        if std_m in standard_results['training']:
            data = standard_results['training'][std_m]
            if 'final_100_mean' not in data:
                rewards = data['rewards'][-100:]
                standard_means.append(np.mean(rewards))
                standard_stds.append(np.std(rewards))
            else:
                standard_means.append(data['final_100_mean'])
                standard_stds.append(data['final_100_std'])
        else:
            standard_means.append(0)
            standard_stds.append(0)

        if duel_m in dueling_results['training']:
            data = dueling_results['training'][duel_m]
            if 'final_100_mean' not in data:
                rewards = data['rewards'][-100:]
                dueling_means.append(np.mean(rewards))
                dueling_stds.append(np.std(rewards))
            else:
                dueling_means.append(data['final_100_mean'])
                dueling_stds.append(data['final_100_std'])
        else:
            dueling_means.append(0)
            dueling_stds.append(0)

    x = np.arange(len(labels))
    width = 0.35

    bars1 = ax.bar(x - width/2, standard_means, width, label='Standard DQN',
                   color='steelblue', yerr=standard_stds, capsize=5)
    bars2 = ax.bar(x + width/2, dueling_means, width, label='Dueling DQN',
                   color='coral', yerr=dueling_stds, capsize=5)

    ax.set_xlabel('Method Configuration', fontsize=12)
    ax.set_ylabel('Mean Reward (Final 100 Episodes)', fontsize=12)
    ax.set_title('Final Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=200, color='black', linestyle=':', linewidth=1, label='Success threshold')

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.0f}',
                       ha='center', va='bottom', fontsize=9)

    # Plot 4: QBound Improvement Analysis
    ax = axes[1, 1]

    # Calculate improvements
    standard_improvements = []
    dueling_improvements = []

    if standard_means[0] != 0:  # baseline exists
        for mean in standard_means:
            improvement = ((mean - standard_means[0]) / abs(standard_means[0])) * 100
            standard_improvements.append(improvement)

    if dueling_means[0] != 0:  # baseline exists
        for mean in dueling_means:
            improvement = ((mean - dueling_means[0]) / abs(dueling_means[0])) * 100
            dueling_improvements.append(improvement)

    bars1 = ax.bar(x - width/2, standard_improvements, width, label='Standard DQN',
                   color='steelblue')
    bars2 = ax.bar(x + width/2, dueling_improvements, width, label='Dueling DQN',
                   color='coral')

    ax.set_xlabel('Method Configuration', fontsize=12)
    ax.set_ylabel('Improvement vs Baseline (%)', fontsize=12)
    ax.set_title('QBound Effectiveness Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:+.0f}%',
                   ha='center', va='bottom' if height > 0 else 'top', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Comparison plot saved to: {save_path}")
    plt.close()


def print_summary_table(standard_results, dueling_results):
    """Print formatted summary table"""

    print("\n" + "=" * 100)
    print("ARCHITECTURAL GENERALIZATION ANALYSIS")
    print("=" * 100)

    print("\nSTANDARD DQN RESULTS (Final 100 Episodes)")
    print("-" * 100)
    print(f"{'Method':<35} {'Mean ± Std':<25} {'Max':<10} {'Success Rate':<15}")
    print("-" * 100)

    methods_standard = [
        ('dqn', 'Baseline DQN'),
        ('qbound_static_dqn', 'QBound DQN'),
        ('ddqn', 'Double DQN'),
        ('qbound_static_ddqn', 'QBound+Double DQN')
    ]

    for method, label in methods_standard:
        if method in standard_results['training']:
            data = standard_results['training'][method]

            # Compute final_100 stats if not present
            if 'final_100_mean' not in data:
                rewards = data['rewards'][-100:]
                mean = np.mean(rewards)
                std = np.std(rewards)
                max_reward = np.max(rewards)
            else:
                mean = data['final_100_mean']
                std = data['final_100_std']
                max_reward = data['final_100_max']
                rewards = data['rewards'][-100:]

            success_rate = sum(1 for r in rewards if r > 200) / len(rewards) * 100

            print(f"{label:<35} {mean:>8.2f} ± {std:<8.2f} {max_reward:>8.2f}   {success_rate:>5.1f}%")

    print("\n" + "-" * 100)
    print("DUELING DQN RESULTS (Final 100 Episodes)")
    print("-" * 100)
    print(f"{'Method':<35} {'Mean ± Std':<25} {'Max':<10} {'Success Rate':<15}")
    print("-" * 100)

    methods_dueling = [
        ('dueling_dqn', 'Baseline Dueling DQN'),
        ('qbound_dueling_dqn', 'QBound Dueling DQN'),
        ('double_dueling_dqn', 'Double Dueling DQN'),
        ('qbound_double_dueling_dqn', 'QBound+Double Dueling DQN')
    ]

    for method, label in methods_dueling:
        if method in dueling_results['training']:
            data = dueling_results['training'][method]
            mean = data['final_100_mean']
            std = data['final_100_std']
            max_reward = data['final_100_max']
            rewards = data['rewards'][-100:]
            success_rate = sum(1 for r in rewards if r > 200) / len(rewards) * 100

            print(f"{label:<35} {mean:>8.2f} ± {std:<8.2f} {max_reward:>8.2f}   {success_rate:>5.1f}%")

    # Calculate QBound improvement consistency
    print("\n" + "=" * 100)
    print("QBOUND IMPROVEMENT CONSISTENCY")
    print("=" * 100)

    # Get or compute stats for standard DQN
    std_data = standard_results['training']['dqn']
    if 'final_100_mean' not in std_data:
        std_baseline = np.mean(std_data['rewards'][-100:])
    else:
        std_baseline = std_data['final_100_mean']

    std_qbound_data = standard_results['training']['qbound_static_dqn']
    if 'final_100_mean' not in std_qbound_data:
        std_qbound = np.mean(std_qbound_data['rewards'][-100:])
    else:
        std_qbound = std_qbound_data['final_100_mean']

    std_improvement = ((std_qbound - std_baseline) / abs(std_baseline)) * 100

    duel_baseline = dueling_results['training']['dueling_dqn']['final_100_mean']
    duel_qbound = dueling_results['training']['qbound_dueling_dqn']['final_100_mean']
    duel_improvement = ((duel_qbound - duel_baseline) / abs(duel_baseline)) * 100

    print(f"\nStandard DQN:  Baseline = {std_baseline:.2f} → QBound = {std_qbound:.2f} ({std_improvement:+.1f}%)")
    print(f"Dueling DQN:   Baseline = {duel_baseline:.2f} → QBound = {duel_qbound:.2f} ({duel_improvement:+.1f}%)")

    if std_improvement > 0 and duel_improvement > 0:
        print("\n✓ QBound improves BOTH architectures → Generalization confirmed!")
    elif std_improvement > 0 or duel_improvement > 0:
        print("\n⚠ QBound improves ONE architecture → Partial generalization")
    else:
        print("\n✗ QBound does not improve either → Generalization failed")

    print("=" * 100)


def main():
    print("=" * 100)
    print("ANALYZING DUELING DQN vs STANDARD DQN")
    print("=" * 100)

    # Load results
    print("\nLoading Standard DQN results...")
    standard_results = load_latest_results("4way_comparison_*.json")

    print("\nLoading Dueling DQN results...")
    dueling_results = load_latest_results("dueling_4way_*.json")

    # Print summary
    print_summary_table(standard_results, dueling_results)

    # Generate comparison plot
    plot_path = PLOTS_DIR / "dueling_vs_standard_comparison.pdf"
    print(f"\nGenerating comparison plot...")
    plot_comparison(standard_results, dueling_results, plot_path)

    print("\n" + "=" * 100)
    print("ANALYSIS COMPLETE")
    print("=" * 100)
    print(f"\nResults:")
    print(f"  - Summary table: printed above")
    print(f"  - Comparison plot: {plot_path}")


if __name__ == '__main__':
    main()
