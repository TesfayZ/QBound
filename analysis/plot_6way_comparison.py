"""
Visualization for 6-Way Comparison Results

Creates comprehensive plots comparing:
1. Standard DDPG
2. Standard TD3
3. Simple DDPG (no target networks)
4. QBound + Simple DDPG
5. QBound + DDPG
6. QBound + TD3
"""

import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_results(filepath):
    """Load results from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)


def smooth_rewards(rewards, window=10):
    """Apply moving average smoothing"""
    if len(rewards) < window:
        return rewards
    smoothed = []
    for i in range(len(rewards)):
        start = max(0, i - window + 1)
        smoothed.append(np.mean(rewards[start:i+1]))
    return smoothed


def plot_6way_learning_curves(results, output_path):
    """Plot learning curves for all 6 methods"""
    plt.figure(figsize=(16, 10))

    # Define colors and styles
    colors = {
        'ddpg': '#1f77b4',           # blue
        'td3': '#ff7f0e',            # orange
        'simple_ddpg': '#d62728',    # red
        'qbound_simple': '#2ca02c',  # green
        'qbound_ddpg': '#9467bd',    # purple
        'qbound_td3': '#8c564b'      # brown
    }

    labels = {
        'ddpg': '1. Standard DDPG',
        'td3': '2. Standard TD3',
        'simple_ddpg': '3. Simple DDPG (no targets)',
        'qbound_simple': '4. QBound + Simple DDPG',
        'qbound_ddpg': '5. QBound + DDPG',
        'qbound_td3': '6. QBound + TD3'
    }

    linestyles = {
        'ddpg': '-',
        'td3': '-',
        'simple_ddpg': '--',
        'qbound_simple': '-',
        'qbound_ddpg': '-',
        'qbound_td3': '-'
    }

    # Plot raw rewards
    plt.subplot(2, 2, 1)
    for method in ['ddpg', 'td3', 'simple_ddpg', 'qbound_simple', 'qbound_ddpg', 'qbound_td3']:
        rewards = results['training'][method]['rewards']
        plt.plot(rewards, color=colors[method], linestyle=linestyles[method],
                alpha=0.3, linewidth=0.8)

    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Episode Reward', fontsize=12)
    plt.title('Raw Learning Curves', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)

    # Plot smoothed rewards
    plt.subplot(2, 2, 2)
    for method in ['ddpg', 'td3', 'simple_ddpg', 'qbound_simple', 'qbound_ddpg', 'qbound_td3']:
        rewards = results['training'][method]['rewards']
        smoothed = smooth_rewards(rewards, window=20)
        plt.plot(smoothed, color=colors[method], linestyle=linestyles[method],
                label=labels[method], linewidth=2)

    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Smoothed Reward', fontsize=12)
    plt.title('Smoothed Learning Curves (window=20)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10, loc='lower right')
    plt.grid(True, alpha=0.3)

    # Plot cumulative rewards
    plt.subplot(2, 2, 3)
    for method in ['ddpg', 'td3', 'simple_ddpg', 'qbound_simple', 'qbound_ddpg', 'qbound_td3']:
        rewards = results['training'][method]['rewards']
        cumulative = np.cumsum(rewards)
        plt.plot(cumulative, color=colors[method], linestyle=linestyles[method],
                label=labels[method], linewidth=2)

    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Cumulative Reward', fontsize=12)
    plt.title('Cumulative Reward Over Training', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10, loc='lower right')
    plt.grid(True, alpha=0.3)

    # Plot bar chart of final performance
    plt.subplot(2, 2, 4)
    methods = ['ddpg', 'td3', 'simple_ddpg', 'qbound_simple', 'qbound_ddpg', 'qbound_td3']
    method_labels = ['1. DDPG', '2. TD3', '3. Simple\nDDPG', '4. QBound+\nSimple',
                     '5. QBound+\nDDPG', '6. QBound+\nTD3']
    total_rewards = [results['training'][m]['total_reward'] for m in methods]
    bar_colors = [colors[m] for m in methods]

    bars = plt.bar(method_labels, total_rewards, color=bar_colors, alpha=0.7, edgecolor='black')
    plt.ylabel('Total Cumulative Reward', fontsize=12)
    plt.title('Total Training Performance', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Learning curves saved to: {output_path}")
    plt.close()


def plot_comparison_summary(results, output_path):
    """Create a detailed comparison summary plot"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    colors = {
        'ddpg': '#1f77b4',
        'td3': '#ff7f0e',
        'simple_ddpg': '#d62728',
        'qbound_simple': '#2ca02c',
        'qbound_ddpg': '#9467bd',
        'qbound_td3': '#8c564b'
    }

    # 1. Average episode reward
    ax = axes[0, 0]
    methods = ['ddpg', 'td3', 'simple_ddpg', 'qbound_simple', 'qbound_ddpg', 'qbound_td3']
    method_labels = ['DDPG', 'TD3', 'Simple\nDDPG', 'QBound+\nSimple', 'QBound+\nDDPG', 'QBound+\nTD3']
    avg_rewards = [results['training'][m]['mean_reward'] for m in methods]
    bar_colors = [colors[m] for m in methods]

    bars = ax.bar(method_labels, avg_rewards, color=bar_colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Average Episode Reward', fontsize=11)
    ax.set_title('Average Training Performance', fontsize=12, fontweight='bold')
    ax.tick_params(axis='x', labelsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.1f}',
               ha='center', va='bottom', fontsize=8, fontweight='bold')

    # 2. Evaluation performance
    ax = axes[0, 1]
    eval_means = [results['evaluation'][m]['mean'] for m in methods]
    eval_stds = [results['evaluation'][m]['std'] for m in methods]

    bars = ax.bar(method_labels, eval_means, yerr=eval_stds, color=bar_colors,
                  alpha=0.7, edgecolor='black', capsize=5)
    ax.set_ylabel('Evaluation Reward', fontsize=11)
    ax.set_title('Final Evaluation Performance', fontsize=12, fontweight='bold')
    ax.tick_params(axis='x', labelsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.1f}',
               ha='center', va='bottom', fontsize=8, fontweight='bold')

    # 3. Q1: Can QBound replace target networks?
    ax = axes[0, 2]
    comparison_methods = ['simple_ddpg', 'qbound_simple']
    comparison_labels = ['Simple DDPG\n(no targets)', 'QBound +\nSimple DDPG']
    comparison_rewards = [results['training'][m]['total_reward'] for m in comparison_methods]
    comparison_colors = [colors[m] for m in comparison_methods]

    bars = ax.bar(comparison_labels, comparison_rewards, color=comparison_colors,
                  alpha=0.7, edgecolor='black')
    ax.set_ylabel('Total Cumulative Reward', fontsize=11)
    ax.set_title('Q1: Can QBound Replace Target Networks?', fontsize=12, fontweight='bold')
    ax.tick_params(axis='x', labelsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    improvement = ((comparison_rewards[1] - comparison_rewards[0]) /
                   abs(comparison_rewards[0])) * 100
    ax.text(0.5, 0.95, f'Improvement: {improvement:+.1f}%',
           transform=ax.transAxes, ha='center', va='top',
           fontsize=11, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{int(height)}',
               ha='center', va='bottom', fontsize=9, fontweight='bold')

    # 4. Q2: Can QBound enhance DDPG?
    ax = axes[1, 0]
    comparison_methods = ['ddpg', 'qbound_ddpg']
    comparison_labels = ['Standard\nDDPG', 'QBound +\nDDPG']
    comparison_rewards = [results['training'][m]['total_reward'] for m in comparison_methods]
    comparison_colors = [colors[m] for m in comparison_methods]

    bars = ax.bar(comparison_labels, comparison_rewards, color=comparison_colors,
                  alpha=0.7, edgecolor='black')
    ax.set_ylabel('Total Cumulative Reward', fontsize=11)
    ax.set_title('Q2: Can QBound Enhance DDPG?', fontsize=12, fontweight='bold')
    ax.tick_params(axis='x', labelsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    improvement = ((comparison_rewards[1] - comparison_rewards[0]) /
                   abs(comparison_rewards[0])) * 100
    ax.text(0.5, 0.95, f'Improvement: {improvement:+.1f}%',
           transform=ax.transAxes, ha='center', va='top',
           fontsize=11, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{int(height)}',
               ha='center', va='bottom', fontsize=9, fontweight='bold')

    # 5. Q3: Can QBound enhance TD3?
    ax = axes[1, 1]
    comparison_methods = ['td3', 'qbound_td3']
    comparison_labels = ['Standard\nTD3', 'QBound +\nTD3']
    comparison_rewards = [results['training'][m]['total_reward'] for m in comparison_methods]
    comparison_colors = [colors[m] for m in comparison_methods]

    bars = ax.bar(comparison_labels, comparison_rewards, color=comparison_colors,
                  alpha=0.7, edgecolor='black')
    ax.set_ylabel('Total Cumulative Reward', fontsize=11)
    ax.set_title('Q3: Can QBound Enhance TD3?', fontsize=12, fontweight='bold')
    ax.tick_params(axis='x', labelsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    improvement = ((comparison_rewards[1] - comparison_rewards[0]) /
                   abs(comparison_rewards[0])) * 100
    ax.text(0.5, 0.95, f'Improvement: {improvement:+.1f}%',
           transform=ax.transAxes, ha='center', va='top',
           fontsize=11, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{int(height)}',
               ha='center', va='bottom', fontsize=9, fontweight='bold')

    # 6. Overall ranking
    ax = axes[1, 2]
    methods = ['ddpg', 'td3', 'simple_ddpg', 'qbound_simple', 'qbound_ddpg', 'qbound_td3']
    method_labels_short = ['1. DDPG', '2. TD3', '3. Simple', '4. QBound+S', '5. QBound+D', '6. QBound+T']
    total_rewards = [results['training'][m]['total_reward'] for m in methods]

    # Sort by performance
    sorted_pairs = sorted(zip(method_labels_short, total_rewards,
                             [colors[m] for m in methods]),
                         key=lambda x: x[1], reverse=True)
    sorted_labels, sorted_rewards, sorted_colors = zip(*sorted_pairs)

    bars = ax.barh(sorted_labels, sorted_rewards, color=sorted_colors,
                   alpha=0.7, edgecolor='black')
    ax.set_xlabel('Total Cumulative Reward', fontsize=11)
    ax.set_title('Overall Performance Ranking', fontsize=12, fontweight='bold')
    ax.tick_params(axis='y', labelsize=9)
    ax.grid(True, alpha=0.3, axis='x')

    for i, (bar, reward) in enumerate(zip(bars, sorted_rewards)):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2.,
               f' {int(reward)}',
               ha='left', va='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Comparison summary saved to: {output_path}")
    plt.close()


def generate_markdown_report(results, output_path):
    """Generate a markdown summary report"""
    with open(output_path, 'w') as f:
        f.write("# 6-Way Comparison Results\n\n")
        f.write("## Experimental Setup\n\n")
        f.write(f"- **Environment**: {results['config']['env']}\n")
        f.write(f"- **Episodes**: {results['config']['episodes']}\n")
        f.write(f"- **Max Steps**: {results['config']['max_steps']}\n")
        f.write(f"- **Discount Factor (γ)**: {results['config']['gamma']}\n")
        f.write(f"- **QBound Range**: [{results['config']['qbound_min']:.2f}, {results['config']['qbound_max']:.2f}]\n")
        f.write(f"- **Timestamp**: {results['timestamp']}\n\n")

        f.write("## Training Performance\n\n")
        f.write("| Method | Total Reward | Average Reward |\n")
        f.write("|--------|--------------|----------------|\n")

        methods = {
            'ddpg': '1. Standard DDPG',
            'td3': '2. Standard TD3',
            'simple_ddpg': '3. Simple DDPG',
            'qbound_simple': '4. QBound + Simple DDPG',
            'qbound_ddpg': '5. QBound + DDPG',
            'qbound_td3': '6. QBound + TD3'
        }

        for key, label in methods.items():
            total = results['training'][key]['total_reward']
            mean = results['training'][key]['mean_reward']
            f.write(f"| {label} | {total:.0f} | {mean:.2f} |\n")

        f.write("\n## Evaluation Performance\n\n")
        f.write("| Method | Mean ± Std |\n")
        f.write("|--------|------------|\n")

        for key, label in methods.items():
            mean = results['evaluation'][key]['mean']
            std = results['evaluation'][key]['std']
            f.write(f"| {label} | {mean:.2f} ± {std:.2f} |\n")

        f.write("\n## Key Comparisons\n\n")

        # Q1: Can QBound replace target networks?
        simple_total = results['training']['simple_ddpg']['total_reward']
        qbound_simple_total = results['training']['qbound_simple']['total_reward']
        improvement_1 = ((qbound_simple_total - simple_total) / abs(simple_total)) * 100

        f.write("### Q1: Can QBound Replace Target Networks?\n\n")
        f.write(f"- **Comparison**: QBound + Simple DDPG vs Simple DDPG\n")
        f.write(f"- **Improvement**: {improvement_1:+.1f}%\n")
        if improvement_1 > 10:
            f.write("- **Conclusion**: ✅ YES! QBound significantly stabilizes learning without target networks\n\n")
        elif improvement_1 > 5:
            f.write("- **Conclusion**: ✅ YES! QBound helps stabilize learning without target networks\n\n")
        elif improvement_1 > -5:
            f.write("- **Conclusion**: ➖ NEUTRAL: QBound has minimal impact\n\n")
        else:
            f.write("- **Conclusion**: ❌ NO: Target networks still needed\n\n")

        # Q2: Can QBound enhance DDPG?
        ddpg_total = results['training']['ddpg']['total_reward']
        qbound_ddpg_total = results['training']['qbound_ddpg']['total_reward']
        improvement_2 = ((qbound_ddpg_total - ddpg_total) / abs(ddpg_total)) * 100

        f.write("### Q2: Can QBound Enhance Standard DDPG?\n\n")
        f.write(f"- **Comparison**: QBound + DDPG vs Standard DDPG\n")
        f.write(f"- **Improvement**: {improvement_2:+.1f}%\n")
        if improvement_2 > 5:
            f.write("- **Conclusion**: ✅ YES! QBound improves DDPG with target networks\n\n")
        elif improvement_2 > -5:
            f.write("- **Conclusion**: ➖ NEUTRAL: QBound doesn't hurt but doesn't help much\n\n")
        else:
            f.write("- **Conclusion**: ❌ NO: QBound hurts DDPG performance\n\n")

        # Q3: Can QBound enhance TD3?
        td3_total = results['training']['td3']['total_reward']
        qbound_td3_total = results['training']['qbound_td3']['total_reward']
        improvement_3 = ((qbound_td3_total - td3_total) / abs(td3_total)) * 100

        f.write("### Q3: Can QBound Enhance TD3?\n\n")
        f.write(f"- **Comparison**: QBound + TD3 vs Standard TD3\n")
        f.write(f"- **Improvement**: {improvement_3:+.1f}%\n")
        if improvement_3 > 5:
            f.write("- **Conclusion**: ✅ YES! QBound improves even the most advanced method\n\n")
        elif improvement_3 > -5:
            f.write("- **Conclusion**: ➖ NEUTRAL: TD3 already well-stabilized\n\n")
        else:
            f.write("- **Conclusion**: ❌ NO: QBound conflicts with TD3's mechanisms\n\n")

        # Best method
        best_method = max(
            [('1. Standard DDPG', ddpg_total),
             ('2. Standard TD3', td3_total),
             ('3. Simple DDPG', simple_total),
             ('4. QBound + Simple DDPG', qbound_simple_total),
             ('5. QBound + DDPG', qbound_ddpg_total),
             ('6. QBound + TD3', qbound_td3_total)],
            key=lambda x: x[1]
        )

        f.write(f"## Best Overall Method\n\n")
        f.write(f"**{best_method[0]}** (Total Reward: {best_method[1]:.0f})\n")

    print(f"✓ Markdown report saved to: {output_path}")


def main():
    # Find the most recent 6-way comparison results
    results_dir = Path("/root/projects/QBound/results/pendulum")
    result_files = sorted(results_dir.glob("6way_comparison_*.json"))

    if not result_files:
        print("Error: No 6-way comparison results found!")
        print("Please run: python3 experiments/pendulum/train_6way_comparison.py")
        return

    latest_result = result_files[-1]
    print(f"Loading results from: {latest_result}")

    results = load_results(latest_result)
    timestamp = results['timestamp']

    # Create output directory
    plots_dir = Path("/root/projects/QBound/results/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Generate plots
    print("\nGenerating visualizations...")

    learning_curves_path = plots_dir / f"6way_learning_curves_{timestamp}.png"
    plot_6way_learning_curves(results, learning_curves_path)

    comparison_summary_path = plots_dir / f"6way_comparison_summary_{timestamp}.png"
    plot_comparison_summary(results, comparison_summary_path)

    # Generate markdown report
    report_path = results_dir / f"6way_report_{timestamp}.md"
    generate_markdown_report(results, report_path)

    print("\n" + "=" * 80)
    print("Visualization Complete!")
    print("=" * 80)
    print(f"\nGenerated files:")
    print(f"  - {learning_curves_path}")
    print(f"  - {comparison_summary_path}")
    print(f"  - {report_path}")


if __name__ == "__main__":
    main()
