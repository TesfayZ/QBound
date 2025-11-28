"""
Comprehensive Analysis and Visualization for All 6-Way Comparison Results

Processes results from all completed experiments:
- GridWorld (DQN-based)
- FrozenLake (DQN-based)
- CartPole (DQN-based)
- LunarLander (DQN-based)
- Pendulum (DDPG/TD3-based)

Generates plots for each environment and saves to paper directory.
"""

import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime


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


def is_dqn_based(env_name):
    """Check if environment uses DQN-based methods"""
    dqn_envs = ['gridworld', 'frozenlake', 'cartpole', 'lunarlander']
    return any(env in env_name.lower() for env in dqn_envs)


def get_method_info(env_name, results):
    """Get method names, labels, colors based on environment type and actual data"""
    # Get actual methods from results
    actual_methods = list(results['training'].keys())

    if is_dqn_based(env_name):
        # DQN-based methods - handle both 'baseline' and 'baseline_dqn' naming
        methods = actual_methods  # Use actual methods from results

        # Build labels based on actual method names
        labels = {}
        colors = {}

        for method in methods:
            if 'baseline' in method and 'ddqn' not in method:
                labels[method] = '1. Baseline DQN'
                colors[method] = '#1f77b4'  # blue
            elif 'static_qbound' in method and 'ddqn' not in method:
                labels[method] = '2. Static QBound + DQN'
                colors[method] = '#ff7f0e'  # orange
            elif 'dynamic_qbound' in method and 'ddqn' not in method:
                labels[method] = '3. Dynamic QBound + DQN'
                colors[method] = '#2ca02c'  # green
            elif 'baseline_ddqn' in method:
                labels[method] = '4. Baseline DDQN'
                colors[method] = '#d62728'  # red
            elif 'static_qbound_ddqn' in method:
                labels[method] = '5. Static QBound + DDQN'
                colors[method] = '#9467bd'  # purple
            elif 'dynamic_qbound_ddqn' in method:
                labels[method] = '6. Dynamic QBound + DDQN'
                colors[method] = '#8c564b'  # brown
    else:
        # DDPG/TD3-based methods
        methods = actual_methods

        labels = {}
        colors = {}

        for method in methods:
            if method == 'ddpg':
                labels[method] = '1. Standard DDPG'
                colors[method] = '#1f77b4'  # blue
            elif method == 'td3':
                labels[method] = '2. Standard TD3'
                colors[method] = '#ff7f0e'  # orange
            elif method == 'simple_ddpg':
                labels[method] = '3. Simple DDPG (no targets)'
                colors[method] = '#d62728'  # red
            elif method == 'qbound_simple':
                labels[method] = '4. QBound + Simple DDPG'
                colors[method] = '#2ca02c'  # green
            elif method == 'qbound_ddpg':
                labels[method] = '5. QBound + DDPG'
                colors[method] = '#9467bd'  # purple
            elif method == 'qbound_td3':
                labels[method] = '6. QBound + TD3'
                colors[method] = '#8c564b'  # brown

    return methods, labels, colors


def plot_environment_results(results, env_name, output_dir):
    """Create comprehensive plots for a single environment"""
    methods, labels, colors = get_method_info(env_name, results)

    # Create main figure with learning curves and summary
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Environment title
    env_display = env_name.replace('_', ' ').title()
    fig.suptitle(f'{env_display}: 6-Way Comparison Results',
                 fontsize=18, fontweight='bold', y=0.98)

    # 1. Smoothed Learning Curves (large plot)
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    window = 20
    for method in methods:
        if method in results['training']:
            rewards = results['training'][method]['rewards']
            smoothed = smooth_rewards(rewards, window=window)
            ax1.plot(smoothed, color=colors[method],
                    label=labels[method], linewidth=2.5, alpha=0.9)

    ax1.set_xlabel('Episode', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Reward (smoothed)', fontsize=13, fontweight='bold')
    ax1.set_title(f'Learning Curves (window={window})', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10, loc='best', framealpha=0.9)
    ax1.grid(True, alpha=0.3)

    # 2. Total Cumulative Reward
    ax2 = fig.add_subplot(gs[0, 2])
    method_list = [m for m in methods if m in results['training']]
    total_rewards = [results['training'][m]['total_reward'] for m in method_list]
    bar_colors = [colors[m] for m in method_list]
    method_short = [labels[m].split('.')[0] + '.\n' + labels[m].split('. ')[1]
                    for m in method_list]

    bars = ax2.bar(range(len(method_list)), total_rewards,
                   color=bar_colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Total Cumulative Reward', fontsize=11, fontweight='bold')
    ax2.set_title('Total Training Performance', fontsize=12, fontweight='bold')
    ax2.set_xticks(range(len(method_list)))
    ax2.set_xticklabels(method_short, fontsize=8, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')

    for bar, reward in zip(bars, total_rewards):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(reward)}', ha='center', va='bottom',
                fontsize=8, fontweight='bold')

    # 3. Average Episode Reward
    ax3 = fig.add_subplot(gs[1, 2])
    avg_rewards = [results['training'][m]['mean_reward'] for m in method_list]

    bars = ax3.bar(range(len(method_list)), avg_rewards,
                   color=bar_colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax3.set_ylabel('Average Episode Reward', fontsize=11, fontweight='bold')
    ax3.set_title('Average Performance', fontsize=12, fontweight='bold')
    ax3.set_xticks(range(len(method_list)))
    ax3.set_xticklabels(method_short, fontsize=8, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3, axis='y')

    for bar, reward in zip(bars, avg_rewards):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{reward:.1f}', ha='center', va='bottom',
                fontsize=8, fontweight='bold')

    # 4. Evaluation Performance (if available)
    ax4 = fig.add_subplot(gs[2, 0])
    if 'evaluation' in results:
        eval_means = [results['evaluation'][m]['mean'] for m in method_list]
        eval_stds = [results['evaluation'][m]['std'] for m in method_list]

        bars = ax4.bar(range(len(method_list)), eval_means, yerr=eval_stds,
                       color=bar_colors, alpha=0.7, edgecolor='black',
                       linewidth=1.5, capsize=5)
        ax4.set_ylabel('Evaluation Reward', fontsize=11, fontweight='bold')
        ax4.set_title('Final Evaluation (mean ± std)', fontsize=12, fontweight='bold')
        ax4.set_xticks(range(len(method_list)))
        ax4.set_xticklabels(method_short, fontsize=8, rotation=45, ha='right')
        ax4.grid(True, alpha=0.3, axis='y')
    else:
        # No evaluation data - show final 100 episode average instead
        final_100_means = [np.mean(results['training'][m]['rewards'][-100:]) for m in method_list]

        bars = ax4.bar(range(len(method_list)), final_100_means,
                       color=bar_colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        ax4.set_ylabel('Reward (final 100 episodes)', fontsize=11, fontweight='bold')
        ax4.set_title('Final 100 Episodes Average', fontsize=12, fontweight='bold')
        ax4.set_xticks(range(len(method_list)))
        ax4.set_xticklabels(method_short, fontsize=8, rotation=45, ha='right')
        ax4.grid(True, alpha=0.3, axis='y')

    # 5. Performance Ranking
    ax5 = fig.add_subplot(gs[2, 1])
    sorted_pairs = sorted(zip(method_list, total_rewards, bar_colors),
                         key=lambda x: x[1], reverse=True)
    sorted_methods, sorted_rewards, sorted_colors = zip(*sorted_pairs)
    sorted_labels = [labels[m] for m in sorted_methods]

    bars = ax5.barh(range(len(sorted_methods)), sorted_rewards,
                    color=sorted_colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax5.set_xlabel('Total Cumulative Reward', fontsize=11, fontweight='bold')
    ax5.set_title('Performance Ranking', fontsize=12, fontweight='bold')
    ax5.set_yticks(range(len(sorted_methods)))
    ax5.set_yticklabels(sorted_labels, fontsize=9)
    ax5.invert_yaxis()
    ax5.grid(True, alpha=0.3, axis='x')

    for i, (bar, reward) in enumerate(zip(bars, sorted_rewards)):
        width = bar.get_width()
        ax5.text(width, bar.get_y() + bar.get_height()/2.,
                f' {int(reward)}', ha='left', va='center',
                fontsize=9, fontweight='bold')

    # 6. Configuration Info
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.axis('off')

    config_text = f"""Configuration:

Environment: {results['config']['env']}
Episodes: {results['config']['episodes']}
Max Steps: {results['config']['max_steps']}
Discount γ: {results['config']['gamma']}
QBound: [{results['config']['qbound_min']:.1f}, {results['config']['qbound_max']:.1f}]

Best Method:
{sorted_labels[0]}
Total: {int(sorted_rewards[0])}
"""

    ax6.text(0.1, 0.9, config_text, transform=ax6.transAxes,
            fontsize=10, verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    # Save figure
    timestamp = results['timestamp']
    output_path = output_dir / f'{env_name}_6way_comprehensive_{timestamp}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path.name}")
    plt.close()

    return output_path


def generate_summary_report(all_results, output_path):
    """Generate markdown summary report for all environments"""
    with open(output_path, 'w') as f:
        f.write("# QBound: Comprehensive 6-Way Comparison Results\n\n")
        f.write(f"**Report Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")

        for env_name, results in all_results.items():
            env_display = env_name.replace('_', ' ').title()
            f.write(f"## {env_display}\n\n")

            methods, labels, colors = get_method_info(env_name, results)
            method_list = [m for m in methods if m in results['training']]

            # Configuration
            f.write("### Configuration\n\n")
            f.write(f"- **Environment**: {results['config']['env']}\n")
            f.write(f"- **Episodes**: {results['config']['episodes']}\n")
            f.write(f"- **Max Steps**: {results['config']['max_steps']}\n")
            f.write(f"- **Discount Factor (γ)**: {results['config']['gamma']}\n")
            f.write(f"- **QBound Range**: [{results['config']['qbound_min']:.2f}, {results['config']['qbound_max']:.2f}]\n\n")

            # Training Performance
            f.write("### Training Performance\n\n")
            f.write("| Rank | Method | Total Reward | Avg Reward |\n")
            f.write("|------|--------|--------------|------------|\n")

            # Sort by total reward
            perf_data = [(labels[m],
                         results['training'][m]['total_reward'],
                         results['training'][m]['mean_reward'])
                        for m in method_list]
            perf_data.sort(key=lambda x: x[1], reverse=True)

            for rank, (label, total, mean) in enumerate(perf_data, 1):
                medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else ""
                f.write(f"| {rank} {medal} | {label} | {total:.0f} | {mean:.2f} |\n")

            f.write("\n")

            # Evaluation Performance (if available)
            if 'evaluation' in results:
                f.write("### Evaluation Performance\n\n")
                f.write("| Method | Mean ± Std |\n")
                f.write("|--------|------------|\n")

                for m in method_list:
                    mean = results['evaluation'][m]['mean']
                    std = results['evaluation'][m]['std']
                    f.write(f"| {labels[m]} | {mean:.2f} ± {std:.2f} |\n")

                f.write("\n")

            f.write("---\n\n")

        # Overall Summary
        f.write("## Summary Across All Environments\n\n")

        for env_name, results in all_results.items():
            methods, labels, colors = get_method_info(env_name, results)
            method_list = [m for m in methods if m in results['training']]

            # Find best method
            best_method = max(method_list,
                            key=lambda m: results['training'][m]['total_reward'])
            best_reward = results['training'][best_method]['total_reward']

            env_display = env_name.replace('_', ' ').title()
            f.write(f"- **{env_display}**: {labels[best_method]} ({best_reward:.0f})\n")

    print(f"  ✓ Saved: {output_path.name}")


def main():
    """Main analysis function"""
    print("\n" + "=" * 80)
    print("QBound: Comprehensive 6-Way Analysis")
    print("=" * 80 + "\n")

    # Define completed experiments
    experiments = {
        'gridworld': '/root/projects/QBound/results/gridworld/6way_comparison_20251028_093746.json',
        'frozenlake': '/root/projects/QBound/results/frozenlake/6way_comparison_20251028_095909.json',
        'cartpole': '/root/projects/QBound/results/cartpole/6way_comparison_20251028_104649.json',
        'lunarlander': '/root/projects/QBound/results/lunarlander/6way_comparison_20251028_123338.json',
    }

    # Create output directories
    plots_dir = Path("/root/projects/QBound/results/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    paper_figures_dir = Path("/root/projects/QBound/LatexDocs/figures")
    paper_figures_dir.mkdir(parents=True, exist_ok=True)

    # Process each environment
    all_results = {}
    generated_plots = []

    print("Processing experiments:\n")

    for env_name, filepath in experiments.items():
        filepath = Path(filepath)
        if not filepath.exists():
            print(f"⚠️  Skipping {env_name}: file not found")
            continue

        print(f"📊 {env_name.upper()}")
        results = load_results(filepath)
        all_results[env_name] = results

        # Generate plots
        plot_path = plot_environment_results(results, env_name, plots_dir)
        generated_plots.append((env_name, plot_path))

        print()

    # Generate summary report
    print("📝 Generating summary report...")
    report_path = plots_dir / "6way_comprehensive_report.md"
    generate_summary_report(all_results, report_path)

    # Copy plots to paper directory
    print("\n📋 Copying plots to paper directory...")
    for env_name, plot_path in generated_plots:
        dest = paper_figures_dir / f'{env_name}_6way_results.pdf'
        # Convert PNG to PDF for paper (just copy as png for now, can convert later)
        dest_png = paper_figures_dir / f'{env_name}_6way_results.png'
        import shutil
        shutil.copy(plot_path, dest_png)
        print(f"  ✓ Copied: {dest_png.name}")

    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("=" * 80)
    print(f"\n📂 Plots saved to: {plots_dir}")
    print(f"📂 Paper figures: {paper_figures_dir}")
    print(f"📄 Summary report: {report_path}")
    print("\nGenerated plots:")
    for env_name, _ in generated_plots:
        print(f"  • {env_name}_6way_results.png")


if __name__ == "__main__":
    main()
