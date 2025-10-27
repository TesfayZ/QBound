#!/usr/bin/env python3
"""
Unified analysis comparing QBound performance across all environments
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

def load_latest_results(env_name, result_dir):
    """Load the latest results for a given environment"""
    results_path = Path(result_dir)
    if not results_path.exists():
        return None

    result_files = sorted(results_path.glob('*.json'))
    if not result_files:
        return None

    with open(result_files[-1], 'r') as f:
        return json.load(f)

def analyze_environment(env_name, results, normalize=False):
    """Extract performance metrics from results"""
    if not results:
        return None

    metrics = {}

    # Get all agent types
    training_data = results.get('training', {})

    for agent_key, agent_data in training_data.items():
        rewards = np.array(agent_data['rewards'])
        final_100 = rewards[-100:]

        metrics[agent_key] = {
            'mean': np.mean(final_100),
            'std': np.std(final_100),
            'max': np.max(rewards),
            'final_max': np.max(final_100)
        }

        # Normalize if requested (for cross-environment comparison)
        if normalize and 'config' in results:
            # Normalize by theoretical max
            if 'qbound_max' in results['config']:
                max_possible = results['config']['qbound_max']
                metrics[agent_key]['normalized_mean'] = metrics[agent_key]['mean'] / max_possible * 100

    return metrics

def main():
    print("=" * 100)
    print("UNIFIED QBOUND ANALYSIS ACROSS ALL ENVIRONMENTS")
    print("=" * 100)

    # Define environments and their result directories
    environments = {
        'GridWorld': '/root/projects/QBound/results/gridworld',
        'FrozenLake': '/root/projects/QBound/results/frozenlake',
        'CartPole': '/root/projects/QBound/results/cartpole',
        'MountainCar': '/root/projects/QBound/results/mountaincar',
        'Acrobot': '/root/projects/QBound/results/acrobot',
        'LunarLander': '/root/projects/QBound/results/lunarlander',
        'CartPole-Corrected': '/root/projects/QBound/results/cartpole_corrected'
    }

    # Load all results
    all_results = {}
    for env_name, result_dir in environments.items():
        results = load_latest_results(env_name, result_dir)
        if results:
            all_results[env_name] = results
            print(f"✓ Loaded {env_name}")
        else:
            print(f"✗ No results found for {env_name}")

    if not all_results:
        print("No results found!")
        return

    print()

    # Analyze each environment
    print("=" * 100)
    print("ENVIRONMENT-BY-ENVIRONMENT COMPARISON")
    print("=" * 100)

    all_metrics = {}
    for env_name, results in all_results.items():
        metrics = analyze_environment(env_name, results)
        if metrics:
            all_metrics[env_name] = metrics

            print(f"\n{env_name}:")
            print(f"{'Agent':<35} {'Final 100 Mean±Std':<25} {'Max Reward':<15}")
            print("-" * 80)

            for agent_key, agent_metrics in metrics.items():
                # Clean up agent name for display
                agent_name = agent_key.replace('_', ' ').title()
                print(f"{agent_name:<35} "
                      f"{agent_metrics['mean']:>10.1f}±{agent_metrics['std']:<10.1f} "
                      f"{agent_metrics['max']:>13.1f}")

    # Cross-environment comparison: DQN vs QBound DQN
    print("\n" + "=" * 100)
    print("CROSS-ENVIRONMENT: DQN VS QBOUND DQN")
    print("=" * 100)

    comparison = []
    for env_name, metrics in all_metrics.items():
        if 'dqn' in metrics and 'qbound_static_dqn' in metrics:
            dqn_mean = metrics['dqn']['mean']
            qbound_mean = metrics['qbound_static_dqn']['mean']
            improvement = qbound_mean - dqn_mean
            improvement_pct = (improvement / abs(dqn_mean)) * 100 if dqn_mean != 0 else 0

            comparison.append({
                'env': env_name,
                'dqn': dqn_mean,
                'qbound': qbound_mean,
                'improvement': improvement,
                'improvement_pct': improvement_pct
            })

    print(f"\n{'Environment':<25} {'DQN':<15} {'QBound DQN':<15} {'Improvement':<15} {'% Change':<15}")
    print("-" * 90)
    for comp in sorted(comparison, key=lambda x: x['improvement_pct'], reverse=True):
        print(f"{comp['env']:<25} "
              f"{comp['dqn']:>13.1f} "
              f"{comp['qbound']:>13.1f} "
              f"{comp['improvement']:>13.1f} "
              f"{comp['improvement_pct']:>13.1f}%")

    # Cross-environment comparison: DQN vs Double DQN
    print("\n" + "=" * 100)
    print("CROSS-ENVIRONMENT: DQN VS DOUBLE DQN")
    print("=" * 100)

    ddqn_comparison = []
    for env_name, metrics in all_metrics.items():
        if 'dqn' in metrics and 'ddqn' in metrics:
            dqn_mean = metrics['dqn']['mean']
            ddqn_mean = metrics['ddqn']['mean']
            improvement = ddqn_mean - dqn_mean
            improvement_pct = (improvement / abs(dqn_mean)) * 100 if dqn_mean != 0 else 0

            ddqn_comparison.append({
                'env': env_name,
                'dqn': dqn_mean,
                'ddqn': ddqn_mean,
                'improvement': improvement,
                'improvement_pct': improvement_pct
            })

    print(f"\n{'Environment':<25} {'DQN':<15} {'Double DQN':<15} {'Improvement':<15} {'% Change':<15}")
    print("-" * 90)
    for comp in sorted(ddqn_comparison, key=lambda x: x['improvement_pct'], reverse=True):
        print(f"{comp['env']:<25} "
              f"{comp['dqn']:>13.1f} "
              f"{comp['ddqn']:>13.1f} "
              f"{comp['improvement']:>13.1f} "
              f"{comp['improvement_pct']:>13.1f}%")

    # Generate summary plots
    print("\n" + "=" * 100)
    print("GENERATING SUMMARY PLOTS")
    print("=" * 100)

    plots_dir = Path('/root/projects/QBound/results/plots')
    plots_dir.mkdir(exist_ok=True)

    # Plot 1: Improvement heatmap
    if comparison:
        fig, ax = plt.subplots(figsize=(12, 8))

        envs = [c['env'] for c in comparison]
        improvements = [c['improvement_pct'] for c in comparison]
        colors_list = ['green' if x > 0 else 'red' for x in improvements]

        bars = ax.barh(envs, improvements, color=colors_list, alpha=0.7, edgecolor='black')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=2)
        ax.set_xlabel('Improvement (%)', fontsize=14)
        ax.set_title('QBound DQN vs Baseline DQN: Improvement Across Environments',
                     fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, improvements)):
            x_pos = val + (5 if val > 0 else -5)
            ha = 'left' if val > 0 else 'right'
            ax.text(x_pos, bar.get_y() + bar.get_height()/2, f'{val:.1f}%',
                   ha=ha, va='center', fontsize=10, fontweight='bold')

        plt.tight_layout()
        plot_file = plots_dir / 'unified_qbound_improvement.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved improvement plot: {plot_file.name}")

        plot_file_pdf = plots_dir / 'unified_qbound_improvement.pdf'
        plt.savefig(plot_file_pdf, bbox_inches='tight')
        print(f"✓ Saved PDF version: {plot_file_pdf.name}")
        plt.close()

    # Plot 2: Grouped bar chart for select environments
    if len(all_metrics) >= 3:
        # Select most interesting environments
        selected_envs = ['LunarLander', 'CartPole-Corrected', 'Acrobot', 'MountainCar']
        selected_envs = [e for e in selected_envs if e in all_metrics]

        if selected_envs:
            fig, ax = plt.subplots(figsize=(14, 8))

            agents_to_compare = ['dqn', 'qbound_static_dqn', 'ddqn', 'qbound_static_ddqn']
            agent_labels = ['DQN', 'QBound DQN', 'Double DQN', 'QBound Double DQN']

            x = np.arange(len(selected_envs))
            width = 0.2

            for i, (agent_key, label) in enumerate(zip(agents_to_compare, agent_labels)):
                means = []
                for env in selected_envs:
                    if agent_key in all_metrics[env]:
                        means.append(all_metrics[env][agent_key]['mean'])
                    else:
                        means.append(0)

                ax.bar(x + i*width, means, width, label=label, alpha=0.8)

            ax.set_xlabel('Environment', fontsize=14)
            ax.set_ylabel('Mean Reward (Final 100 Episodes)', fontsize=14)
            ax.set_title('Performance Comparison Across Environments', fontsize=16, fontweight='bold')
            ax.set_xticks(x + width * 1.5)
            ax.set_xticklabels(selected_envs, rotation=20, ha='right')
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3, axis='y')

            plt.tight_layout()
            plot_file = plots_dir / 'unified_grouped_comparison.png'
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"✓ Saved grouped comparison: {plot_file.name}")

            plot_file_pdf = plots_dir / 'unified_grouped_comparison.pdf'
            plt.savefig(plot_file_pdf, bbox_inches='tight')
            print(f"✓ Saved PDF version: {plot_file_pdf.name}")
            plt.close()

    # Summary statistics
    print("\n" + "=" * 100)
    print("SUMMARY STATISTICS")
    print("=" * 100)

    if comparison:
        improvements = [c['improvement_pct'] for c in comparison]
        print(f"\nQBound DQN vs Baseline DQN:")
        print(f"  Environments improved: {sum(1 for x in improvements if x > 0)}/{len(improvements)}")
        print(f"  Average improvement: {np.mean(improvements):.1f}%")
        print(f"  Best improvement: {max(improvements):.1f}% ({comparison[np.argmax(improvements)]['env']})")
        print(f"  Worst change: {min(improvements):.1f}% ({comparison[np.argmin(improvements)]['env']})")

    if ddqn_comparison:
        ddqn_improvements = [c['improvement_pct'] for c in ddqn_comparison]
        print(f"\nDouble DQN vs Baseline DQN:")
        print(f"  Environments improved: {sum(1 for x in ddqn_improvements if x > 0)}/{len(ddqn_improvements)}")
        print(f"  Average improvement: {np.mean(ddqn_improvements):.1f}%")
        print(f"  Best improvement: {max(ddqn_improvements):.1f}% ({ddqn_comparison[np.argmax(ddqn_improvements)]['env']})")
        print(f"  Worst change: {min(ddqn_improvements):.1f}% ({ddqn_comparison[np.argmin(ddqn_improvements)]['env']})")

    print("\n✅ Unified analysis complete!")

if __name__ == '__main__':
    main()
