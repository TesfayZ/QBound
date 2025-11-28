#!/usr/bin/env python3
"""
Multi-Seed Analysis for QBound Paper

Analyzes all 5-seed experiment results and generates publication-quality plots
with mean ± std error bars for the paper.

Processes results from organized experiments (Option 1: QBound Variant Coverage):
- Time-step dependent (Static + Dynamic QBound): CartPole, Pendulum
- Sparse/State-dependent (Static QBound only): GridWorld, FrozenLake, MountainCar, Acrobot
"""

import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from collections import defaultdict
import os

# Use non-interactive backend for server
matplotlib.use('Agg')

# Seeds used in experiments
SEEDS = [42, 43, 44, 45, 46]

# Environment configurations
ENVIRONMENTS = {
    # Time-step dependent (Static + Dynamic QBound)
    'cartpole': {
        'name': 'CartPole-v1',
        'experiments': ['dqn_full_qbound', 'dueling_full_qbound'],
        'methods': ['baseline', 'static_qbound', 'dynamic_qbound',
                   'baseline_ddqn', 'static_qbound_ddqn', 'dynamic_qbound_ddqn'],
        'category': 'timestep'
    },
    'pendulum': {
        'name': 'Pendulum-v1',
        'experiments': ['dqn_full_qbound', 'ddpg_full_qbound', 'td3_full_qbound', 'ppo_full_qbound'],
        'category': 'timestep'
    },
    # Sparse/State-dependent (Static QBound only)
    'gridworld': {
        'name': 'GridWorld',
        'experiments': ['dqn_static_qbound'],
        'methods': ['baseline', 'static_qbound', 'baseline_ddqn', 'static_qbound_ddqn'],
        'category': 'sparse'
    },
    'frozenlake': {
        'name': 'FrozenLake-v1',
        'experiments': ['dqn_static_qbound'],
        'methods': ['baseline', 'static_qbound', 'baseline_ddqn', 'static_qbound_ddqn'],
        'category': 'sparse'
    },
    'mountaincar': {
        'name': 'MountainCar-v0',
        'experiments': ['dqn_static_qbound'],
        'methods': ['baseline', 'static_qbound', 'baseline_ddqn', 'static_qbound_ddqn'],
        'category': 'sparse'
    },
    'acrobot': {
        'name': 'Acrobot-v1',
        'experiments': ['dqn_static_qbound'],
        'methods': ['baseline', 'static_qbound', 'baseline_ddqn', 'static_qbound_ddqn'],
        'category': 'sparse'
    }
}


def load_result_files(env_dir, experiment_name, seeds):
    """Load result files for all seeds of an experiment"""
    results = []
    for seed in seeds:
        pattern = f'results/{env_dir}/{experiment_name}_seed{seed}_*.json'
        files = glob.glob(pattern)
        if files:
            # Take the most recent if multiple exist
            files.sort(key=os.path.getmtime, reverse=True)
            with open(files[0], 'r') as f:
                data = json.load(f)
                results.append(data)
        else:
            print(f"  Warning: Missing {env_dir}/{experiment_name} seed {seed}")
    return results


def extract_rewards(results, method_name):
    """Extract reward trajectories for a method across seeds"""
    trajectories = []
    for result in results:
        training = result.get('training', {})
        method_data = training.get(method_name, {})
        rewards = method_data.get('rewards', [])
        if rewards:
            trajectories.append(rewards)
    return trajectories


def compute_statistics(trajectories):
    """Compute mean and std across trajectories"""
    if not trajectories:
        return None, None

    # Find minimum length
    min_len = min(len(t) for t in trajectories)

    # Truncate all to minimum length
    truncated = [t[:min_len] for t in trajectories]

    # Compute statistics
    mean = np.mean(truncated, axis=0)
    std = np.std(truncated, axis=0)

    return mean, std


def smooth_curve(data, window=10):
    """Smooth curve using moving average"""
    if len(data) < window:
        return data
    weights = np.ones(window) / window
    return np.convolve(data, weights, mode='valid')


def plot_learning_curves_comparison(env_name, all_stats, output_path, title=None):
    """Plot learning curves with mean ± std shaded regions"""
    plt.figure(figsize=(12, 6))

    colors = {
        'baseline': '#1f77b4',  # blue
        'static_qbound': '#ff7f0e',  # orange
        'dynamic_qbound': '#2ca02c',  # green
        'baseline_ddqn': '#d62728',  # red
        'static_qbound_ddqn': '#9467bd',  # purple
        'dynamic_qbound_ddqn': '#8c564b',  # brown
    }

    labels = {
        'baseline': 'DQN Baseline',
        'static_qbound': 'DQN + Static QBound',
        'dynamic_qbound': 'DQN + Dynamic QBound',
        'baseline_ddqn': 'Double DQN Baseline',
        'static_qbound_ddqn': 'Double DQN + Static QBound',
        'dynamic_qbound_ddqn': 'Double DQN + Dynamic QBound',
    }

    for method_name, (mean, std) in all_stats.items():
        if mean is None:
            continue

        color = colors.get(method_name, '#000000')
        label = labels.get(method_name, method_name)

        # Smooth curves
        smoothing = min(50, len(mean) // 10)
        mean_smooth = smooth_curve(mean, window=smoothing)
        std_smooth = smooth_curve(std, window=smoothing)

        episodes = np.arange(len(mean_smooth))

        plt.plot(episodes, mean_smooth, label=label, color=color, linewidth=2)
        plt.fill_between(episodes,
                        mean_smooth - std_smooth,
                        mean_smooth + std_smooth,
                        alpha=0.2, color=color)

    plt.xlabel('Episode', fontsize=14)
    plt.ylabel('Average Reward', fontsize=14)
    if title:
        plt.title(title, fontsize=16)
    else:
        plt.title(f'{env_name} Learning Curves (Mean ± Std, 5 seeds)', fontsize=16)
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save PDF and PNG
    plt.savefig(output_path.replace('.png', '.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_path}")


def plot_final_performance_bars(env_name, final_perfs, output_path):
    """Plot bar chart of final performance with error bars"""
    if not final_perfs:
        return

    plt.figure(figsize=(10, 6))

    methods = list(final_perfs.keys())
    means = [final_perfs[m]['mean'] for m in methods]
    stds = [final_perfs[m]['std'] for m in methods]

    # Create labels
    labels_map = {
        'baseline': 'DQN\nBaseline',
        'static_qbound': 'DQN +\nStatic QBound',
        'dynamic_qbound': 'DQN +\nDynamic QBound',
        'baseline_ddqn': 'Double DQN\nBaseline',
        'static_qbound_ddqn': 'Double DQN +\nStatic QBound',
        'dynamic_qbound_ddqn': 'Double DQN +\nDynamic QBound',
    }
    labels = [labels_map.get(m, m) for m in methods]

    x = np.arange(len(methods))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'][:len(methods)]

    bars = plt.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.8, edgecolor='black')

    plt.xlabel('Method', fontsize=14)
    plt.ylabel('Final Performance (Last 100 Episodes)', fontsize=14)
    plt.title(f'{env_name} Final Performance Comparison', fontsize=16)
    plt.xticks(x, labels, fontsize=10)
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()

    # Save
    plt.savefig(output_path.replace('.png', '.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_path}")


def compute_final_performance(trajectories, window=100):
    """Compute mean of last window episodes"""
    if not trajectories:
        return None

    final_perfs = []
    for traj in trajectories:
        if len(traj) >= window:
            final_perfs.append(np.mean(traj[-window:]))
        else:
            final_perfs.append(np.mean(traj))

    return {
        'mean': np.mean(final_perfs),
        'std': np.std(final_perfs),
        'values': final_perfs
    }


def generate_summary_table(all_results):
    """Generate summary table of all experiments"""
    print("\n" + "="*100)
    print("MULTI-SEED SUMMARY TABLE (5 seeds: 42, 43, 44, 45, 46)")
    print("="*100)

    for env_dir, env_config in ENVIRONMENTS.items():
        print(f"\n{env_config['name']} ({env_config['category'].upper()}):")
        print("-" * 100)

        for exp_name in env_config['experiments']:
            key = f"{env_dir}_{exp_name}"
            if key not in all_results:
                print(f"  {exp_name}: NO DATA")
                continue

            result_data = all_results[key]
            print(f"\n  {exp_name}:")

            for method, stats in result_data['final_performance'].items():
                if stats:
                    mean = stats['mean']
                    std = stats['std']
                    print(f"    {method:30s}: {mean:10.2f} ± {std:6.2f}")


def main():
    print("="*80)
    print("MULTI-SEED ANALYSIS FOR QBOUND PAPER")
    print("="*80)

    # Create output directory
    os.makedirs('results/plots', exist_ok=True)
    os.makedirs('LatexDocs/figures', exist_ok=True)

    all_results = {}

    # Process each environment
    for env_dir, env_config in ENVIRONMENTS.items():
        print(f"\n{'='*80}")
        print(f"Processing: {env_config['name']}")
        print(f"{'='*80}")

        for exp_name in env_config['experiments']:
            print(f"\n  Experiment: {exp_name}")

            # Load all seed results
            results = load_result_files(env_dir, exp_name, SEEDS)

            if not results:
                print(f"    No results found, skipping...")
                continue

            print(f"    Loaded {len(results)}/{len(SEEDS)} seeds")

            # Get method names from first result
            methods = list(results[0].get('training', {}).keys())

            # Extract statistics for each method
            all_stats = {}
            final_perfs = {}

            for method in methods:
                trajectories = extract_rewards(results, method)
                if trajectories:
                    mean, std = compute_statistics(trajectories)
                    all_stats[method] = (mean, std)
                    final_perfs[method] = compute_final_performance(trajectories)

            # Store results
            key = f"{env_dir}_{exp_name}"
            all_results[key] = {
                'stats': all_stats,
                'final_performance': final_perfs
            }

            # Plot learning curves
            output_file = f"results/plots/{env_dir}_{exp_name}_multiseed.png"
            plot_learning_curves_comparison(
                env_config['name'],
                all_stats,
                output_file,
                title=f"{env_config['name']} - {exp_name.replace('_', ' ').title()}"
            )

            # Plot final performance bars
            bar_output = f"results/plots/{env_dir}_{exp_name}_final_performance.png"
            plot_final_performance_bars(env_config['name'], final_perfs, bar_output)

    # Generate summary table
    generate_summary_table(all_results)

    # Copy key plots to LatexDocs/figures/ for LaTeX
    print(f"\n{'='*80}")
    print("COPYING KEY PLOTS TO LatexDocs/figures/")
    print(f"{'='*80}")

    import shutil

    # Copy main learning curves
    for env_dir in ENVIRONMENTS.keys():
        for exp_name in ENVIRONMENTS[env_dir]['experiments']:
            src_pdf = f"results/plots/{env_dir}_{exp_name}_multiseed.pdf"
            src_png = f"results/plots/{env_dir}_{exp_name}_multiseed.png"

            if os.path.exists(src_pdf):
                dst_pdf = f"LatexDocs/figures/{env_dir}_{exp_name}_5seed.pdf"
                shutil.copy2(src_pdf, dst_pdf)
                print(f"  Copied: {dst_pdf}")

            if os.path.exists(src_png):
                dst_png = f"LatexDocs/figures/{env_dir}_{exp_name}_5seed.png"
                shutil.copy2(src_png, dst_png)

    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE!")
    print(f"{'='*80}")
    print(f"\nPlots saved to:")
    print(f"  - results/plots/        (all plots)")
    print(f"  - LatexDocs/figures/       (key plots for paper)")


if __name__ == '__main__':
    main()
