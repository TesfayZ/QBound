#!/usr/bin/env python3
"""
Comprehensive Paper Analysis for QBound

Analyzes all 5-seed experiment results and generates:
1. Summary statistics tables
2. Publication-quality plots
3. LaTeX-ready tables for the paper
"""

import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from collections import defaultdict
import os
from scipy import stats

# Use non-interactive backend
matplotlib.use('Agg')

# Seeds used in experiments
SEEDS = [42, 43, 44, 45, 46]

# Base results directory
RESULTS_DIR = 'results'


def load_all_results():
    """Load all experiment results organized by environment and method."""
    all_results = {}

    environments = {
        'gridworld': {'prefix': 'dqn_static_qbound', 'name': 'GridWorld'},
        'frozenlake': {'prefix': 'dqn_static_qbound', 'name': 'FrozenLake-v1'},
        'cartpole_dqn': {'prefix': 'dqn_full_qbound', 'dir': 'cartpole', 'name': 'CartPole (DQN)'},
        'cartpole_dueling': {'prefix': 'dueling_full_qbound', 'dir': 'cartpole', 'name': 'CartPole (Dueling)'},
        'pendulum_dqn': {'prefix': 'dqn_full_qbound', 'dir': 'pendulum', 'name': 'Pendulum (DQN)'},
        'pendulum_ddpg': {'prefix': 'ddpg_full_qbound', 'dir': 'pendulum', 'name': 'Pendulum (DDPG)'},
        'pendulum_td3': {'prefix': 'td3_full_qbound', 'dir': 'pendulum', 'name': 'Pendulum (TD3)'},
        'pendulum_ppo': {'prefix': 'ppo_full_qbound', 'dir': 'pendulum', 'name': 'Pendulum (PPO)'},
        'mountaincar': {'prefix': 'dqn_static_qbound', 'name': 'MountainCar'},
        'acrobot': {'prefix': 'dqn_static_qbound', 'name': 'Acrobot'},
    }

    for env_key, config in environments.items():
        env_dir = config.get('dir', env_key)
        prefix = config['prefix']
        env_name = config['name']

        results_by_seed = []
        for seed in SEEDS:
            pattern = f"{RESULTS_DIR}/{env_dir}/{prefix}_seed{seed}_*.json"
            files = glob.glob(pattern)
            if files:
                # Get most recent file
                files.sort(key=os.path.getmtime, reverse=True)
                with open(files[0], 'r') as f:
                    results_by_seed.append(json.load(f))

        if results_by_seed:
            all_results[env_key] = {
                'name': env_name,
                'data': results_by_seed,
                'n_seeds': len(results_by_seed)
            }
            print(f"Loaded {len(results_by_seed)} seeds for {env_name}")
        else:
            print(f"WARNING: No results found for {env_name}")

    return all_results


def compute_final_performance(rewards, window=100):
    """Compute mean of last N episodes."""
    if len(rewards) >= window:
        return np.mean(rewards[-window:])
    return np.mean(rewards)


def analyze_environment(env_data):
    """Analyze results for a single environment across all seeds."""
    results = env_data['data']
    analysis = {}

    # Get all methods from first result
    if not results:
        return analysis

    methods = list(results[0].get('training', {}).keys())

    for method in methods:
        rewards_per_seed = []
        final_perfs = []

        for result in results:
            method_data = result.get('training', {}).get(method, {})
            rewards = method_data.get('rewards', [])
            if rewards:
                rewards_per_seed.append(rewards)
                final_perfs.append(compute_final_performance(rewards))

        if final_perfs:
            analysis[method] = {
                'final_mean': np.mean(final_perfs),
                'final_std': np.std(final_perfs),
                'final_values': final_perfs,
                'n_seeds': len(final_perfs),
                'rewards_per_seed': rewards_per_seed
            }

    return analysis


def compute_improvement(baseline_mean, method_mean, is_negative_reward=False):
    """Compute improvement percentage."""
    if baseline_mean == 0:
        return 0

    if is_negative_reward:
        # For negative rewards, less negative is better
        # Improvement = (baseline - method) / |baseline| * 100
        return (baseline_mean - method_mean) / abs(baseline_mean) * 100
    else:
        # For positive rewards, more is better
        return (method_mean - baseline_mean) / abs(baseline_mean) * 100


def print_summary_table(all_analysis):
    """Print comprehensive summary table."""
    print("\n" + "=" * 120)
    print("COMPREHENSIVE QBOUND RESULTS SUMMARY (5 Seeds: 42, 43, 44, 45, 46)")
    print("=" * 120)

    # Group by category
    positive_envs = ['cartpole_dqn', 'cartpole_dueling']
    negative_envs = ['pendulum_dqn', 'pendulum_ddpg', 'pendulum_td3', 'pendulum_ppo']
    sparse_envs = ['gridworld', 'frozenlake', 'mountaincar', 'acrobot']

    env_names = {
        'cartpole_dqn': 'CartPole (DQN)',
        'cartpole_dueling': 'CartPole (Dueling DQN)',
        'pendulum_dqn': 'Pendulum (DQN)',
        'pendulum_ddpg': 'Pendulum (DDPG)',
        'pendulum_td3': 'Pendulum (TD3)',
        'pendulum_ppo': 'Pendulum (PPO)',
        'gridworld': 'GridWorld',
        'frozenlake': 'FrozenLake',
        'mountaincar': 'MountainCar',
        'acrobot': 'Acrobot'
    }

    def print_env_results(env_key, is_negative=False):
        if env_key not in all_analysis:
            return

        env_analysis = all_analysis[env_key]
        env_name = env_names.get(env_key, env_key)
        print(f"\n{env_name}:")
        print("-" * 100)

        # Find baseline
        baseline_key = None
        for key in env_analysis.keys():
            if 'baseline' in key.lower() and 'ddqn' not in key.lower():
                baseline_key = key
                break

        baseline_mean = env_analysis[baseline_key]['final_mean'] if baseline_key else 0

        for method, stats in env_analysis.items():
            improvement = compute_improvement(baseline_mean, stats['final_mean'], is_negative)
            imp_str = f"+{improvement:.1f}%" if improvement > 0 else f"{improvement:.1f}%"
            if method == baseline_key:
                imp_str = "(baseline)"

            print(f"  {method:40s}: {stats['final_mean']:10.2f} +/- {stats['final_std']:7.2f}  [{imp_str}] (n={stats['n_seeds']})")

    print("\n" + "=" * 80)
    print("POSITIVE DENSE REWARDS (CartPole - Expected: QBound helps)")
    print("=" * 80)
    for env_key in positive_envs:
        if env_key in all_analysis:
            print_env_results(env_key, is_negative=False)

    print("\n" + "=" * 80)
    print("NEGATIVE DENSE REWARDS (Pendulum - Expected: Architectural QBound may help)")
    print("=" * 80)
    for env_key in negative_envs:
        if env_key in all_analysis:
            print_env_results(env_key, is_negative=True)

    print("\n" + "=" * 80)
    print("SPARSE/STATE-DEPENDENT REWARDS (GridWorld, FrozenLake, MountainCar, Acrobot)")
    print("=" * 80)
    for env_key in sparse_envs:
        if env_key in all_analysis:
            print_env_results(env_key, is_negative=env_key in ['mountaincar', 'acrobot'])


def generate_latex_table(all_analysis):
    """Generate LaTeX table for paper."""

    latex = r"""
\begin{table}[htbp]
\centering
\caption{QBound Performance Summary Across All Environments (5 Seeds)}
\label{tab:comprehensive-results}
\small
\begin{tabular}{llrrr}
\toprule
\textbf{Environment} & \textbf{Method} & \textbf{Final Perf.} & \textbf{Std.} & \textbf{Improvement} \\
\midrule
"""

    # Process each environment
    env_order = [
        ('cartpole_dqn', False),
        ('cartpole_dueling', False),
        ('pendulum_dqn', True),
        ('pendulum_ddpg', True),
        ('pendulum_td3', True),
        ('pendulum_ppo', True),
        ('gridworld', False),
        ('frozenlake', False),
        ('mountaincar', True),
        ('acrobot', True),
    ]

    for env_key, is_negative in env_order:
        if env_key not in all_analysis:
            continue

        env_analysis = all_analysis[env_key]
        env_name = env_key.replace('_', ' ').title()

        # Find baseline
        baseline_key = None
        for key in env_analysis.keys():
            if 'baseline' in key.lower() and 'ddqn' not in key.lower():
                baseline_key = key
                break

        baseline_mean = env_analysis[baseline_key]['final_mean'] if baseline_key else 0

        first = True
        for method, stats in sorted(env_analysis.items()):
            improvement = compute_improvement(baseline_mean, stats['final_mean'], is_negative)
            imp_str = f"+{improvement:.1f}\\%" if improvement > 0 else f"{improvement:.1f}\\%"
            if 'baseline' in method.lower() and 'ddqn' not in method.lower():
                imp_str = "---"

            method_clean = method.replace('_', ' ').replace('architectural qbound', 'Arch. QBound')

            if first:
                latex += f"\\multirow{{{len(env_analysis)}}}{{*}}{{{env_name}}} & {method_clean} & {stats['final_mean']:.2f} & {stats['final_std']:.2f} & {imp_str} \\\\\n"
                first = False
            else:
                latex += f" & {method_clean} & {stats['final_mean']:.2f} & {stats['final_std']:.2f} & {imp_str} \\\\\n"

        latex += "\\midrule\n"

    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""

    return latex


def plot_comparison_bars(all_analysis, output_dir):
    """Generate bar charts comparing methods."""

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs('QBound/figures', exist_ok=True)

    # CartPole comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, (env_key, title) in enumerate([
        ('cartpole_dqn', 'CartPole - DQN Variants'),
        ('cartpole_dueling', 'CartPole - Dueling DQN')
    ]):
        if env_key not in all_analysis:
            continue

        ax = axes[idx]
        env_analysis = all_analysis[env_key]

        methods = list(env_analysis.keys())
        means = [env_analysis[m]['final_mean'] for m in methods]
        stds = [env_analysis[m]['final_std'] for m in methods]

        colors = []
        for m in methods:
            if 'baseline' in m.lower():
                colors.append('#1f77b4')
            elif 'static' in m.lower():
                colors.append('#ff7f0e')
            elif 'dynamic' in m.lower():
                colors.append('#2ca02c')
            else:
                colors.append('#d62728')

        x = np.arange(len(methods))
        bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.8, edgecolor='black')

        ax.set_ylabel('Final Performance (Last 100 Episodes)', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.set_xticks(x)
        labels = [m.replace('_', '\n') for m in methods]
        ax.set_xticklabels(labels, fontsize=9, rotation=0)
        ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/cartpole_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/cartpole_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('QBound/figures/cartpole_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/cartpole_comparison.pdf")

    # Pendulum comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    pendulum_envs = ['pendulum_dqn', 'pendulum_ddpg', 'pendulum_td3', 'pendulum_ppo']
    titles = ['Pendulum - DQN', 'Pendulum - DDPG', 'Pendulum - TD3', 'Pendulum - PPO']

    for idx, (env_key, title) in enumerate(zip(pendulum_envs, titles)):
        if env_key not in all_analysis:
            continue

        ax = axes[idx]
        env_analysis = all_analysis[env_key]

        methods = list(env_analysis.keys())
        means = [env_analysis[m]['final_mean'] for m in methods]
        stds = [env_analysis[m]['final_std'] for m in methods]

        colors = []
        for m in methods:
            if 'baseline' in m.lower():
                colors.append('#1f77b4')
            elif 'architectural' in m.lower():
                colors.append('#2ca02c')
            else:
                colors.append('#ff7f0e')

        x = np.arange(len(methods))
        bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.8, edgecolor='black')

        ax.set_ylabel('Final Performance', fontsize=11)
        ax.set_title(title, fontsize=13)
        ax.set_xticks(x)
        labels = [m.replace('_', '\n').replace('architectural', 'arch.') for m in methods]
        ax.set_xticklabels(labels, fontsize=8, rotation=0)
        ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/pendulum_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/pendulum_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('QBound/figures/pendulum_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/pendulum_comparison.pdf")

    # Sparse environments comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    sparse_envs = ['gridworld', 'frozenlake', 'mountaincar', 'acrobot']
    titles = ['GridWorld', 'FrozenLake', 'MountainCar', 'Acrobot']

    for idx, (env_key, title) in enumerate(zip(sparse_envs, titles)):
        if env_key not in all_analysis:
            continue

        ax = axes[idx]
        env_analysis = all_analysis[env_key]

        methods = list(env_analysis.keys())
        means = [env_analysis[m]['final_mean'] for m in methods]
        stds = [env_analysis[m]['final_std'] for m in methods]

        colors = []
        for m in methods:
            if 'baseline' in m.lower() and 'ddqn' not in m.lower():
                colors.append('#1f77b4')
            elif 'static_qbound' in m.lower() and 'ddqn' not in m.lower():
                colors.append('#ff7f0e')
            elif 'baseline_ddqn' in m.lower():
                colors.append('#d62728')
            else:
                colors.append('#9467bd')

        x = np.arange(len(methods))
        bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.8, edgecolor='black')

        ax.set_ylabel('Final Performance', fontsize=11)
        ax.set_title(title, fontsize=13)
        ax.set_xticks(x)
        labels = [m.replace('_', '\n') for m in methods]
        ax.set_xticklabels(labels, fontsize=9, rotation=0)
        ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/sparse_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/sparse_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('QBound/figures/sparse_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/sparse_comparison.pdf")


def plot_learning_curves(all_analysis, output_dir):
    """Generate learning curve plots with shaded std regions."""

    os.makedirs(output_dir, exist_ok=True)

    def smooth_curve(data, window=20):
        if len(data) < window:
            return data
        weights = np.ones(window) / window
        return np.convolve(data, weights, mode='valid')

    # CartPole DQN learning curves
    if 'cartpole_dqn' in all_analysis:
        fig, ax = plt.subplots(figsize=(10, 6))

        env_analysis = all_analysis['cartpole_dqn']
        colors = {'baseline': '#1f77b4', 'static_qbound': '#ff7f0e', 'dynamic_qbound': '#2ca02c',
                  'baseline_ddqn': '#d62728', 'static_qbound_ddqn': '#9467bd', 'dynamic_qbound_ddqn': '#8c564b'}

        for method, stats in env_analysis.items():
            if 'rewards_per_seed' not in stats:
                continue

            rewards = stats['rewards_per_seed']
            min_len = min(len(r) for r in rewards)
            aligned = np.array([r[:min_len] for r in rewards])

            mean = np.mean(aligned, axis=0)
            std = np.std(aligned, axis=0)

            window = min(50, len(mean) // 10)
            mean_smooth = smooth_curve(mean, window)
            std_smooth = smooth_curve(std, window)
            episodes = np.arange(len(mean_smooth))

            color = colors.get(method, '#000000')
            label = method.replace('_', ' ').title()

            ax.plot(episodes, mean_smooth, label=label, color=color, linewidth=2)
            ax.fill_between(episodes, mean_smooth - std_smooth, mean_smooth + std_smooth,
                           alpha=0.2, color=color)

        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Reward', fontsize=12)
        ax.set_title('CartPole DQN Learning Curves (5 Seeds)', fontsize=14)
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{output_dir}/cartpole_dqn_learning_curves.pdf', dpi=300, bbox_inches='tight')
        plt.savefig('QBound/figures/cartpole_dqn_learning_curves.pdf', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_dir}/cartpole_dqn_learning_curves.pdf")

    # Pendulum DDPG learning curves
    if 'pendulum_ddpg' in all_analysis:
        fig, ax = plt.subplots(figsize=(10, 6))

        env_analysis = all_analysis['pendulum_ddpg']
        colors = {'baseline': '#1f77b4', 'architectural_qbound_ddpg': '#2ca02c'}

        for method, stats in env_analysis.items():
            if 'rewards_per_seed' not in stats:
                continue

            rewards = stats['rewards_per_seed']
            min_len = min(len(r) for r in rewards)
            aligned = np.array([r[:min_len] for r in rewards])

            mean = np.mean(aligned, axis=0)
            std = np.std(aligned, axis=0)

            window = min(30, len(mean) // 10)
            mean_smooth = smooth_curve(mean, window)
            std_smooth = smooth_curve(std, window)
            episodes = np.arange(len(mean_smooth))

            color = colors.get(method, '#ff7f0e')
            label = method.replace('_', ' ').replace('architectural qbound ddpg', 'Arch. QBound')
            if method == 'baseline':
                label = 'Baseline'

            ax.plot(episodes, mean_smooth, label=label, color=color, linewidth=2)
            ax.fill_between(episodes, mean_smooth - std_smooth, mean_smooth + std_smooth,
                           alpha=0.2, color=color)

        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Reward', fontsize=12)
        ax.set_title('Pendulum DDPG Learning Curves (5 Seeds)', fontsize=14)
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{output_dir}/pendulum_ddpg_learning_curves.pdf', dpi=300, bbox_inches='tight')
        plt.savefig('QBound/figures/pendulum_ddpg_learning_curves.pdf', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_dir}/pendulum_ddpg_learning_curves.pdf")


def statistical_tests(all_analysis):
    """Perform statistical significance tests."""
    print("\n" + "=" * 80)
    print("STATISTICAL SIGNIFICANCE TESTS (Welch's t-test)")
    print("=" * 80)

    for env_key, env_analysis in all_analysis.items():
        print(f"\n{env_key}:")

        # Find baseline
        baseline_key = None
        for key in env_analysis.keys():
            if 'baseline' in key.lower() and 'ddqn' not in key.lower():
                baseline_key = key
                break

        if not baseline_key:
            continue

        baseline_values = env_analysis[baseline_key]['final_values']

        for method, stats in env_analysis.items():
            if method == baseline_key:
                continue

            method_values = stats['final_values']

            # Welch's t-test (unequal variance)
            t_stat, p_value = stats.ttest_ind(baseline_values, method_values, equal_var=False)

            significance = ""
            if p_value < 0.001:
                significance = "***"
            elif p_value < 0.01:
                significance = "**"
            elif p_value < 0.05:
                significance = "*"

            print(f"  {baseline_key} vs {method}: t={t_stat:.3f}, p={p_value:.4f} {significance}")


def save_analysis_json(all_analysis, output_file):
    """Save analysis results as JSON for paper generation."""

    output = {}
    for env_key, env_analysis in all_analysis.items():
        output[env_key] = {}
        for method, stats in env_analysis.items():
            output[env_key][method] = {
                'final_mean': float(stats['final_mean']),
                'final_std': float(stats['final_std']),
                'n_seeds': stats['n_seeds']
            }

    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved analysis JSON to: {output_file}")


def main():
    print("=" * 80)
    print("COMPREHENSIVE QBOUND PAPER ANALYSIS")
    print("=" * 80)

    # Load all results
    all_results = load_all_results()

    if not all_results:
        print("ERROR: No results found!")
        return

    # Analyze each environment
    all_analysis = {}
    for env_key, env_data in all_results.items():
        all_analysis[env_key] = analyze_environment(env_data)

    # Print summary
    print_summary_table(all_analysis)

    # Generate LaTeX table
    latex_table = generate_latex_table(all_analysis)
    print("\n" + "=" * 80)
    print("LATEX TABLE FOR PAPER")
    print("=" * 80)
    print(latex_table)

    # Save LaTeX table
    with open('QBound/results_table.tex', 'w') as f:
        f.write(latex_table)
    print("Saved LaTeX table to: QBound/results_table.tex")

    # Generate plots
    plot_comparison_bars(all_analysis, 'results/plots')
    plot_learning_curves(all_analysis, 'results/plots')

    # Statistical tests
    try:
        statistical_tests(all_analysis)
    except Exception as e:
        print(f"Statistical tests skipped: {e}")

    # Save JSON
    save_analysis_json(all_analysis, 'results/analysis_summary.json')

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)


if __name__ == '__main__':
    main()
