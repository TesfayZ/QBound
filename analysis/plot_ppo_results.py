"""
Generate plots for PPO + QBound results

Creates:
1. Learning curves for each environment
2. Bar chart comparison across environments
3. 2x2 grid showing action space × reward structure
"""

import json
import glob
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime


def load_latest_results(results_dir="/root/projects/QBound/results/ppo"):
    """Load the latest results for each environment."""
    results = {}

    environments = {
        'cartpole_pilot': 'CartPole',
        'lunarlander_pilot': 'LunarLander',
        'acrobot': 'Acrobot',
        'mountaincar': 'MountainCar',
        'pendulum': 'Pendulum',
        'lunarlander_continuous': 'LunarLanderContinuous',
    }

    for env_key, env_name in environments.items():
        pattern = f"{results_dir}/{env_key}*.json"
        files = glob.glob(pattern)

        if files:
            latest_file = max(files, key=lambda x: Path(x).stat().st_mtime)
            with open(latest_file, 'r') as f:
                data = json.load(f)
            results[env_name] = data
            print(f"Loaded: {env_name}")

    return results


def smooth_curve(data, window=20):
    """Apply moving average smoothing."""
    if len(data) < window:
        return data
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode='valid')


def plot_learning_curves(results, output_dir="/root/projects/QBound/results/ppo/plots"):
    """Plot learning curves for each environment."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for env_name, data in results.items():
        fig, ax = plt.subplots(figsize=(10, 6))

        baseline = data.get('Baseline PPO', {})
        qbound = data.get('PPO + QBound', {})

        if baseline and 'episode_rewards' in baseline:
            rewards = baseline['episode_rewards']
            smoothed = smooth_curve(rewards)
            episodes = range(len(smoothed))
            ax.plot(episodes, smoothed, label='Baseline PPO', color='blue', linewidth=2)

        if qbound and 'episode_rewards' in qbound:
            rewards = qbound['episode_rewards']
            smoothed = smooth_curve(rewards)
            episodes = range(len(smoothed))
            ax.plot(episodes, smoothed, label='PPO + QBound', color='red', linewidth=2)

        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Episode Reward (smoothed)', fontsize=12)
        ax.set_title(f'{env_name} Learning Curves', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        # Save figure
        plt.tight_layout()
        filename = f"{output_dir}/{env_name.lower()}_learning_curve.pdf"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.savefig(filename.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  Saved: {filename}")


def plot_bar_comparison(results, output_dir="/root/projects/QBound/results/ppo/plots"):
    """Plot bar chart comparing all environments."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    env_names = []
    baseline_means = []
    qbound_means = []
    improvements = []

    for env_name, data in results.items():
        baseline = data.get('Baseline PPO', {}).get('final_100_episodes', {})
        qbound = data.get('PPO + QBound', {}).get('final_100_episodes', {})

        if baseline and qbound:
            env_names.append(env_name)
            baseline_mean = baseline['mean']
            qbound_mean = qbound['mean']

            baseline_means.append(baseline_mean)
            qbound_means.append(qbound_mean)

            improvement = ((qbound_mean - baseline_mean) / abs(baseline_mean) * 100)
            improvements.append(improvement)

    # Create bar plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left plot: Absolute performance
    x = np.arange(len(env_names))
    width = 0.35

    ax1.bar(x - width/2, baseline_means, width, label='Baseline PPO', color='blue', alpha=0.7)
    ax1.bar(x + width/2, qbound_means, width, label='PPO + QBound', color='red', alpha=0.7)

    ax1.set_xlabel('Environment', fontsize=12)
    ax1.set_ylabel('Final Performance (Mean Reward)', fontsize=12)
    ax1.set_title('PPO vs PPO+QBound: Absolute Performance', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(env_names, rotation=45, ha='right')
    ax1.legend(fontsize=11)
    ax1.grid(True, axis='y', alpha=0.3)

    # Right plot: Improvement percentages
    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    ax2.bar(range(len(env_names)), improvements, color=colors, alpha=0.7)

    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Environment', fontsize=12)
    ax2.set_ylabel('Improvement over Baseline (%)', fontsize=12)
    ax2.set_title('QBound Improvement', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(len(env_names)))
    ax2.set_xticklabels(env_names, rotation=45, ha='right')
    ax2.grid(True, axis='y', alpha=0.3)

    # Add percentage labels on bars
    for i, imp in enumerate(improvements):
        ax2.text(i, imp + (2 if imp > 0 else -5), f'{imp:+.1f}%',
                ha='center', va='bottom' if imp > 0 else 'top', fontsize=9, fontweight='bold')

    plt.tight_layout()
    filename = f"{output_dir}/ppo_qbound_comparison.pdf"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.savefig(filename.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {filename}")


def plot_2x2_grid(results, output_dir="/root/projects/QBound/results/ppo/plots"):
    """Plot 2x2 grid: Action Space × Reward Structure."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Categorize environments
    discrete_sparse = {}
    discrete_dense = {}
    continuous_sparse = {}
    continuous_dense = {}

    for env_name, data in results.items():
        if 'CartPole' in env_name:
            discrete_dense[env_name] = data
        elif 'Pendulum' in env_name:
            continuous_dense[env_name] = data
        elif 'Continuous' in env_name or 'continuous' in env_name.lower():
            continuous_sparse[env_name] = data
        else:
            discrete_sparse[env_name] = data

    # Create 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    categories = [
        (discrete_sparse, 'Discrete Actions + Sparse Rewards', axes[0, 0]),
        (discrete_dense, 'Discrete Actions + Dense Rewards', axes[0, 1]),
        (continuous_sparse, 'Continuous Actions + Sparse Rewards', axes[1, 0]),
        (continuous_dense, 'Continuous Actions + Dense Rewards', axes[1, 1]),
    ]

    for category_data, title, ax in categories:
        if not category_data:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=14)
            ax.set_title(title, fontsize=12, fontweight='bold')
            continue

        env_names = []
        improvements = []

        for env_name, data in category_data.items():
            baseline = data.get('Baseline PPO', {}).get('final_100_episodes', {})
            qbound = data.get('PPO + QBound', {}).get('final_100_episodes', {})

            if baseline and qbound:
                baseline_mean = baseline['mean']
                qbound_mean = qbound['mean']
                improvement = ((qbound_mean - baseline_mean) / abs(baseline_mean) * 100)

                env_names.append(env_name)
                improvements.append(improvement)

        if improvements:
            colors = ['green' if imp > 0 else 'red' for imp in improvements]
            ax.bar(range(len(env_names)), improvements, color=colors, alpha=0.7)

            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax.set_ylabel('Improvement (%)', fontsize=11)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xticks(range(len(env_names)))
            ax.set_xticklabels(env_names, rotation=45, ha='right', fontsize=9)
            ax.grid(True, axis='y', alpha=0.3)

            # Add labels
            for i, imp in enumerate(improvements):
                ax.text(i, imp + (2 if imp > 0 else -5), f'{imp:+.1f}%',
                       ha='center', va='bottom' if imp > 0 else 'top', fontsize=9)

    plt.suptitle('QBound Performance by Environment Type', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    filename = f"{output_dir}/ppo_qbound_2x2_grid.pdf"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.savefig(filename.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {filename}")


def main():
    """Generate all plots."""
    print("="*60)
    print("Generating PPO + QBound Plots")
    print("="*60)

    results = load_latest_results()

    if not results:
        print("\nERROR: No results found. Run experiments first.")
        return

    print("\nGenerating learning curves...")
    plot_learning_curves(results)

    print("\nGenerating bar comparison...")
    plot_bar_comparison(results)

    print("\nGenerating 2x2 grid...")
    plot_2x2_grid(results)

    print("\n" + "="*60)
    print("All plots generated!")
    print("="*60)
    print(f"\nPlots saved to: /root/projects/QBound/results/ppo/plots/")


if __name__ == "__main__":
    main()
