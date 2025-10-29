#!/usr/bin/env python3
"""
Generate plots for Pendulum DDPG 6-way and PPO experiments
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Configure matplotlib
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 11

def moving_average(data, window=20):
    """Calculate moving average"""
    if len(data) < window:
        return data
    cumsum = np.cumsum(np.insert(data, 0, 0))
    return (cumsum[window:] - cumsum[:-window]) / window

def plot_pendulum_6way():
    """Plot Pendulum DDPG 6-way comparison"""
    print("\n📊 Analyzing Pendulum DDPG 6-way results...")

    # Load the latest results
    results_file = Path("results/pendulum/6way_comparison_20251028_150148.json")
    if not results_file.exists():
        print(f"   ✗ File not found: {results_file}")
        return

    with open(results_file, 'r') as f:
        data = json.load(f)

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Pendulum-v1: 6-Way DDPG/TD3 Comparison with Soft QBound',
                 fontsize=16, fontweight='bold')

    # Method names and colors
    methods = {
        'ddpg': ('1. DDPG', 'blue'),
        'td3': ('2. TD3', 'green'),
        'simple_ddpg': ('3. Simple DDPG (Baseline)', 'red'),
        'qbound_simple': ('4. QBound + Simple DDPG', 'orange'),
        'qbound_ddpg': ('5. QBound + DDPG', 'purple'),
        'qbound_td3': ('6. QBound + TD3', 'brown')
    }

    # Plot each method
    for idx, (key, (name, color)) in enumerate(methods.items()):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]

        # Handle both old and new JSON formats
        method_data = None
        if key in data:
            method_data = data[key]
        elif 'training' in data and key in data['training']:
            method_data = data['training'][key]

        if method_data is None or 'rewards' not in method_data:
            ax.set_title(f"{name}\n(No data)", fontsize=12, fontweight='bold')
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
            continue

        rewards = method_data['rewards']
        smoothed = moving_average(rewards, window=20)

        # Plot raw data (light)
        ax.plot(rewards, alpha=0.2, color=color, linewidth=0.5)

        # Plot smoothed data
        start_idx = len(rewards) - len(smoothed)
        ax.plot(range(start_idx, len(rewards)), smoothed,
                color=color, linewidth=2, label=name)

        # Calculate statistics
        mean_reward = np.mean(rewards[-100:])
        std_reward = np.std(rewards[-100:])
        max_reward = np.max(rewards)

        # Formatting
        ax.set_title(f"{name}\nFinal: {mean_reward:.1f}±{std_reward:.1f} | Max: {max_reward:.1f}",
                     fontsize=11, fontweight='bold')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    plt.tight_layout()

    # Save figure
    output_file = Path("results/plots/pendulum_6way_comprehensive_20251028_150148.png")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"   ✓ Saved: {output_file.name}")

    # Copy to paper directory
    paper_file = Path("QBound/figures/pendulum_6way_results.png")
    paper_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(paper_file, dpi=150, bbox_inches='tight')
    print(f"   ✓ Copied to paper: {paper_file.name}")

    plt.close()

    return data

def plot_ppo_results():
    """Plot PPO results for Pendulum and LunarLander Continuous"""
    print("\n📊 Analyzing PPO results...")

    # PPO experiments
    experiments = [
        ('results/ppo/pendulum_20251029_103110.json', 'Pendulum-v1 PPO'),
        ('results/ppo/lunarlander_continuous_20251029_102354.json', 'LunarLander Continuous PPO')
    ]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('PPO with QBound: Continuous Action Spaces',
                 fontsize=16, fontweight='bold')

    for idx, (results_file, title) in enumerate(experiments):
        ax = axes[idx]
        results_file = Path(results_file)

        if not results_file.exists():
            ax.set_title(f"{title}\n(No data)", fontsize=12, fontweight='bold')
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
            continue

        with open(results_file, 'r') as f:
            data = json.load(f)

        # Get methods
        methods = {
            'Baseline PPO': ('blue', 'Baseline'),
            'QBound PPO (Soft)': ('orange', 'QBound Soft'),
            'QBound PPO (Hard)': ('red', 'QBound Hard')
        }

        for method_name, (color, label) in methods.items():
            if method_name not in data:
                continue

            rewards = data[method_name]['episode_rewards']
            smoothed = moving_average(rewards, window=20)

            # Plot raw data (light)
            ax.plot(rewards, alpha=0.15, color=color, linewidth=0.5)

            # Plot smoothed data
            start_idx = len(rewards) - len(smoothed)
            ax.plot(range(start_idx, len(rewards)), smoothed,
                    color=color, linewidth=2.5, label=label)

        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlabel('Episode', fontsize=11)
        ax.set_ylabel('Reward', fontsize=11)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_file = Path("results/plots/ppo_continuous_comparison.png")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"   ✓ Saved: {output_file.name}")

    # Copy to paper directory
    paper_file = Path("QBound/figures/ppo_continuous_comparison.png")
    paper_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(paper_file, dpi=150, bbox_inches='tight')
    print(f"   ✓ Copied to paper: {paper_file.name}")

    plt.close()

def generate_summary_table(pendulum_data):
    """Generate summary table for Pendulum results"""
    print("\n📝 Generating summary table...")

    methods = {
        'ddpg': '1. DDPG',
        'td3': '2. TD3',
        'simple_ddpg': '3. Simple DDPG (Baseline)',
        'qbound_simple': '4. QBound + Simple DDPG',
        'qbound_ddpg': '5. QBound + DDPG',
        'qbound_td3': '6. QBound + TD3'
    }

    print("\n" + "="*80)
    print("Pendulum-v1: Final Performance (Last 100 Episodes)")
    print("="*80)
    print(f"{'Method':<40} {'Mean ± Std':>20} {'Max':>15}")
    print("-"*80)

    for key, name in methods.items():
        # Handle both old and new JSON formats
        method_data = None
        if key in pendulum_data:
            method_data = pendulum_data[key]
        elif 'training' in pendulum_data and key in pendulum_data['training']:
            method_data = pendulum_data['training'][key]

        if method_data is None or 'rewards' not in method_data:
            continue

        rewards = method_data['rewards']
        mean_reward = np.mean(rewards[-100:])
        std_reward = np.std(rewards[-100:])
        max_reward = np.max(rewards)

        print(f"{name:<40} {mean_reward:>10.1f} ± {std_reward:<8.1f} {max_reward:>15.1f}")

    print("="*80)

def main():
    print("\n" + "="*80)
    print("QBound: Pendulum & PPO Analysis")
    print("="*80)

    # Plot Pendulum 6-way results
    pendulum_data = plot_pendulum_6way()

    # Plot PPO results
    plot_ppo_results()

    # Generate summary
    if pendulum_data:
        generate_summary_table(pendulum_data)

    print("\n" + "="*80)
    print("Analysis Complete!")
    print("="*80)
    print("\n📂 Plots saved to: /root/projects/QBound/results/plots")
    print("📂 Paper figures: /root/projects/QBound/QBound/figures")

if __name__ == "__main__":
    main()
