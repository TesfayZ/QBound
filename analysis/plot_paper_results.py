#!/usr/bin/env python3
"""
Generate publication-quality plots for the QBound paper using existing experiment results.
"""

import sys
sys.path.insert(0, '/root/projects/QBound/src')

import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
from datetime import datetime
import os

# Set publication-quality defaults
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']


def load_latest_results():
    """Load the most recent experiment results."""
    results_file = 'results/combined/experiment_results_20251025_132043.json'

    with open(results_file, 'r') as f:
        data = json.load(f)

    return data


def smooth_curve(values, window=50):
    """Smooth a curve using moving average."""
    if len(values) < window:
        return values, np.arange(len(values))

    smoothed = np.convolve(values, np.ones(window)/window, mode='valid')
    x = np.arange(window-1, len(values))
    return smoothed, x


def plot_learning_curves(results, output_dir='results/plots'):
    """Create learning curve plots for all three environments."""
    os.makedirs(output_dir, exist_ok=True)

    # Create a combined figure with all three environments
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    environments = [
        ('GridWorld', 'GridWorld', 50, 'Success Rate (%)'),
        ('FrozenLake', 'FrozenLake', 100, 'Success Rate (%)'),
        ('CartPole', 'CartPole', 50, 'Episode Reward')
    ]

    for idx, (env_key, env_name, window, ylabel) in enumerate(environments):
        ax = axes[idx]

        # Get data
        env_data = results[env_key]

        # Extract episode rewards
        qbound_rewards = env_data['rewards_qbound']
        baseline_rewards = env_data['rewards_baseline']

        # For GridWorld and FrozenLake, convert to success rate (reward > 0)
        if env_key in ['GridWorld', 'FrozenLake']:
            qbound_success = [1.0 if r > 0 else 0.0 for r in qbound_rewards]
            baseline_success = [1.0 if r > 0 else 0.0 for r in baseline_rewards]

            # Smooth
            qbound_smooth, qbound_x = smooth_curve(qbound_success, window)
            baseline_smooth, baseline_x = smooth_curve(baseline_success, window)

            # Convert to percentage
            qbound_smooth = qbound_smooth * 100
            baseline_smooth = baseline_smooth * 100
        else:
            # For CartPole, plot raw rewards
            qbound_smooth, qbound_x = smooth_curve(qbound_rewards, window)
            baseline_smooth, baseline_x = smooth_curve(baseline_rewards, window)

        # Plot
        ax.plot(qbound_x, qbound_smooth, 'b-', linewidth=2, label='QBound')
        ax.plot(baseline_x, baseline_smooth, 'r-', linewidth=2, alpha=0.7, label='Baseline')

        # Add target line for GridWorld and FrozenLake
        if env_key == 'GridWorld':
            ax.axhline(y=80, color='g', linestyle='--', linewidth=1, alpha=0.5, label='Target (80%)')
        elif env_key == 'FrozenLake':
            ax.axhline(y=70, color='g', linestyle='--', linewidth=1, alpha=0.5, label='Target (70%)')

        ax.set_xlabel('Episode')
        ax.set_ylabel(ylabel)
        ax.set_title(f'{env_name}')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        # Add text box with improvement
        if env_key == 'GridWorld':
            text = '20.2% faster\n(205 vs 257 episodes)'
        elif env_key == 'FrozenLake':
            text = '5.0% faster\n(209 vs 220 episodes)'
        else:
            text = '31.5% higher\ncumulative reward'

        ax.text(0.05, 0.95, text, transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=9)

    plt.tight_layout()

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'{output_dir}/learning_curves_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    plt.close()

    # Also save as PDF for LaTeX
    filename_pdf = f'{output_dir}/learning_curves_{timestamp}.pdf'

    # Recreate the figure for PDF
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for idx, (env_key, env_name, window, ylabel) in enumerate(environments):
        ax = axes[idx]

        env_data = results[env_key]
        qbound_rewards = env_data['rewards_qbound']
        baseline_rewards = env_data['rewards_baseline']

        if env_key in ['GridWorld', 'FrozenLake']:
            qbound_success = [1.0 if r > 0 else 0.0 for r in qbound_rewards]
            baseline_success = [1.0 if r > 0 else 0.0 for r in baseline_rewards]
            qbound_smooth, qbound_x = smooth_curve(qbound_success, window)
            baseline_smooth, baseline_x = smooth_curve(baseline_success, window)
            qbound_smooth = qbound_smooth * 100
            baseline_smooth = baseline_smooth * 100
        else:
            qbound_smooth, qbound_x = smooth_curve(qbound_rewards, window)
            baseline_smooth, baseline_x = smooth_curve(baseline_rewards, window)

        ax.plot(qbound_x, qbound_smooth, 'b-', linewidth=2, label='QBound')
        ax.plot(baseline_x, baseline_smooth, 'r-', linewidth=2, alpha=0.7, label='Baseline')

        if env_key == 'GridWorld':
            ax.axhline(y=80, color='g', linestyle='--', linewidth=1, alpha=0.5, label='Target (80%)')
        elif env_key == 'FrozenLake':
            ax.axhline(y=70, color='g', linestyle='--', linewidth=1, alpha=0.5, label='Target (70%)')

        ax.set_xlabel('Episode')
        ax.set_ylabel(ylabel)
        ax.set_title(f'{env_name}')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        if env_key == 'GridWorld':
            text = '20.2% faster\n(205 vs 257 episodes)'
        elif env_key == 'FrozenLake':
            text = '5.0% faster\n(209 vs 220 episodes)'
        else:
            text = '31.5% higher\ncumulative reward'

        ax.text(0.05, 0.95, text, transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=9)

    plt.tight_layout()
    plt.savefig(filename_pdf, dpi=300, bbox_inches='tight', format='pdf')
    print(f"✓ Saved: {filename_pdf}")
    plt.close()


def plot_individual_environments(results, output_dir='results/plots'):
    """Create individual detailed plots for each environment."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    environments = {
        'GridWorld': ('GridWorld', 50, 80, 'Success Rate (%)'),
        'FrozenLake': ('FrozenLake', 100, 70, 'Success Rate (%)'),
        'CartPole': ('CartPole', 50, None, 'Episode Reward')
    }

    for env_key, (env_name, window, target, ylabel) in environments.items():
        fig, ax = plt.subplots(figsize=(8, 5))

        env_data = results[env_key]

        qbound_rewards = env_data['rewards_qbound']
        baseline_rewards = env_data['rewards_baseline']

        # Convert to success rate for GridWorld/FrozenLake
        if env_key in ['GridWorld', 'FrozenLake']:
            qbound_values = [1.0 if r > 0 else 0.0 for r in qbound_rewards]
            baseline_values = [1.0 if r > 0 else 0.0 for r in baseline_rewards]
            qbound_smooth, qbound_x = smooth_curve(qbound_values, window)
            baseline_smooth, baseline_x = smooth_curve(baseline_values, window)
            qbound_smooth = qbound_smooth * 100
            baseline_smooth = baseline_smooth * 100
        else:
            qbound_smooth, qbound_x = smooth_curve(qbound_rewards, window)
            baseline_smooth, baseline_x = smooth_curve(baseline_rewards, window)

        # Plot
        ax.plot(qbound_x, qbound_smooth, 'b-', linewidth=2.5, label='QBound', zorder=3)
        ax.plot(baseline_x, baseline_smooth, 'r-', linewidth=2.5, alpha=0.7, label='Baseline', zorder=2)

        # Add target line
        if target:
            ax.axhline(y=target, color='g', linestyle='--', linewidth=1.5, alpha=0.6,
                      label=f'Target ({target}%)', zorder=1)

        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f'{env_name} Learning Curves', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save PNG
        filename = f'{output_dir}/{env_key.lower()}_learning_curve_{timestamp}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filename}")

        # Save PDF
        filename_pdf = f'{output_dir}/{env_key.lower()}_learning_curve_{timestamp}.pdf'
        plt.savefig(filename_pdf, dpi=300, bbox_inches='tight', format='pdf')
        print(f"✓ Saved: {filename_pdf}")

        plt.close()


def plot_comparison_bar_chart(results, output_dir='results/plots'):
    """Create bar chart comparing QBound vs Baseline."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Data
    environments = ['GridWorld\n(episodes to 80%)', 'FrozenLake\n(episodes to 70%)', 'CartPole\n(total reward)']
    baseline_values = [257, 220, 131438]
    qbound_values = [205, 209, 172904]

    # Normalize CartPole values to be on similar scale
    normalized_baseline = [257, 220, 131438/1000]  # Divide CartPole by 1000 for display
    normalized_qbound = [205, 209, 172904/1000]

    x = np.arange(len(environments))
    width = 0.35

    bars1 = ax.bar(x - width/2, normalized_baseline, width, label='Baseline', color='#d62728', alpha=0.8)
    bars2 = ax.bar(x + width/2, normalized_qbound, width, label='QBound', color='#1f77b4', alpha=0.8)

    # Add value labels on bars
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        height1 = bar1.get_height()
        height2 = bar2.get_height()

        if i == 2:  # CartPole - show original values
            ax.text(bar1.get_x() + bar1.get_width()/2., height1,
                   f'{baseline_values[i]:,}', ha='center', va='bottom', fontsize=9)
            ax.text(bar2.get_x() + bar2.get_width()/2., height2,
                   f'{qbound_values[i]:,}', ha='center', va='bottom', fontsize=9)
        else:
            ax.text(bar1.get_x() + bar1.get_width()/2., height1,
                   f'{int(height1)}', ha='center', va='bottom', fontsize=9)
            ax.text(bar2.get_x() + bar2.get_width()/2., height2,
                   f'{int(height2)}', ha='center', va='bottom', fontsize=9)

    # Add improvement percentages
    improvements = ['-20.2%', '-5.0%', '+31.5%']
    for i, (imp, x_pos) in enumerate(zip(improvements, x)):
        y_pos = max(normalized_baseline[i], normalized_qbound[i]) * 1.15
        ax.text(x_pos, y_pos, imp, ha='center', va='bottom', fontsize=11,
               fontweight='bold', color='green')

    ax.set_xlabel('Environment', fontsize=12)
    ax.set_ylabel('Episodes / Total Reward (×1000 for CartPole)', fontsize=12)
    ax.set_title('QBound vs Baseline: Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(environments)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    # Save
    filename = f'{output_dir}/comparison_bar_chart_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filename}")

    filename_pdf = f'{output_dir}/comparison_bar_chart_{timestamp}.pdf'
    plt.savefig(filename_pdf, dpi=300, bbox_inches='tight', format='pdf')
    print(f"✓ Saved: {filename_pdf}")

    plt.close()


def main():
    """Generate all plots for the paper."""
    print("\n" + "="*70)
    print("GENERATING PUBLICATION-QUALITY PLOTS FOR QBOUND PAPER")
    print("="*70)

    # Load results
    print("\n📊 Loading experiment results...")
    results = load_latest_results()

    # Create plots directory
    os.makedirs('results/plots', exist_ok=True)

    # Generate plots
    print("\n📈 Generating combined learning curves...")
    plot_learning_curves(results)

    print("\n📈 Generating individual environment plots...")
    plot_individual_environments(results)

    print("\n📊 Generating comparison bar chart...")
    plot_comparison_bar_chart(results)

    # Copy plots to LaTeX figures directory for self-contained paper
    print("\n📁 Copying plots to LaTeX figures directory...")
    latex_figures_dir = 'LatexDocs/figures'
    os.makedirs(latex_figures_dir, exist_ok=True)

    import shutil
    for pdf_file in os.listdir('results/plots'):
        if pdf_file.endswith('.pdf'):
            src = os.path.join('results/plots', pdf_file)
            dst = os.path.join(latex_figures_dir, pdf_file)
            shutil.copy2(src, dst)
            print(f"  ✓ Copied {pdf_file}")

    print("\n" + "="*70)
    print("✓ ALL PLOTS GENERATED SUCCESSFULLY!")
    print("="*70)
    print("\nPlots saved to:")
    print("  • results/plots/ (archive)")
    print("  • LatexDocs/figures/ (for LaTeX compilation)")
    print("\nGenerated files:")
    print("  • learning_curves_*.png/pdf - Combined 3-panel figure")
    print("  • gridworld_learning_curve_*.png/pdf")
    print("  • frozenlake_learning_curve_*.png/pdf")
    print("  • cartpole_learning_curve_*.png/pdf")
    print("  • comparison_bar_chart_*.png/pdf")
    print("\n✓ LatexDocs/ directory is now self-contained for Overleaf upload!")


if __name__ == "__main__":
    main()
