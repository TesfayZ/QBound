"""
Comprehensive Analysis of PPO + QBound Results

Analyzes results across all environments and generates:
1. Per-environment comparison (Baseline vs QBound)
2. Cross-environment patterns (action space, reward structure)
3. Statistical significance tests
4. Summary tables and findings
"""

import json
import glob
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from datetime import datetime


def load_latest_results(results_dir="/root/projects/QBound/results/ppo"):
    """Load the latest results for each environment."""
    results = {}

    environments = {
        'cartpole': 'CartPole-v1 (Discrete + Dense)',
        'lunarlander': 'LunarLander-v3 (Discrete + Sparse)',
        'acrobot': 'Acrobot-v1 (Discrete + Sparse)',
        'mountaincar': 'MountainCar-v0 (Discrete + Sparse)',
        'pendulum': 'Pendulum-v1 (Continuous + Dense)',
        'lunarlander_continuous': 'LunarLanderContinuous-v3 (Continuous + Sparse)',
    }

    for env_key, env_name in environments.items():
        # Find all result files for this environment
        pattern = f"{results_dir}/{env_key}*.json"
        files = glob.glob(pattern)

        if files:
            # Get the latest file
            latest_file = max(files, key=lambda x: Path(x).stat().st_mtime)

            with open(latest_file, 'r') as f:
                data = json.load(f)

            results[env_name] = {
                'file': latest_file,
                'data': data
            }
            print(f"Loaded: {env_name}")
            print(f"  File: {latest_file}")
        else:
            print(f"WARNING: No results found for {env_name}")

    return results


def analyze_single_environment(env_name, env_data):
    """Analyze results for a single environment."""
    print(f"\n{'='*60}")
    print(f"{env_name}")
    print(f"{'='*60}")

    data = env_data['data']

    baseline = data.get('Baseline PPO', {})
    qbound = data.get('PPO + QBound', {})

    if not baseline or not qbound:
        print("ERROR: Missing data")
        return None

    # Extract final 100 episodes statistics
    baseline_stats = baseline['final_100_episodes']
    qbound_stats = qbound['final_100_episodes']

    baseline_mean = baseline_stats['mean']
    qbound_mean = qbound_stats['mean']

    improvement = ((qbound_mean - baseline_mean) / abs(baseline_mean) * 100) if baseline_mean != 0 else 0

    print(f"\nBaseline PPO:")
    print(f"  Mean: {baseline_mean:.2f} ± {baseline_stats['std']:.2f}")
    print(f"  Max: {baseline_stats['max']:.2f}")
    if 'success_rate' in baseline_stats:
        print(f"  Success Rate: {baseline_stats['success_rate']:.1f}%")

    print(f"\nPPO + QBound:")
    print(f"  Mean: {qbound_mean:.2f} ± {qbound_stats['std']:.2f}")
    print(f"  Max: {qbound_stats['max']:.2f}")
    if 'success_rate' in qbound_stats:
        print(f"  Success Rate: {qbound_stats['success_rate']:.1f}%")

    print(f"\n{'Result:':<20} {improvement:+.1f}%")

    # Interpret result
    if improvement < -50:
        verdict = "❌ CATASTROPHIC FAILURE"
    elif improvement < -10:
        verdict = "⚠️  SIGNIFICANT DEGRADATION"
    elif improvement < 5:
        verdict = "➖ NEUTRAL"
    elif improvement < 20:
        verdict = "✓ MODERATE IMPROVEMENT"
    elif improvement < 50:
        verdict = "✅ STRONG IMPROVEMENT"
    else:
        verdict = "🎉 EXCEPTIONAL IMPROVEMENT"

    print(f"{'Verdict:':<20} {verdict}")

    # Statistical significance test (t-test on last 100 episodes)
    baseline_rewards = baseline['episode_rewards'][-100:]
    qbound_rewards = qbound['episode_rewards'][-100:]

    t_stat, p_value = stats.ttest_ind(qbound_rewards, baseline_rewards)

    print(f"\nStatistical Test:")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.4f}")
    if p_value < 0.05:
        print(f"  Result: ✓ Statistically significant (p < 0.05)")
    else:
        print(f"  Result: ✗ Not statistically significant")

    return {
        'env_name': env_name,
        'baseline_mean': baseline_mean,
        'baseline_std': baseline_stats['std'],
        'qbound_mean': qbound_mean,
        'qbound_std': qbound_stats['std'],
        'improvement_pct': improvement,
        'p_value': p_value,
        'verdict': verdict,
        'baseline_success': baseline_stats.get('success_rate', None),
        'qbound_success': qbound_stats.get('success_rate', None),
    }


def cross_environment_analysis(summary_data):
    """Analyze patterns across environments."""
    print(f"\n{'='*60}")
    print("CROSS-ENVIRONMENT ANALYSIS")
    print(f"{'='*60}")

    # Categorize by action space and reward structure
    discrete_sparse = []
    discrete_dense = []
    continuous_sparse = []
    continuous_dense = []

    for data in summary_data:
        env = data['env_name']
        improvement = data['improvement_pct']

        if 'CartPole' in env:
            discrete_dense.append((env, improvement))
        elif 'Pendulum' in env:
            continuous_dense.append((env, improvement))
        elif 'Continuous' in env:
            continuous_sparse.append((env, improvement))
        else:
            discrete_sparse.append((env, improvement))

    # Print categorized results
    print("\n1. DISCRETE ACTION SPACES:")
    print(f"\n   Sparse Rewards:")
    for env, imp in discrete_sparse:
        print(f"     {env:<50} {imp:+.1f}%")
    if discrete_sparse:
        avg = np.mean([imp for _, imp in discrete_sparse])
        print(f"     {'Average:':<50} {avg:+.1f}%")

    print(f"\n   Dense Rewards:")
    for env, imp in discrete_dense:
        print(f"     {env:<50} {imp:+.1f}%")
    if discrete_dense:
        avg = np.mean([imp for _, imp in discrete_dense])
        print(f"     {'Average:':<50} {avg:+.1f}%")

    print("\n2. CONTINUOUS ACTION SPACES (CRITICAL TEST):")
    print(f"\n   Sparse Rewards:")
    for env, imp in continuous_sparse:
        print(f"     {env:<50} {imp:+.1f}%")
        if imp < -50:
            print(f"       ❌ FAILED: QBound breaks continuous actions")
        else:
            print(f"       ✅ SUCCESS: QBound works on continuous actions!")
    if continuous_sparse:
        avg = np.mean([imp for _, imp in continuous_sparse])
        print(f"     {'Average:':<50} {avg:+.1f}%")

    print(f"\n   Dense Rewards:")
    for env, imp in continuous_dense:
        print(f"     {env:<50} {imp:+.1f}%")
        if imp < -50:
            print(f"       ❌ FAILED: Like DDPG/TD3 (-893%)")
        else:
            print(f"       ✅ SUCCESS: Unlike DDPG/TD3!")
    if continuous_dense:
        avg = np.mean([imp for _, imp in continuous_dense])
        print(f"     {'Average:':<50} {avg:+.1f}%")

    # Overall statistics
    print("\n3. OVERALL STATISTICS:")
    all_improvements = [d['improvement_pct'] for d in summary_data]
    print(f"   Mean improvement: {np.mean(all_improvements):+.1f}%")
    print(f"   Median improvement: {np.median(all_improvements):+.1f}%")
    print(f"   Std deviation: {np.std(all_improvements):.1f}%")
    print(f"   Min: {np.min(all_improvements):+.1f}%")
    print(f"   Max: {np.max(all_improvements):+.1f}%")

    # Count successes
    positive = sum(1 for imp in all_improvements if imp > 5)
    neutral = sum(1 for imp in all_improvements if -5 <= imp <= 5)
    negative = sum(1 for imp in all_improvements if imp < -5)

    print(f"\n   Environments improved: {positive}/{len(all_improvements)}")
    print(f"   Environments neutral: {neutral}/{len(all_improvements)}")
    print(f"   Environments degraded: {negative}/{len(all_improvements)}")

    success_rate = positive / len(all_improvements) * 100
    print(f"\n   Success rate: {success_rate:.1f}%")


def generate_summary_table(summary_data):
    """Generate markdown table for documentation."""
    print(f"\n{'='*60}")
    print("SUMMARY TABLE (Markdown)")
    print(f"{'='*60}\n")

    print("| Environment | Action Space | Reward | Baseline | QBound | Improvement |")
    print("|-------------|--------------|--------|----------|--------|-------------|")

    for data in summary_data:
        env = data['env_name'].split('(')[0].strip()

        # Parse action space and reward type
        if 'Continuous' in data['env_name']:
            action = "Continuous"
        else:
            action = "Discrete"

        if 'Sparse' in data['env_name']:
            reward = "Sparse"
        else:
            reward = "Dense"

        baseline = f"{data['baseline_mean']:.1f}"
        qbound = f"{data['qbound_mean']:.1f}"
        improvement = f"{data['improvement_pct']:+.1f}%"

        # Add emoji for visual clarity
        if data['improvement_pct'] > 20:
            improvement += " ✅"
        elif data['improvement_pct'] < -20:
            improvement += " ❌"

        print(f"| {env:<11} | {action:<12} | {reward:<6} | {baseline:<8} | {qbound:<6} | {improvement:<11} |")


def main():
    """Main analysis pipeline."""
    print("="*60)
    print("PPO + QBound Results Analysis")
    print("="*60)

    # Load results
    results = load_latest_results()

    if not results:
        print("\nERROR: No results found. Run experiments first.")
        return

    # Analyze each environment
    summary_data = []
    for env_name, env_data in results.items():
        result = analyze_single_environment(env_name, env_data)
        if result:
            summary_data.append(result)

    # Cross-environment analysis
    if summary_data:
        cross_environment_analysis(summary_data)
        generate_summary_table(summary_data)

    # Save summary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = f"/root/projects/QBound/results/ppo/analysis_summary_{timestamp}.json"

    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=2)

    print(f"\n\nSummary saved to: {summary_file}")


if __name__ == "__main__":
    main()
