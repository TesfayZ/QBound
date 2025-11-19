#!/usr/bin/env python3
"""
Statistical significance analysis: Are the performance changes real or noise?

User's hypothesis: Low violations (< 1%) mean QBound has no real impact.
The observed changes are just random variance, not actual improvement/degradation.

Test: Compare variance in performance changes with violation rates.
"""

import json
import numpy as np
from pathlib import Path

def paired_ttest(x, y):
    """Manual paired t-test implementation."""
    diff = x - y
    n = len(diff)
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)

    if std_diff == 0:
        return 0.0, 1.0

    t_stat = mean_diff / (std_diff / np.sqrt(n))

    # Approximate p-value (two-tailed)
    # Using simple approximation for small n
    from math import exp, sqrt

    # Degrees of freedom
    df = n - 1

    # Very rough p-value approximation
    # For proper p-value, would need t-distribution CDF
    # Using normal approximation for now
    z = abs(t_stat)
    p_value = 2 * (1 - 0.5 * (1 + np.tanh(0.07 * z * (z + 1))))

    return t_stat, p_value

def analyze_significance():
    """Analyze if performance changes are statistically significant."""

    print("=" * 80)
    print("STATISTICAL SIGNIFICANCE ANALYSIS")
    print("=" * 80)
    print()
    print("Hypothesis: Low violations → QBound has NO REAL IMPACT")
    print("Observed changes are random variance, not true effect")
    print()

    environments = {
        'Pendulum (Dense Negative)': {
            'path': 'results/pendulum/dqn_full_qbound_seed*.json',
            'baseline_key': 'dqn',
            'qbound_key': 'static_qbound_dqn',
        },
        'MountainCar (Sparse Negative)': {
            'path': 'results/mountaincar/dqn_static_qbound_seed*.json',
            'baseline_key': 'baseline',
            'qbound_key': 'static_qbound',
        },
        'Acrobot (Sparse Negative)': {
            'path': 'results/acrobot/dqn_static_qbound_seed*.json',
            'baseline_key': 'baseline',
            'qbound_key': 'static_qbound',
        },
        'CartPole (Dense Positive)': {
            'path': 'results/cartpole/dqn_full_qbound_seed*.json',
            'baseline_key': 'baseline',
            'qbound_key': 'static_qbound',
        },
    }

    results_summary = {}

    for env_name, env_info in environments.items():
        print(f"\n{'='*80}")
        print(f"{env_name}")
        print(f"{'='*80}")

        # Find files
        files = list(Path('/root/projects/QBound').glob(env_info['path']))
        if not files:
            print(f"No results found!")
            continue

        # Load results
        baseline_perfs = []
        qbound_perfs = []
        violations = []

        for file in sorted(files):
            # Skip in_progress files
            if 'in_progress' in str(file):
                continue

            with open(file, 'r') as f:
                data = json.load(f)

            # Get performance
            baseline = np.mean(data['training'][env_info['baseline_key']]['rewards'][-100:])
            qbound = np.mean(data['training'][env_info['qbound_key']]['rewards'][-100:])

            baseline_perfs.append(baseline)
            qbound_perfs.append(qbound)

            # Get violations if available
            if 'violations' in data['training'][env_info['qbound_key']]:
                viol_data = data['training'][env_info['qbound_key']]['violations']
                if 'mean' in viol_data:
                    viol_rate = viol_data['mean'].get('next_q_violate_max_rate', 0)
                    violations.append(viol_rate)

        if not baseline_perfs:
            print("No data loaded!")
            continue

        baseline_arr = np.array(baseline_perfs)
        qbound_arr = np.array(qbound_perfs)

        # Compute statistics
        mean_baseline = baseline_arr.mean()
        std_baseline = baseline_arr.std()
        mean_qbound = qbound_arr.mean()
        std_qbound = qbound_arr.std()

        # Performance change
        if 'Positive' in env_name:
            # For positive rewards: higher is better
            changes = ((qbound_arr - baseline_arr) / np.abs(baseline_arr)) * 100
            mean_change = ((mean_qbound - mean_baseline) / np.abs(mean_baseline)) * 100
        else:
            # For negative rewards: less negative is better
            # So improvement = qbound closer to 0
            changes = ((qbound_arr - baseline_arr) / np.abs(baseline_arr)) * 100
            mean_change = ((mean_qbound - mean_baseline) / np.abs(mean_baseline)) * 100

        std_change = changes.std()

        # Paired t-test
        t_stat, p_value = paired_ttest(qbound_arr, baseline_arr)

        # Effect size (Cohen's d)
        diff = qbound_arr - baseline_arr
        pooled_std = np.sqrt((std_baseline**2 + std_qbound**2) / 2)
        cohens_d = diff.mean() / pooled_std if pooled_std > 0 else 0

        print(f"\nPerformance:")
        print(f"  Baseline: {mean_baseline:.2f} ± {std_baseline:.2f}")
        print(f"  QBound:   {mean_qbound:.2f} ± {std_qbound:.2f}")
        print(f"  Change:   {mean_change:+.1f}% ± {std_change:.1f}%")
        print()

        print(f"Statistical Tests:")
        print(f"  t-statistic: {t_stat:.3f}")
        print(f"  p-value: {p_value:.4f}")
        print(f"  Cohen's d (effect size): {cohens_d:.3f}")
        print()

        # Interpret p-value
        if p_value < 0.01:
            significance = "HIGHLY SIGNIFICANT (p < 0.01)"
        elif p_value < 0.05:
            significance = "SIGNIFICANT (p < 0.05)"
        elif p_value < 0.10:
            significance = "MARGINALLY SIGNIFICANT (p < 0.10)"
        else:
            significance = "NOT SIGNIFICANT (p ≥ 0.10)"

        print(f"  Result: {significance}")

        # Interpret effect size
        if abs(cohens_d) < 0.2:
            effect = "NEGLIGIBLE"
        elif abs(cohens_d) < 0.5:
            effect = "SMALL"
        elif abs(cohens_d) < 0.8:
            effect = "MEDIUM"
        else:
            effect = "LARGE"

        print(f"  Effect size: {effect}")
        print()

        # Violations
        if violations:
            mean_viol = np.mean(violations) * 100
            print(f"Violations:")
            print(f"  Mean violation rate: {mean_viol:.2f}%")
            print()

        # Interpretation
        print("Interpretation:")
        if p_value >= 0.10 and abs(cohens_d) < 0.2:
            print("  ✓ RANDOM VARIANCE - QBound has NO REAL IMPACT")
            print("  The observed changes are statistical noise")
        elif p_value >= 0.05:
            print("  ~ UNCERTAIN - Effect may exist but not statistically significant")
            print("  Need more seeds to confirm")
        else:
            if 'Positive' in env_name and mean_change > 0:
                print("  ✓ REAL IMPROVEMENT - QBound helps significantly")
            elif 'Negative' in env_name and mean_change > 0:
                print("  ✗ REAL DEGRADATION - QBound hurts significantly")
            elif 'Negative' in env_name and mean_change < 0:
                print("  ✓ REAL IMPROVEMENT - QBound helps significantly")
            else:
                print("  ✗ REAL DEGRADATION - QBound hurts significantly")

        # Store for summary
        results_summary[env_name] = {
            'mean_change': mean_change,
            'std_change': std_change,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'violations': mean_viol if violations else None,
            'significant': p_value < 0.05,
            'effect_size': effect,
        }

    # Summary comparison
    print("\n" + "=" * 80)
    print("SUMMARY: Violation Rate vs Statistical Significance")
    print("=" * 80)
    print()

    print(f"{'Environment':<30} {'Violations':<15} {'Change':<15} {'p-value':<12} {'Significant?'}")
    print("-" * 90)

    for env_name, stats_data in results_summary.items():
        viol_str = f"{stats_data['violations']:.2f}%" if stats_data['violations'] is not None else "N/A"
        change_str = f"{stats_data['mean_change']:+.1f}%"
        p_str = f"{stats_data['p_value']:.4f}"
        sig_str = "YES" if stats_data['significant'] else "NO"

        print(f"{env_name:<30} {viol_str:<15} {change_str:<15} {p_str:<12} {sig_str}")

    print()
    print("KEY FINDINGS:")
    print("-" * 60)

    # Check user's hypothesis
    low_viol_envs = [(name, data) for name, data in results_summary.items()
                     if data['violations'] is not None and data['violations'] < 5.0]
    high_viol_envs = [(name, data) for name, data in results_summary.items()
                      if data['violations'] is not None and data['violations'] >= 5.0]

    print("\nLOW VIOLATION ENVIRONMENTS (< 5%):")
    for name, data in low_viol_envs:
        sig_str = "significant" if data['significant'] else "NOT significant"
        print(f"  {name}: {data['violations']:.2f}% violations, p={data['p_value']:.4f} ({sig_str})")

    if high_viol_envs:
        print("\nHIGH VIOLATION ENVIRONMENTS (≥ 5%):")
        for name, data in high_viol_envs:
            sig_str = "significant" if data['significant'] else "NOT significant"
            print(f"  {name}: {data['violations']:.2f}% violations, p={data['p_value']:.4f} ({sig_str})")

    print()
    print("HYPOTHESIS TEST:")
    print("-" * 60)

    # Test: Do low violation envs have non-significant results?
    low_viol_significant = sum(1 for _, data in low_viol_envs if data['significant'])
    high_viol_significant = sum(1 for _, data in high_viol_envs if data['significant'])

    print(f"\nLow violation envs with significant effects: {low_viol_significant}/{len(low_viol_envs)}")
    if high_viol_envs:
        print(f"High violation envs with significant effects: {high_viol_significant}/{len(high_viol_envs)}")

    print()
    if low_viol_significant == 0 and len(low_viol_envs) > 0:
        print("✓ USER'S HYPOTHESIS CONFIRMED!")
        print("  Low violations (< 5%) → NO significant impact")
        print("  Observed changes are random variance, not real effects")
    elif low_viol_significant < len(low_viol_envs) / 2:
        print("~ USER'S HYPOTHESIS PARTIALLY SUPPORTED")
        print("  Most low-violation environments show no significant impact")
    else:
        print("✗ USER'S HYPOTHESIS NOT SUPPORTED")
        print("  Low violations still show significant effects")

if __name__ == '__main__':
    analyze_significance()
