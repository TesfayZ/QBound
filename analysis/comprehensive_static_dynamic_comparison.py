#!/usr/bin/env python3
"""
Comprehensive Static vs Dynamic QBound Comparison

Analyzes ALL experiment results to determine when dynamic QBound
outperforms static QBound across all architectures.
"""

import json
import glob
import numpy as np
from pathlib import Path

def load_result_file(filepath):
    """Load a result JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def get_final_performance(rewards, window=100):
    """Get mean reward over last window episodes"""
    if len(rewards) < window:
        return np.mean(rewards)
    return np.mean(rewards[-window:])

def compare_all_variants(result_data):
    """Compare all static vs dynamic variants in a result file"""
    training = result_data.get('training', {})
    comparisons = []

    # Define all possible static/dynamic pairs
    pairs = [
        ('static_qbound', 'dynamic_qbound', 'DQN'),
        ('static_qbound_ddqn', 'dynamic_qbound_ddqn', 'DDQN'),
        ('static_qbound_dueling_dqn', 'dynamic_qbound_dueling_dqn', 'Dueling DQN'),
        ('static_qbound_double_dueling_dqn', 'dynamic_qbound_double_dueling_dqn', 'Double Dueling DQN'),
    ]

    for static_name, dynamic_name, label in pairs:
        if static_name in training and dynamic_name in training:
            static_rewards = training[static_name].get('rewards', [])
            dynamic_rewards = training[dynamic_name].get('rewards', [])

            if static_rewards and dynamic_rewards:
                static_perf = get_final_performance(static_rewards, 100)
                dynamic_perf = get_final_performance(dynamic_rewards, 100)

                improvement = ((dynamic_perf - static_perf) / static_perf * 100) if static_perf > 0 else 0

                comparisons.append({
                    'architecture': label,
                    'static': static_perf,
                    'dynamic': dynamic_perf,
                    'dynamic_better': dynamic_perf > static_perf,
                    'improvement': improvement
                })

    return comparisons

def main():
    # Find all result files from organized experiments
    result_files = []
    result_dirs = ['cartpole', 'pendulum']

    for env_dir in result_dirs:
        pattern = f'results/{env_dir}/*_full_qbound_seed*.json'
        result_files.extend(glob.glob(pattern))

    if not result_files:
        print("No result files found!")
        return

    print("=" * 80)
    print("COMPREHENSIVE STATIC vs DYNAMIC QBOUND COMPARISON")
    print("=" * 80)
    print(f"\nAnalyzing {len(result_files)} result files\n")

    # Track all comparisons
    all_results = []

    # Analyze each result file
    for filepath in sorted(result_files):
        result_data = load_result_file(filepath)

        # Extract experiment info
        filename = Path(filepath).name
        env = Path(filepath).parent.name
        seed = result_data.get('config', {}).get('seed', 'unknown')

        # Get all comparisons for this file
        comparisons = compare_all_variants(result_data)

        if comparisons:
            print(f"\n{'─' * 80}")
            print(f"Environment: {env.upper()} | Seed: {seed}")
            print(f"File: {filename}")
            print(f"{'─' * 80}")

            for comp in comparisons:
                print(f"\n{comp['architecture']}:")
                print(f"  Static:  {comp['static']:.2f}")
                print(f"  Dynamic: {comp['dynamic']:.2f}")

                if comp['dynamic_better']:
                    print(f"  ✓ DYNAMIC WINS by {comp['improvement']:.2f}%")
                else:
                    print(f"  ✗ Static wins by {-comp['improvement']:.2f}%")

                # Add to overall results
                all_results.append({
                    'env': env,
                    'seed': seed,
                    'architecture': comp['architecture'],
                    'static': comp['static'],
                    'dynamic': comp['dynamic'],
                    'dynamic_better': comp['dynamic_better'],
                    'improvement': comp['improvement']
                })

    # Overall summary
    print(f"\n{'=' * 80}")
    print("OVERALL SUMMARY")
    print(f"{'=' * 80}")

    if all_results:
        total = len(all_results)
        dynamic_wins = sum(1 for r in all_results if r['dynamic_better'])
        static_wins = total - dynamic_wins

        print(f"\nTotal comparisons: {total}")
        print(f"Dynamic QBound wins: {dynamic_wins} ({dynamic_wins/total*100:.1f}%)")
        print(f"Static QBound wins: {static_wins} ({static_wins/total*100:.1f}%)")

        # Average improvements
        dynamic_improvements = [r['improvement'] for r in all_results if r['dynamic_better']]
        static_improvements = [-r['improvement'] for r in all_results if not r['dynamic_better']]

        if dynamic_improvements:
            print(f"\nWhen dynamic wins, average improvement: {np.mean(dynamic_improvements):.2f}%")
            print(f"  Best dynamic improvement: {np.max(dynamic_improvements):.2f}%")
        if static_improvements:
            print(f"\nWhen static wins, average improvement: {np.mean(static_improvements):.2f}%")
            print(f"  Best static improvement: {np.max(static_improvements):.2f}%")

        # By architecture
        print(f"\n{'─' * 80}")
        print("BY ARCHITECTURE:")
        print(f"{'─' * 80}")

        architectures = sorted(set(r['architecture'] for r in all_results))
        for arch in architectures:
            arch_results = [r for r in all_results if r['architecture'] == arch]
            arch_dynamic_wins = sum(1 for r in arch_results if r['dynamic_better'])

            print(f"\n{arch}:")
            print(f"  Dynamic wins: {arch_dynamic_wins}/{len(arch_results)}")
            print(f"  Static wins: {len(arch_results) - arch_dynamic_wins}/{len(arch_results)}")

            # Show specific cases where dynamic won
            if arch_dynamic_wins > 0:
                print(f"  Dynamic won in:")
                for r in arch_results:
                    if r['dynamic_better']:
                        print(f"    - {r['env']} (seed {r['seed']}): +{r['improvement']:.1f}%")

        # By environment
        print(f"\n{'─' * 80}")
        print("BY ENVIRONMENT:")
        print(f"{'─' * 80}")

        envs = sorted(set(r['env'] for r in all_results))
        for env in envs:
            env_results = [r for r in all_results if r['env'] == env]
            env_dynamic_wins = sum(1 for r in env_results if r['dynamic_better'])

            print(f"\n{env.upper()}:")
            print(f"  Dynamic wins: {env_dynamic_wins}/{len(env_results)}")
            print(f"  Static wins: {len(env_results) - env_dynamic_wins}/{len(env_results)}")

    else:
        print("\nNo valid comparisons found!")

    # Conclusion
    print(f"\n{'=' * 80}")
    print("CONCLUSION")
    print(f"{'=' * 80}")

    if all_results:
        dynamic_wins = sum(1 for r in all_results if r['dynamic_better'])
        total = len(all_results)

        print(f"\nDynamic QBound outperforms Static QBound in {dynamic_wins}/{total} cases ({dynamic_wins/total*100:.1f}%)")

        if dynamic_wins > 0:
            print("\nSpecific cases where Dynamic QBound helped:")
            for r in sorted(all_results, key=lambda x: x['improvement'], reverse=True):
                if r['dynamic_better']:
                    print(f"  • {r['env'].upper()} - {r['architecture']} (seed {r['seed']}): +{r['improvement']:.1f}%")
        else:
            print("\n⚠️  Dynamic QBound did NOT outperform Static QBound in any tested case!")
            print("    Static QBound appears to be the more reliable choice.")

    print(f"\n{'=' * 80}\n")

if __name__ == '__main__':
    main()
