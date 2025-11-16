#!/usr/bin/env python3
"""
Compare Static vs Dynamic QBound Performance

Analyzes all experiment results to determine when dynamic QBound
outperforms static QBound.
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

def analyze_experiment(result_data):
    """Analyze a single experiment result"""
    training = result_data.get('training', {})

    # Extract performance metrics
    methods = {}

    for method_name, method_data in training.items():
        if 'rewards' in method_data:
            rewards = method_data['rewards']
            methods[method_name] = {
                'mean_final_100': get_final_performance(rewards, 100),
                'mean_all': np.mean(rewards),
                'max_reward': np.max(rewards),
                'total_episodes': len(rewards)
            }

    return methods

def compare_static_vs_dynamic(methods):
    """Compare static and dynamic QBound performance"""
    comparisons = {}

    # Compare DQN variants
    if 'static_qbound' in methods and 'dynamic_qbound' in methods:
        static = methods['static_qbound']['mean_final_100']
        dynamic = methods['dynamic_qbound']['mean_final_100']
        comparisons['dqn'] = {
            'static': static,
            'dynamic': dynamic,
            'dynamic_better': dynamic > static,
            'improvement': ((dynamic - static) / static * 100) if static > 0 else 0
        }

    # Compare DDQN variants
    if 'static_qbound_ddqn' in methods and 'dynamic_qbound_ddqn' in methods:
        static = methods['static_qbound_ddqn']['mean_final_100']
        dynamic = methods['dynamic_qbound_ddqn']['mean_final_100']
        comparisons['ddqn'] = {
            'static': static,
            'dynamic': dynamic,
            'dynamic_better': dynamic > static,
            'improvement': ((dynamic - static) / static * 100) if static > 0 else 0
        }

    # Compare Dueling variants
    if 'static_qbound_dueling' in methods and 'dynamic_qbound_dueling' in methods:
        static = methods['static_qbound_dueling']['mean_final_100']
        dynamic = methods['dynamic_qbound_dueling']['mean_final_100']
        comparisons['dueling'] = {
            'static': static,
            'dynamic': dynamic,
            'dynamic_better': dynamic > static,
            'improvement': ((dynamic - static) / static * 100) if static > 0 else 0
        }

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
    print("STATIC vs DYNAMIC QBOUND COMPARISON")
    print("=" * 80)
    print(f"\nFound {len(result_files)} result files\n")

    # Track overall statistics
    all_comparisons = {
        'dynamic_wins': 0,
        'static_wins': 0,
        'details': []
    }

    # Analyze each result file
    for filepath in sorted(result_files):
        result_data = load_result_file(filepath)

        # Extract experiment info
        filename = Path(filepath).name
        env = Path(filepath).parent.name
        seed = result_data.get('config', {}).get('seed', 'unknown')

        # Analyze methods
        methods = analyze_experiment(result_data)

        # Compare static vs dynamic
        comparisons = compare_static_vs_dynamic(methods)

        if comparisons:
            print(f"\n{'─' * 80}")
            print(f"Environment: {env.upper()}")
            print(f"File: {filename}")
            print(f"Seed: {seed}")
            print(f"{'─' * 80}")

            for algo, comp in comparisons.items():
                print(f"\n{algo.upper()}:")
                print(f"  Static QBound:  {comp['static']:.2f} (final 100 eps)")
                print(f"  Dynamic QBound: {comp['dynamic']:.2f} (final 100 eps)")

                if comp['dynamic_better']:
                    print(f"  ✓ DYNAMIC WINS by {comp['improvement']:.2f}%")
                    all_comparisons['dynamic_wins'] += 1
                else:
                    print(f"  ✗ Static wins by {-comp['improvement']:.2f}%")
                    all_comparisons['static_wins'] += 1

                all_comparisons['details'].append({
                    'env': env,
                    'seed': seed,
                    'algo': algo,
                    'static': comp['static'],
                    'dynamic': comp['dynamic'],
                    'dynamic_better': comp['dynamic_better'],
                    'improvement': comp['improvement']
                })

    # Print overall summary
    print(f"\n{'=' * 80}")
    print("OVERALL SUMMARY")
    print(f"{'=' * 80}")

    total = all_comparisons['dynamic_wins'] + all_comparisons['static_wins']
    if total > 0:
        print(f"\nTotal comparisons: {total}")
        print(f"Dynamic QBound wins: {all_comparisons['dynamic_wins']} ({all_comparisons['dynamic_wins']/total*100:.1f}%)")
        print(f"Static QBound wins: {all_comparisons['static_wins']} ({all_comparisons['static_wins']/total*100:.1f}%)")

        # Calculate average improvements
        dynamic_improvements = [d['improvement'] for d in all_comparisons['details'] if d['dynamic_better']]
        static_improvements = [-d['improvement'] for d in all_comparisons['details'] if not d['dynamic_better']]

        if dynamic_improvements:
            print(f"\nWhen dynamic wins, average improvement: {np.mean(dynamic_improvements):.2f}%")
        if static_improvements:
            print(f"When static wins, average improvement: {np.mean(static_improvements):.2f}%")

        # Group by environment
        print(f"\n{'─' * 80}")
        print("BY ENVIRONMENT:")
        print(f"{'─' * 80}")

        envs = set(d['env'] for d in all_comparisons['details'])
        for env in sorted(envs):
            env_details = [d for d in all_comparisons['details'] if d['env'] == env]
            dynamic_wins = sum(1 for d in env_details if d['dynamic_better'])
            static_wins = len(env_details) - dynamic_wins

            print(f"\n{env.upper()}:")
            print(f"  Dynamic wins: {dynamic_wins}/{len(env_details)}")
            print(f"  Static wins: {static_wins}/{len(env_details)}")

            # Show which algorithms
            for algo in ['dqn', 'ddqn', 'dueling']:
                algo_details = [d for d in env_details if d['algo'] == algo]
                if algo_details:
                    algo_dynamic_wins = sum(1 for d in algo_details if d['dynamic_better'])
                    print(f"    {algo.upper()}: Dynamic wins {algo_dynamic_wins}/{len(algo_details)}")

    else:
        print("\nNo valid comparisons found!")

    print(f"\n{'=' * 80}\n")

if __name__ == '__main__':
    main()
