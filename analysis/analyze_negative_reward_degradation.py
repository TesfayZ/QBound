#!/usr/bin/env python3
"""
Analyze why QBound degrades performance on negative reward environments (Pendulum).

Theoretically, with negative rewards, Q-values should naturally stay ≤ 0,
making QBound redundant. But experiments show -3% to -47% decline.

This script investigates:
1. Which bound is violated (Q_min or Q_max)?
2. Violation patterns across seeds
3. Q-value distributions
4. Performance correlation with violations
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
import glob

def load_pendulum_results():
    """Load all Pendulum experiment results."""
    results = {
        'dqn': [],
        'ddpg': [],
        'td3': [],
        'ppo': []
    }

    results_dir = Path('/root/projects/QBound/results/pendulum')

    # Load DQN results (seeds 42-46)
    for seed in [42, 43, 44, 45, 46]:
        pattern = f"dqn_full_qbound_seed{seed}_*.json"
        files = list(results_dir.glob(pattern))
        if files:
            # Get the timestamped one (not in_progress)
            file = [f for f in files if 'in_progress' not in str(f)]
            if file:
                with open(file[0], 'r') as f:
                    results['dqn'].append((seed, json.load(f)))

    # Load DDPG results
    for seed in [42, 43, 44, 45, 46]:
        pattern = f"ddpg_full_qbound_seed{seed}_*.json"
        files = list(results_dir.glob(pattern))
        if files:
            file = [f for f in files if 'in_progress' not in str(f)]
            if file:
                with open(file[0], 'r') as f:
                    results['ddpg'].append((seed, json.load(f)))

    # Load TD3 results
    for seed in [42, 43, 44, 45, 46]:
        pattern = f"td3_full_qbound_seed{seed}_*.json"
        files = list(results_dir.glob(pattern))
        if files:
            file = [f for f in files if 'in_progress' not in str(f)]
            if file:
                with open(file[0], 'r') as f:
                    results['td3'].append((seed, json.load(f)))

    # Load PPO results
    for seed in [42, 43, 44, 45, 46]:
        pattern = f"ppo_full_qbound_seed{seed}_*.json"
        files = list(results_dir.glob(pattern))
        if files:
            file = [f for f in files if 'in_progress' not in str(f)]
            if file:
                with open(file[0], 'r') as f:
                    results['ppo'].append((seed, json.load(f)))

    return results

def analyze_violations(data: dict, method: str, algo: str) -> dict:
    """Analyze violation patterns for a specific method."""
    # Map method names to actual keys in data
    key_map = {
        'dqn': {
            'baseline': 'dqn',
            'static_qbound': 'static_qbound_dqn',
            'dynamic_qbound': 'dynamic_qbound_dqn'
        },
        'ddpg': {
            'baseline': 'baseline',
            'static_qbound': 'static_qbound',
            'dynamic_qbound': 'dynamic_qbound'
        },
        'td3': {
            'baseline': 'baseline',
            'static_qbound': 'static_qbound',
            'dynamic_qbound': 'dynamic_qbound'
        },
        'ppo': {
            'baseline': 'baseline',
            'static_qbound': 'static_qbound',
            'dynamic_qbound': 'dynamic_qbound'
        }
    }

    actual_key = key_map.get(algo, {}).get(method, method)

    if actual_key not in data['training']:
        return None

    method_data = data['training'][actual_key]

    # Check if violations exist
    if 'violations' not in method_data or not method_data['violations']:
        return {
            'has_violations': False,
            'method': method
        }

    violations = method_data['violations']

    # Check for the actual violation structure (next_q_violate_max_rate, etc.)
    if 'mean' in violations:
        mean_viol = violations['mean']
        final_100_viol = violations.get('final_100', {})

        return {
            'has_violations': True,
            'method': method,
            'type': 'dqn_style',
            'mean_upper_rate': mean_viol.get('next_q_violate_max_rate', 0),
            'mean_lower_rate': mean_viol.get('next_q_violate_min_rate', 0),
            'mean_upper_magnitude': mean_viol.get('violation_magnitude_max_next', 0),
            'mean_lower_magnitude': mean_viol.get('violation_magnitude_min_next', 0),
            'final_100_upper_rate': final_100_viol.get('next_q_violate_max_rate', 0),
            'final_100_lower_rate': final_100_viol.get('next_q_violate_min_rate', 0),
            'qbound_min': mean_viol.get('qbound_min', 0),
            'qbound_max': mean_viol.get('qbound_max', 0),
        }

    return {
        'has_violations': False,
        'method': method
    }

def get_performance(data: dict, method: str, algo: str) -> float:
    """Get final performance (last 100 episodes mean reward)."""
    # Map method names to actual keys in data
    key_map = {
        'dqn': {
            'baseline': 'dqn',
            'static_qbound': 'static_qbound_dqn',
            'dynamic_qbound': 'dynamic_qbound_dqn'
        },
        'ddpg': {
            'baseline': 'baseline',
            'static_qbound': 'static_qbound',
            'dynamic_qbound': 'dynamic_qbound'
        },
        'td3': {
            'baseline': 'baseline',
            'static_qbound': 'static_qbound',
            'dynamic_qbound': 'dynamic_qbound'
        },
        'ppo': {
            'baseline': 'baseline',
            'static_qbound': 'static_qbound',
            'dynamic_qbound': 'dynamic_qbound'
        }
    }

    actual_key = key_map.get(algo, {}).get(method, method)

    if actual_key not in data['training']:
        return None

    rewards = data['training'][actual_key]['rewards']
    return np.mean(rewards[-100:])

def print_analysis():
    """Print comprehensive analysis of negative reward degradation."""
    print("=" * 80)
    print("ANALYSIS: Why QBound Degrades on Negative Reward Environments")
    print("=" * 80)
    print()

    results = load_pendulum_results()

    for algo in ['dqn', 'ddpg', 'td3', 'ppo']:
        if not results[algo]:
            continue

        print(f"\n{'='*80}")
        print(f"{algo.upper()} - Pendulum (Negative Rewards: -16.2 per step)")
        print(f"{'='*80}")
        print(f"Theoretical Q_bounds: Q_min=-3240, Q_max=0 (all Q-values should be ≤ 0)")
        print()

        # Analyze each seed
        for seed, data in results[algo]:
            print(f"\n--- Seed {seed} ---")

            # Get performance for all methods
            baseline_perf = get_performance(data, 'baseline', algo)
            static_perf = get_performance(data, 'static_qbound', algo)
            dynamic_perf = get_performance(data, 'dynamic_qbound', algo)

            print(f"Performance (mean reward, last 100 episodes):")
            if baseline_perf is not None:
                print(f"  Baseline:       {baseline_perf:.2f}")
            if static_perf is not None and baseline_perf is not None:
                print(f"  Static QBound:  {static_perf:.2f} ({((static_perf/baseline_perf - 1)*100):.1f}%)")
            if dynamic_perf is not None and baseline_perf is not None:
                print(f"  Dynamic QBound: {dynamic_perf:.2f} ({((dynamic_perf/baseline_perf - 1)*100):.1f}%)")

            # Analyze violations for static QBound
            print(f"\nStatic QBound Violations:")
            static_viol = analyze_violations(data, 'static_qbound', algo)
            if static_viol and static_viol['has_violations']:
                print(f"  Type: {static_viol['type']}")
                print(f"  Mean upper (Q > Q_max=0) rate: {static_viol['mean_upper_rate']:.2%}")
                print(f"  Mean lower (Q < Q_min=-3240) rate: {static_viol['mean_lower_rate']:.2%}")
                print(f"  Final 100 upper rate: {static_viol['final_100_upper_rate']:.2%}")
                print(f"  Final 100 lower rate: {static_viol['final_100_lower_rate']:.2%}")
                print(f"  Mean upper magnitude: {static_viol['mean_upper_magnitude']:.2f}")
                print(f"  Mean lower magnitude: {static_viol['mean_lower_magnitude']:.2f}")

                # Key insight
                if static_viol['mean_lower_rate'] > 0.01:
                    print(f"  ⚠️  Q_min violations detected! Q-values going below -3240")
                if static_viol['mean_upper_rate'] > 0.01:
                    print(f"  ⚠️  Q_max violations detected! Q-values going above 0 (should be impossible!)")
            else:
                print(f"  No violation data available")

            # Analyze violations for dynamic QBound
            if dynamic_perf:
                print(f"\nDynamic QBound Violations:")
                dynamic_viol = analyze_violations(data, 'dynamic_qbound', algo)
                if dynamic_viol and dynamic_viol['has_violations']:
                    print(f"  Type: {dynamic_viol['type']}")
                    print(f"  Mean upper rate: {dynamic_viol['mean_upper_rate']:.2%}")
                    print(f"  Mean lower rate: {dynamic_viol['mean_lower_rate']:.2%}")
                    print(f"  Final 100 upper rate: {dynamic_viol['final_100_upper_rate']:.2%}")
                    print(f"  Final 100 lower rate: {dynamic_viol['final_100_lower_rate']:.2%}")
                else:
                    print(f"  No violation data available")

    # Summary across all seeds
    print(f"\n{'='*80}")
    print("SUMMARY: Performance Degradation Patterns")
    print(f"{'='*80}")

    for algo in ['dqn', 'ddpg', 'td3', 'ppo']:
        if not results[algo]:
            continue

        print(f"\n{algo.upper()}:")

        static_degrades = []
        dynamic_degrades = []

        for seed, data in results[algo]:
            baseline = get_performance(data, 'baseline', algo)
            static = get_performance(data, 'static_qbound', algo)
            dynamic = get_performance(data, 'dynamic_qbound', algo)

            if baseline is not None and static is not None:
                static_change = ((static / baseline) - 1) * 100
                static_degrades.append(static_change)

            if dynamic is not None and baseline is not None:
                dynamic_change = ((dynamic / baseline) - 1) * 100
                dynamic_degrades.append(dynamic_change)

        if static_degrades:
            print(f"  Static QBound:  {np.mean(static_degrades):.1f}% ± {np.std(static_degrades):.1f}% (range: {min(static_degrades):.1f}% to {max(static_degrades):.1f}%)")
        if dynamic_degrades:
            print(f"  Dynamic QBound: {np.mean(dynamic_degrades):.1f}% ± {np.std(dynamic_degrades):.1f}% (range: {min(dynamic_degrades):.1f}% to {max(dynamic_degrades):.1f}%)")

if __name__ == '__main__':
    print_analysis()
