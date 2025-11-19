#!/usr/bin/env python3
"""
Analyze if negative reward degradation affects all algorithms or just DQN.

Key question: Do algorithms with built-in overestimation mitigation
(DDQN, TD3, PPO) also suffer from QBound on negative rewards?

Analysis:
1. DQN - No overestimation mitigation
2. DDQN (Double DQN) - Uses target network selection to reduce overestimation
3. TD3 - Clipped double Q-learning for actor-critic
4. DDPG - Actor-critic (can overestimate)
5. PPO - Policy gradient (no Q-values, uses value function V(s))
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

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

def get_performance(data: dict, method: str, algo: str) -> float:
    """Get final performance (last 100 episodes mean reward)."""
    # Map method names to actual keys in data
    key_map = {
        'dqn': {
            'baseline': 'dqn',
            'baseline_double': 'double_dqn',
            'static_qbound': 'static_qbound_dqn',
            'static_qbound_double': 'static_qbound_double_dqn',
            'dynamic_qbound': 'dynamic_qbound_dqn',
            'dynamic_qbound_double': 'dynamic_qbound_double_dqn',
        },
        'ddpg': {
            'baseline': 'baseline',
            'static_qbound': 'static_soft_qbound',
            'dynamic_qbound': 'dynamic_soft_qbound'
        },
        'td3': {
            'baseline': 'baseline',
            'static_qbound': 'static_soft_qbound',
            'dynamic_qbound': 'dynamic_soft_qbound'
        },
        'ppo': {
            'baseline': 'baseline',
            'static_qbound': 'static_soft_qbound',
            'dynamic_qbound': 'dynamic_soft_qbound'
        }
    }

    actual_key = key_map.get(algo, {}).get(method, method)

    if actual_key not in data['training']:
        return None

    rewards = data['training'][actual_key]['rewards']
    return np.mean(rewards[-100:])

def analyze_all_algorithms():
    """Analyze degradation across all algorithms."""
    print("=" * 80)
    print("ANALYSIS: Does Negative Reward Degradation Affect All Algorithms?")
    print("=" * 80)
    print()

    results = load_pendulum_results()

    # Storage for summary
    summary = {}

    for algo in ['dqn', 'ddpg', 'td3', 'ppo']:
        if not results[algo]:
            continue

        print(f"\n{'='*80}")
        print(f"{algo.upper()} Analysis")
        print(f"{'='*80}")

        # Check what methods are available
        seed, data = results[algo][0]
        available_methods = list(data['training'].keys())
        print(f"Available methods: {available_methods}")
        print()

        # Analyze based on algorithm
        if algo == 'dqn':
            # DQN has: dqn, double_dqn, static_qbound_dqn, static_qbound_double_dqn, etc.
            methods_to_check = [
                ('DQN (no overestimation mitigation)', 'baseline', 'static_qbound'),
                ('DDQN (double Q-learning)', 'baseline_double', 'static_qbound_double')
            ]
        else:
            # DDPG/TD3/PPO: baseline, static_soft_qbound, dynamic_soft_qbound
            methods_to_check = [
                (f'{algo.upper()} (soft QBound)', 'baseline', 'static_qbound')
            ]

        for desc, baseline_key, qbound_key in methods_to_check:
            print(f"\n{desc}:")
            print("-" * 60)

            baseline_perfs = []
            qbound_perfs = []
            degradations = []

            for seed, data in results[algo]:
                baseline = get_performance(data, baseline_key, algo)
                qbound = get_performance(data, qbound_key, algo)

                if baseline is not None and qbound is not None:
                    degradation = ((qbound / baseline) - 1) * 100
                    baseline_perfs.append(baseline)
                    qbound_perfs.append(qbound)
                    degradations.append(degradation)

                    print(f"  Seed {seed}: Baseline={baseline:7.2f}, QBound={qbound:7.2f}, "
                          f"Degradation={degradation:+6.1f}%")

            if degradations:
                mean_deg = np.mean(degradations)
                std_deg = np.std(degradations)
                print()
                print(f"  Summary: {mean_deg:+.1f}% ± {std_deg:.1f}% "
                      f"(range: {min(degradations):+.1f}% to {max(degradations):+.1f}%)")

                # Store for summary
                if algo not in summary:
                    summary[algo] = {}
                summary[algo][desc] = {
                    'mean': mean_deg,
                    'std': std_deg,
                    'range': (min(degradations), max(degradations)),
                    'count': len(degradations)
                }

    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY: Degradation Across All Algorithms")
    print("=" * 80)
    print()

    print(f"{'Algorithm':<30} {'Mean Degradation':<20} {'Range':<25} {'N Seeds'}")
    print("-" * 95)

    for algo in ['dqn', 'ddpg', 'td3', 'ppo']:
        if algo in summary:
            for desc, stats in summary[algo].items():
                mean_str = f"{stats['mean']:+.1f}% ± {stats['std']:.1f}%"
                range_str = f"{stats['range'][0]:+.1f}% to {stats['range'][1]:+.1f}%"
                print(f"{desc:<30} {mean_str:<20} {range_str:<25} {stats['count']}")

    # Analysis
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    print()

    if 'dqn' in summary:
        if 'DQN (no overestimation mitigation)' in summary['dqn']:
            dqn_deg = summary['dqn']['DQN (no overestimation mitigation)']['mean']
            print(f"1. DQN (no mitigation): {dqn_deg:+.1f}% degradation")

        if 'DDQN (double Q-learning)' in summary['dqn']:
            ddqn_deg = summary['dqn']['DDQN (double Q-learning)']['mean']
            print(f"2. DDQN (with mitigation): {ddqn_deg:+.1f}% degradation")

            if 'DQN (no overestimation mitigation)' in summary['dqn']:
                dqn_deg = summary['dqn']['DQN (no overestimation mitigation)']['mean']
                print(f"   → DDQN vs DQN: {abs(ddqn_deg - dqn_deg):.1f}% difference")
                if abs(ddqn_deg) < abs(dqn_deg):
                    print("   → DDQN degrades LESS (double Q-learning helps!)")
                elif abs(ddqn_deg) > abs(dqn_deg):
                    print("   → DDQN degrades MORE (surprising!)")
                else:
                    print("   → Similar degradation (no difference)")

    if 'td3' in summary:
        for desc, stats in summary['td3'].items():
            print(f"3. TD3 (clipped double Q): {stats['mean']:+.1f}% degradation")

    if 'ddpg' in summary:
        for desc, stats in summary['ddpg'].items():
            print(f"4. DDPG (actor-critic): {stats['mean']:+.1f}% degradation")

    if 'ppo' in summary:
        for desc, stats in summary['ppo'].items():
            print(f"5. PPO (policy gradient, V(s) not Q(s,a)): {stats['mean']:+.1f}% degradation")

    print()
    print("INTERPRETATION:")
    print("-" * 80)

    # Check if algorithms with mitigation perform better
    has_mitigation = []
    no_mitigation = []

    if 'dqn' in summary:
        if 'DQN (no overestimation mitigation)' in summary['dqn']:
            no_mitigation.append(('DQN', summary['dqn']['DQN (no overestimation mitigation)']['mean']))
        if 'DDQN (double Q-learning)' in summary['dqn']:
            has_mitigation.append(('DDQN', summary['dqn']['DDQN (double Q-learning)']['mean']))

    if 'td3' in summary:
        for desc, stats in summary['td3'].items():
            has_mitigation.append(('TD3', stats['mean']))

    if 'ddpg' in summary:
        for desc, stats in summary['ddpg'].items():
            no_mitigation.append(('DDPG', stats['mean']))

    if has_mitigation and no_mitigation:
        avg_mitigation = np.mean([deg for _, deg in has_mitigation])
        avg_no_mitigation = np.mean([deg for _, deg in no_mitigation])

        print(f"\nAlgorithms WITH overestimation mitigation (DDQN, TD3): {avg_mitigation:+.1f}%")
        print(f"Algorithms WITHOUT mitigation (DQN, DDPG): {avg_no_mitigation:+.1f}%")

        if abs(avg_mitigation) < abs(avg_no_mitigation):
            print("\n✓ Algorithms with built-in mitigation degrade LESS!")
            print("  → Double Q-learning / clipped Q helps reduce QBound harm")
        elif abs(avg_mitigation) > abs(avg_no_mitigation):
            print("\n✗ Algorithms with built-in mitigation degrade MORE!")
            print("  → Surprising! QBound may interfere with existing mitigation")
        else:
            print("\n~ Similar degradation regardless of mitigation")
            print("  → QBound's clipping affects all algorithms equally")

if __name__ == '__main__':
    analyze_all_algorithms()
