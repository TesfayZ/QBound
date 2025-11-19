#!/usr/bin/env python3
"""
Analyze QBound effect on:
1. Positive rewards (CartPole - dense positive)
2. Sparse rewards (GridWorld, FrozenLake, MountainCar, Acrobot)

Compare with negative reward analysis to understand full picture.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List

def analyze_cartpole():
    """Analyze positive reward environment (CartPole)."""
    print("=" * 80)
    print("POSITIVE REWARDS: CartPole Analysis")
    print("=" * 80)
    print()
    print("Environment: CartPole-v1")
    print("  Reward: +1 per timestep (dense positive)")
    print("  Max episode length: 500 steps")
    print("  Theoretical Q_max: 99.34 (geometric sum with γ=0.99)")
    print()

    results_dir = Path('/root/projects/QBound/results/cartpole')

    # Find all CartPole DQN results
    dqn_files = list(results_dir.glob('dqn_full_qbound_seed*.json'))
    dueling_files = list(results_dir.glob('dueling_full_qbound_seed*.json'))

    if not dqn_files:
        print("No CartPole DQN results found!")
        return

    print(f"Found {len(dqn_files)} DQN result files")
    print(f"Found {len(dueling_files)} Dueling DQN result files")
    print()

    # Analyze DQN
    print("DQN Results:")
    print("-" * 60)

    dqn_summary = {'baseline': [], 'static': [], 'dynamic': [], 'seeds': []}

    for file in sorted(dqn_files):
        with open(file, 'r') as f:
            data = json.load(f)

        seed = data['config']['seed']
        dqn_summary['seeds'].append(seed)

        # Get performance for each method
        baseline = np.mean(data['training']['baseline']['rewards'][-100:])
        static = np.mean(data['training']['static_qbound']['rewards'][-100:])
        dqn_summary['baseline'].append(baseline)
        dqn_summary['static'].append(static)

        # Check if dynamic exists
        if 'dynamic_qbound' in data['training']:
            dynamic = np.mean(data['training']['dynamic_qbound']['rewards'][-100:])
            dqn_summary['dynamic'].append(dynamic)
        else:
            dqn_summary['dynamic'].append(None)

        static_deg = ((static / baseline) - 1) * 100 if baseline != 0 else 0
        print(f"  Seed {seed}: Baseline={baseline:6.1f}, Static={static:6.1f}, "
              f"Change={static_deg:+5.1f}%")

    print()

    # Calculate statistics
    baseline_arr = np.array(dqn_summary['baseline'])
    static_arr = np.array(dqn_summary['static'])
    degradations = ((static_arr / baseline_arr) - 1) * 100

    print(f"DQN Summary:")
    print(f"  Baseline: {baseline_arr.mean():.1f} ± {baseline_arr.std():.1f}")
    print(f"  Static QBound: {static_arr.mean():.1f} ± {static_arr.std():.1f}")
    print(f"  Mean change: {degradations.mean():+.1f}% ± {degradations.std():.1f}%")
    print(f"  Range: {degradations.min():+.1f}% to {degradations.max():+.1f}%")

    if degradations.mean() > 5:
        print(f"  ✓ QBound HELPS! (+{degradations.mean():.1f}% improvement)")
    elif degradations.mean() < -5:
        print(f"  ✗ QBound HURTS! ({degradations.mean():.1f}% degradation)")
    else:
        print(f"  ~ Neutral effect ({degradations.mean():+.1f}%)")

    print()

    # Analyze Dueling DQN if available
    if dueling_files:
        print("Dueling DQN Results:")
        print("-" * 60)

        dueling_summary = {'baseline': [], 'static': [], 'seeds': []}

        for file in sorted(dueling_files):
            with open(file, 'r') as f:
                data = json.load(f)

            seed = data['config']['seed']
            dueling_summary['seeds'].append(seed)

            baseline = np.mean(data['training']['dueling_dqn']['rewards'][-100:])
            static = np.mean(data['training']['static_qbound_dueling_dqn']['rewards'][-100:])
            dueling_summary['baseline'].append(baseline)
            dueling_summary['static'].append(static)

            static_deg = ((static / baseline) - 1) * 100 if baseline != 0 else 0
            print(f"  Seed {seed}: Baseline={baseline:6.1f}, Static={static:6.1f}, "
                  f"Change={static_deg:+5.1f}%")

        print()

        baseline_arr = np.array(dueling_summary['baseline'])
        static_arr = np.array(dueling_summary['static'])
        degradations = ((static_arr / baseline_arr) - 1) * 100

        print(f"Dueling DQN Summary:")
        print(f"  Baseline: {baseline_arr.mean():.1f} ± {baseline_arr.std():.1f}")
        print(f"  Static QBound: {static_arr.mean():.1f} ± {static_arr.std():.1f}")
        print(f"  Mean change: {degradations.mean():+.1f}% ± {degradations.std():.1f}%")
        print(f"  Range: {degradations.min():+.1f}% to {degradations.max():+.1f}%")

def analyze_sparse():
    """Analyze sparse reward environments."""
    print("\n" + "=" * 80)
    print("SPARSE REWARDS: Analysis")
    print("=" * 80)
    print()

    environments = {
        'gridworld': {
            'path': 'results/gridworld',
            'pattern': 'dqn_static_qbound_seed*.json',
            'reward': '+1 at goal only (sparse terminal)',
            'name': 'GridWorld'
        },
        'frozenlake': {
            'path': 'results/frozenlake',
            'pattern': 'dqn_static_qbound_seed*.json',
            'reward': '+1 at goal only (sparse terminal)',
            'name': 'FrozenLake-v1'
        },
        'mountaincar': {
            'path': 'results/mountaincar',
            'pattern': 'dqn_static_qbound_seed*.json',
            'reward': '-1 until goal (state-dependent negative)',
            'name': 'MountainCar-v0'
        },
        'acrobot': {
            'path': 'results/acrobot',
            'pattern': 'dqn_static_qbound_seed*.json',
            'reward': '-1 until swing-up (state-dependent negative)',
            'name': 'Acrobot-v1'
        }
    }

    for env_key, env_info in environments.items():
        print(f"\n{env_info['name']}:")
        print(f"  Reward structure: {env_info['reward']}")
        print("-" * 60)

        results_dir = Path(f"/root/projects/QBound/{env_info['path']}")
        files = list(results_dir.glob(env_info['pattern']))

        if not files:
            print(f"  No results found!")
            continue

        baseline_perfs = []
        static_perfs = []
        seeds = []

        for file in sorted(files):
            with open(file, 'r') as f:
                data = json.load(f)

            seed = data['config']['seed']
            seeds.append(seed)

            # Get baseline and static QBound performance
            baseline = np.mean(data['training']['baseline']['rewards'][-100:])
            static = np.mean(data['training']['static_qbound']['rewards'][-100:])

            baseline_perfs.append(baseline)
            static_perfs.append(static)

            deg = ((static / baseline) - 1) * 100 if baseline != 0 else 0
            print(f"  Seed {seed}: Baseline={baseline:7.2f}, Static={static:7.2f}, Change={deg:+6.1f}%")

        if baseline_perfs:
            baseline_arr = np.array(baseline_perfs)
            static_arr = np.array(static_perfs)
            degradations = ((static_arr / baseline_arr) - 1) * 100

            print(f"\n  Summary:")
            print(f"    Baseline: {baseline_arr.mean():.2f} ± {baseline_arr.std():.2f}")
            print(f"    Static QBound: {static_arr.mean():.2f} ± {static_arr.std():.2f}")
            print(f"    Mean change: {degradations.mean():+.1f}% ± {degradations.std():.1f}%")
            print(f"    Range: {degradations.min():+.1f}% to {degradations.max():+.1f}%")

            if degradations.mean() > 5:
                print(f"    ✓ QBound HELPS! (+{degradations.mean():.1f}% improvement)")
            elif degradations.mean() < -5:
                print(f"    ✗ QBound HURTS! ({degradations.mean():.1f}% degradation)")
            else:
                print(f"    ~ Neutral effect ({degradations.mean():+.1f}%)")

def summary():
    """Print summary comparison."""
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY: QBound Across All Reward Structures")
    print("=" * 80)
    print()

    print("Expected patterns:")
    print()
    print("1. POSITIVE REWARDS (CartPole):")
    print("   Theory: Q-values can overestimate → QBound should help")
    print("   Expectation: Improvement or neutral")
    print()
    print("2. NEGATIVE REWARDS (Pendulum):")
    print("   Theory: Depends on clipping mechanism")
    print("   - Hard clipping (DQN): Degradation")
    print("   - Two-level (DDPG/TD3): Improvement")
    print()
    print("3. SPARSE REWARDS (GridWorld, FrozenLake):")
    print("   Theory: Sparse terminal rewards → hard to learn")
    print("   Expectation: QBound may help or hurt depending on sparsity")
    print()
    print("4. STATE-DEPENDENT NEGATIVE (MountainCar, Acrobot):")
    print("   Theory: Similar to dense negative → likely degradation")
    print("   Expectation: Degradation with hard clipping")

if __name__ == '__main__':
    analyze_cartpole()
    analyze_sparse()
    summary()
