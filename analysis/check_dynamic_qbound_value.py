#!/usr/bin/env python3
"""
Analysis: Was Dynamic QBound Ever Beneficial?

This script examines all experimental results to determine if dynamic QBound
ever showed positive improvements over static QBound.

Focus areas:
1. CartPole (positive time-step rewards)
2. Pendulum (negative time-step rewards) - buggy and fixed versions
3. All algorithms: DQN, DDQN, Dueling, DDPG, TD3, PPO
"""

import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple

# Add src to path
sys.path.insert(0, '/root/projects/QBound/src')

def load_json_safe(filepath: Path) -> Dict:
    """Safely load JSON file."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def get_final_performance(rewards: List[float], window: int = 100) -> float:
    """Get average performance over final window."""
    if len(rewards) < window:
        return sum(rewards) / len(rewards) if rewards else 0.0
    return sum(rewards[-window:]) / window

def analyze_3way_results(filepath: Path, env_name: str) -> Dict:
    """Analyze 3-way comparison (baseline, static, dynamic)."""
    data = load_json_safe(filepath)
    if not data or 'training' not in data:
        return None

    results = {}

    # Check for DQN variants
    for variant in ['dqn', 'ddqn']:
        baseline_key = f'baseline_{variant}'
        static_key = f'static_qbound_{variant}'
        dynamic_key = f'dynamic_qbound_{variant}'

        if all(k in data['training'] for k in [baseline_key, static_key, dynamic_key]):
            baseline_perf = get_final_performance(data['training'][baseline_key]['rewards'])
            static_perf = get_final_performance(data['training'][static_key]['rewards'])
            dynamic_perf = get_final_performance(data['training'][dynamic_key]['rewards'])

            results[variant] = {
                'baseline': baseline_perf,
                'static': static_perf,
                'dynamic': dynamic_perf,
                'static_improvement': static_perf - baseline_perf,
                'dynamic_improvement': dynamic_perf - baseline_perf,
                'dynamic_vs_static': dynamic_perf - static_perf
            }

    return results

def analyze_actor_critic_results(filepath: Path, algo_name: str) -> Dict:
    """Analyze DDPG/TD3/PPO results (baseline, static, dynamic)."""
    data = load_json_safe(filepath)
    if not data or 'training' not in data:
        return None

    results = {}

    # Check for methods
    if all(k in data['training'] for k in ['baseline', 'static_qbound', 'dynamic_qbound']):
        baseline_perf = get_final_performance(data['training']['baseline']['rewards'])
        static_perf = get_final_performance(data['training']['static_qbound']['rewards'])
        dynamic_perf = get_final_performance(data['training']['dynamic_qbound']['rewards'])

        results[algo_name] = {
            'baseline': baseline_perf,
            'static': static_perf,
            'dynamic': dynamic_perf,
            'static_improvement': static_perf - baseline_perf,
            'dynamic_improvement': dynamic_perf - baseline_perf,
            'dynamic_vs_static': dynamic_perf - static_perf
        }

    return results

def main():
    print("=" * 80)
    print("DYNAMIC QBOUND VALUE ANALYSIS")
    print("=" * 80)
    print()

    results_dir = Path('/root/projects/QBound/results')

    # Track all findings
    all_results = {}

    # ========================================================================
    # 1. CARTPOLE (Positive Time-Step Rewards)
    # ========================================================================
    print("1. CARTPOLE (Positive Time-Step Rewards)")
    print("-" * 80)

    # DQN experiments
    cartpole_dqn_files = list((results_dir / 'cartpole').glob('dqn_full_qbound_seed*.json'))
    if cartpole_dqn_files:
        print(f"\n  Found {len(cartpole_dqn_files)} CartPole DQN seed experiments:")
        for filepath in sorted(cartpole_dqn_files):
            seed = filepath.stem.split('_seed')[1].split('_')[0]
            res = analyze_3way_results(filepath, 'cartpole')
            if res:
                all_results[f'cartpole_dqn_seed{seed}'] = res
                for variant, metrics in res.items():
                    print(f"\n  Seed {seed} - {variant.upper()}:")
                    print(f"    Baseline:  {metrics['baseline']:7.2f}")
                    print(f"    Static:    {metrics['static']:7.2f} (Δ={metrics['static_improvement']:+7.2f})")
                    print(f"    Dynamic:   {metrics['dynamic']:7.2f} (Δ={metrics['dynamic_improvement']:+7.2f})")
                    print(f"    Dynamic vs Static: {metrics['dynamic_vs_static']:+7.2f}")
                    if metrics['dynamic_vs_static'] > 0:
                        print(f"    ✓ DYNAMIC WINS!")

    # Dueling DQN experiments
    cartpole_dueling_files = list((results_dir / 'cartpole').glob('dueling_full_qbound_seed*.json'))
    if cartpole_dueling_files:
        print(f"\n  Found {len(cartpole_dueling_files)} CartPole Dueling DQN seed experiments:")
        for filepath in sorted(cartpole_dueling_files):
            seed = filepath.stem.split('_seed')[1].split('_')[0]
            res = analyze_3way_results(filepath, 'cartpole_dueling')
            if res:
                all_results[f'cartpole_dueling_seed{seed}'] = res
                for variant, metrics in res.items():
                    print(f"\n  Seed {seed} - Dueling {variant.upper()}:")
                    print(f"    Baseline:  {metrics['baseline']:7.2f}")
                    print(f"    Static:    {metrics['static']:7.2f} (Δ={metrics['static_improvement']:+7.2f})")
                    print(f"    Dynamic:   {metrics['dynamic']:7.2f} (Δ={metrics['dynamic_improvement']:+7.2f})")
                    print(f"    Dynamic vs Static: {metrics['dynamic_vs_static']:+7.2f}")
                    if metrics['dynamic_vs_static'] > 0:
                        print(f"    ✓ DYNAMIC WINS!")

    # 6-way comparison (legacy)
    cartpole_6way = results_dir / 'cartpole' / '6way_comparison_20251028_104649.json'
    if cartpole_6way.exists():
        print(f"\n  Legacy 6-way comparison:")
        res = analyze_3way_results(cartpole_6way, 'cartpole_6way')
        if res:
            all_results['cartpole_6way'] = res
            for variant, metrics in res.items():
                print(f"\n  6-way - {variant.upper()}:")
                print(f"    Baseline:  {metrics['baseline']:7.2f}")
                print(f"    Static:    {metrics['static']:7.2f} (Δ={metrics['static_improvement']:+7.2f})")
                print(f"    Dynamic:   {metrics['dynamic']:7.2f} (Δ={metrics['dynamic_improvement']:+7.2f})")
                print(f"    Dynamic vs Static: {metrics['dynamic_vs_static']:+7.2f}")
                if metrics['dynamic_vs_static'] > 0:
                    print(f"    ✓ DYNAMIC WINS!")

    # ========================================================================
    # 2. PENDULUM (Negative Time-Step Rewards) - BUGGY VERSIONS
    # ========================================================================
    print("\n\n2. PENDULUM (Negative Rewards) - BUGGY DYNAMIC QBOUND")
    print("-" * 80)
    print("  (These had bugs in dynamic QBound implementation for experience replay)")

    buggy_dir = results_dir / 'pendulum' / 'backup_buggy_dynamic_20251114_061928'
    if buggy_dir.exists():
        # DQN buggy results
        buggy_dqn_files = list(buggy_dir.glob('dqn_full_qbound_seed*.json'))
        if buggy_dqn_files:
            print(f"\n  Found {len(buggy_dqn_files)} buggy Pendulum DQN experiments:")
            for filepath in sorted(buggy_dqn_files):
                seed = filepath.stem.split('_seed')[1].split('_')[0]
                res = analyze_3way_results(filepath, 'pendulum_dqn_buggy')
                if res:
                    all_results[f'pendulum_dqn_buggy_seed{seed}'] = res
                    for variant, metrics in res.items():
                        print(f"\n  Buggy Seed {seed} - {variant.upper()}:")
                        print(f"    Baseline:  {metrics['baseline']:7.2f}")
                        print(f"    Static:    {metrics['static']:7.2f} (Δ={metrics['static_improvement']:+7.2f})")
                        print(f"    Dynamic:   {metrics['dynamic']:7.2f} (Δ={metrics['dynamic_improvement']:+7.2f})")
                        print(f"    Dynamic vs Static: {metrics['dynamic_vs_static']:+7.2f}")
                        if metrics['dynamic_vs_static'] > 0:
                            print(f"    ✓ DYNAMIC WINS!")

        # DDPG buggy results
        buggy_ddpg_files = list(buggy_dir.glob('ddpg_full_qbound_seed*.json'))
        if buggy_ddpg_files:
            print(f"\n  Found {len(buggy_ddpg_files)} buggy Pendulum DDPG experiments:")
            for filepath in sorted(buggy_ddpg_files):
                seed = filepath.stem.split('_seed')[1].split('_')[0]
                res = analyze_actor_critic_results(filepath, 'ddpg')
                if res:
                    all_results[f'pendulum_ddpg_buggy_seed{seed}'] = res
                    for algo, metrics in res.items():
                        print(f"\n  Buggy Seed {seed} - {algo.upper()}:")
                        print(f"    Baseline:  {metrics['baseline']:7.2f}")
                        print(f"    Static:    {metrics['static']:7.2f} (Δ={metrics['static_improvement']:+7.2f})")
                        print(f"    Dynamic:   {metrics['dynamic']:7.2f} (Δ={metrics['dynamic_improvement']:+7.2f})")
                        print(f"    Dynamic vs Static: {metrics['dynamic_vs_static']:+7.2f}")
                        if metrics['dynamic_vs_static'] > 0:
                            print(f"    ✓ DYNAMIC WINS!")

        # TD3 buggy results
        buggy_td3_files = list(buggy_dir.glob('td3_full_qbound_seed*.json'))
        if buggy_td3_files:
            print(f"\n  Found {len(buggy_td3_files)} buggy Pendulum TD3 experiments:")
            for filepath in sorted(buggy_td3_files):
                seed = filepath.stem.split('_seed')[1].split('_')[0]
                res = analyze_actor_critic_results(filepath, 'td3')
                if res:
                    all_results[f'pendulum_td3_buggy_seed{seed}'] = res
                    for algo, metrics in res.items():
                        print(f"\n  Buggy Seed {seed} - {algo.upper()}:")
                        print(f"    Baseline:  {metrics['baseline']:7.2f}")
                        print(f"    Static:    {metrics['static']:7.2f} (Δ={metrics['static_improvement']:+7.2f})")
                        print(f"    Dynamic:   {metrics['dynamic']:7.2f} (Δ={metrics['dynamic_improvement']:+7.2f})")
                        print(f"    Dynamic vs Static: {metrics['dynamic_vs_static']:+7.2f}")
                        if metrics['dynamic_vs_static'] > 0:
                            print(f"    ✓ DYNAMIC WINS!")

    # ========================================================================
    # 3. PENDULUM (Negative Rewards) - FIXED VERSIONS (currently running)
    # ========================================================================
    print("\n\n3. PENDULUM (Negative Rewards) - FIXED DYNAMIC QBOUND")
    print("-" * 80)
    print("  (Fixed version with correct dynamic QBound in experience replay)")
    print("  Note: Some experiments may still be running...")

    # Check completed fixed experiments
    pendulum_dir = results_dir / 'pendulum'

    # DQN fixed
    fixed_dqn_files = [f for f in pendulum_dir.glob('dqn_full_qbound_seed*.json')
                       if not f.name.endswith('_in_progress.json')]
    if fixed_dqn_files:
        print(f"\n  Found {len(fixed_dqn_files)} completed Pendulum DQN experiments:")
        for filepath in sorted(fixed_dqn_files):
            seed = filepath.stem.split('_seed')[1].split('_')[0]
            res = analyze_3way_results(filepath, 'pendulum_dqn_fixed')
            if res:
                all_results[f'pendulum_dqn_fixed_seed{seed}'] = res
                for variant, metrics in res.items():
                    print(f"\n  Fixed Seed {seed} - {variant.upper()}:")
                    print(f"    Baseline:  {metrics['baseline']:7.2f}")
                    print(f"    Static:    {metrics['static']:7.2f} (Δ={metrics['static_improvement']:+7.2f})")
                    print(f"    Dynamic:   {metrics['dynamic']:7.2f} (Δ={metrics['dynamic_improvement']:+7.2f})")
                    print(f"    Dynamic vs Static: {metrics['dynamic_vs_static']:+7.2f}")
                    if metrics['dynamic_vs_static'] > 0:
                        print(f"    ✓ DYNAMIC WINS!")

    # DDPG fixed
    fixed_ddpg_files = [f for f in pendulum_dir.glob('ddpg_full_qbound_seed*.json')
                        if not f.name.endswith('_in_progress.json')]
    if fixed_ddpg_files:
        print(f"\n  Found {len(fixed_ddpg_files)} completed Pendulum DDPG experiments:")
        for filepath in sorted(fixed_ddpg_files):
            seed = filepath.stem.split('_seed')[1].split('_')[0]
            res = analyze_actor_critic_results(filepath, 'ddpg')
            if res:
                all_results[f'pendulum_ddpg_fixed_seed{seed}'] = res
                for algo, metrics in res.items():
                    print(f"\n  Fixed Seed {seed} - {algo.upper()}:")
                    print(f"    Baseline:  {metrics['baseline']:7.2f}")
                    print(f"    Static:    {metrics['static']:7.2f} (Δ={metrics['static_improvement']:+7.2f})")
                    print(f"    Dynamic:   {metrics['dynamic']:7.2f} (Δ={metrics['dynamic_improvement']:+7.2f})")
                    print(f"    Dynamic vs Static: {metrics['dynamic_vs_static']:+7.2f}")
                    if metrics['dynamic_vs_static'] > 0:
                        print(f"    ✓ DYNAMIC WINS!")

    # TD3 fixed
    fixed_td3_files = [f for f in pendulum_dir.glob('td3_full_qbound_seed*.json')
                       if not f.name.endswith('_in_progress.json')]
    if fixed_td3_files:
        print(f"\n  Found {len(fixed_td3_files)} completed Pendulum TD3 experiments:")
        for filepath in sorted(fixed_td3_files):
            seed = filepath.stem.split('_seed')[1].split('_')[0]
            res = analyze_actor_critic_results(filepath, 'td3')
            if res:
                all_results[f'pendulum_td3_fixed_seed{seed}'] = res
                for algo, metrics in res.items():
                    print(f"\n  Fixed Seed {seed} - {algo.upper()}:")
                    print(f"    Baseline:  {metrics['baseline']:7.2f}")
                    print(f"    Static:    {metrics['static']:7.2f} (Δ={metrics['static_improvement']:+7.2f})")
                    print(f"    Dynamic:   {metrics['dynamic']:7.2f} (Δ={metrics['dynamic_improvement']:+7.2f})")
                    print(f"    Dynamic vs Static: {metrics['dynamic_vs_static']:+7.2f}")
                    if metrics['dynamic_vs_static'] > 0:
                        print(f"    ✓ DYNAMIC WINS!")

    # PPO
    ppo_files = [f for f in (results_dir / 'ppo').glob('pendulum_*_full_qbound_seed*.json')
                 if not f.name.endswith('_in_progress.json')]
    if ppo_files:
        print(f"\n  Found {len(ppo_files)} completed Pendulum PPO experiments:")
        for filepath in sorted(ppo_files):
            seed = filepath.stem.split('_seed')[1].split('_')[0]
            res = analyze_actor_critic_results(filepath, 'ppo')
            if res:
                all_results[f'pendulum_ppo_seed{seed}'] = res
                for algo, metrics in res.items():
                    print(f"\n  Seed {seed} - {algo.upper()}:")
                    print(f"    Baseline:  {metrics['baseline']:7.2f}")
                    print(f"    Static:    {metrics['static']:7.2f} (Δ={metrics['static_improvement']:+7.2f})")
                    print(f"    Dynamic:   {metrics['dynamic']:7.2f} (Δ={metrics['dynamic_improvement']:+7.2f})")
                    print(f"    Dynamic vs Static: {metrics['dynamic_vs_static']:+7.2f}")
                    if metrics['dynamic_vs_static'] > 0:
                        print(f"    ✓ DYNAMIC WINS!")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n\n" + "=" * 80)
    print("SUMMARY: Did Dynamic QBound Ever Win?")
    print("=" * 80)

    dynamic_wins = []
    dynamic_losses = []

    for exp_name, exp_results in all_results.items():
        for variant, metrics in exp_results.items():
            comparison = metrics['dynamic_vs_static']
            entry = (exp_name, variant, comparison)
            if comparison > 0:
                dynamic_wins.append(entry)
            else:
                dynamic_losses.append(entry)

    print(f"\nDynamic QBound WINS (better than Static): {len(dynamic_wins)}")
    if dynamic_wins:
        print("\n  Cases where Dynamic > Static:")
        for exp, variant, delta in sorted(dynamic_wins, key=lambda x: x[2], reverse=True):
            print(f"    {exp:40s} {variant:10s}: +{delta:7.2f}")

    print(f"\nDynamic QBound LOSSES (worse than Static): {len(dynamic_losses)}")
    if dynamic_losses:
        print("\n  Top 10 worst cases where Static > Dynamic:")
        for exp, variant, delta in sorted(dynamic_losses, key=lambda x: x[2])[:10]:
            print(f"    {exp:40s} {variant:10s}: {delta:7.2f}")

    # Overall verdict
    print("\n" + "=" * 80)
    print("VERDICT:")
    print("=" * 80)
    if len(dynamic_wins) > 0:
        win_rate = len(dynamic_wins) / (len(dynamic_wins) + len(dynamic_losses)) * 100
        print(f"\nDynamic QBound won in {len(dynamic_wins)}/{len(dynamic_wins) + len(dynamic_losses)} cases ({win_rate:.1f}%)")
        print("\nConclusion: Dynamic QBound shows some benefit in specific cases.")
    else:
        print("\nDynamic QBound NEVER outperformed Static QBound in any experiment.")
        print("\nConclusion: Static QBound is sufficient. Dynamic QBound adds complexity without benefit.")

    print("\n")

if __name__ == '__main__':
    main()
