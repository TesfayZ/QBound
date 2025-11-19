#!/usr/bin/env python3
"""
Deep investigation: Why do MountainCar and Acrobot IMPROVE with QBound,
but Pendulum DEGRADES?

Key questions:
1. What Q-bounds were used?
2. What are violation patterns?
3. How do these environments differ from Pendulum?
4. Why does DQN work here but not Pendulum?
"""

import json
import numpy as np
from pathlib import Path

def analyze_environment(env_name, results_pattern):
    """Analyze a specific environment."""
    print("=" * 80)
    print(f"{env_name.upper()} ANALYSIS")
    print("=" * 80)
    print()

    results_dir = Path(f'/root/projects/QBound/results/{env_name.lower()}')
    files = list(results_dir.glob(results_pattern))

    if not files:
        print(f"No results found for {env_name}!")
        return None

    # Load first file to check structure
    with open(files[0], 'r') as f:
        data = json.load(f)

    # Print config
    config = data['config']
    print("Configuration:")
    print(f"  Environment: {config.get('env', 'N/A')}")
    print(f"  Episodes: {config.get('episodes', 'N/A')}")
    print(f"  Max steps: {config.get('max_steps', 'N/A')}")
    print(f"  Gamma: {config.get('gamma', 'N/A')}")
    print(f"  Q_min: {config.get('qbound_min', 'N/A')}")
    print(f"  Q_max: {config.get('qbound_max', 'N/A')}")
    print(f"  Reward structure: {config.get('reward_structure', 'N/A')}")
    print()

    # Analyze all seeds
    all_results = {
        'seeds': [],
        'baseline_rewards': [],
        'qbound_rewards': [],
        'baseline_steps': [],
        'qbound_steps': [],
        'violations': []
    }

    for file in sorted(files):
        with open(file, 'r') as f:
            data = json.load(f)

        seed = data['config']['seed']
        all_results['seeds'].append(seed)

        # Get performance
        baseline_rewards = data['training']['baseline']['rewards']
        qbound_rewards = data['training']['static_qbound']['rewards']

        all_results['baseline_rewards'].append(np.mean(baseline_rewards[-100:]))
        all_results['qbound_rewards'].append(np.mean(qbound_rewards[-100:]))

        # Get episode lengths if available
        if 'steps' in data['training']['baseline']:
            baseline_steps = data['training']['baseline']['steps']
            qbound_steps = data['training']['static_qbound']['steps']
            all_results['baseline_steps'].append(np.mean(baseline_steps[-100:]))
            all_results['qbound_steps'].append(np.mean(qbound_steps[-100:]))

        # Get violations if available
        if 'violations' in data['training']['static_qbound']:
            all_results['violations'].append(data['training']['static_qbound']['violations'])

    # Print performance summary
    print("Performance Summary (Final 100 Episodes):")
    print("-" * 60)
    for i, seed in enumerate(all_results['seeds']):
        baseline = all_results['baseline_rewards'][i]
        qbound = all_results['qbound_rewards'][i]
        change = ((qbound / baseline) - 1) * 100 if baseline != 0 else 0

        print(f"Seed {seed}:")
        print(f"  Baseline reward: {baseline:7.2f}")
        print(f"  QBound reward:   {qbound:7.2f} ({change:+6.1f}%)")

        if all_results['baseline_steps']:
            baseline_steps = all_results['baseline_steps'][i]
            qbound_steps = all_results['qbound_steps'][i]
            step_change = ((qbound_steps / baseline_steps) - 1) * 100 if baseline_steps != 0 else 0
            print(f"  Baseline steps:  {baseline_steps:7.2f}")
            print(f"  QBound steps:    {qbound_steps:7.2f} ({step_change:+6.1f}%)")
        print()

    # Aggregate statistics
    baseline_arr = np.array(all_results['baseline_rewards'])
    qbound_arr = np.array(all_results['qbound_rewards'])
    changes = ((qbound_arr / baseline_arr) - 1) * 100

    print("Aggregate Statistics:")
    print(f"  Mean baseline reward: {baseline_arr.mean():.2f} ± {baseline_arr.std():.2f}")
    print(f"  Mean QBound reward:   {qbound_arr.mean():.2f} ± {qbound_arr.std():.2f}")
    print(f"  Mean change: {changes.mean():+.1f}% ± {changes.std():.1f}%")
    print()

    if all_results['baseline_steps']:
        baseline_steps_arr = np.array(all_results['baseline_steps'])
        qbound_steps_arr = np.array(all_results['qbound_steps'])
        step_changes = ((qbound_steps_arr / baseline_steps_arr) - 1) * 100

        print(f"  Mean baseline steps: {baseline_steps_arr.mean():.2f} ± {baseline_steps_arr.std():.2f}")
        print(f"  Mean QBound steps:   {qbound_steps_arr.mean():.2f} ± {qbound_steps_arr.std():.2f}")
        print(f"  Mean step change: {step_changes.mean():+.1f}% ± {step_changes.std():.1f}%")
        print()

    # Analyze violations
    if all_results['violations'] and all_results['violations'][0]:
        print("Violation Analysis:")
        print("-" * 60)

        violation_data = all_results['violations'][0]

        if 'mean' in violation_data:
            mean_viol = violation_data['mean']
            final_viol = violation_data.get('final_100', {})

            print(f"Q-bounds used:")
            print(f"  Q_min: {mean_viol.get('qbound_min', 'N/A')}")
            print(f"  Q_max: {mean_viol.get('qbound_max', 'N/A')}")
            print()

            print(f"Mean violation rates (across all episodes):")
            print(f"  Upper (Q > Q_max): {mean_viol.get('next_q_violate_max_rate', 0):.2%}")
            print(f"  Lower (Q < Q_min): {mean_viol.get('next_q_violate_min_rate', 0):.2%}")
            print()

            print(f"Final 100 episodes violation rates:")
            print(f"  Upper (Q > Q_max): {final_viol.get('next_q_violate_max_rate', 0):.2%}")
            print(f"  Lower (Q < Q_min): {final_viol.get('next_q_violate_min_rate', 0):.2%}")
            print()

            print(f"Violation magnitudes:")
            print(f"  Upper magnitude: {mean_viol.get('violation_magnitude_max_next', 0):.4f}")
            print(f"  Lower magnitude: {mean_viol.get('violation_magnitude_min_next', 0):.4f}")
            print()

    return all_results, config

def compare_environments():
    """Compare Pendulum, MountainCar, and Acrobot."""
    print("\n" + "=" * 80)
    print("COMPARATIVE ANALYSIS: Why Different Results?")
    print("=" * 80)
    print()

    # Analyze each
    print("Loading Pendulum data for comparison...")
    pendulum_file = list(Path('/root/projects/QBound/results/pendulum').glob('dqn_full_qbound_seed42_*.json'))
    if pendulum_file:
        with open([f for f in pendulum_file if 'in_progress' not in str(f)][0], 'r') as f:
            pendulum_data = json.load(f)
        pendulum_config = pendulum_data['config']
    else:
        print("No Pendulum data found!")
        return

    print("\n" + "-" * 80)
    print("ENVIRONMENT CHARACTERISTICS COMPARISON")
    print("-" * 80)
    print()

    # Print comparison table
    print(f"{'Characteristic':<30} {'Pendulum':<20} {'MountainCar':<20} {'Acrobot':<20}")
    print("-" * 90)

    # This will be filled in after analyzing
    print("\nAnalyzing MountainCar...")
    mc_results, mc_config = analyze_environment('mountaincar', 'dqn_static_qbound_seed*.json')

    print("\nAnalyzing Acrobot...")
    acrobot_results, acrobot_config = analyze_environment('acrobot', 'dqn_static_qbound_seed*.json')

    # Now compare
    print("\n" + "=" * 80)
    print("KEY DIFFERENCES")
    print("=" * 80)
    print()

    comparison = {
        'Environment': ['Pendulum-v1', mc_config.get('env', 'MountainCar'), acrobot_config.get('env', 'Acrobot')],
        'Reward per step': ['-16.2', '-1', '-1'],
        'Reward type': ['Dense negative', 'Sparse goal-dependent', 'Sparse goal-dependent'],
        'Max steps': [pendulum_config.get('max_steps', 200), mc_config.get('max_steps', 200), acrobot_config.get('max_steps', 500)],
        'Q_min': [pendulum_config.get('qbound_min', -1409), mc_config.get('qbound_min', 'N/A'), acrobot_config.get('qbound_min', 'N/A')],
        'Q_max': [pendulum_config.get('qbound_max', 0), mc_config.get('qbound_max', 'N/A'), acrobot_config.get('qbound_max', 'N/A')],
    }

    for key, values in comparison.items():
        print(f"{key:<25} {str(values[0]):<20} {str(values[1]):<20} {str(values[2]):<20}")

    print()
    print("CRITICAL DIFFERENCES:")
    print("-" * 60)
    print()
    print("1. REWARD DENSITY:")
    print("   Pendulum: EVERY step gives -16.2 (dense)")
    print("   MountainCar/Acrobot: -1 per step (sparser magnitude)")
    print()
    print("2. REWARD STRUCTURE:")
    print("   Pendulum: Angle-dependent cost (varies -0.1 to -16.2)")
    print("   MountainCar/Acrobot: Constant -1 until goal")
    print()
    print("3. Q-VALUE BOUNDS:")
    print(f"   Pendulum: Q_min={pendulum_config.get('qbound_min', -1409)}, Q_max={pendulum_config.get('qbound_max', 0)}")
    print(f"   MountainCar: Q_min={mc_config.get('qbound_min', 'N/A')}, Q_max={mc_config.get('qbound_max', 'N/A')}")
    print(f"   Acrobot: Q_min={acrobot_config.get('qbound_min', 'N/A')}, Q_max={acrobot_config.get('qbound_max', 'N/A')}")
    print()

    # Analyze violation rates
    print("4. VIOLATION PATTERNS:")
    print()

    # Load Pendulum violations
    pendulum_viol = pendulum_data['training']['static_qbound_dqn']['violations']['mean']
    print(f"   Pendulum Q > Q_max violations: {pendulum_viol.get('next_q_violate_max_rate', 0):.2%}")

    if mc_results and mc_results['violations'] and mc_results['violations'][0]:
        mc_viol = mc_results['violations'][0]['mean']
        print(f"   MountainCar Q > Q_max violations: {mc_viol.get('next_q_violate_max_rate', 0):.2%}")

    if acrobot_results and acrobot_results['violations'] and acrobot_results['violations'][0]:
        acrobot_viol = acrobot_results['violations'][0]['mean']
        print(f"   Acrobot Q > Q_max violations: {acrobot_viol.get('next_q_violate_max_rate', 0):.2%}")

    print()

def hypothesis_analysis():
    """Formulate and test hypotheses."""
    print("\n" + "=" * 80)
    print("HYPOTHESIS ANALYSIS")
    print("=" * 80)
    print()

    print("HYPOTHESIS 1: Reward Magnitude Matters")
    print("-" * 60)
    print("Pendulum: -16.2 per step → large negative Q-values")
    print("MountainCar/Acrobot: -1 per step → smaller negative Q-values")
    print()
    print("If Q-values are smaller (closer to 0), maybe:")
    print("  - Fewer violations of Q_max=0")
    print("  - Less underestimation bias from clipping")
    print("  - QBound acts as stabilizer instead of bias source")
    print()

    print("HYPOTHESIS 2: Sparse vs Dense Reward Learning")
    print("-" * 60)
    print("Pendulum: Dense feedback every step")
    print("  - Network learns detailed value function")
    print("  - Clipping interferes with fine-grained learning")
    print("  - Loss of granularity is costly")
    print()
    print("MountainCar/Acrobot: Constant -1 until goal")
    print("  - Network learns coarse 'distance to goal'")
    print("  - Clipping provides useful bound on exploration")
    print("  - Stabilization effect outweighs bias")
    print()

    print("HYPOTHESIS 3: Exploration Difficulty")
    print("-" * 60)
    print("Pendulum: Continuous state, easy to explore")
    print("  - Baseline learns well")
    print("  - QBound only adds bias")
    print()
    print("MountainCar/Acrobot: Difficult exploration")
    print("  - MountainCar: Need momentum to reach goal")
    print("  - Acrobot: Complex swing-up dynamics")
    print("  - Baseline struggles → QBound stabilization helps")
    print()

    print("HYPOTHESIS 4: Q-Value Distribution")
    print("-" * 60)
    print("Pendulum: Wide Q-value range [-1409, 0]")
    print("  - Many Q-values near 0 (near terminal)")
    print("  - Clipping at 0 affects many states")
    print()
    print("MountainCar/Acrobot: Narrower effective range?")
    print("  - Most Q-values far from bounds")
    print("  - Clipping only affects edge cases")
    print("  - Less impact on learning")
    print()

if __name__ == '__main__':
    compare_environments()
    hypothesis_analysis()
