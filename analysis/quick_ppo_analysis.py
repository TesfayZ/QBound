"""Quick analysis of PPO pilot results (no scipy dependency)."""

import json
import glob
import numpy as np
from pathlib import Path


def analyze_results():
    results_dir = "/root/projects/QBound/results/ppo"

    print("="*60)
    print("PPO + QBound Pilot Results Analysis")
    print("="*60)

    # Find result files
    files = glob.glob(f"{results_dir}/*.json")

    for file in sorted(files):
        print(f"\n{'='*60}")
        print(f"File: {Path(file).name}")
        print(f"{'='*60}")

        with open(file, 'r') as f:
            data = json.load(f)

        # Extract results
        for agent_name in sorted(data.keys()):
            agent_data = data[agent_name]
            stats = agent_data.get('final_100_episodes', {})

            print(f"\n{agent_name}:")
            print(f"  Mean: {stats.get('mean', 0):.2f} ± {stats.get('std', 0):.2f}")
            print(f"  Max: {stats.get('max', 0):.2f}")
            print(f"  Min: {stats.get('min', 0):.2f}")

            if 'success_rate' in stats:
                print(f"  Success Rate: {stats['success_rate']:.1f}%")

        # Compute improvements
        if 'Baseline PPO' in data:
            baseline_mean = data['Baseline PPO']['final_100_episodes']['mean']

            for agent_name in data.keys():
                if agent_name != 'Baseline PPO':
                    qbound_mean = data[agent_name]['final_100_episodes']['mean']
                    improvement = ((qbound_mean - baseline_mean) / abs(baseline_mean) * 100)

                    print(f"\n{agent_name} vs Baseline: {improvement:+.1f}%")

                    if improvement > 10:
                        print("  ✅ STRONG IMPROVEMENT")
                    elif improvement > 0:
                        print("  ✓ Moderate improvement")
                    elif improvement > -10:
                        print("  ➖ Neutral")
                    else:
                        print("  ❌ DEGRADATION")

    # Summary
    print(f"\n\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    print("\nCartPole (Dense Reward):")
    print("  Static QBound: +0.4% (neutral)")
    print("  Dynamic QBound: +17.9% ✅ (step-aware bounds work!)")

    print("\nLunarLander (Sparse Reward):")
    print("  QBound: -30.9% ❌ (unexpected degradation)")
    print("  Success rate dropped from 80% to 38%")

    print("\n" + "="*60)
    print("KEY INSIGHTS")
    print("="*60)

    print("\n1. Dynamic bounds help dense reward tasks (CartPole +17.9%)")
    print("\n2. QBound hurts PPO on sparse rewards (LunarLander -30.9%)")
    print("   - Different from DQN where QBound helped (+263.9%)")
    print("   - PPO's GAE may already provide implicit stabilization")

    print("\n3. PPO baseline already strong on LunarLander (80% success)")
    print("   - Adding QBound may over-constrain the value function")
    print("   - V_min=-100, V_max=200 bounds may be too restrictive")

    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)

    print("\n1. Try looser bounds on LunarLander (e.g., V_min=-300, V_max=300)")
    print("\n2. Run remaining experiments to see full pattern:")
    print("   - Pendulum (continuous + dense) - CRITICAL TEST")
    print("   - LunarLanderContinuous (continuous + sparse)")
    print("   - Acrobot, MountainCar (discrete + sparse)")

    print("\n3. Hypothesis: QBound may not help PPO as much as DQN")
    print("   - PPO's GAE already stabilizes advantages")
    print("   - Value clipping may conflict with GAE")


if __name__ == "__main__":
    analyze_results()
