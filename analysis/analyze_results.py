"""
Analyze experiment results to check QBound design.
Displays key metrics: rewards, discount factor, steps, episodes.
"""

import json
import numpy as np

# Load results
with open('results/combined/experiment_results_20251024_192918.json', 'r') as f:
    results = json.load(f)

print("="*80)
print("QBOUND EXPERIMENTAL ANALYSIS - Design Issue Check")
print("="*80)

for env_name, data in results.items():
    print(f"\n{'='*80}")
    print(f"{env_name.upper()}")
    print(f"{'='*80}")

    config = data['config']

    # Configuration parameters
    print(f"\n📋 Configuration:")
    print(f"   Discount Factor (γ):     {config['gamma']}")
    print(f"   Learning Rate (α):       {config['learning_rate']}")
    print(f"   QBound Max:              {config['qclip_max']}")
    print(f"   QBound Min:              {config['qclip_min']}")
    print(f"   Epsilon Decay:           {config['epsilon_decay']}")
    print(f"   Total Episodes:          {config['num_episodes']}")
    print(f"   Max Steps per Episode:   {config['max_steps']}")
    print(f"   Target Success Rate:     {config['target_success']*100:.0f}%")

    # Reward statistics
    qbound_rewards = np.array(data['rewards_qbound'])
    baseline_rewards = np.array(data['rewards_baseline'])

    print(f"\n📊 Reward Statistics (QBound):")
    print(f"   Total Cumulative Reward: {data['qbound_total_reward']:.1f}")
    print(f"   Mean Episode Reward:     {qbound_rewards.mean():.3f}")
    print(f"   Std Episode Reward:      {qbound_rewards.std():.3f}")
    print(f"   Max Episode Reward:      {qbound_rewards.max():.1f}")
    print(f"   Min Episode Reward:      {qbound_rewards.min():.1f}")
    print(f"   Success Rate (r>0):      {(qbound_rewards > 0).sum() / len(qbound_rewards) * 100:.1f}%")

    print(f"\n📊 Reward Statistics (Baseline):")
    print(f"   Total Cumulative Reward: {data['baseline_total_reward']:.1f}")
    print(f"   Mean Episode Reward:     {baseline_rewards.mean():.3f}")
    print(f"   Std Episode Reward:      {baseline_rewards.std():.3f}")
    print(f"   Max Episode Reward:      {baseline_rewards.max():.1f}")
    print(f"   Min Episode Reward:      {baseline_rewards.min():.1f}")
    print(f"   Success Rate (r>0):      {(baseline_rewards > 0).sum() / len(baseline_rewards) * 100:.1f}%")

    # Episode convergence
    print(f"\n🎯 Convergence:")
    qbound_ep = data['qbound_episodes']
    baseline_ep = data['baseline_episodes']
    print(f"   QBound Episodes to Target:   {qbound_ep if qbound_ep else 'Not achieved'}")
    print(f"   Baseline Episodes to Target: {baseline_ep if baseline_ep else 'Not achieved'}")

    if data['improvement_percent'] is not None:
        improvement = data['improvement_percent']
        print(f"   Improvement:                 {improvement:+.1f}%")
        if improvement < 0:
            print(f"   ⚠️  WARNING: QBound is SLOWER by {abs(improvement):.1f}%")
        else:
            print(f"   ✓  QBound is FASTER by {improvement:.1f}%")
    else:
        print(f"   ⚠️  Neither method reached target")

    # Reward improvement
    reward_imp = data['reward_improvement_percent']
    print(f"\n💰 Total Reward Comparison:")
    print(f"   Reward Improvement:          {reward_imp:+.1f}%")
    if reward_imp < -10:
        print(f"   ⚠️  WARNING: QBound has significantly LOWER total reward")

    # Calculate learning efficiency (early vs late performance)
    early_qbound = qbound_rewards[:100].mean()
    late_qbound = qbound_rewards[-100:].mean()
    early_baseline = baseline_rewards[:100].mean()
    late_baseline = baseline_rewards[-100:].mean()

    print(f"\n📈 Learning Progression:")
    print(f"   QBound - First 100 eps:      {early_qbound:.3f}")
    print(f"   QBound - Last 100 eps:       {late_qbound:.3f}")
    print(f"   QBound - Improvement:        {((late_qbound-early_qbound)/max(early_qbound,0.001)*100):+.1f}%")
    print(f"")
    print(f"   Baseline - First 100 eps:    {early_baseline:.3f}")
    print(f"   Baseline - Last 100 eps:     {late_baseline:.3f}")
    print(f"   Baseline - Improvement:      {((late_baseline-early_baseline)/max(early_baseline,0.001)*100):+.1f}%")

print(f"\n{'='*80}")
print("DESIGN ISSUE ANALYSIS")
print(f"{'='*80}")

# Check for potential design issues
issues_found = []

for env_name, data in results.items():
    config = data['config']

    # Issue 1: QBound bounds too restrictive?
    max_theoretical_return = config['qclip_max'] / (1 - config['gamma'])
    print(f"\n{env_name}:")
    print(f"  Max theoretical return (Q_max/(1-γ)): {max_theoretical_return:.1f}")
    print(f"  Actual QBound max set to:              {config['qclip_max']}")

    # Issue 2: Performance degradation
    if data['improvement_percent'] is not None and data['improvement_percent'] < -15:
        issues_found.append(f"  ⚠️  {env_name}: QBound is {abs(data['improvement_percent']):.1f}% SLOWER")

    # Issue 3: Reward degradation
    if data['reward_improvement_percent'] < -15:
        issues_found.append(f"  ⚠️  {env_name}: QBound has {abs(data['reward_improvement_percent']):.1f}% LOWER total reward")

    # Issue 4: Neither converged
    if data['qbound_episodes'] is None and data['baseline_episodes'] is None:
        issues_found.append(f"  ⚠️  {env_name}: Neither QBound nor Baseline converged")

if issues_found:
    print(f"\n🚨 POTENTIAL DESIGN ISSUES DETECTED:")
    for issue in issues_found:
        print(issue)
else:
    print(f"\n✓ No major design issues detected")

print(f"\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}")
print(f"\nEnvironments where QBound performs BETTER:")
for env_name, data in results.items():
    if data['improvement_percent'] and data['improvement_percent'] > 0:
        print(f"  ✓ {env_name}: {data['improvement_percent']:.1f}% faster convergence")

print(f"\nEnvironments where QBound performs WORSE:")
for env_name, data in results.items():
    if data['improvement_percent'] and data['improvement_percent'] < 0:
        print(f"  ✗ {env_name}: {abs(data['improvement_percent']):.1f}% slower convergence")

print(f"\nEnvironments where neither converged:")
for env_name, data in results.items():
    if data['improvement_percent'] is None:
        print(f"  ? {env_name}: No convergence")

print("\n")
