"""
Generate a clean summary table of Q_min and Q_max for the paper.
"""

import json

# Load results
with open('results/combined/experiment_results_20251024_192918.json', 'r') as f:
    results = json.load(f)

print("\n" + "="*90)
print(" " * 25 + "QBOUND CONFIGURATION SUMMARY TABLE")
print("="*90)

# Header
print(f"\n{'Environment':<15} {'Q_min':<10} {'Q_max':<10} {'γ':<8} {'Theory Max':<12} {'Episodes':<12} {'Performance':<15}")
print("-"*90)

for env_name, data in results.items():
    config = data['config']
    qmin = config['qclip_min']
    qmax = config['qclip_max']
    gamma = config['gamma']
    theory_max = qmax / (1 - gamma)

    # Determine episodes
    episodes = config['num_episodes']

    # Performance indicator
    if data['improvement_percent'] is not None:
        improvement = data['improvement_percent']
        if improvement > 0:
            perf = f"+{improvement:.1f}% ✓"
        else:
            perf = f"{improvement:.1f}% ✗"
    else:
        perf = "No converge"

    print(f"{env_name:<15} {qmin:<10.1f} {qmax:<10.1f} {gamma:<8.2f} {theory_max:<12.1f} {episodes:<12} {perf:<15}")

print("\n" + "="*90)
print("ACTUAL Q-VALUE RANGES NEEDED")
print("="*90)

print(f"\n{'Environment':<15} {'Reward Type':<20} {'Max Episode Reward':<20} {'Recommended Q_max':<20}")
print("-"*90)

recommendations = {
    'GridWorld': {
        'reward_type': 'Sparse (0 or 1)',
        'max_episode': '1.0',
        'recommended_qmax': '≥10.0 (for value prop)',
        'reason': 'Need to propagate value backwards through states'
    },
    'FrozenLake': {
        'reward_type': 'Sparse (0 or 1)',
        'max_episode': '1.0',
        'recommended_qmax': '1.0 ✓ (current OK)',
        'reason': 'Stochastic env, lower γ=0.95, bounds help'
    },
    'CartPole': {
        'reward_type': 'Dense (1 per step)',
        'max_episode': '500.0',
        'recommended_qmax': '≥500.0',
        'reason': 'Current 100 prevents learning optimal policy'
    }
}

for env_name, rec in recommendations.items():
    print(f"{env_name:<15} {rec['reward_type']:<20} {rec['max_episode']:<20} {rec['recommended_qmax']:<20}")
    print(f"{'':>15} Reason: {rec['reason']}")
    print()

print("="*90)
print("PERFORMANCE COMPARISON")
print("="*90)

print(f"\n{'Environment':<15} {'QBound Eps':<15} {'Baseline Eps':<15} {'QBound Reward':<15} {'Baseline Reward':<15}")
print("-"*90)

for env_name, data in results.items():
    qbound_eps = str(data['qbound_episodes']) if data['qbound_episodes'] else "N/A"
    baseline_eps = str(data['baseline_episodes']) if data['baseline_episodes'] else "N/A"
    qbound_reward = f"{data['qbound_total_reward']:.0f}"
    baseline_reward = f"{data['baseline_total_reward']:.0f}"

    print(f"{env_name:<15} {qbound_eps:<15} {baseline_eps:<15} {qbound_reward:<15} {baseline_reward:<15}")

print("\n" + "="*90)
print("KEY FINDINGS")
print("="*90)

print("""
1. Q_max must be set based on MAXIMUM EPISODE RETURN, not step reward:
   - GridWorld: Step reward = 1, but need Q_max > 1 for value propagation
   - FrozenLake: Step reward = 1, Q_max = 1 works (lower γ, stochastic)
   - CartPole: Max episode = 500, but Q_max = 100 (TOO LOW!)

2. Discount factor γ affects required Q_max:
   - High γ (0.99): Values accumulate more, need higher Q_max
   - Low γ (0.95): Less accumulation, tighter bounds acceptable

3. Environment type matters:
   - Stochastic + Sparse: QBound helps (FrozenLake ✓)
   - Deterministic + High γ: QBound hurts (GridWorld ✗)
   - Dense rewards + High γ: QBound severely hurts (CartPole ✗✗)

4. Violation of the bound = prevention of learning:
   - When true optimal Q > Q_max, clipping prevents convergence
   - CartPole: Optimal Q ≈ 500, but clipped at 100 → -41% performance!

CONCLUSION:
Current QBound settings are FUNDAMENTALLY FLAWED for 2 out of 3 environments.
Paper cannot proceed without fixing Q_max values or acknowledging severe limitations.
""")

print("="*90)
