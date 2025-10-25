"""
Quick display of QBound configuration and theoretical analysis.
"""

import json

# Load results
with open('results/combined/experiment_results_20251024_192918.json', 'r') as f:
    results = json.load(f)

print("="*80)
print("QBOUND CONFIGURATION AND THEORETICAL ANALYSIS")
print("="*80)

for env_name, data in results.items():
    config = data['config']

    print(f"\n{'='*80}")
    print(f"{env_name.upper()}")
    print(f"{'='*80}")

    # Configuration
    gamma = config['gamma']
    qmin = config['qclip_min']
    qmax = config['qclip_max']

    print(f"\n📋 QBound Configuration:")
    print(f"   Q_min (configured): {qmin}")
    print(f"   Q_max (configured): {qmax}")
    print(f"   Discount factor γ:  {gamma}")

    # Theoretical analysis
    print(f"\n🔬 Theoretical Analysis:")

    # Maximum possible return under discount
    # For an infinite horizon with constant reward R:
    # V = R + γR + γ²R + ... = R/(1-γ)

    max_theoretical = qmax / (1 - gamma)
    print(f"   Max theoretical return (Q_max/(1-γ)): {max_theoretical:.2f}")
    print(f"   This assumes constant max reward per step")

    # For sparse rewards (like GridWorld where reward is 0 or 1):
    if env_name == "GridWorld":
        print(f"\n   GridWorld specifics:")
        print(f"   - Reward: 1.0 for reaching goal, 0.0 otherwise")
        print(f"   - Single reward episode: Q ≈ 1.0")
        print(f"   - QBound [0, 1] seems appropriate for immediate rewards")
        print(f"   - But Q-values represent discounted future returns!")
        print(f"   - With γ=0.99, value should propagate backwards")

    elif env_name == "FrozenLake":
        print(f"\n   FrozenLake specifics:")
        print(f"   - Reward: 1.0 for reaching goal, 0.0 otherwise")
        print(f"   - Stochastic environment (slippery)")
        print(f"   - QBound [0, 1] appropriate for sparse binary rewards")
        print(f"   - γ=0.95 (lower) reduces future value propagation")

    elif env_name == "CartPole":
        print(f"\n   CartPole specifics:")
        print(f"   - Reward: 1.0 per timestep survived")
        print(f"   - Max episode length: 500")
        print(f"   - Optimal strategy: survive all 500 steps")
        print(f"   - Expected return: up to 500")
        print(f"   - QBound [0, 100] may be TOO RESTRICTIVE!")
        print(f"   - Optimal Q-value could be ~500 with γ=0.99")

    # Actual performance
    print(f"\n📊 Actual Performance:")
    print(f"   QBound total reward:   {data['qbound_total_reward']:.1f}")
    print(f"   Baseline total reward: {data['baseline_total_reward']:.1f}")
    print(f"   Difference:            {data['reward_improvement_percent']:+.1f}%")

    if data['improvement_percent'] is not None:
        print(f"\n🎯 Convergence Speed:")
        print(f"   QBound episodes:  {data['qbound_episodes']}")
        print(f"   Baseline episodes: {data['baseline_episodes']}")
        print(f"   Improvement:       {data['improvement_percent']:+.1f}%")

print(f"\n{'='*80}")
print("KEY INSIGHTS")
print(f"{'='*80}")

print(f"""
1. GridWorld (γ=0.99, QBound=[0,1]):
   - QBound UNDERPERFORMS by 22.1%
   - Possible issue: Bounds prevent value propagation?
   - With γ=0.99, Q-values should build up over training
   - Clipping at 1.0 may prevent learning long-term strategy

2. FrozenLake (γ=0.95, QBound=[0,1]):
   - QBound OUTPERFORMS by 19.4%!
   - Success! Stochastic environment benefits from bounds
   - Lower γ=0.95 means less value propagation needed
   - Bounds help prevent overestimation in uncertain environment

3. CartPole (γ=0.99, QBound=[0,100]):
   - QBound SEVERELY UNDERPERFORMS (-41.4% reward)
   - MAJOR ISSUE: Bounds are too restrictive!
   - Optimal return ≈ 500, but QBound limits Q to 100
   - This fundamentally prevents learning optimal policy

RECOMMENDATION:
- QBound works well in stochastic, sparse-reward environments (FrozenLake)
- QBound struggles when Q-values need to accumulate (GridWorld, CartPole)
- Consider: Q_max should be set based on max possible episode return
- For CartPole: Q_max should be ≥ max_episode_length * reward_per_step
- For GridWorld: May need higher Q_max to allow value propagation
""")

print("="*80)
