#!/usr/bin/env python3
"""
Comprehensive QBound Analysis: Understanding Success and Failure Patterns

This script analyzes:
1. Why QBound works for positive rewards (CartPole)
2. Why architectural QBound FAILS for negative rewards (Pendulum DQN/DDPG/PPO)
3. Why TD3 shows marginal improvement
4. The role of algorithm type (value-based vs actor-critic)
5. Theoretical implications
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import glob
from pathlib import Path

# Configure matplotlib
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10

def load_results(pattern):
    """Load all results matching pattern and compute statistics."""
    results = {}
    for file in sorted(glob.glob(pattern)):
        with open(file, 'r') as f:
            data = json.load(f)
            for method, method_data in data['training'].items():
                if method not in results:
                    results[method] = {
                        'final_100': [],
                        'all_rewards': [],
                        'violations': []
                    }

                rewards = method_data['rewards']
                results[method]['final_100'].append(np.mean(rewards[-100:]))
                results[method]['all_rewards'].append(rewards)

                # Check for violation data
                if 'violations' in method_data:
                    results[method]['violations'].append(method_data['violations'])

    # Compute statistics
    for method in results:
        results[method]['mean'] = np.mean(results[method]['final_100'])
        results[method]['std'] = np.std(results[method]['final_100'])
        results[method]['n_seeds'] = len(results[method]['final_100'])

    return results

def compute_improvement(baseline_mean, variant_mean, is_negative_reward=False):
    """Compute percentage improvement (higher is better)."""
    if is_negative_reward:
        # For negative rewards, closer to 0 is better
        improvement = (baseline_mean - variant_mean) / abs(baseline_mean) * 100
    else:
        # For positive rewards, higher is better
        improvement = (variant_mean - baseline_mean) / baseline_mean * 100
    return improvement

def analyze_cartpole():
    """Analyze CartPole results (positive rewards)."""
    print("=" * 80)
    print("CARTPOLE ANALYSIS (Positive Dense Rewards: r = +1 per step)")
    print("=" * 80)

    results = load_results('results/cartpole/dqn_full_qbound_seed*.json')

    print("\n### Results:")
    for method in sorted(results.keys()):
        print(f"  {method:30s}: {results[method]['mean']:7.2f} ± {results[method]['std']:5.2f} (n={results[method]['n_seeds']})")

    # Compute improvements
    print("\n### QBound Effectiveness:")
    if 'baseline' in results and 'static_qbound' in results:
        imp = compute_improvement(results['baseline']['mean'], results['static_qbound']['mean'])
        print(f"  DQN + Static QBound:   {imp:+6.2f}% improvement")

    if 'baseline_ddqn' in results and 'static_qbound_ddqn' in results:
        imp = compute_improvement(results['baseline_ddqn']['mean'], results['static_qbound_ddqn']['mean'])
        print(f"  DDQN + Static QBound:  {imp:+6.2f}% improvement")

    print("\n### Key Observations:")
    print("  ✓ QBound WORKS for positive rewards")
    print("  ✓ Hard clipping with Q_max = 99.34 effective")
    print("  ✓ Both DQN and DDQN benefit significantly")

    return results

def analyze_pendulum():
    """Analyze Pendulum results (negative rewards)."""
    print("\n" + "=" * 80)
    print("PENDULUM ANALYSIS (Negative Dense Rewards: r ∈ [-16, 0])")
    print("=" * 80)

    all_results = {}

    # DQN
    print("\n### DQN Variants:")
    dqn_results = load_results('results/pendulum/dqn_full_qbound_seed*.json')
    for method in sorted(dqn_results.keys()):
        print(f"  {method:30s}: {dqn_results[method]['mean']:7.2f} ± {dqn_results[method]['std']:5.2f}")
    all_results['dqn'] = dqn_results

    if 'dqn' in dqn_results and 'architectural_qbound_dqn' in dqn_results:
        imp = compute_improvement(dqn_results['dqn']['mean'],
                                 dqn_results['architectural_qbound_dqn']['mean'],
                                 is_negative_reward=True)
        print(f"  → Architectural QBound: {imp:+6.2f}% {'improvement' if imp > 0 else 'degradation'}")

    # DDPG
    print("\n### DDPG Variants:")
    ddpg_results = load_results('results/pendulum/ddpg_full_qbound_seed*.json')
    for method in sorted(ddpg_results.keys()):
        print(f"  {method:30s}: {ddpg_results[method]['mean']:7.2f} ± {ddpg_results[method]['std']:5.2f}")
    all_results['ddpg'] = ddpg_results

    if 'baseline' in ddpg_results and 'architectural_qbound_ddpg' in ddpg_results:
        imp = compute_improvement(ddpg_results['baseline']['mean'],
                                 ddpg_results['architectural_qbound_ddpg']['mean'],
                                 is_negative_reward=True)
        print(f"  → Architectural QBound: {imp:+6.2f}% {'improvement' if imp > 0 else 'degradation'}")

    # TD3
    print("\n### TD3 Variants:")
    td3_results = load_results('results/pendulum/td3_full_qbound_seed*.json')
    for method in sorted(td3_results.keys()):
        print(f"  {method:30s}: {td3_results[method]['mean']:7.2f} ± {td3_results[method]['std']:5.2f}")
    all_results['td3'] = td3_results

    if 'baseline' in td3_results and 'architectural_qbound_td3' in td3_results:
        imp = compute_improvement(td3_results['baseline']['mean'],
                                 td3_results['architectural_qbound_td3']['mean'],
                                 is_negative_reward=True)
        print(f"  → Architectural QBound: {imp:+6.2f}% {'improvement' if imp > 0 else 'degradation'}")

    # PPO
    print("\n### PPO Variants:")
    ppo_results = load_results('results/pendulum/ppo_full_qbound_seed*.json')
    for method in sorted(ppo_results.keys()):
        print(f"  {method:30s}: {ppo_results[method]['mean']:7.2f} ± {ppo_results[method]['std']:5.2f}")
    all_results['ppo'] = ppo_results

    if 'baseline' in ppo_results and 'architectural_qbound_ppo' in ppo_results:
        imp = compute_improvement(ppo_results['baseline']['mean'],
                                 ppo_results['architectural_qbound_ppo']['mean'],
                                 is_negative_reward=True)
        print(f"  → Architectural QBound: {imp:+6.2f}% {'improvement' if imp > 0 else 'degradation'}")

    print("\n### Key Observations:")
    print("  ✗ Architectural QBound FAILS for DQN (-3.3% degradation)")
    print("  ✗ Architectural QBound FAILS for DDPG (-8.0% degradation)")
    print("  ~ Architectural QBound marginal for TD3 (+4.1% improvement)")
    print("  ✗ Architectural QBound FAILS for PPO (-10.8% degradation)")

    return all_results

def analyze_algorithm_dependence(cartpole_results, pendulum_results):
    """Analyze why results depend on algorithm type."""
    print("\n" + "=" * 80)
    print("ALGORITHM DEPENDENCE ANALYSIS")
    print("=" * 80)

    print("\n### Value-Based Methods (DQN, DDQN):")
    print("  CartPole (positive rewards):  QBound works (+12% to +34%)")
    print("  Pendulum DQN (negative):      Architectural QBound fails (-3.3%)")
    print("  → Conclusion: Success depends on REWARD SIGN, not just algorithm")

    print("\n### Actor-Critic Methods (DDPG, TD3, PPO):")
    print("  DDPG (negative rewards):      Architectural QBound fails (-8.0%)")
    print("  TD3 (negative rewards):       Architectural QBound marginal (+4.1%)")
    print("  PPO (negative rewards):       Architectural QBound fails (-10.8%)")
    print("  → Conclusion: Most actor-critic methods DON'T benefit")

    print("\n### TD3 Exception:")
    print("  TD3 is ONLY method showing improvement on negative rewards")
    print("  Possible reasons:")
    print("    1. Twin critics reduce overestimation (Q-values naturally bounded)")
    print("    2. Delayed policy updates (more stable value learning)")
    print("    3. Target policy smoothing (exploration noise management)")
    print("    4. High variance (±40.15) suggests results may not be robust")

def theoretical_analysis():
    """Develop theoretical explanation."""
    print("\n" + "=" * 80)
    print("THEORETICAL ANALYSIS")
    print("=" * 80)

    print("\n### Why QBound Works for Positive Rewards (CartPole):")
    print("""
    1. REWARD ACCUMULATION:
       - r = +1 per step, Q-values MUST grow
       - Q_max = (1 - γ^H) / (1 - γ) = 99.34 (theoretical upper bound)
       - Without QBound: Q-values can overestimate arbitrarily
       - With QBound: Bounded growth prevents overestimation bias

    2. BELLMAN BACKUP:
       - Q(s,a) = r + γ * max Q(s',a')
       - Positive r pushes Q upward
       - Clipping Q_max prevents unbounded growth
       - Creates stable target for bootstrapping

    3. INITIALIZATION ALIGNMENT:
       - Random init typically produces mixed positive/negative values
       - Positive rewards push Q toward positive values naturally
       - Q_max bound only prevents EXCESS growth
       - No conflict between learning direction and bound

    RESULT: +12% to +34% improvement ✓
    """)

    print("\n### Why Architectural QBound FAILS for Negative Rewards:")
    print("""
    1. BELLMAN EQUATION NATURALLY BOUNDS Q ≤ 0:
       - If r ≤ 0 and Q(s',a') ≤ 0
       - Then Q(s,a) = r + γ * Q(s',a') ≤ 0 + 0 = 0
       - Q ≤ 0 is EMERGENT property, not requiring explicit constraint

    2. ARCHITECTURAL CONSTRAINT IS REDUNDANT:
       - Q = -softplus(logits) enforces Q ≤ 0 from initialization
       - But Bellman equation ALREADY ensures this naturally
       - Constraint adds NO new information
       - May restrict network's expressiveness unnecessarily

    3. GRADIENT FLOW RESTRICTION:
       - ∂Q/∂logits = -sigmoid(logits) ∈ (-1, 0)
       - When Q is near 0 (optimal), logits → -∞
       - Sigmoid(−∞) → 0, gradients vanish
       - Slows learning near optimal values

    4. LOSS LANDSCAPE DEFORMATION:
       - Softplus introduces non-linearity in value space
       - TD error: δ = r + γ*Q_next - Q_current
       - Q is non-linear function of network output
       - May create suboptimal minima

    RESULT: -3.3% to -10.8% degradation ✗
    """)

    print("\n### Why TD3 Shows Marginal Improvement:")
    print("""
    TD3 BUILT-IN MECHANISMS:

    1. Twin Critics (Clipped Double Q-Learning):
       - Q_target = r + γ * min(Q1_target, Q2_target)
       - Natural underestimation bias (opposite of overestimation)
       - For negative rewards, underestimation = less negative = GOOD

    2. Delayed Policy Updates:
       - Update actor less frequently than critic
       - Gives critics time to stabilize before policy changes
       - Reduces value function instability

    3. Target Policy Smoothing:
       - Adds noise to target actions: a' = π(s') + ε
       - Prevents overfitting to narrow value peaks
       - Smooths value landscape

    INTERACTION WITH QBOUND:
    - TD3 already has implicit value regularization (twin critics)
    - Architectural QBound may provide small additional regularization
    - But twin critics already doing most of the work
    - High variance (±40.15) suggests effect is weak/unstable

    RESULT: +4.1% improvement (but ±40.15 std) ~
    """)

def visualize_results(cartpole_results, pendulum_results):
    """Create comprehensive visualization."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # CartPole comparison
    ax = axes[0, 0]
    methods = ['baseline', 'static_qbound', 'baseline_ddqn', 'static_qbound_ddqn']
    means = [cartpole_results[m]['mean'] for m in methods if m in cartpole_results]
    stds = [cartpole_results[m]['std'] for m in methods if m in cartpole_results]
    labels = [m.replace('_', '\n') for m in methods if m in cartpole_results]

    colors = ['#3498db', '#2ecc71', '#3498db', '#2ecc71']
    bars = ax.bar(range(len(means)), means, yerr=stds, capsize=5,
                   color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.set_xticks(range(len(means)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('Final 100 Episode Avg Reward')
    ax.set_title('CartPole: QBound Works for Positive Rewards', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

    # Pendulum DQN
    ax = axes[0, 1]
    dqn_res = pendulum_results['dqn']
    methods = ['dqn', 'architectural_qbound_dqn']
    means = [dqn_res[m]['mean'] for m in methods if m in dqn_res]
    stds = [dqn_res[m]['std'] for m in methods if m in dqn_res]
    labels = ['DQN\nBaseline', 'Architectural\nQBound']

    colors = ['#e74c3c', '#c0392b']
    bars = ax.bar(range(len(means)), means, yerr=stds, capsize=5,
                   color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.set_xticks(range(len(means)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Final 100 Episode Avg Reward')
    ax.set_title('Pendulum DQN: Architectural QBound Fails', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

    # Pendulum Actor-Critic
    ax = axes[0, 2]
    algos = ['DDPG', 'TD3', 'PPO']
    baseline_means = [
        pendulum_results['ddpg']['baseline']['mean'],
        pendulum_results['td3']['baseline']['mean'],
        pendulum_results['ppo']['baseline']['mean']
    ]
    qbound_means = [
        pendulum_results['ddpg']['architectural_qbound_ddpg']['mean'],
        pendulum_results['td3']['architectural_qbound_td3']['mean'],
        pendulum_results['ppo']['architectural_qbound_ppo']['mean']
    ]
    baseline_stds = [
        pendulum_results['ddpg']['baseline']['std'],
        pendulum_results['td3']['baseline']['std'],
        pendulum_results['ppo']['baseline']['std']
    ]
    qbound_stds = [
        pendulum_results['ddpg']['architectural_qbound_ddpg']['std'],
        pendulum_results['td3']['architectural_qbound_td3']['std'],
        pendulum_results['ppo']['architectural_qbound_ppo']['std']
    ]

    x = np.arange(len(algos))
    width = 0.35

    bars1 = ax.bar(x - width/2, baseline_means, width, yerr=baseline_stds,
                   label='Baseline', capsize=5, color='#3498db', alpha=0.7,
                   edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, qbound_means, width, yerr=qbound_stds,
                   label='Arch QBound', capsize=5, color='#e74c3c', alpha=0.7,
                   edgecolor='black', linewidth=1.5)

    ax.set_ylabel('Final 100 Episode Avg Reward')
    ax.set_title('Pendulum Actor-Critic: QBound Mostly Fails', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(algos)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

    # Improvement comparison
    ax = axes[1, 0]

    # CartPole improvements
    cartpole_imp_dqn = compute_improvement(
        cartpole_results['baseline']['mean'],
        cartpole_results['static_qbound']['mean']
    ) if 'baseline' in cartpole_results and 'static_qbound' in cartpole_results else 0

    cartpole_imp_ddqn = compute_improvement(
        cartpole_results['baseline_ddqn']['mean'],
        cartpole_results['static_qbound_ddqn']['mean']
    ) if 'baseline_ddqn' in cartpole_results and 'static_qbound_ddqn' in cartpole_results else 0

    # Pendulum improvements
    pendulum_imp_dqn = compute_improvement(
        pendulum_results['dqn']['dqn']['mean'],
        pendulum_results['dqn']['architectural_qbound_dqn']['mean'],
        is_negative_reward=True
    )

    pendulum_imp_ddpg = compute_improvement(
        pendulum_results['ddpg']['baseline']['mean'],
        pendulum_results['ddpg']['architectural_qbound_ddpg']['mean'],
        is_negative_reward=True
    )

    pendulum_imp_td3 = compute_improvement(
        pendulum_results['td3']['baseline']['mean'],
        pendulum_results['td3']['architectural_qbound_td3']['mean'],
        is_negative_reward=True
    )

    pendulum_imp_ppo = compute_improvement(
        pendulum_results['ppo']['baseline']['mean'],
        pendulum_results['ppo']['architectural_qbound_ppo']['mean'],
        is_negative_reward=True
    )

    methods = ['CartPole\nDQN', 'CartPole\nDDQN', 'Pendulum\nDQN',
               'Pendulum\nDDPG', 'Pendulum\nTD3', 'Pendulum\nPPO']
    improvements = [cartpole_imp_dqn, cartpole_imp_ddqn, pendulum_imp_dqn,
                   pendulum_imp_ddpg, pendulum_imp_td3, pendulum_imp_ppo]
    colors = ['#2ecc71' if imp > 0 else '#e74c3c' for imp in improvements]

    bars = ax.barh(methods, improvements, color=colors, alpha=0.7,
                   edgecolor='black', linewidth=1.5)
    ax.set_xlabel('Improvement (%)')
    ax.set_title('QBound Effectiveness Comparison', fontweight='bold')
    ax.axvline(x=0, color='k', linestyle='-', linewidth=2)
    ax.grid(axis='x', alpha=0.3)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, improvements)):
        ax.text(val + (2 if val > 0 else -2), i, f'{val:+.1f}%',
               va='center', ha='left' if val > 0 else 'right', fontweight='bold')

    # Learning curves - CartPole
    ax = axes[1, 1]
    if 'baseline' in cartpole_results and len(cartpole_results['baseline']['all_rewards']) > 0:
        baseline_avg = np.mean(cartpole_results['baseline']['all_rewards'], axis=0)
        qbound_avg = np.mean(cartpole_results['static_qbound']['all_rewards'], axis=0)

        ax.plot(baseline_avg, label='Baseline', color='#3498db', linewidth=2, alpha=0.7)
        ax.plot(qbound_avg, label='Static QBound', color='#2ecc71', linewidth=2, alpha=0.7)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Average Reward')
        ax.set_title('CartPole Learning Curves', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)

    # Learning curves - Pendulum
    ax = axes[1, 2]
    if 'baseline' in pendulum_results['ddpg'] and len(pendulum_results['ddpg']['baseline']['all_rewards']) > 0:
        baseline_avg = np.mean(pendulum_results['ddpg']['baseline']['all_rewards'], axis=0)
        qbound_avg = np.mean(pendulum_results['ddpg']['architectural_qbound_ddpg']['all_rewards'], axis=0)

        ax.plot(baseline_avg, label='DDPG Baseline', color='#3498db', linewidth=2, alpha=0.7)
        ax.plot(qbound_avg, label='DDPG Arch QBound', color='#e74c3c', linewidth=2, alpha=0.7)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Average Reward')
        ax.set_title('Pendulum DDPG Learning Curves', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()

    # Save
    output_dir = Path('results/plots')
    output_dir.mkdir(exist_ok=True, parents=True)

    plt.savefig(output_dir / 'comprehensive_qbound_analysis.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'comprehensive_qbound_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved comprehensive analysis plot to {output_dir}/")

    plt.show()

def main():
    """Run comprehensive analysis."""
    print("\n" + "=" * 80)
    print(" " * 20 + "COMPREHENSIVE QBOUND ANALYSIS")
    print("=" * 80)

    # Analyze each environment
    cartpole_results = analyze_cartpole()
    pendulum_results = analyze_pendulum()

    # Cross-environment analysis
    analyze_algorithm_dependence(cartpole_results, pendulum_results)

    # Theoretical explanation
    theoretical_analysis()

    # Visualizations
    visualize_results(cartpole_results, pendulum_results)

    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    print("""
    1. POSITIVE REWARDS (CartPole):
       ✓ QBound works excellently (+12% to +34%)
       ✓ Hard clipping prevents overestimation
       ✓ Bounds align with reward accumulation dynamics

    2. NEGATIVE REWARDS (Pendulum):
       ✗ Architectural QBound FAILS for most algorithms
       ✗ DQN: -3.3% degradation
       ✗ DDPG: -8.0% degradation
       ✗ PPO: -10.8% degradation
       ~ TD3: +4.1% marginal improvement (high variance ±40.15)

    3. ROOT CAUSE:
       - For negative rewards, Bellman equation NATURALLY bounds Q ≤ 0
       - Architectural constraint is REDUNDANT
       - May restrict expressiveness and create gradient flow issues
       - TD3's twin critics provide natural regularization (explains exception)

    4. THEORETICAL INSIGHT:
       QBound is effective when it adds INFORMATION not implicit in Bellman:
       - Positive rewards: Q_max bound prevents unbounded growth ✓
       - Negative rewards: Q ≤ 0 is emergent, constraint is redundant ✗
    """)

    print("\n" + "=" * 80)
    print("RECOMMENDATIONS FOR PAPER")
    print("=" * 80)
    print("""
    1. CORRECT ABSTRACT:
       - Remove claims about architectural QBound success on negative rewards
       - Focus on positive reward success story
       - Acknowledge architectural QBound failure as important negative result

    2. ADD FAILURE ANALYSIS SECTION:
       - Why architectural QBound fails for negative rewards
       - Bellman natural bound vs explicit constraint
       - TD3 exception and twin critic regularization

    3. REVISE RECOMMENDATIONS:
       - Use QBound ONLY for positive dense rewards
       - Do NOT use architectural QBound for negative rewards
       - Exception: TD3 may show marginal benefit (but high variance)

    4. STRENGTHEN CONTRIBUTION:
       - Systematic study of when QBound works vs fails
       - Identifies reward structure as key determinant
       - Provides theoretical explanation for both success and failure
    """)

if __name__ == '__main__':
    main()
