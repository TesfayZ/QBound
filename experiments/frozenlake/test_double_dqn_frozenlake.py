"""
FrozenLake: Double DQN vs. QBound Comparison

This experiment tests whether Double DQN's failure in CartPole extends to FrozenLake.
FrozenLake previously showed QBound success (19.4% faster convergence).

Comparison:
1. Baseline DQN - Standard DQN (may overestimate)
2. Double DQN - Industry standard solution (soft pessimism)
3. QBound - Hard Q-value clipping (causes underestimation)

Expected Results:
- FrozenLake: Stochastic, sparse terminal rewards
- Hypothesis: Pessimism may HELP here (opposite of CartPole)
- If both Double DQN and QBound succeed → Validates environment-specific theory
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from datetime import datetime
import gymnasium as gym
import sys
import os

sys.path.insert(0, '/root/projects/QBound/src')

from dqn_agent import DQNAgent
from double_dqn_agent import DoubleDQNAgent

# Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)
import torch
torch.manual_seed(SEED)
import random
random.seed(SEED)


class FrozenLakeWrapper:
    """
    Wrapper for FrozenLake to work with our DQN agent.
    Converts discrete state to one-hot encoding.
    """

    def __init__(self, is_slippery=True, map_name="4x4", seed=None):
        self.env = gym.make('FrozenLake-v1', is_slippery=is_slippery, map_name=map_name)
        self.observation_space = self.env.observation_space.n  # One-hot size
        self.action_space = self.env.action_space.n
        self.seed = seed

    def reset(self):
        """Reset and return one-hot encoded state."""
        if self.seed is not None:
            state, _ = self.env.reset(seed=self.seed)
            self.seed += 1
        else:
            state, _ = self.env.reset()
        return self._state_to_onehot(state)

    def _state_to_onehot(self, state):
        """Convert discrete state to one-hot encoding."""
        onehot = np.zeros(self.observation_space)
        onehot[state] = 1.0
        return onehot

    def step(self, action):
        """Take step and return one-hot encoded next state."""
        next_state, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return self._state_to_onehot(next_state), reward, done, info


def train_agent(env, agent, num_episodes, max_steps, desc="Training"):
    """Train agent and return episode rewards."""
    episode_rewards = []
    success_count = 0

    for episode in tqdm(range(num_episodes), desc=desc):
        state = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)

            agent.store_transition(state, action, reward, next_state, done)
            agent.train_step()

            episode_reward += reward
            state = next_state

            if done:
                if reward > 0:  # Goal reached
                    success_count += 1
                break

        episode_rewards.append(episode_reward)

        if (episode + 1) % 400 == 0:
            recent_avg = np.mean(episode_rewards[-min(100, len(episode_rewards)):])
            recent_success_rate = success_count / (episode + 1)
            print(f"  Episode {episode+1}/{num_episodes} - Avg reward: {recent_avg:.3f}, Success rate: {recent_success_rate:.2%}, ε={agent.epsilon:.3f}")

    return episode_rewards


def evaluate_agent(env, agent, num_eval_episodes=100, max_steps=100):
    """Evaluate agent without exploration."""
    successes = 0
    total_rewards = []

    for _ in range(num_eval_episodes):
        state = env.reset()
        episode_reward = 0

        for _ in range(max_steps):
            action = agent.select_action(state, eval_mode=True)
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            state = next_state

            if done:
                if reward > 0:  # Goal reached
                    successes += 1
                break

        total_rewards.append(episode_reward)

    success_rate = successes / num_eval_episodes
    avg_reward = np.mean(total_rewards)
    return success_rate, avg_reward


def main():
    gamma = 0.95  # Lower gamma for FrozenLake
    max_steps = 100
    num_episodes = 2000
    learning_rate = 0.001
    epsilon_decay = 0.995
    map_name = "4x4"
    is_slippery = True

    # Compute theoretical Q_max (for QBound)
    # FrozenLake: Terminal reward of +1, sparse rewards
    qmax_theoretical = 1.0  # Terminal reward only

    print("\n" + "="*80)
    print("FrozenLake: Double DQN vs. QBound Comparison")
    print("="*80)
    print("Testing three approaches to Q-value estimation:")
    print("1. Baseline DQN (may overestimate)")
    print("2. Double DQN (soft pessimism - RECOMMENDED)")
    print(f"3. QBound (hard clip at Q_max={qmax_theoretical:.2f})")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Episodes: {num_episodes}")
    print(f"  Max steps per episode: {max_steps}")
    print(f"  Map: {map_name}")
    print(f"  Slippery: {is_slippery}")
    print(f"  Discount factor γ: {gamma}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Theoretical Q_max: {qmax_theoretical:.2f}")
    print("="*80)

    # =====================================================================
    # 1. BASELINE DQN
    # =====================================================================
    print("\n>>> Training Baseline DQN...")
    env_baseline = FrozenLakeWrapper(is_slippery=is_slippery, map_name=map_name, seed=SEED)
    agent_baseline = DQNAgent(
        state_dim=env_baseline.observation_space,
        action_dim=env_baseline.action_space,
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=epsilon_decay,
        buffer_size=10000,
        batch_size=64,
        target_update_freq=100,
        use_qclip=False,  # No clipping
        device="cpu"
    )

    rewards_baseline = train_agent(env_baseline, agent_baseline, num_episodes, max_steps,
                                   desc="Baseline DQN")

    # =====================================================================
    # 2. DOUBLE DQN
    # =====================================================================
    print("\n>>> Training Double DQN...")
    env_double = FrozenLakeWrapper(is_slippery=is_slippery, map_name=map_name, seed=SEED)
    agent_double = DoubleDQNAgent(
        state_dim=env_double.observation_space,
        action_dim=env_double.action_space,
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=epsilon_decay,
        buffer_size=10000,
        batch_size=64,
        target_update_freq=100,
        device="cpu",
        use_huber_loss=True,  # More robust
        gradient_clip=1.0  # Prevent exploding gradients
    )

    rewards_double = train_agent(env_double, agent_double, num_episodes, max_steps,
                                 desc="Double DQN")

    # =====================================================================
    # 3. QBOUND (for comparison)
    # =====================================================================
    print("\n>>> Training QBound (for comparison)...")
    env_qbound = FrozenLakeWrapper(is_slippery=is_slippery, map_name=map_name, seed=SEED)
    agent_qbound = DQNAgent(
        state_dim=env_qbound.observation_space,
        action_dim=env_qbound.action_space,
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=epsilon_decay,
        buffer_size=10000,
        batch_size=64,
        target_update_freq=100,
        use_qclip=True,  # Enable QBound
        qclip_max=qmax_theoretical,
        qclip_min=0.0,
        device="cpu"
    )

    rewards_qbound = train_agent(env_qbound, agent_qbound, num_episodes, max_steps,
                                 desc="QBound")

    # =====================================================================
    # EVALUATION
    # =====================================================================
    print("\n" + "="*80)
    print("Final Evaluation (100 episodes)")
    print("="*80)

    print("\n>>> Evaluating agents...")
    # Reset environment for consistent evaluation
    eval_env = FrozenLakeWrapper(is_slippery=is_slippery, map_name=map_name, seed=SEED + 10000)

    baseline_success, baseline_reward = evaluate_agent(eval_env, agent_baseline, 100, max_steps)
    print(f"  Baseline DQN:  Success rate: {baseline_success:.2%}, Avg reward: {baseline_reward:.3f}")

    eval_env = FrozenLakeWrapper(is_slippery=is_slippery, map_name=map_name, seed=SEED + 10000)
    double_success, double_reward = evaluate_agent(eval_env, agent_double, 100, max_steps)
    print(f"  Double DQN:    Success rate: {double_success:.2%}, Avg reward: {double_reward:.3f}")

    eval_env = FrozenLakeWrapper(is_slippery=is_slippery, map_name=map_name, seed=SEED + 10000)
    qbound_success, qbound_reward = evaluate_agent(eval_env, agent_qbound, 100, max_steps)
    print(f"  QBound:        Success rate: {qbound_success:.2%}, Avg reward: {qbound_reward:.3f}")

    # =====================================================================
    # RESULTS SUMMARY
    # =====================================================================
    print("\n" + "="*80)
    print("Training Results Summary")
    print("="*80)

    total_baseline = sum(rewards_baseline)
    total_double = sum(rewards_double)
    total_qbound = sum(rewards_qbound)

    print("\nTotal cumulative reward:")
    print(f"  Baseline DQN:  {total_baseline:.1f}")
    print(f"  Double DQN:    {total_double:.1f} ({((total_double/total_baseline-1)*100) if total_baseline > 0 else 0:+.1f}%)")
    print(f"  QBound:        {total_qbound:.1f} ({((total_qbound/total_baseline-1)*100) if total_baseline > 0 else 0:+.1f}%)")

    print("\nAverage episode reward:")
    print(f"  Baseline DQN:  {np.mean(rewards_baseline):.3f}")
    print(f"  Double DQN:    {np.mean(rewards_double):.3f}")
    print(f"  QBound:        {np.mean(rewards_qbound):.3f}")

    # Convergence speed analysis (episodes to reach 50% success)
    def find_convergence_episode(rewards, window=100, threshold=0.5):
        """Find first episode where rolling success rate exceeds threshold."""
        for i in range(window, len(rewards)):
            rolling_success = np.mean(rewards[i-window:i])
            if rolling_success >= threshold:
                return i
        return None

    baseline_conv = find_convergence_episode(rewards_baseline)
    double_conv = find_convergence_episode(rewards_double)
    qbound_conv = find_convergence_episode(rewards_qbound)

    print("\nConvergence speed (episodes to 50% success rate):")
    print(f"  Baseline DQN:  {baseline_conv if baseline_conv else 'Not reached'}")
    print(f"  Double DQN:    {double_conv if double_conv else 'Not reached'}")
    print(f"  QBound:        {qbound_conv if qbound_conv else 'Not reached'}")

    if baseline_conv and double_conv:
        speedup_double = baseline_conv / double_conv
        print(f"  Double DQN speedup: {speedup_double:.2f}x {'faster' if speedup_double > 1 else 'slower'}")

    if baseline_conv and qbound_conv:
        speedup_qbound = baseline_conv / qbound_conv
        print(f"  QBound speedup: {speedup_qbound:.2f}x {'faster' if speedup_qbound > 1 else 'slower'}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        'timestamp': timestamp,
        'config': {
            'num_episodes': num_episodes,
            'max_steps': max_steps,
            'learning_rate': learning_rate,
            'gamma': gamma,
            'epsilon_decay': epsilon_decay,
            'seed': SEED,
            'map_name': map_name,
            'is_slippery': is_slippery,
            'qmax_theoretical': qmax_theoretical
        },
        'training': {
            'baseline': {
                'rewards': rewards_baseline,
                'total_reward': total_baseline,
                'mean_reward': float(np.mean(rewards_baseline)),
                'convergence_episode': baseline_conv
            },
            'double_dqn': {
                'rewards': rewards_double,
                'total_reward': total_double,
                'mean_reward': float(np.mean(rewards_double)),
                'convergence_episode': double_conv
            },
            'qbound': {
                'rewards': rewards_qbound,
                'total_reward': total_qbound,
                'mean_reward': float(np.mean(rewards_qbound)),
                'convergence_episode': qbound_conv
            }
        },
        'evaluation': {
            'baseline': {'success_rate': float(baseline_success), 'avg_reward': float(baseline_reward)},
            'double_dqn': {'success_rate': float(double_success), 'avg_reward': float(double_reward)},
            'qbound': {'success_rate': float(qbound_success), 'avg_reward': float(qbound_reward)}
        }
    }

    os.makedirs('results/frozenlake', exist_ok=True)
    results_path = f'results/frozenlake/double_dqn_comparison_{timestamp}.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to: {results_path}")

    # =====================================================================
    # PLOT
    # =====================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Learning curves
    window = 100
    smooth_baseline = np.convolve(rewards_baseline, np.ones(window)/window, mode='valid')
    smooth_double = np.convolve(rewards_double, np.ones(window)/window, mode='valid')
    smooth_qbound = np.convolve(rewards_qbound, np.ones(window)/window, mode='valid')

    axes[0].plot(smooth_baseline, label='Baseline DQN', linewidth=2, alpha=0.8)
    axes[0].plot(smooth_double, label='Double DQN', linewidth=2, alpha=0.8)
    axes[0].plot(smooth_qbound, label=f'QBound (Q_max={qmax_theoretical:.1f})', linewidth=2, alpha=0.8)
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Success Rate (100-episode moving avg)')
    axes[0].set_title('FrozenLake: Double DQN vs. QBound')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Success rate comparison
    variants = ['Baseline\nDQN', 'Double\nDQN', 'QBound']
    success_rates = [baseline_success * 100, double_success * 100, qbound_success * 100]

    bars = axes[1].bar(variants, success_rates, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    axes[1].set_ylabel('Success Rate (%)')
    axes[1].set_title('Final Success Rate (100 eval episodes)')
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].set_ylim([0, 105])

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=12)

    plt.tight_layout()

    os.makedirs('results/plots', exist_ok=True)
    plot_path = f'results/plots/frozenlake_double_dqn_comparison_{timestamp}.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to: {plot_path}")

    print("\n" + "="*80)
    print("FrozenLake Double DQN Comparison Complete!")
    print("="*80)

    # Print conclusion
    print("\n🎯 CONCLUSION:")
    if double_success >= baseline_success * 0.95 and qbound_success >= baseline_success * 0.95:
        print(f"✅ Both pessimistic approaches work in FrozenLake!")
        print(f"   Double DQN: {double_success:.1%}, QBound: {qbound_success:.1%}, Baseline: {baseline_success:.1%}")
        print("   → Confirms environment-specific hypothesis:")
        print("   → Sparse rewards + stochasticity → pessimism helps")
        print("   → Dense rewards + long horizon → pessimism hurts (CartPole)")
    elif double_success >= baseline_success * 0.95:
        print(f"✅ Double DQN works (success: {double_success:.1%})")
        print(f"❌ QBound fails (success: {qbound_success:.1%})")
        print("   → Soft pessimism > hard clipping")
    else:
        print(f"⚠️  Both pessimistic approaches struggle in FrozenLake")
        print(f"   Baseline: {baseline_success:.1%}, Double DQN: {double_success:.1%}, QBound: {qbound_success:.1%}")


if __name__ == "__main__":
    main()
