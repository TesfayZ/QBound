"""
CartPole: Double DQN vs. QBound Comparison

This experiment tests the hypothesis that Double DQN (soft pessimism)
outperforms QBound (hard clipping) for addressing overestimation bias.

Comparison:
1. Baseline DQN - Standard DQN (may overestimate)
2. Double DQN - Industry standard solution (soft pessimism)
3. QBound - Hard Q-value clipping (causes underestimation)

Expected Results:
- Double DQN ≥ Baseline > QBound
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

from dqn_agent import DQNAgent  # Standard DQN
from double_dqn_agent import DoubleDQNAgent  # Our new implementation

# Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)
import torch
torch.manual_seed(SEED)
import random
random.seed(SEED)


class CartPoleWrapper:
    """Wrapper for CartPole to work with our DQN agent."""

    def __init__(self, seed=None, max_episode_steps=500):
        self.env = gym.make('CartPole-v1', max_episode_steps=max_episode_steps)
        self.observation_space = self.env.observation_space.shape[0]
        self.action_space = self.env.action_space.n
        self.seed = seed

    def reset(self):
        """Reset and return state."""
        if self.seed is not None:
            state, _ = self.env.reset(seed=self.seed)
            self.seed += 1
        else:
            state, _ = self.env.reset()
        return state

    def step(self, action):
        """Take step and return next state."""
        next_state, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return next_state, reward, done, info


def train_agent(env, agent, num_episodes, max_steps, desc="Training"):
    """Train agent and return episode rewards."""
    episode_rewards = []

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
                break

        episode_rewards.append(episode_reward)

        if (episode + 1) % 100 == 0:
            recent_avg = np.mean(episode_rewards[-min(50, len(episode_rewards)):])
            print(f"  Episode {episode+1}/{num_episodes} - Recent avg: {recent_avg:.1f}, ε={agent.epsilon:.3f}")

    return episode_rewards


def evaluate_agent(env, agent, num_eval_episodes=10, max_steps=500):
    """Evaluate agent without exploration."""
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
                break

        total_rewards.append(episode_reward)

    return np.mean(total_rewards), np.std(total_rewards)


def main():
    gamma = 0.99
    max_steps = 500
    num_episodes = 500
    learning_rate = 0.001
    epsilon_decay = 0.995

    # Compute theoretical Q_max (for QBound)
    qmax_theoretical = (1 - gamma**max_steps) / (1 - gamma)

    print("\n" + "="*80)
    print("CartPole: Double DQN vs. QBound Comparison")
    print("="*80)
    print("Testing three approaches to Q-value estimation:")
    print("1. Baseline DQN (may overestimate)")
    print("2. Double DQN (soft pessimism - RECOMMENDED)")
    print(f"3. QBound (hard clip at Q_max={qmax_theoretical:.2f})")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Episodes: {num_episodes}")
    print(f"  Max steps per episode: {max_steps}")
    print(f"  Discount factor γ: {gamma}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Theoretical Q_max: {qmax_theoretical:.2f}")
    print("="*80)

    # =====================================================================
    # 1. BASELINE DQN
    # =====================================================================
    print("\n>>> Training Baseline DQN...")
    env_baseline = CartPoleWrapper(seed=SEED, max_episode_steps=max_steps)
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
    env_double = CartPoleWrapper(seed=SEED, max_episode_steps=max_steps)
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
    env_qbound = CartPoleWrapper(seed=SEED, max_episode_steps=max_steps)
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
    print("Blind Evaluation (10 episodes each)")
    print("="*80)

    print("\n>>> Evaluation at 500 steps (training length)...")
    eval_env_500 = CartPoleWrapper(seed=SEED + 1000, max_episode_steps=500)

    baseline_500_mean, baseline_500_std = evaluate_agent(eval_env_500, agent_baseline, 10, 500)
    eval_env_500 = CartPoleWrapper(seed=SEED + 1000, max_episode_steps=500)
    double_500_mean, double_500_std = evaluate_agent(eval_env_500, agent_double, 10, 500)
    eval_env_500 = CartPoleWrapper(seed=SEED + 1000, max_episode_steps=500)
    qbound_500_mean, qbound_500_std = evaluate_agent(eval_env_500, agent_qbound, 10, 500)

    print(f"  Baseline DQN:  {baseline_500_mean:.1f} ± {baseline_500_std:.1f}")
    print(f"  Double DQN:    {double_500_mean:.1f} ± {double_500_std:.1f}")
    print(f"  QBound:        {qbound_500_mean:.1f} ± {qbound_500_std:.1f}")

    print("\n>>> Evaluation at 1000 steps (2x training length)...")
    eval_env_1000 = CartPoleWrapper(seed=SEED + 2000, max_episode_steps=1000)

    baseline_1000_mean, baseline_1000_std = evaluate_agent(eval_env_1000, agent_baseline, 10, 1000)
    eval_env_1000 = CartPoleWrapper(seed=SEED + 2000, max_episode_steps=1000)
    double_1000_mean, double_1000_std = evaluate_agent(eval_env_1000, agent_double, 10, 1000)
    eval_env_1000 = CartPoleWrapper(seed=SEED + 2000, max_episode_steps=1000)
    qbound_1000_mean, qbound_1000_std = evaluate_agent(eval_env_1000, agent_qbound, 10, 1000)

    print(f"  Baseline DQN:  {baseline_1000_mean:.1f} ± {baseline_1000_std:.1f}")
    print(f"  Double DQN:    {double_1000_mean:.1f} ± {double_1000_std:.1f}")
    print(f"  QBound:        {qbound_1000_mean:.1f} ± {qbound_1000_std:.1f}")

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
    print(f"  Baseline DQN:  {total_baseline:,}")
    print(f"  Double DQN:    {total_double:,} ({((total_double/total_baseline-1)*100):+.1f}%)")
    print(f"  QBound:        {total_qbound:,} ({((total_qbound/total_baseline-1)*100):+.1f}%)")

    print("\nAverage episode reward:")
    print(f"  Baseline DQN:  {np.mean(rewards_baseline):.1f}")
    print(f"  Double DQN:    {np.mean(rewards_double):.1f}")
    print(f"  QBound:        {np.mean(rewards_qbound):.1f}")

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
            'qmax_theoretical': qmax_theoretical
        },
        'training': {
            'baseline': {
                'rewards': rewards_baseline,
                'total_reward': total_baseline,
                'mean_reward': float(np.mean(rewards_baseline))
            },
            'double_dqn': {
                'rewards': rewards_double,
                'total_reward': total_double,
                'mean_reward': float(np.mean(rewards_double))
            },
            'qbound': {
                'rewards': rewards_qbound,
                'total_reward': total_qbound,
                'mean_reward': float(np.mean(rewards_qbound))
            }
        },
        'evaluation': {
            'max_steps_500': {
                'baseline': {'mean': float(baseline_500_mean), 'std': float(baseline_500_std)},
                'double_dqn': {'mean': float(double_500_mean), 'std': float(double_500_std)},
                'qbound': {'mean': float(qbound_500_mean), 'std': float(qbound_500_std)}
            },
            'max_steps_1000': {
                'baseline': {'mean': float(baseline_1000_mean), 'std': float(baseline_1000_std)},
                'double_dqn': {'mean': float(double_1000_mean), 'std': float(double_1000_std)},
                'qbound': {'mean': float(qbound_1000_mean), 'std': float(qbound_1000_std)}
            }
        }
    }

    os.makedirs('results/cartpole', exist_ok=True)
    results_path = f'results/cartpole/double_dqn_comparison_{timestamp}.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to: {results_path}")

    # =====================================================================
    # PLOT
    # =====================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Learning curves
    window = 25
    smooth_baseline = np.convolve(rewards_baseline, np.ones(window)/window, mode='valid')
    smooth_double = np.convolve(rewards_double, np.ones(window)/window, mode='valid')
    smooth_qbound = np.convolve(rewards_qbound, np.ones(window)/window, mode='valid')

    axes[0].plot(smooth_baseline, label='Baseline DQN', linewidth=2, alpha=0.8)
    axes[0].plot(smooth_double, label='Double DQN', linewidth=2, alpha=0.8)
    axes[0].plot(smooth_qbound, label=f'QBound (Q_max={qmax_theoretical:.1f})', linewidth=2, alpha=0.8)
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Reward (25-episode moving avg)')
    axes[0].set_title('CartPole: Double DQN vs. QBound')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Evaluation comparison
    variants = ['Baseline\nDQN', 'Double\nDQN', 'QBound']
    training_means = [
        np.mean(rewards_baseline),
        np.mean(rewards_double),
        np.mean(rewards_qbound)
    ]
    eval_500 = [baseline_500_mean, double_500_mean, qbound_500_mean]
    eval_1000 = [baseline_1000_mean, double_1000_mean, qbound_1000_mean]

    x = np.arange(len(variants))
    width = 0.25

    bars1 = axes[1].bar(x - width, training_means, width, label='Training Mean')
    bars2 = axes[1].bar(x, eval_500, width, label='Eval (500 steps)')
    bars3 = axes[1].bar(x + width, eval_1000, width, label='Eval (1000 steps)')

    axes[1].set_ylabel('Mean Reward')
    axes[1].set_title('Training vs Evaluation Performance')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(variants)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.0f}',
                        ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    os.makedirs('results/plots', exist_ok=True)
    plot_path = f'results/plots/double_dqn_comparison_{timestamp}.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to: {plot_path}")

    print("\n" + "="*80)
    print("Double DQN Comparison Complete!")
    print("="*80)

    # Print conclusion
    print("\n🎯 CONCLUSION:")
    if total_double > total_qbound:
        improvement = ((total_double / total_qbound - 1) * 100)
        print(f"✅ Double DQN outperforms QBound by {improvement:.1f}%")
        print("   Soft pessimism > hard clipping!")
    else:
        print("⚠️  Unexpected result - needs investigation")


if __name__ == "__main__":
    main()
