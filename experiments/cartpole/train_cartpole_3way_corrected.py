"""
Three-way comparison for CartPole with CORRECTED Q-bounds:
1. Baseline (no QBound)
2. QBound with static bounds (Q_max = 99.34, computed with γ=0.99)
3. QBound with dynamic bounds (Q_max(t) = discounted sum from step t)

IMPORTANT: This uses the CORRECT formula with discount factor γ:
Q_max(t) = (1 - γ^(H-t)) / (1 - γ)

NOT the naive undiscounted formula:
Q_max(t) = H - t  (WRONG - ignores discounting!)
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from datetime import datetime
import gymnasium as gym
import sys
sys.path.insert(0, '/root/projects/QBound/src')

from dqn_agent import DQNAgent

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
        # Remove the default 500-step limit from CartPole-v1
        self.env = gym.make('CartPole-v1', max_episode_steps=max_episode_steps)
        self.observation_space = self.env.observation_space.shape[0]
        self.action_space = self.env.action_space.n
        self.seed = seed

    def reset(self):
        """Reset and return state."""
        if self.seed is not None:
            state, _ = self.env.reset(seed=self.seed)
            self.seed += 1  # Increment for next reset
        else:
            state, _ = self.env.reset()
        return state

    def step(self, action):
        """Take step and return next state."""
        next_state, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return next_state, reward, done, info


def compute_discounted_qmax(remaining_steps, gamma=0.99):
    """
    Compute correct Q_max with discounting.

    Q_max = r + γr + γ²r + ... + γ^(H-1)r
    For r=1: Q_max = (1 - γ^H) / (1 - γ)
    """
    if remaining_steps == 0:
        return 0.0
    return (1 - gamma**remaining_steps) / (1 - gamma)


def train_agent(env, agent, num_episodes, max_steps, use_step_aware=False, gamma=0.99):
    """Train agent and return episode rewards."""
    episode_rewards = []

    for episode in tqdm(range(num_episodes), desc="Training"):
        state = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)

            # Store transition
            if use_step_aware:
                # Pass current step to agent (agent computes dynamic Q_max internally)
                agent.store_transition(state, action, reward, next_state, done,
                                     current_step=step)
            else:
                agent.store_transition(state, action, reward, next_state, done,
                                     current_step=None)

            # Train
            agent.train_step()

            episode_reward += reward
            state = next_state

            if done:
                break

        episode_rewards.append(episode_reward)

        # Print progress
        if (episode + 1) % 100 == 0:
            recent_avg = np.mean(episode_rewards[-min(50, len(episode_rewards)):])
            print(f"  Episode {episode+1}/{num_episodes} - Recent avg: {recent_avg:.3f}, ε={agent.epsilon:.3f}")

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

    # Compute correct Q_max with discounting
    qmax_static_correct = compute_discounted_qmax(max_steps, gamma)

    print("\n" + "="*70)
    print("CartPole Three-Way Comparison (CORRECTED Q-BOUNDS)")
    print("="*70)
    print("1. Baseline (no QBound)")
    print(f"2. QBound with static bounds (Q_max = {qmax_static_correct:.2f})")
    print(f"3. QBound with dynamic bounds (Q_max(t) using discounted formula)")
    print("="*70)
    print(f"\nDiscount factor γ = {gamma}")
    print(f"Max steps = {max_steps}")
    print(f"Correct Q_max (static) = {qmax_static_correct:.2f}")
    print(f"Formula: Q_max = (1 - γ^H) / (1 - γ)")
    print("="*70)

    # Configuration
    num_episodes = 500
    learning_rate = 0.001
    epsilon_decay = 0.995

    env = CartPoleWrapper(seed=SEED, max_episode_steps=max_steps)

    # =====================================================================
    # 1. BASELINE (No QBound)
    # =====================================================================
    print("\n>>> Training Baseline (no QBound)...")
    agent_baseline = DQNAgent(
        state_dim=env.observation_space,
        action_dim=env.action_space,
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=epsilon_decay,
        buffer_size=10000,
        batch_size=64,
        target_update_freq=100,
        use_qclip=False,
        device="cpu"
    )

    rewards_baseline = train_agent(env, agent_baseline, num_episodes, max_steps,
                                   use_step_aware=False, gamma=gamma)

    # =====================================================================
    # 2. STATIC QBOUND (Correct discounted Q_max)
    # =====================================================================
    print("\n>>> Training QBound with STATIC bounds (correct discounting)...")
    env_static = CartPoleWrapper(seed=SEED, max_episode_steps=max_steps)
    agent_static = DQNAgent(
        state_dim=env_static.observation_space,
        action_dim=env_static.action_space,
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=epsilon_decay,
        buffer_size=10000,
        batch_size=64,
        target_update_freq=100,
        use_qclip=True,
        qclip_max=qmax_static_correct,
        qclip_min=0.0,
        device="cpu"
    )

    rewards_static = train_agent(env_static, agent_static, num_episodes, max_steps,
                                use_step_aware=False, gamma=gamma)

    # =====================================================================
    # 3. DYNAMIC QBOUND (Correct discounted formula)
    # =====================================================================
    print("\n>>> Training QBound with DYNAMIC bounds (correct discounted formula)...")
    env_dynamic = CartPoleWrapper(seed=SEED, max_episode_steps=max_steps)
    agent_dynamic = DQNAgent(
        state_dim=env_dynamic.observation_space,
        action_dim=env_dynamic.action_space,
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=epsilon_decay,
        buffer_size=10000,
        batch_size=64,
        target_update_freq=100,
        use_qclip=True,
        qclip_max=qmax_static_correct,  # Will be overridden by step-aware
        qclip_min=0.0,
        device="cpu",
        use_step_aware_qbound=True,
        max_episode_steps=max_steps,
        step_reward=1.0  # This is still used in the agent (for reference)
    )

    rewards_dynamic = train_agent(env_dynamic, agent_dynamic, num_episodes, max_steps,
                                 use_step_aware=True, gamma=gamma)

    # =====================================================================
    # SAVE MODELS
    # =====================================================================
    print("\n" + "="*70)
    print("Saving Trained Models")
    print("="*70)

    import os
    os.makedirs('models/cartpole', exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    baseline_path = f'models/cartpole/baseline_corrected_{timestamp}.pt'
    static_path = f'models/cartpole/static_qbound_corrected_{timestamp}.pt'
    dynamic_path = f'models/cartpole/dynamic_qbound_corrected_{timestamp}.pt'

    agent_baseline.save(baseline_path)
    agent_static.save(static_path)
    agent_dynamic.save(dynamic_path)

    print(f"✓ Baseline model saved to: {baseline_path}")
    print(f"✓ Static QBound model saved to: {static_path}")
    print(f"✓ Dynamic QBound model saved to: {dynamic_path}")

    # =====================================================================
    # BLIND EVALUATION
    # =====================================================================
    print("\n" + "="*70)
    print("Blind Evaluation (Models Don't Know Episode Length)")
    print("="*70)
    print("Testing with max_steps=500 and max_steps=1000")
    print("Dynamic QBound was trained expecting 500 steps.")
    print("="*70)

    print("\n>>> Evaluation with max_steps=500 (training length)...")
    eval_env_500 = CartPoleWrapper(seed=SEED + 1000, max_episode_steps=500)
    baseline_500_mean, baseline_500_std = evaluate_agent(eval_env_500, agent_baseline, 10, 500)

    eval_env_500 = CartPoleWrapper(seed=SEED + 1000, max_episode_steps=500)
    static_500_mean, static_500_std = evaluate_agent(eval_env_500, agent_static, 10, 500)

    eval_env_500 = CartPoleWrapper(seed=SEED + 1000, max_episode_steps=500)
    dynamic_500_mean, dynamic_500_std = evaluate_agent(eval_env_500, agent_dynamic, 10, 500)

    print(f"  Baseline:       {baseline_500_mean:.1f} ± {baseline_500_std:.1f}")
    print(f"  Static QBound:  {static_500_mean:.1f} ± {static_500_std:.1f}")
    print(f"  Dynamic QBound: {dynamic_500_mean:.1f} ± {dynamic_500_std:.1f}")

    print("\n>>> Evaluation with max_steps=1000 (2x training length)...")
    print("NOTE: Dynamic QBound trained assuming 500 max steps!")
    eval_env_1000 = CartPoleWrapper(seed=SEED + 2000, max_episode_steps=1000)
    baseline_1000_mean, baseline_1000_std = evaluate_agent(eval_env_1000, agent_baseline, 10, 1000)

    eval_env_1000 = CartPoleWrapper(seed=SEED + 2000, max_episode_steps=1000)
    static_1000_mean, static_1000_std = evaluate_agent(eval_env_1000, agent_static, 10, 1000)

    eval_env_1000 = CartPoleWrapper(seed=SEED + 2000, max_episode_steps=1000)
    dynamic_1000_mean, dynamic_1000_std = evaluate_agent(eval_env_1000, agent_dynamic, 10, 1000)

    print(f"  Baseline:       {baseline_1000_mean:.1f} ± {baseline_1000_std:.1f}")
    print(f"  Static QBound:  {static_1000_mean:.1f} ± {static_1000_std:.1f}")
    print(f"  Dynamic QBound: {dynamic_1000_mean:.1f} ± {dynamic_1000_std:.1f}")

    # =====================================================================
    # RESULTS SUMMARY
    # =====================================================================
    print("\n" + "="*70)
    print("Training Results Summary")
    print("="*70)

    total_baseline = sum(rewards_baseline)
    total_static = sum(rewards_static)
    total_dynamic = sum(rewards_dynamic)

    print("\nTotal cumulative reward:")
    print(f"  Baseline:       {total_baseline:,}")
    print(f"  Static QBound:  {total_static:,} ({((total_static/total_baseline-1)*100):+.1f}%)")
    print(f"  Dynamic QBound: {total_dynamic:,} ({((total_dynamic/total_baseline-1)*100):+.1f}%)")

    print("\nAverage episode reward:")
    print(f"  Baseline:       {np.mean(rewards_baseline):.1f}")
    print(f"  Static QBound:  {np.mean(rewards_static):.1f}")
    print(f"  Dynamic QBound: {np.mean(rewards_dynamic):.1f}")

    # Save results
    results = {
        'timestamp': timestamp,
        'config': {
            'num_episodes': num_episodes,
            'max_steps': max_steps,
            'learning_rate': learning_rate,
            'gamma': gamma,
            'epsilon_decay': epsilon_decay,
            'seed': SEED,
            'qmax_static_correct': qmax_static_correct,
            'formula': 'Q_max = (1 - gamma^H) / (1 - gamma)'
        },
        'training': {
            'baseline': {
                'rewards': rewards_baseline,
                'total_reward': total_baseline,
                'mean_reward': float(np.mean(rewards_baseline))
            },
            'static_qbound': {
                'rewards': rewards_static,
                'total_reward': total_static,
                'mean_reward': float(np.mean(rewards_static)),
                'qmax': qmax_static_correct
            },
            'dynamic_qbound': {
                'rewards': rewards_dynamic,
                'total_reward': total_dynamic,
                'mean_reward': float(np.mean(rewards_dynamic)),
                'qmax_formula': 'Q_max(t) = (1 - gamma^(H-t)) / (1 - gamma)'
            }
        },
        'evaluation': {
            'max_steps_500': {
                'baseline': {'mean': float(baseline_500_mean), 'std': float(baseline_500_std)},
                'static_qbound': {'mean': float(static_500_mean), 'std': float(static_500_std)},
                'dynamic_qbound': {'mean': float(dynamic_500_mean), 'std': float(dynamic_500_std)}
            },
            'max_steps_1000': {
                'baseline': {'mean': float(baseline_1000_mean), 'std': float(baseline_1000_std)},
                'static_qbound': {'mean': float(static_1000_mean), 'std': float(static_1000_std)},
                'dynamic_qbound': {'mean': float(dynamic_1000_mean), 'std': float(dynamic_1000_std)}
            }
        },
        'model_paths': {
            'baseline': baseline_path,
            'static_qbound': static_path,
            'dynamic_qbound': dynamic_path
        }
    }

    os.makedirs('results/cartpole', exist_ok=True)
    results_path = f'results/cartpole/three_way_comparison_corrected_{timestamp}.json'
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
    smooth_static = np.convolve(rewards_static, np.ones(window)/window, mode='valid')
    smooth_dynamic = np.convolve(rewards_dynamic, np.ones(window)/window, mode='valid')

    axes[0].plot(smooth_baseline, label='Baseline', linewidth=2, alpha=0.8)
    axes[0].plot(smooth_static, label=f'Static QBound (Q_max={qmax_static_correct:.1f})', linewidth=2, alpha=0.8)
    axes[0].plot(smooth_dynamic, label='Dynamic QBound (discounted)', linewidth=2, alpha=0.8)
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Reward (25-episode moving avg)')
    axes[0].set_title('CartPole Learning Curves (Corrected Q-Bounds)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Cumulative rewards
    axes[1].plot(np.cumsum(rewards_baseline), label='Baseline', linewidth=2, alpha=0.8)
    axes[1].plot(np.cumsum(rewards_static), label=f'Static QBound (Q_max={qmax_static_correct:.1f})', linewidth=2, alpha=0.8)
    axes[1].plot(np.cumsum(rewards_dynamic), label='Dynamic QBound (discounted)', linewidth=2, alpha=0.8)
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Cumulative Reward')
    axes[1].set_title('Sample Efficiency')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    os.makedirs('results/plots', exist_ok=True)
    plot_path = f'results/plots/cartpole_3way_comparison_corrected_{timestamp}.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to: {plot_path}")

    # Save PDF for paper
    os.makedirs('QBound/figures', exist_ok=True)
    plot_pdf_path = f'QBound/figures/cartpole_3way_comparison_corrected_{timestamp}.pdf'
    plt.savefig(plot_pdf_path, dpi=300, bbox_inches='tight')
    print(f"✓ Plot (PDF) saved to: {plot_pdf_path}")

    print("\n" + "="*70)
    print("Three-way comparison complete (CORRECTED Q-BOUNDS)!")
    print("="*70)


if __name__ == "__main__":
    main()
