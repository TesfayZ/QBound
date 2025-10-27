"""
Training script for DQN with and without QClip.
Compares learning convergence and sample efficiency.
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from datetime import datetime

import sys
sys.path.insert(0, '/root/projects/QBound/src')

from environment import GridWorldEnv
from dqn_agent import DQNAgent


def train_agent(
    env,
    agent,
    num_episodes: int = 1000,
    max_steps: int = 100,
    eval_interval: int = 50,
    verbose: bool = True
):
    """
    Train a DQN agent and track performance metrics.

    Returns:
        dict: Training metrics including episode rewards, steps to goal, and success rate
    """
    episode_rewards = []
    episode_steps = []
    eval_success_rates = []
    eval_episodes = []

    for episode in tqdm(range(num_episodes), desc="Training"):
        state = env.reset()
        episode_reward = 0
        steps = 0

        for step in range(max_steps):
            # Select and perform action
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)

            # Store transition
            agent.store_transition(state, action, reward, next_state, done)

            # Train
            agent.train_step()

            episode_reward += reward
            state = next_state
            steps += 1

            if done:
                break

        episode_rewards.append(episode_reward)
        episode_steps.append(steps)

        # Periodic evaluation
        if (episode + 1) % eval_interval == 0:
            success_rate = evaluate_agent(env, agent, num_eval_episodes=20)
            eval_success_rates.append(success_rate)
            eval_episodes.append(episode + 1)

            if verbose:
                print(f"\nEpisode {episode + 1}/{num_episodes}")
                print(f"  Success Rate: {success_rate:.2%}")
                print(f"  Epsilon: {agent.epsilon:.3f}")
                print(f"  Recent Avg Reward: {np.mean(episode_rewards[-50:]):.3f}")

    return {
        'episode_rewards': episode_rewards,
        'episode_steps': episode_steps,
        'eval_success_rates': eval_success_rates,
        'eval_episodes': eval_episodes
    }


def evaluate_agent(env, agent, num_eval_episodes: int = 100, max_steps: int = 100):
    """
    Evaluate agent performance.

    Returns:
        float: Success rate (fraction of episodes where goal was reached)
    """
    successes = 0

    for _ in range(num_eval_episodes):
        state = env.reset()
        done = False

        for _ in range(max_steps):
            action = agent.select_action(state, eval_mode=True)
            state, reward, done, _ = env.step(action)

            if done:
                successes += 1
                break

    return successes / num_eval_episodes


def plot_comparison(results_qbound, results_baseline, save_path: str = 'comparison.png'):
    """
    Plot comparison between QBound and baseline.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Episode Rewards (smoothed)
    window = 50
    qbound_rewards_smooth = np.convolve(
        results_qbound['episode_rewards'],
        np.ones(window) / window,
        mode='valid'
    )
    baseline_rewards_smooth = np.convolve(
        results_baseline['episode_rewards'],
        np.ones(window) / window,
        mode='valid'
    )

    axes[0, 0].plot(qbound_rewards_smooth, label='QBound', linewidth=2)
    axes[0, 0].plot(baseline_rewards_smooth, label='Baseline (No QBound)', linewidth=2)
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Average Reward (smoothed)')
    axes[0, 0].set_title('Learning Curve: Episode Rewards')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Success Rate
    axes[0, 1].plot(
        results_qbound['eval_episodes'],
        results_qbound['eval_success_rates'],
        marker='o',
        label='QBound',
        linewidth=2
    )
    axes[0, 1].plot(
        results_baseline['eval_episodes'],
        results_baseline['eval_success_rates'],
        marker='s',
        label='Baseline (No QBound)',
        linewidth=2
    )
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Success Rate')
    axes[0, 1].set_title('Evaluation Success Rate')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([-0.05, 1.05])

    # Plot 3: Steps to Goal (smoothed)
    qbound_steps_smooth = np.convolve(
        results_qbound['episode_steps'],
        np.ones(window) / window,
        mode='valid'
    )
    baseline_steps_smooth = np.convolve(
        results_baseline['episode_steps'],
        np.ones(window) / window,
        mode='valid'
    )

    axes[1, 0].plot(qbound_steps_smooth, label='QBound', linewidth=2)
    axes[1, 0].plot(baseline_steps_smooth, label='Baseline (No QBound)', linewidth=2)
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Steps per Episode (smoothed)')
    axes[1, 0].set_title('Sample Efficiency: Steps to Goal')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Cumulative Success (sample efficiency)
    qbound_cumsum = np.cumsum(results_qbound['episode_rewards'])
    baseline_cumsum = np.cumsum(results_baseline['episode_rewards'])

    axes[1, 1].plot(qbound_cumsum, label='QBound', linewidth=2)
    axes[1, 1].plot(baseline_cumsum, label='Baseline (No QBound)', linewidth=2)
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Cumulative Reward')
    axes[1, 1].set_title('Sample Efficiency: Cumulative Reward')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nComparison plot saved to: {save_path}")


def main():
    """Main training and evaluation pipeline."""
    # Set random seeds for reproducibility
    np.random.seed(42)
    import torch
    torch.manual_seed(42)
    import random
    random.seed(42)

    # Environment setup
    env = GridWorldEnv(size=10, goal_pos=(9, 9))

    # Training parameters
    num_episodes = 1000
    max_steps = 100

    print("=" * 60)
    print("QBound vs Baseline DQN Comparison")
    print("=" * 60)
    print(f"Environment: {env.size}x{env.size} Grid World")
    print(f"Goal Position: {env.goal_pos}")
    print(f"Training Episodes: {num_episodes}")
    print(f"Max Steps per Episode: {max_steps}")
    print("=" * 60)

    # Train agent WITH QBound
    print("\n>>> Training agent WITH QBound...")
    agent_qbound = DQNAgent(
        state_dim=env.observation_space,
        action_dim=env.action_space,
        learning_rate=0.001,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        buffer_size=10000,
        batch_size=64,
        target_update_freq=100,
        use_qclip=True,  # Internal flag, using QBound methodology
        qclip_max=1.0,  # Maximum possible reward in this environment
        qclip_min=0.0,  # Minimum possible reward
        device="cpu"
    )

    results_qbound = train_agent(
        env,
        agent_qbound,
        num_episodes=num_episodes,
        max_steps=max_steps,
        eval_interval=50,
        verbose=True
    )

    # Train agent WITHOUT QBound (baseline)
    print("\n>>> Training agent WITHOUT QBound (Baseline)...")
    agent_baseline = DQNAgent(
        state_dim=env.observation_space,
        action_dim=env.action_space,
        learning_rate=0.001,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        buffer_size=10000,
        batch_size=64,
        target_update_freq=100,
        use_qclip=False,  # No QBound
        device="cpu"
    )

    results_baseline = train_agent(
        env,
        agent_baseline,
        num_episodes=num_episodes,
        max_steps=max_steps,
        eval_interval=50,
        verbose=True
    )

    # Final evaluation
    print("\n" + "=" * 60)
    print("Final Evaluation (100 episodes)")
    print("=" * 60)

    final_qbound_success = evaluate_agent(env, agent_qbound, num_eval_episodes=100)
    final_baseline_success = evaluate_agent(env, agent_baseline, num_eval_episodes=100)

    print(f"QBound Success Rate: {final_qbound_success:.2%}")
    print(f"Baseline Success Rate: {final_baseline_success:.2%}")
    print(f"Improvement: {(final_qbound_success - final_baseline_success) * 100:.2f} percentage points")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        'qbound': results_qbound,
        'baseline': results_baseline,
        'final_evaluation': {
            'qbound_success_rate': final_qbound_success,
            'baseline_success_rate': final_baseline_success
        },
        'config': {
            'num_episodes': num_episodes,
            'max_steps': max_steps,
            'grid_size': env.size,
            'goal_pos': env.goal_pos
        }
    }

    results_file = f'results_{timestamp}.json'
    with open(results_file, 'w') as f:
        json.dump({k: v for k, v in results.items() if k != 'qbound' and k != 'baseline'}, f, indent=2)
        # Save numpy arrays separately
        np.savez(
            f'results_{timestamp}.npz',
            qbound_rewards=results_qbound['episode_rewards'],
            qbound_steps=results_qbound['episode_steps'],
            baseline_rewards=results_baseline['episode_rewards'],
            baseline_steps=results_baseline['episode_steps']
        )

    print(f"\nResults saved to: {results_file}")

    # Plot comparison
    plot_comparison(results_qbound, results_baseline, save_path=f'comparison_{timestamp}.png')

    # Print summary statistics
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)

    # Find episode where 80% success rate was first achieved
    qbound_80_episode = next(
        (ep for ep, sr in zip(results_qbound['eval_episodes'], results_qbound['eval_success_rates']) if sr >= 0.8),
        None
    )
    baseline_80_episode = next(
        (ep for ep, sr in zip(results_baseline['eval_episodes'], results_baseline['eval_success_rates']) if sr >= 0.8),
        None
    )

    print(f"\nEpisodes to reach 80% success rate:")
    print(f"  QBound: {qbound_80_episode if qbound_80_episode else 'Not achieved'}")
    print(f"  Baseline: {baseline_80_episode if baseline_80_episode else 'Not achieved'}")

    if qbound_80_episode and baseline_80_episode:
        speedup = baseline_80_episode / qbound_80_episode
        print(f"  Speedup: {speedup:.2f}x faster with QBound")

    # Total cumulative reward (sample efficiency)
    qbound_total = sum(results_qbound['episode_rewards'])
    baseline_total = sum(results_baseline['episode_rewards'])
    print(f"\nTotal cumulative reward:")
    print(f"  QBound: {qbound_total:.1f}")
    print(f"  Baseline: {baseline_total:.1f}")
    print(f"  Improvement: {((qbound_total / baseline_total - 1) * 100):.1f}%")

    print("\n" + "=" * 60)
    print("Experiment completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
