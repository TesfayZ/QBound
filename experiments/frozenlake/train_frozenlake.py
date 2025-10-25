"""
Training script for DQN with and without QClip on FrozenLake-v1.
FrozenLake is a sparse reward environment where the agent navigates a slippery grid.
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from datetime import datetime
import gymnasium as gym

from dqn_agent import DQNAgent


class FrozenLakeWrapper:
    """
    Wrapper for FrozenLake to work with our DQN agent.
    Converts discrete state to one-hot encoding.
    """

    def __init__(self, is_slippery=True, map_name="4x4"):
        self.env = gym.make('FrozenLake-v1', is_slippery=is_slippery, map_name=map_name)
        self.observation_space = self.env.observation_space.n  # One-hot size
        self.action_space = self.env.action_space.n

    def reset(self):
        """Reset and return one-hot encoded state."""
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


def train_agent(
    env,
    agent,
    num_episodes: int = 2000,
    max_steps: int = 100,
    eval_interval: int = 100,
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
            success_rate = evaluate_agent(env, agent, num_eval_episodes=100)
            eval_success_rates.append(success_rate)
            eval_episodes.append(episode + 1)

            if verbose:
                print(f"\nEpisode {episode + 1}/{num_episodes}")
                print(f"  Success Rate: {success_rate:.2%}")
                print(f"  Epsilon: {agent.epsilon:.3f}")
                print(f"  Recent Avg Reward: {np.mean(episode_rewards[-100:]):.3f}")

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

            if done and reward > 0:  # Only count as success if reached goal (not fell in hole)
                successes += 1
                break

            if done:  # Fell in hole
                break

    return successes / num_eval_episodes


def plot_comparison(results_qclip, results_baseline, save_path: str = 'comparison_frozenlake.png'):
    """
    Plot comparison between QClip and baseline.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Episode Rewards (smoothed)
    window = 100
    qclip_rewards_smooth = np.convolve(
        results_qclip['episode_rewards'],
        np.ones(window) / window,
        mode='valid'
    )
    baseline_rewards_smooth = np.convolve(
        results_baseline['episode_rewards'],
        np.ones(window) / window,
        mode='valid'
    )

    axes[0, 0].plot(qclip_rewards_smooth, label='QClip', linewidth=2)
    axes[0, 0].plot(baseline_rewards_smooth, label='Baseline (No QClip)', linewidth=2)
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Average Reward (smoothed)')
    axes[0, 0].set_title('Learning Curve: Episode Rewards')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Success Rate
    axes[0, 1].plot(
        results_qclip['eval_episodes'],
        results_qclip['eval_success_rates'],
        marker='o',
        label='QClip',
        linewidth=2
    )
    axes[0, 1].plot(
        results_baseline['eval_episodes'],
        results_baseline['eval_success_rates'],
        marker='s',
        label='Baseline (No QClip)',
        linewidth=2
    )
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Success Rate')
    axes[0, 1].set_title('Evaluation Success Rate')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([-0.05, 1.05])

    # Plot 3: Steps to Goal (smoothed)
    qclip_steps_smooth = np.convolve(
        results_qclip['episode_steps'],
        np.ones(window) / window,
        mode='valid'
    )
    baseline_steps_smooth = np.convolve(
        results_baseline['episode_steps'],
        np.ones(window) / window,
        mode='valid'
    )

    axes[1, 0].plot(qclip_steps_smooth, label='QClip', linewidth=2)
    axes[1, 0].plot(baseline_steps_smooth, label='Baseline (No QClip)', linewidth=2)
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Steps per Episode (smoothed)')
    axes[1, 0].set_title('Sample Efficiency: Steps to Goal')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Cumulative Success (sample efficiency)
    qclip_cumsum = np.cumsum(results_qclip['episode_rewards'])
    baseline_cumsum = np.cumsum(results_baseline['episode_rewards'])

    axes[1, 1].plot(qclip_cumsum, label='QClip', linewidth=2)
    axes[1, 1].plot(baseline_cumsum, label='Baseline (No QClip)', linewidth=2)
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
    env = FrozenLakeWrapper(is_slippery=True, map_name="4x4")

    # Training parameters
    num_episodes = 2000  # FrozenLake is harder, needs more episodes
    max_steps = 100

    print("=" * 60)
    print("QClip vs Baseline DQN Comparison - FrozenLake-v1")
    print("=" * 60)
    print(f"Environment: FrozenLake-v1 (4x4, slippery)")
    print(f"State Space: {env.observation_space} (one-hot)")
    print(f"Action Space: {env.action_space}")
    print(f"Training Episodes: {num_episodes}")
    print(f"Max Steps per Episode: {max_steps}")
    print("=" * 60)

    # Train agent WITH QClip
    print("\n>>> Training agent WITH QClip...")
    agent_qclip = DQNAgent(
        state_dim=env.observation_space,
        action_dim=env.action_space,
        learning_rate=0.001,
        gamma=0.95,  # Lower gamma for FrozenLake
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        buffer_size=10000,
        batch_size=64,
        target_update_freq=100,
        use_qclip=True,
        qclip_max=1.0,  # Maximum possible reward
        qclip_min=0.0,  # Minimum possible reward
        device="cpu"
    )

    results_qclip = train_agent(
        env,
        agent_qclip,
        num_episodes=num_episodes,
        max_steps=max_steps,
        eval_interval=100,
        verbose=True
    )

    # Train agent WITHOUT QClip (baseline)
    print("\n>>> Training agent WITHOUT QClip (Baseline)...")
    agent_baseline = DQNAgent(
        state_dim=env.observation_space,
        action_dim=env.action_space,
        learning_rate=0.001,
        gamma=0.95,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        buffer_size=10000,
        batch_size=64,
        target_update_freq=100,
        use_qclip=False,  # No QClip
        device="cpu"
    )

    results_baseline = train_agent(
        env,
        agent_baseline,
        num_episodes=num_episodes,
        max_steps=max_steps,
        eval_interval=100,
        verbose=True
    )

    # Final evaluation
    print("\n" + "=" * 60)
    print("Final Evaluation (200 episodes)")
    print("=" * 60)

    final_qclip_success = evaluate_agent(env, agent_qclip, num_eval_episodes=200)
    final_baseline_success = evaluate_agent(env, agent_baseline, num_eval_episodes=200)

    print(f"QClip Success Rate: {final_qclip_success:.2%}")
    print(f"Baseline Success Rate: {final_baseline_success:.2%}")
    print(f"Improvement: {(final_qclip_success - final_baseline_success) * 100:.2f} percentage points")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        'final_evaluation': {
            'qclip_success_rate': final_qclip_success,
            'baseline_success_rate': final_baseline_success
        },
        'config': {
            'num_episodes': num_episodes,
            'max_steps': max_steps,
            'environment': 'FrozenLake-v1',
            'map_size': '4x4',
            'is_slippery': True
        }
    }

    results_file = f'results_frozenlake_{timestamp}.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Save numpy arrays separately
    np.savez(
        f'results_frozenlake_{timestamp}.npz',
        qclip_rewards=results_qclip['episode_rewards'],
        qclip_steps=results_qclip['episode_steps'],
        qclip_eval_episodes=results_qclip['eval_episodes'],
        qclip_eval_success_rates=results_qclip['eval_success_rates'],
        baseline_rewards=results_baseline['episode_rewards'],
        baseline_steps=results_baseline['episode_steps'],
        baseline_eval_episodes=results_baseline['eval_episodes'],
        baseline_eval_success_rates=results_baseline['eval_success_rates']
    )

    print(f"\nResults saved to: {results_file}")

    # Plot comparison
    plot_comparison(results_qclip, results_baseline, save_path=f'comparison_frozenlake_{timestamp}.png')

    # Print summary statistics
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)

    # Find episode where 50% success rate was first achieved (FrozenLake is harder)
    qclip_50_episode = next(
        (ep for ep, sr in zip(results_qclip['eval_episodes'], results_qclip['eval_success_rates']) if sr >= 0.5),
        None
    )
    baseline_50_episode = next(
        (ep for ep, sr in zip(results_baseline['eval_episodes'], results_baseline['eval_success_rates']) if sr >= 0.5),
        None
    )

    print(f"\nEpisodes to reach 50% success rate:")
    print(f"  QClip: {qclip_50_episode if qclip_50_episode else 'Not achieved'}")
    print(f"  Baseline: {baseline_50_episode if baseline_50_episode else 'Not achieved'}")

    if qclip_50_episode and baseline_50_episode:
        speedup = baseline_50_episode / qclip_50_episode
        improvement = ((baseline_50_episode - qclip_50_episode) / baseline_50_episode) * 100
        print(f"  Speedup: {speedup:.2f}x faster with QClip")
        print(f"  Sample Efficiency Improvement: {improvement:.1f}%")

    # Total cumulative reward (sample efficiency)
    qclip_total = sum(results_qclip['episode_rewards'])
    baseline_total = sum(results_baseline['episode_rewards'])
    print(f"\nTotal cumulative reward:")
    print(f"  QClip: {qclip_total:.1f}")
    print(f"  Baseline: {baseline_total:.1f}")
    if baseline_total > 0:
        print(f"  Improvement: {((qclip_total / baseline_total - 1) * 100):.1f}%")

    print("\n" + "=" * 60)
    print("Experiment completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
