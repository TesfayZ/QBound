"""
Training script for DQN with and without QClip on CartPole-v1.
CartPole is a dense reward environment (r=+1 per timestep) for contrast with sparse rewards.
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


class CartPoleWrapper:
    """
    Wrapper for CartPole to work with our DQN agent.
    CartPole has continuous state space that we use directly.
    """

    def __init__(self):
        self.env = gym.make('CartPole-v1')
        self.observation_space = self.env.observation_space.shape[0]  # 4 continuous values
        self.action_space = self.env.action_space.n  # 2 discrete actions

    def reset(self):
        """Reset and return state."""
        state, _ = self.env.reset()
        return state

    def step(self, action):
        """Take step and return next state."""
        next_state, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return next_state, reward, done, info


def train_agent(
    env,
    agent,
    num_episodes: int = 500,
    max_steps: int = 500,
    eval_interval: int = 25,
    verbose: bool = True,
    use_step_aware: bool = False
):
    """
    Train a DQN agent and track performance metrics.

    Args:
        use_step_aware: If True, pass current_step to agent for dynamic Q-bounds

    Returns:
        dict: Training metrics including episode rewards and steps
    """
    episode_rewards = []
    episode_steps = []
    eval_avg_rewards = []
    eval_episodes = []

    for episode in tqdm(range(num_episodes), desc="Training"):
        state = env.reset()
        episode_reward = 0
        steps = 0

        for step in range(max_steps):
            # Select and perform action
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)

            # Store transition with current step for step-aware Q-bounds
            if use_step_aware:
                agent.store_transition(state, action, reward, next_state, done, current_step=step)
            else:
                agent.store_transition(state, action, reward, next_state, done, current_step=None)

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
            avg_reward = evaluate_agent(env, agent, num_eval_episodes=10)
            eval_avg_rewards.append(avg_reward)
            eval_episodes.append(episode + 1)

            if verbose:
                print(f"\nEpisode {episode + 1}/{num_episodes}")
                print(f"  Eval Avg Reward: {avg_reward:.1f}")
                print(f"  Epsilon: {agent.epsilon:.3f}")
                print(f"  Recent Avg Reward: {np.mean(episode_rewards[-25:]):.1f}")

    return {
        'episode_rewards': episode_rewards,
        'episode_steps': episode_steps,
        'eval_avg_rewards': eval_avg_rewards,
        'eval_episodes': eval_episodes
    }


def evaluate_agent(env, agent, num_eval_episodes: int = 10, max_steps: int = 500):
    """
    Evaluate agent performance.

    Returns:
        float: Average reward over evaluation episodes
    """
    total_reward = 0

    for _ in range(num_eval_episodes):
        state = env.reset()
        done = False
        episode_reward = 0

        for _ in range(max_steps):
            action = agent.select_action(state, eval_mode=True)
            state, reward, done, _ = env.step(action)
            episode_reward += reward

            if done:
                break

        total_reward += episode_reward

    return total_reward / num_eval_episodes


def plot_comparison(results_qbound, results_baseline, save_path: str = 'comparison_cartpole.png'):
    """
    Plot comparison between QBound (step-aware) and baseline.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Episode Rewards (smoothed)
    window = 25
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

    axes[0, 0].plot(qbound_rewards_smooth, label='QBound (Step-Aware)', linewidth=2)
    axes[0, 0].plot(baseline_rewards_smooth, label='Baseline', linewidth=2)
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Average Reward (smoothed)')
    axes[0, 0].set_title('Learning Curve: Episode Rewards')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Evaluation Average Rewards
    axes[0, 1].plot(
        results_qbound['eval_episodes'],
        results_qbound['eval_avg_rewards'],
        marker='o',
        label='QBound (Step-Aware)',
        linewidth=2
    )
    axes[0, 1].plot(
        results_baseline['eval_episodes'],
        results_baseline['eval_avg_rewards'],
        marker='s',
        label='Baseline',
        linewidth=2
    )
    axes[0, 1].axhline(y=200, color='gray', linestyle='--', alpha=0.5, label='Target (200)')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Average Reward')
    axes[0, 1].set_title('Evaluation Performance')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Episode Duration (smoothed)
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

    axes[1, 0].plot(qbound_steps_smooth, label='QBound (Step-Aware)', linewidth=2)
    axes[1, 0].plot(baseline_steps_smooth, label='Baseline', linewidth=2)
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Steps per Episode (smoothed)')
    axes[1, 0].set_title('Episode Duration (Higher is Better)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Cumulative Reward (sample efficiency)
    qbound_cumsum = np.cumsum(results_qbound['episode_rewards'])
    baseline_cumsum = np.cumsum(results_baseline['episode_rewards'])

    axes[1, 1].plot(qbound_cumsum, label='QBound (Step-Aware)', linewidth=2)
    axes[1, 1].plot(baseline_cumsum, label='Baseline', linewidth=2)
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
    env = CartPoleWrapper()

    # Training parameters
    num_episodes = 500
    max_steps = 500

    print("=" * 60)
    print("QBound (Step-Aware) vs Baseline DQN - CartPole-v1")
    print("=" * 60)
    print(f"Environment: CartPole-v1 (dense rewards)")
    print(f"State Space: {env.observation_space} (continuous)")
    print(f"Action Space: {env.action_space}")
    print(f"Training Episodes: {num_episodes}")
    print(f"Max Steps per Episode: {max_steps}")
    print("=" * 60)

    # Train agent WITH QBound (Step-Aware)
    print("\n>>> Training agent WITH QBound (Step-Aware Dynamic Bounds)...")
    # For CartPole with dense rewards:
    # Q_max = (max_steps - current_step) * reward_per_step
    # At step 0: Q_max = 500 * 1 = 500
    # At step 250: Q_max = 250 * 1 = 250
    # At step 499: Q_max = 1 * 1 = 1
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
        use_qclip=True,
        qclip_max=500.0,  # Static max (overridden by step-aware)
        qclip_min=0.0,
        device="cpu",
        use_step_aware_qbound=True,  # Enable step-aware dynamic Q-bounds
        max_episode_steps=500,
        step_reward=1.0
    )

    results_qbound = train_agent(
        env,
        agent_qbound,
        num_episodes=num_episodes,
        max_steps=max_steps,
        eval_interval=25,
        verbose=True,
        use_step_aware=True
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
        eval_interval=25,
        verbose=True,
        use_step_aware=False
    )

    # Final evaluation
    print("\n" + "=" * 60)
    print("Final Evaluation (50 episodes)")
    print("=" * 60)

    final_qbound_reward = evaluate_agent(env, agent_qbound, num_eval_episodes=50)
    final_baseline_reward = evaluate_agent(env, agent_baseline, num_eval_episodes=50)

    print(f"QBound (Step-Aware) Average Reward: {final_qbound_reward:.1f}")
    print(f"Baseline Average Reward: {final_baseline_reward:.1f}")
    print(f"Improvement: {final_qbound_reward - final_baseline_reward:.1f} reward points")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        'final_evaluation': {
            'qbound_avg_reward': final_qbound_reward,
            'baseline_avg_reward': final_baseline_reward
        },
        'config': {
            'num_episodes': num_episodes,
            'max_steps': max_steps,
            'environment': 'CartPole-v1'
        }
    }

    results_file = f'results_cartpole_{timestamp}.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Save numpy arrays separately
    np.savez(
        f'results_cartpole_{timestamp}.npz',
        qbound_rewards=results_qbound['episode_rewards'],
        qbound_steps=results_qbound['episode_steps'],
        qbound_eval_episodes=results_qbound['eval_episodes'],
        qbound_eval_avg_rewards=results_qbound['eval_avg_rewards'],
        baseline_rewards=results_baseline['episode_rewards'],
        baseline_steps=results_baseline['episode_steps'],
        baseline_eval_episodes=results_baseline['eval_episodes'],
        baseline_eval_avg_rewards=results_baseline['eval_avg_rewards']
    )

    print(f"\nResults saved to: {results_file}")

    # Plot comparison
    plot_comparison(results_qbound, results_baseline, save_path=f'comparison_cartpole_{timestamp}.png')

    # Print summary statistics
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)

    # Find episode where 200 average reward was first achieved (CartPole "solved" threshold)
    qbound_200_episode = next(
        (ep for ep, r in zip(results_qbound['eval_episodes'], results_qbound['eval_avg_rewards']) if r >= 200),
        None
    )
    baseline_200_episode = next(
        (ep for ep, r in zip(results_baseline['eval_episodes'], results_baseline['eval_avg_rewards']) if r >= 200),
        None
    )

    print(f"\nEpisodes to reach 200 average reward (solved):")
    print(f"  QBound (Step-Aware): {qbound_200_episode if qbound_200_episode else 'Not achieved'}")
    print(f"  Baseline: {baseline_200_episode if baseline_200_episode else 'Not achieved'}")

    if qbound_200_episode and baseline_200_episode:
        speedup = baseline_200_episode / qbound_200_episode
        improvement = ((baseline_200_episode - qbound_200_episode) / baseline_200_episode) * 100
        print(f"  Speedup: {speedup:.2f}x faster with QBound")
        print(f"  Sample Efficiency Improvement: {improvement:.1f}%")

    # Total cumulative reward (sample efficiency)
    qbound_total = sum(results_qbound['episode_rewards'])
    baseline_total = sum(results_baseline['episode_rewards'])
    print(f"\nTotal cumulative reward:")
    print(f"  QBound (Step-Aware): {qbound_total:.1f}")
    print(f"  Baseline: {baseline_total:.1f}")
    if baseline_total > 0:
        print(f"  Improvement: {((qbound_total / baseline_total - 1) * 100):.1f}%")

    print("\n" + "=" * 60)
    print("Experiment completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
