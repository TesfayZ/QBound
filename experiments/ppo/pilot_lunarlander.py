"""
Pilot Experiment: PPO vs PPO+QBound on LunarLander-v3

LunarLander-v3:
- State: 8D continuous (x, y, vx, vy, angle, angular velocity, leg contacts)
- Action: 4 discrete (do nothing, fire left, fire main, fire right)
- Reward: Sparse shaped rewards (landing bonus, crash penalty, fuel cost)
- Max steps: 1000
- Success threshold: Reward > 200 (safe landing)

QBound Configuration:
- V_min = -100 (crash penalty)
- V_max = 200 (conservative landing bonus estimate)
- Static bounds (sparse terminal rewards)
"""

import sys
sys.path.insert(0, '/root/projects/QBound/src')

import gymnasium as gym
import numpy as np
import torch
import json
from datetime import datetime
from ppo_agent import PPOAgent
from ppo_qbound_agent import PPOQBoundAgent


# Set seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


def collect_trajectory(env, agent, max_steps=1000):
    """Collect a single trajectory using the current policy."""
    trajectory = []
    state, _ = env.reset()
    episode_reward = 0
    episode_length = 0

    for step in range(max_steps):
        # Get action from policy
        action, log_prob = agent.get_action(state)

        # Environment step
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Store transition
        trajectory.append((state, action, reward, next_state, done, log_prob.item()))

        episode_reward += reward
        episode_length += 1

        state = next_state

        if done:
            break

    return trajectory, episode_reward, episode_length


def train_agent(agent_type, num_episodes=500, trajectory_length=2048):
    """Train an agent and return training results."""
    print(f"\n{'='*60}")
    print(f"Training: {agent_type}")
    print(f"{'='*60}")

    # Create environment
    env = gym.make('LunarLander-v3')
    env.reset(seed=SEED)

    # Create agent
    if agent_type == "Baseline PPO":
        agent = PPOAgent(
            state_dim=8,
            action_dim=4,
            continuous_action=False,
            hidden_sizes=[128, 128],  # Larger network for harder task
            lr_actor=3e-4,
            lr_critic=1e-3,
            gamma=0.99,
            gae_lambda=0.95,
            clip_epsilon=0.2,
            entropy_coef=0.01,
            ppo_epochs=10,
            minibatch_size=64,
            device='cpu'
        )
    elif agent_type == "PPO + QBound":
        agent = PPOQBoundAgent(
            state_dim=8,
            action_dim=4,
            continuous_action=False,
            V_min=-100.0,
            V_max=200.0,
            use_step_aware_bounds=False,
            hidden_sizes=[128, 128],
            lr_actor=3e-4,
            lr_critic=1e-3,
            gamma=0.99,
            gae_lambda=0.95,
            clip_epsilon=0.2,
            entropy_coef=0.01,
            ppo_epochs=10,
            minibatch_size=64,
            device='cpu'
        )

    # Training loop
    episode_rewards = []
    episode_lengths = []
    training_info_log = []
    success_count = []  # Track success rate (reward > 200)

    trajectory_buffer = []
    total_steps = 0

    for episode in range(num_episodes):
        # Collect trajectory
        trajectory, episode_reward, episode_length = collect_trajectory(env, agent)

        trajectory_buffer.extend(trajectory)
        total_steps += episode_length

        # Store metrics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        success_count.append(1 if episode_reward > 200 else 0)

        # Update policy when buffer is full
        if total_steps >= trajectory_length:
            training_info = agent.update(trajectory_buffer)
            training_info_log.append(training_info)

            # Clear buffer
            trajectory_buffer = []
            total_steps = 0

        # Logging
        if (episode + 1) % 10 == 0:
            recent_rewards = episode_rewards[-10:]
            recent_success = success_count[-10:]

            avg_reward = np.mean(recent_rewards)
            std_reward = np.std(recent_rewards)
            success_rate = np.mean(recent_success) * 100

            print(f"Episode {episode + 1}/{num_episodes} | "
                  f"Avg Reward: {avg_reward:.2f} ± {std_reward:.2f} | "
                  f"Success Rate: {success_rate:.0f}% | "
                  f"Avg Length: {np.mean(episode_lengths[-10:]):.1f}")

            # Print QBound statistics if available
            if training_info_log and 'qbound_violations_bootstrap' in training_info_log[-1]:
                info = training_info_log[-1]
                print(f"  QBound Stats: "
                      f"Bootstrap Violations: {info['qbound_violations_bootstrap']:.2%} | "
                      f"Clipped Fraction: {info['qbound_clipped_fraction']:.2%}")

    env.close()

    # Compute success rate for last 100 episodes
    success_rate_100 = np.mean(success_count[-100:]) * 100

    return {
        'agent_type': agent_type,
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'success_count': success_count,
        'training_info': training_info_log,
        'final_100_episodes': {
            'mean': float(np.mean(episode_rewards[-100:])),
            'std': float(np.std(episode_rewards[-100:])),
            'max': float(np.max(episode_rewards[-100:])),
            'min': float(np.min(episode_rewards[-100:])),
            'success_rate': float(success_rate_100),
        }
    }


def main():
    """Run pilot experiment comparing both methods."""
    print("="*60)
    print("Pilot Experiment: PPO on LunarLander-v3")
    print("Comparing: Baseline PPO vs PPO+QBound")
    print("="*60)

    results = {}

    # Train both methods
    for agent_type in ["Baseline PPO", "PPO + QBound"]:
        result = train_agent(agent_type, num_episodes=500)
        results[agent_type] = result

    # Print summary
    print("\n" + "="*60)
    print("FINAL RESULTS (Last 100 Episodes)")
    print("="*60)

    baseline_mean = results["Baseline PPO"]['final_100_episodes']['mean']

    for agent_type in ["Baseline PPO", "PPO + QBound"]:
        stats = results[agent_type]['final_100_episodes']
        mean = stats['mean']
        std = stats['std']
        success_rate = stats['success_rate']

        improvement = ((mean - baseline_mean) / baseline_mean * 100) if baseline_mean != 0 else 0

        print(f"\n{agent_type}:")
        print(f"  Mean: {mean:.2f} ± {std:.2f}")
        print(f"  Max: {stats['max']:.2f}")
        print(f"  Min: {stats['min']:.2f}")
        print(f"  Success Rate: {success_rate:.1f}%")
        if agent_type != "Baseline PPO":
            print(f"  vs Baseline: {improvement:+.1f}%")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"/root/projects/QBound/results/ppo/lunarlander_pilot_{timestamp}.json"

    import os
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
