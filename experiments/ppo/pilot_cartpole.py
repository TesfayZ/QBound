"""
Pilot Experiment: PPO vs PPO+QBound on CartPole-v1

This is a quick validation experiment to test both implementations.

CartPole-v1:
- State: 4D continuous (position, velocity, angle, angular velocity)
- Action: 2 discrete (left, right)
- Reward: +1 per timestep (dense reward)
- Max steps: 500
- Success threshold: Average reward > 475

QBound Configuration:
- V_min = 0 (minimum possible return)
- V_max = 100 (static) or dynamic V_max(t) = 500 - t
- Test both static and dynamic bounds
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


def collect_trajectory(env, agent, max_steps=500):
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

        # Store transition (including step for dynamic bounds)
        trajectory.append((state, action, reward, next_state, done, log_prob.item(), step))

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
    env = gym.make('CartPole-v1')
    env.reset(seed=SEED)

    # Create agent
    if agent_type == "Baseline PPO":
        agent = PPOAgent(
            state_dim=4,
            action_dim=2,
            continuous_action=False,
            hidden_sizes=[64, 64],
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
    elif agent_type == "PPO + QBound (Static)":
        agent = PPOQBoundAgent(
            state_dim=4,
            action_dim=2,
            continuous_action=False,
            V_min=0.0,
            V_max=100.0,
            use_step_aware_bounds=False,
            hidden_sizes=[64, 64],
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
    elif agent_type == "PPO + QBound (Dynamic)":
        agent = PPOQBoundAgent(
            state_dim=4,
            action_dim=2,
            continuous_action=False,
            V_min=0.0,
            V_max=500.0,  # Will be adjusted dynamically
            use_step_aware_bounds=True,
            max_episode_steps=500,
            step_reward=1.0,
            hidden_sizes=[64, 64],
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
            avg_reward = np.mean(recent_rewards)
            std_reward = np.std(recent_rewards)

            print(f"Episode {episode + 1}/{num_episodes} | "
                  f"Avg Reward: {avg_reward:.2f} ± {std_reward:.2f} | "
                  f"Avg Length: {np.mean(episode_lengths[-10:]):.1f}")

            # Print QBound statistics if available
            if training_info_log and 'qbound_violations_bootstrap' in training_info_log[-1]:
                info = training_info_log[-1]
                print(f"  QBound Stats: "
                      f"Bootstrap Violations: {info['qbound_violations_bootstrap']:.2%} | "
                      f"Clipped Fraction: {info['qbound_clipped_fraction']:.2%}")

    env.close()

    return {
        'agent_type': agent_type,
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'training_info': training_info_log,
        'final_100_episodes': {
            'mean': float(np.mean(episode_rewards[-100:])),
            'std': float(np.std(episode_rewards[-100:])),
            'max': float(np.max(episode_rewards[-100:])),
            'min': float(np.min(episode_rewards[-100:])),
        }
    }


def main():
    """Run pilot experiment comparing all three methods."""
    print("="*60)
    print("Pilot Experiment: PPO on CartPole-v1")
    print("Comparing: Baseline PPO, PPO+QBound (Static), PPO+QBound (Dynamic)")
    print("="*60)

    results = {}

    # Train all three methods
    for agent_type in ["Baseline PPO", "PPO + QBound (Static)", "PPO + QBound (Dynamic)"]:
        result = train_agent(agent_type, num_episodes=300)
        results[agent_type] = result

    # Print summary
    print("\n" + "="*60)
    print("FINAL RESULTS (Last 100 Episodes)")
    print("="*60)

    baseline_mean = results["Baseline PPO"]['final_100_episodes']['mean']

    for agent_type in ["Baseline PPO", "PPO + QBound (Static)", "PPO + QBound (Dynamic)"]:
        stats = results[agent_type]['final_100_episodes']
        mean = stats['mean']
        std = stats['std']

        improvement = ((mean - baseline_mean) / baseline_mean * 100) if baseline_mean != 0 else 0

        print(f"\n{agent_type}:")
        print(f"  Mean: {mean:.2f} ± {std:.2f}")
        print(f"  Max: {stats['max']:.2f}")
        print(f"  Min: {stats['min']:.2f}")
        if agent_type != "Baseline PPO":
            print(f"  vs Baseline: {improvement:+.1f}%")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"/root/projects/QBound/results/ppo/cartpole_pilot_{timestamp}.json"

    import os
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
