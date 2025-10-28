"""
PPO vs PPO+QBound on MountainCar-v0

MountainCar-v0:
- State: 2D continuous (position, velocity)
- Action: 3 discrete (left, none, right)
- Reward: -1 per step until success
- Max steps: 200
- Success: Reach goal position (0.5)
- Success threshold: Episode reward > -110 (solve in < 110 steps)

QBound Configuration:
- V_min = -200 (worst case: never solve)
- V_max = 0 (best case: immediate success)
- Static bounds (sparse terminal reward)
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


SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


def collect_trajectory(env, agent, max_steps=200):
    """Collect a single trajectory."""
    trajectory = []
    state, _ = env.reset()
    episode_reward = 0
    episode_length = 0

    for step in range(max_steps):
        action, log_prob = agent.get_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        trajectory.append((state, action, reward, next_state, done, log_prob.item()))

        episode_reward += reward
        episode_length += 1
        state = next_state

        if done:
            break

    return trajectory, episode_reward, episode_length


def train_agent(agent_type, num_episodes=1000, trajectory_length=2048):
    """Train an agent."""
    print(f"\n{'='*60}")
    print(f"Training: {agent_type}")
    print(f"{'='*60}")

    env = gym.make('MountainCar-v0')
    env.reset(seed=SEED)

    if agent_type == "Baseline PPO":
        agent = PPOAgent(
            state_dim=2,
            action_dim=3,
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
    elif agent_type == "PPO + QBound":
        agent = PPOQBoundAgent(
            state_dim=2,
            action_dim=3,
            continuous_action=False,
            V_min=-200.0,
            V_max=0.0,
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

    episode_rewards = []
    episode_lengths = []
    training_info_log = []
    success_count = []

    trajectory_buffer = []
    total_steps = 0

    for episode in range(num_episodes):
        trajectory, episode_reward, episode_length = collect_trajectory(env, agent)

        trajectory_buffer.extend(trajectory)
        total_steps += episode_length

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        success_count.append(1 if episode_reward > -110 else 0)

        if total_steps >= trajectory_length:
            training_info = agent.update(trajectory_buffer)
            training_info_log.append(training_info)
            trajectory_buffer = []
            total_steps = 0

        if (episode + 1) % 50 == 0:
            recent_rewards = episode_rewards[-50:]
            recent_success = success_count[-50:]
            avg_reward = np.mean(recent_rewards)
            std_reward = np.std(recent_rewards)
            success_rate = np.mean(recent_success) * 100

            print(f"Episode {episode + 1}/{num_episodes} | "
                  f"Avg Reward: {avg_reward:.2f} ± {std_reward:.2f} | "
                  f"Success Rate: {success_rate:.0f}%")

            if training_info_log and 'qbound_violations_bootstrap' in training_info_log[-1]:
                info = training_info_log[-1]
                print(f"  QBound: Violations={info['qbound_violations_bootstrap']:.2%}, "
                      f"Clipped={info['qbound_clipped_fraction']:.2%}")

    env.close()

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
    print("="*60)
    print("PPO on MountainCar-v0 (Discrete + Sparse Reward)")
    print("="*60)

    results = {}

    for agent_type in ["Baseline PPO", "PPO + QBound"]:
        result = train_agent(agent_type, num_episodes=1000)
        results[agent_type] = result

    print("\n" + "="*60)
    print("FINAL RESULTS (Last 100 Episodes)")
    print("="*60)

    baseline_mean = results["Baseline PPO"]['final_100_episodes']['mean']

    for agent_type in ["Baseline PPO", "PPO + QBound"]:
        stats = results[agent_type]['final_100_episodes']
        improvement = ((stats['mean'] - baseline_mean) / abs(baseline_mean) * 100) if baseline_mean != 0 else 0

        print(f"\n{agent_type}:")
        print(f"  Mean: {stats['mean']:.2f} ± {stats['std']:.2f}")
        print(f"  Success Rate: {stats['success_rate']:.1f}%")
        if agent_type != "Baseline PPO":
            print(f"  vs Baseline: {improvement:+.1f}%")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"/root/projects/QBound/results/ppo/mountaincar_{timestamp}.json"

    import os
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
