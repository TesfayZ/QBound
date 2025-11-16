"""
PPO vs PPO+QBound on Pendulum-v1 (CRITICAL TEST)

Pendulum-v1:
- State: 3D continuous (cos(θ), sin(θ), angular velocity)
- Action: 1D continuous (torque ∈ [-2, 2])
- Reward: -(θ^2 + 0.1*θ_dot^2 + 0.001*action^2), dense negative reward
- Max steps: 200
- Goal: Keep pendulum upright (θ ≈ 0)
- Success threshold: Average reward > -200

CRITICAL TEST: Unlike DDPG/TD3 which failed catastrophically (-893%),
PPO+QBound should work because it bounds V(s) not Q(s,a).

QBound Configuration:
- V_min = -3200 (worst case: max negative reward for 200 steps)
- V_max = 0 (best case: perfect upright position)
- Static bounds (or dynamic with V_max(t) = 0 always)
"""

import sys
sys.path.insert(0, '/root/projects/QBound/src')

import gymnasium as gym
import numpy as np
import torch
import json
import argparse
from datetime import datetime
from ppo_agent import PPOAgent
from ppo_qbound_agent import PPOQBoundAgent

# Parse command line arguments
parser = argparse.ArgumentParser(description='PPO on Pendulum-v1 with QBound')
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility (default: 42)')
args = parser.parse_args()

SEED = args.seed
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


def train_agent(agent_type, num_episodes=500, trajectory_length=2048):
    """Train an agent."""
    print(f"\n{'='*60}")
    print(f"Training: {agent_type}")
    print(f"{'='*60}")

    env = gym.make('Pendulum-v1')
    env.reset(seed=SEED)

    if agent_type == "Baseline PPO":
        agent = PPOAgent(
            state_dim=3,
            action_dim=1,
            continuous_action=True,
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
    elif agent_type == "PPO + Static QBound":
        agent = PPOQBoundAgent(
            state_dim=3,
            action_dim=1,
            continuous_action=True,
            V_min=-3200.0,
            V_max=0.0,
            use_step_aware_bounds=False,  # Static bounds
            use_soft_qbound=True,  # Enable soft QBound for gradient flow
            qbound_penalty_weight=0.1,
            qbound_penalty_type='quadratic',
            soft_clip_beta=0.1,
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
    elif agent_type == "PPO + Dynamic QBound":
        agent = PPOQBoundAgent(
            state_dim=3,
            action_dim=1,
            continuous_action=True,
            V_min=-3200.0,
            V_max=0.0,
            use_step_aware_bounds=True,  # Enable dynamic bounds
            max_episode_steps=200,
            step_reward=-16.27,  # Approximate step reward
            use_soft_qbound=True,  # Enable soft QBound for gradient flow
            qbound_penalty_weight=0.1,
            qbound_penalty_type='quadratic',
            soft_clip_beta=0.1,
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

    trajectory_buffer = []
    total_steps = 0

    for episode in range(num_episodes):
        trajectory, episode_reward, episode_length = collect_trajectory(env, agent)

        trajectory_buffer.extend(trajectory)
        total_steps += episode_length

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        if total_steps >= trajectory_length:
            training_info = agent.update(trajectory_buffer)
            training_info_log.append(training_info)
            trajectory_buffer = []
            total_steps = 0

        if (episode + 1) % 25 == 0:
            recent_rewards = episode_rewards[-25:]
            avg_reward = np.mean(recent_rewards)
            std_reward = np.std(recent_rewards)

            print(f"Episode {episode + 1}/{num_episodes} | "
                  f"Avg Reward: {avg_reward:.2f} ± {std_reward:.2f}")

            if training_info_log and 'qbound_violations_bootstrap' in training_info_log[-1]:
                info = training_info_log[-1]
                print(f"  QBound: Violations={info['qbound_violations_bootstrap']:.2%}, "
                      f"Clipped={info['qbound_clipped_fraction']:.2%}")

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
    print("="*60)
    print("PPO on Pendulum-v1 (Continuous + Dense Reward)")
    print("CRITICAL TEST: Does QBound work on continuous actions?")
    print("="*60)

    # Configuration for reproducibility
    config = {
        'env': 'Pendulum-v1',
        'num_episodes': 500,
        'trajectory_length': 2048,
        'max_steps': 200,
        'seed': SEED,
        'hidden_sizes': [64, 64],
        'lr_actor': 3e-4,
        'lr_critic': 1e-3,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_epsilon': 0.2,
        'entropy_coef': 0.01,
        'ppo_epochs': 10,
        'minibatch_size': 64,
        # QBound parameters
        'V_min': -3200.0,
        'V_max': 0.0,
        # Soft QBound parameters (for QBound methods)
        'soft_qbound_params': {
            'use_soft_qbound': True,
            'qbound_penalty_weight': 0.1,
            'qbound_penalty_type': 'quadratic',
            'soft_clip_beta': 0.1
        },
        # Dynamic QBound parameters (for dynamic method)
        'dynamic_qbound_params': {
            'use_step_aware_bounds': True,
            'max_episode_steps': 200,
            'step_reward': -16.27,
        }
    }

    results = {'config': config, 'training': {}}

    for agent_type in ["Baseline PPO", "PPO + Static QBound", "PPO + Dynamic QBound"]:
        result = train_agent(agent_type, num_episodes=500)
        results['training'][agent_type] = result

    print("\n" + "="*60)
    print("FINAL RESULTS (Last 100 Episodes)")
    print("="*60)

    baseline_mean = results['training']["Baseline PPO"]['final_100_episodes']['mean']

    for agent_type in ["Baseline PPO", "PPO + Static QBound", "PPO + Dynamic QBound"]:
        stats = results['training'][agent_type]['final_100_episodes']
        improvement = ((stats['mean'] - baseline_mean) / abs(baseline_mean) * 100) if baseline_mean != 0 else 0

        print(f"\n{agent_type}:")
        print(f"  Mean: {stats['mean']:.2f} ± {stats['std']:.2f}")
        print(f"  Max: {stats['max']:.2f}")
        print(f"  Min: {stats['min']:.2f}")
        if agent_type != "Baseline PPO":
            print(f"  vs Baseline: {improvement:+.1f}%")
            if improvement < -50:
                print("  ⚠️  WARNING: CATASTROPHIC FAILURE (like DDPG/TD3)")
            elif improvement > 0:
                print("  ✅ SUCCESS: QBound works on continuous actions!")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"/root/projects/QBound/results/ppo/pendulum_seed{SEED}_{timestamp}.json"

    import os
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
