"""
Run all experiments to generate real results for the paper.
Optimized for speed while maintaining scientific validity.
"""

import sys
sys.path.insert(0, '/root/projects/QBound/src')

import numpy as np
import json
from datetime import datetime
from environment import GridWorldEnv
from dqn_agent import DQNAgent
import gymnasium as gym


# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
import torch
torch.manual_seed(SEED)
import random
random.seed(SEED)


class FrozenLakeWrapper:
    """Wrapper for FrozenLake to work with our DQN agent."""
    def __init__(self, is_slippery=True, map_name="4x4", seed=None):
        self.env = gym.make('FrozenLake-v1', is_slippery=is_slippery, map_name=map_name)
        self.observation_space = self.env.observation_space.n
        self.action_space = self.env.action_space.n
        self.seed = seed

    def reset(self):
        if self.seed is not None:
            state, _ = self.env.reset(seed=self.seed)
        else:
            state, _ = self.env.reset()
        onehot = np.zeros(self.observation_space)
        onehot[state] = 1.0
        return onehot

    def step(self, action):
        next_state, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        onehot = np.zeros(self.observation_space)
        onehot[next_state] = 1.0
        return onehot, reward, done, info


class CartPoleWrapper:
    """Wrapper for CartPole to work with our DQN agent."""
    def __init__(self, seed=None):
        self.env = gym.make('CartPole-v1')
        self.observation_space = self.env.observation_space.shape[0]
        self.action_space = self.env.action_space.n
        self.seed = seed

    def reset(self):
        if self.seed is not None:
            state, _ = self.env.reset(seed=self.seed)
        else:
            state, _ = self.env.reset()
        return state

    def step(self, action):
        next_state, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return next_state, reward, done, info


def train_agent_efficient(env, agent, num_episodes, max_steps, target_success=0.8, window=50):
    """
    Train agent efficiently, tracking when it reaches target performance.
    Returns episode rewards and episode when target was reached.
    """
    episode_rewards = []
    episodes_to_target = None

    for episode in range(num_episodes):
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

        # Check if target reached
        if episodes_to_target is None and episode >= window:
            recent_avg = np.mean(episode_rewards[-window:])
            if recent_avg >= target_success:
                episodes_to_target = episode
                print(f"  ✓ Reached {target_success*100:.0f}% success at episode {episode}")

        # Print progress
        if (episode + 1) % 100 == 0:
            recent_avg = np.mean(episode_rewards[-min(window, len(episode_rewards)):])
            print(f"  Episode {episode+1}/{num_episodes} - Recent avg: {recent_avg:.3f}, ε={agent.epsilon:.3f}")

    return episode_rewards, episodes_to_target


def run_experiment(env_name, env_creator, config):
    """Run a single experiment comparing QBound vs Baseline."""
    print(f"\n{'='*70}")
    print(f"{env_name} Experiment")
    print(f"{'='*70}")

    num_episodes = config['num_episodes']
    max_steps = config['max_steps']
    target_success = config['target_success']

    # Create environment
    env = env_creator()

    # Train with QBound
    print(f"\n>>> Training WITH QBound...")
    agent_qbound = DQNAgent(
        state_dim=env.observation_space,
        action_dim=env.action_space,
        learning_rate=config['learning_rate'],
        gamma=config['gamma'],
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=config['epsilon_decay'],
        buffer_size=10000,
        batch_size=64,
        target_update_freq=100,
        use_qclip=True,
        qclip_max=config['qclip_max'],
        qclip_min=config['qclip_min'],
        aux_weight=0.5,
        device="cpu"
    )
    rewards_qbound, episodes_qbound = train_agent_efficient(
        env, agent_qbound, num_episodes, max_steps, target_success
    )

    # Train baseline
    print(f"\n>>> Training WITHOUT QBound (Baseline)...")
    env = env_creator()  # Fresh environment
    agent_baseline = DQNAgent(
        state_dim=env.observation_space,
        action_dim=env.action_space,
        learning_rate=config['learning_rate'],
        gamma=config['gamma'],
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=config['epsilon_decay'],
        buffer_size=10000,
        batch_size=64,
        target_update_freq=100,
        use_qclip=False,
        device="cpu"
    )
    rewards_baseline, episodes_baseline = train_agent_efficient(
        env, agent_baseline, num_episodes, max_steps, target_success
    )

    # Calculate results
    print(f"\n{'='*70}")
    print(f"{env_name} Results")
    print(f"{'='*70}")

    qbound_text = str(episodes_qbound) if episodes_qbound else "Not achieved"
    baseline_text = str(episodes_baseline) if episodes_baseline else "Not achieved"

    print(f"\nEpisodes to {target_success*100:.0f}% success rate:")
    print(f"  QBound:   {qbound_text}")
    print(f"  Baseline: {baseline_text}")

    improvement = None
    if episodes_qbound and episodes_baseline:
        improvement = ((episodes_baseline - episodes_qbound) / episodes_baseline) * 100
        speedup = episodes_baseline / episodes_qbound
        print(f"  Improvement: {improvement:.1f}%")
        print(f"  Speedup: {speedup:.2f}x")

    # Total reward
    total_qbound = sum(rewards_qbound)
    total_baseline = sum(rewards_baseline)
    reward_improvement = ((total_qbound - total_baseline) / total_baseline) * 100

    print(f"\nTotal cumulative reward:")
    print(f"  QBound:   {total_qbound:.1f}")
    print(f"  Baseline: {total_baseline:.1f}")
    print(f"  Improvement: {reward_improvement:+.1f}%")

    return {
        'env_name': env_name,
        'qbound_episodes': episodes_qbound,
        'baseline_episodes': episodes_baseline,
        'improvement_percent': improvement,
        'qbound_total_reward': float(total_qbound),
        'baseline_total_reward': float(total_baseline),
        'reward_improvement_percent': float(reward_improvement),
        'rewards_qbound': [float(r) for r in rewards_qbound],
        'rewards_baseline': [float(r) for r in rewards_baseline],
        'config': config
    }


def main():
    """Run all experiments."""
    print("\n" + "="*70)
    print("QBound Experiments - Generating Real Results for Paper")
    print("="*70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    experiments = {
        'GridWorld': {
            'creator': lambda: GridWorldEnv(size=10, goal_pos=(9, 9)),
            'config': {
                'num_episodes': 500,
                'max_steps': 100,
                'target_success': 0.8,
                'qclip_max': 1.0,
                'qclip_min': 0.0,
                'learning_rate': 0.001,
                'gamma': 0.99,
                'epsilon_decay': 0.995,
                'seed': SEED
            }
        },
        'FrozenLake': {
            'creator': lambda: FrozenLakeWrapper(is_slippery=True, map_name="4x4", seed=SEED),
            'config': {
                'num_episodes': 2000,
                'max_steps': 100,
                'target_success': 0.7,  # Lower target due to stochasticity
                'qclip_max': 1.0,
                'qclip_min': 0.0,
                'learning_rate': 0.001,
                'gamma': 0.95,
                'epsilon_decay': 0.999,
                'seed': SEED
            }
        },
        'CartPole': {
            'creator': lambda: CartPoleWrapper(seed=SEED),
            'config': {
                'num_episodes': 500,
                'max_steps': 500,
                'target_success': 475.0,  # Near max score
                'qclip_max': 100.0,  # Based on max episode length
                'qclip_min': 0.0,
                'learning_rate': 0.001,
                'gamma': 0.99,
                'epsilon_decay': 0.995,
                'seed': SEED
            }
        }
    }

    all_results = {}

    # Run experiments
    for env_name, exp_config in experiments.items():
        result = run_experiment(env_name, exp_config['creator'], exp_config['config'])
        all_results[env_name] = result

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save combined results
    combined_output = f'results/combined/experiment_results_{timestamp}.json'
    with open(combined_output, 'w') as f:
        json.dump(all_results, f, indent=2)

    # Save individual environment results
    for env_name, result in all_results.items():
        env_dir = f'results/{env_name.lower()}'
        env_output = f'{env_dir}/results_{timestamp}.json'
        with open(env_output, 'w') as f:
            json.dump(result, f, indent=2)

    print(f"\n{'='*70}")
    print("All Experiments Complete!")
    print(f"{'='*70}")
    print(f"Results saved to:")
    print(f"  Combined: {combined_output}")
    for env_name in all_results.keys():
        print(f"  {env_name}: results/{env_name.lower()}/results_{timestamp}.json")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Summary table
    print(f"\n{'='*70}")
    print("Summary Table (for paper)")
    print(f"{'='*70}")
    print(f"{'Environment':<15} {'Baseline':>10} {'QBound':>10} {'Improvement':>12}")
    print("-" * 70)
    for env_name, result in all_results.items():
        baseline = result['baseline_episodes'] if result['baseline_episodes'] else "N/A"
        qbound = result['qbound_episodes'] if result['qbound_episodes'] else "N/A"
        improvement = f"{result['improvement_percent']:.1f}%" if result['improvement_percent'] else "N/A"
        print(f"{env_name:<15} {str(baseline):>10} {str(qbound):>10} {improvement:>12}")


if __name__ == "__main__":
    main()
