import sys
sys.path.insert(0, '/root/projects/QBound/src')

"""
Track actual Q-values during training to compare with QBound limits.
This will show if the bounds are appropriate or too restrictive.
"""

import numpy as np
import json
from datetime import datetime
from environment import GridWorldEnv
from dqn_agent import DQNAgent
import gymnasium as gym
import torch

# Set random seeds
SEED = 42
np.random.seed(SEED)
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


def track_q_values(env, agent, num_episodes, max_steps, check_interval=50):
    """
    Train agent and track Q-value statistics.
    """
    q_stats = {
        'min_q': [],
        'max_q': [],
        'mean_q': [],
        'episodes': []
    }

    for episode in range(num_episodes):
        state = env.reset()

        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            agent.train_step()
            state = next_state

            if done:
                break

        # Check Q-values periodically
        if (episode + 1) % check_interval == 0:
            # Get Q-values from the network
            with torch.no_grad():
                # Sample some states from replay buffer to check Q-values
                if len(agent.replay_buffer) >= agent.batch_size:
                    batch = agent.replay_buffer.sample(min(1000, len(agent.replay_buffer)))
                    states = torch.FloatTensor(np.array(batch['states'])).to(agent.device)
                    q_values = agent.q_network(states).cpu().numpy()

                    q_stats['episodes'].append(episode + 1)
                    q_stats['min_q'].append(float(q_values.min()))
                    q_stats['max_q'].append(float(q_values.max()))
                    q_stats['mean_q'].append(float(q_values.mean()))

    return q_stats


def run_q_tracking(env_name, env_creator, config):
    """Run experiment and track Q-values."""
    print(f"\n{'='*70}")
    print(f"{env_name} - Tracking Q-values")
    print(f"{'='*70}")
    print(f"QBound limits: [{config['qclip_min']}, {config['qclip_max']}]")

    env = env_creator()

    # Track QBound agent
    print(f"\n>>> Tracking QBound Q-values...")
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
    qbound_stats = track_q_values(env, agent_qbound, config['num_episodes'], config['max_steps'])

    # Track Baseline agent
    print(f">>> Tracking Baseline Q-values...")
    env = env_creator()
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
    baseline_stats = track_q_values(env, agent_baseline, config['num_episodes'], config['max_steps'])

    # Display results
    print(f"\n{'='*70}")
    print(f"{env_name} - Q-Value Statistics")
    print(f"{'='*70}")
    print(f"\nConfigured QBound: [{config['qclip_min']}, {config['qclip_max']}]")
    print(f"Discount factor (γ): {config['gamma']}")
    print(f"Theoretical max: {config['qclip_max'] / (1 - config['gamma']):.2f}")

    print(f"\n📊 QBound Agent:")
    if qbound_stats['max_q']:
        print(f"   Min Q observed:  {min(qbound_stats['min_q']):.4f}")
        print(f"   Max Q observed:  {max(qbound_stats['max_q']):.4f}")
        print(f"   Final min Q:     {qbound_stats['min_q'][-1]:.4f}")
        print(f"   Final max Q:     {qbound_stats['max_q'][-1]:.4f}")
        print(f"   Final mean Q:    {qbound_stats['mean_q'][-1]:.4f}")

    print(f"\n📊 Baseline Agent:")
    if baseline_stats['max_q']:
        print(f"   Min Q observed:  {min(baseline_stats['min_q']):.4f}")
        print(f"   Max Q observed:  {max(baseline_stats['max_q']):.4f}")
        print(f"   Final min Q:     {baseline_stats['min_q'][-1]:.4f}")
        print(f"   Final max Q:     {baseline_stats['max_q'][-1]:.4f}")
        print(f"   Final mean Q:    {baseline_stats['mean_q'][-1]:.4f}")

    # Check if bounds are hit
    if qbound_stats['max_q']:
        hitting_upper = max(qbound_stats['max_q']) >= config['qclip_max'] * 0.99
        hitting_lower = min(qbound_stats['min_q']) <= config['qclip_min'] * 1.01

        print(f"\n🔍 Bound Analysis:")
        if hitting_upper:
            print(f"   ⚠️  Q-values are hitting UPPER bound ({config['qclip_max']})")
            print(f"      Max observed: {max(qbound_stats['max_q']):.4f}")
        else:
            print(f"   ✓  Q-values NOT hitting upper bound")

        if hitting_lower:
            print(f"   ⚠️  Q-values are hitting LOWER bound ({config['qclip_min']})")
            print(f"      Min observed: {min(qbound_stats['min_q']):.4f}")
        else:
            print(f"   ✓  Q-values NOT hitting lower bound")

        # Compare with baseline
        baseline_range = max(baseline_stats['max_q']) - min(baseline_stats['min_q'])
        qbound_range = max(qbound_stats['max_q']) - min(qbound_stats['min_q'])
        print(f"\n   Baseline Q-value range: {baseline_range:.4f}")
        print(f"   QBound Q-value range:   {qbound_range:.4f}")
        print(f"   Restriction factor:     {(1 - qbound_range/baseline_range)*100:.1f}%")

    return {
        'env_name': env_name,
        'config': config,
        'qbound_stats': qbound_stats,
        'baseline_stats': baseline_stats
    }


def main():
    """Track Q-values for all environments."""
    print("\n" + "="*70)
    print("Q-VALUE TRACKING - Checking if QBound limits are appropriate")
    print("="*70)

    experiments = {
        'GridWorld': {
            'creator': lambda: GridWorldEnv(size=10, goal_pos=(9, 9)),
            'config': {
                'num_episodes': 500,
                'max_steps': 100,
                'qclip_max': 1.0,
                'qclip_min': 0.0,
                'learning_rate': 0.001,
                'gamma': 0.99,
                'epsilon_decay': 0.995,
            }
        },
        'FrozenLake': {
            'creator': lambda: FrozenLakeWrapper(is_slippery=True, map_name="4x4", seed=SEED),
            'config': {
                'num_episodes': 2000,
                'max_steps': 100,
                'qclip_max': 1.0,
                'qclip_min': 0.0,
                'learning_rate': 0.001,
                'gamma': 0.95,
                'epsilon_decay': 0.999,
            }
        },
        'CartPole': {
            'creator': lambda: CartPoleWrapper(seed=SEED),
            'config': {
                'num_episodes': 500,
                'max_steps': 500,
                'qclip_max': 100.0,
                'qclip_min': 0.0,
                'learning_rate': 0.001,
                'gamma': 0.99,
                'epsilon_decay': 0.995,
            }
        }
    }

    all_results = {}

    for env_name, exp_config in experiments.items():
        result = run_q_tracking(env_name, exp_config['creator'], exp_config['config'])
        all_results[env_name] = result

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f'results/q_tracking_{timestamp}.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*70}")
    print("Q-VALUE TRACKING COMPLETE")
    print(f"{'='*70}")
    print(f"Results saved to: {output_file}")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY - Are QBound limits appropriate?")
    print(f"{'='*70}")

    for env_name, result in all_results.items():
        config = result['config']
        qbound_stats = result['qbound_stats']
        baseline_stats = result['baseline_stats']

        if qbound_stats['max_q']:
            max_q_qbound = max(qbound_stats['max_q'])
            max_q_baseline = max(baseline_stats['max_q'])

            print(f"\n{env_name}:")
            print(f"  Configured bounds: [{config['qclip_min']}, {config['qclip_max']}]")
            print(f"  QBound max Q:      {max_q_qbound:.4f}")
            print(f"  Baseline max Q:    {max_q_baseline:.4f}")

            if max_q_qbound >= config['qclip_max'] * 0.99:
                print(f"  ⚠️  PROBLEM: QBound is hitting the upper limit!")
                print(f"      This may prevent proper learning.")
            elif max_q_baseline > config['qclip_max'] * 1.5:
                print(f"  ⚠️  WARNING: Baseline Q-values exceed QBound limit by >50%")
                print(f"      QBound may be too restrictive.")
            else:
                print(f"  ✓  Bounds seem reasonable for this environment")


if __name__ == "__main__":
    main()
