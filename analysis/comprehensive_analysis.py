import sys
sys.path.insert(0, '/root/projects/QBound/src')

"""
Comprehensive analysis for QBound paper:
- Track Q-values (min/max) over time
- Track QBound violation rates
- Generate publication-quality plots
"""

import numpy as np
import json
from datetime import datetime
from environment import GridWorldEnv
from dqn_agent import DQNAgent
import gymnasium as gym
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # For non-interactive plotting

# Set random seeds
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
import random
random.seed(SEED)


class FrozenLakeWrapper:
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


def comprehensive_tracking(env, agent, num_episodes, max_steps, check_interval=50,
                          qclip_min=0.0, qclip_max=1.0, use_qclip=False):
    """
    Train agent and comprehensively track:
    - Q-value statistics (min, max, mean)
    - QBound violation rates
    - Episode rewards
    - Success rates
    """
    stats = {
        'episodes': [],
        'min_q': [],
        'max_q': [],
        'mean_q': [],
        'std_q': [],
        'violation_rate_upper': [],
        'violation_rate_lower': [],
        'episode_rewards': [],
        'success_rate': [],
    }

    episode_rewards = []

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

        # Track statistics periodically
        if (episode + 1) % check_interval == 0:
            with torch.no_grad():
                # Sample states from replay buffer
                if len(agent.replay_buffer) >= agent.batch_size:
                    batch = agent.replay_buffer.sample(min(1000, len(agent.replay_buffer)))

                    # Extract states from tuple list
                    states = np.array([transition[0] for transition in batch])
                    states_tensor = torch.FloatTensor(states).to(agent.device)

                    # Get Q-values
                    q_values = agent.q_network(states_tensor).cpu().numpy()

                    stats['episodes'].append(episode + 1)
                    stats['min_q'].append(float(q_values.min()))
                    stats['max_q'].append(float(q_values.max()))
                    stats['mean_q'].append(float(q_values.mean()))
                    stats['std_q'].append(float(q_values.std()))

                    # Calculate violation rates (if using QBound)
                    if use_qclip:
                        violations_upper = (q_values > qclip_max).sum()
                        violations_lower = (q_values < qclip_min).sum()
                        total_q_values = q_values.size

                        violation_rate_upper = violations_upper / total_q_values
                        violation_rate_lower = violations_lower / total_q_values
                    else:
                        # For baseline, check how many would violate if bounds were applied
                        violations_upper = (q_values > qclip_max).sum()
                        violations_lower = (q_values < qclip_min).sum()
                        total_q_values = q_values.size

                        violation_rate_upper = violations_upper / total_q_values
                        violation_rate_lower = violations_lower / total_q_values

                    stats['violation_rate_upper'].append(float(violation_rate_upper))
                    stats['violation_rate_lower'].append(float(violation_rate_lower))

                    # Success rate (last 100 episodes)
                    recent_rewards = episode_rewards[-min(100, len(episode_rewards)):]
                    success_rate = np.mean([r > 0 for r in recent_rewards])
                    stats['success_rate'].append(float(success_rate))

    stats['episode_rewards'] = [float(r) for r in episode_rewards]
    return stats


def run_comprehensive_experiment(env_name, env_creator, config):
    """Run comprehensive analysis for one environment."""
    print(f"\n{'='*70}")
    print(f"{env_name} - Comprehensive Analysis")
    print(f"{'='*70}")

    num_episodes = config['num_episodes']
    max_steps = config['max_steps']

    # Train with QBound
    print(f"\n>>> Training WITH QBound and tracking metrics...")
    env = env_creator()
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
    qbound_stats = comprehensive_tracking(
        env, agent_qbound, num_episodes, max_steps,
        qclip_min=config['qclip_min'],
        qclip_max=config['qclip_max'],
        use_qclip=True
    )

    # Train baseline
    print(f">>> Training WITHOUT QBound (Baseline) and tracking metrics...")
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
    baseline_stats = comprehensive_tracking(
        env, agent_baseline, num_episodes, max_steps,
        qclip_min=config['qclip_min'],
        qclip_max=config['qclip_max'],
        use_qclip=False
    )

    # Display summary
    print(f"\n{'='*70}")
    print(f"{env_name} - Summary")
    print(f"{'='*70}")
    print(f"\nQBound Configuration: [{config['qclip_min']}, {config['qclip_max']}]")
    print(f"Discount factor (γ): {config['gamma']}")

    print(f"\n📊 Q-Value Range (Final):")
    if qbound_stats['max_q']:
        print(f"   QBound:   [{qbound_stats['min_q'][-1]:.4f}, {qbound_stats['max_q'][-1]:.4f}]")
        print(f"   Baseline: [{baseline_stats['min_q'][-1]:.4f}, {baseline_stats['max_q'][-1]:.4f}]")

    print(f"\n⚠️  QBound Violation Rate (Final):")
    if qbound_stats['violation_rate_upper']:
        print(f"   QBound Upper:   {qbound_stats['violation_rate_upper'][-1]*100:.2f}%")
        print(f"   QBound Lower:   {qbound_stats['violation_rate_lower'][-1]*100:.2f}%")
        print(f"   Baseline Upper: {baseline_stats['violation_rate_upper'][-1]*100:.2f}%")
        print(f"   Baseline Lower: {baseline_stats['violation_rate_lower'][-1]*100:.2f}%")

    return {
        'env_name': env_name,
        'config': config,
        'qbound_stats': qbound_stats,
        'baseline_stats': baseline_stats
    }


def create_publication_plots(all_results, output_dir='results/plots'):
    """Create publication-quality plots."""
    import os
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for env_name, result in all_results.items():
        qbound = result['qbound_stats']
        baseline = result['baseline_stats']
        config = result['config']

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'{env_name} - QBound vs Baseline Analysis', fontsize=16, fontweight='bold')

        # Plot 1: Q-value bounds over time
        ax1 = axes[0, 0]
        if qbound['episodes']:
            ax1.plot(qbound['episodes'], qbound['max_q'], 'b-', label='QBound Max Q', linewidth=2)
            ax1.plot(qbound['episodes'], qbound['min_q'], 'b--', label='QBound Min Q', linewidth=2)
            ax1.plot(baseline['episodes'], baseline['max_q'], 'r-', label='Baseline Max Q', linewidth=2, alpha=0.7)
            ax1.plot(baseline['episodes'], baseline['min_q'], 'r--', label='Baseline Min Q', linewidth=2, alpha=0.7)
            ax1.axhline(y=config['qclip_max'], color='g', linestyle=':', label=f'QBound Limit ({config["qclip_max"]})', linewidth=2)
            ax1.axhline(y=config['qclip_min'], color='g', linestyle=':', linewidth=2)
            ax1.set_xlabel('Episodes', fontsize=12)
            ax1.set_ylabel('Q-value', fontsize=12)
            ax1.set_title('Q-value Bounds Over Time', fontsize=14, fontweight='bold')
            ax1.legend(fontsize=10)
            ax1.grid(True, alpha=0.3)

        # Plot 2: Violation rates over time
        ax2 = axes[0, 1]
        if qbound['violation_rate_upper']:
            violation_upper_pct = [v * 100 for v in qbound['violation_rate_upper']]
            violation_lower_pct = [v * 100 for v in qbound['violation_rate_lower']]
            baseline_upper_pct = [v * 100 for v in baseline['violation_rate_upper']]

            ax2.plot(qbound['episodes'], violation_upper_pct, 'b-', label='QBound Upper Violations', linewidth=2)
            ax2.plot(qbound['episodes'], violation_lower_pct, 'b--', label='QBound Lower Violations', linewidth=2)
            ax2.plot(baseline['episodes'], baseline_upper_pct, 'r-', label='Baseline (would violate)', linewidth=2, alpha=0.7)
            ax2.set_xlabel('Episodes', fontsize=12)
            ax2.set_ylabel('Violation Rate (%)', fontsize=12)
            ax2.set_title('QBound Violation Rate Decrease Over Time', fontsize=14, fontweight='bold')
            ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.3)

        # Plot 3: Learning curves (smoothed rewards)
        ax3 = axes[1, 0]
        window = 50
        qbound_rewards = np.array(qbound['episode_rewards'])
        baseline_rewards = np.array(baseline['episode_rewards'])

        if len(qbound_rewards) >= window:
            qbound_smooth = np.convolve(qbound_rewards, np.ones(window)/window, mode='valid')
            baseline_smooth = np.convolve(baseline_rewards, np.ones(window)/window, mode='valid')
            episodes_smooth = np.arange(window-1, len(qbound_rewards))

            ax3.plot(episodes_smooth, qbound_smooth, 'b-', label='QBound', linewidth=2)
            ax3.plot(episodes_smooth, baseline_smooth, 'r-', label='Baseline', linewidth=2, alpha=0.7)
            ax3.set_xlabel('Episodes', fontsize=12)
            ax3.set_ylabel(f'Average Reward (window={window})', fontsize=12)
            ax3.set_title('Learning Curves', fontsize=14, fontweight='bold')
            ax3.legend(fontsize=10)
            ax3.grid(True, alpha=0.3)

        # Plot 4: Mean Q-value over time
        ax4 = axes[1, 1]
        if qbound['mean_q']:
            ax4.plot(qbound['episodes'], qbound['mean_q'], 'b-', label='QBound Mean Q', linewidth=2)
            ax4.plot(baseline['episodes'], baseline['mean_q'], 'r-', label='Baseline Mean Q', linewidth=2, alpha=0.7)
            ax4.fill_between(qbound['episodes'],
                            [m - s for m, s in zip(qbound['mean_q'], qbound['std_q'])],
                            [m + s for m, s in zip(qbound['mean_q'], qbound['std_q'])],
                            alpha=0.2, color='blue', label='QBound ±1 std')
            ax4.fill_between(baseline['episodes'],
                            [m - s for m, s in zip(baseline['mean_q'], baseline['std_q'])],
                            [m + s for m, s in zip(baseline['mean_q'], baseline['std_q'])],
                            alpha=0.2, color='red', label='Baseline ±1 std')
            ax4.set_xlabel('Episodes', fontsize=12)
            ax4.set_ylabel('Mean Q-value', fontsize=12)
            ax4.set_title('Mean Q-value with Standard Deviation', fontsize=14, fontweight='bold')
            ax4.legend(fontsize=10)
            ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        plot_file = f'{output_dir}/{env_name.lower()}_analysis_{timestamp}.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"   Saved plot: {plot_file}")
        plt.close()

    print(f"\n✓ All plots saved to {output_dir}/")


def main():
    """Run comprehensive analysis for all environments."""
    print("\n" + "="*70)
    print("COMPREHENSIVE QBOUND ANALYSIS FOR PAPER")
    print("="*70)
    print("Tracking: Q-values, Violations, Learning Curves, Performance")

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
        result = run_comprehensive_experiment(env_name, exp_config['creator'], exp_config['config'])
        all_results[env_name] = result

    # Save numerical results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f'results/comprehensive_analysis_{timestamp}.json'

    # Convert to JSON-serializable format
    json_results = {}
    for env_name, result in all_results.items():
        json_results[env_name] = {
            'env_name': result['env_name'],
            'config': result['config'],
            'qbound_stats': result['qbound_stats'],
            'baseline_stats': result['baseline_stats']
        }

    with open(output_file, 'w') as f:
        json.dump(json_results, f, indent=2)

    print(f"\n{'='*70}")
    print("Numerical results saved")
    print(f"{'='*70}")
    print(f"Results file: {output_file}")

    # Generate plots
    print(f"\n{'='*70}")
    print("Generating publication-quality plots...")
    print(f"{'='*70}")
    create_publication_plots(all_results)

    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
