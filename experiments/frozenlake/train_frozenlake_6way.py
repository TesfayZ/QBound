"""
FrozenLake-v1: 6-Way Comprehensive Comparison
Tests QBound on Sparse Terminal Reward Environment (Stochastic)

Environment: FrozenLake-v1 (4x4 grid, slippery)
Reward: +1 for reaching goal, 0 otherwise (sparse terminal reward)
Max steps: 100

Comparison:
1. Baseline DQN - No QBound, no Double-Q
2. Static QBound + DQN - Q ∈ [0, 1.0]
3. Dynamic QBound + DQN - Q ∈ [0, Q_max(t)] with step-aware bounds
4. Baseline DDQN - No QBound, with Double-Q
5. Static QBound + DDQN - Q ∈ [0, 1.0] + Double-Q
6. Dynamic QBound + DDQN - Q ∈ [0, Q_max(t)] + Double-Q

Note: FrozenLake is stochastic (slippery) with sparse terminal rewards.
"""

import sys
sys.path.insert(0, '/root/projects/QBound/src')

import numpy as np
import torch
import json
import random
import os
import gymnasium as gym
from datetime import datetime
from dqn_agent import DQNAgent
from double_dqn_agent import DoubleDQNAgent
from tqdm import tqdm


# Reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

# Results file for crash recovery
RESULTS_FILE = "/root/projects/QBound/results/frozenlake/6way_comparison_in_progress.json"

# Environment parameters
ENV_NAME = "FrozenLake-v1"
MAX_EPISODES = 2000  # FrozenLake needs more episodes due to stochasticity
MAX_STEPS = 100
EVAL_EPISODES = 10

# Shared hyperparameters
LR = 0.001
GAMMA = 0.95  # FrozenLake uses γ=0.95
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
BATCH_SIZE = 64
BUFFER_SIZE = 10000
TARGET_UPDATE_FREQ = 100

# QBound parameters for FrozenLake (Sparse Terminal Reward)
# Reward: +1 for reaching goal (terminal state)
# Since reward is terminal, max Q-value is 1.0 (discounting doesn't accumulate)
QBOUND_MIN = 0.0
QBOUND_MAX = 1.0


class FrozenLakeWrapper:
    """Wrapper for FrozenLake to work with DQN agent (one-hot encoding)."""

    def __init__(self, is_slippery=True, map_name="4x4", seed=None):
        self.env = gym.make('FrozenLake-v1', is_slippery=is_slippery, map_name=map_name)
        self.observation_space = self.env.observation_space.n  # One-hot size
        self.action_space = self.env.action_space.n
        self.seed = seed

    def reset(self):
        """Reset and return one-hot encoded state."""
        if self.seed is not None:
            state, _ = self.env.reset(seed=self.seed)
            self.seed += 1  # Increment for next reset
        else:
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


def load_existing_results():
    """Load existing results if the experiment was interrupted"""
    if os.path.exists(RESULTS_FILE):
        print(f"\n🔄 Found existing results file: {RESULTS_FILE}")
        print("   Loading previous progress...")
        with open(RESULTS_FILE, 'r') as f:
            results = json.load(f)

        completed = [k for k in results.get('training', {}).keys()]
        if completed:
            print(f"   ✓ Already completed: {', '.join(completed)}")
        return results
    return None


def save_intermediate_results(results):
    """Save results after each method completes (crash recovery)"""
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"   💾 Progress saved to: {RESULTS_FILE}")


def is_method_completed(results, method_name):
    """Check if a method has already been completed"""
    return method_name in results.get('training', {})


def train_agent(env, agent, agent_name, max_episodes=MAX_EPISODES, use_step_aware=False):
    """Train agent and return results"""
    print(f"\n>>> Training {agent_name}...")

    episode_rewards = []
    episode_steps = []
    success_count = 0

    for episode in tqdm(range(max_episodes), desc=agent_name):
        state = env.reset()

        episode_reward = 0
        done = False
        step = 0

        while not done and step < MAX_STEPS:
            # Select action
            action = agent.select_action(state, eval_mode=False)

            # Take step
            next_state, reward, done, _ = env.step(action)

            # Store transition (with step info if using step-aware bounds)
            if use_step_aware:
                agent.store_transition(state, action, reward, next_state, done, current_step=step)
            else:
                agent.store_transition(state, action, reward, next_state, done)

            # Train
            loss = agent.train_step()

            episode_reward += reward
            state = next_state
            step += 1

        if reward > 0:  # Success (reached goal)
            success_count += 1

        episode_rewards.append(episode_reward)
        episode_steps.append(step)

        # Progress update
        if (episode + 1) % 400 == 0:
            recent_avg_reward = np.mean(episode_rewards[-400:])
            recent_success_rate = sum(1 for r in episode_rewards[-400:] if r > 0) / 400
            print(f"  Episode {episode + 1}/{max_episodes} - Avg reward: {recent_avg_reward:.3f}, "
                  f"Success rate: {recent_success_rate:.2%}, ε={agent.epsilon:.3f}")

    success_rate = success_count / max_episodes
    return episode_rewards, episode_steps, success_rate


def main():
    print("=" * 80)
    print("FrozenLake-v1: 6-Way Comprehensive Comparison")
    print("Testing QBound on Sparse Terminal Reward (Stochastic)")
    print("=" * 80)

    print(f"\nConfiguration:")
    print(f"  Environment: {ENV_NAME} (4x4, slippery)")
    print(f"  Episodes: {MAX_EPISODES}")
    print(f"  Max steps per episode: {MAX_STEPS}")
    print(f"  Discount factor γ: {GAMMA}")
    print(f"  Learning rate: {LR}")
    print(f"  QBound range: [{QBOUND_MIN:.2f}, {QBOUND_MAX:.2f}]")
    print("=" * 80)

    # Load existing results or create new
    results = load_existing_results()
    if results is None:
        print("\n🆕 Starting fresh experiment...")
        results = {
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'config': {
                'env': ENV_NAME,
                'episodes': MAX_EPISODES,
                'max_steps': MAX_STEPS,
                'gamma': GAMMA,
                'lr': LR,
                'qbound_min': QBOUND_MIN,
                'qbound_max': QBOUND_MAX,
                'seed': SEED
            },
            'training': {}
        }
    else:
        print("   ⏩ Resuming experiment...\n")

    # Create environment
    env = FrozenLakeWrapper(seed=SEED)
    state_dim = env.observation_space
    action_dim = env.action_space

    # ===== 1. Baseline DQN =====
    print("\n" + "=" * 80)
    print("METHOD 1: Baseline DQN")
    print("=" * 80)

    if is_method_completed(results, 'baseline'):
        print("⏭️  Already completed, skipping...")
    else:
        baseline_agent = DQNAgent(
            state_dim=state_dim, action_dim=action_dim, learning_rate=LR,
            gamma=GAMMA, epsilon_start=EPSILON_START, epsilon_end=EPSILON_END,
            epsilon_decay=EPSILON_DECAY, buffer_size=BUFFER_SIZE,
            batch_size=BATCH_SIZE, target_update_freq=TARGET_UPDATE_FREQ,
            use_qclip=False, device='cpu'
        )
        env = FrozenLakeWrapper(seed=SEED)
        baseline_rewards, baseline_steps, baseline_success = train_agent(
            env, baseline_agent, "1. Baseline DQN", use_step_aware=False)
        results['training']['baseline'] = {
            'rewards': baseline_rewards, 'steps': baseline_steps,
            'total_reward': float(np.sum(baseline_rewards)),
            'mean_reward': float(np.mean(baseline_rewards)),
            'mean_steps': float(np.mean(baseline_steps)),
            'success_rate': float(baseline_success),
            'final_100_mean': float(np.mean(baseline_rewards[-100:])),
            'final_100_std': float(np.std(baseline_rewards[-100:]))
        }
        save_intermediate_results(results)

    # ===== 2. Static QBound + DQN =====
    print("\n" + "=" * 80)
    print("METHOD 2: Static QBound + DQN")
    print("=" * 80)

    if is_method_completed(results, 'static_qbound'):
        print("⏭️  Already completed, skipping...")
    else:
        static_agent = DQNAgent(
            state_dim=state_dim, action_dim=action_dim, learning_rate=LR,
            gamma=GAMMA, epsilon_start=EPSILON_START, epsilon_end=EPSILON_END,
            epsilon_decay=EPSILON_DECAY, buffer_size=BUFFER_SIZE,
            batch_size=BATCH_SIZE, target_update_freq=TARGET_UPDATE_FREQ,
            use_qclip=True, qclip_min=QBOUND_MIN, qclip_max=QBOUND_MAX,
            device='cpu'
        )
        env = FrozenLakeWrapper(seed=SEED)
        static_rewards, static_steps, static_success = train_agent(
            env, static_agent, "2. Static QBound + DQN", use_step_aware=False)
        results['training']['static_qbound'] = {
            'rewards': static_rewards, 'steps': static_steps,
            'total_reward': float(np.sum(static_rewards)),
            'mean_reward': float(np.mean(static_rewards)),
            'mean_steps': float(np.mean(static_steps)),
            'success_rate': float(static_success),
            'final_100_mean': float(np.mean(static_rewards[-100:])),
            'final_100_std': float(np.std(static_rewards[-100:]))
        }
        save_intermediate_results(results)

    # ===== 3. Dynamic QBound + DQN =====
    print("\n" + "=" * 80)
    print("METHOD 3: Dynamic QBound + DQN")
    print("=" * 80)

    if is_method_completed(results, 'dynamic_qbound'):
        print("⏭️  Already completed, skipping...")
    else:
        dynamic_agent = DQNAgent(
            state_dim=state_dim, action_dim=action_dim, learning_rate=LR,
            gamma=GAMMA, epsilon_start=EPSILON_START, epsilon_end=EPSILON_END,
            epsilon_decay=EPSILON_DECAY, buffer_size=BUFFER_SIZE,
            batch_size=BATCH_SIZE, target_update_freq=TARGET_UPDATE_FREQ,
            use_qclip=True, qclip_min=QBOUND_MIN, qclip_max=QBOUND_MAX,
            use_step_aware_qbound=True, max_episode_steps=MAX_STEPS,
            step_reward=1.0, reward_is_negative=False, device='cpu'
        )
        env = FrozenLakeWrapper(seed=SEED)
        dynamic_rewards, dynamic_steps, dynamic_success = train_agent(
            env, dynamic_agent, "3. Dynamic QBound + DQN", use_step_aware=True)
        results['training']['dynamic_qbound'] = {
            'rewards': dynamic_rewards, 'steps': dynamic_steps,
            'total_reward': float(np.sum(dynamic_rewards)),
            'mean_reward': float(np.mean(dynamic_rewards)),
            'mean_steps': float(np.mean(dynamic_steps)),
            'success_rate': float(dynamic_success),
            'final_100_mean': float(np.mean(dynamic_rewards[-100:])),
            'final_100_std': float(np.std(dynamic_rewards[-100:]))
        }
        save_intermediate_results(results)

    # ===== 4. Baseline DDQN =====
    print("\n" + "=" * 80)
    print("METHOD 4: Baseline DDQN")
    print("=" * 80)

    if is_method_completed(results, 'baseline_ddqn'):
        print("⏭️  Already completed, skipping...")
    else:
        baseline_ddqn_agent = DoubleDQNAgent(
            state_dim=state_dim, action_dim=action_dim, learning_rate=LR,
            gamma=GAMMA, epsilon_start=EPSILON_START, epsilon_end=EPSILON_END,
            epsilon_decay=EPSILON_DECAY, buffer_size=BUFFER_SIZE,
            batch_size=BATCH_SIZE, target_update_freq=TARGET_UPDATE_FREQ,
            use_qclip=False, device='cpu'
        )
        env = FrozenLakeWrapper(seed=SEED)
        baseline_ddqn_rewards, baseline_ddqn_steps, baseline_ddqn_success = train_agent(
            env, baseline_ddqn_agent, "4. Baseline DDQN", use_step_aware=False)
        results['training']['baseline_ddqn'] = {
            'rewards': baseline_ddqn_rewards, 'steps': baseline_ddqn_steps,
            'total_reward': float(np.sum(baseline_ddqn_rewards)),
            'mean_reward': float(np.mean(baseline_ddqn_rewards)),
            'mean_steps': float(np.mean(baseline_ddqn_steps)),
            'success_rate': float(baseline_ddqn_success),
            'final_100_mean': float(np.mean(baseline_ddqn_rewards[-100:])),
            'final_100_std': float(np.std(baseline_ddqn_rewards[-100:]))
        }
        save_intermediate_results(results)

    # ===== 5. Static QBound + DDQN =====
    print("\n" + "=" * 80)
    print("METHOD 5: Static QBound + DDQN")
    print("=" * 80)

    if is_method_completed(results, 'static_qbound_ddqn'):
        print("⏭️  Already completed, skipping...")
    else:
        static_ddqn_agent = DoubleDQNAgent(
            state_dim=state_dim, action_dim=action_dim, learning_rate=LR,
            gamma=GAMMA, epsilon_start=EPSILON_START, epsilon_end=EPSILON_END,
            epsilon_decay=EPSILON_DECAY, buffer_size=BUFFER_SIZE,
            batch_size=BATCH_SIZE, target_update_freq=TARGET_UPDATE_FREQ,
            use_qclip=True, qclip_min=QBOUND_MIN, qclip_max=QBOUND_MAX,
            device='cpu'
        )
        env = FrozenLakeWrapper(seed=SEED)
        static_ddqn_rewards, static_ddqn_steps, static_ddqn_success = train_agent(
            env, static_ddqn_agent, "5. Static QBound + DDQN", use_step_aware=False)
        results['training']['static_qbound_ddqn'] = {
            'rewards': static_ddqn_rewards, 'steps': static_ddqn_steps,
            'total_reward': float(np.sum(static_ddqn_rewards)),
            'mean_reward': float(np.mean(static_ddqn_rewards)),
            'mean_steps': float(np.mean(static_ddqn_steps)),
            'success_rate': float(static_ddqn_success),
            'final_100_mean': float(np.mean(static_ddqn_rewards[-100:])),
            'final_100_std': float(np.std(static_ddqn_rewards[-100:]))
        }
        save_intermediate_results(results)

    # ===== 6. Dynamic QBound + DDQN =====
    print("\n" + "=" * 80)
    print("METHOD 6: Dynamic QBound + DDQN")
    print("=" * 80)

    if is_method_completed(results, 'dynamic_qbound_ddqn'):
        print("⏭️  Already completed, skipping...")
    else:
        dynamic_ddqn_agent = DoubleDQNAgent(
            state_dim=state_dim, action_dim=action_dim, learning_rate=LR,
            gamma=GAMMA, epsilon_start=EPSILON_START, epsilon_end=EPSILON_END,
            epsilon_decay=EPSILON_DECAY, buffer_size=BUFFER_SIZE,
            batch_size=BATCH_SIZE, target_update_freq=TARGET_UPDATE_FREQ,
            use_qclip=True, qclip_min=QBOUND_MIN, qclip_max=QBOUND_MAX,
            use_step_aware_qbound=True, max_episode_steps=MAX_STEPS,
            step_reward=1.0, reward_is_negative=False, device='cpu'
        )
        env = FrozenLakeWrapper(seed=SEED)
        dynamic_ddqn_rewards, dynamic_ddqn_steps, dynamic_ddqn_success = train_agent(
            env, dynamic_ddqn_agent, "6. Dynamic QBound + DDQN", use_step_aware=True)
        results['training']['dynamic_qbound_ddqn'] = {
            'rewards': dynamic_ddqn_rewards, 'steps': dynamic_ddqn_steps,
            'total_reward': float(np.sum(dynamic_ddqn_rewards)),
            'mean_reward': float(np.mean(dynamic_ddqn_rewards)),
            'mean_steps': float(np.mean(dynamic_ddqn_steps)),
            'success_rate': float(dynamic_ddqn_success),
            'final_100_mean': float(np.mean(dynamic_ddqn_rewards[-100:])),
            'final_100_std': float(np.std(dynamic_ddqn_rewards[-100:]))
        }
        save_intermediate_results(results)

    # ===== Analysis and Summary =====
    print("\n" + "=" * 80)
    print("FROZENLAKE RESULTS SUMMARY (Final 100 Episodes)")
    print("=" * 80)

    methods = ['baseline', 'static_qbound', 'dynamic_qbound',
               'baseline_ddqn', 'static_qbound_ddqn', 'dynamic_qbound_ddqn']
    labels = ['Baseline DQN', 'Static QBound', 'Dynamic QBound',
              'Baseline DDQN', 'Static QBound+DDQN', 'Dynamic QBound+DDQN']

    print(f"\n{'Method':<25} {'Mean ± Std':<25} {'Success Rate':<15} {'vs Baseline':<15}")
    print("-" * 80)

    baseline_mean = results['training']['baseline']['final_100_mean']

    for method, label in zip(methods, labels):
        data = results['training'][method]
        mean = data['final_100_mean']
        std = data['final_100_std']
        success = data['success_rate']

        improvement = ((mean - baseline_mean) / abs(baseline_mean)) * 100 if baseline_mean != 0 else 0
        improvement_str = f"{improvement:+.1f}%"

        print(f"{label:<25} {mean:>8.3f} ± {std:<8.3f} {success:>10.2%}   {improvement_str:>10}")

    # ===== Save Final Results =====
    timestamp = results['timestamp']
    final_output_file = f"/root/projects/QBound/results/frozenlake/6way_comparison_{timestamp}.json"

    with open(final_output_file, 'w') as f:
        json.dump(results, f, indent=2)

    if os.path.exists(RESULTS_FILE):
        os.remove(RESULTS_FILE)

    print(f"\n✓ Results saved to: {final_output_file}")
    print("\n" + "=" * 80)
    print("FrozenLake 6-Way Comparison Complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
