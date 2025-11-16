"""
CartPole DQN with Full QBound (Static + Dynamic)

Organization: Time-step Dependent Rewards
This script tests DQN and Double DQN with all QBound variants.

Environment: CartPole-v1 (continuous state, discrete actions)
Reward Type: Dense positive (+1 per step) - TIME-STEP DEPENDENT
Max steps: 500

Methods Tested (6 total):
1. Baseline DQN - No QBound, no Double-Q
2. Static QBound + DQN - Q ∈ [0, 99.34]
3. Dynamic QBound + DQN - Q ∈ [0, Q_max(t)] with step-aware bounds
4. Baseline DDQN - No QBound, with Double-Q
5. Static QBound + DDQN - Q ∈ [0, 99.34] + Double-Q
6. Dynamic QBound + DDQN - Q ∈ [0, Q_max(t)] + Double-Q

QBound Applicability:
✓ Static QBound - Applicable (time-step dependent rewards)
✓ Dynamic QBound - Applicable (time-step dependent rewards)

Q_max calculation: Q_max = (1 - γ^H) / (1 - γ) for H=500, γ=0.99 ≈ 99.34
"""

import sys
sys.path.insert(0, '/root/projects/QBound/src')

import numpy as np
import torch
import json
import random
import os
import argparse
import gymnasium as gym
from datetime import datetime
from dqn_agent import DQNAgent
from double_dqn_agent import DoubleDQNAgent
from tqdm import tqdm

# Parse command line arguments
parser = argparse.ArgumentParser(description='CartPole DQN with full QBound (static + dynamic)')
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility (default: 42)')
args = parser.parse_args()

# Reproducibility
SEED = args.seed
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

# Results file for crash recovery
RESULTS_FILE = f"/root/projects/QBound/results/cartpole/dqn_full_qbound_seed{SEED}_in_progress.json"

# Environment parameters
ENV_NAME = "CartPole-v1"
MAX_EPISODES = 500
MAX_STEPS = 500
EVAL_EPISODES = 10

# Shared hyperparameters
LR = 0.001
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
BATCH_SIZE = 64
BUFFER_SIZE = 10000
TARGET_UPDATE_FREQ = 100

# QBound parameters for CartPole (Dense Positive Reward)
QBOUND_MIN = 0.0
QBOUND_MAX = (1 - GAMMA**MAX_STEPS) / (1 - GAMMA)  # ≈ 99.34


class CartPoleWrapper:
    """Wrapper for CartPole to work with our DQN agent."""

    def __init__(self, seed=None, max_episode_steps=500):
        self.env = gym.make('CartPole-v1', max_episode_steps=max_episode_steps)
        self.observation_space = self.env.observation_space.shape[0]
        self.action_space = self.env.action_space.n
        self.seed = seed

    def reset(self):
        """Reset and return state."""
        if self.seed is not None:
            state, _ = self.env.reset(seed=self.seed)
            self.seed += 1  # Increment for next reset
        else:
            state, _ = self.env.reset()
        return state

    def step(self, action):
        """Take step and return next state."""
        next_state, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return next_state, reward, done, info


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


def train_agent(env, agent, agent_name, max_episodes=MAX_EPISODES, use_step_aware=False, track_violations=False):
    """Train agent and return results with optional violation tracking"""
    print(f"\n>>> Training {agent_name}...")

    episode_rewards = []
    episode_steps = []
    episode_violations = [] if track_violations else None

    for episode in tqdm(range(max_episodes), desc=agent_name):
        state = env.reset()

        episode_reward = 0
        done = False
        step = 0
        violation_stats_episode = [] if track_violations else None

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

            # Train and collect violation stats
            loss, violations = agent.train_step()

            if track_violations and violations is not None:
                violation_stats_episode.append(violations)

            episode_reward += reward
            state = next_state
            step += 1

        episode_rewards.append(episode_reward)
        episode_steps.append(step)

        # Aggregate violations for this episode
        if track_violations and violation_stats_episode:
            episode_violation_summary = {
                k: np.mean([v[k] for v in violation_stats_episode])
                for k in violation_stats_episode[0].keys()
            }
            episode_violations.append(episode_violation_summary)
        elif track_violations:
            episode_violations.append(None)

        # Progress update
        if (episode + 1) % 100 == 0:
            recent_avg_reward = np.mean(episode_rewards[-100:])
            recent_avg_steps = np.mean(episode_steps[-100:])
            progress_msg = f"  Episode {episode + 1}/{max_episodes} - Avg reward: {recent_avg_reward:.1f}, Avg steps: {recent_avg_steps:.1f}, ε={agent.epsilon:.3f}"

            if track_violations and episode_violations and episode_violations[-1] is not None:
                recent_violations = [v for v in episode_violations[-100:] if v is not None]
                if recent_violations:
                    avg_violation_rate = np.mean([v['total_violation_rate'] for v in recent_violations])
                    progress_msg += f", Violations: {avg_violation_rate:.1%}"

            print(progress_msg)

    return episode_rewards, episode_steps, episode_violations


def main():
    print("=" * 80)
    print("CartPole DQN with Full QBound (Static + Dynamic)")
    print("Organization: Time-step Dependent Rewards")
    print("=" * 80)

    print(f"\nConfiguration:")
    print(f"  Environment: {ENV_NAME}")
    print(f"  Reward Type: Time-step dependent (+1 per step)")
    print(f"  Episodes: {MAX_EPISODES}")
    print(f"  Max steps per episode: {MAX_STEPS}")
    print(f"  Discount factor γ: {GAMMA}")
    print(f"  Learning rate: {LR}")
    print(f"  QBound range: [{QBOUND_MIN:.2f}, {QBOUND_MAX:.2f}]")
    print(f"  Formula: Q_max = (1 - γ^H) / (1 - γ) = {QBOUND_MAX:.2f}")
    print(f"  Random seed: {SEED}")
    print("=" * 80)

    # Load existing results or create new
    results = load_existing_results()
    if results is None:
        print("\n🆕 Starting fresh experiment...")
        results = {
            'experiment_type': 'time_step_dependent',
            'script_name': 'train_cartpole_dqn_full_qbound.py',
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'config': {
                'env': ENV_NAME,
                'episodes': MAX_EPISODES,
                'max_steps': MAX_STEPS,
                'gamma': GAMMA,
                'lr': LR,
                'epsilon_start': EPSILON_START,
                'epsilon_end': EPSILON_END,
                'epsilon_decay': EPSILON_DECAY,
                'batch_size': BATCH_SIZE,
                'buffer_size': BUFFER_SIZE,
                'target_update_freq': TARGET_UPDATE_FREQ,
                'qbound_min': QBOUND_MIN,
                'qbound_max': QBOUND_MAX,
                'formula': 'Q_max = (1 - gamma^H) / (1 - gamma)',
                'seed': SEED,
                'dynamic_qbound_params': {
                    'max_episode_steps': MAX_STEPS,
                    'step_reward': 1.0,
                    'reward_is_negative': False
                }
            },
            'training': {}
        }
    else:
        print("   ⏩ Resuming experiment...\n")

    # ===== 1. Baseline DQN =====
    print("\n" + "=" * 80)
    print("METHOD 1: Baseline DQN")
    print("=" * 80)

    if is_method_completed(results, 'baseline'):
        print("⏭️  Already completed, skipping...")
    else:
        env = CartPoleWrapper(seed=SEED, max_episode_steps=MAX_STEPS)
        state_dim = env.observation_space
        action_dim = env.action_space

        baseline_agent = DQNAgent(
            state_dim=state_dim, action_dim=action_dim, learning_rate=LR,
            gamma=GAMMA, epsilon_start=EPSILON_START, epsilon_end=EPSILON_END,
            epsilon_decay=EPSILON_DECAY, buffer_size=BUFFER_SIZE,
            batch_size=BATCH_SIZE, target_update_freq=TARGET_UPDATE_FREQ,
            use_qclip=False, device='cpu'
        )
        baseline_rewards, baseline_steps, _ = train_agent(
            env, baseline_agent, "1. Baseline DQN", use_step_aware=False, track_violations=False)
        results['training']['baseline'] = {
            'rewards': baseline_rewards, 'steps': baseline_steps,
            'total_reward': float(np.sum(baseline_rewards)),
            'mean_reward': float(np.mean(baseline_rewards)),
            'mean_steps': float(np.mean(baseline_steps)),
            'final_100_mean': float(np.mean(baseline_rewards[-100:])),
            'final_100_std': float(np.std(baseline_rewards[-100:])),
            'violations': None  # No QBound
        }
        save_intermediate_results(results)

    # ===== 2. Static QBound + DQN =====
    print("\n" + "=" * 80)
    print("METHOD 2: Static QBound + DQN")
    print("=" * 80)

    if is_method_completed(results, 'static_qbound'):
        print("⏭️  Already completed, skipping...")
    else:
        env = CartPoleWrapper(seed=SEED, max_episode_steps=MAX_STEPS)
        state_dim = env.observation_space
        action_dim = env.action_space

        static_agent = DQNAgent(
            state_dim=state_dim, action_dim=action_dim, learning_rate=LR,
            gamma=GAMMA, epsilon_start=EPSILON_START, epsilon_end=EPSILON_END,
            epsilon_decay=EPSILON_DECAY, buffer_size=BUFFER_SIZE,
            batch_size=BATCH_SIZE, target_update_freq=TARGET_UPDATE_FREQ,
            use_qclip=True, qclip_min=QBOUND_MIN, qclip_max=QBOUND_MAX,
            device='cpu'
        )
        static_rewards, static_steps, static_violations = train_agent(
            env, static_agent, "2. Static QBound + DQN", use_step_aware=False, track_violations=True)

        # Compute violation statistics
        valid_violations = [v for v in static_violations if v is not None]
        violation_summary = {
            'per_episode': valid_violations,
            'mean': {k: float(np.mean([v[k] for v in valid_violations])) for k in valid_violations[0].keys()} if valid_violations else {},
            'final_100': {k: float(np.mean([v[k] for v in valid_violations[-100:]])) for k in valid_violations[0].keys()} if valid_violations else {}
        }

        results['training']['static_qbound'] = {
            'rewards': static_rewards, 'steps': static_steps,
            'total_reward': float(np.sum(static_rewards)),
            'mean_reward': float(np.mean(static_rewards)),
            'mean_steps': float(np.mean(static_steps)),
            'final_100_mean': float(np.mean(static_rewards[-100:])),
            'final_100_std': float(np.std(static_rewards[-100:])),
            'qmax': float(QBOUND_MAX),
            'violations': violation_summary
        }
        save_intermediate_results(results)

    # ===== 3. Dynamic QBound + DQN =====
    print("\n" + "=" * 80)
    print("METHOD 3: Dynamic QBound + DQN")
    print("=" * 80)

    if is_method_completed(results, 'dynamic_qbound'):
        print("⏭️  Already completed, skipping...")
    else:
        env = CartPoleWrapper(seed=SEED, max_episode_steps=MAX_STEPS)
        state_dim = env.observation_space
        action_dim = env.action_space

        dynamic_agent = DQNAgent(
            state_dim=state_dim, action_dim=action_dim, learning_rate=LR,
            gamma=GAMMA, epsilon_start=EPSILON_START, epsilon_end=EPSILON_END,
            epsilon_decay=EPSILON_DECAY, buffer_size=BUFFER_SIZE,
            batch_size=BATCH_SIZE, target_update_freq=TARGET_UPDATE_FREQ,
            use_qclip=True, qclip_min=QBOUND_MIN, qclip_max=QBOUND_MAX,
            use_step_aware_qbound=True, max_episode_steps=MAX_STEPS,
            step_reward=1.0, reward_is_negative=False, device='cpu'
        )
        dynamic_rewards, dynamic_steps, dynamic_violations = train_agent(
            env, dynamic_agent, "3. Dynamic QBound + DQN", use_step_aware=True, track_violations=True)

        # Compute violation statistics
        valid_violations = [v for v in dynamic_violations if v is not None]
        violation_summary = {
            'per_episode': valid_violations,
            'mean': {k: float(np.mean([v[k] for v in valid_violations])) for k in valid_violations[0].keys()} if valid_violations else {},
            'final_100': {k: float(np.mean([v[k] for v in valid_violations[-100:]])) for k in valid_violations[0].keys()} if valid_violations else {}
        }

        results['training']['dynamic_qbound'] = {
            'rewards': dynamic_rewards, 'steps': dynamic_steps,
            'total_reward': float(np.sum(dynamic_rewards)),
            'mean_reward': float(np.mean(dynamic_rewards)),
            'mean_steps': float(np.mean(dynamic_steps)),
            'final_100_mean': float(np.mean(dynamic_rewards[-100:])),
            'final_100_std': float(np.std(dynamic_rewards[-100:])),
            'qmax_formula': 'Q_max(t) = (1 - gamma^(H-t)) / (1 - gamma)',
            'violations': violation_summary
        }
        save_intermediate_results(results)

    # ===== 4. Baseline DDQN =====
    print("\n" + "=" * 80)
    print("METHOD 4: Baseline DDQN")
    print("=" * 80)

    if is_method_completed(results, 'baseline_ddqn'):
        print("⏭️  Already completed, skipping...")
    else:
        env = CartPoleWrapper(seed=SEED, max_episode_steps=MAX_STEPS)
        state_dim = env.observation_space
        action_dim = env.action_space

        baseline_ddqn_agent = DoubleDQNAgent(
            state_dim=state_dim, action_dim=action_dim, learning_rate=LR,
            gamma=GAMMA, epsilon_start=EPSILON_START, epsilon_end=EPSILON_END,
            epsilon_decay=EPSILON_DECAY, buffer_size=BUFFER_SIZE,
            batch_size=BATCH_SIZE, target_update_freq=TARGET_UPDATE_FREQ,
            use_qclip=False, device='cpu'
        )
        baseline_ddqn_rewards, baseline_ddqn_steps, _ = train_agent(
            env, baseline_ddqn_agent, "4. Baseline DDQN", use_step_aware=False, track_violations=False)
        results['training']['baseline_ddqn'] = {
            'rewards': baseline_ddqn_rewards, 'steps': baseline_ddqn_steps,
            'total_reward': float(np.sum(baseline_ddqn_rewards)),
            'mean_reward': float(np.mean(baseline_ddqn_rewards)),
            'mean_steps': float(np.mean(baseline_ddqn_steps)),
            'final_100_mean': float(np.mean(baseline_ddqn_rewards[-100:])),
            'final_100_std': float(np.std(baseline_ddqn_rewards[-100:])),
            'violations': None  # No QBound
        }
        save_intermediate_results(results)

    # ===== 5. Static QBound + DDQN =====
    print("\n" + "=" * 80)
    print("METHOD 5: Static QBound + DDQN")
    print("=" * 80)

    if is_method_completed(results, 'static_qbound_ddqn'):
        print("⏭️  Already completed, skipping...")
    else:
        env = CartPoleWrapper(seed=SEED, max_episode_steps=MAX_STEPS)
        state_dim = env.observation_space
        action_dim = env.action_space

        static_ddqn_agent = DoubleDQNAgent(
            state_dim=state_dim, action_dim=action_dim, learning_rate=LR,
            gamma=GAMMA, epsilon_start=EPSILON_START, epsilon_end=EPSILON_END,
            epsilon_decay=EPSILON_DECAY, buffer_size=BUFFER_SIZE,
            batch_size=BATCH_SIZE, target_update_freq=TARGET_UPDATE_FREQ,
            use_qclip=True, qclip_min=QBOUND_MIN, qclip_max=QBOUND_MAX,
            device='cpu'
        )
        static_ddqn_rewards, static_ddqn_steps, static_ddqn_violations = train_agent(
            env, static_ddqn_agent, "5. Static QBound + DDQN", use_step_aware=False, track_violations=True)

        # Compute violation statistics
        valid_violations = [v for v in static_ddqn_violations if v is not None]
        violation_summary = {
            'per_episode': valid_violations,
            'mean': {k: float(np.mean([v[k] for v in valid_violations])) for k in valid_violations[0].keys()} if valid_violations else {},
            'final_100': {k: float(np.mean([v[k] for v in valid_violations[-100:]])) for k in valid_violations[0].keys()} if valid_violations else {}
        }

        results['training']['static_qbound_ddqn'] = {
            'rewards': static_ddqn_rewards, 'steps': static_ddqn_steps,
            'total_reward': float(np.sum(static_ddqn_rewards)),
            'mean_reward': float(np.mean(static_ddqn_rewards)),
            'mean_steps': float(np.mean(static_ddqn_steps)),
            'final_100_mean': float(np.mean(static_ddqn_rewards[-100:])),
            'final_100_std': float(np.std(static_ddqn_rewards[-100:])),
            'violations': violation_summary
        }
        save_intermediate_results(results)

    # ===== 6. Dynamic QBound + DDQN =====
    print("\n" + "=" * 80)
    print("METHOD 6: Dynamic QBound + DDQN")
    print("=" * 80)

    if is_method_completed(results, 'dynamic_qbound_ddqn'):
        print("⏭️  Already completed, skipping...")
    else:
        env = CartPoleWrapper(seed=SEED, max_episode_steps=MAX_STEPS)
        state_dim = env.observation_space
        action_dim = env.action_space

        dynamic_ddqn_agent = DoubleDQNAgent(
            state_dim=state_dim, action_dim=action_dim, learning_rate=LR,
            gamma=GAMMA, epsilon_start=EPSILON_START, epsilon_end=EPSILON_END,
            epsilon_decay=EPSILON_DECAY, buffer_size=BUFFER_SIZE,
            batch_size=BATCH_SIZE, target_update_freq=TARGET_UPDATE_FREQ,
            use_qclip=True, qclip_min=QBOUND_MIN, qclip_max=QBOUND_MAX,
            use_step_aware_qbound=True, max_episode_steps=MAX_STEPS,
            step_reward=1.0, reward_is_negative=False, device='cpu'
        )
        dynamic_ddqn_rewards, dynamic_ddqn_steps, dynamic_ddqn_violations = train_agent(
            env, dynamic_ddqn_agent, "6. Dynamic QBound + DDQN", use_step_aware=True, track_violations=True)

        # Compute violation statistics
        valid_violations = [v for v in dynamic_ddqn_violations if v is not None]
        violation_summary = {
            'per_episode': valid_violations,
            'mean': {k: float(np.mean([v[k] for v in valid_violations])) for k in valid_violations[0].keys()} if valid_violations else {},
            'final_100': {k: float(np.mean([v[k] for v in valid_violations[-100:]])) for k in valid_violations[0].keys()} if valid_violations else {}
        }

        results['training']['dynamic_qbound_ddqn'] = {
            'rewards': dynamic_ddqn_rewards, 'steps': dynamic_ddqn_steps,
            'total_reward': float(np.sum(dynamic_ddqn_rewards)),
            'mean_reward': float(np.mean(dynamic_ddqn_rewards)),
            'mean_steps': float(np.mean(dynamic_ddqn_steps)),
            'final_100_mean': float(np.mean(dynamic_ddqn_rewards[-100:])),
            'final_100_std': float(np.std(dynamic_ddqn_rewards[-100:])),
            'violations': violation_summary
        }
        save_intermediate_results(results)

    # ===== Analysis and Summary =====
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY (Final 100 Episodes)")
    print("=" * 80)

    methods = ['baseline', 'static_qbound', 'dynamic_qbound',
               'baseline_ddqn', 'static_qbound_ddqn', 'dynamic_qbound_ddqn']
    labels = ['Baseline DQN', 'Static QBound', 'Dynamic QBound',
              'Baseline DDQN', 'Static QBound+DDQN', 'Dynamic QBound+DDQN']

    print(f"\n{'Method':<25} {'Mean ± Std':<30} {'vs Baseline':<15}")
    print("-" * 80)

    baseline_mean = results['training']['baseline']['final_100_mean']

    for method, label in zip(methods, labels):
        data = results['training'][method]
        mean = data['final_100_mean']
        std = data['final_100_std']

        improvement = ((mean - baseline_mean) / abs(baseline_mean)) * 100 if baseline_mean != 0 else 0
        improvement_str = f"{improvement:+.1f}%"

        print(f"{label:<25} {mean:>8.1f} ± {std:<15.1f}   {improvement_str:>10}")

    # ===== Save Final Results =====
    timestamp = results['timestamp']
    final_output_file = f"/root/projects/QBound/results/cartpole/dqn_full_qbound_seed{SEED}_{timestamp}.json"
    os.makedirs('/root/projects/QBound/results/cartpole', exist_ok=True)

    with open(final_output_file, 'w') as f:
        json.dump(results, f, indent=2)

    if os.path.exists(RESULTS_FILE):
        os.remove(RESULTS_FILE)

    print(f"\n✓ Results saved to: {final_output_file}")
    print("\n" + "=" * 80)
    print("Experiment Complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
