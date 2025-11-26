"""
Acrobot-v1 DQN with Static QBound

Organization: Time-step Dependent Rewards
This script tests DQN and Double DQN with static QBound on Acrobot.

Environment: Acrobot-v1 (continuous state, discrete actions)
Reward Type: Dense negative (-1 per step until goal) - TIME-STEP DEPENDENT
Max steps: 500

Methods Tested (4 total):
1. Baseline DQN - No QBound, no Double-Q
2. Static QBound + DQN - Q ∈ [-99.34, 0]
3. Baseline DDQN - No QBound, with Double-Q
4. Static QBound + DDQN - Q ∈ [-99.34, 0] + Double-Q

QBound Applicability:
✓ Static QBound - Applicable (time-step dependent negative rewards)

Q_min calculation: Q_min = -1 * (1 - γ^H) / (1 - γ) for H=500, γ=0.99 ≈ -99.34
Q_max = 0 (best case: reach goal immediately)
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
parser = argparse.ArgumentParser(description='Acrobot DQN with static QBound')
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility (default: 42)')
args = parser.parse_args()

# Reproducibility
SEED = args.seed
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

# Results file for crash recovery
RESULTS_FILE = f"/root/projects/QBound/results/acrobot/dqn_full_qbound_seed{SEED}_in_progress.json"

# Environment parameters
ENV_NAME = "Acrobot-v1"
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

# QBound parameters for Acrobot (Dense Negative Reward)
# Reward: -1 per step until terminal
# Q_max = 0 (best case: reach goal immediately)
# Q_min = -1 * (1 - γ^H) / (1 - γ) for H=500, γ=0.99
QBOUND_MIN = -(1 - GAMMA**MAX_STEPS) / (1 - GAMMA)  # ≈ -99.34
QBOUND_MAX = 0.0


class AcrobotWrapper:
    """Wrapper for Acrobot to work with our DQN agent."""

    def __init__(self, seed=None, max_episode_steps=500):
        self.env = gym.make('Acrobot-v1')
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
    results_dir = os.path.dirname(RESULTS_FILE)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)
        
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"   💾 Progress saved to: {RESULTS_FILE}")


def is_method_completed(results, method_name):
    """Check if a method has already been completed"""
    return method_name in results.get('training', {})


def train_agent(env, agent, agent_name, max_episodes=MAX_EPISODES, track_violations=False):
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

            # Store transition
            agent.store_transition(state, action, reward, next_state, done)

            # Train
            loss, violations = agent.train_step()
            if track_violations and violations is not None:
                violation_stats_episode.append(violations)

            state = next_state
            episode_reward += reward
            step += 1

        episode_rewards.append(episode_reward)
        episode_steps.append(step)

        if track_violations and violation_stats_episode:
            # Average violations over the episode
            avg_violations = {}
            for key in violation_stats_episode[0].keys():
                values = [v[key] for v in violation_stats_episode if key in v]
                avg_violations[key] = float(np.mean(values)) if values else 0.0
            episode_violations.append(avg_violations)
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

    return episode_rewards, episode_violations


def main():
    print("=" * 80)
    print("Acrobot-v1: DQN 4-Way Comparison (Time-Step Dependent)")
    print("Organization: Time-step Dependent Rewards (Negative Dense)")
    print("Testing: Static QBound on DQN and Double DQN")
    print("=" * 80)

    print(f"\nConfiguration:")
    print(f"  Environment: {ENV_NAME}")
    print(f"  Reward Structure: Dense negative (-1 per step, time-step dependent)")
    print(f"  Episodes: {MAX_EPISODES}")
    print(f"  Max steps: {MAX_STEPS}")
    print(f"  Gamma: {GAMMA}")
    print(f"  Q_min: {QBOUND_MIN:.4f}")
    print(f"  Q_max: {QBOUND_MAX}")
    print(f"  Seed: {SEED}")

    # Load or initialize results
    results = load_existing_results()
    if results is None:
        print("\n🆕 Starting fresh experiment...")
        results = {
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'experiment_type': 'time_step_dependent',
            'script_name': 'train_acrobot_dqn_full_qbound.py',
            'config': {
                'env': ENV_NAME,
                'reward_structure': 'time_step_dependent_negative',
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
                'seed': SEED
            },
            'training': {}
        }
    else:
        print("   ⏩ Resuming experiment...\n")

    # Create environment
    env = AcrobotWrapper(seed=SEED)

    # ===== 1. Baseline DQN =====
    print("\n" + "=" * 80)
    print("METHOD 1: Baseline DQN")
    print("=" * 80)

    if is_method_completed(results, 'dqn'):
        print("⏭️  Already completed, skipping...")
    else:
        dqn_agent = DQNAgent(
            state_dim=env.observation_space,
            action_dim=env.action_space,
            learning_rate=LR,
            gamma=GAMMA,
            epsilon_start=EPSILON_START,
            epsilon_end=EPSILON_END,
            epsilon_decay=EPSILON_DECAY,
            buffer_size=BUFFER_SIZE,
            batch_size=BATCH_SIZE,
            target_update_freq=TARGET_UPDATE_FREQ,
            use_qclip=False,
            device='cpu'
        )

        dqn_rewards, _ = train_agent(env, dqn_agent, "1. Baseline DQN", track_violations=False)
        results['training']['dqn'] = {
            'rewards': dqn_rewards,
            'total_reward': float(np.sum(dqn_rewards)),
            'mean_reward': float(np.mean(dqn_rewards)),
            'final_100_mean': float(np.mean(dqn_rewards[-100:])),
            'final_100_std': float(np.std(dqn_rewards[-100:])),
            'violations': None
        }
        save_intermediate_results(results)

    # ===== 2. Static QBound + DQN =====
    print("\n" + "=" * 80)
    print("METHOD 2: Static QBound + DQN")
    print("=" * 80)

    if is_method_completed(results, 'static_qbound_dqn'):
        print("⏭️  Already completed, skipping...")
    else:
        static_dqn_agent = DQNAgent(
            state_dim=env.observation_space,
            action_dim=env.action_space,
            learning_rate=LR,
            gamma=GAMMA,
            epsilon_start=EPSILON_START,
            epsilon_end=EPSILON_END,
            epsilon_decay=EPSILON_DECAY,
            buffer_size=BUFFER_SIZE,
            batch_size=BATCH_SIZE,
            target_update_freq=TARGET_UPDATE_FREQ,
            use_qclip=True,
            qclip_min=QBOUND_MIN,
            qclip_max=QBOUND_MAX,
            device='cpu'
        )

        static_dqn_rewards, static_dqn_violations = train_agent(env, static_dqn_agent, "2. Static QBound + DQN", track_violations=True)
        
        # Compute violation statistics
        valid_violations = [v for v in static_dqn_violations if v is not None]
        violation_summary = {
            'per_episode': valid_violations,
            'mean': {k: float(np.mean([v[k] for v in valid_violations])) for k in valid_violations[0].keys()} if valid_violations else {},
            'final_100': {k: float(np.mean([v[k] for v in valid_violations[-100:]])) for k in valid_violations[0].keys()} if valid_violations else {}
        }

        results['training']['static_qbound_dqn'] = {
            'rewards': static_dqn_rewards,
            'total_reward': float(np.sum(static_dqn_rewards)),
            'mean_reward': float(np.mean(static_dqn_rewards)),
            'final_100_mean': float(np.mean(static_dqn_rewards[-100:])),
            'final_100_std': float(np.std(static_dqn_rewards[-100:])),
            'violations': violation_summary
        }
        save_intermediate_results(results)

    # ===== 3. Baseline DDQN =====
    print("\n" + "=" * 80)
    print("METHOD 3: Baseline DDQN")
    print("=" * 80)

    if is_method_completed(results, 'double_dqn'):
        print("⏭️  Already completed, skipping...")
    else:
        ddqn_agent = DoubleDQNAgent(
            state_dim=env.observation_space,
            action_dim=env.action_space,
            learning_rate=LR,
            gamma=GAMMA,
            epsilon_start=EPSILON_START,
            epsilon_end=EPSILON_END,
            epsilon_decay=EPSILON_DECAY,
            buffer_size=BUFFER_SIZE,
            batch_size=BATCH_SIZE,
            target_update_freq=TARGET_UPDATE_FREQ,
            use_qclip=False,
            device='cpu'
        )

        ddqn_rewards, _ = train_agent(env, ddqn_agent, "3. Baseline DDQN", track_violations=False)
        results['training']['double_dqn'] = {
            'rewards': ddqn_rewards,
            'total_reward': float(np.sum(ddqn_rewards)),
            'mean_reward': float(np.mean(ddqn_rewards)),
            'final_100_mean': float(np.mean(ddqn_rewards[-100:])),
            'final_100_std': float(np.std(ddqn_rewards[-100:])),
            'violations': None
        }
        save_intermediate_results(results)

    # ===== 4. Static QBound + DDQN =====
    print("\n" + "=" * 80)
    print("METHOD 4: Static QBound + DDQN")
    print("=" * 80)

    if is_method_completed(results, 'static_qbound_double_dqn'):
        print("⏭️  Already completed, skipping...")
    else:
        static_ddqn_agent = DoubleDQNAgent(
            state_dim=env.observation_space,
            action_dim=env.action_space,
            learning_rate=LR,
            gamma=GAMMA,
            epsilon_start=EPSILON_START,
            epsilon_end=EPSILON_END,
            epsilon_decay=EPSILON_DECAY,
            buffer_size=BUFFER_SIZE,
            batch_size=BATCH_SIZE,
            target_update_freq=TARGET_UPDATE_FREQ,
            use_qclip=True,
            qclip_min=QBOUND_MIN,
            qclip_max=QBOUND_MAX,
            device='cpu'
        )

        static_ddqn_rewards, static_ddqn_violations = train_agent(env, static_ddqn_agent, "4. Static QBound + DDQN", track_violations=True)
        
        # Compute violation statistics
        valid_violations = [v for v in static_ddqn_violations if v is not None]
        violation_summary = {
            'per_episode': valid_violations,
            'mean': {k: float(np.mean([v[k] for v in valid_violations])) for k in valid_violations[0].keys()} if valid_violations else {},
            'final_100': {k: float(np.mean([v[k] for v in valid_violations[-100:]])) for k in valid_violations[0].keys()} if valid_violations else {}
        }

        results['training']['static_qbound_double_dqn'] = {
            'rewards': static_ddqn_rewards,
            'total_reward': float(np.sum(static_ddqn_rewards)),
            'mean_reward': float(np.mean(static_ddqn_rewards)),
            'final_100_mean': float(np.mean(static_ddqn_rewards[-100:])),
            'final_100_std': float(np.std(static_ddqn_rewards[-100:])),
            'violations': violation_summary
        }
        save_intermediate_results(results)

    # ===== Analysis and Summary =====
    print("\n" + "=" * 80)
    print("ACROBOT DQN RESULTS SUMMARY (Final 100 Episodes)")
    print("=" * 80)
    print()
    print(f"{'Method':<30} {'Mean ± Std':<25} {'vs DQN Baseline'}")
    print("-" * 80)

    baseline_mean = results['training']['dqn']['final_100_mean']
    baseline_std = results['training']['dqn']['final_100_std']

    methods = [
        ('Baseline DQN', 'dqn'),
        ('Static QBound + DQN', 'static_qbound_dqn'),
        ('Baseline DDQN', 'double_dqn'),
        ('Static QBound + DDQN', 'static_qbound_double_dqn')
    ]

    for method_name, method_key in methods:
        if method_key in results['training']:
            mean = results['training'][method_key]['final_100_mean']
            std = results['training'][method_key]['final_100_std']

            if method_key == 'dqn':
                print(f"{method_name:<30} {mean:>8.2f} ± {std:<8.2f}   +0.0%")
            else:
                improvement = ((mean - baseline_mean) / abs(baseline_mean)) * 100 if baseline_mean != 0 else 0
                print(f"{method_name:<30} {mean:>8.2f} ± {std:<8.2f}   {improvement:+.1f}%")

    # ===== Save Final Results =====
    timestamp = results['timestamp']
    final_output_file = f"/root/projects/QBound/results/acrobot/dqn_full_qbound_seed{SEED}_{timestamp}.json"

    # Ensure directory exists
    os.makedirs(os.path.dirname(final_output_file), exist_ok=True)

    with open(final_output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to: {final_output_file}")

    # Delete progress file
    if os.path.exists(RESULTS_FILE):
        os.remove(RESULTS_FILE)

    print("\n" + "=" * 80)
    print("Acrobot DQN 4-Way Comparison Complete!")
    print("=" * 80)
    print()
    print("📊 Key Takeaways:")
    print("  - Tests Static QBound on time-step dependent NEGATIVE rewards")
    print("  - Validates QBound effectiveness on different negative reward structure")
    print("  - Q_min = -99.34, Q_max = 0 (negative dense reward)")


if __name__ == "__main__":
    main()
