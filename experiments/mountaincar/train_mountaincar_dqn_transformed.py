"""
MountainCar-v0 DQN with TRANSFORMED Q-Values (Negative → Positive)

EXPERIMENT GOAL: Test if QBound's failure on negative rewards is due to the
negative value range itself, or due to other factors.

TRANSFORMATION:
- Original: Q ∈ [-86.60, 0] (negative reward environment)
- Transformed: Q ∈ [0, 86.60] (shifted to positive range like CartPole)
- Method: Add abs(Q_min) = 86.60 to all Q-values

This makes MountainCar's Q-value distribution similar to CartPole's Q ∈ [0, 99.34],
where QBound was successful.

Environment: MountainCar-v0 (continuous state, discrete actions)
Reward Type: Dense negative (-1 per step until goal)
Max steps: 200

Methods Tested (2 total):
1. Baseline DQN - Transformed Q-values, no QBound
2. Static QBound + DQN - Transformed Q ∈ [0, 86.60]

Expected Outcome:
- If transformation fixes QBound → Issue was negative value range
- If QBound still degrades → Issue is violation rate or other factors
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
from dqn_agent_transformed import TransformedDQNAgent
from tqdm import tqdm

# Parse command line arguments
parser = argparse.ArgumentParser(description='MountainCar DQN with Transformed Q-values')
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility (default: 42)')
args = parser.parse_args()

# Reproducibility
SEED = args.seed
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

# Results file for crash recovery
RESULTS_FILE = f"/root/projects/QBound/results/mountaincar/dqn_transformed_seed{SEED}_in_progress.json"

# Environment parameters
ENV_NAME = "MountainCar-v0"
MAX_EPISODES = 500
MAX_STEPS = 200
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

# QBound parameters for MountainCar (ORIGINAL negative bounds)
# Original: Q ∈ [-86.60, 0]
# Transformed: Q ∈ [0, 86.60]
QBOUND_MIN_ORIGINAL = -(1 - GAMMA**MAX_STEPS) / (1 - GAMMA)  # ≈ -86.60
QBOUND_MAX_ORIGINAL = 0.0


class MountainCarWrapper:
    """Wrapper for MountainCar to work with our DQN agent."""

    def __init__(self, seed=None, max_episode_steps=200):
        self.env = gym.make('MountainCar-v0')
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
    print("MountainCar-v0: TRANSFORMED DQN (Negative → Positive Q-Values)")
    print("Testing: Does positive Q-value range fix QBound degradation?")
    print("=" * 80)

    print(f"\nConfiguration:")
    print(f"  Environment: {ENV_NAME}")
    print(f"  Reward Structure: Dense negative (-1 per step)")
    print(f"  Original Q bounds: [{QBOUND_MIN_ORIGINAL:.2f}, {QBOUND_MAX_ORIGINAL:.2f}]")
    print(f"  Transformed Q bounds: [0.0, {abs(QBOUND_MIN_ORIGINAL):.2f}]")
    print(f"  Episodes: {MAX_EPISODES}")
    print(f"  Max steps: {MAX_STEPS}")
    print(f"  Gamma: {GAMMA}")
    print(f"  Seed: {SEED}")

    # Load or initialize results
    results = load_existing_results()
    if results is None:
        print("\n🆕 Starting fresh experiment...")
        results = {
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'experiment_type': 'transformed_qvalues',
            'script_name': 'train_mountaincar_dqn_transformed.py',
            'config': {
                'env': ENV_NAME,
                'reward_structure': 'time_step_dependent_negative',
                'transformation': 'shift_to_positive',
                'original_qbound_min': QBOUND_MIN_ORIGINAL,
                'original_qbound_max': QBOUND_MAX_ORIGINAL,
                'transformed_qbound_min': 0.0,
                'transformed_qbound_max': abs(QBOUND_MIN_ORIGINAL),
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
                'seed': SEED
            },
            'training': {}
        }
    else:
        print("   ⏩ Resuming experiment...\n")

    # Create environment
    env = MountainCarWrapper(seed=SEED)

    # ===== 1. Baseline Transformed DQN =====
    print("\n" + "=" * 80)
    print("METHOD 1: Baseline Transformed DQN (No QBound)")
    print("=" * 80)

    if is_method_completed(results, 'transformed_dqn'):
        print("⏭️  Already completed, skipping...")
    else:
        dqn_agent = TransformedDQNAgent(
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
            qclip_max_original=QBOUND_MAX_ORIGINAL,
            qclip_min_original=QBOUND_MIN_ORIGINAL,
            device='cpu'
        )

        dqn_rewards, _ = train_agent(env, dqn_agent, "1. Baseline Transformed DQN", track_violations=False)
        results['training']['transformed_dqn'] = {
            'rewards': dqn_rewards,
            'total_reward': float(np.sum(dqn_rewards)),
            'mean_reward': float(np.mean(dqn_rewards)),
            'final_100_mean': float(np.mean(dqn_rewards[-100:])),
            'final_100_std': float(np.std(dqn_rewards[-100:])),
            'violations': None
        }
        save_intermediate_results(results)

    # ===== 2. Static QBound + Transformed DQN =====
    print("\n" + "=" * 80)
    print("METHOD 2: Static QBound + Transformed DQN")
    print("=" * 80)

    if is_method_completed(results, 'transformed_static_qbound_dqn'):
        print("⏭️  Already completed, skipping...")
    else:
        static_dqn_agent = TransformedDQNAgent(
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
            qclip_max_original=QBOUND_MAX_ORIGINAL,
            qclip_min_original=QBOUND_MIN_ORIGINAL,
            device='cpu'
        )

        static_dqn_rewards, static_dqn_violations = train_agent(env, static_dqn_agent, "2. Static QBound + Transformed DQN", track_violations=True)

        # Compute violation statistics
        valid_violations = [v for v in static_dqn_violations if v is not None]
        violation_summary = {
            'per_episode': valid_violations,
            'mean': {k: float(np.mean([v[k] for v in valid_violations])) for k in valid_violations[0].keys()} if valid_violations else {},
            'final_100': {k: float(np.mean([v[k] for v in valid_violations[-100:]])) for k in valid_violations[0].keys()} if valid_violations else {}
        }

        results['training']['transformed_static_qbound_dqn'] = {
            'rewards': static_dqn_rewards,
            'total_reward': float(np.sum(static_dqn_rewards)),
            'mean_reward': float(np.mean(static_dqn_rewards)),
            'final_100_mean': float(np.mean(static_dqn_rewards[-100:])),
            'final_100_std': float(np.std(static_dqn_rewards[-100:])),
            'violations': violation_summary
        }
        save_intermediate_results(results)

    # ===== Analysis and Summary =====
    print("\n" + "=" * 80)
    print("MOUNTAINCAR TRANSFORMED DQN RESULTS (Final 100 Episodes)")
    print("=" * 80)
    print()
    print(f"{'Method':<40} {'Mean ± Std':<25} {'vs Baseline'}")
    print("-" * 80)

    baseline_mean = results['training']['transformed_dqn']['final_100_mean']
    baseline_std = results['training']['transformed_dqn']['final_100_std']

    methods = [
        ('Baseline Transformed DQN', 'transformed_dqn'),
        ('Static QBound + Transformed DQN', 'transformed_static_qbound_dqn')
    ]

    for method_name, method_key in methods:
        if method_key in results['training']:
            mean = results['training'][method_key]['final_100_mean']
            std = results['training'][method_key]['final_100_std']

            if method_key == 'transformed_dqn':
                print(f"{method_name:<40} {mean:>8.2f} ± {std:<8.2f}   +0.0%")
            else:
                improvement = ((mean - baseline_mean) / abs(baseline_mean)) * 100 if baseline_mean != 0 else 0
                print(f"{method_name:<40} {mean:>8.2f} ± {std:<8.2f}   {improvement:+.1f}%")

    # ===== Save Final Results =====
    timestamp = results['timestamp']
    final_output_file = f"/root/projects/QBound/results/mountaincar/dqn_transformed_seed{SEED}_{timestamp}.json"

    # Ensure directory exists
    os.makedirs(os.path.dirname(final_output_file), exist_ok=True)

    with open(final_output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to: {final_output_file}")

    # Delete progress file
    if os.path.exists(RESULTS_FILE):
        os.remove(RESULTS_FILE)

    print("\n" + "=" * 80)
    print("MountainCar Transformed DQN Experiment Complete!")
    print("=" * 80)
    print()
    print("📊 Key Insights:")
    print("  - Original Q bounds: [-86.60, 0]")
    print("  - Transformed Q bounds: [0, 86.60] (similar to CartPole)")
    print("  - Tests if positive Q-value range fixes QBound degradation")


if __name__ == "__main__":
    main()
