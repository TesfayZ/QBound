"""
Pendulum-v1: DQN/DDQN 6-Way Comparison - Time-Step Dependent Rewards
Organization: Time-step Dependent Rewards

Tests whether DQN and Double DQN benefit from QBound on continuous control tasks
with dense time-step dependent negative rewards.

Environment: Pendulum-v1 (discrete actions via discretization, dense negative rewards)
Reward Structure: Dense negative rewards per time step (time-step dependent)

Comparison:
1. Standard DQN - No QBound, no Double-Q
2. Static QBound + DQN - Q ∈ [-1616, 0]
3. Dynamic QBound + DQN - Q ∈ [Q_min(t), 0] with step-aware bounds
4. Standard Double DQN - No QBound, with Double-Q
5. Static QBound + Double DQN - Q ∈ [-1616, 0] + Double-Q
6. Dynamic QBound + Double DQN - Q ∈ [Q_min(t), 0] + Double-Q

Note: This tests QBound on time-step dependent negative rewards, complementing
the CartPole experiments which test time-step dependent positive rewards.
"""

import sys
sys.path.insert(0, '/root/projects/QBound/src')

import gymnasium as gym
import numpy as np
import torch
import json
import random
import os
import argparse
from datetime import datetime
from dqn_agent import DQNAgent
from double_dqn_agent import DoubleDQNAgent
from tqdm import tqdm

# Parse command line arguments
parser = argparse.ArgumentParser(description='Pendulum DQN/DDQN 6-way comparison with QBound (Time-step Dependent)')
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility (default: 42)')
args = parser.parse_args()

# Reproducibility
SEED = args.seed
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

# Results file for crash recovery (include seed to avoid conflicts with parallel runs)
RESULTS_FILE = f"/root/projects/QBound/results/pendulum/dqn_full_qbound_seed{SEED}_in_progress.json"

# Environment parameters
ENV_NAME = "Pendulum-v1"
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

# Discretize continuous action space
ACTION_BINS = 11  # [-2, -1.6, -1.2, ..., 1.6, 2.0]
ACTION_SPACE = np.linspace(-2.0, 2.0, ACTION_BINS)

# QBound parameters for Pendulum
# Reward range: approximately [-16.27, 0]
# Q_max = 0 (best case: perfect balance from start)
# Q_min = -16.27 * sum(gamma^k for k in 0..199) = -16.27 * (1-gamma^200)/(1-gamma)
# With gamma=0.99, H=200: Q_min = -16.27 * 86.60 = -1409.33
QBOUND_MIN = -1409.3272174664303
QBOUND_MAX = 0.0


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
    best_reward = -np.inf

    for episode in tqdm(range(max_episodes), desc=agent_name):
        # Incremental seeding for reproducibility
        env_seed = SEED + episode
        state, _ = env.reset(seed=env_seed)

        episode_reward = 0
        done = False
        step = 0
        violation_stats_episode = [] if track_violations else None

        while not done and step < MAX_STEPS:
            # Select discrete action
            discrete_action = agent.select_action(state, eval_mode=False)

            # Convert to continuous action
            continuous_action = np.array([ACTION_SPACE[discrete_action]])

            # Take step
            next_state, reward, terminated, truncated, _ = env.step(continuous_action)
            done = terminated or truncated

            # Store transition (with step info if using step-aware bounds)
            if use_step_aware:
                agent.store_transition(state, discrete_action, reward, next_state, done, current_step=step)
            else:
                agent.store_transition(state, discrete_action, reward, next_state, done)

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

        # Track best performance
        if episode_reward > best_reward:
            best_reward = episode_reward

        # Progress update
        if (episode + 1) % 100 == 0:
            recent_avg = np.mean(episode_rewards[-100:])
            progress_msg = f"  Episode {episode + 1}/{max_episodes} - Recent avg: {recent_avg:.2f}, Best: {best_reward:.2f}, ε={agent.epsilon:.3f}"

            if track_violations and episode_violations and episode_violations[-1] is not None:
                recent_violations = [v for v in episode_violations[-100:] if v is not None]
                if recent_violations:
                    avg_violation_rate = np.mean([v['total_violation_rate'] for v in recent_violations])
                    progress_msg += f", Violations: {avg_violation_rate:.1%}"

            print(progress_msg)

    return episode_rewards, episode_steps, episode_violations


def main():
    print("=" * 80)
    print("Pendulum-v1: DQN/DDQN 6-Way Comparison (Time-Step Dependent)")
    print("Organization: Time-step Dependent Rewards")
    print("Testing: DQN + Double DQN + QBound variants")
    print("=" * 80)

    print(f"\nConfiguration:")
    print(f"  Environment: {ENV_NAME}")
    print(f"  Reward Structure: Dense negative per time step (time-step dependent)")
    print(f"  Episodes: {MAX_EPISODES}")
    print(f"  Max steps per episode: {MAX_STEPS}")
    print(f"  Discount factor γ: {GAMMA}")
    print(f"  Learning rate: {LR}")
    print(f"  Action discretization: {ACTION_BINS} bins")
    print(f"  QBound range: [{QBOUND_MIN:.2f}, {QBOUND_MAX:.2f}]")
    print("=" * 80)

    # Load existing results or create new
    results = load_existing_results()
    if results is None:
        print("\n🆕 Starting fresh experiment...")
        results = {
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'experiment_type': 'time_step_dependent',
            'script_name': 'train_pendulum_dqn_full_qbound.py',
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
                'action_bins': ACTION_BINS,
                'qbound_min': QBOUND_MIN,
                'qbound_max': QBOUND_MAX,
                'seed': SEED,
                # Dynamic QBound parameters (for methods that use them)
                'dynamic_qbound_params': {
                    'max_episode_steps': MAX_STEPS,
                    'step_reward': -16.27,
                    'reward_is_negative': True
                }
            },
            'training': {}
        }
    else:
        print("   ⏩ Resuming experiment...\n")

    # Create environment
    env = gym.make(ENV_NAME)
    env.reset(seed=SEED)

    state_dim = env.observation_space.shape[0]
    action_dim = ACTION_BINS  # Discretized actions

    # ===== 1. Standard DQN =====
    print("\n" + "=" * 80)
    print("METHOD 1: Standard DQN")
    print("=" * 80)

    if is_method_completed(results, 'dqn'):
        print("⏭️  Already completed, skipping...")
    else:
        dqn_agent = DQNAgent(
            state_dim=state_dim, action_dim=action_dim, learning_rate=LR,
            gamma=GAMMA, epsilon_start=EPSILON_START, epsilon_end=EPSILON_END,
            epsilon_decay=EPSILON_DECAY, buffer_size=BUFFER_SIZE,
            batch_size=BATCH_SIZE, target_update_freq=TARGET_UPDATE_FREQ,
            use_qclip=False, device='cpu'
        )
        dqn_rewards, dqn_steps, _ = train_agent(
            env, dqn_agent, "1. Standard DQN", use_step_aware=False, track_violations=False)
        results['training']['dqn'] = {
            'rewards': dqn_rewards,
            'steps': dqn_steps,
            'total_reward': float(np.sum(dqn_rewards)),
            'mean_reward': float(np.mean(dqn_rewards)),
            'mean_steps': float(np.mean(dqn_steps)),
            'final_100_mean': float(np.mean(dqn_rewards[-100:])),
            'final_100_std': float(np.std(dqn_rewards[-100:])),
            'violations': None  # No QBound
        }
        save_intermediate_results(results)

    # ===== 2. Static QBound + DQN =====
    print("\n" + "=" * 80)
    print("METHOD 2: Static QBound + DQN")
    print("=" * 80)

    if is_method_completed(results, 'static_qbound_dqn'):
        print("⏭️  Already completed, skipping...")
    else:
        static_qbound_dqn_agent = DQNAgent(
            state_dim=state_dim, action_dim=action_dim, learning_rate=LR,
            gamma=GAMMA, epsilon_start=EPSILON_START, epsilon_end=EPSILON_END,
            epsilon_decay=EPSILON_DECAY, buffer_size=BUFFER_SIZE,
            batch_size=BATCH_SIZE, target_update_freq=TARGET_UPDATE_FREQ,
            use_qclip=True, qclip_min=QBOUND_MIN, qclip_max=QBOUND_MAX,
            device='cpu'
        )
        static_qbound_dqn_rewards, static_qbound_dqn_steps, static_qbound_dqn_violations = train_agent(
            env, static_qbound_dqn_agent, "2. Static QBound + DQN", use_step_aware=False, track_violations=True)

        # Compute violation statistics
        valid_violations = [v for v in static_qbound_dqn_violations if v is not None]
        violation_summary = {
            'per_episode': valid_violations,
            'mean': {k: float(np.mean([v[k] for v in valid_violations])) for k in valid_violations[0].keys()} if valid_violations else {},
            'final_100': {k: float(np.mean([v[k] for v in valid_violations[-100:]])) for k in valid_violations[0].keys()} if valid_violations else {}
        }

        results['training']['static_qbound_dqn'] = {
            'rewards': static_qbound_dqn_rewards,
            'steps': static_qbound_dqn_steps,
            'total_reward': float(np.sum(static_qbound_dqn_rewards)),
            'mean_reward': float(np.mean(static_qbound_dqn_rewards)),
            'mean_steps': float(np.mean(static_qbound_dqn_steps)),
            'final_100_mean': float(np.mean(static_qbound_dqn_rewards[-100:])),
            'final_100_std': float(np.std(static_qbound_dqn_rewards[-100:])),
            'violations': violation_summary
        }
        save_intermediate_results(results)

    # ===== 3. Dynamic QBound + DQN =====
    print("\n" + "=" * 80)
    print("METHOD 3: Dynamic QBound + DQN")
    print("=" * 80)

    if is_method_completed(results, 'dynamic_qbound_dqn'):
        print("⏭️  Already completed, skipping...")
    else:
        dynamic_qbound_dqn_agent = DQNAgent(
            state_dim=state_dim, action_dim=action_dim, learning_rate=LR,
            gamma=GAMMA, epsilon_start=EPSILON_START, epsilon_end=EPSILON_END,
            epsilon_decay=EPSILON_DECAY, buffer_size=BUFFER_SIZE,
            batch_size=BATCH_SIZE, target_update_freq=TARGET_UPDATE_FREQ,
            use_qclip=True, qclip_min=QBOUND_MIN, qclip_max=QBOUND_MAX,
            use_step_aware_qbound=True, max_episode_steps=MAX_STEPS,
            step_reward=-16.27, reward_is_negative=True,
            device='cpu'
        )
        dynamic_qbound_dqn_rewards, dynamic_qbound_dqn_steps, dynamic_qbound_dqn_violations = train_agent(
            env, dynamic_qbound_dqn_agent, "3. Dynamic QBound + DQN", use_step_aware=True, track_violations=True)

        # Compute violation statistics
        valid_violations = [v for v in dynamic_qbound_dqn_violations if v is not None]
        violation_summary = {
            'per_episode': valid_violations,
            'mean': {k: float(np.mean([v[k] for v in valid_violations])) for k in valid_violations[0].keys()} if valid_violations else {},
            'final_100': {k: float(np.mean([v[k] for v in valid_violations[-100:]])) for k in valid_violations[0].keys()} if valid_violations else {}
        }

        results['training']['dynamic_qbound_dqn'] = {
            'rewards': dynamic_qbound_dqn_rewards,
            'steps': dynamic_qbound_dqn_steps,
            'total_reward': float(np.sum(dynamic_qbound_dqn_rewards)),
            'mean_reward': float(np.mean(dynamic_qbound_dqn_rewards)),
            'mean_steps': float(np.mean(dynamic_qbound_dqn_steps)),
            'final_100_mean': float(np.mean(dynamic_qbound_dqn_rewards[-100:])),
            'final_100_std': float(np.std(dynamic_qbound_dqn_rewards[-100:])),
            'violations': violation_summary
        }
        save_intermediate_results(results)

    # ===== 4. Double DQN =====
    print("\n" + "=" * 80)
    print("METHOD 4: Double DQN")
    print("=" * 80)

    if is_method_completed(results, 'double_dqn'):
        print("⏭️  Already completed, skipping...")
    else:
        double_dqn_agent = DoubleDQNAgent(
            state_dim=state_dim, action_dim=action_dim, learning_rate=LR,
            gamma=GAMMA, epsilon_start=EPSILON_START, epsilon_end=EPSILON_END,
            epsilon_decay=EPSILON_DECAY, buffer_size=BUFFER_SIZE,
            batch_size=BATCH_SIZE, target_update_freq=TARGET_UPDATE_FREQ,
            use_qclip=False, device='cpu'
        )
        double_dqn_rewards, double_dqn_steps, _ = train_agent(
            env, double_dqn_agent, "4. Double DQN", use_step_aware=False, track_violations=False)
        results['training']['double_dqn'] = {
            'rewards': double_dqn_rewards,
            'steps': double_dqn_steps,
            'total_reward': float(np.sum(double_dqn_rewards)),
            'mean_reward': float(np.mean(double_dqn_rewards)),
            'mean_steps': float(np.mean(double_dqn_steps)),
            'final_100_mean': float(np.mean(double_dqn_rewards[-100:])),
            'final_100_std': float(np.std(double_dqn_rewards[-100:])),
            'violations': None  # No QBound
        }
        save_intermediate_results(results)

    # ===== 5. Static QBound + Double DQN =====
    print("\n" + "=" * 80)
    print("METHOD 5: Static QBound + Double DQN")
    print("=" * 80)

    if is_method_completed(results, 'static_qbound_double_dqn'):
        print("⏭️  Already completed, skipping...")
    else:
        static_qbound_double_dqn_agent = DoubleDQNAgent(
            state_dim=state_dim, action_dim=action_dim, learning_rate=LR,
            gamma=GAMMA, epsilon_start=EPSILON_START, epsilon_end=EPSILON_END,
            epsilon_decay=EPSILON_DECAY, buffer_size=BUFFER_SIZE,
            batch_size=BATCH_SIZE, target_update_freq=TARGET_UPDATE_FREQ,
            use_qclip=True, qclip_min=QBOUND_MIN, qclip_max=QBOUND_MAX,
            device='cpu'
        )
        static_qbound_double_dqn_rewards, static_qbound_double_dqn_steps, static_qbound_double_dqn_violations = train_agent(
            env, static_qbound_double_dqn_agent, "5. Static QBound + Double DQN", use_step_aware=False, track_violations=True)

        # Compute violation statistics
        valid_violations = [v for v in static_qbound_double_dqn_violations if v is not None]
        violation_summary = {
            'per_episode': valid_violations,
            'mean': {k: float(np.mean([v[k] for v in valid_violations])) for k in valid_violations[0].keys()} if valid_violations else {},
            'final_100': {k: float(np.mean([v[k] for v in valid_violations[-100:]])) for k in valid_violations[0].keys()} if valid_violations else {}
        }

        results['training']['static_qbound_double_dqn'] = {
            'rewards': static_qbound_double_dqn_rewards,
            'steps': static_qbound_double_dqn_steps,
            'total_reward': float(np.sum(static_qbound_double_dqn_rewards)),
            'mean_reward': float(np.mean(static_qbound_double_dqn_rewards)),
            'mean_steps': float(np.mean(static_qbound_double_dqn_steps)),
            'final_100_mean': float(np.mean(static_qbound_double_dqn_rewards[-100:])),
            'final_100_std': float(np.std(static_qbound_double_dqn_rewards[-100:])),
            'violations': violation_summary
        }
        save_intermediate_results(results)

    # ===== 6. Dynamic QBound + Double DQN =====
    print("\n" + "=" * 80)
    print("METHOD 6: Dynamic QBound + Double DQN")
    print("=" * 80)

    if is_method_completed(results, 'dynamic_qbound_double_dqn'):
        print("⏭️  Already completed, skipping...")
    else:
        dynamic_qbound_double_dqn_agent = DoubleDQNAgent(
            state_dim=state_dim, action_dim=action_dim, learning_rate=LR,
            gamma=GAMMA, epsilon_start=EPSILON_START, epsilon_end=EPSILON_END,
            epsilon_decay=EPSILON_DECAY, buffer_size=BUFFER_SIZE,
            batch_size=BATCH_SIZE, target_update_freq=TARGET_UPDATE_FREQ,
            use_qclip=True, qclip_min=QBOUND_MIN, qclip_max=QBOUND_MAX,
            use_step_aware_qbound=True, max_episode_steps=MAX_STEPS,
            step_reward=-16.27, reward_is_negative=True,
            device='cpu'
        )
        dynamic_qbound_double_dqn_rewards, dynamic_qbound_double_dqn_steps, dynamic_qbound_double_dqn_violations = train_agent(
            env, dynamic_qbound_double_dqn_agent, "6. Dynamic QBound + Double DQN", use_step_aware=True, track_violations=True)

        # Compute violation statistics
        valid_violations = [v for v in dynamic_qbound_double_dqn_violations if v is not None]
        violation_summary = {
            'per_episode': valid_violations,
            'mean': {k: float(np.mean([v[k] for v in valid_violations])) for k in valid_violations[0].keys()} if valid_violations else {},
            'final_100': {k: float(np.mean([v[k] for v in valid_violations[-100:]])) for k in valid_violations[0].keys()} if valid_violations else {}
        }

        results['training']['dynamic_qbound_double_dqn'] = {
            'rewards': dynamic_qbound_double_dqn_rewards,
            'steps': dynamic_qbound_double_dqn_steps,
            'total_reward': float(np.sum(dynamic_qbound_double_dqn_rewards)),
            'mean_reward': float(np.mean(dynamic_qbound_double_dqn_rewards)),
            'mean_steps': float(np.mean(dynamic_qbound_double_dqn_steps)),
            'final_100_mean': float(np.mean(dynamic_qbound_double_dqn_rewards[-100:])),
            'final_100_std': float(np.std(dynamic_qbound_double_dqn_rewards[-100:])),
            'violations': violation_summary
        }
        save_intermediate_results(results)

    # ===== Analysis and Summary =====
    print("\n" + "=" * 80)
    print("DQN/DDQN RESULTS SUMMARY (Final 100 Episodes)")
    print("=" * 80)

    methods = ['dqn', 'static_qbound_dqn', 'dynamic_qbound_dqn',
               'double_dqn', 'static_qbound_double_dqn', 'dynamic_qbound_double_dqn']
    labels = ['Standard DQN', 'Static QBound DQN', 'Dynamic QBound DQN',
              'Double DQN', 'Static QBound+DDQN', 'Dynamic QBound+DDQN']

    print(f"\n{'Method':<30} {'Mean ± Std':<25} {'vs Baseline':<15}")
    print("-" * 80)

    baseline_mean = results['training']['dqn']['final_100_mean']

    for method, label in zip(methods, labels):
        data = results['training'][method]
        mean = data['final_100_mean']
        std = data['final_100_std']

        improvement = ((mean - baseline_mean) / abs(baseline_mean)) * 100 if baseline_mean != 0 else 0
        improvement_str = f"{improvement:+.1f}%"

        print(f"{label:<30} {mean:>8.2f} ± {std:<8.2f}   {improvement_str:>10}")

    # ===== Save Final Results =====
    timestamp = results['timestamp']
    final_output_file = f"/root/projects/QBound/results/pendulum/dqn_full_qbound_seed{SEED}_{timestamp}.json"

    with open(final_output_file, 'w') as f:
        json.dump(results, f, indent=2)

    if os.path.exists(RESULTS_FILE):
        os.remove(RESULTS_FILE)

    print(f"\n✓ Results saved to: {final_output_file}")
    print("\n" + "=" * 80)
    print("Pendulum DQN/DDQN 6-Way Comparison Complete!")
    print("=" * 80)

    print("\n📊 Key Takeaways:")
    print("  - Tests QBound on time-step dependent NEGATIVE rewards")
    print("  - Complements CartPole (positive rewards) experiments")
    print("  - Dynamic QBound uses step-aware bounds for negative rewards")
    print("  - Compares standard DQN vs Double DQN with QBound variants")

    env.close()


if __name__ == '__main__':
    main()
