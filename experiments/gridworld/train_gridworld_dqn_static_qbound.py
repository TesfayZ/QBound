"""
GridWorld: DQN Static QBound Comparison (4-Way)
Organization: Sparse/State-Dependent Rewards (Static QBound Only)

Environment: Custom GridWorld (10x10 grid)
Reward: +1 for reaching goal, 0 otherwise (sparse terminal reward)
Max steps: 100

WHY NO DYNAMIC QBOUND:
GridWorld has sparse terminal rewards (only +1 when reaching the goal).
Dynamic QBound only applies when rewards accumulate predictably with time steps.
Since rewards are sparse and not time-step dependent, only static bounds are appropriate.

Comparison:
1. Baseline DQN - No QBound, no Double-Q
2. Static QBound + DQN - Q ∈ [0, 1.0]
3. Baseline DDQN - No QBound, with Double-Q
4. Static QBound + DDQN - Q ∈ [0, 1.0] + Double-Q
"""

import sys
sys.path.insert(0, '/root/projects/QBound/src')

import numpy as np
import torch
import json
import random
import os
import argparse
from datetime import datetime
from environment import GridWorldEnv
from dqn_agent import DQNAgent
from double_dqn_agent import DoubleDQNAgent
from tqdm import tqdm

# Parse command line arguments
parser = argparse.ArgumentParser(description='GridWorld DQN with static QBound only')
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility (default: 42)')
args = parser.parse_args()

# Reproducibility
SEED = args.seed
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

# Results file for crash recovery
RESULTS_FILE = f"/root/projects/QBound/results/gridworld/dqn_static_qbound_seed{SEED}_in_progress.json"

# Environment parameters
ENV_NAME = "GridWorld"
MAX_EPISODES = 500
MAX_STEPS = 100
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

# QBound parameters for GridWorld (Sparse Terminal Reward)
# Reward: +1 for reaching goal (terminal state)
# Since reward is terminal, max Q-value is 1.0 (discounting doesn't accumulate)
QBOUND_MIN = 0.0
QBOUND_MAX = 1.0


def load_existing_results():
    """Load existing results if the experiment was interrupted"""
    if os.path.exists(RESULTS_FILE):
        print(f"\nFound existing results file: {RESULTS_FILE}")
        print("   Loading previous progress...")
        with open(RESULTS_FILE, 'r') as f:
            results = json.load(f)

        completed = [k for k in results.get('training', {}).keys()]
        if completed:
            print(f"   Already completed: {', '.join(completed)}")
        return results
    return None


def save_intermediate_results(results):
    """Save results after each method completes (crash recovery)"""
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"   Progress saved to: {RESULTS_FILE}")


def is_method_completed(results, method_name):
    """Check if a method has already been completed"""
    return method_name in results.get('training', {})


def train_agent(env, agent, agent_name, max_episodes=MAX_EPISODES, track_violations=False):
    """Train agent and return results with optional violation tracking"""
    print(f"\n>>> Training {agent_name}...")

    episode_rewards = []
    episode_steps = []
    episode_violations = [] if track_violations else None
    success_count = 0

    for episode in tqdm(range(max_episodes), desc=agent_name):
        # Incremental seeding for reproducibility
        env_seed = SEED + episode
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

            # Store transition (no step-aware for sparse terminal rewards)
            agent.store_transition(state, action, reward, next_state, done)

            # Train and collect violation stats
            loss, violations = agent.train_step()

            if track_violations and violations is not None:
                violation_stats_episode.append(violations)

            episode_reward += reward
            state = next_state
            step += 1

        if reward > 0:  # Success (reached goal)
            success_count += 1

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
            recent_success_rate = sum(1 for r in episode_rewards[-100:] if r > 0) / 100
            progress_msg = f"  Episode {episode + 1}/{max_episodes} - Avg reward: {recent_avg_reward:.3f}, " \
                          f"Success rate: {recent_success_rate:.2%}, ε={agent.epsilon:.3f}"

            if track_violations and episode_violations and episode_violations[-1] is not None:
                recent_violations = [v for v in episode_violations[-100:] if v is not None]
                if recent_violations:
                    avg_violation_rate = np.mean([v['total_violation_rate'] for v in recent_violations])
                    progress_msg += f", Violations: {avg_violation_rate:.1%}"

            print(progress_msg)

    success_rate = success_count / max_episodes
    return episode_rewards, episode_steps, episode_violations


def main():
    print("=" * 80)
    print("GridWorld: DQN Static QBound Comparison (4-Way)")
    print("Organization: Sparse/State-Dependent Rewards (Static QBound Only)")
    print("=" * 80)

    print(f"\nConfiguration:")
    print(f"  Environment: {ENV_NAME}")
    print(f"  Experiment Type: Sparse/State-Dependent Rewards")
    print(f"  Reward Structure: Sparse terminal reward")
    print(f"  Episodes: {MAX_EPISODES}")
    print(f"  Max steps per episode: {MAX_STEPS}")
    print(f"  Discount factor γ: {GAMMA}")
    print(f"  Learning rate: {LR}")
    print(f"  QBound range: [{QBOUND_MIN:.2f}, {QBOUND_MAX:.2f}]")
    print(f"  Random seed: {SEED}")
    print("  Note: No dynamic QBound (rewards are sparse terminal, not time-step dependent)")
    print("=" * 80)

    # Load existing results or create new
    results = load_existing_results()
    if results is None:
        print("\nStarting fresh experiment...")
        results = {
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'script_name': 'train_gridworld_dqn_static_qbound.py',
            'experiment_type': 'sparse_state_dependent',
            'reward_structure': 'sparse_terminal',
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
        print("   Resuming experiment...\n")

    # Create environment
    env = GridWorldEnv()
    state_dim = env.observation_space  # GridWorldEnv returns dimension directly
    action_dim = env.action_space  # GridWorldEnv returns dimension directly

    # ===== 1. Baseline DQN =====
    print("\n" + "=" * 80)
    print("METHOD 1: Baseline DQN (no QBound, no Double-Q)")
    print("=" * 80)

    if is_method_completed(results, 'baseline'):
        print("Already completed, skipping...")
    else:
        baseline_agent = DQNAgent(
            state_dim=state_dim, action_dim=action_dim, learning_rate=LR,
            gamma=GAMMA, epsilon_start=EPSILON_START, epsilon_end=EPSILON_END,
            epsilon_decay=EPSILON_DECAY, buffer_size=BUFFER_SIZE,
            batch_size=BATCH_SIZE, target_update_freq=TARGET_UPDATE_FREQ,
            use_qclip=False, device='cpu'
        )
        baseline_rewards, baseline_steps, _ = train_agent(
            env, baseline_agent, "1. Baseline DQN", track_violations=False)
        results['training']['baseline'] = {
            'rewards': baseline_rewards, 'steps': baseline_steps,
            'total_reward': float(np.sum(baseline_rewards)),
            'mean_reward': float(np.mean(baseline_rewards)),
            'mean_steps': float(np.mean(baseline_steps)),
            'success_rate': float(np.sum([1 for r in baseline_rewards if r > 0]) / len(baseline_rewards)),
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
        print("Already completed, skipping...")
    else:
        static_agent = DQNAgent(
            state_dim=state_dim, action_dim=action_dim, learning_rate=LR,
            gamma=GAMMA, epsilon_start=EPSILON_START, epsilon_end=EPSILON_END,
            epsilon_decay=EPSILON_DECAY, buffer_size=BUFFER_SIZE,
            batch_size=BATCH_SIZE, target_update_freq=TARGET_UPDATE_FREQ,
            use_qclip=True, qclip_min=QBOUND_MIN, qclip_max=QBOUND_MAX,
            device='cpu'
        )
        static_rewards, static_steps, static_violations = train_agent(
            env, static_agent, "2. Static QBound + DQN", track_violations=True)

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
            'success_rate': float(np.sum([1 for r in static_rewards if r > 0]) / len(static_rewards)),
            'final_100_mean': float(np.mean(static_rewards[-100:])),
            'final_100_std': float(np.std(static_rewards[-100:])),
            'qmax': float(QBOUND_MAX),
            'violations': violation_summary
        }
        save_intermediate_results(results)

    # ===== 3. Baseline DDQN =====
    print("\n" + "=" * 80)
    print("METHOD 3: Baseline DDQN (no QBound, with Double-Q)")
    print("=" * 80)

    if is_method_completed(results, 'baseline_ddqn'):
        print("Already completed, skipping...")
    else:
        baseline_ddqn_agent = DoubleDQNAgent(
            state_dim=state_dim, action_dim=action_dim, learning_rate=LR,
            gamma=GAMMA, epsilon_start=EPSILON_START, epsilon_end=EPSILON_END,
            epsilon_decay=EPSILON_DECAY, buffer_size=BUFFER_SIZE,
            batch_size=BATCH_SIZE, target_update_freq=TARGET_UPDATE_FREQ,
            use_qclip=False, device='cpu'
        )
        baseline_ddqn_rewards, baseline_ddqn_steps, _ = train_agent(
            env, baseline_ddqn_agent, "3. Baseline DDQN", track_violations=False)
        results['training']['baseline_ddqn'] = {
            'rewards': baseline_ddqn_rewards, 'steps': baseline_ddqn_steps,
            'total_reward': float(np.sum(baseline_ddqn_rewards)),
            'mean_reward': float(np.mean(baseline_ddqn_rewards)),
            'mean_steps': float(np.mean(baseline_ddqn_steps)),
            'success_rate': float(np.sum([1 for r in baseline_ddqn_rewards if r > 0]) / len(baseline_ddqn_rewards)),
            'final_100_mean': float(np.mean(baseline_ddqn_rewards[-100:])),
            'final_100_std': float(np.std(baseline_ddqn_rewards[-100:])),
            'violations': None  # No QBound
        }
        save_intermediate_results(results)

    # ===== 4. Static QBound + DDQN =====
    print("\n" + "=" * 80)
    print("METHOD 4: Static QBound + DDQN")
    print("=" * 80)

    if is_method_completed(results, 'static_qbound_ddqn'):
        print("Already completed, skipping...")
    else:
        static_ddqn_agent = DoubleDQNAgent(
            state_dim=state_dim, action_dim=action_dim, learning_rate=LR,
            gamma=GAMMA, epsilon_start=EPSILON_START, epsilon_end=EPSILON_END,
            epsilon_decay=EPSILON_DECAY, buffer_size=BUFFER_SIZE,
            batch_size=BATCH_SIZE, target_update_freq=TARGET_UPDATE_FREQ,
            use_qclip=True, qclip_min=QBOUND_MIN, qclip_max=QBOUND_MAX,
            device='cpu'
        )
        static_ddqn_rewards, static_ddqn_steps, static_ddqn_violations = train_agent(
            env, static_ddqn_agent, "4. Static QBound + DDQN", track_violations=True)

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
            'success_rate': float(np.sum([1 for r in static_ddqn_rewards if r > 0]) / len(static_ddqn_rewards)),
            'final_100_mean': float(np.mean(static_ddqn_rewards[-100:])),
            'final_100_std': float(np.std(static_ddqn_rewards[-100:])),
            'violations': violation_summary
        }
        save_intermediate_results(results)

    # ===== Analysis and Summary =====
    print("\n" + "=" * 80)
    print("GRIDWORLD RESULTS SUMMARY (Final 100 Episodes)")
    print("=" * 80)

    methods = ['baseline', 'static_qbound', 'baseline_ddqn', 'static_qbound_ddqn']
    labels = ['Baseline DQN', 'Static QBound+DQN', 'Baseline DDQN', 'Static QBound+DDQN']

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
    final_output_file = f"/root/projects/QBound/results/gridworld/dqn_static_qbound_seed{SEED}_{timestamp}.json"
    os.makedirs('/root/projects/QBound/results/gridworld', exist_ok=True)

    with open(final_output_file, 'w') as f:
        json.dump(results, f, indent=2)

    if os.path.exists(RESULTS_FILE):
        os.remove(RESULTS_FILE)

    print(f"\nResults saved to: {final_output_file}")
    print("\n" + "=" * 80)
    print("GridWorld DQN Static QBound 4-Way Comparison Complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
