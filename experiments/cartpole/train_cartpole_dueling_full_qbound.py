"""
CartPole-v1: Dueling DQN 6-Way Comparison - Time-Step Dependent Rewards
Organization: Time-step Dependent Rewards

Validates that QBound generalizes to architecturally different DQN variants
in environments with dense, time-step dependent reward structure.

Architecture: Dueling DQN with separate value V(s) and advantage A(s,a) streams
Environment: CartPole-v1 (discrete actions, dense positive rewards per step)
Reward Structure: +1 per time step (time-step dependent)

Comparison:
1. Baseline Dueling DQN - No QBound, no Double-Q
2. Static QBound + Dueling DQN - Q ∈ [0, 99.34]
3. Dynamic QBound + Dueling DQN - Q ∈ [0, Q_max(t)] with step-aware bounds
4. Baseline Double Dueling DQN - No QBound, with Double-Q
5. Static QBound + Double Dueling DQN - Q ∈ [0, 99.34] + Double-Q
6. Dynamic QBound + Double Dueling DQN - Q ∈ [0, Q_max(t)] + Double-Q

Purpose: Demonstrate QBound works with architecturally different networks (not just standard DQN)
Note: CartPole has time-step dependent rewards, so dynamic QBound is applicable (Rule 3).
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
from dueling_dqn_agent import DuelingDQNAgent
from tqdm import tqdm

# Parse command line arguments
parser = argparse.ArgumentParser(description='CartPole Dueling DQN 6-way comparison with QBound (Time-step Dependent)')
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility (default: 42)')
args = parser.parse_args()

# Reproducibility
SEED = args.seed
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

# Results file for crash recovery (include seed to avoid conflicts with parallel runs)
RESULTS_FILE = f"/root/projects/QBound/results/cartpole/dueling_full_qbound_seed{SEED}_in_progress.json"

# Environment parameters
ENV_NAME = "CartPole-v1"
MAX_EPISODES = 500
MAX_STEPS = 500
EVAL_EPISODES = 10

# Shared hyperparameters (same as standard DQN for fair comparison)
LR = 0.001
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
BATCH_SIZE = 64
BUFFER_SIZE = 10000
TARGET_UPDATE_FREQ = 100

# QBound parameters (same as standard CartPole experiments)
# Q_max = (1 - γ^H) / (1 - γ) = (1 - 0.99^500) / (1 - 0.99) ≈ 99.34
QBOUND_MIN = 0.0
QBOUND_MAX = 99.34


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
            # Select action
            action = agent.select_action(state, eval_mode=False)

            # Take step
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

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

        # Track best performance
        if episode_reward > best_reward:
            best_reward = episode_reward

        # Progress update
        if (episode + 1) % 100 == 0:
            recent_avg_reward = np.mean(episode_rewards[-100:])
            recent_avg_steps = np.mean(episode_steps[-100:])
            progress_msg = f"  Episode {episode + 1}/{max_episodes} - Avg reward: {recent_avg_reward:.2f}, " \
                          f"Avg steps: {recent_avg_steps:.1f}, Best: {best_reward:.2f}"

            if track_violations and episode_violations and episode_violations[-1] is not None:
                recent_violations = [v for v in episode_violations[-100:] if v is not None]
                if recent_violations:
                    avg_violation_rate = np.mean([v['total_violation_rate'] for v in recent_violations])
                    progress_msg += f", Violations: {avg_violation_rate:.1%}"

            print(progress_msg)

    return episode_rewards, episode_steps, episode_violations


def main():
    print("=" * 80)
    print("CartPole-v1: Dueling DQN 6-Way Comparison (Time-Step Dependent)")
    print("Organization: Time-step Dependent Rewards")
    print("Validating QBound Generalization to Different Architectures")
    print("=" * 80)
    print("Architecture: Dueling DQN (separate V(s) and A(s,a) streams)")
    print("=" * 80)

    print(f"\nConfiguration:")
    print(f"  Environment: {ENV_NAME}")
    print(f"  Reward Structure: +1 per time step (time-step dependent)")
    print(f"  Architecture: Dueling DQN (value + advantage streams)")
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
            'experiment_type': 'time_step_dependent',
            'script_name': 'train_cartpole_dueling_full_qbound.py',
            'architecture': 'dueling_dqn',
            'config': {
                'env': ENV_NAME,
                'reward_structure': 'time_step_dependent',
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
                'seed': SEED,
                # Dynamic QBound parameters (for methods that use them)
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

    # Create environment
    env = gym.make(ENV_NAME)
    env.reset(seed=SEED)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # ===== 1. Baseline Dueling DQN =====
    print("\n" + "=" * 80)
    print("METHOD 1: Baseline Dueling DQN")
    print("=" * 80)

    if is_method_completed(results, 'dueling_dqn'):
        print("⏭️  Already completed, skipping...")
    else:
        dueling_dqn_agent = DuelingDQNAgent(
            state_dim=state_dim, action_dim=action_dim, learning_rate=LR,
            gamma=GAMMA, epsilon_start=EPSILON_START, epsilon_end=EPSILON_END,
            epsilon_decay=EPSILON_DECAY, buffer_size=BUFFER_SIZE,
            batch_size=BATCH_SIZE, target_update_freq=TARGET_UPDATE_FREQ,
            use_qclip=False, use_double_dqn=False, device='cpu'
        )
        dueling_rewards, dueling_steps, _ = train_agent(env, dueling_dqn_agent, "1. Baseline Dueling DQN",
                                                        use_step_aware=False, track_violations=False)
        results['training']['dueling_dqn'] = {
            'rewards': dueling_rewards, 'steps': dueling_steps,
            'total_reward': float(np.sum(dueling_rewards)),
            'mean_reward': float(np.mean(dueling_rewards)),
            'mean_steps': float(np.mean(dueling_steps)),
            'final_100_mean': float(np.mean(dueling_rewards[-100:])),
            'final_100_std': float(np.std(dueling_rewards[-100:])),
            'final_100_max': float(np.max(dueling_rewards[-100:])),
            'violations': None  # No QBound
        }
        save_intermediate_results(results)

    # ===== 2. Static QBound + Dueling DQN =====
    print("\n" + "=" * 80)
    print("METHOD 2: Static QBound + Dueling DQN")
    print("=" * 80)

    if is_method_completed(results, 'static_qbound_dueling_dqn'):
        print("⏭️  Already completed, skipping...")
    else:
        static_qbound_dueling_agent = DuelingDQNAgent(
            state_dim=state_dim, action_dim=action_dim, learning_rate=LR,
            gamma=GAMMA, epsilon_start=EPSILON_START, epsilon_end=EPSILON_END,
            epsilon_decay=EPSILON_DECAY, buffer_size=BUFFER_SIZE,
            batch_size=BATCH_SIZE, target_update_freq=TARGET_UPDATE_FREQ,
            use_qclip=True, qclip_min=QBOUND_MIN, qclip_max=QBOUND_MAX,
            use_double_dqn=False, device='cpu'
        )
        static_qbound_dueling_rewards, static_qbound_dueling_steps, static_violations = train_agent(
            env, static_qbound_dueling_agent, "2. Static QBound + Dueling DQN",
            use_step_aware=False, track_violations=True)

        # Compute violation statistics
        valid_violations = [v for v in static_violations if v is not None]
        violation_summary = {
            'per_episode': valid_violations,
            'mean': {k: float(np.mean([v[k] for v in valid_violations])) for k in valid_violations[0].keys()} if valid_violations else {},
            'final_100': {k: float(np.mean([v[k] for v in valid_violations[-100:]])) for k in valid_violations[0].keys()} if valid_violations else {}
        }

        results['training']['static_qbound_dueling_dqn'] = {
            'rewards': static_qbound_dueling_rewards, 'steps': static_qbound_dueling_steps,
            'total_reward': float(np.sum(static_qbound_dueling_rewards)),
            'mean_reward': float(np.mean(static_qbound_dueling_rewards)),
            'mean_steps': float(np.mean(static_qbound_dueling_steps)),
            'final_100_mean': float(np.mean(static_qbound_dueling_rewards[-100:])),
            'final_100_std': float(np.std(static_qbound_dueling_rewards[-100:])),
            'final_100_max': float(np.max(static_qbound_dueling_rewards[-100:])),
            'violations': violation_summary
        }
        save_intermediate_results(results)

    # ===== 3. Dynamic QBound + Dueling DQN =====
    print("\n" + "=" * 80)
    print("METHOD 3: Dynamic QBound + Dueling DQN")
    print("=" * 80)

    if is_method_completed(results, 'dynamic_qbound_dueling_dqn'):
        print("⏭️  Already completed, skipping...")
    else:
        dynamic_qbound_dueling_agent = DuelingDQNAgent(
            state_dim=state_dim, action_dim=action_dim, learning_rate=LR,
            gamma=GAMMA, epsilon_start=EPSILON_START, epsilon_end=EPSILON_END,
            epsilon_decay=EPSILON_DECAY, buffer_size=BUFFER_SIZE,
            batch_size=BATCH_SIZE, target_update_freq=TARGET_UPDATE_FREQ,
            use_qclip=True, qclip_min=QBOUND_MIN, qclip_max=QBOUND_MAX,
            use_step_aware_qbound=True, max_episode_steps=MAX_STEPS,
            step_reward=1.0, reward_is_negative=False,
            use_double_dqn=False, device='cpu'
        )
        dynamic_qbound_dueling_rewards, dynamic_qbound_dueling_steps, dynamic_violations = train_agent(
            env, dynamic_qbound_dueling_agent, "3. Dynamic QBound + Dueling DQN",
            use_step_aware=True, track_violations=True)

        # Compute violation statistics
        valid_violations = [v for v in dynamic_violations if v is not None]
        violation_summary = {
            'per_episode': valid_violations,
            'mean': {k: float(np.mean([v[k] for v in valid_violations])) for k in valid_violations[0].keys()} if valid_violations else {},
            'final_100': {k: float(np.mean([v[k] for v in valid_violations[-100:]])) for k in valid_violations[0].keys()} if valid_violations else {}
        }

        results['training']['dynamic_qbound_dueling_dqn'] = {
            'rewards': dynamic_qbound_dueling_rewards, 'steps': dynamic_qbound_dueling_steps,
            'total_reward': float(np.sum(dynamic_qbound_dueling_rewards)),
            'mean_reward': float(np.mean(dynamic_qbound_dueling_rewards)),
            'mean_steps': float(np.mean(dynamic_qbound_dueling_steps)),
            'final_100_mean': float(np.mean(dynamic_qbound_dueling_rewards[-100:])),
            'final_100_std': float(np.std(dynamic_qbound_dueling_rewards[-100:])),
            'final_100_max': float(np.max(dynamic_qbound_dueling_rewards[-100:])),
            'violations': violation_summary
        }
        save_intermediate_results(results)

    # ===== 4. Double Dueling DQN =====
    print("\n" + "=" * 80)
    print("METHOD 4: Double Dueling DQN")
    print("=" * 80)

    if is_method_completed(results, 'double_dueling_dqn'):
        print("⏭️  Already completed, skipping...")
    else:
        double_dueling_agent = DuelingDQNAgent(
            state_dim=state_dim, action_dim=action_dim, learning_rate=LR,
            gamma=GAMMA, epsilon_start=EPSILON_START, epsilon_end=EPSILON_END,
            epsilon_decay=EPSILON_DECAY, buffer_size=BUFFER_SIZE,
            batch_size=BATCH_SIZE, target_update_freq=TARGET_UPDATE_FREQ,
            use_qclip=False, use_double_dqn=True, device='cpu'
        )
        double_dueling_rewards, double_dueling_steps, _ = train_agent(
            env, double_dueling_agent, "4. Double Dueling DQN",
            use_step_aware=False, track_violations=False)
        results['training']['double_dueling_dqn'] = {
            'rewards': double_dueling_rewards, 'steps': double_dueling_steps,
            'total_reward': float(np.sum(double_dueling_rewards)),
            'mean_reward': float(np.mean(double_dueling_rewards)),
            'mean_steps': float(np.mean(double_dueling_steps)),
            'final_100_mean': float(np.mean(double_dueling_rewards[-100:])),
            'final_100_std': float(np.std(double_dueling_rewards[-100:])),
            'final_100_max': float(np.max(double_dueling_rewards[-100:])),
            'violations': None  # No QBound
        }
        save_intermediate_results(results)

    # ===== 5. Static QBound + Double Dueling DQN =====
    print("\n" + "=" * 80)
    print("METHOD 5: Static QBound + Double Dueling DQN")
    print("=" * 80)

    if is_method_completed(results, 'static_qbound_double_dueling_dqn'):
        print("⏭️  Already completed, skipping...")
    else:
        static_qbound_double_dueling_agent = DuelingDQNAgent(
            state_dim=state_dim, action_dim=action_dim, learning_rate=LR,
            gamma=GAMMA, epsilon_start=EPSILON_START, epsilon_end=EPSILON_END,
            epsilon_decay=EPSILON_DECAY, buffer_size=BUFFER_SIZE,
            batch_size=BATCH_SIZE, target_update_freq=TARGET_UPDATE_FREQ,
            use_qclip=True, qclip_min=QBOUND_MIN, qclip_max=QBOUND_MAX,
            use_double_dqn=True, device='cpu'
        )
        static_qbound_double_dueling_rewards, static_qbound_double_dueling_steps, static_double_violations = train_agent(
            env, static_qbound_double_dueling_agent, "5. Static QBound + Double Dueling DQN",
            use_step_aware=False, track_violations=True)

        # Compute violation statistics
        valid_violations = [v for v in static_double_violations if v is not None]
        violation_summary = {
            'per_episode': valid_violations,
            'mean': {k: float(np.mean([v[k] for v in valid_violations])) for k in valid_violations[0].keys()} if valid_violations else {},
            'final_100': {k: float(np.mean([v[k] for v in valid_violations[-100:]])) for k in valid_violations[0].keys()} if valid_violations else {}
        }

        results['training']['static_qbound_double_dueling_dqn'] = {
            'rewards': static_qbound_double_dueling_rewards, 'steps': static_qbound_double_dueling_steps,
            'total_reward': float(np.sum(static_qbound_double_dueling_rewards)),
            'mean_reward': float(np.mean(static_qbound_double_dueling_rewards)),
            'mean_steps': float(np.mean(static_qbound_double_dueling_steps)),
            'final_100_mean': float(np.mean(static_qbound_double_dueling_rewards[-100:])),
            'final_100_std': float(np.std(static_qbound_double_dueling_rewards[-100:])),
            'final_100_max': float(np.max(static_qbound_double_dueling_rewards[-100:])),
            'violations': violation_summary
        }
        save_intermediate_results(results)

    # ===== 6. Dynamic QBound + Double Dueling DQN =====
    print("\n" + "=" * 80)
    print("METHOD 6: Dynamic QBound + Double Dueling DQN")
    print("=" * 80)

    if is_method_completed(results, 'dynamic_qbound_double_dueling_dqn'):
        print("⏭️  Already completed, skipping...")
    else:
        dynamic_qbound_double_dueling_agent = DuelingDQNAgent(
            state_dim=state_dim, action_dim=action_dim, learning_rate=LR,
            gamma=GAMMA, epsilon_start=EPSILON_START, epsilon_end=EPSILON_END,
            epsilon_decay=EPSILON_DECAY, buffer_size=BUFFER_SIZE,
            batch_size=BATCH_SIZE, target_update_freq=TARGET_UPDATE_FREQ,
            use_qclip=True, qclip_min=QBOUND_MIN, qclip_max=QBOUND_MAX,
            use_step_aware_qbound=True, max_episode_steps=MAX_STEPS,
            step_reward=1.0, reward_is_negative=False,
            use_double_dqn=True, device='cpu'
        )
        dynamic_qbound_double_dueling_rewards, dynamic_qbound_double_dueling_steps, dynamic_double_violations = train_agent(
            env, dynamic_qbound_double_dueling_agent, "6. Dynamic QBound + Double Dueling DQN",
            use_step_aware=True, track_violations=True)

        # Compute violation statistics
        valid_violations = [v for v in dynamic_double_violations if v is not None]
        violation_summary = {
            'per_episode': valid_violations,
            'mean': {k: float(np.mean([v[k] for v in valid_violations])) for k in valid_violations[0].keys()} if valid_violations else {},
            'final_100': {k: float(np.mean([v[k] for v in valid_violations[-100:]])) for k in valid_violations[0].keys()} if valid_violations else {}
        }

        results['training']['dynamic_qbound_double_dueling_dqn'] = {
            'rewards': dynamic_qbound_double_dueling_rewards, 'steps': dynamic_qbound_double_dueling_steps,
            'total_reward': float(np.sum(dynamic_qbound_double_dueling_rewards)),
            'mean_reward': float(np.mean(dynamic_qbound_double_dueling_rewards)),
            'mean_steps': float(np.mean(dynamic_qbound_double_dueling_steps)),
            'final_100_mean': float(np.mean(dynamic_qbound_double_dueling_rewards[-100:])),
            'final_100_std': float(np.std(dynamic_qbound_double_dueling_rewards[-100:])),
            'final_100_max': float(np.max(dynamic_qbound_double_dueling_rewards[-100:])),
            'violations': violation_summary
        }
        save_intermediate_results(results)

    # ===== Analysis and Summary =====
    print("\n" + "=" * 80)
    print("DUELING DQN RESULTS SUMMARY (Final 100 Episodes)")
    print("=" * 80)

    methods = ['dueling_dqn', 'static_qbound_dueling_dqn', 'dynamic_qbound_dueling_dqn',
               'double_dueling_dqn', 'static_qbound_double_dueling_dqn', 'dynamic_qbound_double_dueling_dqn']
    labels = ['Baseline Dueling', 'Static QBound Dueling', 'Dynamic QBound Dueling',
              'Double Dueling', 'Static QBound+Double', 'Dynamic QBound+Double']

    print(f"\n{'Method':<30} {'Mean ± Std':<25} {'Max':<10} {'vs Baseline':<15}")
    print("-" * 80)

    baseline_mean = results['training']['dueling_dqn']['final_100_mean']

    for method, label in zip(methods, labels):
        data = results['training'][method]
        mean = data['final_100_mean']
        std = data['final_100_std']
        max_reward = data['final_100_max']

        improvement = ((mean - baseline_mean) / abs(baseline_mean)) * 100 if baseline_mean != 0 else 0
        improvement_str = f"{improvement:+.1f}%"

        print(f"{label:<30} {mean:>8.2f} ± {std:<8.2f} {max_reward:>8.2f}   {improvement_str:>10}")

    # Calculate success rates (reward > 475)
    print("\n" + "=" * 80)
    print("SUCCESS RATE (Reward > 475 in final 100 episodes)")
    print("=" * 80)

    for method, label in zip(methods, labels):
        rewards = results['training'][method]['rewards'][-100:]
        success_rate = sum(1 for r in rewards if r > 475) / len(rewards) * 100
        print(f"{label:<30} {success_rate:.1f}%")

    # ===== Save Final Results =====
    timestamp = results['timestamp']
    final_output_file = f"/root/projects/QBound/results/cartpole/dueling_full_qbound_seed{SEED}_{timestamp}.json"

    with open(final_output_file, 'w') as f:
        json.dump(results, f, indent=2)

    if os.path.exists(RESULTS_FILE):
        os.remove(RESULTS_FILE)

    print(f"\n✓ Results saved to: {final_output_file}")
    print("\n" + "=" * 80)
    print("Dueling DQN 6-Way Comparison Complete!")
    print("=" * 80)

    print("\n📊 Key Takeaways:")
    print("  - If QBound improves Dueling DQN, it demonstrates architectural generalization")
    print("  - Compare to standard DQN results to assess relative performance")
    print("  - Look for similar improvement patterns as standard DQN experiments")
    print("  - Dynamic QBound tests step-aware bounds on time-dependent rewards")


if __name__ == '__main__':
    main()
