"""
Pendulum-v1: PPO with Soft QBound - Time-Step Dependent Rewards
Organization: Time-step Dependent Rewards

Tests whether PPO benefits from Soft QBound on continuous control with
dense time-step dependent negative rewards.

Environment: Pendulum-v1 (continuous actions, dense negative rewards)
Reward Structure: Dense negative rewards per time step (time-step dependent)

Comparison:
1. Baseline PPO - No QBound
2. Static Soft QBound + PPO - Fixed bounds on V(s)

2 total methods (baseline + static QBound only)

Note: PPO bounds the value function V(s), NOT Q(s,a) like DDPG/TD3.
This should work better because V(s) doesn't affect policy gradient directly.

CRITICAL: Unlike DDPG/TD3 which may fail catastrophically with QBound,
PPO+QBound should work because it bounds V(s) not Q(s,a).

This is part of the time-step dependent reward experiments, testing
QBound on negative dense rewards (complementing CartPole's positive rewards).
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
from ppo_agent import PPOAgent
from ppo_qbound_agent import PPOQBoundAgent

# Parse command line arguments
parser = argparse.ArgumentParser(description='PPO on Pendulum-v1 with Static Soft QBound (Time-step Dependent)')
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility (default: 42)')
args = parser.parse_args()

# Reproducibility
SEED = args.seed
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

# Results file for crash recovery
RESULTS_FILE = f"/root/projects/QBound/results/pendulum/ppo_full_qbound_seed{SEED}_in_progress.json"

# Environment parameters
ENV_NAME = "Pendulum-v1"
MAX_EPISODES = 500
MAX_STEPS = 200
TRAJECTORY_LENGTH = 2048

# PPO hyperparameters
HIDDEN_SIZES = [64, 64]
LR_ACTOR = 3e-4
LR_CRITIC = 1e-3
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPSILON = 0.2
ENTROPY_COEF = 0.01
PPO_EPOCHS = 10
MINIBATCH_SIZE = 64

# QBound parameters for Pendulum
# V(s) bounds: Maximum episode return = -16.27 * 200 = -3254 (undiscounted)
# With γ=0.99, H=200: V_min = -16.27 * 86.60 = -1409.33 (geometric series)
# V_max = 0 (best case: perfect balance from start)
V_MIN = -1409.3272174664303
V_MAX = 0.0


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


def collect_trajectory(env, agent, max_steps=200):
    """Collect a single trajectory."""
    trajectory = []
    state, _ = env.reset()
    episode_reward = 0
    episode_length = 0

    for step in range(max_steps):
        action, log_prob = agent.get_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        
        trajectory.append((state, action, reward, next_state, done, log_prob.item(), step))

        episode_reward += reward
        episode_length += 1
        state = next_state

        if done:
            break

    return trajectory, episode_reward, episode_length


def train_agent(env, agent, agent_name, max_episodes=MAX_EPISODES, track_violations=False):
    """Train agent and return results with optional violation tracking"""
    print(f"\n{'='*60}")
    print(f"Training: {agent_name}")
    print(f"{'='*60}")

    episode_rewards = []
    episode_lengths = []
    episode_violations = [] if track_violations else None
    training_info_log = []

    trajectory_buffer = []
    total_steps = 0

    for episode in range(max_episodes):
        trajectory, episode_reward, episode_length = collect_trajectory(env, agent)

        trajectory_buffer.extend(trajectory)
        total_steps += episode_length

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        # Collect violations from update (PPO updates every trajectory_length steps)
        episode_violation = None
        if total_steps >= TRAJECTORY_LENGTH:
            training_info = agent.update(trajectory_buffer)
            training_info_log.append(training_info)

            # Extract violation metrics if tracking
            if track_violations and 'v_violation_rate' in training_info:
                episode_violation = {
                    'v_violation_rate': training_info['v_violation_rate'],
                    'v_clipped_fraction': training_info['v_clipped_fraction'],
                    'penalty_activation_rate': training_info['penalty_activation_rate']
                }

            trajectory_buffer = []
            total_steps = 0

        # Append violation (even if None) to maintain alignment with episodes
        if track_violations:
            episode_violations.append(episode_violation)

        if (episode + 1) % 25 == 0:
            recent_rewards = episode_rewards[-25:]
            avg_reward = np.mean(recent_rewards)
            std_reward = np.std(recent_rewards)

            progress_msg = f"Episode {episode + 1}/{max_episodes} | Avg Reward: {avg_reward:.2f} ± {std_reward:.2f}"

            # Show violation rate if tracking
            if track_violations and episode_violations:
                recent_violations = [v for v in episode_violations[-25:] if v is not None]
                if recent_violations:
                    avg_violation_rate = np.mean([v['v_violation_rate'] for v in recent_violations])
                    progress_msg += f" | Violations: {avg_violation_rate:.1%}"

            print(progress_msg)

    return episode_rewards, episode_violations


def main():
    print("="*60)
    print("Pendulum-v1: PPO 2-Way Comparison (Time-Step Dependent)")
    print("Organization: Time-step Dependent Rewards")
    print("Testing: Static Soft QBound on V(s) for continuous control")
    print("="*60)

    print(f"\nConfiguration:")
    print(f"  Environment: {ENV_NAME}")
    print(f"  Episodes: {MAX_EPISODES}")
    print(f"  Max steps per episode: {MAX_STEPS}")
    print(f"  Seed: {SEED}")
    print(f"  V(s) bounds: [{V_MIN:.2f}, {V_MAX:.2f}]")
    print("="*60)

    # Load existing results or create new
    results = load_existing_results()
    if results is None:
        print("\n🆕 Starting fresh experiment...")
        results = {
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'experiment_type': 'time_step_dependent',
            'script_name': 'train_pendulum_ppo_full_qbound.py',
            'config': {
                'env': ENV_NAME,
                'episodes': MAX_EPISODES,
                'max_steps': MAX_STEPS,
                'trajectory_length': TRAJECTORY_LENGTH,
                'gamma': GAMMA,
                'gae_lambda': GAE_LAMBDA,
                'v_min': V_MIN,
                'v_max': V_MAX,
                'seed': SEED
            },
            'training': {}
        }
    else:
        print("   ⏩ Resuming experiment...\n")

    # Create environment
    env = gym.make(ENV_NAME)
    env.reset(seed=SEED)

    # ===== 1. Baseline PPO =====
    print("\n" + "=" * 60)
    print("METHOD 1: Baseline PPO")
    print("=" * 60)

    if is_method_completed(results, 'baseline'):
        print("⏭️  Already completed, skipping...")
    else:
        baseline_agent = PPOAgent(
            state_dim=3,
            action_dim=1,
            continuous_action=True,
            hidden_sizes=HIDDEN_SIZES,
            lr_actor=LR_ACTOR,
            lr_critic=LR_CRITIC,
            gamma=GAMMA,
            gae_lambda=GAE_LAMBDA,
            clip_epsilon=CLIP_EPSILON,
            entropy_coef=ENTROPY_COEF,
            ppo_epochs=PPO_EPOCHS,
            minibatch_size=MINIBATCH_SIZE,
            device='cpu'
        )

        baseline_rewards, _ = train_agent(env, baseline_agent, "Baseline PPO", track_violations=False)

        results['training']['baseline'] = {
            'rewards': baseline_rewards,
            'final_100_mean': float(np.mean(baseline_rewards[-100:])),
            'final_100_std': float(np.std(baseline_rewards[-100:])),
            'max': float(np.max(baseline_rewards[-100:])),
            'min': float(np.min(baseline_rewards[-100:])),
            'violations': None
        }
        save_intermediate_results(results)

    # ===== 2. Architectural QBound + PPO (Negative Softplus) =====
    print("\n" + "=" * 60)
    print("METHOD 2: Architectural QBound + PPO (Negative Softplus Activation)")
    print("=" * 60)
    print("NOTE: Replaces algorithmic clipping with activation function for negative rewards")
    print("      Uses -softplus(logits) to enforce V ≤ 0 naturally")

    if is_method_completed(results, 'architectural_qbound_ppo'):
        print("⏭️  Already completed, skipping...")
    else:
        architectural_agent = PPOAgent(
            state_dim=3,
            action_dim=1,
            continuous_action=True,
            hidden_sizes=HIDDEN_SIZES,
            lr_actor=LR_ACTOR,
            lr_critic=LR_CRITIC,
            gamma=GAMMA,
            gae_lambda=GAE_LAMBDA,
            clip_epsilon=CLIP_EPSILON,
            entropy_coef=ENTROPY_COEF,
            ppo_epochs=PPO_EPOCHS,
            minibatch_size=MINIBATCH_SIZE,
            use_architectural_qbound=True,  # Use activation function instead of clipping
            device='cpu'
        )

        architectural_rewards, architectural_violations = train_agent(env, architectural_agent, "PPO + Architectural QBound", track_violations=False)  # No violations by construction

        results['training']['architectural_qbound_ppo'] = {
            'rewards': architectural_rewards,
            'final_100_mean': float(np.mean(architectural_rewards[-100:])),
            'final_100_std': float(np.std(architectural_rewards[-100:])),
            'max': float(np.max(architectural_rewards[-100:])),
            'min': float(np.min(architectural_rewards[-100:])),
            'note': 'Architectural bound via -softplus(logits), 0% violations by construction'
        }
        save_intermediate_results(results)

    # ===== Analysis and Summary =====
    print("\n" + "=" * 60)
    print("FINAL RESULTS (Last 100 Episodes)")
    print("=" * 60)
    print()
    print(f"{'Method':<30} {'Mean ± Std':<25} {'vs Baseline'}")
    print("-" * 60)

    baseline_mean = results['training']['baseline']['final_100_mean']
    baseline_std = results['training']['baseline']['final_100_std']

    methods = [
        ('Baseline PPO', 'baseline'),
        ('Static Soft QBound + PPO', 'static_soft_qbound')
    ]

    for method_name, method_key in methods:
        if method_key in results['training']:
            mean = results['training'][method_key]['final_100_mean']
            std = results['training'][method_key]['final_100_std']

            if method_key == 'baseline':
                print(f"{method_name:<30} {mean:>8.2f} ± {std:<8.2f}   +0.0%")
            else:
                improvement = ((mean - baseline_mean) / abs(baseline_mean)) * 100 if baseline_mean != 0 else 0
                print(f"{method_name:<30} {mean:>8.2f} ± {std:<8.2f}   {improvement:+.1f}%")

    # Check for catastrophic failure warning
    for method_name, method_key in methods[1:]:  # Skip baseline
        if method_key in results['training']:
            mean = results['training'][method_key]['final_100_mean']
            improvement = ((mean - baseline_mean) / abs(baseline_mean)) * 100 if baseline_mean != 0 else 0

            if improvement < -50:
                print(f"\n  ⚠️  WARNING: {method_name} shows CATASTROPHIC FAILURE (>{abs(improvement):.0f}% worse)")

    # ===== Save Final Results =====
    timestamp = results['timestamp']
    final_output_file = f"/root/projects/QBound/results/pendulum/ppo_full_qbound_seed{SEED}_{timestamp}.json"

    with open(final_output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to: {final_output_file}")

    # Delete progress file
    if os.path.exists(RESULTS_FILE):
        os.remove(RESULTS_FILE)

    print("\n" + "=" * 60)
    print("📊 Key Takeaways:")
    print("  - Tests Static Soft QBound on PPO (bounds V(s) not Q(s,a))")
    print("  - Environment has time-step dependent NEGATIVE rewards")
    print("  - Compares baseline vs static QBound only")
    print("  - PPO should work better than DDPG/TD3 with QBound")
    print("=" * 60)


if __name__ == "__main__":
    main()
