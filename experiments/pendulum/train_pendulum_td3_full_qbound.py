"""
Pendulum-v1: TD3 with Soft QBound - Time-Step Dependent Rewards
Organization: Time-step Dependent Rewards

Tests whether TD3 benefits from Soft QBound on continuous control with
dense time-step dependent negative rewards.

Environment: Pendulum-v1 (continuous actions, dense negative rewards)
Reward Structure: Dense negative rewards per time step (time-step dependent)

Comparison:
1. Baseline TD3 - No QBound
2. Static Soft QBound + TD3 - Fixed bounds via softplus_clip

Note: ONLY uses Soft QBound (softplus_clip) because TD3 requires
smooth gradients for actor-critic learning. Hard clipping breaks the
gradient flow needed for continuous action optimization.

TD3 Features:
- Twin critics (clipped double-Q learning)
- Delayed policy updates
- Target policy smoothing

This is part of the time-step dependent reward experiments, testing
QBound on negative dense rewards (complementing CartPole's positive rewards).

Total Methods: 2
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
from td3_agent import TD3Agent
from tqdm import tqdm

# Parse command line arguments
parser = argparse.ArgumentParser(description='TD3 on Pendulum-v1 with Static Soft QBound (Time-step Dependent)')
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility (default: 42)')
args = parser.parse_args()

# Reproducibility
SEED = args.seed
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

# Results file for crash recovery
RESULTS_FILE = f"/root/projects/QBound/results/pendulum/td3_full_qbound_seed{SEED}_in_progress.json"

# Environment parameters
ENV_NAME = "Pendulum-v1"
MAX_EPISODES = 500
MAX_STEPS = 200
EVAL_EPISODES = 10

# TD3 hyperparameters
LR_ACTOR = 0.001
LR_CRITIC = 0.001
GAMMA = 0.99
TAU = 0.005
BATCH_SIZE = 256
WARMUP_EPISODES = 10
POLICY_NOISE = 0.2  # Target policy smoothing noise
NOISE_CLIP = 0.5    # Noise clipping
POLICY_FREQ = 2     # Delayed policy updates

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


def train_agent(env, agent, agent_name, max_episodes=MAX_EPISODES, track_violations=False):
    """Train agent and return results with optional violation tracking"""
    print(f"\n>>> Training {agent_name}...")

    episode_rewards = []
    episode_violations = [] if track_violations else None
    best_reward = -np.inf

    for episode in tqdm(range(max_episodes), desc=agent_name):
        state, _ = env.reset()
        agent.reset_noise()
        episode_reward = 0
        done = False
        step = 0
        violation_stats_episode = [] if track_violations else None

        while not done and step < MAX_STEPS:
            # Select action
            if episode < WARMUP_EPISODES:
                # Random exploration during warmup
                action = env.action_space.sample()
            else:
                action = agent.select_action(state, add_noise=True)

            # Take step
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Store transition
            agent.replay_buffer.push(state, action, reward, next_state, done, step)

            # Train if enough samples and collect violation stats
            if episode >= WARMUP_EPISODES:
                
                current_step = None
                critic_loss, actor_loss, violations = agent.train(batch_size=BATCH_SIZE, current_step=current_step)

                if track_violations and violations is not None:
                    violation_stats_episode.append(violations)

            episode_reward += reward
            state = next_state
            step += 1

        episode_rewards.append(episode_reward)

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
            progress_msg = f"  Episode {episode + 1}/{max_episodes} - Recent avg: {recent_avg:.2f}, Best: {best_reward:.2f}"

            if track_violations and episode_violations and episode_violations[-1] is not None:
                recent_violations = [v for v in episode_violations[-100:] if v is not None]
                if recent_violations:
                    avg_violation_rate = np.mean([v['total_violation_rate'] for v in recent_violations])
                    progress_msg += f", Violations: {avg_violation_rate:.1%}"

            print(progress_msg)

    return episode_rewards, episode_violations


def main():
    print("\n" + "=" * 80)
    print("TD3 + SOFT QBOUND EXPERIMENT ON PENDULUM-v1")
    print("Time-Step Dependent Rewards (Dense Negative)")
    print("=" * 80)

    print(f"\n  Environment: {ENV_NAME}")
    print(f"  Episodes: {MAX_EPISODES}")
    print(f"  Max steps per episode: {MAX_STEPS}")
    print(f"  Seed: {SEED}")
    print(f"  QBound range: [{QBOUND_MIN:.2f}, {QBOUND_MAX:.2f}]")
    print(f"  QBound type: SOFT (softplus_clip, preserves gradients)")
    print("=" * 80)

    # Load existing results or create new
    results = load_existing_results()
    if results is None:
        print("\n🆕 Starting fresh experiment...")
        results = {
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'experiment_type': 'time_step_dependent',
            'script_name': 'train_pendulum_td3_full_qbound.py',
            'config': {
                'env': ENV_NAME,
                'reward_structure': 'time_step_dependent_negative',
                'episodes': MAX_EPISODES,
                'max_steps': MAX_STEPS,
                'gamma': GAMMA,
                'lr_actor': LR_ACTOR,
                'lr_critic': LR_CRITIC,
                'tau': TAU,
                'policy_noise': POLICY_NOISE,
                'noise_clip': NOISE_CLIP,
                'policy_freq': POLICY_FREQ,
                'batch_size': BATCH_SIZE,
                'warmup_episodes': WARMUP_EPISODES,
                'qbound_min': QBOUND_MIN,
                'qbound_max': QBOUND_MAX,
                'seed': SEED,
                # Soft QBound parameters
                'soft_qbound_params': {
                    'soft_clip_beta': 0.1,
                    'method': 'softplus_clip'
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
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # ===== 1. Baseline TD3 =====
    print("\n" + "=" * 80)
    print("METHOD 1: Baseline TD3")
    print("=" * 80)

    if is_method_completed(results, 'baseline'):
        print("⏭️  Already completed, skipping...")
    else:
        baseline_agent = TD3Agent(
            state_dim=state_dim,
            action_dim=action_dim,
            max_action=max_action,
            lr_actor=LR_ACTOR,
            lr_critic=LR_CRITIC,
            gamma=GAMMA,
            tau=TAU,
            policy_noise=POLICY_NOISE,
            noise_clip=NOISE_CLIP,
            policy_freq=POLICY_FREQ,
            use_qbound=False,
            device='cpu'
        )

        baseline_rewards, _ = train_agent(env, baseline_agent, "1. Baseline TD3",
                                          track_violations=False)
        results['training']['baseline'] = {
            'rewards': baseline_rewards,
            'total_reward': float(np.sum(baseline_rewards)),
            'mean_reward': float(np.mean(baseline_rewards)),
            'final_100_mean': float(np.mean(baseline_rewards[-100:])),
            'final_100_std': float(np.std(baseline_rewards[-100:])),
            'violations': None  # No QBound
        }
        save_intermediate_results(results)

    # ===== 2. Static Soft QBound + TD3 =====
    print("\n" + "=" * 80)
    print("METHOD 2: Static Soft QBound + TD3")
    print("=" * 80)

    if is_method_completed(results, 'static_soft_qbound'):
        print("⏭️  Already completed, skipping...")
    else:
        static_qbound_agent = TD3Agent(
            state_dim=state_dim,
            action_dim=action_dim,
            max_action=max_action,
            lr_actor=LR_ACTOR,
            lr_critic=LR_CRITIC,
            gamma=GAMMA,
            tau=TAU,
            policy_noise=POLICY_NOISE,
            noise_clip=NOISE_CLIP,
            policy_freq=POLICY_FREQ,
            use_qbound=True,
            qbound_min=QBOUND_MIN,
            qbound_max=QBOUND_MAX,
            use_soft_clip=True,  # CRITICAL: Use soft clipping for continuous control
            soft_clip_beta=0.1,
            # Static bounds
            device='cpu'
        )

        static_rewards, static_violations = train_agent(env, static_qbound_agent, "2. Static Soft QBound + TD3",
                                                        track_violations=True)

        # Compute violation statistics
        valid_violations = [v for v in static_violations if v is not None]
        violation_summary = {
            'per_episode': valid_violations,
            'mean': {k: float(np.mean([v[k] for v in valid_violations])) for k in valid_violations[0].keys()} if valid_violations else {},
            'final_100': {k: float(np.mean([v[k] for v in valid_violations[-100:]])) for k in valid_violations[0].keys()} if valid_violations else {}
        }

        results['training']['static_soft_qbound'] = {
            'rewards': static_rewards,
            'total_reward': float(np.sum(static_rewards)),
            'mean_reward': float(np.mean(static_rewards)),
            'final_100_mean': float(np.mean(static_rewards[-100:])),
            'final_100_std': float(np.std(static_rewards[-100:])),
            'violations': violation_summary
        }
        save_intermediate_results(results)

    # ===== Analysis and Summary =====
    print("\n" + "=" * 80)
    print("TD3 RESULTS SUMMARY (Final 100 Episodes)")
    print("=" * 80)
    print()
    print(f"{'Method':<30} {'Mean ± Std':<25} {'vs Baseline'}")
    print("-" * 80)

    baseline_mean = results['training']['baseline']['final_100_mean']
    baseline_std = results['training']['baseline']['final_100_std']

    methods = [
        ('Baseline TD3', 'baseline'),
        ('Static Soft QBound + TD3', 'static_soft_qbound')
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

    # ===== Save Final Results =====
    timestamp = results['timestamp']
    final_output_file = f"/root/projects/QBound/results/pendulum/td3_full_qbound_seed{SEED}_{timestamp}.json"

    with open(final_output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to: {final_output_file}")

    # Delete progress file
    if os.path.exists(RESULTS_FILE):
        os.remove(RESULTS_FILE)

    print("\n" + "=" * 80)
    print("Pendulum TD3 2-Way Comparison Complete!")
    print("=" * 80)
    print()
    print("📊 Key Takeaways:")
    print("  - Tests Soft QBound (softplus_clip) on time-step dependent NEGATIVE rewards")
    print("  - Compares baseline vs static QBound for TD3")
    print("  - TD3 features: twin critics, delayed updates, target policy smoothing")
    print("  - Soft clipping preserves gradients needed for policy optimization")


if __name__ == "__main__":
    main()
