"""
Pendulum-v1 DDPG with TRANSFORMED Q-Values (Negative → Positive)

EXPERIMENT GOAL: Test if transforming to positive Q-value range improves
QBound performance even further on Pendulum.

CONTEXT: Original Pendulum DDPG showed QBound WORKING (not failing):
- Baseline: -391.29 → QBound: -150.46 (significant improvement!)
- This experiment tests if positive transformation helps even more

TRANSFORMATION:
- Original: Q ∈ [-1409.33, 0] (negative reward environment)
- Transformed: Q ∈ [0, 1409.33] (shifted to positive range)
- Method: Add abs(Q_min) = 1409.33 to all Q-values

Environment: Pendulum-v1 (continuous actions, dense negative rewards)
Reward Type: Dense negative rewards per time step
Max steps: 200

Methods Tested (2 total):
1. Baseline Transformed DDPG - No QBound
2. QBound + Transformed DDPG

Expected Outcome:
- Compare against original Pendulum DDPG results
- If further improvement → Transformation enhances QBound
- If similar → QBound already working optimally in negative space
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
from ddpg_agent_transformed import TransformedDDPGAgent
from tqdm import tqdm

# Parse command line arguments
parser = argparse.ArgumentParser(description='Pendulum DDPG with Transformed Q-values')
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility (default: 42)')
args = parser.parse_args()

# Reproducibility
SEED = args.seed
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

# Results file for crash recovery
RESULTS_FILE = f"/root/projects/QBound/results/pendulum/ddpg_transformed_seed{SEED}_in_progress.json"

# Environment parameters
ENV_NAME = "Pendulum-v1"
MAX_EPISODES = 500
MAX_STEPS = 200
EVAL_EPISODES = 10

# DDPG hyperparameters
LR_ACTOR = 0.001
LR_CRITIC = 0.001
GAMMA = 0.99
TAU = 0.005
BATCH_SIZE = 256
WARMUP_EPISODES = 10

# QBound parameters for Pendulum (ORIGINAL negative bounds)
# Original: Q ∈ [-1409.33, 0]
# Transformed: Q ∈ [0, 1409.33]
QBOUND_MIN_ORIGINAL = -1409.3272174664303
QBOUND_MAX_ORIGINAL = 0.0


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


def train_agent(env, agent, agent_name, max_episodes=MAX_EPISODES, warmup_episodes=WARMUP_EPISODES, track_violations=False):
    """Train agent and return results with optional violation tracking"""
    print(f"\n>>> Training {agent_name}...")

    episode_rewards = []
    episode_violations = [] if track_violations else None

    for episode in tqdm(range(max_episodes), desc=agent_name):
        state, _ = env.reset(seed=SEED + episode)
        episode_reward = 0
        violation_stats_episode = [] if track_violations else None

        for step in range(MAX_STEPS):
            # Select action (with exploration noise during training)
            if episode < warmup_episodes:
                # Random warmup
                action = env.action_space.sample()
            else:
                action = agent.select_action(state, noise_scale=0.1, eval_mode=False)

            # Take step
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Store transition
            agent.store_transition(state, action, reward, next_state, done)

            # Train (after warmup)
            if episode >= warmup_episodes:
                critic_loss, actor_loss, violations = agent.train_step()
                if track_violations and violations is not None:
                    violation_stats_episode.append(violations)

            state = next_state
            episode_reward += reward

            if done:
                break

        episode_rewards.append(episode_reward)

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
            progress_msg = f"  Episode {episode + 1}/{max_episodes} - Avg reward: {recent_avg_reward:.1f}"

            if track_violations and episode_violations and episode_violations[-1] is not None:
                recent_violations = [v for v in episode_violations[-100:] if v is not None]
                if recent_violations:
                    avg_violation_rate = np.mean([v['total_violation_rate'] for v in recent_violations])
                    progress_msg += f", Violations: {avg_violation_rate:.1%}"

            print(progress_msg)

    return episode_rewards, episode_violations


def main():
    print("=" * 80)
    print("Pendulum-v1: TRANSFORMED DDPG (Negative → Positive Q-Values)")
    print("Testing: Does positive Q-value range improve QBound further?")
    print("=" * 80)

    print(f"\nConfiguration:")
    print(f"  Environment: {ENV_NAME}")
    print(f"  Reward Structure: Dense negative (time-step dependent)")
    print(f"  Original Q bounds: [{QBOUND_MIN_ORIGINAL:.2f}, {QBOUND_MAX_ORIGINAL:.2f}]")
    print(f"  Transformed Q bounds: [0.0, {abs(QBOUND_MIN_ORIGINAL):.2f}]")
    print(f"  Episodes: {MAX_EPISODES}")
    print(f"  Max steps: {MAX_STEPS}")
    print(f"  Gamma: {GAMMA}")
    print(f"  Seed: {SEED}")
    print(f"\n  Note: Original Pendulum DDPG showed QBound WORKING")
    print(f"        This tests if transformation improves it further")

    # Load or initialize results
    results = load_existing_results()
    if results is None:
        print("\n🆕 Starting fresh experiment...")
        results = {
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'experiment_type': 'transformed_qvalues',
            'script_name': 'train_pendulum_ddpg_transformed.py',
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
                'lr_actor': LR_ACTOR,
                'lr_critic': LR_CRITIC,
                'tau': TAU,
                'batch_size': BATCH_SIZE,
                'warmup_episodes': WARMUP_EPISODES,
                'seed': SEED
            },
            'training': {}
        }
    else:
        print("   ⏩ Resuming experiment...\n")

    # Create environment
    env = gym.make(ENV_NAME)

    # ===== 1. Baseline Transformed DDPG =====
    print("\n" + "=" * 80)
    print("METHOD 1: Baseline Transformed DDPG (No QBound)")
    print("=" * 80)

    if is_method_completed(results, 'transformed_ddpg'):
        print("⏭️  Already completed, skipping...")
    else:
        ddpg_agent = TransformedDDPGAgent(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            max_action=float(env.action_space.high[0]),
            learning_rate_actor=LR_ACTOR,
            learning_rate_critic=LR_CRITIC,
            gamma=GAMMA,
            tau=TAU,
            batch_size=BATCH_SIZE,
            use_qbound=False,
            qbound_min_original=QBOUND_MIN_ORIGINAL,
            qbound_max_original=QBOUND_MAX_ORIGINAL,
            device='cpu'
        )

        ddpg_rewards, _ = train_agent(env, ddpg_agent, "1. Baseline Transformed DDPG", track_violations=False)
        results['training']['transformed_ddpg'] = {
            'rewards': ddpg_rewards,
            'total_reward': float(np.sum(ddpg_rewards)),
            'mean_reward': float(np.mean(ddpg_rewards)),
            'final_100_mean': float(np.mean(ddpg_rewards[-100:])),
            'final_100_std': float(np.std(ddpg_rewards[-100:])),
            'violations': None
        }
        save_intermediate_results(results)

    # ===== 2. QBound + Transformed DDPG =====
    print("\n" + "=" * 80)
    print("METHOD 2: QBound + Transformed DDPG")
    print("=" * 80)

    if is_method_completed(results, 'transformed_qbound_ddpg'):
        print("⏭️  Already completed, skipping...")
    else:
        qbound_ddpg_agent = TransformedDDPGAgent(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            max_action=float(env.action_space.high[0]),
            learning_rate_actor=LR_ACTOR,
            learning_rate_critic=LR_CRITIC,
            gamma=GAMMA,
            tau=TAU,
            batch_size=BATCH_SIZE,
            use_qbound=True,
            qbound_min_original=QBOUND_MIN_ORIGINAL,
            qbound_max_original=QBOUND_MAX_ORIGINAL,
            use_soft_clip=True,   # CRITICAL: Use soft clipping for DDPG
            soft_clip_beta=0.1,   # Steepness parameter (same as regular DDPG)
            device='cpu'
        )

        qbound_ddpg_rewards, qbound_ddpg_violations = train_agent(env, qbound_ddpg_agent, "2. QBound + Transformed DDPG", track_violations=True)

        # Compute violation statistics
        valid_violations = [v for v in qbound_ddpg_violations if v is not None]
        violation_summary = {
            'per_episode': valid_violations,
            'mean': {k: float(np.mean([v[k] for v in valid_violations])) for k in valid_violations[0].keys()} if valid_violations else {},
            'final_100': {k: float(np.mean([v[k] for v in valid_violations[-100:]])) for k in valid_violations[0].keys()} if valid_violations else {}
        }

        results['training']['transformed_qbound_ddpg'] = {
            'rewards': qbound_ddpg_rewards,
            'total_reward': float(np.sum(qbound_ddpg_rewards)),
            'mean_reward': float(np.mean(qbound_ddpg_rewards)),
            'final_100_mean': float(np.mean(qbound_ddpg_rewards[-100:])),
            'final_100_std': float(np.std(qbound_ddpg_rewards[-100:])),
            'violations': violation_summary
        }
        save_intermediate_results(results)

    # ===== Analysis and Summary =====
    print("\n" + "=" * 80)
    print("PENDULUM TRANSFORMED DDPG RESULTS (Final 100 Episodes)")
    print("=" * 80)
    print()
    print(f"{'Method':<40} {'Mean ± Std':<25} {'vs Baseline'}")
    print("-" * 80)

    baseline_mean = results['training']['transformed_ddpg']['final_100_mean']
    baseline_std = results['training']['transformed_ddpg']['final_100_std']

    methods = [
        ('Baseline Transformed DDPG', 'transformed_ddpg'),
        ('QBound + Transformed DDPG', 'transformed_qbound_ddpg')
    ]

    for method_name, method_key in methods:
        if method_key in results['training']:
            mean = results['training'][method_key]['final_100_mean']
            std = results['training'][method_key]['final_100_std']

            if method_key == 'transformed_ddpg':
                print(f"{method_name:<40} {mean:>8.2f} ± {std:<8.2f}   +0.0%")
            else:
                improvement = ((mean - baseline_mean) / abs(baseline_mean)) * 100 if baseline_mean != 0 else 0
                print(f"{method_name:<40} {mean:>8.2f} ± {std:<8.2f}   {improvement:+.1f}%")

    # ===== Save Final Results =====
    timestamp = results['timestamp']
    final_output_file = f"/root/projects/QBound/results/pendulum/ddpg_transformed_seed{SEED}_{timestamp}.json"

    # Ensure directory exists
    os.makedirs(os.path.dirname(final_output_file), exist_ok=True)

    with open(final_output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to: {final_output_file}")

    # Delete progress file
    if os.path.exists(RESULTS_FILE):
        os.remove(RESULTS_FILE)

    print("\n" + "=" * 80)
    print("Pendulum Transformed DDPG Experiment Complete!")
    print("=" * 80)
    print()
    print("📊 Key Insights:")
    print("  - Original Q bounds: [-1409.33, 0]")
    print("  - Transformed Q bounds: [0, 1409.33]")
    print("  - Original Pendulum DDPG: QBound already worked (-391 → -150)")
    print("  - This tests if positive range improves it further")


if __name__ == "__main__":
    main()
