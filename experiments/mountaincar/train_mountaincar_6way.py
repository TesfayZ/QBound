"""
MountainCar-v0: 6-Way Comprehensive Comparison
Testing QBound on Sparse Negative Reward with Dynamic Bounds

Environment: MountainCar-v0 (discrete actions, sparse negative rewards)
Reward: -1 per step until goal reached
This tests dynamic QBound that INCREASES over time (opposite of dense positive rewards)

Comparison:
1. Baseline DQN - No QBound, no Double-Q
2. Static QBound + DQN - Q ∈ [-100, 0]
3. Dynamic QBound + DQN - Q ∈ [Q_min(t), 0], step-aware increasing bounds
4. Baseline DDQN - No QBound, with Double-Q
5. Static QBound + DDQN - Q ∈ [-100, 0] + Double-Q
6. Dynamic QBound + DDQN - Q ∈ [Q_min(t), 0] + Double-Q

Key Hypothesis:
- Dynamic QBound should work for negative sparse rewards with INCREASING bounds
- Q_min(t) = -(1 - γ^(H-t)) / (1 - γ) becomes LESS negative as t increases
- This is the opposite of CartPole where Q_max(t) DECREASES with t
"""

import sys
sys.path.insert(0, '/root/projects/QBound/src')

import gymnasium as gym
import numpy as np
import torch
import json
import random
import os
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
RESULTS_FILE = "/root/projects/QBound/results/mountaincar/6way_comparison_in_progress.json"

# Environment parameters
ENV_NAME = "MountainCar-v0"
MAX_EPISODES = 500
MAX_STEPS = 200  # MountainCar default max steps
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

# QBound parameters for MountainCar (Sparse Negative Reward)
# Reward: -1 per step until goal
# Max episode length: 200 steps
# Q_min = -1 * (1 - γ^200) / (1 - γ) = -1 * 99.34 ≈ -100
# Q_max = 0 (reach goal immediately)
QBOUND_MIN = -100.0
QBOUND_MAX = 0.0
STEP_REWARD = 1.0  # Magnitude of reward per step


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


def evaluate_agent(env, agent, num_episodes=10):
    """Evaluate agent performance"""
    total_rewards = []
    total_steps = []

    for _ in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        step = 0

        while not done and step < MAX_STEPS:
            action = agent.select_action(state, eval_mode=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            state = next_state
            step += 1

        total_rewards.append(episode_reward)
        total_steps.append(step)

    return np.mean(total_rewards), np.std(total_rewards), np.mean(total_steps)


def train_agent(env, agent, agent_name, max_episodes=MAX_EPISODES, use_step_aware=False):
    """Train agent and return results"""
    print(f"\n>>> Training {agent_name}...")

    episode_rewards = []
    episode_steps = []
    best_reward = -np.inf

    for episode in tqdm(range(max_episodes), desc=agent_name):
        # Incremental seeding for reproducibility
        env_seed = SEED + episode
        state, _ = env.reset(seed=env_seed)

        episode_reward = 0
        done = False
        step = 0

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

            # Train
            loss = agent.train_step()

            episode_reward += reward
            state = next_state
            step += 1

        episode_rewards.append(episode_reward)
        episode_steps.append(step)

        # Track best performance
        if episode_reward > best_reward:
            best_reward = episode_reward

        # Progress update
        if (episode + 1) % 100 == 0:
            recent_avg_reward = np.mean(episode_rewards[-100:])
            recent_avg_steps = np.mean(episode_steps[-100:])
            print(f"  Episode {episode + 1}/{max_episodes} - Avg reward: {recent_avg_reward:.2f}, "
                  f"Avg steps: {recent_avg_steps:.1f}, Best: {best_reward:.2f}")

    return episode_rewards, episode_steps


def main():
    print("=" * 80)
    print("MountainCar-v0: 6-Way Comprehensive Comparison")
    print("Testing Dynamic QBound on Sparse Negative Rewards")
    print("=" * 80)
    print("Hypothesis: Dynamic QBound with INCREASING bounds works for negative rewards")
    print("=" * 80)

    print(f"\nConfiguration:")
    print(f"  Environment: {ENV_NAME}")
    print(f"  Episodes: {MAX_EPISODES}")
    print(f"  Max steps per episode: {MAX_STEPS}")
    print(f"  Discount factor γ: {GAMMA}")
    print(f"  Learning rate: {LR}")
    print(f"  Static QBound range: [{QBOUND_MIN:.2f}, {QBOUND_MAX:.2f}]")
    print(f"  Dynamic QBound: Q_min(t) = -(1 - γ^(H-t)) / (1-γ), Q_max = 0")
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
    env = gym.make(ENV_NAME)
    env.reset(seed=SEED)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # ===== 1. Baseline DQN (no QBound, no Double-Q) =====
    print("\n" + "=" * 80)
    print("METHOD 1: Baseline DQN (no QBound, no Double-Q)")
    print("=" * 80)

    if is_method_completed(results, 'dqn'):
        print("⏭️  Already completed, skipping...")
    else:
        dqn_agent = DQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
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

        dqn_rewards, dqn_steps = train_agent(env, dqn_agent, "1. Baseline DQN", use_step_aware=False)
        results['training']['dqn'] = {
            'rewards': dqn_rewards,
            'steps': dqn_steps,
            'total_reward': float(np.sum(dqn_rewards)),
            'mean_reward': float(np.mean(dqn_rewards)),
            'mean_steps': float(np.mean(dqn_steps))
        }
        save_intermediate_results(results)

    # ===== 2. Static QBound + DQN =====
    print("\n" + "=" * 80)
    print("METHOD 2: Static QBound + DQN")
    print(f"Q-bounds: [{QBOUND_MIN:.2f}, {QBOUND_MAX:.2f}]")
    print("=" * 80)

    if is_method_completed(results, 'qbound_static_dqn'):
        print("⏭️  Already completed, skipping...")
    else:
        qbound_static_dqn_agent = DQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
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

        qbound_static_rewards, qbound_static_steps = train_agent(
            env, qbound_static_dqn_agent, "2. Static QBound + DQN", use_step_aware=False)
        results['training']['qbound_static_dqn'] = {
            'rewards': qbound_static_rewards,
            'steps': qbound_static_steps,
            'total_reward': float(np.sum(qbound_static_rewards)),
            'mean_reward': float(np.mean(qbound_static_rewards)),
            'mean_steps': float(np.mean(qbound_static_steps))
        }
        save_intermediate_results(results)

    # ===== 3. Dynamic QBound + DQN (Step-Aware) =====
    print("\n" + "=" * 80)
    print("METHOD 3: Dynamic QBound + DQN (Step-Aware INCREASING bounds)")
    print("Q_min(t) = -(1 - γ^(H-t)) / (1-γ) [increases to 0 as t → H]")
    print("=" * 80)

    if is_method_completed(results, 'qbound_dynamic_dqn'):
        print("⏭️  Already completed, skipping...")
    else:
        qbound_dynamic_dqn_agent = DQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
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
            use_step_aware_qbound=True,
            max_episode_steps=MAX_STEPS,
            step_reward=STEP_REWARD,
            reward_is_negative=True,  # Key: negative sparse reward
            device='cpu'
        )

        qbound_dynamic_rewards, qbound_dynamic_steps = train_agent(
            env, qbound_dynamic_dqn_agent, "3. Dynamic QBound + DQN", use_step_aware=True)
        results['training']['qbound_dynamic_dqn'] = {
            'rewards': qbound_dynamic_rewards,
            'steps': qbound_dynamic_steps,
            'total_reward': float(np.sum(qbound_dynamic_rewards)),
            'mean_reward': float(np.mean(qbound_dynamic_rewards)),
            'mean_steps': float(np.mean(qbound_dynamic_steps))
        }
        save_intermediate_results(results)

    # ===== 4. Baseline DDQN (no QBound, with Double-Q) =====
    print("\n" + "=" * 80)
    print("METHOD 4: Baseline DDQN (no QBound, with Double-Q)")
    print("=" * 80)

    if is_method_completed(results, 'ddqn'):
        print("⏭️  Already completed, skipping...")
    else:
        ddqn_agent = DoubleDQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
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

        ddqn_rewards, ddqn_steps = train_agent(env, ddqn_agent, "4. Baseline DDQN", use_step_aware=False)
        results['training']['ddqn'] = {
            'rewards': ddqn_rewards,
            'steps': ddqn_steps,
            'total_reward': float(np.sum(ddqn_rewards)),
            'mean_reward': float(np.mean(ddqn_rewards)),
            'mean_steps': float(np.mean(ddqn_steps))
        }
        save_intermediate_results(results)

    # ===== 5. Static QBound + DDQN =====
    print("\n" + "=" * 80)
    print("METHOD 5: Static QBound + DDQN")
    print(f"Q-bounds: [{QBOUND_MIN:.2f}, {QBOUND_MAX:.2f}] + Double-Q")
    print("=" * 80)

    if is_method_completed(results, 'qbound_static_ddqn'):
        print("⏭️  Already completed, skipping...")
    else:
        qbound_static_ddqn_agent = DoubleDQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
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

        qbound_static_ddqn_rewards, qbound_static_ddqn_steps = train_agent(
            env, qbound_static_ddqn_agent, "5. Static QBound + DDQN", use_step_aware=False)
        results['training']['qbound_static_ddqn'] = {
            'rewards': qbound_static_ddqn_rewards,
            'steps': qbound_static_ddqn_steps,
            'total_reward': float(np.sum(qbound_static_ddqn_rewards)),
            'mean_reward': float(np.mean(qbound_static_ddqn_rewards)),
            'mean_steps': float(np.mean(qbound_static_ddqn_steps))
        }
        save_intermediate_results(results)

    # ===== 6. Dynamic QBound + DDQN (Step-Aware) =====
    print("\n" + "=" * 80)
    print("METHOD 6: Dynamic QBound + DDQN (Step-Aware INCREASING bounds + Double-Q)")
    print("Q_min(t) = -(1 - γ^(H-t)) / (1-γ) [increases to 0 as t → H]")
    print("=" * 80)

    if is_method_completed(results, 'qbound_dynamic_ddqn'):
        print("⏭️  Already completed, skipping...")
    else:
        qbound_dynamic_ddqn_agent = DoubleDQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
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
            use_step_aware_qbound=True,
            max_episode_steps=MAX_STEPS,
            step_reward=STEP_REWARD,
            reward_is_negative=True,  # Key: negative sparse reward
            device='cpu'
        )

        qbound_dynamic_ddqn_rewards, qbound_dynamic_ddqn_steps = train_agent(
            env, qbound_dynamic_ddqn_agent, "6. Dynamic QBound + DDQN", use_step_aware=True)
        results['training']['qbound_dynamic_ddqn'] = {
            'rewards': qbound_dynamic_ddqn_rewards,
            'steps': qbound_dynamic_ddqn_steps,
            'total_reward': float(np.sum(qbound_dynamic_ddqn_rewards)),
            'mean_reward': float(np.mean(qbound_dynamic_ddqn_rewards)),
            'mean_steps': float(np.mean(qbound_dynamic_ddqn_steps))
        }
        save_intermediate_results(results)

    # ===== Summary =====
    print("\n" + "=" * 80)
    print("Training Results Summary")
    print("=" * 80)

    print(f"\nTotal cumulative reward:")
    print(f"  1. Baseline DQN:          {results['training']['dqn']['total_reward']:.0f}")
    print(f"  2. Static QBound + DQN:   {results['training']['qbound_static_dqn']['total_reward']:.0f}")
    print(f"  3. Dynamic QBound + DQN:  {results['training']['qbound_dynamic_dqn']['total_reward']:.0f}")
    print(f"  4. Baseline DDQN:         {results['training']['ddqn']['total_reward']:.0f}")
    print(f"  5. Static QBound + DDQN:  {results['training']['qbound_static_ddqn']['total_reward']:.0f}")
    print(f"  6. Dynamic QBound + DDQN: {results['training']['qbound_dynamic_ddqn']['total_reward']:.0f}")

    print(f"\nAverage episode reward:")
    print(f"  1. Baseline DQN:          {results['training']['dqn']['mean_reward']:.2f}")
    print(f"  2. Static QBound + DQN:   {results['training']['qbound_static_dqn']['mean_reward']:.2f}")
    print(f"  3. Dynamic QBound + DQN:  {results['training']['qbound_dynamic_dqn']['mean_reward']:.2f}")
    print(f"  4. Baseline DDQN:         {results['training']['ddqn']['mean_reward']:.2f}")
    print(f"  5. Static QBound + DDQN:  {results['training']['qbound_static_ddqn']['mean_reward']:.2f}")
    print(f"  6. Dynamic QBound + DDQN: {results['training']['qbound_dynamic_ddqn']['mean_reward']:.2f}")

    print(f"\nAverage steps to goal:")
    print(f"  1. Baseline DQN:          {results['training']['dqn']['mean_steps']:.1f}")
    print(f"  2. Static QBound + DQN:   {results['training']['qbound_static_dqn']['mean_steps']:.1f}")
    print(f"  3. Dynamic QBound + DQN:  {results['training']['qbound_dynamic_dqn']['mean_steps']:.1f}")
    print(f"  4. Baseline DDQN:         {results['training']['ddqn']['mean_steps']:.1f}")
    print(f"  5. Static QBound + DDQN:  {results['training']['qbound_static_ddqn']['mean_steps']:.1f}")
    print(f"  6. Dynamic QBound + DDQN: {results['training']['qbound_dynamic_ddqn']['mean_steps']:.1f}")

    # ===== Key Comparisons =====
    print("\n" + "=" * 80)
    print("KEY COMPARISONS")
    print("=" * 80)

    # Q1: Does static QBound help DQN?
    dqn_total = results['training']['dqn']['total_reward']
    static_dqn_total = results['training']['qbound_static_dqn']['total_reward']
    improvement_1 = ((static_dqn_total - dqn_total) / abs(dqn_total)) * 100

    print(f"\nQ1: Does STATIC QBound help DQN on sparse negative rewards?")
    print(f"    Static QBound+DQN vs Baseline DQN: {improvement_1:+.1f}%")
    if improvement_1 > 5:
        print(f"    ✅ YES! Static QBound improves DQN")
    elif improvement_1 > -5:
        print(f"    ➖ NEUTRAL: Minimal impact")
    else:
        print(f"    ❌ NO: Static QBound hurts DQN")

    # Q2: Does dynamic QBound help DQN?
    dynamic_dqn_total = results['training']['qbound_dynamic_dqn']['total_reward']
    improvement_2 = ((dynamic_dqn_total - dqn_total) / abs(dqn_total)) * 100

    print(f"\nQ2: Does DYNAMIC QBound help DQN on sparse negative rewards?")
    print(f"    Dynamic QBound+DQN vs Baseline DQN: {improvement_2:+.1f}%")
    if improvement_2 > 5:
        print(f"    ✅ YES! Dynamic QBound improves DQN")
    elif improvement_2 > -5:
        print(f"    ➖ NEUTRAL: Minimal impact")
    else:
        print(f"    ❌ NO: Dynamic QBound hurts DQN")

    # Q3: Does QBound enhance DDQN?
    ddqn_total = results['training']['ddqn']['total_reward']
    static_ddqn_total = results['training']['qbound_static_ddqn']['total_reward']
    dynamic_ddqn_total = results['training']['qbound_dynamic_ddqn']['total_reward']
    improvement_3a = ((static_ddqn_total - ddqn_total) / abs(ddqn_total)) * 100
    improvement_3b = ((dynamic_ddqn_total - ddqn_total) / abs(ddqn_total)) * 100

    print(f"\nQ3: Can QBound ENHANCE DDQN?")
    print(f"    Static QBound+DDQN vs Baseline DDQN: {improvement_3a:+.1f}%")
    print(f"    Dynamic QBound+DDQN vs Baseline DDQN: {improvement_3b:+.1f}%")
    if improvement_3b > 5:
        print(f"    ✅ YES! Dynamic QBound enhances even DDQN")
    elif improvement_3a > 5:
        print(f"    ✅ YES! Static QBound enhances DDQN")
    elif improvement_3a > -5 and improvement_3b > -5:
        print(f"    ➖ NEUTRAL: DDQN already handles overestimation well")
    else:
        print(f"    ❌ NO: QBound conflicts with DDQN")

    # Best overall method
    best_method = max(
        [('1. Baseline DQN', dqn_total),
         ('2. Static QBound + DQN', static_dqn_total),
         ('3. Dynamic QBound + DQN', dynamic_dqn_total),
         ('4. Baseline DDQN', ddqn_total),
         ('5. Static QBound + DDQN', static_ddqn_total),
         ('6. Dynamic QBound + DDQN', dynamic_ddqn_total)],
        key=lambda x: x[1]
    )

    print(f"\n🎯 BEST OVERALL METHOD: {best_method[0]} (total reward: {best_method[1]:.0f})")

    # Save final results (rename from in_progress to timestamped)
    timestamp = results['timestamp']
    final_output_file = f"/root/projects/QBound/results/mountaincar/6way_comparison_{timestamp}.json"

    with open(final_output_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Remove the in-progress file
    if os.path.exists(RESULTS_FILE):
        os.remove(RESULTS_FILE)
        print(f"\n✓ Final results saved to: {final_output_file}")
        print(f"✓ Removed in-progress file")
    else:
        print(f"\n✓ Results saved to: {final_output_file}")

    print("\n" + "=" * 80)
    print("6-Way Comparison Complete!")
    print("=" * 80)

    env.close()


if __name__ == "__main__":
    main()
