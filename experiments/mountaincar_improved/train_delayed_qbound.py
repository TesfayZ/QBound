"""
MountainCar-v0: Delayed QBound Experiment (Priority 1.1)

Hypothesis: QBound hurts exploration-critical tasks by over-constraining Q-values
           during initial discovery phase.

Solution: Allow initial exploration without bounds, then apply QBound after
         agent discovers the momentum solution.

Experiment Design:
- Episodes 0-499: No QBound (free exploration)
- Episodes 500-1999: QBound active with [-200, 0]

Expected Outcome: +10-20% improvement vs always-on QBound
Success Criteria: Performance ≥ Baseline DQN (break even or better)

Reference: FAILED_EXPERIMENTS_ANALYSIS.md - Experiment 1.1
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
from qbound_adaptive import DelayedQBound
from tqdm import tqdm

# Reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

# Experiment configuration
ENV_NAME = "MountainCar-v0"
MAX_EPISODES = 2000  # Increased from 500 for better learning
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

# QBound parameters
QBOUND_MIN = -200.0
QBOUND_MAX = 0.0

# Delayed QBound configuration
DELAY_EPISODES = 500  # Apply QBound after 500 episodes of exploration


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


def train_agent(env, agent, agent_name, max_episodes, delayed_qbound=None):
    """
    Train agent with optional delayed QBound activation.

    Args:
        env: Gym environment
        agent: DQN agent
        agent_name: Name for logging
        max_episodes: Number of training episodes
        delayed_qbound: DelayedQBound instance (None for no delay)

    Returns:
        episode_rewards, episode_steps, qbound_active_episodes
    """
    print(f"\n>>> Training {agent_name}...")

    episode_rewards = []
    episode_steps = []
    qbound_active_episodes = []  # Track when QBound is active
    best_reward = -np.inf

    for episode in tqdm(range(max_episodes), desc=agent_name):
        # Incremental seeding for reproducibility
        env_seed = SEED + episode
        state, _ = env.reset(seed=env_seed)

        # Check if QBound should be active for this episode
        qbound_active = False
        if delayed_qbound is not None:
            Q_min, Q_max = delayed_qbound.get_bounds(episode)
            if Q_min is not None:
                # Activate QBound for this episode
                agent.qclip_min = Q_min
                agent.qclip_max = Q_max
                agent.use_qclip = True
                qbound_active = True
            else:
                # Disable QBound for exploration
                agent.use_qclip = False

        qbound_active_episodes.append(qbound_active)

        episode_reward = 0
        done = False
        step = 0

        while not done and step < MAX_STEPS:
            action = agent.select_action(state, eval_mode=False)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store_transition(state, action, reward, next_state, done)
            loss = agent.train_step()

            episode_reward += reward
            state = next_state
            step += 1

        episode_rewards.append(episode_reward)
        episode_steps.append(step)

        if episode_reward > best_reward:
            best_reward = episode_reward

        # Progress update
        if (episode + 1) % 100 == 0:
            recent_avg_reward = np.mean(episode_rewards[-100:])
            recent_avg_steps = np.mean(episode_steps[-100:])
            qbound_status = "QBound ON" if qbound_active else "QBound OFF"
            print(f"  Episode {episode + 1}/{max_episodes} - Avg reward: {recent_avg_reward:.2f}, "
                  f"Avg steps: {recent_avg_steps:.1f}, Best: {best_reward:.2f} [{qbound_status}]")

    return episode_rewards, episode_steps, qbound_active_episodes


def main():
    print("=" * 80)
    print("MountainCar-v0: Delayed QBound Experiment (Priority 1.1)")
    print("=" * 80)
    print("Hypothesis: Allow exploration first, then apply QBound")
    print(f"Configuration: No QBound for {DELAY_EPISODES} episodes, then activate")
    print("=" * 80)

    print(f"\nExperiment Settings:")
    print(f"  Environment: {ENV_NAME}")
    print(f"  Total episodes: {MAX_EPISODES}")
    print(f"  QBound delay: {DELAY_EPISODES} episodes")
    print(f"  QBound range (after delay): [{QBOUND_MIN:.2f}, {QBOUND_MAX:.2f}]")
    print(f"  Discount factor γ: {GAMMA}")
    print(f"  Learning rate: {LR}")
    print("=" * 80)

    # Create environment
    env = gym.make(ENV_NAME)
    env.reset(seed=SEED)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

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
            'delay_episodes': DELAY_EPISODES,
            'seed': SEED
        },
        'training': {}
    }

    # ===== 1. Baseline DQN (no QBound) =====
    print("\n" + "=" * 80)
    print("METHOD 1: Baseline DQN (no QBound)")
    print("=" * 80)

    baseline_agent = DQNAgent(
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

    baseline_rewards, baseline_steps, _ = train_agent(
        env, baseline_agent, "1. Baseline DQN", MAX_EPISODES)

    results['training']['baseline'] = {
        'rewards': baseline_rewards,
        'steps': baseline_steps,
        'final_100_mean': float(np.mean(baseline_rewards[-100:])),
        'total_reward': float(np.sum(baseline_rewards)),
        'mean_reward': float(np.mean(baseline_rewards)),
        'mean_steps': float(np.mean(baseline_steps))
    }

    # ===== 2. Always-On QBound (original failing version) =====
    print("\n" + "=" * 80)
    print("METHOD 2: Always-On QBound (original)")
    print(f"Q-bounds: [{QBOUND_MIN:.2f}, {QBOUND_MAX:.2f}] from episode 0")
    print("=" * 80)

    always_on_agent = DQNAgent(
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

    always_on_rewards, always_on_steps, _ = train_agent(
        env, always_on_agent, "2. Always-On QBound", MAX_EPISODES)

    results['training']['always_on_qbound'] = {
        'rewards': always_on_rewards,
        'steps': always_on_steps,
        'final_100_mean': float(np.mean(always_on_rewards[-100:])),
        'total_reward': float(np.sum(always_on_rewards)),
        'mean_reward': float(np.mean(always_on_rewards)),
        'mean_steps': float(np.mean(always_on_steps))
    }

    # ===== 3. Delayed QBound (NEW - Priority 1.1) =====
    print("\n" + "=" * 80)
    print("METHOD 3: Delayed QBound (NEW - Experiment 1.1)")
    print(f"Episodes 0-{DELAY_EPISODES-1}: No QBound (exploration)")
    print(f"Episodes {DELAY_EPISODES}-{MAX_EPISODES-1}: QBound [{QBOUND_MIN:.2f}, {QBOUND_MAX:.2f}]")
    print("=" * 80)

    delayed_qbound = DelayedQBound(
        Q_min=QBOUND_MIN,
        Q_max=QBOUND_MAX,
        delay_episodes=DELAY_EPISODES
    )

    delayed_agent = DQNAgent(
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
        use_qclip=False,  # Will be enabled dynamically
        qclip_min=QBOUND_MIN,
        qclip_max=QBOUND_MAX,
        device='cpu'
    )

    delayed_rewards, delayed_steps, qbound_active = train_agent(
        env, delayed_agent, "3. Delayed QBound", MAX_EPISODES, delayed_qbound=delayed_qbound)

    results['training']['delayed_qbound'] = {
        'rewards': delayed_rewards,
        'steps': delayed_steps,
        'qbound_active': qbound_active,
        'final_100_mean': float(np.mean(delayed_rewards[-100:])),
        'total_reward': float(np.sum(delayed_rewards)),
        'mean_reward': float(np.mean(delayed_rewards)),
        'mean_steps': float(np.mean(delayed_steps))
    }

    # ===== Analysis =====
    print("\n" + "=" * 80)
    print("EXPERIMENT RESULTS")
    print("=" * 80)

    baseline_final = results['training']['baseline']['final_100_mean']
    always_on_final = results['training']['always_on_qbound']['final_100_mean']
    delayed_final = results['training']['delayed_qbound']['final_100_mean']

    print(f"\nFinal 100 Episodes Average Reward:")
    print(f"  Baseline DQN:       {baseline_final:.2f}")
    print(f"  Always-On QBound:   {always_on_final:.2f}")
    print(f"  Delayed QBound:     {delayed_final:.2f}")

    # Compute improvements
    always_on_vs_baseline = ((always_on_final - baseline_final) / abs(baseline_final)) * 100
    delayed_vs_baseline = ((delayed_final - baseline_final) / abs(baseline_final)) * 100
    delayed_vs_always_on = ((delayed_final - always_on_final) / abs(always_on_final)) * 100

    print(f"\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)

    print(f"\n1. Always-On QBound vs Baseline: {always_on_vs_baseline:+.1f}%")
    if always_on_vs_baseline < -5:
        print(f"   ❌ Confirmed: Always-On QBound HURTS performance (expected from paper)")

    print(f"\n2. Delayed QBound vs Baseline: {delayed_vs_baseline:+.1f}%")
    if delayed_vs_baseline >= 0:
        print(f"   ✅ SUCCESS! Delayed QBound ≥ Baseline (broke even or better)")
    elif delayed_vs_baseline > -5:
        print(f"   ➖ Partial success: Small degradation but much better than always-on")
    else:
        print(f"   ❌ Still fails: Delayed QBound still hurts performance")

    print(f"\n3. Delayed QBound vs Always-On QBound: {delayed_vs_always_on:+.1f}%")
    if delayed_vs_always_on > 10:
        print(f"   ✅ EXCELLENT! Delayed QBound is {delayed_vs_always_on:.1f}% better!")
    elif delayed_vs_always_on > 5:
        print(f"   ✅ GOOD! Delayed QBound is {delayed_vs_always_on:.1f}% better")
    elif delayed_vs_always_on > 0:
        print(f"   ✅ Delayed QBound is slightly better")
    else:
        print(f"   ❌ Delayed QBound does not improve over always-on")

    # Success evaluation
    print(f"\n" + "=" * 80)
    print("SUCCESS EVALUATION")
    print("=" * 80)

    if delayed_vs_baseline >= 0:
        print(f"✅ MINIMUM SUCCESS: Delayed QBound ≥ Baseline (broke even)")
    if delayed_vs_baseline >= 5:
        print(f"✅ GOOD SUCCESS: Delayed QBound ≥ Baseline + 5%")
    if delayed_vs_baseline >= 10:
        print(f"✅ EXCELLENT SUCCESS: Delayed QBound ≥ Baseline + 10%")

    if delayed_vs_baseline < 0:
        print(f"⚠️  Did not meet minimum success criteria")
        print(f"   Recommendation: Try longer delay or adaptive bounds (Experiment 1.4)")

    # Save results
    output_file = f"/root/projects/QBound/results/mountaincar/delayed_qbound_{results['timestamp']}.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to: {output_file}")
    print("\n" + "=" * 80)
    print("Experiment Complete!")
    print("=" * 80)

    env.close()


if __name__ == "__main__":
    main()
