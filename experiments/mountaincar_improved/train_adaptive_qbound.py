"""
MountainCar-v0: Adaptive QBound Experiment (Priority 1.4)

Hypothesis: Static bounds are too tight from the start. Q-values need room to
           explore initially, then can be constrained as learning stabilizes.

Solution: Start with LOOSE bounds, progressively tighten to final bounds over
         training. This matches the natural learning dynamics.

Experiment Design:
- Episodes 0-1999: Bounds linearly tighten
- Initial bounds: [-600, +200] (3x looser than standard)
- Final bounds: [-200, 0] (standard tight bounds)

Expected Outcome: +20-30% improvement vs always-on static QBound
Success Criteria: Performance ≥ Baseline DQN

Reference: FAILED_EXPERIMENTS_ANALYSIS.md - Experiment 1.4
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
from qbound_adaptive import AdaptiveQBound
from tqdm import tqdm

# Reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

# Experiment configuration
ENV_NAME = "MountainCar-v0"
MAX_EPISODES = 2000
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

# Standard QBound parameters
QBOUND_MIN_STANDARD = -200.0
QBOUND_MAX_STANDARD = 0.0

# Adaptive QBound parameters (3x looser initially)
QBOUND_MIN_INIT = -600.0
QBOUND_MAX_INIT = +200.0
QBOUND_MIN_FINAL = -200.0
QBOUND_MAX_FINAL = 0.0


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


def train_agent(env, agent, agent_name, max_episodes, adaptive_qbound=None):
    """
    Train agent with optional adaptive QBound.

    Args:
        env: Gym environment
        agent: DQN agent
        agent_name: Name for logging
        max_episodes: Number of training episodes
        adaptive_qbound: AdaptiveQBound instance (None for static bounds)

    Returns:
        episode_rewards, episode_steps, bounds_history
    """
    print(f"\n>>> Training {agent_name}...")

    episode_rewards = []
    episode_steps = []
    bounds_history = []  # Track bound evolution
    best_reward = -np.inf

    for episode in tqdm(range(max_episodes), desc=agent_name):
        # Incremental seeding for reproducibility
        env_seed = SEED + episode
        state, _ = env.reset(seed=env_seed)

        # Update bounds if using adaptive strategy
        if adaptive_qbound is not None:
            Q_min, Q_max = adaptive_qbound.get_bounds(episode, max_episodes)
            agent.qclip_min = Q_min
            agent.qclip_max = Q_max
            bounds_history.append({'episode': episode, 'Q_min': Q_min, 'Q_max': Q_max})

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
            if adaptive_qbound is not None:
                Q_min, Q_max = adaptive_qbound.get_bounds(episode, max_episodes)
                bounds_str = f"Bounds: [{Q_min:.1f}, {Q_max:.1f}]"
            else:
                bounds_str = "Static bounds"
            print(f"  Episode {episode + 1}/{max_episodes} - Avg reward: {recent_avg_reward:.2f}, "
                  f"Avg steps: {recent_avg_steps:.1f}, Best: {best_reward:.2f} [{bounds_str}]")

    return episode_rewards, episode_steps, bounds_history


def main():
    print("=" * 80)
    print("MountainCar-v0: Adaptive QBound Experiment (Priority 1.4)")
    print("=" * 80)
    print("Hypothesis: Start with loose bounds, progressively tighten during training")
    print(f"Configuration: [{QBOUND_MIN_INIT:.0f}, {QBOUND_MAX_INIT:+.0f}] → "
          f"[{QBOUND_MIN_FINAL:.0f}, {QBOUND_MAX_FINAL:.0f}]")
    print("=" * 80)

    print(f"\nExperiment Settings:")
    print(f"  Environment: {ENV_NAME}")
    print(f"  Total episodes: {MAX_EPISODES}")
    print(f"  Initial bounds (loose): [{QBOUND_MIN_INIT:.0f}, {QBOUND_MAX_INIT:+.0f}]")
    print(f"  Final bounds (tight):   [{QBOUND_MIN_FINAL:.0f}, {QBOUND_MAX_FINAL:.0f}]")
    print(f"  Standard bounds:        [{QBOUND_MIN_STANDARD:.0f}, {QBOUND_MAX_STANDARD:.0f}]")
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
            'qbound_min_init': QBOUND_MIN_INIT,
            'qbound_max_init': QBOUND_MAX_INIT,
            'qbound_min_final': QBOUND_MIN_FINAL,
            'qbound_max_final': QBOUND_MAX_FINAL,
            'qbound_min_standard': QBOUND_MIN_STANDARD,
            'qbound_max_standard': QBOUND_MAX_STANDARD,
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

    # ===== 2. Static QBound (original failing version) =====
    print("\n" + "=" * 80)
    print("METHOD 2: Static QBound (original)")
    print(f"Q-bounds: [{QBOUND_MIN_STANDARD:.2f}, {QBOUND_MAX_STANDARD:.2f}] (constant)")
    print("=" * 80)

    static_agent = DQNAgent(
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
        qclip_min=QBOUND_MIN_STANDARD,
        qclip_max=QBOUND_MAX_STANDARD,
        device='cpu'
    )

    static_rewards, static_steps, _ = train_agent(
        env, static_agent, "2. Static QBound", MAX_EPISODES)

    results['training']['static_qbound'] = {
        'rewards': static_rewards,
        'steps': static_steps,
        'final_100_mean': float(np.mean(static_rewards[-100:])),
        'total_reward': float(np.sum(static_rewards)),
        'mean_reward': float(np.mean(static_rewards)),
        'mean_steps': float(np.mean(static_steps))
    }

    # ===== 3. Adaptive QBound (NEW - Priority 1.4) =====
    print("\n" + "=" * 80)
    print("METHOD 3: Adaptive QBound (NEW - Experiment 1.4)")
    print(f"Start: [{QBOUND_MIN_INIT:.0f}, {QBOUND_MAX_INIT:+.0f}] (3x looser)")
    print(f"  End: [{QBOUND_MIN_FINAL:.0f}, {QBOUND_MAX_FINAL:.0f}] (standard)")
    print(f"Linearly adapt over {MAX_EPISODES} episodes")
    print("=" * 80)

    adaptive_qbound = AdaptiveQBound(
        Q_min_init=QBOUND_MIN_INIT,
        Q_max_init=QBOUND_MAX_INIT,
        Q_min_final=QBOUND_MIN_FINAL,
        Q_max_final=QBOUND_MAX_FINAL
    )

    adaptive_agent = DQNAgent(
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
        qclip_min=QBOUND_MIN_INIT,  # Will be updated dynamically
        qclip_max=QBOUND_MAX_INIT,
        device='cpu'
    )

    adaptive_rewards, adaptive_steps, bounds_history = train_agent(
        env, adaptive_agent, "3. Adaptive QBound", MAX_EPISODES, adaptive_qbound=adaptive_qbound)

    results['training']['adaptive_qbound'] = {
        'rewards': adaptive_rewards,
        'steps': adaptive_steps,
        'bounds_history': bounds_history,
        'final_100_mean': float(np.mean(adaptive_rewards[-100:])),
        'total_reward': float(np.sum(adaptive_rewards)),
        'mean_reward': float(np.mean(adaptive_rewards)),
        'mean_steps': float(np.mean(adaptive_steps))
    }

    # ===== Analysis =====
    print("\n" + "=" * 80)
    print("EXPERIMENT RESULTS")
    print("=" * 80)

    baseline_final = results['training']['baseline']['final_100_mean']
    static_final = results['training']['static_qbound']['final_100_mean']
    adaptive_final = results['training']['adaptive_qbound']['final_100_mean']

    print(f"\nFinal 100 Episodes Average Reward:")
    print(f"  Baseline DQN:       {baseline_final:.2f}")
    print(f"  Static QBound:      {static_final:.2f}")
    print(f"  Adaptive QBound:    {adaptive_final:.2f}")

    # Compute improvements
    static_vs_baseline = ((static_final - baseline_final) / abs(baseline_final)) * 100
    adaptive_vs_baseline = ((adaptive_final - baseline_final) / abs(baseline_final)) * 100
    adaptive_vs_static = ((adaptive_final - static_final) / abs(static_final)) * 100

    print(f"\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)

    print(f"\n1. Static QBound vs Baseline: {static_vs_baseline:+.1f}%")
    if static_vs_baseline < -5:
        print(f"   ❌ Confirmed: Static QBound HURTS performance (expected from paper)")

    print(f"\n2. Adaptive QBound vs Baseline: {adaptive_vs_baseline:+.1f}%")
    if adaptive_vs_baseline >= 10:
        print(f"   ✅ EXCELLENT! Adaptive QBound is {adaptive_vs_baseline:.1f}% better than baseline!")
    elif adaptive_vs_baseline >= 5:
        print(f"   ✅ GOOD! Adaptive QBound is {adaptive_vs_baseline:.1f}% better than baseline")
    elif adaptive_vs_baseline >= 0:
        print(f"   ✅ SUCCESS! Adaptive QBound ≥ Baseline (broke even or better)")
    elif adaptive_vs_baseline > -5:
        print(f"   ➖ Partial success: Small degradation but much better than static")
    else:
        print(f"   ❌ Still fails: Adaptive QBound still hurts performance")

    print(f"\n3. Adaptive QBound vs Static QBound: {adaptive_vs_static:+.1f}%")
    if adaptive_vs_static > 20:
        print(f"   ✅ EXCELLENT! Adaptive is {adaptive_vs_static:.1f}% better than static!")
    elif adaptive_vs_static > 10:
        print(f"   ✅ VERY GOOD! Adaptive is {adaptive_vs_static:.1f}% better")
    elif adaptive_vs_static > 5:
        print(f"   ✅ GOOD! Adaptive is {adaptive_vs_static:.1f}% better")
    elif adaptive_vs_static > 0:
        print(f"   ✅ Adaptive is better than static")
    else:
        print(f"   ❌ Adaptive does not improve over static")

    # Success evaluation
    print(f"\n" + "=" * 80)
    print("SUCCESS EVALUATION")
    print("=" * 80)

    if adaptive_vs_baseline >= 0:
        print(f"✅ MINIMUM SUCCESS: Adaptive QBound ≥ Baseline (broke even)")
    if adaptive_vs_baseline >= 5:
        print(f"✅ GOOD SUCCESS: Adaptive QBound ≥ Baseline + 5%")
    if adaptive_vs_baseline >= 10:
        print(f"✅ EXCELLENT SUCCESS: Adaptive QBound ≥ Baseline + 10%")

    if adaptive_vs_baseline < 0:
        print(f"⚠️  Did not meet minimum success criteria")
        print(f"   Recommendation: Try different adaptation schedule or exploration bonus")

    # Bounds evolution summary
    print(f"\n" + "=" * 80)
    print("BOUNDS EVOLUTION")
    print("=" * 80)

    if len(bounds_history) > 0:
        print(f"\nEpisode 0:    Q ∈ [{bounds_history[0]['Q_min']:.1f}, {bounds_history[0]['Q_max']:.1f}]")
        if len(bounds_history) > 500:
            print(f"Episode 500:  Q ∈ [{bounds_history[500]['Q_min']:.1f}, {bounds_history[500]['Q_max']:.1f}]")
        if len(bounds_history) > 1000:
            print(f"Episode 1000: Q ∈ [{bounds_history[1000]['Q_min']:.1f}, {bounds_history[1000]['Q_max']:.1f}]")
        if len(bounds_history) > 1500:
            print(f"Episode 1500: Q ∈ [{bounds_history[1500]['Q_min']:.1f}, {bounds_history[1500]['Q_max']:.1f}]")
        print(f"Episode {len(bounds_history)-1}:   Q ∈ [{bounds_history[-1]['Q_min']:.1f}, {bounds_history[-1]['Q_max']:.1f}]")

    # Save results
    output_file = f"/root/projects/QBound/results/mountaincar/adaptive_qbound_{results['timestamp']}.json"
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
