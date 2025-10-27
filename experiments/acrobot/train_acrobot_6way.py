"""
Acrobot-v1: 6-Way Comprehensive Comparison
Testing QBound on Sparse Negative Reward (Swing-Up Task)

Environment: Acrobot-v1 (discrete actions, sparse negative rewards)
Reward: -1 per step until goal reached (swing up to height)
Max steps: 500

This tests QBound on another sparse negative reward environment with INCREASING bounds.

Comparison:
1. Baseline DQN - No QBound, no Double-Q
2. Static QBound + DQN - Q ∈ [-500, 0]
3. Dynamic QBound + DQN - Q ∈ [Q_min(t), 0], step-aware increasing bounds
4. Baseline DDQN - No QBound, with Double-Q
5. Static QBound + DDQN - Q ∈ [-500, 0] + Double-Q
6. Dynamic QBound + DDQN - Q ∈ [Q_min(t), 0] + Double-Q
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
RESULTS_FILE = "/root/projects/QBound/results/acrobot/6way_comparison_in_progress.json"

# Environment parameters
ENV_NAME = "Acrobot-v1"
MAX_EPISODES = 500
MAX_STEPS = 500  # Acrobot default max steps
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

# QBound parameters for Acrobot (Sparse Negative Reward)
# Reward: -1 per step until swing up
# Max episode length: 500 steps
# Q_min = -1 * (1 - γ^500) / (1 - γ) ≈ -100 (using γ=0.99)
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
    print("Acrobot-v1: 6-Way Comprehensive Comparison")
    print("Testing QBound on Sparse Negative Rewards (Swing-Up Task)")
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

    # ===== 1. Baseline DQN =====
    print("\n" + "=" * 80)
    print("METHOD 1: Baseline DQN")
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
        dqn_rewards, dqn_steps = train_agent(env, dqn_agent, "1. Baseline DQN", use_step_aware=False)
        results['training']['dqn'] = {
            'rewards': dqn_rewards, 'steps': dqn_steps,
            'total_reward': float(np.sum(dqn_rewards)),
            'mean_reward': float(np.mean(dqn_rewards)),
            'mean_steps': float(np.mean(dqn_steps))
        }
        save_intermediate_results(results)

    # ===== 2. Static QBound + DQN =====
    print("\n" + "=" * 80)
    print("METHOD 2: Static QBound + DQN")
    print("=" * 80)

    if is_method_completed(results, 'qbound_static_dqn'):
        print("⏭️  Already completed, skipping...")
    else:
        qbound_static_dqn_agent = DQNAgent(
            state_dim=state_dim, action_dim=action_dim, learning_rate=LR,
            gamma=GAMMA, epsilon_start=EPSILON_START, epsilon_end=EPSILON_END,
            epsilon_decay=EPSILON_DECAY, buffer_size=BUFFER_SIZE,
            batch_size=BATCH_SIZE, target_update_freq=TARGET_UPDATE_FREQ,
            use_qclip=True, qclip_min=QBOUND_MIN, qclip_max=QBOUND_MAX, device='cpu'
        )
        qbound_static_rewards, qbound_static_steps = train_agent(
            env, qbound_static_dqn_agent, "2. Static QBound + DQN", use_step_aware=False)
        results['training']['qbound_static_dqn'] = {
            'rewards': qbound_static_rewards, 'steps': qbound_static_steps,
            'total_reward': float(np.sum(qbound_static_rewards)),
            'mean_reward': float(np.mean(qbound_static_rewards)),
            'mean_steps': float(np.mean(qbound_static_steps))
        }
        save_intermediate_results(results)

    # ===== 3. Dynamic QBound + DQN =====
    print("\n" + "=" * 80)
    print("METHOD 3: Dynamic QBound + DQN (INCREASING bounds)")
    print("=" * 80)

    if is_method_completed(results, 'qbound_dynamic_dqn'):
        print("⏭️  Already completed, skipping...")
    else:
        qbound_dynamic_dqn_agent = DQNAgent(
            state_dim=state_dim, action_dim=action_dim, learning_rate=LR,
            gamma=GAMMA, epsilon_start=EPSILON_START, epsilon_end=EPSILON_END,
            epsilon_decay=EPSILON_DECAY, buffer_size=BUFFER_SIZE,
            batch_size=BATCH_SIZE, target_update_freq=TARGET_UPDATE_FREQ,
            use_qclip=True, qclip_min=QBOUND_MIN, qclip_max=QBOUND_MAX,
            use_step_aware_qbound=True, max_episode_steps=MAX_STEPS,
            step_reward=STEP_REWARD, reward_is_negative=True, device='cpu'
        )
        qbound_dynamic_rewards, qbound_dynamic_steps = train_agent(
            env, qbound_dynamic_dqn_agent, "3. Dynamic QBound + DQN", use_step_aware=True)
        results['training']['qbound_dynamic_dqn'] = {
            'rewards': qbound_dynamic_rewards, 'steps': qbound_dynamic_steps,
            'total_reward': float(np.sum(qbound_dynamic_rewards)),
            'mean_reward': float(np.mean(qbound_dynamic_rewards)),
            'mean_steps': float(np.mean(qbound_dynamic_steps))
        }
        save_intermediate_results(results)

    # ===== 4. Baseline DDQN =====
    print("\n" + "=" * 80)
    print("METHOD 4: Baseline DDQN")
    print("=" * 80)

    if is_method_completed(results, 'ddqn'):
        print("⏭️  Already completed, skipping...")
    else:
        ddqn_agent = DoubleDQNAgent(
            state_dim=state_dim, action_dim=action_dim, learning_rate=LR,
            gamma=GAMMA, epsilon_start=EPSILON_START, epsilon_end=EPSILON_END,
            epsilon_decay=EPSILON_DECAY, buffer_size=BUFFER_SIZE,
            batch_size=BATCH_SIZE, target_update_freq=TARGET_UPDATE_FREQ,
            use_qclip=False, device='cpu'
        )
        ddqn_rewards, ddqn_steps = train_agent(env, ddqn_agent, "4. Baseline DDQN", use_step_aware=False)
        results['training']['ddqn'] = {
            'rewards': ddqn_rewards, 'steps': ddqn_steps,
            'total_reward': float(np.sum(ddqn_rewards)),
            'mean_reward': float(np.mean(ddqn_rewards)),
            'mean_steps': float(np.mean(ddqn_steps))
        }
        save_intermediate_results(results)

    # ===== 5. Static QBound + DDQN =====
    print("\n" + "=" * 80)
    print("METHOD 5: Static QBound + DDQN")
    print("=" * 80)

    if is_method_completed(results, 'qbound_static_ddqn'):
        print("⏭️  Already completed, skipping...")
    else:
        qbound_static_ddqn_agent = DoubleDQNAgent(
            state_dim=state_dim, action_dim=action_dim, learning_rate=LR,
            gamma=GAMMA, epsilon_start=EPSILON_START, epsilon_end=EPSILON_END,
            epsilon_decay=EPSILON_DECAY, buffer_size=BUFFER_SIZE,
            batch_size=BATCH_SIZE, target_update_freq=TARGET_UPDATE_FREQ,
            use_qclip=True, qclip_min=QBOUND_MIN, qclip_max=QBOUND_MAX, device='cpu'
        )
        qbound_static_ddqn_rewards, qbound_static_ddqn_steps = train_agent(
            env, qbound_static_ddqn_agent, "5. Static QBound + DDQN", use_step_aware=False)
        results['training']['qbound_static_ddqn'] = {
            'rewards': qbound_static_ddqn_rewards, 'steps': qbound_static_ddqn_steps,
            'total_reward': float(np.sum(qbound_static_ddqn_rewards)),
            'mean_reward': float(np.mean(qbound_static_ddqn_rewards)),
            'mean_steps': float(np.mean(qbound_static_ddqn_steps))
        }
        save_intermediate_results(results)

    # ===== 6. Dynamic QBound + DDQN =====
    print("\n" + "=" * 80)
    print("METHOD 6: Dynamic QBound + DDQN (INCREASING bounds)")
    print("=" * 80)

    if is_method_completed(results, 'qbound_dynamic_ddqn'):
        print("⏭️  Already completed, skipping...")
    else:
        qbound_dynamic_ddqn_agent = DoubleDQNAgent(
            state_dim=state_dim, action_dim=action_dim, learning_rate=LR,
            gamma=GAMMA, epsilon_start=EPSILON_START, epsilon_end=EPSILON_END,
            epsilon_decay=EPSILON_DECAY, buffer_size=BUFFER_SIZE,
            batch_size=BATCH_SIZE, target_update_freq=TARGET_UPDATE_FREQ,
            use_qclip=True, qclip_min=QBOUND_MIN, qclip_max=QBOUND_MAX,
            use_step_aware_qbound=True, max_episode_steps=MAX_STEPS,
            step_reward=STEP_REWARD, reward_is_negative=True, device='cpu'
        )
        qbound_dynamic_ddqn_rewards, qbound_dynamic_ddqn_steps = train_agent(
            env, qbound_dynamic_ddqn_agent, "6. Dynamic QBound + DDQN", use_step_aware=True)
        results['training']['qbound_dynamic_ddqn'] = {
            'rewards': qbound_dynamic_ddqn_rewards, 'steps': qbound_dynamic_ddqn_steps,
            'total_reward': float(np.sum(qbound_dynamic_ddqn_rewards)),
            'mean_reward': float(np.mean(qbound_dynamic_ddqn_rewards)),
            'mean_steps': float(np.mean(qbound_dynamic_ddqn_steps))
        }
        save_intermediate_results(results)

    # ===== Save Final Results =====
    timestamp = results['timestamp']
    final_output_file = f"/root/projects/QBound/results/acrobot/6way_comparison_{timestamp}.json"

    with open(final_output_file, 'w') as f:
        json.dump(results, f, indent=2)

    if os.path.exists(RESULTS_FILE):
        os.remove(RESULTS_FILE)

    print(f"\n✓ Results saved to: {final_output_file}")
    print("\n" + "=" * 80)
    print("Acrobot 6-Way Comparison Complete!")
    print("=" * 80)

    env.close()


if __name__ == "__main__":
    main()
