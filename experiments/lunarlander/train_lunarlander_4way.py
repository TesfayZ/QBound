"""
LunarLander-v3: 4-Way Comparison (Static QBound Only)
Testing QBound on Shaped/Mixed Rewards

Environment: LunarLander-v3 (discrete actions, shaped rewards)
Reward: Mixed (negative for fuel, positive for landing, large penalty for crash)
This is a SHAPED reward environment, so we only test STATIC QBound.

Dynamic QBound doesn't apply well to shaped rewards since the reward structure
is complex and doesn't follow a simple step-based pattern.

Comparison:
1. Baseline DQN - No QBound, no Double-Q
2. Static QBound + DQN - Q ∈ [-100, 200]
3. Baseline DDQN - No QBound, with Double-Q
4. Static QBound + DDQN - Q ∈ [-100, 200] + Double-Q
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
RESULTS_FILE = "/root/projects/QBound/results/lunarlander/4way_comparison_in_progress.json"

# Environment parameters
ENV_NAME = "LunarLander-v3"
MAX_EPISODES = 500
MAX_STEPS = 1000  # LunarLander max steps
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

# QBound parameters for LunarLander (Shaped/Mixed Rewards)
# Reward range: approximately -100 (crash) to +200 (perfect landing)
# Using conservative static bounds
QBOUND_MIN = -100.0
QBOUND_MAX = 200.0


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


def train_agent(env, agent, agent_name, max_episodes=MAX_EPISODES):
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

            # Store transition (no step-aware for shaped rewards)
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
    print("LunarLander-v3: 4-Way Comparison (Static QBound Only)")
    print("Testing QBound on Shaped/Mixed Rewards")
    print("=" * 80)
    print("Note: Dynamic QBound not used - shaped rewards don't follow step pattern")
    print("=" * 80)

    print(f"\nConfiguration:")
    print(f"  Environment: {ENV_NAME}")
    print(f"  Episodes: {MAX_EPISODES}")
    print(f"  Max steps per episode: {MAX_STEPS}")
    print(f"  Discount factor γ: {GAMMA}")
    print(f"  Learning rate: {LR}")
    print(f"  Static QBound range: [{QBOUND_MIN:.2f}, {QBOUND_MAX:.2f}]")
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
        dqn_rewards, dqn_steps = train_agent(env, dqn_agent, "1. Baseline DQN")
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
            env, qbound_static_dqn_agent, "2. Static QBound + DQN")
        results['training']['qbound_static_dqn'] = {
            'rewards': qbound_static_rewards, 'steps': qbound_static_steps,
            'total_reward': float(np.sum(qbound_static_rewards)),
            'mean_reward': float(np.mean(qbound_static_rewards)),
            'mean_steps': float(np.mean(qbound_static_steps))
        }
        save_intermediate_results(results)

    # ===== 3. Baseline DDQN =====
    print("\n" + "=" * 80)
    print("METHOD 3: Baseline DDQN")
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
        ddqn_rewards, ddqn_steps = train_agent(env, ddqn_agent, "3. Baseline DDQN")
        results['training']['ddqn'] = {
            'rewards': ddqn_rewards, 'steps': ddqn_steps,
            'total_reward': float(np.sum(ddqn_rewards)),
            'mean_reward': float(np.mean(ddqn_rewards)),
            'mean_steps': float(np.mean(ddqn_steps))
        }
        save_intermediate_results(results)

    # ===== 4. Static QBound + DDQN =====
    print("\n" + "=" * 80)
    print("METHOD 4: Static QBound + DDQN")
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
            env, qbound_static_ddqn_agent, "4. Static QBound + DDQN")
        results['training']['qbound_static_ddqn'] = {
            'rewards': qbound_static_ddqn_rewards, 'steps': qbound_static_ddqn_steps,
            'total_reward': float(np.sum(qbound_static_ddqn_rewards)),
            'mean_reward': float(np.mean(qbound_static_ddqn_rewards)),
            'mean_steps': float(np.mean(qbound_static_ddqn_steps))
        }
        save_intermediate_results(results)

    # ===== Save Final Results =====
    timestamp = results['timestamp']
    final_output_file = f"/root/projects/QBound/results/lunarlander/4way_comparison_{timestamp}.json"

    with open(final_output_file, 'w') as f:
        json.dump(results, f, indent=2)

    if os.path.exists(RESULTS_FILE):
        os.remove(RESULTS_FILE)

    print(f"\n✓ Results saved to: {final_output_file}")
    print("\n" + "=" * 80)
    print("LunarLander 4-Way Comparison Complete!")
    print("=" * 80)

    env.close()


if __name__ == "__main__":
    main()
