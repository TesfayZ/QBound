"""
LunarLander-v3: 6-Way Comprehensive Comparison
Tests QBound on Shaped/Mixed Reward Environment

Environment: LunarLander-v3 (discrete actions, shaped rewards)
Reward: Mixed (negative for fuel, positive for landing, penalty for crash)
Max steps: 1000

Comparison:
1. Baseline DQN - No QBound, no Double-Q
2. Static QBound + DQN - Q ∈ [-100, 200]
3. Dynamic QBound + DQN - Q ∈ [Q_min(t), Q_max(t)] with step-aware bounds
4. Baseline DDQN - No QBound, with Double-Q
5. Static QBound + DDQN - Q ∈ [-100, 200] + Double-Q
6. Dynamic QBound + DDQN - Q ∈ [Q_min(t), Q_max(t)] + Double-Q

Note: LunarLander has shaped rewards, so dynamic QBound is less applicable.
Static QBound should be the most relevant for this environment.
"""

import sys
sys.path.insert(0, '/root/projects/QBound/src')

import numpy as np
import torch
import json
import random
import os
import gymnasium as gym
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
RESULTS_FILE = "/root/projects/QBound/results/lunarlander/6way_comparison_in_progress.json"

# Environment parameters
ENV_NAME = "LunarLander-v3"
MAX_EPISODES = 500
MAX_STEPS = 1000
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
            # Note: step-aware is less meaningful for shaped rewards
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
                  f"Avg steps: {recent_avg_steps:.1f}, Best: {best_reward:.2f}, ε={agent.epsilon:.3f}")

    return episode_rewards, episode_steps


def main():
    print("=" * 80)
    print("LunarLander-v3: 6-Way Comprehensive Comparison")
    print("Testing QBound on Shaped/Mixed Reward")
    print("=" * 80)

    print(f"\nConfiguration:")
    print(f"  Environment: {ENV_NAME}")
    print(f"  Episodes: {MAX_EPISODES}")
    print(f"  Max steps per episode: {MAX_STEPS}")
    print(f"  Discount factor γ: {GAMMA}")
    print(f"  Learning rate: {LR}")
    print(f"  QBound range: [{QBOUND_MIN:.2f}, {QBOUND_MAX:.2f}]")
    print("  Note: Dynamic QBound less applicable for shaped rewards")
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

    if is_method_completed(results, 'baseline'):
        print("⏭️  Already completed, skipping...")
    else:
        baseline_agent = DQNAgent(
            state_dim=state_dim, action_dim=action_dim, learning_rate=LR,
            gamma=GAMMA, epsilon_start=EPSILON_START, epsilon_end=EPSILON_END,
            epsilon_decay=EPSILON_DECAY, buffer_size=BUFFER_SIZE,
            batch_size=BATCH_SIZE, target_update_freq=TARGET_UPDATE_FREQ,
            use_qclip=False, device='cpu'
        )
        baseline_rewards, baseline_steps = train_agent(
            env, baseline_agent, "1. Baseline DQN", use_step_aware=False)
        results['training']['baseline'] = {
            'rewards': baseline_rewards, 'steps': baseline_steps,
            'total_reward': float(np.sum(baseline_rewards)),
            'mean_reward': float(np.mean(baseline_rewards)),
            'mean_steps': float(np.mean(baseline_steps)),
            'final_100_mean': float(np.mean(baseline_rewards[-100:])),
            'final_100_std': float(np.std(baseline_rewards[-100:])),
            'final_100_max': float(np.max(baseline_rewards[-100:]))
        }
        save_intermediate_results(results)

    # ===== 2. Static QBound + DQN =====
    print("\n" + "=" * 80)
    print("METHOD 2: Static QBound + DQN")
    print("=" * 80)

    if is_method_completed(results, 'static_qbound'):
        print("⏭️  Already completed, skipping...")
    else:
        static_agent = DQNAgent(
            state_dim=state_dim, action_dim=action_dim, learning_rate=LR,
            gamma=GAMMA, epsilon_start=EPSILON_START, epsilon_end=EPSILON_END,
            epsilon_decay=EPSILON_DECAY, buffer_size=BUFFER_SIZE,
            batch_size=BATCH_SIZE, target_update_freq=TARGET_UPDATE_FREQ,
            use_qclip=True, qclip_min=QBOUND_MIN, qclip_max=QBOUND_MAX,
            device='cpu'
        )
        static_rewards, static_steps = train_agent(
            env, static_agent, "2. Static QBound + DQN", use_step_aware=False)
        results['training']['static_qbound'] = {
            'rewards': static_rewards, 'steps': static_steps,
            'total_reward': float(np.sum(static_rewards)),
            'mean_reward': float(np.mean(static_rewards)),
            'mean_steps': float(np.mean(static_steps)),
            'final_100_mean': float(np.mean(static_rewards[-100:])),
            'final_100_std': float(np.std(static_rewards[-100:])),
            'final_100_max': float(np.max(static_rewards[-100:]))
        }
        save_intermediate_results(results)

    # ===== 3. Dynamic QBound + DQN =====
    print("\n" + "=" * 80)
    print("METHOD 3: Dynamic QBound + DQN")
    print("=" * 80)
    print("Note: This is included for completeness, but dynamic bounds")
    print("      are less meaningful for shaped rewards like LunarLander.")

    if is_method_completed(results, 'dynamic_qbound'):
        print("⏭️  Already completed, skipping...")
    else:
        dynamic_agent = DQNAgent(
            state_dim=state_dim, action_dim=action_dim, learning_rate=LR,
            gamma=GAMMA, epsilon_start=EPSILON_START, epsilon_end=EPSILON_END,
            epsilon_decay=EPSILON_DECAY, buffer_size=BUFFER_SIZE,
            batch_size=BATCH_SIZE, target_update_freq=TARGET_UPDATE_FREQ,
            use_qclip=True, qclip_min=QBOUND_MIN, qclip_max=QBOUND_MAX,
            use_step_aware_qbound=True, max_episode_steps=MAX_STEPS,
            step_reward=1.0, reward_is_negative=False, device='cpu'
        )
        dynamic_rewards, dynamic_steps = train_agent(
            env, dynamic_agent, "3. Dynamic QBound + DQN", use_step_aware=True)
        results['training']['dynamic_qbound'] = {
            'rewards': dynamic_rewards, 'steps': dynamic_steps,
            'total_reward': float(np.sum(dynamic_rewards)),
            'mean_reward': float(np.mean(dynamic_rewards)),
            'mean_steps': float(np.mean(dynamic_steps)),
            'final_100_mean': float(np.mean(dynamic_rewards[-100:])),
            'final_100_std': float(np.std(dynamic_rewards[-100:])),
            'final_100_max': float(np.max(dynamic_rewards[-100:]))
        }
        save_intermediate_results(results)

    # ===== 4. Baseline DDQN =====
    print("\n" + "=" * 80)
    print("METHOD 4: Baseline DDQN")
    print("=" * 80)

    if is_method_completed(results, 'baseline_ddqn'):
        print("⏭️  Already completed, skipping...")
    else:
        baseline_ddqn_agent = DoubleDQNAgent(
            state_dim=state_dim, action_dim=action_dim, learning_rate=LR,
            gamma=GAMMA, epsilon_start=EPSILON_START, epsilon_end=EPSILON_END,
            epsilon_decay=EPSILON_DECAY, buffer_size=BUFFER_SIZE,
            batch_size=BATCH_SIZE, target_update_freq=TARGET_UPDATE_FREQ,
            use_qclip=False, device='cpu'
        )
        baseline_ddqn_rewards, baseline_ddqn_steps = train_agent(
            env, baseline_ddqn_agent, "4. Baseline DDQN", use_step_aware=False)
        results['training']['baseline_ddqn'] = {
            'rewards': baseline_ddqn_rewards, 'steps': baseline_ddqn_steps,
            'total_reward': float(np.sum(baseline_ddqn_rewards)),
            'mean_reward': float(np.mean(baseline_ddqn_rewards)),
            'mean_steps': float(np.mean(baseline_ddqn_steps)),
            'final_100_mean': float(np.mean(baseline_ddqn_rewards[-100:])),
            'final_100_std': float(np.std(baseline_ddqn_rewards[-100:])),
            'final_100_max': float(np.max(baseline_ddqn_rewards[-100:]))
        }
        save_intermediate_results(results)

    # ===== 5. Static QBound + DDQN =====
    print("\n" + "=" * 80)
    print("METHOD 5: Static QBound + DDQN")
    print("=" * 80)

    if is_method_completed(results, 'static_qbound_ddqn'):
        print("⏭️  Already completed, skipping...")
    else:
        static_ddqn_agent = DoubleDQNAgent(
            state_dim=state_dim, action_dim=action_dim, learning_rate=LR,
            gamma=GAMMA, epsilon_start=EPSILON_START, epsilon_end=EPSILON_END,
            epsilon_decay=EPSILON_DECAY, buffer_size=BUFFER_SIZE,
            batch_size=BATCH_SIZE, target_update_freq=TARGET_UPDATE_FREQ,
            use_qclip=True, qclip_min=QBOUND_MIN, qclip_max=QBOUND_MAX,
            device='cpu'
        )
        static_ddqn_rewards, static_ddqn_steps = train_agent(
            env, static_ddqn_agent, "5. Static QBound + DDQN", use_step_aware=False)
        results['training']['static_qbound_ddqn'] = {
            'rewards': static_ddqn_rewards, 'steps': static_ddqn_steps,
            'total_reward': float(np.sum(static_ddqn_rewards)),
            'mean_reward': float(np.mean(static_ddqn_rewards)),
            'mean_steps': float(np.mean(static_ddqn_steps)),
            'final_100_mean': float(np.mean(static_ddqn_rewards[-100:])),
            'final_100_std': float(np.std(static_ddqn_rewards[-100:])),
            'final_100_max': float(np.max(static_ddqn_rewards[-100:]))
        }
        save_intermediate_results(results)

    # ===== 6. Dynamic QBound + DDQN =====
    print("\n" + "=" * 80)
    print("METHOD 6: Dynamic QBound + DDQN")
    print("=" * 80)
    print("Note: This is included for completeness, but dynamic bounds")
    print("      are less meaningful for shaped rewards like LunarLander.")

    if is_method_completed(results, 'dynamic_qbound_ddqn'):
        print("⏭️  Already completed, skipping...")
    else:
        dynamic_ddqn_agent = DoubleDQNAgent(
            state_dim=state_dim, action_dim=action_dim, learning_rate=LR,
            gamma=GAMMA, epsilon_start=EPSILON_START, epsilon_end=EPSILON_END,
            epsilon_decay=EPSILON_DECAY, buffer_size=BUFFER_SIZE,
            batch_size=BATCH_SIZE, target_update_freq=TARGET_UPDATE_FREQ,
            use_qclip=True, qclip_min=QBOUND_MIN, qclip_max=QBOUND_MAX,
            use_step_aware_qbound=True, max_episode_steps=MAX_STEPS,
            step_reward=1.0, reward_is_negative=False, device='cpu'
        )
        dynamic_ddqn_rewards, dynamic_ddqn_steps = train_agent(
            env, dynamic_ddqn_agent, "6. Dynamic QBound + DDQN", use_step_aware=True)
        results['training']['dynamic_qbound_ddqn'] = {
            'rewards': dynamic_ddqn_rewards, 'steps': dynamic_ddqn_steps,
            'total_reward': float(np.sum(dynamic_ddqn_rewards)),
            'mean_reward': float(np.mean(dynamic_ddqn_rewards)),
            'mean_steps': float(np.mean(dynamic_ddqn_steps)),
            'final_100_mean': float(np.mean(dynamic_ddqn_rewards[-100:])),
            'final_100_std': float(np.std(dynamic_ddqn_rewards[-100:])),
            'final_100_max': float(np.max(dynamic_ddqn_rewards[-100:]))
        }
        save_intermediate_results(results)

    # ===== Analysis and Summary =====
    print("\n" + "=" * 80)
    print("LUNARLANDER RESULTS SUMMARY (Final 100 Episodes)")
    print("=" * 80)

    methods = ['baseline', 'static_qbound', 'dynamic_qbound',
               'baseline_ddqn', 'static_qbound_ddqn', 'dynamic_qbound_ddqn']
    labels = ['Baseline DQN', 'Static QBound', 'Dynamic QBound',
              'Baseline DDQN', 'Static QBound+DDQN', 'Dynamic QBound+DDQN']

    print(f"\n{'Method':<25} {'Mean ± Std':<30} {'Max':<12} {'vs Baseline':<15}")
    print("-" * 85)

    baseline_mean = results['training']['baseline']['final_100_mean']

    for method, label in zip(methods, labels):
        data = results['training'][method]
        mean = data['final_100_mean']
        std = data['final_100_std']
        max_reward = data['final_100_max']

        improvement = ((mean - baseline_mean) / abs(baseline_mean)) * 100 if baseline_mean != 0 else 0
        improvement_str = f"{improvement:+.1f}%"

        print(f"{label:<25} {mean:>8.2f} ± {std:<12.2f} {max_reward:>8.2f}   {improvement_str:>10}")

    # Calculate success rates (reward > 200 = solved)
    print("\n" + "=" * 80)
    print("SUCCESS RATE (Reward > 200 in final 100 episodes)")
    print("=" * 80)

    for method, label in zip(methods, labels):
        rewards = results['training'][method]['rewards'][-100:]
        success_rate = sum(1 for r in rewards if r > 200) / len(rewards) * 100
        print(f"{label:<25} {success_rate:>6.1f}%")

    # ===== Save Final Results =====
    timestamp = results['timestamp']
    final_output_file = f"/root/projects/QBound/results/lunarlander/6way_comparison_{timestamp}.json"
    os.makedirs('/root/projects/QBound/results/lunarlander', exist_ok=True)

    with open(final_output_file, 'w') as f:
        json.dump(results, f, indent=2)

    if os.path.exists(RESULTS_FILE):
        os.remove(RESULTS_FILE)

    print(f"\n✓ Results saved to: {final_output_file}")
    print("\n" + "=" * 80)
    print("LunarLander 6-Way Comparison Complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
