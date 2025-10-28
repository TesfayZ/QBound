"""
LunarLander-v3: Dueling DQN 4-Way Comparison
Validates that QBound generalizes to architecturally different DQN variants

Architecture: Dueling DQN with separate value V(s) and advantage A(s,a) streams
Environment: LunarLander-v3 (discrete actions, shaped rewards)

Comparison:
1. Baseline Dueling DQN - No QBound, no Double-Q
2. QBound + Dueling DQN - Q ∈ [-100, 200]
3. Double Dueling DQN - No QBound, with Double-Q
4. QBound + Double Dueling DQN - Q ∈ [-100, 200] + Double-Q

Purpose: Demonstrate QBound works with architecturally different networks (not just standard DQN)
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
from dueling_dqn_agent import DuelingDQNAgent
from tqdm import tqdm


# Reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

# Results file for crash recovery
RESULTS_FILE = "/root/projects/QBound/results/lunarlander/dueling_4way_in_progress.json"

# Environment parameters
ENV_NAME = "LunarLander-v3"
MAX_EPISODES = 500
MAX_STEPS = 1000
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

# QBound parameters (same as standard DQN experiments)
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
    print("LunarLander-v3: Dueling DQN 4-Way Comparison")
    print("Validating QBound Generalization to Different Architectures")
    print("=" * 80)
    print("Architecture: Dueling DQN (separate V(s) and A(s,a) streams)")
    print("=" * 80)

    print(f"\nConfiguration:")
    print(f"  Environment: {ENV_NAME}")
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
            'architecture': 'dueling_dqn',
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
        dueling_rewards, dueling_steps = train_agent(env, dueling_dqn_agent, "1. Baseline Dueling DQN")
        results['training']['dueling_dqn'] = {
            'rewards': dueling_rewards, 'steps': dueling_steps,
            'total_reward': float(np.sum(dueling_rewards)),
            'mean_reward': float(np.mean(dueling_rewards)),
            'mean_steps': float(np.mean(dueling_steps)),
            'final_100_mean': float(np.mean(dueling_rewards[-100:])),
            'final_100_std': float(np.std(dueling_rewards[-100:])),
            'final_100_max': float(np.max(dueling_rewards[-100:]))
        }
        save_intermediate_results(results)

    # ===== 2. QBound + Dueling DQN =====
    print("\n" + "=" * 80)
    print("METHOD 2: QBound + Dueling DQN")
    print("=" * 80)

    if is_method_completed(results, 'qbound_dueling_dqn'):
        print("⏭️  Already completed, skipping...")
    else:
        qbound_dueling_agent = DuelingDQNAgent(
            state_dim=state_dim, action_dim=action_dim, learning_rate=LR,
            gamma=GAMMA, epsilon_start=EPSILON_START, epsilon_end=EPSILON_END,
            epsilon_decay=EPSILON_DECAY, buffer_size=BUFFER_SIZE,
            batch_size=BATCH_SIZE, target_update_freq=TARGET_UPDATE_FREQ,
            use_qclip=True, qclip_min=QBOUND_MIN, qclip_max=QBOUND_MAX,
            use_double_dqn=False, device='cpu'
        )
        qbound_dueling_rewards, qbound_dueling_steps = train_agent(
            env, qbound_dueling_agent, "2. QBound + Dueling DQN")
        results['training']['qbound_dueling_dqn'] = {
            'rewards': qbound_dueling_rewards, 'steps': qbound_dueling_steps,
            'total_reward': float(np.sum(qbound_dueling_rewards)),
            'mean_reward': float(np.mean(qbound_dueling_rewards)),
            'mean_steps': float(np.mean(qbound_dueling_steps)),
            'final_100_mean': float(np.mean(qbound_dueling_rewards[-100:])),
            'final_100_std': float(np.std(qbound_dueling_rewards[-100:])),
            'final_100_max': float(np.max(qbound_dueling_rewards[-100:]))
        }
        save_intermediate_results(results)

    # ===== 3. Double Dueling DQN =====
    print("\n" + "=" * 80)
    print("METHOD 3: Double Dueling DQN")
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
        double_dueling_rewards, double_dueling_steps = train_agent(
            env, double_dueling_agent, "3. Double Dueling DQN")
        results['training']['double_dueling_dqn'] = {
            'rewards': double_dueling_rewards, 'steps': double_dueling_steps,
            'total_reward': float(np.sum(double_dueling_rewards)),
            'mean_reward': float(np.mean(double_dueling_rewards)),
            'mean_steps': float(np.mean(double_dueling_steps)),
            'final_100_mean': float(np.mean(double_dueling_rewards[-100:])),
            'final_100_std': float(np.std(double_dueling_rewards[-100:])),
            'final_100_max': float(np.max(double_dueling_rewards[-100:]))
        }
        save_intermediate_results(results)

    # ===== 4. QBound + Double Dueling DQN =====
    print("\n" + "=" * 80)
    print("METHOD 4: QBound + Double Dueling DQN")
    print("=" * 80)

    if is_method_completed(results, 'qbound_double_dueling_dqn'):
        print("⏭️  Already completed, skipping...")
    else:
        qbound_double_dueling_agent = DuelingDQNAgent(
            state_dim=state_dim, action_dim=action_dim, learning_rate=LR,
            gamma=GAMMA, epsilon_start=EPSILON_START, epsilon_end=EPSILON_END,
            epsilon_decay=EPSILON_DECAY, buffer_size=BUFFER_SIZE,
            batch_size=BATCH_SIZE, target_update_freq=TARGET_UPDATE_FREQ,
            use_qclip=True, qclip_min=QBOUND_MIN, qclip_max=QBOUND_MAX,
            use_double_dqn=True, device='cpu'
        )
        qbound_double_dueling_rewards, qbound_double_dueling_steps = train_agent(
            env, qbound_double_dueling_agent, "4. QBound + Double Dueling DQN")
        results['training']['qbound_double_dueling_dqn'] = {
            'rewards': qbound_double_dueling_rewards, 'steps': qbound_double_dueling_steps,
            'total_reward': float(np.sum(qbound_double_dueling_rewards)),
            'mean_reward': float(np.mean(qbound_double_dueling_rewards)),
            'mean_steps': float(np.mean(qbound_double_dueling_steps)),
            'final_100_mean': float(np.mean(qbound_double_dueling_rewards[-100:])),
            'final_100_std': float(np.std(qbound_double_dueling_rewards[-100:])),
            'final_100_max': float(np.max(qbound_double_dueling_rewards[-100:]))
        }
        save_intermediate_results(results)

    # ===== Analysis and Summary =====
    print("\n" + "=" * 80)
    print("DUELING DQN RESULTS SUMMARY (Final 100 Episodes)")
    print("=" * 80)

    methods = ['dueling_dqn', 'qbound_dueling_dqn', 'double_dueling_dqn', 'qbound_double_dueling_dqn']
    labels = ['Baseline Dueling', 'QBound Dueling', 'Double Dueling', 'QBound+Double Dueling']

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

    # Calculate success rates (reward > 200)
    print("\n" + "=" * 80)
    print("SUCCESS RATE (Reward > 200 in final 100 episodes)")
    print("=" * 80)

    for method, label in zip(methods, labels):
        rewards = results['training'][method]['rewards'][-100:]
        success_rate = sum(1 for r in rewards if r > 200) / len(rewards) * 100
        print(f"{label:<30} {success_rate:.1f}%")

    # ===== Save Final Results =====
    timestamp = results['timestamp']
    final_output_file = f"/root/projects/QBound/results/lunarlander/dueling_4way_{timestamp}.json"

    with open(final_output_file, 'w') as f:
        json.dump(results, f, indent=2)

    if os.path.exists(RESULTS_FILE):
        os.remove(RESULTS_FILE)

    print(f"\n✓ Results saved to: {final_output_file}")
    print("\n" + "=" * 80)
    print("Dueling DQN 4-Way Comparison Complete!")
    print("=" * 80)

    print("\n📊 Key Takeaways:")
    print("  - If QBound improves Dueling DQN, it demonstrates architectural generalization")
    print("  - Compare to standard DQN results to assess relative performance")
    print("  - Look for similar improvement patterns as standard DQN experiments")


if __name__ == '__main__':
    main()
