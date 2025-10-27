"""
6-Way Comparison: Complete QBound vs DDPG vs TD3 Analysis

Tests the core hypothesis: Can QBound replace or enhance complex stabilization mechanisms?

Comparison:
1. Standard DDPG - With target networks (no QBound)
2. Standard TD3 - Full complexity (2 critics + target networks + clipped double-Q)
3. Simple DDPG - No target networks (no QBound) - BASELINE INSTABILITY
4. QBound + Simple DDPG - No target networks + QBound - TESTS IF QBOUND REPLACES TARGETS
5. QBound + DDPG - With target networks + QBound - TESTS IF QBOUND ENHANCES DDPG
6. QBound + TD3 - Full TD3 + QBound - TESTS IF QBOUND ENHANCES TD3

Environment: Pendulum-v1 (dense negative rewards)
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
from ddpg_agent import DDPGAgent
from simple_ddpg_agent import SimpleDDPGAgent
from td3_agent import TD3Agent
from tqdm import tqdm


# Reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

# Results file for crash recovery
RESULTS_FILE = "/root/projects/QBound/results/pendulum/6way_comparison_in_progress.json"

# Environment parameters
ENV_NAME = "Pendulum-v1"
MAX_EPISODES = 500
MAX_STEPS = 200
EVAL_EPISODES = 10

# Shared hyperparameters
LR_ACTOR = 0.001
LR_CRITIC = 0.001
GAMMA = 0.99
TAU = 0.005
BATCH_SIZE = 256
WARMUP_EPISODES = 10

# QBound parameters for Pendulum
# Reward range: approximately [-16.27, 0]
# Q_max = 0 (best case: perfect balance from start)
# Q_min = -16.27 * sum(gamma^k for k in 0..199) = -16.27 * (1-gamma^200)/(1-gamma)
# With gamma=0.99: Q_min ≈ -16.27 * 99.34 ≈ -1616
QBOUND_MIN = -1616.0
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


def evaluate_agent(env, agent, num_episodes=10):
    """Evaluate agent performance"""
    total_rewards = []

    for _ in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        step = 0

        while not done and step < MAX_STEPS:
            action = agent.select_action(state, add_noise=False)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            state = next_state
            step += 1

        total_rewards.append(episode_reward)

    return np.mean(total_rewards), np.std(total_rewards)


def train_agent(env, agent, agent_name, max_episodes=MAX_EPISODES):
    """Train agent and return results"""
    print(f"\n>>> Training {agent_name}...")

    episode_rewards = []
    best_reward = -np.inf

    for episode in tqdm(range(max_episodes), desc=agent_name):
        state, _ = env.reset()
        agent.reset_noise()
        episode_reward = 0
        done = False
        step = 0

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
            agent.replay_buffer.push(state, action, reward, next_state, done)

            # Train if enough samples
            if episode >= WARMUP_EPISODES:
                critic_loss, actor_loss = agent.train(batch_size=BATCH_SIZE)

            episode_reward += reward
            state = next_state
            step += 1

        episode_rewards.append(episode_reward)

        # Track best performance
        if episode_reward > best_reward:
            best_reward = episode_reward

        # Progress update
        if (episode + 1) % 100 == 0:
            recent_avg = np.mean(episode_rewards[-100:])
            print(f"  Episode {episode + 1}/{max_episodes} - Recent avg: {recent_avg:.2f}, Best: {best_reward:.2f}")

    return episode_rewards


def main():
    print("=" * 80)
    print("Pendulum-v1: 6-Way Comprehensive Comparison")
    print("=" * 80)
    print("Testing: DDPG vs TD3 vs QBound variants")
    print("=" * 80)

    print(f"\nConfiguration:")
    print(f"  Environment: {ENV_NAME}")
    print(f"  Episodes: {MAX_EPISODES}")
    print(f"  Max steps per episode: {MAX_STEPS}")
    print(f"  Discount factor γ: {GAMMA}")
    print(f"  Learning rates: {LR_ACTOR} (actor), {LR_CRITIC} (critic)")
    print(f"  QBound range: [{QBOUND_MIN:.2f}, {QBOUND_MAX:.2f}]")
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
                'lr_actor': LR_ACTOR,
                'lr_critic': LR_CRITIC,
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
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # ===== 1. Standard DDPG (with target networks, no QBound) =====
    print("\n" + "=" * 80)
    print("METHOD 1: Standard DDPG (with target networks, no QBound)")
    print("=" * 80)

    if is_method_completed(results, 'ddpg'):
        print("⏭️  Already completed, skipping...")
        ddpg_agent = None  # Will be created for evaluation if needed
    else:
        ddpg_agent = DDPGAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            max_action=max_action,
            lr_actor=LR_ACTOR,
            lr_critic=LR_CRITIC,
            gamma=GAMMA,
            tau=TAU,
            use_qbound=False,
            device='cpu'
        )

        ddpg_rewards = train_agent(env, ddpg_agent, "1. Standard DDPG")
        results['training']['ddpg'] = {
            'rewards': ddpg_rewards,
            'total_reward': float(np.sum(ddpg_rewards)),
            'mean_reward': float(np.mean(ddpg_rewards))
        }
        save_intermediate_results(results)

    # ===== 2. Standard TD3 (full complexity, no QBound) =====
    print("\n" + "=" * 80)
    print("METHOD 2: Standard TD3 (2 critics + target networks + clipped double-Q)")
    print("=" * 80)

    if is_method_completed(results, 'td3'):
        print("⏭️  Already completed, skipping...")
        td3_agent = None
    else:
        td3_agent = TD3Agent(
            state_dim=state_dim,
            action_dim=action_dim,
            max_action=max_action,
            lr_actor=LR_ACTOR,
            lr_critic=LR_CRITIC,
            gamma=GAMMA,
            tau=TAU,
            use_qbound=False,
            device='cpu'
        )

        td3_rewards = train_agent(env, td3_agent, "2. Standard TD3")
        results['training']['td3'] = {
            'rewards': td3_rewards,
            'total_reward': float(np.sum(td3_rewards)),
            'mean_reward': float(np.mean(td3_rewards))
        }
        save_intermediate_results(results)

    # ===== 3. Simple DDPG (no target networks, no QBound) =====
    print("\n" + "=" * 80)
    print("METHOD 3: Simple DDPG (NO target networks, NO QBound) - BASELINE")
    print("=" * 80)

    if is_method_completed(results, 'simple_ddpg'):
        print("⏭️  Already completed, skipping...")
        simple_ddpg_agent = None
    else:
        simple_ddpg_agent = SimpleDDPGAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            max_action=max_action,
            lr_actor=LR_ACTOR,
            lr_critic=LR_CRITIC,
            gamma=GAMMA,
            use_qbound=False,
            device='cpu'
        )

        simple_ddpg_rewards = train_agent(env, simple_ddpg_agent, "3. Simple DDPG")
        results['training']['simple_ddpg'] = {
            'rewards': simple_ddpg_rewards,
            'total_reward': float(np.sum(simple_ddpg_rewards)),
            'mean_reward': float(np.mean(simple_ddpg_rewards))
        }
        save_intermediate_results(results)

    # ===== 4. QBound + Simple DDPG (no target networks + QBound) =====
    print("\n" + "=" * 80)
    print("METHOD 4: QBound + Simple DDPG (NO target networks + QBound)")
    print("Testing: Can QBound REPLACE target networks?")
    print("=" * 80)

    if is_method_completed(results, 'qbound_simple'):
        print("⏭️  Already completed, skipping...")
        qbound_simple_agent = None
    else:
        qbound_simple_agent = SimpleDDPGAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            max_action=max_action,
            lr_actor=LR_ACTOR,
            lr_critic=LR_CRITIC,
            gamma=GAMMA,
            use_qbound=True,
            qbound_min=QBOUND_MIN,
            qbound_max=QBOUND_MAX,
            device='cpu'
        )

        qbound_simple_rewards = train_agent(env, qbound_simple_agent, "4. QBound + Simple DDPG")
        results['training']['qbound_simple'] = {
            'rewards': qbound_simple_rewards,
            'total_reward': float(np.sum(qbound_simple_rewards)),
            'mean_reward': float(np.mean(qbound_simple_rewards))
        }
        save_intermediate_results(results)

    # ===== 5. QBound + Standard DDPG (with target networks + QBound) =====
    print("\n" + "=" * 80)
    print("METHOD 5: QBound + Standard DDPG (with target networks + QBound)")
    print("Testing: Can QBound ENHANCE standard DDPG?")
    print("=" * 80)

    if is_method_completed(results, 'qbound_ddpg'):
        print("⏭️  Already completed, skipping...")
        qbound_ddpg_agent = None
    else:
        qbound_ddpg_agent = DDPGAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            max_action=max_action,
            lr_actor=LR_ACTOR,
            lr_critic=LR_CRITIC,
            gamma=GAMMA,
            tau=TAU,
            use_qbound=True,
            qbound_min=QBOUND_MIN,
            qbound_max=QBOUND_MAX,
            device='cpu'
        )

        qbound_ddpg_rewards = train_agent(env, qbound_ddpg_agent, "5. QBound + DDPG")
        results['training']['qbound_ddpg'] = {
            'rewards': qbound_ddpg_rewards,
            'total_reward': float(np.sum(qbound_ddpg_rewards)),
            'mean_reward': float(np.mean(qbound_ddpg_rewards))
        }
        save_intermediate_results(results)

    # ===== 6. QBound + TD3 (full complexity + QBound) =====
    print("\n" + "=" * 80)
    print("METHOD 6: QBound + TD3 (full TD3 + QBound)")
    print("Testing: Can QBound ENHANCE the most advanced method?")
    print("=" * 80)

    if is_method_completed(results, 'qbound_td3'):
        print("⏭️  Already completed, skipping...")
        qbound_td3_agent = None
    else:
        qbound_td3_agent = TD3Agent(
            state_dim=state_dim,
            action_dim=action_dim,
            max_action=max_action,
            lr_actor=LR_ACTOR,
            lr_critic=LR_CRITIC,
            gamma=GAMMA,
            tau=TAU,
            use_qbound=True,
            qbound_min=QBOUND_MIN,
            qbound_max=QBOUND_MAX,
            device='cpu'
        )

        qbound_td3_rewards = train_agent(env, qbound_td3_agent, "6. QBound + TD3")
        results['training']['qbound_td3'] = {
            'rewards': qbound_td3_rewards,
            'total_reward': float(np.sum(qbound_td3_rewards)),
            'mean_reward': float(np.mean(qbound_td3_rewards))
        }
        save_intermediate_results(results)

    # ===== Evaluation =====
    print("\n" + "=" * 80)
    print("Final Evaluation")
    print("=" * 80)

    # Only evaluate agents that were just trained (we don't save models, so can't reload)
    agents_to_eval = {
        'ddpg': ddpg_agent,
        'td3': td3_agent,
        'simple_ddpg': simple_ddpg_agent,
        'qbound_simple': qbound_simple_agent,
        'qbound_ddpg': qbound_ddpg_agent,
        'qbound_td3': qbound_td3_agent
    }

    # Check if we have all agents (no skipped methods)
    if all(agent is not None for agent in agents_to_eval.values()):
        print(f"\n>>> Evaluating all agents ({EVAL_EPISODES} episodes)...")

        ddpg_eval_mean, ddpg_eval_std = evaluate_agent(env, ddpg_agent, EVAL_EPISODES)
        td3_eval_mean, td3_eval_std = evaluate_agent(env, td3_agent, EVAL_EPISODES)
        simple_ddpg_eval_mean, simple_ddpg_eval_std = evaluate_agent(env, simple_ddpg_agent, EVAL_EPISODES)
        qbound_simple_eval_mean, qbound_simple_eval_std = evaluate_agent(env, qbound_simple_agent, EVAL_EPISODES)
        qbound_ddpg_eval_mean, qbound_ddpg_eval_std = evaluate_agent(env, qbound_ddpg_agent, EVAL_EPISODES)
        qbound_td3_eval_mean, qbound_td3_eval_std = evaluate_agent(env, qbound_td3_agent, EVAL_EPISODES)

        print(f"  1. Standard DDPG:        {ddpg_eval_mean:.2f} ± {ddpg_eval_std:.2f}")
        print(f"  2. Standard TD3:         {td3_eval_mean:.2f} ± {td3_eval_std:.2f}")
        print(f"  3. Simple DDPG:          {simple_ddpg_eval_mean:.2f} ± {simple_ddpg_eval_std:.2f}")
        print(f"  4. QBound + Simple DDPG: {qbound_simple_eval_mean:.2f} ± {qbound_simple_eval_std:.2f}")
        print(f"  5. QBound + DDPG:        {qbound_ddpg_eval_mean:.2f} ± {qbound_ddpg_eval_std:.2f}")
        print(f"  6. QBound + TD3:         {qbound_td3_eval_mean:.2f} ± {qbound_td3_eval_std:.2f}")

        results['evaluation'] = {
            'ddpg': {'mean': float(ddpg_eval_mean), 'std': float(ddpg_eval_std)},
            'td3': {'mean': float(td3_eval_mean), 'std': float(td3_eval_std)},
            'simple_ddpg': {'mean': float(simple_ddpg_eval_mean), 'std': float(simple_ddpg_eval_std)},
            'qbound_simple': {'mean': float(qbound_simple_eval_mean), 'std': float(qbound_simple_eval_std)},
            'qbound_ddpg': {'mean': float(qbound_ddpg_eval_mean), 'std': float(qbound_ddpg_eval_std)},
            'qbound_td3': {'mean': float(qbound_td3_eval_mean), 'std': float(qbound_td3_eval_std)}
        }
        save_intermediate_results(results)
    else:
        print("\n⚠️  Skipping evaluation - some methods were trained in a previous session")
        print("    (Model weights not saved, cannot reload for evaluation)")
        if 'evaluation' not in results:
            results['evaluation'] = {}

    # ===== Summary =====
    print("\n" + "=" * 80)
    print("Training Results Summary")
    print("=" * 80)

    print(f"\nTotal cumulative reward:")
    print(f"  1. Standard DDPG:        {results['training']['ddpg']['total_reward']:.0f}")
    print(f"  2. Standard TD3:         {results['training']['td3']['total_reward']:.0f}")
    print(f"  3. Simple DDPG:          {results['training']['simple_ddpg']['total_reward']:.0f}")
    print(f"  4. QBound + Simple DDPG: {results['training']['qbound_simple']['total_reward']:.0f}")
    print(f"  5. QBound + DDPG:        {results['training']['qbound_ddpg']['total_reward']:.0f}")
    print(f"  6. QBound + TD3:         {results['training']['qbound_td3']['total_reward']:.0f}")

    print(f"\nAverage episode reward:")
    print(f"  1. Standard DDPG:        {results['training']['ddpg']['mean_reward']:.2f}")
    print(f"  2. Standard TD3:         {results['training']['td3']['mean_reward']:.2f}")
    print(f"  3. Simple DDPG:          {results['training']['simple_ddpg']['mean_reward']:.2f}")
    print(f"  4. QBound + Simple DDPG: {results['training']['qbound_simple']['mean_reward']:.2f}")
    print(f"  5. QBound + DDPG:        {results['training']['qbound_ddpg']['mean_reward']:.2f}")
    print(f"  6. QBound + TD3:         {results['training']['qbound_td3']['mean_reward']:.2f}")

    # ===== Key Comparisons =====
    print("\n" + "=" * 80)
    print("KEY COMPARISONS")
    print("=" * 80)

    # Q1: Can QBound replace target networks?
    simple_total = results['training']['simple_ddpg']['total_reward']
    qbound_simple_total = results['training']['qbound_simple']['total_reward']
    improvement_1 = ((qbound_simple_total - simple_total) / abs(simple_total)) * 100

    print(f"\nQ1: Can QBound REPLACE target networks?")
    print(f"    QBound+Simple DDPG vs Simple DDPG: {improvement_1:+.1f}%")
    if improvement_1 > 10:
        print(f"    ✅ YES! QBound significantly stabilizes learning without target networks")
    elif improvement_1 > 5:
        print(f"    ✅ YES! QBound helps stabilize learning without target networks")
    elif improvement_1 > -5:
        print(f"    ➖ NEUTRAL: QBound has minimal impact")
    else:
        print(f"    ❌ NO: Target networks still needed")

    # Q2: Can QBound enhance standard DDPG?
    ddpg_total = results['training']['ddpg']['total_reward']
    qbound_ddpg_total = results['training']['qbound_ddpg']['total_reward']
    improvement_2 = ((qbound_ddpg_total - ddpg_total) / abs(ddpg_total)) * 100

    print(f"\nQ2: Can QBound ENHANCE standard DDPG?")
    print(f"    QBound+DDPG vs Standard DDPG: {improvement_2:+.1f}%")
    if improvement_2 > 5:
        print(f"    ✅ YES! QBound improves DDPG with target networks")
    elif improvement_2 > -5:
        print(f"    ➖ NEUTRAL: QBound doesn't hurt but doesn't help much")
    else:
        print(f"    ❌ NO: QBound hurts DDPG performance")

    # Q3: Can QBound enhance TD3?
    td3_total = results['training']['td3']['total_reward']
    qbound_td3_total = results['training']['qbound_td3']['total_reward']
    improvement_3 = ((qbound_td3_total - td3_total) / abs(td3_total)) * 100

    print(f"\nQ3: Can QBound ENHANCE TD3?")
    print(f"    QBound+TD3 vs Standard TD3: {improvement_3:+.1f}%")
    if improvement_3 > 5:
        print(f"    ✅ YES! QBound improves even the most advanced method")
    elif improvement_3 > -5:
        print(f"    ➖ NEUTRAL: TD3 already well-stabilized")
    else:
        print(f"    ❌ NO: QBound conflicts with TD3's mechanisms")

    # Best overall method
    best_method = max(
        [('1. Standard DDPG', ddpg_total),
         ('2. Standard TD3', td3_total),
         ('3. Simple DDPG', simple_total),
         ('4. QBound + Simple DDPG', qbound_simple_total),
         ('5. QBound + DDPG', qbound_ddpg_total),
         ('6. QBound + TD3', qbound_td3_total)],
        key=lambda x: x[1]
    )

    print(f"\n🎯 BEST OVERALL METHOD: {best_method[0]} (total reward: {best_method[1]:.0f})")

    # Save final results (rename from in_progress to timestamped)
    timestamp = results['timestamp']
    final_output_file = f"/root/projects/QBound/results/pendulum/6way_comparison_{timestamp}.json"

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
