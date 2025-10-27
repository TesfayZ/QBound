"""
DDPG vs QBound-DDPG Comparison on Pendulum-v1

Tests whether DDPG benefits from QBound critic bounds.
Pendulum has dense negative rewards, testing if bounds help or hurt.
"""

import sys
sys.path.insert(0, '/root/projects/QBound/src')

import gymnasium as gym
import numpy as np
import torch
import json
import random
from datetime import datetime
from ddpg_agent import DDPGAgent
from tqdm import tqdm


# Reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

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

# QBound parameters for Pendulum
# Reward range: approximately [-16.27, 0]
# Q_max = 0 (best case: perfect balance from start)
# Q_min = -16.27 * sum(gamma^k for k in 0..199) = -16.27 * (1-gamma^200)/(1-gamma)
# With gamma=0.99: Q_min ≈ -16.27 * 99.34 ≈ -1616
QBOUND_MIN = -1616.0
QBOUND_MAX = 0.0


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
    print("Pendulum-v1: DDPG vs. QBound-DDPG Comparison")
    print("=" * 80)
    print("Testing whether Q-value bounds help DDPG on continuous control.")
    print("=" * 80)

    print(f"\nConfiguration:")
    print(f"  Environment: {ENV_NAME}")
    print(f"  Episodes: {MAX_EPISODES}")
    print(f"  Max steps per episode: {MAX_STEPS}")
    print(f"  Discount factor γ: {GAMMA}")
    print(f"  Learning rates: {LR_ACTOR} (actor), {LR_CRITIC} (critic)")
    print(f"  QBound range: [{QBOUND_MIN:.2f}, {QBOUND_MAX:.2f}]")
    print("=" * 80)

    # Create environment
    env = gym.make(ENV_NAME)
    env.reset(seed=SEED)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

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

    # ===== Baseline DDPG =====
    baseline_agent = DDPGAgent(
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

    baseline_rewards = train_agent(env, baseline_agent, "Baseline DDPG")
    results['training']['baseline'] = {
        'rewards': baseline_rewards,
        'total_reward': float(np.sum(baseline_rewards)),
        'mean_reward': float(np.mean(baseline_rewards))
    }

    # ===== QBound-DDPG =====
    qbound_agent = DDPGAgent(
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

    qbound_rewards = train_agent(env, qbound_agent, "QBound-DDPG")
    results['training']['qbound'] = {
        'rewards': qbound_rewards,
        'total_reward': float(np.sum(qbound_rewards)),
        'mean_reward': float(np.mean(qbound_rewards))
    }

    # ===== Evaluation =====
    print("\n" + "=" * 80)
    print("Final Evaluation")
    print("=" * 80)

    baseline_eval_mean, baseline_eval_std = evaluate_agent(env, baseline_agent, EVAL_EPISODES)
    qbound_eval_mean, qbound_eval_std = evaluate_agent(env, qbound_agent, EVAL_EPISODES)

    print(f"\n>>> Evaluating agents ({EVAL_EPISODES} episodes)...")
    print(f"  Baseline DDPG: {baseline_eval_mean:.2f} ± {baseline_eval_std:.2f}")
    print(f"  QBound-DDPG:   {qbound_eval_mean:.2f} ± {qbound_eval_std:.2f}")

    results['evaluation'] = {
        'baseline': {
            'mean': float(baseline_eval_mean),
            'std': float(baseline_eval_std)
        },
        'qbound': {
            'mean': float(qbound_eval_mean),
            'std': float(qbound_eval_std)
        }
    }

    # ===== Summary =====
    print("\n" + "=" * 80)
    print("Training Results Summary")
    print("=" * 80)

    baseline_total = results['training']['baseline']['total_reward']
    qbound_total = results['training']['qbound']['total_reward']
    improvement = ((qbound_total - baseline_total) / abs(baseline_total)) * 100

    print(f"\nTotal cumulative reward:")
    print(f"  Baseline DDPG: {baseline_total:.0f}")
    print(f"  QBound-DDPG:   {qbound_total:.0f} ({improvement:+.1f}%)")

    print(f"\nAverage episode reward:")
    print(f"  Baseline DDPG: {results['training']['baseline']['mean_reward']:.2f}")
    print(f"  QBound-DDPG:   {results['training']['qbound']['mean_reward']:.2f}")

    # Save results
    timestamp = results['timestamp']
    output_file = f"/root/projects/QBound/results/pendulum/ddpg_comparison_{timestamp}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to: {output_file}")

    print("\n" + "=" * 80)
    print("Pendulum DDPG Comparison Complete!")
    print("=" * 80)

    # Analysis
    if improvement > 5:
        print("\n🎯 CONCLUSION:")
        print("✅ QBound helps DDPG on Pendulum (dense negative rewards)")
    elif improvement < -5:
        print("\n🎯 CONCLUSION:")
        print("⚠️  QBound hurts DDPG on Pendulum - bounds may be too restrictive")
    else:
        print("\n🎯 CONCLUSION:")
        print("➖ QBound has minimal impact on DDPG for Pendulum")

    env.close()


if __name__ == "__main__":
    import random
    main()
