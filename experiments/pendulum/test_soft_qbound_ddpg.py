"""
Test Soft QBound DDPG on Pendulum-v1

Pendulum-v1 is a continuous control environment with:
- Reward range: approximately [-16.3, 0] (negative, shaped reward)
- Goal: Swing pendulum upright and keep it balanced
- Action space: Continuous torque [-2, 2]

This test compares:
1. Baseline DDPG (no QBound)
2. Hard QBound DDPG (hard clipping - EXPECTED TO FAIL)
3. Soft QBound DDPG (smooth penalties - SHOULD WORK!)

Expected Results:
- Baseline: Works normally
- Hard QBound: Fails due to zero gradients
- Soft QBound: Works as well or better than baseline!
"""

import sys
sys.path.insert(0, '/root/projects/QBound/src')

import numpy as np
import torch
import gymnasium as gym
import random
import json
import os
from datetime import datetime
from tqdm import tqdm

from ddpg_agent import DDPGAgent
from soft_qbound_ddpg_agent import SoftQBoundDDPGAgent


# Reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

# Environment parameters
ENV_NAME = "Pendulum-v1"
MAX_EPISODES = 200  # Quick test
MAX_STEPS = 200

# DDPG hyperparameters
LR_ACTOR = 0.001
LR_CRITIC = 0.001
GAMMA = 0.99
TAU = 0.005
BATCH_SIZE = 128

# QBound parameters for Pendulum
# Reward range: approximately [-16.3, 0]
# Max episode return: ~0 (ideal), worst: ~-1630
# Conservative Q-bounds for episode return
QBOUND_MIN = -1630.0  # Worst possible episode return
QBOUND_MAX = 0.0      # Best possible episode return

# Soft QBound parameters
PENALTY_WEIGHT = 0.1  # λ in loss = TD_loss + λ * penalty
PENALTY_TYPE = 'quadratic'  # 'quadratic', 'huber', or 'exponential'


def train_ddpg(agent, env, agent_name, max_episodes=MAX_EPISODES):
    """Train DDPG agent and return results"""
    print(f"\n>>> Training {agent_name}...")

    episode_rewards = []
    critic_losses = []
    actor_losses = []
    penalties = [] if hasattr(agent, 'get_penalty_stats') else None

    for episode in tqdm(range(max_episodes), desc=agent_name):
        state, _ = env.reset(seed=SEED + episode)
        agent.reset_noise()

        episode_reward = 0
        episode_critic_loss = []
        episode_actor_loss = []

        for step in range(MAX_STEPS):
            # Select action
            action = agent.select_action(state, add_noise=True)

            # Take step
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Store transition
            agent.replay_buffer.push(state, action, reward, next_state, done)

            # Train
            train_result = agent.train(batch_size=BATCH_SIZE)
            if train_result[0] is not None:
                if len(train_result) == 3:  # Soft QBound returns penalty too
                    c_loss, a_loss, penalty = train_result
                    episode_critic_loss.append(c_loss)
                    episode_actor_loss.append(a_loss)
                else:  # Hard QBound or baseline
                    c_loss, a_loss = train_result
                    episode_critic_loss.append(c_loss)
                    episode_actor_loss.append(a_loss)

            episode_reward += reward
            state = next_state

            if done:
                break

        episode_rewards.append(episode_reward)
        if episode_critic_loss:
            critic_losses.append(np.mean(episode_critic_loss))
            actor_losses.append(np.mean(episode_actor_loss))

        # Track penalties for soft QBound
        if penalties is not None and hasattr(agent, 'get_penalty_stats'):
            penalty_stats = agent.get_penalty_stats()
            penalties.append(penalty_stats['mean'])

        # Progress update
        if (episode + 1) % 50 == 0:
            recent_avg = np.mean(episode_rewards[-50:])
            recent_critic = np.mean(critic_losses[-50:]) if critic_losses else 0
            recent_actor = np.mean(actor_losses[-50:]) if actor_losses else 0
            print(f"  Episode {episode + 1}/{max_episodes} - "
                  f"Avg reward: {recent_avg:.1f}, "
                  f"Critic loss: {recent_critic:.4f}, "
                  f"Actor loss: {recent_actor:.4f}")

            if penalties is not None:
                recent_penalty = np.mean(penalties[-50:]) if penalties else 0
                print(f"    QBound penalty: {recent_penalty:.4f}")

    return {
        'rewards': episode_rewards,
        'critic_losses': critic_losses,
        'actor_losses': actor_losses,
        'penalties': penalties
    }


def main():
    print("=" * 80)
    print("Testing Soft QBound DDPG on Pendulum-v1")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Environment: {ENV_NAME}")
    print(f"  Episodes: {MAX_EPISODES}")
    print(f"  Max steps: {MAX_STEPS}")
    print(f"  QBound range: [{QBOUND_MIN:.1f}, {QBOUND_MAX:.1f}]")
    print(f"  Penalty type: {PENALTY_TYPE}")
    print(f"  Penalty weight (λ): {PENALTY_WEIGHT}")
    print("=" * 80)

    env = gym.make(ENV_NAME)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    results = {
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
        'config': {
            'env': ENV_NAME,
            'episodes': MAX_EPISODES,
            'max_steps': MAX_STEPS,
            'qbound_min': QBOUND_MIN,
            'qbound_max': QBOUND_MAX,
            'penalty_type': PENALTY_TYPE,
            'penalty_weight': PENALTY_WEIGHT,
            'seed': SEED
        },
        'training': {}
    }

    # ===== 1. Baseline DDPG =====
    print("\n" + "=" * 80)
    print("METHOD 1: Baseline DDPG (No QBound)")
    print("=" * 80)

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

    baseline_results = train_ddpg(baseline_agent, env, "Baseline DDPG")
    results['training']['baseline'] = {
        'rewards': baseline_results['rewards'],
        'final_50_mean': float(np.mean(baseline_results['rewards'][-50:])),
        'final_50_std': float(np.std(baseline_results['rewards'][-50:])),
        'best': float(np.max(baseline_results['rewards']))
    }

    # ===== 2. Hard QBound DDPG (EXPECTED TO FAIL) =====
    print("\n" + "=" * 80)
    print("METHOD 2: Hard QBound DDPG (Hard Clipping)")
    print("=" * 80)
    print("⚠️  WARNING: This is expected to perform WORSE due to zero gradients!")

    hard_agent = DDPGAgent(
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

    hard_results = train_ddpg(hard_agent, env, "Hard QBound DDPG")
    results['training']['hard_qbound'] = {
        'rewards': hard_results['rewards'],
        'final_50_mean': float(np.mean(hard_results['rewards'][-50:])),
        'final_50_std': float(np.std(hard_results['rewards'][-50:])),
        'best': float(np.max(hard_results['rewards']))
    }

    # ===== 3. Soft QBound DDPG (SHOULD WORK!) =====
    print("\n" + "=" * 80)
    print("METHOD 3: Soft QBound DDPG (Smooth Penalties)")
    print("=" * 80)
    print("✓ This should work well - gradients flow even when Q violates bounds!")

    soft_agent = SoftQBoundDDPGAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        lr_actor=LR_ACTOR,
        lr_critic=LR_CRITIC,
        gamma=GAMMA,
        tau=TAU,
        use_soft_qbound=True,
        qbound_min=QBOUND_MIN,
        qbound_max=QBOUND_MAX,
        qbound_penalty_weight=PENALTY_WEIGHT,
        qbound_penalty_type=PENALTY_TYPE,
        device='cpu'
    )

    soft_results = train_ddpg(soft_agent, env, "Soft QBound DDPG")
    results['training']['soft_qbound'] = {
        'rewards': soft_results['rewards'],
        'final_50_mean': float(np.mean(soft_results['rewards'][-50:])),
        'final_50_std': float(np.std(soft_results['rewards'][-50:])),
        'best': float(np.max(soft_results['rewards'])),
        'penalties': soft_results['penalties']
    }

    # ===== Analysis =====
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY (Final 50 Episodes)")
    print("=" * 80)

    methods = ['baseline', 'hard_qbound', 'soft_qbound']
    labels = ['Baseline DDPG', 'Hard QBound (clipping)', 'Soft QBound (penalty)']

    print(f"\n{'Method':<30} {'Mean ± Std':<25} {'Best':<12} {'vs Baseline':<15}")
    print("-" * 85)

    baseline_mean = results['training']['baseline']['final_50_mean']

    for method, label in zip(methods, labels):
        data = results['training'][method]
        mean = data['final_50_mean']
        std = data['final_50_std']
        best = data['best']

        improvement = ((mean - baseline_mean) / abs(baseline_mean)) * 100
        improvement_str = f"{improvement:+.1f}%"

        # Add warning if performance is bad
        warning = ""
        if improvement < -20:
            warning = "  ⚠️ FAILED!"
        elif improvement > 5:
            warning = "  ✓ BETTER!"

        print(f"{label:<30} {mean:>8.1f} ± {std:<8.1f} {best:>8.1f}   {improvement_str:>10}{warning}")

    print("\n" + "=" * 80)
    print("INTERPRETATION:")
    print("=" * 80)
    print("  - Baseline: Should work normally")
    print("  - Hard QBound: Expected to FAIL (zero gradients kill actor learning)")
    print("  - Soft QBound: Should match or BEAT baseline (gradients preserved!)")
    print("=" * 80)

    # Save results
    os.makedirs('/root/projects/QBound/results/pendulum', exist_ok=True)
    results_file = f"/root/projects/QBound/results/pendulum/soft_qbound_test_{results['timestamp']}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to: {results_file}")
    print("\n" + "=" * 80)
    print("Soft QBound DDPG Test Complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
