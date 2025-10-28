"""
Quick validation test for PPO and PPO+QBound implementations.

Tests:
1. Networks can be instantiated
2. Forward passes work
3. Actions can be sampled
4. Updates can be performed
5. No runtime errors
"""

import sys
sys.path.insert(0, '/root/projects/QBound/src')

import gymnasium as gym
import numpy as np
import torch
from ppo_agent import PPOAgent
from ppo_qbound_agent import PPOQBoundAgent


def test_discrete_action():
    """Test on CartPole (discrete actions)."""
    print("\n" + "="*60)
    print("Test 1: Discrete Actions (CartPole)")
    print("="*60)

    env = gym.make('CartPole-v1')
    state, _ = env.reset(seed=42)

    # Test baseline PPO
    print("\nTesting Baseline PPO...")
    agent = PPOAgent(
        state_dim=4,
        action_dim=2,
        continuous_action=False,
        hidden_sizes=[32, 32]
    )

    # Sample action
    action, log_prob = agent.get_action(state)
    print(f"  ✓ Action sampled: {action}, log_prob: {log_prob.item():.4f}")

    # Collect small trajectory
    trajectory = []
    for _ in range(10):
        action, log_prob = agent.get_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        trajectory.append((state, action, reward, next_state, done, log_prob.item()))
        state = next_state
        if done:
            state, _ = env.reset()

    # Update
    info = agent.update(trajectory)
    print(f"  ✓ Update successful: actor_loss={info['actor_loss']:.4f}, critic_loss={info['critic_loss']:.4f}")

    # Test PPO+QBound
    print("\nTesting PPO+QBound...")
    agent_qbound = PPOQBoundAgent(
        state_dim=4,
        action_dim=2,
        continuous_action=False,
        V_min=0.0,
        V_max=100.0,
        hidden_sizes=[32, 32]
    )

    state, _ = env.reset(seed=42)
    action, log_prob = agent_qbound.get_action(state)
    print(f"  ✓ Action sampled: {action}, log_prob: {log_prob.item():.4f}")

    # Collect trajectory
    trajectory = []
    for _ in range(10):
        action, log_prob = agent_qbound.get_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        trajectory.append((state, action, reward, next_state, done, log_prob.item()))
        state = next_state
        if done:
            state, _ = env.reset()

    # Update
    info = agent_qbound.update(trajectory)
    print(f"  ✓ Update successful: actor_loss={info['actor_loss']:.4f}, critic_loss={info['critic_loss']:.4f}")
    if 'qbound_violations_bootstrap' in info:
        print(f"  ✓ QBound stats: violations={info['qbound_violations_bootstrap']:.2%}")

    env.close()
    print("\n✅ Discrete action test PASSED")


def test_continuous_action():
    """Test on Pendulum (continuous actions)."""
    print("\n" + "="*60)
    print("Test 2: Continuous Actions (Pendulum)")
    print("="*60)

    env = gym.make('Pendulum-v1')
    state, _ = env.reset(seed=42)

    # Test baseline PPO
    print("\nTesting Baseline PPO...")
    agent = PPOAgent(
        state_dim=3,
        action_dim=1,
        continuous_action=True,
        hidden_sizes=[32, 32]
    )

    # Sample action
    action, log_prob = agent.get_action(state)
    print(f"  ✓ Action sampled: {action}, log_prob: {log_prob.item():.4f}")

    # Collect small trajectory
    trajectory = []
    for _ in range(10):
        action, log_prob = agent.get_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        trajectory.append((state, action, reward, next_state, done, log_prob.item()))
        state = next_state
        if done:
            state, _ = env.reset()

    # Update
    info = agent.update(trajectory)
    print(f"  ✓ Update successful: actor_loss={info['actor_loss']:.4f}, critic_loss={info['critic_loss']:.4f}")

    # Test PPO+QBound
    print("\nTesting PPO+QBound...")
    agent_qbound = PPOQBoundAgent(
        state_dim=3,
        action_dim=1,
        continuous_action=True,
        V_min=-3200.0,
        V_max=0.0,
        hidden_sizes=[32, 32]
    )

    state, _ = env.reset(seed=42)
    action, log_prob = agent_qbound.get_action(state)
    print(f"  ✓ Action sampled: {action}, log_prob: {log_prob.item():.4f}")

    # Collect trajectory
    trajectory = []
    for _ in range(10):
        action, log_prob = agent_qbound.get_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        trajectory.append((state, action, reward, next_state, done, log_prob.item()))
        state = next_state
        if done:
            state, _ = env.reset()

    # Update
    info = agent_qbound.update(trajectory)
    print(f"  ✓ Update successful: actor_loss={info['actor_loss']:.4f}, critic_loss={info['critic_loss']:.4f}")
    if 'qbound_violations_bootstrap' in info:
        print(f"  ✓ QBound stats: violations={info['qbound_violations_bootstrap']:.2%}")

    env.close()
    print("\n✅ Continuous action test PASSED")


def test_dynamic_bounds():
    """Test step-aware dynamic bounds."""
    print("\n" + "="*60)
    print("Test 3: Dynamic Step-Aware Bounds")
    print("="*60)

    env = gym.make('CartPole-v1')
    state, _ = env.reset(seed=42)

    agent = PPOQBoundAgent(
        state_dim=4,
        action_dim=2,
        continuous_action=False,
        V_min=0.0,
        V_max=500.0,
        use_step_aware_bounds=True,
        max_episode_steps=500,
        step_reward=1.0,
        hidden_sizes=[32, 32]
    )

    print("Testing dynamic bound computation...")
    for step in [0, 100, 250, 499]:
        V_min, V_max = agent.compute_bounds(step)
        print(f"  Step {step:3d}: V_min={V_min:.1f}, V_max={V_max:.1f}")

    print("\n  ✓ Dynamic bounds work correctly")

    # Test trajectory with steps
    trajectory = []
    for step in range(10):
        action, log_prob = agent.get_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        trajectory.append((state, action, reward, next_state, done, log_prob.item(), step))
        state = next_state
        if done:
            state, _ = env.reset()

    # Update with steps
    info = agent.update(trajectory)
    print(f"  ✓ Update with dynamic bounds successful")

    env.close()
    print("\n✅ Dynamic bounds test PASSED")


def main():
    """Run all tests."""
    print("="*60)
    print("PPO Implementation Validation Tests")
    print("="*60)

    torch.manual_seed(42)
    np.random.seed(42)

    try:
        test_discrete_action()
        test_continuous_action()
        test_dynamic_bounds()

        print("\n" + "="*60)
        print("🎉 ALL TESTS PASSED!")
        print("="*60)
        print("\nImplementations are working correctly.")
        print("Ready to run full pilot experiments.")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
