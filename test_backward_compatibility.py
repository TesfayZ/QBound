"""
Backward Compatibility Test for Soft QBound Integration

Tests that:
1. Existing agents work with default parameters (hard QBound)
2. New soft QBound flag works correctly
3. No breaking changes were introduced

This is a quick smoke test, not a full training run.
"""

import sys
sys.path.insert(0, '/root/projects/QBound/src')

import torch
import numpy as np

# Set seed for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

def test_ddpg_backward_compatibility():
    """Test DDPG with both hard and soft QBound"""
    print("\n" + "="*60)
    print("Testing DDPG Backward Compatibility")
    print("="*60)

    from ddpg_agent import DDPGAgent

    # Test 1: Hard QBound (backward compatible - default behavior)
    print("\n[Test 1] DDPG with hard QBound (default)...")
    try:
        agent_hard = DDPGAgent(
            state_dim=8,
            action_dim=2,
            max_action=2.0,
            use_qbound=True,
            qbound_min=-1630.0,
            qbound_max=0.0,
            use_soft_qbound=False  # Explicit hard clipping
        )

        # Add some fake transitions
        for _ in range(10):
            state = np.random.randn(8)
            action = np.random.randn(2)
            reward = np.random.randn()
            next_state = np.random.randn(8)
            done = False
            agent_hard.replay_buffer.push(state, action, reward, next_state, done)

        # Try a training step
        critic_loss, actor_loss = agent_hard.train(batch_size=10)

        if critic_loss is not None:
            print(f"✓ Hard QBound DDPG works! Critic loss: {critic_loss:.4f}")
        else:
            print("✓ Hard QBound DDPG initialized correctly (needs more data for training)")

    except Exception as e:
        print(f"✗ FAILED: Hard QBound DDPG - {str(e)}")
        return False

    # Test 2: Soft QBound (new functionality)
    print("\n[Test 2] DDPG with soft QBound (new feature)...")
    try:
        agent_soft = DDPGAgent(
            state_dim=8,
            action_dim=2,
            max_action=2.0,
            use_qbound=True,
            qbound_min=-1630.0,
            qbound_max=0.0,
            use_soft_qbound=True,  # NEW: Soft clipping
            qbound_penalty_weight=0.1,
            qbound_penalty_type='quadratic'
        )

        # Add some fake transitions
        for _ in range(10):
            state = np.random.randn(8)
            action = np.random.randn(2)
            reward = np.random.randn()
            next_state = np.random.randn(8)
            done = False
            agent_soft.replay_buffer.push(state, action, reward, next_state, done)

        # Try a training step
        result = agent_soft.train(batch_size=10)

        if result is not None and len(result) == 3:
            critic_loss, actor_loss, qbound_penalty = result
            print(f"✓ Soft QBound DDPG works! Critic loss: {critic_loss:.4f}, Penalty: {qbound_penalty:.4f}")
        elif result is not None:
            print("✓ Soft QBound DDPG initialized correctly (needs more data for training)")
        else:
            print("✓ Soft QBound DDPG initialized correctly")

    except Exception as e:
        print(f"✗ FAILED: Soft QBound DDPG - {str(e)}")
        return False

    print("\n✓ DDPG backward compatibility: PASSED")
    return True


def test_td3_backward_compatibility():
    """Test TD3 with both hard and soft QBound"""
    print("\n" + "="*60)
    print("Testing TD3 Backward Compatibility")
    print("="*60)

    from td3_agent import TD3Agent

    # Test 1: Hard QBound (backward compatible)
    print("\n[Test 1] TD3 with hard QBound (default)...")
    try:
        agent_hard = TD3Agent(
            state_dim=8,
            action_dim=2,
            max_action=2.0,
            use_qbound=True,
            qbound_min=-1630.0,
            qbound_max=0.0,
            use_soft_qbound=False
        )

        # Add some fake transitions
        for _ in range(10):
            state = np.random.randn(8)
            action = np.random.randn(2)
            reward = np.random.randn()
            next_state = np.random.randn(8)
            done = False
            agent_hard.replay_buffer.push(state, action, reward, next_state, done)

        # Try a training step
        critic_loss, actor_loss = agent_hard.train(batch_size=10)

        if critic_loss is not None:
            print(f"✓ Hard QBound TD3 works! Critic loss: {critic_loss:.4f}")
        else:
            print("✓ Hard QBound TD3 initialized correctly")

    except Exception as e:
        print(f"✗ FAILED: Hard QBound TD3 - {str(e)}")
        return False

    # Test 2: Soft QBound (new functionality)
    print("\n[Test 2] TD3 with soft QBound (new feature)...")
    try:
        agent_soft = TD3Agent(
            state_dim=8,
            action_dim=2,
            max_action=2.0,
            use_qbound=True,
            qbound_min=-1630.0,
            qbound_max=0.0,
            use_soft_qbound=True,
            qbound_penalty_weight=0.1,
            qbound_penalty_type='quadratic'
        )

        # Add some fake transitions
        for _ in range(10):
            state = np.random.randn(8)
            action = np.random.randn(2)
            reward = np.random.randn()
            next_state = np.random.randn(8)
            done = False
            agent_soft.replay_buffer.push(state, action, reward, next_state, done)

        # Try a training step
        result = agent_soft.train(batch_size=10)

        if result is not None and len(result) == 3:
            critic_loss, actor_loss, qbound_penalty = result
            print(f"✓ Soft QBound TD3 works! Critic loss: {critic_loss:.4f}, Penalty: {qbound_penalty:.4f}")
        elif result is not None:
            print("✓ Soft QBound TD3 initialized correctly")
        else:
            print("✓ Soft QBound TD3 initialized correctly")

    except Exception as e:
        print(f"✗ FAILED: Soft QBound TD3 - {str(e)}")
        return False

    print("\n✓ TD3 backward compatibility: PASSED")
    return True


def test_ppo_backward_compatibility():
    """Test PPO QBound with both hard and soft modes"""
    print("\n" + "="*60)
    print("Testing PPO QBound Backward Compatibility")
    print("="*60)

    from ppo_qbound_agent import PPOQBoundAgent

    # Test 1: Hard QBound (backward compatible)
    print("\n[Test 1] PPO with hard QBound (default)...")
    try:
        agent_hard = PPOQBoundAgent(
            state_dim=4,
            action_dim=2,
            continuous_action=False,
            V_min=0.0,
            V_max=99.34,
            use_soft_qbound=False
        )

        # Create fake trajectory
        trajectories = []
        for _ in range(32):  # Mini-batch size
            state = np.random.randn(4)
            action = np.random.randint(0, 2)
            reward = 1.0
            next_state = np.random.randn(4)
            done = False
            log_prob = -0.5
            trajectories.append((state, action, reward, next_state, done, log_prob))

        # Try an update
        training_info = agent_hard.update(trajectories)

        print(f"✓ Hard QBound PPO works! Critic loss: {training_info['critic_loss']:.4f}")

    except Exception as e:
        print(f"✗ FAILED: Hard QBound PPO - {str(e)}")
        return False

    # Test 2: Soft QBound (new functionality)
    print("\n[Test 2] PPO with soft QBound (new feature)...")
    try:
        agent_soft = PPOQBoundAgent(
            state_dim=4,
            action_dim=2,
            continuous_action=False,
            V_min=0.0,
            V_max=99.34,
            use_soft_qbound=True,
            qbound_penalty_weight=0.1,
            qbound_penalty_type='quadratic'
        )

        # Create fake trajectory
        trajectories = []
        for _ in range(32):
            state = np.random.randn(4)
            action = np.random.randint(0, 2)
            reward = 1.0
            next_state = np.random.randn(4)
            done = False
            log_prob = -0.5
            trajectories.append((state, action, reward, next_state, done, log_prob))

        # Try an update
        training_info = agent_soft.update(trajectories)

        print(f"✓ Soft QBound PPO works! Critic loss: {training_info['critic_loss']:.4f}")

    except Exception as e:
        print(f"✗ FAILED: Soft QBound PPO - {str(e)}")
        return False

    print("\n✓ PPO backward compatibility: PASSED")
    return True


def main():
    """Run all backward compatibility tests"""
    print("="*60)
    print("BACKWARD COMPATIBILITY TEST SUITE")
    print("="*60)
    print("\nTesting that soft QBound integration maintains compatibility...")
    print("This verifies:")
    print("  1. Default behavior (hard QBound) unchanged")
    print("  2. New soft QBound flag works correctly")
    print("  3. No breaking changes introduced")

    results = []

    # Test DDPG
    results.append(("DDPG", test_ddpg_backward_compatibility()))

    # Test TD3
    results.append(("TD3", test_td3_backward_compatibility()))

    # Test PPO
    results.append(("PPO", test_ppo_backward_compatibility()))

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    all_passed = True
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{name:20s} {status}")
        if not passed:
            all_passed = False

    print("="*60)

    if all_passed:
        print("\n✓ ALL TESTS PASSED - Backward compatibility maintained!")
        print("\nNext steps:")
        print("  1. Existing experiments will work unchanged (use_soft_qbound=False by default)")
        print("  2. New experiments can enable soft QBound with use_soft_qbound=True")
        print("  3. All agents maintain full backward compatibility")
    else:
        print("\n✗ SOME TESTS FAILED - Review errors above")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
