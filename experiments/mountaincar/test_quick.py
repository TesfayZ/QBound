"""
Quick test of MountainCar 6-way experiment
Tests just 2 episodes of each method to verify everything works
"""

import sys
sys.path.insert(0, '/root/projects/QBound/src')

import gymnasium as gym
import numpy as np
import torch
import random
from dqn_agent import DQNAgent
from double_dqn_agent import DoubleDQNAgent

# Reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

# Quick test parameters
TEST_EPISODES = 2
MAX_STEPS = 200

# Hyperparameters
LR = 0.001
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
BATCH_SIZE = 64
BUFFER_SIZE = 10000
TARGET_UPDATE_FREQ = 100

# QBound parameters
QBOUND_MIN = -100.0
QBOUND_MAX = 0.0
STEP_REWARD = 1.0

print("=" * 60)
print("MountainCar Quick Test (2 episodes per method)")
print("=" * 60)

env = gym.make("MountainCar-v0")
env.reset(seed=SEED)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

print(f"\nEnvironment: MountainCar-v0")
print(f"State dim: {state_dim}, Action dim: {action_dim}")
print(f"Test episodes: {TEST_EPISODES}")

def test_agent(agent, name, use_step_aware=False):
    """Quick test of an agent"""
    print(f"\n>>> Testing: {name}")

    for episode in range(TEST_EPISODES):
        state, _ = env.reset(seed=SEED + episode)
        episode_reward = 0
        step = 0

        while step < MAX_STEPS:
            action = agent.select_action(state, eval_mode=False)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Store transition
            if use_step_aware:
                agent.store_transition(state, action, reward, next_state, done, current_step=step)
            else:
                agent.store_transition(state, action, reward, next_state, done)

            # Train
            loss = agent.train_step()

            episode_reward += reward
            state = next_state
            step += 1

            if done:
                break

        print(f"  Episode {episode+1}: reward={episode_reward:.1f}, steps={step}")

    print(f"  ✓ {name} passed!")

# Test 1: Baseline DQN
print("\n" + "=" * 60)
print("TEST 1: Baseline DQN")
print("=" * 60)
agent1 = DQNAgent(
    state_dim=state_dim, action_dim=action_dim, learning_rate=LR,
    gamma=GAMMA, epsilon_start=EPSILON_START, epsilon_end=EPSILON_END,
    epsilon_decay=EPSILON_DECAY, buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE,
    target_update_freq=TARGET_UPDATE_FREQ, use_qclip=False, device='cpu'
)
test_agent(agent1, "Baseline DQN", use_step_aware=False)

# Test 2: Static QBound + DQN
print("\n" + "=" * 60)
print("TEST 2: Static QBound + DQN")
print("=" * 60)
agent2 = DQNAgent(
    state_dim=state_dim, action_dim=action_dim, learning_rate=LR,
    gamma=GAMMA, epsilon_start=EPSILON_START, epsilon_end=EPSILON_END,
    epsilon_decay=EPSILON_DECAY, buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE,
    target_update_freq=TARGET_UPDATE_FREQ, use_qclip=True,
    qclip_min=QBOUND_MIN, qclip_max=QBOUND_MAX, device='cpu'
)
test_agent(agent2, "Static QBound + DQN", use_step_aware=False)

# Test 3: Dynamic QBound + DQN
print("\n" + "=" * 60)
print("TEST 3: Dynamic QBound + DQN (INCREASING bounds)")
print("=" * 60)
agent3 = DQNAgent(
    state_dim=state_dim, action_dim=action_dim, learning_rate=LR,
    gamma=GAMMA, epsilon_start=EPSILON_START, epsilon_end=EPSILON_END,
    epsilon_decay=EPSILON_DECAY, buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE,
    target_update_freq=TARGET_UPDATE_FREQ, use_qclip=True,
    qclip_min=QBOUND_MIN, qclip_max=QBOUND_MAX,
    use_step_aware_qbound=True, max_episode_steps=MAX_STEPS,
    step_reward=STEP_REWARD, reward_is_negative=True, device='cpu'
)
test_agent(agent3, "Dynamic QBound + DQN", use_step_aware=True)

# Test 4: Baseline DDQN
print("\n" + "=" * 60)
print("TEST 4: Baseline DDQN")
print("=" * 60)
agent4 = DoubleDQNAgent(
    state_dim=state_dim, action_dim=action_dim, learning_rate=LR,
    gamma=GAMMA, epsilon_start=EPSILON_START, epsilon_end=EPSILON_END,
    epsilon_decay=EPSILON_DECAY, buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE,
    target_update_freq=TARGET_UPDATE_FREQ, use_qclip=False, device='cpu'
)
test_agent(agent4, "Baseline DDQN", use_step_aware=False)

# Test 5: Static QBound + DDQN
print("\n" + "=" * 60)
print("TEST 5: Static QBound + DDQN")
print("=" * 60)
agent5 = DoubleDQNAgent(
    state_dim=state_dim, action_dim=action_dim, learning_rate=LR,
    gamma=GAMMA, epsilon_start=EPSILON_START, epsilon_end=EPSILON_END,
    epsilon_decay=EPSILON_DECAY, buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE,
    target_update_freq=TARGET_UPDATE_FREQ, use_qclip=True,
    qclip_min=QBOUND_MIN, qclip_max=QBOUND_MAX, device='cpu'
)
test_agent(agent5, "Static QBound + DDQN", use_step_aware=False)

# Test 6: Dynamic QBound + DDQN
print("\n" + "=" * 60)
print("TEST 6: Dynamic QBound + DDQN (INCREASING bounds)")
print("=" * 60)
agent6 = DoubleDQNAgent(
    state_dim=state_dim, action_dim=action_dim, learning_rate=LR,
    gamma=GAMMA, epsilon_start=EPSILON_START, epsilon_end=EPSILON_END,
    epsilon_decay=EPSILON_DECAY, buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE,
    target_update_freq=TARGET_UPDATE_FREQ, use_qclip=True,
    qclip_min=QBOUND_MIN, qclip_max=QBOUND_MAX,
    use_step_aware_qbound=True, max_episode_steps=MAX_STEPS,
    step_reward=STEP_REWARD, reward_is_negative=True, device='cpu'
)
test_agent(agent6, "Dynamic QBound + DDQN", use_step_aware=True)

print("\n" + "=" * 60)
print("✅ ALL TESTS PASSED!")
print("=" * 60)
print("\nReady to run full experiment:")
print("python3 experiments/mountaincar/train_mountaincar_6way.py")

env.close()
