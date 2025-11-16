# QBound Usage Guide

This guide provides detailed instructions on how to use QBound in your own reinforcement learning projects.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Choosing QBound Configuration](#choosing-qbound-configuration)
3. [DQN with QBound](#dqn-with-qbound)
4. [DDPG with Soft QBound](#ddpg-with-soft-qbound)
5. [PPO with Soft QBound](#ppo-with-soft-qbound)
6. [Computing Bounds](#computing-bounds)
7. [Dynamic Bounds](#dynamic-bounds)
8. [Troubleshooting](#troubleshooting)

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run a Sample Experiment

```bash
# DQN on FrozenLake (works well)
python experiments/frozenlake/train_frozenlake_6way.py

# DDPG on Pendulum (demonstrates soft QBound)
python experiments/pendulum/train_6way_comparison.py
```

---

## Choosing QBound Configuration

### Decision Tree

```
Is your action space discrete or continuous?
│
├─ DISCRETE (DQN, DDQN, Dueling DQN)
│  └─ Use HARD QBOUND (torch.clamp)
│     │
│     └─ What's your reward structure?
│        │
│        ├─ Sparse terminal (GridWorld, FrozenLake)
│        │  └─ Use STATIC bounds: Q_max = terminal_reward
│        │
│        ├─ Dense positive step (CartPole)
│        │  └─ Use DYNAMIC bounds: Q_max(t) = (1-γ^(H-t))/(1-γ)
│        │
│        └─ Shaped rewards (LunarLander)
│           └─ Use STATIC bounds from domain knowledge
│
└─ CONTINUOUS (DDPG, TD3, PPO)
   └─ Use SOFT QBOUND (differentiable soft clipping)
      │
      └─ What's your reward structure?
         │
         ├─ Dense negative (Pendulum)
         │  └─ Use STATIC bounds: Q_min = r×(1-γ^H)/(1-γ), Q_max = 0
         │
         └─ Sparse/shaped
            └─ Use STATIC bounds from domain knowledge
```

---

## DQN with QBound

### Basic Setup

```python
import gymnasium as gym
from src.dqn_agent import DQNAgent

# Create environment
env = gym.make('FrozenLake-v1')

# Compute bounds
Q_min = 0.0
Q_max = 1.0  # Terminal reward
gamma = 0.95

# Create agent with QBound
agent = DQNAgent(
    state_dim=env.observation_space.n,
    action_dim=env.action_space.n,
    hidden_dim=128,
    lr=0.001,
    gamma=gamma,
    target_update_freq=100,
    use_qclip=True,      # Enable QBound
    qclip_min=Q_min,
    qclip_max=Q_max,
    device='cpu'
)

# Training loop
num_episodes = 2000
batch_size = 64
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01

for episode in range(num_episodes):
    state, _ = env.reset()
    done = False
    episode_reward = 0

    while not done:
        # Select action with epsilon-greedy
        action = agent.select_action(state, epsilon=epsilon)

        # Take step
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Store transition
        agent.replay_buffer.push(state, action, reward, next_state, done)

        # Train if enough samples
        if len(agent.replay_buffer) >= batch_size:
            agent.train(batch_size)

        episode_reward += reward
        state = next_state

    # Decay epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    if (episode + 1) % 100 == 0:
        print(f"Episode {episode+1}, Reward: {episode_reward}, Epsilon: {epsilon:.3f}")
```

### With Dynamic Bounds (Dense Positive Rewards)

```python
from src.dqn_agent import DQNAgent

# For CartPole-like environments
env = gym.make('CartPole-v1')
gamma = 0.99
max_steps = 500
reward_per_step = 1.0

# Compute static Q_max
Q_max = (1 - gamma**max_steps) / (1 - gamma)  # ≈ 99.34

# Create agent with dynamic bounds
agent = DQNAgent(
    state_dim=4,
    action_dim=2,
    use_qclip=True,
    qclip_min=0.0,
    qclip_max=Q_max,
    use_step_aware_qbound=True,  # Enable dynamic
    max_episode_steps=max_steps,
    step_reward=reward_per_step,
    reward_is_negative=False,
    gamma=gamma
)

# Training loop with step tracking
for episode in range(num_episodes):
    state, _ = env.reset()
    done = False
    step = 0

    while not done and step < max_steps:
        action = agent.select_action(state, epsilon=epsilon)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Pass current step for dynamic bounds
        agent.replay_buffer.push(
            state, action, reward, next_state, done,
            current_step=step  # Track step for dynamic Q_max(t)
        )

        if len(agent.replay_buffer) >= batch_size:
            agent.train(batch_size, current_steps=[step] * batch_size)

        state = next_state
        step += 1
```

---

## DDPG with Soft QBound

### Setup with Soft Clipping

```python
import gymnasium as gym
from src.ddpg_agent import DDPGAgent

# Create environment
env = gym.make('Pendulum-v1')

# Compute bounds for Pendulum
gamma = 0.99
max_steps = 200
min_reward_per_step = -16.27  # Approximate min reward

# Geometric sum for negative rewards
geometric_sum = (1 - gamma**max_steps) / (1 - gamma)  # ≈ 99.34
Q_min = min_reward_per_step * geometric_sum  # ≈ -1616
Q_max = 0.0

# Create DDPG agent with Soft QBound
agent = DDPGAgent(
    state_dim=3,
    action_dim=1,
    max_action=2.0,
    lr_actor=0.001,
    lr_critic=0.001,
    gamma=gamma,
    tau=0.005,
    use_qbound=True,
    use_soft_clip=True,       # CRITICAL: Enable soft clipping (differentiable)
    qbound_min=Q_min,
    qbound_max=Q_max,
    device='cpu'
)

# Training loop
num_episodes = 500
batch_size = 256
warmup_episodes = 10

for episode in range(num_episodes):
    state, _ = env.reset()
    agent.reset_noise()  # Reset OU noise
    done = False
    episode_reward = 0

    while not done:
        # Select action with exploration noise
        if episode < warmup_episodes:
            action = env.action_space.sample()
        else:
            action = agent.select_action(state, add_noise=True)

        # Take step
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Store transition
        agent.replay_buffer.push(state, action, reward, next_state, done)

        # Train
        if episode >= warmup_episodes and len(agent.replay_buffer) >= batch_size:
            critic_loss, actor_loss = agent.train(batch_size)

            if episode % 10 == 0:
                print(f"  Critic Loss: {critic_loss:.4f}, "
                      f"Actor Loss: {actor_loss:.4f}")

        episode_reward += reward
        state = next_state

    print(f"Episode {episode+1}/{num_episodes}, Reward: {episode_reward:.2f}")
```

### Simple DDPG (No Target Networks)

```python
from src.simple_ddpg_agent import SimpleDDPGAgent

# Same setup as DDPG, but without target networks
agent = SimpleDDPGAgent(
    state_dim=3,
    action_dim=1,
    max_action=2.0,
    use_qbound=True,
    use_soft_clip=True,  # Essential for learning
    qbound_min=Q_min,
    qbound_max=Q_max
)

# Result: With Soft QBound, achieves near-competitive performance
# without the complexity of target networks!
```

---

## PPO with Soft QBound

### Setup for Value Bounding

```python
import gymnasium as gym
from src.ppo_qbound_agent import PPOQBoundAgent

# Create environment
env = gym.make('LunarLanderContinuous-v3')

# Compute bounds
V_min = -100.0  # Crash penalty
V_max = 200.0   # Landing bonus

# Create PPO agent with QBound
agent = PPOQBoundAgent(
    state_dim=8,
    action_dim=2,
    continuous_action=True,
    V_min=V_min,
    V_max=V_max,
    use_step_aware_bounds=False,  # Static bounds for sparse rewards
    use_soft_qbound=True,          # Enable soft clipping
    hidden_sizes=[128, 128],
    lr_actor=3e-4,
    lr_critic=1e-3,
    gamma=0.99,
    gae_lambda=0.95,
    clip_epsilon=0.2,
    entropy_coef=0.01,
    ppo_epochs=10,
    minibatch_size=64
)

# Training loop
num_episodes = 500
trajectory_length = 2048

for episode in range(num_episodes):
    trajectory = []
    state, _ = env.reset()
    episode_reward = 0

    # Collect trajectory
    for step in range(trajectory_length):
        action, log_prob = agent.get_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        trajectory.append((state, action, reward, next_state, done, log_prob.item()))

        episode_reward += reward
        state = next_state

        if done:
            state, _ = env.reset()

    # Update policy
    agent.update(trajectory)

    print(f"Episode {episode+1}, Reward: {episode_reward:.2f}")
```

---

## Computing Bounds

### Step-by-Step Guide

#### 1. Identify Reward Structure

```python
# Analyze your environment
env = gym.make('YourEnv-v1')

# Questions to ask:
# - What's the reward per step? (constant, variable, terminal only?)
# - What's the maximum episode length?
# - Are rewards positive, negative, or mixed?
# - Is the reward sparse or dense?
```

#### 2. Compute Bounds Based on Structure

**Sparse Terminal Rewards:**

```python
# Example: GridWorld, FrozenLake
# Reward: +1 at goal, 0 elsewhere

Q_min = 0.0
Q_max = terminal_reward  # e.g., 1.0

# Q_max is independent of horizon for sparse terminal rewards
```

**Dense Step Rewards (Positive):**

```python
# Example: CartPole
# Reward: +1 per step, max 500 steps

gamma = 0.99
max_steps = 500
reward_per_step = 1.0

# Geometric sum
Q_max = reward_per_step * (1 - gamma**max_steps) / (1 - gamma)
Q_min = 0.0

print(f"Q_max = {Q_max:.2f}")  # ≈ 99.34
```

**Dense Step Rewards (Negative):**

```python
# Example: Pendulum
# Reward: approximately -16.27 per step, max 200 steps

gamma = 0.99
max_steps = 200
min_reward_per_step = -16.27

# Geometric sum
Q_min = min_reward_per_step * (1 - gamma**max_steps) / (1 - gamma)
Q_max = 0.0

print(f"Q_min = {Q_min:.2f}")  # ≈ -1616
```

**Shaped Rewards (Mixed):**

```python
# Example: LunarLander
# Complex reward structure with bonuses and penalties

# Use domain knowledge
Q_min = -100.0  # Crash penalty
Q_max = 200.0   # Safe landing + bonuses

# Or estimate from actual returns
episode_returns = []
for episode in range(1000):
    total_reward = 0
    state = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        _, reward, done, _ = env.step(action)
        total_reward += reward
    episode_returns.append(total_reward)

Q_min = min(episode_returns) * 1.1  # Add 10% margin
Q_max = max(episode_returns) * 1.1
```

---

## Dynamic Bounds

### When to Use

Dynamic bounds are beneficial for **dense positive step rewards** where the maximum possible return decreases as the episode progresses.

### Implementation

```python
# Example: CartPole with dynamic bounds

gamma = 0.99
max_steps = 500
reward_per_step = 1.0

def compute_Q_max_dynamic(current_step):
    """Compute Q_max at current timestep"""
    remaining_steps = max_steps - current_step
    return reward_per_step * (1 - gamma**remaining_steps) / (1 - gamma)

# At step 0: Q_max(0) = 99.34
# At step 250: Q_max(250) ≈ 49.67
# At step 499: Q_max(499) = 1.0

# This provides tighter bounds as episode progresses,
# reducing overestimation
```

### Enable in DQN

```python
agent = DQNAgent(
    ...,
    use_qclip=True,
    qclip_max=99.34,  # Max possible Q_max
    use_step_aware_qbound=True,  # Enable dynamic
    max_episode_steps=500,
    step_reward=1.0,
    reward_is_negative=False
)

# During training, pass current step
agent.train(batch_size, current_steps=steps_batch)
```

---

## Troubleshooting

### Issue 1: Performance Degradation

**Symptom**: QBound makes performance worse

**Possible Causes**:

1. **Bounds too restrictive**
   ```python
   # Check if Q_max is too low
   # For CartPole, Q_max=100 is too low (should be ~99.34 but episode returns can exceed this)
   # Solution: Use correct geometric sum formula
   ```

2. **Wrong bound type**
   ```python
   # Using static bounds on dense positive rewards
   # Solution: Try dynamic bounds
   agent = DQNAgent(..., use_step_aware_qbound=True)
   ```

3. **Hard clipping on continuous actions**
   ```python
   # WRONG: use_soft_qbound=False for DDPG
   # Solution: Always use soft QBound for continuous
   agent = DDPGAgent(..., use_soft_qbound=True)
   ```

### Issue 2: Gradient Death (Continuous Control)

**Symptom**: Policy doesn't improve, loss becomes NaN

**Cause**: Using hard clipping instead of soft penalty

**Solution**:
```python
# CORRECT:
agent = DDPGAgent(
    ...,
    use_qbound=True,
    use_soft_qbound=True,  # CRITICAL!
    qbound_penalty_weight=0.1
)

# WRONG:
agent = DDPGAgent(
    ...,
    use_qbound=True,
    use_soft_qbound=False  # Will cause gradient death!
)
```

### Issue 3: QBound Penalty Too Large

**Symptom**: QBound penalty dominates training, Q-values don't learn

**Solution**: Reduce penalty weight
```python
# Default λ = 0.1
agent = DDPGAgent(..., qbound_penalty_weight=0.1)

# If penalty too strong, try:
agent = DDPGAgent(..., qbound_penalty_weight=0.05)

# If penalty too weak, try:
agent = DDPGAgent(..., qbound_penalty_weight=0.2)

# Monitor penalty during training
critic_loss, actor_loss, penalty = agent.train(batch_size)
print(f"Penalty: {penalty:.4f}")  # Should be < 10.0 typically
```

### Issue 4: Conflicts with Algorithm Mechanisms

**TD3 + QBound Failure**:
- TD3's clipped double-Q conflicts with QBound
- Solution: Use DDPG instead, or vanilla TD3 without QBound

**PPO + QBound on Dense Rewards**:
- GAE temporal smoothing disrupted by hard clipping
- Solution: Only use on sparse rewards or with dynamic bounds

**DDQN on Dense Positive Rewards**:
- Double-Q pessimism + dense rewards = underestimation
- Solution: Use vanilla DQN or dynamic QBound with DDQN

---

## Summary Checklist

Before using QBound:

- [ ] Identified reward structure (sparse/dense/shaped)
- [ ] Computed correct Q_min and Q_max bounds
- [ ] Chose correct implementation (hard vs soft)
- [ ] Selected bound type (static vs dynamic)
- [ ] Verified gradient flow (soft for continuous)
- [ ] Set appropriate penalty weight (λ = 0.1 default)
- [ ] Monitored training (check penalty values)
- [ ] Validated results (compare with baseline)

---

For more details, see:
- [Paper](QBound/main.pdf) - Full methodology
- [Implementation Verification](QBOUND_IMPLEMENTATION_VERIFICATION.md) - Code verification
- [README](README.md) - Overview and results
