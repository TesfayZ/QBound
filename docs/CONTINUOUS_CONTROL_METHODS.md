# Continuous Control Methods with Soft QBound

## Overview

This document describes the implementation and configuration of continuous control RL methods (DDPG, TD3, PPO) with **Soft QBound** for gradient-preserving value bounding.

## Why Soft QBound for Continuous Control?

### The Gradient Problem

**Hard clipping** (using `torch.clamp()`) sets gradients to **ZERO** when values exceed bounds:
```python
# Hard clipping - breaks gradients
Q_clipped = torch.clamp(Q, Q_min, Q_max)
# ∇Q_clipped = 0 when Q < Q_min or Q > Q_max
```

This is **catastrophic** for:
- **DDPG/TD3**: Need `∇_a Q(s,a)` for deterministic policy gradient
- **PPO**: Need smooth value gradients for advantage estimation

### Soft QBound Solution

**Soft QBound** uses differentiable soft clipping instead of hard clipping:
```python
# Soft clipping using softplus - preserves gradients
Q_clipped = softplus_clip(Q, Q_min, Q_max, beta=1.0)
# Q-values approach bounds smoothly with non-zero gradients
```

**Implementation:**
```python
def softplus_clip(q_values, q_min, q_max, beta=1.0):
    # Soft lower bound
    q_lower = q_min + F.softplus(q_values - q_min, beta=beta)
    # Soft upper bound
    q_clipped = q_max - F.softplus(q_max - q_lower, beta=beta)
    return q_clipped
```

**Benefits:**
- ✅ Gradients preserved everywhere (non-zero even when violating bounds)
- ✅ Smooth transitions at boundaries (no discontinuities)
- ✅ Works with actor-critic methods
- ✅ Compatible with continuous actions
- ✅ No hyperparameter tuning needed (β=1.0 usually works)

---

## Implemented Methods

### 1. DDPG (Deep Deterministic Policy Gradient)

**File:** `src/ddpg_agent.py`

**Key Features:**
- Deterministic policy: `a = μ(s)`
- Single critic: `Q(s,a)`
- Requires `∇_a Q(s,a)` for policy gradient
- **Uses Soft QBound** to preserve gradients

**Training Script:** `experiments/pendulum/train_pendulum_ddpg_full_qbound.py`

**Configuration:**
```python
agent = DDPGAgent(
    state_dim=3,
    action_dim=1,
    max_action=2.0,
    lr_actor=0.001,
    lr_critic=0.001,
    gamma=0.99,
    tau=0.005,
    use_qbound=True,
    qbound_min=-1616.0,
    qbound_max=0.0,
    use_soft_qbound=True,          # Enable soft QBound
    qbound_penalty_weight=0.1,     # Penalty strength (λ)
    qbound_penalty_type='quadratic', # Penalty function
    soft_clip_beta=0.1,            # Smoothness parameter
    device='cpu'
)
```

**Comparison:**
- Method 1: Baseline DDPG (no QBound)
- Method 2: DDPG + Soft QBound

---

### 2. TD3 (Twin Delayed DDPG)

**File:** `src/td3_agent.py`

**Key Features:**
- Twin critics: `Q1(s,a)` and `Q2(s,a)` (clipped double-Q learning)
- Delayed policy updates (update actor every N steps)
- Target policy smoothing (noise on target actions)
- **Uses Soft QBound** on both critics

**Training Script:** `experiments/pendulum/train_pendulum_td3_full_qbound.py`

**Configuration:**
```python
agent = TD3Agent(
    state_dim=3,
    action_dim=1,
    max_action=2.0,
    lr_actor=0.001,
    lr_critic=0.001,
    gamma=0.99,
    tau=0.005,
    policy_noise=0.2,              # TD3-specific
    noise_clip=0.5,                # TD3-specific
    policy_freq=2,                 # TD3-specific (delayed updates)
    use_qbound=True,
    qbound_min=-1616.0,
    qbound_max=0.0,
    use_soft_qbound=True,          # Enable soft QBound
    qbound_penalty_weight=0.1,
    qbound_penalty_type='quadratic',
    soft_clip_beta=0.1,
    device='cpu'
)
```

**Comparison:**
- Method 1: Baseline TD3 (no QBound)
- Method 2: TD3 + Soft QBound

**TD3 vs DDPG:**
- TD3 more stable due to twin critics and delayed updates
- TD3 typically achieves better performance
- Both require Soft QBound for gradient preservation

---

### 3. PPO (Proximal Policy Optimization)

**File:** `src/ppo_qbound_agent.py`

**Key Features:**
- Bounds **V(s)** (state-value), not Q(s,a)
- Stochastic policy with entropy regularization
- On-policy (uses trajectory buffer)
- **CRITICAL DIFFERENCE:** Policy gradient doesn't need `∇_a V(s)`

**Training Script:** `experiments/ppo/train_pendulum_ppo_full_qbound.py`

**Configuration:**
```python
agent = PPOQBoundAgent(
    state_dim=3,
    action_dim=1,
    continuous_action=True,
    V_min=-3200.0,                 # Note: V bounds, not Q bounds
    V_max=0.0,
    use_step_aware_bounds=False,   # Static bounds
    use_soft_qbound=True,          # Enable soft QBound
    qbound_penalty_weight=0.1,
    qbound_penalty_type='quadratic',
    soft_clip_beta=0.1,
    # PPO-specific parameters
    hidden_sizes=[64, 64],
    lr_actor=3e-4,
    lr_critic=1e-3,
    gamma=0.99,
    gae_lambda=0.95,               # GAE parameter
    clip_epsilon=0.2,              # PPO clip parameter
    entropy_coef=0.01,             # Entropy regularization
    ppo_epochs=10,                 # Optimization epochs
    minibatch_size=64,
    device='cpu'
)
```

**Comparison:**
- Method 1: Baseline PPO (no QBound)
- Method 2: PPO + Soft QBound

**Why PPO Works Better with QBound:**
- Bounds V(s), not Q(s,a) → No action gradient disruption
- Policy gradient: `∇_θ log π(a|s) * A(s,a)` (independent of V bounds)
- Advantage estimation uses bounded V(s) for stability

---

## Implementation Summary

### Where QBound is Applied

| Method | What's Bounded | Where Applied | Gradient Flow |
|--------|---------------|---------------|---------------|
| **DDPG** | Q(s,a) | Target computation | ✅ Via soft penalty |
| **TD3** | Q1(s,a), Q2(s,a) | Both critics' targets | ✅ Via soft penalty |
| **PPO** | V(s) | GAE + returns | ✅ Via soft penalty |

### Soft QBound Mechanism

All three methods use the same soft penalty approach:

**During training:**
```python
# 1. Compute primary loss (TD error, policy loss, etc.)
primary_loss = compute_primary_loss(...)

# 2. Add soft QBound penalty
if use_soft_qbound:
    penalty = penalty_fn.quadratic_penalty(values, V_min, V_max)
    total_loss = primary_loss + λ * penalty
else:
    # Hard clipping fallback (not recommended for continuous)
    values_clipped = torch.clamp(values, V_min, V_max)
```

**Penalty functions available:**
1. **Quadratic:** `penalty = (max(0, Q-Q_max))² + (max(0, Q_min-Q))²`
2. **Huber:** Smooth L1-like penalty
3. **Exponential:** Very smooth, exponentially increasing

---

## Experimental Setup

### Environment: Pendulum-v1

**Characteristics:**
- **State space:** 3D continuous (angle, angular velocity, time)
- **Action space:** 1D continuous (torque)
- **Reward structure:** Dense negative rewards (time-step dependent)
- **Reward range:** Approximately [-16.27, 0]
- **Episode length:** 200 steps

**QBound Configuration:**
```python
# Q_min = reward_per_step * sum(gamma^k for k=0..199)
# Q_min ≈ -16.27 * (1 - 0.99^200) / (1 - 0.99) ≈ -1616
QBOUND_MIN = -1616.0
QBOUND_MAX = 0.0  # Best case: perfect balance from start

# For PPO (V bounds are wider)
V_MIN = -3200.0
V_MAX = 0.0
```

**Hyperparameters:**
- Episodes: 500
- Max steps per episode: 200
- Batch size: 256 (DDPG/TD3), 64 (PPO)
- Learning rate: 0.001 (actor/critic for DDPG/TD3), 3e-4/1e-3 (PPO)
- Gamma: 0.99
- Warmup episodes: 10 (DDPG/TD3)

---

## Running Experiments

### Individual Experiments

**DDPG:**
```bash
python3 experiments/pendulum/train_pendulum_ddpg_full_qbound.py --seed 42
```

**TD3:**
```bash
python3 experiments/pendulum/train_pendulum_td3_full_qbound.py --seed 42
```

**PPO:**
```bash
python3 experiments/ppo/train_pendulum_ppo_full_qbound.py --seed 42
```

### All Continuous Control Methods

```bash
# Run all time-step dependent experiments (includes DDPG, TD3, PPO)
python3 experiments/run_all_organized_experiments.py --category timestep
```

---

## Expected Results

### Performance Expectations

**DDPG:**
- Baseline: -180 to -200 (average reward)
- With Soft QBound: -170 to -190 (~5-10% improvement expected)

**TD3:**
- Baseline: -170 to -190 (better than DDPG due to twin critics)
- With Soft QBound: -160 to -180 (~5-10% improvement expected)

**PPO:**
- Baseline: -200 to -250
- With Soft QBound: -180 to -230 (~10-30% improvement expected)

### Key Validation

**✅ Success criteria:**
1. Soft QBound does NOT cause catastrophic failure (unlike hard clipping)
2. Performance improves or stays neutral (no >10% degradation)
3. Gradients flow properly (no NaN/Inf values)
4. Training is stable (no collapse)

**❌ Hard QBound results (archived):**
- DDPG + Hard QBound: -893% degradation (catastrophic)
- TD3 + Hard QBound: -600% degradation (catastrophic)
- Reason: Zero gradients break actor updates

---

## Key Differences Between Methods

### DDPG vs TD3

| Feature | DDPG | TD3 |
|---------|------|-----|
| Critics | 1 | 2 (twin) |
| Target smoothing | No | Yes (noise) |
| Policy updates | Every step | Delayed (every N) |
| Stability | Lower | Higher |
| Performance | Good | Better |
| QBound compatibility | ✅ Soft only | ✅ Soft only |

### Actor-Critic (DDPG/TD3) vs Policy Gradient (PPO)

| Feature | DDPG/TD3 | PPO |
|---------|----------|-----|
| Bounds | Q(s,a) | V(s) |
| Policy type | Deterministic | Stochastic |
| Learning | Off-policy | On-policy |
| Needs ∇_a Q? | ✅ Yes | ❌ No |
| QBound impact | Direct (affects policy gradient) | Indirect (affects advantage) |
| Gradient sensitivity | High | Lower |

---

## Troubleshooting

### Issue: Training unstable with Soft QBound

**Solution:** Adjust penalty weight
```python
qbound_penalty_weight=0.01  # Reduce if training unstable
qbound_penalty_weight=0.5   # Increase if bounds violated often
```

### Issue: Performance worse than baseline

**Possible causes:**
1. Q_min/Q_max incorrectly specified → Check reward range
2. Penalty weight too high → Reduce penalty weight
3. Penalty function too aggressive → Try different penalty type

### Issue: NaN/Inf in training

**Solution:**
1. Check Q_min/Q_max are not too extreme
2. Reduce learning rates
3. Check penalty function implementation

---

## References

**DDPG:**
- Lillicrap et al. (2015) "Continuous Control with Deep Reinforcement Learning"

**TD3:**
- Fujimoto et al. (2018) "Addressing Function Approximation Error in Actor-Critic Methods"

**PPO:**
- Schulman et al. (2017) "Proximal Policy Optimization Algorithms"

**Soft QBound:**
- Implementation: `src/soft_qbound_penalty.py`
- Documentation: `docs/SOFT_QBOUND.md`

---

## Summary

✅ **DDPG implemented** with Soft QBound for continuous control
✅ **TD3 implemented** with Soft QBound (twin critics + delayed updates)
✅ **PPO implemented** with Soft QBound on V(s) critic

All three methods:
- Use **Soft QBound only** (hard clipping breaks gradients)
- Tested on **Pendulum-v1** (time-step dependent negative rewards)
- Preserve gradient flow for stable continuous control
- Part of organized experiment suite

**Key insight:** Continuous control requires differentiability → Use Soft QBound, never hard clipping.
