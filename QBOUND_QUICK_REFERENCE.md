# QBound Quick Reference Card

## TL;DR: Does QBound Work?

**Short Answer:** Yes, but only for specific environments.

---

## ✅ USE QBOUND FOR:

### 1. Dense Positive Rewards (BEST CASE)
- **Example:** CartPole (+1 per timestep)
- **Result:** +12% to +34% improvement
- **Method:** Hard QBound with auxiliary loss
- **All DQN variants benefit**

### 2. Continuous Control (GOOD CASE)
- **Example:** DDPG/TD3 on Pendulum
- **Result:** +15% to +25% improvement
- **Method:** Soft QBound (softplus clipping)
- **Bonus:** 51-87% variance reduction

---

## ❌ DON'T USE QBOUND FOR:

### 1. Sparse Rewards
- **Examples:** GridWorld, FrozenLake
- **Result:** No improvement (0%)
- **Why:** No dense signal to bound

### 2. State-Dependent Rewards
- **Examples:** MountainCar, Acrobot
- **Result:** -3% to -47% degradation
- **Why:** Bounds don't match value distribution

### 3. On-Policy Methods
- **Example:** PPO
- **Result:** -20% degradation
- **Why:** Interferes with policy gradients

### 4. Negative Dense Rewards (Discrete Actions)
- **Example:** Pendulum DQN
- **Result:** -7% degradation
- **Why:** Bounds interfere with learning dynamics

---

## Quick Decision Flowchart

```
START: Should I use QBound?
│
├─ Is your environment CartPole or similar (dense +1 rewards)?
│  └─ YES → ✅ USE HARD QBOUND (expect +12-34% gain)
│
├─ Using DDPG or TD3 with continuous control?
│  └─ YES → ✅ USE SOFT QBOUND (expect +15-25% gain)
│
├─ Does your environment have sparse rewards?
│  └─ YES → ❌ DON'T USE QBOUND (no benefit)
│
├─ Using PPO or other on-policy methods?
│  └─ YES → ❌ DON'T USE QBOUND (expect -20% loss)
│
└─ None of the above?
   └─ ❌ DON'T USE QBOUND (likely -3 to -8% loss)
```

---

## Results Summary Table

| Environment Type | Best Algorithm | Improvement | Use QBound? |
|------------------|----------------|-------------|-------------|
| Dense Positive (CartPole) | DQN/DDQN/Dueling | **+12% to +34%** | ✅ YES |
| Continuous Control (Pendulum) | DDPG/TD3 | **+15% to +25%** | ✅ YES |
| Discrete Negative (Pendulum) | DQN/DDQN | **-3% to -7%** | ❌ NO |
| On-Policy (Pendulum) | PPO | **-20%** | ❌ NO |
| Sparse Rewards (GridWorld, FrozenLake) | DQN/DDQN | **0%** | ❌ NO |
| State-Dependent (MountainCar, Acrobot) | DQN/DDQN | **-3% to -47%** | ❌ NO |

---

## Implementation Code Snippets

### Hard QBound (for CartPole-like environments):

```python
# Dense positive rewards, discrete actions
agent = DQNAgent(
    use_qclip=True,
    qclip_min=0.0,
    qclip_max=99.34,  # (1 - 0.99^500) / (1 - 0.99)
    aux_loss_weight=0.1
)
```

### Soft QBound (for DDPG/TD3):

```python
# Continuous control, actor-critic
agent = DDPGAgent(
    use_soft_qbound=True,
    qclip_method="softplus_clip",
    qclip_min=-1800,  # Based on environment
    qclip_max=0
)
```

### Baseline (for sparse/state-dependent):

```python
# Sparse rewards or state-dependent rewards
agent = DQNAgent(
    use_qclip=False  # Just use baseline
)
```

---

## Success Statistics

**Total Experiments:** 15 algorithm-environment combinations × 5 seeds = 75 runs

**Successful (>10% improvement):** 6 combinations (40%)
- CartPole DQN: +12.0%
- CartPole DDQN: +33.6%
- CartPole Dueling: +22.5%
- CartPole Double-Dueling: +15.5%
- Pendulum DDPG: +25.0%
- Pendulum TD3: +15.3%

**Neutral (±5%):** 3 combinations (20%)
- GridWorld, FrozenLake, Acrobot DDQN

**Failed (<-5%):** 6 combinations (40%)
- Pendulum DQN/DDQN: -3% to -7%
- Pendulum PPO: -20%
- MountainCar DQN/DDQN: -8% to -47%
- Acrobot DQN: -5%

**Overall Success Rate: 40% (6/15 combinations)**

---

## Key Takeaway

**QBound is NOT a universal improvement. It's a specialized technique for:**
1. ✅ Dense positive reward environments (CartPole-style)
2. ✅ Continuous control with actor-critic (DDPG/TD3)

**Use it ONLY after verifying your environment matches these criteria.**

---

## Reference Documents

- **Detailed Analysis:** `docs/QBOUND_FINDINGS.md`
- **Executive Summary:** `EXECUTIVE_SUMMARY.md`
- **Visualizations:** `results/plots/qbound_summary_all_experiments.pdf`
- **Raw Results:** `results/organized_experiments_log.json`

---

## Citation Template

If you use QBound in your work, acknowledge the limitations:

> "We apply QBound to [environment name], which has [dense positive/continuous control]
> characteristics. QBound has been shown to improve performance in such environments
> (+12-34% for CartPole, +15-25% for DDPG/TD3), but is not effective for sparse or
> state-dependent reward structures."

---

**Last Updated:** November 19, 2025
**Data Source:** 5-seed experiments (seeds: 42, 43, 44, 45, 46)
