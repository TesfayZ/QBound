# Understanding `aux_weight` in QBound

## What is `aux_weight`?

`aux_weight` (auxiliary weight) is a **hyperparameter** that controls the balance between two different loss functions in the QBound training process.

**Value in your experiments: 0.5**

---

## The Two-Loss Training System

QBound uses a **dual-loss approach**:

### 1. **PRIMARY LOSS** (Standard TD Loss)
- **Purpose:** Learn optimal Q-values through temporal difference learning
- **What it does:** Updates Q-values based on Bellman equation
- **Formula:** `primary_loss = MSE(Q(s,a), r + γ * max_a' Q(s',a'))`
- **Weight:** Always 1.0 (implicit)

### 2. **AUXILIARY LOSS** (Bound Enforcement Loss)
- **Purpose:** Teach the network to naturally output Q-values within bounds
- **What it does:** Supervised learning to keep Q-values in [Q_min, Q_max]
- **Formula:** `aux_loss = MSE(violated_Q_values, scaled_Q_values)`
- **Weight:** `aux_weight` (in your case: 0.5)

---

## How Auxiliary Loss Works

When Q-values violate the bounds, the auxiliary loss:

1. **Detects violations** - Finds next-state Q-values outside [Q_min, Q_max]
2. **Scales proportionally** - Rescales ALL actions at that state to fit in bounds
3. **Preserves preferences** - Maintains relative ordering of action values
4. **Creates supervision target** - Teaches network to output scaled values

### Example:

```
Original Q-values for 4 actions at next state:
Q(s', a1) = 120  # Violates Q_max = 100
Q(s', a2) = 150  # Violates Q_max = 100
Q(s', a3) = 80
Q(s', a4) = 50

After proportional scaling to [0, 100]:
Q(s', a1) = 70   # Scaled down
Q(s', a2) = 100  # Scaled down (but still best)
Q(s', a3) = 30   # Scaled down
Q(s', a4) = 0    # Scaled down

Auxiliary loss = MSE(original, scaled)
Teaches network: "output these scaled values instead"
```

---

## Combined Loss Formula

```python
total_loss = primary_loss + aux_weight * aux_loss
```

With `aux_weight = 0.5`:
```python
total_loss = primary_loss + 0.5 * aux_loss
```

---

## What Does `aux_weight` Control?

The `aux_weight` parameter controls **how aggressively** the network is pushed to stay within bounds:

| aux_weight | Effect | Pros | Cons |
|------------|--------|------|------|
| **0.0** | No auxiliary loss | Pure RL learning | Q-values can violate bounds freely |
| **0.1-0.3** | Gentle nudging | Mostly learns optimally | May still violate bounds |
| **0.5** | **Balanced** (your setting) | Balance between optimality and bounds | Moderate restriction |
| **0.7-0.9** | Strong enforcement | Stays within bounds | May sacrifice optimality |
| **1.0+** | Very aggressive | Forces bounds strictly | Can harm learning significantly |

---

## In Your Experiments

**Setting: `aux_weight = 0.5`**

This means:
- **50% focus** on learning optimal Q-values (primary loss)
- **50% focus** on enforcing bounds (auxiliary loss)

### Impact on Results:

1. **GridWorld (failure case):**
   - Auxiliary loss actively prevents Q-values from exceeding 1.0
   - But optimal learning needs Q-values > 1.0 for value propagation
   - `aux_weight = 0.5` is strong enough to harm learning

2. **FrozenLake (success case):**
   - Auxiliary loss helps prevent overestimation
   - Bounds are appropriate, so aux loss doesn't hurt
   - `aux_weight = 0.5` helps stabilize learning

3. **CartPole (severe failure):**
   - Auxiliary loss enforces Q_max = 100
   - But optimal Q-values should be ~500
   - `aux_weight = 0.5` causes severe performance degradation

---

## Visualization of Loss Components

```
Episode Training:
┌─────────────────────────────────────────┐
│ Total Loss = Primary + 0.5 × Auxiliary  │
└─────────────────────────────────────────┘
           │                    │
           │                    │
           ▼                    ▼
    ┌──────────────┐    ┌──────────────┐
    │ Primary Loss │    │ Aux Loss     │
    │              │    │              │
    │ Learn to     │    │ Stay within  │
    │ maximize     │    │ [Q_min,      │
    │ rewards      │    │  Q_max]      │
    └──────────────┘    └──────────────┘
         Weight: 1.0      Weight: 0.5
```

---

## Key Code Section

From `dqn_agent.py:237`:

```python
# Combined loss
total_loss = primary_loss + self.aux_weight * aux_loss
```

From `dqn_agent.py:200-232`:
```python
if self.use_qclip and self.aux_weight > 0:
    # Calculate auxiliary loss for violated samples
    # ... (proportional scaling logic)
    aux_loss = nn.MSELoss()(violated_q, scaled_q)
```

---

## Critical Insight

**The auxiliary loss is a DOUBLE-EDGED SWORD:**

✅ **Helps when:**
- Bounds are correctly set
- Environment is stochastic
- Overestimation is a problem

❌ **Hurts when:**
- Bounds are too restrictive (your case!)
- Optimal Q-values exceed Q_max
- Environment needs value propagation

**In 2 out of 3 of your environments, the bounds are wrong, so aux_weight = 0.5 is actively making things worse.**

---

## Recommendations

1. **If you keep current Q_max values:**
   - Try `aux_weight = 0.0` (disable auxiliary loss)
   - This will reduce the harm from incorrect bounds

2. **If you fix Q_max values:**
   - GridWorld: Set Q_max = 10.0, keep aux_weight = 0.5
   - CartPole: Set Q_max = 500.0, keep aux_weight = 0.5
   - FrozenLake: Keep as is (working well!)

3. **For experimentation:**
   - Try aux_weight ∈ {0.0, 0.1, 0.3, 0.5, 0.7, 1.0}
   - Plot performance vs aux_weight
   - Include in paper as ablation study

---

## Summary

**`aux_weight = 0.5`** means:
- Equal importance to learning optimal values vs staying in bounds
- Works well when bounds are correct (FrozenLake)
- Actively harms performance when bounds are wrong (GridWorld, CartPole)
- Is essentially fighting against optimal learning in 2/3 environments

**Bottom line:** The aux_weight isn't the problem—the Q_max values are! Fix those first, then aux_weight = 0.5 should work fine.
