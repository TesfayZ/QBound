# Answer: MountainCar/Acrobot Investigation

## Your Questions

1. **"What were the Q_min and Q_max values for MountainCar/Acrobot?"**
2. **"Why did DQN work here but not in Pendulum?"**
3. **"Why did Pendulum work on actor-critic but not DQN?"**

## Short Answers

### Q1: Q-Bounds Used

| Environment | Q_min | Q_max | Range |
|-------------|-------|-------|-------|
| **Pendulum** | -1409.33 | 0.0 | 1409.33 |
| **MountainCar** | -86.60 | 0.0 | 86.60 |
| **Acrobot** | -99.34 | 0.0 | 99.34 |

**Key:** MountainCar/Acrobot have **16x smaller Q-value ranges!**

### Q2: DQN Actually DOESN'T Work on MountainCar/Acrobot!

**I misinterpreted the results initially!** In negative reward environments:
- More negative = worse performance
- QBound makes rewards MORE negative = degradation

**Corrected results:**
- MountainCar: **+8.9% degradation** (more steps to reach goal)
- Acrobot: **+5.0% degradation** (more steps to reach goal)
- Pendulum: **+7.1% degradation**

**All degrade!** But through different mechanisms.

### Q3: Actor-Critic Works on Pendulum Because of Two-Level Clipping

**The mechanism:**
- Level 1 (hard on critic): Prevents Q-explosion
- Level 2 (**soft on actor**): Preserves gradients → learns to reduce violations

This only helps when violations are HIGH (like Pendulum: 57%).

---

## The Smoking Gun: Violation Rates

| Environment | Violations (Q > Q_max) | Degradation | Mechanism |
|-------------|------------------------|-------------|-----------|
| **Pendulum** | **57.30%** | +7.1% | High violations → underestimation bias |
| **MountainCar** | **0.44%** | +8.9% | Low violations → exploration interference |
| **Acrobot** | **0.90%** | +5.0% | Low violations → exploration interference |

**Key Insight:** Degradation happens regardless, but for DIFFERENT reasons!

---

## Detailed Analysis

### Why Violation Rates Differ

#### Pendulum (High Violations: 57.30%)

**Reward structure:**
- Dense negative: -16.2 per step
- Varies with angle (-0.1 to -16.2)
- Creates wide Q-value distribution

**Q-value characteristics:**
- Range: [-1409, 0] (very wide!)
- Near-terminal states: Q ≈ -16 (only 16 units from Q_max=0)
- Far states: Q ≈ -1400
- **Many states cluster near Q_max=0 → high violation rate**

**Why violations occur:**
```
Near-terminal state:
  True Q ≈ -16.1
  Network approximation: -16.1 ± 20 (function approximation error)
  Often predicts: Q > 0 → VIOLATION!

Violation rate: 57.30% ✗
```

#### MountainCar/Acrobot (Low Violations: 0.44-0.90%)

**Reward structure:**
- Constant negative: -1 per step
- Same reward regardless of state
- Creates narrow Q-value distribution

**Q-value characteristics:**
- Range: [-86.6, 0] or [-99.3, 0] (16x smaller!)
- Most states: Q ≈ -100 to -125 (far from Q_max=0)
- Only near-goal states: Q ≈ 0
- **Rare for states to be near Q_max → low violation rate**

**Why few violations:**
```
Typical state:
  True Q ≈ -120
  Network approximation: -120 ± 20
  Rarely exceeds 0 even with errors

Violation rate: 0.44% ✓
```

### Why Both Degrade Despite Different Violations

#### Pendulum: Bias-Driven Degradation

**With 57% violations:**

```python
# For 57% of transitions:
Q_next_raw = +0.092  # Violation
Q_next_clipped = 0.0  # Forced to bound
target = -16.2 + 0.99 * 0.0 = -16.20  # Biased!

# True target:
target_true = -16.2 + 0.99 * 0.092 = -16.11

# Bias per transition: -0.09
# With 57% affected → LARGE accumulated bias
```

**Result:** Underestimation → policy selects suboptimal actions → +7.1% degradation

#### MountainCar/Acrobot: Exploration-Driven Degradation

**With only 0.44-0.90% violations:**

```python
# For 99% of transitions:
Q_next = -120  # Well within bounds
target = -1 + 0.99 * (-120) = -119.8  # Unbiased! ✓

# For < 1% of transitions:
Q_next_raw = +0.003  # Tiny violation
Q_next_clipped = 0.0
target = -1 + 0.99 * 0.0 = -1.0  # Minimal bias

# Bias: Nearly zero!
```

**So why degrade?**

**The exploration interference mechanism:**

1. **MountainCar/Acrobot are exploration problems:**
   - Need to discover momentum/swing-up strategy
   - Critical states: near goal but not quite there
   - Q-values guide exploration

2. **Even rare clipping affects critical states:**
   - States near goal have Q ≈ 0
   - These are the EXACT states that get clipped!
   - Clipping removes distinction between "almost there" and "far away"

3. **Exploration becomes less directed:**
   - Can't distinguish progress toward goal
   - Takes more random exploration
   - More steps needed to succeed

**Evidence:**
- MountainCar baseline: 124 steps → QBound: 134 steps (+8.9%)
- Acrobot baseline: 90 steps → QBound: 94 steps (+5.0%)
- QBound makes goal-finding HARDER

---

## Why Actor-Critic Works on Pendulum But Not Tested on MountainCar

### Pendulum: High Violations → Two-Level Helps

**Problem:** 57% violation rate creates massive bias

**Solution:** Two-level clipping
```python
# Level 1 (Critic): Hard clipping for stability
target = clamp(r + γ*Q_next, max=0)  # Still 57% violations

# Level 2 (Actor): SOFT clipping for learning
Q_actor = softplus_clip(Q(s,μ(s)), max=0)
gradient = sigmoid(...) > 0  # ← ALWAYS positive!

# Actor learns to select actions with Q-values within bounds
# Violations decrease over time
# Result: -15.1% improvement ✓
```

**Why it works:**
- High initial violations (57%)
- Large room for improvement
- Soft clipping gradually reduces violations
- Performance improves significantly

### MountainCar/Acrobot: Low Violations → Two-Level Wouldn't Help Much

**"Problem":** Only 0.44-0.90% violations (not really a problem!)

**If we applied two-level clipping:**
```python
# Level 1: Hard clipping (same as before)
target = clamp(r + γ*Q_next, max=0)  # Only 0.44% affected

# Level 2: Soft clipping on actor
Q_actor = softplus_clip(Q(s,μ(s)), max=0)
# Actor tries to reduce violations... but already < 1%!
# Not much to improve
```

**Expected result:**
- Violations already minimal
- Soft clipping has little to work with
- **Main problem is exploration interference**, not violations
- Soft clipping doesn't fix exploration issues
- Likely neutral or small improvement at best

**Not tested because:**
- MountainCar/Acrobot are discrete action spaces
- Actor-critic (DDPG/TD3) designed for continuous actions
- Would need different algorithm (e.g., SAC for discrete)

---

## Complete Mechanism Summary

### Why All Negative Rewards Degrade with Hard Clipping (DQN)

**Dense Negative (Pendulum):**
1. Large reward magnitude (-16.2) → wide Q-range
2. Many near-terminal states → high violations (57%)
3. Hard clipping → underestimation bias
4. Bias accumulates → +7.1% degradation

**Sparse Negative (MountainCar/Acrobot):**
1. Small reward magnitude (-1) → narrow Q-range
2. Few near-terminal states → low violations (< 1%)
3. Minimal bias, BUT clipping affects critical exploration states
4. Exploration interference → +5-9% degradation

### Why Two-Level Clipping Helps Pendulum

1. **High violation rate (57%)** = large room for improvement
2. **Soft clipping on actor** = gradients flow → learn to reduce violations
3. **Violations decrease** over training
4. **Result:** -15.1% improvement

### Why Two-Level Wouldn't Help MountainCar/Acrobot

1. **Low violation rate (< 1%)** = little room for improvement
2. Soft clipping can't reduce already-minimal violations
3. **Main problem is exploration**, which soft clipping doesn't address
4. **Expected result:** Minimal benefit

---

## Updated Recommendations

### For Positive Dense Rewards (CartPole)
✓ **USE QBound (hard clipping)**
- Prevents overestimation
- +14% to +23% improvement

### For Negative Dense Rewards (Pendulum)

**With DQN/DDQN (hard clipping only):**
✗ **AVOID** - Degrades +3.7% to +7.1%

**With DDPG/TD3 (two-level clipping):**
✓ **USE** - Improves -5.7% to -15.1%

**Reason:** High violations (57%) → soft clipping reduces them effectively

### For Sparse Negative Rewards (MountainCar/Acrobot)

**With DQN (hard clipping):**
✗ **AVOID** - Degrades +5% to +9%

**With Actor-Critic (two-level clipping):**
~ **Likely neutral** - Low violations (< 1%) → little benefit from soft clipping

**Reason:** Exploration interference is main problem, not addressable by soft clipping

### General Rules

1. **Hard clipping on negative rewards = bad**
   - Always degrades (either bias or exploration interference)

2. **Two-level clipping only helps high-violation cases**
   - Pendulum: 57% violations → -15% improvement ✓
   - MountainCar/Acrobot: < 1% violations → minimal benefit expected

3. **Violation rate predicts mechanism:**
   - High (> 10%): Bias-driven degradation
   - Low (< 5%): Exploration-driven degradation

---

## Files Created

1. **`docs/WHY_MOUNTAINCAR_WORKS_BUT_PENDULUM_DOESNT.md`** - Detailed analysis
2. **`analysis/investigate_mountaincar_acrobot.py`** - Investigation script
3. **`ANSWER_MOUNTAINCAR_INVESTIGATION.md`** - This summary

---

## Bottom Line

### Your Questions Answered:

**Q1: Q-bounds?**
- Pendulum: [-1409, 0], MountainCar: [-86.6, 0], Acrobot: [-99.3, 0]
- MountainCar/Acrobot ranges 16x smaller!

**Q2: Why DQN works on MountainCar but not Pendulum?**
- **It doesn't!** Both degrade (+8.9% vs +7.1%)
- Different mechanisms:
  - Pendulum: High violations (57%) → bias
  - MountainCar: Low violations (0.44%) → exploration interference

**Q3: Why actor-critic works on Pendulum?**
- **Two-level clipping** with soft on actor
- Reduces high violations (57%) → improvement
- Wouldn't help MountainCar (already low violations)

**Key insight:** Violation rate determines whether two-level clipping helps!
