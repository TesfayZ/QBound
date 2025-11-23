# CORRECTED ANALYSIS: QBound Empirical Results

**Date:** November 22, 2025
**Status:** URGENT - Paper contains INCORRECT claims that must be corrected

---

## Executive Summary: What the Data Actually Shows

### CORRECT Results (50 Experiments: 10 environments × 5 seeds)

**1. POSITIVE REWARDS (CartPole): QBound WORKS ✓**
- DQN + Static QBound: **+12.01% improvement** (351.07 → 393.24)
- DDQN + Static QBound: **+33.60% improvement** (147.83 → 197.50)
- Implementation: Hard clipping with Q_max = 99.34
- Mechanism: Prevents unbounded Q-value growth

**2. NEGATIVE REWARDS (Pendulum): Architectural QBound MOSTLY FAILS ✗**
- DQN + Architectural QBound: **-3.27% degradation** (-156.25 → -161.36)
- DDPG + Architectural QBound: **-8.02% degradation** (-188.63 → -203.76)
- **TD3 + Architectural QBound: +4.14% improvement** (-183.25 → -175.66) ✓ **ONLY SUCCESS**
- PPO + Architectural QBound: **-10.79% degradation** (-784.96 → -869.63)

### What the Paper Currently Claims (INCORRECT):

> "Architectural QBound achieves +2.5% to +7.2% improvements on Pendulum DQN/DDPG/TD3"

### What the Data Actually Shows (CORRECT):

> "Architectural QBound fails for most algorithms on Pendulum (DQN: -3.3%, DDPG: -8.0%, PPO: -10.8%), with TD3 being the ONLY exception showing marginal improvement (+4.1%) but with high variance (±40.15)"

---

## Detailed Results Verification

### CartPole (Positive Rewards: r = +1 per step)

| Method | Mean ± Std | Change | Status |
|--------|-----------|--------|--------|
| DQN Baseline | 351.07 ± 41.50 | - | Baseline |
| DQN + Static QBound | 393.24 ± 33.01 | **+12.01%** | ✓ Success |
| DDQN Baseline | 147.83 ± 87.13 | - | Baseline |
| DDQN + Static QBound | 197.50 ± 45.46 | **+33.60%** | ✓ Success |

**Conclusion:** Hard clipping QBound works excellently for positive rewards.

---

### Pendulum (Negative Rewards: r ∈ [-16, 0])

#### DQN Variants

| Method | Mean ± Std | Vs Baseline | Status |
|--------|-----------|-------------|--------|
| DQN Baseline | -156.25 ± 4.26 | - | Baseline |
| **Architectural QBound DQN** | **-161.36 ± 6.23** | **-3.27%** | ✗ **WORSE** |
| Double DQN | -170.01 ± 6.90 | -8.81% | Worse than baseline |
| Static QBound DDQN | -182.05 ± 4.94 | -16.52% | Much worse |

**Interpretation:**
- Less negative = BETTER (closer to 0)
- Architectural QBound makes DQN MORE negative = WORSE performance
- This CONTRADICTS the paper's claims

#### DDPG (Continuous Control)

| Method | Mean ± Std | Vs Baseline | Status |
|--------|-----------|-------------|--------|
| DDPG Baseline | -188.63 ± 18.72 | - | Baseline |
| **Architectural QBound DDPG** | **-203.76 ± 38.41** | **-8.02%** | ✗ **WORSE** |

**Interpretation:**
- Architectural QBound makes DDPG significantly WORSE
- Also increases variance (18.72 → 38.41, +105% variance increase)
- This CONTRADICTS the paper's claim of "+4.8% improvement"

#### TD3 (Twin Delayed DDPG)

| Method | Mean ± Std | Vs Baseline | Status |
|--------|-----------|-------------|--------|
| TD3 Baseline | -183.25 ± 23.36 | - | Baseline |
| **Architectural QBound TD3** | **-175.66 ± 40.15** | **+4.14%** | ✓ **BETTER** |

**Interpretation:**
- **ONLY algorithm where architectural QBound works**
- But high variance (±40.15) suggests weak/unstable effect
- Marginal improvement (7.59 units on -183 scale = 4.14%)

#### PPO (On-Policy Actor-Critic)

| Method | Mean ± Std | Vs Baseline | Status |
|--------|-----------|-------------|--------|
| PPO Baseline | -784.96 ± 269.14 | - | Baseline |
| **Architectural QBound PPO** | **-869.63 ± 133.55** | **-10.79%** | ✗ **WORSE** |

**Interpretation:**
- Architectural QBound significantly degrades PPO
- Makes performance 10.79% worse
- This CONFIRMS the paper's claim that QBound fails for on-policy methods

---

## Critical Error in Paper Claims

### Abstract (Lines 76-78) - INCORRECT

**Current claim:**
> "Architectural QBound achieves +2.5% to +7.2% improvements on Pendulum DQN/DDPG/TD3"

**Actual data:**
- Pendulum DQN: **-3.27% degradation** ✗
- Pendulum DDPG: **-8.02% degradation** ✗
- Pendulum TD3: **+4.14% improvement** ✓
- Pendulum PPO: **-10.79% degradation** ✗

**Source of error:** The documentation files (QBOUND_CLIPPING_VS_ARCHITECTURAL.md, etc.) contain incorrect percentages that don't match the actual JSON results.

---

## Why Architectural QBound Fails (Revised Theoretical Explanation)

### The Paper's Current Explanation (Lines 403-421)

Current theory claims architectural QBound works because:
1. Smooth gradients (∂Q/∂logits = -sigmoid)
2. No gradient blocking
3. Guides exploration space
4. Aligns with initialization bias

### The CORRECT Explanation (Based on Data)

**Architectural QBound FAILS because:**

1. **Bellman Equation Already Enforces Q ≤ 0 for Negative Rewards**
   - If r ≤ 0 and Q(s',a') ≤ 0
   - Then Q(s,a) = r + γ * Q(s',a') ≤ 0 + 0 = 0
   - Q ≤ 0 emerges NATURALLY from the Bellman recursion
   - Architectural constraint adds NO new information

2. **Redundant Constraint Restricts Network Expressiveness**
   - Q = -softplus(logits) forces Q ∈ (-∞, 0]
   - But network needs to learn specific negative values, not just "any negative value"
   - The constraint is too coarse—doesn't help discriminate between different action values
   - Example: Q(left) = -5.2, Q(right) = -5.1 (slight difference matters)

3. **Loss Landscape Deformation**
   - Softplus introduces non-linearity BETWEEN network output and Q-value
   - TD error: δ = r + γ*Q_next - Q_current
   - With architectural constraint: δ = r + γ*(-softplus(logits_next)) - (-softplus(logits_current))
   - Non-linear transformation changes the optimization landscape
   - May create sub-optimal local minima

4. **Gradient Magnitude Issues Near Optimal**
   - When Q is near 0 (optimal for small negative rewards):
     - logits → -∞ (to make softplus(logits) → 0)
     - ∂Q/∂logits = -sigmoid(logits) → 0 (gradient vanishes)
   - Slows learning when close to optimal values

### Why TD3 is the Exception

**TD3 has built-in mechanisms that interact positively with architectural QBound:**

1. **Twin Critics (Clipped Double Q-Learning)**
   - Q_target = r + γ * min(Q1_target, Q2_target)
   - Creates natural underestimation bias
   - For negative rewards, underestimation = less negative = BETTER
   - Twin critics already do implicit regularization
   - Architectural QBound adds small additional regularization

2. **Delayed Policy Updates**
   - Actor updates less frequently than critics
   - Gives value functions time to stabilize
   - Reduces interference between value learning and policy updates

3. **Target Policy Smoothing**
   - Adds noise to target actions
   - Smooths value landscape
   - Makes value learning more robust to architectural constraints

**BUT:** Even for TD3, the improvement is marginal (+4.14%) with very high variance (±40.15 std), suggesting the effect is weak and unstable.

---

## Correct Summary Table

| Environment | Algorithm | QBound Type | Result | Status |
|------------|-----------|-------------|--------|--------|
| CartPole (+) | DQN | Hard Clipping | +12.01% | ✓ Works |
| CartPole (+) | DDQN | Hard Clipping | +33.60% | ✓ Works |
| Pendulum (-) | DQN | Architectural | -3.27% | ✗ Fails |
| Pendulum (-) | DDPG | Architectural | -8.02% | ✗ Fails |
| Pendulum (-) | TD3 | Architectural | +4.14% | ~ Marginal (high variance) |
| Pendulum (-) | PPO | Architectural | -10.79% | ✗ Fails |

**Success Rate: 40% (2/5 for CartPole, 1/4 for Pendulum)**

---

## What Needs to Change in the Paper

### 1. Abstract (Lines 71-82)

**REMOVE ALL CLAIMS about architectural QBound success on Pendulum**

Replace with:

> For negative reward environments (Pendulum: r ∈ [-16, 0]), architectural QBound (Q = -softplus(logits)) **fails for most algorithms** (DQN: -3.3%, DDPG: -8.0%, PPO: -10.8% degradation). The failure mechanism: the Bellman equation **already constrains Q ≤ 0 naturally** for negative rewards through recursive bootstrapping—the architectural constraint is **redundant** and restricts network expressiveness without adding information. **TD3 is the ONLY exception** (+4.1% improvement), likely due to its twin critic architecture providing complementary regularization, but with high variance (±40.15) suggesting weak effect.

### 2. Section 3.2.3 - Architectural QBound (Lines 403-441)

**ADD FAILURE ANALYSIS:**

> **Empirical Results Contradict Initial Hypothesis:** Despite theoretical appeal, comprehensive evaluation shows architectural QBound **fails for 3 out of 4 algorithms** on Pendulum:
>
> - DQN: -156.25 → -161.36 (-3.27% degradation)
> - DDPG: -188.63 → -203.76 (-8.02% degradation)
> - PPO: -784.96 → -869.63 (-10.79% degradation)
> - TD3: -183.25 → -175.66 (+4.14% improvement, but ±40.15 std)
>
> **Root Cause:** For negative rewards, the Bellman equation **naturally enforces Q ≤ 0** through the recursion Q(s,a) = r + γ*Q(s',a'). When r ≤ 0 and Q(s',a') ≤ 0, then Q(s,a) ≤ 0 emerges automatically. The architectural constraint adds no new information—it's a **redundant constraint** that restricts expressiveness without improving learning. The network must still learn to discriminate between different negative values (e.g., Q = -5.2 vs -5.1), and the architectural constraint doesn't help with this discrimination task.

### 3. Recommendations Section

**CORRECT to:**

**When to Use QBound:**
1. ✓ Positive dense rewards (CartPole) → Use hard clipping (+12% to +34%)
2. ~ Negative rewards with TD3 only → Consider architectural QBound (+4.1%, high variance)
3. ✗ Negative rewards with DQN/DDPG/PPO → Do NOT use architectural QBound (degradation)
4. ✗ On-policy methods (PPO) → Do NOT use QBound

### 4. Contributions Section (Lines 183-195)

**REVISE:**

- Remove claim about "architectural constraints outperform algorithmic clipping"
- Change to: "We demonstrate architectural QBound's **limited applicability**, working only for TD3 on negative rewards (+4.1%) while failing for DQN/DDPG/PPO (-3.3% to -10.8%). This negative result is valuable: it shows that Bellman recursion naturally enforces value bounds for negative rewards, making explicit architectural constraints redundant."

---

## Files Containing Incorrect Data

These documents contain incorrect performance numbers and should be corrected or archived:

1. `docs/QBOUND_CLIPPING_VS_ARCHITECTURAL.md` - Claims +2.5% to +7.2% (WRONG)
2. `docs/ARCHITECTURAL_QBOUND_DESIGN.md` - Claims architectural QBound works (WRONG for most algorithms)
3. `docs/FINAL_SUMMARY_ARCHITECTURAL_QBOUND.md` - Contains incorrect improvement percentages

**Action:** Archive these as `docs/archive/INCORRECT_CLAIMS_*.md` with warnings

---

## Correct Theoretical Contribution

**What QBound Actually Contributes:**

1. **Positive Rewards:** Hard clipping QBound successfully prevents overestimation (+12% to +34% on CartPole)

2. **Negative Rewards:** Architectural QBound **fails for most algorithms**, revealing that:
   - Bellman equation naturally bounds Q ≤ 0 for negative rewards
   - Explicit architectural constraints are redundant
   - TD3's twin critics provide the actual mechanism for improvement (+4.1%)

3. **Key Insight:** QBound effectiveness depends on whether bounds add **new information**:
   - Positive rewards: Q_max bounds unbounded growth (helpful) ✓
   - Negative rewards: Q ≤ 0 is emergent from Bellman (redundant) ✗

**This is still a valuable contribution** - it shows when and why value bounds work or don't work!

---

## Honest Abstract (What We Should Say)

> We present QBound, a Q-value bounding technique for reinforcement learning. Comprehensive evaluation (50 runs across 10 environments) shows QBound is **highly effective for positive dense rewards** (CartPole: +12% to +34% improvement across all DQN variants) but **fails for most negative reward scenarios**.
>
> For positive rewards, hard clipping prevents unbounded Q-value growth. For negative rewards, we tested architectural constraints (Q = -softplus(logits)) and found they **degrade performance in 75% of cases** (DQN: -3.3%, DDPG: -8.0%, PPO: -10.8%), with TD3 being the only exception (+4.1%, high variance ±40.15).
>
> **Theoretical insight:** QBound works when bounds add information not implicit in Bellman equation. For positive rewards, Q_max prevents unbounded growth. For negative rewards, Q ≤ 0 emerges naturally from Bellman recursion, making architectural constraints redundant.
>
> **Recommendations:** Use QBound only for positive dense rewards with hard clipping. Do not use architectural QBound for negative rewards (except possibly TD3 with caution). This work provides systematic evidence of when and why value bounds succeed or fail.

---

## Next Steps

1. ✓ Create comprehensive analysis script (DONE: `analysis/comprehensive_qbound_analysis.py`)
2. ⚠️ **URGENT:** Correct paper abstract and all sections claiming architectural QBound success
3. ⚠️ Archive incorrect documentation files with warnings
4. ⚠️ Add "Failure Analysis" section explaining why architectural QBound fails
5. ✓ Generate corrected plots and tables
6. Re-compile paper with honest results
7. Consider this a **negative result** paper - still valuable!

---

## Why This is Still Publishable

**Negative results are valuable!** This paper:

1. Shows QBound works great for positive rewards (+12% to +34%)
2. Rigorously demonstrates where it fails (negative rewards: 3/4 algorithms)
3. Provides theoretical explanation for both success and failure
4. Contributes insight about Bellman-emergent bounds vs explicit constraints
5. Includes comprehensive experiments (50 runs, full reproducibility)

**Honest science > inflated claims**
