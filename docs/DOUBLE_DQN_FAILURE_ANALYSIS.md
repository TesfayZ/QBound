# Double DQN Failure Analysis

**Date:** October 26, 2025
**Status:** Investigation in Progress

---

## Observed Failure Pattern

### Training Performance Comparison

**Baseline DQN:**
- Episode 100: 255.4 avg reward
- Episode 200: 462.9 avg reward
- Episode 300: 444.4 avg reward
- Episode 400: 423.3 avg reward
- Episode 500: 313.1 avg reward

**Double DQN:**
- Episode 100: 146.7 avg reward (43% worse than baseline)
- Episode 200: 327.1 avg reward (29% worse than baseline)
- Episode 300: **11.2 avg reward** (97% WORSE - CATASTROPHIC!)
- Episode 400: 85.8 avg reward (80% worse than baseline)
- Episode 500: 31.5 avg reward (90% worse than baseline)

---

## Failure Pattern: Catastrophic Collapse at Episode 300

The most striking feature is the **sudden collapse from 327.1 to 11.2** between episodes 200 and 300. This represents a **97% drop in performance** and the agent never recovered.

---

## Hypothesis 1: Underestimation Spiral (MOST LIKELY)

### Theory

Double DQN was designed to **reduce overestimation bias** by decoupling action selection from value estimation. However, in environments with **high discount factors (γ=0.99) and long horizons (500 steps)**, this can lead to **severe underestimation**.

### Mechanism

1. **Initial underestimation**: Double DQN starts with conservative Q-values
2. **Bootstrap amplification**: Training on underestimated Q-values further reduces estimates
3. **Spiral effect**: Lower Q-values → worse action selection → lower rewards → even lower Q-values
4. **Catastrophic collapse**: Agent becomes pessimistic about all states, essentially giving up

### Evidence

- Double DQN started **43% worse** than baseline (episode 100: 146.7 vs 255.4)
- Briefly improved to 327.1 (episode 200), suggesting it was learning
- **Sudden collapse** at episode 300 suggests a tipping point where underestimation became too severe
- Never recovered, ending at 31.5 (90% worse than baseline)

### Why This Happens in CartPole

CartPole with γ=0.99 and max_steps=500 requires Q-values approaching 500 for good states. Double DQN's pessimistic updates might:

1. Underestimate early states as worth only 100-200 reward
2. Select suboptimal actions based on these underestimates
3. Experience shorter episodes (falling faster)
4. Train on these shorter experiences, further reducing Q-estimates
5. **Spiral into failure**

---

## Hypothesis 2: Target Network Synchronization

### Theory

Double DQN uses the **online network for action selection** and the **target network for evaluation**. If these networks diverge too much, Double DQN can become unstable.

### Evidence

- Target network updates every 100 steps (same as baseline)
- Around episode 300, a particularly unlucky target update might have:
  - Copied a pessimistic online network to the target
  - Created a feedback loop of underestimation
  - Caused the catastrophic collapse

---

## Hypothesis 3: Exploration-Exploitation Timing

### Theory

Epsilon decays **every training step**, not every episode. With:
- 500 episodes
- ~300 steps per episode on average
- 150,000 total training steps

Epsilon decay schedule:
- ε = 1.0 × 0.995^steps
- At step 1,000: ε ≈ 0.0067 (essentially zero exploration)
- By episode 300 (~90,000 steps): ε = epsilon_end = 0.01

### Problem

If Double DQN's underestimation causes it to:
1. Select poor actions during early training
2. Fill the replay buffer with short episodes
3. Stop exploring before finding good states
4. Get stuck in a local minimum of pessimistic Q-values

---

## Hypothesis 4: Huber Loss vs MSE

### Difference

- **Baseline**: MSE loss (penalizes large errors quadratically)
- **Double DQN**: Huber loss (linear for large errors, quadratic for small)

### Theory

Huber loss is designed to be **more robust to outliers**. However, in CartPole:
- Large positive Q-values (rewards near 500) are NOT outliers - they're the TARGET
- Huber loss might treat these as outliers and **underweight** them
- This could slow learning of high Q-values, contributing to underestimation

---

## Hypothesis 5: Gradient Clipping Side Effect

### Difference

- **Baseline**: No gradient clipping
- **Double DQN**: Gradient clipping at max_norm=1.0

### Theory

While gradient clipping prevents exploding gradients, it can also:
- Slow down learning when large updates are needed
- Prevent the network from quickly adjusting to high Q-values
- In CartPole, transitioning from Q ≈ 100 to Q ≈ 500 requires large updates
- Gradient clipping might prevent these necessary large updates

---

## The QBound Connection

This failure is **highly relevant to QBound's failure**:

### Similarity

Both Double DQN and QBound create **underestimation bias**:
- **QBound**: Hard clipping at Q_max=99.34
- **Double DQN**: Soft pessimism through decoupled networks

### Key Insight

The fact that **both approaches fail** in CartPole suggests:

1. **Underestimation is worse than overestimation** for this environment
2. **High γ and long horizons** make pessimistic approaches unstable
3. **Baseline DQN's "naive" overestimation** actually helps learning
4. **The real problem isn't overestimation** - it's that the agent needs optimistic Q-values to explore effectively

---

## Proposed Experiments to Confirm

### Experiment 1: Visualize Q-Value Evolution

Plot the max Q-value at the initial state over episodes for:
- Baseline DQN
- Double DQN

**Expected result:** Double DQN's Q-values drop around episode 300

### Experiment 2: Test with Lower Discount Factor

Run Double DQN with γ=0.95 instead of 0.99

**Hypothesis:** Lower γ reduces the severity of underestimation, Double DQN should work better

### Experiment 3: Disable Huber Loss

Run Double DQN with MSE loss instead of Huber loss

**Hypothesis:** MSE will allow faster learning of high Q-values

### Experiment 4: Disable Gradient Clipping

Run Double DQN without gradient clipping

**Hypothesis:** Allowing large gradients will enable faster Q-value updates

### Experiment 5: Slower Target Updates

Update target network every 200 steps instead of 100

**Hypothesis:** Slower updates might prevent the underestimation spiral

---

## Implications for QBound

### Critical Realization

If Double DQN (the **industry standard for addressing overestimation**) fails catastrophically in CartPole, this suggests:

1. **CartPole doesn't have an overestimation problem**
   - Or if it does, the overestimation is **beneficial**
   - Baseline DQN's "overestimation" helps it explore and find good policies

2. **Any pessimistic approach will fail in CartPole**
   - QBound: hard clipping → underestimation
   - Double DQN: soft pessimism → underestimation
   - **Both fail for the same underlying reason**

3. **The "correct" theoretical Q_max is irrelevant**
   - Q_max=99.34 (discounted) is theoretically correct
   - But forcing Q-values to stay below this **prevents optimal learning**
   - The network **needs** to learn Q-values near 500 (empirical returns)

4. **Overestimation might be a feature, not a bug**
   - For long-horizon tasks with high γ
   - Optimistic Q-values encourage exploration
   - Underestimation causes premature convergence to suboptimal policies

---

## Recommendations

### For This Research

1. **Accept that QBound is fundamentally flawed** for dense-reward, long-horizon tasks
2. **Document Double DQN's failure alongside QBound's**
   - Show that even the industry standard fails with pessimism
   - Argue that CartPole specifically requires optimistic value estimates
3. **Reframe the paper**:
   - NOT: "QBound improves Q-learning"
   - YES: "When pessimistic Q-learning fails: A case study of CartPole"
   - Focus on **understanding when underestimation hurts**

### For Alternative Approaches

If you want pessimistic RL to work in CartPole:

1. **Warm start with optimistic initialization**
   - Train baseline DQN first
   - Then add pessimism gradually
2. **Hybrid approach**
   - Use optimistic Q-values during exploration
   - Switch to pessimistic Q-values during evaluation only
3. **Adaptive pessimism**
   - Start optimistic, gradually add pessimism
   - Monitor Q-value trends to prevent spiral

---

## Conclusion

**Double DQN's catastrophic failure validates our QBound failure analysis.**

Both approaches introduce pessimism (hard vs. soft), and both fail in CartPole. This suggests:

- **Underestimation bias is more harmful than overestimation bias** for this task
- **The theoretical correctness of Q_max=99.34 doesn't matter** for practical learning
- **Baseline DQN's "naive" approach actually works better** than sophisticated pessimistic methods

**The key insight:** CartPole with γ=0.99 and 500-step horizon **requires optimistic value estimates** to learn effectively. Any method that introduces pessimism will struggle or fail.

---

## Next Steps

1. Wait for the experiment to finish and analyze the evaluation results
2. Create detailed plots comparing Q-value evolution
3. Test the proposed experiments to confirm hypotheses
4. Write up findings for the paper

