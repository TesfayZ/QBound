# Critical Bug Analysis: Dynamic QBound Sign Error on Negative Rewards

## Executive Summary

**Found a SIGN ERROR in the dynamic QBound implementation for negative rewards!**

The bug causes Q-bounds to become INVERTED, making Q_min positive and invalid, which explains the catastrophic -500% to -746% performance degradation.

---

## The Bug

### Location
`src/dqn_agent.py`, lines 204-209:

```python
if self.reward_is_negative:
    # Negative rewards: Q ∈ [Q_min(t), 0]
    dynamic_qmin_current = -geometric_sum_current * self.step_reward  # ← BUG HERE
    dynamic_qmax_current = torch.zeros_like(dynamic_qmin_current)
    dynamic_qmin_next = -geometric_sum_next * self.step_reward        # ← BUG HERE
    dynamic_qmax_next = torch.zeros_like(dynamic_qmin_next)
```

### What Goes Wrong

For Pendulum with dynamic QBound:
- **Input**: `step_reward = -16.27` (negative value)
- **Input**: `reward_is_negative = True`
- **Calculation**: `geometric_sum_current ≈ 86.60` (always positive)

**Line 206 computes**:
```python
dynamic_qmin_current = -geometric_sum_current * self.step_reward
                     = -(86.60) * (-16.27)
                     = +1409.33  ← WRONG! Q_min should be NEGATIVE!
```

**Line 207 computes**:
```python
dynamic_qmax_current = 0.0  ← Correct
```

**Result**: The bounds become `Q ∈ [+1409, 0]`, which is **inverted and invalid**!

This forces all Q-values into an impossible range, completely breaking the learning.

---

## Why Static QBound Works

### Static Bounds Setup
In `experiments/pendulum/train_pendulum_dqn_full_qbound.py`:

```python
# Lines 72-77
QBOUND_MIN = -1409.3272174664303  # PRE-CALCULATED correctly
QBOUND_MAX = 0.0
```

For static QBound agents:
```python
static_qbound_dqn_agent = DQNAgent(
    ...
    use_qclip=True,
    qclip_min=QBOUND_MIN,  # = -1409.33 (correct)
    qclip_max=QBOUND_MAX,  # = 0.0 (correct)
    use_step_aware_qbound=False  # ← Uses static bounds
)
```

**Static bounds path** (line 217-221):
```python
else:
    # Use static Q bounds (same for current and next state)
    dynamic_qmax_current = torch.full((self.batch_size,), self.qclip_max, device=self.device)  # = 0.0
    dynamic_qmin_current = torch.full((self.batch_size,), self.qclip_min, device=self.device)  # = -1409.33
```

This directly uses the pre-calculated `qclip_min = -1409.33`, which is **already negative and correct**.

---

## Theoretical vs Implementation

### Correct Theoretical Formula

For negative rewards where `r < 0`:

**Q-value at time t**:
```
Q(s_t, a) = sum_{k=0}^{H-t-1} γ^k * r
          = r * sum_{k=0}^{H-t-1} γ^k
          = r * (1 - γ^(H-t)) / (1 - γ)
```

Where:
- `r = -16.27` (negative)
- `(1 - γ^(H-t)) / (1 - γ) = geometric_sum` (positive)

**Result**: `Q_min(t) = r * geometric_sum = (-16.27) * (86.60) = -1409.33` ✅ **NEGATIVE**

### Buggy Implementation

```python
dynamic_qmin_current = -geometric_sum_current * self.step_reward
                     = -geometric_sum_current * (-16.27)
                     = geometric_sum_current * 16.27
                     = +1409.33  ✗ POSITIVE (WRONG!)
```

The bug multiplies by `-1` TWICE:
1. Once explicitly in the code: `-geometric_sum_current`
2. Once implicitly because `step_reward` is already negative: `* (-16.27)`

---

## The Correct Fix

### Option 1: Remove the Leading Minus Sign

```python
if self.reward_is_negative:
    # Negative rewards: Q ∈ [Q_min(t), 0]
    dynamic_qmin_current = geometric_sum_current * self.step_reward  # ← FIXED: removed leading minus
    dynamic_qmax_current = torch.zeros_like(dynamic_qmin_current)
    dynamic_qmin_next = geometric_sum_next * self.step_reward        # ← FIXED: removed leading minus
    dynamic_qmax_next = torch.zeros_like(dynamic_qmin_next)
```

**Why this works**:
- `geometric_sum_current = 86.60` (positive)
- `self.step_reward = -16.27` (negative)
- `dynamic_qmin_current = 86.60 * (-16.27) = -1409.33` ✅ **NEGATIVE**

### Option 2: Use Absolute Value for step_reward

```python
if self.reward_is_negative:
    # Negative rewards: Q ∈ [Q_min(t), 0]
    # step_reward should be POSITIVE magnitude, code applies sign
    dynamic_qmin_current = -geometric_sum_current * abs(self.step_reward)
    dynamic_qmax_current = torch.zeros_like(dynamic_qmin_current)
    dynamic_qmin_next = -geometric_sum_next * abs(self.step_reward)
    dynamic_qmax_next = torch.zeros_like(dynamic_qmin_next)
```

And pass `step_reward=16.27` (positive magnitude) instead of `step_reward=-16.27`.

**Recommendation**: Use **Option 1** (simpler, requires no parameter changes).

---

## Why This Explains the Catastrophic Failure

### Invalid Bounds Break Learning

With inverted bounds `Q ∈ [+1409, 0]`:

1. **All Q-values are clipped to 0** (the max of the invalid range)
2. **Target computation**: `target = r + γ * clip(next_Q, +1409, 0)`
   - If `next_Q = -200` (reasonable), it gets clipped to `0`
   - `target = -16.27 + 0.99 * 0 = -16.27`
3. **Network tries to predict -16.27 for all states**
4. **Loss explodes** because this contradicts the Bellman equation

**Result**: The agent learns complete nonsense, achieving -1200 to -1300 reward (vs baseline -180).

### Consistency Across All Architectures

The bug affects **all DQN variants**:
- Standard DQN: -746% degradation
- Double DQN: -638% degradation
- DDPG (continuous): -538% degradation
- TD3 (continuous): -523% degradation

All use the same buggy dynamic bound calculation → all fail catastrophically.

---

## Expected Performance After Fix

### Hypothesis
If we fix the sign error, dynamic QBound should:

1. **Work correctly on Pendulum** (negative rewards)
2. **Potentially show benefit** similar to CartPole (positive rewards)
3. **Maintain Double DQN synergy** (if the algorithm interaction holds)

### Predicted Results (after fix)
- **Pendulum DQN + Dynamic QBound**: -140 to -170 (similar to static, or better)
- **Pendulum DDPG + Dynamic QBound**: -170 to -190 (similar to baseline)
- **Pendulum TD3 + Dynamic QBound**: -150 to -180 (potentially improved)

Currently:
- **Pendulum DQN + Dynamic QBound**: -1322 (catastrophic failure)
- **Pendulum DDPG + Dynamic QBound**: -1203 (catastrophic failure)
- **Pendulum TD3 + Dynamic QBound**: -1217 (catastrophic failure)

---

## Verification of Static QBound

### Why Static Works
The pre-calculated value is correct:
```python
# Manual calculation for Pendulum
r = -16.27
H = 200
γ = 0.99
geometric_sum = (1 - 0.99^200) / (1 - 0.99) = 86.596...
Q_min = r * geometric_sum = -16.27 * 86.596 = -1409.33 ✅
```

Static QBound directly uses `qclip_min = -1409.33` (correct value) → Works fine.

Dynamic QBound tries to COMPUTE this but has a sign error → Fails catastrophically.

---

## Impact on Paper

### Current Narrative (INCORRECT)
"Dynamic QBound is theoretically invalid for negative rewards"

### Corrected Narrative (AFTER FIX)
"Dynamic QBound implementation had a sign error for negative rewards, which has been corrected. After the fix, dynamic QBound [results pending re-run]"

### Action Items
1. ✅ **Bug identified**: Sign error in `src/dqn_agent.py` line 206, 208
2. ⚠️ **Fix required**: Remove leading minus sign in negative reward case
3. ⚠️ **Re-run needed**: All Pendulum dynamic QBound experiments (30 runs: 3 architectures × 2 DQN variants × 5 seeds)
4. 📝 **Paper update**: Revise dynamic QBound discussion after obtaining corrected results

---

## Conclusion

The catastrophic failure of dynamic QBound on negative rewards is **NOT a theoretical limitation** but a **sign error in the implementation**.

Static QBound works because it uses pre-calculated (correct) bounds.

Dynamic QBound fails because it tries to compute bounds at runtime but applies the negative sign twice.

**This is FIXABLE** and should be corrected before paper submission.
