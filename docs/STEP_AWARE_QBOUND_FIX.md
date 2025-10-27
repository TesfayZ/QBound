# Step-Aware Q-Bound Correction

## Issue Identified

The original step-aware Q-bound implementation had a subtle but critical bug: it used the **same Q_max value** for both clipping next-state Q-values and clipping the final target.

### The Problem

For a transition `(s_t, a_t, r_t, s_{t+1})` at step `t`:

**Incorrect approach:**
```python
# Compute Q_max for current step t
Q_max(t) = (1 - γ^(H-t)) / (1 - γ)

# Clip next-state Q-value using Q_max(t)  ❌ WRONG!
next_q = clip(Q(s_{t+1}), 0, Q_max(t))

# Compute target
target = r_t + γ * next_q

# Clip target using Q_max(t)
target = clip(target, 0, Q_max(t))
```

**Why is this wrong?**

The next state `s_{t+1}` is at step `t+1`, not step `t`. The maximum return from `s_{t+1}` onwards is `Q_max(t+1)`, not `Q_max(t)`.

Using `Q_max(t)` to clip `Q(s_{t+1})` is **too loose** - it allows the next-state Q-value to be larger than theoretically possible.

## The Correct Approach

We need **two different Q_max values**:

1. **Q_max(t+1)** - for clipping next-state Q-values
2. **Q_max(t)** - for validating the final target (optional safety check)

### Mathematical Justification

For CartPole with reward `r = 1` at each step:

At step `t`:
- Q_max(t) = (1 - γ^(H-t)) / (1 - γ)
- Q_max(t+1) = (1 - γ^(H-t-1)) / (1 - γ)

If we properly clip the next-state Q-value:
```
next_q ≤ Q_max(t+1) = (1 - γ^(H-t-1)) / (1 - γ)
```

Then the target becomes:
```
target = r + γ * next_q
      ≤ 1 + γ * Q_max(t+1)
      = 1 + γ * (1 - γ^(H-t-1)) / (1 - γ)
      = 1 + (γ - γ^(H-t)) / (1 - γ)
      = (1 - γ + γ - γ^(H-t)) / (1 - γ)
      = (1 - γ^(H-t)) / (1 - γ)
      = Q_max(t)  ✓
```

**Key insight:** If `next_q` is properly bounded by `Q_max(t+1)`, then `target` will automatically satisfy `Q_max(t)` without additional clipping!

The final clipping to `Q_max(t)` is kept as a **safety check** for numerical stability, but should be redundant in theory.

## Implementation Fix

### Before (Incorrect):
```python
# Compute Q_max for current state only
remaining_steps = max_episode_steps - current_step
dynamic_qmax = (1 - gamma^remaining_steps) / (1 - gamma)

# Use same Q_max for both clipping operations ❌
next_q_values = clamp(next_q_values, 0, dynamic_qmax)
target = r + gamma * next_q_values
target = clamp(target, 0, dynamic_qmax)
```

### After (Correct):
```python
# Compute Q_max for BOTH current state (t) and next state (t+1)
remaining_steps_current = max_episode_steps - current_step
remaining_steps_next = remaining_steps_current - 1

# Q_max for current state (step t)
dynamic_qmax_current = (1 - gamma^remaining_steps_current) / (1 - gamma)

# Q_max for next state (step t+1)
dynamic_qmax_next = (1 - gamma^remaining_steps_next) / (1 - gamma)

# Clip next-state Q using Q_max(t+1) ✓
next_q_values = clamp(next_q_values, 0, dynamic_qmax_next)

# Compute target
target = r + gamma * next_q_values

# Safety clip using Q_max(t) (should be redundant)
target = clamp(target, 0, dynamic_qmax_current)
```

## Impact

This fix ensures that:
1. ✅ Next-state Q-values are bounded by the **correct** maximum (from step t+1)
2. ✅ Targets automatically satisfy current-state bounds through the Bellman equation
3. ✅ The mathematical relationship between Q_max(t) and Q_max(t+1) is preserved
4. ✅ The algorithm correctly implements the theoretical step-aware Q-bound

## Files Modified

- `src/dqn_agent.py` - Updated `train_step()` method to compute separate bounds for current and next states

## Testing Required

After this fix, re-run the CartPole 3-way comparison to verify:
1. Dynamic Q-bound still outperforms or matches baseline
2. Learning curves are stable
3. Final performance is maintained or improved

## Related Documents

- `docs/DISCOUNT_FACTOR_CORRECTION.md` - Explains the correct Q_max formula
- `docs/BLIND_EVALUATION.md` - Describes evaluation methodology
- `CLAUDE.md` - Project overview and conventions
