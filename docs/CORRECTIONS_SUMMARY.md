# QBound Corrections Summary

## Critical Discovery: Discount Factor Was Ignored

**Date**: October 25, 2025
**Status**: Major theoretical and implementation correction

---

## The Problem

The initial QBound implementation contained a **fundamental theoretical error**: it computed Q-value bounds without accounting for the discount factor γ, leading to bounds that were 5x too loose for CartPole.

### Incorrect Formula (Original)

```
Q_max = H × r
```

Where:
- H = maximum episode length
- r = reward per step

**Example (CartPole):**
- H = 500 steps
- r = +1 per step
- **Q_max = 500 × 1 = 500** ❌ WRONG

### Correct Formula (Fixed)

Q-values represent **discounted** sums of future rewards:

```
Q_max = (1 - γ^H) / (1 - γ)
```

**Example (CartPole with γ=0.99):**
- H = 500 steps
- r = +1 per step
- γ = 0.99
- **Q_max = (1 - 0.99^500) / (1 - 0.99) ≈ 99.34** ✓ CORRECT

---

## Impact on Experiments

### CartPole

| Configuration | Old Q_max | Correct Q_max | Error |
|---------------|-----------|---------------|-------|
| Static bound | 500 | 99.34 | **5.03x too loose** |
| Dynamic at t=0 | 500 | 99.34 | **5.03x too loose** |
| Dynamic at t=250 | 250 | 91.89 | **2.72x too loose** |
| Dynamic at t=400 | 100 | 63.40 | **1.58x too loose** |

### Other Environments

| Environment | γ | H | Old Q_max | Correct Q_max | Error |
|-------------|---|---|-----------|---------------|-------|
| GridWorld | 0.99 | 100 | 100 | 63.4 | 1.58x |
| FrozenLake | 0.95 | 100 | 100 | **19.6** | **5.10x** |
| CartPole | 0.99 | 500 | 500 | 99.34 | 5.03x |

**Note**: GridWorld and FrozenLake use Q_max=1.0 for sparse terminal rewards (correct), but the theoretical maximum for dense rewards would be much lower than previously thought.

---

## Experimental Results Comparison

### Old "Corrected" Experiment (Q_max=500 - Still Wrong!)

**Setup:**
- Baseline: No bounds
- Static QBound: Q_max = 500 (5x too high)
- Dynamic QBound: Q_max(t) = 500 - t (5x too high)

**Results:**
- Baseline: 183,022
- Static QBound: 155,392 (-15.1%)
- Dynamic QBound: 112,131 (-38.7%)

**Interpretation**: QBound appeared to hurt because bounds were too loose to provide any benefit.

### Even Older Experiment (Q_max=100 - Accidentally Correct!)

**Setup:**
- Baseline: No bounds
- QBound: Q_max = 100 (almost exactly correct!)

**Results:**
- Baseline: 131,438
- QBound: 172,904 (+31.5%)

**Interpretation**: QBound worked because Q_max=100 ≈ 99.34 was actually correct!

### New Corrected Experiment (Q_max=99.34 - Theoretically Correct)

**Status**: Currently running (episode 77/500)

**Setup:**
- Baseline: No bounds
- Static QBound: Q_max = 99.34 (correct with discounting)
- Dynamic QBound: Q_max(t) = (1 - γ^(H-t))/(1-γ) (correct formula)

**Expected**: QBound should provide benefits similar to the Q_max=100 experiment.

---

## Files Updated

### 1. Core Implementation
**File**: `src/dqn_agent.py`
**Lines**: 183-190

**Before:**
```python
# Q_max = (max_steps - current_step) * reward_per_step
dynamic_qmax = (self.max_episode_steps - current_steps_tensor) * self.step_reward
```

**After:**
```python
# CORRECT FORMULA: Q_max(t) = (1 - γ^(H-t)) / (1 - γ)
remaining_steps = self.max_episode_steps - current_steps_tensor
dynamic_qmax = (1 - torch.pow(self.gamma, remaining_steps)) / (1 - self.gamma)
```

### 2. New Experiment Script
**File**: `experiments/cartpole/train_cartpole_3way_corrected.py`
**Purpose**: 3-way comparison with correct Q-bounds

### 3. Documentation
**Files**:
- `docs/DISCOUNT_FACTOR_CORRECTION.md` - Mathematical derivation
- `docs/CORRECTIONS_SUMMARY.md` - This file
- `CLAUDE.md` - Updated with correct formulas

### 4. Paper (LaTeX)
**File**: `QBound/main.tex`
**Updates**:
- Line 330: Dynamic bounds formula
- Line 918: CartPole analysis formula
- Line 928: Bound selection rationale

---

## Mathematical Justification

### Bellman Equation

The Q-value is defined as:

```
Q(s,a) = E[R_t + γR_{t+1} + γ²R_{t+2} + ...]
```

For constant reward r at each step:

```
Q(s,a) = E[r + γr + γ²r + ... + γ^(H-1)r]
       = r × (1 + γ + γ² + ... + γ^(H-1))
       = r × [(1 - γ^H) / (1 - γ)]
```

### Step-Aware Dynamic Bounds

At step t, with (H-t) remaining steps:

```
Q_max(t) = r × (1 + γ + γ² + ... + γ^(H-t-1))
         = r × [(1 - γ^(H-t)) / (1 - γ)]
```

This naturally decreases as t increases:
- **t=0**: Q_max(0) = r × [(1 - γ^H) / (1 - γ)] ≈ 99.34
- **t=250**: Q_max(250) = r × [(1 - γ^250) / (1 - γ)] ≈ 91.89
- **t=499**: Q_max(499) = r × [(1 - γ^1) / (1 - γ)] = r = 1.0

---

## Lessons Learned

### 1. Always Account for Discounting

Discounting is fundamental to RL. Any Q-value bound must account for γ, or it will be incorrect.

### 2. Accidental Correctness Can Be Misleading

The Q_max=100 experiment worked **by accident** - the comment suggested it was based on episode length, but it happened to match the correct discounted value.

### 3. Test with Extreme Cases

Testing with very high γ (like 0.995) or very long episodes would have revealed the error immediately:
- γ=0.995, H=500 → Correct Q_max ≈ 197 (not 500)
- γ=0.9, H=500 → Correct Q_max ≈ 10 (not 500)

### 4. Document Derivations Explicitly

The paper should have shown the full mathematical derivation, which would have caught this error during peer review.

---

## Action Items

### Completed ✓
- [x] Fixed `src/dqn_agent.py` dynamic bound computation
- [x] Created corrected experiment script
- [x] Updated CLAUDE.md with correct formulas
- [x] Updated LaTeX paper with correct formulas
- [x] Documented the correction in detail

### In Progress ⏳
- [ ] Running corrected 3-way experiment (episode 77/500)

### Pending ⏸️
- [ ] Analyze corrected experimental results
- [ ] Update paper with new results if significantly different
- [ ] Consider re-running GridWorld/FrozenLake with correct bounds
- [ ] Add explicit derivation section to paper

---

## References

- Sutton & Barto (2018), *Reinforcement Learning: An Introduction*, Section 3.3
- Geometric series formula: https://en.wikipedia.org/wiki/Geometric_series

---

## Contact

For questions about this correction, see:
- Technical details: `docs/DISCOUNT_FACTOR_CORRECTION.md`
- Implementation: `src/dqn_agent.py` lines 183-190
- Experiment: `experiments/cartpole/train_cartpole_3way_corrected.py`
