# Discount Factor Correction for QBound

## Critical Discovery: Q_max Must Account for Discounting

### The Problem

The initial QBound implementation made a **fundamental error** in computing Q-value bounds: it ignored the discount factor γ.

### Incorrect Formula (Old)

```
Q_max = H × r
```

Where:
- H = maximum episode length
- r = reward per step

**Example for CartPole:**
- H = 500 steps
- r = +1 per step
- **Q_max = 500 × 1 = 500** ❌

### Correct Formula (New)

The Q-value is the **discounted** sum of future rewards:

```
Q(s,a) = r₁ + γr₂ + γ²r₃ + ... + γ^(H-1)r_H
```

For constant reward r=1, this is a geometric series:

```
Q_max = Σ(t=0 to H-1) γ^t = (1 - γ^H) / (1 - γ)
```

**Example for CartPole with γ=0.99:**
- H = 500 steps
- γ = 0.99
- **Q_max = (1 - 0.99^500) / (1 - 0.99) = 99.34** ✓

### Impact on Results

| Configuration | Old Q_max | Correct Q_max | Error Factor |
|---------------|-----------|---------------|--------------|
| CartPole static | 500 | 99.34 | 5.03x too high |
| CartPole at step 0 | 500 | 99.34 | 5.03x too high |
| CartPole at step 100 | 400 | 98.20 | 4.07x too high |
| CartPole at step 250 | 250 | 91.89 | 2.72x too high |
| CartPole at step 400 | 100 | 63.40 | 1.58x too high |

The old "Q_max=100" experiment was actually **almost correct by accident** - it just wasn't properly explained!

## Mathematical Derivation

### Static Bounds

For an environment with:
- Maximum episode length: H
- Reward per step: r
- Discount factor: γ

The maximum Q-value at the start of an episode is:

```
Q_max = r·(1 + γ + γ² + ... + γ^(H-1))
      = r·[(1 - γ^H) / (1 - γ)]
```

### Dynamic Step-Aware Bounds

At step t, the remaining steps are (H - t), so:

```
Q_max(t) = r·[(1 - γ^(H-t)) / (1 - γ)]
```

This naturally decreases as the episode progresses:
- **At t=0:** Q_max(0) = r·[(1 - γ^H) / (1 - γ)]
- **At t=H-1:** Q_max(H-1) = r·[(1 - γ^1) / (1 - γ)] = r
- **At t=H:** Q_max(H) = 0 (episode ended)

## Implementation Fix

### Before (Incorrect)

```python
# In dqn_agent.py train_step()
if self.use_step_aware_qbound:
    remaining_steps = self.max_episode_steps - current_steps
    dynamic_qmax = remaining_steps * self.step_reward  # WRONG!
```

### After (Correct)

```python
# In dqn_agent.py train_step()
if self.use_step_aware_qbound:
    remaining_steps = self.max_episode_steps - current_steps
    # Correct formula with discounting
    dynamic_qmax = (1 - gamma**remaining_steps) / (1 - gamma)  # CORRECT!
```

## Experimental Validation

### Old Experiment (Incorrect Bounds)

**Setup:**
- Baseline: No bounds
- Static QBound: Q_max = 500 (5x too high)
- Dynamic QBound: Q_max(t) = 500 - t (5x too high at start)

**Results:**
- Baseline: 183,022
- Static QBound: 155,392 (-15.1%) ❌
- Dynamic QBound: 112,131 (-38.7%) ❌

**Conclusion:** QBound appeared to hurt performance because bounds were too loose.

### Corrected Experiment (Correct Bounds)

**Setup:**
- Baseline: No bounds
- Static QBound: Q_max = 99.34 (correct)
- Dynamic QBound: Q_max(t) = (1 - γ^(H-t))/(1-γ) (correct)

**Results:** Running now...

## Why This Matters

### 1. Theoretical Correctness

QBound is based on bounding Q-values to their **theoretically possible** range. Ignoring discounting means the bounds are not actually valid - they allow Q-values far beyond what's achievable.

### 2. Practical Performance

Loose bounds (5x too high) provide no benefit:
- They don't prevent Q-value overestimation
- They don't provide regularization
- They just add computational overhead

Tight, correct bounds should:
- Prevent impossible Q-values
- Guide learning toward realistic estimates
- Improve sample efficiency

### 3. Environment-Specific Implications

| Environment | γ | H | Reward | Q_max (wrong) | Q_max (correct) | Ratio |
|-------------|---|---|--------|---------------|-----------------|-------|
| GridWorld | 0.99 | 100 | 1 | 100 | 63.4 | 1.58x |
| FrozenLake | 0.95 | 100 | 1 | 100 | 19.6 | 5.10x |
| CartPole | 0.99 | 500 | 1 | 500 | 99.3 | 5.03x |

**FrozenLake has the biggest error!** With γ=0.95, the correct Q_max is only 19.6, not 100!

## Recommendations

### For All Environments

Always compute Q_max using the discounted formula:

```python
def compute_qmax(max_steps, reward_per_step, gamma):
    """Compute correct Q_max with discounting."""
    return reward_per_step * (1 - gamma**max_steps) / (1 - gamma)
```

### For Sparse Rewards

In environments like GridWorld where only the terminal state gives reward:
- The discount factor still matters
- Q_max = γ^(distance_to_goal) × terminal_reward

### For Dense Rewards

In environments like CartPole where every step gives reward:
- Use the geometric series formula
- Consider step-aware bounds: Q_max(t) decreases over episode

## Updated Code Files

1. **src/dqn_agent.py** - Fixed dynamic Q-max computation
2. **experiments/cartpole/train_cartpole_3way_corrected.py** - New experiment with correct bounds
3. **docs/DISCOUNT_FACTOR_CORRECTION.md** - This document

## References

- Sutton & Barto, *Reinforcement Learning: An Introduction*, Chapter 3.3 (Returns and Episodes)
- Geometric series formula: https://en.wikipedia.org/wiki/Geometric_series
