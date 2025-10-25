# QBound Experimental Analysis Summary

**Date:** October 24-25, 2025
**Experiments:** GridWorld, FrozenLake, CartPole
**Comparison:** QBound vs Baseline DQN

---

## Executive Summary

QBound shows **mixed results** across the three test environments:
- ✅ **FrozenLake:** 19.4% faster convergence (SUCCESS)
- ❌ **GridWorld:** 22.1% slower convergence (FAILURE)
- ❌ **CartPole:** 41.4% lower total reward (MAJOR FAILURE)

### Key Finding
**QBound has significant design issues** - it only works well in one specific scenario (stochastic, sparse-reward environments with low discount factors).

---

## Detailed Results by Environment

### 1. GridWorld (10x10, Goal at [9,9])

#### Configuration
- **Discount Factor (γ):** 0.99
- **QBound Limits:** [0.0, 1.0]
- **Episodes:** 500
- **Max Steps/Episode:** 100
- **Target Success Rate:** 80%

#### Performance Metrics
| Metric | QBound | Baseline | Difference |
|--------|--------|----------|------------|
| Episodes to 80% Success | 326 | 267 | -22.1% ⚠️ |
| Total Cumulative Reward | 227.0 | 279.0 | -18.6% ⚠️ |
| Final Success Rate | 86.0% | 100% | -14.0% |
| Mean Episode Reward | 0.454 | 0.558 | -18.6% |

#### Learning Progression
- **QBound:** First 100 eps: 0.180 → Last 100 eps: 0.890 (+394%)
- **Baseline:** First 100 eps: 0.060 → Last 100 eps: 1.000 (+1567%)

#### Analysis
⚠️ **Problem Identified:**
- QBound limits Q-values to [0, 1] but with γ=0.99, Q-values need to propagate backwards through states
- The theoretical maximum return is Q_max/(1-γ) = 1.0/0.01 = 100
- Clipping at 1.0 prevents proper value function learning for states far from the goal
- Baseline learns much more efficiently (1567% vs 394% improvement)

---

### 2. FrozenLake (4x4, Slippery)

#### Configuration
- **Discount Factor (γ):** 0.95
- **QBound Limits:** [0.0, 1.0]
- **Episodes:** 2000
- **Max Steps/Episode:** 100
- **Target Success Rate:** 70%

#### Performance Metrics
| Metric | QBound | Baseline | Difference |
|--------|--------|----------|------------|
| Episodes to 70% Success | 203 | 252 | +19.4% ✅ |
| Total Cumulative Reward | 1698.0 | 1753.0 | -3.1% |
| Final Success Rate | 84.0% | 96.0% | -12.5% |
| Mean Episode Reward | 0.849 | 0.876 | -3.1% |

#### Learning Progression
- **QBound:** First 100 eps: 0.030 → Last 100 eps: 0.900 (+2900%)
- **Baseline:** First 100 eps: 0.060 → Last 100 eps: 0.970 (+1517%)

#### Analysis
✅ **Success Case:**
- QBound converges 19.4% faster!
- Stochastic environment benefits from bounded Q-values
- Lower γ=0.95 means less aggressive value propagation needed
- Bounds help prevent overestimation in uncertain, slippery environment
- This is the **only environment where QBound works as intended**

---

### 3. CartPole (Balance Control)

#### Configuration
- **Discount Factor (γ):** 0.99
- **QBound Limits:** [0.0, 100.0]
- **Episodes:** 500
- **Max Steps/Episode:** 500
- **Target Success Rate:** 475.0 avg reward

#### Performance Metrics
| Metric | QBound | Baseline | Difference |
|--------|--------|----------|------------|
| Episodes to Target | Not achieved | Not achieved | N/A |
| Total Cumulative Reward | 92,111 | 157,123 | -41.4% ⚠️⚠️ |
| Mean Episode Reward | 184.2 | 314.2 | -41.4% |
| Max Episode Reward | 500 | 500 | 0% |

#### Learning Progression
- **QBound:** First 100 eps: 106.4 → Last 100 eps: 187.7 (+76%)
- **Baseline:** First 100 eps: 190.3 → Last 100 eps: 405.7 (+113%)

#### Analysis
⚠️⚠️ **CRITICAL FAILURE:**
- QBound severely underperforms with 41.4% lower total reward
- **Root cause:** QBound max of 100 is too restrictive
- Optimal episode reward is 500 (survive all steps)
- With γ=0.99, optimal Q-value approaches 500, but QBound clips at 100
- This **fundamentally prevents learning the optimal policy**
- Theoretical max return: Q_max/(1-γ) = 100/0.01 = 10,000 (way too high)
- But practical max episode return is 500, so Q_max should be ≥500

---

## QBound Configuration Analysis

### Current Configurations

| Environment | Q_min | Q_max | γ | Theoretical Max |
|------------|-------|-------|---|-----------------|
| GridWorld | 0.0 | 1.0 | 0.99 | 100.0 |
| FrozenLake | 0.0 | 1.0 | 0.95 | 20.0 |
| CartPole | 0.0 | 100.0 | 0.99 | 10,000.0 |

### Issues Identified

1. **GridWorld:**
   - Q_max=1.0 matches immediate reward but ignores value propagation
   - States far from goal need Q-values to accumulate through Bellman updates
   - Clipping prevents this accumulation

2. **FrozenLake:**
   - Configuration is appropriate (ONLY success case)
   - Stochastic transitions benefit from bounded estimates
   - Lower γ=0.95 aligns with Q_max=1.0

3. **CartPole:**
   - Q_max=100 is far too low for max episode return of 500
   - Should be Q_max ≥ 500 to allow optimal policy learning

---

## Design Issues Summary

### 🚨 Critical Problems

1. **QBound limits must match episode return, not step reward**
   - Current approach: Set Q_max to max step reward
   - Correct approach: Set Q_max to max cumulative episode return

2. **High discount factors incompatible with tight bounds**
   - γ=0.99 with Q_max=1.0 prevents value propagation
   - Either increase Q_max or decrease γ

3. **Only works in specific scenario:**
   - ✅ Stochastic environments
   - ✅ Sparse rewards
   - ✅ Lower discount factors (γ ≤ 0.95)
   - ❌ Fails otherwise

### Performance Summary

**Environments where QBound is BETTER:**
- ✅ FrozenLake: +19.4% convergence speed

**Environments where QBound is WORSE:**
- ❌ GridWorld: -22.1% convergence speed, -18.6% total reward
- ❌ CartPole: -41.4% total reward (SEVERE)

**Environments where neither converged:**
- ⚠️ CartPole: Both methods failed to reach target (but baseline performed much better)

---

## Recommendations for Paper

### 1. Acknowledge Limitations
QBound is **not a general-purpose improvement** to DQN. It has a narrow applicability window.

### 2. Revise QBound Configuration
For environments to be included in the paper:
- **GridWorld:** Increase Q_max to 10.0 or higher
- **CartPole:** Increase Q_max to 500.0

### 3. Focus on Success Case
The paper should emphasize:
- QBound's effectiveness in **stochastic, sparse-reward environments**
- FrozenLake as the primary success demonstration
- Clear guidelines for when to use/not use QBound

### 4. Additional Experiments Needed
To validate the hypothesis:
- Test with adjusted Q_max values
- Test with various γ values (0.9, 0.95, 0.99)
- Track actual Q-value ranges during training
- Measure QBound violation rates over time

### 5. Theoretical Analysis
Include in paper:
- Relationship between Q_max and max episode return
- Impact of discount factor on required Q_max
- When bounds help vs. hurt learning

---

## Next Steps

1. ✅ **Run comprehensive analysis** (tracking Q-values, violations, plots)
2. ⏳ **Generate publication plots** (currently running)
3. 🔄 **Re-run experiments with corrected Q_max values**
4. 📊 **Show violation rate decrease over time**
5. 📝 **Update paper with honest assessment**

---

## Conclusion

The current QBound implementation has **significant design flaws**. The bounds are set incorrectly (based on step rewards rather than episode returns), causing it to:
- ✅ Help in 1/3 environments (FrozenLake)
- ❌ Hurt in 2/3 environments (GridWorld, CartPole)

**The paper cannot be published with these results** without:
1. Fixing the Q_max configurations
2. Clearly defining when QBound should be used
3. Providing theoretical justification for bound selection
4. Being transparent about failure cases
