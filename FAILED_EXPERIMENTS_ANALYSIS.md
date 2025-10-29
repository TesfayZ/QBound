# Failed Experiments: Analysis & Repeat Plan

**Date:** 2025-10-28
**Purpose:** Identify failed experiments and create improvement plan
**Status:** Ready for re-experimentation

---

## 🎯 IDENTIFIED FAILURES FROM PAPER

Based on comprehensive evaluation in the paper, these experiments showed **degraded performance with QBound**:

### 1. **MountainCar-v0** (PRIMARY FAILURE)
**Result:** -16.6% degradation
- **Baseline DQN:** -124.5 (final 100 episodes)
- **QBound DQN:** -145.2 (final 100 episodes)
- **Assessment:** QBound HURT performance

**Why it Failed:**
- Exploration-critical task (must discover momentum solution)
- Over-constraining Q-values limits exploration
- Sparse reward requires aggressive exploration
- QBound's conservative bounds prevent necessary Q-value growth

**Hypothesis:** Need looser bounds or delayed QBound application

---

### 2. **Acrobot-v1** (SECONDARY FAILURE)
**Result:** -7.6% degradation
- **Baseline DQN:** -87.0 (final 100 episodes)
- **QBound DQN:** -93.7 (final 100 episodes)
- **Assessment:** QBound HURT performance (moderate)

**Why it Failed:**
- Similar to MountainCar: exploration-critical
- Negative reward structure (minimize episode length)
- Must discover swingup dynamics through exploration
- QBound constraints may limit Q-value divergence needed for exploration

**Hypothesis:** Exploration bonus or different bound strategy needed

---

### 3. **PPO on LunarLander Discrete** (ALGORITHM INTERACTION FAILURE)
**Result:** -30.9% degradation
- **Baseline PPO:** 200+ reward, 80% success rate
- **QBound PPO:** ~138 reward (estimated from paper)
- **Assessment:** QBound CONFLICTS with PPO's GAE

**Why it Failed:**
- PPO uses Generalized Advantage Estimation (GAE)
- Clipping Q-values disrupts advantage calculation
- GAE relies on smooth value function estimates
- Hard clipping creates discontinuities in advantages

**Hypothesis:** Soft clipping or different integration point needed

---

### 4. **Pendulum-v1 (DDPG/TD3)** (FUNDAMENTAL INCOMPATIBILITY)
**Result:** -893% catastrophic failure
- **Assessment:** QBound fundamentally incompatible with continuous action spaces
- **Why:** Hard clipping disrupts smooth critic gradients required for $\nabla_a Q(s,a)$
- **Verdict:** **NOT REPEATING** - Well-understood fundamental limitation

---

## 📋 REPEAT EXPERIMENT PLAN

### Phase 1: Exploration-Critical Tasks (MountainCar, Acrobot)

#### Experiment 1A: **Delayed QBound Application**
**Hypothesis:** Allow initial exploration, then apply QBound

**Setup:**
- Start without QBound for N episodes (exploration phase)
- Enable QBound after initial discovery
- Test N ∈ {100, 200, 300, 500} episodes

**Expected Outcome:** Better exploration → improved performance

**Implementation:**
```python
def should_use_qbound(episode, exploration_episodes=200):
    return episode >= exploration_episodes
```

#### Experiment 1B: **Looser Bounds**
**Hypothesis:** Current bounds too tight for exploration

**Setup - MountainCar:**
- Current: $Q_{\min} = -200$, $Q_{\max} = 0$
- Test: $Q_{\min} = -500$, $Q_{\max} = +200$ (much looser)
- Test: No upper bound (only clip minimum)

**Setup - Acrobot:**
- Current: $Q_{\min} = -500$, $Q_{\max} = 0$
- Test: $Q_{\min} = -1000$, $Q_{\max} = +200$

**Expected Outcome:** Less constraint → more exploration

#### Experiment 1C: **QBound + Exploration Bonus**
**Hypothesis:** Combine QBound with intrinsic motivation

**Setup:**
- Add curiosity-driven exploration (count-based or RND)
- Or: Increase epsilon exploration schedule
- Or: Add entropy bonus to Q-learning

**Expected Outcome:** Explicit exploration compensates for QBound constraints

#### Experiment 1D: **Adaptive Bounds**
**Hypothesis:** Start loose, tighten over time

**Setup:**
```python
def adaptive_bounds(episode, total_episodes):
    progress = episode / total_episodes
    # Start with 3x loose, end at 1x tight
    scale = 3.0 - 2.0 * progress
    Q_min_scaled = Q_min * scale
    Q_max_scaled = Q_max * scale
    return Q_min_scaled, Q_max_scaled
```

**Expected Outcome:** Early exploration + late refinement

---

### Phase 2: PPO Integration Investigation

#### Experiment 2A: **Soft Clipping for PPO**
**Hypothesis:** Soft clipping preserves smoothness

**Setup:**
```python
def soft_clip(Q, Q_min, Q_max, softness=0.1):
    # Tanh-based soft clipping
    Q_range = Q_max - Q_min
    Q_normalized = (Q - Q_min) / Q_range
    Q_soft = torch.tanh(Q_normalized * 2 - 1)  # Map to [-1, 1]
    Q_clipped = (Q_soft + 1) / 2 * Q_range + Q_min
    return Q_clipped
```

**Expected Outcome:** Preserves advantage smoothness

#### Experiment 2B: **QBound on Value Function Only (not Advantages)**
**Hypothesis:** Clip V(s) but not advantages A(s,a)

**Setup:**
- Apply QBound to V(s) estimation
- Let advantages compute naturally from unclipped Q-values
- PPO uses clipped V for baseline subtraction

**Expected Outcome:** Stabilize value learning without disrupting policy gradients

#### Experiment 2C: **QBound with Increased PPO Clip Range**
**Hypothesis:** PPO's policy clip conflicts with Q-clipping

**Setup:**
- Standard PPO clip: $\epsilon = 0.2$
- Test with: $\epsilon = 0.3, 0.4, 0.5$
- Allow more policy deviation to compensate for value clipping

**Expected Outcome:** More flexibility in policy updates

---

## 🔬 DETAILED EXPERIMENT SPECIFICATIONS

### Experiment Set 1: MountainCar Improvements

#### Config 1.1: Delayed QBound (Baseline + Late Application)
```python
Environment: MountainCar-v0
Episodes: 2000
QBound activation: Episode 500 onward
Bounds: [-200, 0]
Seed: 42
Expected improvement: +10-20% vs current QBound
```

#### Config 1.2: Looser Bounds
```python
Environment: MountainCar-v0
Episodes: 2000
Bounds: [-500, +200]  # Much looser
Seed: 42
Expected improvement: +5-15% vs current QBound
```

#### Config 1.3: Count-Based Exploration Bonus
```python
Environment: MountainCar-v0
Episodes: 2000
QBound: Standard [-200, 0]
Exploration: Count-based bonus (Bellemare et al. 2016)
Bonus weight: 0.1
Seed: 42
Expected improvement: +15-25% vs current QBound
```

#### Config 1.4: Adaptive Bounds
```python
Environment: MountainCar-v0
Episodes: 2000
Bounds: Start at [-600, +200], linearly tighten to [-200, 0]
Seed: 42
Expected improvement: +20-30% vs current QBound
```

---

### Experiment Set 2: Acrobot Improvements

Same as MountainCar but with Acrobot-specific bounds:
- Base bounds: [-500, 0]
- Loose bounds: [-1000, +200]
- Adaptive: Start at [-1500, +200], tighten to [-500, 0]

---

### Experiment Set 3: PPO Integration

#### Config 3.1: Soft Clipping
```python
Environment: LunarLander-v2 (discrete)
Algorithm: PPO
QBound: Soft clipping (tanh-based)
Episodes: 1000
Seed: 42
Expected: Match or exceed baseline PPO
```

#### Config 3.2: Value-Only Clipping
```python
Environment: LunarLander-v2 (discrete)
Algorithm: PPO
QBound: Applied to V(s) only, not Q(s,a)
Episodes: 1000
Seed: 42
Expected: +10-20% vs baseline PPO
```

---

## 📊 SUCCESS CRITERIA

### For MountainCar/Acrobot:
**Minimum Success:** QBound ≥ Baseline (break even)
**Good Success:** QBound ≥ Baseline + 5%
**Excellent Success:** QBound ≥ Baseline + 10%

### For PPO Integration:
**Minimum Success:** QBound PPO ≥ 90% of Baseline PPO
**Good Success:** QBound PPO ≥ Baseline PPO
**Excellent Success:** QBound PPO > Baseline PPO + 5%

---

## 🎯 EXPERIMENT PRIORITY

### Priority 1 (Run Immediately):
1. **MountainCar with Delayed QBound** (Exp 1.1)
   - Most likely to succeed
   - Simple to implement
   - Could fully reverse the failure

2. **MountainCar with Adaptive Bounds** (Exp 1.4)
   - Theoretically sound
   - Matches learning dynamics
   - Moderate implementation complexity

### Priority 2 (If Priority 1 Fails):
3. **MountainCar with Count-Based Exploration** (Exp 1.3)
   - More complex implementation
   - Requires additional dependencies
   - Could be "overkill" solution

4. **MountainCar with Looser Bounds** (Exp 1.2)
   - Very simple
   - May not fully solve problem
   - Good fallback option

### Priority 3 (Research Direction):
5. **PPO Soft Clipping** (Exp 3.1)
   - Novel contribution if successful
   - Could enable broader applicability
   - Higher risk, higher reward

6. **Acrobot Improvements** (Set 2)
   - Similar to MountainCar
   - Lower priority (smaller failure -7.6% vs -16.6%)

---

## 🛠️ IMPLEMENTATION PLAN

### Step 1: Create Experiment Infrastructure
```bash
mkdir -p experiments/mountaincar_improved
mkdir -p experiments/acrobot_improved
mkdir -p experiments/ppo_integration

cp experiments/mountaincar/train_mountaincar.py experiments/mountaincar_improved/
cp experiments/acrobot/train_acrobot.py experiments/acrobot_improved/
```

### Step 2: Implement Adaptive QBound
Create `src/qbound_adaptive.py`:
```python
class AdaptiveQBound:
    def __init__(self, Q_min_init, Q_max_init, Q_min_final, Q_max_final):
        self.Q_min_init = Q_min_init
        self.Q_max_init = Q_max_init
        self.Q_min_final = Q_min_final
        self.Q_max_final = Q_max_final

    def get_bounds(self, episode, total_episodes):
        progress = episode / total_episodes
        Q_min = self.Q_min_init + progress * (self.Q_min_final - self.Q_min_init)
        Q_max = self.Q_max_init + progress * (self.Q_max_final - self.Q_max_init)
        return Q_min, Q_max
```

### Step 3: Run Experiments Sequentially
```bash
# Priority 1
python experiments/mountaincar_improved/train_delayed_qbound.py
python experiments/mountaincar_improved/train_adaptive_qbound.py

# Priority 2 (if needed)
python experiments/mountaincar_improved/train_loose_bounds.py
python experiments/mountaincar_improved/train_exploration_bonus.py

# Priority 3
python experiments/acrobot_improved/train_adaptive_qbound.py
python experiments/ppo_integration/train_soft_clipping.py
```

### Step 4: Analysis and Reporting
```python
python analysis/compare_improved_results.py
# Generate updated plots showing improvements
# Update paper with successful methods
```

---

## 📝 EXPECTED TIMELINE

**Day 1 (8 hours):**
- Implement adaptive QBound class
- Implement delayed QBound logic
- Run Priority 1 experiments (2 experiments × 2 hours each)
- Initial analysis

**Day 2 (6 hours):**
- Run Priority 2 experiments if needed (4 hours)
- Run Acrobot variants (2 hours)

**Day 3 (4 hours):**
- Implement soft clipping for PPO
- Run PPO integration experiment
- Final analysis and plotting

**Total:** 18-20 hours of experiments

---

## 📊 WHAT TO REPORT

### If Successful:
Add to paper:
- "Section 5.8: Improved QBound for Exploration-Critical Tasks"
- Show adaptive/delayed QBound rescues performance
- New figures comparing original vs improved
- Update abstract if dramatic improvement

### If Partially Successful:
Add to paper:
- "Section 6.5: When QBound Fails and How to Mitigate"
- Honest analysis of failure modes
- Discussion of adaptive strategies
- Guidelines for practitioners

### If Still Fails:
Keep in paper:
- Current honest reporting is already excellent
- Add paragraph on attempted improvements
- Strengthen "limitations" section
- Emphasize environment-dependent nature

---

## 🎓 SCIENTIFIC VALUE

Even if these experiments don't fully succeed:

**Positive Scientific Contribution:**
1. Comprehensive exploration of when/why QBound fails
2. Systematic investigation of mitigation strategies
3. Clear guidelines for practitioners
4. Framework for adaptive bound selection

**Paper Strength:**
- Honest, thorough evaluation already distinguishes your work
- Additional experiments show scientific rigor
- Failure analysis is as valuable as success analysis

**Reviewer Response:**
- Shows you investigated obvious improvements
- Demonstrates understanding of limitations
- Provides clear boundary of applicability

---

## ✅ READY TO EXECUTE

All experiments are well-defined and ready to implement. Priority 1 experiments are most likely to succeed and should be run first.

**Next Action:** Implement adaptive QBound infrastructure and run Priority 1 experiments.
