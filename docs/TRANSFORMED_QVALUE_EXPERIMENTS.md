# Transformed Q-Value Experiments: Testing Negative → Positive Transformation

## Motivation

QBound showed degradation on negative reward environments (Pendulum, MountainCar, Acrobot) but success on positive reward environments (CartPole). This raises the question:

**Is QBound's failure due to the negative value range itself, or due to other factors like violation rates?**

## Hypothesis

If we transform negative reward environments to have positive Q-value ranges (similar to CartPole), QBound should work better.

## Transformation Method

### Mathematical Transformation

For environments with original bounds `Q ∈ [Q_min, 0]` where `Q_min < 0`:

1. Compute shift: `abs_Q_min = |Q_min|`
2. Transform bounds: `Q_transformed ∈ [0, abs_Q_min]`
3. Apply to targets: `Q_target_transformed = Q_target_original + abs_Q_min`

### Implementation

The transformation is applied in the DQN training loop:

```python
# Original TD target (negative range)
target_original = r + γ * max Q(s', a')

# Transform to positive range
target_transformed = target_original + abs_Q_min

# Apply QBound to transformed range [0, abs_Q_min]
target_clipped = clip(target_transformed, 0, abs_Q_min)

# Compute loss with transformed values
loss = MSE(Q_current, target_clipped)
```

## Experiments

### 1. MountainCar-v0 Transformed (DQN)

**Original bounds:** Q ∈ [-86.60, 0]
**Transformed bounds:** Q ∈ [0, 86.60]

Similar range to CartPole: Q ∈ [0, 99.34]

**Context:** Original MountainCar showed mixed results with QBound

**Methods:**
1. Baseline Transformed DQN (no QBound)
2. Static QBound + Transformed DQN

**Script:** `experiments/mountaincar/train_mountaincar_dqn_transformed.py`

### 2. Acrobot-v1 Transformed (DQN)

**Original bounds:** Q ∈ [-99.34, 0]
**Transformed bounds:** Q ∈ [0, 99.34]

Nearly identical range to CartPole: Q ∈ [0, 99.34]

**Context:** Original Acrobot showed slight improvement with QBound

**Methods:**
1. Baseline Transformed DQN (no QBound)
2. Static QBound + Transformed DQN

**Script:** `experiments/acrobot/train_acrobot_dqn_transformed.py`

### 3. Pendulum-v1 Transformed (DDPG)

**Original bounds:** Q ∈ [-1409.33, 0]
**Transformed bounds:** Q ∈ [0, 1409.33]

Larger range than CartPole but same positive structure

**Context:** Original Pendulum DDPG showed QBound WORKING (-391 → -150, 61% improvement!)

**Goal:** Test if positive transformation improves QBound performance even further

**Methods:**
1. Baseline Transformed DDPG (no QBound)
2. QBound + Transformed DDPG

**Script:** `experiments/pendulum/train_pendulum_ddpg_transformed.py`

## Expected Outcomes

### Scenario 1: Transformation Fixes QBound
If QBound shows improvement on transformed environments:
- **Conclusion:** QBound's failure was due to negative value range
- **Implication:** QBound is fundamentally incompatible with negative rewards
- **Recommendation:** Only use QBound on positive reward environments

### Scenario 2: QBound Still Degrades
If QBound still shows degradation on transformed environments:
- **Conclusion:** Issue is NOT the negative value range
- **Possible causes:**
  - High violation rates (like Pendulum)
  - Reward magnitude
  - Q-value distribution width
- **Recommendation:** Investigate violation rates and clipping bias

### Scenario 3: Mixed Results
- MountainCar/Acrobot may behave differently
- Need to analyze violation rates in transformed space
- Compare to original negative Q-value results

## Running the Experiments

### Run only transformed experiments (recommended for quick test)
```bash
python3 experiments/run_all_organized_experiments.py --category transformed --seeds 42 43 44 45 46

# 3 experiments (MountainCar, Acrobot, Pendulum)
# 2 methods each (baseline + QBound)
# Est. time: ~195 minutes per seed, ~16.25 hours for 5 seeds
```

### Run all experiments including transformed
```bash
python3 experiments/run_all_organized_experiments.py --seeds 42 43 44 45 46

# Crash recovery will skip already-completed experiments
```

### Run single seed for quick validation
```bash
python3 experiments/run_all_organized_experiments.py --category transformed --seed 42

# Est. time: ~195 minutes (~3.25 hours)
```

## Analysis

After experiments complete, compare:

1. **Original negative Q-values:**
   - `results/mountaincar/dqn_full_qbound_seed42_*.json`
   - `results/acrobot/dqn_full_qbound_seed42_*.json`
   - `results/pendulum/ddpg_full_qbound_seed42_*.json`

2. **Transformed positive Q-values:**
   - `results/mountaincar/dqn_transformed_seed42_*.json`
   - `results/acrobot/dqn_transformed_seed42_*.json`
   - `results/pendulum/ddpg_transformed_seed42_*.json`

3. **CartPole positive Q-values (reference):**
   - `results/cartpole/dqn_full_qbound_seed42_*.json`

### Key Metrics to Compare

| Metric | Description |
|--------|-------------|
| Final 100 Episode Mean | Average reward over last 100 episodes |
| QBound Improvement | % change vs baseline |
| Violation Rate | % of Q-values violating bounds |
| Violation Magnitude | How far Q-values exceed bounds |

## Implementation Details

### New Files Created

1. **`src/dqn_agent_transformed.py`**
   - TransformedDQNAgent class (for discrete actions)
   - Handles Q-value transformation in DQN training loop
   - Maintains transformed bounds [0, abs_Q_min]

2. **`src/ddpg_agent_transformed.py`**
   - TransformedDDPGAgent class (for continuous actions)
   - Handles Q-value transformation in DDPG training loop
   - Actor-critic with transformed Q-values

3. **`experiments/mountaincar/train_mountaincar_dqn_transformed.py`**
   - MountainCar with transformed Q-values (DQN)
   - Tests baseline + QBound on positive range

4. **`experiments/acrobot/train_acrobot_dqn_transformed.py`**
   - Acrobot with transformed Q-values (DQN)
   - Tests baseline + QBound on positive range

5. **`experiments/pendulum/train_pendulum_ddpg_transformed.py`**
   - Pendulum with transformed Q-values (DDPG)
   - Tests if positive range improves already-working QBound

6. **`experiments/run_all_organized_experiments.py`** (modified)
   - Added TRANSFORMED_QVALUE_EXPERIMENTS category (3 experiments)
   - New `--category transformed` option
   - Integrated into crash recovery system

## Key Design Decisions

### Why Transform in Target, Not Network Output?

We transform the TD target before computing MSE, not the network output:

**Pros:**
- Keeps network architecture unchanged
- Natural interpretation: network learns transformed Q-values
- Transformation is part of training algorithm, not network

**Alternative considered:**
- Transform network output: `Q_net = net(s) + abs_Q_min`
- Rejected: Adds constant to all outputs, less flexible

### Why Pendulum Is Included

Pendulum uses DDPG (continuous control), not DQN. While the original Pendulum DDPG showed QBound **working** (not failing), we include the transformed version to test if positive Q-value range improves performance even further.

This provides a complete picture across all three negative reward environments.

## Connection to Existing Analysis

This experiment directly tests the hypothesis from:
- `docs/WHY_MOUNTAINCAR_WORKS_BUT_PENDULUM_DOESNT.md`
- `docs/NEGATIVE_REWARD_DEGRADATION_ANALYSIS.md`

By transforming negative rewards to positive range, we isolate the effect of the value range from other factors like violation rates.
