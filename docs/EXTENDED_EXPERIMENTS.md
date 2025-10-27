# Extended Experiments: Dynamic QBound on Sparse Negative Rewards

## Overview

This document describes the extended experiments designed to test Dynamic QBound on **time-dependent sparse reward** environments, addressing the limitation discovered in the Pendulum experiments.

## Key Insight: Reward Type Determines Q-Value Trajectory

### Dense Positive Rewards (e.g., CartPole)
- Reward: +1 per step
- Q-values **DECREASE** over time: Q(t) = remaining_steps × reward
- At t=0: Q_max = 500 (500 steps remaining)
- At t=499: Q_max = 1 (1 step remaining)
- **Dynamic QBound must DECREASE**: `Q_max(t) = (1 - γ^(H-t)) / (1-γ)`

### Sparse Negative Rewards (e.g., MountainCar)
- Reward: -1 per step until goal
- Q-values **INCREASE** over time (become less negative)
- At t=0: Q_min ≈ -100 (up to 200 steps of penalties)
- At t=199: Q_min ≈ -1 (only 1 step of penalty left)
- **Dynamic QBound must INCREASE**: `Q_min(t) = -(1 - γ^(H-t)) / (1-γ)`

## MountainCar-v0 Experiment

### Environment Characteristics
- **Action Space**: Discrete (3 actions: left, nothing, right)
- **Reward**: -1 per step until goal reached
- **Episode Length**: Max 200 steps
- **Goal**: Reach flag on mountain top

### Q-Value Bounds

**Static QBound:**
```
Q_min = -1 × (1 - γ^200) / (1 - γ) = -1 × 99.34 ≈ -100
Q_max = 0 (reach goal immediately)
Bounds: Q ∈ [-100, 0]
```

**Dynamic QBound (Increasing):**
```
At step t, remaining steps: H - t = 200 - t
Q_min(t) = -1 × (1 - γ^(200-t)) / (1 - γ)
Q_max(t) = 0 (always possible to reach goal)

Examples:
- t=0:   Q ∈ [-99.34, 0]  (strictest bound)
- t=100: Q ∈ [-63.40, 0]
- t=199: Q ∈ [-1.00, 0]   (loosest bound)
```

The bounds **increase** (become less negative) as the episode progresses, reflecting that fewer penalty steps remain.

## Double Clipping Strategy

Dynamic QBound requires **TWO clipping operations** (as pointed out):

### Step 1: Clip Next-State Q-Values
```python
# Clip to bounds at time t+1
next_q_values = torch.clamp(next_q_values,
                            min=Q_min(t+1),
                            max=Q_max(t+1))
```

### Step 2: Clip Final Target (IMPORTANT!)
```python
# Compute Bellman target
target = reward + γ × next_q_values

# Clip to bounds at time t
target = torch.clamp(target,
                    min=Q_min(t),
                    max=Q_max(t))
```

**Why both are needed:**
1. First clipping ensures next-state Q-values respect their bounds
2. Second clipping ensures the final target respects the current state's bounds
3. The second clipping is MORE IMPORTANT as it directly constrains the learning target

## 6-Way Comparison

The experiment compares:

| Method | QBound Type | Double-Q | Description |
|--------|-------------|----------|-------------|
| 1. Baseline DQN | None | No | Standard DQN (baseline) |
| 2. Static QBound + DQN | Static: [-100, 0] | No | Tests if static bounds help |
| 3. Dynamic QBound + DQN | Dynamic: [Q_min(t), 0] | No | Tests increasing bounds |
| 4. Baseline DDQN | None | Yes | Industry-standard Double DQN |
| 5. Static QBound + DDQN | Static: [-100, 0] | Yes | QBound + Double-Q |
| 6. Dynamic QBound + DDQN | Dynamic: [Q_min(t), 0] | Yes | Full enhancement |

## Crash-Resistant Implementation

Following the Pendulum 6-way pattern:

```python
RESULTS_FILE = "results/mountaincar/6way_comparison_in_progress.json"

# Load previous progress if interrupted
results = load_existing_results()

# Skip completed methods
if is_method_completed(results, 'method_name'):
    print("⏭️  Already completed, skipping...")
else:
    # Train method
    ...
    # Save immediately after completion
    save_intermediate_results(results)
```

**Benefits:**
- If computer crashes, resume from last completed method
- No need to re-run completed experiments
- Progress saved after each method finishes

## Running the Experiment

```bash
# Run full 6-way comparison (crash-resistant)
python3 experiments/mountaincar/train_mountaincar_6way.py

# If interrupted, simply re-run - it will resume automatically
python3 experiments/mountaincar/train_mountaincar_6way.py
```

## Expected Results

### Hypotheses

**H1: Static QBound helps sparse negative rewards**
- GridWorld and FrozenLake showed static QBound works for sparse rewards
- Expected: Static QBound + DQN > Baseline DQN

**H2: Dynamic QBound with increasing bounds helps**
- CartPole failed because it used DECREASING bounds on dense rewards
- MountainCar uses INCREASING bounds matching the Q-value trajectory
- Expected: Dynamic QBound + DQN > Static QBound + DQN

**H3: QBound enhances Double DQN**
- DDQN already reduces overestimation bias
- Testing if QBound provides additional benefit
- Expected: Neutral or slight improvement

### Key Metrics

1. **Total Cumulative Reward**: Sum of rewards over 500 episodes
2. **Average Episode Reward**: Mean reward per episode
3. **Average Steps to Goal**: Lower is better (faster convergence)

### Success Criteria

Dynamic QBound is considered successful if:
- Dynamic QBound + DQN achieves >5% improvement over Static QBound + DQN
- Dynamic QBound shows the bounds are correctly increasing with time
- The approach generalizes to sparse negative reward settings

## Next Steps

After MountainCar validation:

1. **Test on other sparse reward environments**:
   - Acrobot-v1 (discrete, sparse)
   - LunarLander-v2 (discrete, shaped/sparse)

2. **Test on dense reward discrete environments**:
   - CartPole with proper DECREASING bounds
   - Verify dense reward + discrete actions works

3. **Document findings** and update paper

## Implementation Files

- **Experiment**: `experiments/mountaincar/train_mountaincar_6way.py`
- **DQN Agent**: `src/dqn_agent.py` (with double clipping)
- **DoubleDQN Agent**: `src/double_dqn_agent.py` (with double clipping)
- **Results**: `results/mountaincar/6way_comparison_*.json`

## Technical Notes

### Reproducibility
- Seed: 42
- Incremental episode seeds: SEED + episode_num
- Device: CPU (for full determinism)

### Hyperparameters
- Episodes: 500
- Learning rate: 0.001
- Gamma: 0.99
- Epsilon decay: 0.995
- Batch size: 64
- Target update frequency: 100 steps

### Double Clipping Math

For negative sparse rewards at time t:

**Current state bounds (for target clipping):**
```
remaining_steps = H - t
Q_min(t) = -(1 - γ^(H-t)) / (1-γ) × step_reward
Q_max(t) = 0
```

**Next state bounds (for next-Q clipping):**
```
remaining_steps_next = H - t - 1
Q_min(t+1) = -(1 - γ^(H-t-1)) / (1-γ) × step_reward
Q_max(t+1) = 0
```

This ensures:
1. `Q(s_{t+1}) ∈ [Q_min(t+1), Q_max(t+1)]`
2. `target = r + γ × Q(s_{t+1}) ∈ [Q_min(t), Q_max(t)]`

## References

- Original QBound paper implementation: GridWorld, FrozenLake, CartPole
- Pendulum experiments: Revealed limitations with dense rewards
- MountainCar experiments: Testing sparse negative rewards with increasing bounds
