# Dynamic QBound Removal Documentation

**Date**: 2025-11-16
**Reason**: Analysis showed Dynamic QBound never outperformed Static QBound

## Analysis Summary

### Experiments Analyzed

1. **CartPole DDQN (Positive Time-Step Rewards)**
   - Static QBound: 186.90
   - Dynamic QBound: 148.25
   - **Result**: Static wins by +38.65

2. **Pendulum DQN (Negative Time-Step Rewards) - 3 seeds**
   - Seed 43: Static -168.67 vs Dynamic -175.39 (Δ = -6.72)
   - Seed 44: Static -157.72 vs Dynamic -158.33 (Δ = -0.61)
   - Seed 45: Static -174.10 vs Dynamic -180.55 (Δ = -6.45)
   - **Result**: Static avg -166.83 vs Dynamic avg -171.42

### Verdict

**0 wins out of 4 comparisons** - Dynamic QBound never outperformed Static QBound.

## What Was Removed

### 1. Experiment Scripts

Changed from **3-way comparison** (baseline, static, dynamic) to **2-way comparison** (baseline, static):

- `experiments/cartpole/train_cartpole_dqn_full_qbound.py`
- `experiments/cartpole/train_cartpole_dueling_full_qbound.py`
- `experiments/pendulum/train_pendulum_dqn_full_qbound.py`
- `experiments/pendulum/train_pendulum_ddpg_full_qbound.py`
- `experiments/pendulum/train_pendulum_td3_full_qbound.py`
- `experiments/ppo/train_pendulum_ppo_full_qbound.py`

### 2. Method Variants Removed

**DQN/DDQN/Dueling:**
- `dynamic_qbound_dqn`
- `dynamic_qbound_ddqn`

**DDPG/TD3/PPO:**
- `dynamic_qbound`

### 3. Code Removed

**Dynamic QBound computation logic:**
```python
# REMOVED: Step-aware dynamic bounds
if use_step_aware_qbound:
    remaining_steps = max_episode_steps - step
    Q_max_t = compute_dynamic_qbound(
        remaining_steps=remaining_steps,
        step_reward=step_reward,
        gamma=gamma,
        reward_is_negative=reward_is_negative
    )
```

**Experience replay modifications:**
- Removed `current_step` tracking in replay buffer
- Removed dynamic bound recomputation during sampling
- Simplified to static bounds only

## What Was Preserved (In Git History)

All dynamic QBound experiments are preserved in git history (commit `acad862` and earlier):

- Original 3-way experiment scripts
- Dynamic QBound implementation in agents
- Bug fixes for dynamic QBound in experience replay
- Full experimental results in `results/pendulum/backup_buggy_dynamic_20251114_061928/`

To recover dynamic QBound experiments:
```bash
git checkout acad862 -- experiments/
```

## Simplified Architecture

### New Experiment Structure (2-way comparison)

Each experiment now tests:
1. **Baseline** - No QBound
2. **Static QBound** - Fixed Q_min and Q_max for entire episode

### Benefits of Removal

1. **Simpler code** - Less complexity in agents and training loops
2. **Easier to maintain** - Fewer parameters and edge cases
3. **Better performance** - Static QBound consistently outperformed dynamic
4. **Clearer theory** - Static bounds are theoretically justified for both sparse and time-step dependent rewards

## Future Work

If dynamic QBound is ever reconsidered, key questions to investigate:

1. **Why did it fail?**
   - Over-restrictive early in episode?
   - Interfered with exploration?
   - Incorrect bound computation?

2. **Alternative approaches:**
   - Learned adaptive bounds (not hand-crafted)
   - Softer dynamic constraints
   - Episode-progress-aware weighting

## References

- Analysis script: `analysis/check_dynamic_qbound_value.py`
- Backup results: `results/pendulum/backup_buggy_dynamic_20251114_061928/`
- Historic experiments: Git commits `acad862` and earlier
