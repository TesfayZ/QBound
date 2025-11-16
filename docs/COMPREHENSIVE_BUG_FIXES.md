# Comprehensive Bug Fixes for Dynamic QBound on Negative Rewards

## Summary of Bugs Found

**All dynamic QBound implementations have sign errors for negative rewards:**

1. **DQN agents** (`dqn_agent.py`, `double_dqn_agent.py`, `dueling_dqn_agent.py`):
   - Apply negative sign twice → inverted bounds `[+1409, 0]`

2. **Continuous control agents** (`ddpg_agent.py`, `td3_agent.py`, `ppo_qbound_agent.py`):
   - Return wrong bound as Q_max → invalid bounds `[-1409, -1409]`

## Root Cause

For **negative step rewards** (e.g., Pendulum: r = -16.27 per step):

**Correct bounds**:
- Q_min(t) = r × (remaining steps) = becomes LESS negative as t increases
- Q_max = 0 (always, best case = perfect balance)

**Example at t=0 (start)**:
- Q_min(0) = -16.27 × 86.60 = -1409.33
- Q_max = 0.0
- Range: `[-1409, 0]` ✅

**Example at t=199 (last step)**:
- Q_min(199) = -16.27 × 1 = -16.27
- Q_max = 0.0
- Range: `[-16, 0]` ✅

---

## Fixes Required

### Fix 1: DQN/Double DQN/Dueling DQN

**Files**: `src/dqn_agent.py`, `src/double_dqn_agent.py`, `src/dueling_dqn_agent.py`

**Location**: Lines ~204-209 (dqn), ~213-218 (double_dqn), ~234-239 (dueling)

**Current Code** (BUGGY):
```python
if self.reward_is_negative:
    # Negative rewards: Q ∈ [Q_min(t), 0]
    dynamic_qmin_current = -geometric_sum_current * self.step_reward  # BUG: double negative!
    dynamic_qmax_current = torch.zeros_like(dynamic_qmin_current)
    dynamic_qmin_next = -geometric_sum_next * self.step_reward
    dynamic_qmax_next = torch.zeros_like(dynamic_qmin_next)
```

**Fixed Code**:
```python
if self.reward_is_negative:
    # Negative rewards: Q ∈ [Q_min(t), 0]
    # step_reward is already negative (e.g., -16.27)
    # Q_min(t) = step_reward * geometric_sum = (-16.27) * 86.60 = -1409.33 ✓
    dynamic_qmin_current = geometric_sum_current * self.step_reward
    dynamic_qmax_current = torch.zeros_like(dynamic_qmin_current)
    dynamic_qmin_next = geometric_sum_next * self.step_reward
    dynamic_qmax_next = torch.zeros_like(dynamic_qmin_next)
```

**Change**: Remove the leading `-` sign

---

### Fix 2: DDPG/TD3

**Files**: `src/ddpg_agent.py`, `src/td3_agent.py`

**Location**: Lines ~200-230 (`compute_qbound` method)

**Current Code** (BUGGY):
```python
def compute_qbound(self, current_step=None):
    if self.use_step_aware_qbound and current_step is not None:
        remaining_steps = self.max_episode_steps - current_step
        if remaining_steps > 0:
            if abs(self.gamma - 1.0) < 1e-6:
                Q_max_dynamic = self.step_reward * remaining_steps
            else:
                Q_max_dynamic = self.step_reward * (1 - self.gamma ** remaining_steps) / (1 - self.gamma)
            return self.qbound_min, Q_max_dynamic  # BUG: returns wrong bound for negative rewards!
        else:
            return self.qbound_min, 0.0
    else:
        return self.qbound_min, self.qbound_max
```

**Fixed Code**:
```python
def compute_qbound(self, current_step=None):
    if self.use_step_aware_qbound and current_step is not None:
        remaining_steps = self.max_episode_steps - current_step
        if remaining_steps > 0:
            # Compute dynamic bound based on remaining steps
            if abs(self.gamma - 1.0) < 1e-6:
                # Undiscounted case
                bound_dynamic = self.step_reward * remaining_steps
            else:
                # Discounted geometric series
                bound_dynamic = self.step_reward * (1 - self.gamma ** remaining_steps) / (1 - self.gamma)

            # For positive rewards: Q ∈ [0, Q_max(t)] where Q_max(t) decreases
            # For negative rewards: Q ∈ [Q_min(t), 0] where Q_min(t) increases (becomes less negative)
            if self.step_reward >= 0:
                # Positive rewards: Q_max decreases, Q_min stays at 0
                return 0.0, bound_dynamic
            else:
                # Negative rewards: Q_min decreases in magnitude, Q_max stays at 0
                return bound_dynamic, 0.0
        else:
            # No remaining steps
            return self.qbound_min, 0.0
    else:
        # Static bounds
        return self.qbound_min, self.qbound_max
```

**Change**: Check sign of `step_reward` and return appropriate bound pair

---

### Fix 3: PPO

**File**: `src/ppo_qbound_agent.py`

**Location**: Lines ~50-85 (`compute_vbound` method)

**Current Code** (BUGGY):
```python
def compute_vbound(self, current_step=None):
    if self.use_step_aware_qbound and current_step is not None:
        remaining_steps = self.max_episode_steps - current_step
        if remaining_steps > 0:
            V_max_dynamic = remaining_steps * self.step_reward  # BUG: for negative rewards!
            return self.qbound_min, V_max_dynamic
        else:
            return self.qbound_min, 0.0
    else:
        return self.qbound_min, self.qbound_max
```

**Fixed Code**:
```python
def compute_vbound(self, current_step=None):
    if self.use_step_aware_qbound and current_step is not None:
        remaining_steps = self.max_episode_steps - current_step
        if remaining_steps > 0:
            # Compute dynamic bound (typically undiscounted for PPO)
            bound_dynamic = remaining_steps * self.step_reward

            # For positive rewards: V ∈ [0, V_max(t)] where V_max(t) decreases
            # For negative rewards: V ∈ [V_min(t), 0] where V_min(t) increases (becomes less negative)
            if self.step_reward >= 0:
                # Positive rewards: V_max decreases, V_min stays at 0
                return 0.0, bound_dynamic
            else:
                # Negative rewards: V_min decreases in magnitude, V_max stays at 0
                return bound_dynamic, 0.0
        else:
            # No remaining steps
            return self.qbound_min, 0.0
    else:
        # Static bounds
        return self.qbound_min, self.qbound_max
```

**Change**: Check sign of `step_reward` and return appropriate bound pair

---

## Files to Modify

1. `src/dqn_agent.py` - line ~206, ~208
2. `src/double_dqn_agent.py` - line ~215, ~217
3. `src/dueling_dqn_agent.py` - line ~236, ~238
4. `src/ddpg_agent.py` - method `compute_qbound` (~line 200-229)
5. `src/td3_agent.py` - method `compute_qbound` (~line 238-267)
6. `src/ppo_qbound_agent.py` - method `compute_vbound` (~line 50-85)

---

## Experiments Affected (Need Re-run)

### Pendulum Experiments (All Dynamic QBound):
1. `train_pendulum_dqn_full_qbound.py` - DQN/DDQN dynamic (30 runs: 2 variants × 5 seeds × 3 methods but only dynamic affected)
2. `train_pendulum_ddpg_full_qbound.py` - DDPG dynamic (5 runs: 1 variant × 5 seeds × 1 dynamic method)
3. `train_pendulum_td3_full_qbound.py` - TD3 dynamic (5 runs: 1 variant × 5 seeds × 1 dynamic method)
4. `train_pendulum_ppo_full_qbound.py` - PPO dynamic (5 runs: 1 variant × 5 seeds × 1 dynamic method)

**Total**: 20 experiments to re-run (4 scripts × 5 seeds each)

### CartPole Experiments (Positive Rewards - NOT Affected):
- No bug for positive rewards
- Dynamic QBound works correctly on CartPole
- **No re-run needed**

---

## Expected Results After Fix

### Pendulum DQN/DDQN Dynamic:
- **Before fix**: -1322 to -1305 (catastrophic failure)
- **After fix**: -150 to -180 (similar to baseline or improved)

### Pendulum DDPG Dynamic:
- **Before fix**: -1203 (catastrophic failure)
- **After fix**: -170 to -200 (similar to or better than baseline -189)

### Pendulum TD3 Dynamic:
- **Before fix**: -1217 (catastrophic failure)
- **After fix**: -150 to -180 (potentially improved vs baseline -196)

### Pendulum PPO Dynamic:
- **Before fix**: -1223 (catastrophic failure)
- **After fix**: -250 to -300 (TBD, might work better)

---

## Verification Steps

After fixes:
1. ✅ Check bounds are correct: `Q ∈ [negative, 0]` for negative rewards
2. ✅ Verify violation rates drop from 90-100% to <10%
3. ✅ Confirm performance improves to baseline level or better
4. ✅ Test that CartPole results remain unchanged (positive rewards unaffected)
