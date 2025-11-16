# QBound Experiment Organization (Option 1)

## Overview

This document describes the organizational structure of QBound experiments based on **QBound Variant Coverage**. Experiments are categorized by whether the environment supports both static and dynamic QBound, or only static QBound.

## Organization Philosophy

### Rule 3: QBound Applicability

**Dynamic QBound is ONLY applicable to time-step dependent rewards.**

- **Time-step dependent**: Rewards accumulate predictably with elapsed time steps
  - Example: CartPole (+1 per step), Pendulum (constant negative reward per step)
  - **Supports**: Static QBound AND Dynamic QBound

- **Sparse/State-dependent**: Rewards depend on state transitions, not time
  - Example: GridWorld (sparse terminal), MountainCar (state-dependent termination)
  - **Supports**: Static QBound ONLY

---

## Category 1: Time-Step Dependent Rewards

### Environments: CartPole, Pendulum

These environments have rewards that accumulate predictably with time steps. Both static and dynamic QBound are applicable.

### Scripts (5 total)

| Script | Environment | Algorithm | Methods | QBound Types |
|--------|-------------|-----------|---------|--------------|
| `train_cartpole_dqn_full_qbound.py` | CartPole-v1 | DQN/DDQN | 6 | Static + Dynamic |
| `train_cartpole_dueling_full_qbound.py` | CartPole-v1 | Dueling DQN | 6 | Static + Dynamic |
| `train_pendulum_dqn_full_qbound.py` | Pendulum-v1 | DQN/DDQN | 6 | Static + Dynamic |
| `train_pendulum_ddpg_full_qbound.py` | Pendulum-v1 | DDPG | 2 | Soft only* |
| `train_pendulum_ppo_full_qbound.py` | Pendulum-v1 | PPO | 2 | Soft only* |

*Actor-critic methods (DDPG, PPO) use soft QBound only (differentiable for gradient backprop)

### Methods Tested

**Value-based (DQN/DDQN/Dueling)**: 6 methods
1. Baseline DQN
2. Static QBound + DQN
3. Dynamic QBound + DQN
4. Baseline DDQN (or Double Dueling DQN)
5. Static QBound + DDQN
6. Dynamic QBound + DDQN

**Actor-critic (DDPG/PPO)**: 2 methods
1. Baseline (no QBound)
2. Soft QBound

### CartPole-v1

**Reward structure**: Dense positive (+1 per step)
- **Reward type**: Time-step dependent
- **Max steps**: 500
- **Q_max formula**: `Q_max = (1 - γ^H) / (1 - γ) ≈ 99.34` for γ=0.99, H=500
- **QBound range**: [0.0, 99.34]

**Why dynamic QBound works**: Each additional time step adds exactly +1 reward, making Q_max(t) = (1 - γ^(H-t)) / (1 - γ) a precise bound.

**Scripts**:
- `experiments/cartpole/train_cartpole_dqn_full_qbound.py` (DQN/DDQN)
- `experiments/cartpole/train_cartpole_dueling_full_qbound.py` (Dueling DQN)

### Pendulum-v1

**Reward structure**: Dense negative (time-step dependent cost)
- **Reward type**: Time-step dependent negative
- **Max steps**: 200
- **Reward per step**: Negative cost (angle + velocity penalty)
- **QBound range**: [-1409.33, 0.0]

**Why dynamic QBound works**: Similar cost per step, predictable accumulation.

**Scripts**:
- `experiments/pendulum/train_pendulum_dqn_full_qbound.py` (DQN/DDQN, discretized)
- `experiments/pendulum/train_pendulum_ddpg_full_qbound.py` (DDPG, continuous, soft QBound)
- `experiments/ppo/train_pendulum_ppo_full_qbound.py` (PPO, continuous, soft QBound)

---

## Category 2: Sparse/State-Dependent Rewards

### Environments: GridWorld, FrozenLake, MountainCar, Acrobot

These environments have rewards that depend on state transitions or are sparse terminal. **Dynamic QBound is NOT applicable** (violates Rule 3).

### Scripts (4 total)

| Script | Environment | Algorithm | Methods | QBound Types |
|--------|-------------|-----------|---------|--------------|
| `train_gridworld_dqn_static_qbound.py` | GridWorld | DQN/DDQN | 4 | Static only |
| `train_frozenlake_dqn_static_qbound.py` | FrozenLake-v1 | DQN/DDQN | 4 | Static only |
| `train_mountaincar_dqn_static_qbound.py` | MountainCar-v0 | DQN/DDQN | 4 | Static only |
| `train_acrobot_dqn_static_qbound.py` | Acrobot-v1 | DQN/DDQN | 4 | Static only |

### Methods Tested

**All environments**: 4 methods only
1. Baseline DQN
2. Static QBound + DQN
3. Baseline DDQN
4. Static QBound + DDQN

### GridWorld

**Reward structure**: Sparse terminal (+1 at goal only)
- **Reward type**: Sparse terminal
- **QBound range**: [0.0, 1.0]
- **Why NO dynamic QBound**: Only terminal reward, no time-step accumulation

**Script**: `experiments/gridworld/train_gridworld_dqn_static_qbound.py`

### FrozenLake-v1

**Reward structure**: Sparse terminal (+1 at goal only)
- **Reward type**: Sparse terminal
- **Stochasticity**: Slippery surface (actions have noise)
- **QBound range**: [0.0, 1.0]
- **Why NO dynamic QBound**: Only terminal reward, no time-step accumulation

**Script**: `experiments/frozenlake/train_frozenlake_dqn_static_qbound.py`

### MountainCar-v0

**Reward structure**: State-dependent negative (-1 per step until goal)
- **Reward type**: State-dependent (episode length depends on when goal reached)
- **QBound range**: [-86.60, 0.0]
- **Why NO dynamic QBound**: Total reward depends on WHEN goal is reached (state event)

**Script**: `experiments/mountaincar/train_mountaincar_dqn_static_qbound.py`

### Acrobot-v1

**Reward structure**: State-dependent negative (-1 per step until swing-up)
- **Reward type**: State-dependent (episode length depends on when goal reached)
- **QBound range**: [-99.34, 0.0]
- **Why NO dynamic QBound**: Total reward depends on WHEN swing-up succeeds (state event)

**Script**: `experiments/acrobot/train_acrobot_dqn_static_qbound.py`

---

## Directory Structure

```
experiments/
├── cartpole/
│   ├── train_cartpole_dqn_full_qbound.py          # Time-step dependent
│   └── train_cartpole_dueling_full_qbound.py      # Time-step dependent
│
├── pendulum/
│   ├── train_pendulum_dqn_full_qbound.py          # Time-step dependent
│   ├── train_pendulum_ddpg_full_qbound.py         # Time-step dependent (soft)
│   └── ...
│
├── ppo/
│   └── train_pendulum_ppo_full_qbound.py          # Time-step dependent (soft)
│
├── gridworld/
│   └── train_gridworld_dqn_static_qbound.py       # Sparse/state-dependent
│
├── frozenlake/
│   └── train_frozenlake_dqn_static_qbound.py      # Sparse/state-dependent
│
├── mountaincar/
│   └── train_mountaincar_dqn_static_qbound.py     # Sparse/state-dependent
│
├── acrobot/
│   └── train_acrobot_dqn_static_qbound.py         # Sparse/state-dependent
│
└── run_all_organized_experiments.py               # Master orchestrator
```

---

## Running Experiments

### Run All Experiments

```bash
# Run everything with default seed (42)
python3 experiments/run_all_organized_experiments.py

# Run with custom seed
python3 experiments/run_all_organized_experiments.py --seed 123
```

### Run by Category

```bash
# Only time-step dependent experiments (5 scripts)
python3 experiments/run_all_organized_experiments.py --category timestep

# Only sparse/state-dependent experiments (4 scripts)
python3 experiments/run_all_organized_experiments.py --category sparse
```

### Run Individual Experiments

```bash
# Time-step dependent examples
python3 experiments/cartpole/train_cartpole_dqn_full_qbound.py --seed 42
python3 experiments/pendulum/train_pendulum_ddpg_full_qbound.py --seed 43

# Sparse/state-dependent examples
python3 experiments/gridworld/train_gridworld_dqn_static_qbound.py --seed 42
python3 experiments/mountaincar/train_mountaincar_dqn_static_qbound.py --seed 44
```

### Dry Run

```bash
# Preview what would be executed
python3 experiments/run_all_organized_experiments.py --dry-run
```

---

## Results Structure

### File Naming Convention

**Time-step dependent**:
```
results/cartpole/dqn_full_qbound_seed42_20251029_143000.json
results/cartpole/dueling_full_qbound_seed42_20251029_150000.json
results/pendulum/dqn_full_qbound_seed42_20251029_160000.json
results/pendulum/ddpg_full_qbound_seed42_20251029_180000.json
results/ppo/pendulum_full_qbound_seed42_20251029_190000.json
```

**Sparse/state-dependent**:
```
results/gridworld/dqn_static_qbound_seed42_20251029_220000.json
results/frozenlake/dqn_static_qbound_seed42_20251029_230000.json
results/mountaincar/dqn_static_qbound_seed42_20251030_000000.json
results/acrobot/dqn_static_qbound_seed42_20251030_010000.json
```

### Metadata Fields

All results files include:

```json
{
  "experiment_type": "time_step_dependent" | "sparse_state_dependent",
  "script_name": "train_<env>_<algo>_<qbound_type>.py",
  "reward_structure": "time_step_dependent" | "sparse_terminal" | "state_dependent",
  "timestamp": "20251029_143000",
  "config": { ... },
  "training": { ... }
}
```

---

## Estimated Execution Times

### Time-Step Dependent (Total: ~5.75 hours)

| Experiment | Methods | Time (min) |
|------------|---------|-----------|
| CartPole DQN Full QBound | 6 | 30 |
| CartPole Dueling Full QBound | 6 | 30 |
| Pendulum DQN Full QBound | 6 | 120 |
| Pendulum DDPG Full QBound | 2 | 60 |
| Pendulum PPO Full QBound | 2 | 45 |
| **Total** | **22** | **285 min** |

### Sparse/State-Dependent (Total: ~2.33 hours)

| Experiment | Methods | Time (min) |
|------------|---------|-----------|
| GridWorld DQN Static | 4 | 15 |
| FrozenLake DQN Static | 4 | 20 |
| MountainCar DQN Static | 4 | 60 |
| Acrobot DQN Static | 4 | 45 |
| **Total** | **16** | **140 min** |

### Grand Total

- **9 experiments**
- **38 total methods**
- **~7.1 hours** sequential execution
- **~2.4 hours** with 3-way parallelization (3 seeds simultaneously)

---

## Multi-Seed Runs

All scripts support the `--seed` argument for reproducibility:

```bash
# Run same experiment with 3 different seeds
for seed in 42 43 44; do
    python3 experiments/cartpole/train_cartpole_dqn_full_qbound.py --seed $seed
done
```

Results will be saved separately:
```
results/cartpole/dqn_full_qbound_seed42_20251029_143000.json
results/cartpole/dqn_full_qbound_seed43_20251029_150000.json
results/cartpole/dqn_full_qbound_seed44_20251029_153000.json
```

---

## QBound Type Reference

### Hard QBound (Clipping)

**Used for**: Value-based methods (DQN, DDQN, Dueling DQN)

**Why**: These methods don't need differentiable Q-values for actor updates. Hard clipping is simpler and works well.

**Implementation**:
```python
q_values = torch.clamp(q_values, min=qbound_min, max=qbound_max)
```

### Soft QBound (Penalty)

**Used for**: Actor-critic methods (DDPG, PPO)

**Why**: These methods backpropagate gradients through Q/V values to update the actor. Hard clipping causes zero gradients at boundaries, breaking learning. Soft penalties maintain differentiability.

**Implementation**:
```python
# Penalty loss added to primary loss
penalty = torch.mean((q_values - qbound_max).clamp(min=0)**2) + \
          torch.mean((qbound_min - q_values).clamp(min=0)**2)
total_loss = primary_loss + aux_weight * penalty
```

---

## Key Takeaways

1. **Organization by QBound variant coverage** clearly separates:
   - Environments that support dynamic QBound (time-step dependent)
   - Environments that only support static QBound (sparse/state-dependent)

2. **Rule 3 compliance**: Dynamic QBound ONLY used where theoretically justified

3. **Consistent naming**: `*_full_qbound.py` vs `*_static_qbound.py`

4. **Complete coverage**:
   - 5 time-step dependent scripts (22 methods total)
   - 6 sparse/state-dependent scripts (24 methods total)
   - 46 total algorithm × QBound combinations

5. **Flexible execution**: Run all, by category, or individual experiments

6. **Multi-seed ready**: All scripts accept `--seed` for reproducibility

---

## Next Steps

1. **Run experiments**:
   ```bash
   python3 experiments/run_all_organized_experiments.py
   ```

2. **Multi-seed runs** (for statistical significance):
   ```bash
   for seed in 42 43 44; do
       python3 experiments/run_all_organized_experiments.py --seed $seed
   done
   ```

3. **Analyze results**: Create aggregation scripts to compute mean ± std across seeds

4. **Update paper**: Use organized results to structure paper sections clearly

---

## References

- **Rule 3**: Dynamic QBound only for time-step dependent rewards
- **CLAUDE.md**: Project conventions and structure
- **MULTI_SEED_SETUP_COMPLETE.md**: Multi-seed implementation details
