# Reproducibility Guide

## Random Seed Configuration

All experiments in this project use **deterministic seeding** to ensure reproducibility. The global seed is `SEED = 42` and is applied to all random number generators.

## Seeding Strategy

### 1. Global Seeds (Applied to All Experiments)

Every experiment script sets the following seeds at the beginning:

```python
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)
```

This ensures:
- NumPy operations are deterministic
- PyTorch operations (network initialization, sampling) are deterministic
- Python's built-in random module is deterministic

### 2. Environment Seeds

**CartPole and FrozenLake (Gymnasium Environments):**

The environment wrapper uses an incremental seeding strategy:

```python
class CartPoleWrapper:
    def __init__(self, seed=None):
        self.seed = seed

    def reset(self):
        if self.seed is not None:
            state, _ = self.env.reset(seed=self.seed)
            self.seed += 1  # Increment for next reset
        return state
```

Why increment?
- Each episode needs different initial conditions
- But the sequence must be reproducible
- Starting from SEED=42, episodes get seeds: 42, 43, 44, ...
- This ensures different episodes while maintaining reproducibility

**GridWorld (Custom Environment):**

GridWorld is deterministic and doesn't require seeding.

### 3. Agent Initialization

DQN agents use PyTorch, which is seeded globally with `torch.manual_seed(SEED)`. This ensures:
- Network weight initialization is identical across runs
- Replay buffer sampling is deterministic (uses Python's `random.sample()`, seeded globally)

## Reproducing Experiments

### Single Environment

To reproduce any single experiment:

```bash
# GridWorld
python3 experiments/gridworld/train_gridworld.py

# FrozenLake
python3 experiments/frozenlake/train_frozenlake.py

# CartPole (original 2-way comparison)
python3 experiments/cartpole/train_cartpole.py

# CartPole (3-way comparison: baseline vs static vs dynamic)
python3 experiments/cartpole/train_cartpole_3way.py
```

All scripts use `SEED = 42` by default.

### All Experiments

To reproduce all experiments at once:

```bash
python3 experiments/combined/run_all_experiments.py
```

This runs GridWorld, FrozenLake, and CartPole experiments sequentially with the same seed.

## Expected Results (with SEED=42)

### GridWorld
- **QBound**: Reaches 80% success at episode 205
- **Baseline**: Reaches 80% success at episode 257
- **Improvement**: 20.2% faster convergence

### FrozenLake
- **QBound**: Reaches 70% success at episode 209
- **Baseline**: Reaches 70% success at episode 220
- **Improvement**: 5.0% faster convergence

### CartPole (Dynamic QBound)
- **QBound**: Total reward = 172,904
- **Baseline**: Total reward = 131,438
- **Improvement**: 31.5% higher cumulative reward

### CartPole 3-Way Comparison (New)
- **Baseline** (no QBound)
- **Static QBound** (Q_max = 100)
- **Dynamic QBound** (Q_max(t) = 500 - t)

Results pending (experiment currently running).

## Important Notes

### Why Multiple Seeds Matter

For statistical significance, you should run experiments with multiple seeds:

```python
SEEDS = [42, 123, 456, 789, 1024]

for seed in SEEDS:
    # Run experiment with this seed
    # Collect results
# Average results across seeds
```

However, for **demonstrating the method**, we use a single seed (42) to:
- Keep training time reasonable
- Show clear differences between methods
- Allow exact reproduction of paper results

### Verifying Reproducibility

To verify seeding works correctly:

1. Run experiment twice with same seed → should get identical results
2. Run experiment with different seed → should get different episode rewards but similar final performance

Example verification:

```bash
# Run 1
python3 experiments/cartpole/train_cartpole_3way.py > run1.log

# Run 2
python3 experiments/cartpole/train_cartpole_3way.py > run2.log

# Compare
diff run1.log run2.log  # Should be identical
```

## Seed Independence Test

To test that results are robust across seeds:

```python
# Modify experiment script temporarily
for seed in [42, 123, 456]:
    SEED = seed
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    random.seed(SEED)

    # Run experiment
    # Save results with seed in filename
```

The method should show improvement across different seeds, not just seed=42.

## Hardware Determinism

**Note**: PyTorch operations on GPU may not be fully deterministic even with seeding. For exact reproducibility:

1. Use CPU (as we do: `device="cpu"`)
2. Or enable PyTorch deterministic mode:
   ```python
   torch.backends.cudnn.deterministic = True
   torch.backends.cudnn.benchmark = False
   ```

All our experiments use CPU to ensure complete reproducibility.

## Summary

**For other researchers to reproduce our results:**

1. Use the same random seed: `SEED = 42`
2. Use the same hyperparameters (documented in each experiment file)
3. Use CPU device (not GPU)
4. Run the exact experiment scripts provided

**Expected outcome**: Identical episode-by-episode results to those reported in the paper.
