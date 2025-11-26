# Quick Start: Transformed Q-Value Experiments

## What This Tests

**Hypothesis:** QBound fails on negative rewards because of the negative value range.

**Test:** Transform negative Q-values to positive range (like CartPole where QBound works).

**Environments:**
- MountainCar (DQN): Q ∈ [-86.60, 0] → Q ∈ [0, 86.60]
- Acrobot (DQN): Q ∈ [-99.34, 0] → Q ∈ [0, 99.34]
- Pendulum (DDPG): Q ∈ [-1409.33, 0] → Q ∈ [0, 1409.33]

## Run Commands

### Option 1: Run Only Transformed Experiments (Recommended)
```bash
# Run with 5 seeds for statistical significance
python3 experiments/run_all_organized_experiments.py --category transformed --seeds 42 43 44 45 46

# 3 experiments (MountainCar, Acrobot, Pendulum)
# Estimated time: ~16.25 hours (975 minutes total)
```

### Option 2: Run All Experiments (Includes Transformed)
```bash
# The master script now includes transformed experiments
python3 experiments/run_all_organized_experiments.py

# Crash recovery will skip already-completed experiments
# Only transformed experiments will run (if others are done)
```

### Option 3: Quick Test with Single Seed
```bash
# Fast test to verify everything works
python3 experiments/run_all_organized_experiments.py --category transformed --seed 42

# Estimated time: ~195 minutes (~3.25 hours)
```

## Results Location

Results will be saved to:
- `results/mountaincar/dqn_transformed_seed{SEED}_{timestamp}.json`
- `results/acrobot/dqn_transformed_seed{SEED}_{timestamp}.json`
- `results/pendulum/ddpg_transformed_seed{SEED}_{timestamp}.json`

## What Gets Compared

After experiments complete, you can compare:

1. **Original negative Q-values:**
   - `results/mountaincar/dqn_full_qbound_seed42_*.json`
   - `results/acrobot/dqn_full_qbound_seed42_*.json`
   - `results/pendulum/ddpg_full_qbound_seed42_*.json`

2. **Transformed positive Q-values (NEW):**
   - `results/mountaincar/dqn_transformed_seed42_*.json`
   - `results/acrobot/dqn_transformed_seed42_*.json`
   - `results/pendulum/ddpg_transformed_seed42_*.json`

3. **CartPole positive Q-values (reference):**
   - `results/cartpole/dqn_full_qbound_seed42_*.json`

## Expected Results

### If Transformation Fixes/Improves QBound:
- MountainCar/Acrobot: QBound shows improvement on transformed experiments
- Pendulum: QBound shows even better improvement than original
- Conclusion: Negative range was (at least part of) the problem
- Implication: QBound works better with positive rewards

### If QBound Still Has Mixed Results:
- QBound still shows mixed performance on transformed experiments
- Conclusion: Problem is NOT just the negative range
- Need to investigate violation rates and clipping bias
- Other factors (reward magnitude, Q-value distribution) may be more important

## Crash Recovery

If experiments are interrupted:
```bash
# Just re-run the same command
python3 experiments/run_all_organized_experiments.py --category transformed --seeds 42 43 44 45 46

# The system will:
# - Load progress from results/organized_experiments_log.json
# - Skip already-completed experiments
# - Continue from where it left off
```

## Monitoring Progress

Check progress log:
```bash
cat results/organized_experiments_log.json | python3 -m json.tool | grep -A 5 "transformed"
```

Check existing results:
```bash
ls -lh results/mountaincar/dqn_transformed_*
ls -lh results/acrobot/dqn_transformed_*
```

## Troubleshooting

### If script doesn't find experiments:
```bash
# Verify the category exists
python3 experiments/run_all_organized_experiments.py --dry-run --category transformed
```

### If import errors:
```bash
# Check Python can find the modules
python3 -c "import sys; sys.path.insert(0, 'src'); from dqn_agent_transformed import TransformedDQNAgent; print('✓ Import successful')"
```

### If result files aren't created:
```bash
# Check directories exist
mkdir -p results/mountaincar results/acrobot
```

## Documentation

For more details, see:
- `docs/TRANSFORMED_QVALUE_EXPERIMENTS.md` - Full documentation
- `TRANSFORMED_EXPERIMENTS_SUMMARY.md` - Implementation summary

## Quick Verification

Test everything is working:
```bash
# Dry run to preview
python3 experiments/run_all_organized_experiments.py --category transformed --dry-run --seed 42

# Should show 3 experiments (MountainCar, Acrobot, Pendulum)
```

## Important Notes

- **PPO is NOT included** - Only DQN and DDPG transformed experiments
- **Pendulum context**: Original DDPG showed QBound working (61% improvement)
  - Transformation tests if positive range improves it further
- **MountainCar/Acrobot context**: Original results were mixed
  - Transformation tests if positive range fixes QBound
