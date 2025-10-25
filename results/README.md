# QBound Experiment Results

This directory contains organized experimental results for the QBound paper.

## Directory Structure

```
results/
├── gridworld/          # GridWorld 10x10 experiments
├── frozenlake/         # FrozenLake 4x4 experiments
├── cartpole/           # CartPole-v1 experiments
└── combined/           # Combined results from all environments
```

## File Naming Convention

All result files follow the pattern: `results_YYYYMMDD_HHMMSS.json`

Example: `results_20250124_143052.json` (generated on Jan 24, 2025 at 14:30:52)

## Result File Format

Each JSON file contains:

```json
{
  "env_name": "GridWorld",
  "qbound_episodes": 206,
  "baseline_episodes": 352,
  "improvement_percent": 41.5,
  "qbound_total_reward": 350.5,
  "baseline_total_reward": 245.2,
  "reward_improvement_percent": 42.9,
  "rewards_qbound": [...],  // Full trajectory
  "rewards_baseline": [...],  // Full trajectory
  "config": {...}  // Hyperparameters used
}
```

## Running Experiments

To generate results:

```bash
# Run all experiments (GridWorld, FrozenLake, CartPole)
python run_all_experiments.py

# Results will be saved to:
# - results/combined/experiment_results_<timestamp>.json
# - results/gridworld/results_<timestamp>.json
# - results/frozenlake/results_<timestamp>.json
# - results/cartpole/results_<timestamp>.json
```

## Reproducibility

All experiments use:
- **Random Seed:** 42 (for NumPy, PyTorch, Python random, and Gymnasium environments)
- **Fixed Hyperparameters:** Defined in `run_all_experiments.py`
- **Consistent Evaluation:** Same target metrics across runs

## Updating the Paper

After running experiments, update the paper with real results:

```bash
python update_paper_with_results.py
```

This script:
1. Finds the latest experiment results in `results/combined/`
2. Extracts key metrics (episodes to target, improvement %)
3. Updates the results table in `QBound/main.tex`
4. Creates a backup of the original file

## Key Metrics

For each environment, we track:

1. **Episodes to Target Performance**
   - Primary metric for sample efficiency
   - Target varies by environment (80% for GridWorld, 70% for FrozenLake, 475 avg for CartPole)

2. **Improvement Percentage**
   - Calculated as: (Baseline - QBound) / Baseline × 100%
   - Shows relative speedup

3. **Total Cumulative Reward**
   - Sum of all episode rewards during training
   - Alternative measure of sample efficiency

4. **Full Reward Trajectories**
   - Episode-by-episode rewards for plotting learning curves

## Experimental Configuration

| Environment | Episodes | Max Steps | Target | Q_min | Q_max | γ    |
|-------------|----------|-----------|--------|-------|-------|------|
| GridWorld   | 500      | 100       | 0.8    | 0.0   | 1.0   | 0.99 |
| FrozenLake  | 2000     | 100       | 0.7    | 0.0   | 1.0   | 0.95 |
| CartPole    | 500      | 500       | 475    | 0.0   | 100.0 | 0.99 |

All experiments use:
- Batch size: 64
- Learning rate: 0.001
- Replay buffer: 10,000 transitions
- Network: [128, 128] ReLU
- Auxiliary weight (λ): 0.5

## Notes

- FrozenLake has stochastic transitions (33% success rate for intended action)
- CartPole terminates on failure with max 500 steps
- GridWorld is deterministic navigation with sparse binary reward
