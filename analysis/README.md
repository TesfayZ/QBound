# QBound Analysis Scripts

This directory contains comprehensive analysis scripts for all QBound experiments.

## Available Scripts

### Environment-Specific Analysis

#### 1. **LunarLander Analysis**
```bash
python3 analysis/analyze_lunarlander.py
```

**Output:**
- Learning curves for all 4 agents (DQN, QBound DQN, Double DQN, QBound Double DQN)
- Comparison plot with all agents overlaid
- Bar chart of final performance
- Success rate statistics
- PDF versions for paper

**Generated files:**
- `results/plots/lunarlander_learning_curves_*.{png,pdf}`
- `results/plots/lunarlander_comparison_*.{png,pdf}`
- `results/plots/lunarlander_bar_comparison_*.{png,pdf}`

---

#### 2. **CartPole Corrected Analysis**
```bash
python3 analysis/analyze_cartpole_corrected.py
```

**Output:**
- Learning curves for all 6 agents (including dynamic variants)
- Comparison plot with all agents overlaid
- Bar chart of final performance
- DQN vs Double DQN comparison
- PDF versions for paper

**Generated files:**
- `results/plots/cartpole_corrected_learning_curves_*.{png,pdf}`
- `results/plots/cartpole_corrected_comparison_*.{png,pdf}`
- `results/plots/cartpole_corrected_bar_comparison_*.{png,pdf}`

---

### Unified Analysis

#### 3. **Cross-Environment Comparison**
```bash
python3 analysis/unified_analysis.py
```

**Output:**
- Performance comparison across ALL environments
- QBound improvement analysis
- Double DQN vs DQN comparison
- Summary statistics
- Heatmap of improvements
- Grouped bar charts

**Generated files:**
- `results/plots/unified_qbound_improvement.{png,pdf}`
- `results/plots/unified_grouped_comparison.{png,pdf}`

**Includes:**
- GridWorld
- FrozenLake
- CartPole
- MountainCar
- Acrobot
- LunarLander
- CartPole-Corrected

---

### Legacy Analysis Scripts

- `qbound_summary_table.py` - Quick summary table
- `analyze_results.py` - Detailed analysis (older format)
- `comprehensive_analysis.py` - Full analysis with plots (older format)
- `plot_paper_results.py` - Generate paper-ready plots

---

## Quick Start

### Run All Analyses

```bash
# Individual analyses
python3 analysis/analyze_lunarlander.py
python3 analysis/analyze_cartpole_corrected.py

# Unified cross-environment analysis
python3 analysis/unified_analysis.py
```

### View Generated Plots

```bash
# List all plots
ls -lh results/plots/

# View specific plots
open results/plots/lunarlander_comparison_*.png
open results/plots/cartpole_corrected_comparison_*.png
open results/plots/unified_qbound_improvement.png
```

---

## Understanding the Output

### Key Metrics Reported

1. **Mean ± Std** - Average reward over last 100 episodes with standard deviation
2. **Max Reward** - Highest reward achieved during training
3. **Success Rate** - Percentage of episodes exceeding success threshold
4. **Final 100 Success Rate** - Success rate in last 100 episodes only

### Success Thresholds

- **LunarLander**: 200 (successful landing)
- **CartPole**: 475 (95% of max 500)
- **MountainCar**: -110 (reaching goal)
- **Acrobot**: -100 (swinging up)

### Improvement Calculation

```
Improvement = QBound_Mean - Baseline_Mean
% Change = (Improvement / |Baseline_Mean|) × 100
```

---

## Plot Types

### 1. Learning Curves
- Raw rewards (transparent lines)
- 20-episode moving average (solid lines)
- Success thresholds (dashed lines)
- Individual subplots per agent

### 2. Comparison Plots
- All agents on single plot
- 20-episode moving average
- Easy to compare learning dynamics
- Success threshold overlay

### 3. Bar Charts
- Final 100 episode performance
- Error bars (standard deviation)
- Value labels on bars
- Success threshold line

### 4. Heatmaps (Unified Analysis)
- Environment-wise improvements
- Color-coded (green=good, red=bad)
- Percentage improvements

---

## File Structure

```
analysis/
├── README.md                          # This file
├── analyze_lunarlander.py             # LunarLander 4-way analysis
├── analyze_cartpole_corrected.py      # CartPole 6-way analysis
├── unified_analysis.py                # Cross-environment comparison
├── qbound_summary_table.py            # Quick summary
├── analyze_results.py                 # Legacy detailed analysis
├── comprehensive_analysis.py          # Legacy full analysis
└── plot_paper_results.py              # Paper plot generation

results/
├── plots/                             # All generated plots
│   ├── lunarlander_*.{png,pdf}       # LunarLander plots
│   ├── cartpole_corrected_*.{png,pdf}# CartPole plots
│   └── unified_*.{png,pdf}           # Cross-env plots
└── <environment>/                     # Raw experimental data
    └── *.json                        # JSON result files
```

---

## Tips

### Re-generating Plots

If you update plot styling or want higher resolution:

```bash
# Edit the script to change DPI or style
nano analysis/analyze_lunarlander.py

# Re-run to regenerate
python3 analysis/analyze_lunarlander.py
```

### Custom Analysis

To create custom analysis:

```python
import json
import numpy as np

# Load results
with open('results/lunarlander/4way_comparison_*.json', 'r') as f:
    data = json.load(f)

# Access training data
dqn_rewards = data['training']['dqn']['rewards']

# Analyze
print(f"Mean: {np.mean(dqn_rewards[-100:])}")
```

### Statistical Tests

For rigorous comparison, consider:
- Mann-Whitney U test (non-parametric)
- Bootstrap confidence intervals
- Multiple comparison corrections

---

## Documentation

See also:
- `docs/EXTENDED_ANALYSIS_SUMMARY.md` - Comprehensive results summary
- `docs/EXTENDED_EXPERIMENTS.md` - Experiment details
- `CLAUDE.md` - Project overview and structure

---

## Troubleshooting

### "No results found"
- Check that experiments have completed
- Verify file paths in scripts
- Ensure JSON files exist in results directories

### "Module not found"
- Install dependencies: `pip install numpy matplotlib`
- Check Python version (3.7+)

### Plots look bad
- Increase DPI in scripts (default: 300)
- Try different figure sizes
- Adjust font sizes for readability

---

## Future Enhancements

Planned additions:
- Statistical significance testing
- Interactive plots (Plotly)
- Hyperparameter sensitivity analysis
- Confidence interval visualization
- Q-value distribution analysis

---

**Last Updated:** 2025-10-27
