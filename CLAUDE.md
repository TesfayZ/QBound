# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**QBound** - Q-Value Bounding for Deep Reinforcement Learning

A research project implementing QBound, a technique for bounding Q-values in Deep Q-Networks (DQN) to improve learning in sparse reward environments. Tests on GridWorld, FrozenLake, and CartPole environments.

## ⚠️ CRITICAL: Directory Structure MUST Be Maintained

**This directory structure is MANDATORY for all future work:**

```
QBound/
├── src/                          # CORE COMPONENTS (shared by all experiments)
│   ├── dqn_agent.py             # DQN agent with QBound implementation
│   ├── environment.py           # GridWorld environment
│   └── __init__.py              # Package initialization
│
├── experiments/                  # EXPERIMENT SCRIPTS (one folder per environment)
│   ├── gridworld/               # GridWorld experiments
│   ├── frozenlake/              # FrozenLake experiments
│   ├── cartpole/                # CartPole experiments
│   └── combined/                # Run all experiments
│
├── analysis/                     # ANALYSIS AND VISUALIZATION
│   └── *.py                     # Analysis scripts
│
├── docs/                         # DOCUMENTATION
│   └── *.md                     # Documentation files
│
└── results/                      # EXPERIMENT RESULTS
    ├── gridworld/               # GridWorld results
    ├── frozenlake/              # FrozenLake results
    ├── cartpole/                # CartPole results
    ├── combined/                # Combined results
    └── plots/                   # Generated plots
```

### Rules for Directory Structure:

1. **NEVER** put experiment scripts in the root directory
2. **ALWAYS** put core shared code in `src/`
3. **ALWAYS** create environment-specific folders under `experiments/`
4. **ALWAYS** put analysis tools in `analysis/`
5. **ALWAYS** put documentation in `docs/`
6. **ALWAYS** save results in `results/<environment>/`

## Reproducibility

**CRITICAL**: All experiments use deterministic seeding for reproducibility.

- **Global seed**: `SEED = 42` (applied to NumPy, PyTorch, Python random)
- **Environment seeding**: Incremental seeds (42, 43, 44, ...) for each episode
- **Device**: CPU only (ensures full determinism)

See `docs/REPRODUCIBILITY.md` for complete seeding strategy and reproduction instructions.

Running the same experiment twice with the same seed MUST produce identical results.

## Development Commands

### Run Experiments

```bash
# Individual environments
python3 experiments/gridworld/train_gridworld.py
python3 experiments/frozenlake/train_frozenlake.py
python3 experiments/cartpole/train_cartpole.py

# CartPole 3-way comparison (baseline vs static vs dynamic QBound)
python3 experiments/cartpole/train_cartpole_3way.py

# All experiments
python3 experiments/combined/run_all_experiments.py
```

### Analyze Results

```bash
# Quick summary
python3 analysis/qbound_summary_table.py

# Detailed analysis
python3 analysis/analyze_results.py

# Full analysis with plots
python3 analysis/comprehensive_analysis.py
```

### View Project Structure

```bash
./show_structure.sh
```

## Code Architecture

### Core Components (src/)

**dqn_agent.py** - Main DQN implementation with QBound
- `DQNAgent` class with dual-loss training
- Primary loss: Standard TD loss
- Auxiliary loss: Penalizes Q-values violating [Q_min, Q_max]
- Combined loss: `total_loss = primary_loss + aux_weight * aux_loss`

**environment.py** - GridWorld environment
- 10x10 grid world
- Sparse reward: +1 for goal, 0 otherwise
- Custom implementation for GridWorld experiments

### Import Pattern

**ALL experiment and analysis scripts MUST use this import pattern:**

```python
import sys
sys.path.insert(0, '/root/projects/QBound/src')

from environment import GridWorldEnv
from dqn_agent import DQNAgent
```

This ensures all scripts use the same core components.

## Model Saving and Blind Evaluation

**IMPORTANT**: The 3-way CartPole experiment saves trained models and performs blind evaluation.

### Saved Models

Trained models are saved to `models/cartpole/`:
- `baseline_<timestamp>.pt` - Baseline DQN (no QBound)
- `static_qbound_<timestamp>.pt` - Static QBound (Q_max=100)
- `dynamic_qbound_<timestamp>.pt` - Dynamic QBound (Q_max(t)=500-t)

### Blind Evaluation

Models are evaluated **without step-aware information**:
- Tests if dynamic QBound generalizes beyond training assumptions
- Evaluates at max_steps=500 (training length) and max_steps=1000 (2x training)
- Critical test: Can dynamic QBound perform without knowing current step?

See `docs/BLIND_EVALUATION.md` for complete methodology and expected outcomes.

## Key Parameters

### QBound Configuration

- `use_qclip`: Enable/disable QBound (bool)
- `qclip_min`: Lower bound for Q-values (float)
- `qclip_max`: Upper bound for Q-values (float)
- `use_step_aware_qbound`: Enable dynamic step-aware bounds (bool)
- `max_episode_steps`: Maximum episode length for computing Q_max(t) (int)
- `step_reward`: Reward per step for computing Q_max(t) (float)
- `gamma`: Discount factor (float)

### Per-Environment Settings

**⚠️ CRITICAL**: All Q_max values must account for the discount factor γ!

**Correct Formula**: Q_max = (1 - γ^H) / (1 - γ) where H = max episode steps

**GridWorld:**
- Q_min=0.0, Q_max=1.0, γ=0.99
- Episodes: 500, Max steps: 100
- Theoretical Q_max = 63.4 (but using 1.0 for sparse terminal reward)

**FrozenLake:**
- Q_min=0.0, Q_max=1.0, γ=0.95
- Episodes: 2000, Max steps: 100
- Theoretical Q_max = 19.6 (but using 1.0 for sparse terminal reward)

**CartPole:**
- Q_min=0.0, Q_max=99.34, γ=0.99
- Episodes: 500, Max steps: 500
- Theoretical Q_max = (1 - 0.99^500) / (1 - 0.99) ≈ 99.34

See `docs/DISCOUNT_FACTOR_CORRECTION.md` for full explanation.

## Important Design Decisions

### v2.0 Changes (2025-10-25)

**Auxiliary Loss Update:**
- OLD: Proportionally scaled all actions when one violated
- NEW: Only clips individual Q-values that violate bounds
- Reason: Avoid degrading well-behaved actions

See `docs/CHANGES.md` for details.

## Key Integration Points

1. **All experiments** → Import from `src/`
2. **Training scripts** → Save results to `results/<environment>/`
3. **Analysis scripts** → Read from `results/` folders
4. **Core changes in src/** → Affect ALL experiments

## Important Conventions

1. **File Naming:**
   - Experiment scripts: `train_<environment>.py`
   - Analysis scripts: Descriptive names (e.g., `analyze_results.py`)
   - Documentation: `.md` files in `docs/`

2. **Results Naming:**
   - Use timestamps: `results_YYYYMMDD_HHMMSS.json`
   - Save to environment-specific folders

3. **Code Style:**
   - Use docstrings for all functions
   - Type hints for parameters
   - Clear variable names

4. **Adding New Environments:**
   - Create folder: `experiments/<new_env>/`
   - Create script: `experiments/<new_env>/train_<new_env>.py`
   - Create results folder: `results/<new_env>/`
   - Update `experiments/combined/run_all_experiments.py`

## Known Issues

1. **Q_max values incorrectly set** - Based on step rewards instead of episode returns
2. **CartPole severely limited** - Q_max=100 but optimal return ≈500
3. **GridWorld value propagation** - Q_max=1.0 prevents proper learning

See `docs/ANALYSIS_SUMMARY.md` for detailed analysis.

## Quick Reference

- **Main algorithm:** `src/dqn_agent.py`
- **GridWorld code:** `experiments/gridworld/train_gridworld.py`
- **FrozenLake code:** `experiments/frozenlake/train_frozenlake.py`
- **CartPole code:** `experiments/cartpole/train_cartpole.py`
- **Run all:** `experiments/combined/run_all_experiments.py`
- **Analysis:** `analysis/` folder
- **Documentation:** `docs/` folder
- **Results:** `results/` folder

## Paper Compilation and Overleaf Upload

### Self-Contained LaTeX Directory

The `QBound/` directory is designed to be **self-contained** for easy upload to Overleaf or other LaTeX platforms:

```
QBound/
├── main.tex              # Main paper file
├── references.bib        # Bibliography
├── arxiv.sty            # ArXiv style file
└── figures/             # All plots (PDFs)
    ├── learning_curves_*.pdf
    ├── gridworld_learning_curve_*.pdf
    ├── frozenlake_learning_curve_*.pdf
    ├── cartpole_learning_curve_*.pdf
    └── comparison_bar_chart_*.pdf
```

### Generating Plots for Paper

**IMPORTANT:** After running experiments, always generate plots for the paper:

```bash
# This will:
# 1. Generate plots in results/plots/
# 2. Copy PDFs to QBound/figures/ for LaTeX
python3 analysis/plot_paper_results.py
```

### Uploading to Overleaf

Simply zip and upload the entire `QBound/` directory:

```bash
# From project root
zip -r qbound_paper.zip QBound/
# Upload qbound_paper.zip to Overleaf
```

All figure references in `main.tex` use relative paths: `figures/filename.pdf`

### LaTeX Compilation

```bash
cd QBound
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Remember

**🔴 CRITICAL: The directory structure MUST be maintained for all future experiments!**

See `PROJECT_ORGANIZATION.md` for complete documentation on the structure.

### Key Reminders:

- **Plots must be in QBound/figures/**: Always run `plot_paper_results.py` after experiments to copy plots to the LaTeX directory
- **Self-contained paper**: The QBound/ folder should be uploadable as-is to Overleaf
- **Data preservation**: Raw data in `results/` folders is preserved so plots can be regenerated if needed (e.g., for rebranding)
- **Git commits**: Use git to maintain different versions when experiment settings change
- **LaTeX paths**: All figure paths in main.tex are relative: `figures/filename.pdf` (NOT `../results/plots/`)