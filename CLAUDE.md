# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**QBound** - Q-Value Bounding for Deep Reinforcement Learning

A research project implementing QBound, a technique for bounding Q-values in Deep Q-Networks (DQN) to improve learning in sparse reward environments. Tests on GridWorld, FrozenLake, CartPole, Pendulum, MountainCar, and Acrobot environments.

## Experiment Organization (Option 1: QBound Variant Coverage)

**NEW**: Experiments are organized by QBound applicability according to Rule 3:

### Category 1: Time-Step Dependent Rewards (Static + Dynamic QBound)
**6 scripts, 30 total methods**

Environments where rewards accumulate predictably with time steps:
- **CartPole-v1**: Dense positive reward (+1 per step)
- **Pendulum-v1**: Dense negative reward (time-step dependent cost)

Scripts:
1. `train_cartpole_dqn_full_qbound.py` (6 methods: DQN/DDQN × baseline/static/dynamic)
2. `train_cartpole_dueling_full_qbound.py` (6 methods: Dueling DQN × baseline/static/dynamic)
3. `train_pendulum_dqn_full_qbound.py` (6 methods: DQN/DDQN × baseline/static/dynamic)
4. `train_pendulum_ddpg_full_qbound.py` (3 methods: DDPG × baseline/static/dynamic with softplus_clip)
5. `train_pendulum_td3_full_qbound.py` (3 methods: TD3 × baseline/static/dynamic with softplus_clip)
6. `train_pendulum_ppo_full_qbound.py` (3 methods: PPO × baseline/static/dynamic with softplus_clip on V(s))

### Category 2: Sparse/State-Dependent Rewards (Static QBound Only)
**4 scripts, 16 total methods**

Environments where rewards depend on state transitions, NOT time steps:
- **GridWorld**: Sparse terminal (+1 at goal only)
- **FrozenLake-v1**: Sparse terminal (+1 at goal only)
- **MountainCar-v0**: State-dependent (-1 until goal reached)
- **Acrobot-v1**: State-dependent (-1 until swing-up)

Scripts (4 methods each: DQN/DDQN × baseline/static):
1. `train_gridworld_dqn_static_qbound.py`
2. `train_frozenlake_dqn_static_qbound.py`
3. `train_mountaincar_dqn_static_qbound.py`
4. `train_acrobot_dqn_static_qbound.py`

**Rule 3**: Dynamic QBound is ONLY applicable to time-step dependent rewards. For sparse/state-dependent rewards, only static QBound is theoretically justified.

See `docs/EXPERIMENT_ORGANIZATION.md` for complete details.

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

### Run Organized Experiments (Option 1: QBound Variant Coverage)

**NEW**: Experiments are now organized by QBound applicability with **MULTI-SEED SUPPORT**:

```bash
# Run all organized experiments with default seed (42)
python3 experiments/run_all_organized_experiments.py

# Run with single custom seed
python3 experiments/run_all_organized_experiments.py --seed 123

# 🆕 Run with MULTIPLE seeds (automatic multi-seed runs with crash recovery)
python3 experiments/run_all_organized_experiments.py --seeds 42 43 44

# Multi-seed with 5 seeds (recommended for statistical significance)
python3 experiments/run_all_organized_experiments.py --seeds 42 43 44 45 46

# Run only time-step dependent experiments (static + dynamic QBound)
python3 experiments/run_all_organized_experiments.py --category timestep --seeds 42 43 44

# Run only sparse/state-dependent experiments (static QBound only)
python3 experiments/run_all_organized_experiments.py --category sparse --seeds 42 43 44

# Dry run (preview what would be executed)
python3 experiments/run_all_organized_experiments.py --dry-run --seeds 42 43 44
```

**🔄 Crash Recovery**: If interrupted, simply re-run the same command. The system will:
- Load existing progress from `results/organized_experiments_log.json`
- Skip already-completed seeds automatically
- Continue from where it left off

See `docs/EXPERIMENT_ORGANIZATION.md` for complete documentation on the new structure.

### Run Individual Experiments

**Time-step dependent (Static + Dynamic QBound)**:
```bash
# CartPole experiments
python3 experiments/cartpole/train_cartpole_dqn_full_qbound.py --seed 42
python3 experiments/cartpole/train_cartpole_dueling_full_qbound.py --seed 42

# Pendulum experiments
python3 experiments/pendulum/train_pendulum_dqn_full_qbound.py --seed 42
python3 experiments/pendulum/train_pendulum_ddpg_full_qbound.py --seed 42
python3 experiments/ppo/train_pendulum_ppo_full_qbound.py --seed 42
```

**Sparse/State-dependent (Static QBound only)**:
```bash
python3 experiments/gridworld/train_gridworld_dqn_static_qbound.py --seed 42
python3 experiments/frozenlake/train_frozenlake_dqn_static_qbound.py --seed 42
python3 experiments/mountaincar/train_mountaincar_dqn_static_qbound.py --seed 42
python3 experiments/acrobot/train_acrobot_dqn_static_qbound.py --seed 42
```

### Run Legacy Experiments

```bash
# Old 6-way comparison scripts (still available)
python3 experiments/cartpole/train_cartpole_6way.py
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

## QBound Violation Tracking

**NEW (2025-10-29)**: All QBound experiments now track violations instead of running ablation studies.

### What is Tracked

**Hard QBound (DQN/DDQN/Dueling):**
- Violation rates: % of Q-values violating bounds per minibatch
- Violation magnitudes: How far beyond bounds
- Separate tracking for next-state Q vs TD targets

**Soft QBound (DDPG/TD3/PPO):**
- Clipping activation rate: % of Q-values exceeding bounds (before soft clipping)
- Clipping magnitude: How far Q-values deviate from bounds
- Gradient flow metrics: Verifies non-zero gradients during clipping

### Usage in Results

All experiment results include violation statistics:
```json
{
  "training": {
    "static_qbound": {
      "rewards": [...],
      "violations": {
        "per_episode": [{...}, {...}],
        "mean": {...},
        "final_100": {...}
      }
    }
  }
}
```

### Analysis

See `docs/QBOUND_VIOLATION_TRACKING.md` for:
- Complete metric definitions
- Visualization guidelines
- Interpretation for reviewers

## Known Issues

1. **Q_max values based on theoretical bounds** - May need empirical validation
2. **Early training violations** - Expected as agent explores
3. **Late training violations** - Should decrease as Q-values stabilize

See `docs/ANALYSIS_SUMMARY.md` for detailed analysis.

## Quick Reference

### Core Components
- **Main algorithm:** `src/dqn_agent.py`
- **Analysis tools:** `analysis/` folder
- **Documentation:** `docs/` folder
- **Results:** `results/` folder

### Organized Experiments (NEW)
- **Run all organized:** `experiments/run_all_organized_experiments.py`
- **Organization docs:** `docs/EXPERIMENT_ORGANIZATION.md`

**Time-step dependent (Static + Dynamic QBound)**:
- CartPole DQN: `experiments/cartpole/train_cartpole_dqn_full_qbound.py`
- CartPole Dueling: `experiments/cartpole/train_cartpole_dueling_full_qbound.py`
- Pendulum DQN: `experiments/pendulum/train_pendulum_dqn_full_qbound.py`
- Pendulum DDPG: `experiments/pendulum/train_pendulum_ddpg_full_qbound.py`
- Pendulum PPO: `experiments/ppo/train_pendulum_ppo_full_qbound.py`

**Sparse/State-dependent (Static QBound only)**:
- GridWorld: `experiments/gridworld/train_gridworld_dqn_static_qbound.py`
- FrozenLake: `experiments/frozenlake/train_frozenlake_dqn_static_qbound.py`
- MountainCar: `experiments/mountaincar/train_mountaincar_dqn_static_qbound.py`
- Acrobot: `experiments/acrobot/train_acrobot_dqn_static_qbound.py`

### Legacy Experiments
- **Run all (old):** `experiments/combined/run_all_experiments.py`
- **CartPole 6-way:** `experiments/cartpole/train_cartpole_6way.py`

## Paper Compilation and Overleaf Upload

### Self-Contained LaTeX Directory

The `LatexDocs/` directory is designed to be **self-contained** for easy upload to Overleaf or other LaTeX platforms:

```
LatexDocs/
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
# 2. Copy PDFs to LatexDocs/figures/ for LaTeX
python3 analysis/plot_paper_results.py
```

### Uploading to Overleaf

Simply zip and upload the entire `LatexDocs/` directory:

```bash
# From project root
zip -r qbound_paper.zip LatexDocs/
# Upload qbound_paper.zip to Overleaf
```

All figure references in `main.tex` use relative paths: `figures/filename.pdf`

### LaTeX Compilation

```bash
cd LatexDocs
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Remember

**🔴 CRITICAL: The directory structure MUST be maintained for all future experiments!**

See `docs/EXPERIMENT_ORGANIZATION.md` for complete documentation on experiment organization.

### Key Reminders:

- **Plots must be in LatexDocs/figures/**: Always run `plot_paper_results.py` after experiments to copy plots to the LaTeX directory
- **Self-contained paper**: The LatexDocs/ folder should be uploadable as-is to Overleaf
- **Data preservation**: Raw data in `results/` folders is preserved so plots can be regenerated if needed (e.g., for rebranding)
- **Git commits**: Use git to maintain different versions when experiment settings change
- **LaTeX paths**: All figure paths in main.tex are relative: `figures/filename.pdf` (NOT `../results/plots/`)