# QBound - Q-Value Bounding for Deep Reinforcement Learning

A research project implementing QBound, a technique for bounding Q-values in Deep Q-Networks (DQN) to improve learning in sparse reward environments.

## 📁 Project Structure

```
QBound/
├── src/                          # Core implementation
│   ├── dqn_agent.py             # DQN agent with QBound
│   └── environment.py           # GridWorld environment
│
├── experiments/                  # Experiment scripts
│   ├── gridworld/               # GridWorld experiments
│   │   └── train_gridworld.py   # Train on GridWorld (10x10 grid)
│   ├── frozenlake/              # FrozenLake experiments
│   │   └── train_frozenlake.py  # Train on FrozenLake (4x4, slippery)
│   ├── cartpole/                # CartPole experiments
│   │   └── train_cartpole.py    # Train on CartPole (balance task)
│   └── combined/                # Run all experiments
│       └── run_all_experiments.py  # Run all 3 environments
│
├── analysis/                     # Analysis and visualization
│   ├── analyze_results.py       # Analyze experiment results
│   ├── comprehensive_analysis.py # Full analysis with plots
│   ├── qbound_summary_table.py  # Generate summary tables
│   ├── show_qbound_config.py    # Show configuration analysis
│   ├── track_q_values.py        # Track Q-value statistics
│   ├── generate_plot.py         # Generate plots for paper
│   └── update_paper_with_results.py  # Update paper with results
│
├── docs/                         # Documentation
│   ├── ANALYSIS_SUMMARY.md      # Full analysis of results
│   ├── CHANGES.md               # Code change log
│   └── explain_aux_weight.md    # Explanation of aux_weight parameter
│
├── results/                      # Experiment results
│   ├── gridworld/               # GridWorld results
│   ├── frozenlake/              # FrozenLake results
│   ├── cartpole/                # CartPole results
│   ├── combined/                # Combined results
│   └── plots/                   # Generated plots
│
├── CLAUDE.md                     # Project instructions for Claude
└── README.md                     # This file
```

## 🎯 Environments

### 1. GridWorld (10x10)
- **File:** `experiments/gridworld/train_gridworld.py`
- **Environment:** Custom 10x10 grid, start at (0,0), goal at (9,9)
- **Reward:** +1 for reaching goal, 0 otherwise
- **QBound Config:** Q_min=0.0, Q_max=1.0, γ=0.99
- **Episodes:** 500
- **Status:** ❌ QBound underperforms (-22.1%)

### 2. FrozenLake (4x4, Slippery)
- **File:** `experiments/frozenlake/train_frozenlake.py`
- **Environment:** Gymnasium FrozenLake-v1 (stochastic)
- **Reward:** +1 for reaching goal, 0 otherwise
- **QBound Config:** Q_min=0.0, Q_max=1.0, γ=0.95
- **Episodes:** 2000
- **Status:** ✅ QBound works! (+19.4% faster convergence)

### 3. CartPole
- **File:** `experiments/cartpole/train_cartpole.py`
- **Environment:** Gymnasium CartPole-v1 (balance pole)
- **Reward:** +1 per timestep survived (max 500)
- **QBound Config:** Q_min=0.0, Q_max=100.0, γ=0.99
- **Episodes:** 500
- **Status:** ❌ QBound severely underperforms (-41.4%)

## 🚀 Quick Start

### Run Individual Experiments

```bash
# GridWorld
cd /root/projects/QBound
python experiments/gridworld/train_gridworld.py

# FrozenLake
python experiments/frozenlake/train_frozenlake.py

# CartPole
python experiments/cartpole/train_cartpole.py
```

### Run All Experiments

```bash
python experiments/combined/run_all_experiments.py
```

### Analyze Results

```bash
# Quick summary
python analysis/qbound_summary_table.py

# Detailed analysis
python analysis/analyze_results.py

# Full analysis with plots
python analysis/comprehensive_analysis.py

# Show Q-value configuration
python analysis/show_qbound_config.py
```

## 📊 Key Results

| Environment | QBound Episodes | Baseline Episodes | Performance |
|------------|----------------|------------------|-------------|
| GridWorld  | 326            | 267              | -22.1% ❌   |
| FrozenLake | 203            | 252              | +19.4% ✅   |
| CartPole   | N/A            | N/A              | -41.4% ❌   |

### Key Findings

1. **QBound works well in stochastic environments** (FrozenLake ✅)
2. **QBound struggles with high discount factors** (GridWorld ❌)
3. **QBound fails when Q_max is too restrictive** (CartPole ❌)

## 🔧 Core Components

### DQN Agent (`src/dqn_agent.py`)

Implements DQN with optional QBound using dual-loss training:

**Primary Loss:** Standard TD loss for learning optimal Q-values
```python
primary_loss = MSE(Q(s,a), r + γ * max_a' Q(s',a'))
```

**Auxiliary Loss:** Penalizes only Q-values that violate [Q_min, Q_max]
```python
violation_mask = (Q < Q_min) | (Q > Q_max)
aux_loss = MSE(Q[violation_mask], clip(Q[violation_mask]))
```

**Combined Loss:**
```python
total_loss = primary_loss + aux_weight * aux_loss
```

### Key Parameters

- `use_qclip`: Enable/disable QBound (True/False)
- `qclip_min`: Lower bound for Q-values
- `qclip_max`: Upper bound for Q-values
- `aux_weight`: Weight for auxiliary loss (default: 0.5)
- `gamma`: Discount factor

## 📈 Recent Changes

### v2.0 - Fixed Auxiliary Loss (2025-10-25)

**Changed:** Auxiliary loss now clips only violating Q-values instead of scaling all actions

**Before:**
- When one action violated bounds, ALL actions were scaled proportionally
- Problem: Punished good learners for one bad action

**After:**
- Only Q-values that violate bounds are clipped
- Benefit: Well-behaved actions remain unchanged

See `docs/CHANGES.md` for details.

## 📝 Documentation

- **docs/ANALYSIS_SUMMARY.md** - Comprehensive analysis of all experiments
- **docs/CHANGES.md** - Code change history
- **docs/explain_aux_weight.md** - Detailed explanation of aux_weight parameter

## ⚠️ Known Issues

1. **Q_max values are incorrectly set** - Based on step rewards instead of episode returns
2. **CartPole severely limited** - Q_max=100 but optimal return ≈500
3. **GridWorld value propagation** - Q_max=1.0 prevents proper learning

## 🔮 Future Work

1. Fix Q_max values based on maximum episode returns
2. Experiment with different aux_weight values (0.0 to 1.0)
3. Test with various discount factors
4. Add more environments (Atari, MuJoCo)
5. Implement adaptive Q_max bounds

## 📄 Citation

```bibtex
@article{qbound2025,
  title={QBound: Q-Value Bounding for Deep Reinforcement Learning},
  author={...},
  year={2025}
}
```
