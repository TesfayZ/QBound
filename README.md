# QBound: Environment-Aware Q-Value Bounds for Stable Temporal Difference Learning

[![Paper](https://img.shields.io/badge/Paper-PDF-red)](LatexDocs/main.pdf)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-orange.svg)](https://pytorch.org/)

A comprehensive implementation and evaluation of **QBound**, a stabilization mechanism that exploits environment structure to prevent overestimation bias in deep reinforcement learning by deriving and enforcing Q-value bounds from known reward structures.

**Summary of Results:**
- **Positive dense rewards**: Recommended (+12% to +33%, 80-100% win rate)
- **Sparse rewards**: Neutral/No benefit (violations rare: 0-1.5%)
- **Negative dense + DDPG/TD3**: Mixed (+15% to +25% mean, but only 40-60% win rate)
- **Negative dense + DQN variants**: Degrading (-7.1%, under investigation)

## Research Paper

**QBound: Environment-Aware Q-Value Bounds for Stable Temporal Difference Learning**

*Tesfay*

Value-based reinforcement learning methods suffer from overestimation bias ([Thrun & Schwartz, 1993](https://proceedings.neurips.cc/paper/1993); [Van Hasselt et al., 2016](https://arxiv.org/abs/1509.06461)), where bootstrapped Q-value estimates systematically exceed true values. QBound addresses this at the source by clipping next-state Q-values to environment-specific bounds [Q_min, Q_max].

**Scope:** QBound targets *off-policy* value-based methods (DQN, DDQN, Dueling DQN) and off-policy actor-critic methods (DDPG, TD3) that use experience replay. On-policy methods (e.g., PPO, A2C, REINFORCE) are outside QBound's scope.

[Read the full paper (PDF)](LatexDocs/main.pdf)

---

## Quick Results Summary (5-Seed Evaluation)

All results below are from **5-seed experiments** (seeds 42-46) for statistical validity.

### Complete Results: Mean Change, Win Rate, and Violation Rate

| Reward Type | Environment | Algorithm | Mean Δ | Win Rate | Violation Rate | Status |
|-------------|-------------|-----------|--------|----------|----------------|--------|
| **Positive Dense** | CartPole | DQN | **+12.0%** | 4/5 (80%) | 13.3% | ✓ Recommended |
| **Positive Dense** | CartPole | DDQN | **+33.6%** | - | 13.3% | ✓ Recommended |
| **Positive Dense** | CartPole | Dueling | **+22.5%** | 5/5 (100%) | 13.3% | ✓ Recommended |
| **Sparse Positive** | GridWorld | DQN | -1.0% | 1/5 (20%) | 1.5% | No benefit |
| **Sparse Positive** | FrozenLake | DQN | -1.7% | 3/5 (60%) | 0.0% | No benefit |
| **Sparse Negative** | MountainCar | DQN | -8.2% | 1/5 (20%) | 0.4% | No benefit |
| **Sparse Negative** | Acrobot | DQN | -4.9% | 2/5 (40%) | 0.8% | No benefit |
| **Negative Dense** | Pendulum | DQN | -7.1% | 2/5 (40%) | 55.5% | ⚠️ Under investigation |
| **Negative Dense** | Pendulum | DDQN | -7.1% | 2/5 (40%) | 3.8% | ⚠️ Under investigation |
| **Negative Dense** | Pendulum | DDPG | **+25.0%** | 2/5 (40%) | 29.7% | Mixed |
| **Negative Dense** | Pendulum | TD3 | **+15.3%** | 3/5 (60%) | 3.5% | Mixed |

### Key Findings

**What works:**
- **Positive dense rewards + Hard QBound**: Strong improvement (CartPole: +12% to +33.6%, 80-100% win rate)

**Mixed results:**
- **Negative dense rewards + Soft QBound (DDPG/TD3)**: +15% to +25% mean improvement but only 40-60% win rate

**What doesn't work:**
- **Sparse rewards**: No benefit (violations rare: 0-1.5%)
- **Negative dense rewards + Hard QBound (DQN)**: Degradation despite high violations

### Under Investigation

**DQN variants fail on dense negative rewards despite high violation rates:**
- Pendulum DQN: 55.5% violation rate → -7.1% performance (degradation)
- CartPole DQN: 13.3% violation rate → +12.0% performance (improvement)

Why high violations (55.5%) fail to benefit from clipping on dense negative rewards is under investigation.

**Recommendation**: Use Hard QBound for **positive dense reward environments** with Dueling DQN (100% win rate). For continuous control with negative rewards, Soft QBound with DDPG/TD3 shows mixed results (40-60% win rate).

---

## Theoretical Background

### The Overestimation Problem

In temporal difference learning, Q-values are updated via bootstrapping:

```
Q(s,a) <- r + gamma * max_a' Q(s', a')
```

The max operator introduces **overestimation bias** when Q-values contain approximation errors.

### QBound Solution

QBound exploits known environment structure to derive **theoretically justified bounds**:

**For positive rewards (r >= 0):**
```
Q_max = r_max * (1 - gamma^H) / (1 - gamma)
Q_min = 0
```

**For negative rewards (r <= 0):**
```
Q_max = 0
Q_min = r_min * (1 - gamma^H) / (1 - gamma)
```

**Core mechanism:** Clip next-state Q-values during bootstrapping:
```
Q_target = r + gamma * clip(Q(s', a'), Q_min, Q_max)
```

---

## Installation

### Prerequisites

- Python 3.8 or higher
- PyTorch 1.9+

### Setup

```bash
# Clone the repository
git clone https://github.com/TesfayZ/QBound.git
cd QBound

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Quick Start

### Run Multi-Seed Experiments

```bash
# Run ALL experiments with 5 seeds (recommended)
python experiments/run_all_organized_experiments.py --seeds 42 43 44 45 46

# Run with single seed (quick test)
python experiments/run_all_organized_experiments.py --seed 42

# Run only specific experiment categories
python experiments/run_all_organized_experiments.py --category timestep   # CartPole, Pendulum
python experiments/run_all_organized_experiments.py --category sparse     # GridWorld, FrozenLake, etc.

# Crash Recovery: If interrupted, re-run the same command
# The script automatically skips completed experiments
```

### Run Individual Experiments

```bash
# CartPole (positive dense rewards)
python experiments/cartpole/train_cartpole_dqn_full_qbound.py --seed 42

# Pendulum DDPG/TD3
python experiments/pendulum/train_pendulum_ddpg_full_qbound.py --seed 42

# GridWorld (sparse terminal rewards)
python experiments/gridworld/train_gridworld_dqn_static_qbound.py --seed 42
```

### Generate Plots

```bash
python analysis/analyze_all_6way_results.py
# Plots saved to results/plots/ and LatexDocs/figures/
```

---

## Project Structure

```
QBound/
├── src/                          # Core implementations
│   ├── dqn_agent.py             # DQN with QBound
│   ├── double_dqn_agent.py      # Double DQN with QBound
│   ├── dueling_dqn_agent.py     # Dueling DQN with QBound
│   ├── ddpg_agent.py            # DDPG with Soft QBound
│   ├── td3_agent.py             # TD3 with Soft QBound
│   └── environment.py           # Custom GridWorld
│
├── experiments/                  # Experiment scripts
│   ├── gridworld/               # GridWorld experiments
│   ├── frozenlake/              # FrozenLake experiments
│   ├── cartpole/                # CartPole experiments
│   ├── pendulum/                # Pendulum DDPG/TD3 experiments
│   └── run_all_organized_experiments.py  # Multi-seed runner
│
├── analysis/                     # Analysis scripts
│   └── analyze_all_6way_results.py
│
├── results/                      # Experimental results
│
├── LatexDocs/                    # Paper directory
│   ├── main.tex                 # LaTeX source
│   ├── main.pdf                 # Compiled paper
│   └── figures/                 # Paper figures
│
└── docs/                         # Documentation
```

---

## Usage Guide

### Basic Usage: DQN with QBound

```python
from src.dqn_agent import DQNAgent

# Create agent with QBound
agent = DQNAgent(
    state_dim=4,
    action_dim=2,
    hidden_dim=128,
    lr=0.001,
    gamma=0.99,
    use_qclip=True,           # Enable QBound
    qclip_min=0.0,            # Lower bound
    qclip_max=99.34,          # Upper bound (CartPole)
    device='cpu'
)
```

### Computing Q_min and Q_max

```python
# Dense step rewards (e.g., CartPole)
gamma = 0.99
max_steps = 500
reward_per_step = 1.0

Q_max = reward_per_step * (1 - gamma**max_steps) / (1 - gamma)  # ~99.34
Q_min = 0.0
```

---

## Configuration Guidelines

### QBound Variants

| Variant | Use Case | Implementation | Status |
|---------|----------|----------------|--------|
| **Hard QBound** | Discrete actions (DQN, DDQN, Dueling) | `clip(Q, Q_min, Q_max)` | Evaluated (+12-34% on CartPole) |
| **Soft QBound** | Continuous actions (DDPG, TD3) | Penalty-based loss | Evaluated (mixed results on Pendulum) |

### When to Use QBound

**Important:** QBound is a *specialized technique*, not a universal improvement.

| Environment Type | QBound Type | Recommendation | Reason |
|------------------|-------------|----------------|--------|
| Positive dense rewards | Hard | **Recommended** | +12% to +33.6% improvement |
| Sparse terminal rewards | Hard | Not recommended | Bounds trivially satisfied |
| Negative rewards (DQN) | Hard | **Do not use** | Degradation (-7% to -47%) |
| Negative rewards (DDPG/TD3) | Soft | Mixed | +15-25% mean but high variance |

---

## Known Limitations

1. **Hard QBound on Negative Rewards**: Hard QBound degrades DQN variants on negative reward environments (Pendulum DDQN: -7.1%, MountainCar DDQN: -47%)

2. **Sparse Terminal Rewards**: No benefit on sparse terminal reward tasks (GridWorld, FrozenLake)

3. **High Seed Variance**: Soft QBound on DDPG/TD3 shows high variance across seeds (e.g., DDPG ranges from +61.5% to -4.2%). Always run multiple seeds.

4. **State-Dependent Negative Rewards**: MountainCar and Acrobot show degradation with Hard QBound.

---

## Citation

```bibtex
@article{tesfay2025qbound,
  title={QBound: Environment-Aware Q-Value Bounds for Stable Temporal Difference Learning},
  author={Tesfay},
  journal={arXiv preprint},
  year={2025}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

This research was conducted by Tesfay with [Claude](https://claude.ai) (Anthropic) serving as an AI coding and research assistant. Claude assisted with code implementation, experimental design, data analysis, and manuscript preparation. All research direction, core ideas, and final decisions were made by the author.

**Tools and Resources:**
- Built with [PyTorch](https://pytorch.org/)
- Environments from [Gymnasium](https://gymnasium.farama.org/)
- Theoretical foundations from [Sutton & Barto (2018)](http://incompleteideas.net/book/the-book-2nd.html)
