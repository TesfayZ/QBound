# QBound: Environment-Aware Q-Value Bounds for Stable Temporal Difference Learning

[![Paper](https://img.shields.io/badge/Paper-PDF-red)](LatexDocs/main.pdf)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-orange.svg)](https://pytorch.org/)

A comprehensive implementation and evaluation of **QBound**, a stabilization mechanism that exploits environment structure to prevent overestimation bias in deep reinforcement learning by deriving and enforcing Q-value bounds from known reward structures.

**Important:** QBound is a *specialized technique* effective only for **positive dense reward environments** (e.g., CartPole). It causes degradation on negative reward environments and provides no benefit on sparse terminal rewards.

## Research Paper

**QBound: Environment-Aware Q-Value Bounds for Stable Temporal Difference Learning**

*Tesfay*

Value-based reinforcement learning methods suffer from overestimation bias ([Thrun & Schwartz, 1993](https://proceedings.neurips.cc/paper/1993); [Van Hasselt et al., 2016](https://arxiv.org/abs/1509.06461)), where bootstrapped Q-value estimates systematically exceed true values. QBound addresses this at the source by clipping next-state Q-values to environment-specific bounds [Q_min, Q_max].

**Scope:** QBound targets *off-policy* value-based methods (DQN, DDQN, Dueling DQN) and off-policy actor-critic methods (DDPG, TD3) that use experience replay. On-policy methods (e.g., PPO, A2C, REINFORCE) are outside QBound's scope.

[Read the full paper (PDF)](LatexDocs/main.pdf)

---

## Quick Results Summary (5-Seed Evaluation)

All results below are from **5-seed experiments** (seeds 42-46) for statistical validity.

### Positive Dense Rewards (Recommended)

| Environment | Algorithm | Improvement | Win Rate | Notes |
|-------------|-----------|-------------|----------|-------|
| **CartPole** | DQN | **+12.0%** | 4/5 (80%) | Consistent improvement |
| **CartPole** | DDQN | **+33.6%** | - | Best improvement |
| **CartPole** | Dueling DQN | **+22.5%** | 5/5 (100%) | Most reliable |

### Negative Rewards (Not Recommended)

| Environment | Algorithm | Result | Win Rate | Notes |
|-------------|-----------|--------|----------|-------|
| **Pendulum** | DQN | -3.3% | 2/5 (40%) | Degradation |
| **Pendulum** | DDPG | -8.0% | 2/5 (40%) | Degradation |
| **Pendulum** | TD3 | +4.1% | 4/5 (80%) | Exception (72% higher variance) |

### Sparse Terminal Rewards (No Benefit)

| Environment | Algorithm | Change | Win Rate | Notes |
|-------------|-----------|--------|----------|-------|
| **GridWorld** | DQN | -1.0% | 1/5 (20%) | Bounds trivially satisfied |
| **FrozenLake** | DQN | -1.7% | 3/5 (60%) | No better than chance |

### State-Dependent Negative Rewards (Degradation)

| Environment | Algorithm | Change | Notes |
|-------------|-----------|--------|-------|
| **MountainCar** | DQN | -8.2% | Causes degradation |
| **MountainCar** | DDQN | **-47.4%** | Severe degradation |
| **Acrobot** | DQN | -4.9% | Causes degradation |
| **Acrobot** | DDQN | -3.6% | Causes degradation |

### Key Insight

QBound's effectiveness fundamentally depends on **reward sign and structure**:
- **Positive dense rewards**: Strong improvement (CartPole: +12% to +33.6%)
- **Sparse terminal rewards**: No benefit (Q bounds trivially satisfied)
- **Negative rewards**: Degradation (underlying cause remains open question)

**Recommendation**: Use QBound only for **positive dense reward environments** with Dueling DQN architecture (100% win rate, +22.5% mean improvement).

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
│   ├── ddpg_agent.py            # DDPG with Architectural QBound
│   ├── td3_agent.py             # TD3 with Architectural QBound
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
| **Architectural QBound** | Negative rewards | `Q = -softplus(logits)` | Evaluated (fails for most algorithms) |
| **Soft QBound** | Continuous actions (penalty-based) | Quadratic penalty loss | Not empirically evaluated |

### When to Use QBound

**Important:** QBound is a *specialized technique*, not a universal improvement.

| Environment Type | Recommendation | Reason |
|------------------|----------------|--------|
| Positive dense rewards | **Recommended** | +12% to +33.6% improvement |
| Sparse terminal rewards | Not recommended | Bounds trivially satisfied |
| Negative rewards | **Do not use** | Causes degradation (-3% to -47%) |

---

## Known Limitations

1. **Negative Rewards**: QBound degrades performance on negative reward environments (Pendulum: -3% to -8%, MountainCar: -47%)

2. **Sparse Terminal Rewards**: No benefit on sparse terminal reward tasks (GridWorld, FrozenLake)

3. **Seed Sensitivity**: Even in successful cases (CartPole DQN), 20% of seeds show degradation. Always run multiple seeds.

These are **fundamental limitations based on reward structure**, not implementation bugs.

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
