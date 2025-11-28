# QBound: Environment-Aware Q-Value Bounds for Stable Temporal Difference Learning

[![Paper](https://img.shields.io/badge/Paper-PDF-red)](QBound/main.pdf)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-orange.svg)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Status-Work%20in%20Progress-yellow.svg)]()

> ⚠️ **Work in Progress**: This is ongoing research. Contributions, experiments, and feedback are welcome!

A comprehensive implementation and evaluation of **QBound**, a stabilization mechanism that exploits environment structure to prevent overestimation bias in deep reinforcement learning by deriving and enforcing Q-value bounds from known reward structures.

## 📄 Research Paper (Draft)

**QBound: Environment-Aware Q-Value Bounds for Stable Temporal Difference Learning**

*Tesfay*

> **Note**: This paper is a working draft documenting current findings. Several open questions remain, particularly regarding negative reward environments.

Value-based reinforcement learning methods suffer from overestimation bias ([Thrun & Schwartz, 1993](https://proceedings.neurips.cc/paper/1993); [Van Hasselt et al., 2016](https://arxiv.org/abs/1509.06461)), where bootstrapped Q-value estimates systematically exceed true values. QBound addresses this at the source by clipping next-state Q-values to environment-specific bounds [Q_min, Q_max].

**Scope:** QBound targets *off-policy* value-based methods (DQN, DDQN, Dueling DQN) and off-policy actor-critic methods (DDPG, TD3) that use experience replay. On-policy methods (e.g., PPO, A2C, REINFORCE) are outside QBound's scope because they do not suffer from the same overestimation dynamics.

**Key Findings:**
- **Positive dense rewards** (CartPole-style): 12-34% improvement across DQN variants
- **Negative rewards** (Pendulum-style): Generally degrades DQN; architectural QBound works for DDPG/TD3 (+4-7%)
- **Negligible overhead**: <2% computational cost

[Read the current draft (PDF)](QBound/main.pdf)

---

## 🎯 Key Features

- **Hard QBound**: Direct clipping for discrete action spaces (DQN, DDQN, Dueling DQN)
- **Architectural QBound**: Output activation constraints for continuous control (DDPG, TD3)
- **Dynamic Bounds**: Step-aware bounds for dense positive rewards
- **Comprehensive Evaluation**: 8 environments, off-policy algorithms
- **Verified Implementation**: All bounds mathematically proven correct

---

## 📊 Quick Results Summary

**40 experimental runs** (8 environments × 5 seeds) reveal reward-sign dependent effectiveness:

### Positive Dense Rewards (QBound Recommended ✅)

| Environment | Algorithm | Improvement | Notes |
|-------------|-----------|-------------|-------|
| **CartPole** | DQN | +12% | Consistent improvement |
| **CartPole** | DDQN | +33.6% | Best improvement |
| **CartPole** | Dueling DQN | +22.5% | Works with dueling architecture |

### Negative Rewards (Off-Policy Actor-Critic with Architectural QBound)

| Environment | Algorithm | Result | Notes |
|-------------|-----------|--------|-------|
| **Pendulum** | DQN | -7.0% | Hard QBound degrades performance |
| **Pendulum** | DDPG | +4.8% | Architectural QBound works |
| **Pendulum** | TD3 | +7.2% | Architectural QBound works |

### Sparse Rewards (Mixed Results)

| Environment | Algorithm | Improvement | Notes |
|-------------|-----------|-------------|-------|
| **GridWorld** | DDQN | +87.5% | Works for sparse terminal rewards |
| **FrozenLake** | DQN | +282% | Strong improvement |
| **LunarLander** | DQN | +263.9% | Excellent for shaped sparse rewards |

**Key Insight**: QBound's effectiveness fundamentally depends on **reward sign**. Neural networks with linear output layers have no architectural constraint on positive values, making explicit upper bounds essential for positive rewards. For negative rewards, hard QBound can interfere with learning dynamics; architectural enforcement (softplus clipping) works better for actor-critic methods. Future work will investigate Q-value transformation approaches to unify bounding across reward structures.

**Scope Note**: On-policy methods (PPO, A2C, REINFORCE) are outside QBound's scope. These methods naturally suffer less from overestimation bias because they use recent on-policy samples, have no max operator in value updates, and include built-in value stabilization mechanisms.

---

## 📐 Theoretical Background

### The Overestimation Problem

In temporal difference learning ([Sutton & Barto, 2018](http://incompleteideas.net/book/the-book-2nd.html)), Q-values are updated via bootstrapping:

```
Q(s,a) ← r + γ · max_a' Q(s', a')
```

[Thrun & Schwartz (1993)](https://www.ri.cmu.edu/pub_files/pub1/thrun_sebastian_1993_1/thrun_sebastian_1993_1.pdf) identified that this max operator introduces **overestimation bias** when Q-values contain approximation errors. With function approximation, errors compound through bootstrapping, causing Q-values to diverge ([Tsitsiklis & Van Roy, 1997](https://ieeexplore.ieee.org/document/580874)).

### QBound Solution

QBound exploits known environment structure to derive **theoretically justified bounds**:

**For positive rewards (r ≥ 0):**
```
Q_max = Σ_{t=0}^{H-1} γ^t · r_max = r_max · (1 - γ^H) / (1 - γ)
Q_min = 0
```

**For negative rewards (r ≤ 0):**
```
Q_max = 0
Q_min = r_min · (1 - γ^H) / (1 - γ)
```

**Core mechanism:** Clip next-state Q-values during bootstrapping:
```
Q_target = r + γ · clip(Q(s', a'), Q_min, Q_max)
```

This prevents overestimation **at its source** while preserving gradient flow for learning.

### Relationship to Prior Work

| Method | Mechanism | Limitation |
|--------|-----------|------------|
| **Target Networks** ([Mnih et al., 2015](https://www.nature.com/articles/nature14236)) | Delayed updates | Doesn't prevent overestimation, only slows it |
| **Double Q-learning** ([Van Hasselt et al., 2016](https://arxiv.org/abs/1509.06461)) | Decoupled selection/evaluation | Can underestimate; generic pessimism |
| **Clipped Double-Q** ([Fujimoto et al., 2018](https://arxiv.org/abs/1802.09477)) | Min of two critics | Excessive pessimism possible |
| **QBound** (this work) | Environment-specific bounds | Precise bounds, reward-sign dependent |

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA (optional, for GPU acceleration)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/QBound.git
cd QBound

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🏃 Quick Start

### Run Individual Experiments

#### DQN-Based (Discrete Actions)

```bash
# GridWorld 6-way comparison
python experiments/gridworld/train_gridworld_6way.py

# FrozenLake 6-way comparison
python experiments/frozenlake/train_frozenlake_6way.py

# CartPole 6-way comparison
python experiments/cartpole/train_cartpole_6way.py

# LunarLander 6-way comparison
python experiments/lunarlander/train_lunarlander_6way.py
```

#### DDPG/TD3 (Continuous Control)

```bash
# Pendulum 6-way comparison (DDPG/TD3 variants)
python experiments/pendulum/train_6way_comparison.py
```

#### PPO (Policy Gradient)

```bash
# PPO on Pendulum
python experiments/ppo/train_pendulum.py

# PPO on LunarLander Continuous
python experiments/ppo/train_lunarlander_continuous.py
```

### Reproduce All Paper Results

```bash
# Run all 6-way comparisons sequentially
bash run_all_experiments_sequential.sh

# Or run specific experiment sets
bash experiments/run_all_6way_experiments.sh
```

### Generate Plots

```bash
# Generate all paper plots
python analysis/analyze_all_6way_results.py

# Generate Pendulum and PPO plots
python analysis/plot_pendulum_and_ppo.py

# Plots will be saved to:
# - results/plots/
# - QBound/figures/ (for paper)
```

---

## 📁 Project Structure

```
QBound/
├── src/                          # Core implementations
│   ├── dqn_agent.py             # DQN with QBound
│   ├── double_dqn_agent.py      # Double DQN with QBound
│   ├── dueling_dqn_agent.py     # Dueling DQN with QBound
│   ├── ddpg_agent.py            # DDPG with Soft QBound
│   ├── simple_ddpg_agent.py     # DDPG without target networks
│   ├── td3_agent.py             # TD3 with Soft QBound
│   ├── ppo_agent.py             # PPO baseline
│   ├── ppo_qbound_agent.py      # PPO with Soft QBound
│   ├── soft_qbound_penalty.py   # Soft QBound clipping functions
│   └── environment.py           # Custom GridWorld
│
├── experiments/                  # Experiment scripts
│   ├── gridworld/               # GridWorld experiments
│   ├── frozenlake/              # FrozenLake experiments
│   ├── cartpole/                # CartPole experiments
│   ├── lunarlander/             # LunarLander experiments
│   ├── pendulum/                # Pendulum DDPG/TD3 experiments
│   └── ppo/                     # PPO experiments
│
├── analysis/                     # Analysis scripts
│   ├── analyze_all_6way_results.py  # Comprehensive analysis
│   └── plot_pendulum_and_ppo.py     # Pendulum/PPO visualization
│
├── results/                      # Experimental results
│   ├── gridworld/               # GridWorld results
│   ├── frozenlake/              # FrozenLake results
│   ├── cartpole/                # CartPole results
│   ├── lunarlander/             # LunarLander results
│   ├── pendulum/                # Pendulum results
│   ├── ppo/                     # PPO results
│   └── plots/                   # Generated visualizations
│
├── QBound/                       # Paper directory
│   ├── main.tex                 # LaTeX source
│   ├── main.pdf                 # Compiled paper (45 pages)
│   ├── references.bib           # Bibliography
│   └── figures/                 # All paper figures
│
├── docs/                         # Documentation
│   ├── SOFT_QBOUND.md          # Soft QBound explanation
│   └── SOFT_QBOUND_MIGRATION_GUIDE.md
│
├── QBOUND_IMPLEMENTATION_VERIFICATION.md  # Implementation verification
├── QBOUND_VERIFICATION_SUMMARY.md         # Verification summary
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## 🔧 Usage Guide

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
    qclip_max=100.0,          # Upper bound
    device='cpu'
)

# Train
for episode in range(num_episodes):
    state = env.reset()
    done = False

    while not done:
        action = agent.select_action(state, epsilon=epsilon)
        next_state, reward, done, _ = env.step(action)
        agent.replay_buffer.push(state, action, reward, next_state, done)

        if len(agent.replay_buffer) > batch_size:
            agent.train(batch_size)

        state = next_state
```

### Advanced: DDPG with Soft QBound

```python
from src.ddpg_agent import DDPGAgent

# Create agent with Soft QBound
agent = DDPGAgent(
    state_dim=3,
    action_dim=1,
    max_action=2.0,
    use_qbound=True,
    use_soft_clip=True,        # Enable soft clipping (differentiable bounds)
    qbound_min=-1616.0,
    qbound_max=0.0,
    device='cpu'
)

# Training loop
for episode in range(num_episodes):
    state = env.reset()
    agent.reset_noise()

    while not done:
        action = agent.select_action(state, add_noise=True)
        next_state, reward, done, _ = env.step(action)
        agent.replay_buffer.push(state, action, reward, next_state, done)

        if len(agent.replay_buffer) > batch_size:
            # Returns (critic_loss, actor_loss)
            losses = agent.train(batch_size)

        state = next_state
```

### Computing Q_min and Q_max

#### Sparse Terminal Rewards

```python
# Example: GridWorld, FrozenLake
Q_min = 0.0
Q_max = reward_terminal  # e.g., 1.0
```

#### Dense Step Rewards

```python
# Example: CartPole, Pendulum
import numpy as np

gamma = 0.99
max_steps = 500
reward_per_step = 1.0  # or -16.27 for Pendulum

# Geometric sum formula
geometric_sum = (1 - gamma**max_steps) / (1 - gamma)

# Positive rewards
Q_max = reward_per_step * geometric_sum
Q_min = 0.0

# Negative rewards
Q_min = reward_per_step * geometric_sum  # negative value
Q_max = 0.0
```

#### Dynamic Bounds (Step-Aware)

```python
# For dense positive step rewards
def compute_dynamic_qmax(current_step, max_steps, gamma, reward_per_step):
    remaining_steps = max_steps - current_step
    return reward_per_step * (1 - gamma**remaining_steps) / (1 - gamma)

# Enable in DQN agent
agent = DQNAgent(
    ...,
    use_step_aware_qbound=True,
    max_episode_steps=500,
    step_reward=1.0
)
```

---

## 📖 Configuration Guidelines

### When to Use Hard vs Soft QBound

**Hard QBound (Direct Clipping)**
- ✅ Use for: Discrete action spaces (DQN, Double DQN, Dueling DQN)
- ✅ Reason: Policy is ε-greedy, no action gradients needed
- ✅ Implementation: `Q_target = r + γ · clip(Q(s',a'), Q_min, Q_max)`

**Soft QBound (Differentiable Clipping)**
- ✅ Use for: Continuous action spaces (DDPG, TD3, PPO)
- ✅ Reason: Policy learning requires gradient flow through Q-values
- ✅ Implementation: `Q_clipped = softplus_clip(Q, Q_min, Q_max, β)` (smooth, differentiable bounds)

### When to Use Static vs Dynamic Bounds

**Static Bounds**
- ✅ Sparse terminal rewards (GridWorld, FrozenLake)
- ✅ Shaped rewards (LunarLander)
- ✅ Dense negative rewards (Pendulum)

**Dynamic Bounds**
- ✅ Dense positive step rewards (CartPole)
- ✅ Formula: Q_max(t) = r × (1 - γ^(H-t)) / (1 - γ)
- ✅ Result: +17.9% improvement vs static in CartPole PPO

---

## 🧪 Reproducing Paper Results

### Complete Reproduction

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run all experiments (takes ~8-12 hours on CPU)
bash run_all_experiments_sequential.sh

# 3. Generate all plots
python analysis/analyze_all_6way_results.py
python analysis/plot_pendulum_and_ppo.py

# 4. Compile paper (requires LaTeX)
cd QBound
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

### Quick Validation (Subset)

```bash
# Run key experiments only (~2 hours)
python experiments/frozenlake/train_frozenlake_6way.py
python experiments/pendulum/train_6way_comparison.py
python experiments/ppo/train_lunarlander_continuous.py

# Generate plots
python analysis/analyze_all_6way_results.py
```

---

## 📈 Key Findings

### 1. Soft QBound Can Replace Target Networks

**Simple DDPG (no target networks):**
- Baseline: -1464.9 (catastrophic failure)
- With Soft QBound: -205.6 (+712% improvement!)
- Standard DDPG (with targets): -180.8

**Conclusion**: QBound provides alternative stabilization mechanism.

### 2. Algorithm-Specific Interactions

**Works Well:**
- ✅ DQN on LunarLander (+469%)
- ✅ DDPG on Pendulum (+5%)
- ✅ PPO on LunarLander Continuous (+30.6%)

**Conflicts:**
- ❌ TD3 + QBound (-600%) - conflicts with double-Q
- ❌ PPO + QBound on Pendulum (-162%) - conflicts with GAE
- ❌ DDQN on CartPole (-21%) - pessimistic + dense = bad

### 3. Environment-Dependent Effectiveness

**Sparse Rewards**: QBound excels (FrozenLake +282%, LunarLander +469%)

**Dense Rewards**: Mixed results
- Positive dense (CartPole): Dynamic bounds help
- Negative dense (Pendulum): Static sufficient

**Shaped Rewards**: Static bounds work best

---

## 🔬 Implementation Details

### Soft QBound Clipping Mechanism

The `soft_qbound_penalty.py` module provides **differentiable soft clipping** using softplus:

```python
def softplus_clip(q_values, q_min, q_max, beta=1.0):
    """
    Smooth clipping using softplus (differentiable approximation of ReLU).
    Q-values approach bounds asymptotically while maintaining non-zero gradients.
    """
    # Soft lower bound: q_clipped >= q_min
    q_shifted = q_values - q_min
    q_lower = q_min + F.softplus(q_shifted, beta=beta)

    # Soft upper bound: q_clipped <= q_max
    q_shifted = q_max - q_lower
    q_clipped = q_max - F.softplus(q_shifted, beta=beta)

    return q_clipped
```

**Key Parameters:**
- `beta`: Steepness parameter (default=1.0, higher values → closer to hard clipping)
- No penalty weight λ - the clipped values are used directly

### Gradient Flow Verification

```python
# Hard clipping (BAD for continuous control)
Q_clipped = torch.clamp(Q, Q_min, Q_max)
# ∂Q_clipped/∂a = 0 when Q violates bounds ❌

# Soft clipping (GOOD for continuous control)
Q_clipped = softplus_clip(Q, Q_min, Q_max, beta=1.0)
# ∂Q_clipped/∂a ≠ 0 even when Q violates bounds ✅
```

---

## 📊 Experimental Configurations

### DQN Environments

| Environment | Q_min | Q_max | γ | Episodes | Bound Type |
|-------------|-------|-------|---|----------|------------|
| GridWorld | 0.0 | 1.0 | 0.99 | 500 | Static + Dynamic |
| FrozenLake | 0.0 | 1.0 | 0.95 | 2000 | Static + Dynamic |
| CartPole | 0.0 | 99.34 | 0.99 | 500 | Static + Dynamic |
| LunarLander | -100 | 200 | 0.99 | 500 | Static + Dynamic |

### Continuous Control

| Environment | Q_min | Q_max | γ | Episodes | Implementation |
|-------------|-------|-------|---|----------|----------------|
| Pendulum (DDPG) | -1616 | 0 | 0.99 | 500 | Soft QBound |
| Pendulum (PPO) | -3200 | 0 | 0.99 | 500 | Soft QBound |
| LunarLander Cont. (PPO) | -100 | 200 | 0.99 | 500 | Soft QBound |

---

## 🐛 Known Limitations

1. **TD3 Conflict**: QBound conflicts with TD3's clipped double-Q mechanism
2. **Negative Rewards**: Hard QBound degrades performance for DQN on negative reward environments (Pendulum)
3. **DDQN CartPole**: Double-Q pessimism hurts dense positive reward learning

These are **fundamental algorithmic interactions**, not implementation bugs.

---

## 🔮 Future Work

1. **Transformed Q-Values for Negative Rewards**: Investigate whether transforming Q-values to positive space (e.g., Q' = Q - Q_min) allows hard QBound to work effectively on negative reward environments. This could unify the bounding approach across all reward structures.

2. **Adaptive Bound Tightening**: Explore dynamically tightening bounds as learning progresses based on observed Q-value distributions.

3. **Extension to Model-Based RL**: Apply QBound principles to model-based methods where environment structure is learned rather than given.

---

## 📚 Documentation

- **[Paper (main.pdf)](QBound/main.pdf)** - Complete research paper (45 pages)
- **[Implementation Verification](QBOUND_IMPLEMENTATION_VERIFICATION.md)** - Code verification
- **[Verification Summary](QBOUND_VERIFICATION_SUMMARY.md)** - Quick reference
- **[Soft QBound Guide](docs/SOFT_QBOUND.md)** - Detailed soft QBound explanation

---

## 🤝 Contributing

This is an **open research project** and contributions are highly encouraged! Areas where help is needed:

### Open Research Questions
- **Why does QBound work for positive but not negative rewards?** This is the central open question
- **Q-value transformation**: Can transforming negative Q-values to positive space unify the approach?
- **Other environments**: Testing QBound on additional benchmarks (Atari, MuJoCo, etc.)

### How to Contribute
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Ideas for Contributions
- Run experiments with different seeds or hyperparameters
- Test on new environments
- Investigate the negative reward problem
- Improve documentation
- Add visualizations or analysis tools

---

## 📄 Citation

If you use this code in your research, please cite:

```bibtex
@article{tesfay2025qbound,
  title={QBound: Environment-Aware Q-Value Bounds for Stable Temporal Difference Learning},
  author={Tesfay},
  journal={arXiv preprint},
  year={2025}
}
```

---

## 📚 References

QBound builds upon foundational work in reinforcement learning. Key references include:

### Foundational RL

- **Sutton & Barto (2018)** - [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html) - The definitive textbook on RL theory and algorithms
- **Watkins & Dayan (1992)** - [Q-learning](https://link.springer.com/article/10.1007/BF00992698) - Original Q-learning algorithm with convergence proofs
- **Bellman (1957)** - [A Markovian Decision Process](https://www.jstor.org/stable/24900506) - Foundation of dynamic programming and the Bellman equation

### Deep Q-Learning

- **Mnih et al. (2015)** - [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236) - DQN: Deep Q-Networks with experience replay and target networks
- **Van Hasselt et al. (2016)** - [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461) - Double DQN addressing overestimation bias
- **Wang et al. (2016)** - [Dueling Network Architectures for Deep RL](https://arxiv.org/abs/1511.06581) - Dueling DQN with separate value and advantage streams
- **Hessel et al. (2018)** - [Rainbow: Combining Improvements in Deep RL](https://arxiv.org/abs/1710.02298) - Combining multiple DQN improvements

### Actor-Critic Methods

- **Lillicrap et al. (2015)** - [Continuous Control with Deep RL (DDPG)](https://arxiv.org/abs/1509.02971) - Deep deterministic policy gradient for continuous control
- **Fujimoto et al. (2018)** - [Addressing Function Approximation Error (TD3)](https://arxiv.org/abs/1802.09477) - Twin Delayed DDPG with clipped double-Q
- **Haarnoja et al. (2018)** - [Soft Actor-Critic](https://arxiv.org/abs/1801.01290) - Maximum entropy RL for continuous control
- **Schulman et al. (2017)** - [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347) - PPO: stable policy gradient with clipped objectives

### Overestimation and Stability

- **Thrun & Schwartz (1993)** - [Issues in Using Function Approximation for RL](https://www.ri.cmu.edu/pub_files/pub1/thrun_sebastian_1993_1/thrun_sebastian_1993_1.pdf) - First identification of overestimation bias
- **Tsitsiklis & Van Roy (1997)** - [Analysis of TD Learning with Function Approximation](https://ieeexplore.ieee.org/document/580874) - Theoretical analysis of TD convergence
- **Kumar et al. (2020)** - [Conservative Q-Learning for Offline RL](https://arxiv.org/abs/2006.04779) - CQL: pessimistic value bounds for offline RL

### Related Recent Work

- **Liu et al. (2024)** - [Boosting Soft Q-Learning by Bounding](https://arxiv.org/abs/2406.18033) - Soft Q-learning with value bounds
- **Adamczyk et al. (2023)** - [Bounding Optimal Value in Compositional RL](https://arxiv.org/abs/2303.02557) - Value bounds in compositional settings
- **Wang et al. (2024)** - [Adaptive Pessimism via Target Q-value](https://www.sciencedirect.com/journal/neural-networks) - Adaptive bounds for offline RL

### Tools and Benchmarks

- **Brockman et al. (2016)** - [OpenAI Gym](https://arxiv.org/abs/1606.01540) - Standard RL benchmark environments
- **Raffin et al. (2021)** - [Stable-Baselines3](https://jmlr.org/papers/v22/20-1364.html) - Reliable RL implementations

---

## 📧 Contact

**Author:** Tesfay
**Email:** tzemuy13@gmail.com

For questions or issues:
- Open an [Issue](https://github.com/tzemuy13/QBound/issues)

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

This research was conducted by Tesfay with [Claude](https://claude.ai) (Anthropic) serving as an AI coding and research assistant. Claude assisted with code implementation, experimental design, data analysis, and manuscript preparation. All research direction, core ideas, and final decisions were made by the author.

**Tools and Resources:**
- Built with [PyTorch](https://pytorch.org/)
- Environments from [Gymnasium](https://gymnasium.farama.org/) (formerly OpenAI Gym)
- Theoretical foundations from [Sutton & Barto (2018)](http://incompleteideas.net/book/the-book-2nd.html)
- Inspired by constrained optimization methods from [Boyd & Vandenberghe (2004)](https://web.stanford.edu/~boyd/cvxbook/)
