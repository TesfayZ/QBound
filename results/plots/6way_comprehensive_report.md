# QBound: Comprehensive 6-Way Comparison Results

**Report Generated**: 2025-10-29 12:52:05

---

## Gridworld

### Configuration

- **Environment**: GridWorld
- **Episodes**: 500
- **Max Steps**: 100
- **Discount Factor (γ)**: 0.99
- **QBound Range**: [0.00, 1.00]

### Training Performance

| Rank | Method | Total Reward | Avg Reward |
|------|--------|--------------|------------|
| 1 🥇 | 6. Dynamic QBound + DDQN | 482 | 0.96 |
| 2 🥈 | 4. Baseline DDQN | 476 | 0.95 |
| 3 🥉 | 5. Static QBound + DDQN | 474 | 0.95 |
| 4  | 2. Static QBound + DQN | 350 | 0.70 |
| 5  | 1. Baseline DQN | 257 | 0.51 |
| 6  | 3. Dynamic QBound + DQN | 2 | 0.00 |

---

## Frozenlake

### Configuration

- **Environment**: FrozenLake-v1
- **Episodes**: 2000
- **Max Steps**: 100
- **Discount Factor (γ)**: 0.95
- **QBound Range**: [0.00, 1.00]

### Training Performance

| Rank | Method | Total Reward | Avg Reward |
|------|--------|--------------|------------|
| 1 🥇 | 4. Baseline DDQN | 1065 | 0.53 |
| 2 🥈 | 2. Static QBound + DQN | 982 | 0.49 |
| 3 🥉 | 1. Baseline DQN | 917 | 0.46 |
| 4  | 6. Dynamic QBound + DDQN | 860 | 0.43 |
| 5  | 5. Static QBound + DDQN | 854 | 0.43 |
| 6  | 3. Dynamic QBound + DQN | 710 | 0.35 |

---

## Cartpole

### Configuration

- **Environment**: CartPole-v1
- **Episodes**: 500
- **Max Steps**: 500
- **Discount Factor (γ)**: 0.99
- **QBound Range**: [0.00, 99.34]

### Training Performance

| Rank | Method | Total Reward | Avg Reward |
|------|--------|--------------|------------|
| 1 🥇 | 1. Baseline DQN | 183022 | 366.04 |
| 2 🥈 | 2. Static QBound + DQN | 167840 | 335.68 |
| 3 🥉 | 5. Static QBound + DDQN | 119819 | 239.64 |
| 4  | 3. Dynamic QBound + DQN | 106027 | 212.05 |
| 5  | 6. Dynamic QBound + DDQN | 82557 | 165.11 |
| 6  | 4. Baseline DDQN | 43402 | 86.80 |

---

## Lunarlander

### Configuration

- **Environment**: LunarLander-v3
- **Episodes**: 500
- **Max Steps**: 1000
- **Discount Factor (γ)**: 0.99
- **QBound Range**: [-100.00, 200.00]

### Training Performance

| Rank | Method | Total Reward | Avg Reward |
|------|--------|--------------|------------|
| 1 🥇 | 3. Dynamic QBound + DQN | 82158 | 164.32 |
| 2 🥈 | 6. Dynamic QBound + DDQN | 61684 | 123.37 |
| 3 🥉 | 4. Baseline DDQN | 48069 | 96.14 |
| 4  | 5. Static QBound + DDQN | 33626 | 67.25 |
| 5  | 2. Static QBound + DQN | 31236 | 62.47 |
| 6  | 1. Baseline DQN | -38946 | -77.89 |

---

## Summary Across All Environments

- **Gridworld**: 6. Dynamic QBound + DDQN (482)
- **Frozenlake**: 4. Baseline DDQN (1065)
- **Cartpole**: 1. Baseline DQN (183022)
- **Lunarlander**: 3. Dynamic QBound + DQN (82158)
