# Comparison Methods Research

## Overview
This document outlines the methods to compare against QBound based on the literature review and user requirements.

## User's Specified Methods

### 1. Double DQN (DDQN) - **COMPLETED**
- **Status:** Already tested
- **Key Finding:** Catastrophically fails on dense rewards (-66% on CartPole), succeeds on sparse rewards (+15% on FrozenLake)
- **Mechanism:** Decouples action selection and evaluation using separate networks to reduce overestimation
- **Why it fails:** Algorithm-level pessimism underestimates long-horizon returns in survival tasks

### 2. Deep Deterministic Policy Gradient (DDPG)
- **Status:** TO BE TESTED
- **Reference:** Lillicrap et al. (2015) - "Continuous Control with Deep Reinforcement Learning"
- **Type:** Actor-critic method for continuous control
- **Mechanism:**
  - Deterministic policy gradient
  - Single Q-function critic
  - Learns Q(s,a) for continuous actions
- **DDQN-Like Critic:** User mentioned "DDPG where the critic is like DDQN"
  - Standard DDPG uses single Q-network
  - Could implement DDPG with Double-Q critic (uses two networks for evaluation)
  - This would test if Double-Q helps in actor-critic settings

### 3. Twin Delayed DDPG (TD3)
- **Status:** TO BE TESTED
- **Reference:** Fujimoto et al. (2018) - "Addressing Function Approximation Error in Actor-Critic Methods"
- **Type:** Improved DDPG with multiple stabilization techniques
- **Key Features:**
  - **Clipped Double-Q Learning:** Uses two critics, takes minimum (pessimistic)
  - **Delayed Policy Updates:** Updates actor less frequently than critics
  - **Target Policy Smoothing:** Adds noise to target actions
- **Why it's important:** TD3's clipped double-Q is similar to DDQN's pessimism but in continuous control
- **Hypothesis:** TD3 may exhibit similar environment-dependent failures as DDQN

## Additional Methods from Literature

### 4. Soft Actor-Critic (SAC)
- **Reference:** Haarnoja et al. (2018) - "Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL"
- **Type:** Actor-critic with entropy regularization
- **Key Features:**
  - Entropy-augmented objective encourages exploration
  - Uses two Q-networks (like TD3)
  - Maximum entropy framework
- **Why relevant:** State-of-the-art continuous control method, widely used baseline
- **Comparison value:** SAC is known for robustness - testing if QBound helps or hurts

### 5. Conservative Q-Learning (CQL)
- **Reference:** Kumar et al. (2020) - "Conservative Q-Learning for Offline Reinforcement Learning"
- **Type:** Offline RL method with learned pessimistic bounds
- **Key Features:**
  - Learns conservative Q-values to avoid distribution shift
  - Uses auxiliary loss to push Q-values down on out-of-distribution actions
- **Why relevant:** Already mentioned in paper as related work
- **Comparison angle:** CQL learns pessimism, QBound enforces environment-aware bounds
- **Note:** Primarily for offline RL, may not be fair comparison for online setting

## Methods NOT to Compare (and why)

### Policy Gradient Methods (REINFORCE, TRPO, PPO)
- **Reason:** These don't learn Q-functions/critics
- **Note:** A2C/A3C have critics but are designed for discrete control
- **Decision:** Skip pure policy gradient methods

### Rainbow DQN
- **Reason:** Combination of multiple techniques (dueling, prioritized replay, distributional, etc.)
- **Decision:** Too many confounds; focus on core algorithmic differences

### Dueling DQN / Distributional RL
- **Reason:** Architectural changes rather than algorithmic pessimism/bounds
- **Decision:** Orthogonal to QBound's contribution

## Recommended Comparison Plan

### Priority 1: Value-Based Methods (Discrete Control)
1. ✅ **Baseline DQN** (completed)
2. ✅ **Double DQN** (completed)

### Priority 2: Actor-Critic Methods (Continuous Control)
3. **DDPG** (standard)
4. **DDPG with Double-Q Critic** (DDQN-like variant requested by user)
5. **TD3** (clipped double-Q, similar pessimism to DDQN)
6. **SAC** (entropy-regularized, state-of-the-art baseline)

### Priority 3 (Optional): Offline RL
7. **CQL** (if time permits, mainly for discussion/related work)

## Environments for Comparison

### Sparse Reward Environments
- **GridWorld** (10×10, deterministic navigation) - already tested with DQN/DDQN
- **FrozenLake** (4×4, stochastic navigation) - already tested with DQN/DDQN

### Dense Reward Environments
- **CartPole** (balance task, r=+1/step) - already tested with DQN/DDQN
- **Continuous CartPole** - for DDPG/TD3/SAC testing
- **Pendulum-v1** - classic continuous control benchmark (dense rewards)
- **MountainCarContinuous-v0** - sparse terminal reward + dense negative cost

## Key Research Questions

1. **Does TD3's clipped double-Q fail like DDQN on dense rewards?**
   - Hypothesis: Yes, same pessimism mechanism

2. **Does QBound help DDPG/TD3/SAC critics?**
   - Hypothesis: Yes, stabilizing critic improves actor-critic methods

3. **Is environment-dependent failure universal for pessimistic methods?**
   - Test: DDQN (value-based) vs TD3 (actor-critic) on same tasks

4. **Can QBound replace double-Q mechanisms entirely?**
   - Test: QBound alone vs Double-Q alone vs QBound+Double-Q

## Expected Outcomes

### If QBound succeeds across all comparisons:
- **Title change:** "QBound: Replacing DDQN and TD3 in RL Simply" (as user suggested)
- **Main claim:** Environment-aware bounds > algorithm-level pessimism
- **Impact:** QBound becomes default stabilization technique

### If mixed results:
- **Sparse rewards:** QBound best, DDQN/TD3 moderate, baseline worst
- **Dense rewards:** QBound best, baseline moderate, DDQN/TD3 fail
- **Claim:** QBound is universal, pessimistic methods are environment-specific

## Implementation Notes

### DDPG Implementation
- Use standard DDPG from Spinning Up or Stable-Baselines3
- Implement DDPG variant with Double-Q critic:
  - Two critic networks Q1, Q2
  - For target: use min(Q1_target, Q2_target) like TD3
  - Or use decoupled action selection (Q1 selects, Q2 evaluates) like DDQN

### TD3 Implementation
- Use standard TD3 from Spinning Up or Stable-Baselines3
- Already has clipped double-Q built in
- Test with and without QBound on critics

### SAC Implementation
- Use standard SAC from Stable-Baselines3
- Test with and without QBound on critics
- May need to adjust entropy coefficient

### Continuous Environments
- CartPole continuous version (if available) or Pendulum
- MountainCarContinuous for sparse reward + dense cost structure
- Reacher-v2 or HalfCheetah-v2 for more complex control (if time)

## Timeline Estimate

1. **DDPG implementation and testing:** 2-3 days
2. **TD3 implementation and testing:** 1-2 days (simpler since similar to DDPG)
3. **SAC implementation and testing:** 1-2 days
4. **Analysis and paper update:** 1-2 days
5. **Total:** ~1 week for comprehensive comparison

## References to Add to Paper

- Lillicrap et al. (2015) - DDPG
- Fujimoto et al. (2018) - TD3
- Haarnoja et al. (2018) - SAC
- Kumar et al. (2020) - CQL (already cited)
