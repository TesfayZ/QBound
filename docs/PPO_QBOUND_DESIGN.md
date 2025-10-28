# PPO + QBound Experimental Design

## Research Question

**Can QBound stabilize value function learning in PPO across discrete and continuous action spaces?**

Unlike DDPG/TD3 where QBound failed due to disrupting Q(s,a) gradients in continuous action spaces, PPO uses V(s) critics (state-value, not action-value). This fundamental difference means QBound might work even for continuous actions.

## Key Insight: Why PPO is Different from DDPG/TD3

### DDPG/TD3 (Failed with QBound)
```
Critic: Q(s, a) - action-value function
Policy gradient: ∇θ J = E[∇a Q(s,a) · ∇θ π(a|s)]
Problem: Hard clipping Q(s,a) disrupts smooth gradients w.r.t. actions
Result: 893% performance degradation on Pendulum
```

### PPO (Should Work with QBound)
```
Critic: V(s) - state-value function
Policy gradient: ∇θ J = E[A(s,a) · ∇θ log π(a|s)]
Advantage: A(s,a) = r + γV(s') - V(s)
Key difference: Policy gradient doesn't depend on ∇a V(s)
Hypothesis: Bounding V(s) stabilizes advantages without disrupting policy learning
```

## Experimental Matrix (2x2 Design)

| | **Discrete Actions** | **Continuous Actions** |
|---|---|---|
| **Sparse Rewards** | LunarLander-v3 (8D state, 4 actions) | LunarLanderContinuous-v3 (8D state, 2D actions) |
| **Dense Rewards** | CartPole-v1 (4D state, 2 actions) | Pendulum-v1 (3D state, 1D action) |

### Additional Environments (Extended Evaluation)

**Sparse + Discrete:**
- Acrobot-v1 (6D state, 3 actions, r=-1 per step)
- MountainCar-v0 (2D state, 3 actions, r=-1 per step)

**Sparse + Continuous:**
- BipedalWalker-v3 (24D state, 4D actions, sparse landing rewards)
- MountainCarContinuous-v0 (2D state, 1D action, r=-1 per step)

**Dense + Continuous:**
- HalfCheetah-v4 (Mujoco, forward speed reward)
- Ant-v4 (Mujoco, forward speed reward)

## Implementation Design

### 1. Base PPO Agent (Baseline)

```python
class PPOAgent:
    """
    Standard PPO implementation with GAE.

    Components:
    - Actor: π(a|s) - policy network
    - Critic: V(s) - value network

    Loss functions:
    - Actor loss: -E[min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)]
    - Critic loss: MSE(V(s), G_t)  where G_t = r + γV(s')
    """

    def __init__(self, state_dim, action_dim, continuous_action):
        self.actor = ActorNetwork(state_dim, action_dim, continuous_action)
        self.critic = CriticNetwork(state_dim)  # V(s)
        self.optimizer_actor = Adam(self.actor.parameters())
        self.optimizer_critic = Adam(self.critic.parameters())

        # PPO hyperparameters
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_epsilon = 0.2
        self.ppo_epochs = 10
        self.minibatch_size = 64

    def compute_gae(self, rewards, values, next_values, dones):
        """Generalized Advantage Estimation."""
        advantages = []
        gae = 0

        for t in reversed(range(len(rewards))):
            if dones[t]:
                delta = rewards[t] - values[t]
                gae = delta
            else:
                delta = rewards[t] + self.gamma * next_values[t] - values[t]
                gae = delta + self.gamma * self.gae_lambda * gae

            advantages.insert(0, gae)

        returns = [adv + val for adv, val in zip(advantages, values)]
        return advantages, returns

    def update(self, states, actions, old_log_probs, returns, advantages):
        """Standard PPO update."""
        for _ in range(self.ppo_epochs):
            # Actor update (policy)
            new_log_probs, entropy = self.actor.evaluate(states, actions)
            ratio = torch.exp(new_log_probs - old_log_probs)

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1-self.clip_epsilon, 1+self.clip_epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy.mean()

            # Critic update (value function)
            values = self.critic(states)
            critic_loss = F.mse_loss(values, returns)

            # Optimize
            self.optimizer_actor.zero_grad()
            actor_loss.backward()
            self.optimizer_actor.step()

            self.optimizer_critic.zero_grad()
            critic_loss.backward()
            self.optimizer_critic.step()
```

### 2. PPO + QBound Agent

```python
class PPOQBoundAgent(PPOAgent):
    """
    PPO with QBound applied to the critic V(s).

    Key modifications:
    1. Bound V(s) predictions: V_min ≤ V(s) ≤ V_max
    2. Bound bootstrapped targets: V_min ≤ r + γV(s') ≤ V_max
    3. Use bounded values for advantage computation

    Hypothesis: Bounding V(s) stabilizes advantage estimation,
    improving sample efficiency without disrupting policy gradients.
    """

    def __init__(self, state_dim, action_dim, continuous_action,
                 V_min, V_max, use_step_aware_bounds=False,
                 max_episode_steps=None, step_reward=None):
        super().__init__(state_dim, action_dim, continuous_action)

        # QBound parameters
        self.V_min = V_min
        self.V_max = V_max
        self.use_step_aware_bounds = use_step_aware_bounds
        self.max_episode_steps = max_episode_steps
        self.step_reward = step_reward

    def compute_bounds(self, current_step=None):
        """
        Compute V(s) bounds.

        Static bounds (sparse rewards):
            V_min, V_max = known reward bounds with discount

        Dynamic bounds (dense rewards):
            V_max(t) = (H - t) * step_reward
            Adapts to remaining episode potential
        """
        if self.use_step_aware_bounds and current_step is not None:
            # Dynamic bound for dense reward survival tasks
            remaining_steps = self.max_episode_steps - current_step
            V_max_dynamic = remaining_steps * self.step_reward
            return self.V_min, V_max_dynamic
        else:
            # Static bounds for sparse reward tasks
            return self.V_min, self.V_max

    def compute_gae_with_bounds(self, rewards, values, next_values, dones, steps=None):
        """
        GAE with bounded value estimates.

        Critical: Apply bounds to next_values during bootstrapping,
        preventing value overestimation from propagating through advantages.
        """
        advantages = []
        gae = 0

        for t in reversed(range(len(rewards))):
            # Apply bounds to next value during bootstrapping
            if steps is not None:
                V_min, V_max = self.compute_bounds(steps[t])
            else:
                V_min, V_max = self.V_min, self.V_max

            next_value_bounded = torch.clamp(next_values[t], V_min, V_max)

            if dones[t]:
                delta = rewards[t] - values[t]
                gae = delta
            else:
                # Use bounded next value in TD error
                delta = rewards[t] + self.gamma * next_value_bounded - values[t]
                gae = delta + self.gamma * self.gae_lambda * gae

            advantages.insert(0, gae)

        # Compute returns with bounded advantages
        returns = [adv + val for adv, val in zip(advantages, values)]

        # Apply bounds to returns as well
        if steps is not None:
            returns = [torch.clamp(ret, *self.compute_bounds(step))
                      for ret, step in zip(returns, steps)]
        else:
            returns = [torch.clamp(ret, self.V_min, self.V_max) for ret in returns]

        return advantages, returns

    def update(self, states, actions, old_log_probs, rewards, dones, steps=None):
        """PPO update with bounded value targets."""
        # Get value predictions
        values = self.critic(states).detach()
        next_values = self.critic(states[1:]).detach()  # Shifted by 1

        # Compute advantages and returns with bounds
        advantages, returns = self.compute_gae_with_bounds(
            rewards, values, next_values, dones, steps
        )

        # Convert to tensors
        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = torch.tensor(returns, dtype=torch.float32)

        # Normalize advantages (standard PPO trick)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Standard PPO update with bounded returns
        for _ in range(self.ppo_epochs):
            # Actor update (unchanged - policy gradients not affected by bounds)
            new_log_probs, entropy = self.actor.evaluate(states, actions)
            ratio = torch.exp(new_log_probs - old_log_probs)

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1-self.clip_epsilon, 1+self.clip_epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy.mean()

            # Critic update with bounded targets
            value_pred = self.critic(states)

            # Apply bounds to predictions as well (soft enforcement)
            if steps is not None:
                # Dynamic bounds per step
                value_pred_bounded = torch.stack([
                    torch.clamp(v, *self.compute_bounds(step))
                    for v, step in zip(value_pred, steps)
                ])
            else:
                # Static bounds
                value_pred_bounded = torch.clamp(value_pred, self.V_min, self.V_max)

            critic_loss = F.mse_loss(value_pred_bounded, returns)

            # Optimize
            self.optimizer_actor.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.optimizer_actor.step()

            self.optimizer_critic.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.optimizer_critic.step()
```

## Experimental Protocol

### Training Configuration

**Common Hyperparameters (all environments):**
```python
# PPO parameters
gamma = 0.99
gae_lambda = 0.95
clip_epsilon = 0.2
ppo_epochs = 10
learning_rate_actor = 3e-4
learning_rate_critic = 1e-3
batch_size = 2048  # trajectory length
minibatch_size = 64

# Training
num_episodes = 1000  # or until convergence
eval_frequency = 50  # episodes
eval_episodes = 100  # final evaluation
```

### Environment-Specific Bounds

#### Discrete Action Spaces

**1. LunarLander-v3 (Sparse Reward)**
```python
# Reward structure:
#   Crash: -100
#   Landing: +100 to +140
#   Leg contact: +10 per leg
#   Fuel: -0.3 (main), -0.03 (side)
#
# Theoretical bounds:
#   Worst case: crash immediately = -100
#   Best case: perfect landing = +200 (conservative)

V_min = -100 / (1 - 0.99)  # ≈ -10,000 (but use -100 for one-step)
V_max = 200 / (1 - 0.99)   # ≈ 20,000 (but use 200 for sparse terminal)

# Practical bounds (sparse terminal reward):
V_min = -100.0
V_max = 200.0
use_step_aware_bounds = False
```

**2. CartPole-v1 (Dense Reward)**
```python
# Reward: +1 per timestep, max 500 steps
#
# Theoretical V_max = (1 - γ^H) / (1 - γ)
#                   = (1 - 0.99^500) / (1 - 0.99)
#                   ≈ 99.34

V_min = 0.0
V_max = 100.0  # static bound
use_step_aware_bounds = True  # enable dynamic
max_episode_steps = 500
step_reward = 1.0
# Dynamic: V_max(t) = 500 - t
```

**3. Acrobot-v1 (Sparse Reward)**
```python
# Reward: -1 per step, terminal +0 on success
# Max steps: 500

V_min = -500.0
V_max = 0.0
use_step_aware_bounds = False
```

**4. MountainCar-v0 (Sparse Reward)**
```python
# Reward: -1 per step, terminal +0 on success
# Max steps: 200

V_min = -200.0
V_max = 0.0
use_step_aware_bounds = False
```

#### Continuous Action Spaces

**5. LunarLanderContinuous-v3 (Sparse Reward)**
```python
# Same reward structure as discrete version
# Actions: [main engine power, side engine power] ∈ [-1, 1]^2

V_min = -100.0
V_max = 200.0
use_step_aware_bounds = False
```

**6. Pendulum-v1 (Dense Reward)**
```python
# Reward: -(θ^2 + 0.1*θ_dot^2 + 0.001*action^2)
# Range: approximately [-16, 0] per step
# Max steps: 200
#
# Theoretical max return: 0 (if perfectly upright)
# Theoretical min return: -16 * 200 = -3200

V_min = -3200.0
V_max = 0.0
use_step_aware_bounds = True  # could try dynamic
max_episode_steps = 200
step_reward = 0.0  # target (stays at 0)
```

**7. BipedalWalker-v3 (Sparse Reward)**
```python
# Reward: +300 for walking forward, -100 for falling
# Max steps: 1600

V_min = -100.0
V_max = 300.0
use_step_aware_bounds = False
```

**8. MountainCarContinuous-v0 (Sparse Reward)**
```python
# Reward: +100 on success, -0.1 * action^2 per step
# Max steps: 999

V_min = -100.0  # conservative
V_max = 100.0
use_step_aware_bounds = False
```

## Comparison Methods

For each environment, we run a 2-way comparison:

1. **Baseline PPO** - Standard implementation
2. **PPO + QBound** - V(s) bounded during bootstrapping

### Metrics

**Training metrics (logged every episode):**
- Episode reward
- Episode length
- Critic loss
- Actor loss
- Advantage mean/std
- Value function mean/std
- Fraction of V(s) predictions violating bounds (for QBound)

**Evaluation metrics (final 100 episodes):**
- Mean reward ± std deviation
- Max reward achieved
- Success rate (environment-specific threshold)
- Sample efficiency (episodes to reach threshold)
- Learning stability (variance of learning curve)

**Success thresholds:**
- LunarLander: > 200 reward (safe landing)
- CartPole: > 475 reward (near maximum)
- Acrobot: < -100 reward (solve quickly)
- MountainCar: > -110 reward (solve quickly)
- Pendulum: > -200 reward (good control)
- BipedalWalker: > 300 reward (complete walk)

## Expected Outcomes and Hypotheses

### Hypothesis 1: QBound Helps on Sparse Rewards (Both Action Types)

**Expectation:** PPO + QBound shows significant improvements on sparse reward tasks regardless of action space.

**Reasoning:**
- Sparse rewards cause large value estimate errors early in training
- Bounding V(s) prevents overestimation from corrupting advantages
- Unlike DDPG/TD3, policy gradients don't depend on ∇_a V(s)

**Predicted results:**
- LunarLander (discrete): +50-100% improvement
- LunarLanderContinuous: +50-100% improvement
- Acrobot: +20-40% improvement
- MountainCar: +20-40% improvement
- BipedalWalker: +30-60% improvement

### Hypothesis 2: QBound Works on Continuous Actions (Unlike DDPG/TD3)

**Expectation:** PPO + QBound shows improvements even on continuous action spaces, contradicting Pendulum DDPG/TD3 results.

**Reasoning:**
- DDPG/TD3 failed because clipping Q(s,a) disrupted ∇_a Q needed for deterministic policy
- PPO uses stochastic policy with advantages A(s,a) = V(s') - V(s)
- Bounding V(s) doesn't disrupt policy gradient: ∇_θ log π(a|s)

**Predicted results:**
- Pendulum: -10% to +30% (may help or neutral, but won't catastrophically fail)
- LunarLanderContinuous: +50-100% (sparse rewards benefit)
- BipedalWalker: +30-60% (sparse rewards benefit)

### Hypothesis 3: Dense Rewards with Dynamic Bounds

**Expectation:** Dynamic step-aware bounds help on dense reward survival tasks.

**Reasoning:**
- CartPole gives +1 per step, remaining potential = H - t
- Dynamic V_max(t) = H - t prevents overestimation of future rewards
- Should improve over static bounds

**Predicted results:**
- CartPole with dynamic bounds: +20-40% vs baseline PPO
- CartPole with static bounds: +10-20% vs baseline PPO
- Dynamic > Static for dense rewards

### Hypothesis 4: No Catastrophic Failures

**Expectation:** PPO + QBound never causes catastrophic performance degradation (unlike DDPG/TD3 with QBound).

**Reasoning:**
- Bounding V(s) can't disrupt policy learning mechanism
- Worst case: bounds are too tight, slowing learning (but not breaking it)

**Predicted results:**
- No environment shows > 20% performance degradation
- All environments show -20% to +100% relative performance

## Analysis Plan

### Comparative Analysis

For each environment:

1. **Learning curves:** Plot episode rewards vs training episodes for both methods
2. **Final performance:** Compare final 100 episodes mean ± std
3. **Sample efficiency:** Episodes to reach success threshold
4. **Stability:** Variance of learning curve (smoothness)
5. **Value bound violations:** Track fraction of V(s) predictions outside bounds

### Cross-Environment Analysis

1. **Action space comparison:**
   - Does QBound help equally on discrete vs continuous actions?
   - Plot: QBound improvement vs action space type

2. **Reward structure comparison:**
   - Does QBound help more on sparse vs dense rewards?
   - Plot: QBound improvement vs reward sparsity

3. **Interaction effects:**
   - 2x2 heatmap: Action space × Reward type → QBound improvement

4. **Comparison with DQN results:**
   - LunarLander: PPO+QBound vs DQN+QBound
   - CartPole: PPO+QBound vs DQN+QBound
   - Does algorithm choice matter?

### Statistical Testing

- Paired t-tests between Baseline PPO and PPO+QBound
- Effect size (Cohen's d) for each environment
- Multiple comparison correction (Bonferroni)

## File Structure

```
experiments/ppo/
├── baseline_ppo/
│   ├── lunarlander_discrete.py
│   ├── lunarlander_continuous.py
│   ├── cartpole.py
│   ├── acrobot.py
│   ├── mountaincar.py
│   ├── pendulum.py
│   ├── bipedalwalker.py
│   └── mountaincar_continuous.py
│
├── qbound_ppo/
│   ├── lunarlander_discrete.py
│   ├── lunarlander_continuous.py
│   ├── cartpole.py
│   ├── acrobot.py
│   ├── mountaincar.py
│   ├── pendulum.py
│   ├── bipedalwalker.py
│   └── mountaincar_continuous.py
│
└── run_all_ppo_experiments.py

src/
├── ppo_agent.py          # Base PPO implementation
└── ppo_qbound_agent.py   # PPO with QBound

analysis/
├── analyze_ppo_results.py
├── plot_ppo_comparison.py
└── ppo_cross_environment_analysis.py

results/ppo/
├── lunarlander_discrete/
├── lunarlander_continuous/
├── cartpole/
├── acrobot/
├── mountaincar/
├── pendulum/
├── bipedalwalker/
├── mountaincar_continuous/
└── plots/

docs/
└── PPO_QBOUND_RESULTS.md
```

## Timeline

**Phase 1: Core Implementation (3-4 days)**
- Implement base PPO agent (1 day)
- Implement PPO + QBound agent (1 day)
- Test on CartPole and LunarLander (1 day)
- Debug and validate (1 day)

**Phase 2: Discrete Action Experiments (2-3 days)**
- LunarLander-v3 (1 day)
- CartPole-v1 (0.5 day)
- Acrobot-v1 (0.5 day)
- MountainCar-v0 (0.5 day)
- Analysis (0.5 day)

**Phase 3: Continuous Action Experiments (3-4 days)**
- LunarLanderContinuous-v3 (1 day)
- Pendulum-v1 (1 day) - CRITICAL TEST
- BipedalWalker-v3 (1 day)
- MountainCarContinuous-v0 (0.5 day)
- Analysis (0.5 day)

**Phase 4: Analysis and Documentation (2-3 days)**
- Comprehensive analysis (1 day)
- Plot generation (0.5 day)
- Documentation (1 day)
- Paper section draft (0.5 day)

**Total: 10-14 days**

## Success Criteria

**Minimum viable results:**
1. PPO + QBound shows improvement on ≥ 50% of environments
2. No catastrophic failures (> 50% degradation)
3. At least one environment shows > 50% improvement
4. Continuous action spaces don't break (unlike DDPG/TD3)

**Strong results:**
1. PPO + QBound shows improvement on ≥ 75% of environments
2. Average improvement > 30% across all environments
3. Multiple environments show > 50% improvement
4. Clear pattern: sparse rewards benefit more than dense
5. Continuous actions work as well as discrete

**Groundbreaking results:**
1. PPO + QBound shows improvement on 100% of environments
2. Average improvement > 50% across all environments
3. Continuous action spaces show similar improvements to discrete
4. Clear evidence that QBound generalizes to policy gradient methods
5. Results strong enough to warrant separate publication

## Next Steps

1. **Review and approve this design**
2. **Implement base PPO agent** (start with standard implementation)
3. **Implement PPO + QBound** (careful attention to where bounds apply)
4. **Run pilot experiments** (CartPole + LunarLander to validate)
5. **Scale to full experimental matrix**
6. **Analyze results and document findings**

Would you like me to proceed with implementation?
