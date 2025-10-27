# Blind Evaluation Methodology

## Purpose

The blind evaluation tests whether models can generalize beyond their training assumptions, particularly for **dynamic QBound** which uses step-aware bounds during training.

## The Problem

Dynamic QBound is trained with knowledge of the maximum episode length:

```python
# During training
Q_max(t) = (max_episode_steps - current_step) * reward_per_step

# For CartPole with max_episode_steps=500 and reward=1:
# At step 0: Q_max = 500
# At step 250: Q_max = 250
# At step 499: Q_max = 1
```

**Key Question**: What happens when the model is deployed in an environment where:
1. The episode length is unknown?
2. The episode length is different from training?

## Blind Evaluation Protocol

### Definition

"Blind" means the model operates **without step-aware information** during evaluation:
- No `current_step` is passed to the agent
- The agent cannot compute dynamic Q-bounds
- The agent must rely solely on learned Q-values

### Implementation

```python
def evaluate_agent_blind(env, agent, num_eval_episodes=100, max_steps=500):
    """
    Evaluate agent without step-aware information.

    Key: Agent's select_action() is called in eval_mode=True,
    which means no exploration and no access to current_step.
    """
    total_rewards = []

    for _ in range(num_eval_episodes):
        state = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            # Agent selects action WITHOUT knowing current step
            action = agent.select_action(state, eval_mode=True)
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            state = next_state

            if done:
                break

        total_rewards.append(episode_reward)

    return np.mean(total_rewards), np.std(total_rewards)
```

### Test Scenarios

We evaluate on two scenarios:

#### Scenario 1: Same Episode Length (max_steps=500)

Tests whether models can perform well when deployed "as-is":

- **Baseline**: Should perform well (never saw step info anyway)
- **Static QBound**: Should perform well (bounds are constant)
- **Dynamic QBound**: Critical test - can it perform without step info?

#### Scenario 2: Different Episode Length (max_steps=1000)

Tests generalization to longer episodes:

- **Baseline**: May perform better or worse depending on learning
- **Static QBound**: May hit Q_max=100 ceiling earlier
- **Dynamic QBound**: Was trained assuming 500 max steps - what happens at step 501+?

## Expected Outcomes

### Best Case (Dynamic QBound Generalizes)

Dynamic QBound learns a good policy during training and doesn't actually need step information at test time:

```
Scenario 1 (500 steps):
  Baseline:       200 ± 50
  Static QBound:  250 ± 40
  Dynamic QBound: 300 ± 30  ← Performs well even without step info

Scenario 2 (1000 steps):
  Baseline:       300 ± 60
  Static QBound:  350 ± 50
  Dynamic QBound: 400 ± 40  ← Generalizes to longer episodes
```

### Worst Case (Dynamic QBound Fails)

Dynamic QBound overfits to step-aware bounds and fails without them:

```
Scenario 1 (500 steps):
  Baseline:       200 ± 50
  Static QBound:  250 ± 40
  Dynamic QBound: 150 ± 60  ← Worse than baseline!

Scenario 2 (1000 steps):
  Baseline:       300 ± 60
  Static QBound:  350 ± 50
  Dynamic QBound: 100 ± 70  ← Catastrophic failure
```

### Most Likely (Partial Generalization)

Dynamic QBound performs well on training length but struggles on different lengths:

```
Scenario 1 (500 steps):
  Baseline:       200 ± 50
  Static QBound:  250 ± 40
  Dynamic QBound: 280 ± 35  ← Good, but slightly worse than with step info

Scenario 2 (1000 steps):
  Baseline:       300 ± 60
  Static QBound:  350 ± 50
  Dynamic QBound: 250 ± 55  ← Degrades on longer episodes
```

## Why This Matters

### For Practical Deployment

Real-world RL systems often don't know episode lengths in advance:
- **Robot tasks**: How long will the task take?
- **Game playing**: Episodes vary in length
- **Control problems**: Disturbances can extend episodes

If dynamic QBound requires step-aware information at test time, it's less practical.

### For Understanding QBound

Blind evaluation reveals whether QBound:
1. **Shapes the learning process** (good - helps during training)
2. **Becomes a crutch** (bad - model depends on it at test time)

Ideally, QBound should be like training wheels:
- Helpful during learning
- Not needed once the skill is acquired

## Implementation in Experiments

### CartPole 3-Way Comparison

The `experiments/cartpole/train_cartpole_3way.py` script:

1. **Trains** all three models (baseline, static, dynamic)
2. **Saves** the trained models to `models/cartpole/`
3. **Evaluates blindly** at max_steps=500 and max_steps=1000
4. **Records** all results in JSON

Models are saved so they can be:
- Re-evaluated later with different protocols
- Shared with other researchers
- Tested on completely new environments

### Saved Models

```
models/cartpole/
├── baseline_<timestamp>.pt
├── static_qbound_<timestamp>.pt
└── dynamic_qbound_<timestamp>.pt
```

Each model includes:
- Q-network weights
- Target network weights
- Optimizer state

## Future Experiments

### Transfer Learning

Test whether models trained on CartPole can transfer to:
- CartPole with different physics (longer pole, heavier cart)
- CartPole with different reward structures
- Similar balancing tasks (Acrobot, Pendulum)

### Adaptive QBound

Develop QBound that:
- Doesn't require knowing max_episode_steps in advance
- Adapts bounds based on observed episode lengths
- Uses uncertainty estimates instead of step counts

### Meta-Learning

Train dynamic QBound on episodes of varying lengths:
- Randomly sample max_steps from [100, 500, 1000]
- Force the model to generalize across episode lengths
- Test if this improves blind evaluation performance

## Summary

Blind evaluation is critical for understanding whether dynamic QBound:
1. ✅ Helps learning but isn't needed at test time (ideal)
2. ⚠️ Helps learning but degrades performance without step info
3. ❌ Creates a dependency that makes deployment impractical

The results will inform whether dynamic QBound is:
- Ready for real-world deployment (case 1)
- Needs architectural improvements (case 2)
- Should be limited to environments with known episode lengths (case 3)
