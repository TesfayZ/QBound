# Architectural QBound: Activation Function Replaces Algorithmic Clipping

## The New Approach

**REPLACING algorithmic clipping with architectural bounds for negative rewards.**

### What We're Testing

**Baseline (unchanged):**
```python
# No QBound, unbounded output
Q = network(state)  # Linear output, Q ∈ (-∞, +∞)
```

**NEW Architectural Approach (replaces old QBound):**
```python
# Use activation function to enforce Q ≤ 0 naturally
logits = network(state)
Q = -F.softplus(logits)  # Q ∈ (-∞, 0]
```

**OLD Algorithmic QBound (ABANDONED for negative rewards):**
```python
# What we were doing (56.79% violations, -7% degradation)
Q_raw = network(state)
Q = torch.clamp(Q_raw, max=0)  # DON'T USE THIS ANYMORE
```

## Implementation: Negative Softplus

For Pendulum (negative rewards, Q should be ≤ 0):

```python
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128, 
                 use_negative_activation=False):
        super().__init__()
        self.use_negative_activation = use_negative_activation
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, x):
        logits = self.network(x)
        
        if self.use_negative_activation:
            # Architectural bound: Q ≤ 0
            Q = -F.softplus(logits)
        else:
            # Baseline: unbounded
            Q = logits
        
        return Q
```

**Key:** No torch.clamp anywhere! The activation function IS the bound.

## Experimental Comparison

**Two methods only:**

1. **`dqn` (baseline):** Unbounded output, no activation
2. **`architectural_qbound_dqn` (NEW):** Negative softplus activation

**What we're testing:**
- Does architectural enforcement work better than no bounds?
- Replaces failed algorithmic approach for negative rewards

## Expected Results

**Hypothesis:**
- Architectural approach should work BETTER than baseline
- Network learns correct range from start (no violations by construction)
- No fighting against clipping (smooth gradients)

| Method | Output Range | Violations | Expected Performance |
|--------|--------------|------------|---------------------|
| Baseline | (-∞, +∞) | N/A | Reference (100%) |
| Architectural | (-∞, 0] | 0% (by design) | Better (+5-10%?) |

**Why it should work:**
- Guides learning toward correct range
- No gradient conflicts
- Natural enforcement, not adversarial clipping

## Implementation Files

1. **`src/dqn_agent.py`**
   - Add `use_negative_activation` parameter to QNetwork
   - Remove torch.clamp for negative rewards

2. **`experiments/pendulum/train_pendulum_dqn_full_qbound.py`**
   - Keep `dqn` baseline (unchanged)
   - Replace `static_qbound_dqn` with `architectural_qbound_dqn`
   - Set `use_negative_activation=True` for new method

## Result Naming

Results saved as:
- `dqn` - Baseline (unbounded)
- `architectural_qbound_dqn` - New approach (negative softplus)

Clear comparison of architectural approach vs no bounds.

## For Positive Rewards (Future)

Keep current approach (positive rewards work fine):
- CartPole continues using algorithmic clipping
- Only negative rewards get architectural treatment

Focus: Fix what's broken (negative rewards) with better approach.
