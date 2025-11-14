# Dynamic QBound Selective Rerun - Quick Start Guide

**All agents fixed! Ready for selective rerun to preserve baseline and static QBound results.**

## What Was Fixed

✅ **4 Agents:** DQN, DDQN (Double DQN), DDPG, TD3
✅ **Bug:** Experience replay used single time step for entire minibatch
✅ **Fix:** Per-transition time steps, correct bounds for each transition
✅ **PPO:** Not affected (on-policy, no replay buffer)

## Quick Start: 3 Steps

### Step 1: Prepare (removes only dynamic QBound from results)

```bash
python3 prepare_selective_rerun.py
```

**Output:** Creates `*_in_progress.json` files with baseline and static QBound preserved

### Step 2: Rerun (only trains dynamic QBound methods)

```bash
# Pendulum DQN/DDQN (skips 4 methods, trains 2)
for seed in 42 43 44 45 46; do
    python3 experiments/pendulum/train_pendulum_dqn_full_qbound.py --seed $seed
done

# Pendulum DDPG (skips 2 methods, trains 1)
for seed in 42 43 44 45 46; do
    python3 experiments/pendulum/train_pendulum_ddpg_full_qbound.py --seed $seed
done

# Pendulum TD3 (skips 2 methods, trains 1)
for seed in 42 43 44 45 46; do
    python3 experiments/pendulum/train_pendulum_td3_full_qbound.py --seed $seed
done
```

### Step 3: Verify (check dynamic QBound improved)

```bash
python3 -c "
import json, glob, numpy as np

for method in ['dqn', 'ddpg', 'td3']:
    files = glob.glob(f'results/pendulum/{method}_full_qbound_seed*.json')
    files = [f for f in files if 'in_progress' not in f and files]

    if not files:
        continue

    with open(files[0], 'r') as f:
        data = json.load(f)

    baseline_key = 'dqn' if method == 'dqn' else 'baseline'
    dynamic_key = 'dynamic_qbound_dqn' if method == 'dqn' else 'dynamic_soft_qbound'

    if baseline_key in data['training'] and dynamic_key in data['training']:
        baseline = data['training'][baseline_key]['final_100_mean']
        dynamic = data['training'][dynamic_key]['final_100_mean']
        improvement = ((dynamic - baseline) / abs(baseline)) * 100
        status = '✓' if improvement > 0 else '✗'
        print(f'{method.upper():6s}: {improvement:+6.1f}% {status}')
"
```

## Time Estimate

- **DQN/DDQN:** ~2 hours per seed × 5 seeds = **10 hours**
- **DDPG:** ~0.5 hours per seed × 5 seeds = **2.5 hours**
- **TD3:** ~0.5 hours per seed × 5 seeds = **2.5 hours**
- **Total: ~15 hours** (vs 105 hours for full rerun)

## What Gets Preserved

✅ **Preserved (not retrained):**
- `dqn` (baseline)
- `static_qbound_dqn`
- `double_dqn` (baseline)
- `static_qbound_double_dqn`
- `baseline` (DDPG/TD3)
- `static_soft_qbound` (DDPG/TD3)

🔄 **Retrained with fix:**
- `dynamic_qbound_dqn`
- `dynamic_qbound_double_dqn`
- `dynamic_soft_qbound` (DDPG/TD3)

## Expected Results

| Method | Old Dynamic | Expected New |
|--------|-------------|--------------|
| DQN | -171.42 (-7.8%) | **-150 to -160** (+0-6%) |
| DDQN | -176.27 (+0.7%) | **-170 to -175** (+2-4%) |
| DDPG | -173.53 (+27.8%) | **-160 to -170** (+30-35%) |
| TD3 | -277.36 (-23.8%) | **-170 to -190** (+15-25%) |

**Key:** All dynamic QBound methods should now beat or match baseline!

## Rollback

If anything goes wrong:

```bash
BACKUP=$(ls -td results/pendulum/backup_buggy_dynamic_* | head -1)
cp $BACKUP/*.json results/pendulum/
rm results/pendulum/*_in_progress.json
```

## Documentation

- **Technical Details:** `docs/COMPLETE_FIX_SUMMARY.md`
- **Bug Analysis:** `docs/TD3_EXPERIENCE_REPLAY_BUG_ANALYSIS.md`
- **TD3 Specific:** `docs/TD3_FIX_SUMMARY.md`

---

**Ready to run!** Execute Step 1, then Step 2, then Step 3.
