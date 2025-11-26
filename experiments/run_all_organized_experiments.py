#!/usr/bin/env python3
"""
Master Script: Run All Organized Experiments

This script runs all experiments with baseline and static QBound only:
- Time-step dependent environments (Baseline + Static QBound)
- Sparse/State-dependent environments (Baseline + Static QBound)
- Transformed Q-value experiments (Negative → Positive, tests QBound on shifted range)

Organization Philosophy:
======================
Time-step dependent: Rewards accumulate predictably with time steps
  → Tests baseline vs static QBound

Sparse/State-dependent: Rewards depend on state transitions, not time
  → Tests baseline vs static QBound

Transformed Q-values: Negative reward environments shifted to positive range
  → Tests if QBound's failure is due to negative value range

Total Scripts: 16 experiments
- 8 time-step dependent (4 DQN scripts with 4 methods each + 4 continuous with 2 methods each)
- 4 sparse/state-dependent (4 methods each: baseline + static × DQN/DDQN)
- 3 transformed Q-value (2 methods each: baseline + QBound with Q-value transformation)

Usage:
------
# Run ALL experiments with default 5 seeds (42, 43, 44, 45, 46) - RECOMMENDED
python3 experiments/run_all_organized_experiments.py

# The script will:
# - Run all categories (timestep + sparse + transformed)
# - Use 5 seeds for statistical significance
# - Auto-confirm (no prompt)
# - Auto-skip completed experiments (crash recovery)

# Run with custom seeds
python3 experiments/run_all_organized_experiments.py --seeds 1 2 3

# Run with single seed
python3 experiments/run_all_organized_experiments.py --seed 42

# Run only time-step dependent experiments
python3 experiments/run_all_organized_experiments.py --category timestep

# Run only sparse/state-dependent experiments
python3 experiments/run_all_organized_experiments.py --category sparse

# Run only transformed Q-value experiments (NEW!)
python3 experiments/run_all_organized_experiments.py --category transformed

# Dry run (list what would be executed)
python3 experiments/run_all_organized_experiments.py --dry-run

# Require manual confirmation (override auto-confirm)
python3 experiments/run_all_organized_experiments.py --no-auto-confirm

# Crash Recovery:
# Simply re-run the same command if interrupted - the script will:
# - Check result files for each (experiment, seed) pair
# - Skip already-completed experiments automatically
# - Continue from where it left off
"""

import subprocess
import sys
import os
import argparse
import json
import glob
from datetime import datetime
from pathlib import Path

# Experiment definitions
TIME_STEP_DEPENDENT_EXPERIMENTS = [
    {
        'name': 'CartPole DQN QBound',
        'script': 'experiments/cartpole/train_cartpole_dqn_full_qbound.py',
        'methods': 4,
        'est_time_min': 30,
        'description': 'Dense positive reward (+1/step), DQN/DDQN with baseline + static QBound'
    },
    {
        'name': 'CartPole Dueling DQN QBound',
        'script': 'experiments/cartpole/train_cartpole_dueling_full_qbound.py',
        'methods': 4,
        'est_time_min': 30,
        'description': 'Dense positive reward (+1/step), Dueling DQN with baseline + static QBound'
    },
    {
        'name': 'Pendulum DDPG QBound',
        'script': 'experiments/pendulum/train_pendulum_ddpg_full_qbound.py',
        'methods': 2,
        'est_time_min': 90,
        'description': 'Continuous control with DDPG (baseline + static softplus_clip) - negative rewards'
    },
    {
        'name': 'Pendulum TD3 QBound',
        'script': 'experiments/pendulum/train_pendulum_td3_full_qbound.py',
        'methods': 2,
        'est_time_min': 90,
        'description': 'Continuous control with TD3 (baseline + static softplus_clip) - negative rewards'
    },
    {
        'name': 'Acrobot DQN QBound',
        'script': 'experiments/acrobot/train_acrobot_dqn_full_qbound.py',
        'methods': 4,
        'est_time_min': 45,
        'description': 'Dense negative reward (-1/step), DQN/DDQN with baseline + static QBound'
    },
    {
        'name': 'MountainCar DQN QBound',
        'script': 'experiments/mountaincar/train_mountaincar_dqn_full_qbound.py',
        'methods': 4,
        'est_time_min': 60,
        'description': 'Dense negative reward (-1/step), DQN/DDQN with baseline + static QBound'
    },
    {
        'name': 'MountainCarContinuous DDPG QBound',
        'script': 'experiments/mountaincar_continuous/train_mountaincar_continuous_ddpg_full_qbound.py',
        'methods': 2,
        'est_time_min': 90,
        'description': 'Continuous control with DDPG (baseline + static softplus_clip) - negative rewards'
    },
    {
        'name': 'MountainCarContinuous TD3 QBound',
        'script': 'experiments/mountaincar_continuous/train_mountaincar_continuous_td3_full_qbound.py',
        'methods': 2,
        'est_time_min': 90,
        'description': 'Continuous control with TD3 (baseline + static softplus_clip) - negative rewards'
    }
]

SPARSE_STATE_DEPENDENT_EXPERIMENTS = [
    {
        'name': 'GridWorld DQN Static QBound',
        'script': 'experiments/gridworld/train_gridworld_dqn_static_qbound.py',
        'methods': 4,
        'est_time_min': 15,
        'description': 'Sparse terminal reward (+1 at goal only)'
    },
    {
        'name': 'FrozenLake DQN Static QBound',
        'script': 'experiments/frozenlake/train_frozenlake_dqn_static_qbound.py',
        'methods': 4,
        'est_time_min': 20,
        'description': 'Stochastic environment, sparse terminal reward'
    },
    {
        'name': 'MountainCar DQN Static QBound',
        'script': 'experiments/mountaincar/train_mountaincar_dqn_static_qbound.py',
        'methods': 4,
        'est_time_min': 60,
        'description': 'State-dependent reward (-1 until goal reached)'
    },
    {
        'name': 'Acrobot DQN Static QBound',
        'script': 'experiments/acrobot/train_acrobot_dqn_static_qbound.py',
        'methods': 4,
        'est_time_min': 45,
        'description': 'State-dependent reward (-1 until swing-up goal)'
    }
]

# NEW: Transformed Q-Value Experiments (Negative → Positive)
# Tests whether transforming to positive Q-value range improves QBound performance
TRANSFORMED_QVALUE_EXPERIMENTS = [
    {
        'name': 'MountainCar DQN Transformed (Negative→Positive)',
        'script': 'experiments/mountaincar/train_mountaincar_dqn_transformed.py',
        'methods': 2,
        'est_time_min': 60,
        'description': 'Transform Q ∈ [-86.6, 0] → Q ∈ [0, 86.6] to test if positive range fixes QBound'
    },
    {
        'name': 'Acrobot DQN Transformed (Negative→Positive)',
        'script': 'experiments/acrobot/train_acrobot_dqn_transformed.py',
        'methods': 2,
        'est_time_min': 45,
        'description': 'Transform Q ∈ [-99.3, 0] → Q ∈ [0, 99.3] to test if positive range fixes QBound'
    },
    {
        'name': 'Pendulum DDPG Transformed (Negative→Positive)',
        'script': 'experiments/pendulum/train_pendulum_ddpg_transformed.py',
        'methods': 2,
        'est_time_min': 90,
        'description': 'Transform Q ∈ [-1409, 0] → Q ∈ [0, 1409] to test if positive range improves QBound further'
    }
]


def load_crash_recovery_state(log_file):
    """Load crash recovery state for progress tracking"""
    if os.path.exists(log_file):
        try:
            with open(log_file, 'r') as f:
                state = json.load(f)
            print(f"\n🔄 CRASH RECOVERY: Found existing progress log")
            print(f"   📊 Previous session data loaded")
            return state
        except Exception as e:
            print(f"\n⚠️  Warning: Could not load crash recovery state: {e}")
            return None
    return None


def save_crash_recovery_state(log_file, state):
    """Save crash recovery state periodically"""
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    with open(log_file, 'w') as f:
        json.dump(state, f, indent=2)
    print(f"   💾 Progress saved to: {log_file}")


def print_experiment_summary(experiments, category_name):
    """Print summary table of experiments"""
    print(f"\n{'='*80}")
    print(f"{category_name} EXPERIMENTS")
    print(f"{'='*80}")

    total_methods = sum(exp['methods'] for exp in experiments)
    total_time = sum(exp['est_time_min'] for exp in experiments)

    print(f"\nTotal: {len(experiments)} experiments, {total_methods} total methods")
    print(f"Estimated time: {total_time} minutes ({total_time/60:.1f} hours)\n")

    print(f"{'#':<4} {'Experiment':<40} {'Methods':<10} {'Time (min)':<12}")
    print("-" * 80)

    for i, exp in enumerate(experiments, 1):
        print(f"{i:<4} {exp['name']:<40} {exp['methods']:<10} {exp['est_time_min']:<12}")

    print(f"\n{'='*80}\n")


def get_experiment_result_pattern(script_path, seed):
    """
    Determine the expected result file pattern for an experiment.
    Returns tuple of (result_dir, file_pattern) to check for existing results.
    """
    # Extract environment and experiment type from script path
    # e.g., "experiments/cartpole/train_cartpole_dqn_full_qbound.py"
    #    -> results/cartpole/dqn_full_qbound_seed42_*.json
    # e.g., "experiments/ppo/train_pendulum_ppo_full_qbound.py"
    #    -> results/pendulum/ppo_full_qbound_seed42_*.json

    script_name = Path(script_path).stem  # e.g., "train_cartpole_dqn_full_qbound"

    # Remove "train_" prefix
    if script_name.startswith('train_'):
        script_name = script_name[6:]  # e.g., "cartpole_dqn_full_qbound"

    # Extract environment (first part before algorithm)
    parts = script_name.split('_')
    env = parts[0]  # e.g., "cartpole" or "pendulum"

    # Algorithm and variant (everything after environment)
    algo_variant = '_'.join(parts[1:])  # e.g., "dqn_full_qbound" or "ppo_full_qbound"

    # Check both timestamped and seed-specific result files
    result_dir = f"results/{env}"
    patterns = [
        f"{algo_variant}_seed{seed}_*.json",  # e.g., dqn_full_qbound_seed42_20251029.json
        f"{algo_variant}_seed{seed}.json",    # Alternative pattern
    ]

    return result_dir, patterns


def is_experiment_completed(script_path, seed):
    """
    Check if an experiment has already completed by looking for result files.
    Returns True if a completed result file exists, False otherwise.
    """
    result_dir, patterns = get_experiment_result_pattern(script_path, seed)

    if not os.path.exists(result_dir):
        return False

    # Check for matching result files
    for pattern in patterns:
        matches = glob.glob(f"{result_dir}/{pattern}")
        for match in matches:
            # Skip in_progress files
            if '_in_progress.json' in match:
                continue
            # Found a completed result file
            return True

    return False


def run_experiment(exp_config, seed, dry_run=False, force=False):
    """Run a single experiment"""
    script_path = exp_config['script']
    exp_name = exp_config['name']

    # Check if already completed (unless force flag is set)
    if not force and not dry_run:
        if is_experiment_completed(script_path, seed):
            print(f"\n{'='*80}")
            print(f"SKIPPING (Already Completed): {exp_name}")
            print(f"Script: {script_path}")
            print(f"Seed: {seed}")
            print(f"{'='*80}\n")
            return {'status': 'skipped', 'reason': 'already_completed'}

    print(f"\n{'='*80}")
    print(f"Running: {exp_name}")
    print(f"Script: {script_path}")
    print(f"Seed: {seed}")
    print(f"Methods: {exp_config['methods']}")
    print(f"Estimated time: {exp_config['est_time_min']} minutes")
    print(f"{'='*80}\n")

    if dry_run:
        print(f"[DRY RUN] Would execute: python3 {script_path} --seed {seed}")
        return {'status': 'skipped', 'dry_run': True}

    # Execute the script
    cmd = [sys.executable, script_path, '--seed', str(seed)]
    start_time = datetime.now()

    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds() / 60

        print(f"\n✅ SUCCESS: {exp_name}")
        print(f"   Duration: {duration:.1f} minutes")

        return {
            'status': 'success',
            'duration_min': duration,
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat()
        }

    except subprocess.CalledProcessError as e:
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds() / 60

        print(f"\n❌ FAILED: {exp_name}")
        print(f"   Error code: {e.returncode}")
        print(f"   Duration before failure: {duration:.1f} minutes")

        return {
            'status': 'failed',
            'error_code': e.returncode,
            'duration_min': duration,
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat()
        }


def main():
    parser = argparse.ArgumentParser(
        description='Run all organized QBound experiments with multi-seed support and crash recovery',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--seed', type=int, default=None,
                        help='Single random seed for all experiments (overrides default multi-seed)')
    parser.add_argument('--seeds', type=int, nargs='+', default=None,
                        help='Multiple random seeds for automatic multi-seed runs (default: 42 43 44 45 46)')
    parser.add_argument('--category', choices=['timestep', 'sparse', 'transformed', 'all'], default='all',
                        help='Which category to run (default: all)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print what would be executed without running')
    parser.add_argument('--log-file', type=str, default='results/organized_experiments_log.json',
                        help='Path to save execution log')
    parser.add_argument('--yes', '-y', action='store_true', default=True,
                        help='Skip confirmation prompt and start immediately (default: True)')
    parser.add_argument('--no-auto-confirm', action='store_false', dest='yes',
                        help='Require manual confirmation before starting')

    args = parser.parse_args()

    # Determine which seeds to use (DEFAULT: 5 seeds for statistical significance)
    if args.seed is not None:
        # Single seed mode (overrides everything)
        seeds_to_run = [args.seed]
    elif args.seeds is not None:
        # Multiple seeds specified
        seeds_to_run = args.seeds
    else:
        # DEFAULT: Run with 5 seeds for statistical significance
        seeds_to_run = [42, 43, 44, 45, 46]

    # Select experiments based on category
    experiments_to_run = []

    if args.category in ['timestep', 'all']:
        experiments_to_run.extend(TIME_STEP_DEPENDENT_EXPERIMENTS)

    if args.category in ['sparse', 'all']:
        experiments_to_run.extend(SPARSE_STATE_DEPENDENT_EXPERIMENTS)

    if args.category in ['transformed', 'all']:
        experiments_to_run.extend(TRANSFORMED_QVALUE_EXPERIMENTS)

    # Print summary
    print("\n" + "="*80)
    print("ORGANIZED QBOUND EXPERIMENTS - MULTI-SEED MODE")
    print("="*80)
    print(f"\nSeeds: {seeds_to_run}")
    print(f"Category: {args.category}")
    print(f"Dry run: {args.dry_run}")
    print(f"Total experiments per seed: {len(experiments_to_run)}")
    print(f"Total runs (experiments × seeds): {len(experiments_to_run) * len(seeds_to_run)}")

    if args.category == 'timestep' or args.category == 'all':
        print_experiment_summary(TIME_STEP_DEPENDENT_EXPERIMENTS, "TIME-STEP DEPENDENT")

    if args.category == 'sparse' or args.category == 'all':
        print_experiment_summary(SPARSE_STATE_DEPENDENT_EXPERIMENTS, "SPARSE/STATE-DEPENDENT")

    if args.category == 'transformed' or args.category == 'all':
        print_experiment_summary(TRANSFORMED_QVALUE_EXPERIMENTS, "TRANSFORMED Q-VALUES (Negative → Positive)")

    # Confirm before running
    if not args.dry_run:
        total_time_per_seed = sum(exp['est_time_min'] for exp in experiments_to_run)
        total_time_all_seeds = total_time_per_seed * len(seeds_to_run)
        print(f"\nEstimated time per seed: {total_time_per_seed} minutes ({total_time_per_seed/60:.1f} hours)")
        print(f"Total estimated time (all {len(seeds_to_run)} seeds): {total_time_all_seeds} minutes ({total_time_all_seeds/60:.1f} hours)")

        if not args.yes:
            response = input("\nProceed with experiments? [y/N]: ")
            if response.lower() != 'y':
                print("Aborted by user.")
                return
        else:
            print("\n--yes flag provided, proceeding automatically...")

    # Load crash recovery state
    crash_recovery_state = load_crash_recovery_state(args.log_file)
    if crash_recovery_state is None:
        crash_recovery_state = {
            'session_start': datetime.now().isoformat(),
            'seeds': seeds_to_run,
            'category': args.category,
            'dry_run': args.dry_run,
            'seed_results': {}
        }

    # Run experiments for each seed
    session_start = datetime.now()
    total_experiments_run = 0
    total_experiments_skipped = 0

    for seed_idx, seed in enumerate(seeds_to_run, 1):
        print(f"\n{'█'*80}")
        print(f"█ SEED {seed_idx}/{len(seeds_to_run)}: {seed}")
        print(f"{'█'*80}\n")

        # Initialize results for this seed if not exists
        seed_key = str(seed)
        if seed_key not in crash_recovery_state['seed_results']:
            crash_recovery_state['seed_results'][seed_key] = {
                'seed': seed,
                'start_time': datetime.now().isoformat(),
                'experiments': []
            }

        seed_results = crash_recovery_state['seed_results'][seed_key]

        # Run all experiments for this seed
        for exp_idx, exp in enumerate(experiments_to_run, 1):
            print(f"\n{'#'*80}")
            print(f"# SEED {seed_idx}/{len(seeds_to_run)} | EXPERIMENT {exp_idx}/{len(experiments_to_run)}")
            print(f"{'#'*80}")

            # Check if this experiment was already recorded in the progress log
            already_recorded = any(
                e['script'] == exp['script']
                for e in seed_results['experiments']
            )

            # If already recorded, verify that result files actually exist
            # If files were deleted, we need to re-run even if it's in the log
            if already_recorded:
                if is_experiment_completed(exp['script'], seed):
                    print(f"\n{'='*80}")
                    print(f"ALREADY RECORDED IN LOG (files verified): {exp['name']}")
                    print(f"Script: {exp['script']}")
                    print(f"Seed: {seed}")
                    print(f"{'='*80}\n")
                    total_experiments_skipped += 1
                    continue
                else:
                    print(f"\n{'='*80}")
                    print(f"⚠️  RECORDED IN LOG BUT FILES MISSING: {exp['name']}")
                    print(f"Script: {exp['script']}")
                    print(f"Seed: {seed}")
                    print(f"Re-running experiment...")
                    print(f"{'='*80}\n")
                    # Remove from log so it gets re-recorded
                    seed_results['experiments'] = [
                        e for e in seed_results['experiments']
                        if e['script'] != exp['script']
                    ]

            # Run experiment (will auto-skip if already completed)
            exp_result = run_experiment(exp, seed, args.dry_run)

            if exp_result['status'] == 'skipped' and exp_result.get('reason') == 'already_completed':
                total_experiments_skipped += 1
            else:
                total_experiments_run += 1

            # Record result
            seed_results['experiments'].append({
                'name': exp['name'],
                'script': exp['script'],
                'methods': exp['methods'],
                'estimated_time_min': exp['est_time_min'],
                'result': exp_result
            })

            # Save progress after each experiment (for fine-grained recovery)
            if not args.dry_run and exp_result['status'] != 'skipped':
                save_crash_recovery_state(args.log_file, crash_recovery_state)

        # Mark seed as completed
        seed_results['end_time'] = datetime.now().isoformat()

        # Save checkpoint after each seed
        if not args.dry_run:
            print(f"\n✅ Seed {seed} completed - saving checkpoint...")
            save_crash_recovery_state(args.log_file, crash_recovery_state)

    # Final summary
    session_end = datetime.now()
    crash_recovery_state['session_end'] = session_end.isoformat()
    crash_recovery_state['total_duration_min'] = (session_end - session_start).total_seconds() / 60

    print(f"\n{'='*80}")
    print("MULTI-SEED FINAL SUMMARY")
    print(f"{'='*80}")

    print(f"\nSeeds processed: {len(crash_recovery_state['seed_results'])}/{len(seeds_to_run)}")
    print(f"Seed list: {seeds_to_run}")

    # Count successes and failures across all seeds
    total_successful = 0
    total_failed = 0

    for seed_key, seed_data in crash_recovery_state['seed_results'].items():
        successful = sum(1 for exp in seed_data['experiments'] if exp['result']['status'] == 'success')
        failed = sum(1 for exp in seed_data['experiments'] if exp['result']['status'] == 'failed')
        skipped = sum(1 for exp in seed_data['experiments'] if exp['result']['status'] == 'skipped')
        total_successful += successful
        total_failed += failed

        print(f"\nSeed {seed_key}:")
        print(f"  ✓ Successful: {successful}/{len(experiments_to_run)}")
        print(f"  ⏭️  Skipped (already done): {skipped}/{len(experiments_to_run)}")
        print(f"  ✗ Failed: {failed}/{len(experiments_to_run)}")

    print(f"\nOverall Statistics:")
    print(f"  Total experiments executed: {total_experiments_run}")
    print(f"  Skipped (already completed): {total_experiments_skipped}")
    print(f"  Successful: {total_successful}")
    print(f"  Failed: {total_failed}")

    if not args.dry_run:
        total_actual_time = crash_recovery_state['total_duration_min']
        print(f"  Total time: {total_actual_time:.1f} minutes ({total_actual_time/60:.1f} hours)")
        print(f"\nLog saved to: {args.log_file}")

    print(f"\n{'='*80}\n")

    # Save final results
    if not args.dry_run:
        save_crash_recovery_state(args.log_file, crash_recovery_state)

    # Exit with error code if any experiments failed
    if total_failed > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
