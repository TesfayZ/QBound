#!/usr/bin/env python3
"""
Prepare experiments for selective dynamic QBound rerun

This script:
1. Backs up original results with buggy dynamic QBound
2. Removes only dynamic QBound results from JSON files
3. Prepares in_progress.json files for crash recovery
4. Preserves baseline and static QBound results

After running this script, simply rerun the experiments and they will:
- Skip baseline (already completed)
- Skip static QBound (already completed)
- Only train dynamic QBound methods with the fix
"""

import json
import glob
import shutil
import os
from datetime import datetime

def main():
    print("=" * 80)
    print("SELECTIVE DYNAMIC QBOUND RERUN PREPARATION")
    print("=" * 80)

    # Experiments to process
    experiments = [
        {
            'name': 'Pendulum DQN',
            'pattern': 'results/pendulum/dqn_full_qbound_seed*.json',
            'dynamic_methods': ['dynamic_qbound_dqn', 'dynamic_qbound_double_dqn'],
            'seeds': []
        },
        {
            'name': 'Pendulum DDPG',
            'pattern': 'results/pendulum/ddpg_full_qbound_seed*.json',
            'dynamic_methods': ['dynamic_soft_qbound'],
            'seeds': []
        },
        {
            'name': 'Pendulum TD3',
            'pattern': 'results/pendulum/td3_full_qbound_seed*.json',
            'dynamic_methods': ['dynamic_soft_qbound'],
            'seeds': []
        }
    ]

    # Create backup directory
    backup_dir = f"results/pendulum/backup_buggy_dynamic_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(backup_dir, exist_ok=True)
    print(f"\n📦 Backup directory: {backup_dir}\n")

    total_files = 0
    total_methods_removed = 0

    for exp in experiments:
        files = glob.glob(exp['pattern'])
        files = [f for f in files if 'in_progress' not in f]

        if not files:
            print(f"⏭️  {exp['name']}: No results found, skipping")
            continue

        print(f"\n{'=' * 80}")
        print(f"{exp['name']}: Found {len(files)} files")
        print(f"{'=' * 80}")

        for fpath in sorted(files):
            filename = os.path.basename(fpath)

            # Extract seed
            seed = fpath.split('seed')[1].split('_')[0]
            exp['seeds'].append(int(seed))

            print(f"\n  Processing: {filename}")

            # Backup original
            backup_path = os.path.join(backup_dir, filename)
            shutil.copy2(fpath, backup_path)
            print(f"    ✓ Backed up to: {backup_path}")

            # Load results
            with open(fpath, 'r') as f:
                data = json.load(f)

            # Check which dynamic methods exist
            removed = []
            for method in exp['dynamic_methods']:
                if method in data.get('training', {}):
                    removed.append(method)
                    data['training'].pop(method)

            if removed:
                print(f"    ✓ Removed dynamic methods: {', '.join(removed)}")
                total_methods_removed += len(removed)

                # Save as in_progress file (crash recovery format)
                out_file = fpath.replace('.json', '').split('_202')[0] + '_in_progress.json'
                with open(out_file, 'w') as f:
                    json.dump(data, f, indent=2)

                print(f"    ✓ Created: {os.path.basename(out_file)}")
                total_files += 1
            else:
                print(f"    ⚠️  No dynamic methods found to remove")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"\n  Files processed: {total_files}")
    print(f"  Dynamic methods removed: {total_methods_removed}")
    print(f"  Backup location: {backup_dir}")

    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)

    print("""
1. Verify the in_progress.json files were created:
   ls -lh results/pendulum/*_in_progress.json

2. Rerun experiments (they will use crash recovery):

   # Pendulum DQN - will skip baseline/static, only train dynamic
   for seed in 42 43 44 45 46; do
       python3 experiments/pendulum/train_pendulum_dqn_full_qbound.py --seed $seed
   done

   # Pendulum DDPG - will skip baseline/static, only train dynamic
   for seed in 42 43 44 45 46; do
       python3 experiments/pendulum/train_pendulum_ddpg_full_qbound.py --seed $seed
   done

   # Pendulum TD3 - will skip baseline/static, only train dynamic
   for seed in 42 43 44 45 46; do
       python3 experiments/pendulum/train_pendulum_td3_full_qbound.py --seed $seed
   done

3. After successful completion:
   - Final results saved as: results/pendulum/*_seed*_TIMESTAMP.json
   - in_progress files automatically deleted
   - Baseline and static QBound results preserved!

4. If anything goes wrong:
   - Original results backed up in: {backup_dir}
   - Can restore: cp {backup_dir}/*.json results/pendulum/
""".format(backup_dir=backup_dir))

    print("\n" + "=" * 80)
    print("ESTIMATED TIME")
    print("=" * 80)

    # Calculate estimated time (only dynamic methods)
    time_per_method = {
        'Pendulum DQN': 40,      # ~40 min per dynamic method (2 methods × 40min)
        'Pendulum DDPG': 30,     # ~30 min per dynamic method (1 method × 30min)
        'Pendulum TD3': 30,      # ~30 min per dynamic method (1 method × 30min)
    }

    total_time_per_seed = 0
    for exp in experiments:
        if exp['seeds']:
            time = time_per_method.get(exp['name'], 30) * len(exp['dynamic_methods'])
            total_time_per_seed += time
            print(f"\n  {exp['name']}: ~{time} min/seed ({len(exp['dynamic_methods'])} dynamic methods)")

    num_seeds = 5  # Typically 42-46
    total_time = total_time_per_seed * num_seeds

    print(f"\n  Total per seed: ~{total_time_per_seed} minutes")
    print(f"  Total for {num_seeds} seeds: ~{total_time} minutes ({total_time/60:.1f} hours)")
    print(f"\n  vs. Full rerun: ~400 minutes (6.7 hours)")
    print(f"  Time saved: ~{400-total_time} minutes ({(400-total_time)/60:.1f} hours)")

    print("\n✓ Ready for selective rerun!")
    print("  Run the commands above to start training only dynamic QBound methods.\n")

if __name__ == '__main__':
    main()
