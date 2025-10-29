#!/bin/bash
#
# Resume Master Experiment Runner - From LunarLander
#
# This script resumes the experiment run that was interrupted at 12:27 on Oct 28.
#
# ✓ Already completed (DO NOT re-run):
#   - GridWorld 6-way
#   - FrozenLake 6-way
#   - CartPole 6-way
#
# 🔄 To be run:
#   - LunarLander 6-way (restart from beginning - no mid-method checkpoint)
#   - Pendulum DDPG 6-way
#   - LunarLander Continuous PPO
#   - Pendulum PPO
#

set -e  # Exit on error

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs"
mkdir -p $LOG_DIR

echo "================================================================================"
echo "QBound Experiments - RESUMING FROM LUNARLANDER"
echo "================================================================================"
echo "Start time: $(date)"
echo "Log directory: $LOG_DIR"
echo ""
echo "⏭️  Skipping already completed experiments:"
echo "   ✓ GridWorld 6-way (completed at 09:59)"
echo "   ✓ FrozenLake 6-way (completed at 10:46)"
echo "   ✓ CartPole 6-way (completed at 12:11)"
echo ""
echo "🔄 Will run:"
echo "   1. LunarLander 6-way (restarting from beginning)"
echo "   2. Pendulum DDPG 6-way"
echo "   3. LunarLander Continuous PPO"
echo "   4. Pendulum PPO"
echo "================================================================================"

# Function to run experiment with error handling
run_experiment() {
    local name=$1
    local script=$2
    local logfile="${LOG_DIR}/${name}_${TIMESTAMP}.log"

    echo ""
    echo "################################################################################"
    echo "# $name"
    echo "################################################################################"
    echo "Script: $script"
    echo "Log: $logfile"
    echo "Start: $(date)"
    echo ""

    if python3 "$script" 2>&1 | tee "$logfile"; then
        echo ""
        echo "✓ SUCCESS: $name completed"
        echo "End: $(date)"
    else
        echo ""
        echo "✗ FAILED: $name encountered an error"
        echo "End: $(date)"
        echo "Check log: $logfile"
        return 1
    fi
}

# Track successes and failures
declare -a SUCCESS
declare -a FAILED

# ===== Resume from LunarLander =====
echo ""
echo "================================================================================"
echo "PART 1: DQN 6-Way Experiment (Resuming from LunarLander)"
echo "================================================================================"

if run_experiment "LunarLander_6way" "experiments/lunarlander/train_lunarlander_6way.py"; then
    SUCCESS+=("LunarLander_6way")
else
    FAILED+=("LunarLander_6way")
fi

# ===== DDPG Experiments =====
echo ""
echo "================================================================================"
echo "PART 2: DDPG 6-Way Experiment (Continuous Action Spaces - SOFT QBound)"
echo "================================================================================"

if run_experiment "Pendulum_DDPG_6way" "experiments/pendulum/train_6way_comparison.py"; then
    SUCCESS+=("Pendulum_DDPG_6way")
else
    FAILED+=("Pendulum_DDPG_6way")
fi

# ===== PPO Experiments =====
echo ""
echo "================================================================================"
echo "PART 3: PPO Experiments (Continuous Control - SOFT QBound)"
echo "================================================================================"

if run_experiment "LunarLander_Continuous_PPO" "experiments/ppo/train_lunarlander_continuous.py"; then
    SUCCESS+=("LunarLander_Continuous_PPO")
else
    FAILED+=("LunarLander_Continuous_PPO")
fi

if run_experiment "Pendulum_PPO" "experiments/ppo/train_pendulum.py"; then
    SUCCESS+=("Pendulum_PPO")
else
    FAILED+=("Pendulum_PPO")
fi

# ===== Summary =====
echo ""
echo "================================================================================"
echo "RESUMED EXPERIMENT SUMMARY"
echo "================================================================================"
echo "End time: $(date)"
echo ""
echo "Previously completed (not re-run):"
echo "  ✓ GridWorld_6way"
echo "  ✓ FrozenLake_6way"
echo "  ✓ CartPole_6way"
echo ""
echo "Newly completed experiments (${#SUCCESS[@]}):"
for exp in "${SUCCESS[@]}"; do
    echo "  ✓ $exp"
done

if [ ${#FAILED[@]} -gt 0 ]; then
    echo ""
    echo "Failed experiments (${#FAILED[@]}):"
    for exp in "${FAILED[@]}"; do
        echo "  ✗ $exp"
    done
    echo ""
    echo "⚠️  Some experiments failed. Check logs in $LOG_DIR"
else
    echo ""
    echo "✓ ALL RESUMED EXPERIMENTS COMPLETED SUCCESSFULLY!"
fi

echo "================================================================================"
echo "All results saved to respective results/ directories"
echo "Logs saved to: $LOG_DIR"
echo "================================================================================"
