#!/bin/bash
#
# Resume Experiments - Continue from Pendulum
#
# Status:
# ✓ Already completed (DO NOT re-run):
#   - GridWorld 6-way (Oct 28, 09:37)
#   - FrozenLake 6-way (Oct 28, 09:59)
#   - CartPole 6-way (Oct 28, 10:46)
#   - LunarLander 6-way (Oct 28, 12:33) ← THIS IS COMPLETE!
#
# 🔄 To be run:
#   - Pendulum DDPG 6-way (resume - has crash recovery, 3/6 methods done)
#   - LunarLander Continuous PPO
#   - Pendulum PPO
#

set -e  # Exit on error

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs"
mkdir -p $LOG_DIR

echo "================================================================================"
echo "QBound Experiments - RESUMING FROM PENDULUM"
echo "================================================================================"
echo "Start time: $(date)"
echo "Log directory: $LOG_DIR"
echo ""
echo "⏭️  Skipping already completed experiments:"
echo "   ✓ GridWorld 6-way (completed Oct 28, 09:37)"
echo "   ✓ FrozenLake 6-way (completed Oct 28, 09:59)"
echo "   ✓ CartPole 6-way (completed Oct 28, 10:46)"
echo "   ✓ LunarLander 6-way (completed Oct 28, 12:33)"
echo ""
echo "🔄 Will run (3 remaining experiments):"
echo "   1. Pendulum DDPG 6-way (resume from 3/6 methods completed)"
echo "   2. LunarLander Continuous PPO"
echo "   3. Pendulum PPO"
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
        return 0
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

# ===== DDPG Experiments =====
echo ""
echo "================================================================================"
echo "PART 1: DDPG 6-Way Experiment (Continuous Action Spaces - SOFT QBound)"
echo "================================================================================"
echo "Note: This experiment has crash recovery. It will resume from method 4/6."

if run_experiment "Pendulum_DDPG_6way" "experiments/pendulum/train_6way_comparison.py"; then
    SUCCESS+=("Pendulum_DDPG_6way")
else
    FAILED+=("Pendulum_DDPG_6way")
fi

# ===== PPO Experiments =====
echo ""
echo "================================================================================"
echo "PART 2: PPO Experiments (Continuous Control - SOFT QBound)"
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
echo "EXPERIMENT SUMMARY"
echo "================================================================================"
echo "End time: $(date)"
echo ""
echo "Previously completed (not re-run):"
echo "  ✓ GridWorld_6way (Oct 28, 09:37)"
echo "  ✓ FrozenLake_6way (Oct 28, 09:59)"
echo "  ✓ CartPole_6way (Oct 28, 10:46)"
echo "  ✓ LunarLander_6way (Oct 28, 12:33)"
echo ""
echo "Newly completed experiments (${#SUCCESS[@]}/3):"
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
    echo ""
    echo "Total completion: $((4 + ${#SUCCESS[@]}))/7 experiments"
else
    echo ""
    echo "✓ ALL REMAINING EXPERIMENTS COMPLETED SUCCESSFULLY!"
    echo "✓ FULL 7/7 EXPERIMENT SUITE COMPLETE!"
fi

echo "================================================================================"
echo "All results saved to respective results/ directories"
echo "Logs saved to: $LOG_DIR"
echo "================================================================================"
