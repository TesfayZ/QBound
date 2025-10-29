#!/bin/bash
#
# Master Experiment Runner - Sequential Execution
#
# Runs all QBound experiments in sequence with proper error handling.
# This ensures experiments complete one by one even if some fail.
#
# Order:
# 1. DQN 6-way experiments (GridWorld, FrozenLake, CartPole, LunarLander)
# 2. DDPG 6-way experiment (Pendulum) - with SOFT QBound
# 3. PPO experiments (LunarLander-Continuous, Pendulum) - with SOFT QBound
#

set -e  # Exit on error

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs"
mkdir -p $LOG_DIR

echo "================================================================================"
echo "QBound Experiments - Sequential Execution"
echo "================================================================================"
echo "Start time: $(date)"
echo "Log directory: $LOG_DIR"
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

# ===== DQN Experiments =====
echo ""
echo "================================================================================"
echo "PART 1: DQN 6-Way Experiments (Discrete Action Spaces)"
echo "================================================================================"

if run_experiment "GridWorld_6way" "experiments/gridworld/train_gridworld_6way.py"; then
    SUCCESS+=("GridWorld_6way")
else
    FAILED+=("GridWorld_6way")
fi

if run_experiment "FrozenLake_6way" "experiments/frozenlake/train_frozenlake_6way.py"; then
    SUCCESS+=("FrozenLake_6way")
else
    FAILED+=("FrozenLake_6way")
fi

if run_experiment "CartPole_6way" "experiments/cartpole/train_cartpole_6way.py"; then
    SUCCESS+=("CartPole_6way")
else
    FAILED+=("CartPole_6way")
fi

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
echo "EXPERIMENT SUMMARY"
echo "================================================================================"
echo "End time: $(date)"
echo ""
echo "Successful experiments (${#SUCCESS[@]}):"
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
    echo "✓ ALL EXPERIMENTS COMPLETED SUCCESSFULLY!"
fi

echo "================================================================================"
echo "Logs saved to: $LOG_DIR"
echo "================================================================================"
