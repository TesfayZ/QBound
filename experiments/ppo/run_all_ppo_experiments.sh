#!/bin/bash
#
# Run all PPO+QBound experiments
#
# Usage:
#   ./run_all_ppo_experiments.sh              # Run all experiments
#   ./run_all_ppo_experiments.sh cartpole     # Run specific experiment
#

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "PPO + QBound Experimental Suite"
echo "============================================================"
echo ""

# Parse arguments
RUN_ALL=true
if [ $# -gt 0 ]; then
    RUN_ALL=false
    EXPERIMENT=$1
fi

# Function to run experiment
run_experiment() {
    local name=$1
    local script=$2
    local category=$3

    if [ "$RUN_ALL" = true ] || [ "$EXPERIMENT" = "$name" ]; then
        echo ""
        echo "============================================================"
        echo "Running: $name ($category)"
        echo "============================================================"
        python3 "$script"
        echo "✓ $name completed"
    fi
}

# Discrete + Sparse Reward
run_experiment "lunarlander" "pilot_lunarlander.py" "Discrete + Sparse"
run_experiment "acrobot" "train_acrobot.py" "Discrete + Sparse"
run_experiment "mountaincar" "train_mountaincar.py" "Discrete + Sparse"

# Discrete + Dense Reward
run_experiment "cartpole" "pilot_cartpole.py" "Discrete + Dense"

# Continuous + Dense Reward
run_experiment "pendulum" "train_pendulum.py" "Continuous + Dense (CRITICAL TEST)"

# Continuous + Sparse Reward
run_experiment "lunarlander_continuous" "train_lunarlander_continuous.py" "Continuous + Sparse (CRITICAL TEST)"

echo ""
echo "============================================================"
echo "All experiments completed!"
echo "============================================================"
echo ""
echo "Results saved to: /root/projects/QBound/results/ppo/"
echo ""
echo "Next steps:"
echo "  1. Run analysis: python3 ../../analysis/analyze_ppo_results.py"
echo "  2. Generate plots: python3 ../../analysis/plot_ppo_results.py"
echo ""
