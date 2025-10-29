#!/bin/bash
#
# Master script to run all 6-way DQN comparison experiments
# Each experiment compares:
#   1. Baseline DQN
#   2. Static QBound + DQN
#   3. Dynamic QBound + DQN
#   4. Baseline DDQN
#   5. Static QBound + DDQN
#   6. Dynamic QBound + DDQN
#

set -e  # Exit on error

echo "================================================================================"
echo "Running All 6-Way DQN Comparison Experiments"
echo "================================================================================"
echo ""
echo "This will run experiments for:"
echo "  1. GridWorld (sparse terminal reward)"
echo "  2. FrozenLake (sparse terminal reward, stochastic)"
echo "  3. CartPole (dense positive reward)"
echo "  4. LunarLander (shaped/mixed reward)"
echo ""
echo "Each environment tests 6 variants:"
echo "  - Baseline DQN, Static QBound, Dynamic QBound"
echo "  - Baseline DDQN, Static QBound+DDQN, Dynamic QBound+DDQN"
echo ""
echo "================================================================================"
echo ""

# Create results directories
mkdir -p /root/projects/QBound/results/gridworld
mkdir -p /root/projects/QBound/results/frozenlake
mkdir -p /root/projects/QBound/results/cartpole
mkdir -p /root/projects/QBound/results/lunarlander

# Track total time
START_TIME=$(date +%s)

# ===============================================================================
# 1. GridWorld (Fast - ~15 minutes)
# ===============================================================================
echo ""
echo "================================================================================"
echo "1/4: Running GridWorld 6-Way Comparison"
echo "================================================================================"
echo "Estimated time: ~15 minutes"
echo ""

python3 /root/projects/QBound/experiments/gridworld/train_gridworld_6way.py

echo ""
echo "✓ GridWorld complete!"
echo ""

# ===============================================================================
# 2. FrozenLake (Moderate - ~60 minutes)
# ===============================================================================
echo ""
echo "================================================================================"
echo "2/4: Running FrozenLake 6-Way Comparison"
echo "================================================================================"
echo "Estimated time: ~60 minutes (2000 episodes per variant)"
echo ""

python3 /root/projects/QBound/experiments/frozenlake/train_frozenlake_6way.py

echo ""
echo "✓ FrozenLake complete!"
echo ""

# ===============================================================================
# 3. CartPole (Fast - ~15 minutes)
# ===============================================================================
echo ""
echo "================================================================================"
echo "3/4: Running CartPole 6-Way Comparison"
echo "================================================================================"
echo "Estimated time: ~15 minutes"
echo ""

python3 /root/projects/QBound/experiments/cartpole/train_cartpole_6way.py

echo ""
echo "✓ CartPole complete!"
echo ""

# ===============================================================================
# 4. LunarLander (Moderate - ~30 minutes)
# ===============================================================================
echo ""
echo "================================================================================"
echo "4/4: Running LunarLander 6-Way Comparison"
echo "================================================================================"
echo "Estimated time: ~30 minutes"
echo ""

python3 /root/projects/QBound/experiments/lunarlander/train_lunarlander_6way.py

echo ""
echo "✓ LunarLander complete!"
echo ""

# ===============================================================================
# Summary
# ===============================================================================
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))

echo ""
echo "================================================================================"
echo "ALL 6-WAY EXPERIMENTS COMPLETE!"
echo "================================================================================"
echo ""
echo "Total time: ${HOURS}h ${MINUTES}m"
echo ""
echo "Results saved to:"
echo "  - /root/projects/QBound/results/gridworld/6way_comparison_*.json"
echo "  - /root/projects/QBound/results/frozenlake/6way_comparison_*.json"
echo "  - /root/projects/QBound/results/cartpole/6way_comparison_*.json"
echo "  - /root/projects/QBound/results/lunarlander/6way_comparison_*.json"
echo ""
echo "Next steps:"
echo "  1. Run analysis script: python3 analysis/analyze_6way_results.py"
echo "  2. Generate plots for paper: python3 analysis/plot_6way_paper_results.py"
echo ""
echo "================================================================================"
