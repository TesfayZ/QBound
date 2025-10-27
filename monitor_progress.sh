#!/bin/bash
# Live progress monitor for all experiments

clear
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║          QBound Extended Experiments - Live Progress              ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""

while true; do
    # Clear previous output (keep header)
    tput cup 4 0
    tput ed

    echo "┌─ MOUNTAINCAR (6-way) ──────────────────────────────────────────┐"
    if [ -f "/root/projects/QBound/results/mountaincar/experiment.log" ]; then
        LAST_LINE=$(tail -1 /root/projects/QBound/results/mountaincar/experiment.log)
        echo "│ $LAST_LINE"

        # Check if completed
        if grep -q "6-Way Comparison Complete" /root/projects/QBound/results/mountaincar/experiment.log; then
            echo "│ Status: ✓ COMPLETED"
        else
            echo "│ Status: ⏳ RUNNING"
        fi
    else
        echo "│ Status: ⏰ QUEUED"
    fi
    echo "└────────────────────────────────────────────────────────────────┘"
    echo ""

    echo "┌─ ACROBOT (6-way) ──────────────────────────────────────────────┐"
    if [ -f "/root/projects/QBound/results/acrobot/experiment.log" ]; then
        LAST_LINE=$(tail -1 /root/projects/QBound/results/acrobot/experiment.log)
        echo "│ $LAST_LINE"

        if grep -q "6-Way Comparison Complete" /root/projects/QBound/results/acrobot/experiment.log; then
            echo "│ Status: ✓ COMPLETED"
        else
            echo "│ Status: ⏳ RUNNING"
        fi
    else
        echo "│ Status: ⏰ QUEUED (waits for MountainCar)"
    fi
    echo "└────────────────────────────────────────────────────────────────┘"
    echo ""

    echo "┌─ LUNARLANDER (4-way) ──────────────────────────────────────────┐"
    if [ -f "/root/projects/QBound/results/lunarlander/experiment.log" ]; then
        LAST_LINE=$(tail -1 /root/projects/QBound/results/lunarlander/experiment.log)
        echo "│ $LAST_LINE"

        if grep -q "4-Way Comparison Complete" /root/projects/QBound/results/lunarlander/experiment.log; then
            echo "│ Status: ✓ COMPLETED"
        else
            echo "│ Status: ⏳ RUNNING"
        fi
    else
        echo "│ Status: ⏰ QUEUED (waits for Acrobot)"
    fi
    echo "└────────────────────────────────────────────────────────────────┘"
    echo ""

    echo "┌─ CARTPOLE CORRECTED (6-way) ───────────────────────────────────┐"
    if [ -f "/root/projects/QBound/results/cartpole_corrected/experiment.log" ]; then
        LAST_LINE=$(tail -1 /root/projects/QBound/results/cartpole_corrected/experiment.log)
        echo "│ $LAST_LINE"

        if grep -q "6-Way Comparison Complete" /root/projects/QBound/results/cartpole_corrected/experiment.log; then
            echo "│ Status: ✓ COMPLETED"
        else
            echo "│ Status: ⏳ RUNNING"
        fi
    else
        echo "│ Status: ⏰ QUEUED (waits for LunarLander)"
    fi
    echo "└────────────────────────────────────────────────────────────────┘"
    echo ""

    echo "═══════════════════════════════════════════════════════════════════"
    echo "Updated: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Press Ctrl+C to exit monitor"

    sleep 10
done
