#!/bin/bash
#
# Monitor PPO experimental progress
#

echo "============================================================"
echo "PPO + QBound Experimental Progress Monitor"
echo "============================================================"
echo ""

# Check if experiments are running
if ps aux | grep -v grep | grep "run_all_ppo_experiments.sh" > /dev/null; then
    echo "✓ Experiments are RUNNING"
else
    echo "✗ Experiments are NOT running"
fi

echo ""
echo "Results found:"
echo "----------------------------------------"
ls -lh results/ppo/*.json 2>/dev/null | awk '{print $9, "("$5")"}'

echo ""
echo "Completed experiments:"
echo "----------------------------------------"
for file in results/ppo/*.json; do
    if [ -f "$file" ]; then
        basename "$file" .json
    fi
done

echo ""
echo "Last 20 lines of log:"
echo "----------------------------------------"
tail -20 /tmp/ppo_full_suite.log 2>/dev/null || echo "No log file found"

echo ""
echo "============================================================"
echo "To view live progress:"
echo "  tail -f /tmp/ppo_full_suite.log"
echo ""
echo "To analyze results when complete:"
echo "  python3 analysis/quick_ppo_analysis.py"
echo "============================================================"
