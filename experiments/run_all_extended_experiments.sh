#!/bin/bash
# Master script to run all extended experiments sequentially
# This ensures experiments run one after another without conflicts

set -e  # Exit on error

PROJECT_ROOT="/root/projects/QBound"
LOG_DIR="$PROJECT_ROOT/results"

echo "============================================================"
echo "Running All Extended Experiments Sequentially"
echo "============================================================"
echo "Started at: $(date)"
echo ""

# Function to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Function to run experiment and check success
run_experiment() {
    local name=$1
    local script=$2
    local log_file=$3

    log "Starting: $name"
    echo "------------------------------------------------------------"

    if python3 "$script" > "$log_file" 2>&1; then
        log "✓ COMPLETED: $name"
    else
        log "✗ FAILED: $name (check $log_file for details)"
        exit 1
    fi

    echo ""
}

# Check if MountainCar is still running
MOUNTAINCAR_PID=$(pgrep -f "train_mountaincar_6way.py" || echo "")

if [ -n "$MOUNTAINCAR_PID" ]; then
    log "Waiting for MountainCar experiment to complete (PID: $MOUNTAINCAR_PID)..."
    echo "This may take ~80 minutes total"

    # Wait for MountainCar to finish
    while kill -0 $MOUNTAINCAR_PID 2>/dev/null; do
        sleep 60  # Check every minute
        # Show progress
        if [ -f "$LOG_DIR/mountaincar/experiment.log" ]; then
            PROGRESS=$(tail -1 "$LOG_DIR/mountaincar/experiment.log" | grep -oP '\d+%' | tail -1 || echo "?%")
            log "MountainCar progress: $PROGRESS"
        fi
    done

    log "✓ MountainCar experiment completed!"
else
    log "MountainCar experiment already completed or not running"
fi

echo ""
echo "============================================================"
echo "Starting Remaining Experiments"
echo "============================================================"
echo ""

# Run Acrobot
run_experiment \
    "Acrobot 6-Way Comparison" \
    "$PROJECT_ROOT/experiments/acrobot/train_acrobot_6way.py" \
    "$LOG_DIR/acrobot/experiment.log"

# Run LunarLander
run_experiment \
    "LunarLander 4-Way Comparison" \
    "$PROJECT_ROOT/experiments/lunarlander/train_lunarlander_4way.py" \
    "$LOG_DIR/lunarlander/experiment.log"

# Run Corrected CartPole
run_experiment \
    "CartPole (Corrected) 6-Way Comparison" \
    "$PROJECT_ROOT/experiments/cartpole_corrected/train_cartpole_6way.py" \
    "$LOG_DIR/cartpole_corrected/experiment.log"

echo ""
echo "============================================================"
echo "ALL EXPERIMENTS COMPLETED!"
echo "============================================================"
echo "Completed at: $(date)"
echo ""
echo "Results saved to:"
echo "  - $LOG_DIR/mountaincar/"
echo "  - $LOG_DIR/acrobot/"
echo "  - $LOG_DIR/lunarlander/"
echo "  - $LOG_DIR/cartpole_corrected/"
echo ""
