#!/bin/bash
# Auto-run CartPole after LunarLander completes

set -e

PROJECT_ROOT="/root/projects/QBound"
LOG_DIR="$PROJECT_ROOT/results"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Wait for LunarLander to complete
LUNARLANDER_PID=$(pgrep -f "train_lunarlander_4way.py" || echo "")

if [ -n "$LUNARLANDER_PID" ]; then
    log "Waiting for LunarLander to complete (PID: $LUNARLANDER_PID)..."

    while kill -0 $LUNARLANDER_PID 2>/dev/null; do
        sleep 60
        if [ -f "$LOG_DIR/lunarlander/experiment.log" ]; then
            PROGRESS=$(tail -1 "$LOG_DIR/lunarlander/experiment.log" | grep -oP '\d+%' | tail -1 || echo "")
            if [ -n "$PROGRESS" ]; then
                log "LunarLander progress: $PROGRESS"
            fi
        fi
    done

    log "✓ LunarLander completed!"
else
    log "LunarLander already completed or not running"
fi

# Run CartPole
log "Starting CartPole corrected 6-way experiment..."
python3 "$PROJECT_ROOT/experiments/cartpole_corrected/train_cartpole_6way.py" \
    > "$LOG_DIR/cartpole_corrected/experiment.log" 2>&1

if [ $? -eq 0 ]; then
    log "✓ COMPLETED: CartPole corrected 6-way experiment"
else
    log "✗ FAILED: CartPole experiment"
    exit 1
fi

log "✅ ALL EXPERIMENTS COMPLETED!"
