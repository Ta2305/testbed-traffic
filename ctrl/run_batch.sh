#!/bin/bash
# =============================================================================
# WAFL Batch Experiment Runner
# =============================================================================
# This script runs multiple WAFL experiments sequentially with automatic
# virtual environment reactivation between experiments.
#
# Usage:
#   1. Edit the EXPERIMENTS array below to define your experiments
#   2. Run: chmod +x ctrl/run_batch.sh && ./ctrl/run_batch.sh
#
# Features:
#   - Automatic venv deactivation/reactivation between experiments
#   - Single sudo password input at the beginning
#   - Logging of experiment progress
# =============================================================================

set -e  # Exit on error

# =============================================================================
# CONFIGURATION - Edit these values
# =============================================================================

# Virtual environment path (relative to project root)
VENV_PATH=".venv/bin/activate"

# Define experiments as an array of argument strings
# Each line is one experiment with its parameters
EXPERIMENTS=(
    "--experiment_name Re_efficient_exp_1_1_128_2048 --K 1 --Q 1 --self_epoch 128 --wafl_epoch 2048 --wafl_script src/main4.py"
)
    

# =============================================================================
# SCRIPT START - Do not edit below unless you know what you're doing
# =============================================================================

# Change to project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "========================================"
echo "WAFL Batch Experiment Runner"
echo "========================================"
echo "Project Root: $PROJECT_ROOT"
echo "Total Experiments: ${#EXPERIMENTS[@]}"
echo "========================================"

# Get sudo password once at the beginning
echo ""
read -s -p "Please enter the sudo password (will be used for all experiments): " SUDO_PASSWORD
echo ""
echo "========================================"

# Export for child processes
export SUDO_PASSWORD

COMPLETED=0
FAILED=0
START_TIME=$(date +%s)

for i in "${!EXPERIMENTS[@]}"; do
    EXPERIMENT_NUM=$((i + 1))
    EXPERIMENT_ARGS="${EXPERIMENTS[$i]}"
    
    echo ""
    echo "========================================"
    echo "[$EXPERIMENT_NUM/${#EXPERIMENTS[@]}] Starting experiment..."
    echo "Arguments: $EXPERIMENT_ARGS"
    echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "========================================"
    
    # Deactivate current venv if active, then reactivate
    # This is a workaround for the venv state corruption issue
    if [[ -n "$VIRTUAL_ENV" ]]; then
        echo "🔄 Deactivating current virtual environment..."
        deactivate 2>/dev/null || true
    fi
    
    echo "🔄 Activating virtual environment..."
    source "$VENV_PATH"
    
    # Refresh SSH agent socket for tmux compatibility
    # This fixes "No authentication methods available" errors
    # Always search for the latest valid socket in /tmp/ssh-* regardless of current SSH_AUTH_SOCK
    LATEST_SSH_SOCK=$(find /tmp/ssh-* -name "agent.*" -user "$USER" 2>/dev/null | head -n 1)
    if [ -n "$LATEST_SSH_SOCK" ] && [ -S "$LATEST_SSH_SOCK" ]; then
        export SSH_AUTH_SOCK="$LATEST_SSH_SOCK"
        echo "🔑 Updated SSH_AUTH_SOCK to: $SSH_AUTH_SOCK"
    else
        echo "⚠️ Warning: No valid SSH agent socket found in /tmp/ssh-*"
        echo "   Current SSH_AUTH_SOCK: $SSH_AUTH_SOCK"
    fi
    
    # Run the experiment
    if python ctrl/main.py $EXPERIMENT_ARGS --sudo_password "$SUDO_PASSWORD"; then
        echo "✅ Experiment completed successfully"
        COMPLETED=$((COMPLETED + 1))
    else
        echo "❌ Experiment failed"
        FAILED=$((FAILED + 1))
    fi
    
    # Wait between experiments so sockets left in TIME_WAIT from the previous
    # run's TCP connections are released before the next experiment starts
    if [ $EXPERIMENT_NUM -lt ${#EXPERIMENTS[@]} ]; then
        echo "⏳ Waiting 120 seconds before next experiment..."
        sleep 120
    fi
    
    echo ""
done

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo ""
echo "========================================"
echo "Batch Experiment Complete!"
echo "========================================"
echo "Total Experiments: ${#EXPERIMENTS[@]}"
echo "Completed: $COMPLETED"
echo "Failed: $FAILED"
echo "Total Time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "========================================"

# Clear sudo password from environment
unset SUDO_PASSWORD
