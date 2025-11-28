#!/bin/bash
# =============================================================================
# Deploy and Run MMM Model on EC2
#
# This script:
#   1. Syncs your code to the EC2 instance
#   2. Sets up the Python environment with GPU support
#   3. Runs the model
#   4. Downloads results back to your local machine
#
# Prerequisites:
#   - EC2 instance running (run ec2_setup.sh first)
#   - instance_info.txt exists with instance details
#
# Usage:
#   ./deploy_model.sh [command]
#
# Commands:
#   setup    - Initial setup (sync code, install deps)
#   run      - Run the model
#   results  - Download results
#   ssh      - SSH into the instance
#   stop     - Stop the instance
#   terminate - Terminate the instance
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Load instance info
if [ -f "$SCRIPT_DIR/instance_info.txt" ]; then
    source "$SCRIPT_DIR/instance_info.txt"
else
    echo "Error: instance_info.txt not found. Run ec2_setup.sh first."
    exit 1
fi

KEY_FILE="$SCRIPT_DIR/${KEY_NAME}.pem"
SSH_OPTS="-i $KEY_FILE -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
SSH_CMD="ssh $SSH_OPTS ubuntu@$PUBLIC_IP"
SCP_CMD="scp $SSH_OPTS"
RSYNC_CMD="rsync -avz -e \"ssh $SSH_OPTS\""

REMOTE_DIR="/home/ubuntu/mmm_platform"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# =============================================================================
# COMMANDS
# =============================================================================

cmd_setup() {
    log "Setting up EC2 instance..."

    # Wait for instance to be ready
    log "Waiting for SSH to be available..."
    for i in {1..30}; do
        if $SSH_CMD "echo 'SSH ready'" 2>/dev/null; then
            break
        fi
        echo -n "."
        sleep 10
    done
    echo ""

    # Sync code
    log "Syncing code to EC2..."
    rsync -avz -e "ssh $SSH_OPTS" \
        --exclude 'venv' \
        --exclude '__pycache__' \
        --exclude '*.pyc' \
        --exclude '.git' \
        --exclude 'saved_models' \
        --exclude '*.idata' \
        --exclude 'deploy/*.pem' \
        --exclude 'deploy/instance_info.txt' \
        "$PROJECT_DIR/" "ubuntu@$PUBLIC_IP:$REMOTE_DIR/"

    # Setup Python environment with GPU support
    log "Setting up Python environment with JAX GPU support..."
    $SSH_CMD << 'REMOTE_SETUP'
set -e

cd /home/ubuntu/mmm_platform

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install JAX with CUDA support first
pip install --upgrade "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Install numpyro for GPU sampling
pip install numpyro

# Install project requirements
pip install -r requirements.txt

# Install nutpie as backup
pip install nutpie

# Verify GPU is available
python3 -c "
import jax
print('JAX devices:', jax.devices())
print('GPU available:', any('gpu' in str(d).lower() for d in jax.devices()))
"

echo "Setup complete!"
REMOTE_SETUP

    log "EC2 setup complete!"
}

cmd_sync() {
    log "Syncing code to EC2..."
    rsync -avz -e "ssh $SSH_OPTS" \
        --exclude 'venv' \
        --exclude '__pycache__' \
        --exclude '*.pyc' \
        --exclude '.git' \
        --exclude 'saved_models' \
        --exclude '*.idata' \
        --exclude 'deploy/*.pem' \
        --exclude 'deploy/instance_info.txt' \
        "$PROJECT_DIR/" "ubuntu@$PUBLIC_IP:$REMOTE_DIR/"
    log "Sync complete!"
}

cmd_run() {
    # Check for config and data file arguments
    CONFIG_FILE="${1:-config.yaml}"
    DATA_FILE="${2:-data.csv}"

    log "Running model on EC2..."
    log "Config: $CONFIG_FILE"
    log "Data: $DATA_FILE"

    # Sync latest code
    cmd_sync

    # Run the model
    $SSH_CMD << REMOTE_RUN
set -e

cd /home/ubuntu/mmm_platform
source venv/bin/activate

# Run the model script
python3 run_ec2_model.py --config "$CONFIG_FILE" --data "$DATA_FILE"

echo "Model run complete!"
REMOTE_RUN

    log "Model fitting complete!"

    # Download results
    cmd_results
}

cmd_run_script() {
    # Run a specific Python script
    SCRIPT_NAME="${1:-bevmo_offline_model.py}"

    log "Running $SCRIPT_NAME on EC2..."

    # Sync latest code
    cmd_sync

    # Run the script
    $SSH_CMD << REMOTE_RUN
set -e

cd /home/ubuntu/mmm_platform
source venv/bin/activate

# Set JAX to use GPU
export JAX_PLATFORM_NAME=gpu

# Run the script
python3 "$SCRIPT_NAME"

echo "Script complete!"
REMOTE_RUN

    log "Script complete!"
}

cmd_results() {
    log "Downloading results..."

    # Create local results directory
    mkdir -p "$PROJECT_DIR/results"

    # Download saved models
    rsync -avz -e "ssh $SSH_OPTS" \
        "ubuntu@$PUBLIC_IP:$REMOTE_DIR/saved_models/" \
        "$PROJECT_DIR/saved_models/" 2>/dev/null || true

    # Download any result files
    rsync -avz -e "ssh $SSH_OPTS" \
        "ubuntu@$PUBLIC_IP:$REMOTE_DIR/results/" \
        "$PROJECT_DIR/results/" 2>/dev/null || true

    log "Results downloaded to $PROJECT_DIR/results/"
}

cmd_ssh() {
    log "Connecting to EC2..."
    $SSH_CMD
}

cmd_status() {
    log "Checking instance status..."
    aws ec2 describe-instances \
        --region $AWS_REGION \
        --instance-ids $INSTANCE_ID \
        --query 'Reservations[0].Instances[0].[State.Name,PublicIpAddress,InstanceType]' \
        --output table
}

cmd_stop() {
    log "Stopping instance $INSTANCE_ID..."
    aws ec2 stop-instances \
        --region $AWS_REGION \
        --instance-ids $INSTANCE_ID
    log "Instance stopping. You won't be charged for compute while stopped."
    log "Note: You'll still be charged for EBS storage."
}

cmd_start() {
    log "Starting instance $INSTANCE_ID..."
    aws ec2 start-instances \
        --region $AWS_REGION \
        --instance-ids $INSTANCE_ID

    # Wait for running
    aws ec2 wait instance-running \
        --region $AWS_REGION \
        --instance-ids $INSTANCE_ID

    # Update public IP (it changes on stop/start)
    NEW_IP=$(aws ec2 describe-instances \
        --region $AWS_REGION \
        --instance-ids $INSTANCE_ID \
        --query 'Reservations[0].Instances[0].PublicIpAddress' \
        --output text)

    # Update instance_info.txt
    sed -i "s/PUBLIC_IP=.*/PUBLIC_IP=$NEW_IP/" "$SCRIPT_DIR/instance_info.txt"
    PUBLIC_IP=$NEW_IP

    log "Instance started. New IP: $PUBLIC_IP"
}

cmd_terminate() {
    read -p "Are you sure you want to TERMINATE instance $INSTANCE_ID? This cannot be undone. (yes/no): " confirm
    if [ "$confirm" == "yes" ]; then
        log "Terminating instance $INSTANCE_ID..."
        aws ec2 terminate-instances \
            --region $AWS_REGION \
            --instance-ids $INSTANCE_ID
        log "Instance terminating."
        rm -f "$SCRIPT_DIR/instance_info.txt"
    else
        log "Termination cancelled."
    fi
}

cmd_help() {
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  setup      - Initial setup (sync code, install deps)"
    echo "  sync       - Sync code to EC2"
    echo "  run        - Run model with config.yaml and data.csv"
    echo "  run-script - Run a specific Python script (e.g., ./deploy_model.sh run-script bevmo_offline_model.py)"
    echo "  results    - Download results from EC2"
    echo "  ssh        - SSH into the instance"
    echo "  status     - Check instance status"
    echo "  stop       - Stop the instance (saves money)"
    echo "  start      - Start a stopped instance"
    echo "  terminate  - Terminate the instance (permanent)"
    echo "  help       - Show this help"
    echo ""
    echo "Instance: $INSTANCE_ID"
    echo "IP: $PUBLIC_IP"
}

# =============================================================================
# MAIN
# =============================================================================

COMMAND="${1:-help}"
shift || true

case "$COMMAND" in
    setup)      cmd_setup "$@" ;;
    sync)       cmd_sync "$@" ;;
    run)        cmd_run "$@" ;;
    run-script) cmd_run_script "$@" ;;
    results)    cmd_results "$@" ;;
    ssh)        cmd_ssh "$@" ;;
    status)     cmd_status "$@" ;;
    stop)       cmd_stop "$@" ;;
    start)      cmd_start "$@" ;;
    terminate)  cmd_terminate "$@" ;;
    help|*)     cmd_help ;;
esac
