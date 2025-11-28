#!/bin/bash
# =============================================================================
# EC2 Initial Setup Script
#
# Run this ONCE on a new EC2 instance to set up everything.
# After this, GitHub Actions will handle all updates automatically.
#
# Usage (run on EC2):
#   curl -sSL https://raw.githubusercontent.com/YOUR_ORG/mmm_platform/main/deploy/ec2_initial_setup.sh | bash
#
# Or copy-paste the commands below.
# =============================================================================

set -e

echo "=========================================="
echo "MMM Platform - EC2 Initial Setup"
echo "=========================================="

# Configuration - UPDATE THESE
GITHUB_REPO="https://github.com/YOUR_ORG/mmm_platform.git"  # Change this!
PROJECT_DIR="$HOME/mmm_platform"

# Update system
echo "Updating system..."
sudo apt update && sudo apt install -y python3-pip python3-venv git

# Clone repository
echo "Cloning repository..."
if [ -d "$PROJECT_DIR" ]; then
    echo "Directory exists, pulling latest..."
    cd "$PROJECT_DIR"
    git pull
else
    git clone "$GITHUB_REPO" "$PROJECT_DIR"
    cd "$PROJECT_DIR"
fi

# Create virtual environment
echo "Setting up Python environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Install package in dev mode
pip install -e .

# Verify installation
echo "Verifying installation..."
python -c "import pymc; print('PyMC:', pymc.__version__)"
python -c "from mmm_platform.model.mmm import MMMWrapper; print('MMM Platform: OK')"

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Add GitHub secrets (see deploy/README.md)"
echo "2. Push to main branch to trigger auto-deploy"
echo ""
echo "To manually update: cd ~/mmm_platform && git pull"
echo "=========================================="
