#!/bin/bash
# =============================================================================
# EC2 Setup Script for MMM Platform
#
# This script launches a GPU-enabled EC2 instance for fast PyMC model fitting.
#
# Prerequisites:
#   - AWS CLI installed and configured (aws configure)
#   - Appropriate IAM permissions for EC2
#
# Usage:
#   chmod +x ec2_setup.sh
#   ./ec2_setup.sh
# =============================================================================

set -e

# =============================================================================
# CONFIGURATION - EDIT THESE VALUES
# =============================================================================

# AWS Region
AWS_REGION="us-west-2"

# Instance type (g4dn.xlarge has NVIDIA T4 GPU, good price/performance)
INSTANCE_TYPE="g4dn.xlarge"

# AMI - Deep Learning AMI (Ubuntu 20.04) with CUDA pre-installed
# This AMI ID is for us-west-2, update for your region
# Find AMIs: aws ec2 describe-images --owners amazon --filters "Name=name,Values=Deep Learning AMI GPU PyTorch*Ubuntu 20.04*" --query 'Images[*].[ImageId,Name]' --output table
AMI_ID="ami-0a7d4d5d8d8d8d8d8"  # Placeholder - will be looked up

# Key pair name (must exist in your AWS account)
KEY_NAME="mmm-platform-key"

# Security group name
SECURITY_GROUP_NAME="mmm-platform-sg"

# Instance name tag
INSTANCE_NAME="mmm-platform-gpu"

# Root volume size (GB)
VOLUME_SIZE=100

# Spot instance (much cheaper, but can be interrupted)
USE_SPOT="true"
SPOT_MAX_PRICE="0.25"  # Max hourly price for spot

# =============================================================================
# FUNCTIONS
# =============================================================================

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

error() {
    echo "[ERROR] $1" >&2
    exit 1
}

# Get the latest Deep Learning AMI
get_dl_ami() {
    log "Finding latest Deep Learning AMI..."
    AMI_ID=$(aws ec2 describe-images \
        --region $AWS_REGION \
        --owners amazon \
        --filters "Name=name,Values=Deep Learning OSS Nvidia Driver AMI GPU PyTorch*Ubuntu 20.04*" \
                  "Name=state,Values=available" \
        --query 'Images | sort_by(@, &CreationDate) | [-1].ImageId' \
        --output text)

    if [ "$AMI_ID" == "None" ] || [ -z "$AMI_ID" ]; then
        # Try alternative AMI pattern
        AMI_ID=$(aws ec2 describe-images \
            --region $AWS_REGION \
            --owners amazon \
            --filters "Name=name,Values=Deep Learning AMI GPU PyTorch*Ubuntu 20.04*" \
                      "Name=state,Values=available" \
            --query 'Images | sort_by(@, &CreationDate) | [-1].ImageId' \
            --output text)
    fi

    if [ "$AMI_ID" == "None" ] || [ -z "$AMI_ID" ]; then
        error "Could not find Deep Learning AMI. Please specify AMI_ID manually."
    fi

    log "Using AMI: $AMI_ID"
}

# Create key pair if it doesn't exist
create_key_pair() {
    if ! aws ec2 describe-key-pairs --region $AWS_REGION --key-names $KEY_NAME >/dev/null 2>&1; then
        log "Creating key pair: $KEY_NAME"
        aws ec2 create-key-pair \
            --region $AWS_REGION \
            --key-name $KEY_NAME \
            --query 'KeyMaterial' \
            --output text > "${KEY_NAME}.pem"
        chmod 400 "${KEY_NAME}.pem"
        log "Key pair saved to ${KEY_NAME}.pem - KEEP THIS FILE SAFE!"
    else
        log "Key pair $KEY_NAME already exists"
    fi
}

# Create security group if it doesn't exist
create_security_group() {
    # Check if security group exists
    SG_ID=$(aws ec2 describe-security-groups \
        --region $AWS_REGION \
        --filters "Name=group-name,Values=$SECURITY_GROUP_NAME" \
        --query 'SecurityGroups[0].GroupId' \
        --output text 2>/dev/null || echo "None")

    if [ "$SG_ID" == "None" ] || [ -z "$SG_ID" ]; then
        log "Creating security group: $SECURITY_GROUP_NAME"

        # Get default VPC
        VPC_ID=$(aws ec2 describe-vpcs \
            --region $AWS_REGION \
            --filters "Name=is-default,Values=true" \
            --query 'Vpcs[0].VpcId' \
            --output text)

        SG_ID=$(aws ec2 create-security-group \
            --region $AWS_REGION \
            --group-name $SECURITY_GROUP_NAME \
            --description "Security group for MMM Platform" \
            --vpc-id $VPC_ID \
            --query 'GroupId' \
            --output text)

        # Allow SSH from anywhere (restrict in production!)
        aws ec2 authorize-security-group-ingress \
            --region $AWS_REGION \
            --group-id $SG_ID \
            --protocol tcp \
            --port 22 \
            --cidr 0.0.0.0/0

        log "Security group created: $SG_ID"
    else
        log "Security group already exists: $SG_ID"
    fi
}

# Launch instance
launch_instance() {
    log "Launching EC2 instance..."

    # User data script to install dependencies
    USER_DATA=$(cat <<'USERDATA'
#!/bin/bash
set -e

# Log everything
exec > >(tee /var/log/user-data.log) 2>&1

echo "Starting setup..."

# Update system
sudo apt-get update -y

# Install Python 3.11 if not present
if ! command -v python3.11 &> /dev/null; then
    sudo add-apt-repository ppa:deadsnakes/ppa -y
    sudo apt-get update -y
    sudo apt-get install -y python3.11 python3.11-venv python3.11-dev
fi

# Create working directory
mkdir -p /home/ubuntu/mmm_platform
chown ubuntu:ubuntu /home/ubuntu/mmm_platform

# Create setup complete flag
touch /home/ubuntu/.setup_complete

echo "Base setup complete!"
USERDATA
)

    USER_DATA_BASE64=$(echo "$USER_DATA" | base64 -w 0)

    if [ "$USE_SPOT" == "true" ]; then
        log "Requesting spot instance (max price: \$$SPOT_MAX_PRICE/hr)..."

        # Create launch specification
        LAUNCH_SPEC=$(cat <<EOF
{
    "ImageId": "$AMI_ID",
    "InstanceType": "$INSTANCE_TYPE",
    "KeyName": "$KEY_NAME",
    "SecurityGroupIds": ["$SG_ID"],
    "BlockDeviceMappings": [
        {
            "DeviceName": "/dev/sda1",
            "Ebs": {
                "VolumeSize": $VOLUME_SIZE,
                "VolumeType": "gp3",
                "DeleteOnTermination": true
            }
        }
    ],
    "UserData": "$USER_DATA_BASE64"
}
EOF
)

        # Request spot instance
        SPOT_REQUEST_ID=$(aws ec2 request-spot-instances \
            --region $AWS_REGION \
            --spot-price "$SPOT_MAX_PRICE" \
            --instance-count 1 \
            --type "one-time" \
            --launch-specification "$LAUNCH_SPEC" \
            --query 'SpotInstanceRequests[0].SpotInstanceRequestId' \
            --output text)

        log "Spot request ID: $SPOT_REQUEST_ID"
        log "Waiting for spot instance to be fulfilled..."

        # Wait for spot request to be fulfilled
        aws ec2 wait spot-instance-request-fulfilled \
            --region $AWS_REGION \
            --spot-instance-request-ids $SPOT_REQUEST_ID

        INSTANCE_ID=$(aws ec2 describe-spot-instance-requests \
            --region $AWS_REGION \
            --spot-instance-request-ids $SPOT_REQUEST_ID \
            --query 'SpotInstanceRequests[0].InstanceId' \
            --output text)
    else
        # Launch on-demand instance
        INSTANCE_ID=$(aws ec2 run-instances \
            --region $AWS_REGION \
            --image-id $AMI_ID \
            --instance-type $INSTANCE_TYPE \
            --key-name $KEY_NAME \
            --security-group-ids $SG_ID \
            --block-device-mappings "DeviceName=/dev/sda1,Ebs={VolumeSize=$VOLUME_SIZE,VolumeType=gp3,DeleteOnTermination=true}" \
            --user-data "$USER_DATA" \
            --query 'Instances[0].InstanceId' \
            --output text)
    fi

    log "Instance ID: $INSTANCE_ID"

    # Tag the instance
    aws ec2 create-tags \
        --region $AWS_REGION \
        --resources $INSTANCE_ID \
        --tags "Key=Name,Value=$INSTANCE_NAME"

    # Wait for instance to be running
    log "Waiting for instance to be running..."
    aws ec2 wait instance-running \
        --region $AWS_REGION \
        --instance-ids $INSTANCE_ID

    # Get public IP
    PUBLIC_IP=$(aws ec2 describe-instances \
        --region $AWS_REGION \
        --instance-ids $INSTANCE_ID \
        --query 'Reservations[0].Instances[0].PublicIpAddress' \
        --output text)

    log "Instance is running!"
    log "Public IP: $PUBLIC_IP"

    # Save instance info
    cat > instance_info.txt <<EOF
INSTANCE_ID=$INSTANCE_ID
PUBLIC_IP=$PUBLIC_IP
AWS_REGION=$AWS_REGION
KEY_NAME=$KEY_NAME
INSTANCE_TYPE=$INSTANCE_TYPE
EOF

    log "Instance info saved to instance_info.txt"
    echo ""
    echo "============================================="
    echo "EC2 Instance Ready!"
    echo "============================================="
    echo "Instance ID: $INSTANCE_ID"
    echo "Public IP:   $PUBLIC_IP"
    echo "Instance:    $INSTANCE_TYPE"
    echo ""
    echo "Connect with:"
    echo "  ssh -i ${KEY_NAME}.pem ubuntu@$PUBLIC_IP"
    echo ""
    echo "Next steps:"
    echo "  1. Wait 2-3 minutes for instance setup to complete"
    echo "  2. Run: ./deploy_model.sh"
    echo "============================================="
}

# =============================================================================
# MAIN
# =============================================================================

log "Starting EC2 setup for MMM Platform..."

# Check AWS CLI
if ! command -v aws &> /dev/null; then
    error "AWS CLI not found. Please install it first."
fi

# Check AWS credentials
if ! aws sts get-caller-identity >/dev/null 2>&1; then
    error "AWS credentials not configured. Run 'aws configure' first."
fi

get_dl_ami
create_key_pair
create_security_group
launch_instance

log "Setup complete!"
