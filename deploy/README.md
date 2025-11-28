# EC2 Deployment Guide for MMM Platform

Run your MMM models on AWS EC2 - much faster than local!

## Quick Reference

| Instance | Cost | Model Time | Recommended For |
|----------|------|------------|-----------------|
| `c5.4xlarge` (16 CPU) | ~$0.68/hr | 30-60 min | **Start here** |
| `c5.2xlarge` (8 CPU) | ~$0.34/hr | 45-90 min | Budget option |
| `g4dn.xlarge` (GPU) | ~$0.52/hr | 5-15 min | Production (advanced setup) |

---

## Setup Guide (CPU Instance - Recommended)

### Step 1: Launch EC2 Instance

1. **Go to EC2 Console**: https://console.aws.amazon.com/ec2

2. **Click "Launch Instance"**

3. **Configure:**

   | Setting | Value |
   |---------|-------|
   | **Name** | `mmm-platform` |
   | **AMI** | Ubuntu Server 22.04 LTS (free tier eligible AMI is fine) |
   | **Instance type** | `c5.4xlarge` (16 vCPU, 32 GB RAM) |
   | **Key pair** | Click "Create new key pair" → Name: `mmm-platform-key` → **Download the `.pem` file!** |
   | **Network settings** | Check "Allow SSH traffic from" → Select "My IP" |
   | **Storage** | Change to `50 GB` gp3 |

4. **Click "Launch Instance"**

5. **Wait 1-2 minutes**, then click on your instance to see the **Public IPv4 address**

### Step 2: Connect to EC2

**Option A: Browser-based (easiest)**
- Click your instance → "Connect" button → "EC2 Instance Connect" tab → "Connect"

**Option B: SSH from PowerShell**
```powershell
ssh -i "C:\Users\YourName\Downloads\mmm-platform-key.pem" ubuntu@<PUBLIC_IP>
```

### Step 3: Install Dependencies on EC2

Run these commands on the EC2 instance:

```bash
# Update system
sudo apt update && sudo apt install -y python3-pip python3-venv

# Create project directory
cd ~
mkdir -p mmm_platform
cd mmm_platform

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyMC ecosystem with nutpie (fast CPU sampler)
pip install pymc pymc-marketing pymc-extras nutpie arviz pandas numpy scipy matplotlib pyyaml

# Verify installation
python -c "import pymc; print('PyMC version:', pymc.__version__)"
```

This takes about 5 minutes.

### Step 4: Upload Your Code to EC2

From your **local machine** (PowerShell), run:

```powershell
# Navigate to your project folder first
cd C:\localProjects\pymc_testing\mmm_platform

# Upload the mmm_platform package
scp -i "deploy\mmm-platform-key.pem" -r mmm_platform ubuntu@<PUBLIC_IP>:~/mmm_platform/

# Upload other needed files
scp -i "deploy\mmm-platform-key.pem" requirements.txt ubuntu@<PUBLIC_IP>:~/mmm_platform/
scp -i "deploy\mmm-platform-key.pem" run_ec2_model.py ubuntu@<PUBLIC_IP>:~/mmm_platform/
```

Then on EC2:
```bash
cd ~/mmm_platform
source venv/bin/activate
pip install -e .  # Install your package in dev mode
```

### Step 5: Configure Your Local Streamlit

1. **Move your `.pem` file** to `deploy/mmm-platform-key.pem`

2. **Create `deploy/instance_info.txt`** with this content:
   ```
   INSTANCE_ID=i-0abc123def456
   PUBLIC_IP=<YOUR_EC2_PUBLIC_IP>
   AWS_REGION=us-east-1
   KEY_NAME=mmm-platform-key
   ```
   (Replace with your actual instance ID and IP from the AWS console)

3. **Run your Streamlit app locally** - the EC2 option will appear on the Run Model page!

### Step 6: Stop Instance When Done (Save Money!)

**Important**: EC2 charges by the hour. When you're done:

1. Go to EC2 Console
2. Select your instance
3. Click "Instance state" → "Stop instance"

You can start it again later. The IP will change, so update `instance_info.txt`.

---

## Troubleshooting

### "Permission denied" when using SSH
```powershell
# On Windows, you may need to fix key permissions
icacls "deploy\mmm-platform-key.pem" /inheritance:r /grant:r "%USERNAME%:R"
```

### "Connection timed out"
- Check your instance is running (green "Running" state)
- Check security group allows SSH (port 22) from your IP
- Your IP may have changed - update security group with new IP

### Instance runs slow
- Check you selected `c5.4xlarge` not `t2.micro`
- Run `htop` on EC2 to see CPU usage

---

## Advanced: GPU Setup (Optional)

For even faster runs (5-15 min), use GPU:

1. Launch `g4dn.xlarge` with **Deep Learning AMI GPU PyTorch 2.0 (Ubuntu 20.04)**

2. Additional setup on EC2:
   ```bash
   pip install "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
   pip install numpyro

   # Verify GPU
   python -c "import jax; print(jax.devices())"
   # Should show: [CudaDevice(id=0)]
   ```

3. The code will automatically use GPU when available.

---

## Auto-Deploy with GitHub Actions

Once set up, every push to `main` automatically updates your EC2 instance.

### One-Time Setup

#### 1. Push your code to GitHub

```bash
cd C:\localProjects\pymc_testing\mmm_platform
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_ORG/mmm_platform.git
git push -u origin main
```

#### 2. Add GitHub Secrets

Go to your repo → **Settings** → **Secrets and variables** → **Actions** → **New repository secret**

Add these secrets:

| Secret Name | Value |
|-------------|-------|
| `EC2_HOST` | Your EC2 public IP (e.g., `1.2.3.4`) |
| `EC2_SSH_KEY` | Contents of your `.pem` file (copy the entire file content) |

#### 3. Initial EC2 Setup (run once on EC2)

Connect to EC2 and run:

```bash
# Update these values first!
GITHUB_REPO="https://github.com/YOUR_ORG/mmm_platform.git"

# Install git and clone
sudo apt update && sudo apt install -y python3-pip python3-venv git
git clone $GITHUB_REPO ~/mmm_platform
cd ~/mmm_platform

# Setup Python environment
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

#### 4. Test the Workflow

Push a change to `main`:
```bash
git add .
git commit -m "Test deploy"
git push
```

Check **Actions** tab in GitHub to see the deployment run.

### How It Works

```
Local code change
       ↓
   git push
       ↓
GitHub Actions triggers
       ↓
SSH into EC2 → git pull → pip install
       ↓
EC2 has latest code!
```

### Manual Deploy (if needed)

SSH into EC2 and run:
```bash
cd ~/mmm_platform
git pull
source venv/bin/activate
pip install -r requirements.txt
```

### Update EC2 IP After Restart

If you stop/start EC2, the IP changes. Update the `EC2_HOST` secret in GitHub.

---

## Option 2: AWS CLI (Automated)

### Prerequisites

1. **AWS CLI installed and configured**
   ```bash
   # Install AWS CLI
   # Windows: https://aws.amazon.com/cli/
   # Mac: brew install awscli

   # Configure credentials
   aws configure
   # Enter your AWS Access Key ID, Secret, and region (e.g., us-west-2)
   ```

2. **Git Bash or WSL** (for Windows users to run the shell scripts)

### Quick Start

#### Step 1: Launch EC2 Instance

```bash
cd deploy
chmod +x ec2_setup.sh
./ec2_setup.sh
```

This will:
- Create a key pair (`mmm-platform-key.pem`)
- Create a security group
- Launch a g4dn.xlarge spot instance (~$0.15-0.25/hr) with GPU
- Save instance info to `instance_info.txt`

### Step 2: Setup the Instance

Wait 2-3 minutes for the instance to initialize, then:

```bash
./deploy_model.sh setup
```

This will:
- Sync your code to EC2
- Install Python dependencies
- Install JAX with CUDA support
- Install numpyro for GPU sampling

### Step 3: Upload Your Data

```bash
# From the project root directory
scp -i deploy/mmm-platform-key.pem your_data.csv ubuntu@<EC2_IP>:/home/ubuntu/mmm_platform/data/
```

Or create a `data/` directory and sync:
```bash
./deploy_model.sh sync
```

### Step 4: Run Your Model

**Option A: Run the BevMo model**
```bash
./deploy_model.sh ssh
# Then on EC2:
cd /home/ubuntu/mmm_platform
source venv/bin/activate
python run_ec2_model.py --bevmo
```

**Option B: Run any Python script**
```bash
./deploy_model.sh run-script your_model.py
```

**Option C: Run with config file**
```bash
./deploy_model.sh run config.yaml data.csv
```

### Step 5: Download Results

```bash
./deploy_model.sh results
```

Results will be downloaded to `results/` directory.

### Step 6: Stop/Terminate Instance

```bash
# Stop (keeps data, no compute charges)
./deploy_model.sh stop

# Start again later
./deploy_model.sh start

# Terminate (deletes everything)
./deploy_model.sh terminate
```

## Commands Reference

| Command | Description |
|---------|-------------|
| `./deploy_model.sh setup` | Initial setup (sync + install deps) |
| `./deploy_model.sh sync` | Sync code to EC2 |
| `./deploy_model.sh ssh` | SSH into the instance |
| `./deploy_model.sh run-script <file.py>` | Run a Python script |
| `./deploy_model.sh results` | Download results |
| `./deploy_model.sh status` | Check instance status |
| `./deploy_model.sh stop` | Stop instance (save money) |
| `./deploy_model.sh start` | Start stopped instance |
| `./deploy_model.sh terminate` | Delete instance |

## Expected Performance

| Setup | Time for 1500 draws, 4 chains, 12 channels |
|-------|-------------------------------------------|
| Local CPU | 2-4 hours |
| EC2 CPU (c5.4xlarge) | 30-60 minutes |
| EC2 GPU (g4dn.xlarge) + numpyro | **5-15 minutes** |

## Cost Estimates

| Instance | On-Demand | Spot |
|----------|-----------|------|
| g4dn.xlarge | $0.526/hr | ~$0.15-0.20/hr |
| g4dn.2xlarge | $0.752/hr | ~$0.25-0.35/hr |
| c5.4xlarge (CPU) | $0.68/hr | ~$0.25-0.35/hr |

**Tip**: Use spot instances for ~70% savings. They can be interrupted but are fine for batch jobs.

## Troubleshooting

### "GPU not available"
- The Deep Learning AMI should have CUDA pre-installed
- Verify with: `nvidia-smi`
- Check JAX sees GPU: `python -c "import jax; print(jax.devices())"`

### "Connection refused" during SSH
- Wait 2-3 minutes after launch
- Check instance is running: `./deploy_model.sh status`
- Check security group allows SSH (port 22)

### "Permission denied" for key file
```bash
chmod 400 deploy/mmm-platform-key.pem
```

### Spot instance terminated
- Spot instances can be interrupted when demand is high
- Your data on EBS is preserved
- Just launch a new instance and re-attach or run setup again

## Manual GPU Setup (if needed)

If GPU isn't working, SSH in and run:

```bash
# Check NVIDIA driver
nvidia-smi

# Install JAX with CUDA
pip install --upgrade "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Verify
python -c "import jax; print(jax.devices())"
```

## Files Created

- `deploy/ec2_setup.sh` - Launches EC2 instance
- `deploy/deploy_model.sh` - Deployment and management commands
- `deploy/instance_info.txt` - Instance details (created after setup)
- `deploy/mmm-platform-key.pem` - SSH key (KEEP SAFE!)
- `run_ec2_model.py` - GPU-enabled model runner
