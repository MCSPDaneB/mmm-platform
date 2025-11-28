"""
EC2 Remote Model Runner

This module handles offloading model fitting to an EC2 GPU instance.
"""

import os
import json
import tempfile
import subprocess
import time
from pathlib import Path
from typing import Optional, Callable
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class EC2Config:
    """EC2 connection configuration."""
    host: str  # EC2 public IP or hostname
    key_file: str  # Path to .pem file
    user: str = "ubuntu"
    remote_dir: str = "/home/ubuntu/mmm_platform"
    python_path: str = "/home/ubuntu/mmm_platform/venv/bin/python"

    def ssh_command(self) -> list[str]:
        """Get base SSH command."""
        return [
            "ssh",
            "-i", self.key_file,
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "ConnectTimeout=10",
            f"{self.user}@{self.host}"
        ]

    def scp_command(self, local_path: str, remote_path: str) -> list[str]:
        """Get SCP upload command."""
        return [
            "scp",
            "-i", self.key_file,
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            local_path,
            f"{self.user}@{self.host}:{remote_path}"
        ]

    def scp_download_command(self, remote_path: str, local_path: str) -> list[str]:
        """Get SCP download command."""
        return [
            "scp",
            "-i", self.key_file,
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            f"{self.user}@{self.host}:{remote_path}",
            local_path
        ]


class EC2Runner:
    """Runs PyMC models on a remote EC2 instance."""

    def __init__(self, config: EC2Config):
        self.config = config
        self._connected = False

    def test_connection(self) -> tuple[bool, str]:
        """Test SSH connection to EC2."""
        try:
            cmd = self.config.ssh_command() + ["echo 'connected'"]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0 and "connected" in result.stdout:
                self._connected = True
                return True, "Connection successful"
            else:
                return False, f"Connection failed: {result.stderr}"
        except subprocess.TimeoutExpired:
            return False, "Connection timed out"
        except FileNotFoundError:
            return False, "SSH not found. Please install OpenSSH."
        except Exception as e:
            return False, f"Error: {str(e)}"

    def check_gpu(self) -> tuple[bool, str]:
        """Check if GPU is available on EC2."""
        try:
            cmd = self.config.ssh_command() + [
                f"{self.config.python_path} -c \"import jax; devs=jax.devices(); print('GPU:' + str(any('gpu' in str(d).lower() for d in devs)))\""
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if "GPU:True" in result.stdout:
                return True, "GPU available"
            else:
                return False, "GPU not detected"
        except Exception as e:
            return False, f"Error checking GPU: {str(e)}"

    def upload_file(self, local_path: str, remote_path: str) -> bool:
        """Upload a file to EC2."""
        try:
            cmd = self.config.scp_command(local_path, remote_path)
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            return result.returncode == 0
        except Exception as e:
            logger.error(f"Upload failed: {e}")
            return False

    def download_file(self, remote_path: str, local_path: str) -> bool:
        """Download a file from EC2."""
        try:
            cmd = self.config.scp_download_command(remote_path, local_path)
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            return result.returncode == 0
        except Exception as e:
            logger.error(f"Download failed: {e}")
            return False

    def run_command(self, command: str, timeout: int = 3600) -> tuple[int, str, str]:
        """Run a command on EC2."""
        try:
            cmd = self.config.ssh_command() + [command]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return -1, "", "Command timed out"
        except Exception as e:
            return -1, "", str(e)

    def fit_model(
        self,
        config_dict: dict,
        data_df,
        draws: int = 1500,
        tune: int = 1500,
        chains: int = 4,
        progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> dict:
        """
        Fit a model on EC2 and return results.

        Parameters
        ----------
        config_dict : dict
            Model configuration as dictionary
        data_df : pd.DataFrame
            Data to fit
        draws, tune, chains : int
            Sampling parameters
        progress_callback : callable
            Function to call with (progress_pct, message)

        Returns
        -------
        dict
            Results including idata path, fit stats, etc.
        """
        import pandas as pd

        def update_progress(pct, msg):
            if progress_callback:
                progress_callback(pct, msg)
            logger.info(f"[{pct}%] {msg}")

        update_progress(5, "Testing EC2 connection...")

        # Test connection
        connected, msg = self.test_connection()
        if not connected:
            raise ConnectionError(f"Cannot connect to EC2: {msg}")

        update_progress(10, "Connection OK. Preparing data...")

        # Create temp directory for files
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Save config
            config_path = tmpdir / "config.json"
            with open(config_path, 'w') as f:
                json.dump(config_dict, f, default=str)

            # Save data
            data_path = tmpdir / "data.csv"
            data_df.to_csv(data_path, index=False)

            # Save run parameters
            params = {
                "draws": draws,
                "tune": tune,
                "chains": chains,
                "config_file": "config.json",
                "data_file": "data.csv",
            }
            params_path = tmpdir / "params.json"
            with open(params_path, 'w') as f:
                json.dump(params, f)

            update_progress(15, "Uploading data to EC2...")

            # Create remote temp dir
            remote_tmp = f"{self.config.remote_dir}/tmp_run_{int(time.time())}"
            self.run_command(f"mkdir -p {remote_tmp}")

            # Upload files
            self.upload_file(str(config_path), f"{remote_tmp}/config.json")
            self.upload_file(str(data_path), f"{remote_tmp}/data.csv")
            self.upload_file(str(params_path), f"{remote_tmp}/params.json")

            update_progress(25, "Starting model fitting on EC2 GPU...")

            # Create and upload the runner script
            runner_script = self._create_runner_script()
            runner_path = tmpdir / "run_fit.py"
            with open(runner_path, 'w') as f:
                f.write(runner_script)
            self.upload_file(str(runner_path), f"{remote_tmp}/run_fit.py")

            # Run the model
            update_progress(30, f"Fitting model ({draws} draws, {tune} tune, {chains} chains)...")

            returncode, stdout, stderr = self.run_command(
                f"cd {remote_tmp} && {self.config.python_path} run_fit.py",
                timeout=7200  # 2 hour timeout
            )

            if returncode != 0:
                raise RuntimeError(f"Model fitting failed on EC2:\n{stderr}\n{stdout}")

            update_progress(85, "Downloading results...")

            # Download results
            results_dir = Path("ec2_results")
            results_dir.mkdir(exist_ok=True)

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            local_idata = results_dir / f"idata_{timestamp}.nc"
            local_summary = results_dir / f"summary_{timestamp}.csv"
            local_stats = results_dir / f"stats_{timestamp}.json"

            self.download_file(f"{remote_tmp}/results/idata.nc", str(local_idata))
            self.download_file(f"{remote_tmp}/results/summary.csv", str(local_summary))
            self.download_file(f"{remote_tmp}/results/stats.json", str(local_stats))

            update_progress(95, "Loading results...")

            # Load results
            results = {
                "idata_path": str(local_idata),
                "summary_path": str(local_summary),
            }

            # Load stats
            if local_stats.exists():
                with open(local_stats) as f:
                    results["fit_stats"] = json.load(f)

            # Cleanup remote
            self.run_command(f"rm -rf {remote_tmp}")

            update_progress(100, "Complete!")

            return results

    def _create_runner_script(self) -> str:
        """Create the Python script to run on EC2."""
        return '''#!/usr/bin/env python3
"""EC2 Model Fitting Script - Auto-generated"""

import json
import time
import numpy as np
import pandas as pd
from pathlib import Path

# Load parameters
with open("params.json") as f:
    params = json.load(f)

with open("config.json") as f:
    config_dict = json.load(f)

# Load data
df = pd.read_csv("data.csv")
print(f"Data loaded: {df.shape}")

# Check for GPU (optional)
gpu_available = False
try:
    import jax
    devices = jax.devices()
    gpu_available = any('gpu' in str(d).lower() for d in devices)
    print(f"JAX devices: {devices}")
    print(f"GPU available: {gpu_available}")
except ImportError:
    print("JAX not installed - using CPU with nutpie")

# Import PyMC
from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation
from pymc_extras.prior import Prior

# Reconstruct config
from mmm_platform.config.schema import ModelConfig
config = ModelConfig(**config_dict)

# Create wrapper
from mmm_platform.model.mmm import MMMWrapper

wrapper = MMMWrapper(config)
wrapper.df_raw = df

# Ensure date column is datetime
date_col = config.data.date_column
if date_col in wrapper.df_raw.columns:
    wrapper.df_raw[date_col] = pd.to_datetime(wrapper.df_raw[date_col])

# Prepare and build
print("Preparing data...")
wrapper.prepare_data()

print("Building model...")
wrapper.build_model()

# Determine best sampler
if gpu_available:
    sampler = "numpyro"
    print("Using numpyro (GPU)")
else:
    try:
        import nutpie
        sampler = "nutpie"
        print("Using nutpie (fast CPU)")
    except ImportError:
        sampler = "pymc"
        print("Using default PyMC sampler")

print(f"Fitting model: {params['draws']} draws, {params['tune']} tune, {params['chains']} chains")
start_time = time.time()

X = wrapper._get_feature_dataframe()
y = wrapper.df_scaled[config.data.target_column]

fit_kwargs = {
    "draws": params["draws"],
    "tune": params["tune"],
    "chains": params["chains"],
    "target_accept": config.sampling.target_accept,
    "random_seed": config.sampling.random_seed,
}

if sampler != "pymc":
    fit_kwargs["nuts_sampler"] = sampler

wrapper.idata = wrapper.mmm.fit(X, y, **fit_kwargs)

duration = time.time() - start_time
print(f"Model fitted in {duration:.1f} seconds ({duration/60:.1f} minutes)")

# Save results
results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

# Save idata
wrapper.idata.to_netcdf(results_dir / "idata.nc")

# Save summary
import arviz as az
summary = az.summary(wrapper.idata)
summary.to_csv(results_dir / "summary.csv")

# Save stats
try:
    stats = wrapper.get_fit_statistics()
    stats["fit_duration_seconds"] = duration
    with open(results_dir / "stats.json", "w") as f:
        json.dump(stats, f)
except Exception as e:
    print(f"Could not compute fit stats: {e}")
    with open(results_dir / "stats.json", "w") as f:
        json.dump({"fit_duration_seconds": duration}, f)

print("Results saved!")
'''


def get_ec2_config_from_file(config_path: str = "deploy/instance_info.txt") -> Optional[EC2Config]:
    """Load EC2 config from instance_info.txt file."""
    config_path = Path(config_path)
    if not config_path.exists():
        return None

    config = {}
    with open(config_path) as f:
        for line in f:
            if "=" in line:
                key, value = line.strip().split("=", 1)
                config[key] = value

    key_file = config_path.parent / f"{config.get('KEY_NAME', 'mmm-platform-key')}.pem"

    if "PUBLIC_IP" not in config:
        return None

    return EC2Config(
        host=config["PUBLIC_IP"],
        key_file=str(key_file),
    )
