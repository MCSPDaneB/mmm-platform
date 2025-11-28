#!/usr/bin/env python3
"""
EC2 Model Runner for MMM Platform

This script is designed to run on EC2 with GPU support.
It uses JAX/numpyro for GPU-accelerated sampling.

Usage:
    python run_ec2_model.py --config config.yaml --data data.csv
    python run_ec2_model.py --script bevmo_offline_model.py
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Set JAX to use GPU before importing
os.environ["JAX_PLATFORM_NAME"] = "gpu"


def check_gpu():
    """Check if GPU is available."""
    try:
        import jax
        devices = jax.devices()
        gpu_available = any('gpu' in str(d).lower() for d in devices)
        print(f"JAX devices: {devices}")
        print(f"GPU available: {gpu_available}")
        return gpu_available
    except Exception as e:
        print(f"Error checking GPU: {e}")
        return False


def setup_numpyro_sampler():
    """Configure PyMC to use numpyro with JAX backend."""
    try:
        import pymc as pm
        # Set numpyro as the default sampler
        print("Configuring numpyro sampler for GPU...")
        return True
    except Exception as e:
        print(f"Error setting up numpyro: {e}")
        return False


def run_model_from_config(config_path: str, data_path: str):
    """Run model using configuration file."""
    import yaml
    import pandas as pd
    from mmm_platform.config.schema import ModelConfig
    from mmm_platform.model.mmm import MMMWrapper

    print(f"Loading config from: {config_path}")
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)

    config = ModelConfig(**config_dict)

    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path, parse_dates=[config.data.date_column], dayfirst=config.data.dayfirst)

    print(f"Data shape: {df.shape}")
    print(f"Channels: {len(config.channels)}")
    print(f"Controls: {len(config.controls)}")

    # Create wrapper
    wrapper = MMMWrapper(config)
    wrapper.df_raw = df

    # Prepare data
    print("Preparing data...")
    wrapper.prepare_data()

    # Build model
    print("Building model...")
    wrapper.build_model()

    # Fit with numpyro for GPU acceleration
    print("Fitting model with GPU acceleration...")
    start_time = time.time()

    X = wrapper._get_feature_dataframe()
    y = wrapper.df_scaled[config.data.target_column]

    # Use numpyro sampler for GPU
    wrapper.idata = wrapper.mmm.fit(
        X, y,
        draws=config.sampling.draws,
        tune=config.sampling.tune,
        chains=config.sampling.chains,
        target_accept=config.sampling.target_accept,
        random_seed=config.sampling.random_seed,
        nuts_sampler="numpyro",  # GPU-accelerated sampler
    )

    duration = time.time() - start_time
    print(f"Model fitted in {duration:.1f} seconds ({duration/60:.1f} minutes)")

    # Save results
    save_results(wrapper, config.name)

    return wrapper


def run_bevmo_model():
    """Run the BevMo offline model with GPU support."""
    import numpy as np
    import pandas as pd
    from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation
    from pymc_extras.prior import Prior

    print("=" * 60)
    print("BevMo Offline MMM - GPU Accelerated")
    print("=" * 60)

    # Configuration
    DATA_FILE_PATH = "data/bevmo_data_offline.csv"  # Update this path
    DATE_COL = "time"
    TARGET_COL = "kpi_offline_revenue"
    SPEND_SCALE = 1000.0

    SAMPLING_DRAWS = 1500
    SAMPLING_TUNE = 1500
    SAMPLING_CHAINS = 4
    TARGET_ACCEPT = 0.9
    RANDOM_SEED = 42

    # Channel columns
    CHANNEL_COLS = [
        "PaidMedia_Bing_Brand_spend",
        "PaidMedia_Criteo_Criteo_spend",
        "PaidMedia_GooglePMAX_GooglePMAX_spend",
        "PaidMedia_GoogleSearch_Brand_spend",
        "PaidMedia_GoogleSearch_Nonbrand_spend",
        "PaidMedia_Meta_Meta_spend",
        "PaidMedia_Pinterest_Pinterest_spend",
        "PaidMedia_Snapchat_Snapchat_spend",
        "PaidMedia_Tiktok_Tiktok_spend",
        "PaidMedia_Vistar_Vistar_spend",
        "PaidMedia_Yelp_Yelp_spend",
        "PaidMedia_Youtube_Youtube_spend"
    ]

    # Check for data file
    if not os.path.exists(DATA_FILE_PATH):
        print(f"Data file not found: {DATA_FILE_PATH}")
        print("Please upload your data file to the EC2 instance.")
        print("You can use: scp your_data.csv ubuntu@<EC2_IP>:/home/ubuntu/mmm_platform/data/")
        return None

    # Load data
    print(f"Loading data from {DATA_FILE_PATH}...")
    df = pd.read_csv(DATA_FILE_PATH, parse_dates=[DATE_COL], dayfirst=True)
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    df["t"] = np.arange(1, len(df) + 1)

    print(f"Data shape: {df.shape}")
    print(f"Date range: {df[DATE_COL].min()} to {df[DATE_COL].max()}")

    # Scale data
    SCALE_COLS = [TARGET_COL] + CHANNEL_COLS
    df_scaled = df.copy()
    for col in SCALE_COLS:
        if col in df_scaled.columns:
            df_scaled[col] = df_scaled[col] / SPEND_SCALE

    # Simple control columns (time trend)
    CONTROL_COLS = ["t"]

    # Build model
    print("Building model...")
    model_config = {
        "intercept": Prior("Normal", mu=0, sigma=2),
        "saturation_beta": Prior("HalfNormal", sigma=1, dims="channel"),
        "adstock_alpha": Prior("Beta", alpha=3, beta=3, dims="channel"),
        "saturation_lam": Prior("HalfNormal", sigma=1, dims="channel"),
        "gamma_control": Prior("Normal", mu=0, sigma=1, dims="control"),
        "likelihood": Prior("Normal", sigma=Prior("HalfNormal", sigma=2)),
    }

    mmm = MMM(
        model_config=model_config,
        sampler_config={"target_accept": TARGET_ACCEPT},
        date_column=DATE_COL,
        channel_columns=CHANNEL_COLS,
        control_columns=CONTROL_COLS,
        adstock=GeometricAdstock(l_max=8),
        saturation=LogisticSaturation(),
        yearly_seasonality=2,
    )

    # Prepare feature dataframe
    X = df_scaled[[DATE_COL] + CHANNEL_COLS + CONTROL_COLS]
    y = df_scaled[TARGET_COL]

    # Build model
    mmm.build_model(X, y)

    # Fit with numpyro (GPU)
    print("Fitting model with GPU (numpyro)...")
    print(f"Settings: {SAMPLING_DRAWS} draws, {SAMPLING_TUNE} tune, {SAMPLING_CHAINS} chains")
    start_time = time.time()

    idata = mmm.fit(
        X, y,
        draws=SAMPLING_DRAWS,
        tune=SAMPLING_TUNE,
        chains=SAMPLING_CHAINS,
        target_accept=TARGET_ACCEPT,
        random_seed=RANDOM_SEED,
        nuts_sampler="numpyro",  # GPU-accelerated
    )

    duration = time.time() - start_time
    print(f"\nModel fitted in {duration:.1f} seconds ({duration/60:.1f} minutes)")

    # Save results
    print("Saving results...")
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # Save inference data
    import arviz as az
    idata.to_netcdf(results_dir / "bevmo_offline_idata.nc")

    # Save summary
    summary = az.summary(idata)
    summary.to_csv(results_dir / "bevmo_offline_summary.csv")

    print(f"Results saved to {results_dir}/")
    print("\nModel Summary:")
    print(summary.head(20))

    return mmm, idata


def save_results(wrapper, model_name: str):
    """Save model results."""
    from pathlib import Path
    import arviz as az

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    prefix = f"{model_name}_{timestamp}"

    # Save inference data
    if wrapper.idata is not None:
        idata_path = results_dir / f"{prefix}_idata.nc"
        wrapper.idata.to_netcdf(idata_path)
        print(f"Saved inference data to {idata_path}")

        # Save summary
        summary = az.summary(wrapper.idata)
        summary_path = results_dir / f"{prefix}_summary.csv"
        summary.to_csv(summary_path)
        print(f"Saved summary to {summary_path}")

    # Save fit statistics
    try:
        stats = wrapper.get_fit_statistics()
        stats_path = results_dir / f"{prefix}_stats.txt"
        with open(stats_path, 'w') as f:
            for k, v in stats.items():
                f.write(f"{k}: {v}\n")
        print(f"Saved fit statistics to {stats_path}")
    except Exception as e:
        print(f"Could not save fit statistics: {e}")


def main():
    parser = argparse.ArgumentParser(description="Run MMM model on EC2 with GPU")
    parser.add_argument("--config", type=str, help="Path to config YAML file")
    parser.add_argument("--data", type=str, help="Path to data CSV file")
    parser.add_argument("--bevmo", action="store_true", help="Run BevMo offline model")
    parser.add_argument("--check-gpu", action="store_true", help="Just check GPU availability")

    args = parser.parse_args()

    # Check GPU
    print("Checking GPU availability...")
    gpu_available = check_gpu()

    if args.check_gpu:
        sys.exit(0 if gpu_available else 1)

    if not gpu_available:
        print("WARNING: GPU not available. Model will run on CPU (slower).")

    # Setup numpyro
    setup_numpyro_sampler()

    # Run model
    if args.bevmo:
        run_bevmo_model()
    elif args.config and args.data:
        run_model_from_config(args.config, args.data)
    else:
        print("Please specify either --bevmo or both --config and --data")
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
