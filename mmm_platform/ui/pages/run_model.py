"""
Model running page for MMM Platform.
"""

import os
import streamlit as st
import pandas as pd
import time
from datetime import datetime
from pathlib import Path

from mmm_platform.model.mmm import MMMWrapper
from mmm_platform.model.fitting import ModelFitter
from mmm_platform.api.client import EC2ModelClient, JobStatus
import numpy as np


def show():
    """Show the model running page."""
    st.title("Run Model")

    # Check for demo mode
    if st.session_state.get("demo_mode", False):
        st.info("**Demo Mode**: The model is already fitted with simulated data. Go to **Results** to explore!")
        st.stop()

    # Check prerequisites
    if st.session_state.get("current_data") is None:
        st.warning("Please upload data first!")
        st.stop()

    if st.session_state.get("current_config") is None:
        st.warning("Please configure the model first!")
        st.stop()

    config = st.session_state.current_config
    df = st.session_state.current_data

    # Check model state
    model_fitted = st.session_state.get("model_fitted", False)
    has_model = st.session_state.get("current_model") is not None

    # If model is already fitted, show summary stats and redirect to Results
    if model_fitted and has_model:
        wrapper = st.session_state.current_model
        st.success("Model is fitted! Go to **Results** to view the full analysis.")

        # Show quick summary stats
        st.subheader("Fit Summary")
        try:
            fit_stats = wrapper.get_fit_statistics()
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("R²", f"{fit_stats['r2']:.3f}")
            with col2:
                st.metric("MAPE", f"{fit_stats['mape']:.1f}%")
            with col3:
                st.metric("RMSE", f"{fit_stats['rmse']:.1f}")
            with col4:
                if fit_stats.get('fit_duration_seconds'):
                    st.metric("Fit Time", f"{fit_stats['fit_duration_seconds']:.0f}s")
                else:
                    st.metric("Fit Time", "N/A")
        except Exception:
            st.info("Fit statistics not available")

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.info("Navigate to **Results** in the sidebar to explore model output.")
        with col2:
            if st.button("New Model", type="primary", width="stretch"):
                st.session_state.model_fitted = False
                st.session_state.current_model = None
                st.rerun()
        return
    elif model_fitted and not has_model:
        st.warning("Model was fitted but model object is missing. Please run again.")
        st.session_state.model_fitted = False
    elif has_model and not model_fitted:
        st.session_state.model_fitted = True
        st.rerun()

    # Show configuration summary
    st.subheader("Configuration Summary")

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Model Name", config.name)
    with col2:
        st.metric("Channels", len(config.channels))
    with col3:
        st.metric("Controls", len(config.controls))
    with col4:
        st.metric("Dummy Variables", len(config.dummy_variables))
    with col5:
        st.metric("Observations", len(df))

    # Show channels
    with st.expander("Channels"):
        for ch in config.channels:
            st.write(f"- **{ch.name}**: ROI prior [{ch.roi_prior_low}, {ch.roi_prior_mid}, {ch.roi_prior_high}], Adstock: {ch.adstock_type}")

    # Show dummy variables if any
    if config.dummy_variables:
        with st.expander("Dummy Variables", expanded=True):
            for dv in config.dummy_variables:
                st.write(f"- **{dv.name}**: {dv.start_date} to {dv.end_date} ({dv.sign_constraint.value})")
    else:
        # Check if there are dummy variables in config_state but not in config
        config_state_dummies = st.session_state.get("config_state", {}).get("dummy_variables", [])
        if config_state_dummies:
            st.warning(f"You have {len(config_state_dummies)} dummy variable(s) configured but they're NOT in the current config! "
                      "Go back to Configure Model and click 'Build Configuration' to include them.")

    # Show sampling settings
    with st.expander("Sampling Settings"):
        st.write(f"- Draws: {config.sampling.draws}")
        st.write(f"- Tune: {config.sampling.tune}")
        st.write(f"- Chains: {config.sampling.chains}")
        st.write(f"- Target Accept: {config.sampling.target_accept}")

    st.markdown("---")

    # Run Options
    st.subheader("Run Options")

    # Compute location selection
    run_location = st.radio(
        "Where to run the model?",
        options=["local", "ec2"],
        format_func=lambda x: "Local (slow, 2-4 hours)" if x == "local" else "EC2 GPU (fast, 5-15 min)",
        horizontal=True,
        index=1 if _ec2_available() else 0,
    )

    # EC2 Configuration
    if run_location == "ec2":
        ec2_ok = _show_ec2_config()
        if not ec2_ok:
            st.stop()

    col1, col2 = st.columns(2)
    with col1:
        quick_run = st.checkbox(
            "Quick run (fewer samples)",
            value=False,
            help="Use fewer samples for faster testing"
        )

    with col2:
        save_model = st.checkbox(
            "Save model after fitting",
            value=True,
        )

    # Adjust sampling for quick run
    if quick_run:
        draws = 500
        tune = 500
        chains = 2
        st.info(f"Quick run: {draws} draws, {tune} tune, {chains} chains")
    else:
        draws = config.sampling.draws
        tune = config.sampling.tune
        chains = config.sampling.chains

    # Estimated time
    if run_location == "ec2":
        est_time = "5-15 minutes" if not quick_run else "2-5 minutes"
    else:
        est_time = "2-4 hours" if not quick_run else "30-60 minutes"
    st.info(f"Estimated time: {est_time}")

    # Run button
    st.markdown("---")

    if st.button("Run Model", type="primary", width="stretch"):
        if run_location == "ec2":
            run_model_ec2(config, df, draws, tune, chains, save_model)
        else:
            run_model_local(config, df, draws, tune, chains, save_model)


def _ec2_available() -> bool:
    """Check if EC2 API URL is configured."""
    # Check env var or config file
    ec2_url = os.getenv("EC2_API_URL")
    if ec2_url:
        return True
    # Check config file
    config_file = Path("deploy/ec2_api_url.txt")
    return config_file.exists()


def _get_ec2_url() -> str:
    """Get EC2 API URL from env or config."""
    ec2_url = os.getenv("EC2_API_URL")
    if ec2_url:
        return ec2_url
    config_file = Path("deploy/ec2_api_url.txt")
    if config_file.exists():
        return config_file.read_text().strip()
    return ""


def _show_ec2_config() -> bool:
    """Show EC2 configuration UI. Returns True if ready to run."""
    st.markdown("#### EC2 API Configuration")

    # Get configured URL (from env/file) as default
    default_url = _get_ec2_url()

    # Show help if no URL configured
    if not default_url:
        with st.expander("How to set up EC2 API", expanded=True):
            st.markdown("""
            1. Start the EC2 instance with the MMM API running
            2. Get the public IP address
            3. Enter the API URL below (e.g., `http://1.2.3.4:8000`)

            **Or** set the `EC2_API_URL` environment variable.
            """)

    # Always show editable text input (pre-filled with default if available)
    ec2_url = st.text_input(
        "EC2 API URL",
        value=default_url,
        placeholder="http://your-ec2-ip:8000",
        help="The URL of the FastAPI server running on EC2"
    )

    if not ec2_url:
        st.warning("Enter the EC2 API URL to continue")
        return False

    st.session_state.ec2_api_url = ec2_url

    # Show option to save if URL differs from saved config
    if default_url and ec2_url != default_url:
        st.info("URL differs from saved configuration")
        if st.button("Save as Default", key="save_ec2_url"):
            config_file = Path("deploy/ec2_api_url.txt")
            config_file.parent.mkdir(parents=True, exist_ok=True)
            config_file.write_text(ec2_url)
            st.success("URL saved to deploy/ec2_api_url.txt")
            st.rerun()
    elif not default_url and ec2_url:
        # No saved URL yet, offer to save
        if st.button("Save URL for Future Use", key="save_ec2_url_new"):
            config_file = Path("deploy/ec2_api_url.txt")
            config_file.parent.mkdir(parents=True, exist_ok=True)
            config_file.write_text(ec2_url)
            st.success("URL saved to deploy/ec2_api_url.txt")
            st.rerun()

    # Connection test button
    col1, col2 = st.columns([1, 2])
    with col1:
        test_clicked = st.button("Test Connection", type="primary", key="test_ec2_conn")

    if test_clicked:
        with st.spinner("Testing connection..."):
            try:
                client = EC2ModelClient(ec2_url)
                if client.health_check():
                    st.session_state.ec2_connected = True
                    st.session_state.ec2_connected_url = ec2_url
                    st.success("Connected to EC2 API!")
                else:
                    st.session_state.ec2_connected = False
                    st.error("EC2 API health check failed - is the server running?")
            except Exception as e:
                st.session_state.ec2_connected = False
                st.error(f"Connection failed: {e}")

    # Check if already connected to this URL
    if (st.session_state.get("ec2_connected") and
        st.session_state.get("ec2_connected_url") == ec2_url):
        st.success("EC2 connection verified")
        return True
    else:
        st.warning("Test connection before running model")
        return False


def run_model_ec2(config, df, draws, tune, chains, save_model):
    """Run model on EC2 via FastAPI."""
    ec2_url = st.session_state.get("ec2_api_url")
    if not ec2_url:
        st.error("EC2 API URL not configured!")
        return

    progress_container = st.container()

    with progress_container:
        st.subheader("Model Fitting Progress (EC2)")

        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            client = EC2ModelClient(ec2_url)

            # Build channel configs
            channels = []
            for ch in config.channels:
                channels.append({
                    "name": ch.name,
                    "roi_prior_low": ch.roi_prior_low,
                    "roi_prior_mid": ch.roi_prior_mid,
                    "roi_prior_high": ch.roi_prior_high,
                    "adstock_type": ch.adstock_type.value if hasattr(ch.adstock_type, 'value') else str(ch.adstock_type),
                })

            # Build control configs
            controls = []
            for ctrl in config.controls:
                controls.append({
                    "name": ctrl.name,
                    "expected_sign": ctrl.sign_constraint.value if hasattr(ctrl.sign_constraint, 'value') else str(ctrl.sign_constraint),
                })

            # Build dummy variable configs
            dummy_variables = []
            for dv in config.dummy_variables:
                dummy_variables.append({
                    "name": dv.name,
                    "start_date": dv.start_date,
                    "end_date": dv.end_date,
                    "categories": dv.categories if hasattr(dv, 'categories') else {},
                    "sign_constraint": dv.sign_constraint.value if hasattr(dv.sign_constraint, 'value') else str(dv.sign_constraint),
                })

            # Data config
            data_config = {
                "target_column": config.data.target_column,
                "date_column": config.data.date_column,
                "spend_scale": config.data.spend_scale,
                "revenue_scale": config.data.revenue_scale,
            }

            # Sampling config
            sampling_config = {
                "draws": draws,
                "tune": tune,
                "chains": chains,
                "target_accept": config.sampling.target_accept,
                "sampler": getattr(config.sampling, 'sampler', 'nutpie'),
            }

            status_text.text("Submitting job to EC2...")
            progress_bar.progress(5)

            # Submit job
            job_id = client.submit_model_run(
                model_name=config.name,
                data=df,
                channels=channels,
                controls=controls,
                dummy_variables=dummy_variables,
                data_config=data_config,
                sampling_config=sampling_config,
            )

            st.info(f"Job submitted: `{job_id}`")

            # Poll for completion
            start_time = time.time()
            max_wait = 3600  # 1 hour max

            while True:
                try:
                    status = client.get_status(job_id)

                    # Update progress
                    progress_bar.progress(int(status.progress * 100))
                    status_text.text(f"{status.message} ({status.progress*100:.0f}%)")

                    if status.status == JobStatus.COMPLETED:
                        break

                    if status.status == JobStatus.FAILED:
                        st.error(f"Job failed: {status.error}")
                        return

                except Exception as poll_error:
                    # Log but continue polling - might be temporary network issue
                    status_text.text(f"Connection issue, retrying... ({poll_error})")

                elapsed = time.time() - start_time
                if elapsed > max_wait:
                    st.error(f"Timeout: Job did not complete within {max_wait/60:.0f} minutes")
                    return

                time.sleep(5)  # Poll every 5 seconds

            total_time = time.time() - start_time

            # Get results
            results = client.get_results(job_id)

            status_text.text("Downloading model data from EC2...")
            progress_bar.progress(90)

            # Download all model data
            import arviz as az
            import json
            import numpy as np
            from pathlib import Path

            download_dir = Path("ec2_results") / job_id
            download_dir.mkdir(parents=True, exist_ok=True)

            # Download inference data
            idata_path = download_dir / "inference_data.nc"
            client.download_inference_data(job_id, str(idata_path))

            status_text.text("Downloading model state from EC2...")
            progress_bar.progress(92)

            # Download model data bundle (df_scaled, df_raw, model_state)
            try:
                model_data_paths = client.download_model_data(job_id, str(download_dir))
            except Exception as download_err:
                st.warning(f"Could not download model data bundle: {download_err}. Falling back to local rebuild.")
                model_data_paths = None

            status_text.text("Loading model data...")
            progress_bar.progress(95)

            # Load the inference data
            idata = az.from_netcdf(str(idata_path))

            # Create MMMWrapper with the downloaded data
            wrapper = MMMWrapper(config)
            wrapper.idata = idata
            wrapper.fitted_at = datetime.now()
            wrapper.fit_duration_seconds = total_time

            # Load the exact data that was used on EC2
            if model_data_paths and Path(model_data_paths["df_scaled"]).exists():
                # Use the exact df_scaled from EC2
                wrapper.df_scaled = pd.read_parquet(model_data_paths["df_scaled"])
                wrapper.df_raw = pd.read_parquet(model_data_paths["df_raw"])

                # Load model state
                with open(model_data_paths["model_state"], "r") as f:
                    model_state = json.load(f)

                wrapper.control_cols = model_state.get("control_cols")
                wrapper.lam_vec = np.array(model_state["lam_vec"]) if model_state.get("lam_vec") else None
                # beta_mu/beta_sigma can be scalars or arrays
                beta_mu = model_state.get("beta_mu")
                beta_sigma = model_state.get("beta_sigma")
                wrapper.beta_mu = np.array(beta_mu) if isinstance(beta_mu, list) else beta_mu
                wrapper.beta_sigma = np.array(beta_sigma) if isinstance(beta_sigma, list) else beta_sigma

                # Build model with the correct data
                try:
                    wrapper.build_model()
                    wrapper.mmm.idata = idata
                    wrapper.mmm.is_fitted_ = True
                except Exception as build_error:
                    st.warning(f"Could not build model: {build_error}")
            else:
                # Fallback: prepare data locally (may have slight differences)
                wrapper.df_raw = df.copy()
                try:
                    wrapper.prepare_data()
                    wrapper.build_model()
                    wrapper.mmm.idata = idata
                    wrapper.mmm.is_fitted_ = True
                except Exception as prep_error:
                    st.warning(f"Could not fully prepare wrapper: {prep_error}")

            # Store in session state (same as local model)
            st.session_state.current_model = wrapper
            st.session_state.model_fitted = True
            st.session_state.ec2_job_id = job_id
            # Don't set ec2_mode - we now have a full wrapper like local mode

            progress_bar.progress(100)
            status_text.text("Complete!")

            # Show results
            st.success(f"Model fitted on EC2 in {total_time:.1f} seconds!")

            fit_stats = results.fit_statistics or {}

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                r2 = fit_stats.get("r2", "N/A")
                st.metric("R²", f"{r2:.3f}" if isinstance(r2, (int, float)) else r2)
            with col2:
                mape = fit_stats.get("mape", "N/A")
                st.metric("MAPE", f"{mape:.1f}%" if isinstance(mape, (int, float)) else mape)
            with col3:
                rmse = fit_stats.get("rmse", "N/A")
                st.metric("RMSE", f"{rmse:.1f}" if isinstance(rmse, (int, float)) else rmse)
            with col4:
                st.metric("Location", "EC2")

            # Save model if requested
            if save_model:
                from mmm_platform.model.persistence import ModelPersistence, get_models_dir
                import json

                models_dir = get_models_dir()
                save_dir = models_dir / f"{config.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                try:
                    ModelPersistence.save(wrapper, save_dir, include_data=True)

                    # Also save session state for config restoration
                    session_state = {
                        "category_columns": st.session_state.get("category_columns", []),
                        "config_state": st.session_state.get("config_state", {}),
                        "date_column": st.session_state.get("date_column"),
                        "target_column": st.session_state.get("target_column"),
                        "detected_channels": st.session_state.get("detected_channels", []),
                        "dayfirst": st.session_state.get("dayfirst", False),
                    }
                    with open(save_dir / "session_state.json", "w") as f:
                        json.dump(session_state, f, indent=2)

                    st.info(f"Model saved to: {save_dir}")
                except Exception as e:
                    st.warning(f"Could not save model: {e}")

            st.balloons()

            # Rerun to show the fitted model view
            time.sleep(1)
            st.rerun()

        except Exception as e:
            progress_bar.progress(100)
            status_text.text("Error!")
            st.error(f"EC2 model fitting failed: {str(e)}")
            st.exception(e)


def run_model_local(config, df, draws, tune, chains, save_model):
    """Run the model fitting process locally."""

    progress_container = st.container()

    with progress_container:
        st.subheader("Model Fitting Progress (Local)")

        progress_bar = st.progress(0)
        status_text = st.empty()

        status_text.text("Initializing model...")
        progress_bar.progress(5)

        try:
            wrapper = MMMWrapper(config)

            status_text.text("Loading and validating data...")
            progress_bar.progress(10)

            wrapper.df_raw = df.copy()

            # Ensure date column is datetime
            date_col = config.data.date_column
            if date_col in wrapper.df_raw.columns:
                wrapper.df_raw[date_col] = pd.to_datetime(
                    wrapper.df_raw[date_col],
                    dayfirst=config.data.dayfirst
                )

            status_text.text("Preparing data (scaling, transforms)...")
            progress_bar.progress(20)
            wrapper.prepare_data()

            status_text.text("Building PyMC model...")
            progress_bar.progress(30)
            wrapper.build_model()

            status_text.text(f"Fitting model ({draws} draws, {tune} tune, {chains} chains)...")
            status_text.text("This may take several minutes...")
            progress_bar.progress(40)

            start_time = time.time()

            wrapper.idata = wrapper.mmm.fit(
                wrapper._get_feature_dataframe(),
                wrapper.df_scaled[config.data.target_column],
                draws=draws,
                tune=tune,
                chains=chains,
                target_accept=config.sampling.target_accept,
                random_seed=config.sampling.random_seed,
            )

            wrapper.fitted_at = datetime.now()
            wrapper.fit_duration_seconds = time.time() - start_time

            progress_bar.progress(90)
            status_text.text("Computing results...")

            st.session_state.current_model = wrapper
            st.session_state.model_fitted = True

            fit_stats = wrapper.get_fit_statistics()
            convergence = ModelFitter.check_convergence(wrapper.idata)

            progress_bar.progress(100)
            status_text.text("Complete!")

            st.success(f"Model fitted successfully in {wrapper.fit_duration_seconds:.1f} seconds!")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("R2", f"{fit_stats['r2']:.3f}")
            with col2:
                st.metric("MAPE", f"{fit_stats['mape']:.1f}%")
            with col3:
                st.metric("RMSE", f"{fit_stats['rmse']:.1f}")
            with col4:
                convergence_status = "Yes" if convergence["converged"] else "No"
                st.metric("Converged", convergence_status)

            if convergence["warnings"]:
                st.warning("Convergence warnings:")
                for warn in convergence["warnings"]:
                    st.write(f"- {warn}")

            if save_model:
                from mmm_platform.model.persistence import ModelPersistence, get_models_dir
                import json

                models_dir = get_models_dir()
                save_dir = models_dir / f"{config.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                ModelPersistence.save(wrapper, save_dir, include_data=True)

                # Also save session state for config restoration
                session_state = {
                    "category_columns": st.session_state.get("category_columns", []),
                    "config_state": st.session_state.get("config_state", {}),
                    "date_column": st.session_state.get("date_column"),
                    "target_column": st.session_state.get("target_column"),
                    "detected_channels": st.session_state.get("detected_channels", []),
                    "dayfirst": st.session_state.get("dayfirst", False),
                }
                with open(save_dir / "session_state.json", "w") as f:
                    json.dump(session_state, f, indent=2)

                st.info(f"Model saved to: {save_dir}")

            st.balloons()

            # Rerun to show the fitted model view
            time.sleep(1)
            st.rerun()

        except Exception as e:
            progress_bar.progress(100)
            status_text.text("Error!")
            st.error(f"Model fitting failed: {str(e)}")
            st.exception(e)
