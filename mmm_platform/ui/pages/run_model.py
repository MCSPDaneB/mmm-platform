"""
Model running page for MMM Platform.
"""

import streamlit as st
import pandas as pd
import time
from datetime import datetime
from pathlib import Path

from mmm_platform.model.mmm import MMMWrapper
from mmm_platform.model.fitting import ModelFitter


def show():
    """Show the model running page."""
    st.title("Run Model")

    # Check prerequisites
    if st.session_state.get("current_data") is None:
        st.warning("Please upload data first!")
        st.stop()

    if st.session_state.get("current_config") is None:
        st.warning("Please configure the model first!")
        st.stop()

    config = st.session_state.current_config
    df = st.session_state.current_data

    # Show configuration summary
    st.subheader("Configuration Summary")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Model Name", config.name)
    with col2:
        st.metric("Channels", len(config.channels))
    with col3:
        st.metric("Controls", len(config.controls))
    with col4:
        st.metric("Observations", len(df))

    # Show channels
    with st.expander("Channels"):
        for ch in config.channels:
            st.write(f"- **{ch.name}**: ROI prior [{ch.roi_prior_low}, {ch.roi_prior_mid}, {ch.roi_prior_high}], Adstock: {ch.adstock_type}")

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

    if st.button("Run Model", type="primary", use_container_width=True):
        if run_location == "ec2":
            run_model_ec2(config, df, draws, tune, chains, save_model)
        else:
            run_model_local(config, df, draws, tune, chains, save_model)


def _ec2_available() -> bool:
    """Check if EC2 config exists."""
    return Path("deploy/instance_info.txt").exists()


def _show_ec2_config() -> bool:
    """Show EC2 configuration UI. Returns True if ready to run."""
    from mmm_platform.core.ec2_runner import EC2Config, EC2Runner, get_ec2_config_from_file

    st.markdown("#### EC2 Configuration")

    # Try to load from file
    ec2_config = get_ec2_config_from_file()

    if ec2_config is None:
        st.warning("EC2 not configured. Please set up your EC2 instance first.")

        with st.expander("How to set up EC2"):
            st.markdown("""
            1. Go to [AWS EC2 Console](https://console.aws.amazon.com/ec2)
            2. Launch a **g4dn.xlarge** instance with Deep Learning AMI
            3. Download the key pair (.pem file)
            4. Enter the details below
            """)

        # Manual configuration
        col1, col2 = st.columns(2)
        with col1:
            ec2_ip = st.text_input("EC2 Public IP", placeholder="1.2.3.4")
        with col2:
            key_file = st.text_input("Key file path", placeholder="C:/path/to/key.pem")

        if ec2_ip and key_file:
            ec2_config = EC2Config(host=ec2_ip, key_file=key_file)
            st.session_state.ec2_config = ec2_config
        else:
            return False
    else:
        st.success(f"EC2 configured: `{ec2_config.host}`")
        st.session_state.ec2_config = ec2_config

        # Test connection button
        if st.button("Test Connection"):
            with st.spinner("Testing connection..."):
                runner = EC2Runner(ec2_config)
                connected, msg = runner.test_connection()
                if connected:
                    st.success(f"Connected to EC2!")
                    # Check GPU
                    gpu_ok, gpu_msg = runner.check_gpu()
                    if gpu_ok:
                        st.success("GPU available!")
                    else:
                        st.warning(f"GPU check: {gpu_msg}")
                else:
                    st.error(f"Connection failed: {msg}")
                    return False

    return True


def run_model_ec2(config, df, draws, tune, chains, save_model):
    """Run model on EC2 GPU instance."""
    from mmm_platform.core.ec2_runner import EC2Runner

    ec2_config = st.session_state.get("ec2_config")
    if ec2_config is None:
        st.error("EC2 not configured!")
        return

    progress_container = st.container()

    with progress_container:
        st.subheader("Model Fitting Progress (EC2 GPU)")

        progress_bar = st.progress(0)
        status_text = st.empty()

        def update_progress(pct, msg):
            progress_bar.progress(pct / 100)
            status_text.text(msg)

        try:
            runner = EC2Runner(ec2_config)

            # Convert config to dict for serialization
            config_dict = config.model_dump()

            start_time = time.time()

            # Run on EC2
            results = runner.fit_model(
                config_dict=config_dict,
                data_df=df,
                draws=draws,
                tune=tune,
                chains=chains,
                progress_callback=update_progress
            )

            total_time = time.time() - start_time

            # Load results
            import arviz as az

            idata_path = results.get("idata_path")
            if idata_path and Path(idata_path).exists():
                idata = az.from_netcdf(idata_path)

                # Create a wrapper to store results
                wrapper = MMMWrapper(config)
                wrapper.df_raw = df.copy()
                wrapper.idata = idata
                wrapper.fitted_at = datetime.now()
                wrapper.fit_duration_seconds = results.get("fit_stats", {}).get(
                    "fit_duration_seconds", total_time
                )

                # Store in session state
                st.session_state.current_model = wrapper
                st.session_state.model_fitted = True
                st.session_state.ec2_results = results

                # Show results
                st.success(f"Model fitted on EC2 in {wrapper.fit_duration_seconds:.1f} seconds!")

                fit_stats = results.get("fit_stats", {})

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    r2 = fit_stats.get("r2", "N/A")
                    st.metric("R2", f"{r2:.3f}" if isinstance(r2, float) else r2)
                with col2:
                    mape = fit_stats.get("mape", "N/A")
                    st.metric("MAPE", f"{mape:.1f}%" if isinstance(mape, float) else mape)
                with col3:
                    rmse = fit_stats.get("rmse", "N/A")
                    st.metric("RMSE", f"{rmse:.1f}" if isinstance(rmse, float) else rmse)
                with col4:
                    st.metric("Location", "EC2 GPU")

                # Save if requested
                if save_model:
                    from mmm_platform.model.persistence import ModelPersistence

                    save_dir = Path("saved_models") / f"{config.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    try:
                        ModelPersistence.save(wrapper, save_dir)
                        st.info(f"Model saved to: {save_dir}")
                    except Exception as e:
                        st.warning(f"Could not save model: {e}")

                st.balloons()
                st.markdown("---")
                st.success("Navigate to **Results** in the sidebar to explore the model output.")

            else:
                st.error("Results file not found. Check EC2 logs.")

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
                from mmm_platform.model.persistence import ModelPersistence

                save_dir = Path("saved_models") / f"{config.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                ModelPersistence.save(wrapper, save_dir)
                st.info(f"Model saved to: {save_dir}")

            st.balloons()
            st.markdown("---")
            st.success("Navigate to **Results** in the sidebar to explore the model output.")

        except Exception as e:
            progress_bar.progress(100)
            status_text.text("Error!")
            st.error(f"Model fitting failed: {str(e)}")
            st.exception(e)
