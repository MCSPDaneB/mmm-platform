"""
Model execution utilities for MMM Platform.

Contains helper functions for running models locally and on EC2,
extracted from run_model.py for reuse in configure_model.py.
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
from mmm_platform.analysis.pre_fit_checks import run_pre_fit_checks


def ec2_available() -> bool:
    """Check if EC2 API URL is configured."""
    ec2_url = os.getenv("EC2_API_URL")
    if ec2_url:
        return True
    config_file = Path("deploy/ec2_api_url.txt")
    return config_file.exists()


def get_ec2_url() -> str:
    """Get EC2 API URL from env or config."""
    ec2_url = os.getenv("EC2_API_URL")
    if ec2_url:
        return ec2_url
    config_file = Path("deploy/ec2_api_url.txt")
    if config_file.exists():
        return config_file.read_text().strip()
    return ""


def show_ec2_config() -> bool:
    """Show EC2 configuration UI. Returns True if ready to run."""
    st.markdown("#### EC2 API Configuration")

    default_url = get_ec2_url()

    if not default_url:
        with st.expander("How to set up EC2 API", expanded=True):
            st.markdown("""
            1. Start the EC2 instance with the MMM API running
            2. Get the public IP address
            3. Enter the API URL below (e.g., `http://1.2.3.4:8000`)

            **Or** set the `EC2_API_URL` environment variable.
            """)

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

    if default_url and ec2_url != default_url:
        st.info("URL differs from saved configuration")
        if st.button("Save as Default", key="save_ec2_url"):
            config_file = Path("deploy/ec2_api_url.txt")
            config_file.parent.mkdir(parents=True, exist_ok=True)
            config_file.write_text(ec2_url)
            st.success("URL saved to deploy/ec2_api_url.txt")
            st.rerun()
    elif not default_url and ec2_url:
        if st.button("Save URL for Future Use", key="save_ec2_url_new"):
            config_file = Path("deploy/ec2_api_url.txt")
            config_file.parent.mkdir(parents=True, exist_ok=True)
            config_file.write_text(ec2_url)
            st.success("URL saved to deploy/ec2_api_url.txt")
            st.rerun()

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

    if (st.session_state.get("ec2_connected") and
        st.session_state.get("ec2_connected_url") == ec2_url):
        st.success("EC2 connection verified")
        return True
    else:
        st.warning("Test connection before running model")
        return False


def show_pre_fit_warnings(config, df):
    """Display pre-fit warnings if any issues detected."""
    try:
        warnings = run_pre_fit_checks(config, df)
        if warnings:
            with st.expander(f"⚠️ Pre-Fit Warnings ({len(warnings)})", expanded=False):
                # Explain what's being checked
                st.caption(
                    "**Checks:** Low spend (<3%), high zero periods (>50%), "
                    "high variability (CV>2), wide ROI priors (>20x range)"
                )

                # Build table data
                rows = []
                for w in warnings:
                    severity_icon = {"critical": "🔴", "warning": "⚠️", "info": "ℹ️"}.get(w.severity, "")
                    rows.append({
                        "Channel": w.channel,
                        "Severity": severity_icon,
                        "Issue": w.issue,
                        "Recommendation": w.recommendation
                    })

                # Display as table
                import pandas as pd
                df_warnings = pd.DataFrame(rows)
                st.dataframe(df_warnings, hide_index=True)
    except Exception:
        pass  # Silently skip if pre-fit checks fail


def run_model_ec2(config, df, draws, tune, chains, save_model):
    """Run model on EC2 via FastAPI."""
    import arviz as az
    import json
    import numpy as np

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

            # Build owned media configs
            owned_media = []
            for om in config.owned_media:
                owned_media.append({
                    "name": om.name,
                    "display_name": om.display_name,
                    "categories": om.categories,
                    "adstock_type": om.adstock_type.value if hasattr(om.adstock_type, 'value') else str(om.adstock_type),
                    "curve_sharpness_override": om.curve_sharpness_override,
                    "include_roi": om.include_roi,
                    "roi_prior_low": om.roi_prior_low,
                    "roi_prior_mid": om.roi_prior_mid,
                    "roi_prior_high": om.roi_prior_high,
                })

            # Build competitor configs
            competitors = []
            for comp in config.competitors:
                competitors.append({
                    "name": comp.name,
                    "display_name": comp.display_name,
                    "categories": comp.categories,
                    "adstock_type": comp.adstock_type.value if hasattr(comp.adstock_type, 'value') else str(comp.adstock_type),
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
                "target_scale": config.data.target_scale,
                "model_start_date": config.data.model_start_date,
                "model_end_date": config.data.model_end_date,
                "dayfirst": config.data.dayfirst,
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
                owned_media=owned_media,
                competitors=competitors,
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
                    progress_bar.progress(int(status.progress * 100))
                    status_text.text(f"{status.message} ({status.progress*100:.0f}%)")

                    if status.status == JobStatus.COMPLETED:
                        break

                    if status.status == JobStatus.FAILED:
                        st.error(f"Job failed: {status.error}")
                        return

                except Exception as poll_error:
                    status_text.text(f"Connection issue, retrying... ({poll_error})")

                elapsed = time.time() - start_time
                if elapsed > max_wait:
                    st.error(f"Timeout: Job did not complete within {max_wait/60:.0f} minutes")
                    return

                time.sleep(5)

            total_time = time.time() - start_time

            # Get results
            results = client.get_results(job_id)

            status_text.text("Downloading model data from EC2...")
            progress_bar.progress(90)

            download_dir = Path("ec2_results") / job_id
            download_dir.mkdir(parents=True, exist_ok=True)

            # Download inference data
            idata_path = download_dir / "inference_data.nc"
            client.download_inference_data(job_id, str(idata_path))

            status_text.text("Downloading model state from EC2...")
            progress_bar.progress(92)

            try:
                model_data_paths = client.download_model_data(job_id, str(download_dir))
            except Exception as download_err:
                st.warning(f"Could not download model data bundle: {download_err}. Falling back to local rebuild.")
                model_data_paths = None

            status_text.text("Loading model data...")
            progress_bar.progress(95)

            idata = az.from_netcdf(str(idata_path))

            wrapper = MMMWrapper(config)
            wrapper.idata = idata
            wrapper.fitted_at = datetime.now()
            wrapper.fit_duration_seconds = total_time

            if model_data_paths and Path(model_data_paths["df_scaled"]).exists():
                wrapper.df_scaled = pd.read_parquet(model_data_paths["df_scaled"])
                wrapper.df_raw = pd.read_parquet(model_data_paths["df_raw"])

                with open(model_data_paths["model_state"], "r") as f:
                    model_state = json.load(f)

                wrapper.control_cols = model_state.get("control_cols")
                wrapper.lam_vec = np.array(model_state["lam_vec"]) if model_state.get("lam_vec") else None
                beta_mu = model_state.get("beta_mu")
                beta_sigma = model_state.get("beta_sigma")
                wrapper.beta_mu = np.array(beta_mu) if isinstance(beta_mu, list) else beta_mu
                wrapper.beta_sigma = np.array(beta_sigma) if isinstance(beta_sigma, list) else beta_sigma

                try:
                    wrapper.build_model()
                    wrapper.mmm.idata = idata
                    wrapper.mmm.is_fitted_ = True
                except Exception as build_error:
                    st.warning(f"Could not build model: {build_error}")
            else:
                wrapper.df_raw = df.copy()
                try:
                    wrapper.prepare_data()
                    wrapper.build_model()
                    wrapper.mmm.idata = idata
                    wrapper.mmm.is_fitted_ = True
                except Exception as prep_error:
                    st.warning(f"Could not fully prepare wrapper: {prep_error}")

            st.session_state.current_model = wrapper
            st.session_state.model_fitted = True
            st.session_state.ec2_job_id = job_id

            progress_bar.progress(100)
            status_text.text("Complete!")

            st.success(f"Model fitted on EC2 in {total_time:.1f} seconds!")

            _show_fit_results(wrapper, results.fit_statistics or {}, results.convergence, draws, tune, chains, config)

            if save_model:
                _save_model(wrapper, config)

            st.balloons()

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
            ess_stats = ModelFitter.get_effective_sample_size(wrapper.idata)

            convergence_dict = {
                "converged": convergence["converged"],
                "divergences": convergence["divergences"],
                "high_rhat_params": convergence["high_rhat_params"],
                "warnings": convergence["warnings"],
                "ess_bulk_min": ess_stats.get("ess_bulk_min"),
                "ess_tail_min": ess_stats.get("ess_tail_min"),
                "ess_sufficient": ess_stats.get("sufficient", True),
            }

            progress_bar.progress(100)
            status_text.text("Complete!")

            st.success(f"Model fitted successfully in {wrapper.fit_duration_seconds:.1f} seconds!")

            _show_fit_results(wrapper, fit_stats, convergence_dict, draws, tune, chains, config)

            if save_model:
                _save_model(wrapper, config)

            st.balloons()

        except Exception as e:
            progress_bar.progress(100)
            status_text.text("Error!")
            st.error(f"Model fitting failed: {str(e)}")
            st.exception(e)


def _show_fit_results(wrapper, fit_stats, convergence, draws, tune, chains, config):
    """Display fit results and convergence diagnostics."""
    col1, col2, col3 = st.columns(3)
    with col1:
        r2 = fit_stats.get("r2", "N/A")
        st.metric("R²", f"{r2:.3f}" if isinstance(r2, (int, float)) else r2)
    with col2:
        mape = fit_stats.get("mape", "N/A")
        st.metric("MAPE", f"{mape:.1f}%" if isinstance(mape, (int, float)) else mape)
    with col3:
        if convergence:
            converged = convergence.get("converged", True)
            st.metric("Converged", "Yes" if converged else "No")
        else:
            st.metric("Fit Time", f"{wrapper.fit_duration_seconds:.0f}s" if wrapper.fit_duration_seconds else "N/A")

    if convergence:
        from mmm_platform.model.diagnostics import DiagnosticsAdvisor

        advisor = DiagnosticsAdvisor()
        sampling_config = {
            "draws": draws,
            "tune": tune,
            "chains": chains,
            "target_accept": config.sampling.target_accept if config.sampling else 0.9,
        }
        diagnostics = advisor.analyze_from_convergence_dict(convergence, sampling_config)

        if diagnostics:
            with st.expander("⚠️ Convergence Recommendations", expanded=True):
                for diag in diagnostics:
                    if diag.severity == "critical":
                        st.error(f"**{diag.issue}**: {diag.details}")
                    else:
                        st.warning(f"**{diag.issue}**: {diag.details}")

                    if diag.recommendations:
                        st.markdown("**Suggested changes:**")
                        for rec in diag.recommendations:
                            st.write(f"- **{rec.setting}**: {rec.current} → {rec.suggested} ({rec.reason})")
                st.info("💡 Adjust these settings in the Sampling Settings tab and re-run the model.")
        else:
            with st.expander("✅ Convergence Status", expanded=False):
                divergences = convergence.get("divergences", 0)
                st.success("**Model converged well!** No action required.")
                st.write(f"- Divergent transitions: {divergences}")
                st.write("- All R-hat values acceptable")
                st.write("- Effective sample sizes sufficient")


def _save_model(wrapper, config):
    """Save the fitted model to disk."""
    from mmm_platform.model.persistence import ModelPersistence, get_models_dir, get_client_models_dir
    import json

    client = getattr(config, 'client', None)
    if client:
        models_dir = get_client_models_dir(client)
    else:
        models_dir = get_models_dir()

    save_dir = models_dir / f"{config.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    try:
        ModelPersistence.save(wrapper, save_dir, include_data=True)

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
