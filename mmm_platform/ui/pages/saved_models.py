"""
Saved models and configurations page for MMM Platform.

Shows two sections:
1. Saved Configurations - configs that haven't been run yet
2. Fitted Models - models that have been trained
"""

import streamlit as st
from pathlib import Path
from datetime import datetime

from mmm_platform.model.persistence import (
    ModelPersistence,
    ConfigPersistence,
    get_workspace_dir,
    set_workspace_dir,
    get_configs_dir,
    get_models_dir,
    restore_config_to_session,
)


def show():
    """Show the saved models page."""
    st.title("💾 Saved Models & Configurations")

    # Workspace directory setting
    _show_workspace_setting()

    st.markdown("---")

    # Two-column layout for the two sections
    tab1, tab2 = st.tabs(["📋 Saved Configurations", "📦 Fitted Models"])

    with tab1:
        _show_configs_section()

    with tab2:
        _show_models_section()


def _show_workspace_setting():
    """Show workspace directory setting."""
    with st.expander("⚙️ Workspace Directory", expanded=False):
        current_workspace = get_workspace_dir()

        st.write(f"**Current workspace:** `{current_workspace.absolute()}`")

        new_workspace = st.text_input(
            "Change workspace directory",
            value=str(current_workspace),
            key="workspace_dir_input",
            help="All configurations and models will be saved to this directory",
        )

        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("Update", key="update_workspace_btn"):
                try:
                    set_workspace_dir(new_workspace)
                    st.success(f"Workspace updated to: {new_workspace}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error setting workspace: {e}")

        with col2:
            st.caption(
                "Configs saved to: `{workspace}/configs/`  \n"
                "Models saved to: `{workspace}/models/`"
            )


def _show_configs_section():
    """Show saved configurations section."""
    st.subheader("Saved Configurations")
    st.caption("Configurations that have been saved but not yet fitted")

    # Save current config button
    if st.session_state.get("current_data") is not None:
        with st.expander("💾 Save Current Configuration", expanded=False):
            config_name = st.text_input(
                "Configuration name",
                value=st.session_state.get("current_config").name
                if st.session_state.get("current_config")
                and hasattr(st.session_state.get("current_config"), "name")
                else "My Config",
                key="save_config_name",
            )

            if st.button("Save Configuration", key="save_config_btn", type="primary"):
                _save_current_config(config_name)

    # List saved configs
    configs = ConfigPersistence.list_saved_configs()

    if not configs:
        st.info("No saved configurations found. Configure a model and save it here.")
        return

    st.write(f"Found **{len(configs)}** saved configuration(s)")

    # Display configs
    for config in configs:
        config_path = config.get("path", "")
        config_name = config.get("name", "Unknown")
        created_at = config.get("created_at", "Unknown")[:16].replace("T", " ")

        with st.expander(f"📋 {config_name} - {created_at}"):
            col1, col2 = st.columns(2)

            with col1:
                st.write(f"**Created:** {created_at}")
                st.write(f"**Channels:** {config.get('n_channels', 'N/A')}")
                st.write(f"**Controls:** {config.get('n_controls', 'N/A')}")

            with col2:
                st.write(f"**Data rows:** {config.get('n_rows', 'N/A')}")
                st.write(f"**Path:** `{config_path}`")

            # Actions
            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button(
                    "📂 Load Config", key=f"load_config_{config_path}", type="primary"
                ):
                    _load_config(config_path)

            with col2:
                pass  # Placeholder for future actions

            with col3:
                if st.button("🗑️ Delete", key=f"delete_config_{config_path}"):
                    _delete_path(config_path, "Configuration")


def _show_models_section():
    """Show fitted models section."""
    st.subheader("Fitted Models")
    st.caption("Models that have been trained and saved")

    # Get models directory
    models_dir = get_models_dir()

    # List saved models
    models = ModelPersistence.list_saved_models(models_dir)

    if not models:
        st.info(
            "No fitted models found. Run a model and it will be automatically saved here."
        )
        return

    st.write(f"Found **{len(models)}** fitted model(s)")

    # Display models
    for model in models:
        model_path = model.get("path", "")
        model_name = model.get("config_name", "Unknown")
        created_at = model.get("created_at", "Unknown")[:16].replace("T", " ")

        with st.expander(f"📦 {model_name} - {created_at}"):
            col1, col2 = st.columns(2)

            with col1:
                st.write(f"**Created:** {created_at}")
                fitted_at = model.get("fitted_at")
                if fitted_at:
                    st.write(f"**Fitted:** {fitted_at[:16].replace('T', ' ')}")
                duration = model.get("fit_duration_seconds", 0)
                if duration:
                    st.write(f"**Duration:** {duration:.1f}s")

            with col2:
                st.write(f"**Channels:** {model.get('n_channels', 'N/A')}")
                st.write(f"**Controls:** {model.get('n_controls', 'N/A')}")
                st.write(f"**Path:** `{model_path}`")

            # Actions
            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button(
                    "📂 Load Model", key=f"load_model_{model_path}", type="primary"
                ):
                    _load_model(model_path)

            with col2:
                if st.button("📊 View Results", key=f"view_model_{model_path}"):
                    _load_model(model_path, navigate_to_results=True)

            with col3:
                if st.button("🗑️ Delete", key=f"delete_model_{model_path}"):
                    _delete_path(model_path, "Model")


def _save_current_config(name: str):
    """Save the current configuration."""
    try:
        config = st.session_state.get("current_config")
        data = st.session_state.get("current_data")

        if data is None:
            st.error("No data loaded. Please upload data first.")
            return

        # Gather session state to save
        session_state = {
            "category_columns": st.session_state.get("category_columns", []),
            "config_state": st.session_state.get("config_state", {}),
            "date_column": st.session_state.get("date_column"),
            "target_column": st.session_state.get("target_column"),
            "detected_channels": st.session_state.get("detected_channels", []),
            "dayfirst": st.session_state.get("dayfirst", False),
        }

        path = ConfigPersistence.save(
            name=name,
            config=config,
            data=data,
            session_state=session_state,
        )

        st.success(f"Configuration saved to: {path}")
        st.rerun()

    except Exception as e:
        st.error(f"Error saving configuration: {e}")


def _load_config(path: str):
    """Load a saved configuration and restore to session."""
    try:
        config, data, session_state = ConfigPersistence.load(path)

        # Use restore_config_to_session to build updates
        updates = restore_config_to_session(config, data, session_state)

        # Apply updates to session state
        for key, value in updates.items():
            if value is not None:
                st.session_state[key] = value

        st.success(
            "Configuration loaded! Navigate to **Configure Model** to continue editing."
        )
        st.rerun()

    except Exception as e:
        st.error(f"Error loading configuration: {e}")


def _load_model(path: str, navigate_to_results: bool = False):
    """Load a saved model and restore config to session."""
    try:
        from mmm_platform.model.mmm import MMMWrapper

        wrapper = ModelPersistence.load(path, MMMWrapper)

        # Store model in session
        st.session_state.current_model = wrapper
        st.session_state.current_config = wrapper.config
        st.session_state.model_fitted = wrapper.idata is not None

        # Also restore data if available
        if wrapper.df_raw is not None:
            st.session_state.current_data = wrapper.df_raw
        elif wrapper.df_scaled is not None:
            st.session_state.current_data = wrapper.df_scaled

        # Try to load associated session state from config path
        # Models saved with new system should have config.json alongside
        model_path = Path(path)
        config_file = model_path / "config.json"

        if config_file.exists():
            # Check if there's session state saved with the model
            session_file = model_path / "session_state.json"
            if session_file.exists():
                import json

                with open(session_file, "r") as f:
                    session_state = json.load(f)

                # Restore session state
                if "category_columns" in session_state:
                    st.session_state.category_columns = session_state["category_columns"]
                if "config_state" in session_state:
                    st.session_state.config_state = session_state["config_state"]
                if session_state.get("date_column"):
                    st.session_state.date_column = session_state["date_column"]
                if session_state.get("target_column"):
                    st.session_state.target_column = session_state["target_column"]
                if session_state.get("detected_channels"):
                    st.session_state.detected_channels = session_state["detected_channels"]
                if "dayfirst" in session_state:
                    st.session_state.dayfirst = session_state["dayfirst"]

        if navigate_to_results:
            st.success("Model loaded! Navigating to Results...")
        else:
            st.success(
                "Model loaded! Navigate to **Results** to view analysis or **Configure Model** to see settings."
            )

        st.rerun()

    except Exception as e:
        st.error(f"Error loading model: {e}")


def _delete_path(path: str, item_type: str = "Item"):
    """Delete a saved model or config."""
    import shutil

    try:
        shutil.rmtree(path)
        st.success(f"{item_type} deleted")
        st.rerun()
    except Exception as e:
        st.error(f"Error deleting {item_type.lower()}: {e}")
