"""
Saved models and configurations page for MMM Platform.

Shows two sections:
1. Saved Configurations - configs that haven't been run yet
2. Fitted Models - models that have been trained
"""

import json
import streamlit as st
from pathlib import Path
from datetime import datetime
from typing import Optional

from mmm_platform.model.persistence import (
    ModelPersistence,
    ConfigPersistence,
    get_workspace_dir,
    set_workspace_dir,
    get_configs_dir,
    get_models_dir,
    restore_config_to_session,
    list_clients,
)
from mmm_platform.config.schema import ModelConfig, CategoryColumnConfig


def show():
    """Show the saved models page."""
    st.title("💾 Saved Models & Configurations")

    # Workspace directory setting
    _show_workspace_setting()

    st.markdown("---")

    # Client filter
    selected_client = _show_client_filter()

    st.markdown("---")

    # Two-column layout for the two sections
    tab1, tab2 = st.tabs(["📋 Saved Configurations", "📦 Fitted Models"])

    with tab1:
        _show_configs_section(client=selected_client)

    with tab2:
        _show_models_section(client=selected_client)


def _show_client_filter():
    """Show client filter dropdown at top of page."""
    clients = list_clients()

    if not clients:
        st.info("No clients found. Create one in **Configure Model** page.")
        return "all"

    client_options = ["All Clients"] + clients
    selected = st.selectbox(
        "Filter by Client",
        options=client_options,
        key="saved_models_client_filter",
        help="Show configs and models for a specific client"
    )

    return "all" if selected == "All Clients" else selected


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


def _show_configs_section(client: str = "all"):
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

    # Filter controls
    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        show_archived_configs = st.checkbox(
            "Show archived",
            value=False,
            key="show_archived_configs",
            help="Show archived configurations"
        )
    with filter_col2:
        config_filter = st.selectbox(
            "Filter",
            options=["All", "Favorites only", "Non-favorites"],
            key="config_favorite_filter",
            help="Filter by favorite status"
        )

    # List saved configs (filtered by client)
    configs = ConfigPersistence.list_saved_configs(client=client)

    # Apply archive filter
    if not show_archived_configs:
        configs = [c for c in configs if not c.get("is_archived", False)]

    # Apply favorites filter
    if config_filter == "Favorites only":
        configs = [c for c in configs if c.get("is_favorite", False)]
    elif config_filter == "Non-favorites":
        configs = [c for c in configs if not c.get("is_favorite", False)]

    if not configs:
        if client != "all":
            st.info(f"No saved configurations found for client '{client}'.")
        else:
            st.info("No saved configurations found. Configure a model and save it here.")
        return

    st.write(f"Found **{len(configs)}** saved configuration(s)")

    # Display configs
    for config in configs:
        config_path = config.get("path", "")
        config_name = config.get("name", "Unknown")
        config_client = config.get("client", "")
        created_at = config.get("created_at", "Unknown")[:16].replace("T", " ")
        is_favorite = config.get("is_favorite", False)
        is_archived = config.get("is_archived", False)

        # Show client in expander title if viewing all
        favorite_icon = "⭐ " if is_favorite else ""
        title = f"📋 {favorite_icon}{config_name} - {created_at}"
        if client == "all" and config_client:
            title = f"📋 {favorite_icon}[{config_client}] {config_name} - {created_at}"
        if is_archived:
            title += " (archived)"

        with st.expander(title):
            col1, col2 = st.columns(2)

            with col1:
                st.write(f"**Created:** {created_at}")
                if config_client:
                    st.write(f"**Client:** {config_client}")
                st.write(f"**Channels:** {config.get('n_channels', 'N/A')}")
                st.write(f"**Controls:** {config.get('n_controls', 'N/A')}")

            with col2:
                st.write(f"**Data rows:** {config.get('n_rows', 'N/A')}")
                st.write(f"**Path:** `{config_path}`")

            # Actions row 1: Load, Favorite, Archive
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                if st.button(
                    "📂 Load Config", key=f"load_config_{config_path}", type="primary"
                ):
                    _load_config(config_path)

            with col2:
                fav_label = "☆ Unfavorite" if is_favorite else "⭐ Favorite"
                if st.button(fav_label, key=f"fav_config_{config_path}"):
                    ConfigPersistence.set_favorite(config_path, not is_favorite)
                    st.rerun()

            with col3:
                arch_label = "📤 Unarchive" if is_archived else "📦 Archive"
                if st.button(arch_label, key=f"arch_config_{config_path}"):
                    ConfigPersistence.set_archived(config_path, not is_archived)
                    st.rerun()

            with col4:
                if st.button("🗑️ Delete", key=f"delete_config_{config_path}"):
                    _delete_path(config_path, "Configuration")


def _show_models_section(client: str = "all"):
    """Show fitted models section."""
    st.subheader("Fitted Models")
    st.caption("Models that have been trained and saved")

    # Initialize comparison selection state
    if "models_to_compare" not in st.session_state:
        st.session_state.models_to_compare = []

    # Filter controls
    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        show_archived_models = st.checkbox(
            "Show archived",
            value=False,
            key="show_archived_models",
            help="Show archived models"
        )
    with filter_col2:
        model_filter = st.selectbox(
            "Filter",
            options=["All", "Favorites only", "Non-favorites"],
            key="model_favorite_filter",
            help="Filter by favorite status"
        )

    # List saved models (filtered by client)
    models = ModelPersistence.list_saved_models(client=client)

    # Apply archive filter
    if not show_archived_models:
        models = [m for m in models if not m.get("is_archived", False)]

    # Apply favorites filter
    if model_filter == "Favorites only":
        models = [m for m in models if m.get("is_favorite", False)]
    elif model_filter == "Non-favorites":
        models = [m for m in models if not m.get("is_favorite", False)]

    if not models:
        if client != "all":
            st.info(f"No fitted models found for client '{client}'.")
        else:
            st.info(
                "No fitted models found. Run a model and it will be automatically saved here."
            )
        return

    # Show compare button if 2 models selected
    n_selected = len(st.session_state.models_to_compare)
    if n_selected == 2:
        st.success(f"2 models selected for comparison")
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("🔍 Compare Models", type="primary", width="stretch"):
                st.session_state.active_comparison = st.session_state.models_to_compare.copy()
                st.rerun()
        with col2:
            if st.button("Clear Selection", width="stretch"):
                st.session_state.models_to_compare = []
                st.rerun()
    elif n_selected == 1:
        st.info("Select one more model to compare (2 required)")
    elif n_selected > 2:
        st.warning(f"{n_selected} models selected. Please select exactly 2 for comparison.")
        if st.button("Clear Selection"):
            st.session_state.models_to_compare = []
            st.rerun()

    st.write(f"Found **{len(models)}** fitted model(s)")

    # Display models
    for model in models:
        model_path = model.get("path", "")
        model_name = model.get("config_name", "Unknown")
        model_client = model.get("client", "")
        created_at = model.get("created_at", "Unknown")[:16].replace("T", " ")
        is_favorite = model.get("is_favorite", False)
        is_archived = model.get("is_archived", False)

        # Check if this model is selected for comparison
        is_selected = model_path in st.session_state.models_to_compare

        # Build expander title
        favorite_icon = "⭐ " if is_favorite else ""
        title = f"📦 {favorite_icon}{model_name} - {created_at}"
        if client == "all" and model_client:
            title = f"📦 {favorite_icon}[{model_client}] {model_name} - {created_at}"
        if is_selected:
            title += " ✓"
        if is_archived:
            title += " (archived)"

        # Header with checkbox
        header_col1, header_col2 = st.columns([0.05, 0.95])
        with header_col1:
            # Checkbox for comparison selection
            selected = st.checkbox(
                "Compare",
                value=is_selected,
                key=f"compare_check_{model_path}",
                help="Select for comparison",
                label_visibility="collapsed"
            )
            # Update selection state
            if selected and model_path not in st.session_state.models_to_compare:
                st.session_state.models_to_compare.append(model_path)
                st.rerun()
            elif not selected and model_path in st.session_state.models_to_compare:
                st.session_state.models_to_compare.remove(model_path)
                st.rerun()

        with header_col2:
            with st.expander(title):
                col1, col2 = st.columns(2)

                with col1:
                    st.write(f"**Created:** {created_at}")
                    if model_client:
                        st.write(f"**Client:** {model_client}")
                    fitted_at = model.get("fitted_at")
                    if fitted_at:
                        st.write(f"**Fitted:** {fitted_at[:16].replace('T', ' ')}")
                    st.write(f"**Channels:** {model.get('n_channels', 'N/A')}")
                    st.write(f"**Controls:** {model.get('n_controls', 'N/A')}")

                with col2:
                    # Fit statistics
                    r2 = model.get("r2")
                    mape = model.get("mape")
                    rmse = model.get("rmse")
                    duration = model.get("fit_duration_seconds", 0)

                    if r2 is not None:
                        st.write(f"**R²:** {r2:.2f}")
                    if mape is not None:
                        st.write(f"**MAPE:** {mape:.2f}%")
                    if rmse is not None:
                        st.write(f"**RMSE:** {rmse:.2f}")
                    if duration:
                        st.write(f"**Fit Time:** {duration:.0f}s")

                # Actions row 1
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

                # Actions row 2: Favorite, Archive
                col1, col2, col3 = st.columns(3)

                with col1:
                    fav_label = "☆ Unfavorite" if is_favorite else "⭐ Favorite"
                    if st.button(fav_label, key=f"fav_model_{model_path}"):
                        ModelPersistence.set_favorite(model_path, not is_favorite)
                        st.rerun()

                with col2:
                    arch_label = "📤 Unarchive" if is_archived else "📦 Archive"
                    if st.button(arch_label, key=f"arch_model_{model_path}"):
                        ModelPersistence.set_archived(model_path, not is_archived)
                        st.rerun()

                # Edit Categories expander
                with st.expander("Edit Categories", expanded=False):
                    _show_category_editor(model_path, model_path.replace("\\", "_").replace("/", "_"))


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

        # Store data and config in session
        st.session_state.current_data = data
        st.session_state.current_config = config
        st.session_state.model_fitted = False

        # ALWAYS rebuild config_state from ModelConfig to ensure UI displays correctly
        # This ensures channels, priors, and all settings are properly serialized
        st.session_state.config_state = _build_config_state_from_model(config)

        # Restore category_columns from config
        st.session_state.category_columns = [
            col.model_dump(mode="json") for col in config.category_columns
        ]

        # Sort category columns alphabetically for consistent display order
        st.session_state.category_columns.sort(key=lambda x: x["name"])

        # Populate category column options from actual values in channels/controls
        category_options: dict[str, set] = {}

        for ch in config.channels:
            for col_name, value in ch.categories.items():
                if col_name not in category_options:
                    category_options[col_name] = set()
                if value:
                    category_options[col_name].add(value)

        for om in config.owned_media:
            for col_name, value in om.categories.items():
                if col_name not in category_options:
                    category_options[col_name] = set()
                if value:
                    category_options[col_name].add(value)

        for comp in config.competitors:
            for col_name, value in comp.categories.items():
                if col_name not in category_options:
                    category_options[col_name] = set()
                if value:
                    category_options[col_name].add(value)

        for ctrl in config.controls:
            for col_name, value in ctrl.categories.items():
                if col_name not in category_options:
                    category_options[col_name] = set()
                if value:
                    category_options[col_name].add(value)

        # Update category_columns with collected options
        for cat_col in st.session_state.category_columns:
            col_name = cat_col["name"]
            if col_name in category_options:
                cat_col["options"] = sorted(list(category_options[col_name]))

        # Set date/target columns from config
        if config.data:
            st.session_state.date_column = config.data.date_column
            st.session_state.target_column = config.data.target_column
            st.session_state.dayfirst = config.data.dayfirst

        # Set detected_channels from config's channel list
        if config.channels:
            st.session_state.detected_channels = [ch.name for ch in config.channels]

        # Clear widget keys that need to be re-initialized from config
        # This forces configure_model.py to re-read from config_state instead of using stale widget values
        widget_keys_to_clear = [
            "channel_multiselect",
            "owned_media_multiselect",
            "competitor_multiselect",
            "control_multiselect",
            "channels_data_editor",
            "owned_media_data_editor",
            "competitor_data_editor",
            "controls_data_editor",
        ]
        for key in widget_keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]

        # Increment config version to force widget reset in configure_model page
        st.session_state["config_version"] = st.session_state.get("config_version", 0) + 1

        st.success(
            "Configuration loaded! Navigate to **Configure Model** to continue editing."
        )
        st.rerun()

    except Exception as e:
        st.error(f"Error loading configuration: {e}")


def _build_config_state_from_model(config: ModelConfig) -> dict:
    """Build config_state dictionary from a ModelConfig object.

    This ensures the UI can display all settings from a loaded model,
    even if no session_state.json file exists.
    """
    config_state = {
        "name": config.name,
        "client": config.client,
        "description": config.description,
    }

    # Data settings
    if config.data:
        config_state["date_col"] = config.data.date_column
        config_state["target_col"] = config.data.target_column
        config_state["revenue_scale"] = config.data.revenue_scale
        config_state["spend_scale"] = config.data.spend_scale
        config_state["dayfirst"] = config.data.dayfirst
        config_state["include_trend"] = config.data.include_trend
        config_state["model_start_date"] = config.data.model_start_date
        config_state["model_end_date"] = config.data.model_end_date
        config_state["brand"] = config.data.brand

    # Adstock settings - include all decay rates
    if config.adstock:
        config_state["l_max"] = config.adstock.l_max
        config_state["short_decay"] = config.adstock.short_decay
        config_state["medium_decay"] = config.adstock.medium_decay
        config_state["long_decay"] = config.adstock.long_decay

    # Saturation settings
    if config.saturation:
        config_state["curve_sharpness"] = config.saturation.curve_sharpness

    # Sampling settings - include all fields
    if config.sampling:
        config_state["draws"] = config.sampling.draws
        config_state["tune"] = config.sampling.tune
        config_state["chains"] = config.sampling.chains
        config_state["target_accept"] = config.sampling.target_accept
        config_state["random_seed"] = config.sampling.random_seed

    # Seasonality
    if config.seasonality:
        config_state["yearly_seasonality"] = config.seasonality.yearly_seasonality

    # Control prior
    if config.control_prior:
        config_state["control_prior_sigma"] = config.control_prior.sigma

    # Channels - include full config with ROI priors
    # Use mode="json" to serialize enums as strings (e.g., adstock_type: "medium" not AdstockType.MEDIUM)
    config_state["channels"] = [ch.model_dump(mode="json") for ch in config.channels]

    # Owned media
    config_state["owned_media"] = [om.model_dump(mode="json") for om in config.owned_media]

    # Competitors
    config_state["competitors"] = [comp.model_dump(mode="json") for comp in config.competitors]

    # Controls
    config_state["controls"] = [ctrl.model_dump(mode="json") for ctrl in config.controls]

    # Dummy variables
    config_state["dummy_variables"] = [dv.model_dump(mode="json") for dv in config.dummy_variables]

    # Month dummies
    if config.month_dummies:
        config_state["month_dummies"] = config.month_dummies.model_dump(mode="json")

    return config_state


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
        session_loaded = False

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
                    session_loaded = True
                if session_state.get("date_column"):
                    st.session_state.date_column = session_state["date_column"]
                if session_state.get("target_column"):
                    st.session_state.target_column = session_state["target_column"]
                if session_state.get("detected_channels"):
                    st.session_state.detected_channels = session_state["detected_channels"]
                if "dayfirst" in session_state:
                    st.session_state.dayfirst = session_state["dayfirst"]

        # ALWAYS rebuild config_state from ModelConfig to ensure UI displays correctly
        # This handles models saved before session_state.json existed
        st.session_state.config_state = _build_config_state_from_model(wrapper.config)

        # Also restore category_columns from config
        st.session_state.category_columns = [
            col.model_dump(mode="json") for col in wrapper.config.category_columns
        ]

        # Sort category columns alphabetically for consistent display order
        st.session_state.category_columns.sort(key=lambda x: x["name"])

        # Populate category column options from actual values in channels/controls
        # The config stores category_columns with empty options, but the actual values
        # are stored in each variable's categories dict. We need to collect these values
        # to populate the SelectboxColumn options in the UI.
        category_options: dict[str, set] = {}  # {column_name: set of values}

        for ch in wrapper.config.channels:
            for col_name, value in ch.categories.items():
                if col_name not in category_options:
                    category_options[col_name] = set()
                if value:
                    category_options[col_name].add(value)

        for om in wrapper.config.owned_media:
            for col_name, value in om.categories.items():
                if col_name not in category_options:
                    category_options[col_name] = set()
                if value:
                    category_options[col_name].add(value)

        for comp in wrapper.config.competitors:
            for col_name, value in comp.categories.items():
                if col_name not in category_options:
                    category_options[col_name] = set()
                if value:
                    category_options[col_name].add(value)

        for ctrl in wrapper.config.controls:
            for col_name, value in ctrl.categories.items():
                if col_name not in category_options:
                    category_options[col_name] = set()
                if value:
                    category_options[col_name].add(value)

        # Update category_columns with collected options
        for cat_col in st.session_state.category_columns:
            col_name = cat_col["name"]
            if col_name in category_options:
                cat_col["options"] = sorted(list(category_options[col_name]))

        # Set date/target columns from config
        if wrapper.config.data:
            st.session_state.date_column = wrapper.config.data.date_column
            st.session_state.target_column = wrapper.config.data.target_column
            st.session_state.dayfirst = wrapper.config.data.dayfirst

        # Set detected_channels from config's channel list
        if wrapper.config.channels:
            st.session_state.detected_channels = [ch.name for ch in wrapper.config.channels]

        # Clear widget keys that need to be re-initialized from config
        # This forces configure_model.py to re-read from config_state instead of using stale widget values
        widget_keys_to_clear = [
            "channel_multiselect",
            "owned_media_multiselect",
            "competitor_multiselect",
            "control_multiselect",
            "channels_data_editor",
            "owned_media_data_editor",
            "competitor_data_editor",
            "controls_data_editor",
        ]
        for key in widget_keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]

        # Increment config version to force widget reset in configure_model page
        st.session_state["config_version"] = st.session_state.get("config_version", 0) + 1

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


def _load_model_config(model_path: str) -> Optional[ModelConfig]:
    """Load config from a saved model directory."""
    config_file = Path(model_path) / "config.json"
    if not config_file.exists():
        return None

    try:
        with open(config_file, "r") as f:
            config_dict = json.load(f)
        return ModelConfig(**config_dict)
    except Exception as e:
        st.error(f"Error loading config: {e}")
        return None


def _generate_categories_csv(config: ModelConfig) -> str:
    """Generate CSV content with all variables and their categories."""
    import pandas as pd
    import io

    # Fixed columns
    rows = []

    # Get all existing category column names
    existing_cat_cols = [col.name for col in config.category_columns]

    # Add channels
    for ch in config.channels:
        row = {
            "variable_name": ch.name,
            "variable_type": "channel",
            "display_name": ch.display_name or ch.name,
        }
        for cat_col in existing_cat_cols:
            row[cat_col] = ch.categories.get(cat_col, "")
        rows.append(row)

    # Add owned media
    for om in config.owned_media:
        row = {
            "variable_name": om.name,
            "variable_type": "owned_media",
            "display_name": om.display_name or om.name,
        }
        for cat_col in existing_cat_cols:
            row[cat_col] = om.categories.get(cat_col, "")
        rows.append(row)

    # Add competitors
    for comp in config.competitors:
        row = {
            "variable_name": comp.name,
            "variable_type": "competitor",
            "display_name": comp.display_name or comp.name,
        }
        for cat_col in existing_cat_cols:
            row[cat_col] = comp.categories.get(cat_col, "")
        rows.append(row)

    # Add controls
    for ctrl in config.controls:
        row = {
            "variable_name": ctrl.name,
            "variable_type": "control",
            "display_name": ctrl.display_name or ctrl.name,
        }
        for cat_col in existing_cat_cols:
            row[cat_col] = ctrl.categories.get(cat_col, "")
        rows.append(row)

    # Add dummy variables
    for dummy in config.dummy_variables:
        row = {
            "variable_name": dummy.name,
            "variable_type": "dummy",
            "display_name": dummy.name,  # DummyVariableConfig doesn't have display_name
        }
        for cat_col in existing_cat_cols:
            row[cat_col] = dummy.categories.get(cat_col, "")
        rows.append(row)

    # Add base components (intercept, trend, seasonality)
    # Read from stored categories or default to empty
    base_cats = config.base_component_categories if hasattr(config, 'base_component_categories') else {}

    # Intercept - always present
    intercept_cats = base_cats.get("intercept", {})
    rows.append({
        "variable_name": "intercept",
        "variable_type": "base",
        "display_name": "Intercept",
        **{cat_col: intercept_cats.get(cat_col, "") for cat_col in existing_cat_cols}
    })

    # Trend (if enabled)
    if config.data.include_trend:
        trend_cats = base_cats.get("trend", {})
        rows.append({
            "variable_name": "trend",
            "variable_type": "base",
            "display_name": "Trend",
            **{cat_col: trend_cats.get(cat_col, "") for cat_col in existing_cat_cols}
        })

    # Fourier/Seasonality terms
    if config.seasonality and config.seasonality.yearly_seasonality:
        for i in range(1, config.seasonality.yearly_seasonality + 1):
            sin_name = f"sin_order_{i}"
            cos_name = f"cos_order_{i}"
            sin_cats = base_cats.get(sin_name, {})
            cos_cats = base_cats.get(cos_name, {})
            rows.append({
                "variable_name": sin_name,
                "variable_type": "seasonality",
                "display_name": f"Seasonality Sin {i}",
                **{cat_col: sin_cats.get(cat_col, "") for cat_col in existing_cat_cols}
            })
            rows.append({
                "variable_name": cos_name,
                "variable_type": "seasonality",
                "display_name": f"Seasonality Cos {i}",
                **{cat_col: cos_cats.get(cat_col, "") for cat_col in existing_cat_cols}
            })

    if not rows:
        # Empty config - return template with just headers
        return "variable_name,variable_type,display_name\n"

    df = pd.DataFrame(rows)
    # Ensure column order: fixed cols first, then category cols
    fixed_cols = ["variable_name", "variable_type", "display_name"]
    col_order = fixed_cols + [c for c in df.columns if c not in fixed_cols]
    df = df[col_order]

    return df.to_csv(index=False)


def _parse_categories_csv(csv_content: str, config: ModelConfig) -> tuple[dict, list[str], list[str], list[str], list[dict]]:
    """
    Parse uploaded CSV and extract category changes.

    Returns:
        (new_categories_by_var, added_columns, removed_columns, final_columns, changes_list)
    """
    import pandas as pd
    import io

    df = pd.read_csv(io.StringIO(csv_content))

    # Fixed columns that are read-only
    fixed_cols = {"variable_name", "variable_type", "display_name"}

    # Detect category columns in CSV (anything not in fixed cols)
    csv_category_cols = [c for c in df.columns if c not in fixed_cols]

    # Get current category columns
    current_cat_cols = {col.name for col in config.category_columns}

    # Check which columns have at least one non-empty value
    cols_with_values = set()
    for col in csv_category_cols:
        for val in df[col]:
            if pd.notna(val) and str(val).strip():
                cols_with_values.add(col)
                break

    # Determine column changes:
    # - Added: in CSV with values, not in current
    # - Removed: in current but not in CSV, OR in CSV but completely empty
    added_columns = [c for c in cols_with_values if c not in current_cat_cols]
    removed_columns = [c for c in current_cat_cols if c not in cols_with_values]

    # Final columns = columns with at least one value
    final_columns = list(cols_with_values)

    # Build categories dict keyed by variable_name
    new_categories = {}
    for _, row in df.iterrows():
        var_name = row["variable_name"]
        cats = {}
        for col in final_columns:  # Only include columns that have values
            val = row.get(col, "")
            if pd.notna(val) and str(val).strip():
                cats[col] = str(val).strip()
        new_categories[var_name] = cats

    # Build list of changes for preview
    changes = []

    # Helper to get current categories for a variable
    def get_current_cats(var_config):
        return var_config.categories if var_config else {}

    # Check all variables
    all_vars = (
        [(ch, "channel") for ch in config.channels] +
        [(om, "owned_media") for om in config.owned_media] +
        [(comp, "competitor") for comp in config.competitors] +
        [(ctrl, "control") for ctrl in config.controls] +
        [(dummy, "dummy") for dummy in config.dummy_variables]
    )

    for var_config, var_type in all_vars:
        var_name = var_config.name
        current_cats = get_current_cats(var_config)
        new_cats = new_categories.get(var_name, {})

        # Check each category column (current and new)
        all_cols = set(current_cats.keys()) | set(new_cats.keys())
        for col in all_cols:
            old_val = current_cats.get(col, "")
            new_val = new_cats.get(col, "")
            if old_val != new_val:
                changes.append({
                    "variable": var_name,
                    "column": col,
                    "old_value": old_val or "(empty)",
                    "new_value": new_val or "(empty)",
                })

    # Add base components to change detection
    base_cats = config.base_component_categories if hasattr(config, 'base_component_categories') else {}
    base_vars = ["intercept"]
    if config.data.include_trend:
        base_vars.append("trend")
    if config.seasonality and config.seasonality.yearly_seasonality:
        for i in range(1, config.seasonality.yearly_seasonality + 1):
            base_vars.extend([f"sin_order_{i}", f"cos_order_{i}"])

    for var_name in base_vars:
        current_cats = base_cats.get(var_name, {})
        new_cats = new_categories.get(var_name, {})
        all_cols = set(current_cats.keys()) | set(new_cats.keys())
        for col in all_cols:
            old_val = current_cats.get(col, "")
            new_val = new_cats.get(col, "")
            if old_val != new_val:
                changes.append({
                    "variable": var_name,
                    "column": col,
                    "old_value": old_val or "(empty)",
                    "new_value": new_val or "(empty)",
                })

    return new_categories, added_columns, removed_columns, final_columns, changes


def _apply_categories_from_csv(config: ModelConfig, new_categories: dict, all_category_cols: list[str]) -> ModelConfig:
    """Apply category changes from parsed CSV to config."""
    # Update category columns list
    config.category_columns = [
        CategoryColumnConfig(name=col, options=[])
        for col in all_category_cols
    ]

    # Update channels
    for ch in config.channels:
        if ch.name in new_categories:
            ch.categories = new_categories[ch.name]

    # Update owned media
    for om in config.owned_media:
        if om.name in new_categories:
            om.categories = new_categories[om.name]

    # Update competitors
    for comp in config.competitors:
        if comp.name in new_categories:
            comp.categories = new_categories[comp.name]

    # Update controls
    for ctrl in config.controls:
        if ctrl.name in new_categories:
            ctrl.categories = new_categories[ctrl.name]

    # Update dummy variables
    for dummy in config.dummy_variables:
        if dummy.name in new_categories:
            dummy.categories = new_categories[dummy.name]

    # Update base component categories (intercept, trend, seasonality)
    base_vars = ["intercept"]
    if config.data.include_trend:
        base_vars.append("trend")
    if config.seasonality and config.seasonality.yearly_seasonality:
        for i in range(1, config.seasonality.yearly_seasonality + 1):
            base_vars.extend([f"sin_order_{i}", f"cos_order_{i}"])

    for var_name in base_vars:
        if var_name in new_categories:
            config.base_component_categories[var_name] = new_categories[var_name]

    return config


def _show_category_editor(model_path: str, unique_key: str):
    """Show CSV-based category editing UI for a saved model."""
    import pandas as pd

    config = _load_model_config(model_path)
    if config is None:
        st.warning("Could not load configuration for this model.")
        return

    edit_key = f"cat_edit_{unique_key}"

    # Initialize session state
    if edit_key not in st.session_state:
        st.session_state[edit_key] = {
            "uploaded_csv": None,
            "preview_data": None,
        }

    edit_state = st.session_state[edit_key]

    st.caption("Download the categories template, edit in Excel/Sheets, then upload to update.")

    # Download button
    csv_content = _generate_categories_csv(config)
    model_name = Path(model_path).name
    st.download_button(
        label="Download Categories CSV",
        data=csv_content,
        file_name=f"{model_name}_categories.csv",
        mime="text/csv",
        key=f"{edit_key}_download",
    )

    st.markdown("---")

    # Upload section
    uploaded_file = st.file_uploader(
        "Upload Updated CSV",
        type=["csv"],
        key=f"{edit_key}_upload",
        help="Upload a modified CSV to update categories. Add new columns to create new category groupings."
    )

    if uploaded_file is not None:
        try:
            csv_content = uploaded_file.getvalue().decode("utf-8")
            new_categories, added_cols, removed_cols, final_cols, changes = _parse_categories_csv(csv_content, config)

            # Store for apply
            edit_state["preview_data"] = {
                "new_categories": new_categories,
                "added_cols": added_cols,
                "removed_cols": removed_cols,
                "final_cols": final_cols,
                "changes": changes,
                "csv_content": csv_content,
            }

            # Show preview
            st.markdown("##### Preview Changes")

            if added_cols:
                st.success(f"New category columns: **{', '.join(added_cols)}**")

            if removed_cols:
                st.warning(f"Removed category columns: **{', '.join(removed_cols)}**")

            if changes:
                st.markdown(f"**{len(changes)} value change(s) detected:**")

                # Show changes as a table
                changes_df = pd.DataFrame(changes)
                changes_df.columns = ["Variable", "Category Column", "Old Value", "New Value"]
                st.dataframe(changes_df, width="stretch", hide_index=True)
            elif not added_cols and not removed_cols:
                st.info("No changes detected.")

            # Apply / Cancel buttons
            col1, col2 = st.columns(2)
            with col1:
                has_changes = changes or added_cols or removed_cols
                if has_changes:
                    if st.button("Apply Changes", key=f"{edit_key}_apply", type="primary"):
                        try:
                            # Apply changes using final_cols (only columns with values)
                            updated_config = _apply_categories_from_csv(config, new_categories, final_cols)

                            # Save
                            ModelPersistence.update_config(model_path, updated_config)

                            # Clear state
                            del st.session_state[edit_key]

                            st.success("Categories updated successfully!")
                            st.rerun()

                        except Exception as e:
                            st.error(f"Error applying changes: {e}")

            with col2:
                if st.button("Cancel", key=f"{edit_key}_cancel"):
                    del st.session_state[edit_key]
                    st.rerun()

        except Exception as e:
            st.error(f"Error parsing CSV: {e}")
