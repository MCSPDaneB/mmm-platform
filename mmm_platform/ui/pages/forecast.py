"""
Media Incrementality Forecast page for MMM Platform.

Provides interface for:
- Uploading planned/actual spend CSVs (aggregated or granular format)
- Previewing seasonal adjustments
- Forecasting incremental response with uncertainty
- Sanity checking against historical data
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import logging
import io
import zipfile

from mmm_platform.config.schema import ForecastSpendMapping
from mmm_platform.forecasting import (
    detect_spend_format,
    get_level_columns,
    validate_against_saved_mapping,
    aggregate_granular_spend,
    check_forecast_overlap,
    OverlapAnalysis,
)
from mmm_platform.model.persistence import ForecastPersistence
from mmm_platform.ui.kpi_labels import KPILabels

logger = logging.getLogger(__name__)


def show():
    """Display the media incrementality forecast page."""
    st.title("📈 Media Incrementality Forecast")
    st.caption(
        "Forecast incremental response from planned or actual media spend. "
        "Uses the same channel effectiveness indices as the optimizer."
    )

    # Check for fitted model
    if not st.session_state.get("model_fitted") or st.session_state.get("current_model") is None:
        st.warning("Please fit a model first on the Configure Model page.")
        st.info("Forecasting requires a fitted model with posterior samples.")
        return

    wrapper = st.session_state.current_model

    # Get KPI labels for proper display
    kpi_labels = KPILabels(wrapper.config)

    try:
        from mmm_platform.forecasting import SpendForecastEngine
    except ImportError as e:
        st.error(f"Forecasting module not available: {e}")
        return

    # Initialize forecast engine
    try:
        engine = SpendForecastEngine(wrapper)
    except Exception as e:
        st.error(f"Error initializing forecast engine: {e}")
        logger.exception("Forecast engine init failed")
        return

    # Create tabs for workflow
    upload_tab, results_tab, history_tab = st.tabs([
        "📤 Upload & Configure",
        "📊 Results",
        "📜 History"
    ])

    with upload_tab:
        _show_upload_tab(engine, wrapper)

    with results_tab:
        _show_results_tab(engine, wrapper)

    with history_tab:
        _show_history_tab(wrapper, kpi_labels)


def _show_upload_tab(engine, wrapper):
    """Display the upload and configuration tab."""
    st.subheader("Upload Spend Data")

    # Show expected format
    with st.expander("CSV Format Requirements", expanded=False):
        st.markdown("""
        **Two formats are supported:**

        **1. Aggregated Format** (direct - matches model channels):
        ```csv
        date,search_spend,social_spend,tv_spend
        2025-01-06,50000,30000,100000
        2025-01-13,55000,32000,95000
        ```

        **2. Granular Format** (requires mapping):
        ```csv
        date,media_channel_lvl1,media_channel_lvl2,media_channel_lvl3,spend
        2025-01-06,Google,Search,Brand,50000
        2025-01-06,Google,Search,NonBrand,30000
        2025-01-06,Facebook,Social,Awareness,25000
        ```

        **Notes:**
        - Dates must start AFTER your model's training data ends
        - Spend should be in the same units as your model's training data
        - For granular format, a mapping UI will appear on first upload
        """)

    # Show available channels and model date range
    channel_names = engine.get_channel_display_names()
    channel_cols = list(channel_names.keys())

    # Get model's date range and frequency
    model_dates = pd.to_datetime(wrapper.df_scaled["date"])
    model_end = model_dates.max()

    # Detect frequency from model data (days between observations)
    if len(model_dates) > 1:
        date_diffs = model_dates.sort_values().diff().dropna()
        median_days = date_diffs.dt.days.median()
        next_forecast_date = model_end + pd.Timedelta(days=int(median_days))
        freq_label = f"{int(median_days)}-day" if median_days != 7 else "weekly"
    else:
        next_forecast_date = model_end + pd.Timedelta(days=7)
        freq_label = "weekly"

    next_date_str = next_forecast_date.strftime("%Y-%m-%d")

    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**Model channels:** {', '.join(channel_names.values())}")
    with col2:
        st.warning(f"**Model is {freq_label}. Next forecast date:** {next_date_str}")

    # Check for cached data from previous upload
    cached_spend = st.session_state.get("forecast_df_spend")
    cached_file_info = st.session_state.get("forecast_processed_file")

    # File upload
    uploaded_file = st.file_uploader(
        "Upload spend CSV",
        type=["csv"],
        key="forecast_csv_upload",
        help="Upload a new file or use previously uploaded data below"
    )

    # Show option to use cached data if available and no new file uploaded
    if uploaded_file is None and cached_spend is not None and cached_file_info:
        st.info(f"Previously uploaded: **{cached_file_info.get('name', 'unknown')}** ({len(cached_spend)} rows)")
        col1, col2 = st.columns([1, 3])
        with col1:
            use_cached = st.button("Use Previous Data", type="secondary")
        with col2:
            clear_cached = st.button("Clear & Upload New")

        if clear_cached:
            st.session_state.forecast_df_spend = None
            st.session_state.forecast_processed_file = None
            st.session_state.forecast_result = None
            st.rerun()

        if use_cached:
            uploaded_file = "USE_CACHED"  # Flag to use cached data

    if uploaded_file is not None:
        try:
            # Handle cached data vs new upload
            if uploaded_file == "USE_CACHED":
                # Use previously processed data - skip all parsing
                df_spend = cached_spend
                st.success(f"Using previously processed data ({len(df_spend)} rows)")

                # Show preview of cached data
                st.markdown("**Data Preview:**")
                st.dataframe(df_spend.head(10), width="stretch")
            else:
                # New file upload - process it
                df_raw = pd.read_csv(uploaded_file)
                st.success(f"Loaded {len(df_raw)} rows")

                # Parse date column with dayfirst=True for DD/MM/YYYY format
                date_col_candidates = [c for c in df_raw.columns if 'date' in c.lower()]
                for date_col in date_col_candidates:
                    if df_raw[date_col].dtype == 'object':  # String column
                        try:
                            df_raw[date_col] = pd.to_datetime(df_raw[date_col], dayfirst=True)
                            logger.info(f"Parsed '{date_col}' with dayfirst=True")
                        except Exception:
                            pass  # Leave as-is if parsing fails

                # Show preview
                st.markdown("**Data Preview:**")
                st.dataframe(df_raw.head(10), width="stretch")

                # Detect format
                data_format = detect_spend_format(df_raw, channel_cols)

                if data_format == "granular":
                    # Check if we already have processed data for THIS file
                    current_file_id = {"name": uploaded_file.name, "size": uploaded_file.size}
                    stored_file_id = st.session_state.get("forecast_processed_file")
                    prev_cached_spend = st.session_state.get("forecast_df_spend")

                    if current_file_id == stored_file_id and prev_cached_spend is not None:
                        # Same file, use cached processed data
                        df_spend = prev_cached_spend
                        st.success(f"Using processed data ({len(df_spend)} rows)")
                    else:
                        # New file or no cached data, process from scratch
                        df_spend = _handle_granular_format(df_raw, channel_cols, wrapper)
                        if df_spend is not None:
                            # Store file identity with processed data
                            st.session_state.forecast_processed_file = current_file_id

                    if df_spend is None:
                        return  # User needs to complete mapping
                elif data_format == "aggregated":
                    st.success("Detected **aggregated format** (direct channel columns)")
                    df_spend = df_raw
                    # Store file identity for aggregated format too
                    st.session_state.forecast_processed_file = {"name": uploaded_file.name, "size": uploaded_file.size}
                else:
                    # Unknown format - show helpful error
                    st.error("**Format not recognized**")
                    st.warning(
                        f"Expected either:\n"
                        f"- Aggregated: columns matching model channels ({', '.join(channel_cols[:3])}...)\n"
                        f"- Granular: columns containing 'lvl' or 'level' (e.g., media_channel_lvl1)"
                    )
                    return

                # Store the processed spend data
                st.session_state.forecast_df_spend = df_spend

            # Validate
            errors = engine.validate_spend_csv(df_spend)
            if errors:
                st.error("**Validation Errors:**")
                for err in errors:
                    st.error(f"- {err.field}: {err.message}")
                return

            st.success("CSV validation passed")

            # Check for overlap with historical forecasts
            overlap_ok = _check_and_show_overlap(df_spend, wrapper)
            if not overlap_ok:
                return  # User chose to cancel or needs to resolve overlap

            # Show seasonality preview
            _show_seasonality_preview(engine, df_spend)

            # Configuration options
            st.markdown("---")
            st.subheader("Forecast Settings")

            apply_seasonal = st.checkbox(
                "Apply seasonal effectiveness adjustment",
                value=True,
                key="forecast_apply_seasonal",
                help="Apply channel effectiveness indices based on the forecast period"
            )

            # Custom indices override (optional)
            use_custom_indices = st.checkbox(
                "Use custom seasonal indices",
                value=False,
                key="forecast_use_custom_indices"
            )

            custom_indices = None
            if use_custom_indices:
                st.markdown("**Custom Channel Indices:**")
                custom_indices = {}
                cols = st.columns(3)
                for i, (ch_col, ch_name) in enumerate(channel_names.items()):
                    with cols[i % 3]:
                        default_idx = st.session_state.get(f"preview_index_{ch_col}", 1.0)
                        custom_indices[ch_col] = st.number_input(
                            ch_name,
                            min_value=0.1,
                            max_value=3.0,
                            value=float(default_idx),
                            step=0.05,
                            key=f"custom_idx_{ch_col}"
                        )

            st.session_state.forecast_custom_indices = custom_indices

            # Save options
            st.markdown("---")

            # Check if model is saved (required for forecast persistence)
            model_is_saved = hasattr(wrapper, '_saved_model_path') and wrapper._saved_model_path

            if not model_is_saved:
                st.warning("Model not saved to disk. Save your model first to enable forecast history.")
                save_forecast = False
                forecast_notes = ""
            else:
                save_forecast = st.checkbox(
                    "Save forecast to history",
                    value=True,
                    key="forecast_save_to_history",
                    help="Save this forecast for future reference and overlap detection"
                )
                forecast_notes = ""
                if save_forecast:
                    forecast_notes = st.text_input(
                        "Notes (optional)",
                        key="forecast_notes",
                        placeholder="e.g., Q1 2025 plan v1"
                    )

            # Run forecast button
            st.markdown("---")
            if st.button("🚀 Generate Forecast", type="primary", width="stretch"):
                with st.spinner("Computing forecast..."):
                    try:
                        result = engine.forecast(
                            df_spend=df_spend,
                            apply_seasonal=apply_seasonal,
                            custom_seasonal_indices=custom_indices,
                        )
                        st.session_state.forecast_result = result
                        st.session_state.forecast_input_spend = df_spend

                        # Save to history if requested (model_is_saved already validated above)
                        if save_forecast:
                            try:
                                forecast_id = ForecastPersistence.save_forecast(
                                    wrapper._saved_model_path,
                                    result,
                                    df_spend,
                                    notes=forecast_notes if forecast_notes else None
                                )
                                st.success(f"Forecast complete and saved to history (ID: {forecast_id})")
                            except Exception as e:
                                st.warning(f"Forecast completed but save failed: {e}")
                                logger.exception("Forecast save failed")
                                st.success("Forecast complete! View results in the Results tab.")
                        else:
                            st.success("Forecast complete! View results in the Results tab.")
                    except Exception as e:
                        st.error(f"Forecast failed: {e}")
                        logger.exception("Forecast failed")

        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            logger.exception("Error reading forecast CSV")


def _check_and_show_overlap(df_spend: pd.DataFrame, wrapper) -> bool:
    """
    Check for overlap with historical forecasts and show warning if found.

    Parameters
    ----------
    df_spend : pd.DataFrame
        New spend data being uploaded.
    wrapper : MMMWrapper
        The fitted model wrapper.

    Returns
    -------
    bool
        True if OK to proceed, False if user needs to address overlap.
    """
    # Check if we have a saved model path
    if not hasattr(wrapper, '_saved_model_path') or not wrapper._saved_model_path:
        return True  # No persistence, skip overlap check

    # Get historical spend data
    try:
        historical_spend = ForecastPersistence.get_historical_spend(wrapper._saved_model_path)
    except Exception as e:
        logger.warning(f"Could not load historical spend: {e}")
        return True  # Can't check, proceed anyway

    if historical_spend.empty:
        return True  # No history, proceed

    # Check for overlap
    overlap = check_forecast_overlap(df_spend, historical_spend)

    if not overlap.has_overlap:
        return True  # No overlap, proceed

    # Show overlap warning
    st.markdown("---")
    st.warning(f"**Overlap Detected:** Your upload contains {len(overlap.overlapping_dates)} dates that were previously forecasted.")

    # Summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Previous Spend", f"${overlap.total_spend_old:,.0f}")
    with col2:
        st.metric("New Spend", f"${overlap.total_spend_new:,.0f}")
    with col3:
        delta_str = f"{overlap.pct_change:+.1f}%" if overlap.pct_change != 0 else "0%"
        st.metric("Change", f"${overlap.spend_difference:+,.0f}", delta=delta_str)

    # Show comparison table
    if not overlap.spend_comparison.empty:
        with st.expander("View detailed comparison", expanded=True):
            display_df = overlap.spend_comparison.copy()

            # Format currency columns
            for col in ["old_spend", "new_spend", "diff"]:
                if col in display_df.columns:
                    display_df[col] = display_df[col].apply(lambda x: f"${x:,.0f}")

            # Format pct_change
            if "pct_change" in display_df.columns:
                display_df["pct_change"] = display_df["pct_change"].apply(lambda x: f"{x:+.1f}%")

            # Only show rows with differences
            st.dataframe(display_df, width="stretch", hide_index=True)

    # User options
    st.markdown("**How would you like to proceed?**")

    overlap_action = st.radio(
        "Select action",
        options=["override", "append", "cancel"],
        format_func=lambda x: {
            "override": "Override - Replace previous forecast for overlapping dates",
            "append": "Append only - Only forecast new dates, keep previous for overlap",
            "cancel": "Cancel - Go back and review data"
        }.get(x, x),
        key="forecast_overlap_action",
        label_visibility="collapsed"
    )

    if overlap_action == "cancel":
        st.info("Upload different data or adjust dates to continue.")
        return False

    if overlap_action == "append":
        # Filter to only new dates
        if overlap.new_dates:
            df_spend_filtered = df_spend[
                df_spend["date"].astype(str).isin(overlap.new_dates)
            ]
            st.session_state.forecast_df_spend = df_spend_filtered
            st.info(f"Proceeding with {len(overlap.new_dates)} new dates only.")
        else:
            st.warning("No new dates to forecast. All dates overlap with history.")
            return False

    # For "override", we just proceed with all dates
    st.success("Proceeding with forecast...")
    return True


def _handle_granular_format(df_granular: pd.DataFrame, model_channels: list[str], wrapper) -> pd.DataFrame | None:
    """
    Handle granular format upload - show mapping UI and aggregate.

    Parameters
    ----------
    df_granular : pd.DataFrame
        Granular spend data with level columns.
    model_channels : list[str]
        List of model channel column names.
    wrapper : MMMWrapper
        The fitted model wrapper.

    Returns
    -------
    pd.DataFrame or None
        Aggregated data if mapping is complete, None if user needs to finish mapping.
    """
    level_cols = get_level_columns(df_granular)
    st.success(f"Detected **granular format** (level columns: {', '.join(level_cols)})")

    # Check for saved mapping
    saved_mapping = wrapper.config.forecast_spend_mapping

    if saved_mapping is not None:
        return _handle_saved_mapping(df_granular, level_cols, model_channels, saved_mapping, wrapper)
    else:
        return _handle_first_time_mapping(df_granular, level_cols, model_channels, wrapper)


def _handle_saved_mapping(
    df_granular: pd.DataFrame,
    level_cols: list[str],
    model_channels: list[str],
    saved_mapping: ForecastSpendMapping,
    wrapper
) -> pd.DataFrame | None:
    """Handle upload when a saved mapping exists."""
    st.markdown("---")
    st.subheader("Mapping Status")

    # Check if level columns match
    if saved_mapping.level_columns != level_cols:
        st.warning(
            f"Level columns changed: expected {saved_mapping.level_columns}, "
            f"got {level_cols}. Please create a new mapping."
        )
        return _handle_first_time_mapping(df_granular, level_cols, model_channels, wrapper)

    # Validate entities against saved mapping
    validation = validate_against_saved_mapping(
        df_granular,
        level_cols,
        saved_mapping.entity_mappings,
        skipped_entities=saved_mapping.skipped_entities
    )

    # Show validation status
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Matched", len(validation["matched"]))
    with col2:
        st.metric("Skipped", len(validation.get("skipped", [])))
    with col3:
        st.metric("New", len(validation["new_entities"]))
    with col4:
        st.metric("Missing", len(validation["missing"]))

    if validation["is_valid"]:
        st.success("All entities match saved mapping")

        # Auto-apply or let user choose
        if st.button("Apply Saved Mapping", type="primary"):
            try:
                df_aggregated = aggregate_granular_spend(
                    df_granular,
                    level_cols,
                    saved_mapping.entity_mappings,
                    date_column=saved_mapping.date_column,
                    spend_column=saved_mapping.spend_column
                )
                st.success(f"Aggregated to {len(df_aggregated)} rows")
                return df_aggregated
            except Exception as e:
                st.error(f"Aggregation failed: {e}")
                return None

        # Option to edit mapping
        if st.checkbox("Edit mapping", key="forecast_edit_mapping"):
            return _show_mapping_editor(
                df_granular, level_cols, model_channels, wrapper,
                existing_mapping=saved_mapping.entity_mappings
            )

        return None  # User needs to click button

    else:
        # Has new entities - need to map them
        st.warning(f"Found {len(validation['new_entities'])} new entities that need mapping")

        if validation["missing"]:
            with st.expander(f"Missing entities ({len(validation['missing'])})"):
                st.caption("These entities are in saved mapping but not in uploaded file:")
                for entity in validation["missing"][:20]:
                    st.text(f"  {entity}")
                if len(validation["missing"]) > 20:
                    st.text(f"  ... and {len(validation['missing']) - 20} more")

        # Show mapping for new entities
        return _show_new_entity_mapping(
            df_granular, level_cols, model_channels, wrapper,
            validation, saved_mapping
        )


def _handle_first_time_mapping(
    df_granular: pd.DataFrame,
    level_cols: list[str],
    model_channels: list[str],
    wrapper
) -> pd.DataFrame | None:
    """Handle first-time mapping when no saved mapping exists."""
    st.markdown("---")
    st.subheader("Map Entities to Model Channels")
    st.info("First granular upload - map each entity to a model channel.")

    return _show_mapping_editor(df_granular, level_cols, model_channels, wrapper)


def _show_mapping_editor(
    df_granular: pd.DataFrame,
    level_cols: list[str],
    model_channels: list[str],
    wrapper,
    existing_mapping: dict[str, str] | None = None
) -> pd.DataFrame | None:
    """Show the full mapping editor UI with table-based editing."""
    # Get unique entities
    entities_df = df_granular[level_cols].drop_duplicates()
    entity_keys = []
    for _, row in entities_df.iterrows():
        entity_key = "|".join(str(row[c]) for c in level_cols)
        entity_keys.append(entity_key)

    entity_keys = sorted(entity_keys)

    st.caption(f"Map {len(entity_keys)} entities to model channels")

    # Channel options
    channel_options = ["-- Skip --"] + model_channels

    # Create mapping table dataframe
    mapping_data = []
    for entity in entity_keys:
        parts = entity.split("|")
        row_data = {"entity_key": entity}
        # Add level columns for display
        for i, col in enumerate(level_cols):
            row_data[col] = parts[i] if i < len(parts) else ""
        # Add current mapping
        default_val = existing_mapping.get(entity, "-- Skip --") if existing_mapping else "-- Skip --"
        row_data["model_channel"] = default_val
        mapping_data.append(row_data)

    mapping_df = pd.DataFrame(mapping_data)

    # Download/Upload section
    st.markdown("---")
    st.markdown("**Bulk Mapping Options:**")
    col_dl, col_ul = st.columns(2)

    with col_dl:
        # Create downloadable template
        template_df = mapping_df[[*level_cols, "model_channel"]].copy()

        # Build valid channels reference CSV
        channels_df = pd.DataFrame({
            "model_channel": channel_options,
            "description": ["Exclude from mapping"] + [f"Map to {ch}" for ch in model_channels]
        })

        # Create ZIP with both files
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("mapping_template.csv", template_df.to_csv(index=False))
            zf.writestr("valid_channels.csv", channels_df.to_csv(index=False))
        zip_buffer.seek(0)

        st.download_button(
            "Download Mapping Template",
            data=zip_buffer.getvalue(),
            file_name="mapping_template.zip",
            mime="application/zip",
            help=f"ZIP with mapping template ({len(entity_keys)} entities) + valid channels reference"
        )

    with col_ul:
        uploaded_mapping = st.file_uploader(
            "Upload Filled Mapping",
            type=["csv"],
            key="mapping_csv_upload",
            help="Upload the filled template CSV"
        )

    # Process uploaded mapping
    if uploaded_mapping is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_mapping)
            # Validate columns
            if "model_channel" not in uploaded_df.columns:
                st.error("Uploaded CSV must have 'model_channel' column")
            else:
                # Merge uploaded mappings
                for _, row in uploaded_df.iterrows():
                    # Build entity key from level columns
                    entity_key = "|".join(str(row[c]) for c in level_cols if c in row.index)
                    channel = row.get("model_channel", "-- Skip --")
                    if entity_key in entity_keys and channel in channel_options:
                        # Update mapping_df
                        mask = mapping_df["entity_key"] == entity_key
                        mapping_df.loc[mask, "model_channel"] = channel
                st.success("Mapping loaded from CSV")
        except Exception as e:
            st.error(f"Error reading mapping CSV: {e}")

    # Table-based editor
    st.markdown("---")
    st.markdown("**Entity Mapping Table:**")
    st.caption("Edit the 'model_channel' column to map each entity")

    # Configure columns for data_editor
    column_config = {
        "entity_key": None,  # Hide the entity_key column
        "model_channel": st.column_config.SelectboxColumn(
            "Model Channel",
            options=channel_options,
            required=True,
            width="medium",
        )
    }
    # Make level columns read-only
    for col in level_cols:
        column_config[col] = st.column_config.TextColumn(
            col.replace("_", " ").title(),
            disabled=True,
        )

    # Display editable table
    display_cols = [*level_cols, "model_channel"]
    edited_df = st.data_editor(
        mapping_df[display_cols],
        column_config=column_config,
        width="stretch",
        hide_index=True,
        num_rows="fixed",
        key="mapping_table_editor"
    )

    # Build mapping and skipped list from edited table
    new_mapping = {}
    skipped_entities = []
    for i, row in edited_df.iterrows():
        entity_key = mapping_df.iloc[i]["entity_key"]
        channel = row["model_channel"]
        if channel != "-- Skip --":
            new_mapping[entity_key] = channel
        else:
            skipped_entities.append(entity_key)

    # Summary
    mapped_count = len(new_mapping)
    skipped_count = len(skipped_entities)
    st.markdown(f"**Summary:** {mapped_count} mapped, {skipped_count} skipped")

    if mapped_count == 0:
        st.warning("No entities mapped. Map at least one entity to continue.")
        return None

    # Date and spend column configuration
    st.markdown("---")
    st.markdown("**Column Configuration:**")
    col1, col2 = st.columns(2)

    # Get available columns (excluding level columns)
    available_cols = [c for c in df_granular.columns if c not in level_cols]

    # Find default date column index
    date_default_idx = 0
    for i, c in enumerate(available_cols):
        if c.lower() == "date":
            date_default_idx = i
            break

    with col1:
        date_col = st.selectbox(
            "Date column",
            options=available_cols,
            index=date_default_idx,
            key="forecast_date_col"
        )

    # Get spend column options (excluding selected date column)
    spend_options = [c for c in available_cols if c != date_col]

    # Find default spend column index
    spend_default_idx = 0
    for i, c in enumerate(spend_options):
        if c.lower() == "spend":
            spend_default_idx = i
            break

    with col2:
        spend_col = st.selectbox(
            "Spend column",
            options=spend_options,
            index=spend_default_idx,
            key="forecast_spend_col"
        )

    # Save option
    save_mapping = st.checkbox(
        "Save mapping for future uploads",
        value=True,
        key="forecast_save_mapping",
        help="Stores mapping with the model so future uploads auto-apply"
    )

    # Apply button
    if st.button("Apply Mapping", type="primary", key="forecast_apply_mapping"):
        try:
            df_aggregated = aggregate_granular_spend(
                df_granular,
                level_cols,
                new_mapping,
                date_column=date_col,
                spend_column=spend_col
            )

            if save_mapping:
                _save_mapping_to_model(
                    wrapper, level_cols, new_mapping, date_col, spend_col,
                    skipped_entities=skipped_entities
                )
                st.success("Mapping saved for future uploads")

            st.success(f"Aggregated to {len(df_aggregated)} rows")
            return df_aggregated

        except Exception as e:
            st.error(f"Aggregation failed: {e}")
            logger.exception("Granular aggregation failed")
            return None

    return None  # User needs to click button


def _show_new_entity_mapping(
    df_granular: pd.DataFrame,
    level_cols: list[str],
    model_channels: list[str],
    wrapper,
    validation: dict,
    saved_mapping: ForecastSpendMapping
) -> pd.DataFrame | None:
    """Show mapping UI for new entities only using table-based editing."""
    st.markdown("**Map new entities:**")

    channel_options = ["-- Skip --"] + model_channels
    new_entities = sorted(validation["new_entities"])

    # Create table for new entities
    mapping_data = []
    for entity in new_entities:
        parts = entity.split("|")
        row_data = {"entity_key": entity}
        for i, col in enumerate(level_cols):
            row_data[col] = parts[i] if i < len(parts) else ""
        row_data["model_channel"] = "-- Skip --"
        mapping_data.append(row_data)

    mapping_df = pd.DataFrame(mapping_data)

    # Configure columns
    column_config = {
        "entity_key": None,
        "model_channel": st.column_config.SelectboxColumn(
            "Model Channel",
            options=channel_options,
            required=True,
            width="medium",
        )
    }
    for col in level_cols:
        column_config[col] = st.column_config.TextColumn(
            col.replace("_", " ").title(),
            disabled=True,
        )

    # Display table
    display_cols = [*level_cols, "model_channel"]
    edited_df = st.data_editor(
        mapping_df[display_cols],
        column_config=column_config,
        width="stretch",
        hide_index=True,
        num_rows="fixed",
        key="new_entity_mapping_table"
    )

    # Build mapping and skipped list from edited table
    new_entity_mapping = {}
    new_skipped_entities = []
    for i, row in edited_df.iterrows():
        entity_key = mapping_df.iloc[i]["entity_key"]
        channel = row["model_channel"]
        if channel != "-- Skip --":
            new_entity_mapping[entity_key] = channel
        else:
            new_skipped_entities.append(entity_key)

    # Combine with existing
    combined_mapping = dict(saved_mapping.entity_mappings)
    combined_mapping.update(new_entity_mapping)

    # Combine skipped entities (existing + new)
    combined_skipped = list(set(saved_mapping.skipped_entities) | set(new_skipped_entities))

    # Update mapping option
    update_saved = st.checkbox(
        "Update saved mapping with new entities",
        value=True,
        key="forecast_update_mapping"
    )

    if st.button("Apply Updated Mapping", type="primary", key="forecast_apply_updated"):
        try:
            df_aggregated = aggregate_granular_spend(
                df_granular,
                level_cols,
                combined_mapping,
                date_column=saved_mapping.date_column,
                spend_column=saved_mapping.spend_column
            )

            if update_saved:
                _save_mapping_to_model(
                    wrapper, level_cols, combined_mapping,
                    saved_mapping.date_column, saved_mapping.spend_column,
                    skipped_entities=combined_skipped
                )
                st.success("Saved mapping updated")

            st.success(f"Aggregated to {len(df_aggregated)} rows")
            return df_aggregated

        except Exception as e:
            st.error(f"Aggregation failed: {e}")
            return None

    return None


def _save_mapping_to_model(
    wrapper,
    level_columns: list[str],
    entity_mappings: dict[str, str],
    date_column: str,
    spend_column: str,
    skipped_entities: list[str] | None = None
):
    """Save the mapping to the model's config."""
    new_mapping = ForecastSpendMapping(
        level_columns=level_columns,
        date_column=date_column,
        spend_column=spend_column,
        entity_mappings=entity_mappings,
        skipped_entities=skipped_entities or []
    )

    # Update config
    wrapper.config.forecast_spend_mapping = new_mapping

    # Persist if we have a saved model path
    try:
        from mmm_platform.model.persistence import ModelPersistence
        if hasattr(wrapper, '_saved_model_path') and wrapper._saved_model_path:
            # Re-save the config
            ModelPersistence.update_config(wrapper._saved_model_path, wrapper.config)
            logger.info(f"Updated forecast mapping in saved model")
    except Exception as e:
        logger.warning(f"Could not persist mapping to disk: {e}")


def _show_seasonality_preview(engine, df_spend: pd.DataFrame):
    """Show preview of seasonal indices for the forecast period."""
    st.markdown("---")
    st.subheader("Seasonality Preview")

    try:
        preview = engine.get_seasonality_preview(df_spend)

        st.markdown(f"**Forecast Period:** {preview['period']} ({preview['num_weeks']} weeks)")

        # Channel effectiveness indices
        st.markdown("**Channel Effectiveness Indices** (applied to forecast):")

        channel_names = engine.get_channel_display_names()
        cols = st.columns(min(4, len(preview['channel_indices'])))

        for i, (ch_col, idx) in enumerate(preview['channel_indices'].items()):
            ch_name = channel_names.get(ch_col, ch_col)
            with cols[i % len(cols)]:
                # Format as percentage deviation
                if idx > 1.0:
                    delta_str = f"+{(idx - 1) * 100:.0f}%"
                    color = "green"
                elif idx < 1.0:
                    delta_str = f"{(idx - 1) * 100:.0f}%"
                    color = "red"
                else:
                    delta_str = "avg"
                    color = "gray"

                st.metric(
                    label=ch_name,
                    value=f"{idx:.2f}",
                    delta=delta_str,
                )
                # Store for potential custom override
                st.session_state[f"preview_index_{ch_col}"] = idx

        # Demand index context
        demand_idx = preview['demand_index']
        if demand_idx > 1.0:
            demand_desc = f"{(demand_idx - 1) * 100:.0f}% above average"
        elif demand_idx < 1.0:
            demand_desc = f"{(1 - demand_idx) * 100:.0f}% below average"
        else:
            demand_desc = "average"

        st.info(
            f"**Context:** Demand Index = {demand_idx:.2f} ({demand_desc} base demand for this period). "
            f"Shown for context only - not applied to forecast."
        )

        # Confidence info
        conf = preview['confidence']
        confidence_colors = {"high": "🟢", "medium": "🟡", "low": "🔴"}
        st.caption(
            f"{confidence_colors.get(conf['confidence_level'], '⚪')} "
            f"Index confidence: {conf['confidence_level'].title()} "
            f"({conf['avg_observations']:.0f} avg observations per month)"
        )

    except Exception as e:
        st.warning(f"Could not compute seasonality preview: {e}")
        logger.exception("Seasonality preview failed")


def _show_results_tab(engine, wrapper):
    """Display the forecast results tab."""
    result = st.session_state.get("forecast_result")

    if result is None:
        st.info("Upload spend data and run a forecast to see results here.")
        return

    # Get KPI labels for proper ROI vs Cost Per display
    kpi_labels = KPILabels(wrapper.config)

    # Main metrics
    st.subheader("Forecast Results")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Budget",
            f"${result.total_spend:,.0f}",
        )
        st.caption(f"{result.num_weeks} weeks")

    with col2:
        # Format response based on KPI type
        if kpi_labels.is_revenue_type:
            response_value = f"${result.total_response:,.0f}"
        else:
            response_value = f"{result.total_response:,.0f}"
        st.metric(
            f"Media {kpi_labels.target_label}",
            response_value,
        )
        st.caption("Incremental media-driven response")

    with col3:
        # Use KPI-appropriate efficiency label
        if kpi_labels.is_revenue_type:
            st.metric(
                "ROI",
                f"${result.blended_roi:,.2f}",
            )
            st.caption("Revenue per dollar spent")
        else:
            # For count KPIs, show Cost Per X
            cost_per = result.total_spend / result.total_response if result.total_response > 0 else float('inf')
            st.metric(
                f"Cost Per {kpi_labels.target_label}",
                f"${cost_per:,.2f}",
            )
            st.caption("Spend per acquisition")

    # Seasonality annotation
    if result.seasonal_applied:
        st.info(
            f"Includes channel effectiveness adjustment (seasonally adjusted for {result.forecast_period}). "
            f"Context: Demand index {result.demand_index:.2f} for this period."
        )
    else:
        st.info("No seasonal adjustment applied.")

    # Charts
    st.markdown("---")
    st.subheader("Detailed Breakdown")

    chart_tab1, chart_tab2, chart_tab3 = st.tabs([
        "📅 Weekly Response",
        "📊 Channel Contributions",
        "📥 Download"
    ])

    with chart_tab1:
        _show_weekly_chart(result, wrapper, kpi_labels)

    with chart_tab2:
        _show_channel_chart(result, engine, wrapper, kpi_labels)

    with chart_tab3:
        _show_download_options(result, engine)


def _show_weekly_chart(result, wrapper, kpi_labels):
    """Show weekly response breakdown chart with historical context."""
    forecast_df = result.weekly_df.copy()
    forecast_df["date"] = pd.to_datetime(forecast_df["date"])
    forecast_df["period"] = "Forecast"

    # Get historical data (last 6 weeks)
    try:
        contribs = wrapper.get_contributions()
        channel_cols = wrapper.transform_engine.get_effective_channel_columns()
        spend_cols = [c for c in channel_cols if c in wrapper.df_scaled.columns]

        # Sum channel contributions per week for total response
        if len(contribs) > 0 and any(c in contribs.columns for c in channel_cols):
            available_channel_cols = [c for c in channel_cols if c in contribs.columns]
            historical_response = contribs[available_channel_cols].sum(axis=1)

            # Get spend from df_scaled
            historical_spend = wrapper.df_scaled[spend_cols].sum(axis=1) if spend_cols else pd.Series(0, index=contribs.index)

            # Take last 6 weeks
            n_weeks = min(6, len(historical_response))
            hist_dates = contribs.index[-n_weeks:]
            hist_response = historical_response.iloc[-n_weeks:].values
            hist_spend = historical_spend.iloc[-n_weeks:].values

            historical_df = pd.DataFrame({
                "date": pd.to_datetime(hist_dates),
                "response": hist_response,
                "spend": hist_spend,
                "ci_low": hist_response,  # No CI for historical
                "ci_high": hist_response,
                "period": "Historical"
            })
        else:
            historical_df = pd.DataFrame()
    except Exception as e:
        logger.warning(f"Could not get historical data: {e}")
        historical_df = pd.DataFrame()

    # Combine historical and forecast data
    if len(historical_df) > 0:
        combined_df = pd.concat([historical_df, forecast_df], ignore_index=True)
    else:
        combined_df = forecast_df

    fig = go.Figure()

    # Plot historical spend bars (if available)
    if len(historical_df) > 0:
        fig.add_trace(go.Bar(
            x=historical_df["date"],
            y=historical_df["spend"],
            name="Budget (Historical)",
            opacity=0.2,
            marker_color="gray",
            yaxis="y2",
            hovertemplate="Budget: $%{y:,.0f}<extra></extra>"
        ))

    # Add forecast spend on secondary axis
    fig.add_trace(go.Bar(
        x=forecast_df["date"],
        y=forecast_df["spend"],
        name="Budget (Forecast)",
        opacity=0.3,
        marker_color="rgb(0, 100, 200)",
        yaxis="y2",
        hovertemplate="Budget: $%{y:,.0f}<extra></extra>"
    ))

    # Plot combined response line (historical + forecast connected)
    if len(historical_df) > 0:
        # Combine dates, responses, and spend for one continuous line
        all_dates = list(historical_df["date"]) + list(forecast_df["date"])
        all_responses = list(historical_df["response"]) + list(forecast_df["response"])
        all_spend = list(historical_df["spend"]) + list(forecast_df["spend"])
        n_hist = len(historical_df)

        # Build hover text with ROI or Cost Per (Budget already shown by bar trace)
        if kpi_labels.is_revenue_type:
            hover_text = [
                f"Response: ${r:,.0f}<br>ROI: ${r/s:,.2f}" if s > 0 else f"Response: ${r:,.0f}"
                for r, s in zip(all_responses, all_spend)
            ]
        else:
            hover_text = [
                f"Response: {r:,.0f}<br>Cost Per: ${s/r:,.2f}" if r > 0 else f"Response: {r:,.0f}"
                for r, s in zip(all_responses, all_spend)
            ]

        # Plot as one continuous line with different marker colors
        fig.add_trace(go.Scatter(
            x=all_dates,
            y=all_responses,
            mode="lines+markers",
            name="Response",
            line=dict(color="rgb(100, 100, 100)", width=2),
            marker=dict(
                size=6,
                color=["rgb(100, 100, 100)"] * n_hist + ["rgb(0, 100, 200)"] * len(forecast_df)
            ),
            hovertemplate="%{text}<extra></extra>",
            text=hover_text
        ))
    else:
        # No historical data - just plot forecast
        forecast_spend = list(forecast_df["spend"])
        forecast_responses = list(forecast_df["response"])

        # Build hover text with ROI or Cost Per (Budget already shown by bar trace)
        if kpi_labels.is_revenue_type:
            hover_text = [
                f"Response: ${r:,.0f}<br>ROI: ${r/s:,.2f}" if s > 0 else f"Response: ${r:,.0f}"
                for r, s in zip(forecast_responses, forecast_spend)
            ]
        else:
            hover_text = [
                f"Response: {r:,.0f}<br>Cost Per: ${s/r:,.2f}" if r > 0 else f"Response: {r:,.0f}"
                for r, s in zip(forecast_responses, forecast_spend)
            ]

        fig.add_trace(go.Scatter(
            x=forecast_df["date"],
            y=forecast_df["response"],
            mode="lines+markers",
            name="Response",
            line=dict(color="rgb(0, 100, 200)", width=2),
            marker=dict(size=6),
            hovertemplate="%{text}<extra></extra>",
            text=hover_text
        ))

    # Add vertical line to separate historical from forecast
    if len(historical_df) > 0:
        boundary_date = forecast_df["date"].min()
        # Use add_shape instead of add_vline for date axis compatibility
        fig.add_shape(
            type="line",
            x0=boundary_date, x1=boundary_date,
            y0=0, y1=1,
            yref="paper",
            line=dict(color="red", width=2, dash="dash")
        )
        fig.add_annotation(
            x=boundary_date,
            y=1.05,
            yref="paper",
            text="Forecast Start",
            showarrow=False,
            font=dict(color="red", size=10)
        )

    # Build y-axis label using KPI
    y_axis_label = f"Media {kpi_labels.target_label}"

    fig.update_layout(
        title="Weekly Response: Historical vs Forecast",
        xaxis_title="Week",
        yaxis=dict(
            title=y_axis_label,
            tickprefix="$" if kpi_labels.is_revenue_type else "",
            tickformat=",."
        ),
        yaxis2=dict(
            title="Budget ($)",
            overlaying="y",
            side="right",
            showgrid=False,
            tickprefix="$",
            tickformat=",."
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified"
    )

    st.plotly_chart(fig, width="stretch")

    # Show data table
    with st.expander("View Weekly Data"):
        display_df = combined_df.copy()
        display_df["date"] = display_df["date"].dt.strftime("%Y-%m-%d")

        # Calculate efficiency metric based on KPI type
        if kpi_labels.is_revenue_type:
            # ROI = response / spend
            display_df["ROI"] = display_df.apply(
                lambda r: f"${r['response'] / r['spend']:,.2f}" if r['spend'] > 0 else "N/A", axis=1
            )
        else:
            # Cost Per = spend / response
            display_df[f"Cost Per {kpi_labels.target_label}"] = display_df.apply(
                lambda r: f"${r['spend'] / r['response']:,.2f}" if r['response'] > 0 else "N/A", axis=1
            )

        # Remove CI columns, format remaining numeric columns
        display_df = display_df.drop(columns=["ci_low", "ci_high"], errors="ignore")
        for col in ["response", "spend"]:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"{x:,.0f}")

        # Reorder columns: date, spend, response, ROI/Cost Per, period
        efficiency_col = "ROI" if kpi_labels.is_revenue_type else f"Cost Per {kpi_labels.target_label}"
        col_order = ["date", "spend", "response", efficiency_col, "period"]
        display_df = display_df[[c for c in col_order if c in display_df.columns]]

        st.dataframe(display_df, width="stretch")


def _show_channel_chart(result, engine, wrapper, kpi_labels):
    """Show channel contribution breakdown chart with historical context."""
    forecast_df = result.channel_contributions.copy()
    forecast_df["date"] = pd.to_datetime(forecast_df["date"])
    forecast_df["period"] = "Forecast"

    # Get display names
    channel_names = engine.get_channel_display_names()
    forecast_df["channel_name"] = forecast_df["channel"].map(channel_names)

    # Get historical data (last 6 weeks)
    try:
        contribs = wrapper.get_contributions()
        channel_cols = wrapper.transform_engine.get_effective_channel_columns()

        if len(contribs) > 0 and any(c in contribs.columns for c in channel_cols):
            available_channel_cols = [c for c in channel_cols if c in contribs.columns]

            # Take last 6 weeks
            n_weeks = min(6, len(contribs))
            hist_contribs = contribs.iloc[-n_weeks:]

            # Melt to long format for plotting
            hist_records = []
            for date_val in hist_contribs.index:
                for channel in available_channel_cols:
                    if channel in hist_contribs.columns:
                        hist_records.append({
                            "date": pd.to_datetime(date_val),
                            "channel": channel,
                            "contribution": hist_contribs.loc[date_val, channel],
                            "period": "Historical",
                            "channel_name": channel_names.get(channel, channel)
                        })
            historical_df = pd.DataFrame(hist_records) if hist_records else pd.DataFrame()
        else:
            historical_df = pd.DataFrame()
    except Exception as e:
        logger.warning(f"Could not get historical channel data: {e}")
        historical_df = pd.DataFrame()

    # Combine historical and forecast data
    if len(historical_df) > 0:
        combined_df = pd.concat([historical_df, forecast_df], ignore_index=True)
    else:
        combined_df = forecast_df

    # Build y-axis label using KPI
    y_axis_label = f"Incremental Media Driven {kpi_labels.target_label}"

    # Stacked bar chart
    fig = px.bar(
        combined_df,
        x="date",
        y="contribution",
        color="channel_name",
        title="Channel Contributions: Historical vs Forecast",
        labels={"contribution": y_axis_label, "date": "Week", "channel_name": "Channel"},
        barmode="stack"
    )

    # Add vertical line to separate historical from forecast
    if len(historical_df) > 0:
        boundary_date = forecast_df["date"].min()
        # Use add_shape instead of add_vline for date axis compatibility
        fig.add_shape(
            type="line",
            x0=boundary_date, x1=boundary_date,
            y0=0, y1=1,
            yref="paper",
            line=dict(color="red", width=2, dash="dash")
        )
        fig.add_annotation(
            x=boundary_date,
            y=1.05,
            yref="paper",
            text="Forecast Start",
            showarrow=False,
            font=dict(color="red", size=10)
        )

    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified"
    )

    st.plotly_chart(fig, width="stretch")

    # Channel totals table (forecast only)
    st.markdown("**Forecast Channel Totals:**")
    totals = forecast_df.groupby("channel_name")["contribution"].sum().reset_index()
    totals = totals.sort_values("contribution", ascending=False)
    totals["contribution"] = totals["contribution"].apply(lambda x: f"{x:,.0f}")
    totals.columns = ["Channel", "Total Contribution"]
    st.dataframe(totals, width="stretch", hide_index=True)


def _show_download_options(result, engine):
    """Show download options for forecast results."""
    st.subheader("Download Results")

    # Weekly summary
    st.markdown("**Weekly Forecast:**")
    weekly_csv = result.weekly_df.to_csv(index=False)
    st.download_button(
        label="Download Weekly Forecast CSV",
        data=weekly_csv,
        file_name="forecast_weekly.csv",
        mime="text/csv"
    )

    # Channel contributions
    st.markdown("**Channel Contributions:**")
    channel_csv = result.channel_contributions.to_csv(index=False)
    st.download_button(
        label="Download Channel Contributions CSV",
        data=channel_csv,
        file_name="forecast_channels.csv",
        mime="text/csv"
    )

    # Summary report
    st.markdown("**Summary Report:**")
    summary = _generate_summary_report(result, engine)
    st.download_button(
        label="Download Summary Report (TXT)",
        data=summary,
        file_name="forecast_summary.txt",
        mime="text/plain"
    )


def _generate_summary_report(result, engine) -> str:
    """Generate a text summary report."""
    lines = [
        "=" * 60,
        "MEDIA INCREMENTALITY FORECAST REPORT",
        "=" * 60,
        "",
        f"Forecast Period: {result.forecast_period}",
        f"Number of Weeks: {result.num_weeks}",
        f"Seasonal Adjustment Applied: {'Yes' if result.seasonal_applied else 'No'}",
        "",
        "-" * 60,
        "SUMMARY METRICS",
        "-" * 60,
        f"Total Media Spend: ${result.total_spend:,.0f}",
        f"Forecast Response: ${result.total_response:,.0f}",
        f"  95% CI Low:      ${result.total_ci_low:,.0f}",
        f"  95% CI High:     ${result.total_ci_high:,.0f}",
        f"Blended ROI:       ${result.blended_roi:,.2f}",
        "",
    ]

    if result.seasonal_applied:
        lines.extend([
            "-" * 60,
            "SEASONAL INDICES APPLIED",
            "-" * 60,
        ])
        channel_names = engine.get_channel_display_names()
        for ch_col, idx in result.seasonal_indices.items():
            ch_name = channel_names.get(ch_col, ch_col)
            lines.append(f"  {ch_name}: {idx:.2f}")
        lines.extend([
            "",
            f"Demand Index (context only): {result.demand_index:.2f}",
            "",
        ])

    lines.extend([
        "",
        "=" * 60,
        "Generated by MMM Platform - Spend Forecast Engine",
        "=" * 60,
    ])

    return "\n".join(lines)


def _show_history_tab(wrapper, kpi_labels):
    """Display the forecast history tab."""
    st.subheader("Forecast History")

    # Check if we have a saved model path
    if not hasattr(wrapper, '_saved_model_path') or not wrapper._saved_model_path:
        st.info("Forecast history is only available for saved models. Save your model first to enable history tracking.")
        return

    # Load forecasts
    try:
        forecasts = ForecastPersistence.list_forecasts(wrapper._saved_model_path)
    except Exception as e:
        st.error(f"Error loading forecast history: {e}")
        logger.exception("Failed to load forecast history")
        return

    if not forecasts:
        st.info("No forecasts saved yet. Run a forecast with 'Save to history' enabled to see it here.")
        return

    st.caption(f"Found {len(forecasts)} saved forecasts")

    # Show forecasts as a table with KPI-appropriate labels
    forecast_data = []
    for f in forecasts:
        # Format response based on KPI type
        if kpi_labels.is_revenue_type:
            response_str = f"${f.total_response:,.0f}"
            efficiency_str = f"${f.blended_roi:,.2f}"
        else:
            response_str = f"{f.total_response:,.0f}"
            cost_per = f.total_spend / f.total_response if f.total_response > 0 else float('inf')
            efficiency_str = f"${cost_per:.2f}"

        forecast_data.append({
            "ID": f.id,
            "Period": f.forecast_period,
            "Weeks": f.num_weeks,
            "Spend": f"${f.total_spend:,.0f}",
            kpi_labels.target_label: response_str,
            kpi_labels.efficiency_label: efficiency_str,
            "Seasonal": "Yes" if f.seasonal_applied else "No",
            "Created": f.created_at[:16].replace("T", " "),
            "Notes": f.notes or "",
        })

    df_forecasts = pd.DataFrame(forecast_data)
    st.dataframe(df_forecasts, width="stretch", hide_index=True)

    # Detail view
    st.markdown("---")
    st.subheader("View Forecast Details")

    forecast_ids = [f.id for f in forecasts]
    selected_id = st.selectbox(
        "Select a forecast to view",
        options=forecast_ids,
        format_func=lambda x: next(
            (f"{f.id} - {f.forecast_period} ({f.created_at[:10]})" for f in forecasts if f.id == x),
            x
        ),
        key="history_selected_forecast"
    )

    if selected_id:
        # Action buttons
        btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 2])

        with btn_col1:
            load_clicked = st.button("Load to Results", type="primary", key="load_forecast_btn")

        with btn_col2:
            if st.button("Delete Forecast", type="secondary", key="delete_forecast_btn"):
                if ForecastPersistence.delete_forecast(wrapper._saved_model_path, selected_id):
                    st.success(f"Deleted {selected_id}")
                    st.rerun()
                else:
                    st.error("Failed to delete forecast")

        try:
            result, input_spend, metadata = ForecastPersistence.load_forecast(
                wrapper._saved_model_path, selected_id
            )

            # Handle load to results
            if load_clicked:
                st.session_state["forecast_result"] = result
                st.session_state["forecast_input_spend"] = input_spend
                st.success("Forecast loaded! Switch to the Results tab to view it.")

            # Show summary
            st.markdown(f"**Period:** {metadata.forecast_period}")
            st.markdown(f"**Dates:** {metadata.start_date} to {metadata.end_date}")
            if metadata.notes:
                st.markdown(f"**Notes:** {metadata.notes}")

            # Metrics with KPI-appropriate labels
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Total Spend", f"${metadata.total_spend:,.0f}")
            with c2:
                if kpi_labels.is_revenue_type:
                    st.metric(f"Total {kpi_labels.target_label}", f"${metadata.total_response:,.0f}")
                else:
                    st.metric(f"Total {kpi_labels.target_label}", f"{metadata.total_response:,.0f}")
            with c3:
                if kpi_labels.is_revenue_type:
                    st.metric(f"Blended {kpi_labels.efficiency_label}", f"${metadata.blended_roi:,.2f}")
                else:
                    cost_per = metadata.total_spend / metadata.total_response if metadata.total_response > 0 else float('inf')
                    st.metric(f"Blended {kpi_labels.efficiency_label}", f"${cost_per:.2f}")
            with c4:
                st.metric("Weeks", metadata.num_weeks)

            # Weekly data
            with st.expander("Weekly Data"):
                st.dataframe(result.weekly_df, width="stretch")

            # Channel contributions
            with st.expander("Channel Contributions"):
                st.dataframe(result.channel_contributions, width="stretch")

            # Input spend
            with st.expander("Input Spend Data"):
                st.dataframe(input_spend, width="stretch")

        except Exception as e:
            st.error(f"Error loading forecast details: {e}")
            logger.exception(f"Failed to load forecast {selected_id}")
