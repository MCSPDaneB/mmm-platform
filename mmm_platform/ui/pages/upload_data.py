"""
Data upload page for MMM Platform.

Supports two-file upload:
1. Media data file (long format) with channel hierarchy
2. Other data file (wide format) with KPIs, promos, etc.
"""

import streamlit as st
import pandas as pd
from io import StringIO

from mmm_platform.core.validation import DataValidator, ValidationResult
from mmm_platform.config.schema import DataConfig
from mmm_platform.core.data_processing import (
    parse_media_file,
    get_unique_channel_combinations,
    aggregate_media_data,
    merge_datasets,
    detect_date_column,
    detect_metric_columns,
)


def _merge_data_incrementally(
    existing_data: pd.DataFrame,
    new_data: pd.DataFrame,
    date_col: str,
    dayfirst: bool = True,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Merge new data columns into existing data.

    Used when a model is loaded and user uploads additional data columns.
    New columns are added; existing columns with same name are overwritten.

    Args:
        existing_data: The current model's data
        new_data: New data to merge in
        date_col: Name of the date column to join on
        dayfirst: Whether dates are day-first format

    Returns:
        Tuple of (merged_df, added_columns, overwritten_columns)
    """
    # Ensure date columns are datetime
    existing_data = existing_data.copy()
    new_data = new_data.copy()

    if date_col in new_data.columns:
        new_data[date_col] = pd.to_datetime(new_data[date_col], dayfirst=dayfirst)
    if date_col in existing_data.columns:
        existing_data[date_col] = pd.to_datetime(existing_data[date_col], dayfirst=dayfirst)

    # Get new columns (excluding date)
    new_cols = [c for c in new_data.columns if c != date_col]
    existing_cols = [c for c in existing_data.columns if c != date_col]

    # Identify what's being overwritten vs added
    overwritten = [c for c in new_cols if c in existing_cols]
    added = [c for c in new_cols if c not in existing_cols]

    # Merge on date
    merged_df = existing_data.copy()
    for col in new_cols:
        # Map new column values by date
        date_to_value = dict(zip(new_data[date_col], new_data[col]))
        merged_df[col] = merged_df[date_col].map(date_to_value)

    return merged_df, added, overwritten


def _check_config_compatibility(new_df: pd.DataFrame):
    """Check if existing config is compatible with new dataset columns.

    If columns are missing, prompts user to reset or keep config.
    If columns match, shows a subtle confirmation.
    """
    config = st.session_state.get("current_config")

    if config is None:
        return  # No config to check

    new_columns = set(new_df.columns)

    # Get columns referenced by config
    config_columns = set()
    config_columns.add(config.data.date_column)
    config_columns.add(config.data.target_column)
    config_columns.update(ch.name for ch in config.channels)
    config_columns.update(ctrl.name for ctrl in config.controls)

    # Check if all config columns exist in new data
    missing_columns = config_columns - new_columns

    if missing_columns:
        # Columns don't match - prompt user
        st.warning(f"⚠️ New dataset is missing columns used in current config: **{', '.join(sorted(missing_columns))}**")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Reset Config", type="primary", key="reset_config_btn"):
                st.session_state.current_config = None
                st.session_state.model_fitted = False
                st.session_state.current_model = None
                # Also reset config_state to prevent cross-brand contamination
                st.session_state.config_state = {
                    "name": "my_mmm_model",
                    "channels": [],
                    "controls": [],
                    "owned_media": [],
                    "competitors": [],
                    "dummy_variables": [],
                }
                # Clear widget multiselect states
                for key in ["channel_multiselect", "owned_media_multiselect",
                            "competitor_multiselect", "control_multiselect"]:
                    if key in st.session_state:
                        del st.session_state[key]
                # Clear category columns
                st.session_state.category_columns = []
                st.success("Config reset!")
                st.rerun()
        with col2:
            if st.button("Keep Config", key="keep_config_btn"):
                st.info("Config kept. Some features may not work correctly.")
    else:
        # Columns match - keep config silently with subtle confirmation
        st.info(f"✅ Existing config ({config.name}) is compatible with new dataset")


def _show_media_upload_section():
    """Show media data upload section."""
    st.subheader("1️⃣ Media Data (Long Format)")
    st.markdown("""
    Upload your media spend data in **long format** with channel hierarchy levels.
    Expected columns: `date`, `media_channel_lvl1`, `media_channel_lvl2`, etc., `spend`, `impressions`, `clicks`
    """)

    media_file = st.file_uploader(
        "Upload Media Data (CSV)",
        type=["csv"],
        key="media_file_uploader",
        help="Long format file with one row per channel combination per date"
    )

    if media_file is not None:
        try:
            raw_df = pd.read_csv(media_file)
            st.success(f"Media file loaded: {len(raw_df)} rows, {len(raw_df.columns)} columns")

            # Parse and clean
            df, level_cols = parse_media_file(raw_df)

            if not level_cols:
                st.error("Could not detect channel level columns. Expected columns like 'media_channel_lvl1', 'media_channel_lvl2', etc.")
                return None, None, None

            st.info(f"Detected {len(level_cols)} channel levels: {', '.join(level_cols)}")

            # Store raw data
            st.session_state.media_raw_data = df
            st.session_state.media_level_cols = level_cols

            # Show preview
            with st.expander("Preview Media Data", expanded=False):
                st.dataframe(df.head(20), width="stretch")

            # Detect metrics
            metrics = detect_metric_columns(df)
            metric_info = []
            if metrics['spend']:
                metric_info.append(f"Spend: {metrics['spend']}")
            if metrics['impressions']:
                metric_info.append(f"Impressions: {metrics['impressions']}")
            if metrics['clicks']:
                metric_info.append(f"Clicks: {metrics['clicks']}")
            if metric_info:
                st.info(f"Detected metrics: {', '.join(metric_info)}")

            return df, level_cols, metrics

        except Exception as e:
            st.error(f"Error loading media file: {e}")
            return None, None, None

    return None, None, None


def _show_channel_mapping_ui(media_df: pd.DataFrame, level_cols: list[str]):
    """Show channel mapping UI where user assigns variable names."""
    st.subheader("2️⃣ Map Media Channels to Variables")
    st.markdown("""
    Assign a **variable name** to each channel combination. Rows with the same variable name will be **aggregated (summed)**.
    Leave the variable name empty to **exclude** that channel from the final data.
    """)

    # Get unique combinations
    unique_combos = get_unique_channel_combinations(media_df, level_cols)

    if unique_combos.empty:
        st.warning("No channel combinations found.")
        return {}

    st.info(f"Found {len(unique_combos)} unique channel combinations")

    # Initialize or reset mapping in session state if columns don't match
    stored_cols = set(st.session_state.get('channel_mapping_df', pd.DataFrame()).columns) - {'variable_name'}
    current_cols = set(level_cols)
    if 'channel_mapping_df' not in st.session_state or stored_cols != current_cols:
        st.session_state.channel_mapping_df = unique_combos.copy()
        # Clear upload processed flag when mapping is reset
        if "_mapping_upload_processed" in st.session_state:
            del st.session_state["_mapping_upload_processed"]

    # Create editable dataframe
    # Only include level columns and variable_name for editing
    display_cols = level_cols + ['variable_name']

    # === Bulk Edit via CSV ===
    st.markdown("#### Bulk Edit via CSV")
    st.caption("Download the template, edit in Excel/Sheets, then upload to apply mappings in bulk.")

    col_download, col_upload = st.columns(2)

    with col_download:
        # Generate CSV from current mapping
        csv_data = st.session_state.channel_mapping_df[display_cols].to_csv(index=False)
        st.download_button(
            label="📥 Download Mapping CSV",
            data=csv_data,
            file_name="channel_mapping.csv",
            mime="text/csv",
            help="Download current mapping to edit in Excel/Sheets"
        )

    with col_upload:
        uploaded_mapping = st.file_uploader(
            "Upload Mapping CSV",
            type=["csv"],
            key="channel_mapping_upload",
            help="Upload edited CSV to update all mappings at once",
            label_visibility="collapsed"
        )

    if uploaded_mapping is not None:
        # Check if we've already processed this file (prevent reprocessing on rerun)
        if st.session_state.get("_mapping_upload_processed"):
            # Clear the flag and show success (the data is already loaded)
            n_mapped = sum(1 for v in st.session_state.channel_mapping_df['variable_name'] if v.strip())
            st.success(f"Mappings loaded! {n_mapped} channels assigned.")
        else:
            try:
                uploaded_df = pd.read_csv(uploaded_mapping)

                # Validate columns match
                expected_cols = set(display_cols)
                actual_cols = set(uploaded_df.columns)

                if expected_cols != actual_cols:
                    missing = expected_cols - actual_cols
                    extra = actual_cols - expected_cols
                    error_msg = "Column mismatch in uploaded CSV."
                    if missing:
                        error_msg += f" Missing: {list(missing)}."
                    if extra:
                        error_msg += f" Unexpected: {list(extra)}."
                    st.error(error_msg)
                else:
                    # Merge uploaded variable_names with existing combinations
                    # Use left merge to ensure we only accept mappings for existing channel combinations
                    merged = st.session_state.channel_mapping_df[level_cols].merge(
                        uploaded_df,
                        on=level_cols,
                        how='left'
                    )

                    # Fill NaN variable_names with empty string
                    merged['variable_name'] = merged['variable_name'].fillna('')

                    # Update session state
                    st.session_state.channel_mapping_df = merged
                    # Mark as processed so we don't reprocess on rerun
                    st.session_state["_mapping_upload_processed"] = True
                    # Delete the data_editor widget key to force re-initialization
                    if "channel_mapping_editor" in st.session_state:
                        del st.session_state["channel_mapping_editor"]
                    st.rerun()

            except Exception as e:
                st.error(f"Error loading CSV: {e}")

    st.markdown("---")

    # === Manual Edit Table ===
    edit_df = st.session_state.channel_mapping_df[display_cols].copy()

    edited_df = st.data_editor(
        edit_df,
        column_config={
            'variable_name': st.column_config.TextColumn(
                "Variable Name",
                help="Enter the variable name for this channel combination. Same names will be aggregated.",
                default="",
            ),
            **{col: st.column_config.TextColumn(col, disabled=True) for col in level_cols}
        },
        hide_index=True,
        width="stretch",
        key="channel_mapping_editor"
    )

    # Save button to apply changes (prevents constant refreshing)
    if st.button("💾 Save Channel Mapping", key="save_channel_mapping_btn"):
        st.session_state.channel_mapping_df = edited_df
        st.success("Channel mapping saved!")
        st.rerun()

    # Build mapping dict from SAVED state (not edited_df)
    mapping = {}
    for idx, row in st.session_state.channel_mapping_df.iterrows():
        key = tuple(row[col] for col in level_cols)
        var_name = row['variable_name'].strip() if pd.notna(row['variable_name']) else ''
        mapping[key] = var_name

    # Show summary
    assigned = sum(1 for v in mapping.values() if v)
    excluded = sum(1 for v in mapping.values() if not v)
    unique_vars = len(set(v for v in mapping.values() if v))

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Assigned", assigned)
    with col2:
        st.metric("Excluded", excluded)
    with col3:
        st.metric("Unique Variables", unique_vars)

    return mapping


def _show_other_data_upload_section():
    """Show other data upload section."""
    st.subheader("3️⃣ Other Data (Wide Format)")
    st.markdown("""
    Upload your **other data** file containing KPIs, promotions, email, DM, and control variables.
    This should be in **wide format** with one row per date.
    """)

    other_file = st.file_uploader(
        "Upload Other Data (CSV)",
        type=["csv"],
        key="other_file_uploader",
        help="Wide format file with KPIs, promos, controls - one row per date"
    )

    if other_file is not None:
        try:
            df = pd.read_csv(other_file)
            st.success(f"Other data file loaded: {len(df)} rows, {len(df.columns)} columns")

            # Store in session state
            st.session_state.other_raw_data = df

            # Show preview
            with st.expander("Preview Other Data", expanded=False):
                st.dataframe(df.head(20), width="stretch")

            # Detect date column
            date_col = detect_date_column(df)
            if date_col:
                st.info(f"Detected date column: {date_col}")

            return df

        except Exception as e:
            st.error(f"Error loading other data file: {e}")
            return None

    return None


def _show_merge_preview(media_df: pd.DataFrame, level_cols: list[str],
                        mapping: dict, other_df: pd.DataFrame):
    """Show preview of merged data and allow final configuration."""
    st.subheader("4️⃣ Preview & Merge")

    # Date configuration
    col1, col2 = st.columns(2)
    with col1:
        date_col = st.text_input("Date column name", value="date", key="merge_date_col")
    with col2:
        dayfirst = st.checkbox("Dates are day-first format (DD/MM/YYYY)", value=True, key="merge_dayfirst")

    if st.button("🔄 Aggregate & Merge Data", type="primary"):
        with st.spinner("Aggregating media data..."):
            # Aggregate media data
            aggregated_media = aggregate_media_data(
                media_df, level_cols, mapping,
                date_col=date_col, dayfirst=dayfirst
            )

            if aggregated_media.empty:
                st.error("No data after aggregation. Check your variable name mappings.")
                return

            st.success(f"Aggregated media data: {len(aggregated_media)} rows, {len(aggregated_media.columns)} columns")

            # Show aggregated media preview
            with st.expander("Aggregated Media Data", expanded=True):
                st.dataframe(aggregated_media.head(20), width="stretch")

            # Store aggregated media
            st.session_state.aggregated_media = aggregated_media

        with st.spinner("Merging datasets..."):
            # Merge with other data
            merged_df, stats = merge_datasets(
                aggregated_media, other_df,
                date_col=date_col, dayfirst=dayfirst
            )

            st.success(f"Merged dataset: {len(merged_df)} rows, {len(merged_df.columns)} columns")

            # Show merge stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Common Dates", stats['common_dates'])
            with col2:
                st.metric("Media-only Dates", stats['media_only_dates'])
            with col3:
                st.metric("Other-only Dates", stats['other_only_dates'])

            if stats['media_only_dates'] > 0 or stats['other_only_dates'] > 0:
                st.warning(f"Date range aligned: {stats['media_only_dates']} media-only dates "
                           f"and {stats['other_only_dates']} other-only dates excluded.")

            # Show merged preview
            with st.expander("Merged Dataset Preview", expanded=True):
                st.dataframe(merged_df.head(20), width="stretch")

            # Check if we should merge with existing model data (incremental add mode)
            existing_config = st.session_state.get("current_config")
            existing_data = st.session_state.get("current_data")

            if existing_config is not None and existing_data is not None:
                # Incremental add mode - merge this new data with existing model data
                final_merged_df, added, overwritten = _merge_data_incrementally(
                    existing_data, merged_df, date_col, dayfirst
                )

                st.session_state.current_data = final_merged_df
                st.session_state.date_column = date_col
                st.session_state.dayfirst = dayfirst
                st.session_state.merge_complete = True

                # Show what happened
                if added:
                    st.success(f"Added new columns to model: {', '.join(added)}")
                if overwritten:
                    st.info(f"Overwritten columns: {', '.join(overwritten)}")

                st.success(f"Data merged with existing model! Total: {len(final_merged_df)} rows, {len(final_merged_df.columns)} columns")

                # DO NOT reset config_state - keep existing config
            else:
                # Fresh upload mode - existing behavior
                st.session_state.current_data = merged_df
                st.session_state.date_column = date_col
                st.session_state.dayfirst = dayfirst
                st.session_state.merge_complete = True

                # RESET config_state for new data to prevent cross-brand contamination
                # This ensures owned_media, dummies, etc. from previous brands don't carry over
                st.session_state.config_state = {
                    "name": "my_mmm_model",
                    "channels": [],
                    "controls": [],
                    "owned_media": [],
                    "competitors": [],
                    "dummy_variables": [],
                }
                # Clear widget multiselect states so they re-initialize from fresh config_state
                for key in ["channel_multiselect", "owned_media_multiselect",
                            "competitor_multiselect", "control_multiselect"]:
                    if key in st.session_state:
                        del st.session_state[key]
                # Clear category columns
                st.session_state.category_columns = []
                # Clear current_config and model state
                st.session_state.current_config = None
                st.session_state.model_fitted = False
                st.session_state.current_model = None


def _show_final_configuration():
    """Show final configuration after merge."""
    st.subheader("5️⃣ Configure Target Column")

    df = st.session_state.get("current_data")
    if df is None:
        return

    # Get numeric columns
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    # Try to detect KPI columns
    potential_kpi_cols = [col for col in numeric_cols
                         if 'kpi' in col.lower() or 'revenue' in col.lower()
                         or 'sales' in col.lower() or 'conversion' in col.lower()]

    default_idx = 0
    if potential_kpi_cols:
        default_idx = numeric_cols.index(potential_kpi_cols[0])

    target_col = st.selectbox(
        "Select target (KPI) column",
        options=numeric_cols,
        index=default_idx,
        help="This is the variable you want to model (e.g., revenue, conversions)"
    )

    if target_col:
        # Check if target column changed - may need to review priors
        old_target = st.session_state.get("target_column")
        if old_target and old_target != target_col and st.session_state.get("current_config"):
            st.session_state.priors_need_review = True
            st.session_state.priors_set_for_target = old_target

        st.session_state.target_column = target_col

        # Show target stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mean", f"{df[target_col].mean():,.2f}")
        with col2:
            st.metric("Total", f"{df[target_col].sum():,.2f}")
        with col3:
            st.metric("Min", f"{df[target_col].min():,.2f}")
        with col4:
            st.metric("Max", f"{df[target_col].max():,.2f}")

    # Channel detection
    st.subheader("Detected Media Channels")

    # Detect spend columns from merged data
    spend_cols = [col for col in numeric_cols if '_spend' in col.lower() or col.endswith('_spend')]

    if spend_cols:
        st.write(f"Detected {len(spend_cols)} media channel columns:")
        st.write(spend_cols)
        st.session_state.detected_channels = spend_cols

        # Show channel spend totals
        channel_totals = df[spend_cols].sum().sort_values(ascending=False)
        st.bar_chart(channel_totals)
    else:
        st.info("No media channel columns detected. You can configure them in the next step.")

    # Check config compatibility
    _check_config_compatibility(df)

    # Next step button
    st.markdown("---")
    if st.button("✅ Proceed to Configure Model", type="primary"):
        st.session_state.data_ready = True
        st.success("Data ready! Navigate to 'Configure Model' in the sidebar.")


def _show_single_file_upload():
    """Show traditional single file upload (fallback option)."""
    st.markdown("""
    Upload your marketing data in CSV format. The data should contain:
    - A date column (weekly or daily granularity)
    - A target/KPI column (e.g., revenue, conversions)
    - Media spend columns (one per channel)
    - Optional: control variables (promotions, events, etc.)
    """)

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        help="Upload a CSV file with your marketing data"
    )

    if uploaded_file is not None:
        try:
            # Read the file
            df = pd.read_csv(uploaded_file)
            st.success(f"File loaded: {len(df)} rows, {len(df.columns)} columns")

            # Check if we should merge with existing data (incremental add mode)
            existing_config = st.session_state.get("current_config")
            existing_data = st.session_state.get("current_data")

            if existing_config is not None and existing_data is not None:
                # Incremental add mode - merge new data with existing
                date_col = st.session_state.get("date_column", "date")
                dayfirst = st.session_state.get("dayfirst", True)

                if date_col not in df.columns:
                    st.error(f"Date column '{date_col}' not found in uploaded file. "
                             f"Available columns: {list(df.columns)}")
                else:
                    # Merge the data
                    merged_df, added, overwritten = _merge_data_incrementally(
                        existing_data, df, date_col, dayfirst
                    )

                    st.session_state.current_data = merged_df
                    st.session_state.uploaded_filename = uploaded_file.name

                    # Show what happened
                    if added:
                        st.success(f"Added new columns: {', '.join(added)}")
                    if overwritten:
                        st.info(f"Overwritten columns: {', '.join(overwritten)}")

                    # DO NOT reset config_state - keep existing config
                    # Just update the data reference

                    # Show merged data preview
                    st.subheader("Merged Data Preview")
                    st.dataframe(merged_df.head(20), width="stretch")

                    st.success(f"Data merged! Total: {len(merged_df)} rows, {len(merged_df.columns)} columns")
                    return  # Exit early - no need to show the rest of the single file flow

            # Fresh upload mode - existing behavior
            st.session_state.current_data = df
            st.session_state.uploaded_filename = uploaded_file.name

            # RESET config_state for new data to prevent cross-brand contamination
            # This ensures owned_media, dummies, etc. from previous brands don't carry over
            st.session_state.config_state = {
                "name": "my_mmm_model",
                "channels": [],
                "controls": [],
                "owned_media": [],
                "competitors": [],
                "dummy_variables": [],
            }
            # Clear widget multiselect states so they re-initialize from fresh config_state
            for key in ["channel_multiselect", "owned_media_multiselect",
                        "competitor_multiselect", "control_multiselect"]:
                if key in st.session_state:
                    del st.session_state[key]
            # Clear category columns
            st.session_state.category_columns = []
            # Clear current_config and model state
            st.session_state.current_config = None
            st.session_state.model_fitted = False
            st.session_state.current_model = None

            # Check config compatibility with new data
            _check_config_compatibility(df)

            # Show preview
            st.subheader("Data Preview")
            st.dataframe(df.head(20), width="stretch")

            # Column info
            st.subheader("Column Information")

            col1, col2 = st.columns(2)

            with col1:
                st.write("**All Columns:**")
                st.write(list(df.columns))

            with col2:
                st.write("**Data Types:**")
                dtype_df = pd.DataFrame({
                    "Column": df.columns,
                    "Type": df.dtypes.astype(str).values,
                    "Non-Null": df.notna().sum().values,
                })
                st.dataframe(dtype_df, width="stretch", hide_index=True)

            # Basic statistics
            st.subheader("Basic Statistics")

            # Identify numeric columns
            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

            if numeric_cols:
                st.dataframe(
                    df[numeric_cols].describe().round(2),
                    width="stretch"
                )

            # Date column detection
            st.subheader("Configure Date Column")

            # Try to detect date column
            potential_date_cols = []
            for col in df.columns:
                if "date" in col.lower() or "time" in col.lower():
                    potential_date_cols.append(col)

            date_col = st.selectbox(
                "Select date column",
                options=df.columns.tolist(),
                index=df.columns.tolist().index(potential_date_cols[0]) if potential_date_cols else 0,
            )

            dayfirst = st.checkbox("Dates are day-first format (DD/MM/YYYY)", value=True)

            # Try to parse dates
            if date_col:
                try:
                    df[date_col] = pd.to_datetime(df[date_col], dayfirst=dayfirst)
                    st.success(f"Date range: {df[date_col].min()} to {df[date_col].max()}")

                    # Update session state with parsed dates
                    st.session_state.current_data = df
                    st.session_state.date_column = date_col
                    st.session_state.dayfirst = dayfirst

                except Exception as e:
                    st.error(f"Could not parse dates: {e}")

            # Target column selection
            st.subheader("Configure Target Column")

            target_col = st.selectbox(
                "Select target (KPI) column",
                options=numeric_cols,
                index=0 if numeric_cols else None,
                help="This is the variable you want to model (e.g., revenue, conversions)"
            )

            if target_col:
                # Check if target column changed - may need to review priors
                old_target = st.session_state.get("target_column")
                if old_target and old_target != target_col and st.session_state.get("current_config"):
                    st.session_state.priors_need_review = True
                    st.session_state.priors_set_for_target = old_target

                st.session_state.target_column = target_col

                # Show target stats
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Mean", f"{df[target_col].mean():,.2f}")
                with col2:
                    st.metric("Total", f"{df[target_col].sum():,.2f}")
                with col3:
                    st.metric("Min", f"{df[target_col].min():,.2f}")
                with col4:
                    st.metric("Max", f"{df[target_col].max():,.2f}")

            # Channel detection
            st.subheader("Detect Media Channels")

            # Try to auto-detect spend columns
            potential_channels = [
                col for col in numeric_cols
                if "spend" in col.lower() or "cost" in col.lower() or "media" in col.lower()
            ]

            if potential_channels:
                st.write(f"Detected {len(potential_channels)} potential channel columns:")
                st.write(potential_channels)
                st.session_state.detected_channels = potential_channels

                # Show channel spend totals
                channel_totals = df[potential_channels].sum().sort_values(ascending=False)
                st.bar_chart(channel_totals)

            else:
                st.info("No channel columns auto-detected. You can configure them in the next step.")

            # Next step button
            st.markdown("---")
            if st.button("✅ Proceed to Configure Model", type="primary"):
                st.session_state.data_ready = True
                st.success("Data ready! Navigate to 'Configure Model' in the sidebar.")

        except Exception as e:
            st.error(f"Error loading file: {e}")

    else:
        # Show sample data option
        st.markdown("---")
        st.subheader("Or use sample data")

        if st.button("Load Sample Data"):
            # Create sample data
            import numpy as np

            np.random.seed(42)
            dates = pd.date_range(start="2023-01-01", periods=104, freq="W")

            sample_df = pd.DataFrame({
                "time": dates,
                "revenue": np.random.normal(100000, 15000, 104).clip(50000),
                "channel_a_spend": np.random.uniform(5000, 20000, 104),
                "channel_b_spend": np.random.uniform(3000, 15000, 104),
                "channel_c_spend": np.random.uniform(1000, 8000, 104),
                "promo_event": np.random.choice([0, 1], 104, p=[0.85, 0.15]),
            })

            st.session_state.current_data = sample_df
            st.session_state.date_column = "time"
            st.session_state.target_column = "revenue"
            st.session_state.detected_channels = ["channel_a_spend", "channel_b_spend", "channel_c_spend"]

            st.success("Sample data loaded!")
            st.dataframe(sample_df.head(10))
            st.rerun()


def show():
    """Show the data upload page."""
    st.title("📁 Upload Data")

    # Check for demo mode
    if st.session_state.get("demo_mode", False):
        st.info("**Demo Mode**: Using simulated data. Go to **Results** to explore!")

        # Show demo data preview
        demo = st.session_state.get("demo")
        if demo is not None:
            st.subheader("Demo Data Preview")
            st.dataframe(demo.df_scaled.head(10), width="stretch")
        st.stop()

    # Check if model is loaded - show incremental add message
    if st.session_state.get("current_config") is not None:
        config_name = st.session_state.current_config.name
        st.info(
            f"**Model loaded: {config_name}**\n\n"
            "Any data uploaded will be **merged** with the existing model's data "
            "(joined on date column). New columns will be added; existing columns "
            "with the same name will be overwritten."
        )

        # Option to start fresh
        if st.button("Start Fresh (Clear Model)", key="clear_model_for_new_data"):
            st.session_state.current_config = None
            st.session_state.current_model = None
            st.session_state.model_fitted = False
            st.session_state.current_data = None
            st.session_state.config_state = {
                "name": "my_mmm_model",
                "channels": [],
                "controls": [],
                "owned_media": [],
                "competitors": [],
                "dummy_variables": [],
            }
            # Clear widget multiselect states
            for key in ["channel_multiselect", "owned_media_multiselect",
                        "competitor_multiselect", "control_multiselect"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.session_state.category_columns = []
            st.rerun()

        st.markdown("---")

    # Upload mode selection
    upload_mode = st.radio(
        "Select upload mode",
        options=["Two-file upload (Media + Other)", "Single file upload"],
        horizontal=True,
        help="Use two-file upload if you have separate media data (long format) and other data (wide format)"
    )

    st.markdown("---")

    if upload_mode == "Two-file upload (Media + Other)":
        # Two-file upload flow
        media_df, level_cols, metrics = _show_media_upload_section()

        if media_df is not None and level_cols:
            st.markdown("---")
            mapping = _show_channel_mapping_ui(media_df, level_cols)

            if mapping and any(v for v in mapping.values()):
                st.markdown("---")
                other_df = _show_other_data_upload_section()

                if other_df is not None:
                    st.markdown("---")
                    _show_merge_preview(media_df, level_cols, mapping, other_df)

                    # Show final configuration if merge is complete
                    if st.session_state.get("merge_complete"):
                        st.markdown("---")
                        _show_final_configuration()
    else:
        # Single file upload (original behavior)
        _show_single_file_upload()
