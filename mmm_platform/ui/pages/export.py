"""
Export page for MMM Platform.

Generates CSV files in specific formats for upload to external visualization platforms.
"""

import io
import zipfile
import streamlit as st
import pandas as pd
from datetime import datetime


def show():
    """Show the export page."""
    # Lazy import to avoid circular import deadlock
    from mmm_platform.analysis.export import (
        generate_decomps_stacked,
        generate_media_results,
        generate_actual_vs_fitted,
        generate_combined_decomps_stacked,
        generate_combined_media_results,
        generate_combined_actual_vs_fitted,
    )
    from mmm_platform.model.persistence import ModelPersistence, list_clients
    from mmm_platform.model.mmm import MMMWrapper

    st.title("Export Data")
    st.caption("Generate CSV files for upload to external visualization platforms")

    # Model Selection Section
    st.subheader("Select Models")

    # Filter controls
    filter_col1, filter_col2 = st.columns(2)

    # Client filter
    clients = list_clients()
    with filter_col1:
        if clients:
            client_options = ["All Clients"] + clients

            # Sync widget state FROM active_client before render
            active = st.session_state.get("active_client")
            widget_value = active if active and active in clients else "All Clients"
            st.session_state.export_client_filter = widget_value

            def on_export_client_change():
                val = st.session_state.export_client_filter
                st.session_state.active_client = None if val == "All Clients" else val

            selected_client = st.selectbox(
                "Filter by Client",
                options=client_options,
                key="export_client_filter",
                on_change=on_export_client_change,
                help="Show models for a specific client"
            )

            client_filter = "all" if selected_client == "All Clients" else selected_client
        else:
            st.info("No clients found. Please run some models first.")
            st.stop()

    with filter_col2:
        export_fav_filter = st.selectbox(
            "Filter",
            options=["All", "Favorites only", "Non-favorites"],
            key="export_favorite_filter",
            help="Filter by favorite status"
        )

    # Get saved models (client-aware)
    saved_models = ModelPersistence.list_saved_models(client=client_filter)

    # Apply favorites filter
    if export_fav_filter == "Favorites only":
        saved_models = [m for m in saved_models if m.get("is_favorite", False)]
    elif export_fav_filter == "Non-favorites":
        saved_models = [m for m in saved_models if not m.get("is_favorite", False)]

    # Sort favorites to the top, then by created_at descending (most recent first)
    saved_models.sort(key=lambda m: (not m.get("is_favorite", False), m.get("created_at", "")), reverse=True)

    if not saved_models:
        st.warning("No fitted models found. Please run and save at least 1 model first.")
        st.stop()

    # Create options for multiselect with rich labels
    model_options = {}
    for model in saved_models:
        r2 = model.get("r2")
        r2_str = f"R²={r2:.3f}" if r2 is not None else "R²=N/A"
        n_channels = model.get("n_channels", "?")
        created = model.get("created_at", "")[:10]
        name = model.get("config_name", "Unknown")
        model_client = model.get("client", "")
        is_favorite = model.get("is_favorite", False)

        # Add star icon for favorites
        fav_icon = "⭐ " if is_favorite else ""

        # Show client in label when viewing all clients
        if client_filter == "all" and model_client:
            option_label = f"{fav_icon}[{model_client}] {name} ({created}) - {n_channels} channels, {r2_str}"
        else:
            option_label = f"{fav_icon}{name} ({created}) - {n_channels} channels, {r2_str}"
        model_options[option_label] = {
            "path": model["path"],
            "name": name,
            "client": model_client
        }

    selected_options = st.multiselect(
        "Select models to export",
        options=list(model_options.keys()),
        default=[],
        help="Select 1 model for single export, or 2+ models for combined export",
        key="export_model_multiselect"
    )

    # Handle based on selection count
    if len(selected_options) == 0:
        st.info("Please select 1 or more models to export.")
        st.stop()

    st.markdown("---")

    if len(selected_options) == 1:
        # Single model export
        model_info = model_options[selected_options[0]]
        try:
            with st.spinner("Loading model..."):
                wrapper = ModelPersistence.load(model_info["path"], MMMWrapper)
        except Exception as e:
            st.error(f"Error loading model: {e}")
            st.stop()

        _show_single_model_export(
            wrapper=wrapper,
            config=wrapper.config,
            brand=model_info["client"] or "unknown",
            model_path=model_info["path"],
            generate_decomps_stacked=generate_decomps_stacked,
            generate_media_results=generate_media_results,
            generate_actual_vs_fitted=generate_actual_vs_fitted
        )
    else:
        # Combined model export (2+ models)
        st.warning(
            "⚠️ **KPIs must be in the same units to be summable** (e.g., both in revenue £). "
            "The kpi_total column is the sum of all model contributions."
        )

        _show_combined_model_export(
            selected_options=selected_options,
            model_options=model_options,
            client_filter=client_filter,
            ModelPersistence=ModelPersistence,
            MMMWrapper=MMMWrapper,
            generate_combined_decomps_stacked=generate_combined_decomps_stacked,
            generate_combined_media_results=generate_combined_media_results,
            generate_combined_actual_vs_fitted=generate_combined_actual_vs_fitted
        )


def _show_disaggregation_ui(wrapper, config, brand: str, model_path: str = None, key_suffix: str = ""):
    """
    Show disaggregation UI for a single model.

    Parameters
    ----------
    wrapper : MMMWrapper
        The fitted model wrapper
    config : ModelConfig
        The model configuration
    brand : str
        Brand name for exports
    model_path : str, optional
        Path to model directory (for loading/saving disaggregation configs)
    key_suffix : str
        Suffix to add to all widget keys (for multiple instances)

    Returns
    -------
    tuple or None
        (mapped_df, granular_name_cols, date_col, weight_col) if ready, None otherwise
        granular_name_cols is a list of column names forming the entity identifier
    """
    from mmm_platform.model.persistence import (
        load_disaggregation_configs,
        save_disaggregation_config,
        validate_disaggregation_config
    )

    model_channels = [ch.name for ch in config.channels]

    # Load saved disaggregation configs if model_path provided
    saved_configs = []
    if model_path:
        saved_configs = load_disaggregation_configs(model_path)

    # Show saved configs selector if any exist
    selected_saved_config = None
    if saved_configs:
        st.markdown("#### Saved Configurations")

        config_options = ["+ Create New"] + [
            f"{c['name']} ({c['created_at'][:10]})" for c in saved_configs
        ]

        selected_option = st.selectbox(
            "Use saved configuration",
            options=config_options,
            key=f"disagg_saved_config{key_suffix}",
            help="Select a saved configuration or create a new one"
        )

        if selected_option != "+ Create New":
            # Find the selected config
            idx = config_options.index(selected_option) - 1  # -1 for "Create New"
            selected_saved_config = saved_configs[idx]

            # Validate against current model channels
            is_valid, error_msg = validate_disaggregation_config(selected_saved_config, model_channels)

            if not is_valid:
                st.error(f"⚠️ Saved config is invalid: {error_msg}")
                st.info("Please create a new configuration or edit this one.")
                selected_saved_config = None
            else:
                st.success(f"Using saved config: **{selected_saved_config['name']}**")
                with st.expander("Config details"):
                    st.write(f"**Entity columns:** {', '.join(selected_saved_config['granular_name_cols'])}")
                    st.write(f"**Date column:** {selected_saved_config['date_column']}")
                    st.write(f"**Weight column:** {selected_saved_config['weight_column']}")
                    st.write(f"**Mappings:** {len(selected_saved_config['entity_to_channel_mapping'])} entities")

    # File upload
    granular_file = st.file_uploader(
        "Upload granular mapping file",
        type=["csv", "xlsx"],
        help="CSV or Excel file with granular-level data (e.g., placement-level spend/attribution)",
        key=f"granular_file_uploader{key_suffix}"
    )

    if granular_file is None:
        return None

    # Load the file
    try:
        if granular_file.name.endswith('.csv'):
            granular_df = pd.read_csv(granular_file)
        else:
            granular_df = pd.read_excel(granular_file)

        st.success(f"Loaded {len(granular_df):,} rows × {len(granular_df.columns)} columns")

        # Get all columns for mapping
        all_columns = granular_df.columns.tolist()
        numeric_columns = granular_df.select_dtypes(include=['number']).columns.tolist()

        # Column mapping UI
        st.markdown("#### Column Mapping")

        col1, col2 = st.columns(2)

        # Pre-populate from saved config if available
        default_entity_cols = selected_saved_config['granular_name_cols'] if selected_saved_config else []
        default_entity_cols = [c for c in default_entity_cols if c in all_columns]  # Filter to valid columns

        default_date_col = selected_saved_config['date_column'] if selected_saved_config else None
        default_date_idx = all_columns.index(default_date_col) if default_date_col and default_date_col in all_columns else 0

        default_weight_col = selected_saved_config['weight_column'] if selected_saved_config else None
        weight_options = numeric_columns if numeric_columns else all_columns
        default_weight_idx = weight_options.index(default_weight_col) if default_weight_col and default_weight_col in weight_options else 0

        with col1:
            granular_name_cols = st.multiselect(
                "Entity Identifier Column(s)",
                options=all_columns,
                default=default_entity_cols,
                help="Select one or more columns to form the unique entity identifier (e.g., Region + Store_ID)",
                key=f"granular_name_cols{key_suffix}"
            )

            date_col_granular = st.selectbox(
                "Date Column",
                options=all_columns,
                index=default_date_idx,
                help="Column containing dates",
                key=f"granular_date_col{key_suffix}"
            )

        with col2:
            weight_col = st.selectbox(
                "Weight Column",
                options=weight_options,
                index=default_weight_idx,
                help="Numeric column to use for proportional allocation (e.g., spend, impressions, attribution)",
                key=f"granular_weight_col{key_suffix}"
            )

        # Validate entity columns selected
        if not granular_name_cols:
            st.warning("Please select at least one entity identifier column.")
            return None

        # Model channel mapping
        st.markdown("#### Map Granular Entities to Model Channels")
        st.caption(
            "For each unique entity in your granular file, select which model channel it maps to. "
            "Leave as '-- Not Mapped --' to exclude from disaggregation."
        )

        # Get unique entity combinations (composite key from multiple columns)
        unique_entities_df = granular_df[granular_name_cols].drop_duplicates()

        # Create composite key for display and mapping
        def make_composite_key(row):
            return " | ".join(str(row[col]) for col in granular_name_cols)

        unique_entities_df["_composite_key"] = unique_entities_df.apply(make_composite_key, axis=1)
        unique_granular = unique_entities_df["_composite_key"].tolist()

        # Create mapping DataFrame
        mapping_options = ["-- Not Mapped --"] + model_channels

        # Initialize mapping from saved config or session state
        mapping_key = f"granular_mapping{key_suffix}"
        if mapping_key not in st.session_state:
            if selected_saved_config:
                # Use saved config mapping
                st.session_state[mapping_key] = selected_saved_config.get('entity_to_channel_mapping', {}).copy()
            else:
                st.session_state[mapping_key] = {}

        # CSV Download/Upload for bulk mapping
        st.markdown("**Bulk Mapping Options**")
        csv_col1, csv_col2 = st.columns(2)

        with csv_col1:
            # Build template with ALL entities (not just first 50)
            template_data = []
            for granular_val in unique_granular:
                current_mapping = st.session_state[mapping_key].get(str(granular_val), "-- Not Mapped --")
                # Auto-match if not already mapped
                if current_mapping == "-- Not Mapped --" or current_mapping not in mapping_options:
                    for ch in model_channels:
                        if ch.lower() in str(granular_val).lower() or str(granular_val).lower() in ch.lower():
                            current_mapping = ch
                            break
                    else:
                        current_mapping = "-- Not Mapped --"
                template_data.append({
                    "entity": str(granular_val),
                    "model_channel": current_mapping
                })

            template_df = pd.DataFrame(template_data)
            st.download_button(
                "Download Mapping Template",
                data=template_df.to_csv(index=False),
                file_name="mapping_template.csv",
                mime="text/csv",
                key=f"download_mapping_template{key_suffix}",
                help=f"Download CSV with all {len(unique_granular)} entities for bulk editing"
            )

        with csv_col2:
            uploaded_mapping = st.file_uploader(
                "Upload Mapping CSV",
                type=["csv"],
                key=f"mapping_upload{key_suffix}",
                help="Upload completed mapping CSV (columns: entity, model_channel)"
            )

        if uploaded_mapping is not None:
            try:
                mapping_csv = pd.read_csv(uploaded_mapping)
                if "entity" in mapping_csv.columns and "model_channel" in mapping_csv.columns:
                    imported_count = 0
                    invalid_channels = []
                    for _, row in mapping_csv.iterrows():
                        entity = str(row["entity"])
                        channel = str(row["model_channel"])
                        if channel in mapping_options:
                            st.session_state[mapping_key][entity] = channel
                            imported_count += 1
                        elif channel and channel != "nan":
                            invalid_channels.append(channel)

                    if imported_count > 0:
                        st.success(f"Imported {imported_count} mappings from CSV")
                    if invalid_channels:
                        st.warning(f"Skipped invalid channels: {set(invalid_channels)}")
                else:
                    st.error("CSV must have 'entity' and 'model_channel' columns")
            except Exception as e:
                st.error(f"Error reading CSV: {e}")

        # Build mapping table data (first 50 for UI display)
        mapping_data = []
        for granular_val in unique_granular[:50]:
            # Check saved mapping first
            if str(granular_val) in st.session_state[mapping_key]:
                saved_mapping = st.session_state[mapping_key][str(granular_val)]
                # Validate the saved mapping is still valid
                if saved_mapping not in mapping_options:
                    saved_mapping = "-- Not Mapped --"
            else:
                # Try to auto-match by checking if entity name contains channel name
                saved_mapping = "-- Not Mapped --"
                for ch in model_channels:
                    if ch.lower() in str(granular_val).lower() or str(granular_val).lower() in ch.lower():
                        saved_mapping = ch
                        break

            mapping_data.append({
                "Entity": str(granular_val),
                "Model Channel": saved_mapping,
            })

        if len(unique_granular) > 50:
            st.info(f"Showing first 50 of {len(unique_granular)} unique entities in the editor. "
                    "Use the CSV download/upload for bulk mapping of all entities.")

        mapping_df = pd.DataFrame(mapping_data)

        # Wrap data_editor in form to prevent rerender on each change
        with st.form(key=f"mapping_form{key_suffix}"):
            edited_mapping = st.data_editor(
                mapping_df,
                column_config={
                    "Entity": st.column_config.TextColumn("Entity", disabled=True),
                    "Model Channel": st.column_config.SelectboxColumn(
                        "Model Channel",
                        options=mapping_options,
                        help="Select which model channel this maps to"
                    ),
                },
                hide_index=True,
                use_container_width=True,
                key=f"granular_mapping_editor{key_suffix}",
            )

            apply_btn = st.form_submit_button("Apply Mappings", type="primary")

        # Save mapping to session state only when form is submitted
        if apply_btn:
            for _, row in edited_mapping.iterrows():
                st.session_state[mapping_key][row["Entity"]] = row["Model Channel"]
            st.rerun()

        # Create composite key column in granular_df for mapping
        granular_df["_composite_key"] = granular_df.apply(make_composite_key, axis=1)

        # Apply mapping to granular DataFrame using composite key
        granular_df["_model_channel"] = granular_df["_composite_key"].map(
            lambda x: st.session_state[mapping_key].get(str(x), "-- Not Mapped --")
        )

        # Filter out unmapped rows
        mapped_df = granular_df[granular_df["_model_channel"] != "-- Not Mapped --"].copy()

        # Show mapping summary
        st.markdown("#### Mapping Summary")
        mapped_count = len(mapped_df)
        total_count = len(granular_df)
        mapped_channels_count = mapped_df["_model_channel"].nunique()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mapped Rows", f"{mapped_count:,} / {total_count:,}")
        with col2:
            st.metric("Mapped Channels", f"{mapped_channels_count} / {len(model_channels)}")
        with col3:
            unmapped_channels = set(model_channels) - set(mapped_df["_model_channel"].unique())
            st.metric("Unmapped Channels", len(unmapped_channels))

        if unmapped_channels:
            with st.expander("Unmapped Model Channels"):
                st.write("These channels have no granular mapping and will be output at channel level:")
                for ch in sorted(unmapped_channels):
                    st.write(f"- {ch}")

        # Save configuration button
        if model_path and mapped_count > 0:
            st.markdown("---")
            st.markdown("#### Save Configuration")
            save_col1, save_col2 = st.columns([3, 1])

            with save_col1:
                config_name = st.text_input(
                    "Configuration name",
                    value=selected_saved_config['name'] if selected_saved_config else f"Disagg by {weight_col}",
                    key=f"disagg_config_name{key_suffix}",
                    placeholder="e.g., Placements by Spend"
                )

            with save_col2:
                if st.button("💾 Save Config", key=f"save_disagg_config{key_suffix}"):
                    if config_name:
                        new_config = {
                            "id": selected_saved_config['id'] if selected_saved_config else None,
                            "name": config_name,
                            "created_at": datetime.now().isoformat(),
                            "granular_name_cols": granular_name_cols,
                            "date_column": date_col_granular,
                            "weight_column": weight_col,
                            "entity_to_channel_mapping": dict(st.session_state[mapping_key]),
                        }
                        save_disaggregation_config(model_path, new_config)
                        st.success(f"Saved configuration: {config_name}")
                        st.rerun()
                    else:
                        st.warning("Please enter a configuration name")

        if mapped_count > 0:
            return (mapped_df, granular_name_cols, date_col_granular, weight_col)
        else:
            st.warning("No rows are mapped to model channels. Please map at least one entity.")
            return None

    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None


def _show_single_model_export(
    wrapper,
    config,
    brand: str,
    model_path: str,
    generate_decomps_stacked,
    generate_media_results,
    generate_actual_vs_fitted
):
    """Show single model export UI."""
    from mmm_platform.analysis.export import generate_disaggregated_results

    # Brand input section
    st.subheader("Export Settings")

    col1, col2 = st.columns([1, 2])
    with col1:
        # Get brand from config or use provided default
        default_brand = config.data.brand if config.data.brand else brand
        brand = st.text_input(
            "Brand Name",
            value=default_brand,
            help="Brand name to include in all export files",
            placeholder="Enter brand name (e.g., bevmo)"
        )

    if not brand:
        st.warning("Please enter a brand name to enable exports.")
        st.stop()

    # Export options
    force_to_actuals = st.checkbox(
        "Force decomposition to actuals",
        value=False,
        help="Absorb residuals into intercept so decomposition sums to actual values instead of fitted",
        key="single_force_to_actuals"
    )

    enable_disagg = st.checkbox(
        "Enable disaggregation",
        value=False,
        help="Split model results to granular level (e.g., placements) using proportional weighting",
        key="enable_disaggregation"
    )

    # Disaggregation settings (if enabled) - BEFORE prepare button
    disagg_config = None
    if enable_disagg:
        st.markdown("---")
        st.subheader("Disaggregation Settings")
        st.caption(
            "Split model results to a more granular level (e.g., placements, campaigns) "
            "using proportional weighting from an uploaded file."
        )
        disagg_config = _show_disaggregation_ui(wrapper, config, brand, model_path=model_path, key_suffix="")

    st.markdown("---")

    # Prepare button
    if st.button("Prepare Export Files", type="primary", key="single_prepare_btn"):
        with st.spinner("Generating export files..."):
            try:
                # Generate main exports
                st.session_state["export_df_decomps"] = generate_decomps_stacked(wrapper, config, brand, force_to_actuals)
                st.session_state["export_df_media"] = generate_media_results(wrapper, config, brand)
                st.session_state["export_df_fit"] = generate_actual_vs_fitted(wrapper, config, brand)

                # Generate disaggregated results if configured
                if enable_disagg and disagg_config is not None:
                    mapped_df, granular_name_cols, date_col, weight_col = disagg_config
                    st.session_state["export_df_disagg"] = generate_disaggregated_results(
                        wrapper=wrapper,
                        config=config,
                        granular_df=mapped_df,
                        granular_name_cols=granular_name_cols,
                        date_col=date_col,
                        model_channel_col="_model_channel",
                        weight_col=weight_col,
                        brand=brand
                    )
                else:
                    st.session_state["export_df_disagg"] = None

                st.session_state["export_files_ready"] = True
                st.session_state["export_brand"] = brand
                st.success("Export files ready!")

            except Exception as e:
                st.error(f"Error generating exports: {e}")
                st.session_state["export_files_ready"] = False

    # Show downloads only if files are ready
    if st.session_state.get("export_files_ready", False):
        st.markdown("---")
        st.subheader("Platform Export Files")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Three columns for the three export types
        col1, col2, col3 = st.columns(3)

        # 1. Decomps Stacked
        with col1:
            st.markdown("### Decomps Stacked")
            df_decomps = st.session_state["export_df_decomps"]
            csv_decomps = df_decomps.to_csv(index=False)

            st.download_button(
                label="Download decomps_stacked.csv",
                data=csv_decomps,
                file_name=f"decomps_stacked_{timestamp}.csv",
                mime="text/csv",
                width="stretch"
            )

            with st.expander("Preview (first 10 rows)"):
                st.dataframe(df_decomps.head(10), width="stretch")

            st.caption(f"{len(df_decomps):,} rows × {len(df_decomps.columns)} columns")

        # 2. Media Results
        with col2:
            st.markdown("### Media Results")
            df_media = st.session_state["export_df_media"]
            csv_media = df_media.to_csv(index=False)

            st.download_button(
                label="Download mmm_media_results.csv",
                data=csv_media,
                file_name=f"mmm_media_results_{timestamp}.csv",
                mime="text/csv",
                width="stretch"
            )

            with st.expander("Preview (first 10 rows)"):
                st.dataframe(df_media.head(10), width="stretch")

            st.caption(f"{len(df_media):,} rows × {len(df_media.columns)} columns")

        # 3. Actual vs Fitted
        with col3:
            st.markdown("### Actual vs Fitted")
            df_fit = st.session_state["export_df_fit"]
            csv_fit = df_fit.to_csv(index=False)

            st.download_button(
                label="Download actual_vs_fitted.csv",
                data=csv_fit,
                file_name=f"actual_vs_fitted_{timestamp}.csv",
                mime="text/csv",
                width="stretch"
            )

            with st.expander("Preview (first 10 rows)"):
                st.dataframe(df_fit.head(10), width="stretch")

            st.caption(f"{len(df_fit):,} rows × {len(df_fit.columns)} columns")

        # Disaggregated results (if generated)
        df_disagg = st.session_state.get("export_df_disagg")
        if df_disagg is not None:
            st.markdown("---")
            st.markdown("### Disaggregated Results")
            csv_disagg = df_disagg.to_csv(index=False)

            st.download_button(
                label="Download disaggregated_results.csv",
                data=csv_disagg,
                file_name=f"disaggregated_results_{timestamp}.csv",
                mime="text/csv",
            )

            with st.expander("Preview (first 20 rows)"):
                st.dataframe(df_disagg.head(20), width="stretch")

            st.caption(f"{len(df_disagg):,} rows × {len(df_disagg.columns)} columns")

        # Download All button
        st.markdown("---")
        st.subheader("Download All Files")

        export_brand = st.session_state.get("export_brand", brand)
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(f"decomps_stacked_{timestamp}.csv", df_decomps.to_csv(index=False))
            zf.writestr(f"mmm_media_results_{timestamp}.csv", df_media.to_csv(index=False))
            zf.writestr(f"actual_vs_fitted_{timestamp}.csv", df_fit.to_csv(index=False))
            if df_disagg is not None:
                zf.writestr(f"disaggregated_results_{timestamp}.csv", df_disagg.to_csv(index=False))

        zip_buffer.seek(0)

        st.download_button(
            label="Download All (ZIP)",
            data=zip_buffer.getvalue(),
            file_name=f"{export_brand}_exports_{timestamp}.zip",
            mime="application/zip",
            type="primary",
            key="single_download_all"
        )
        st.caption("Downloads all CSV files in a single ZIP archive")

    # File format documentation
    st.markdown("---")
    with st.expander("File Format Documentation"):
        st.markdown("""
        ### decomps_stacked.csv

        | Column | Description |
        |--------|-------------|
        | date | Date value |
        | wc_mon | Week commencing Monday |
        | brand | Brand name |
        | decomp | Variable name (e.g., "Google_PMAX_spend", "trend") |
        | decomp_lvl1 | Category level 1 |
        | decomp_lvl2 | Category level 2 |
        | kpi_{target} | Contribution value in real units |

        ### mmm_media_results.csv

        | Column | Description |
        |--------|-------------|
        | date | Date value |
        | wc_mon | Week commencing Monday |
        | brand | Brand name |
        | decomp_lvl1 | Category level 1 |
        | decomp_lvl2 | Category level 2 |
        | spend | Spend in real units |
        | impressions | Impressions (placeholder = 0) |
        | clicks | Clicks (placeholder = 0) |
        | decomp | Channel name |
        | kpi_{target} | Contribution value in real units |

        ### actual_vs_fitted.csv

        | Column | Description |
        |--------|-------------|
        | date | Date value |
        | wc_mon | Week commencing Monday |
        | brand | Brand name |
        | kpi_label | KPI column name |
        | actual_fitted | "Actual" or "Fitted" |
        | value | Value in real units |
        """)

    # Category columns info
    if config.category_columns:
        st.markdown("---")
        st.subheader("Category Columns")
        st.markdown("The following category columns are used for grouping (decomp_lvl1, decomp_lvl2, etc.):")

        for i, cat_col in enumerate(config.category_columns):
            st.markdown(f"**decomp_lvl{i + 1}**: {cat_col.name}")
            if cat_col.options:
                st.caption(f"Options: {', '.join(cat_col.options)}")
    else:
        st.info(
            "No category columns configured. Using display names for decomp_lvl1 and decomp_lvl2. "
            "You can add category columns in the Configure Model page."
        )


def _show_combined_model_export(
    selected_options,
    model_options,
    client_filter,
    ModelPersistence,
    MMMWrapper,
    generate_combined_decomps_stacked,
    generate_combined_media_results,
    generate_combined_actual_vs_fitted
):
    """Show combined model export UI for merging 2+ models."""
    from mmm_platform.analysis.export import generate_disaggregated_results

    # Model Configuration Section
    st.subheader("Configure Labels")
    st.caption("Set custom labels for each model (used in column names like kpi_online, kpi_offline)")

    # Load models first (so we can get KPI names for default labels)
    try:
        with st.spinner("Loading models..."):
            loaded_wrappers = []
            for option in selected_options:
                model_info = model_options[option]
                wrapper = ModelPersistence.load(model_info["path"], MMMWrapper)
                loaded_wrappers.append((wrapper, model_info["name"], model_info["path"]))
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.stop()

    # Build configuration data using KPI name from loaded models as default label
    config_data = []
    for wrapper, name, path in loaded_wrappers:
        # Use the model's target column (KPI name) as the default label
        default_label = wrapper.config.data.target_column

        config_data.append({
            "Model": name,
            "Path": path,
            "Label": default_label,
        })

    # Display editable configuration
    config_df = pd.DataFrame(config_data)

    edited_config = st.data_editor(
        config_df,
        column_config={
            "Model": st.column_config.TextColumn("Model", disabled=True),
            "Path": None,  # Hide path column
            "Label": st.column_config.TextColumn(
                "Label",
                help="Used in column names (e.g., kpi_online_revenue, kpi_instore_revenue)",
                max_chars=30,
            ),
        },
        hide_index=True,
        width="stretch",
        key="export_config_editor",
    )

    # Validate labels
    labels = edited_config["Label"].tolist()
    if len(labels) != len(set(labels)):
        st.error("Labels must be unique for each model.")
        st.stop()

    if any(not label or not label.strip() for label in labels):
        st.warning("Please enter labels for all models.")
        st.stop()

    # Determine brand from client
    # If filtered by a specific client, use that; otherwise use the first model's client
    if client_filter != "all":
        brand = client_filter
    else:
        # Use client from first selected model
        brand = model_options[selected_options[0]]["client"] or "unknown"

    # Build wrappers_with_labels using already-loaded wrappers
    wrappers_with_labels = []
    for i, row in edited_config.iterrows():
        # Find the wrapper that matches this path
        wrapper = next(w for w, n, p in loaded_wrappers if p == row["Path"])
        wrappers_with_labels.append((wrapper, row["Label"]))

    # Export options
    st.subheader("Export Settings")

    force_to_actuals = st.checkbox(
        "Force decomposition to actuals",
        value=False,
        help="Absorb residuals into intercept so decomposition sums to actual values instead of fitted",
        key="combined_force_to_actuals"
    )

    enable_disagg = st.checkbox(
        "Enable disaggregation",
        value=False,
        help="Split each model's results to granular level (e.g., placements) using proportional weighting",
        key="combined_enable_disaggregation"
    )

    # Disaggregation settings per model (if enabled) - BEFORE prepare button
    disagg_configs = {}
    if enable_disagg:
        st.markdown("---")
        st.subheader("Disaggregation Settings")
        st.caption(
            "Split each model's results independently to granular level (e.g., placements, campaigns) "
            "using proportional weighting from uploaded files."
        )

        for idx, row in edited_config.iterrows():
            wrapper = next(w for w, n, p in loaded_wrappers if p == row["Path"])
            label = row["Label"]
            model_path = row["Path"]
            with st.expander(f"Disaggregation for {label}", expanded=False):
                disagg_config = _show_disaggregation_ui(
                    wrapper=wrapper,
                    config=wrapper.config,
                    brand=brand,
                    model_path=model_path,
                    key_suffix=f"_{idx}_{label}"
                )
                if disagg_config is not None:
                    disagg_configs[label] = (wrapper, disagg_config)

    st.markdown("---")

    # Prepare button
    if st.button("Prepare Export Files", type="primary", key="combined_prepare_btn"):
        with st.spinner("Generating export files..."):
            try:
                # Generate main exports
                st.session_state["combined_df_decomps"] = generate_combined_decomps_stacked(wrappers_with_labels, brand, force_to_actuals)
                st.session_state["combined_df_media"] = generate_combined_media_results(wrappers_with_labels, brand)
                st.session_state["combined_df_fit"] = generate_combined_actual_vs_fitted(wrappers_with_labels, brand)

                # Generate disaggregated results for each configured model
                combined_disagg_results = {}
                for label, (wrapper, disagg_config) in disagg_configs.items():
                    mapped_df, granular_name_cols, date_col, weight_col = disagg_config
                    df_disagg = generate_disaggregated_results(
                        wrapper=wrapper,
                        config=wrapper.config,
                        granular_df=mapped_df,
                        granular_name_cols=granular_name_cols,
                        date_col=date_col,
                        model_channel_col="_model_channel",
                        weight_col=weight_col,
                        brand=brand
                    )
                    combined_disagg_results[label] = df_disagg

                st.session_state["combined_disagg_results"] = combined_disagg_results if combined_disagg_results else None

                st.session_state["combined_files_ready"] = True
                st.session_state["combined_brand"] = brand
                st.session_state["combined_labels"] = [label for _, label in wrappers_with_labels]
                st.success("Export files ready!")

            except Exception as e:
                st.error(f"Error generating exports: {e}")
                st.session_state["combined_files_ready"] = False

    # Show downloads only if files are ready
    if st.session_state.get("combined_files_ready", False):
        st.markdown("---")

        label_list = st.session_state.get("combined_labels", [])
        label_display = ", ".join([f"kpi_{l}" for l in label_list])

        st.subheader("Platform Export Files")
        st.info(
            f"These CSV files combine data from all models with separate columns: "
            f"{label_display}, kpi_total"
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Three columns for the three export types
        col1, col2, col3 = st.columns(3)

        # 1. Decomps Stacked
        with col1:
            st.markdown("### Decomps Stacked")
            df_decomps = st.session_state["combined_df_decomps"]
            csv_decomps = df_decomps.to_csv(index=False)

            st.download_button(
                label="Download decomps_stacked.csv",
                data=csv_decomps,
                file_name=f"decomps_stacked_combined_{timestamp}.csv",
                mime="text/csv",
                key="combined_decomps_download",
                type="primary"
            )

            with st.expander("Preview (first 10 rows)"):
                st.dataframe(df_decomps.head(10), width="stretch")

            st.caption(f"{len(df_decomps):,} rows × {len(df_decomps.columns)} columns")

        # 2. Media Results
        with col2:
            st.markdown("### Media Results")
            df_media = st.session_state["combined_df_media"]
            csv_media = df_media.to_csv(index=False)

            st.download_button(
                label="Download mmm_media_results.csv",
                data=csv_media,
                file_name=f"mmm_media_results_combined_{timestamp}.csv",
                mime="text/csv",
                key="combined_media_download",
                type="primary"
            )

            with st.expander("Preview (first 10 rows)"):
                st.dataframe(df_media.head(10), width="stretch")

            st.caption(f"{len(df_media):,} rows × {len(df_media.columns)} columns")

        # 3. Actual vs Fitted
        with col3:
            st.markdown("### Actual vs Fitted")
            df_fit = st.session_state["combined_df_fit"]
            csv_fit = df_fit.to_csv(index=False)

            st.download_button(
                label="Download actual_vs_fitted.csv",
                data=csv_fit,
                file_name=f"actual_vs_fitted_combined_{timestamp}.csv",
                mime="text/csv",
                key="combined_fit_download",
                type="primary"
            )

            with st.expander("Preview (first 10 rows)"):
                st.dataframe(df_fit.head(10), width="stretch")

            st.caption(f"{len(df_fit):,} rows × {len(df_fit.columns)} columns")

        # Disaggregated results (if generated)
        combined_disagg = st.session_state.get("combined_disagg_results")
        if combined_disagg:
            st.markdown("---")
            st.subheader("Disaggregated Results")

            for label, df_disagg in combined_disagg.items():
                st.markdown(f"### {label}")
                csv_disagg = df_disagg.to_csv(index=False)

                st.download_button(
                    label=f"Download disaggregated_{label}.csv",
                    data=csv_disagg,
                    file_name=f"disaggregated_{label}_{timestamp}.csv",
                    mime="text/csv",
                    key=f"combined_disagg_download_{label}"
                )

                with st.expander(f"Preview (first 20 rows)"):
                    st.dataframe(df_disagg.head(20), width="stretch")

                st.caption(f"{len(df_disagg):,} rows × {len(df_disagg.columns)} columns")

        # Download All button
        st.markdown("---")
        st.subheader("Download All Files")

        export_brand = st.session_state.get("combined_brand", brand)
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(f"decomps_stacked_combined_{timestamp}.csv", df_decomps.to_csv(index=False))
            zf.writestr(f"mmm_media_results_combined_{timestamp}.csv", df_media.to_csv(index=False))
            zf.writestr(f"actual_vs_fitted_combined_{timestamp}.csv", df_fit.to_csv(index=False))
            if combined_disagg:
                for label, df_disagg in combined_disagg.items():
                    zf.writestr(f"disaggregated_{label}_{timestamp}.csv", df_disagg.to_csv(index=False))

        zip_buffer.seek(0)

        st.download_button(
            label="Download All (ZIP)",
            data=zip_buffer.getvalue(),
            file_name=f"{export_brand}_combined_exports_{timestamp}.zip",
            mime="application/zip",
            type="primary",
            key="combined_download_all"
        )
        st.caption("Downloads all CSV files in a single ZIP archive")

    # File format documentation for combined exports
    st.markdown("---")
    with st.expander("Combined Export File Format Documentation"):
        st.markdown(f"""
        ### decomps_stacked.csv (Combined)

        | Column | Description |
        |--------|-------------|
        | date | Date value |
        | wc_mon | Week commencing Monday |
        | brand | Brand name |
        | decomp | Variable name |
        | decomp_lvl1 | Category level 1 |
        | decomp_lvl2 | Category level 2 |
        | kpi_{{label}} | Contribution from each model |
        | kpi_total | Sum of all model contributions |

        ### mmm_media_results.csv (Combined)

        | Column | Description |
        |--------|-------------|
        | date | Date value |
        | wc_mon | Week commencing Monday |
        | brand | Brand name |
        | decomp_lvl1 | Category level 1 |
        | decomp_lvl2 | Category level 2 |
        | spend | Business spend (same for all KPIs) |
        | kpi_{{label}} | Contribution from each model |
        | kpi_total | Sum of all model contributions |
        | decomp | Channel name |

        ### actual_vs_fitted.csv (Combined)

        | Column | Description |
        |--------|-------------|
        | date | Date value |
        | wc_mon | Week commencing Monday |
        | brand | Brand name |
        | actual_fitted | "Actual" or "Fitted" |
        | value_{{label}} | Value from each model |
        | value_total | Sum of all model values |
        """)
