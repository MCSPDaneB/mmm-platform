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
            selected_client = st.selectbox(
                "Filter by Client",
                options=client_options,
                key="export_client_filter",
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


def _show_disaggregation_ui(wrapper, config, brand: str, key_suffix: str = ""):
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
    key_suffix : str
        Suffix to add to all widget keys (for multiple instances)
    """
    from mmm_platform.analysis.export import generate_disaggregated_results

    # File upload
    granular_file = st.file_uploader(
        "Upload granular mapping file",
        type=["csv", "xlsx"],
        help="CSV or Excel file with granular-level data (e.g., placement-level spend/attribution)",
        key=f"granular_file_uploader{key_suffix}"
    )

    if granular_file is not None:
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

            # Get model channel names for dropdown
            model_channels = [ch.name for ch in config.channels]

            # Column mapping UI
            st.markdown("#### Column Mapping")

            col1, col2 = st.columns(2)

            with col1:
                granular_name_col = st.selectbox(
                    "Granular Name Column",
                    options=all_columns,
                    help="Column containing granular identifiers (e.g., placement_name, campaign_name)",
                    key=f"granular_name_col{key_suffix}"
                )

                date_col_granular = st.selectbox(
                    "Date Column",
                    options=all_columns,
                    help="Column containing dates",
                    key=f"granular_date_col{key_suffix}"
                )

            with col2:
                weight_col = st.selectbox(
                    "Weight Column",
                    options=numeric_columns if numeric_columns else all_columns,
                    help="Numeric column to use for proportional allocation (e.g., spend, impressions, attribution)",
                    key=f"granular_weight_col{key_suffix}"
                )

            # Model channel mapping
            st.markdown("#### Map Granular Names to Model Channels")
            st.caption(
                "For each unique value in your granular file, select which model channel it maps to. "
                "Leave as '-- Not Mapped --' to exclude from disaggregation."
            )

            # Get unique granular values that need mapping
            unique_granular = granular_df[granular_name_col].unique().tolist()

            # Create mapping DataFrame
            mapping_options = ["-- Not Mapped --"] + model_channels

            # Check for existing mapping in session state (unique per model)
            mapping_key = f"granular_mapping{key_suffix}"
            if mapping_key not in st.session_state:
                st.session_state[mapping_key] = {}

            # Build mapping table data
            mapping_data = []
            for granular_val in unique_granular[:50]:  # Limit to first 50 for performance
                # Try to auto-match by checking if granular name contains channel name
                auto_match = "-- Not Mapped --"
                for ch in model_channels:
                    if ch.lower() in str(granular_val).lower() or str(granular_val).lower() in ch.lower():
                        auto_match = ch
                        break

                saved_mapping = st.session_state[mapping_key].get(str(granular_val), auto_match)
                mapping_data.append({
                    "Granular Name": str(granular_val),
                    "Model Channel": saved_mapping,
                })

            if len(unique_granular) > 50:
                st.warning(f"Showing first 50 of {len(unique_granular)} unique granular values. "
                          "All values will be processed based on exact matches to mapped values.")

            mapping_df = pd.DataFrame(mapping_data)

            edited_mapping = st.data_editor(
                mapping_df,
                column_config={
                    "Granular Name": st.column_config.TextColumn("Granular Name", disabled=True),
                    "Model Channel": st.column_config.SelectboxColumn(
                        "Model Channel",
                        options=mapping_options,
                        help="Select which model channel this maps to"
                    ),
                },
                hide_index=True,
                width="stretch",
                key=f"granular_mapping_editor{key_suffix}",
            )

            # Save mapping to session state
            for _, row in edited_mapping.iterrows():
                st.session_state[mapping_key][row["Granular Name"]] = row["Model Channel"]

            # Apply mapping to granular DataFrame
            granular_df["_model_channel"] = granular_df[granular_name_col].map(
                lambda x: st.session_state[mapping_key].get(str(x), "-- Not Mapped --")
            )

            # Filter out unmapped rows
            mapped_df = granular_df[granular_df["_model_channel"] != "-- Not Mapped --"].copy()

            # Show mapping summary
            st.markdown("#### Mapping Summary")
            mapped_count = len(mapped_df)
            total_count = len(granular_df)
            mapped_channels = mapped_df["_model_channel"].nunique()

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Mapped Rows", f"{mapped_count:,} / {total_count:,}")
            with col2:
                st.metric("Mapped Channels", f"{mapped_channels} / {len(model_channels)}")
            with col3:
                unmapped_channels = set(model_channels) - set(mapped_df["_model_channel"].unique())
                st.metric("Unmapped Channels", len(unmapped_channels))

            if unmapped_channels:
                with st.expander("Unmapped Model Channels"):
                    st.write("These channels have no granular mapping and will be output at channel level:")
                    for ch in sorted(unmapped_channels):
                        st.write(f"- {ch}")

            # Generate and download disaggregated results
            if mapped_count > 0:
                st.markdown("---")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                if st.button("Generate Disaggregated Export", type="primary", key=f"generate_disagg_btn{key_suffix}"):
                    with st.spinner("Generating disaggregated results..."):
                        try:
                            df_disagg = generate_disaggregated_results(
                                wrapper=wrapper,
                                config=config,
                                granular_df=mapped_df,
                                granular_name_col=granular_name_col,
                                date_col=date_col_granular,
                                model_channel_col="_model_channel",
                                weight_col=weight_col,
                                brand=brand
                            )

                            result_key = f"disagg_result{key_suffix}"
                            st.session_state[result_key] = df_disagg

                            st.success(f"Generated {len(df_disagg):,} rows")

                        except Exception as e:
                            st.error(f"Error generating disaggregated results: {e}")

                # Show results and download if generated
                result_key = f"disagg_result{key_suffix}"
                if result_key in st.session_state and st.session_state[result_key] is not None:
                    df_disagg = st.session_state[result_key]

                    with st.expander("Preview (first 20 rows)", expanded=True):
                        st.dataframe(df_disagg.head(20), width="stretch")

                    csv_disagg = df_disagg.to_csv(index=False)
                    st.download_button(
                        label="Download disaggregated_results.csv",
                        data=csv_disagg,
                        file_name=f"disaggregated_results{key_suffix}_{timestamp}.csv",
                        mime="text/csv",
                        key=f"download_disagg_btn{key_suffix}"
                    )
            else:
                st.warning("No rows are mapped to model channels. Please map at least one granular value.")

        except Exception as e:
            st.error(f"Error loading file: {e}")


def _show_single_model_export(
    wrapper,
    config,
    brand: str,
    generate_decomps_stacked,
    generate_media_results,
    generate_actual_vs_fitted
):
    """Show single model export UI."""
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

    st.markdown("---")

    # Export files section
    st.subheader("Platform Export Files")
    st.info(
        "These CSV files are formatted for upload to external visualization platforms. "
        "Each file follows a specific schema with stacked/long format data."
    )

    # Generate timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Three columns for the three export types
    col1, col2, col3 = st.columns(3)

    # 1. Decomps Stacked
    with col1:
        st.markdown("### Decomps Stacked")
        st.markdown(
            "All decomposition components (channels, controls, base) "
            "in stacked format with category groupings."
        )

        with st.spinner("Generating decomps_stacked..."):
            try:
                df_decomps = generate_decomps_stacked(wrapper, config, brand, force_to_actuals)
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

            except Exception as e:
                st.error(f"Error generating decomps_stacked: {e}")

    # 2. Media Results
    with col2:
        st.markdown("### Media Results")
        st.markdown(
            "Media channels only with spend data. "
            "Impressions/clicks are placeholders (0) for now."
        )

        with st.spinner("Generating mmm_media_results..."):
            try:
                df_media = generate_media_results(wrapper, config, brand)
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

            except Exception as e:
                st.error(f"Error generating mmm_media_results: {e}")

    # 3. Actual vs Fitted
    with col3:
        st.markdown("### Actual vs Fitted")
        st.markdown(
            "Model fit comparison with actual and fitted values "
            "in long format (two rows per date)."
        )

        with st.spinner("Generating actual_vs_fitted..."):
            try:
                df_fit = generate_actual_vs_fitted(wrapper, config, brand)
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

            except Exception as e:
                st.error(f"Error generating actual_vs_fitted: {e}")

    # Download All button
    st.markdown("---")
    st.subheader("Download All Files")

    try:
        # Generate all dataframes
        df_decomps = generate_decomps_stacked(wrapper, config, brand, force_to_actuals)
        df_media = generate_media_results(wrapper, config, brand)
        df_fit = generate_actual_vs_fitted(wrapper, config, brand)

        # Create ZIP file in memory
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(f"decomps_stacked_{timestamp}.csv", df_decomps.to_csv(index=False))
            zf.writestr(f"mmm_media_results_{timestamp}.csv", df_media.to_csv(index=False))
            zf.writestr(f"actual_vs_fitted_{timestamp}.csv", df_fit.to_csv(index=False))

        zip_buffer.seek(0)

        st.download_button(
            label="Download All (ZIP)",
            data=zip_buffer.getvalue(),
            file_name=f"{brand}_exports_{timestamp}.zip",
            mime="application/zip",
            type="primary",
            key="single_download_all"
        )
        st.caption("Downloads all three CSV files in a single ZIP archive")

    except Exception as e:
        st.error(f"Error creating ZIP: {e}")

    # Disaggregation Section (only if enabled)
    if enable_disagg:
        st.markdown("---")
        st.subheader("Disaggregation Settings")
        st.caption(
            "Split model results to a more granular level (e.g., placements, campaigns) "
            "using proportional weighting from an uploaded file."
        )
        _show_disaggregation_ui(wrapper, config, brand, key_suffix="")

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

    st.markdown("---")

    # Build wrappers_with_labels using already-loaded wrappers
    wrappers_with_labels = []
    for i, row in edited_config.iterrows():
        # Find the wrapper that matches this path
        wrapper = next(w for w, n, p in loaded_wrappers if p == row["Path"])
        wrappers_with_labels.append((wrapper, row["Label"]))

    # Get labels for display
    label_list = [label for _, label in wrappers_with_labels]
    label_display = ", ".join([f"kpi_{l}" for l in label_list])

    # Export files section
    st.subheader("Platform Export Files")
    st.info(
        f"These CSV files combine data from all models with separate columns: "
        f"{label_display}, kpi_total"
    )

    # Generate timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Three columns for the three export types
    col1, col2, col3 = st.columns(3)

    # 1. Decomps Stacked
    with col1:
        st.markdown("### Decomps Stacked")
        st.markdown(
            f"All decomposition components with {label_display}, `kpi_total` columns."
        )

        with st.spinner("Generating combined decomps_stacked..."):
            try:
                df_decomps = generate_combined_decomps_stacked(wrappers_with_labels, brand, force_to_actuals)
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

            except Exception as e:
                st.error(f"Error generating decomps_stacked: {e}")

    # 2. Media Results
    with col2:
        st.markdown("### Media Results")
        st.markdown(
            f"Media channels with `spend`, {label_display}, `kpi_total`."
        )

        with st.spinner("Generating combined mmm_media_results..."):
            try:
                df_media = generate_combined_media_results(wrappers_with_labels, brand)
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

            except Exception as e:
                st.error(f"Error generating mmm_media_results: {e}")

    # 3. Actual vs Fitted
    with col3:
        st.markdown("### Actual vs Fitted")
        value_label_display = ", ".join([f"value_{l}" for l in label_list])
        st.markdown(
            f"Model fit with {value_label_display}, `value_total` columns."
        )

        with st.spinner("Generating combined actual_vs_fitted..."):
            try:
                df_fit = generate_combined_actual_vs_fitted(wrappers_with_labels, brand)
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

            except Exception as e:
                st.error(f"Error generating actual_vs_fitted: {e}")

    # Download All button for combined exports
    st.markdown("---")
    st.subheader("Download All Files")

    try:
        # Generate all dataframes
        df_decomps = generate_combined_decomps_stacked(wrappers_with_labels, brand, force_to_actuals)
        df_media = generate_combined_media_results(wrappers_with_labels, brand)
        df_fit = generate_combined_actual_vs_fitted(wrappers_with_labels, brand)

        # Create ZIP file in memory
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(f"decomps_stacked_combined_{timestamp}.csv", df_decomps.to_csv(index=False))
            zf.writestr(f"mmm_media_results_combined_{timestamp}.csv", df_media.to_csv(index=False))
            zf.writestr(f"actual_vs_fitted_combined_{timestamp}.csv", df_fit.to_csv(index=False))

        zip_buffer.seek(0)

        st.download_button(
            label="Download All (ZIP)",
            data=zip_buffer.getvalue(),
            file_name=f"{brand}_combined_exports_{timestamp}.zip",
            mime="application/zip",
            type="primary",
            key="combined_download_all"
        )
        st.caption("Downloads all three CSV files in a single ZIP archive")

    except Exception as e:
        st.error(f"Error creating ZIP: {e}")

    # Disaggregation Section - per model (only if enabled)
    if enable_disagg:
        st.markdown("---")
        st.subheader("Disaggregation Settings")
        st.caption(
            "Split each model's results independently to granular level (e.g., placements, campaigns) "
            "using proportional weighting from uploaded files."
        )

        for idx, (wrapper, label) in enumerate(wrappers_with_labels):
            with st.expander(f"Disaggregation for {label}", expanded=False):
                _show_disaggregation_ui(
                    wrapper=wrapper,
                    config=wrapper.config,
                    brand=brand,
                    key_suffix=f"_{idx}_{label}"
                )

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
