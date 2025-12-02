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
    from mmm_platform.model.persistence import ModelPersistence
    from mmm_platform.model.mmm import MMMWrapper

    st.title("Export Data")
    st.caption("Generate CSV files for upload to external visualization platforms")

    # Export mode selection
    st.subheader("Export Mode")
    export_mode = st.radio(
        "Select export mode",
        options=["Single Model", "Combined Models"],
        horizontal=True,
        help="Single Model exports data from the current model. Combined Models merges two models with separate KPI columns."
    )

    st.markdown("---")

    if export_mode == "Single Model":
        _show_single_model_export(
            generate_decomps_stacked,
            generate_media_results,
            generate_actual_vs_fitted
        )
    else:
        _show_combined_model_export(
            ModelPersistence,
            MMMWrapper,
            generate_combined_decomps_stacked,
            generate_combined_media_results,
            generate_combined_actual_vs_fitted
        )


def _show_single_model_export(
    generate_decomps_stacked,
    generate_media_results,
    generate_actual_vs_fitted
):
    """Show single model export UI (original functionality)."""
    # Check if model is fitted
    if not st.session_state.get("model_fitted") or st.session_state.get("current_model") is None:
        st.warning("Please run the model first to export data.")
        st.stop()

    wrapper = st.session_state.current_model
    config = wrapper.config

    # Brand input section
    st.subheader("Export Settings")

    col1, col2 = st.columns([1, 2])
    with col1:
        # Get brand from config or prompt user
        default_brand = config.data.brand if config.data.brand else ""
        brand = st.text_input(
            "Brand Name",
            value=default_brand,
            help="Brand name to include in all export files",
            placeholder="Enter brand name (e.g., bevmo)"
        )

    if not brand:
        st.warning("Please enter a brand name to enable exports.")
        st.stop()

    # Force to actuals checkbox
    force_to_actuals = st.checkbox(
        "Force decomposition to actuals",
        value=False,
        help="Absorb residuals into intercept so decomposition sums to actual values instead of fitted",
        key="single_force_to_actuals"
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
    ModelPersistence,
    MMMWrapper,
    generate_combined_decomps_stacked,
    generate_combined_media_results,
    generate_combined_actual_vs_fitted
):
    """Show combined model export UI for merging two models."""
    from mmm_platform.model.persistence import list_clients

    st.subheader("Combined Model Export")

    # Warning about summability
    st.warning(
        "⚠️ **KPIs must be in the same units to be summable** (e.g., both in revenue £). "
        "The kpi_total column is the sum of both model contributions."
    )

    # Filter controls
    filter_col1, filter_col2 = st.columns(2)

    # Client filter (same design as Combined Analysis)
    clients = list_clients()
    with filter_col1:
        if clients:
            client_options = ["All Clients"] + clients
            selected_client = st.selectbox(
                "Filter by Client",
                options=client_options,
                key="export_client_filter",
                help="Show models for a specific client. Client name will be used as brand."
            )
            client_filter = "all" if selected_client == "All Clients" else selected_client
        else:
            st.error("No clients found. Please run some models first.")
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
        st.warning("No fitted models found. Please run and save at least 2 models first.")
        st.stop()

    if len(saved_models) < 2:
        st.warning(f"At least 2 fitted models are required. Found {len(saved_models)} model(s).")
        st.stop()

    # Model Selection Section (same design as Combined Analysis)
    st.markdown("### 1. Select Models")

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
        "Select models to combine",
        options=list(model_options.keys()),
        default=[],
        help="Select exactly 2 models to combine for export",
        key="export_model_multiselect"
    )

    if len(selected_options) < 2:
        st.info("Please select exactly 2 models to proceed.")
        st.stop()

    if len(selected_options) > 2:
        st.warning("Please select exactly 2 models. You have selected more than 2.")
        st.stop()

    # Model Configuration Section (same design as Combined Analysis)
    st.markdown("### 2. Configure Labels")
    st.caption("Set custom labels for each model (used in column names like kpi_online, kpi_offline)")

    # Build configuration data for selected models
    config_data = []
    default_labels = ["online", "offline"]
    for i, option in enumerate(selected_options):
        model_info = model_options[option]
        name = model_info["name"]

        config_data.append({
            "Model": name,
            "Path": model_info["path"],
            "Label": default_labels[i] if i < len(default_labels) else f"model{i+1}",
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
                help="Used in column names (e.g., kpi_online, kpi_offline)",
                max_chars=20,
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

    # Force to actuals checkbox
    force_to_actuals = st.checkbox(
        "Force decomposition to actuals",
        value=False,
        help="Absorb residuals into intercept so decomposition sums to actual values instead of fitted",
        key="combined_force_to_actuals"
    )

    st.markdown("---")

    # Load models
    try:
        with st.spinner("Loading models..."):
            wrappers_with_labels = []
            for i, row in edited_config.iterrows():
                wrapper = ModelPersistence.load(row["Path"], MMMWrapper)
                wrappers_with_labels.append((wrapper, row["Label"]))
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.stop()

    # Get labels for display
    label1 = wrappers_with_labels[0][1]
    label2 = wrappers_with_labels[1][1]

    # Export files section
    st.subheader("Platform Export Files")
    st.info(
        f"These CSV files combine data from both models with separate columns: "
        f"kpi_{label1}, kpi_{label2}, kpi_total"
    )

    # Generate timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Three columns for the three export types
    col1, col2, col3 = st.columns(3)

    # 1. Decomps Stacked
    with col1:
        st.markdown("### Decomps Stacked")
        st.markdown(
            f"All decomposition components with `kpi_{label1}`, `kpi_{label2}`, `kpi_total` columns."
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
            f"Media channels with `spend`, `kpi_{label1}`, `kpi_{label2}`, `kpi_total`."
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
        st.markdown(
            f"Model fit with `value_{label1}`, `value_{label2}`, `value_total` columns."
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
        | kpi_{label1} | Contribution from Model 1 |
        | kpi_{label2} | Contribution from Model 2 |
        | kpi_total | Sum of both model contributions |

        ### mmm_media_results.csv (Combined)

        | Column | Description |
        |--------|-------------|
        | date | Date value |
        | wc_mon | Week commencing Monday |
        | brand | Brand name |
        | decomp_lvl1 | Category level 1 |
        | decomp_lvl2 | Category level 2 |
        | spend | Business spend (same for all KPIs) |
        | kpi_{label1} | Contribution from Model 1 |
        | kpi_{label2} | Contribution from Model 2 |
        | kpi_total | Sum of both model contributions |
        | decomp | Channel name |

        ### actual_vs_fitted.csv (Combined)

        | Column | Description |
        |--------|-------------|
        | date | Date value |
        | wc_mon | Week commencing Monday |
        | brand | Brand name |
        | actual_fitted | "Actual" or "Fitted" |
        | value_{label1} | Value from Model 1 |
        | value_{label2} | Value from Model 2 |
        | value_total | Sum of both model values |
        """)
