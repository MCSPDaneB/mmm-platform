"""
Export page for MMM Platform.

Generates CSV files in specific formats for upload to external visualization platforms.
"""

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
    )

    st.title("Export Data")
    st.caption("Generate CSV files for upload to external visualization platforms")

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
                df_decomps = generate_decomps_stacked(wrapper, config, brand)
                csv_decomps = df_decomps.to_csv(index=False)

                st.download_button(
                    label="Download decomps_stacked.csv",
                    data=csv_decomps,
                    file_name=f"decomps_stacked_{timestamp}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

                with st.expander("Preview (first 10 rows)"):
                    st.dataframe(df_decomps.head(10), use_container_width=True)

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
                    use_container_width=True
                )

                with st.expander("Preview (first 10 rows)"):
                    st.dataframe(df_media.head(10), use_container_width=True)

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
                    use_container_width=True
                )

                with st.expander("Preview (first 10 rows)"):
                    st.dataframe(df_fit.head(10), use_container_width=True)

                st.caption(f"{len(df_fit):,} rows × {len(df_fit.columns)} columns")

            except Exception as e:
                st.error(f"Error generating actual_vs_fitted: {e}")

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
