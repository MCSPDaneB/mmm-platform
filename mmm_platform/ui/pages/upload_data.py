"""
Data upload page for MMM Platform.
"""

import streamlit as st
import pandas as pd
from io import StringIO

from mmm_platform.core.validation import DataValidator, ValidationResult
from mmm_platform.config.schema import DataConfig


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
                st.success("Config reset!")
                st.rerun()
        with col2:
            if st.button("Keep Config", key="keep_config_btn"):
                st.info("Config kept. Some features may not work correctly.")
    else:
        # Columns match - keep config silently with subtle confirmation
        st.info(f"✅ Existing config ({config.name}) is compatible with new dataset")


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

            # Store in session state
            st.session_state.current_data = df
            st.session_state.uploaded_filename = uploaded_file.name

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
