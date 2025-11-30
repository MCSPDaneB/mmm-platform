"""
Model configuration page for MMM Platform.
"""

import streamlit as st
import pandas as pd
from typing import Optional, List

from mmm_platform.config.schema import (
    ModelConfig, ChannelConfig, ControlConfig, DataConfig,
    SamplingConfig, AdstockConfig, SaturationConfig, SeasonalityConfig,
    AdstockType, SignConstraint, PriorConfig, CategoryColumnConfig
)
from mmm_platform.config.loader import ConfigLoader


# Maximum number of custom category columns allowed
MAX_CATEGORY_COLUMNS = 5


def render_category_columns_manager():
    """
    Render the category columns management section.
    This allows users to add/remove custom category columns that appear in both
    channel and control configuration tables.
    """
    # Initialize session state for category columns if not exists
    if "category_columns" not in st.session_state:
        st.session_state.category_columns = []

    st.subheader("Category Columns")
    st.caption(f"Define custom columns for grouping results (max {MAX_CATEGORY_COLUMNS})")

    # Display existing columns
    if st.session_state.category_columns:
        cols = st.columns(min(len(st.session_state.category_columns) + 1, MAX_CATEGORY_COLUMNS + 1))

        for i, cat_col in enumerate(st.session_state.category_columns):
            with cols[i]:
                st.markdown(f"**{cat_col['name']}**")
                st.caption(f"{len(cat_col.get('options', []))} options")
                if st.button("✕", key=f"remove_cat_col_{i}", help="Remove this column"):
                    st.session_state.category_columns.pop(i)
                    st.rerun()

    # Add new column section
    if len(st.session_state.category_columns) < MAX_CATEGORY_COLUMNS:
        with st.expander("➕ Add Category Column", expanded=False):
            new_col_name = st.text_input(
                "Column Name",
                placeholder="e.g., Funnel Stage",
                key="new_cat_col_name"
            )
            new_col_options = st.text_input(
                "Options (comma-separated)",
                placeholder="e.g., Awareness, Consideration, Conversion",
                key="new_cat_col_options"
            )

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Add Column", key="add_cat_col_btn"):
                    if new_col_name:
                        # Check for duplicate names
                        existing_names = [c["name"] for c in st.session_state.category_columns]
                        if new_col_name in existing_names:
                            st.error(f"Column '{new_col_name}' already exists!")
                        else:
                            options = [o.strip() for o in new_col_options.split(",") if o.strip()]
                            st.session_state.category_columns.append({
                                "name": new_col_name,
                                "options": options if options else ["Other"]
                            })
                            st.rerun()
                    else:
                        st.warning("Please enter a column name")
    else:
        st.info(f"Maximum of {MAX_CATEGORY_COLUMNS} category columns reached")


def show():
    """Show the model configuration page."""
    st.title("⚙️ Configure Model")

    # Check for demo mode
    if st.session_state.get("demo_mode", False):
        st.info("**Demo Mode**: Configuration is pre-set with simulated data. Go to **Results** to explore!")
        st.stop()

    # Check if data is loaded
    if st.session_state.get("current_data") is None:
        st.warning("Please upload data first!")
        st.stop()

    df = st.session_state.current_data

    # Tabs for different configuration sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Data Settings",
        "📺 Channels",
        "🎛️ Controls",
        "⚡ Transforms",
        "🎯 Sampling"
    ])

    # Initialize config state if needed
    if "config_state" not in st.session_state:
        st.session_state.config_state = {
            "name": "my_mmm_model",
            "channels": [],
            "controls": [],
        }

    # =========================================================================
    # Tab 1: Data Settings
    # =========================================================================
    with tab1:
        st.subheader("Data Settings")

        col1, col2 = st.columns(2)

        with col1:
            model_name = st.text_input(
                "Model Name",
                value=st.session_state.config_state.get("name", "my_mmm_model"),
                help="A unique name for this model configuration"
            )

            date_col = st.selectbox(
                "Date Column",
                options=df.columns.tolist(),
                index=df.columns.tolist().index(st.session_state.get("date_column", df.columns[0])),
            )

            target_col = st.selectbox(
                "Target Column",
                options=df.select_dtypes(include=["number"]).columns.tolist(),
                index=0,
            )

        with col2:
            description = st.text_area(
                "Description (optional)",
                value="",
                height=100,
            )

            revenue_scale = st.number_input(
                "Revenue/Spend Scale Factor",
                value=1000.0,
                min_value=1.0,
                help="Divide values by this factor for numerical stability"
            )

            dayfirst = st.checkbox("Dates are day-first", value=True)

        st.session_state.config_state.update({
            "name": model_name,
            "description": description,
            "date_col": date_col,
            "target_col": target_col,
            "revenue_scale": revenue_scale,
            "dayfirst": dayfirst,
        })

    # =========================================================================
    # Tab 2: Channel Configuration
    # =========================================================================
    with tab2:
        st.subheader("Media Channels")

        st.markdown("""
        Configure your media channels. Each channel represents a marketing spend variable.
        Set ROI priors based on your expectations or historical performance.

        **Note:** Channels are auto-detected by the `PaidMedia_` prefix in column names.
        """)

        # Auto-detect channels with PaidMedia_ prefix
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        potential_channels = [
            col for col in numeric_cols
            if col.startswith("PaidMedia_")
        ]

        # Initialize session state for channels if needed
        if "channel_multiselect" not in st.session_state:
            st.session_state.channel_multiselect = potential_channels if potential_channels else []

        # Select All / Clear buttons
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 4])
        with col_btn1:
            if st.button("Select All Channels", key="select_all_channels"):
                st.session_state.channel_multiselect = potential_channels if potential_channels else numeric_cols
                st.rerun()
        with col_btn2:
            if st.button("Clear Channels", key="clear_channels"):
                st.session_state.channel_multiselect = []
                st.rerun()

        # Channel selection multiselect
        selected_channels = st.multiselect(
            "Select channel columns",
            options=numeric_cols,
            key="channel_multiselect",
            help="Select columns that represent media spend (auto-detects PaidMedia_ prefix)"
        )

        if selected_channels:
            st.markdown("---")

            # Option to upload priors CSV
            st.subheader("Channel Priors")

            col_upload, col_template = st.columns(2)

            with col_upload:
                priors_file = st.file_uploader(
                    "Upload Priors CSV (optional)",
                    type=["csv"],
                    help="Upload a CSV with columns: channel, roi_low, roi_mid, roi_high, adstock_type"
                )

            with col_template:
                # Generate template CSV
                template_data = []
                for ch in selected_channels:
                    template_data.append({
                        "channel": ch,
                        "display_name": ch.replace("PaidMedia_", "").replace("_spend", "").replace("_", " ").title(),
                        "roi_low": 0.5,
                        "roi_mid": 2.0,
                        "roi_high": 5.0,
                        "adstock_type": "medium"
                    })
                template_df = pd.DataFrame(template_data)

                st.download_button(
                    "Download Priors Template",
                    data=template_df.to_csv(index=False),
                    file_name="channel_priors_template.csv",
                    mime="text/csv",
                    help="Download a template CSV with all selected channels"
                )

            # Load priors from uploaded file or use defaults
            if priors_file is not None:
                try:
                    uploaded_priors = pd.read_csv(priors_file)
                    st.success(f"Loaded priors for {len(uploaded_priors)} channels")

                    # Merge with selected channels
                    priors_dict = {}
                    for _, row in uploaded_priors.iterrows():
                        priors_dict[row["channel"]] = {
                            "display_name": row.get("display_name", row["channel"]),
                            "categories": {},  # Categories are now set via the UI category columns
                            "roi_low": row.get("roi_low", 0.5),
                            "roi_mid": row.get("roi_mid", 2.0),
                            "roi_high": row.get("roi_high", 5.0),
                            "adstock_type": row.get("adstock_type", "medium"),
                        }
                except Exception as e:
                    st.error(f"Error loading priors CSV: {e}")
                    priors_dict = {}
            else:
                priors_dict = {}

            # Build channel config table
            st.markdown("---")

            # Category columns manager
            render_category_columns_manager()

            st.markdown("---")
            st.subheader("Channel Settings Table")
            st.markdown("Edit the table below to configure channel priors:")

            # Get category columns from session state
            category_cols = st.session_state.get("category_columns", [])

            # Prepare data for editable table
            channel_table_data = []
            for ch in selected_channels:
                row_data = {
                    "Channel": ch,
                    "Display Name": ch.replace("PaidMedia_", "").replace("_spend", "").replace("_", " ").title(),
                    "ROI Low": 0.5,
                    "ROI Mid": 2.0,
                    "ROI High": 5.0,
                    "Adstock": "medium",
                    "Curve Shape": "Default",  # Per-channel curve sharpness override
                    "Total Spend": df[ch].sum(),
                }

                # Override with priors if available
                if ch in priors_dict:
                    prior = priors_dict[ch]
                    row_data["Display Name"] = prior.get("display_name", row_data["Display Name"])
                    row_data["ROI Low"] = prior.get("roi_low", 0.5)
                    row_data["ROI Mid"] = prior.get("roi_mid", 2.0)
                    row_data["ROI High"] = prior.get("roi_high", 5.0)
                    row_data["Adstock"] = prior.get("adstock_type", "medium")
                    # Get curve shape override (convert from stored value to label)
                    curve_override = prior.get("curve_sharpness_override")
                    if curve_override:
                        row_data["Curve Shape"] = curve_override.title()
                    else:
                        row_data["Curve Shape"] = "Default"

                    # Add category values from priors (categories dict)
                    prior_categories = prior.get("categories", {})
                    for cat_col in category_cols:
                        row_data[cat_col["name"]] = prior_categories.get(cat_col["name"], "")
                else:
                    # Add empty category values
                    for cat_col in category_cols:
                        row_data[cat_col["name"]] = ""

                channel_table_data.append(row_data)

            channel_df = pd.DataFrame(channel_table_data)

            # Build column config dynamically
            column_config = {
                "Channel": st.column_config.TextColumn("Channel", disabled=True),
                "Display Name": st.column_config.TextColumn("Display Name"),
                "ROI Low": st.column_config.NumberColumn("ROI Low", min_value=0.0, format="%.2f"),
                "ROI Mid": st.column_config.NumberColumn("ROI Mid", min_value=0.0, format="%.2f"),
                "ROI High": st.column_config.NumberColumn("ROI High", min_value=0.0, format="%.2f"),
                "Adstock": st.column_config.SelectboxColumn(
                    "Adstock",
                    options=["short", "medium", "long"],
                ),
                "Curve Shape": st.column_config.SelectboxColumn(
                    "Curve Shape",
                    options=["Default", "Gradual", "Balanced", "Sharp"],
                    help="Override curve sharpness for this channel (Default uses global setting)"
                ),
                "Total Spend": st.column_config.NumberColumn("Total Spend", disabled=True, format="%.2f"),
            }

            # Add dynamic category columns
            for cat_col in category_cols:
                column_config[cat_col["name"]] = st.column_config.SelectboxColumn(
                    cat_col["name"],
                    options=cat_col.get("options", ["Other"]),
                    help=f"Select {cat_col['name']} for grouping in results"
                )

            # Editable data table
            edited_channels = st.data_editor(
                channel_df,
                column_config=column_config,
                hide_index=True,
                use_container_width=True,
                num_rows="fixed",
            )

            # Convert edited table to channels config
            channels_config = []
            for _, row in edited_channels.iterrows():
                # Build categories dict from dynamic columns
                categories = {}
                for cat_col in category_cols:
                    if cat_col["name"] in row and row[cat_col["name"]]:
                        categories[cat_col["name"]] = row[cat_col["name"]]

                # Handle curve shape override (convert label to stored value)
                curve_shape = row.get("Curve Shape", "Default")
                curve_override = None if curve_shape == "Default" else curve_shape.lower()

                channels_config.append({
                    "name": row["Channel"],
                    "display_name": row["Display Name"],
                    "categories": categories,
                    "adstock_type": row["Adstock"],
                    "roi_prior_low": row["ROI Low"],
                    "roi_prior_mid": row["ROI Mid"],
                    "roi_prior_high": row["ROI High"],
                    "curve_sharpness_override": curve_override,
                })

            st.session_state.config_state["channels"] = channels_config

            # Show summary stats
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Channels Selected", len(selected_channels))
            with col2:
                total_spend = edited_channels["Total Spend"].sum()
                st.metric("Total Media Spend", f"{total_spend:,.0f}")
            with col3:
                avg_roi = edited_channels["ROI Mid"].mean()
                st.metric("Avg Expected ROI", f"{avg_roi:.2f}")

    # =========================================================================
    # Tab 3: Control Variables
    # =========================================================================
    with tab3:
        st.subheader("Control Variables")

        st.markdown("""
        Add control variables that might influence your target but aren't media channels.
        Examples: promotions, holidays, seasonality events, economic factors.

        **Sign Constraints:**
        - **positive**: Variable has a positive effect on target (e.g., promotions increase sales)
        - **negative**: Variable has a negative effect on target (e.g., competitor activity decreases sales)
        - **unconstrained**: Effect direction is unknown
        """)

        # Available columns (excluding channels and target)
        excluded_cols = selected_channels + [
            st.session_state.config_state.get("date_col", ""),
            st.session_state.config_state.get("target_col", ""),
        ]
        available_controls = [c for c in df.columns if c not in excluded_cols]

        # Initialize session state for controls if needed
        if "control_multiselect" not in st.session_state:
            st.session_state.control_multiselect = []

        # Select All / Clear buttons
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 4])
        with col_btn1:
            if st.button("Select All Controls", key="select_all_controls"):
                st.session_state.control_multiselect = available_controls
                st.rerun()
        with col_btn2:
            if st.button("Clear Controls", key="clear_controls"):
                st.session_state.control_multiselect = []
                st.rerun()

        # Control selection multiselect
        selected_controls = st.multiselect(
            "Select control variables",
            options=available_controls,
            key="control_multiselect",
            help="Select columns to use as control variables"
        )

        if selected_controls:
            st.markdown("---")

            # Option to upload controls CSV
            st.subheader("Control Settings")

            col_upload, col_template = st.columns(2)

            with col_upload:
                controls_file = st.file_uploader(
                    "Upload Controls CSV (optional)",
                    type=["csv"],
                    key="controls_csv_uploader",
                    help="Upload a CSV with columns: control, sign_constraint, is_dummy, scale"
                )

            with col_template:
                # Generate template CSV
                template_data = []
                for ctrl in selected_controls:
                    is_dummy = df[ctrl].isin([0, 1]).all() if ctrl in df.columns else False
                    template_data.append({
                        "control": ctrl,
                        "sign_constraint": "positive",
                        "is_dummy": is_dummy,
                        "scale": not is_dummy,
                    })
                template_df = pd.DataFrame(template_data)

                st.download_button(
                    "Download Controls Template",
                    data=template_df.to_csv(index=False),
                    file_name="control_settings_template.csv",
                    mime="text/csv",
                    help="Download a template CSV with all selected controls"
                )

            # Load controls from uploaded file or use defaults
            if controls_file is not None:
                try:
                    uploaded_controls = pd.read_csv(controls_file)
                    st.success(f"Loaded settings for {len(uploaded_controls)} controls")

                    # Merge with selected controls
                    controls_dict = {}
                    for _, row in uploaded_controls.iterrows():
                        controls_dict[row["control"]] = {
                            "categories": row.get("categories", {}),
                            "sign_constraint": row.get("sign_constraint", "positive"),
                            "is_dummy": bool(row.get("is_dummy", False)),
                            "scale": bool(row.get("scale", True)),
                        }
                except Exception as e:
                    st.error(f"Error loading controls CSV: {e}")
                    controls_dict = {}
            else:
                controls_dict = {}

            # Build control config table
            st.markdown("---")
            st.subheader("Control Settings Table")
            st.markdown("Edit the table below to configure control variables:")

            # Get category columns from session state (same as channels)
            category_cols = st.session_state.get("category_columns", [])

            # Prepare data for editable table
            control_table_data = []
            for ctrl in selected_controls:
                is_dummy_default = df[ctrl].isin([0, 1]).all() if ctrl in df.columns else False

                row_data = {
                    "Control": ctrl,
                    "Sign Constraint": "positive",
                    "Is Dummy": is_dummy_default,
                    "Scale": not is_dummy_default,
                    "Mean": df[ctrl].mean() if ctrl in df.columns else 0,
                    "Std": df[ctrl].std() if ctrl in df.columns else 0,
                }

                # Override with settings if available
                if ctrl in controls_dict:
                    settings = controls_dict[ctrl]
                    row_data["Sign Constraint"] = settings.get("sign_constraint", "positive")
                    row_data["Is Dummy"] = settings.get("is_dummy", is_dummy_default)
                    row_data["Scale"] = settings.get("scale", not is_dummy_default)

                    # Add category values from settings (categories dict)
                    settings_categories = settings.get("categories", {})
                    for cat_col in category_cols:
                        row_data[cat_col["name"]] = settings_categories.get(cat_col["name"], "")
                else:
                    # Add empty category values
                    for cat_col in category_cols:
                        row_data[cat_col["name"]] = ""

                control_table_data.append(row_data)

            control_df = pd.DataFrame(control_table_data)

            # Build column config dynamically
            control_column_config = {
                "Control": st.column_config.TextColumn("Control", disabled=True),
                "Sign Constraint": st.column_config.SelectboxColumn(
                    "Sign Constraint",
                    options=["positive", "negative", "unconstrained"],
                    help="Expected direction of effect"
                ),
                "Is Dummy": st.column_config.CheckboxColumn(
                    "Is Dummy",
                    help="Check if this is a binary 0/1 variable"
                ),
                "Scale": st.column_config.CheckboxColumn(
                    "Scale",
                    help="Check to scale this variable (usually False for dummies)"
                ),
                "Mean": st.column_config.NumberColumn("Mean", disabled=True, format="%.2f"),
                "Std": st.column_config.NumberColumn("Std", disabled=True, format="%.2f"),
            }

            # Add dynamic category columns
            for cat_col in category_cols:
                control_column_config[cat_col["name"]] = st.column_config.SelectboxColumn(
                    cat_col["name"],
                    options=cat_col.get("options", ["Other"]),
                    help=f"Select {cat_col['name']} for grouping in results"
                )

            # Editable data table
            edited_controls = st.data_editor(
                control_df,
                column_config=control_column_config,
                hide_index=True,
                use_container_width=True,
                num_rows="fixed",
            )

            # Convert edited table to controls config
            controls_config = []
            for _, row in edited_controls.iterrows():
                # Build categories dict from dynamic columns
                categories = {}
                for cat_col in category_cols:
                    if cat_col["name"] in row and row[cat_col["name"]]:
                        categories[cat_col["name"]] = row[cat_col["name"]]

                controls_config.append({
                    "name": row["Control"],
                    "categories": categories,
                    "sign_constraint": row["Sign Constraint"],
                    "is_dummy": row["Is Dummy"],
                    "scale": row["Scale"],
                })

            st.session_state.config_state["controls"] = controls_config

            # Show summary stats
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Controls Selected", len(selected_controls))
            with col2:
                n_positive = sum(1 for _, r in edited_controls.iterrows() if r["Sign Constraint"] == "positive")
                st.metric("Positive Constraints", n_positive)
            with col3:
                n_dummy = sum(1 for _, r in edited_controls.iterrows() if r["Is Dummy"])
                st.metric("Dummy Variables", n_dummy)

    # =========================================================================
    # Tab 4: Transform Settings
    # =========================================================================
    with tab4:
        st.subheader("Transform Settings")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Adstock Settings**")
            l_max = st.slider(
                "Maximum Lag (weeks)",
                min_value=1,
                max_value=52,
                value=8,
                help="Maximum carryover period"
            )

            short_decay = st.slider(
                "Short Decay Rate",
                min_value=0.0,
                max_value=1.0,
                value=0.15,
            )
            medium_decay = st.slider(
                "Medium Decay Rate",
                min_value=0.0,
                max_value=1.0,
                value=0.40,
            )
            long_decay = st.slider(
                "Long Decay Rate",
                min_value=0.0,
                max_value=1.0,
                value=0.70,
            )

        with col2:
            st.markdown("**Saturation Curve Shape**")
            st.caption("Controls how quickly channels reach diminishing returns")

            curve_sharpness = st.slider(
                "Curve Sharpness",
                min_value=0,
                max_value=100,
                value=50,
                help="0 = Very gradual (slow saturation), 100 = Very sharp (quick saturation)",
                key="curve_sharpness_slider"
            )

            # Visual labels for the slider
            label_col1, label_col2, label_col3 = st.columns(3)
            with label_col1:
                st.caption("← Gradual")
            with label_col2:
                st.caption("Balanced")
            with label_col3:
                st.caption("Sharp →")

            # Show the effective percentile
            from mmm_platform.config.schema import sharpness_to_percentile
            effective_percentile = sharpness_to_percentile(curve_sharpness)
            st.info(f"Half-saturation at {effective_percentile}th percentile of spend")

            st.markdown("**Seasonality**")
            yearly_seasonality = st.slider(
                "Fourier Terms",
                min_value=0,
                max_value=10,
                value=2,
                help="Number of Fourier terms for yearly seasonality"
            )

        st.session_state.config_state.update({
            "l_max": l_max,
            "short_decay": short_decay,
            "medium_decay": medium_decay,
            "long_decay": long_decay,
            "curve_sharpness": curve_sharpness,
            "yearly_seasonality": yearly_seasonality,
        })

    # =========================================================================
    # Tab 5: Sampling Settings
    # =========================================================================
    with tab5:
        st.subheader("MCMC Sampling Settings")

        col1, col2 = st.columns(2)

        with col1:
            draws = st.number_input(
                "Posterior Draws",
                min_value=100,
                max_value=10000,
                value=1500,
                step=100,
            )

            tune = st.number_input(
                "Tuning Steps",
                min_value=100,
                max_value=10000,
                value=1500,
                step=100,
            )

        with col2:
            chains = st.number_input(
                "Number of Chains",
                min_value=1,
                max_value=8,
                value=4,
            )

            target_accept = st.slider(
                "Target Acceptance Rate",
                min_value=0.5,
                max_value=0.99,
                value=0.9,
            )

        random_seed = st.number_input(
            "Random Seed",
            min_value=0,
            value=42,
            help="For reproducibility"
        )

        st.session_state.config_state.update({
            "draws": draws,
            "tune": tune,
            "chains": chains,
            "target_accept": target_accept,
            "random_seed": random_seed,
        })

    # =========================================================================
    # Build and Save Configuration
    # =========================================================================
    st.markdown("---")
    st.subheader("Build Configuration")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🔨 Build Configuration", type="primary"):
            try:
                config = build_config_from_state()
                st.session_state.current_config = config
                st.success(f"Configuration '{config.name}' built successfully!")

                # Show summary
                st.json({
                    "name": config.name,
                    "channels": len(config.channels),
                    "controls": len(config.controls),
                    "target": config.data.target_column,
                })

            except Exception as e:
                st.error(f"Error building configuration: {e}")

    with col2:
        if st.button("💾 Export to YAML"):
            if st.session_state.get("current_config"):
                config = st.session_state.current_config
                yaml_content = ConfigLoader.get_template()  # Would need to implement proper export
                st.download_button(
                    "Download YAML",
                    data=str(yaml_content),
                    file_name=f"{config.name}.yaml",
                    mime="text/yaml"
                )

    with col3:
        if st.button("📂 Load from YAML"):
            st.info("Upload a YAML configuration file")
            # Would add file uploader here


def build_config_from_state() -> ModelConfig:
    """Build a ModelConfig from session state."""
    state = st.session_state.config_state

    # Build category columns from session state
    category_columns = [
        CategoryColumnConfig(name=col["name"], options=col.get("options", []))
        for col in st.session_state.get("category_columns", [])
    ]

    # Build channel configs with categories dict
    channels = [
        ChannelConfig(
            name=ch["name"],
            display_name=ch.get("display_name"),
            categories=ch.get("categories", {}),
            adstock_type=AdstockType(ch.get("adstock_type", "medium")),
            roi_prior_low=ch.get("roi_prior_low", 0.5),
            roi_prior_mid=ch.get("roi_prior_mid", 2.0),
            roi_prior_high=ch.get("roi_prior_high", 5.0),
            curve_sharpness_override=ch.get("curve_sharpness_override"),
        )
        for ch in state.get("channels", [])
    ]

    # Build control configs with categories dict
    controls = [
        ControlConfig(
            name=ctrl["name"],
            categories=ctrl.get("categories", {}),
            sign_constraint=SignConstraint(ctrl.get("sign_constraint", "unconstrained")),
            is_dummy=ctrl.get("is_dummy", False),
            scale=ctrl.get("scale", False),
        )
        for ctrl in state.get("controls", [])
    ]

    # Build full config
    config = ModelConfig(
        name=state.get("name", "my_mmm_model"),
        description=state.get("description"),
        data=DataConfig(
            date_column=state.get("date_col", "time"),
            target_column=state.get("target_col", "revenue"),
            dayfirst=state.get("dayfirst", True),
            revenue_scale=state.get("revenue_scale", 1000.0),
            spend_scale=state.get("revenue_scale", 1000.0),
        ),
        channels=channels,
        controls=controls,
        category_columns=category_columns,
        adstock=AdstockConfig(
            l_max=state.get("l_max", 8),
            short_decay=state.get("short_decay", 0.15),
            medium_decay=state.get("medium_decay", 0.40),
            long_decay=state.get("long_decay", 0.70),
        ),
        saturation=SaturationConfig(
            curve_sharpness=state.get("curve_sharpness", 50),
        ),
        seasonality=SeasonalityConfig(
            yearly_seasonality=state.get("yearly_seasonality", 2),
        ),
        sampling=SamplingConfig(
            draws=state.get("draws", 1500),
            tune=state.get("tune", 1500),
            chains=state.get("chains", 4),
            target_accept=state.get("target_accept", 0.9),
            random_seed=state.get("random_seed", 42),
        ),
        control_prior=PriorConfig(
            distribution="HalfNormal",
            sigma=1.0,
        ),
    )

    return config
