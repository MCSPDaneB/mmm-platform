"""
Model configuration page for MMM Platform.
"""

import streamlit as st
import pandas as pd
from typing import Optional, List

from mmm_platform.config.schema import (
    ModelConfig, ChannelConfig, ControlConfig, DataConfig,
    SamplingConfig, AdstockConfig, SaturationConfig, SeasonalityConfig,
    AdstockType, SignConstraint, PriorConfig, CategoryColumnConfig,
    DummyVariableConfig
)
from mmm_platform.config.loader import ConfigLoader
from mmm_platform.model.persistence import list_clients, get_client_configs_dir
from mmm_platform.core.trend_detection import detect_trend


# Maximum number of custom category columns allowed
MAX_CATEGORY_COLUMNS = 5


def render_category_columns_manager(key_prefix: str = ""):
    """
    Render the category columns management section.
    This allows users to add/remove custom category columns that appear in both
    channel and control configuration tables.

    Parameters
    ----------
    key_prefix : str
        Prefix for widget keys to avoid duplicates when rendered multiple times
    """
    # Initialize session state for category columns if not exists
    if "category_columns" not in st.session_state:
        st.session_state.category_columns = []

    st.subheader("Category Columns")
    st.caption(f"Define custom columns for grouping results (max {MAX_CATEGORY_COLUMNS})")

    # Display existing columns with edit capability
    if st.session_state.category_columns:
        for i, cat_col in enumerate(st.session_state.category_columns):
            with st.expander(f"**{cat_col['name']}** ({len(cat_col.get('options', []))} options)", expanded=False):
                # Show current options as editable text
                current_options = ", ".join(cat_col.get("options", []))
                new_options = st.text_input(
                    "Options (comma-separated)",
                    value=current_options,
                    key=f"{key_prefix}edit_cat_options_{i}",
                    help="Edit the options for this category"
                )

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Update", key=f"{key_prefix}update_cat_{i}"):
                        options = [o.strip() for o in new_options.split(",") if o.strip()]
                        if options:
                            st.session_state.category_columns[i]["options"] = options
                            st.success("Updated!")
                            st.rerun()
                        else:
                            st.warning("At least one option is required")
                with col2:
                    if st.button("Remove Column", key=f"{key_prefix}remove_cat_col_{i}", type="secondary"):
                        st.session_state.category_columns.pop(i)
                        st.rerun()

    # Add new column section
    if len(st.session_state.category_columns) < MAX_CATEGORY_COLUMNS:
        with st.expander("➕ Add Category Column", expanded=False):
            new_col_name = st.text_input(
                "Column Name",
                placeholder="e.g., Funnel Stage",
                key=f"{key_prefix}new_cat_col_name"
            )
            new_col_options = st.text_input(
                "Options (comma-separated)",
                placeholder="e.g., Awareness, Consideration, Conversion",
                key=f"{key_prefix}new_cat_col_options"
            )

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Add Column", key=f"{key_prefix}add_cat_col_btn"):
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

    # Check if priors need review due to KPI change
    if st.session_state.get("priors_need_review"):
        old_target = st.session_state.get("priors_set_for_target", "unknown")
        new_target = st.session_state.get("target_column", "unknown")

        st.warning(
            f"⚠️ **Target KPI changed from '{old_target}' to '{new_target}'**. "
            "ROI priors may need to be updated to reflect the new KPI scale."
        )

        if st.button("✅ I've reviewed the priors", key="dismiss_prior_warning"):
            st.session_state.priors_need_review = False
            st.rerun()

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

        # Get saved values from config_state
        saved_data = st.session_state.config_state

        # Get config version for widget keys (forces re-init when config loaded)
        config_version = st.session_state.get("config_version", 0)

        col1, col2 = st.columns(2)

        with col1:
            model_name = st.text_input(
                "Model Name",
                value=saved_data.get("name", "my_mmm_model"),
                help="A unique name for this model configuration"
            )

            # Client selection
            clients = list_clients()
            client_options = ["(Select or create new)"] + clients
            saved_client = saved_data.get("client", "(Select or create new)")
            if saved_client and saved_client in clients:
                client_index = client_options.index(saved_client)
            else:
                client_index = 0

            client_col1, client_col2 = st.columns([3, 1])
            with client_col1:
                selected_client = st.selectbox(
                    "Client",
                    options=client_options,
                    index=client_index,
                    help="Organize configs/models by client"
                )
            with client_col2:
                new_client = st.text_input("New client", key="new_client_name", label_visibility="hidden", placeholder="New client")
                if st.button("Add", key="add_client_btn") and new_client:
                    # Create the client folder
                    get_client_configs_dir(new_client)
                    st.session_state.config_state["client"] = new_client
                    st.success(f"Client '{new_client}' created!")
                    st.rerun()

            # Determine final client value
            if selected_client != "(Select or create new)":
                client_value = selected_client
            elif new_client:
                client_value = new_client
            else:
                client_value = None

            # Date column - try to find saved value in columns
            date_options = df.columns.tolist()
            saved_date = st.session_state.get("date_column") or saved_data.get("date_col")
            date_index = date_options.index(saved_date) if saved_date and saved_date in date_options else 0
            date_col = st.selectbox(
                "Date Column",
                options=date_options,
                index=date_index,
                key=f"date_col_select_{config_version}",
            )

            # Target column - try to find saved value in numeric columns
            target_options = df.select_dtypes(include=["number"]).columns.tolist()
            saved_target = st.session_state.get("target_column") or saved_data.get("target_col")
            target_index = target_options.index(saved_target) if saved_target and saved_target in target_options else 0
            target_col = st.selectbox(
                "Target Column",
                options=target_options,
                index=target_index,
                key=f"target_col_select_{config_version}",
            )

        with col2:
            description = st.text_area(
                "Description (optional)",
                value=saved_data.get("description", ""),
                height=100,
            )

            revenue_scale = st.number_input(
                "Revenue/Spend Scale Factor",
                value=saved_data.get("revenue_scale", 1000.0),
                min_value=1.0,
                help="Divide values by this factor for numerical stability"
            )

            dayfirst = st.checkbox(
                "Dates are day-first",
                value=st.session_state.get("dayfirst", saved_data.get("dayfirst", True))
            )

        # Trend Detection section
        st.markdown("---")
        st.markdown("**Time Trend Analysis**")

        # Run trend detection on selected KPI
        if target_col and target_col in df.columns:
            trend_result = detect_trend(df[target_col])

            # Show mini chart with trend line
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(6, 2.5))
            ax.plot(df[target_col].values, alpha=0.7, linewidth=1, label='KPI')
            ax.plot(trend_result['trend_line'], '--', color='red', linewidth=2, label='Trend')
            ax.set_xlabel('Time Period', fontsize=9)
            ax.set_ylabel(target_col, fontsize=9)
            ax.tick_params(axis='both', labelsize=8)
            ax.legend(fontsize=8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            # Show test result
            trend_col1, trend_col2 = st.columns([2, 1])
            with trend_col1:
                if trend_result['has_trend']:
                    direction_emoji = "📈" if trend_result['direction'] == 'increasing' else "📉"
                    st.success(f"{direction_emoji} Significant {trend_result['direction']} trend detected (p={trend_result['p_value']:.3f}, R²={trend_result['r_squared']:.2f})")
                    trend_default = True
                else:
                    st.info(f"No significant trend detected (p={trend_result['p_value']:.3f})")
                    trend_default = False

            with trend_col2:
                # Use saved value if available, otherwise fall back to trend detection
                saved_include_trend = saved_data.get("include_trend")
                checkbox_default = saved_include_trend if saved_include_trend is not None else trend_default

                include_trend = st.checkbox(
                    "Include Trend",
                    value=checkbox_default,
                    key=f"include_trend_check_{config_version}",
                    help="Add a linear time trend (t=1,2,3,...) as a control variable"
                )
        else:
            include_trend = st.checkbox(
                "Include Time Trend",
                value=saved_data.get("include_trend", True),
                key=f"include_trend_check_fallback_{config_version}",
                help="Add a linear time trend (t=1,2,3,...) as a control variable"
            )

        # Model Date Range section
        st.markdown("---")
        st.markdown("**Model Date Range**")
        st.caption("Optionally limit the date range used for modeling")

        # Parse date column to get min/max dates
        date_col_for_range = st.session_state.config_state.get("date_col") or date_col
        try:
            df_dates = pd.to_datetime(df[date_col_for_range], dayfirst=dayfirst)
            data_min_date = df_dates.min().date()
            data_max_date = df_dates.max().date()
        except Exception:
            data_min_date = None
            data_max_date = None

        date_range_col1, date_range_col2 = st.columns(2)

        with date_range_col1:
            # Get saved start date or use data min
            saved_start = saved_data.get("model_start_date")
            if saved_start:
                try:
                    default_start = pd.to_datetime(saved_start).date()
                except Exception:
                    default_start = data_min_date
            else:
                default_start = data_min_date

            model_start_date = st.date_input(
                "Model Start Date",
                value=default_start,
                min_value=data_min_date,
                max_value=data_max_date,
                help="Start date for modeling (leave at min to use all data)"
            )

        with date_range_col2:
            # Get saved end date or use data max
            saved_end = saved_data.get("model_end_date")
            if saved_end:
                try:
                    default_end = pd.to_datetime(saved_end).date()
                except Exception:
                    default_end = data_max_date
            else:
                default_end = data_max_date

            model_end_date = st.date_input(
                "Model End Date",
                value=default_end,
                min_value=data_min_date,
                max_value=data_max_date,
                help="End date for modeling (leave at max to use all data)"
            )

        # Show row count for selected range
        if data_min_date and data_max_date:
            try:
                mask = (df_dates >= pd.Timestamp(model_start_date)) & (df_dates <= pd.Timestamp(model_end_date))
                rows_in_range = mask.sum()
                total_rows = len(df)
                st.info(f"📊 Selected range: **{rows_in_range}** of {total_rows} rows ({model_start_date} to {model_end_date})")
            except Exception:
                pass

        # Convert to string for storage (None if using full range)
        model_start_str = model_start_date.strftime("%Y-%m-%d") if model_start_date and model_start_date != data_min_date else None
        model_end_str = model_end_date.strftime("%Y-%m-%d") if model_end_date and model_end_date != data_max_date else None

        st.session_state.config_state.update({
            "name": model_name,
            "client": client_value,
            "description": description,
            "date_col": date_col,
            "target_col": target_col,
            "revenue_scale": revenue_scale,
            "dayfirst": dayfirst,
            "include_trend": include_trend,
            "model_start_date": model_start_str,
            "model_end_date": model_end_str,
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
        # If channel_multiselect exists (e.g. from loaded config), filter to valid columns
        if "channel_multiselect" not in st.session_state:
            # Check if we have saved channels in config_state to restore
            saved_channels = st.session_state.get("config_state", {}).get("channels", [])
            if saved_channels:
                # Use saved channel names that exist in numeric_cols
                saved_channel_names = [c["name"] for c in saved_channels]
                st.session_state.channel_multiselect = [c for c in saved_channel_names if c in numeric_cols]
            else:
                st.session_state.channel_multiselect = potential_channels if potential_channels else []
        else:
            # Filter existing selection to only valid columns
            st.session_state.channel_multiselect = [c for c in st.session_state.channel_multiselect if c in numeric_cols]

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
                # Generate template CSV - include category columns with current values
                category_cols_for_template = st.session_state.get("category_columns", [])

                # Get current channel config from session state if available
                current_channels_config = st.session_state.get("config_state", {}).get("channels", [])
                channels_config_dict = {ch["name"]: ch for ch in current_channels_config}

                template_data = []
                for ch in selected_channels:
                    # Get existing config for this channel if available
                    existing = channels_config_dict.get(ch, {})

                    row = {
                        "channel": ch,
                        "display_name": existing.get("display_name", ch.replace("PaidMedia_", "").replace("_spend", "").replace("_", " ").title()),
                        "roi_low": existing.get("roi_prior_low", 0.5),
                        "roi_mid": existing.get("roi_prior_mid", 2.0),
                        "roi_high": existing.get("roi_prior_high", 5.0),
                        "adstock_type": existing.get("adstock_type", "medium")
                    }
                    # Add category columns with current values
                    existing_categories = existing.get("categories", {})
                    for cat_col in category_cols_for_template:
                        row[cat_col["name"]] = existing_categories.get(cat_col["name"], "")
                    template_data.append(row)
                template_df = pd.DataFrame(template_data)

                st.download_button(
                    "Download Priors Template",
                    data=template_df.to_csv(index=False),
                    file_name="channel_priors_template.csv",
                    mime="text/csv",
                    help="Download a template CSV with all selected channels"
                )

            # Load priors from uploaded file, session state, or use defaults
            if priors_file is not None:
                try:
                    uploaded_priors = pd.read_csv(priors_file)
                    st.success(f"Loaded priors for {len(uploaded_priors)} channels")

                    # Get category columns for reading from CSV
                    category_cols_for_upload = st.session_state.get("category_columns", [])

                    # Merge with selected channels
                    priors_dict = {}
                    for _, row in uploaded_priors.iterrows():
                        # Read category values from CSV
                        categories = {}
                        for cat_col in category_cols_for_upload:
                            if cat_col["name"] in row and pd.notna(row[cat_col["name"]]) and row[cat_col["name"]]:
                                cat_value = str(row[cat_col["name"]]).strip()
                                if cat_value:
                                    categories[cat_col["name"]] = cat_value
                                    # Auto-add new values to category options
                                    if cat_value not in cat_col["options"]:
                                        cat_col["options"].append(cat_value)

                        priors_dict[row["channel"]] = {
                            "display_name": row.get("display_name", row["channel"]),
                            "categories": categories,
                            "roi_low": row.get("roi_low", 0.5),
                            "roi_mid": row.get("roi_mid", 2.0),
                            "roi_high": row.get("roi_high", 5.0),
                            "adstock_type": row.get("adstock_type", "medium"),
                        }
                except Exception as e:
                    st.error(f"Error loading priors CSV: {e}")
                    priors_dict = {}
            else:
                # Initialize priors_dict from saved config_state if available
                priors_dict = {}
                saved_channels = st.session_state.get("config_state", {}).get("channels", [])
                for ch_config in saved_channels:
                    priors_dict[ch_config["name"]] = {
                        "display_name": ch_config.get("display_name", ch_config["name"]),
                        "categories": ch_config.get("categories", {}),
                        "roi_low": ch_config.get("roi_prior_low", 0.5),
                        "roi_mid": ch_config.get("roi_prior_mid", 2.0),
                        "roi_high": ch_config.get("roi_prior_high", 5.0),
                        "adstock_type": ch_config.get("adstock_type", "medium"),
                        "curve_sharpness_override": ch_config.get("curve_sharpness_override"),
                    }

            # Build channel config table
            st.markdown("---")

            # Category columns manager
            render_category_columns_manager(key_prefix="channels_")

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
                width="stretch",
                num_rows="fixed",
                key="channels_data_editor",
            )

            # Save button to apply changes
            if st.button("💾 Save Channel Settings", key="save_channels_btn"):
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
                st.success("Channel settings saved!")

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
        # If control_multiselect exists (e.g. from loaded config), filter to valid columns
        if "control_multiselect" not in st.session_state:
            # Check if we have saved controls in config_state to restore
            saved_controls = st.session_state.get("config_state", {}).get("controls", [])
            if saved_controls:
                # Use saved control names that exist in available_controls
                saved_control_names = [c["name"] for c in saved_controls]
                st.session_state.control_multiselect = [c for c in saved_control_names if c in available_controls]
            else:
                st.session_state.control_multiselect = []
        else:
            # Filter existing selection to only valid columns
            st.session_state.control_multiselect = [c for c in st.session_state.control_multiselect if c in available_controls]

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

            # Category columns manager (same as channels tab)
            render_category_columns_manager(key_prefix="controls_")

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
                # Generate template CSV - include category columns with current values
                category_cols_for_ctrl_template = st.session_state.get("category_columns", [])

                # Get current control config from session state if available
                current_controls_config = st.session_state.get("config_state", {}).get("controls", [])
                controls_config_dict = {ctrl["name"]: ctrl for ctrl in current_controls_config}

                template_data = []
                for ctrl in selected_controls:
                    is_dummy_default = df[ctrl].isin([0, 1]).all() if ctrl in df.columns else False

                    # Get existing config for this control if available
                    existing = controls_config_dict.get(ctrl, {})

                    row = {
                        "control": ctrl,
                        "sign_constraint": existing.get("sign_constraint", "positive"),
                        "is_dummy": existing.get("is_dummy", is_dummy_default),
                        "scale": existing.get("scale", not is_dummy_default),
                    }
                    # Add category columns with current values
                    existing_categories = existing.get("categories", {})
                    for cat_col in category_cols_for_ctrl_template:
                        row[cat_col["name"]] = existing_categories.get(cat_col["name"], "")
                    template_data.append(row)
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

                    # Get category columns for reading from CSV
                    category_cols_for_ctrl_upload = st.session_state.get("category_columns", [])

                    # Merge with selected controls
                    controls_dict = {}
                    for _, row in uploaded_controls.iterrows():
                        # Read category values from CSV
                        categories = {}
                        for cat_col in category_cols_for_ctrl_upload:
                            if cat_col["name"] in row and pd.notna(row[cat_col["name"]]) and row[cat_col["name"]]:
                                cat_value = str(row[cat_col["name"]]).strip()
                                if cat_value:
                                    categories[cat_col["name"]] = cat_value
                                    # Auto-add new values to category options
                                    if cat_value not in cat_col["options"]:
                                        cat_col["options"].append(cat_value)

                        controls_dict[row["control"]] = {
                            "categories": categories,
                            "sign_constraint": row.get("sign_constraint", "positive"),
                            "is_dummy": bool(row.get("is_dummy", False)),
                            "scale": bool(row.get("scale", True)),
                        }
                except Exception as e:
                    st.error(f"Error loading controls CSV: {e}")
                    controls_dict = {}
            else:
                # Initialize controls_dict from saved config_state if available
                controls_dict = {}
                saved_controls = st.session_state.get("config_state", {}).get("controls", [])
                for ctrl_config in saved_controls:
                    controls_dict[ctrl_config["name"]] = {
                        "categories": ctrl_config.get("categories", {}),
                        "sign_constraint": ctrl_config.get("sign_constraint", "positive"),
                        "is_dummy": ctrl_config.get("is_dummy", False),
                        "scale": ctrl_config.get("scale", True),
                    }

            # Build control config table
            st.markdown("---")
            st.subheader("Control Settings Table")
            st.markdown("Edit the table below to configure control variables:")

            # Get category columns from session state (same as channels)
            category_cols = st.session_state.get("category_columns", [])

            # Get configured dummy variables from config state
            config_dummies = st.session_state.get("config_state", {}).get("dummy_variables", [])
            dummy_names = [d["name"] for d in config_dummies]

            # Prepare data for editable table
            control_table_data = []

            # First add regular controls from data columns
            for ctrl in selected_controls:
                is_dummy_default = df[ctrl].isin([0, 1]).all() if ctrl in df.columns else False
                default_display = ctrl.replace("_", " ").title()

                row_data = {
                    "Control": ctrl,
                    "Display Name": default_display,
                    "Sign Constraint": "positive",
                    "Is Dummy": is_dummy_default,
                    "Scale": not is_dummy_default,
                    "Mean": df[ctrl].mean() if ctrl in df.columns else 0,
                    "Std": df[ctrl].std() if ctrl in df.columns else 0,
                    "Source": "data",  # Track source for later
                }

                # Override with settings if available
                if ctrl in controls_dict:
                    settings = controls_dict[ctrl]
                    row_data["Display Name"] = settings.get("display_name", default_display)
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

            # Then add configured dummy variables (from Results page)
            for dummy in config_dummies:
                default_display = dummy["name"].replace("_", " ").title()
                row_data = {
                    "Control": dummy["name"],
                    "Display Name": dummy.get("display_name", default_display),
                    "Sign Constraint": dummy.get("sign_constraint", "positive"),
                    "Is Dummy": True,
                    "Scale": False,
                    "Mean": 0.0,  # Will be created dynamically
                    "Std": 0.0,
                    "Source": "dummy",  # Track source
                    "start_date": dummy.get("start_date"),  # Keep date info
                    "end_date": dummy.get("end_date"),
                }
                # Add category values for dummies (use saved values or empty)
                dummy_categories = dummy.get("categories", {})
                for cat_col in category_cols:
                    row_data[cat_col["name"]] = dummy_categories.get(cat_col["name"], "")

                control_table_data.append(row_data)

            control_df = pd.DataFrame(control_table_data)

            # Build column config dynamically - hide internal columns
            display_columns = ["Control", "Display Name", "Sign Constraint", "Is Dummy", "Scale", "Mean", "Std"]
            for cat_col in category_cols:
                display_columns.append(cat_col["name"])

            control_column_config = {
                "Control": st.column_config.TextColumn("Control", disabled=True),
                "Display Name": st.column_config.TextColumn(
                    "Display Name",
                    help="Human-readable name for display in results"
                ),
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

            # Only show display columns (hide Source, start_date, end_date)
            display_df = control_df[display_columns] if len(control_df) > 0 else control_df

            # Editable data table
            edited_controls = st.data_editor(
                display_df,
                column_config=control_column_config,
                hide_index=True,
                width="stretch",
                num_rows="fixed",
                key="controls_data_editor",
            )

            # Save button to apply changes (prevents constant refreshing)
            if st.button("💾 Save Control Settings", key="save_controls_btn"):
                # Convert edited table to controls config and dummy variables config
                controls_config = []
                dummy_variables_config = []

                for i, row in edited_controls.iterrows():
                    # Build categories dict from dynamic columns
                    categories = {}
                    for cat_col in category_cols:
                        if cat_col["name"] in row and row[cat_col["name"]]:
                            categories[cat_col["name"]] = row[cat_col["name"]]

                    # Check if this is a dummy variable (from Results page)
                    original_row = control_table_data[i] if i < len(control_table_data) else {}
                    is_from_dummy = original_row.get("Source") == "dummy"

                    if is_from_dummy:
                        # Keep as dummy variable config
                        dummy_variables_config.append({
                            "name": row["Control"],
                            "display_name": row["Display Name"],
                            "start_date": original_row.get("start_date"),
                            "end_date": original_row.get("end_date"),
                            "categories": categories,
                            "sign_constraint": row["Sign Constraint"],
                        })
                    else:
                        # Regular control
                        controls_config.append({
                            "name": row["Control"],
                            "display_name": row["Display Name"],
                            "categories": categories,
                            "sign_constraint": row["Sign Constraint"],
                            "is_dummy": row["Is Dummy"],
                            "scale": row["Scale"],
                        })

                st.session_state.config_state["controls"] = controls_config
                st.session_state.config_state["dummy_variables"] = dummy_variables_config
                st.success("Control settings saved!")

            # Show summary stats
            st.markdown("---")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Data Controls", len(selected_controls))
            with col2:
                st.metric("Dummy Variables", len(config_dummies))
            with col3:
                n_positive = sum(1 for _, r in edited_controls.iterrows() if r["Sign Constraint"] == "positive")
                st.metric("Positive Constraints", n_positive)
            with col4:
                st.metric("Total Controls", len(edited_controls))

            # Allow removing dummy variables
            if config_dummies:
                st.caption("To remove a dummy variable, click the button below:")
                cols = st.columns(min(len(config_dummies), 4))
                for i, dummy in enumerate(config_dummies):
                    with cols[i % 4]:
                        if st.button(f"Remove {dummy['name']}", key=f"remove_dummy_{i}"):
                            st.session_state.config_state["dummy_variables"].pop(i)
                            st.rerun()

        # =====================================================================
        # Pending Dummy Variables (from Results page)
        # =====================================================================
        if st.session_state.get("pending_dummies"):
            st.markdown("---")
            st.subheader("Pending Dummy Variables")
            st.caption("These were created from the Results page residual analysis. Add them to include in the controls table above.")

            pending_dummies = st.session_state.pending_dummies

            # Display pending dummies in a compact table-like format
            for i, dummy in enumerate(pending_dummies):
                col1, col2, col3, col4, col5 = st.columns([3, 2, 2, 2, 1])
                with col1:
                    st.write(f"**{dummy.name}**")
                with col2:
                    st.write(f"{dummy.start_date}")
                with col3:
                    st.write(f"{dummy.end_date}")
                with col4:
                    sign_label = dummy.sign_constraint.value
                    if sign_label == "positive":
                        st.success(sign_label)
                    elif sign_label == "negative":
                        st.error(sign_label)
                    else:
                        st.info(sign_label)
                with col5:
                    if st.button("✕", key=f"remove_pending_dummy_{i}", help="Remove"):
                        st.session_state.pending_dummies.pop(i)
                        st.rerun()

            # Add to model button
            st.markdown("")
            col1, col2 = st.columns([2, 3])
            with col1:
                if st.button("Add to Controls Table", type="primary", key="add_pending_dummies"):
                    # Add pending dummies to config state
                    if "config_state" not in st.session_state:
                        st.session_state.config_state = {}

                    # Initialize dummy_variables in config state if needed
                    if "dummy_variables" not in st.session_state.config_state:
                        st.session_state.config_state["dummy_variables"] = []

                    # Add each pending dummy (avoid duplicates)
                    existing_names = [d["name"] for d in st.session_state.config_state["dummy_variables"]]
                    added_count = 0
                    for dummy in pending_dummies:
                        if dummy.name not in existing_names:
                            dummy_dict = {
                                "name": dummy.name,
                                "start_date": dummy.start_date,
                                "end_date": dummy.end_date,
                                "sign_constraint": dummy.sign_constraint.value
                            }
                            st.session_state.config_state["dummy_variables"].append(dummy_dict)
                            added_count += 1

                    # Clear pending dummies
                    st.session_state.pending_dummies = []

                    # Invalidate current config since dummy variables changed
                    if "current_config" in st.session_state:
                        del st.session_state["current_config"]

                    st.success(f"Added {added_count} dummy variable(s) to controls table! Please rebuild the configuration.")
                    st.rerun()

            with col2:
                if st.button("Clear All Pending", key="clear_pending_dummies"):
                    st.session_state.pending_dummies = []
                    st.rerun()

    # =========================================================================
    # Tab 4: Transform Settings
    # =========================================================================
    with tab4:
        st.subheader("Transform Settings")

        # Get saved values from config_state
        saved_config = st.session_state.config_state

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Adstock Settings**")
            l_max = st.slider(
                "Maximum Lag (weeks)",
                min_value=1,
                max_value=52,
                value=saved_config.get("l_max", 8),
                help="Maximum carryover period"
            )

            short_decay = st.slider(
                "Short Decay Rate",
                min_value=0.0,
                max_value=1.0,
                value=saved_config.get("short_decay", 0.15),
            )
            medium_decay = st.slider(
                "Medium Decay Rate",
                min_value=0.0,
                max_value=1.0,
                value=saved_config.get("medium_decay", 0.40),
            )
            long_decay = st.slider(
                "Long Decay Rate",
                min_value=0.0,
                max_value=1.0,
                value=saved_config.get("long_decay", 0.70),
            )

        with col2:
            st.markdown("**Saturation Curve Shape**")
            st.caption("Controls how quickly channels reach diminishing returns")

            curve_sharpness = st.slider(
                "Curve Sharpness",
                min_value=0,
                max_value=100,
                value=saved_config.get("curve_sharpness", 50),
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
                value=saved_config.get("yearly_seasonality", 2),
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

        # Get saved values from config_state
        saved_sampling = st.session_state.config_state

        col1, col2 = st.columns(2)

        with col1:
            draws = st.number_input(
                "Posterior Draws",
                min_value=100,
                max_value=10000,
                value=saved_sampling.get("draws", 1500),
                step=100,
            )

            tune = st.number_input(
                "Tuning Steps",
                min_value=100,
                max_value=10000,
                value=saved_sampling.get("tune", 1500),
                step=100,
            )

        with col2:
            chains = st.number_input(
                "Number of Chains",
                min_value=1,
                max_value=8,
                value=saved_sampling.get("chains", 4),
            )

            target_accept = st.slider(
                "Target Acceptance Rate",
                min_value=0.5,
                max_value=0.99,
                value=saved_sampling.get("target_accept", 0.9),
            )

        random_seed = st.number_input(
            "Random Seed",
            min_value=0,
            value=saved_sampling.get("random_seed", 42),
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

    if st.button("🔨 Build Configuration", type="primary", width="stretch"):
        try:
            # Build the config
            config = build_config_from_state()
            st.session_state.current_config = config

            # Auto-save to workspace
            from mmm_platform.model.persistence import ConfigPersistence

            session_state = {
                "category_columns": st.session_state.get("category_columns", []),
                "config_state": st.session_state.get("config_state", {}),
                "date_column": st.session_state.get("date_column"),
                "target_column": st.session_state.get("target_column"),
                "detected_channels": st.session_state.get("detected_channels", []),
                "dayfirst": st.session_state.get("dayfirst", False),
            }

            path = ConfigPersistence.save(
                name=config.name,
                config=config,
                data=st.session_state.current_data,
                session_state=session_state,
            )

            st.success(f"Configuration '{config.name}' built and saved!")

            # Show summary
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Channels", len(config.channels))
            with col2:
                st.metric("Controls", len(config.controls))
            with col3:
                st.metric("Dummies", len(config.dummy_variables))
            with col4:
                st.metric("Target", config.data.target_column)
            with col5:
                st.metric("Status", "Ready")

            st.caption(f"Saved to: `{path}`")

        except Exception as e:
            st.error(f"Error building configuration: {e}")


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

    # Build dummy variable configs from session state
    dummy_variables = [
        DummyVariableConfig(
            name=dv["name"],
            start_date=dv["start_date"],
            end_date=dv["end_date"],
            categories=dv.get("categories", {}),
            sign_constraint=SignConstraint(dv.get("sign_constraint", "unconstrained")),
        )
        for dv in state.get("dummy_variables", [])
    ]

    # Build full config
    config = ModelConfig(
        name=state.get("name", "my_mmm_model"),
        description=state.get("description"),
        client=state.get("client"),
        data=DataConfig(
            date_column=state.get("date_col", "time"),
            target_column=state.get("target_col", "revenue"),
            dayfirst=state.get("dayfirst", True),
            revenue_scale=state.get("revenue_scale", 1000.0),
            spend_scale=state.get("revenue_scale", 1000.0),
            model_start_date=state.get("model_start_date"),
            model_end_date=state.get("model_end_date"),
            include_trend=state.get("include_trend", True),
        ),
        channels=channels,
        controls=controls,
        category_columns=category_columns,
        dummy_variables=dummy_variables,
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
