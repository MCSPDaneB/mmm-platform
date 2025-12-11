"""
Budget Optimization page for MMM Platform.

Provides interface for:
- Budget allocation optimization
- Incremental budget optimization
- Target-based optimization (find budget to hit target)
- Scenario analysis across budget levels
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import logging

logger = logging.getLogger(__name__)


def _fill_budget_callback():
    """Callback for Fill Budget button - sets budget_value (not widget key)."""
    from mmm_platform.optimization import BudgetAllocator

    try:
        wrapper = st.session_state.current_model
        fill_weeks = st.session_state.get("fill_weeks", 8)
        num_periods = st.session_state.get("opt_num_periods", 8)

        allocator = BudgetAllocator(wrapper, num_periods=num_periods)
        spend_dict, start_date, end_date = allocator.bridge.get_last_n_weeks_spend(
            n_weeks=fill_weeks,
            num_periods=num_periods,
        )
        total = sum(spend_dict.values())

        if total > 0:
            # Set our own variable - NOT the widget key
            st.session_state.budget_value = int(total)

            # Also auto-configure comparison to match the filled period
            st.session_state.opt_compare_historical = True
            st.session_state.opt_comparison_mode = "Last N weeks actual"
            st.session_state.opt_comparison_n_weeks = fill_weeks

            st.session_state.budget_fill_info = (
                f"Filled with ${total:,.0f} from {start_date:%Y-%m-%d} to {end_date:%Y-%m-%d}. "
                f"Comparison set to last {fill_weeks} weeks."
            )
        else:
            st.session_state.budget_fill_error = "No spend found for selected period"
    except Exception as e:
        st.session_state.budget_fill_error = str(e)


def show():
    """Display the budget optimization page with unified 2-tab layout."""
    st.title("🎯 Budget Optimization")

    # Check for fitted model
    if not st.session_state.get("model_fitted") or st.session_state.get("current_model") is None:
        st.warning("Please fit a model first on the Configure Model page.")
        st.info("Budget optimization requires a fitted model with posterior samples.")
        return

    wrapper = st.session_state.current_model

    try:
        from mmm_platform.optimization import (
            BudgetAllocator,
            TargetOptimizer,
            TimeDistribution,
            build_bounds_from_constraints,
            SeasonalIndexCalculator,
        )
    except ImportError as e:
        st.error(f"Optimization module not available: {e}")
        return

    # Get channel info for various inputs
    try:
        allocator = BudgetAllocator(wrapper, num_periods=8)
        channels = allocator.channels
        channel_info = allocator.get_channel_info()
    except Exception as e:
        st.error(f"Error initializing optimizer: {e}")
        return

    # Create unified 2-tab layout
    config_tab, results_tab = st.tabs(["⚙️ Configuration", "📊 Results"])

    with config_tab:
        _show_configuration_tab(wrapper, allocator, channel_info)

    with results_tab:
        _show_results_tab(wrapper, channel_info)


def _show_configuration_tab(wrapper, allocator, channel_info):
    """Display the unified configuration tab with mode selector."""
    from mmm_platform.optimization import SeasonalIndexCalculator

    # Mode selector
    st.markdown("### Optimization Mode")
    mode = st.radio(
        "Select optimization mode",
        options=["optimize", "incremental", "target", "scenarios"],
        format_func=lambda x: {
            "optimize": "📊 Optimize Budget",
            "incremental": "➕ Incremental Budget",
            "target": "🎯 Find Target",
            "scenarios": "📈 Scenarios",
        }[x],
        horizontal=True,
        help=(
            "**Optimize Budget**: Allocate total budget optimally across channels.\n\n"
            "**Incremental Budget**: Add extra budget on top of committed spend.\n\n"
            "**Find Target**: Find minimum budget to achieve a target response.\n\n"
            "**Scenarios**: Compare results across multiple budget levels."
        ),
        key="optimization_mode_selector",
    )

    # Store mode in session state
    st.session_state.optimization_mode = mode

    st.markdown("---")

    # Mode-specific inputs
    if mode == "optimize":
        _show_optimize_mode_inputs(channel_info)
    elif mode == "incremental":
        _show_incremental_mode_inputs(channel_info)
    elif mode == "target":
        _show_target_mode_inputs()
    elif mode == "scenarios":
        _show_scenarios_mode_inputs()

    st.markdown("---")

    # Common settings (for all modes except target and scenarios which have simpler needs)
    if mode in ["optimize", "incremental"]:
        _show_common_settings(allocator)
        st.markdown("---")

    # Shared expanders (bounds, seasonality) - for optimize and incremental modes
    if mode in ["optimize", "incremental"]:
        _show_channel_bounds_expander(channel_info)
        _show_seasonality_expander(wrapper, channel_info)
        _show_validation_expander(wrapper)

    # Run button
    st.markdown("---")
    _show_run_button(wrapper, allocator, channel_info, mode)


def _show_optimize_mode_inputs(channel_info):
    """Show inputs specific to optimize budget mode."""
    from mmm_platform.optimization import BudgetAllocator

    st.markdown("### Budget Settings")

    # Get available date range for fill functionality
    try:
        wrapper = st.session_state.current_model
        allocator = BudgetAllocator(wrapper, num_periods=8)
        min_date, max_date, total_periods = allocator.bridge.get_available_date_range()
    except Exception:
        total_periods = 52

    # Initialize our budget storage (separate from widget key)
    if "budget_value" not in st.session_state:
        st.session_state.budget_value = 100000

    # Total budget with quick fill
    col_budget, col_fill = st.columns([2, 1])

    with col_budget:
        # No key parameter - allows programmatic control via value parameter
        total_budget = st.number_input(
            "Total Budget ($)",
            min_value=1000,
            max_value=100000000,
            value=st.session_state.budget_value,
            step=10000,
            format="%d",
        )
        # Sync user edits back to our storage
        st.session_state.budget_value = total_budget

    with col_fill:
        st.markdown("**Quick Fill**")

        fill_weeks = st.number_input(
            "Last N weeks",
            min_value=1,
            max_value=total_periods,
            value=min(8, total_periods),
            step=1,
            key="fill_weeks",
            help="Fill budget with actual spend from the last N weeks",
        )

        # Use on_click callback - runs BEFORE widgets render on next run
        st.button(
            "📥 Fill",
            key="fill_budget_btn",
            on_click=_fill_budget_callback,
            help="Fill budget from historical spend",
        )

    # Show fill info/error messages
    if "budget_fill_info" in st.session_state:
        st.info(st.session_state.budget_fill_info)
    if "budget_fill_error" in st.session_state:
        st.error(f"Could not fill budget: {st.session_state.budget_fill_error}")
        del st.session_state.budget_fill_error

    # Optimization objective
    optimization_objective = st.selectbox(
        "Optimization Objective",
        options=["Maximize Response", "ROI Floor", "CPA Floor"],
        help=(
            "**Maximize Response**: Allocate budget to maximize expected response.\n\n"
            "**ROI Floor**: Maximize response while maintaining a minimum ROI. "
            "May return unallocated budget if the target can't be met at full spend.\n\n"
            "**CPA Floor**: Maximize response while keeping cost-per-acquisition below a threshold."
        ),
        key="opt_objective",
    )

    # Efficiency target inputs (only show for ROI/CPA floor)
    if optimization_objective == "ROI Floor":
        st.number_input(
            "Minimum ROI",
            min_value=0.1,
            value=2.0,
            step=0.1,
            help="Minimum return on investment required (e.g., 2.0 = 2x return)",
            key="opt_efficiency_target",
        )
    elif optimization_objective == "CPA Floor":
        st.number_input(
            "Maximum CPA ($)",
            min_value=0.01,
            value=10.0,
            step=1.0,
            help="Maximum cost per acquisition allowed",
            key="opt_efficiency_target",
        )

    # Compare to current toggle
    compare_to_current = st.checkbox(
        "Compare to historical spend",
        value=False,
        key="opt_compare_historical",
    )

    if compare_to_current:
        _show_comparison_options()


def _show_incremental_mode_inputs(channel_info):
    """Show inputs specific to incremental budget mode."""
    st.markdown("### Incremental Budget Settings")

    st.caption(
        "You have a committed budget plan. Enter your current allocation below, "
        "then specify the extra budget to optimize."
    )

    # Incremental budget amount
    incremental_budget = st.number_input(
        "Incremental Budget ($)",
        min_value=1000,
        max_value=100000000,
        value=st.session_state.get("inc_budget", 50000),
        step=5000,
        format="%d",
        help="The additional budget you want to allocate on top of your committed plan",
        key="inc_budget",
    )

    # Current allocation input
    st.markdown("### Committed Budget")
    st.caption(
        "Enter your current committed spend per channel. "
        "Pre-filled with average spend × periods."
    )

    # Get num_periods for default calculation (use session state if available)
    num_periods = st.session_state.get("opt_num_periods", 8)

    # Initialize base allocation in session state if not exists
    if "base_allocation" not in st.session_state:
        st.session_state.base_allocation = {}

    base_allocation = {}
    for _, row in channel_info.iterrows():
        ch = row["channel"]
        display = row["display_name"]
        avg = row["avg_period_spend"]
        default_value = int(avg * num_periods)

        # Use session state value if available, otherwise use default
        current_val = st.session_state.base_allocation.get(ch, default_value)

        base_allocation[ch] = float(st.number_input(
            f"{display}",
            min_value=0,
            value=int(current_val),
            step=1000,
            key=f"base_{ch}",
        ))

    # Store in session state
    st.session_state.base_allocation = base_allocation

    # Show totals
    committed_total = sum(base_allocation.values())
    total_with_incremental = committed_total + incremental_budget

    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    col1.metric("Committed Total", f"${committed_total:,.0f}")
    col2.metric("+ Incremental", f"${incremental_budget:,.0f}")
    col3.metric("**Total Budget**", f"${total_with_incremental:,.0f}")


def _show_target_mode_inputs():
    """Show inputs specific to target mode."""
    st.markdown("### Target Settings")

    st.caption("Find the minimum budget needed to achieve a target response.")

    # Target response
    st.number_input(
        "Target Response ($)",
        min_value=1000,
        max_value=100000000,
        value=st.session_state.get("target_response", 500000),
        step=50000,
        format="%d",
        help="The revenue/conversions you want to achieve",
        key="target_response",
    )

    # Budget search range
    st.markdown("**Budget Search Range**")
    col1, col2 = st.columns(2)
    with col1:
        st.number_input(
            "Minimum Budget ($)",
            min_value=1000,
            value=st.session_state.get("target_min_budget", 10000),
            step=5000,
            key="target_min_budget",
        )
    with col2:
        st.number_input(
            "Maximum Budget ($)",
            min_value=10000,
            value=st.session_state.get("target_max_budget", 1000000),
            step=50000,
            key="target_max_budget",
        )

    # Number of periods
    st.slider(
        "Forecast Periods (weeks)",
        min_value=1,
        max_value=52,
        value=st.session_state.get("target_num_periods", 8),
        key="target_num_periods",
    )


def _show_scenarios_mode_inputs():
    """Show inputs specific to scenarios mode."""
    st.markdown("### Scenario Settings")

    st.caption("Compare optimization results across multiple budget levels.")

    # Budget scenarios
    st.markdown("**Budget Scenarios**")
    scenario_input = st.text_area(
        "Enter budgets (one per line or comma-separated)",
        value=st.session_state.get("scenario_input", "50000\n100000\n150000\n200000\n250000"),
        height=150,
        key="scenario_input",
    )

    # Parse scenarios
    try:
        scenarios = []
        for line in scenario_input.strip().split("\n"):
            for val in line.split(","):
                val = val.strip().replace("$", "").replace(",", "")
                if val:
                    scenarios.append(float(val))
        scenarios = sorted(set(scenarios))
        st.session_state.parsed_scenarios = scenarios
    except ValueError:
        st.error("Invalid budget values. Enter numbers only.")
        scenarios = []
        st.session_state.parsed_scenarios = []

    if scenarios:
        st.caption(f"Analyzing {len(scenarios)} scenarios: ${min(scenarios):,.0f} - ${max(scenarios):,.0f}")

    # Number of periods
    st.slider(
        "Forecast Periods (weeks)",
        min_value=1,
        max_value=52,
        value=st.session_state.get("scenario_num_periods", 8),
        key="scenario_num_periods",
    )


def _show_comparison_options():
    """Show comparison baseline options for historical spend comparison."""
    # Get available date range for context
    try:
        from mmm_platform.optimization import BudgetAllocator
        wrapper = st.session_state.current_model
        allocator = BudgetAllocator(wrapper, num_periods=8)
        min_date, max_date, total_periods = allocator.bridge.get_available_date_range()
        st.caption(
            f"Data: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')} "
            f"({total_periods} periods)"
        )
    except Exception:
        total_periods = 52

    num_periods = st.session_state.get("opt_num_periods", 8)

    comparison_options = {
        "Average (all data)": "average",
        "Last N weeks actual": "last_n_weeks",
        f"Most recent {num_periods} weeks": "most_recent_period",
    }

    comparison_label = st.selectbox(
        "Comparison baseline",
        options=list(comparison_options.keys()),
        help=(
            "How to calculate the 'historical' spend for comparison:\n\n"
            "- **Average (all data)**: Average weekly spend across all historical data, "
            "multiplied by the forecast periods.\n\n"
            "- **Last N weeks actual**: Actual spend from the last N weeks of data, "
            "extrapolated to match the forecast horizon.\n\n"
            "- **Most recent N weeks**: Actual spend from the most recent weeks "
            "matching your optimization period (no extrapolation)."
        ),
        key="opt_comparison_mode",
    )
    comparison_mode = comparison_options[comparison_label]

    # Show N weeks input if "Last N weeks" is selected
    if comparison_mode == "last_n_weeks":
        st.number_input(
            "Number of weeks to use",
            min_value=1,
            max_value=total_periods,
            value=min(52, total_periods),
            step=1,
            help=f"Look back this many weeks from the most recent date. Max available: {total_periods} weeks.",
            key="opt_comparison_n_weeks",
        )


def _show_common_settings(allocator):
    """Show common settings for optimize and incremental modes."""
    st.markdown("### Optimization Period")

    col1, col2 = st.columns(2)

    with col1:
        # Number of periods
        st.slider(
            "Forecast Periods (weeks)",
            min_value=1,
            max_value=52,
            value=st.session_state.get("opt_num_periods", 8),
            key="opt_num_periods",
        )

    with col2:
        # Starting month
        month_names = [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"
        ]
        st.selectbox(
            "Starting Month",
            options=list(range(1, 13)),
            format_func=lambda x: month_names[x - 1],
            index=0,
            help="Which month does your optimization period start? This affects seasonal adjustments.",
            key="opt_start_month",
        )

    # Calculate end period info for display
    num_periods = st.session_state.get("opt_num_periods", 8)
    start_month = st.session_state.get("opt_start_month", 1)
    end_month_idx = (start_month - 1 + (num_periods // 4)) % 12
    period_info = f"{month_names[start_month - 1]}"
    if num_periods > 4:
        period_info += f" → {month_names[end_month_idx]}"
    st.caption(f"Period: {period_info} ({num_periods} weeks)")

    # Risk profile / Utility function
    utility_options = {
        "Mean (Risk Neutral)": "mean",
        "Value at Risk (Conservative)": "var",
        "Expected Shortfall (Very Conservative)": "cvar",
        "Sharpe Ratio (Risk-Adjusted)": "sharpe",
    }
    st.selectbox(
        "Risk Profile",
        options=list(utility_options.keys()),
        key="opt_utility",
    )


def _show_channel_bounds_expander(channel_info):
    """Show the channel bounds configuration expander."""
    with st.expander("Channel Bounds", expanded=False):
        st.caption(
            "Constrain how much each channel can change from historical spend."
        )

        bounds_mode = st.radio(
            "Bounds mode",
            ["No bounds", "Max % change", "Custom bounds"],
            horizontal=True,
            key="bounds_mode",
            help=(
                "**No bounds**: Optimizer can freely reallocate budget.\n\n"
                "**Max % change**: Limit each channel to ±X% of historical spend.\n\n"
                "**Custom bounds**: Set specific min/max for each channel."
            ),
        )

        num_periods = st.session_state.get("opt_num_periods", 8)

        if bounds_mode == "Max % change":
            max_delta = st.slider(
                "Maximum change per channel (%)",
                min_value=10,
                max_value=100,
                value=30,
                step=5,
                key="max_delta_pct",
                help="No channel can increase or decrease by more than this percentage from historical",
            )

            # Calculate bounds from historical spend
            bounds_config = {}
            for _, row in channel_info.iterrows():
                ch = row["channel"]
                avg = row["avg_period_spend"]
                historical = avg * num_periods
                min_val = historical * (1 - max_delta / 100)
                max_val = historical * (1 + max_delta / 100)
                bounds_config[ch] = (max(0.0, min_val), max_val)

            st.session_state.bounds_config = bounds_config

            # Show preview table
            st.caption(f"Each channel limited to ±{max_delta}% of historical spend:")
            preview_data = []
            for _, row in channel_info.iterrows():
                ch = row["channel"]
                display = row["display_name"]
                min_b, max_b = bounds_config[ch]
                historical = row["avg_period_spend"] * num_periods
                preview_data.append({
                    "Channel": display,
                    "Historical": f"${historical:,.0f}",
                    "Min": f"${min_b:,.0f}",
                    "Max": f"${max_b:,.0f}",
                })
            import pandas as pd
            st.dataframe(pd.DataFrame(preview_data), hide_index=True, use_container_width=True)

        elif bounds_mode == "Custom bounds":
            st.caption(
                "Set minimum and maximum spend per channel. "
                "Use **Min** for pre-committed budgets. "
                "Use **Max** to cap investment."
            )
            bounds_config = {}
            for _, row in channel_info.iterrows():
                ch = row["channel"]
                display = row["display_name"]
                avg = row["avg_period_spend"]

                col_min, col_max = st.columns(2)
                with col_min:
                    min_val = st.number_input(
                        f"{display} Min",
                        min_value=0,
                        value=0,
                        step=1000,
                        key=f"bound_min_{ch}",
                    )
                with col_max:
                    max_val = st.number_input(
                        f"{display} Max",
                        min_value=0,
                        value=int(avg * num_periods * 3),
                        step=1000,
                        key=f"bound_max_{ch}",
                    )
                bounds_config[ch] = (float(min_val), float(max_val))

            st.session_state.bounds_config = bounds_config

        else:  # No bounds
            st.session_state.bounds_config = None


def _show_seasonality_expander(wrapper, channel_info):
    """Show the seasonal adjustments configuration expander."""
    from mmm_platform.optimization import SeasonalIndexCalculator

    with st.expander("Seasonal Adjustments", expanded=False):
        st.caption(
            "Seasonality affects both overall demand and channel effectiveness. "
            "These indices help the optimizer account for time-of-year variations."
        )

        try:
            seasonal_calc = SeasonalIndexCalculator(wrapper)

            start_month = st.session_state.get("opt_start_month", 1)
            num_periods = st.session_state.get("opt_num_periods", 8)
            num_months = max(1, num_periods // 4)

            # Get demand index for period
            demand_index = seasonal_calc.get_demand_index_for_period(
                start_month=start_month,
                num_months=num_months,
            )

            # Get channel effectiveness indices for period
            seasonal_indices = seasonal_calc.get_indices_for_period(
                start_month=start_month,
                num_months=num_months,
            )

            # Get confidence info
            confidence_info = seasonal_calc.get_confidence_info(
                start_month=start_month,
                num_months=num_months,
            )

            # Show demand seasonality prominently
            st.markdown("**Demand Seasonality**")
            demand_pct = (demand_index - 1) * 100
            if demand_index > 1.05:
                st.success(
                    f"📈 Demand Index: **{demand_index:.2f}** — "
                    f"{abs(demand_pct):.0f}% higher than average for this period"
                )
            elif demand_index < 0.95:
                st.warning(
                    f"📉 Demand Index: **{demand_index:.2f}** — "
                    f"{abs(demand_pct):.0f}% lower than average for this period"
                )
            else:
                st.info(
                    f"➡️ Demand Index: **{demand_index:.2f}** — "
                    "Average demand for this period"
                )

            # Store demand index in session state
            st.session_state.demand_index = demand_index

            # Download demand seasonality
            demand_csv_data = [{
                "Period": f"{start_month} ({num_months} months)",
                "Demand Index": demand_index,
                "Interpretation": (
                    "Above average" if demand_index > 1.05 else
                    "Below average" if demand_index < 0.95 else
                    "Average"
                ),
            }]
            demand_df = pd.DataFrame(demand_csv_data)
            demand_csv = demand_df.to_csv(index=False)
            st.download_button(
                label="Download Demand Seasonality",
                data=demand_csv,
                file_name="demand_seasonality.csv",
                mime="text/csv",
                key="download_demand_seasonality",
            )

            st.markdown("---")
            st.markdown("**Channel Effectiveness**")

            # Show confidence indicator
            if confidence_info["confidence_level"] == "high":
                st.success(
                    f"High confidence indices based on {confidence_info['avg_observations']:.0f} "
                    f"observations per month"
                )
            elif confidence_info["confidence_level"] == "medium":
                st.info(
                    f"Medium confidence indices based on {confidence_info['avg_observations']:.0f} "
                    f"observations per month"
                )
            else:
                st.warning(
                    f"Low confidence - only {confidence_info['avg_observations']:.0f} "
                    f"observations per month. Consider using quarterly indices."
                )

            if confidence_info["using_quarterly"]:
                st.caption("Using quarterly indices due to limited monthly data.")

            # Display indices for selected period
            indices_data = []
            for ch, idx in seasonal_indices.items():
                display_name = channel_info[channel_info["channel"] == ch]["display_name"].values
                display_name = display_name[0] if len(display_name) > 0 else ch
                indices_data.append({
                    "Channel": display_name,
                    "Index": idx,
                    "Interpretation": (
                        "More effective" if idx > 1.05 else
                        "Less effective" if idx < 0.95 else
                        "Average"
                    ),
                })

            indices_df = pd.DataFrame(indices_data).sort_values("Index", ascending=False)

            # Allow user to override indices
            use_custom_indices = st.checkbox(
                "Customize seasonal indices",
                value=False,
                help="Override the computed indices based on your business knowledge",
                key="use_custom_seasonal",
            )

            if use_custom_indices:
                st.caption(
                    "Edit the indices below. Values > 1.0 mean more effective, < 1.0 means less effective."
                )

                # Create editable dataframe
                edited_df = st.data_editor(
                    indices_df,
                    column_config={
                        "Channel": st.column_config.TextColumn("Channel", disabled=True),
                        "Index": st.column_config.NumberColumn(
                            "Index",
                            min_value=0.1,
                            max_value=3.0,
                            step=0.05,
                            format="%.2f",
                        ),
                        "Interpretation": st.column_config.TextColumn("Interpretation", disabled=True),
                    },
                    hide_index=True,
                    width="stretch",
                    key="seasonal_editor",
                )

                # Extract edited indices back to dict
                edited_indices = {}
                for _, row in edited_df.iterrows():
                    display_name = row["Channel"]
                    for ch in seasonal_indices.keys():
                        ch_display = channel_info[channel_info["channel"] == ch]["display_name"].values
                        ch_display = ch_display[0] if len(ch_display) > 0 else ch
                        if ch_display == display_name:
                            edited_indices[ch] = float(row["Index"])
                            break

                st.session_state.seasonal_indices = edited_indices
            else:
                # Display read-only table
                display_indices_df = indices_df.copy()
                display_indices_df["Index"] = display_indices_df["Index"].apply(lambda x: f"{x:.2f}")
                st.dataframe(display_indices_df, width="stretch", hide_index=True)

                # Store computed indices
                st.session_state.seasonal_indices = seasonal_indices

            # Download channel seasonality
            channel_csv_data = []
            for ch, idx in seasonal_indices.items():
                display_name = channel_info[channel_info["channel"] == ch]["display_name"].values
                display_name = display_name[0] if len(display_name) > 0 else ch
                channel_csv_data.append({
                    "Channel": ch,
                    "Display Name": display_name,
                    "Index": idx,
                })
            channel_df = pd.DataFrame(channel_csv_data)
            channel_csv = channel_df.to_csv(index=False)
            st.download_button(
                label="Download Channel Seasonality",
                data=channel_csv,
                file_name="channel_seasonality.csv",
                mime="text/csv",
                key="download_channel_seasonality",
            )

            # Upload custom channel indices
            uploaded_file = st.file_uploader(
                "Upload custom channel indices (CSV)",
                type=["csv"],
                help="Upload a CSV with Channel and Index columns to override computed indices",
                key="seasonal_upload",
            )

            if uploaded_file is not None:
                try:
                    custom_df = pd.read_csv(uploaded_file)
                    updated_count = 0
                    for ch in seasonal_indices.keys():
                        if "Channel" in custom_df.columns and ch in custom_df["Channel"].values:
                            row = custom_df[custom_df["Channel"] == ch]
                            if "Index" in custom_df.columns:
                                seasonal_indices[ch] = float(row["Index"].values[0])
                                updated_count += 1
                    if updated_count > 0:
                        st.success(f"Loaded custom indices for {updated_count} channels")
                        st.session_state.seasonal_indices = seasonal_indices
                    else:
                        st.warning("No matching channels found in uploaded file. Ensure CSV has 'Channel' and 'Index' columns.")
                except Exception as e:
                    st.error(f"Error reading file: {e}")

        except Exception as e:
            import traceback
            error_msg = f"{type(e).__name__}: {str(e)}" if str(e) else type(e).__name__
            st.warning(f"Could not compute seasonal indices: {error_msg}")
            st.code(traceback.format_exc())
            st.session_state.seasonal_indices = None


def _show_validation_expander(wrapper):
    """Show the optimizer validation expander."""
    with st.expander("Validate Optimizer Accuracy", expanded=False):
        st.caption("Compare model predictions vs actual results for historical data")

        validation_periods = st.slider(
            "Periods to validate",
            min_value=4,
            max_value=min(52, len(wrapper.df_scaled) if wrapper.df_scaled is not None else 52),
            value=min(26, len(wrapper.df_scaled) if wrapper.df_scaled is not None else 26),
            key="validation_periods",
        )

        if st.button("Run Validation", key="run_validation"):
            with st.spinner("Running backtest validation..."):
                try:
                    from mmm_platform.analysis.backtest import BacktestValidator

                    validator = BacktestValidator(wrapper)
                    backtest_df = validator.run_backtest(validation_periods)
                    metrics = validator.get_metrics(backtest_df)

                    st.session_state.backtest_metrics = metrics
                    st.session_state.backtest_df = backtest_df

                except Exception as e:
                    st.error(f"Validation failed: {e}")
                    logger.exception("Backtest validation error")

        # Display validation results if available
        if "backtest_metrics" in st.session_state:
            metrics = st.session_state.backtest_metrics
            backtest_df = st.session_state.backtest_df

            m1, m2, m3 = st.columns(3)
            m1.metric("R²", f"{metrics['r2']:.3f}")
            m2.metric("MAPE", f"{metrics['mape']:.1f}%")
            m3.metric("Correlation", f"{metrics['correlation']:.3f}")

            if metrics['r2'] > 0.7 and metrics['mape'] < 15:
                st.success("Excellent fit - optimizer predictions are reliable")
            elif metrics['r2'] > 0.5 and metrics['mape'] < 25:
                st.info("Good fit - optimizer predictions are reasonably accurate")
            elif metrics['r2'] > 0.3:
                st.warning("Moderate fit - use optimizer results with caution")
            else:
                st.error("Poor fit - optimizer predictions may not be reliable")

            # Time series chart
            fig_ts = go.Figure()
            fig_ts.add_trace(go.Scatter(
                x=backtest_df['date'],
                y=backtest_df['actual'],
                mode='lines+markers',
                name='Actual',
                line=dict(color='blue'),
            ))
            fig_ts.add_trace(go.Scatter(
                x=backtest_df['date'],
                y=backtest_df['predicted'],
                mode='lines+markers',
                name='Predicted',
                line=dict(color='orange'),
            ))
            fig_ts.update_layout(
                title="Actual vs Predicted Response Over Time",
                xaxis_title="Date",
                yaxis_title="Response",
                height=300,
            )
            st.plotly_chart(fig_ts, width="stretch")

            # Scatter plot
            fig_scatter = px.scatter(
                backtest_df,
                x='actual',
                y='predicted',
                title="Predicted vs Actual (ideal = diagonal line)",
            )
            max_val = max(backtest_df['actual'].max(), backtest_df['predicted'].max())
            fig_scatter.add_trace(go.Scatter(
                x=[0, max_val],
                y=[0, max_val],
                mode='lines',
                name='Perfect fit',
                line=dict(dash='dash', color='gray'),
            ))
            fig_scatter.update_layout(height=300)
            st.plotly_chart(fig_scatter, width="stretch")


def _show_run_button(wrapper, allocator, channel_info, mode):
    """Show the run button and handle optimization execution."""
    from mmm_platform.optimization import BudgetAllocator, TargetOptimizer

    # Get button label based on mode
    button_labels = {
        "optimize": "🚀 Optimize Budget",
        "incremental": "🚀 Optimize Incremental Budget",
        "target": "🚀 Find Required Budget",
        "scenarios": "🚀 Run Scenario Analysis",
    }

    # Disable scenarios button if not enough scenarios
    disabled = False
    if mode == "scenarios":
        scenarios = st.session_state.get("parsed_scenarios", [])
        disabled = len(scenarios) < 2

    run_clicked = st.button(
        button_labels[mode],
        type="primary",
        disabled=disabled,
    )

    if run_clicked:
        if mode == "optimize":
            _run_optimize(wrapper)
        elif mode == "incremental":
            _run_incremental(wrapper)
        elif mode == "target":
            _run_target(wrapper)
        elif mode == "scenarios":
            _run_scenarios(wrapper)


def _run_optimize(wrapper):
    """Run the budget optimization."""
    from mmm_platform.optimization import BudgetAllocator

    with st.spinner("Running optimization..."):
        try:
            num_periods = st.session_state.get("opt_num_periods", 8)
            utility_options = {
                "Mean (Risk Neutral)": "mean",
                "Value at Risk (Conservative)": "var",
                "Expected Shortfall (Very Conservative)": "cvar",
                "Sharpe Ratio (Risk-Adjusted)": "sharpe",
            }
            utility_label = st.session_state.get("opt_utility", "Mean (Risk Neutral)")
            utility = utility_options.get(utility_label, "mean")

            allocator = BudgetAllocator(
                wrapper,
                num_periods=num_periods,
                utility=utility,
            )

            total_budget = st.session_state.get("budget_value", 100000)
            channel_bounds = st.session_state.get("bounds_config")
            seasonal_indices = st.session_state.get("seasonal_indices")
            compare_to_current = st.session_state.get("opt_compare_historical", False)

            # Get comparison settings
            comparison_mode = "average"
            comparison_n_weeks = None
            if compare_to_current:
                comparison_options = {
                    "Average (all data)": "average",
                    "Last N weeks actual": "last_n_weeks",
                }
                comparison_label = st.session_state.get("opt_comparison_mode", "Average (all data)")
                if "Most recent" in comparison_label:
                    comparison_mode = "most_recent_period"
                elif comparison_label in comparison_options:
                    comparison_mode = comparison_options[comparison_label]
                comparison_n_weeks = st.session_state.get("opt_comparison_n_weeks")

            # Get efficiency settings
            optimization_objective = st.session_state.get("opt_objective", "Maximize Response")
            efficiency_metric = None
            efficiency_target = None

            if optimization_objective == "ROI Floor":
                efficiency_metric = "roi"
                efficiency_target = st.session_state.get("opt_efficiency_target", 2.0)
            elif optimization_objective == "CPA Floor":
                efficiency_metric = "cpa"
                efficiency_target = st.session_state.get("opt_efficiency_target", 10.0)

            # Run optimization
            if efficiency_metric is not None and efficiency_target is not None:
                result = allocator.optimize_with_efficiency_floor(
                    total_budget=total_budget,
                    efficiency_metric=efficiency_metric,
                    efficiency_target=efficiency_target,
                    channel_bounds=channel_bounds,
                    seasonal_indices=seasonal_indices,
                    compare_to_current=compare_to_current,
                    comparison_mode=comparison_mode,
                    comparison_n_weeks=comparison_n_weeks,
                )
            else:
                result = allocator.optimize(
                    total_budget=total_budget,
                    channel_bounds=channel_bounds,
                    compare_to_current=compare_to_current,
                    comparison_mode=comparison_mode,
                    comparison_n_weeks=comparison_n_weeks,
                    seasonal_indices=seasonal_indices,
                )

            # Store result
            st.session_state.optimization_result = result
            st.session_state.result_mode = "optimize"

            # Store config for display
            st.session_state.optimizer_config = {
                "mode": "optimize",
                "total_budget": total_budget,
                "num_periods": num_periods,
                "start_month": st.session_state.get("opt_start_month", 1),
                "utility": utility_label,
                "optimization_objective": optimization_objective,
                "compare_to_current": compare_to_current,
                "comparison_mode": comparison_mode if compare_to_current else None,
            }

            st.success("Optimization complete! Switch to Results tab to view.")

        except Exception as e:
            st.error(f"Optimization failed: {e}")
            logger.exception("Optimization error")


def _run_incremental(wrapper):
    """Run the incremental budget optimization."""
    from mmm_platform.optimization import BudgetAllocator

    with st.spinner("Optimizing incremental budget..."):
        try:
            num_periods = st.session_state.get("opt_num_periods", 8)
            utility_options = {
                "Mean (Risk Neutral)": "mean",
                "Value at Risk (Conservative)": "var",
                "Expected Shortfall (Very Conservative)": "cvar",
                "Sharpe Ratio (Risk-Adjusted)": "sharpe",
            }
            utility_label = st.session_state.get("opt_utility", "Mean (Risk Neutral)")
            utility = utility_options.get(utility_label, "mean")

            allocator = BudgetAllocator(
                wrapper,
                num_periods=num_periods,
                utility=utility,
            )

            incremental_budget = st.session_state.get("inc_budget", 50000)
            base_allocation = st.session_state.get("base_allocation", {})

            result = allocator.optimize_incremental(
                base_allocation=base_allocation,
                incremental_budget=incremental_budget,
            )

            # Store result
            st.session_state.optimization_result = result
            st.session_state.result_mode = "incremental"
            st.session_state.incremental_budget_amount = incremental_budget

            # Store config for display
            st.session_state.optimizer_config = {
                "mode": "incremental",
                "incremental_budget": incremental_budget,
                "committed_total": sum(base_allocation.values()),
                "num_periods": num_periods,
                "utility": utility_label,
            }

            st.success("Optimization complete! Switch to Results tab to view.")

        except Exception as e:
            st.error(f"Optimization failed: {e}")
            logger.exception("Incremental optimization error")


def _run_target(wrapper):
    """Run the target-based optimization."""
    from mmm_platform.optimization import BudgetAllocator, TargetOptimizer

    min_budget = st.session_state.get("target_min_budget", 10000)
    max_budget = st.session_state.get("target_max_budget", 1000000)

    if min_budget >= max_budget:
        st.error("Minimum budget must be less than maximum budget")
        return

    with st.spinner("Searching for optimal budget..."):
        try:
            num_periods = st.session_state.get("target_num_periods", 8)
            target_response = st.session_state.get("target_response", 500000)

            allocator = BudgetAllocator(wrapper, num_periods=num_periods)
            target_opt = TargetOptimizer(allocator)

            result = target_opt.find_budget_for_target(
                target_response=target_response,
                budget_range=(min_budget, max_budget),
            )

            # Store result
            st.session_state.optimization_result = result
            st.session_state.result_mode = "target"

            # Store config for display
            st.session_state.optimizer_config = {
                "mode": "target",
                "target_response": target_response,
                "budget_range": (min_budget, max_budget),
                "num_periods": num_periods,
            }

            st.success("Search complete! Switch to Results tab to view.")

        except Exception as e:
            st.error(f"Search failed: {e}")
            logger.exception("Target search error")


def _run_scenarios(wrapper):
    """Run the scenario analysis."""
    from mmm_platform.optimization import BudgetAllocator

    scenarios = st.session_state.get("parsed_scenarios", [])
    if len(scenarios) < 2:
        st.error("Need at least 2 budget scenarios to analyze")
        return

    with st.spinner(f"Analyzing {len(scenarios)} scenarios..."):
        try:
            num_periods = st.session_state.get("scenario_num_periods", 8)

            allocator = BudgetAllocator(wrapper, num_periods=num_periods)
            result = allocator.scenario_analysis(scenarios)

            # Store result
            st.session_state.optimization_result = result
            st.session_state.result_mode = "scenarios"

            # Store config for display
            st.session_state.optimizer_config = {
                "mode": "scenarios",
                "scenarios": scenarios,
                "num_periods": num_periods,
            }

            st.success("Analysis complete! Switch to Results tab to view.")

        except Exception as e:
            st.error(f"Analysis failed: {e}")
            logger.exception("Scenario analysis error")


def _show_results_tab(wrapper, channel_info):
    """Display the results tab with mode-specific results."""
    result_mode = st.session_state.get("result_mode")

    if "optimization_result" not in st.session_state or result_mode is None:
        st.info("Configure optimization settings and click Run to see results.")
        return

    result = st.session_state.optimization_result
    config = st.session_state.get("optimizer_config", {})

    # Config summary expander
    _show_config_summary(config)

    # Display mode-specific results
    if result_mode == "optimize":
        _show_optimize_results(wrapper, result)
    elif result_mode == "incremental":
        _show_incremental_results(wrapper, result)
    elif result_mode == "target":
        _show_target_results(result)
    elif result_mode == "scenarios":
        _show_scenarios_results(result)


def _show_config_summary(config):
    """Show configuration summary in an expander."""
    if not config:
        return

    month_names = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ]

    mode = config.get("mode", "optimize")

    with st.expander("Configuration Summary", expanded=False):
        if mode == "optimize":
            c1, c2, c3 = st.columns(3)
            c1.metric("Budget", f"${config.get('total_budget', 0):,.0f}")
            c2.metric("Period", f"{config.get('num_periods', 8)} weeks")
            c3.metric("Starting", month_names[config.get('start_month', 1) - 1])
            st.caption(f"Objective: {config.get('optimization_objective', 'Maximize Response')} | Risk Profile: {config.get('utility', 'Mean')}")
            if config.get('compare_to_current'):
                st.caption(f"Comparison: {config.get('comparison_mode', 'average')}")

        elif mode == "incremental":
            c1, c2, c3 = st.columns(3)
            c1.metric("Committed", f"${config.get('committed_total', 0):,.0f}")
            c2.metric("Incremental", f"${config.get('incremental_budget', 0):,.0f}")
            c3.metric("Period", f"{config.get('num_periods', 8)} weeks")
            st.caption(f"Risk Profile: {config.get('utility', 'Mean')}")

        elif mode == "target":
            c1, c2, c3 = st.columns(3)
            c1.metric("Target", f"${config.get('target_response', 0):,.0f}")
            budget_range = config.get('budget_range', (0, 0))
            c2.metric("Min Budget", f"${budget_range[0]:,.0f}")
            c3.metric("Max Budget", f"${budget_range[1]:,.0f}")
            st.caption(f"Period: {config.get('num_periods', 8)} weeks")

        elif mode == "scenarios":
            scenarios = config.get('scenarios', [])
            st.caption(f"Analyzed {len(scenarios)} scenarios from ${min(scenarios):,.0f} to ${max(scenarios):,.0f}")
            st.caption(f"Period: {config.get('num_periods', 8)} weeks")


def _show_optimize_results(wrapper, result):
    """Show results for optimize budget mode."""
    if result.success:
        st.success(f"Optimization completed in {result.iterations} iterations")
    else:
        st.warning(f"Optimization message: {result.message}")

    if getattr(result, 'used_fallback', False):
        st.info(
            "Used enhanced gradient-based optimization for better results. "
            "The default optimizer had convergence issues with this model."
        )

    # Determine KPI type for display formatting
    kpi_type = getattr(wrapper.config.data, 'kpi_type', 'revenue')
    target_col = wrapper.config.data.target_column

    # Key metrics
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    with metric_col1:
        st.metric("Total Budget", f"${result.total_budget:,.0f}")
    with metric_col2:
        if kpi_type == "count":
            response_label = f"Expected {target_col.replace('_', ' ').title()}"
            response_value = f"{result.expected_response:,.0f}"
        else:
            response_label = "Expected Response"
            response_value = f"${result.expected_response:,.0f}"
        st.metric(
            response_label,
            response_value,
            delta=f"+{result.response_uplift_pct:.1f}%" if result.response_uplift_pct else None,
        )
    with metric_col3:
        if kpi_type == "count":
            cpa = result.total_budget / result.expected_response if result.expected_response > 0 else 0
            st.metric("Expected CPA", f"${cpa:,.2f}")
        else:
            roi = result.expected_response / result.total_budget if result.total_budget > 0 else 0
            st.metric("Expected ROI", f"{roi:.2f}x")

    # CI caption
    if kpi_type == "count":
        st.caption(f"95% CI: {result.response_ci_low:,.0f} - {result.response_ci_high:,.0f}")
    else:
        st.caption(f"95% CI: ${result.response_ci_low:,.0f} - ${result.response_ci_high:,.0f}")

    # Efficiency floor results (if applicable)
    if result.unallocated_budget is not None and result.unallocated_budget > 0:
        st.warning(
            f"**Unallocated Budget:** ${result.unallocated_budget:,.0f}\n\n"
            f"To achieve {result.efficiency_metric.upper()} target of "
            f"{result.efficiency_target:.2f}, only ${result.total_budget:,.0f} can be efficiently deployed."
        )
    elif result.efficiency_metric is not None:
        st.success(
            f"Full budget achieves {result.efficiency_metric.upper()} target: "
            f"achieved {result.achieved_efficiency:.2f} vs target {result.efficiency_target:.2f}"
        )

    # Allocation chart
    df = result.to_dataframe()
    fig = px.bar(
        df,
        x="channel",
        y="optimal",
        title="Optimal Budget Allocation",
        labels={"channel": "Channel", "optimal": "Budget ($)"},
        text=df["optimal"].apply(lambda x: f"${x:,.0f}"),
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig)

    # Percentage distribution comparison chart (when comparing to historical)
    if "current" in df.columns and result.current_allocation:
        _show_comparison_charts(result, df)

    # Allocation table
    _show_allocation_table(df)

    # Download results
    csv = result.to_dataframe().to_csv(index=False)
    st.download_button(
        "Download Results (CSV)",
        csv,
        "optimization_results.csv",
        "text/csv",
    )


def _show_comparison_charts(result, df):
    """Show comparison charts for historical vs optimal allocation."""
    current_total = sum(result.current_allocation.values())
    optimal_total = result.total_budget

    comparison_data = []
    for ch in df["channel"]:
        current_val = result.current_allocation.get(ch, 0)
        optimal_val = result.optimal_allocation.get(ch, 0)
        current_pct = (current_val / current_total * 100) if current_total > 0 else 0
        optimal_pct = (optimal_val / optimal_total * 100) if optimal_total > 0 else 0
        comparison_data.append({
            "channel": ch,
            "Historical %": current_pct,
            "Optimal %": optimal_pct,
            "Difference": optimal_pct - current_pct,
        })

    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values("Difference", key=abs, ascending=False)

    # Grouped bar chart for percentage comparison
    fig_pct = go.Figure()
    fig_pct.add_trace(go.Bar(
        name="Historical %",
        x=comparison_df["channel"],
        y=comparison_df["Historical %"],
        marker_color="rgba(100, 100, 100, 0.6)",
        text=comparison_df["Historical %"].apply(lambda x: f"{x:.1f}%"),
        textposition="outside",
    ))
    fig_pct.add_trace(go.Bar(
        name="Optimal %",
        x=comparison_df["channel"],
        y=comparison_df["Optimal %"],
        marker_color="rgba(99, 110, 250, 0.8)",
        text=comparison_df["Optimal %"].apply(lambda x: f"{x:.1f}%"),
        textposition="outside",
    ))
    fig_pct.update_layout(
        title="Budget Distribution: Historical vs Optimal",
        xaxis_title="Channel",
        yaxis_title="% of Total Budget",
        barmode="group",
        xaxis_tickangle=-45,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_pct)

    # Difference chart
    fig_diff = px.bar(
        comparison_df,
        x="channel",
        y="Difference",
        title="Allocation Shift: Optimal vs Historical (% points)",
        labels={"channel": "Channel", "Difference": "Change (% points)"},
        text=comparison_df["Difference"].apply(lambda x: f"{x:+.1f}%"),
        color="Difference",
        color_continuous_scale=["#ef553b", "#f0f0f0", "#00cc96"],
        color_continuous_midpoint=0,
    )
    fig_diff.update_traces(textposition="outside")
    fig_diff.update_layout(
        xaxis_tickangle=-45,
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig_diff)


def _show_allocation_table(df):
    """Show the allocation table with formatting."""
    display_df = df.copy()
    display_df["optimal"] = display_df["optimal"].apply(lambda x: f"${x:,.0f}")
    display_df["pct_of_total"] = display_df["pct_of_total"].apply(lambda x: f"{x:.1f}%")

    if "current" in display_df.columns:
        display_df["current"] = display_df["current"].apply(lambda x: f"${x:,.0f}")
        display_df["delta"] = display_df["delta"].apply(lambda x: f"${x:+,.0f}")
        display_df["pct_change"] = display_df["pct_change"].apply(lambda x: f"{x:+.1f}%")

    st.dataframe(display_df, width="stretch", hide_index=True)


def _show_incremental_results(wrapper, result):
    """Show results for incremental budget mode."""
    if result.success:
        st.success(f"Optimization completed in {result.iterations} iterations")
    else:
        st.warning(f"Optimization message: {result.message}")

    # Determine KPI type for display formatting
    kpi_type = getattr(wrapper.config.data, 'kpi_type', 'revenue')
    target_col = wrapper.config.data.target_column

    # Calculate incremental allocation
    incremental_allocation = {
        ch: result.optimal_allocation.get(ch, 0) - result.current_allocation.get(ch, 0)
        for ch in result.optimal_allocation
    }

    # Key insight message
    top_channels = sorted(
        incremental_allocation.items(),
        key=lambda x: x[1],
        reverse=True
    )[:3]
    top_channels = [(ch, amt) for ch, amt in top_channels if amt > 0]

    if top_channels:
        insight_parts = [f"${amt:,.0f} to {ch}" for ch, amt in top_channels]
        st.info(f"**Recommendation:** Put {', '.join(insight_parts)}")

    incremental_budget = st.session_state.get("incremental_budget_amount", 50000)

    # Key metrics
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    with metric_col1:
        st.metric("Incremental Budget", f"${incremental_budget:,.0f}")
    with metric_col2:
        if kpi_type == "count":
            response_label = f"Expected {target_col.replace('_', ' ').title()}"
            response_value = f"{result.expected_response:,.0f}"
        else:
            response_label = "Expected Response"
            response_value = f"${result.expected_response:,.0f}"
        st.metric(
            response_label,
            response_value,
            delta=f"+{result.response_uplift_pct:.1f}%" if result.response_uplift_pct else None,
        )
    with metric_col3:
        if result.current_response and result.current_response > 0:
            uplift = result.expected_response - result.current_response
            if kpi_type == "count":
                st.metric("Incremental Response", f"+{uplift:,.0f}")
            else:
                st.metric("Incremental Response", f"+${uplift:,.0f}")

    # Create comparison dataframe
    comparison_data = []
    for ch in result.optimal_allocation:
        committed = result.current_allocation.get(ch, 0)
        recommended = result.optimal_allocation.get(ch, 0)
        incremental = recommended - committed
        comparison_data.append({
            "Channel": ch,
            "Committed": committed,
            "Recommended": recommended,
            "Incremental": incremental,
        })

    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values("Incremental", ascending=False)

    # Incremental allocation chart
    fig = px.bar(
        comparison_df[comparison_df["Incremental"] > 0],
        x="Channel",
        y="Incremental",
        title="Where Your Incremental Budget Should Go",
        labels={"Incremental": "Additional Spend ($)"},
        text=comparison_df[comparison_df["Incremental"] > 0]["Incremental"].apply(
            lambda x: f"${x:,.0f}"
        ),
        color="Incremental",
        color_continuous_scale="Greens",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        xaxis_tickangle=-45,
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig)

    # Stacked bar chart: Committed vs Incremental
    fig_stacked = go.Figure()
    fig_stacked.add_trace(go.Bar(
        name="Committed",
        x=comparison_df["Channel"],
        y=comparison_df["Committed"],
        marker_color="rgba(100, 100, 100, 0.6)",
    ))
    fig_stacked.add_trace(go.Bar(
        name="Incremental",
        x=comparison_df["Channel"],
        y=comparison_df["Incremental"].clip(lower=0),
        marker_color="rgba(0, 200, 100, 0.8)",
    ))
    fig_stacked.update_layout(
        title="Total Allocation: Committed + Incremental",
        xaxis_title="Channel",
        yaxis_title="Budget ($)",
        barmode="stack",
        xaxis_tickangle=-45,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_stacked)

    # Allocation table
    display_df = comparison_df.copy()
    display_df["Committed"] = display_df["Committed"].apply(lambda x: f"${x:,.0f}")
    display_df["Recommended"] = display_df["Recommended"].apply(lambda x: f"${x:,.0f}")
    display_df["Incremental"] = display_df["Incremental"].apply(lambda x: f"${x:+,.0f}")

    st.dataframe(display_df, width="stretch", hide_index=True)

    # Download results
    csv = comparison_df.to_csv(index=False)
    st.download_button(
        "Download Results (CSV)",
        csv,
        "incremental_optimization_results.csv",
        "text/csv",
    )


def _show_target_results(result):
    """Show results for target mode."""
    if result.achievable:
        st.success(f"Target achievable! Found in {result.iterations} iterations.")
    else:
        st.warning("Target may not be achievable within the budget range.")

    # Key metrics
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    with metric_col1:
        st.metric("Target Response", f"${result.target_response:,.0f}")
    with metric_col2:
        st.metric("Required Budget", f"${result.required_budget:,.0f}")
    with metric_col3:
        st.metric("Expected Response", f"${result.expected_response:,.0f}")

    st.caption(f"95% CI: ${result.response_ci_low:,.0f} - ${result.response_ci_high:,.0f}")

    # Allocation breakdown
    st.markdown("#### Optimal Allocation")
    alloc_df = pd.DataFrame([
        {"Channel": ch, "Budget": f"${amt:,.0f}"}
        for ch, amt in result.optimal_allocation.items()
    ])
    st.dataframe(alloc_df, width="stretch", hide_index=True)

    # Show message
    st.info(result.message)


def _show_scenarios_results(result):
    """Show results for scenario analysis mode."""
    st.success(f"Analyzed {len(result.results)} scenarios")

    # Efficiency curve chart
    curve = result.efficiency_curve
    fig = go.Figure()

    # Response line
    fig.add_trace(go.Scatter(
        x=curve["budget"],
        y=curve["expected_response"],
        mode="lines+markers",
        name="Expected Response",
        line=dict(color="blue", width=2),
    ))

    # Confidence band
    fig.add_trace(go.Scatter(
        x=list(curve["budget"]) + list(curve["budget"][::-1]),
        y=list(curve["response_ci_high"]) + list(curve["response_ci_low"][::-1]),
        fill="toself",
        fillcolor="rgba(0,100,255,0.1)",
        line=dict(color="rgba(0,0,0,0)"),
        name="95% CI",
    ))

    fig.update_layout(
        title="Response vs Budget (Efficiency Frontier)",
        xaxis_title="Budget ($)",
        yaxis_title="Expected Response ($)",
        hovermode="x unified",
    )
    st.plotly_chart(fig)

    # Marginal ROI chart
    fig_marginal = px.line(
        curve,
        x="budget",
        y="marginal_response",
        title="Marginal ROI by Budget Level",
        labels={"budget": "Budget ($)", "marginal_response": "Marginal ROI"},
        markers=True,
    )
    fig_marginal.add_hline(y=1.0, line_dash="dash", line_color="red", annotation_text="Breakeven (ROI=1)")
    st.plotly_chart(fig_marginal)

    # Summary table
    summary_df = curve[["budget", "expected_response", "marginal_response"]].copy()
    summary_df["budget"] = summary_df["budget"].apply(lambda x: f"${x:,.0f}")
    summary_df["expected_response"] = summary_df["expected_response"].apply(lambda x: f"${x:,.0f}")
    summary_df["marginal_response"] = summary_df["marginal_response"].apply(lambda x: f"{x:.2f}")
    summary_df.columns = ["Budget", "Expected Response", "Marginal ROI"]
    st.dataframe(summary_df, width="stretch", hide_index=True)

    # Download
    csv = result.to_dataframe().to_csv(index=False)
    st.download_button(
        "Download Scenario Results (CSV)",
        csv,
        "scenario_results.csv",
        "text/csv",
    )
