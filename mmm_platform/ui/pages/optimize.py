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
        num_periods = st.session_state.get("opt_num_periods", 8)

        allocator = BudgetAllocator(wrapper, num_periods=num_periods)
        spend_dict, start_date, end_date = allocator.bridge.get_last_n_weeks_spend(
            n_weeks=num_periods,
        )
        total = sum(spend_dict.values())

        if total > 0:
            # Set our own variable - NOT the widget key
            st.session_state.budget_value = int(total)

            # Also auto-configure comparison to match the filled period
            st.session_state.opt_compare_historical = True

            st.session_state.budget_fill_info = (
                f"Filled with ${total:,.0f} from {start_date:%Y-%m-%d} to {end_date:%Y-%m-%d}."
            )
        else:
            st.session_state.budget_fill_error = "No spend found for selected period"
    except Exception as e:
        st.session_state.budget_fill_error = str(e)


def _render_settings_status_bar():
    """Render a dark status bar showing current optimization settings."""
    num_periods = st.session_state.get("opt_num_periods", 8)
    start_month = st.session_state.get("opt_start_month", 1)

    month_names = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
    ]
    month_name = month_names[start_month - 1]

    # Check if custom seasonality is applied
    seasonal_indices = st.session_state.get("seasonal_indices")
    has_custom_seasonality = seasonal_indices is not None and len(seasonal_indices) > 0

    # Check if custom bounds are set
    bounds_config = st.session_state.get("bounds_config")
    has_custom_bounds = bounds_config is not None and len(bounds_config) > 0

    # Build status items
    items = [
        f"📅 {num_periods} weeks",
        f"🗓️ Start: {month_name}",
        f"🌊 Seasonality: {'Custom' if has_custom_seasonality else 'None'}",
        f"📊 Bounds: {'Custom' if has_custom_bounds else 'Default'}",
    ]

    # Render dark banner with HTML/CSS
    st.markdown(
        f"""
        <div style="
            background-color: #3D3D3D;
            color: white;
            padding: 10px 20px;
            border-radius: 8px;
            margin-bottom: 16px;
            font-size: 14px;
            display: flex;
            justify-content: flex-start;
            gap: 24px;
            flex-wrap: wrap;
        ">
            {''.join(f'<span>{item}</span>' for item in items)}
        </div>
        """,
        unsafe_allow_html=True
    )


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

    # Create unified 3-tab layout with Settings first
    settings_tab, config_tab, results_tab = st.tabs(["🔧 Settings", "⚙️ Configuration", "📊 Results"])

    with settings_tab:
        _render_settings_status_bar()
        _show_settings_tab(wrapper, allocator, channel_info)

    with config_tab:
        _render_settings_status_bar()
        _show_configuration_tab(wrapper, allocator, channel_info)

    with results_tab:
        _render_settings_status_bar()
        _show_results_tab(wrapper, channel_info)


def _show_settings_tab(wrapper, allocator, channel_info):
    """Display the settings tab with all optimization parameters."""
    st.markdown("### Optimization Period")
    st.caption("These settings apply to ALL optimization modes.")

    col1, col2 = st.columns(2)

    with col1:
        st.slider(
            "Forecast Periods (weeks)",
            min_value=1,
            max_value=52,
            value=st.session_state.get("opt_num_periods", 8),
            key="opt_num_periods",
            help="Number of weeks to optimize for.",
        )

    with col2:
        month_names = [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"
        ]
        current_month = st.session_state.get("opt_start_month", 1)
        st.selectbox(
            "Starting Month",
            options=list(range(1, 13)),
            format_func=lambda x: month_names[x - 1],
            index=current_month - 1,
            help="Which month does your optimization period start?",
            key="opt_start_month",
        )

    st.markdown("---")

    # Channel bounds configuration
    _show_channel_bounds_expander(channel_info)

    st.markdown("---")

    # Seasonal adjustments
    _show_seasonality_expander(wrapper, channel_info)

    st.markdown("---")

    # Optimizer validation
    _show_validation_expander(wrapper)

    st.markdown("---")

    # Debug info
    with st.expander("Debug: Session State Values", expanded=False):
        st.write(f"opt_num_periods: {st.session_state.get('opt_num_periods', 'NOT SET')}")
        st.write(f"opt_start_month: {st.session_state.get('opt_start_month', 'NOT SET')}")
        st.write(f"bounds_config: {'SET' if st.session_state.get('bounds_config') else 'NOT SET'}")


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

    # Initialize parsed_scenarios for scenarios mode button enablement
    # Must happen BEFORE top button renders
    if mode == "scenarios" and "parsed_scenarios" not in st.session_state:
        st.session_state.parsed_scenarios = [50000, 100000, 150000, 200000, 250000]

    # Top run button (for quick access)
    _show_run_button(wrapper, allocator, channel_info, mode, position="top")

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

    # Bottom run button
    _show_run_button(wrapper, allocator, channel_info, mode, position="bottom")


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
        num_periods = st.session_state.get("opt_num_periods", 8)
        st.button(
            f"📥 Fill from last {num_periods} weeks",
            key="fill_budget_btn",
            on_click=_fill_budget_callback,
            help=f"Fill budget with actual spend from the last {num_periods} weeks (set in Settings)",
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

    # Build DataFrame for editable table
    committed_data = []
    for _, row in channel_info.iterrows():
        ch = row["channel"]
        display = row["display_name"]
        avg = row["avg_period_spend"]
        default_value = int(avg * num_periods)
        # Use session state value if available, otherwise use default
        current_val = st.session_state.base_allocation.get(ch, default_value)
        committed_data.append({
            "channel_key": ch,
            "Channel": display,
            "Default": default_value,
            "Committed": int(current_val),
        })

    committed_df = pd.DataFrame(committed_data)

    # Editable table for committed budget
    edited_df = st.data_editor(
        committed_df[["Channel", "Default", "Committed"]],
        column_config={
            "Channel": st.column_config.TextColumn("Channel", disabled=True),
            "Default": st.column_config.NumberColumn("Default ($)", disabled=True, format="$%d"),
            "Committed": st.column_config.NumberColumn("Committed ($)", min_value=0, format="$%d"),
        },
        hide_index=True,
        key="committed_budget_editor",
    )

    # Extract values back to base_allocation dict
    base_allocation = {}
    for i, row in committed_df.iterrows():
        ch = row["channel_key"]
        base_allocation[ch] = float(edited_df.iloc[i]["Committed"])

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


def _show_scenarios_mode_inputs():
    """Show inputs specific to scenarios mode."""
    st.markdown("### Scenario Settings")

    st.caption("Compare optimization results across multiple budget levels.")

    # Toggle for constraint comparison mode
    auto_constraint_analysis = st.toggle(
        "Run Constraint Comparison",
        value=st.session_state.get("auto_constraint_analysis", False),
        help="Automatically compare productivity curves at different constraint levels (Unconstrained, ±50%, ±30%, ±10%)",
        key="auto_constraint_analysis",
    )

    if auto_constraint_analysis:
        # Auto mode - show info about what will be generated
        st.info(
            "Will generate 15 budget scenarios from 50% to 300% of historical spend "
            "and compare curves at: **Unconstrained**, **±50%**, **±30%**, **±10%** bounds"
        )
        st.caption(
            "Constraints scale with budget: at 200% budget, ±10% means ±10% of the "
            "proportionally scaled channel spend, not ±10% of historical."
        )
        # Still need to set parsed_scenarios for button enablement
        st.session_state.parsed_scenarios = [1, 2]  # Dummy values to enable button
    else:
        # Manual mode - show budget input
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
        pass

    num_periods = st.session_state.get("opt_num_periods", 8)

    # Always use actual spend from last N weeks (matches historical response calculation)
    st.caption(f"Comparing against actual spend from the last {num_periods} weeks.")


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


def _show_channel_bounds_expander(channel_info):
    """Show the channel bounds configuration expander."""
    with st.expander("Channel Bounds", expanded=False):
        st.caption(
            "Constrain how much each channel can change from historical spend."
        )

        # Reset button
        if st.button("Reset to Defaults", key="reset_bounds", type="secondary"):
            # Clear bounds-related session state to recalculate from defaults
            for key in ["bounds_config", "bounds_mode", "max_delta_pct", "custom_bounds_editor"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

        bounds_mode = st.radio(
            "Bounds mode",
            ["No bounds", "Max % change", "Custom bounds"],
            index=1,  # Default to "Max % change"
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
                help="No channel can increase or decrease by more than this percentage from historical baseline",
            )

            keep_zero_channels_zero = st.checkbox(
                "Keep zero-spend channels at zero",
                value=True,
                key="keep_zero_channels_zero",
                help="Channels with $0 historical spend will not receive any budget allocation",
            )

            # Show percentage slider only when allowing zero-spend channels to receive budget
            if not keep_zero_channels_zero:
                zero_spend_max_pct = st.slider(
                    "Max % of budget for zero-spend channels",
                    min_value=1,
                    max_value=50,
                    value=10,
                    step=1,
                    key="zero_spend_max_pct",
                    help="Maximum percentage of total budget that can be allocated to each channel with no historical spend",
                )
            else:
                zero_spend_max_pct = 0

            # Always use actual spend from last N weeks (matches historical response calculation)
            from mmm_platform.optimization import BudgetAllocator
            try:
                allocator = BudgetAllocator(st.session_state.current_model)
                baseline_spend, _, _ = allocator.bridge.get_last_n_weeks_spend(num_periods)
            except Exception:
                # Fallback to average if bridge fails
                baseline_spend = {
                    row["channel"]: row["avg_period_spend"] * num_periods
                    for _, row in channel_info.iterrows()
                }

            # Get total budget for scaling
            total_budget = st.session_state.get("budget_value", 100000)

            # Calculate scale factor: how much bigger/smaller is budget vs historical
            total_historical = sum(baseline_spend.values())
            budget_scale = total_budget / total_historical if total_historical > 0 else 1.0

            # Calculate bounds from scaled baseline
            bounds_config = {}
            for _, row in channel_info.iterrows():
                ch = row["channel"]
                historical = baseline_spend.get(ch, 0)

                if historical > 0:
                    # Scale baseline to match target budget, then apply ±delta
                    scaled_val = historical * budget_scale
                    min_val = scaled_val * (1 - max_delta / 100)
                    max_val = scaled_val * (1 + max_delta / 100)
                else:
                    # Channel has no historical spend
                    min_val = 0
                    max_val = total_budget * (zero_spend_max_pct / 100)

                bounds_config[ch] = (max(0.0, min_val), max_val)

            st.session_state.bounds_config = bounds_config

            # Show preview table
            scale_note = f", scaled {budget_scale:.0%}" if abs(budget_scale - 1.0) > 0.01 else ""
            st.caption(f"Bounds based on last {num_periods} weeks{scale_note} (±{max_delta}%):")

            import pandas as pd
            preview_data = []
            for _, row in channel_info.iterrows():
                ch = row["channel"]
                display = row["display_name"]
                min_b, max_b = bounds_config[ch]
                historical = baseline_spend.get(ch, 0)
                scaled = historical * budget_scale
                note = "" if historical > 0 else "(no history, 10% cap)"
                preview_data.append({
                    "Channel": display,
                    "Historical": f"${historical:,.0f}",
                    "Scaled": f"${scaled:,.0f}",
                    "Min": f"${min_b:,.0f}",
                    "Max": f"${max_b:,.0f}",
                    "Note": note,
                })
            st.dataframe(pd.DataFrame(preview_data), hide_index=True, width="stretch")

            # Check if budget exceeds bounds capacity
            total_max = sum(bounds_config[ch][1] for ch in bounds_config)
            current_budget = st.session_state.get("budget_value", 100000)
            if current_budget > total_max:
                unallocated = current_budget - total_max
                st.warning(
                    f"**Budget exceeds bounds capacity:** ${current_budget:,.0f} budget > ${total_max:,.0f} max allocatable.\n\n"
                    f"${unallocated:,.0f} will be unallocated. Loosen bounds or reduce budget to allocate fully."
                )

        elif bounds_mode == "Custom bounds":
            st.caption(
                "Set minimum and maximum spend per channel. "
                "Use **Min** for pre-committed budgets. "
                "Use **Max** to cap investment."
            )

            # Build DataFrame for editing
            bounds_data = []
            for _, row in channel_info.iterrows():
                avg = row["avg_period_spend"]
                bounds_data.append({
                    "Channel": row["display_name"],
                    "channel_key": row["channel"],
                    "Baseline": int(avg * num_periods),
                    "Min": 0,
                    "Max": int(avg * num_periods * 3),
                })
            bounds_df = pd.DataFrame(bounds_data)

            # Editable table
            edited_df = st.data_editor(
                bounds_df[["Channel", "Baseline", "Min", "Max"]],
                column_config={
                    "Channel": st.column_config.TextColumn("Channel", disabled=True),
                    "Baseline": st.column_config.NumberColumn(
                        "Baseline ($)",
                        disabled=True,
                        format="$%d",
                        help="Historical average spend for the optimization period"
                    ),
                    "Min": st.column_config.NumberColumn(
                        "Min ($)",
                        min_value=0,
                        format="$%d",
                        help="Minimum spend (floor)"
                    ),
                    "Max": st.column_config.NumberColumn(
                        "Max ($)",
                        min_value=0,
                        format="$%d",
                        help="Maximum spend (ceiling)"
                    ),
                },
                hide_index=True,
                key="custom_bounds_editor",
            )

            # Convert edited table back to bounds_config dict
            bounds_config = {}
            for i, row in bounds_df.iterrows():
                ch = row["channel_key"]
                bounds_config[ch] = (float(edited_df.iloc[i]["Min"]), float(edited_df.iloc[i]["Max"]))

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

        # Reset button
        if st.button("Reset to Defaults", key="reset_seasonal", type="secondary"):
            # Clear seasonal-related session state to recalculate from model
            for key in ["seasonal_indices", "seasonal_editor"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

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
    """Show the optimizer calibration check expander."""
    with st.expander("Optimizer Calibration Check", expanded=False):
        st.caption(
            "Validates that the optimizer's saturation formula matches the model's "
            "channel contribution calculations. This checks internal consistency, "
            "not future prediction accuracy."
        )

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
                st.success("Excellent fit - optimizer formula accurately matches model")
            elif metrics['r2'] > 0.5 and metrics['mape'] < 25:
                st.info("Good fit - optimizer formula reasonably matches model")
            elif metrics['r2'] > 0.3:
                st.warning("Moderate fit - some discrepancy between optimizer and model")
            else:
                st.error("Poor fit - optimizer formula may diverge from model")

            # Time series chart
            fig_ts = go.Figure()
            fig_ts.add_trace(go.Scatter(
                x=backtest_df['date'],
                y=backtest_df['actual'],
                mode='lines+markers',
                name='Model Contribution',
                line=dict(color='blue'),
            ))
            fig_ts.add_trace(go.Scatter(
                x=backtest_df['date'],
                y=backtest_df['predicted'],
                mode='lines+markers',
                name='Optimizer Prediction',
                line=dict(color='orange'),
            ))
            fig_ts.update_layout(
                title="Calibration: Model vs Optimizer Formula",
                xaxis_title="Date",
                yaxis_title="Media Contribution",
                height=300,
            )
            st.plotly_chart(fig_ts, width="stretch")

            # Scatter plot
            fig_scatter = px.scatter(
                backtest_df,
                x='actual',
                y='predicted',
                title="Optimizer vs Model Media Contribution (ideal = diagonal)",
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


def _show_run_button(wrapper, allocator, channel_info, mode, position="bottom"):
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

    # Add divider for bottom button only
    if position == "bottom":
        st.markdown("---")

    run_clicked = st.button(
        button_labels[mode],
        type="primary",
        disabled=disabled,
        key=f"run_btn_{position}",
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

            allocator = BudgetAllocator(
                wrapper,
                num_periods=num_periods,
                utility="mean",
            )

            total_budget = st.session_state.get("budget_value", 100000)
            channel_bounds = st.session_state.get("bounds_config")
            seasonal_indices = st.session_state.get("seasonal_indices")
            compare_to_current = st.session_state.get("opt_compare_historical", False)

            # Always use actual spend from last N weeks (matches historical response calculation)
            comparison_mode = "last_n_weeks"

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

            # Run optimization (comparison_n_weeks uses num_periods for consistency)
            if efficiency_metric is not None and efficiency_target is not None:
                result = allocator.optimize_with_efficiency_floor(
                    total_budget=total_budget,
                    efficiency_metric=efficiency_metric,
                    efficiency_target=efficiency_target,
                    channel_bounds=channel_bounds,
                    seasonal_indices=seasonal_indices,
                    compare_to_current=compare_to_current,
                    comparison_mode=comparison_mode,
                    comparison_n_weeks=num_periods,
                )
            else:
                result = allocator.optimize(
                    total_budget=total_budget,
                    channel_bounds=channel_bounds,
                    compare_to_current=compare_to_current,
                    comparison_mode=comparison_mode,
                    comparison_n_weeks=num_periods,
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
                "utility": "mean",
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

            allocator = BudgetAllocator(
                wrapper,
                num_periods=num_periods,
                utility="mean",
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
                "utility": "mean",
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
            num_periods = st.session_state.get("opt_num_periods", 8)
            target_response = st.session_state.get("target_response", 500000)

            allocator = BudgetAllocator(wrapper, num_periods=num_periods, utility="mean")
            target_opt = TargetOptimizer(allocator)

            # Get bounds and seasonal settings
            channel_bounds = st.session_state.get("bounds_config")
            seasonal_indices = st.session_state.get("seasonal_indices")

            result = target_opt.find_budget_for_target(
                target_response=target_response,
                budget_range=(min_budget, max_budget),
                channel_bounds=channel_bounds,
                seasonal_indices=seasonal_indices,
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
                "has_bounds": channel_bounds is not None,
                "has_seasonal": seasonal_indices is not None,
            }

            st.success("Search complete! Switch to Results tab to view.")

        except Exception as e:
            st.error(f"Search failed: {e}")
            logger.exception("Target search error")


def _run_scenarios(wrapper):
    """Run the scenario analysis."""
    # Check if constraint comparison mode is enabled
    if st.session_state.get("auto_constraint_analysis"):
        return _run_constraint_comparison(wrapper)

    from mmm_platform.optimization import BudgetAllocator

    scenarios = st.session_state.get("parsed_scenarios", [])
    if len(scenarios) < 2:
        st.error("Need at least 2 budget scenarios to analyze")
        return

    with st.spinner(f"Analyzing {len(scenarios)} scenarios..."):
        try:
            num_periods = st.session_state.get("opt_num_periods", 8)

            allocator = BudgetAllocator(wrapper, num_periods=num_periods, utility="mean")

            # Get bounds and seasonal settings
            channel_bounds = st.session_state.get("bounds_config")
            seasonal_indices = st.session_state.get("seasonal_indices")

            result = allocator.scenario_analysis(
                scenarios,
                channel_bounds=channel_bounds,
                seasonal_indices=seasonal_indices,
            )

            # Store result
            st.session_state.optimization_result = result
            st.session_state.result_mode = "scenarios"

            # Store config for display
            st.session_state.optimizer_config = {
                "mode": "scenarios",
                "scenarios": scenarios,
                "num_periods": num_periods,
                "has_bounds": channel_bounds is not None,
                "has_seasonal": seasonal_indices is not None,
            }

            st.success("Analysis complete! Switch to Results tab to view.")

        except Exception as e:
            st.error(f"Analysis failed: {e}")
            logger.exception("Scenario analysis error")


def _run_constraint_comparison(wrapper):
    """Run scenario analysis at multiple constraint levels with properly scaled bounds."""
    from mmm_platform.optimization import BudgetAllocator
    from mmm_platform.optimization.results import ScenarioResult
    import numpy as np
    import pandas as pd

    num_periods = st.session_state.get("opt_num_periods", 8)

    with st.spinner("Running constraint comparison (this may take a few minutes)..."):
        try:
            allocator = BudgetAllocator(wrapper, num_periods=num_periods)

            # Get historical budget and response for time period
            historical_spend, _, _ = allocator.bridge.get_last_n_weeks_spend(num_periods)
            total_historical = sum(historical_spend.values())

            # Get historical response (actual contributions from this period)
            historical_response = allocator.bridge.get_contributions_for_period(
                num_periods, comparison_mode="last_n_weeks", n_weeks=num_periods
            )

            # Generate 15 budget points from 50% to 300%
            budget_scenarios = np.linspace(
                total_historical * 0.5,
                total_historical * 3.0,
                15
            ).tolist()

            # Define constraint levels
            constraint_levels = [
                ("Unconstrained", None),
                ("Mild (±50%)", 0.50),
                ("Standard (±30%)", 0.30),
                ("Tight (±10%)", 0.10),
            ]

            # Get seasonal indices if configured
            seasonal_indices = st.session_state.get("seasonal_indices")

            # Initialize results structure: {constraint_name: [results per budget]}
            results_by_constraint = {name: [] for name, _ in constraint_levels}

            progress_bar = st.progress(0)
            status_text = st.empty()

            total_runs = len(budget_scenarios) * len(constraint_levels)
            run_count = 0

            # Run optimizations: loop through budgets, then constraints
            # This ensures bounds are properly scaled for each budget level
            for budget in budget_scenarios:
                # Scale factor: how much to scale historical spend for this budget
                budget_scale = budget / total_historical

                for name, delta_pct in constraint_levels:
                    status_text.text(f"Budget ${budget:,.0f} - {name}...")

                    if delta_pct is None:
                        bounds = None
                    else:
                        # Scale bounds proportionally with budget
                        bounds = {}
                        for ch, val in historical_spend.items():
                            if val > 0:
                                # Scale the baseline spend, then apply ±delta
                                scaled_val = val * budget_scale
                                bounds[ch] = (
                                    max(0, scaled_val * (1 - delta_pct)),
                                    scaled_val * (1 + delta_pct)
                                )
                            else:
                                # Zero-spend channel: allow up to 10% of this budget
                                bounds[ch] = (0, budget * 0.10)

                    result = allocator.optimize(
                        total_budget=budget,
                        channel_bounds=bounds,
                        seasonal_indices=seasonal_indices,
                    )
                    results_by_constraint[name].append(result)

                    run_count += 1
                    progress_bar.progress(run_count / total_runs)

            progress_bar.empty()
            status_text.empty()

            # Build ScenarioResult objects for each constraint level
            results = {}
            for name, opt_results in results_by_constraint.items():
                # Build efficiency curve
                curve_data = []
                for i, result in enumerate(opt_results):
                    row = {
                        "budget": result.total_budget,
                        "expected_response": result.expected_response,
                        "response_ci_low": result.response_ci_low,
                        "response_ci_high": result.response_ci_high,
                        "success": result.success,
                    }

                    # Calculate marginal response
                    if i > 0:
                        prev = opt_results[i - 1]
                        budget_delta = result.total_budget - prev.total_budget
                        response_delta = result.expected_response - prev.expected_response
                        row["marginal_response"] = (
                            response_delta / budget_delta if budget_delta > 0 else 0
                        )
                    else:
                        row["marginal_response"] = (
                            result.expected_response / result.total_budget
                            if result.total_budget > 0 else 0
                        )

                    curve_data.append(row)

                efficiency_curve = pd.DataFrame(curve_data)

                results[name] = ScenarioResult(
                    budget_scenarios=budget_scenarios,
                    results=opt_results,
                    efficiency_curve=efficiency_curve,
                )

            # Store results
            st.session_state.constraint_comparison_result = results
            st.session_state.constraint_comparison_config = {
                "num_periods": num_periods,
                "historical_budget": total_historical,
                "historical_response": historical_response,
                "budget_scenarios": budget_scenarios,
                "historical_spend": historical_spend,
            }
            st.session_state.result_mode = "constraint_comparison"

            st.success("Constraint comparison complete! Switch to Results tab to view.")

        except Exception as e:
            st.error(f"Constraint comparison failed: {e}")
            logger.exception("Constraint comparison error")


def _show_results_tab(wrapper, channel_info):
    """Display the results tab with mode-specific results."""
    result_mode = st.session_state.get("result_mode")

    # Handle constraint comparison mode separately (has its own result storage)
    if result_mode == "constraint_comparison":
        results = st.session_state.get("constraint_comparison_result")
        config = st.session_state.get("constraint_comparison_config", {})
        if results:
            _show_constraint_comparison_results(results, config, wrapper)
        else:
            st.info("Configure optimization settings and click Run to see results.")
        return

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
            st.caption(f"Objective: {config.get('optimization_objective', 'Maximize Response')}")
            if config.get('compare_to_current'):
                st.caption(f"Comparison: {config.get('comparison_mode', 'average')}")

        elif mode == "incremental":
            c1, c2, c3 = st.columns(3)
            c1.metric("Committed", f"${config.get('committed_total', 0):,.0f}")
            c2.metric("Incremental", f"${config.get('incremental_budget', 0):,.0f}")
            c3.metric("Period", f"{config.get('num_periods', 8)} weeks")

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
        # Filter out technical optimizer messages that don't provide user value
        skip_messages = [
            "Positive directional derivative",
            "Constraint violation exceeds",
        ]
        if not any(msg in result.message for msg in skip_messages):
            st.warning(f"Optimization message: {result.message}")

    if getattr(result, 'used_fallback', False):
        st.info(
            "Used enhanced gradient-based optimization for better results. "
            "The default optimizer had convergence issues with this model."
        )

    # Always use expected response
    display_response = result.expected_response

    # Determine KPI type for display formatting
    kpi_type = getattr(wrapper.config.data, 'kpi_type', 'revenue')
    target_col = wrapper.config.data.target_column
    # Use kpi_display_name if configured, otherwise format the column name
    kpi_display_name = getattr(wrapper.config.data, 'kpi_display_name', None)
    kpi_label = kpi_display_name or target_col.replace('_', ' ').title()

    # Check if we have historical comparison data
    has_historical = (
        result.current_response is not None
        and result.current_response > 0
        and result.current_allocation is not None
    )

    if has_historical:
        # Calculate historical budget from current allocation
        historical_budget = sum(result.current_allocation.values())

        # Row 1: Budget - show comparison only if budgets differ significantly (>1%)
        budgets_differ = abs(result.total_budget - historical_budget) / historical_budget > 0.01 if historical_budget > 0 else False

        if budgets_differ:
            # Show comparison row
            budget_col1, budget_col2, budget_col3 = st.columns([2, 2, 1])
            with budget_col1:
                st.metric("Historical Spend", f"${historical_budget:,.0f}")
            with budget_col2:
                st.metric("Optimized Budget", f"${result.total_budget:,.0f}")
            with budget_col3:
                budget_delta_pct = ((result.total_budget - historical_budget) / historical_budget) * 100
                # Green if spending less, red if spending more
                color = "#28a745" if budget_delta_pct <= 0 else "#dc3545"
                sign = "+" if budget_delta_pct >= 0 else ""
                st.markdown(f"""
                    <div style="text-align: center; padding-top: 8px;">
                        <p style="margin: 0 0 4px 0; font-size: 14px; color: #666;">Budget Change</p>
                        <span style="background-color: {color}; color: white; padding: 6px 16px;
                                     border-radius: 16px; font-weight: bold; font-size: 16px;">
                            {sign}{budget_delta_pct:.1f}%
                        </span>
                    </div>
                """, unsafe_allow_html=True)
        else:
            # Same budget - show single metric
            st.metric("Total Budget", f"${result.total_budget:,.0f}")

        # Row 2: Response comparison (Historical | Expected | Uplift %)
        resp_col1, resp_col2, resp_col3 = st.columns([2, 2, 1])
        with resp_col1:
            if kpi_type == "count":
                hist_response_label = f"Historical {kpi_label}"
                hist_response_value = f"{result.current_response:,.0f}"
            else:
                hist_response_label = "Historical Response"
                hist_response_value = f"${result.current_response:,.0f}"
            st.metric(hist_response_label, hist_response_value)
        with resp_col2:
            if kpi_type == "count":
                exp_response_label = f"Expected {kpi_label}"
                exp_response_value = f"{display_response:,.0f}"
            else:
                exp_response_label = "Expected Response"
                exp_response_value = f"${display_response:,.0f}"
            st.metric(exp_response_label, exp_response_value)
        with resp_col3:
            # Show response uplift as colored pill (dynamically calculated based on confidence view)
            if result.current_response is not None and result.current_response > 0:
                response_uplift_pct = ((display_response - result.current_response) / result.current_response) * 100
                color = "#28a745" if response_uplift_pct >= 0 else "#dc3545"
                sign = "+" if response_uplift_pct >= 0 else ""
                st.markdown(f"""
                    <div style="text-align: center; padding-top: 8px;">
                        <p style="margin: 0 0 4px 0; font-size: 14px; color: #666;">Response Uplift</p>
                        <span style="background-color: {color}; color: white; padding: 6px 16px;
                                     border-radius: 16px; font-weight: bold; font-size: 16px;">
                            {sign}{response_uplift_pct:.1f}%
                        </span>
                    </div>
                """, unsafe_allow_html=True)

        # Row 3: Efficiency comparison (Historical ROI/CPA | Expected ROI/CPA | Change %)
        eff_col1, eff_col2, eff_col3 = st.columns([2, 2, 1])
        with eff_col1:
            if kpi_type == "count":
                hist_cpa = historical_budget / result.current_response
                st.metric("Historical CPA", f"${hist_cpa:,.2f}")
            else:
                hist_roi = result.current_response / historical_budget if historical_budget > 0 else 0
                st.metric("Historical ROI", f"${hist_roi:.2f}")
        with eff_col2:
            if kpi_type == "count":
                exp_cpa = result.total_budget / display_response if display_response > 0 else 0
                st.metric("Expected CPA", f"${exp_cpa:,.2f}")
            else:
                exp_roi = display_response / result.total_budget if result.total_budget > 0 else 0
                st.metric("Expected ROI", f"${exp_roi:.2f}")
        with eff_col3:
            # Calculate efficiency change percentage and display as colored pill
            if kpi_type == "count":
                # CPA: lower is better, so show improvement as positive when CPA decreases
                hist_cpa = historical_budget / result.current_response
                exp_cpa = result.total_budget / display_response if display_response > 0 else 0
                if hist_cpa > 0:
                    cpa_change_pct = ((hist_cpa - exp_cpa) / hist_cpa) * 100
                    color = "#28a745" if cpa_change_pct >= 0 else "#dc3545"
                    sign = "+" if cpa_change_pct >= 0 else ""
                    st.markdown(f"""
                        <div style="text-align: center; padding-top: 8px;">
                            <p style="margin: 0 0 4px 0; font-size: 14px; color: #666;">CPA Improvement</p>
                            <span style="background-color: {color}; color: white; padding: 6px 16px;
                                         border-radius: 16px; font-weight: bold; font-size: 16px;">
                                {sign}{cpa_change_pct:.1f}%
                            </span>
                        </div>
                    """, unsafe_allow_html=True)
            else:
                # ROI: higher is better
                hist_roi = result.current_response / historical_budget if historical_budget > 0 else 0
                exp_roi = display_response / result.total_budget if result.total_budget > 0 else 0
                if hist_roi > 0:
                    roi_change_pct = ((exp_roi - hist_roi) / hist_roi) * 100
                    color = "#28a745" if roi_change_pct >= 0 else "#dc3545"
                    sign = "+" if roi_change_pct >= 0 else ""
                    st.markdown(f"""
                        <div style="text-align: center; padding-top: 8px;">
                            <p style="margin: 0 0 4px 0; font-size: 14px; color: #666;">ROI Improvement</p>
                            <span style="background-color: {color}; color: white; padding: 6px 16px;
                                         border-radius: 16px; font-weight: bold; font-size: 16px;">
                                {sign}{roi_change_pct:.1f}%
                            </span>
                        </div>
                    """, unsafe_allow_html=True)
    else:
        # No historical comparison - use original simple layout
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        with metric_col1:
            st.metric("Total Budget", f"${result.total_budget:,.0f}")
        with metric_col2:
            if kpi_type == "count":
                response_label = f"Expected {kpi_label}"
                response_value = f"{display_response:,.0f}"
            else:
                response_label = "Expected Response"
                response_value = f"${display_response:,.0f}"
            # Calculate uplift dynamically based on confidence view
            uplift_delta = None
            if result.current_response is not None and result.current_response > 0:
                uplift_pct = ((display_response - result.current_response) / result.current_response) * 100
                sign = "+" if uplift_pct >= 0 else ""
                uplift_delta = f"{sign}{uplift_pct:.1f}%"
            st.metric(
                response_label,
                response_value,
                delta=uplift_delta,
            )
        with metric_col3:
            if kpi_type == "count":
                cpa = result.total_budget / display_response if display_response > 0 else 0
                st.metric("Expected CPA", f"${cpa:,.2f}")
            else:
                roi = display_response / result.total_budget if result.total_budget > 0 else 0
                st.metric("Expected ROI", f"${roi:.2f}")

    # Unallocated budget results (from bounds or efficiency floor)
    if result.unallocated_budget is not None and result.unallocated_budget > 0:
        if result.efficiency_metric is not None:
            # From efficiency floor mode
            st.warning(
                f"**Unallocated Budget:** ${result.unallocated_budget:,.0f}\n\n"
                f"To achieve {result.efficiency_metric.upper()} target of "
                f"{result.efficiency_target:.2f}, only ${result.total_budget:,.0f} can be efficiently deployed."
            )
        else:
            # From bounds constraints
            max_delta = st.session_state.get("max_delta_pct", 30)
            st.warning(
                f"**Unallocated Budget:** ${result.unallocated_budget:,.0f}\n\n"
                f"Channel bounds (±{max_delta}%) prevented full budget allocation. "
                f"Only ${result.total_budget:,.0f} allocated. Loosen bounds or reduce budget."
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
    optimal_total = sum(result.optimal_allocation.values())

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
    # Replace inf/NaN with 0 for display (sprintf can't handle special floats)
    display_df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Build column config for sortable, formatted display
    col_config = {
        "channel": st.column_config.TextColumn("Channel"),
        "optimal": st.column_config.NumberColumn("Optimal ($)", format="$%.0f"),
        "pct_of_total": st.column_config.NumberColumn("% of Total", format="%.1f"),
    }

    if "current" in display_df.columns:
        col_config["current"] = st.column_config.NumberColumn("Current ($)", format="$%.0f")
        col_config["delta"] = st.column_config.NumberColumn("Delta ($)", format="$%+.0f")
        col_config["pct_change"] = st.column_config.NumberColumn("% Change", format="%+.1f")

    st.dataframe(display_df, column_config=col_config, hide_index=True)


def _show_incremental_results(wrapper, result):
    """Show results for incremental budget mode."""
    if result.success:
        st.success(f"Optimization completed in {result.iterations} iterations")
    else:
        # Filter out technical optimizer messages that don't provide user value
        skip_messages = [
            "Positive directional derivative",
            "Constraint violation exceeds",
        ]
        if not any(msg in result.message for msg in skip_messages):
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

    # Allocation table with sortable columns
    # Replace inf/NaN with 0 for display (sprintf can't handle special floats)
    display_comparison_df = comparison_df.replace([np.inf, -np.inf], np.nan).fillna(0)
    st.dataframe(
        display_comparison_df,
        column_config={
            "Channel": st.column_config.TextColumn("Channel"),
            "Committed": st.column_config.NumberColumn("Committed ($)", format="$%.0f"),
            "Recommended": st.column_config.NumberColumn("Recommended ($)", format="$%.0f"),
            "Incremental": st.column_config.NumberColumn("Incremental ($)", format="$%+.0f"),
        },
        hide_index=True,
    )

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

    # Allocation breakdown
    st.markdown("#### Optimal Allocation")
    alloc_df = pd.DataFrame([
        {"Channel": ch, "Budget": amt}
        for ch, amt in result.optimal_allocation.items()
    ])
    # Replace inf/NaN with 0 for display (sprintf can't handle special floats)
    display_alloc_df = alloc_df.replace([np.inf, -np.inf], np.nan).fillna(0)
    st.dataframe(
        display_alloc_df,
        column_config={
            "Channel": st.column_config.TextColumn("Channel"),
            "Budget": st.column_config.NumberColumn("Budget ($)", format="$%.0f"),
        },
        hide_index=True,
    )

    # Show message
    st.info(result.message)


def _show_constraint_comparison_results(results, config, wrapper):
    """Display overlaid productivity curves for constraint comparison."""
    import plotly.graph_objects as go

    st.success(f"Compared 4 constraint levels across {len(config.get('budget_scenarios', []))} budget scenarios")

    # Config summary
    with st.expander("Configuration Summary", expanded=False):
        c1, c2, c3 = st.columns(3)
        c1.metric("Period", f"{config.get('num_periods', 8)} weeks")
        c2.metric("Historical Budget", f"${config.get('historical_budget', 0):,.0f}")
        c3.metric("Budget Range", f"50% - 300%")

    # Get KPI labels
    from mmm_platform.ui.kpi_labels import KPILabels
    labels = KPILabels(wrapper.config) if wrapper else None
    is_count_kpi = labels and not labels.is_revenue_type
    # Use target name for response label (e.g., "Expected Revenue" or "Expected Conversions")
    if labels:
        target_name = getattr(labels, 'target_name', None) or wrapper.config.data.target_column
        response_label = f"Expected {target_name}"
    else:
        response_label = "Expected Response"

    # Color scheme for constraint levels
    colors = {
        "Unconstrained": "#2ecc71",      # Green
        "Mild (±50%)": "#3498db",         # Blue
        "Standard (±30%)": "#f39c12",     # Orange
        "Tight (±10%)": "#e74c3c",        # Red
    }

    # Main productivity curve chart
    st.markdown("### Productivity Curves by Constraint Level")

    fig = go.Figure()

    for name, result in results.items():
        curve = result.efficiency_curve
        fig.add_trace(go.Scatter(
            x=curve["budget"],
            y=curve["expected_response"],
            mode="lines+markers",
            name=name,
            line=dict(color=colors.get(name, "#333"), width=2),
            marker=dict(size=6),
        ))

    # Add vertical line for historical budget
    historical_budget = config.get("historical_budget", 0)
    fig.add_vline(
        x=historical_budget,
        line_dash="dash",
        line_color="gray",
        annotation_text="Historical",
        annotation_position="top",
    )

    fig.update_layout(
        xaxis_title="Budget ($)",
        yaxis_title=response_label,
        legend_title="Constraint Level",
        hovermode="x unified",
        xaxis=dict(tickformat="$,.0f"),
        yaxis=dict(tickformat=",.0f" if is_count_kpi else "$,.0f"),
    )

    st.plotly_chart(fig)

    # Explanation of how constraints work
    with st.expander("How constraints work"):
        st.markdown("""
**Constraint levels represent flexibility around a proportional baseline:**

- **Unconstrained:** Optimizer can allocate freely across channels
- **Mild (±50%):** Each channel can vary up to 50% from its proportional share
- **Standard (±30%):** Each channel can vary up to 30% from its proportional share
- **Tight (±10%):** Each channel stays within 10% of its proportional share

*Example: At 200% of historical budget, a channel that was 30% of spend
can range from 27% to 33% of the new budget under ±10% constraints.*

**Why curves may differ:** Tighter constraints limit the optimizer's ability to
shift budget to high-performing channels, reducing overall efficiency.
        """)

    # Constraint impact summary
    st.markdown("### Constraint Impact Summary")
    _show_constraint_summary_table(results, config, is_count_kpi)

    # Detailed breakdown per constraint level
    with st.expander("Detailed Results by Constraint Level"):
        tabs = st.tabs(list(results.keys()))
        for tab, (name, result) in zip(tabs, results.items()):
            with tab:
                curve = result.efficiency_curve
                df = curve[["budget", "expected_response"]].copy()
                df.columns = ["Budget", response_label]
                df["Budget"] = df["Budget"].apply(lambda x: f"${x:,.0f}")
                if is_count_kpi:
                    df[response_label] = df[response_label].apply(lambda x: f"{x:,.0f}")
                else:
                    df[response_label] = df[response_label].apply(lambda x: f"${x:,.0f}")
                st.dataframe(df, hide_index=True)


def _show_constraint_summary_table(results, config, is_count_kpi=False):
    """Show summary table comparing constraint levels at historical budget."""
    import pandas as pd

    historical_budget = config.get("historical_budget", 0)
    budget_scenarios = config.get("budget_scenarios", [])

    # Find the scenario closest to historical budget
    if not budget_scenarios:
        st.warning("No budget scenarios to compare")
        return

    # Find index closest to historical (should be around index 7 for 15 points at 50%-150%)
    closest_idx = min(range(len(budget_scenarios)), key=lambda i: abs(budget_scenarios[i] - historical_budget))

    summary_data = []
    unconstrained_response = None
    historical_response = config.get("historical_response", 0)

    for name, result in results.items():
        curve = result.efficiency_curve
        if closest_idx < len(curve):
            row = curve.iloc[closest_idx]
            response = row["expected_response"]

            if name == "Unconstrained":
                unconstrained_response = response

            summary_data.append({
                "Constraint Level": name,
                "Response at Historical": response,
                "vs Actual": None,  # Will fill in later
                "vs Unconstrained": None,  # Will fill in later
            })

    # Calculate % difference vs actual and vs unconstrained
    for row in summary_data:
        # vs Actual
        if historical_response and historical_response > 0:
            diff_pct = (row["Response at Historical"] - historical_response) / historical_response * 100
            row["vs Actual"] = diff_pct

        # vs Unconstrained
        if unconstrained_response and unconstrained_response > 0:
            if row["Constraint Level"] != "Unconstrained":
                diff_pct = (row["Response at Historical"] - unconstrained_response) / unconstrained_response * 100
                row["vs Unconstrained"] = diff_pct
            else:
                row["vs Unconstrained"] = 0.0

    df = pd.DataFrame(summary_data)

    # Format for display
    if is_count_kpi:
        df["Response at Historical"] = df["Response at Historical"].apply(lambda x: f"{x:,.0f}")
    else:
        df["Response at Historical"] = df["Response at Historical"].apply(lambda x: f"${x:,.0f}")

    df["vs Actual"] = df["vs Actual"].apply(
        lambda x: f"{x:+.1f}%" if x is not None else "-"
    )
    df["vs Unconstrained"] = df["vs Unconstrained"].apply(
        lambda x: f"{x:+.1f}%" if x is not None else "-"
    )

    # Show historical context
    if is_count_kpi:
        st.caption(f"Historical: ${historical_budget:,.0f} spend → {historical_response:,.0f} {kpi_label}")
    else:
        st.caption(f"Historical: ${historical_budget:,.0f} spend → ${historical_response:,.0f} response")
    st.dataframe(df, hide_index=True)

    # Key insight
    if len(summary_data) >= 2:
        tight_row = next((r for r in summary_data if "±10%" in r["Constraint Level"]), None)
        if tight_row and unconstrained_response:
            loss_pct = (unconstrained_response - tight_row["Response at Historical"]) / unconstrained_response * 100
            if loss_pct > 0:
                st.info(f"Tight constraints (±10%) reduce potential response by {loss_pct:.1f}% compared to unconstrained optimization")


def _show_scenarios_results(result):
    """Show results for scenario analysis mode."""
    st.success(f"Analyzed {len(result.results)} scenarios")

    # Get KPI labels for dynamic labeling
    from mmm_platform.ui.kpi_labels import KPILabels
    wrapper = st.session_state.get("current_model")
    labels = KPILabels(wrapper.config) if wrapper else None
    marginal_label = labels.marginal_efficiency_label if labels else "Marginal ROI"
    efficiency_label = labels.efficiency_column_label if labels else "ROI"
    is_count_kpi = labels and not labels.is_revenue_type

    # For count KPIs, let user set target CPA as breakeven threshold
    target_cpa = None
    if is_count_kpi:
        kpi_name = labels.target_name or "Conversion"
        st.markdown(f"**Set your target {efficiency_label}** to define the breakeven threshold:")
        target_cpa = st.number_input(
            f"Target {efficiency_label} (maximum acceptable cost per {kpi_name})",
            min_value=0.01,
            value=15.0,
            step=1.0,
            format="%.2f",
            help=f"Your target cost per {kpi_name}. Breakeven = budget level where marginal cost exceeds this target.",
            key="scenario_target_cpa_input"
        )
        st.markdown("---")

    # Efficiency curve chart
    curve = result.efficiency_curve.copy()
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

    # Marginal efficiency chart (ROI or Cost Per)
    # For count KPIs, invert marginal_response to show cost-per (dollars per install)
    if is_count_kpi:
        # Handle division by zero: replace 0 with NaN, then invert, then replace inf with NaN
        curve["marginal_response"] = curve["marginal_response"].replace(0, np.nan)
        curve["marginal_response"] = 1 / curve["marginal_response"]
        curve["marginal_response"] = curve["marginal_response"].replace([np.inf, -np.inf], np.nan)

    fig_marginal = px.line(
        curve,
        x="budget",
        y="marginal_response",
        title=f"{marginal_label} by Budget Level",
        labels={"budget": "Budget ($)", "marginal_response": marginal_label},
        markers=True,
    )

    # Add breakeven line: ROI=1.0 for revenue KPIs, target CPA for count KPIs
    if is_count_kpi and target_cpa:
        fig_marginal.add_hline(
            y=target_cpa, line_dash="dash", line_color="red",
            annotation_text=f"Target (${target_cpa:.2f})"
        )
    elif not is_count_kpi:
        fig_marginal.add_hline(
            y=1.0, line_dash="dash", line_color="red",
            annotation_text="Breakeven ($1 ROI)"
        )
    st.plotly_chart(fig_marginal)

    # Summary table with ROI or CPA depending on KPI type
    summary_df = curve[["budget", "expected_response", "marginal_response"]].copy()

    # Add efficiency column (ROI for revenue, CPA for count)
    # Handle division by zero by replacing inf with NaN
    if is_count_kpi:
        summary_df["efficiency"] = curve["budget"] / curve["expected_response"].replace(0, np.nan)
        summary_df["efficiency"] = summary_df["efficiency"].replace([np.inf, -np.inf], np.nan)
        eff_col_name = "CPA"
    else:
        summary_df["efficiency"] = curve["expected_response"] / curve["budget"].replace(0, np.nan)
        summary_df["efficiency"] = summary_df["efficiency"].replace([np.inf, -np.inf], np.nan)
        eff_col_name = "ROI"

    # Keep numeric values, rename columns
    summary_df = summary_df[["budget", "expected_response", "efficiency", "marginal_response"]]
    response_label = labels.target_name if is_count_kpi else "Expected Response"
    summary_df.columns = ["Budget", response_label, eff_col_name, marginal_label]

    # Build column config based on KPI type
    if is_count_kpi:
        col_config = {
            "Budget": st.column_config.NumberColumn("Budget", format="$%.0f"),
            response_label: st.column_config.NumberColumn(response_label, format="%.0f"),
            eff_col_name: st.column_config.NumberColumn(eff_col_name, format="$%.2f"),
            marginal_label: st.column_config.NumberColumn(marginal_label, format="$%.2f"),
        }
    else:
        col_config = {
            "Budget": st.column_config.NumberColumn("Budget", format="$%.0f"),
            response_label: st.column_config.NumberColumn(response_label, format="$%.0f"),
            eff_col_name: st.column_config.NumberColumn(eff_col_name, format="%.2f"),
            marginal_label: st.column_config.NumberColumn(marginal_label, format="%.2f"),
        }

    # Replace NaN/inf with 0 for display (sprintf can't handle special floats)
    display_summary_df = summary_df.replace([np.inf, -np.inf], np.nan).fillna(0)
    st.dataframe(display_summary_df, column_config=col_config, hide_index=True)

    # Scenario drill-down details
    st.markdown("### Scenario Details")
    st.caption("Click to expand and see channel allocation for each budget level")

    for scenario_result in result.results:
        budget = scenario_result.total_budget
        response = scenario_result.expected_response

        # Format expander label based on KPI type
        if is_count_kpi:
            cpa = budget / response if response > 0 else 0
            response_label = labels.target_name or "Response"
            expander_label = f"${budget:,.0f} Budget → {response:,.0f} {response_label} (CPA: ${cpa:.2f})"
        else:
            roi = response / budget if budget > 0 else 0
            expander_label = f"${budget:,.0f} Budget → ${response:,.0f} Response (ROI: {roi:.2f})"

        with st.expander(expander_label):
            df = scenario_result.to_dataframe()

            # Build column config for sortable, formatted display
            col_config = {
                "channel": st.column_config.TextColumn("Channel"),
            }
            if "optimal" in df.columns:
                col_config["optimal"] = st.column_config.NumberColumn("Optimal ($)", format="$%.0f")
            if "current" in df.columns:
                col_config["current"] = st.column_config.NumberColumn("Current ($)", format="$%.0f")
            if "delta" in df.columns:
                col_config["delta"] = st.column_config.NumberColumn("Delta ($)", format="$%+.0f")
            if "pct_change" in df.columns:
                col_config["pct_change"] = st.column_config.NumberColumn("% Change", format="%.0f")
            if "pct_of_total" in df.columns:
                col_config["pct_of_total"] = st.column_config.NumberColumn("% of Total", format="%.1f")

            # Replace inf/NaN with 0 for display (sprintf can't handle special floats)
            display_df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
            st.dataframe(display_df, column_config=col_config, hide_index=True)

    # Download
    csv = result.to_dataframe().to_csv(index=False)
    st.download_button(
        "Download Scenario Results (CSV)",
        csv,
        "scenario_results.csv",
        "text/csv",
    )
