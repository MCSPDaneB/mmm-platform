"""
Budget Optimization page for MMM Platform.

Provides interface for:
- Budget allocation optimization
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


def show():
    """Display the budget optimization page."""
    st.title("Budget Optimization")

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

    # Create tabs
    tab1, tab2, tab3 = st.tabs([
        "Optimize Budget",
        "Find Target",
        "Scenarios",
    ])

    with tab1:
        show_optimize_budget_tab(wrapper)

    with tab2:
        show_find_target_tab(wrapper)

    with tab3:
        show_scenarios_tab(wrapper)


def show_optimize_budget_tab(wrapper):
    """Show the main budget optimization tab."""
    from mmm_platform.optimization import (
        BudgetAllocator,
        build_bounds_from_constraints,
        UTILITY_FUNCTIONS,
        SeasonalIndexCalculator,
    )

    st.subheader("Allocate Budget Optimally")

    # Get channel info
    try:
        allocator = BudgetAllocator(wrapper, num_periods=8)
        channels = allocator.channels
        channel_info = allocator.get_channel_info()
    except Exception as e:
        st.error(f"Error initializing optimizer: {e}")
        return

    # Optimization mode selector
    optimization_mode = st.radio(
        "Optimization Mode",
        ["Full Budget", "Incremental Budget"],
        horizontal=True,
        help=(
            "**Full Budget**: Optimize total budget allocation from scratch.\n\n"
            "**Incremental Budget**: You have a committed budget plan, and extra money to invest. "
            "Find the optimal way to allocate the extra budget on top of your existing plan."
        ),
    )

    if optimization_mode == "Incremental Budget":
        show_incremental_budget_mode(wrapper, allocator, channel_info)
        return

    # Full budget mode (original UI)
    st.caption("Find the optimal allocation of your budget across channels.")

    # Configuration section
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### Configuration")

        # Total budget
        total_budget = st.number_input(
            "Total Budget ($)",
            min_value=1000,
            max_value=100000000,
            value=100000,
            step=10000,
            format="%d",
        )

        # Number of periods
        num_periods = st.slider(
            "Forecast Periods (weeks)",
            min_value=1,
            max_value=52,
            value=8,
        )

        # Optimization period selector
        st.markdown("**Optimization Period**")
        month_names = [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"
        ]
        start_month = st.selectbox(
            "Starting Month",
            options=list(range(1, 13)),
            format_func=lambda x: month_names[x - 1],
            index=0,
            help="Which month does your optimization period start? This affects seasonal adjustments.",
        )

        # Calculate end period info for display
        end_month_idx = (start_month - 1 + (num_periods // 4)) % 12
        period_info = f"{month_names[start_month - 1]}"
        if num_periods > 4:
            period_info += f" → {month_names[end_month_idx]}"
        st.caption(f"Period: {period_info} ({num_periods} weeks)")

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
        )

        # Efficiency target inputs (only show for ROI/CPA floor)
        efficiency_target = None
        efficiency_metric = None

        if optimization_objective == "ROI Floor":
            efficiency_metric = "roi"
            efficiency_target = st.number_input(
                "Minimum ROI",
                min_value=0.1,
                value=2.0,
                step=0.1,
                help="Minimum return on investment required (e.g., 2.0 = 2x return)",
            )
        elif optimization_objective == "CPA Floor":
            efficiency_metric = "cpa"
            efficiency_target = st.number_input(
                "Maximum CPA ($)",
                min_value=0.01,
                value=10.0,
                step=1.0,
                help="Maximum cost per acquisition allowed",
            )

        # Utility function
        utility_options = {
            "Mean (Risk Neutral)": "mean",
            "Value at Risk (Conservative)": "var",
            "Expected Shortfall (Very Conservative)": "cvar",
            "Sharpe Ratio (Risk-Adjusted)": "sharpe",
        }
        utility_label = st.selectbox(
            "Risk Profile",
            options=list(utility_options.keys()),
        )
        utility = utility_options[utility_label]

        # Compare to current toggle
        compare_to_current = st.checkbox(
            "Compare to historical spend",
            value=False,
        )

        # Comparison mode options (only show when compare_to_current is True)
        comparison_mode = "average"
        comparison_n_weeks = None

        if compare_to_current:
            # Get available date range for context
            try:
                min_date, max_date, total_periods = allocator.bridge.get_available_date_range()
                st.caption(
                    f"Data: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')} "
                    f"({total_periods} periods)"
                )
            except Exception:
                total_periods = 52

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
            )
            comparison_mode = comparison_options[comparison_label]

            # Show N weeks input if "Last N weeks" is selected
            if comparison_mode == "last_n_weeks":
                comparison_n_weeks = st.number_input(
                    "Number of weeks to use",
                    min_value=1,
                    max_value=total_periods,
                    value=min(52, total_periods),
                    step=1,
                    help=f"Look back this many weeks from the most recent date. Max available: {total_periods} weeks.",
                )

        # Channel bounds expander
        with st.expander("Channel Bounds", expanded=False):
            st.caption(
                "Set minimum and maximum spend per channel. "
                "Use **Min** for pre-committed budgets that can't be reduced. "
                "Use **Max** to cap investment in any single channel."
            )

            use_custom_bounds = st.checkbox("Use custom bounds", value=False)

            bounds_config = {}
            if use_custom_bounds:
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
                            key=f"min_{ch}",
                        )
                    with col_max:
                        max_val = st.number_input(
                            f"{display} Max",
                            min_value=0,
                            value=int(avg * num_periods * 3),
                            step=1000,
                            key=f"max_{ch}",
                        )
                    bounds_config[ch] = (float(min_val), float(max_val))

        # Seasonal indices expander
        with st.expander("Seasonal Adjustments", expanded=False):
            st.caption(
                "Seasonality affects both overall demand and channel effectiveness. "
                "These indices help the optimizer account for time-of-year variations."
            )

            try:
                seasonal_calc = SeasonalIndexCalculator(wrapper)

                # Get indices for selected period
                # Convert weeks to approximate months
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
                )

                if use_custom_indices:
                    st.caption(
                        "Edit the indices below. Values > 1.0 mean more effective, < 1.0 means less effective."
                    )

                    # Create editable inputs for each channel
                    edited_indices = {}
                    for ch, idx in seasonal_indices.items():
                        display_name = channel_info[channel_info["channel"] == ch]["display_name"].values
                        display_name = display_name[0] if len(display_name) > 0 else ch

                        edited_indices[ch] = st.number_input(
                            f"{display_name}",
                            min_value=0.1,
                            max_value=3.0,
                            value=float(idx),
                            step=0.05,
                            format="%.2f",
                            key=f"seasonal_{ch}",
                        )

                    # Use edited indices
                    st.session_state.seasonal_indices = edited_indices
                else:
                    # Display read-only table
                    indices_df["Index"] = indices_df["Index"].apply(lambda x: f"{x:.2f}")
                    st.dataframe(indices_df, width="stretch", hide_index=True)

                    # Store computed indices
                    st.session_state.seasonal_indices = seasonal_indices

                # Show full monthly/quarterly table
                with st.expander("View Full Seasonal Table"):
                    full_table = seasonal_calc.to_dataframe(
                        use_quarterly=confidence_info["using_quarterly"]
                    )
                    # Keep unformatted version for download
                    full_table_raw = full_table.copy()
                    # Format numbers for display
                    for col in full_table.columns:
                        if col != "Display Name":
                            full_table[col] = full_table[col].apply(lambda x: f"{x:.2f}")
                    st.dataframe(full_table, width="stretch")

                    # Download button for seasonal indices
                    csv = full_table_raw.to_csv(index=True)
                    st.download_button(
                        label="Download Seasonal Indices (CSV)",
                        data=csv,
                        file_name="seasonal_indices.csv",
                        mime="text/csv",
                    )

                # Upload custom seasonal indices
                uploaded_file = st.file_uploader(
                    "Upload custom seasonal indices (CSV)",
                    type=["csv"],
                    help="Upload a CSV with channel names and index values to override computed indices",
                    key="seasonal_upload",
                )

                if uploaded_file is not None:
                    try:
                        custom_df = pd.read_csv(uploaded_file, index_col=0)
                        # Parse uploaded values into seasonal_indices dict
                        updated_count = 0
                        for ch in seasonal_indices.keys():
                            if ch in custom_df.index:
                                # Use the first numeric column for the period index
                                for col in custom_df.columns:
                                    if col != "Display Name":
                                        try:
                                            seasonal_indices[ch] = float(custom_df.loc[ch, col])
                                            updated_count += 1
                                            break
                                        except (ValueError, TypeError):
                                            continue
                        if updated_count > 0:
                            st.success(f"Loaded custom indices for {updated_count} channels")
                            st.session_state.seasonal_indices = seasonal_indices
                        else:
                            st.warning("No matching channels found in uploaded file")
                    except Exception as e:
                        st.error(f"Error reading file: {e}")

            except Exception as e:
                import traceback
                error_msg = f"{type(e).__name__}: {str(e)}" if str(e) else type(e).__name__
                st.warning(f"Could not compute seasonal indices: {error_msg}")
                st.code(traceback.format_exc())  # Show full traceback for debugging
                st.session_state.seasonal_indices = None

        # Validation expander
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

                        # Store in session state for display
                        st.session_state.backtest_metrics = metrics
                        st.session_state.backtest_df = backtest_df

                    except Exception as e:
                        st.error(f"Validation failed: {e}")
                        logger.exception("Backtest validation error")

            # Display validation results if available
            if "backtest_metrics" in st.session_state:
                metrics = st.session_state.backtest_metrics
                backtest_df = st.session_state.backtest_df

                # Metrics row
                m1, m2, m3 = st.columns(3)
                m1.metric("R²", f"{metrics['r2']:.3f}")
                m2.metric("MAPE", f"{metrics['mape']:.1f}%")
                m3.metric("Correlation", f"{metrics['correlation']:.3f}")

                # Interpretation
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
                # Add perfect fit line
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

        # Run button
        optimize_clicked = st.button(
            "Optimize Budget",
            type="primary",
        )

    with col2:
        st.markdown("### Results")

        if optimize_clicked:
            with st.spinner("Running optimization..."):
                try:
                    # Create allocator with selected settings
                    allocator = BudgetAllocator(
                        wrapper,
                        num_periods=num_periods,
                        utility=utility,
                    )

                    # Set bounds
                    channel_bounds = bounds_config if use_custom_bounds and bounds_config else None

                    # Get seasonal indices from session state (computed in seasonal expander)
                    seasonal_indices = st.session_state.get("seasonal_indices")

                    # Run optimization based on objective
                    if efficiency_metric is not None and efficiency_target is not None:
                        # ROI/CPA floor mode
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
                        # Standard maximize response mode
                        result = allocator.optimize(
                            total_budget=total_budget,
                            channel_bounds=channel_bounds,
                            compare_to_current=compare_to_current,
                            comparison_mode=comparison_mode,
                            comparison_n_weeks=comparison_n_weeks,
                            seasonal_indices=seasonal_indices,
                        )

                    # Store result in session state
                    st.session_state.optimization_result = result

                except Exception as e:
                    st.error(f"Optimization failed: {e}")
                    logger.exception("Optimization error")
                    return

        # Display results
        if "optimization_result" in st.session_state:
            result = st.session_state.optimization_result

            if result.success:
                st.success(f"Optimization completed in {result.iterations} iterations")
            else:
                st.warning(f"Optimization message: {result.message}")

            # Show info if fallback optimizer was used
            if getattr(result, 'used_fallback', False):
                st.info(
                    "Used enhanced gradient-based optimization for better results. "
                    "The default optimizer had convergence issues with this model."
                )

            # Determine KPI type for display formatting
            kpi_type = getattr(wrapper.config.data, 'kpi_type', 'revenue')
            target_col = wrapper.config.data.target_column

            # Key metrics - format based on KPI type
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            with metric_col1:
                st.metric("Total Budget", f"${result.total_budget:,.0f}")
            with metric_col2:
                if kpi_type == "count":
                    # For count KPIs (installs, conversions, etc.) - show count without $
                    response_label = f"Expected {target_col.replace('_', ' ').title()}"
                    response_value = f"{result.expected_response:,.0f}"
                else:
                    # For revenue KPIs - show with $
                    response_label = "Expected Response"
                    response_value = f"${result.expected_response:,.0f}"
                st.metric(
                    response_label,
                    response_value,
                    delta=f"+{result.response_uplift_pct:.1f}%" if result.response_uplift_pct else None,
                )
            with metric_col3:
                if kpi_type == "count":
                    # For count KPIs - show CPA (Cost Per Acquisition)
                    cpa = result.total_budget / result.expected_response if result.expected_response > 0 else 0
                    st.metric("Expected CPA", f"${cpa:,.2f}")
                else:
                    # For revenue KPIs - show ROI
                    roi = result.expected_response / result.total_budget if result.total_budget > 0 else 0
                    st.metric("Expected ROI", f"{roi:.2f}x")

            # CI caption - format based on KPI type
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
                # Calculate percentage of total for both allocations
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
                # Sort by absolute difference to highlight biggest changes
                comparison_df = comparison_df.sort_values("Difference", key=abs, ascending=False)

                # Create grouped bar chart for percentage comparison
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

                # Difference chart (what changed)
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

            # Allocation table
            display_df = df.copy()
            display_df["optimal"] = display_df["optimal"].apply(lambda x: f"${x:,.0f}")
            display_df["pct_of_total"] = display_df["pct_of_total"].apply(lambda x: f"{x:.1f}%")

            if "current" in display_df.columns:
                display_df["current"] = display_df["current"].apply(lambda x: f"${x:,.0f}")
                display_df["delta"] = display_df["delta"].apply(lambda x: f"${x:+,.0f}")
                display_df["pct_change"] = display_df["pct_change"].apply(lambda x: f"{x:+.1f}%")

            st.dataframe(display_df, width="stretch", hide_index=True)

            # Download results
            csv = result.to_dataframe().to_csv(index=False)
            st.download_button(
                "Download Results (CSV)",
                csv,
                "optimization_results.csv",
                "text/csv",
            )


def show_incremental_budget_mode(wrapper, allocator, channel_info):
    """Show the incremental budget optimization UI."""
    from mmm_platform.optimization import BudgetAllocator

    st.caption(
        "You have a committed budget plan. Enter your current allocation below, "
        "then specify the extra budget to optimize."
    )

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### Configuration")

        # Number of periods
        num_periods = st.slider(
            "Forecast Periods (weeks)",
            min_value=1,
            max_value=52,
            value=8,
            key="incremental_num_periods",
        )

        # Incremental budget amount
        incremental_budget = st.number_input(
            "Incremental Budget ($)",
            min_value=1000,
            max_value=100000000,
            value=50000,
            step=5000,
            format="%d",
            help="The additional budget you want to allocate on top of your committed plan",
        )

        # Utility function
        utility_options = {
            "Mean (Risk Neutral)": "mean",
            "Value at Risk (Conservative)": "var",
            "Expected Shortfall (Very Conservative)": "cvar",
            "Sharpe Ratio (Risk-Adjusted)": "sharpe",
        }
        utility_label = st.selectbox(
            "Risk Profile",
            options=list(utility_options.keys()),
            key="incremental_utility",
        )
        utility = utility_options[utility_label]

        # Current allocation input
        st.markdown("### Committed Budget")
        st.caption(
            "Enter your current committed spend per channel. "
            "Pre-filled with average spend × periods."
        )

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
        st.metric("Committed Total", f"${committed_total:,.0f}")
        st.metric("+ Incremental", f"${incremental_budget:,.0f}")
        st.metric("**Total Budget**", f"${total_with_incremental:,.0f}")

        # Run button
        optimize_clicked = st.button(
            "Optimize Incremental Budget",
            type="primary",
            key="run_incremental",
        )

    with col2:
        st.markdown("### Results")

        if optimize_clicked:
            with st.spinner("Optimizing incremental budget..."):
                try:
                    # Create allocator with selected settings
                    allocator = BudgetAllocator(
                        wrapper,
                        num_periods=num_periods,
                        utility=utility,
                    )

                    # Run incremental optimization
                    result = allocator.optimize_incremental(
                        base_allocation=base_allocation,
                        incremental_budget=incremental_budget,
                    )

                    # Store result in session state
                    st.session_state.incremental_result = result

                except Exception as e:
                    st.error(f"Optimization failed: {e}")
                    logger.exception("Incremental optimization error")
                    return

        # Display results
        if "incremental_result" in st.session_state:
            result = st.session_state.incremental_result

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
                # Response uplift from incremental budget
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


def show_find_target_tab(wrapper):
    """Show the target-based optimization tab."""
    from mmm_platform.optimization import BudgetAllocator, TargetOptimizer

    st.subheader("Find Budget for Target")
    st.caption("Find the minimum budget needed to achieve a target response.")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### Configuration")

        # Target response
        target_response = st.number_input(
            "Target Response ($)",
            min_value=1000,
            max_value=100000000,
            value=500000,
            step=50000,
            format="%d",
            help="The revenue/conversions you want to achieve",
        )

        # Budget search range
        st.markdown("**Budget Search Range**")
        min_budget = st.number_input(
            "Minimum Budget ($)",
            min_value=1000,
            value=10000,
            step=5000,
        )
        max_budget = st.number_input(
            "Maximum Budget ($)",
            min_value=10000,
            value=1000000,
            step=50000,
        )

        # Number of periods
        num_periods = st.slider(
            "Forecast Periods (weeks)",
            min_value=1,
            max_value=52,
            value=8,
            key="target_num_periods",
        )

        # Search button
        search_clicked = st.button(
            "Find Required Budget",
            type="primary",
        )

    with col2:
        st.markdown("### Results")

        if search_clicked:
            if min_budget >= max_budget:
                st.error("Minimum budget must be less than maximum budget")
                return

            with st.spinner("Searching for optimal budget..."):
                try:
                    allocator = BudgetAllocator(wrapper, num_periods=num_periods)
                    target_opt = TargetOptimizer(allocator)

                    result = target_opt.find_budget_for_target(
                        target_response=target_response,
                        budget_range=(min_budget, max_budget),
                    )

                    st.session_state.target_result = result

                except Exception as e:
                    st.error(f"Search failed: {e}")
                    logger.exception("Target search error")
                    return

        if "target_result" in st.session_state:
            result = st.session_state.target_result

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


def show_scenarios_tab(wrapper):
    """Show the scenario analysis tab."""
    from mmm_platform.optimization import BudgetAllocator

    st.subheader("Scenario Analysis")
    st.caption("Compare optimization results across multiple budget levels.")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### Configuration")

        # Budget scenarios
        st.markdown("**Budget Scenarios**")
        scenario_input = st.text_area(
            "Enter budgets (one per line or comma-separated)",
            value="50000\n100000\n150000\n200000\n250000",
            height=150,
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
        except ValueError:
            st.error("Invalid budget values. Enter numbers only.")
            scenarios = []

        if scenarios:
            st.caption(f"Analyzing {len(scenarios)} scenarios: ${min(scenarios):,.0f} - ${max(scenarios):,.0f}")

        # Number of periods
        num_periods = st.slider(
            "Forecast Periods (weeks)",
            min_value=1,
            max_value=52,
            value=8,
            key="scenario_num_periods",
        )

        # Run button
        analyze_clicked = st.button(
            "Run Scenario Analysis",
            type="primary",
            disabled=len(scenarios) < 2,
        )

    with col2:
        st.markdown("### Results")

        if analyze_clicked and scenarios:
            with st.spinner(f"Analyzing {len(scenarios)} scenarios..."):
                try:
                    allocator = BudgetAllocator(wrapper, num_periods=num_periods)
                    scenario_result = allocator.scenario_analysis(scenarios)
                    st.session_state.scenario_result = scenario_result

                except Exception as e:
                    st.error(f"Analysis failed: {e}")
                    logger.exception("Scenario analysis error")
                    return

        if "scenario_result" in st.session_state:
            result = st.session_state.scenario_result

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
