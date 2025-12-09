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
    )

    st.subheader("Allocate Budget Optimally")
    st.caption("Find the optimal allocation of your budget across channels.")

    # Get channel info
    try:
        allocator = BudgetAllocator(wrapper, num_periods=8)
        channels = allocator.channels
        channel_info = allocator.get_channel_info()
    except Exception as e:
        st.error(f"Error initializing optimizer: {e}")
        return

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

        # Channel bounds expander
        with st.expander("Channel Bounds", expanded=False):
            st.caption("Set min/max spend per channel (optional)")

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

                    # Run optimization
                    result = allocator.optimize(
                        total_budget=total_budget,
                        channel_bounds=channel_bounds,
                        compare_to_current=compare_to_current,
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
