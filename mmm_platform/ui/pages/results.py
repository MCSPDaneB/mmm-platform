"""
Results page for MMM Platform.
"""

import streamlit as st
import pandas as pd
import numpy as np
import datetime
import plotly.express as px
import plotly.graph_objects as go
import arviz as az

from mmm_platform.analysis.diagnostics import ModelDiagnostics
from mmm_platform.analysis.contributions import ContributionAnalyzer, CATEGORY_COLORS
from mmm_platform.analysis.reporting import ReportGenerator
from mmm_platform.analysis.bayesian_significance import (
    BayesianSignificanceAnalyzer,
    get_interpretation_guide,
)
from mmm_platform.analysis.marginal_roi import MarginalROIAnalyzer
from mmm_platform.analysis.executive_summary import ExecutiveSummaryGenerator
from mmm_platform.config.schema import DummyVariableConfig, SignConstraint


def get_contiguous_periods(df: pd.DataFrame, mask: pd.Series, date_col: str) -> list[tuple]:
    """Find contiguous date periods where mask is True.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the dates
    mask : pd.Series
        Boolean mask indicating which rows are in the period
    date_col : str
        Name of the date column

    Returns
    -------
    list[tuple]
        List of (start_date, end_date) tuples for each contiguous period
    """
    periods = []
    in_period = False
    start_date = None

    dates = df[date_col].values if date_col in df.columns else df.index

    for i, (is_active, date) in enumerate(zip(mask.values, dates)):
        if is_active and not in_period:
            # Start new period
            in_period = True
            start_date = date
        elif not is_active and in_period:
            # End current period
            periods.append((start_date, dates[i - 1]))
            in_period = False
            start_date = None

    # Close any open period at the end
    if in_period and start_date is not None:
        periods.append((start_date, dates[-1]))

    return periods


def add_residual_highlights(fig: go.Figure, time_df: pd.DataFrame, threshold_std: float = 1.5) -> go.Figure:
    """Add shaded regions to highlight periods with large residuals.

    Parameters
    ----------
    fig : go.Figure
        Plotly figure to add highlights to
    time_df : pd.DataFrame
        DataFrame with 'Date' and 'Residual' columns
    threshold_std : float
        Number of standard deviations for threshold

    Returns
    -------
    go.Figure
        Updated figure with highlights
    """
    residuals = time_df["Residual"]
    std_resid = residuals.std()

    # Find large positive residuals (underfitting)
    large_positive = residuals > threshold_std * std_resid
    # Find large negative residuals (overfitting)
    large_negative = residuals < -threshold_std * std_resid

    # Add highlights for underfitting periods (coral/salmon - visible on dark)
    for start, end in get_contiguous_periods(time_df, large_positive, "Date"):
        fig.add_vrect(
            x0=start, x1=end,
            fillcolor="rgba(255, 100, 100, 0.25)",
            layer="below",
            line_width=0,
        )

    # Add highlights for overfitting periods (cyan - visible on dark)
    for start, end in get_contiguous_periods(time_df, large_negative, "Date"):
        fig.add_vrect(
            x0=start, x1=end,
            fillcolor="rgba(100, 200, 255, 0.25)",
            layer="below",
            line_width=0,
        )

    return fig


def detect_suggested_dummies(
    time_df: pd.DataFrame,
    threshold_std: float
) -> list[dict]:
    """Detect outlier periods that could benefit from dummy variables.

    Parameters
    ----------
    time_df : pd.DataFrame
        DataFrame with 'Date' and 'Residual' columns
    threshold_std : float
        Number of standard deviations for threshold

    Returns
    -------
    list[dict]
        List of suggested dummies with start_date, end_date, avg_residual,
        num_weeks, suggested_sign, suggested_name
    """
    std_resid = time_df["Residual"].std()
    suggestions = []

    # Detect underfitting periods (positive residuals)
    underfit_mask = time_df["Residual"] > threshold_std * std_resid
    underfit_periods = get_contiguous_periods(time_df, underfit_mask, "Date")

    for start, end in underfit_periods:
        mask = (time_df["Date"] >= start) & (time_df["Date"] <= end)
        avg_res = time_df.loc[mask, "Residual"].mean()
        num_weeks = mask.sum()
        suggestions.append({
            "start_date": pd.to_datetime(start),
            "end_date": pd.to_datetime(end),
            "avg_residual": avg_res,
            "num_weeks": num_weeks,
            "suggested_sign": "positive",
            "suggested_name": f"dummy_{pd.to_datetime(start).strftime('%Y%m%d')}"
        })

    # Detect overfitting periods (negative residuals)
    overfit_mask = time_df["Residual"] < -threshold_std * std_resid
    overfit_periods = get_contiguous_periods(time_df, overfit_mask, "Date")

    for start, end in overfit_periods:
        mask = (time_df["Date"] >= start) & (time_df["Date"] <= end)
        avg_res = time_df.loc[mask, "Residual"].mean()
        num_weeks = mask.sum()
        suggestions.append({
            "start_date": pd.to_datetime(start),
            "end_date": pd.to_datetime(end),
            "avg_residual": avg_res,
            "num_weeks": num_weeks,
            "suggested_sign": "negative",
            "suggested_name": f"dummy_{pd.to_datetime(start).strftime('%Y%m%d')}"
        })

    # Sort by absolute residual magnitude (largest first)
    suggestions.sort(key=lambda x: abs(x["avg_residual"]), reverse=True)
    return suggestions


def show():
    """Show the results page."""
    st.title("Results")

    # Check for demo mode
    is_demo = st.session_state.get("demo_mode", False)

    if is_demo:
        show_demo_results()
        return

    # Check for EC2 results mode
    if st.session_state.get("ec2_mode") and st.session_state.get("ec2_results"):
        show_ec2_results()
        return

    # Check if model is fitted (non-demo mode)
    if not st.session_state.get("model_fitted") or st.session_state.get("current_model") is None:
        st.warning("Please run the model first, or load the demo from the Home page!")
        st.stop()

    wrapper = st.session_state.current_model
    config = wrapper.config

    # Create analyzers
    diagnostics = ModelDiagnostics.from_mmm_wrapper(wrapper)
    contributions = ContributionAnalyzer.from_mmm_wrapper(wrapper)
    marginal_analyzer = MarginalROIAnalyzer.from_mmm_wrapper(wrapper)
    exec_generator = ExecutiveSummaryGenerator(marginal_analyzer)

    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11 = st.tabs([
        "Overview",
        "Channel ROI",
        "Marginal ROI & Priority",
        "Executive Summary",
        "Bayesian Significance",
        "Diagnostics",
        "Time Series",
        "Export",
        "Visualizations",
        "Model Coefficients",
        "Media Curves"
    ])

    # =========================================================================
    # Tab 1: Overview
    # =========================================================================
    with tab1:
        st.subheader("Model Overview")

        # Key metrics
        fit_stats = wrapper.get_fit_statistics()

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("R²", f"{fit_stats['r2']:.2f}")
        with col2:
            st.metric("MAPE", f"{fit_stats['mape']:.2f}%")
        with col3:
            st.metric("RMSE", f"{fit_stats['rmse']:.2f}")
        with col4:
            st.metric("Fit Time", f"{fit_stats['fit_duration_seconds']:.2f}s")

        st.markdown("---")

        # Contribution breakdown
        st.subheader("Contribution Breakdown")

        breakdown = contributions.get_contribution_breakdown()

        # Pie chart
        breakdown_data = pd.DataFrame([
            {"Category": "Base", "Value": abs(breakdown["intercept"]["real_value"])},
            {"Category": "Channels", "Value": abs(breakdown["channels"]["real_value"])},
            {"Category": "Controls", "Value": abs(breakdown["controls"]["real_value"])},
            {"Category": "Seasonality", "Value": abs(breakdown["seasonality"]["real_value"])},
        ])

        fig = px.pie(
            breakdown_data,
            values="Value",
            names="Category",
            title="Contribution by Category"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Category column selector for grouping
        category_column_names = config.get_category_column_names()
        selected_category_col = None

        if category_column_names:
            selected_category_col = st.selectbox(
                "Group results by",
                options=["None"] + category_column_names,
                key="overview_category_grouping",
                help="Select a category column to group results"
            )
            if selected_category_col == "None":
                selected_category_col = None

        # Grouped contributions table (use category mappings from config)
        if selected_category_col:
            channel_categories = config.get_channel_category_map(selected_category_col)
            control_categories = config.get_control_category_map(selected_category_col)
        else:
            channel_categories = None
            control_categories = None

        grouped = contributions.get_grouped_contributions(
            channel_categories=channel_categories,
            control_categories=control_categories,
        )

        # Bar chart with category colors
        if len(grouped) > 0:
            fig_grouped = px.bar(
                grouped,
                x="group",
                y="pct_of_total",
                title="Contribution by Category (%)",
                labels={"group": "Category", "pct_of_total": "% of Total"},
                color="group",
                color_discrete_map=CATEGORY_COLORS,
            )
            fig_grouped.update_layout(showlegend=False, xaxis_tickangle=-45)
            st.plotly_chart(fig_grouped, use_container_width=True)

        st.dataframe(
            grouped[["group", "contribution_real", "pct_of_total"]].rename(columns={
                "group": "Group",
                "contribution_real": "Contribution ($)",
                "pct_of_total": "% of Total"
            }),
            use_container_width=True,
            hide_index=True,
        )

    # =========================================================================
    # Tab 2: Channel ROI
    # =========================================================================
    with tab2:
        st.subheader("Channel ROI Analysis")

        roi_df = contributions.get_channel_roi()

        if len(roi_df) > 0:
            # Category column selector for ROI grouping
            roi_category_column_names = config.get_category_column_names()
            roi_selected_category_col = None

            col1, col2 = st.columns([2, 3])
            with col1:
                # View toggle
                view_by = st.radio(
                    "View by",
                    ["Channel", "Category"],
                    horizontal=True,
                    key="roi_view_toggle"
                )
            with col2:
                if roi_category_column_names and view_by == "Category":
                    roi_selected_category_col = st.selectbox(
                        "Category column",
                        options=roi_category_column_names,
                        key="roi_category_column",
                        help="Select which category column to group by"
                    )

            # Add category column from config
            if roi_selected_category_col:
                channel_categories = config.get_channel_category_map(roi_selected_category_col)
            else:
                channel_categories = config.get_channel_category_map()
            roi_df["category"] = roi_df["channel"].apply(
                lambda ch: channel_categories.get(ch, "Paid Media")
            )

            if view_by == "Category":
                # Aggregate by category
                category_roi = roi_df.groupby("category").agg({
                    "spend_real": "sum",
                    "contribution_real": "sum",
                }).reset_index()
                category_roi["roi"] = category_roi["contribution_real"] / (category_roi["spend_real"] + 1e-9)
                category_roi = category_roi.sort_values("roi", ascending=False)

                fig = px.bar(
                    category_roi,
                    x="category",
                    y="roi",
                    title="ROI by Category",
                    labels={"category": "Category", "roi": "ROI"},
                    color="category",
                    color_discrete_map=CATEGORY_COLORS,
                )
                fig.update_layout(showlegend=False, xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)

                # Category table
                display_df = category_roi[["category", "spend_real", "contribution_real", "roi"]].copy()
                display_df.columns = ["Category", "Spend ($)", "Contribution ($)", "ROI"]
            else:
                # ROI bar chart by channel, colored by category
                fig = px.bar(
                    roi_df.sort_values("roi", ascending=False),
                    x="display_name",
                    y="roi",
                    title="ROI by Channel (colored by Category)",
                    labels={"display_name": "Channel", "roi": "ROI"},
                    color="category",
                    color_discrete_map=CATEGORY_COLORS,
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)

                # Channel table
                display_df = roi_df[["display_name", "category", "spend_real", "contribution_real", "roi"]].copy()
                display_df.columns = ["Channel", "Category", "Spend ($)", "Contribution ($)", "ROI"]

            # Format table
            st.subheader("Details")
            display_df["Spend ($)"] = display_df["Spend ($)"].apply(lambda x: f"${x:,.0f}")
            display_df["Contribution ($)"] = display_df["Contribution ($)"].apply(lambda x: f"${x:,.0f}")
            display_df["ROI"] = display_df["ROI"].apply(lambda x: f"{x:.2f}")

            st.dataframe(display_df, use_container_width=True, hide_index=True)

            # Spend vs Contribution scatter
            st.subheader("Spend vs Contribution")

            fig2 = px.scatter(
                roi_df,
                x="spend_real",
                y="contribution_real",
                text="display_name",
                title="Spend vs Contribution by Channel",
                labels={
                    "spend_real": "Total Spend ($)",
                    "contribution_real": "Total Contribution ($)"
                },
                color="category",
                color_discrete_map=CATEGORY_COLORS,
            )
            fig2.update_traces(textposition="top center")

            # Add break-even line
            max_val = max(roi_df["spend_real"].max(), roi_df["contribution_real"].max())
            fig2.add_trace(go.Scatter(
                x=[0, max_val],
                y=[0, max_val],
                mode="lines",
                name="Break-even (ROI=1)",
                line=dict(dash="dash", color="gray")
            ))

            st.plotly_chart(fig2, use_container_width=True)

    # =========================================================================
    # Tab 3: Marginal ROI & Investment Priority
    # =========================================================================
    with tab3:
        st.subheader("Marginal ROI & Investment Priority")

        st.markdown("""
        This analysis uses **saturation curve derivatives** to calculate the true marginal ROI
        (return on the *next* dollar spent), not just the average ROI.
        """)

        try:
            # Get priority table
            priority_df = marginal_analyzer.get_priority_table()

            # Key metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                n_increase = len(priority_df[priority_df['action'] == 'INCREASE'])
                st.metric("Channels to INCREASE", n_increase)
            with col2:
                n_hold = len(priority_df[priority_df['action'] == 'HOLD'])
                st.metric("Channels to HOLD", n_hold)
            with col3:
                n_reduce = len(priority_df[priority_df['action'] == 'REDUCE'])
                st.metric("Channels to REDUCE", n_reduce)

            st.markdown("---")

            # Priority table
            st.subheader("Investment Priority Table")

            display_df = priority_df.copy()
            display_df['current_spend'] = display_df['current_spend'].apply(lambda x: f"${x:,.0f}")
            display_df['current_roi'] = display_df['current_roi'].apply(lambda x: f"${x:.2f}")
            display_df['marginal_roi'] = display_df['marginal_roi'].apply(lambda x: f"${x:.2f}")
            display_df['breakeven_spend'] = display_df['breakeven_spend'].apply(
                lambda x: f"${x:,.0f}" if x is not None else "N/A"
            )
            display_df['headroom_amount'] = display_df['headroom_amount'].apply(lambda x: f"${x:,.0f}")

            styled_df = display_df[['channel', 'current_spend', 'current_roi', 'marginal_roi',
                                     'priority_rank', 'breakeven_spend', 'headroom_amount', 'action', 'needs_test']]
            styled_df.columns = ['Channel', 'Current Spend', 'Current ROI', 'Marginal ROI',
                                 'Priority', 'Breakeven Spend', 'Headroom', 'Action', 'Needs Test']

            st.dataframe(styled_df, use_container_width=True, hide_index=True)

            st.markdown("""
            **Key Concepts:**
            - **Marginal ROI**: Return on the *next* dollar spent (from saturation curve derivative)
            - **Breakeven Spend**: Spend level where marginal ROI = $1.00
            - **Headroom**: Additional spend available before hitting breakeven
            - **Needs Test**: High uncertainty in ROI estimate - validate with incrementality test
            """)

            # Visualizations
            st.markdown("---")
            st.subheader("Current vs Marginal ROI")

            result = marginal_analyzer.run_full_analysis()
            channel_names = [ch.channel_name for ch in result.channel_analysis]
            current_rois = [ch.current_roi for ch in result.channel_analysis]
            marginal_rois = [ch.marginal_roi for ch in result.channel_analysis]

            fig = go.Figure()
            fig.add_trace(go.Bar(
                name='Current (Avg) ROI',
                x=channel_names,
                y=current_rois,
                marker_color='steelblue'
            ))
            fig.add_trace(go.Bar(
                name='Marginal ROI',
                x=channel_names,
                y=marginal_rois,
                marker_color='orange'
            ))
            fig.add_hline(y=1, line_dash="dash", line_color="red", annotation_text="Breakeven")
            fig.update_layout(
                barmode='group',
                title="Current ROI vs Marginal ROI by Channel",
                xaxis_tickangle=-45
            )
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error running marginal ROI analysis: {str(e)}")
            st.info("Make sure the model has been properly fitted with posterior samples available.")

    # =========================================================================
    # Tab 4: Executive Summary
    # =========================================================================
    with tab4:
        st.subheader("Executive Summary")

        try:
            summary = exec_generator.get_summary_dict()

            # Portfolio Overview
            st.markdown("### Portfolio Overview")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Spend", f"${summary['portfolio']['total_spend']:,.0f}")
            with col2:
                st.metric("Total Contribution", f"${summary['portfolio']['total_contribution']:,.0f}")
            with col3:
                st.metric("Portfolio ROI", f"${summary['portfolio']['portfolio_roi']:.2f}")

            st.markdown("---")

            # Recommendations Summary
            st.markdown("### Investment Recommendations")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("INCREASE", summary['counts']['increase'], delta="High marginal ROI")
            with col2:
                st.metric("HOLD", summary['counts']['hold'], delta="Profitable")
            with col3:
                st.metric("REDUCE", summary['counts']['reduce'], delta="Below breakeven")
            with col4:
                st.metric("Need Validation", summary['counts']['needs_validation'], delta="Run tests")

            st.markdown("---")

            # Channel recommendations
            st.markdown("### Channel Recommendations")

            if summary['recommendations']['increase']:
                st.markdown("**INCREASE Investment:**")
                for ch in summary['recommendations']['increase']:
                    test_note = " *(validate with test)*" if ch['needs_test'] else ""
                    st.markdown(f"- **{ch['channel_name']}**: Marginal ROI ${ch['marginal_roi']:.2f}, "
                               f"Headroom ${ch['headroom_amount']:,.0f}{test_note}")

            if summary['recommendations']['hold']:
                st.markdown("**HOLD Steady:**")
                for ch in summary['recommendations']['hold']:
                    st.markdown(f"- **{ch['channel_name']}**: Marginal ROI ${ch['marginal_roi']:.2f}")

            if summary['recommendations']['reduce']:
                st.markdown("**REDUCE/Reallocate:**")
                for ch in summary['recommendations']['reduce']:
                    st.markdown(f"- **{ch['channel_name']}**: Marginal ROI ${ch['marginal_roi']:.2f} *(below breakeven)*")

            st.markdown("---")

            # Reallocation Opportunity
            st.markdown("### Reallocation Opportunity")
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "Funds from REDUCE channels",
                    f"${summary['portfolio']['reallocation_potential']:,.0f}"
                )
            with col2:
                st.metric(
                    "Headroom in INCREASE channels",
                    f"${summary['portfolio']['headroom_available']:,.0f}"
                )

            if summary['reallocation_moves']:
                st.markdown("**Top Reallocation Moves:**")
                for i, move in enumerate(summary['reallocation_moves'][:3]):
                    test_note = " *(validate first)*" if move['needs_validation'] else ""
                    st.markdown(f"{i+1}. Allocate **${move['amount']:,.0f}** to **{move['to_channel']}** "
                               f"(expected return: ${move['expected_return']:,.0f}){test_note}")

        except Exception as e:
            st.error(f"Error generating executive summary: {str(e)}")
            st.info("Make sure the model has been properly fitted with posterior samples available.")

    # =========================================================================
    # Tab 5: Bayesian Significance Analysis
    # =========================================================================
    with tab5:
        st.subheader("Bayesian Significance Analysis")

        # Get prior ROIs from config
        prior_rois = {}
        for channel_config in config.channels:
            prior_rois[channel_config.name] = channel_config.roi_prior_mid

        # Create analyzer
        try:
            sig_analyzer = BayesianSignificanceAnalyzer(
                idata=wrapper.idata,
                df_scaled=wrapper.df_scaled,
                channel_cols=config.get_channel_columns(),
                target_col=config.data.target_column,
                prior_rois=prior_rois,
            )

            # Run full analysis
            sig_report = sig_analyzer.run_full_analysis()

            # Subsections
            sig_tab1, sig_tab2, sig_tab3, sig_tab4, sig_tab5 = st.tabs([
                "Credible Intervals",
                "Probability of Direction",
                "ROPE Analysis",
                "ROI Posteriors",
                "Prior Sensitivity"
            ])

            # -----------------------------------------------------------------
            # Credible Intervals
            # -----------------------------------------------------------------
            with sig_tab1:
                st.markdown("### Beta Parameter Credible Intervals (95% HDI)")
                st.markdown("The HDI (Highest Density Interval) contains 95% of the posterior probability mass.")

                ci_data = []
                for ci in sig_report.credible_intervals:
                    ci_data.append({
                        "Channel": ci.channel,
                        "Mean": f"{ci.mean:.2f}",
                        "HDI Low": f"{ci.hdi_low:.2f}",
                        "HDI High": f"{ci.hdi_high:.2f}",
                        "Excludes Zero": "Yes" if ci.excludes_zero else "No"
                    })

                ci_df = pd.DataFrame(ci_data)
                st.dataframe(ci_df, use_container_width=True, hide_index=True)

                # Visualization
                st.markdown("---")
                channels = [ci.channel for ci in sig_report.credible_intervals]
                means = [ci.mean for ci in sig_report.credible_intervals]
                lows = [ci.hdi_low for ci in sig_report.credible_intervals]
                highs = [ci.hdi_high for ci in sig_report.credible_intervals]

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=means,
                    y=channels,
                    mode='markers',
                    name='Mean',
                    marker=dict(size=10, color='steelblue')
                ))
                for i, ch in enumerate(channels):
                    fig.add_shape(
                        type="line",
                        x0=lows[i], x1=highs[i],
                        y0=ch, y1=ch,
                        line=dict(color="steelblue", width=3)
                    )
                fig.add_vline(x=0, line_dash="dash", line_color="red")
                fig.update_layout(
                    title="Beta Posteriors with 95% HDI",
                    xaxis_title="Beta (saturation_beta)",
                    yaxis_title="Channel",
                    height=max(400, len(channels) * 40)
                )
                st.plotly_chart(fig, use_container_width=True)

            # -----------------------------------------------------------------
            # Probability of Direction
            # -----------------------------------------------------------------
            with sig_tab2:
                st.markdown("### Probability of Direction (pd)")
                st.markdown("The probability that the effect is positive. Higher values indicate stronger evidence.")

                pd_data = []
                for pd_result in sig_report.probability_of_direction:
                    pd_data.append({
                        "Channel": pd_result.channel,
                        "Probability": f"{pd_result.pd:.1%}",
                        "Interpretation": pd_result.interpretation
                    })

                pd_df = pd.DataFrame(pd_data)
                st.dataframe(pd_df, use_container_width=True, hide_index=True)

                # Visualization
                st.markdown("---")
                channels = [pd_result.channel for pd_result in sig_report.probability_of_direction]
                pds = [pd_result.pd for pd_result in sig_report.probability_of_direction]
                colors = ['green' if pd >= 0.95 else 'orange' if pd >= 0.75 else 'red' for pd in pds]

                fig = go.Figure()
                fig.add_trace(go.Bar(
                    y=channels,
                    x=pds,
                    orientation='h',
                    marker_color=colors
                ))
                fig.add_vline(x=0.95, line_dash="dash", line_color="green", annotation_text="95%")
                fig.add_vline(x=0.75, line_dash="dash", line_color="orange", annotation_text="75%")
                fig.update_layout(
                    title="Probability of Direction (Positive Effect)",
                    xaxis_title="Probability",
                    yaxis_title="Channel",
                    xaxis=dict(range=[0.5, 1.0]),
                    height=max(400, len(channels) * 40)
                )
                st.plotly_chart(fig, use_container_width=True)

            # -----------------------------------------------------------------
            # ROPE Analysis
            # -----------------------------------------------------------------
            with sig_tab3:
                st.markdown("### ROPE Analysis (Region of Practical Equivalence)")
                st.markdown(f"Effects between {sig_analyzer.rope_low} and {sig_analyzer.rope_high} are considered 'practically zero'.")

                rope_data = []
                for rope in sig_report.rope_analysis:
                    rope_data.append({
                        "Channel": rope.channel,
                        "% in ROPE": f"{rope.pct_in_rope:.1%}",
                        "% Below ROPE": f"{rope.pct_below_rope:.1%}",
                        "% Above ROPE": f"{rope.pct_above_rope:.1%}",
                        "Conclusion": rope.conclusion
                    })

                rope_df = pd.DataFrame(rope_data)
                st.dataframe(rope_df, use_container_width=True, hide_index=True)

            # -----------------------------------------------------------------
            # ROI Posteriors
            # -----------------------------------------------------------------
            with sig_tab4:
                st.markdown("### ROI Credible Intervals")
                st.markdown("Full posterior distribution of ROI per channel with uncertainty.")

                roi_data = []
                for roi in sig_report.roi_posteriors:
                    roi_data.append({
                        "Channel": roi.channel,
                        "ROI Mean": f"{roi.roi_mean:.2f}",
                        "ROI Median": f"{roi.roi_median:.2f}",
                        "ROI 5%": f"{roi.roi_5pct:.2f}",
                        "ROI 95%": f"{roi.roi_95pct:.2f}",
                        "Significant": "Yes" if roi.significant else "No"
                    })

                roi_df = pd.DataFrame(roi_data)
                st.dataframe(roi_df, use_container_width=True, hide_index=True)

                # Visualization
                st.markdown("---")
                channels = [roi.channel for roi in sig_report.roi_posteriors]
                roi_means = [roi.roi_mean for roi in sig_report.roi_posteriors]
                roi_5s = [roi.roi_5pct for roi in sig_report.roi_posteriors]
                roi_95s = [roi.roi_95pct for roi in sig_report.roi_posteriors]
                colors = ['steelblue' if roi.significant else 'lightcoral' for roi in sig_report.roi_posteriors]

                fig = go.Figure()
                fig.add_trace(go.Bar(
                    y=channels,
                    x=roi_means,
                    orientation='h',
                    marker_color=colors,
                    error_x=dict(
                        type='data',
                        symmetric=False,
                        array=[roi_95s[i] - roi_means[i] for i in range(len(channels))],
                        arrayminus=[roi_means[i] - roi_5s[i] for i in range(len(channels))]
                    )
                ))
                fig.add_vline(x=0, line_dash="dash", line_color="red")
                fig.add_vline(x=1, line_dash="dash", line_color="green", annotation_text="ROI=1")
                fig.update_layout(
                    title="ROI Posteriors with 90% CI",
                    xaxis_title="ROI",
                    yaxis_title="Channel",
                    height=max(400, len(channels) * 40)
                )
                st.plotly_chart(fig, use_container_width=True)

            # -----------------------------------------------------------------
            # Prior Sensitivity
            # -----------------------------------------------------------------
            with sig_tab5:
                st.markdown("### Prior vs Posterior Sensitivity")
                st.markdown("How much did the posterior move from the prior? Large shifts indicate strong data influence.")

                sens_data = []
                for sens in sig_report.prior_sensitivity:
                    sens_data.append({
                        "Channel": sens.channel,
                        "Prior ROI": f"{sens.prior_roi:.2f}",
                        "Posterior ROI": f"{sens.posterior_roi:.2f}",
                        "Shift": f"{sens.shift:+.2f}",
                        "Data Influence": sens.data_influence
                    })

                sens_df = pd.DataFrame(sens_data)
                st.dataframe(sens_df, use_container_width=True, hide_index=True)

                # Visualization
                st.markdown("---")
                prior_rois_plot = [sens.prior_roi for sens in sig_report.prior_sensitivity]
                post_rois_plot = [sens.posterior_roi for sens in sig_report.prior_sensitivity]
                influence_colors = []
                for sens in sig_report.prior_sensitivity:
                    if sens.data_influence == "Strong":
                        influence_colors.append('green')
                    elif sens.data_influence == "Moderate":
                        influence_colors.append('orange')
                    else:
                        influence_colors.append('red')

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=prior_rois_plot,
                    y=post_rois_plot,
                    mode='markers+text',
                    text=[sens.channel.replace("_spend", "")[:15] for sens in sig_report.prior_sensitivity],
                    textposition="top center",
                    marker=dict(size=12, color=influence_colors)
                ))
                # Add diagonal line (no change)
                max_val = max(max(prior_rois_plot), max(post_rois_plot)) * 1.1
                min_val = min(min(prior_rois_plot) if min(prior_rois_plot) > 0 else 0,
                              min(post_rois_plot) if min(post_rois_plot) > 0 else 0)
                fig.add_trace(go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    name='No change',
                    line=dict(dash='dash', color='gray')
                ))
                fig.update_layout(
                    title="Prior vs Posterior ROI",
                    xaxis_title="Prior ROI",
                    yaxis_title="Posterior ROI",
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)

            # Interpretation Guide
            st.markdown("---")
            with st.expander("Interpretation Guide"):
                st.code(get_interpretation_guide(), language=None)

        except Exception as e:
            st.error(f"Error running Bayesian significance analysis: {str(e)}")
            st.info("Make sure the model has been properly fitted with posterior samples available.")

    # =========================================================================
    # Tab 6: Diagnostics
    # =========================================================================
    with tab6:
        st.subheader("Model Diagnostics")

        # Run diagnostics
        diag_report = diagnostics.run_all_diagnostics()

        # Overall status
        if diag_report.overall_passed:
            st.success("All diagnostics passed!")
        else:
            st.warning("Some diagnostics need attention")

        # Individual results
        for result in diag_report.results:
            icon = "✅" if result.passed else "⚠️"
            st.write(f"{icon} **{result.name}**: {result.message}")

        # Recommendations
        if diag_report.recommendations:
            st.markdown("---")
            st.subheader("Recommendations")
            for rec in diag_report.recommendations:
                st.info(rec)

        # Worst predictions
        st.markdown("---")
        st.subheader("Worst Predictions")

        worst = diagnostics.get_worst_predictions(10)
        worst["date"] = worst["date"].dt.strftime("%Y-%m-%d")
        st.dataframe(worst, use_container_width=True, hide_index=True)

        # Control variable analysis
        st.markdown("---")
        st.subheader("Control Variable Analysis")

        control_df = contributions.get_control_contributions()
        if len(control_df) > 0:
            # Add emoji for sign validation
            control_df["Status"] = control_df["sign_valid"].apply(
                lambda x: "✅" if x else "⚠️"
            )
            # Use display_name if available, otherwise fall back to control
            display_col = "display_name" if "display_name" in control_df.columns else "control"
            columns_to_show = [display_col, "contribution_real", "expected_sign", "actual_sign", "Status"]
            display_ctrl_df = control_df[columns_to_show].copy()
            display_ctrl_df.columns = ["Control", "Contribution ($)", "Expected Sign", "Actual Sign", "Status"]
            st.dataframe(
                display_ctrl_df,
                use_container_width=True,
                hide_index=True
            )

    # =========================================================================
    # Tab 7: Time Series
    # =========================================================================
    with tab7:
        st.subheader("Time Series Analysis")

        # Actual vs Fitted
        contribs = wrapper.get_contributions()
        target_col = config.data.target_column

        df_indexed = wrapper.df_scaled.set_index(config.data.date_column)
        actual = df_indexed[target_col].reindex(contribs.index) * config.data.revenue_scale

        component_cols = [c for c in contribs.columns if c != target_col]
        fitted = contribs[component_cols].sum(axis=1) * config.data.revenue_scale

        time_df = pd.DataFrame({
            "Date": contribs.index,
            "Actual": actual.values,
            "Fitted": fitted.values,
        })

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=time_df["Date"],
            y=time_df["Actual"],
            name="Actual",
            line=dict(color="white", width=2)
        ))
        fig.add_trace(go.Scatter(
            x=time_df["Date"],
            y=time_df["Fitted"],
            name="Fitted",
            line=dict(color="orange", width=2)
        ))
        fig.update_layout(
            title="Actual vs Fitted",
            xaxis_title="Date",
            yaxis_title="Revenue ($)",
            template="plotly_dark",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Residuals
        st.subheader("Residuals")

        time_df["Residual"] = time_df["Actual"] - time_df["Fitted"]

        # Add threshold slider for highlighting
        highlight_threshold = st.slider(
            "Highlight threshold (std deviations)",
            min_value=0.5,
            max_value=3.0,
            value=1.5,
            step=0.25,
            help="Periods with residuals exceeding this many standard deviations will be highlighted",
            key="residual_highlight_threshold"
        )

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=time_df["Date"],
            y=time_df["Residual"],
            mode="lines+markers",  # Add markers for selection
            name="Residual",
            line=dict(color="steelblue"),
            marker=dict(size=6, color="steelblue")
        ))
        fig2.add_hline(y=0, line_dash="dash", line_color="lightgray")

        # Add threshold lines (using lighter colors for dark background)
        std_resid = time_df["Residual"].std()
        fig2.add_hline(
            y=highlight_threshold * std_resid,
            line_dash="dot",
            line_color="salmon",
            annotation_text="Underfitting threshold",
            annotation=dict(font_color="salmon")
        )
        fig2.add_hline(
            y=-highlight_threshold * std_resid,
            line_dash="dot",
            line_color="cyan",
            annotation_text="Overfitting threshold",
            annotation=dict(font_color="cyan")
        )

        # Add shaded regions for large residuals
        fig2 = add_residual_highlights(fig2, time_df, highlight_threshold)

        fig2.update_layout(
            title="Residuals Over Time (Salmon = Underfitting, Cyan = Overfitting)",
            xaxis_title="Date",
            yaxis_title="Residual ($)",
            template="plotly_dark",
        )
        st.plotly_chart(fig2, use_container_width=True)

        # =====================================================================
        # Suggested Dummy Variables Section
        # =====================================================================
        st.markdown("---")
        st.subheader("Suggested Dummy Variables")

        # Initialize pending_dummies if not exists
        if "pending_dummies" not in st.session_state:
            st.session_state.pending_dummies = []

        suggestions = detect_suggested_dummies(time_df, highlight_threshold)

        if not suggestions:
            st.info(f"No outlier periods detected at current threshold (±{highlight_threshold}σ).")
        else:
            # Sort options
            col_sort1, col_sort2 = st.columns([2, 3])
            with col_sort1:
                sort_by = st.selectbox(
                    "Sort by",
                    options=["Magnitude (largest first)", "Date (earliest first)", "Date (latest first)"],
                    key="dummy_sort_by"
                )

            # Apply sorting
            if sort_by == "Magnitude (largest first)":
                suggestions.sort(key=lambda x: abs(x["avg_residual"]), reverse=True)
            elif sort_by == "Date (earliest first)":
                suggestions.sort(key=lambda x: x["start_date"])
            else:  # Date (latest first)
                suggestions.sort(key=lambda x: x["start_date"], reverse=True)

            st.write(f"Found **{len(suggestions)}** period(s) exceeding ±{highlight_threshold}σ threshold:")

            # Initialize selection state
            if "dummy_selections" not in st.session_state:
                st.session_state.dummy_selections = {}

            # Display checkbox list
            for i, sug in enumerate(suggestions):
                key = f"sug_{sug['start_date'].strftime('%Y%m%d')}_{sug['end_date'].strftime('%Y%m%d')}"

                # Format display text
                sign_emoji = "📈" if sug["suggested_sign"] == "positive" else "📉"
                duration = f"{sug['num_weeks']} week{'s' if sug['num_weeks'] > 1 else ''}"
                residual_fmt = f"${sug['avg_residual']:,.0f}"

                label = f"{sign_emoji} {sug['start_date'].strftime('%Y-%m-%d')} to {sug['end_date'].strftime('%Y-%m-%d')} ({duration}, avg residual: {residual_fmt})"

                st.session_state.dummy_selections[key] = st.checkbox(
                    label,
                    value=st.session_state.dummy_selections.get(key, False),
                    key=f"checkbox_{key}"
                )

            # Add Selected button
            if st.button("Add Selected to Pending", key="add_suggested_dummies"):
                added_count = 0
                for sug in suggestions:
                    key = f"sug_{sug['start_date'].strftime('%Y%m%d')}_{sug['end_date'].strftime('%Y%m%d')}"
                    if st.session_state.dummy_selections.get(key, False):
                        # Check not already in pending
                        existing_names = [d.name for d in st.session_state.pending_dummies]
                        if sug["suggested_name"] not in existing_names:
                            dummy_config = DummyVariableConfig(
                                name=sug["suggested_name"],
                                start_date=sug["start_date"].strftime("%Y-%m-%d"),
                                end_date=sug["end_date"].strftime("%Y-%m-%d"),
                                sign_constraint=SignConstraint(sug["suggested_sign"])
                            )
                            st.session_state.pending_dummies.append(dummy_config)
                            added_count += 1
                            # Clear checkbox
                            st.session_state.dummy_selections[key] = False

                if added_count > 0:
                    st.success(f"Added {added_count} dummy variable(s) to pending list")
                    st.rerun()
                else:
                    st.warning("No suggestions selected, or all selected items already exist in pending list.")

        # =====================================================================
        # Create Custom Dummy Variable Section
        # =====================================================================
        st.markdown("---")
        st.subheader("Create Custom Dummy Variable")
        st.caption("Manually select a date range to create a dummy variable.")

        min_date = pd.to_datetime(time_df["Date"].min()).date()
        max_date = pd.to_datetime(time_df["Date"].max()).date()

        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=min_date,
                min_value=min_date,
                max_value=max_date,
                key="dummy_start_date"
            )
        with col2:
            num_weeks = st.number_input(
                "Duration (weeks)",
                min_value=1,
                max_value=52,
                value=1,
                key="dummy_num_weeks",
                help="Number of weeks for this dummy variable (1 = single week)"
            )

        # Calculate end date based on start date and number of weeks
        end_date = start_date + datetime.timedelta(days=(num_weeks - 1) * 7)

        # Show the date range
        if num_weeks == 1:
            st.caption(f"Dummy covers: **{start_date}** (single week)")
        else:
            st.caption(f"Dummy covers: **{start_date}** to **{end_date}** ({num_weeks} weeks)")

        # Calculate stats from date picker selection
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        mask = (pd.to_datetime(time_df["Date"]) >= start_dt) & (pd.to_datetime(time_df["Date"]) <= end_dt)
        selected_residuals = time_df.loc[mask, "Residual"]

        if len(selected_residuals) > 0:
            avg_residual = selected_residuals.mean()
            n_periods = len(selected_residuals)
        else:
            avg_residual = 0
            n_periods = 0

        # Show stats and creation form if we have a valid selection
        if n_periods > 0:
            # Determine suggested sign constraint
            if avg_residual > 0:
                suggested_sign = "positive"
                sign_explanation = "underfitting (model predicting below actual)"
            else:
                suggested_sign = "negative"
                sign_explanation = "overfitting (model predicting above actual)"

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Periods Selected", n_periods)
            with col2:
                st.metric("Avg Residual", f"${avg_residual:,.0f}")
            with col3:
                st.metric("Pattern", sign_explanation.split("(")[0].strip())

            st.info(f"**Suggested sign constraint: {suggested_sign.upper()}** - The model is {sign_explanation}")

            # Dummy creation form - use dynamic key so default updates when dates change
            default_name = f"dummy_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
            dummy_name = st.text_input(
                "Dummy Variable Name",
                value=default_name,
                key=f"dummy_name_{start_date}_{end_date}",
                help="Name for the dummy variable (will be added as a control)"
            )

            sign_override = st.selectbox(
                "Sign Constraint",
                options=["positive", "negative", "unconstrained"],
                index=0 if suggested_sign == "positive" else 1,
                key=f"dummy_sign_{start_date}_{end_date}",
                help="Expected direction of the coefficient"
            )

            if st.button("Create Dummy Variable", type="primary", key="create_dummy_btn"):
                if not dummy_name or dummy_name.strip() == "":
                    st.error("Please enter a name for the dummy variable.")
                elif start_date > end_date:
                    st.error("Start date must be before or equal to end date.")
                else:
                    dummy_config = DummyVariableConfig(
                        name=dummy_name.strip(),
                        start_date=start_date.strftime("%Y-%m-%d"),
                        end_date=end_date.strftime("%Y-%m-%d"),
                        sign_constraint=SignConstraint(sign_override)
                    )

                    if "pending_dummies" not in st.session_state:
                        st.session_state.pending_dummies = []

                    existing_names = [d.name for d in st.session_state.pending_dummies]
                    if dummy_name.strip() in existing_names:
                        st.warning(f"A dummy variable named '{dummy_name}' already exists in pending list.")
                    else:
                        st.session_state.pending_dummies.append(dummy_config)
                        st.success(f"Created dummy variable '{dummy_name}' ({start_date} to {end_date}, {sign_override})")
                        st.info("Go to **Configure Model** page to add this dummy to your model configuration.")
        else:
            st.warning("No data points in selected date range.")

        # Show pending dummies
        if st.session_state.get("pending_dummies"):
            st.markdown("---")
            st.markdown("**Pending Dummy Variables**")
            st.caption("These will be available in Configure Model page")

            for i, dummy in enumerate(st.session_state.pending_dummies):
                col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
                with col1:
                    st.write(f"**{dummy.name}**")
                with col2:
                    st.write(f"{dummy.start_date} to {dummy.end_date}")
                with col3:
                    st.write(f"Sign: {dummy.sign_constraint.value}")
                with col4:
                    if st.button("Remove", key=f"remove_dummy_{i}"):
                        st.session_state.pending_dummies.pop(i)
                        st.rerun()

        # Channel contributions over time
        st.subheader("Channel Contributions Over Time")

        channel_cols = config.get_channel_columns()
        channel_contribs = contribs[channel_cols] * config.data.revenue_scale

        fig3 = go.Figure()
        for ch in channel_cols:
            if ch in channel_contribs.columns:
                fig3.add_trace(go.Scatter(
                    x=contribs.index,
                    y=channel_contribs[ch],
                    name=ch.replace("_spend", ""),
                    stackgroup="one"
                ))

        fig3.update_layout(
            title="Stacked Channel Contributions",
            xaxis_title="Date",
            yaxis_title="Contribution ($)",
            template="plotly_dark",
        )
        st.plotly_chart(fig3, use_container_width=True)

    # =========================================================================
    # Tab 8: Export
    # =========================================================================
    with tab8:
        st.subheader("Export Results")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Contributions CSV**")
            contribs_real = contributions.get_time_series_contributions()
            contribs_csv = contribs_real.reset_index().to_csv(index=False)
            st.download_button(
                "Download Contributions CSV",
                data=contribs_csv,
                file_name="mmm_contributions.csv",
                mime="text/csv",
            )

            st.markdown("**Channel ROI CSV**")
            roi_csv = contributions.get_channel_roi().to_csv(index=False)
            st.download_button(
                "Download Channel ROI CSV",
                data=roi_csv,
                file_name="mmm_channel_roi.csv",
                mime="text/csv",
            )

        with col2:
            st.markdown("**JSON Report**")
            report_gen = ReportGenerator(wrapper)
            json_report = report_gen.generate_json_report()
            st.download_button(
                "Download JSON Report",
                data=json_report,
                file_name="mmm_report.json",
                mime="application/json",
            )

            st.markdown("**HTML Report**")
            html_report = report_gen.generate_html_report()
            st.download_button(
                "Download HTML Report",
                data=html_report,
                file_name="mmm_report.html",
                mime="text/html",
            )

        # Bayesian Significance Report
        st.markdown("---")
        st.markdown("**Bayesian Significance Analysis CSV**")

        try:
            prior_rois_export = {}
            for channel_config in config.channels:
                prior_rois_export[channel_config.name] = channel_config.roi_prior_mid

            sig_analyzer_export = BayesianSignificanceAnalyzer(
                idata=wrapper.idata,
                df_scaled=wrapper.df_scaled,
                channel_cols=config.get_channel_columns(),
                target_col=config.data.target_column,
                prior_rois=prior_rois_export,
            )
            sig_report_export = sig_analyzer_export.run_full_analysis()
            sig_df = sig_report_export.to_dataframe()
            sig_csv = sig_df.to_csv(index=False)

            st.download_button(
                "Download Bayesian Significance CSV",
                data=sig_csv,
                file_name="mmm_bayesian_significance.csv",
                mime="text/csv",
            )
        except Exception as e:
            st.warning(f"Could not generate Bayesian significance report: {str(e)}")

    # =========================================================================
    # Tab 9: Visualizations
    # =========================================================================
    with tab9:
        st.subheader("Executive Visualizations")

        import matplotlib.pyplot as plt
        from mmm_platform.analysis import (
            create_baseline_channels_donut,
            create_contribution_waterfall_chart,
            create_stacked_contributions_area,
            create_response_curves,
            create_roi_effectiveness_bubble,
        )

        viz_option = st.selectbox(
            "Select Visualization",
            [
                "Contribution Waterfall",
                "Base vs Channels (Donut)",
                "Stacked Contributions Over Time",
                "ROI vs Spend (Bubble)",
                "Response Curves",
            ]
        )

        fig = None

        try:
            if viz_option == "Contribution Waterfall":
                # Get contributions by component
                contribs_df = wrapper.get_contributions()
                channel_cols = config.get_channel_columns()

                # Build contribution dict
                contrib_dict = {}

                # Add baseline/intercept
                if 'intercept' in contribs_df.columns:
                    contrib_dict['Base'] = float(contribs_df['intercept'].sum()) * config.data.revenue_scale

                # Add channels
                for ch in channel_cols:
                    if ch in contribs_df.columns:
                        contrib_dict[ch.replace('_spend', '')] = float(contribs_df[ch].sum()) * config.data.revenue_scale

                # Add controls if present
                if wrapper.control_cols:
                    for ctrl in wrapper.control_cols:
                        if ctrl in contribs_df.columns:
                            contrib_dict[ctrl] = float(contribs_df[ctrl].sum()) * config.data.revenue_scale

                # Add seasonality if present
                seasonality_cols = [c for c in contribs_df.columns if 'fourier' in c.lower() or 'seasonal' in c.lower()]
                if seasonality_cols:
                    seasonality_total = sum(float(contribs_df[c].sum()) for c in seasonality_cols) * config.data.revenue_scale
                    contrib_dict['Seasonality'] = seasonality_total

                fig = create_contribution_waterfall_chart(contrib_dict)

            elif viz_option == "Base vs Channels (Donut)":
                breakdown = contributions.get_contribution_breakdown()
                baseline = abs(breakdown["intercept"]["real_value"])
                channels = abs(breakdown["channels"]["real_value"])
                fig = create_baseline_channels_donut(baseline, channels)

            elif viz_option == "Stacked Contributions Over Time":
                contribs_df = wrapper.get_contributions()
                channel_cols = config.get_channel_columns()
                dates = contribs_df.index
                channel_contribs = contribs_df[channel_cols] * config.data.revenue_scale
                fig = create_stacked_contributions_area(dates, channel_contribs)

            elif viz_option == "ROI vs Spend (Bubble)":
                roi_df = contributions.get_channel_roi()
                metrics = []
                for _, row in roi_df.iterrows():
                    metrics.append({
                        'channel': row['channel'].replace('_spend', ''),
                        'roi': row['roi'],
                        'spend': row['spend_real'],
                        'contribution': row['contribution_real'],
                    })
                fig = create_roi_effectiveness_bubble(metrics)

            elif viz_option == "Response Curves":
                # Get response curve data from the model
                channel_cols = config.get_channel_columns()
                curves = []

                # Try to extract saturation parameters from the model
                try:
                    if hasattr(wrapper.mmm, 'saturation') and wrapper.idata is not None:
                        posterior = wrapper.idata.posterior

                        for i, ch in enumerate(channel_cols):
                            # Get spend range
                            spend_data = wrapper.df_scaled[ch].values
                            spend_range = np.linspace(0, spend_data.max() * 1.5, 100)

                            # Get mean saturation parameters
                            if 'saturation_lam' in posterior:
                                lam = float(posterior['saturation_lam'].mean(dim=['chain', 'draw']).values[i])
                            else:
                                lam = 1.0

                            # Simple logistic saturation curve
                            response = lam * (1 - np.exp(-spend_range / (lam * 0.5)))

                            curves.append({
                                'channel': ch.replace('_spend', ''),
                                'spend': spend_range * config.data.spend_scale,
                                'response': response * config.data.revenue_scale,
                                'current_spend': float(spend_data.sum()) * config.data.spend_scale / len(spend_data),
                            })

                        fig = create_response_curves(curves)
                    else:
                        st.warning("Response curves require saturation parameters from the model.")
                except Exception as curve_error:
                    st.warning(f"Could not generate response curves: {curve_error}")

            if fig is not None:
                st.pyplot(fig)
                plt.close(fig)

                # Download button for the current visualization
                import io
                img_buffer = io.BytesIO()
                fig_for_download = None

                # Regenerate for download
                if viz_option == "Contribution Waterfall":
                    fig_for_download = create_contribution_waterfall_chart(contrib_dict)
                elif viz_option == "Base vs Channels (Donut)":
                    fig_for_download = create_baseline_channels_donut(baseline, channels)
                elif viz_option == "Stacked Contributions Over Time":
                    fig_for_download = create_stacked_contributions_area(dates, channel_contribs)
                elif viz_option == "ROI vs Spend (Bubble)":
                    fig_for_download = create_roi_effectiveness_bubble(metrics)

                if fig_for_download is not None:
                    fig_for_download.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
                    plt.close(fig_for_download)

                    st.download_button(
                        f"Download {viz_option} (PNG)",
                        data=img_buffer.getvalue(),
                        file_name=f"mmm_{viz_option.lower().replace(' ', '_')}.png",
                        mime="image/png"
                    )

        except Exception as viz_error:
            st.error(f"Error generating visualization: {viz_error}")
            st.info("Some visualizations require specific model outputs. Try a different visualization.")

    # =========================================================================
    # Tab 10: Model Coefficients
    # =========================================================================
    with tab10:
        st.subheader("Model Coefficients")
        st.caption("All variables in the model with their posterior statistics")

        # Build the variables table from posterior
        variables_data = []

        try:
            idata = wrapper.idata

            # Channels (saturation_beta)
            if "saturation_beta" in idata.posterior:
                beta_summary = az.summary(idata, var_names=["saturation_beta"], hdi_prob=0.95)
                channel_cols = config.get_channel_columns()
                for i, ch in enumerate(channel_cols):
                    try:
                        row = beta_summary.iloc[i]
                        hdi_low = row["hdi_2.5%"]
                        hdi_high = row["hdi_97.5%"]
                        significant = "Yes" if (hdi_low > 0 or hdi_high < 0) else "No"
                        variables_data.append({
                            "Variable": ch,
                            "Type": "Channel",
                            "Coef Mean": f"{row['mean']:.2f}",
                            "Coef Std": f"{row['sd']:.2f}",
                            "95% HDI": f"[{hdi_low:.2f}, {hdi_high:.2f}]",
                            "Significant": significant
                        })
                    except Exception:
                        pass

            # Controls (gamma_control)
            if "gamma_control" in idata.posterior:
                gamma_summary = az.summary(idata, var_names=["gamma_control"], hdi_prob=0.95)
                control_cols = wrapper.control_cols or []
                for i, ctrl in enumerate(control_cols):
                    try:
                        row = gamma_summary.iloc[i]
                        hdi_low = row["hdi_2.5%"]
                        hdi_high = row["hdi_97.5%"]
                        significant = "Yes" if (hdi_low > 0 or hdi_high < 0) else "No"

                        # Determine if this is a dummy variable
                        dummy_names = [dv.name for dv in config.dummy_variables]
                        is_dummy = ctrl in dummy_names or ctrl.replace("_inv", "") in dummy_names
                        var_type = "Dummy" if is_dummy else "Control"

                        variables_data.append({
                            "Variable": ctrl,
                            "Type": var_type,
                            "Coef Mean": f"{row['mean']:.2f}",
                            "Coef Std": f"{row['sd']:.2f}",
                            "95% HDI": f"[{hdi_low:.2f}, {hdi_high:.2f}]",
                            "Significant": significant
                        })
                    except Exception:
                        pass

            # Intercept
            if "intercept" in idata.posterior:
                intercept_summary = az.summary(idata, var_names=["intercept"], hdi_prob=0.95)
                row = intercept_summary.iloc[0]
                hdi_low = row["hdi_2.5%"]
                hdi_high = row["hdi_97.5%"]
                variables_data.append({
                    "Variable": "intercept",
                    "Type": "Intercept",
                    "Coef Mean": f"{row['mean']:.2f}",
                    "Coef Std": f"{row['sd']:.2f}",
                    "95% HDI": f"[{hdi_low:.2f}, {hdi_high:.2f}]",
                    "Significant": "Yes"
                })

        except Exception as e:
            st.warning(f"Could not extract all variable statistics: {e}")

        # Display the table
        if variables_data:
            df_vars = pd.DataFrame(variables_data)
            st.dataframe(df_vars, use_container_width=True, hide_index=True)

            # Summary counts
            st.markdown("---")
            st.subheader("Summary")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Channels", len([v for v in variables_data if v["Type"] == "Channel"]))
            with col2:
                st.metric("Controls", len([v for v in variables_data if v["Type"] == "Control"]))
            with col3:
                st.metric("Dummies", len([v for v in variables_data if v["Type"] == "Dummy"]))
            with col4:
                sig_count = len([v for v in variables_data if v["Significant"] == "Yes"])
                st.metric("Significant", f"{sig_count}/{len(variables_data)}")

            # Download button
            st.markdown("---")
            csv_data = df_vars.to_csv(index=False)
            st.download_button(
                "Download Coefficients (CSV)",
                data=csv_data,
                file_name="model_coefficients.csv",
                mime="text/csv"
            )
        else:
            st.warning("No variable statistics available.")

    # =========================================================================
    # Tab 11: Media Curves
    # =========================================================================
    with tab11:
        st.subheader("Interactive Media Curves")
        st.caption("Explore saturation curves and adstock decay for each channel")

        # Check if we have the required posterior variables
        idata = wrapper.idata
        if idata is None:
            st.warning("No inference data available for curve visualization")
        else:
            posterior = idata.posterior
            has_saturation = "saturation_lam" in posterior and "saturation_beta" in posterior
            has_adstock = "adstock_alpha" in posterior

            if not has_saturation:
                st.warning("Saturation parameters not found in model posterior")
            else:
                # Get channel info
                channel_cols = config.get_channel_columns()

                # Build display names
                display_names = {}
                for ch_config in config.channels:
                    display_names[ch_config.name] = ch_config.get_display_name()

                channel_display_names = [display_names.get(ch, ch) for ch in channel_cols]

                # Controls row
                ctrl_col1, ctrl_col2 = st.columns([1, 2])

                with ctrl_col1:
                    view_type = st.radio(
                        "View",
                        ["Saturation Curves", "Adstock Decay"] if has_adstock else ["Saturation Curves"],
                        horizontal=True
                    )

                with ctrl_col2:
                    channel_options = ["All Channels"] + channel_display_names
                    selected_channel = st.selectbox("Channel", channel_options)

                st.markdown("---")

                # Get scales
                spend_scale = config.data.spend_scale
                revenue_scale = config.data.revenue_scale

                if view_type == "Saturation Curves":
                    _show_saturation_curves(
                        wrapper, config, channel_cols, display_names,
                        selected_channel, spend_scale, revenue_scale
                    )
                else:
                    _show_adstock_curves(
                        wrapper, config, channel_cols, display_names,
                        selected_channel
                    )


def _logistic_saturation(x: np.ndarray, lam: float) -> np.ndarray:
    """Apply logistic saturation: (1 - exp(-lam*x)) / (1 + exp(-lam*x))"""
    return (1 - np.exp(-lam * x)) / (1 + np.exp(-lam * x))


def _show_saturation_curves(wrapper, config, channel_cols, display_names, selected_channel, spend_scale, revenue_scale):
    """Show interactive saturation curves."""
    import plotly.graph_objects as go

    posterior = wrapper.idata.posterior

    # Time period selector
    col1, col2 = st.columns([1, 3])
    with col1:
        time_period = st.selectbox(
            "Time Period",
            ["Weekly", "Yearly"],
            index=0,
            help="Weekly shows response per week. Yearly multiplies by 52 weeks."
        )
    time_multiplier = 52 if time_period == "Yearly" else 1
    period_label = "Yearly" if time_period == "Yearly" else "Weekly"

    # Initialize scenario state
    if 'curve_scenarios' not in st.session_state:
        st.session_state.curve_scenarios = {}

    # Determine which channels to show
    if selected_channel == "All Channels":
        channels_to_show = list(range(len(channel_cols)))
    else:
        # Find index of selected channel
        channel_display_names = [display_names.get(ch, ch) for ch in channel_cols]
        idx = channel_display_names.index(selected_channel)
        channels_to_show = [idx]

    # Get actual contributions to calibrate the curve
    # We calibrate so the curve passes through actual avg weekly contribution at avg weekly spend
    contribs = wrapper.get_contributions()
    n_periods = len(wrapper.df_scaled)

    # Generate curve data for each channel
    curves_data = []
    for i in channels_to_show:
        ch = channel_cols[i]
        display_name = display_names.get(ch, ch)

        # Get posterior mean for lambda (saturation shape parameter)
        lam = float(posterior['saturation_lam'].mean(dim=['chain', 'draw']).values[i])

        # Get spend data (in scaled units, e.g., thousands)
        spend_data = wrapper.df_scaled[ch].values

        # PyMC-Marketing normalizes by max internally, so lambda is calibrated for 0-1 range
        x_max = spend_data.max()
        if x_max == 0:
            x_max = 1.0

        # Get actual values for this channel
        actual_contrib_total = contribs[ch].sum() * revenue_scale if ch in contribs.columns else 0
        actual_spend_total = spend_data.sum() * spend_scale

        # Average WEEKLY values (what the curve should show)
        avg_weekly_spend = spend_data.mean() * spend_scale
        avg_weekly_contrib = actual_contrib_total / n_periods if n_periods > 0 else 0

        # Calculate saturation at current average spend (normalized)
        avg_spend_normalized = avg_weekly_spend / (x_max * spend_scale)
        avg_saturation = _logistic_saturation(avg_spend_normalized, lam)

        # Calibration factor: at avg spend, response should equal avg weekly contribution
        # response = calibration_factor * saturation(spend_normalized)
        calibration_factor = avg_weekly_contrib / avg_saturation if avg_saturation > 0.001 else avg_weekly_contrib

        # Generate spend range in real dollars (0 to 2.5x max weekly spend)
        max_spend_real = x_max * spend_scale * 2.5
        spend_range_real = np.linspace(0.001, max_spend_real, 200)

        # Convert to normalized space (0-1) for saturation calculation
        spend_range_normalized = spend_range_real / (x_max * spend_scale)

        # Apply saturation and calibration to get WEEKLY response, then apply time multiplier
        response_normalized = _logistic_saturation(spend_range_normalized, lam)
        response_weekly = calibration_factor * response_normalized
        response_real = response_weekly * time_multiplier

        curves_data.append({
            'channel': ch,
            'display_name': display_name,
            'spend_range': spend_range_real * time_multiplier,  # Scale spend to match period
            'response': response_real,
            'current_spend': avg_weekly_spend * time_multiplier,
            'current_response': avg_weekly_contrib * time_multiplier,
            'actual_contrib_total': actual_contrib_total,
            'actual_spend_total': actual_spend_total,
            'lam': lam,
            'calibration_factor': calibration_factor,
            'x_max': x_max,
            'max_spend': max_spend_real * time_multiplier,
            'n_periods': n_periods,
            'time_multiplier': time_multiplier,
        })

    # Create Plotly figure
    fig = go.Figure()

    colors = px.colors.qualitative.Set2

    for idx, curve in enumerate(curves_data):
        color = colors[idx % len(colors)]

        # Main curve
        fig.add_trace(go.Scatter(
            x=curve['spend_range'],
            y=curve['response'],
            mode='lines',
            name=curve['display_name'],
            line=dict(color=color, width=3),
            hovertemplate=f"<b>{curve['display_name']}</b><br>" +
                         "Spend: $%{x:,.0f}<br>Response: $%{y:,.0f}<extra></extra>"
        ))

        # Current position marker
        fig.add_trace(go.Scatter(
            x=[curve['current_spend']],
            y=[curve['current_response']],
            mode='markers',
            name=f"{curve['display_name']} (Current)",
            marker=dict(size=12, color=color, symbol='circle',
                       line=dict(width=2, color='black')),
            hovertemplate=f"<b>{curve['display_name']} - Current</b><br>" +
                         "Spend: $%{x:,.0f}<br>Response: $%{y:,.0f}<extra></extra>",
            showlegend=False
        ))

    # Add breakeven line
    fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)

    fig.update_layout(
        title=f"Saturation Curves - {period_label} Response vs {period_label} Spend",
        xaxis_title=f"{period_label} Spend ($)",
        yaxis_title=f"{period_label} Response ($)",
        hovermode='closest',
        showlegend=True,
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, use_container_width=True)

    # Show comparison to actual contributions
    if time_multiplier == 1:
        st.caption("*Curve shows weekly response at each spend level. Calibrated so current position matches actual average weekly contribution.*")
    else:
        st.caption(f"*Curve shows yearly response (weekly × 52). Spend and response are annualized projections.*")

    # Show a comparison table for all channels in the view
    if len(channels_to_show) > 0:
        with st.expander("📊 Compare Curve vs Actual Model Contributions"):
            comparison_data = []
            for curve in curves_data:
                comparison_data.append({
                    "Channel": curve['display_name'],
                    f"Avg {period_label} Spend": f"${curve['current_spend']:,.0f}",
                    f"Avg {period_label} Response": f"${curve['current_response']:,.0f}",
                    "Total Contribution": f"${curve['actual_contrib_total']:,.0f}",
                    "Total Spend": f"${curve['actual_spend_total']:,.0f}",
                })
            st.dataframe(pd.DataFrame(comparison_data), use_container_width=True, hide_index=True)
            if time_multiplier == 1:
                st.caption("*Avg Weekly Response = Total Contribution / Periods.*")
            else:
                st.caption("*Yearly values are weekly × 52.*")

    # Interactive exploration (only for single channel)
    if len(channels_to_show) == 1:
        curve = curves_data[0]
        ch = curve['channel']

        st.markdown("---")
        st.subheader(f"Explore: {curve['display_name']}")

        # Spend slider
        min_spend = 0
        max_spend = int(curve['max_spend'])
        current = int(curve['current_spend'])
        time_mult = curve['time_multiplier']

        slider_spend = st.slider(
            f"Explore {period_label} Spend Level",
            min_value=min_spend,
            max_value=max_spend,
            value=current,
            step=max(1, max_spend // 100),
            format="$%d"
        )

        # Calculate metrics at slider position
        # Convert slider spend back to weekly for saturation calculation
        x_max = curve['x_max']
        calibration_factor = curve['calibration_factor']
        slider_spend_weekly = slider_spend / time_mult
        slider_spend_normalized = slider_spend_weekly / (x_max * spend_scale)
        slider_response_normalized = _logistic_saturation(slider_spend_normalized, curve['lam'])

        # Response using the same calibration as the curve, then apply time multiplier
        slider_response_weekly = calibration_factor * slider_response_normalized
        slider_response = slider_response_weekly * time_mult

        # Marginal ROI calculation (derivative in normalized space, then scale)
        # Marginal ROI is per-dollar so it doesn't change with time period
        from mmm_platform.analysis.marginal_roi import logistic_saturation_derivative
        marginal_normalized = logistic_saturation_derivative(slider_spend_normalized, curve['lam'])
        marginal_roi = calibration_factor * marginal_normalized / (x_max * spend_scale)

        # Saturation percentage (how close to the asymptotic max)
        saturation_pct = slider_response_normalized * 100

        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric(f"{period_label} Spend", f"${slider_spend:,.0f}")
        col2.metric(f"{period_label} Response", f"${slider_response:,.0f}")
        col3.metric("Marginal ROI", f"${marginal_roi:.2f}")
        col4.metric("Saturation Level", f"{saturation_pct:.0f}%")

        # Zone indicator
        if marginal_roi > 1.5:
            st.success("**Efficient Zone** - High marginal returns, room to increase spend")
        elif marginal_roi > 1.0:
            st.info("**Optimal Zone** - Good returns, near optimal spend level")
        else:
            st.warning("**Diminishing Returns Zone** - Low marginal ROI, consider reducing spend")

        # Scenario comparison
        st.markdown("---")
        st.subheader("Scenario Comparison")

        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("Add Scenario", use_container_width=True):
                if ch not in st.session_state.curve_scenarios:
                    st.session_state.curve_scenarios[ch] = []
                st.session_state.curve_scenarios[ch].append({
                    'spend': slider_spend,
                    'response': slider_response,
                    'marginal_roi': marginal_roi,
                    'saturation': saturation_pct,
                })
                st.rerun()

        with col2:
            if ch in st.session_state.curve_scenarios and st.session_state.curve_scenarios[ch]:
                if st.button("Clear Scenarios"):
                    st.session_state.curve_scenarios[ch] = []
                    st.rerun()

        # Display scenarios
        if ch in st.session_state.curve_scenarios and st.session_state.curve_scenarios[ch]:
            scenarios = st.session_state.curve_scenarios[ch]
            scenario_data = []
            for i, s in enumerate(scenarios):
                scenario_data.append({
                    "Scenario": f"#{i+1}",
                    "Spend": f"${s['spend']:,.0f}",
                    "Response": f"${s['response']:,.0f}",
                    "Marginal ROI": f"${s['marginal_roi']:.2f}",
                    "Saturation": f"{s['saturation']:.0f}%"
                })

            st.dataframe(pd.DataFrame(scenario_data), use_container_width=True, hide_index=True)


def _show_adstock_curves(wrapper, config, channel_cols, display_names, selected_channel):
    """Show adstock decay curves."""
    import plotly.graph_objects as go

    posterior = wrapper.idata.posterior

    if "adstock_alpha" not in posterior:
        st.warning("Adstock parameters not found in model")
        return

    # Determine which channels to show
    if selected_channel == "All Channels":
        channels_to_show = list(range(len(channel_cols)))
    else:
        channel_display_names = [display_names.get(ch, ch) for ch in channel_cols]
        idx = channel_display_names.index(selected_channel)
        channels_to_show = [idx]

    # Generate adstock data
    l_max = 12  # Number of periods to show
    periods = np.arange(l_max)

    curves_data = []
    for i in channels_to_show:
        ch = channel_cols[i]
        display_name = display_names.get(ch, ch)

        alpha = float(posterior['adstock_alpha'].mean(dim=['chain', 'draw']).values[i])

        # Generate decay weights
        weights = np.array([alpha ** p for p in periods])
        weights_normalized = weights / weights.sum()

        # Calculate half-life
        half_life = np.log(0.5) / np.log(alpha) if alpha > 0 and alpha < 1 else float('inf')

        curves_data.append({
            'channel': ch,
            'display_name': display_name,
            'alpha': alpha,
            'weights': weights,
            'weights_normalized': weights_normalized,
            'half_life': half_life,
        })

    # Create figure
    fig = go.Figure()

    colors = px.colors.qualitative.Set2

    for idx, curve in enumerate(curves_data):
        color = colors[idx % len(colors)]

        fig.add_trace(go.Bar(
            x=periods,
            y=curve['weights_normalized'] * 100,
            name=f"{curve['display_name']} (α={curve['alpha']:.2f})",
            marker_color=color,
            opacity=0.7,
            hovertemplate=f"<b>{curve['display_name']}</b><br>" +
                         "Period: %{x}<br>Weight: %{y:.1f}%<extra></extra>"
        ))

    fig.update_layout(
        title="Adstock Decay - Carryover Effect by Period",
        xaxis_title="Periods After Spend",
        yaxis_title="Effect Weight (%)",
        barmode='group',
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, use_container_width=True)

    # Summary table
    st.markdown("---")
    st.subheader("Adstock Parameters")

    summary_data = []
    for curve in curves_data:
        summary_data.append({
            "Channel": curve['display_name'],
            "Alpha (Decay Rate)": f"{curve['alpha']:.2f}",
            "Half-Life (periods)": f"{curve['half_life']:.1f}" if curve['half_life'] < 100 else "N/A",
            "Immediate Effect": f"{curve['weights_normalized'][0]*100:.1f}%",
            "Carryover (period 1)": f"{curve['weights_normalized'][1]*100:.1f}%" if len(curve['weights_normalized']) > 1 else "N/A",
        })

    st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

    # Interpretation
    st.markdown("---")
    st.markdown("""
    **Interpretation:**
    - **Alpha (α)**: Decay rate. Higher α = more carryover effect
    - **Half-Life**: Number of periods for effect to decay by 50%
    - **Immediate Effect**: % of total effect in the spend period
    - Channels with high α have longer-lasting advertising effects
    """)


def show_ec2_results():
    """Show results page for EC2 mode with results from the API."""
    results = st.session_state.ec2_results
    job_id = st.session_state.get("ec2_job_id", "unknown")

    st.info(f"Showing results from EC2 model run. Job ID: `{job_id}`")

    # Tabs
    tab1, tab2, tab3 = st.tabs([
        "Overview",
        "Channel ROI",
        "Contributions"
    ])

    # =========================================================================
    # Tab 1: Overview
    # =========================================================================
    with tab1:
        st.subheader("Model Overview")

        fit_stats = results.get("fit_statistics", {})
        diagnostics = results.get("diagnostics", {})

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            r2 = diagnostics.get("r2") or fit_stats.get("r2", "N/A")
            st.metric("R²", f"{r2:.2f}" if isinstance(r2, (int, float)) else r2)
        with col2:
            mape = diagnostics.get("mape") or fit_stats.get("mape", "N/A")
            st.metric("MAPE", f"{mape:.2f}%" if isinstance(mape, (int, float)) else mape)
        with col3:
            rmse = diagnostics.get("rmse") or fit_stats.get("rmse", "N/A")
            st.metric("RMSE", f"{rmse:.2f}" if isinstance(rmse, (int, float)) else rmse)
        with col4:
            n_obs = diagnostics.get("n_observations") or fit_stats.get("n_observations", "N/A")
            st.metric("Observations", n_obs)

        st.markdown("---")
        st.subheader("Fit Statistics")
        st.json(fit_stats)

    # =========================================================================
    # Tab 2: Channel ROI
    # =========================================================================
    with tab2:
        st.subheader("Channel ROI")

        channel_roi = results.get("channel_roi", [])
        if channel_roi:
            roi_df = pd.DataFrame(channel_roi)
            st.dataframe(roi_df, use_container_width=True, hide_index=True)

            # ROI bar chart
            if "channel" in roi_df.columns and "roi" in roi_df.columns:
                fig = px.bar(
                    roi_df.sort_values("roi", ascending=True),
                    x="roi",
                    y="channel",
                    orientation="h",
                    title="ROI by Channel",
                    labels={"roi": "ROI", "channel": "Channel"}
                )
                fig.add_vline(x=1, line_dash="dash", line_color="red", annotation_text="Breakeven")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No channel ROI data available")

    # =========================================================================
    # Tab 3: Contributions
    # =========================================================================
    with tab3:
        st.subheader("Contributions")

        contributions = results.get("contributions", {})

        # Total by channel
        total_by_channel = contributions.get("total_by_channel", {})
        if total_by_channel:
            st.markdown("#### Total Contribution by Channel")
            contrib_df = pd.DataFrame([
                {"Channel": k, "Contribution": v}
                for k, v in total_by_channel.items()
            ]).sort_values("Contribution", ascending=False)

            st.dataframe(contrib_df, use_container_width=True, hide_index=True)

            # Pie chart
            fig = px.pie(
                contrib_df[contrib_df["Contribution"] > 0],
                values="Contribution",
                names="Channel",
                title="Contribution Breakdown"
            )
            st.plotly_chart(fig, use_container_width=True)

        # Time series
        time_series = contributions.get("time_series", [])
        if time_series:
            st.markdown("#### Contribution Time Series")
            ts_df = pd.DataFrame(time_series)
            st.dataframe(ts_df.head(20), use_container_width=True, hide_index=True)
            st.caption(f"Showing first 20 of {len(ts_df)} rows")

    # Reset button
    st.markdown("---")
    if st.button("Clear EC2 Results & Run New Model"):
        st.session_state.ec2_mode = False
        st.session_state.ec2_results = None
        st.session_state.ec2_job_id = None
        st.session_state.model_fitted = False
        st.rerun()


def show_demo_results():
    """Show results page for demo mode with all new features."""
    import matplotlib.pyplot as plt
    from mmm_platform.analysis import (
        create_baseline_channels_donut,
        create_contribution_rank_over_time,
        create_roi_effectiveness_bubble,
        create_response_curves,
        create_current_vs_marginal_roi,
        create_spend_vs_breakeven,
        create_stacked_contributions_area,
        create_contribution_waterfall_chart,
    )

    demo = st.session_state.demo

    st.info("Running in **Demo Mode** with simulated data. All features are fully functional!")

    # Tabs for demo features
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Overview",
        "Marginal ROI & Priority",
        "Executive Summary",
        "Combined Models",
        "Visualizations"
    ])

    # =========================================================================
    # Tab 1: Overview
    # =========================================================================
    with tab1:
        st.subheader("Demo Data Overview")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Channels", len(demo.channel_cols))
        with col2:
            st.metric("Weeks of Data", len(demo.df_scaled))
        with col3:
            total_spend = demo.df_scaled[demo.channel_cols].sum().sum() * demo.spend_scale
            st.metric("Total Spend", f"${total_spend:,.0f}")
        with col4:
            total_revenue = demo.contribs[demo.target_col].sum() * demo.revenue_scale
            st.metric("Total Revenue", f"${total_revenue:,.0f}")

        st.markdown("---")

        # Channel list
        st.subheader("Channels in Demo")
        channel_df = pd.DataFrame({
            "Channel": [ch.replace("_", " ").title() for ch in demo.channel_cols],
            "Total Spend": [f"${demo.df_scaled[ch].sum() * demo.spend_scale:,.0f}" for ch in demo.channel_cols],
            "Total Contribution": [f"${demo.contribs[ch].sum() * demo.revenue_scale:,.0f}" for ch in demo.channel_cols],
        })
        st.dataframe(channel_df, use_container_width=True, hide_index=True)

        # Contribution breakdown
        st.markdown("---")
        st.subheader("Contribution Breakdown")

        breakdown = demo._get_contribution_breakdown()
        fig = px.pie(
            values=[breakdown['baseline'], breakdown['channels']],
            names=['Base', 'All Channels'],
            title="Base vs Channels"
        )
        st.plotly_chart(fig, use_container_width=True)

    # =========================================================================
    # Tab 2: Marginal ROI & Investment Priority
    # =========================================================================
    with tab2:
        st.subheader("Marginal ROI & Investment Priority")

        st.markdown("""
        This analysis uses **saturation curve derivatives** to calculate the true marginal ROI
        (return on the *next* dollar spent), not just the average ROI.
        """)

        # Get priority table
        priority_df = demo.marginal_analyzer.get_priority_table()

        # Key metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            n_increase = len(priority_df[priority_df['action'] == 'INCREASE'])
            st.metric("Channels to INCREASE", n_increase)
        with col2:
            n_hold = len(priority_df[priority_df['action'] == 'HOLD'])
            st.metric("Channels to HOLD", n_hold)
        with col3:
            n_reduce = len(priority_df[priority_df['action'] == 'REDUCE'])
            st.metric("Channels to REDUCE", n_reduce)

        st.markdown("---")

        # Priority table
        st.subheader("Investment Priority Table")

        display_df = priority_df.copy()
        display_df['current_spend'] = display_df['current_spend'].apply(lambda x: f"${x:,.0f}")
        display_df['current_roi'] = display_df['current_roi'].apply(lambda x: f"${x:.2f}")
        display_df['marginal_roi'] = display_df['marginal_roi'].apply(lambda x: f"${x:.2f}")
        display_df['breakeven_spend'] = display_df['breakeven_spend'].apply(
            lambda x: f"${x:,.0f}" if x is not None else "N/A"
        )
        display_df['headroom_amount'] = display_df['headroom_amount'].apply(lambda x: f"${x:,.0f}")

        # Color-code action column
        def color_action(val):
            if val == 'INCREASE':
                return 'background-color: #90EE90'
            elif val == 'REDUCE':
                return 'background-color: #FFB6C1'
            else:
                return 'background-color: #FFFFE0'

        styled_df = display_df[['channel', 'current_spend', 'current_roi', 'marginal_roi',
                                 'priority_rank', 'breakeven_spend', 'headroom_amount', 'action', 'needs_test']]
        styled_df.columns = ['Channel', 'Current Spend', 'Current ROI', 'Marginal ROI',
                             'Priority', 'Breakeven Spend', 'Headroom', 'Action', 'Needs Test']

        st.dataframe(styled_df, use_container_width=True, hide_index=True)

        st.markdown("""
        **Key Concepts:**
        - **Marginal ROI**: Return on the *next* dollar spent (from saturation curve derivative)
        - **Breakeven Spend**: Spend level where marginal ROI = $1.00
        - **Headroom**: Additional spend available before hitting breakeven
        - **Needs Test**: High uncertainty in ROI estimate - validate with incrementality test
        """)

        # Visualizations
        st.markdown("---")
        st.subheader("Current vs Marginal ROI")

        result = demo.marginal_analyzer.run_full_analysis()
        channels = [ch.channel for ch in result.channel_analysis]
        current_rois = [ch.current_roi for ch in result.channel_analysis]
        marginal_rois = [ch.marginal_roi for ch in result.channel_analysis]
        channel_names = [ch.channel_name for ch in result.channel_analysis]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='Current (Avg) ROI',
            x=channel_names,
            y=current_rois,
            marker_color='steelblue'
        ))
        fig.add_trace(go.Bar(
            name='Marginal ROI',
            x=channel_names,
            y=marginal_rois,
            marker_color='orange'
        ))
        fig.add_hline(y=1, line_dash="dash", line_color="red", annotation_text="Breakeven")
        fig.update_layout(
            barmode='group',
            title="Current ROI vs Marginal ROI by Channel",
            xaxis_tickangle=-45
        )
        st.plotly_chart(fig, use_container_width=True)

    # =========================================================================
    # Tab 3: Executive Summary
    # =========================================================================
    with tab3:
        st.subheader("Executive Summary")

        summary = demo.exec_generator.get_summary_dict()

        # Portfolio Overview
        st.markdown("### Portfolio Overview")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Spend", f"${summary['portfolio']['total_spend']:,.0f}")
        with col2:
            st.metric("Total Contribution", f"${summary['portfolio']['total_contribution']:,.0f}")
        with col3:
            st.metric("Portfolio ROI", f"${summary['portfolio']['portfolio_roi']:.2f}")

        st.markdown("---")

        # Recommendations Summary
        st.markdown("### Investment Recommendations")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("INCREASE", summary['counts']['increase'], delta="High marginal ROI")
        with col2:
            st.metric("HOLD", summary['counts']['hold'], delta="Profitable")
        with col3:
            st.metric("REDUCE", summary['counts']['reduce'], delta="Below breakeven")
        with col4:
            st.metric("Need Validation", summary['counts']['needs_validation'], delta="Run tests")

        st.markdown("---")

        # INCREASE channels
        st.markdown("### Channels to INCREASE")
        st.markdown("*Marginal ROI > $1.50 with headroom to grow*")
        if summary['recommendations']['increase']:
            for ch in summary['recommendations']['increase']:
                with st.expander(f"+ {ch['channel_name']}" + (" (Validate)" if ch['needs_test'] else "")):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Current Spend:** ${ch['current_spend']:,.0f}")
                        st.write(f"**Marginal ROI:** ${ch['marginal_roi']:.2f}")
                    with col2:
                        st.write(f"**Headroom:** ${ch['headroom_amount']:,.0f}")
                        if ch['breakeven_spend']:
                            st.write(f"**Breakeven Spend:** ${ch['breakeven_spend']:,.0f}")
        else:
            st.info("No channels qualify for increase.")

        # HOLD channels
        st.markdown("### Channels to HOLD")
        st.markdown("*Marginal ROI $1.00-$1.50 - profitable but limited upside*")
        if summary['recommendations']['hold']:
            for ch in summary['recommendations']['hold']:
                st.write(f"= **{ch['channel_name']}** - Marginal ROI: ${ch['marginal_roi']:.2f}")
        else:
            st.info("No channels in hold range.")

        # REDUCE channels
        st.markdown("### Channels to REDUCE")
        st.markdown("*Marginal ROI < $1.00 - next dollar loses money*")
        if summary['recommendations']['reduce']:
            for ch in summary['recommendations']['reduce']:
                st.write(f"- **{ch['channel_name']}** - Marginal ROI: ${ch['marginal_roi']:.2f}")
        else:
            st.success("No channels below breakeven!")

        # Reallocation Opportunity
        st.markdown("---")
        st.markdown("### Reallocation Opportunity")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Funds from REDUCE channels", f"${summary['portfolio']['reallocation_potential']:,.0f}")
        with col2:
            st.metric("Headroom in INCREASE channels", f"${summary['portfolio']['headroom_available']:,.0f}")

        if summary['reallocation_moves']:
            st.markdown("**Top Reallocation Moves:**")
            for i, move in enumerate(summary['reallocation_moves'][:3]):
                validate = " *(validate with test)*" if move['needs_validation'] else ""
                st.write(f"{i+1}. Allocate **${move['amount']:,.0f}** to **{move['to_channel']}**{validate}")
                st.write(f"   Expected return: ${move['expected_return']:,.0f}")

    # =========================================================================
    # Tab 4: Combined Model Analysis
    # =========================================================================
    with tab4:
        st.subheader("Combined Model Analysis (Online + Offline)")

        st.markdown("""
        This analysis combines two models (e.g., Online Revenue + Offline Revenue) to provide
        unified recommendations across multiple optimization views.
        """)

        combined_summary = demo.combined_analyzer.get_summary_dict(demo.combined_result)

        # Margin settings
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Online Margin", f"{combined_summary['margins']['online']*100:.0f}%")
        with col2:
            st.metric("Offline Margin", f"{combined_summary['margins']['offline']*100:.0f}%")

        st.markdown("---")

        # Master table
        st.subheader("Marginal ROI by View")

        combined_df = demo.combined_analyzer.get_summary_table(demo.combined_result.combined_analysis)
        display_cols = ['channel', 'current_spend', 'marginal_online', 'marginal_offline',
                        'marginal_total', 'marginal_profit', 'action_online', 'action_offline',
                        'action_total', 'action_profit']

        display_df = combined_df[display_cols].copy()
        display_df['current_spend'] = display_df['current_spend'].apply(lambda x: f"${x:,.0f}")
        for col in ['marginal_online', 'marginal_offline', 'marginal_total', 'marginal_profit']:
            display_df[col] = display_df[col].apply(lambda x: f"${x:.2f}")

        display_df.columns = ['Channel', 'Spend', 'Online', 'Offline', 'Total', 'Profit',
                              'Act:Online', 'Act:Offline', 'Act:Total', 'Act:Profit']

        st.dataframe(display_df, use_container_width=True, hide_index=True)

        # Conflicts
        st.markdown("---")
        st.subheader("Conflicting Recommendations")

        if combined_summary['conflicts']:
            st.warning("These channels have different recommendations depending on your optimization goal:")
            for ch in combined_summary['conflicts']:
                st.write(f"- **{ch}**")
            st.info("Decision depends on your business objective (revenue vs profit)!")
        else:
            st.success("All views agree on recommendations - no conflicts!")

        # View-specific summaries
        st.markdown("---")
        st.subheader("Recommendations by View")

        view_tabs = st.tabs(["Online", "Offline", "Total Revenue", "Profit"])

        for i, (view, tab) in enumerate(zip(['online', 'offline', 'total', 'profit'], view_tabs)):
            with tab:
                view_data = combined_summary['views'][view]
                st.markdown(f"**Thresholds:** INCREASE > ${view_data['threshold_high']:.2f}, REDUCE < ${view_data['threshold_low']:.2f}")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("INCREASE", view_data['counts']['increase'])
                with col2:
                    st.metric("HOLD", view_data['counts']['hold'])
                with col3:
                    st.metric("REDUCE", view_data['counts']['reduce'])

    # =========================================================================
    # Tab 5: Visualizations
    # =========================================================================
    with tab5:
        st.subheader("Executive Visualizations")

        viz_option = st.selectbox(
            "Select Visualization",
            [
                "Base vs Channels (Donut)",
                "Contribution Rank Over Time",
                "ROI vs Effectiveness (Bubble)",
                "Response Curves",
                "Current vs Marginal ROI",
                "Spend vs Breakeven",
                "Stacked Contributions Over Time",
                "Contribution Waterfall",
            ]
        )

        # Generate selected visualization
        fig = None

        if viz_option == "Base vs Channels (Donut)":
            breakdown = demo._get_contribution_breakdown()
            fig = create_baseline_channels_donut(
                breakdown['baseline'], breakdown['channels']
            )

        elif viz_option == "Contribution Rank Over Time":
            contrib_ts = demo.contribs[demo.channel_cols].copy()
            contrib_ts.index = demo.dates
            fig = create_contribution_rank_over_time(contrib_ts, resample_freq='Q')

        elif viz_option == "ROI vs Effectiveness (Bubble)":
            metrics = demo._get_channel_metrics()
            fig = create_roi_effectiveness_bubble(metrics)

        elif viz_option == "Response Curves":
            curves = demo._get_response_curves()
            fig = create_response_curves(curves[:7])

        elif viz_option == "Current vs Marginal ROI":
            result = demo.marginal_analyzer.run_full_analysis()
            channels = [ch.channel for ch in result.channel_analysis]
            current_rois = [ch.current_roi for ch in result.channel_analysis]
            marginal_rois = [ch.marginal_roi for ch in result.channel_analysis]
            fig = create_current_vs_marginal_roi(channels, current_rois, marginal_rois)

        elif viz_option == "Spend vs Breakeven":
            result = demo.marginal_analyzer.run_full_analysis()
            channels = [ch.channel for ch in result.channel_analysis]
            current_spends = [ch.current_spend for ch in result.channel_analysis]
            breakeven_spends = [ch.breakeven_spend for ch in result.channel_analysis]
            fig = create_spend_vs_breakeven(channels, current_spends, breakeven_spends)

        elif viz_option == "Stacked Contributions Over Time":
            fig = create_stacked_contributions_area(
                demo.dates,
                demo.contribs[demo.channel_cols]
            )

        elif viz_option == "Contribution Waterfall":
            contrib_dict = {ch: float(demo.contribs[ch].sum()) for ch in demo.channel_cols}
            contrib_dict['Base'] = float(demo.contribs['intercept'].sum())
            fig = create_contribution_waterfall_chart(contrib_dict)

        if fig is not None:
            st.pyplot(fig)
            plt.close(fig)

        # Download option
        st.markdown("---")
        if st.button("Generate All Visualizations"):
            with st.spinner("Generating all visualizations..."):
                import io
                import zipfile

                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                    # Generate each visualization
                    viz_funcs = [
                        ("01_donut.png", lambda: create_baseline_channels_donut(
                            demo._get_contribution_breakdown()['baseline'],
                            demo._get_contribution_breakdown()['channels']
                        )),
                        ("02_rank_over_time.png", lambda: create_contribution_rank_over_time(
                            demo.contribs[demo.channel_cols].set_index(demo.dates)
                        )),
                        ("03_bubble.png", lambda: create_roi_effectiveness_bubble(
                            demo._get_channel_metrics()
                        )),
                        ("04_response_curves.png", lambda: create_response_curves(
                            demo._get_response_curves()[:7]
                        )),
                    ]

                    for filename, func in viz_funcs:
                        fig = func()
                        img_buffer = io.BytesIO()
                        fig.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
                        plt.close(fig)
                        zf.writestr(filename, img_buffer.getvalue())

                st.download_button(
                    "Download All Visualizations (ZIP)",
                    data=zip_buffer.getvalue(),
                    file_name="mmm_visualizations.zip",
                    mime="application/zip"
                )
