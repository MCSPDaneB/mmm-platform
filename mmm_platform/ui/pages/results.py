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
from mmm_platform.analysis.roi_diagnostics import quick_roi_diagnostics
from mmm_platform.config.schema import DummyVariableConfig, SignConstraint, KPIType
from mmm_platform.ui.kpi_labels import KPILabels


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

    # Create KPI-aware labels helper
    kpi_labels = KPILabels(config)
    eff_label = kpi_labels.efficiency_label  # "ROI" or "Cost Per X"

    # Create analyzers
    diagnostics = ModelDiagnostics.from_mmm_wrapper(wrapper)
    contributions = ContributionAnalyzer.from_mmm_wrapper(wrapper)
    marginal_analyzer = MarginalROIAnalyzer.from_mmm_wrapper(wrapper)
    exec_generator = ExecutiveSummaryGenerator(marginal_analyzer)

    # Tabs - use dynamic labels based on KPI type
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12, tab13 = st.tabs([
        "Overview",
        f"Channel {eff_label}",
        f"Marginal {eff_label} & Priority",
        "Executive Summary",
        "Bayesian Significance",
        "Diagnostics",
        "Time Series",
        "Export",
        "Visualizations",
        "Model Coefficients",
        "Media Curves",
        "Owned Media",
        f"{eff_label} Prior Validation"
    ])

    # =========================================================================
    # Tab 1: Overview
    # =========================================================================
    with tab1:
        st.subheader("Model Overview")

        # Key metrics
        fit_stats = wrapper.get_fit_statistics()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("R²", f"{fit_stats['r2']:.2f}")
        with col2:
            st.metric("MAPE", f"{fit_stats['mape']:.2f}%")
        with col3:
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
        st.plotly_chart(fig, width="stretch")

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
            # Include owned media categories (they're treated as channels in ContributionAnalyzer)
            owned_media_categories = config.get_owned_media_category_map(selected_category_col)
            channel_categories.update(owned_media_categories)
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
            st.plotly_chart(fig_grouped, width="stretch")

        st.dataframe(
            grouped[["group", "contribution_real", "pct_of_total"]].rename(columns={
                "group": "Group",
                "contribution_real": "Contribution ($)",
                "pct_of_total": "% of Total"
            }),
            width="stretch",
            hide_index=True,
        )

    # =========================================================================
    # Tab 2: Channel Efficiency (ROI or Cost Per X)
    # =========================================================================
    with tab2:
        st.subheader(f"Channel {eff_label} Analysis")

        roi_df = contributions.get_channel_roi(roi_channels=config.get_channel_columns())

        if len(roi_df) > 0:
            # Category column selector for efficiency grouping
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

                # Convert to display values for count KPIs (efficiency -> cost-per)
                if kpi_labels.is_revenue_type:
                    category_roi["roi_display"] = category_roi["roi"]
                else:
                    category_roi["roi_display"] = category_roi["roi"].apply(
                        lambda x: kpi_labels.convert_internal_to_display(x) if x > 0 else 0
                    )

                category_roi = category_roi.sort_values("roi_display", ascending=kpi_labels.is_revenue_type is False)

                fig = px.bar(
                    category_roi,
                    x="category",
                    y="roi_display",
                    title=f"{eff_label} by Category",
                    labels={"category": "Category", "roi_display": eff_label},
                    color="category",
                    color_discrete_map=CATEGORY_COLORS,
                )
                fig.update_layout(showlegend=False, xaxis_tickangle=-45)
                st.plotly_chart(fig, width="stretch")

                # Category table
                display_df = category_roi[["category", "spend_real", "contribution_real", "roi_display"]].copy()
                display_df.columns = ["Category", "Spend ($)", "Contribution ($)", eff_label]
            else:
                # Convert to display values for count KPIs (efficiency -> cost-per)
                if kpi_labels.is_revenue_type:
                    roi_df["roi_display"] = roi_df["roi"]
                else:
                    roi_df["roi_display"] = roi_df["roi"].apply(
                        lambda x: kpi_labels.convert_internal_to_display(x) if x > 0 else 0
                    )

                # Efficiency bar chart by channel, colored by category
                fig = px.bar(
                    roi_df.sort_values("roi_display", ascending=kpi_labels.is_revenue_type is False),
                    x="display_name",
                    y="roi_display",
                    title=f"{eff_label} by Channel (colored by Category)",
                    labels={"display_name": "Channel", "roi_display": eff_label},
                    color="category",
                    color_discrete_map=CATEGORY_COLORS,
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, width="stretch")

                # Channel table
                display_df = roi_df[["display_name", "category", "spend_real", "contribution_real", "roi_display"]].copy()
                display_df.columns = ["Channel", "Category", "Spend ($)", "Contribution ($)", eff_label]

            # Format table
            st.subheader("Details")
            display_df["Spend ($)"] = display_df["Spend ($)"].apply(lambda x: f"${x:,.0f}")
            display_df["Contribution ($)"] = display_df["Contribution ($)"].apply(lambda x: f"${x:,.0f}")
            display_df[eff_label] = display_df[eff_label].apply(lambda x: f"{x:.2f}")

            st.dataframe(display_df, width="stretch", hide_index=True)

            # Spend vs Contribution scatter
            target_label = kpi_labels.target_name if not kpi_labels.is_revenue_type else "Contribution"
            st.subheader(f"Spend vs {target_label}")

            fig2 = px.scatter(
                roi_df,
                x="spend_real",
                y="contribution_real",
                text="display_name",
                title=f"Spend vs {target_label} by Channel",
                labels={
                    "spend_real": "Total Spend ($)",
                    "contribution_real": f"Total {target_label}" + (" ($)" if kpi_labels.is_revenue_type else "")
                },
                color="category",
                color_discrete_map=CATEGORY_COLORS,
            )
            fig2.update_traces(textposition="top center")

            # Add break-even line (only for revenue KPIs where ROI=1 is meaningful)
            if kpi_labels.is_revenue_type:
                max_val = max(roi_df["spend_real"].max(), roi_df["contribution_real"].max())
                fig2.add_trace(go.Scatter(
                    x=[0, max_val],
                    y=[0, max_val],
                    mode="lines",
                    name=f"Break-even ({eff_label}=1)",
                    line=dict(dash="dash", color="gray")
                ))

            st.plotly_chart(fig2, width="stretch")

    # =========================================================================
    # Tab 3: Marginal Efficiency & Investment Priority
    # =========================================================================
    with tab3:
        st.subheader(f"Marginal {eff_label} & Investment Priority")

        st.markdown(f"""
        This analysis uses **saturation curve derivatives** to calculate the true marginal {eff_label}
        (return on the *next* dollar spent), not just the average {eff_label}.
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

            # Convert ROI values for count KPIs (efficiency -> cost-per)
            if kpi_labels.is_revenue_type:
                display_df['current_roi'] = display_df['current_roi'].apply(lambda x: f"{x:.2f}")
                display_df['marginal_roi'] = display_df['marginal_roi'].apply(lambda x: f"{x:.2f}")
            else:
                display_df['current_roi'] = display_df['current_roi'].apply(
                    lambda x: f"{kpi_labels.convert_internal_to_display(x):.2f}" if x > 0 else "N/A"
                )
                display_df['marginal_roi'] = display_df['marginal_roi'].apply(
                    lambda x: f"{kpi_labels.convert_internal_to_display(x):.2f}" if x > 0 else "N/A"
                )

            display_df['breakeven_spend'] = display_df['breakeven_spend'].apply(
                lambda x: f"${x:,.0f}" if x is not None else "N/A"
            )
            display_df['headroom_amount'] = display_df['headroom_amount'].apply(lambda x: f"${x:,.0f}")

            styled_df = display_df[['channel', 'current_spend', 'current_roi', 'marginal_roi',
                                     'priority_rank', 'breakeven_spend', 'headroom_amount', 'action', 'needs_test']]
            styled_df.columns = ['Channel', 'Current Spend', f'Current {eff_label}', f'Marginal {eff_label}',
                                 'Priority', 'Breakeven Spend', 'Headroom', 'Action', 'Needs Test']

            st.dataframe(styled_df, width="stretch", hide_index=True)

            st.markdown(f"""
            **Key Concepts:**
            - **Marginal {eff_label}**: Return on the *next* dollar spent (from saturation curve derivative)
            - **Breakeven Spend**: Spend level where marginal {eff_label} = $1.00
            - **Headroom**: Additional spend available before hitting breakeven
            - **Needs Test**: High uncertainty in {eff_label} estimate - validate with incrementality test
            """)

            # Visualizations
            st.markdown("---")
            st.subheader(f"Current vs Marginal {eff_label}")

            result = marginal_analyzer.run_full_analysis()
            channel_names = [ch.channel_name for ch in result.channel_analysis]

            # Convert ROI values for count KPIs (efficiency -> cost-per)
            if kpi_labels.is_revenue_type:
                current_rois = [ch.current_roi for ch in result.channel_analysis]
                marginal_rois = [ch.marginal_roi for ch in result.channel_analysis]
                breakeven_val = 1
            else:
                current_rois = [
                    kpi_labels.convert_internal_to_display(ch.current_roi) if ch.current_roi > 0 else 0
                    for ch in result.channel_analysis
                ]
                marginal_rois = [
                    kpi_labels.convert_internal_to_display(ch.marginal_roi) if ch.marginal_roi > 0 else 0
                    for ch in result.channel_analysis
                ]
                breakeven_val = 1  # $1 cost per target is still breakeven

            fig = go.Figure()
            fig.add_trace(go.Bar(
                name=f'Current (Avg) {eff_label}',
                x=channel_names,
                y=current_rois,
                marker_color='steelblue'
            ))
            fig.add_trace(go.Bar(
                name=f'Marginal {eff_label}',
                x=channel_names,
                y=marginal_rois,
                marker_color='orange'
            ))
            fig.add_hline(y=breakeven_val, line_dash="dash", line_color="red", annotation_text="Breakeven")
            fig.update_layout(
                barmode='group',
                title=f"Current {eff_label} vs Marginal {eff_label} by Channel",
                xaxis_tickangle=-45
            )
            st.plotly_chart(fig, width="stretch")

        except Exception as e:
            st.error(f"Error running marginal {eff_label} analysis: {str(e)}")
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
                st.metric(f"Portfolio {eff_label}", f"${summary['portfolio']['portfolio_roi']:.2f}")

            st.markdown("---")

            # Recommendations Summary
            st.markdown("### Investment Recommendations")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("INCREASE", summary['counts']['increase'], delta=f"High marginal {eff_label}")
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
                    st.markdown(f"- **{ch['channel_name']}**: Marginal {eff_label} ${ch['marginal_roi']:.2f}, "
                               f"Headroom ${ch['headroom_amount']:,.0f}{test_note}")

            if summary['recommendations']['hold']:
                st.markdown("**HOLD Steady:**")
                for ch in summary['recommendations']['hold']:
                    st.markdown(f"- **{ch['channel_name']}**: Marginal {eff_label} ${ch['marginal_roi']:.2f}")

            if summary['recommendations']['reduce']:
                st.markdown("**REDUCE/Reallocate:**")
                for ch in summary['recommendations']['reduce']:
                    st.markdown(f"- **{ch['channel_name']}**: Marginal {eff_label} ${ch['marginal_roi']:.2f} *(below breakeven)*")

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

        # Create analyzer (paid media only)
        try:
            sig_analyzer = BayesianSignificanceAnalyzer(
                idata=wrapper.idata,
                df_scaled=wrapper.df_scaled,
                channel_cols=config.get_channel_columns(),  # Paid media only
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
                f"{eff_label} Posteriors",
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
                st.dataframe(ci_df, width="stretch", hide_index=True)

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
                st.plotly_chart(fig, width="stretch")

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
                st.dataframe(pd_df, width="stretch", hide_index=True)

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
                st.plotly_chart(fig, width="stretch")

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
                st.dataframe(rope_df, width="stretch", hide_index=True)

            # -----------------------------------------------------------------
            # Efficiency Posteriors (ROI or Cost Per)
            # -----------------------------------------------------------------
            with sig_tab4:
                st.markdown(f"### {eff_label} Credible Intervals")
                st.markdown(f"Full posterior distribution of {eff_label} per channel with uncertainty.")

                roi_data = []
                for roi in sig_report.roi_posteriors:
                    # Convert efficiency to display value (inverts for count KPIs)
                    display_mean = kpi_labels.convert_internal_to_display(roi.roi_mean)
                    display_median = kpi_labels.convert_internal_to_display(roi.roi_median)
                    display_5pct = kpi_labels.convert_internal_to_display(roi.roi_5pct)
                    display_95pct = kpi_labels.convert_internal_to_display(roi.roi_95pct)

                    # For count KPIs, inversion reverses ordering, so swap 5/95 percentiles
                    # to maintain proper CI semantics (5% = lower bound, 95% = upper bound)
                    if not kpi_labels.is_revenue_type:
                        display_5pct, display_95pct = display_95pct, display_5pct

                    roi_data.append({
                        "Channel": roi.channel,
                        f"{eff_label} Mean": f"{display_mean:.2f}",
                        f"{eff_label} Median": f"{display_median:.2f}",
                        f"{eff_label} 5%": f"{display_5pct:.2f}",
                        f"{eff_label} 95%": f"{display_95pct:.2f}",
                        "Significant": "Yes" if roi.significant else "No"
                    })

                roi_df = pd.DataFrame(roi_data)
                st.dataframe(roi_df, width="stretch", hide_index=True)

                # Visualization
                st.markdown("---")
                channels = [roi.channel for roi in sig_report.roi_posteriors]

                # Convert values for chart display
                if kpi_labels.is_revenue_type:
                    roi_means = [roi.roi_mean for roi in sig_report.roi_posteriors]
                    roi_5s = [roi.roi_5pct for roi in sig_report.roi_posteriors]
                    roi_95s = [roi.roi_95pct for roi in sig_report.roi_posteriors]
                else:
                    # Invert for cost-per display, swap 5/95 to maintain proper CI bounds
                    roi_means = [1/roi.roi_mean if roi.roi_mean > 0 else 0 for roi in sig_report.roi_posteriors]
                    roi_5s = [1/roi.roi_95pct if roi.roi_95pct > 0 else 0 for roi in sig_report.roi_posteriors]
                    roi_95s = [1/roi.roi_5pct if roi.roi_5pct > 0 else 0 for roi in sig_report.roi_posteriors]

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
                fig.add_vline(x=1, line_dash="dash", line_color="green", annotation_text=f"{eff_label}=1")
                fig.update_layout(
                    title=f"{eff_label} Posteriors with 90% CI",
                    xaxis_title=eff_label,
                    yaxis_title="Channel",
                    height=max(400, len(channels) * 40)
                )
                st.plotly_chart(fig, width="stretch")

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
                        f"Prior {eff_label}": f"{sens.prior_roi:.2f}",
                        f"Posterior {eff_label}": f"{sens.posterior_roi:.2f}",
                        "Shift": f"{sens.shift:+.2f}",
                        "Data Influence": sens.data_influence
                    })

                sens_df = pd.DataFrame(sens_data)
                st.dataframe(sens_df, width="stretch", hide_index=True)

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
                    title=f"Prior vs Posterior {eff_label}",
                    xaxis_title=f"Prior {eff_label}",
                    yaxis_title=f"Posterior {eff_label}",
                    height=500
                )
                st.plotly_chart(fig, width="stretch")

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
        st.dataframe(worst, width="stretch", hide_index=True)

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
                width="stretch",
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
        st.plotly_chart(fig, width="stretch")

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
        st.plotly_chart(fig2, width="stretch")

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

        # Channel contributions over time (paid media only)
        st.subheader("Channel Contributions Over Time")

        channel_cols = config.get_channel_columns()  # Paid media only
        channel_contribs = contribs[[c for c in channel_cols if c in contribs.columns]] * config.data.revenue_scale

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
        st.plotly_chart(fig3, width="stretch")

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
                channel_cols=config.get_channel_columns(),  # Paid media only
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
                # Get contributions by component (paid media only)
                contribs_df = wrapper.get_contributions()
                channel_cols = config.get_channel_columns()  # Paid media only

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
                channel_cols = config.get_channel_columns()  # Paid media only
                dates = contribs_df.index
                channel_contribs = contribs_df[[c for c in channel_cols if c in contribs_df.columns]] * config.data.revenue_scale
                fig = create_stacked_contributions_area(dates, channel_contribs)

            elif viz_option == "ROI vs Spend (Bubble)":
                roi_df = contributions.get_channel_roi()
                metrics = []
                for _, row in roi_df.iterrows():
                    # Convert ROI for count KPIs (efficiency -> cost-per)
                    if kpi_labels.is_revenue_type:
                        display_roi = row['roi']
                    else:
                        display_roi = kpi_labels.convert_internal_to_display(row['roi']) if row['roi'] > 0 else 0

                    metrics.append({
                        'channel': row['channel'].replace('_spend', ''),
                        'roi': display_roi,
                        'spend': row['spend_real'],
                        'contribution': row['contribution_real'],
                    })
                fig = create_roi_effectiveness_bubble(metrics)

            elif viz_option == "Response Curves":
                # Get response curve data from the model (paid media only)
                channel_cols = config.get_channel_columns()  # Paid media only
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

            # Channels (saturation_beta) - paid media
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

                # Owned media (also in saturation_beta, after paid channels)
                owned_media_cols = config.get_owned_media_columns()
                paid_channel_count = len(channel_cols)
                for i, om in enumerate(owned_media_cols):
                    try:
                        idx = paid_channel_count + i
                        row = beta_summary.iloc[idx]
                        hdi_low = row["hdi_2.5%"]
                        hdi_high = row["hdi_97.5%"]
                        significant = "Yes" if (hdi_low > 0 or hdi_high < 0) else "No"
                        variables_data.append({
                            "Variable": om,
                            "Type": "Owned Media",
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
            st.dataframe(df_vars, width="stretch", hide_index=True)

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
                # Get channel info (paid media only)
                channel_cols = config.get_channel_columns()  # Paid media only

                # Build display names (paid media only)
                display_names = {}
                for ch_config in config.channels:
                    display_names[ch_config.name] = ch_config.get_display_name()

                channel_display_names = [display_names.get(ch, ch) for ch in channel_cols]

                # Controls row
                ctrl_col1, ctrl_col2 = st.columns([1, 2])

                with ctrl_col1:
                    view_options = ["Saturation Curves", "ROI Curves"]
                    if has_adstock:
                        view_options.append("Adstock Decay")
                    view_type = st.radio("View", view_options, horizontal=True)

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
                elif view_type == "ROI Curves":
                    _show_roi_curves(
                        wrapper, config, channel_cols, display_names,
                        selected_channel, spend_scale, revenue_scale
                    )
                else:
                    _show_adstock_curves(
                        wrapper, config, channel_cols, display_names,
                        selected_channel
                    )

    # =========================================================================
    # Tab 12: Owned Media
    # =========================================================================
    with tab12:
        st.header("Owned Media Analysis")

        owned_media_cols = config.get_owned_media_columns()

        if not owned_media_cols:
            st.info("No owned media configured for this model.")
        else:
            # Build display names for owned media
            om_display_names = {
                om.name: om.get_display_name() for om in config.owned_media
            }

            # Sub-tabs for organized analysis
            om_tab1, om_tab2, om_tab3, om_tab4 = st.tabs([
                "Contributions",
                "Coefficients",
                "Response Curves",
                "Significance"
            ])

            # -----------------------------------------------------------------
            # Sub-tab 1: Contributions
            # -----------------------------------------------------------------
            with om_tab1:
                st.subheader("Owned Media Contributions Over Time")

                try:
                    contribs_df = wrapper.get_contributions()
                    om_contribs = contribs_df[[c for c in owned_media_cols if c in contribs_df.columns]] * config.data.revenue_scale

                    if om_contribs.empty or om_contribs.shape[1] == 0:
                        st.warning("No contribution data available for owned media.")
                    else:
                        fig = go.Figure()
                        for om in owned_media_cols:
                            if om in om_contribs.columns:
                                display_name = om_display_names.get(om, om)
                                fig.add_trace(go.Scatter(
                                    x=contribs_df.index,
                                    y=om_contribs[om],
                                    name=display_name,
                                    stackgroup="one"
                                ))

                        fig.update_layout(
                            title="Stacked Owned Media Contributions",
                            xaxis_title="Date",
                            yaxis_title=f"Contribution ({config.data.target_column})",
                            hovermode="x unified",
                            template="plotly_dark",
                        )
                        st.plotly_chart(fig, width="stretch")

                        # Summary table
                        st.subheader("Contribution Summary")
                        summary_data = []
                        for om in owned_media_cols:
                            if om in contribs_df.columns:
                                total_contrib = contribs_df[om].sum() * config.data.revenue_scale
                                avg_contrib = contribs_df[om].mean() * config.data.revenue_scale
                                summary_data.append({
                                    "Owned Media": om_display_names.get(om, om),
                                    "Total Contribution": f"${total_contrib:,.0f}",
                                    "Avg Weekly": f"${avg_contrib:,.0f}",
                                })
                        if summary_data:
                            st.dataframe(pd.DataFrame(summary_data), width="stretch")

                except Exception as e:
                    st.error(f"Error loading contributions: {e}")

            # -----------------------------------------------------------------
            # Sub-tab 2: Coefficients
            # -----------------------------------------------------------------
            with om_tab2:
                st.subheader("Owned Media Coefficients")

                try:
                    idata = wrapper.idata
                    variables_data = []

                    # Owned media coefficients are in saturation_beta after paid media
                    if "saturation_beta" in idata.posterior:
                        beta_summary = az.summary(idata, var_names=["saturation_beta"], hdi_prob=0.95)
                        paid_channel_count = len(config.get_channel_columns())

                        for i, om in enumerate(owned_media_cols):
                            try:
                                # Owned media starts after paid channels in the index
                                idx = paid_channel_count + i
                                row = beta_summary.iloc[idx]
                                hdi_low = row["hdi_2.5%"]
                                hdi_high = row["hdi_97.5%"]
                                significant = "Yes" if (hdi_low > 0 or hdi_high < 0) else "No"
                                variables_data.append({
                                    "Variable": om_display_names.get(om, om),
                                    "Type": "Owned Media",
                                    "Coef Mean": f"{row['mean']:.4f}",
                                    "Coef Std": f"{row['sd']:.4f}",
                                    "95% HDI": f"[{hdi_low:.4f}, {hdi_high:.4f}]",
                                    "Significant": significant
                                })
                            except Exception:
                                pass

                    if variables_data:
                        coef_df = pd.DataFrame(variables_data)
                        st.dataframe(coef_df, width="stretch")
                    else:
                        st.info("No coefficient data available for owned media.")

                except Exception as e:
                    st.error(f"Error loading coefficients: {e}")

            # -----------------------------------------------------------------
            # Sub-tab 3: Response Curves
            # -----------------------------------------------------------------
            with om_tab3:
                st.subheader("Owned Media Saturation Curves")

                try:
                    idata = wrapper.idata
                    posterior = idata.posterior

                    has_saturation = "saturation_lam" in posterior and "saturation_beta" in posterior
                    has_adstock = "adstock_alpha" in posterior

                    if not has_saturation:
                        st.warning("Saturation parameters not found in model posterior.")
                    else:
                        # Controls
                        view_options = ["Saturation Curves"]
                        if has_adstock:
                            view_options.append("Adstock Decay")
                        view_type = st.radio("View", view_options, horizontal=True, key="om_view_type")

                        om_display_names_list = [om_display_names.get(om, om) for om in owned_media_cols]
                        channel_options = ["All Owned Media"] + om_display_names_list
                        selected_om = st.selectbox("Owned Media", channel_options, key="om_selected")

                        st.markdown("---")

                        spend_scale = config.data.spend_scale
                        revenue_scale = config.data.revenue_scale
                        paid_channel_count = len(config.get_channel_columns())

                        if view_type == "Saturation Curves":
                            _show_owned_media_saturation_curves(
                                wrapper, config, owned_media_cols, om_display_names,
                                selected_om, spend_scale, revenue_scale, paid_channel_count
                            )
                        else:
                            _show_owned_media_adstock_curves(
                                wrapper, config, owned_media_cols, om_display_names,
                                selected_om, paid_channel_count
                            )

                except Exception as e:
                    st.error(f"Error loading response curves: {e}")

            # -----------------------------------------------------------------
            # Sub-tab 4: Significance
            # -----------------------------------------------------------------
            with om_tab4:
                st.subheader("Bayesian Significance Analysis")

                try:
                    # Build prior ROIs for owned media (if applicable)
                    om_prior_rois = {}
                    for om_config in config.owned_media:
                        if om_config.include_roi and om_config.roi_prior_mid:
                            om_prior_rois[om_config.name] = om_config.roi_prior_mid

                    sig_analyzer = BayesianSignificanceAnalyzer(
                        idata=wrapper.idata,
                        df_scaled=wrapper.df_scaled,
                        channel_cols=owned_media_cols,
                        target_col=config.data.target_column,
                        prior_rois=om_prior_rois,
                    )

                    sig_report = sig_analyzer.run_full_analysis()

                    # Display summary
                    if sig_report.channel_results:
                        sig_data = []
                        for channel, result in sig_report.channel_results.items():
                            display_name = om_display_names.get(channel, channel)
                            sig_data.append({
                                "Owned Media": display_name,
                                "Effect Mean": f"{result.effect_mean:.4f}",
                                "95% CI": f"[{result.ci_lower:.4f}, {result.ci_upper:.4f}]",
                                "Prob. Direction": f"{result.prob_direction:.1%}",
                                "Significant": "Yes" if result.is_significant else "No"
                            })
                        sig_df = pd.DataFrame(sig_data)
                        st.dataframe(sig_df, width="stretch")
                    else:
                        st.info("No significance data available.")

                except Exception as e:
                    st.error(f"Error running significance analysis: {e}")

    # =========================================================================
    # Tab 13: Efficiency Prior Validation (ROI or Cost Per)
    # =========================================================================
    with tab13:
        st.subheader(f"{eff_label} Prior Validation")
        st.caption(f"Compares your {eff_label} beliefs (priors) against what the model learned (posterior)")

        try:
            roi_report = quick_roi_diagnostics(wrapper)

            # Overall health status
            health = roi_report.overall_health
            if "Good" in health:
                st.success(f"**{health}**")
            elif "Moderate" in health:
                st.warning(f"**{health}**")
            else:
                st.error(f"**{health}**")

            # Explanation
            with st.expander("What does this mean?", expanded=False):
                st.markdown(f"""
                This validates whether the {eff_label} the model learned matches your prior beliefs:

                - **Prior {eff_label}**: The {eff_label} range you specified when configuring channels
                - **Posterior {eff_label}**: The {eff_label} the model learned from data (with 90% HDI)
                - **Prior in HDI**: ✅ if your belief falls within the learned range
                - **{eff_label} Shift**: How much the learned {eff_label} differs from your prior belief
                - **λ Shift**: How much the saturation curve changed from initial assumptions

                **Large shifts** may indicate:
                - Your prior beliefs need updating based on this data
                - Data quality issues for that channel
                - The model found a different relationship than expected
                """)

            # Summary table
            if roi_report.channel_results:
                roi_df = roi_report.to_dataframe()

                # Convert values for display (invert for count KPIs)
                def convert_for_display(val):
                    return kpi_labels.convert_internal_to_display(val)

                if kpi_labels.is_revenue_type:
                    # Revenue KPI: display raw values
                    display_df = pd.DataFrame({
                        "Channel": roi_df["channel"],
                        f"Prior {eff_label} (Low-Mid-High)": roi_df.apply(
                            lambda r: f"{r['prior_roi_low']:.1f} - {r['prior_roi_mid']:.1f} - {r['prior_roi_high']:.1f}",
                            axis=1
                        ),
                        f"Posterior {eff_label} [90% HDI]": roi_df.apply(
                            lambda r: f"{r['posterior_roi_mean']:.2f} [{r['posterior_roi_hdi_low']:.2f}, {r['posterior_roi_hdi_high']:.2f}]",
                            axis=1
                        ),
                        "Prior in HDI": roi_df["prior_in_hdi"].apply(lambda x: "✅" if x else "⚠️"),
                        f"{eff_label} Shift": roi_df["roi_shift_pct"].apply(lambda x: f"{x:+.0%}" if pd.notna(x) else "-"),
                        "λ Shift": roi_df["lambda_shift_pct"].apply(lambda x: f"{x:+.0%}" if pd.notna(x) else "-"),
                    })
                else:
                    # Count KPI: convert efficiency to cost-per for display
                    # For priors: low efficiency = high cost, so swap order for display
                    display_df = pd.DataFrame({
                        "Channel": roi_df["channel"],
                        f"Prior {eff_label} (Low-Mid-High)": roi_df.apply(
                            lambda r: f"{convert_for_display(r['prior_roi_high']):.1f} - {convert_for_display(r['prior_roi_mid']):.1f} - {convert_for_display(r['prior_roi_low']):.1f}",
                            axis=1
                        ),
                        f"Posterior {eff_label} [90% HDI]": roi_df.apply(
                            lambda r: f"{convert_for_display(r['posterior_roi_mean']):.2f} [{convert_for_display(r['posterior_roi_hdi_high']):.2f}, {convert_for_display(r['posterior_roi_hdi_low']):.2f}]",
                            axis=1
                        ),
                        "Prior in HDI": roi_df["prior_in_hdi"].apply(lambda x: "✅" if x else "⚠️"),
                        f"{eff_label} Shift": roi_df["roi_shift_pct"].apply(lambda x: f"{x:+.0%}" if pd.notna(x) else "-"),
                        "λ Shift": roi_df["lambda_shift_pct"].apply(lambda x: f"{x:+.0%}" if pd.notna(x) else "-"),
                    })

                st.dataframe(display_df, hide_index=True, width='stretch')

            # Warnings
            if roi_report.channels_with_prior_tension:
                st.warning(f"**Channels with prior tension:** {', '.join(roi_report.channels_with_prior_tension)}")

            if roi_report.channels_with_large_shift:
                st.warning(f"**Channels with large {eff_label} shift (>50%):** {', '.join(roi_report.channels_with_large_shift)}")

            # Recommendations
            if roi_report.recommendations:
                st.subheader("Recommendations")
                for rec in roi_report.recommendations:
                    st.info(rec)

            # Detailed Prior Diagnostics (compact table view)
            with st.expander("Detailed Prior Diagnostics", expanded=False):
                st.caption("Parameter shifts between prior beliefs and posterior learnings")

                # Build table data
                table_data = []
                warnings_data = []

                for channel, result in roi_report.channel_results.items():
                    row = {
                        "Channel": channel,
                        "β Shift": f"{result.beta_shift:+.1%}" if result.beta_shift is not None else "N/A",
                        "Prior β": f"{result.prior_beta_median:.6f}" if result.prior_beta_median is not None else "N/A",
                        "Post β": f"{result.posterior_beta_mean:.6f}" if result.posterior_beta_mean is not None else "N/A",
                        "λ Shift": f"{result.lambda_shift:+.1%}" if result.lambda_shift is not None else "N/A",
                        "Spend %": f"{result.channel_spend_pct:.1%}" if result.channel_spend_pct is not None else "N/A",
                        "Filtered %": f"{result.samples_filtered_pct:.1%}" if result.samples_filtered_pct is not None else "N/A",
                    }

                    # Add ROI percentiles
                    if result.raw_sample_percentiles:
                        p = result.raw_sample_percentiles
                        row["P5"] = f"{p.get(5, 0):.2f}"
                        row["P25"] = f"{p.get(25, 0):.2f}"
                        row["Median"] = f"{p.get(50, 0):.2f}"
                        row["P75"] = f"{p.get(75, 0):.2f}"
                        row["P95"] = f"{p.get(95, 0):.2f}"
                    else:
                        row["P5"] = "N/A"
                        row["P25"] = "N/A"
                        row["Median"] = "N/A"
                        row["P75"] = "N/A"
                        row["P95"] = "N/A"

                    table_data.append(row)

                    # Collect warnings separately
                    if result.warnings:
                        for w in result.warnings:
                            warnings_data.append({"Channel": channel, "Warning": w})

                # Display table
                df_diagnostics = pd.DataFrame(table_data)
                st.dataframe(df_diagnostics, use_container_width=True, hide_index=True)

                # Show warnings if any
                if warnings_data:
                    st.markdown("#### Warnings")
                    for item in warnings_data:
                        st.warning(f"**{item['Channel']}:** {item['Warning']}")

        except Exception as e:
            st.info(f"ROI prior validation not available: {str(e)}")


def _show_owned_media_saturation_curves(wrapper, config, owned_media_cols, display_names, selected_om, spend_scale, revenue_scale, paid_channel_count):
    """Show saturation curves for owned media."""
    posterior = wrapper.idata.posterior

    # Time period selector
    col1, col2 = st.columns([1, 3])
    with col1:
        time_period = st.selectbox(
            "Time Period",
            ["Weekly", "Yearly"],
            index=0,
            key="om_time_period",
            help="Weekly shows response per week. Yearly multiplies by 52 weeks."
        )
    time_multiplier = 52 if time_period == "Yearly" else 1
    period_label = "Yearly" if time_period == "Yearly" else "Weekly"

    # Get saturation parameters
    lam_mean = posterior["saturation_lam"].mean(dim=["chain", "draw"]).values
    beta_mean = posterior["saturation_beta"].mean(dim=["chain", "draw"]).values

    fig = go.Figure()

    for i, om in enumerate(owned_media_cols):
        display_name = display_names.get(om, om)

        # Skip if not selected and not "All"
        if selected_om != "All Owned Media" and display_name != selected_om:
            continue

        try:
            # Index for owned media (after paid channels)
            idx = paid_channel_count + i
            lam = float(lam_mean[idx])
            beta = float(beta_mean[idx])

            # Get spend range from data
            if om in wrapper.df_raw.columns:
                max_spend = float(wrapper.df_raw[om].max()) * spend_scale
            else:
                max_spend = 10000  # Default

            # Generate curve
            x_spend = np.linspace(0, max_spend * 1.5, 100)
            x_scaled = x_spend / spend_scale
            y_sat = _logistic_saturation(x_scaled, lam) * beta * revenue_scale * time_multiplier

            fig.add_trace(go.Scatter(
                x=x_spend,
                y=y_sat,
                name=display_name,
                mode="lines"
            ))
        except Exception:
            pass

    fig.update_layout(
        title=f"Owned Media Saturation Curves ({period_label})",
        xaxis_title="Activity Level ($)",
        yaxis_title=f"Response ({config.data.target_column})",
        hovermode="x unified",
        template="plotly_dark",
    )
    st.plotly_chart(fig, width="stretch")


def _show_owned_media_adstock_curves(wrapper, config, owned_media_cols, display_names, selected_om, paid_channel_count):
    """Show adstock decay curves for owned media."""
    posterior = wrapper.idata.posterior

    if "adstock_alpha" not in posterior:
        st.warning("Adstock parameters not found.")
        return

    alpha_mean = posterior["adstock_alpha"].mean(dim=["chain", "draw"]).values

    fig = go.Figure()
    max_periods = 12

    for i, om in enumerate(owned_media_cols):
        display_name = display_names.get(om, om)

        if selected_om != "All Owned Media" and display_name != selected_om:
            continue

        try:
            idx = paid_channel_count + i
            alpha = float(alpha_mean[idx])

            periods = np.arange(max_periods + 1)
            decay = alpha ** periods

            fig.add_trace(go.Scatter(
                x=periods,
                y=decay,
                name=f"{display_name} (α={alpha:.3f})",
                mode="lines+markers"
            ))
        except Exception:
            pass

    fig.update_layout(
        title="Owned Media Adstock Decay",
        xaxis_title="Weeks Since Activity",
        yaxis_title="Remaining Effect (%)",
        hovermode="x unified",
        template="plotly_dark",
    )
    st.plotly_chart(fig, width="stretch")


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

        # Store raw data for validation (last 52 weeks or all if less)
        n_validation_weeks = min(52, n_periods)
        last_n_spend_scaled = spend_data[-n_validation_weeks:].sum()
        last_n_contrib_scaled = contribs[ch].iloc[-n_validation_weeks:].sum() if ch in contribs.columns else 0

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
            # Validation data
            'n_validation_weeks': n_validation_weeks,
            'last_n_spend_scaled': last_n_spend_scaled,
            'last_n_contrib_scaled': last_n_contrib_scaled,
            'spend_scale': spend_scale,
            'revenue_scale': revenue_scale,
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

    st.plotly_chart(fig, width="stretch")

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
            st.dataframe(pd.DataFrame(comparison_data), width="stretch", hide_index=True)
            if time_multiplier == 1:
                st.caption("*Avg Weekly Response = Total Contribution / Periods.*")
            else:
                st.caption("*Yearly values are weekly × 52.*")

    # Validation section - compare curve predictions vs actual 52-week data
    if len(channels_to_show) > 0:
        with st.expander("✅ Validate Curve Against Actual Data"):
            n_val_weeks = curves_data[0]['n_validation_weeks']
            st.write(f"**Validation Period:** Last {n_val_weeks} weeks")
            st.write("Compare what the curve predicts vs actual model contributions:")

            validation_data = []
            for curve in curves_data:
                # Get actual data from last N weeks
                last_n_spend = curve['last_n_spend_scaled'] * curve['spend_scale']
                last_n_contrib = curve['last_n_contrib_scaled'] * curve['revenue_scale']
                actual_roi = last_n_contrib / last_n_spend if last_n_spend > 0 else 0

                # Calculate what curve predicts for that avg weekly spend
                n_val = curve['n_validation_weeks']
                avg_weekly_spend_val = last_n_spend / n_val if n_val > 0 else 0
                avg_spend_normalized = avg_weekly_spend_val / (curve['x_max'] * curve['spend_scale'])
                avg_saturation = _logistic_saturation(avg_spend_normalized, curve['lam'])
                curve_weekly_prediction = curve['calibration_factor'] * avg_saturation
                curve_total_prediction = curve_weekly_prediction * n_val

                # Calculate difference
                pct_diff = ((curve_total_prediction - last_n_contrib) / last_n_contrib * 100) if last_n_contrib != 0 else 0

                validation_data.append({
                    "Channel": curve['display_name'],
                    f"Actual Spend ({n_val}wk)": f"${last_n_spend:,.0f}",
                    f"Actual Contribution ({n_val}wk)": f"${last_n_contrib:,.0f}",
                    "Actual ROI": f"{actual_roi:.2f}",
                    f"Curve Prediction ({n_val}wk)": f"${curve_total_prediction:,.0f}",
                    "Difference": f"{pct_diff:+.1f}%",
                })

            st.dataframe(pd.DataFrame(validation_data), width="stretch", hide_index=True)

            st.caption("*Differences are expected due to varying weekly spend and adstock effects. "
                      "The curve assumes constant weekly spend. Differences <15% are normal.*")

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
            if st.button("Add Scenario", width="stretch"):
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

            st.dataframe(pd.DataFrame(scenario_data), width="stretch", hide_index=True)


def _show_roi_curves(wrapper, config, channel_cols, display_names, selected_channel, spend_scale, revenue_scale):
    """Show Average ROI and Marginal ROI curves."""
    import plotly.graph_objects as go
    from mmm_platform.analysis.marginal_roi import logistic_saturation_derivative

    posterior = wrapper.idata.posterior

    # Controls row: Time period and ROI type toggles
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        time_period = st.selectbox(
            "Time Period",
            ["Weekly", "Yearly"],
            index=0,
            help="Weekly shows ROI per week. Yearly shows annualized ROI.",
            key="roi_time_period"
        )
    with col2:
        show_avg_roi = st.checkbox("Average ROI", value=True, key="show_avg_roi",
                                   help="Total return / total spend at each level")
        show_marginal_roi = st.checkbox("Marginal ROI", value=True, key="show_marginal_roi",
                                        help="Return on the next dollar spent")

    time_multiplier = 52 if time_period == "Yearly" else 1
    period_label = "Yearly" if time_period == "Yearly" else "Weekly"

    # Ensure at least one is selected
    if not show_avg_roi and not show_marginal_roi:
        st.warning("Please select at least one ROI type to display.")
        return

    # Determine which channels to show
    if selected_channel == "All Channels":
        channels_to_show = list(range(len(channel_cols)))
    else:
        channel_display_names = [display_names.get(ch, ch) for ch in channel_cols]
        idx = channel_display_names.index(selected_channel)
        channels_to_show = [idx]

    # Get contributions for calibration
    contribs = wrapper.get_contributions()
    n_periods = len(wrapper.df_scaled)

    # Generate ROI data for each channel
    curves_data = []
    for i in channels_to_show:
        ch = channel_cols[i]
        display_name = display_names.get(ch, ch)

        # Get posterior mean for lambda
        lam = float(posterior['saturation_lam'].mean(dim=['chain', 'draw']).values[i])

        # Get spend data
        spend_data = wrapper.df_scaled[ch].values
        x_max = spend_data.max()
        if x_max == 0:
            x_max = 1.0

        # Get actual values for calibration
        actual_contrib_total = contribs[ch].sum() * revenue_scale if ch in contribs.columns else 0
        avg_weekly_spend = spend_data.mean() * spend_scale
        avg_weekly_contrib = actual_contrib_total / n_periods if n_periods > 0 else 0

        # Calibration factor (same as saturation curves)
        avg_spend_normalized = avg_weekly_spend / (x_max * spend_scale)
        avg_saturation = _logistic_saturation(avg_spend_normalized, lam)
        calibration_factor = avg_weekly_contrib / avg_saturation if avg_saturation > 0.001 else avg_weekly_contrib

        # Generate spend range
        max_spend_real = x_max * spend_scale * 2.5
        spend_range_real = np.linspace(0.01, max_spend_real, 200)  # Avoid division by zero
        spend_range_normalized = spend_range_real / (x_max * spend_scale)

        # Calculate response at each spend level
        response_normalized = _logistic_saturation(spend_range_normalized, lam)
        response_weekly = calibration_factor * response_normalized

        # Average ROI = Response / Spend (same units, so no scale adjustment needed)
        avg_roi = response_weekly / spend_range_real

        # Marginal ROI = d(Response)/d(Spend)
        marginal_normalized = np.array([logistic_saturation_derivative(x, lam) for x in spend_range_normalized])
        marginal_roi = calibration_factor * marginal_normalized / (x_max * spend_scale)

        # Current position
        current_avg_roi = avg_weekly_contrib / avg_weekly_spend if avg_weekly_spend > 0 else 0
        current_marginal_normalized = logistic_saturation_derivative(avg_spend_normalized, lam)
        current_marginal_roi = calibration_factor * current_marginal_normalized / (x_max * spend_scale)

        curves_data.append({
            'channel': ch,
            'display_name': display_name,
            'spend_range': spend_range_real * time_multiplier,
            'avg_roi': avg_roi,
            'marginal_roi': marginal_roi,
            'current_spend': avg_weekly_spend * time_multiplier,
            'current_avg_roi': current_avg_roi,
            'current_marginal_roi': current_marginal_roi,
            'lam': lam,
            'calibration_factor': calibration_factor,
            'x_max': x_max,
        })

    # Create Plotly figure
    fig = go.Figure()

    colors = px.colors.qualitative.Set2

    for idx, curve in enumerate(curves_data):
        color = colors[idx % len(colors)]

        # Average ROI curve (solid) - if toggled on
        if show_avg_roi:
            fig.add_trace(go.Scatter(
                x=curve['spend_range'],
                y=curve['avg_roi'],
                mode='lines',
                name=f"{curve['display_name']} - Avg ROI" if show_marginal_roi else curve['display_name'],
                line=dict(color=color, width=3),
                hovertemplate=f"<b>{curve['display_name']} - Avg ROI</b><br>" +
                             f"{period_label} Spend: $%{{x:,.0f}}<br>Avg ROI: %{{y:.2f}}<extra></extra>"
            ))

        # Marginal ROI curve (dashed) - if toggled on
        if show_marginal_roi:
            fig.add_trace(go.Scatter(
                x=curve['spend_range'],
                y=curve['marginal_roi'],
                mode='lines',
                name=f"{curve['display_name']} - Marginal ROI" if show_avg_roi else curve['display_name'],
                line=dict(color=color, width=2, dash='dash'),
                hovertemplate=f"<b>{curve['display_name']} - Marginal ROI</b><br>" +
                             f"{period_label} Spend: $%{{x:,.0f}}<br>Marginal ROI: %{{y:.2f}}<extra></extra>"
            ))

        # Current position marker - show on whichever curve is visible
        if show_avg_roi:
            marker_y = curve['current_avg_roi']
            marker_label = "Avg ROI"
        else:
            marker_y = curve['current_marginal_roi']
            marker_label = "Marginal ROI"

        fig.add_trace(go.Scatter(
            x=[curve['current_spend']],
            y=[marker_y],
            mode='markers',
            name=f"{curve['display_name']} (Current)",
            marker=dict(size=12, color=color, symbol='circle',
                       line=dict(width=2, color='black')),
            hovertemplate=f"<b>{curve['display_name']} - Current</b><br>" +
                         f"Spend: $%{{x:,.0f}}<br>{marker_label}: %{{y:.2f}}<extra></extra>",
            showlegend=False
        ))

    # Breakeven line at ROI = 1.0
    fig.add_hline(y=1.0, line_dash="dot", line_color="red", opacity=0.7,
                  annotation_text="Breakeven (ROI = 1.0)", annotation_position="right")

    fig.update_layout(
        title=f"ROI Curves - {period_label} Spend",
        xaxis_title=f"{period_label} Spend ($)",
        yaxis_title="ROI ($ return per $ spent)",
        hovermode='closest',
        showlegend=True,
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, width="stretch")

    # Dynamic caption based on what's shown
    if show_avg_roi and show_marginal_roi:
        st.caption("*Solid lines show Average ROI (total return / total spend). "
                   "Dashed lines show Marginal ROI (return on next dollar). "
                   "When Marginal ROI < 1.0, additional spend loses money.*")
    elif show_avg_roi:
        st.caption("*Average ROI = total return / total spend at each level.*")
    else:
        st.caption("*Marginal ROI = return on the next dollar spent. "
                   "When Marginal ROI < 1.0, additional spend loses money.*")

    # Metrics for single channel
    if len(channels_to_show) == 1:
        curve = curves_data[0]

        st.markdown("---")
        st.subheader(f"ROI Analysis: {curve['display_name']}")

        col1, col2, col3 = st.columns(3)
        col1.metric("Current Avg ROI", f"${curve['current_avg_roi']:.2f}")
        col2.metric("Current Marginal ROI", f"${curve['current_marginal_roi']:.2f}")

        # Determine recommendation
        if curve['current_marginal_roi'] > 1.5:
            col3.metric("Recommendation", "INCREASE", delta="High marginal returns")
            st.success("**Strong opportunity to increase spend** - Marginal ROI is well above breakeven")
        elif curve['current_marginal_roi'] > 1.0:
            col3.metric("Recommendation", "HOLD", delta="Good returns")
            st.info("**Near optimal spend level** - Marginal ROI is positive but diminishing")
        else:
            col3.metric("Recommendation", "REDUCE", delta="Below breakeven", delta_color="inverse")
            st.warning("**Consider reducing spend** - Additional dollars lose money (Marginal ROI < 1.0)")


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

    st.plotly_chart(fig, width="stretch")

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

    st.dataframe(pd.DataFrame(summary_data), width="stretch", hide_index=True)

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

        col1, col2, col3 = st.columns(3)
        with col1:
            r2 = diagnostics.get("r2") or fit_stats.get("r2", "N/A")
            st.metric("R²", f"{r2:.2f}" if isinstance(r2, (int, float)) else r2)
        with col2:
            mape = diagnostics.get("mape") or fit_stats.get("mape", "N/A")
            st.metric("MAPE", f"{mape:.2f}%" if isinstance(mape, (int, float)) else mape)
        with col3:
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
            st.dataframe(roi_df, width="stretch", hide_index=True)

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
                st.plotly_chart(fig, width="stretch")
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

            st.dataframe(contrib_df, width="stretch", hide_index=True)

            # Pie chart
            fig = px.pie(
                contrib_df[contrib_df["Contribution"] > 0],
                values="Contribution",
                names="Channel",
                title="Contribution Breakdown"
            )
            st.plotly_chart(fig, width="stretch")

        # Time series
        time_series = contributions.get("time_series", [])
        if time_series:
            st.markdown("#### Contribution Time Series")
            ts_df = pd.DataFrame(time_series)
            st.dataframe(ts_df.head(20), width="stretch", hide_index=True)
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
        st.dataframe(channel_df, width="stretch", hide_index=True)

        # Contribution breakdown
        st.markdown("---")
        st.subheader("Contribution Breakdown")

        breakdown = demo._get_contribution_breakdown()
        fig = px.pie(
            values=[breakdown['baseline'], breakdown['channels']],
            names=['Base', 'All Channels'],
            title="Base vs Channels"
        )
        st.plotly_chart(fig, width="stretch")

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

        st.dataframe(styled_df, width="stretch", hide_index=True)

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
        st.plotly_chart(fig, width="stretch")

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

        st.dataframe(display_df, width="stretch", hide_index=True)

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
