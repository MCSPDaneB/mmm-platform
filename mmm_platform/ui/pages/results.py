"""
Results page for MMM Platform.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from mmm_platform.analysis.diagnostics import ModelDiagnostics
from mmm_platform.analysis.contributions import ContributionAnalyzer
from mmm_platform.analysis.reporting import ReportGenerator
from mmm_platform.analysis.bayesian_significance import (
    BayesianSignificanceAnalyzer,
    get_interpretation_guide,
)


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

    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Overview",
        "Channel ROI",
        "Bayesian Significance",
        "Diagnostics",
        "Time Series",
        "Export"
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
            st.metric("R²", f"{fit_stats['r2']:.3f}")
        with col2:
            st.metric("MAPE", f"{fit_stats['mape']:.1f}%")
        with col3:
            st.metric("RMSE", f"{fit_stats['rmse']:.1f}")
        with col4:
            st.metric("Fit Time", f"{fit_stats['fit_duration_seconds']:.1f}s")

        st.markdown("---")

        # Contribution breakdown
        st.subheader("Contribution Breakdown")

        breakdown = contributions.get_contribution_breakdown()

        # Pie chart
        breakdown_data = pd.DataFrame([
            {"Category": "Baseline", "Value": abs(breakdown["intercept"]["real_value"])},
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

        # Grouped contributions table
        grouped = contributions.get_grouped_contributions()
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
            # ROI bar chart
            fig = px.bar(
                roi_df,
                x="channel",
                y="roi",
                title="ROI by Channel",
                labels={"channel": "Channel", "roi": "ROI"},
                color="roi",
                color_continuous_scale="RdYlGn",
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

            # Detailed table
            st.subheader("Channel Details")

            display_df = roi_df[["channel", "spend_real", "contribution_real", "roi"]].copy()
            display_df.columns = ["Channel", "Spend ($)", "Contribution ($)", "ROI"]
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
                text="channel",
                title="Spend vs Contribution by Channel",
                labels={
                    "spend_real": "Total Spend ($)",
                    "contribution_real": "Total Contribution ($)"
                }
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
    # Tab 3: Bayesian Significance Analysis
    # =========================================================================
    with tab3:
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
                        "Mean": f"{ci.mean:.4f}",
                        "HDI Low": f"{ci.hdi_low:.4f}",
                        "HDI High": f"{ci.hdi_high:.4f}",
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
    # Tab 4: Diagnostics
    # =========================================================================
    with tab4:
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
            st.dataframe(
                control_df[["control", "contribution_real", "expected_sign", "actual_sign", "Status"]],
                use_container_width=True,
                hide_index=True
            )

    # =========================================================================
    # Tab 5: Time Series
    # =========================================================================
    with tab5:
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
            line=dict(color="black", width=2)
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
        )
        st.plotly_chart(fig, use_container_width=True)

        # Residuals
        st.subheader("Residuals")

        time_df["Residual"] = time_df["Actual"] - time_df["Fitted"]

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=time_df["Date"],
            y=time_df["Residual"],
            mode="lines",
            name="Residual"
        ))
        fig2.add_hline(y=0, line_dash="dash", line_color="gray")
        fig2.update_layout(
            title="Residuals Over Time",
            xaxis_title="Date",
            yaxis_title="Residual ($)",
        )
        st.plotly_chart(fig2, use_container_width=True)

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
        )
        st.plotly_chart(fig3, use_container_width=True)

    # =========================================================================
    # Tab 6: Export
    # =========================================================================
    with tab6:
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
            st.metric("R²", f"{r2:.3f}" if isinstance(r2, (int, float)) else r2)
        with col2:
            mape = diagnostics.get("mape") or fit_stats.get("mape", "N/A")
            st.metric("MAPE", f"{mape:.1f}%" if isinstance(mape, (int, float)) else mape)
        with col3:
            rmse = diagnostics.get("rmse") or fit_stats.get("rmse", "N/A")
            st.metric("RMSE", f"{rmse:.1f}" if isinstance(rmse, (int, float)) else rmse)
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
            names=['Baseline', 'All Channels'],
            title="Baseline vs Channels"
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
                "Baseline vs Channels (Donut)",
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

        if viz_option == "Baseline vs Channels (Donut)":
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
            contrib_dict['Baseline'] = float(demo.contribs['intercept'].sum())
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
