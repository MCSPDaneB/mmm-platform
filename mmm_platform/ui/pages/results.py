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

    # Check if model is fitted
    if not st.session_state.get("model_fitted") or st.session_state.get("current_model") is None:
        st.warning("Please run the model first!")
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
