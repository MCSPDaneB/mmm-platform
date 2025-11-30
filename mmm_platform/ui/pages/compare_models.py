"""
Model Comparison page for MMM Platform.

Provides side-by-side comparison of 2 fitted models.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, Optional

from mmm_platform.model.persistence import ModelPersistence, get_models_dir
from mmm_platform.analysis.contributions import ContributionAnalyzer


def show():
    """Show the model comparison page."""
    st.title("🔍 Model Comparison")

    # Check if we have models to compare
    if "active_comparison" not in st.session_state or not st.session_state.active_comparison:
        st.info(
            "No models selected for comparison. "
            "Go to **Saved Configs & Models** and select 2 models to compare."
        )

        # Show quick selection if models exist
        models_dir = get_models_dir()
        models = ModelPersistence.list_saved_models(models_dir)

        if len(models) >= 2:
            st.markdown("---")
            st.subheader("Quick Selection")

            col1, col2 = st.columns(2)

            model_options = {
                f"{m.get('config_name', 'Unknown')} ({m.get('created_at', '')[:10]})": m.get("path")
                for m in models
            }

            with col1:
                model_a_name = st.selectbox(
                    "Model A",
                    options=list(model_options.keys()),
                    key="quick_model_a"
                )

            with col2:
                model_b_name = st.selectbox(
                    "Model B",
                    options=list(model_options.keys()),
                    index=min(1, len(model_options) - 1),
                    key="quick_model_b"
                )

            if model_a_name != model_b_name:
                if st.button("Compare Selected Models", type="primary"):
                    st.session_state.active_comparison = [
                        model_options[model_a_name],
                        model_options[model_b_name]
                    ]
                    st.rerun()
            else:
                st.warning("Please select two different models")

        return

    # Load and compare models
    paths = st.session_state.active_comparison

    if len(paths) != 2:
        st.error("Expected 2 models for comparison")
        return

    # Load models
    with st.spinner("Loading models for comparison..."):
        try:
            data = _load_comparison_data(paths[0], paths[1])
        except Exception as e:
            st.error(f"Error loading models: {e}")
            if st.button("Clear Selection"):
                st.session_state.active_comparison = None
                st.session_state.models_to_compare = []
                st.rerun()
            return

    # Header with model names
    st.markdown(f"**Model A:** {data['model_a']['name']}  vs  **Model B:** {data['model_b']['name']}")

    # Clear comparison button
    if st.button("← Back to Saved Models"):
        st.session_state.active_comparison = None
        st.session_state.models_to_compare = []
        st.rerun()

    st.markdown("---")

    # Tabs for different comparison views
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Overview",
        "💰 Channel ROI",
        "📈 Contributions",
        "⚙️ Coefficients"
    ])

    with tab1:
        _show_overview_tab(data)

    with tab2:
        _show_channel_roi_tab(data)

    with tab3:
        _show_contributions_tab(data)

    with tab4:
        _show_coefficients_tab(data)


def _load_comparison_data(path_a: str, path_b: str) -> Dict[str, Any]:
    """Load both models and extract comparison data."""
    from mmm_platform.model.mmm import MMMWrapper

    wrapper_a = ModelPersistence.load(path_a, MMMWrapper)
    wrapper_b = ModelPersistence.load(path_b, MMMWrapper)

    # Use existing analyzers
    contrib_a = ContributionAnalyzer.from_mmm_wrapper(wrapper_a)
    contrib_b = ContributionAnalyzer.from_mmm_wrapper(wrapper_b)

    return {
        "model_a": {
            "name": wrapper_a.config.name,
            "path": path_a,
            "wrapper": wrapper_a,
            "fit_stats": wrapper_a.get_fit_statistics(),
            "channel_roi": contrib_a.get_channel_roi(),
            "contributions": contrib_a.get_grouped_contributions(),
            "channel_cols": wrapper_a.config.get_channel_columns(),
            "control_cols": wrapper_a.control_cols if hasattr(wrapper_a, 'control_cols') else [],
        },
        "model_b": {
            "name": wrapper_b.config.name,
            "path": path_b,
            "wrapper": wrapper_b,
            "fit_stats": wrapper_b.get_fit_statistics(),
            "channel_roi": contrib_b.get_channel_roi(),
            "contributions": contrib_b.get_grouped_contributions(),
            "channel_cols": wrapper_b.config.get_channel_columns(),
            "control_cols": wrapper_b.control_cols if hasattr(wrapper_b, 'control_cols') else [],
        }
    }


def _format_delta(value_a: float, value_b: float, format_str: str = ".2f", lower_is_better: bool = False) -> str:
    """Format delta between two values with indicator."""
    delta = value_b - value_a
    if abs(delta) < 0.001:
        return "="

    if lower_is_better:
        indicator = "▼" if delta < 0 else "▲"
        color = "green" if delta < 0 else "red"
    else:
        indicator = "▲" if delta > 0 else "▼"
        color = "green" if delta > 0 else "red"

    return f":{color}[{indicator} {delta:+{format_str}}]"


def _show_overview_tab(data: Dict[str, Any]):
    """Show overview comparison tab."""
    st.subheader("Model Fit Comparison")

    model_a = data["model_a"]
    model_b = data["model_b"]

    # Get fit statistics
    stats_a = model_a["fit_stats"]
    stats_b = model_b["fit_stats"]

    # Create comparison metrics
    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        st.markdown(f"### {model_a['name']}")

    with col2:
        st.markdown(f"### {model_b['name']}")

    with col3:
        st.markdown("### Delta")

    st.markdown("---")

    # R²
    r2_a = stats_a.get("r2", 0)
    r2_b = stats_b.get("r2", 0)
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        st.metric("R²", f"{r2_a:.2f}")
    with col2:
        st.metric("R²", f"{r2_b:.2f}")
    with col3:
        delta = r2_b - r2_a
        st.markdown(f"**{delta:+.2f}**" + (" ✓" if delta > 0 else (" ✗" if delta < 0 else "")))

    # MAPE (lower is better)
    mape_a = stats_a.get("mape", 0)
    mape_b = stats_b.get("mape", 0)
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        st.metric("MAPE", f"{mape_a:.2f}%")
    with col2:
        st.metric("MAPE", f"{mape_b:.2f}%")
    with col3:
        delta = mape_b - mape_a
        st.markdown(f"**{delta:+.2f}%**" + (" ✓" if delta < 0 else (" ✗" if delta > 0 else "")))

    # RMSE (lower is better)
    rmse_a = stats_a.get("rmse", 0)
    rmse_b = stats_b.get("rmse", 0)
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        st.metric("RMSE", f"{rmse_a:.2f}")
    with col2:
        st.metric("RMSE", f"{rmse_b:.2f}")
    with col3:
        delta = rmse_b - rmse_a
        st.markdown(f"**{delta:+.2f}**" + (" ✓" if delta < 0 else (" ✗" if delta > 0 else "")))

    # Fit Duration
    duration_a = stats_a.get("fit_duration_seconds", 0)
    duration_b = stats_b.get("fit_duration_seconds", 0)
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        st.metric("Fit Time", f"{duration_a:.0f}s")
    with col2:
        st.metric("Fit Time", f"{duration_b:.0f}s")
    with col3:
        delta = duration_b - duration_a
        st.markdown(f"**{delta:+.0f}s**")

    st.markdown("---")

    # Model structure comparison
    st.subheader("Model Structure")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"**{model_a['name']}**")
        st.write(f"- Channels: {len(model_a['channel_cols'])}")
        st.write(f"- Controls: {len(model_a['control_cols'])}")
        st.write(f"- Channels: {', '.join(model_a['channel_cols'][:5])}" +
                 ("..." if len(model_a['channel_cols']) > 5 else ""))

    with col2:
        st.markdown(f"**{model_b['name']}**")
        st.write(f"- Channels: {len(model_b['channel_cols'])}")
        st.write(f"- Controls: {len(model_b['control_cols'])}")
        st.write(f"- Channels: {', '.join(model_b['channel_cols'][:5])}" +
                 ("..." if len(model_b['channel_cols']) > 5 else ""))

    # Common vs different channels
    channels_a = set(model_a['channel_cols'])
    channels_b = set(model_b['channel_cols'])
    common_channels = channels_a & channels_b
    only_a = channels_a - channels_b
    only_b = channels_b - channels_a

    if only_a or only_b:
        st.markdown("---")
        st.subheader("Channel Differences")
        if common_channels:
            st.write(f"**Common channels ({len(common_channels)}):** {', '.join(sorted(common_channels))}")
        if only_a:
            st.write(f"**Only in Model A ({len(only_a)}):** {', '.join(sorted(only_a))}")
        if only_b:
            st.write(f"**Only in Model B ({len(only_b)}):** {', '.join(sorted(only_b))}")


def _show_channel_roi_tab(data: Dict[str, Any]):
    """Show channel ROI comparison tab."""
    st.subheader("Channel ROI Comparison")

    roi_a = data["model_a"]["channel_roi"]
    roi_b = data["model_b"]["channel_roi"]

    if roi_a.empty and roi_b.empty:
        st.warning("No channel ROI data available")
        return

    # Use display_name if available, otherwise channel
    name_col = "display_name" if "display_name" in roi_a.columns else "channel"

    # Merge on channel name
    roi_a_renamed = roi_a.rename(columns={
        "roi": "ROI (A)",
        "spend_real": "Spend (A)",
        "contribution_real": "Contribution (A)"
    })
    roi_b_renamed = roi_b.rename(columns={
        "roi": "ROI (B)",
        "spend_real": "Spend (B)",
        "contribution_real": "Contribution (B)"
    })

    # Merge dataframes
    if "channel" in roi_a_renamed.columns and "channel" in roi_b_renamed.columns:
        merged = pd.merge(
            roi_a_renamed[["channel", name_col, "ROI (A)", "Spend (A)", "Contribution (A)"]],
            roi_b_renamed[["channel", "ROI (B)", "Spend (B)", "Contribution (B)"]],
            on="channel",
            how="outer"
        )
    else:
        st.warning("Cannot merge ROI data - missing channel column")
        return

    # Calculate differences
    merged["ROI Diff"] = merged["ROI (B)"].fillna(0) - merged["ROI (A)"].fillna(0)

    # Format for display
    display_df = merged.copy()
    display_df["ROI (A)"] = display_df["ROI (A)"].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "N/A")
    display_df["ROI (B)"] = display_df["ROI (B)"].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "N/A")
    display_df["ROI Diff"] = display_df["ROI Diff"].apply(
        lambda x: f"+${x:.2f}" if x > 0 else (f"-${abs(x):.2f}" if x < 0 else "$0.00")
    )

    # Show table
    st.dataframe(
        display_df[[name_col, "ROI (A)", "ROI (B)", "ROI Diff"]].rename(columns={name_col: "Channel"}),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")

    # Grouped bar chart
    st.subheader("ROI Comparison Chart")

    # Prepare data for chart
    chart_data = []
    for _, row in merged.iterrows():
        channel_name = row[name_col] if pd.notna(row[name_col]) else row["channel"]
        if pd.notna(row["ROI (A)"]):
            chart_data.append({
                "Channel": channel_name,
                "Model": data["model_a"]["name"],
                "ROI": row["ROI (A)"]
            })
        if pd.notna(row["ROI (B)"]):
            chart_data.append({
                "Channel": channel_name,
                "Model": data["model_b"]["name"],
                "ROI": row["ROI (B)"]
            })

    if chart_data:
        chart_df = pd.DataFrame(chart_data)
        fig = px.bar(
            chart_df,
            x="Channel",
            y="ROI",
            color="Model",
            barmode="group",
            title="ROI by Channel and Model"
        )
        fig.add_hline(y=1, line_dash="dash", line_color="red", annotation_text="Breakeven")
        st.plotly_chart(fig, use_container_width=True)


def _show_contributions_tab(data: Dict[str, Any]):
    """Show contributions comparison tab."""
    st.subheader("Contribution Breakdown Comparison")

    contrib_a = data["model_a"]["contributions"]
    contrib_b = data["model_b"]["contributions"]

    if contrib_a.empty and contrib_b.empty:
        st.warning("No contribution data available")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"### {data['model_a']['name']}")
        if not contrib_a.empty:
            fig_a = px.pie(
                contrib_a,
                values="pct_of_total",
                names="group",
                title="Contribution by Category",
                color="group",
                color_discrete_map={row["group"]: row["color"] for _, row in contrib_a.iterrows()}
            )
            st.plotly_chart(fig_a, use_container_width=True)

            # Table
            display_a = contrib_a[["group", "pct_of_total", "contribution_real"]].copy()
            display_a["pct_of_total"] = display_a["pct_of_total"].apply(lambda x: f"{x:.1f}%")
            display_a["contribution_real"] = display_a["contribution_real"].apply(lambda x: f"${x:,.0f}")
            display_a.columns = ["Category", "% of Total", "Contribution"]
            st.dataframe(display_a, use_container_width=True, hide_index=True)

    with col2:
        st.markdown(f"### {data['model_b']['name']}")
        if not contrib_b.empty:
            fig_b = px.pie(
                contrib_b,
                values="pct_of_total",
                names="group",
                title="Contribution by Category",
                color="group",
                color_discrete_map={row["group"]: row["color"] for _, row in contrib_b.iterrows()}
            )
            st.plotly_chart(fig_b, use_container_width=True)

            # Table
            display_b = contrib_b[["group", "pct_of_total", "contribution_real"]].copy()
            display_b["pct_of_total"] = display_b["pct_of_total"].apply(lambda x: f"{x:.1f}%")
            display_b["contribution_real"] = display_b["contribution_real"].apply(lambda x: f"${x:,.0f}")
            display_b.columns = ["Category", "% of Total", "Contribution"]
            st.dataframe(display_b, use_container_width=True, hide_index=True)

    # Side-by-side comparison chart
    st.markdown("---")
    st.subheader("Category Comparison")

    # Merge contribution data
    contrib_a_renamed = contrib_a.rename(columns={"pct_of_total": "% (A)", "contribution_real": "Value (A)"})
    contrib_b_renamed = contrib_b.rename(columns={"pct_of_total": "% (B)", "contribution_real": "Value (B)"})

    merged_contrib = pd.merge(
        contrib_a_renamed[["group", "% (A)", "Value (A)"]],
        contrib_b_renamed[["group", "% (B)", "Value (B)"]],
        on="group",
        how="outer"
    ).fillna(0)

    merged_contrib["% Diff"] = merged_contrib["% (B)"] - merged_contrib["% (A)"]

    # Display comparison table
    display_merged = merged_contrib.copy()
    display_merged["% (A)"] = display_merged["% (A)"].apply(lambda x: f"{x:.1f}%")
    display_merged["% (B)"] = display_merged["% (B)"].apply(lambda x: f"{x:.1f}%")
    display_merged["% Diff"] = display_merged["% Diff"].apply(
        lambda x: f"+{x:.1f}%" if x > 0 else f"{x:.1f}%"
    )
    display_merged.columns = ["Category", "% (A)", "Value (A)", "% (B)", "Value (B)", "% Diff"]
    display_merged["Value (A)"] = merged_contrib["Value (A)"].apply(lambda x: f"${x:,.0f}")
    display_merged["Value (B)"] = merged_contrib["Value (B)"].apply(lambda x: f"${x:,.0f}")

    st.dataframe(display_merged, use_container_width=True, hide_index=True)


def _show_coefficients_tab(data: Dict[str, Any]):
    """Show model coefficients comparison tab."""
    st.subheader("Model Coefficients Comparison")

    wrapper_a = data["model_a"]["wrapper"]
    wrapper_b = data["model_b"]["wrapper"]

    # Try to get posterior summaries
    try:
        import arviz as az

        if wrapper_a.idata is None or wrapper_b.idata is None:
            st.warning("Inference data not available for one or both models")
            return

        # Get summary statistics
        summary_a = az.summary(wrapper_a.idata, var_names=["beta_channel"], round_to=4)
        summary_b = az.summary(wrapper_b.idata, var_names=["beta_channel"], round_to=4)

        st.markdown("### Channel Coefficients (beta_channel)")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"**{data['model_a']['name']}**")
            display_a = summary_a[["mean", "sd", "hdi_3%", "hdi_97%"]].copy()
            display_a.columns = ["Mean", "Std", "HDI 3%", "HDI 97%"]
            st.dataframe(display_a, use_container_width=True)

        with col2:
            st.markdown(f"**{data['model_b']['name']}**")
            display_b = summary_b[["mean", "sd", "hdi_3%", "hdi_97%"]].copy()
            display_b.columns = ["Mean", "Std", "HDI 3%", "HDI 97%"]
            st.dataframe(display_b, use_container_width=True)

        # Compare common channels
        st.markdown("---")
        st.subheader("Coefficient Difference (Common Channels)")

        # Get channel names from index
        channels_a = set(summary_a.index)
        channels_b = set(summary_b.index)
        common = channels_a & channels_b

        if common:
            comparison_data = []
            for ch in sorted(common):
                mean_a = summary_a.loc[ch, "mean"]
                mean_b = summary_b.loc[ch, "mean"]
                comparison_data.append({
                    "Channel": ch,
                    "Mean (A)": f"{mean_a:.2f}",
                    "Mean (B)": f"{mean_b:.2f}",
                    "Difference": f"{mean_b - mean_a:+.2f}"
                })

            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        else:
            st.info("No common channels found between models")

        # Saturation parameters if available
        st.markdown("---")
        st.subheader("Saturation Parameters")

        try:
            sat_a = az.summary(wrapper_a.idata, var_names=["lam"], round_to=4)
            sat_b = az.summary(wrapper_b.idata, var_names=["lam"], round_to=4)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"**{data['model_a']['name']} - Lambda (saturation)**")
                st.dataframe(sat_a[["mean", "sd"]], use_container_width=True)

            with col2:
                st.markdown(f"**{data['model_b']['name']} - Lambda (saturation)**")
                st.dataframe(sat_b[["mean", "sd"]], use_container_width=True)

        except Exception:
            st.info("Saturation parameters not available")

        # Adstock parameters if available
        st.markdown("---")
        st.subheader("Adstock Parameters")

        try:
            adstock_a = az.summary(wrapper_a.idata, var_names=["alpha"], round_to=4)
            adstock_b = az.summary(wrapper_b.idata, var_names=["alpha"], round_to=4)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"**{data['model_a']['name']} - Alpha (adstock)**")
                st.dataframe(adstock_a[["mean", "sd"]], use_container_width=True)

            with col2:
                st.markdown(f"**{data['model_b']['name']} - Alpha (adstock)**")
                st.dataframe(adstock_b[["mean", "sd"]], use_container_width=True)

        except Exception:
            st.info("Adstock parameters not available")

    except Exception as e:
        st.error(f"Error loading coefficient data: {e}")
        st.info("This may occur if the models were saved without full inference data.")
