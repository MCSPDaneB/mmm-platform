"""
Combined Model Analysis page for MMM Platform.

Allows users to load multiple fitted models and combine their results
for unified investment recommendations across different outcome streams.
"""

import streamlit as st
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional

from mmm_platform.model.persistence import ModelPersistence, get_models_dir, list_clients
from mmm_platform.model.mmm import MMMWrapper
from mmm_platform.analysis.combined_models import MultiModelAnalyzer, MultiModelResult
from mmm_platform.analysis.marginal_roi import MarginalROIAnalyzer


def show():
    """Show the combined analysis page."""
    st.title("Combined Model Analysis")
    st.caption("Combine multiple fitted models to analyze outcomes across different revenue streams")

    # Initialize session state for combined analysis
    if "combined_models" not in st.session_state:
        st.session_state.combined_models = []
    if "combined_result" not in st.session_state:
        st.session_state.combined_result = None
    if "combined_analyzer" not in st.session_state:
        st.session_state.combined_analyzer = None

    # Client filter
    clients = list_clients()
    if clients:
        client_options = ["All Clients"] + clients
        selected_client = st.selectbox(
            "Filter by Client",
            options=client_options,
            key="combined_analysis_client_filter",
            help="Show models for a specific client"
        )
        client_filter = "all" if selected_client == "All Clients" else selected_client
    else:
        client_filter = "all"

    # Get saved models (client-aware)
    saved_models = ModelPersistence.list_saved_models(client=client_filter)

    if not saved_models:
        st.warning("No fitted models found. Please run and save at least 2 models first.")
        st.info("Go to **Run Model** to fit a model, then it will be automatically saved.")
        return

    if len(saved_models) < 2:
        st.warning("At least 2 fitted models are required for combined analysis.")
        st.info(f"Found {len(saved_models)} model(s). Please fit more models.")
        return

    # Model Selection Section
    st.subheader("1. Select Models")

    # Create options for multiselect
    model_options = {}
    for model in saved_models:
        r2 = model.get("r2")
        r2_str = f"R²={r2:.3f}" if r2 is not None else "R²=N/A"
        n_channels = model.get("n_channels", "?")
        created = model.get("created_at", "")[:10]
        name = model.get("config_name", "Unknown")
        model_client = model.get("client", "")

        # Show client in label when viewing all clients
        if client_filter == "all" and model_client:
            option_label = f"[{model_client}] {name} ({created}) - {n_channels} channels, {r2_str}"
        else:
            option_label = f"{name} ({created}) - {n_channels} channels, {r2_str}"
        model_options[option_label] = model["path"]

    selected_options = st.multiselect(
        "Select models to combine",
        options=list(model_options.keys()),
        default=[],
        help="Select at least 2 models to combine",
    )

    if len(selected_options) < 2:
        st.info("Please select at least 2 models to proceed.")
        return

    # Model Configuration Section
    st.subheader("2. Configure Models")
    st.caption("Set custom labels and profit margins for each model")

    # Build configuration data for selected models
    config_data = []
    for option in selected_options:
        path = model_options[option]
        # Find the model metadata
        model_meta = next((m for m in saved_models if m["path"] == path), {})
        name = model_meta.get("config_name", "Unknown")

        # Check if we already have config for this model
        existing_config = next(
            (m for m in st.session_state.combined_models if m.get("path") == path),
            None
        )

        config_data.append({
            "Model": name,
            "Path": path,
            "Label": existing_config["label"] if existing_config else name,
            "Margin (%)": existing_config["margin"] * 100 if existing_config else 30.0,
        })

    # Display editable configuration
    config_df = pd.DataFrame(config_data)

    edited_config = st.data_editor(
        config_df,
        column_config={
            "Model": st.column_config.TextColumn("Model", disabled=True),
            "Path": st.column_config.TextColumn("Path", disabled=True),
            "Label": st.column_config.TextColumn(
                "Label",
                help="Custom name for this model (e.g., 'Online Revenue')",
                max_chars=50,
            ),
            "Margin (%)": st.column_config.NumberColumn(
                "Profit Margin (%)",
                help="Profit margin for this outcome stream (0-100%)",
                min_value=0,
                max_value=100,
                step=1,
            ),
        },
        hide_index=True,
        width="stretch",
        key="combined_config_editor",
    )

    # Save configuration button
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("Save Configuration", type="secondary"):
            # Update session state with new configuration
            st.session_state.combined_models = [
                {
                    "path": row["Path"],
                    "label": row["Label"],
                    "margin": row["Margin (%)"] / 100.0,
                }
                for _, row in edited_config.iterrows()
            ]
            st.success("Configuration saved!")

    st.markdown("---")

    # Run Analysis Section
    st.subheader("3. Run Combined Analysis")

    # Check for duplicate labels
    labels = [row["Label"] for _, row in edited_config.iterrows()]
    if len(labels) != len(set(labels)):
        st.error("Labels must be unique. Please use different labels for each model.")
        return

    col1, col2 = st.columns([1, 3])
    with col1:
        run_button = st.button("Run Analysis", type="primary", width="stretch")

    if run_button:
        _run_combined_analysis(edited_config)

    # Display Results
    if st.session_state.combined_result is not None:
        st.markdown("---")
        _display_results()


def _run_combined_analysis(config_df: pd.DataFrame):
    """Load models and run combined analysis."""
    with st.spinner("Loading models and running analysis..."):
        try:
            wrappers_with_config = []

            for _, row in config_df.iterrows():
                path = row["Path"]
                label = row["Label"]
                margin = row["Margin (%)"] / 100.0

                # Load the model
                wrapper = ModelPersistence.load(path, MMMWrapper)

                wrappers_with_config.append({
                    "wrapper": wrapper,
                    "label": label,
                    "margin": margin,
                })

            # Run combined analysis
            analyzer, result = MultiModelAnalyzer.from_mmm_wrappers(wrappers_with_config)

            # Store in session state
            st.session_state.combined_analyzer = analyzer
            st.session_state.combined_result = result
            st.session_state.combined_models = [
                {"path": row["Path"], "label": row["Label"], "margin": row["Margin (%)"] / 100.0}
                for _, row in config_df.iterrows()
            ]

            st.success("Analysis complete!")
            st.rerun()

        except Exception as e:
            st.error(f"Error running analysis: {e}")
            import traceback
            st.code(traceback.format_exc())


def _display_results():
    """Display combined analysis results."""
    result: MultiModelResult = st.session_state.combined_result
    analyzer: MultiModelAnalyzer = st.session_state.combined_analyzer

    st.subheader("4. Results")

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Models Combined", len(result.model_configs))
    with col2:
        st.metric("Channels Analyzed", len(result.combined_analysis))
    with col3:
        st.metric("Views Available", len(result.view_recommendations))
    with col4:
        st.metric("Conflicts Found", len(result.conflicting_channels))

    # Tabs for different result views
    tab1, tab2, tab3, tab4 = st.tabs([
        "Combined Summary",
        "By Model",
        "Recommendations",
        "Conflicts",
    ])

    with tab1:
        _show_combined_summary(analyzer, result)

    with tab2:
        _show_by_model(result)

    with tab3:
        _show_recommendations(result)

    with tab4:
        _show_conflicts(result)


def _show_combined_summary(analyzer: MultiModelAnalyzer, result: MultiModelResult):
    """Show combined summary table."""
    st.subheader("Combined Summary")
    st.caption("Marginal ROI and profit metrics across all models")

    # Get summary table
    summary_df = analyzer.get_summary_table()

    # Format columns
    format_dict = {"Current Spend": "${:,.0f}"}
    for col in summary_df.columns:
        if "Marginal" in col or "Current ROI" in col or "Profit" in col:
            format_dict[col] = "${:.2f}"

    st.dataframe(
        summary_df.style.format(format_dict),
        width="stretch",
        hide_index=True,
    )

    # Show model margins
    st.markdown("---")
    st.markdown("**Model Margins:**")
    margin_text = " | ".join([
        f"{c['label']}: {c['margin']*100:.0f}%"
        for c in result.model_configs
    ])
    st.caption(margin_text)


def _show_by_model(result: MultiModelResult):
    """Show breakdown by individual model."""
    st.subheader("Analysis by Model")

    # Model selector
    model_labels = [c["label"] for c in result.model_configs]
    selected_model = st.selectbox(
        "Select model view",
        options=model_labels,
        key="by_model_selector",
    )

    if selected_model:
        rec = result.view_recommendations.get(selected_model)
        if rec:
            st.markdown(f"### {rec.view_title}")
            st.caption(
                f"Thresholds: INCREASE > ${rec.threshold_high:.2f}, "
                f"REDUCE < ${rec.threshold_low:.2f}"
            )

            # Summary counts
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("INCREASE", len(rec.increase_channels), delta="", delta_color="normal")
            with col2:
                st.metric("HOLD", len(rec.hold_channels))
            with col3:
                st.metric("REDUCE", len(rec.reduce_channels), delta="", delta_color="inverse")

            # Ranked table
            data = []
            for rank, ch, action in rec.ranked_channels:
                marginal = ch.marginal_roi_by_model.get(selected_model, 0)
                data.append({
                    "Rank": rank,
                    "Channel": ch.channel_name,
                    "Marginal ROI": marginal,
                    "Current Spend": ch.current_spend,
                    "Action": action,
                })

            df = pd.DataFrame(data)

            # Color-code actions
            def highlight_action(val):
                if val == "INCREASE":
                    return "background-color: #d4edda; color: #155724"
                elif val == "REDUCE":
                    return "background-color: #f8d7da; color: #721c24"
                else:
                    return "background-color: #fff3cd; color: #856404"

            styled_df = df.style.applymap(
                highlight_action, subset=["Action"]
            ).format({
                "Marginal ROI": "${:.2f}",
                "Current Spend": "${:,.0f}",
            })

            st.dataframe(styled_df, width="stretch", hide_index=True)


def _show_recommendations(result: MultiModelResult):
    """Show recommendations comparison across views."""
    st.subheader("Recommendations by View")

    # View selector
    view_options = (
        [c["label"] for c in result.model_configs]
        + ["total", "profit"]
    )
    view_labels = {
        "total": "Total Revenue",
        "profit": "Profit",
    }
    view_labels.update({c["label"]: c["label"] for c in result.model_configs})

    selected_view = st.selectbox(
        "Select view",
        options=view_options,
        format_func=lambda x: view_labels.get(x, x),
        key="recommendations_view_selector",
    )

    if selected_view:
        rec = result.view_recommendations.get(selected_view)
        if rec:
            st.markdown(f"### {rec.view_title}")
            st.caption(
                f"Thresholds: INCREASE > ${rec.threshold_high:.2f}, "
                f"REDUCE < ${rec.threshold_low:.2f}"
            )

            # Summary counts with colored boxes
            col1, col2, col3 = st.columns(3)
            with col1:
                st.success(f"**INCREASE:** {len(rec.increase_channels)} channels")
            with col2:
                st.warning(f"**HOLD:** {len(rec.hold_channels)} channels")
            with col3:
                st.error(f"**REDUCE:** {len(rec.reduce_channels)} channels")

            # Get marginal ROI for this view
            def get_marginal(ch):
                if selected_view == "total":
                    return ch.marginal_total
                elif selected_view == "profit":
                    return ch.marginal_profit
                else:
                    return ch.marginal_roi_by_model.get(selected_view, 0)

            # Ranked table
            st.markdown("---")
            data = []
            for rank, ch, action in rec.ranked_channels:
                marginal = get_marginal(ch)
                data.append({
                    "Rank": rank,
                    "Channel": ch.channel_name,
                    "Marginal ROI": marginal,
                    "Current Spend": ch.current_spend,
                    "Action": action,
                })

            df = pd.DataFrame(data)

            def highlight_action(val):
                if val == "INCREASE":
                    return "background-color: #d4edda; color: #155724"
                elif val == "REDUCE":
                    return "background-color: #f8d7da; color: #721c24"
                else:
                    return "background-color: #fff3cd; color: #856404"

            styled_df = df.style.applymap(
                highlight_action, subset=["Action"]
            ).format({
                "Marginal ROI": "${:.2f}",
                "Current Spend": "${:,.0f}",
            })

            st.dataframe(styled_df, width="stretch", hide_index=True)

    # Comparison table across all views
    st.markdown("---")
    st.subheader("Recommendation Comparison")
    st.caption("Compare recommendations across all views")

    # Build comparison table
    comparison_data = []
    for ch in sorted(result.combined_analysis, key=lambda x: -x.marginal_total):
        row = {"Channel": ch.channel_name}

        for view in view_options:
            rec = result.view_recommendations.get(view)
            if rec:
                for _, c, action in rec.ranked_channels:
                    if c.channel == ch.channel:
                        row[view_labels.get(view, view)] = action
                        break

        comparison_data.append(row)

    comparison_df = pd.DataFrame(comparison_data)

    def highlight_action_cell(val):
        if val == "INCREASE":
            return "background-color: #d4edda"
        elif val == "REDUCE":
            return "background-color: #f8d7da"
        elif val == "HOLD":
            return "background-color: #fff3cd"
        return ""

    styled_comparison = comparison_df.style.applymap(
        highlight_action_cell,
        subset=[c for c in comparison_df.columns if c != "Channel"]
    )

    st.dataframe(styled_comparison, width="stretch", hide_index=True)


def _show_conflicts(result: MultiModelResult):
    """Show conflicting channels."""
    st.subheader("Conflicting Recommendations")

    if not result.conflicting_channels:
        st.success("No conflicts! All views agree on recommendations for all channels.")
        return

    st.warning(
        f"Found **{len(result.conflicting_channels)}** channels with "
        "different recommendations across views."
    )
    st.caption(
        "These channels have different actions (INCREASE/HOLD/REDUCE) "
        "depending on which outcome you optimize for."
    )

    # List conflicts
    st.markdown("**Conflicting Channels:**")
    for ch_name in result.conflicting_channels:
        st.markdown(f"- {ch_name}")

    # Show detailed breakdown for conflicts
    st.markdown("---")
    st.markdown("**Detailed Breakdown:**")

    view_labels = {
        "total": "Total Revenue",
        "profit": "Profit",
    }
    view_labels.update({c["label"]: c["label"] for c in result.model_configs})

    for ch_name in result.conflicting_channels:
        # Find the channel
        ch = next(
            (c for c in result.combined_analysis if c.channel_name == ch_name),
            None
        )
        if ch:
            with st.expander(f"**{ch_name}** - Current Spend: ${ch.current_spend:,.0f}"):
                data = []
                for view, rec in result.view_recommendations.items():
                    for _, c, action in rec.ranked_channels:
                        if c.channel == ch.channel:
                            if view == "total":
                                marginal = ch.marginal_total
                            elif view == "profit":
                                marginal = ch.marginal_profit
                            else:
                                marginal = ch.marginal_roi_by_model.get(view, 0)

                            data.append({
                                "View": view_labels.get(view, view),
                                "Marginal ROI": marginal,
                                "Action": action,
                            })
                            break

                df = pd.DataFrame(data)

                def highlight_action(val):
                    if val == "INCREASE":
                        return "background-color: #d4edda"
                    elif val == "REDUCE":
                        return "background-color: #f8d7da"
                    elif val == "HOLD":
                        return "background-color: #fff3cd"
                    return ""

                styled_df = df.style.applymap(
                    highlight_action, subset=["Action"]
                ).format({"Marginal ROI": "${:.2f}"})

                st.dataframe(styled_df, width="stretch", hide_index=True)

    st.markdown("---")
    st.info(
        "**Decision depends on business objective!** "
        "Choose the view that aligns with your strategic priorities."
    )
