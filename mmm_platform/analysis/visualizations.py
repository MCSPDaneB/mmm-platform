"""Enhanced visualizations for MMM Platform.

Provides comprehensive visualization functions including:
- Forest plots for posterior distributions
- ROI posterior plots with credible intervals
- Probability of Direction plots
- Prior vs Posterior comparison plots
- Model decomposition charts
- Residual analysis plots
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import arviz as az

from .bayesian_significance import (
    BayesianSignificanceReport,
    ROICredibleIntervalResult,
    ProbabilityOfDirectionResult,
    PriorSensitivityResult,
)


def create_forest_plot(
    idata: Any,
    var_names: List[str] = ["saturation_beta"],
    combined: bool = True,
    figsize: Tuple[int, int] = (12, 8),
    title: Optional[str] = None,
    show_zero_line: bool = True
) -> plt.Figure:
    """Create a forest plot for posterior distributions.

    Args:
        idata: ArviZ InferenceData object
        var_names: Variable names to plot
        combined: Whether to combine chains
        figsize: Figure size
        title: Plot title
        show_zero_line: Whether to show vertical line at zero

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    az.plot_forest(
        idata,
        var_names=var_names,
        combined=combined,
        ax=ax
    )

    if show_zero_line:
        ax.axvline(0, color='red', linestyle='--', alpha=0.5, label='Zero effect')

    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"Posterior: {', '.join(var_names)}")

    plt.tight_layout()
    return fig


def create_roi_posterior_plot(
    roi_results: List[ROICredibleIntervalResult],
    channel_display_names: Optional[Dict[str, str]] = None,
    figsize: Tuple[int, int] = (12, 8),
    title: str = "ROI Posteriors with 90% Credible Intervals"
) -> plt.Figure:
    """Create a horizontal bar plot of ROI posteriors with credible intervals.

    Args:
        roi_results: List of ROI credible interval results
        channel_display_names: Optional mapping of channel names to display names
        figsize: Figure size
        title: Plot title

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    channels = [r.channel for r in roi_results]
    roi_means = [r.roi_mean for r in roi_results]
    roi_5s = [r.roi_5pct for r in roi_results]
    roi_95s = [r.roi_95pct for r in roi_results]

    # Create display names
    if channel_display_names:
        display_names = [channel_display_names.get(ch, ch) for ch in channels]
    else:
        # Simplify channel names by removing common prefixes/suffixes
        display_names = [
            ch.replace("PaidMedia_", "")
              .replace("_spend", "")
              .replace("_", " ")
            for ch in channels
        ]

    y_pos = np.arange(len(channels))

    # Calculate error bars
    errors_low = np.array(roi_means) - np.array(roi_5s)
    errors_high = np.array(roi_95s) - np.array(roi_means)

    # Color based on significance
    colors = ['steelblue' if r.significant else 'lightcoral' for r in roi_results]

    ax.barh(y_pos, roi_means,
            xerr=[errors_low, errors_high],
            capsize=3,
            color=colors,
            alpha=0.7)

    ax.axvline(0, color='red', linestyle='--', alpha=0.5, label='ROI = 0')
    ax.axvline(1, color='green', linestyle='--', alpha=0.5, label='ROI = 1 (breakeven)')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(display_names)
    ax.set_xlabel("ROI")
    ax.set_title(title)
    ax.legend(loc='lower right')

    plt.tight_layout()
    return fig


def create_probability_of_direction_plot(
    pd_results: List[ProbabilityOfDirectionResult],
    channel_display_names: Optional[Dict[str, str]] = None,
    figsize: Tuple[int, int] = (12, 8),
    title: str = "Probability of Direction (Positive Effect)"
) -> plt.Figure:
    """Create a horizontal bar plot of probability of direction.

    Args:
        pd_results: List of probability of direction results
        channel_display_names: Optional mapping of channel names to display names
        figsize: Figure size
        title: Plot title

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    channels = [r.channel for r in pd_results]
    pds = [r.pd for r in pd_results]

    # Create display names
    if channel_display_names:
        display_names = [channel_display_names.get(ch, ch) for ch in channels]
    else:
        display_names = [
            ch.replace("PaidMedia_", "")
              .replace("_spend", "")
              .replace("_", " ")
            for ch in channels
        ]

    y_pos = np.arange(len(channels))

    # Color based on strength
    colors = []
    for pd_val in pds:
        if pd_val >= 0.95:
            colors.append('green')
        elif pd_val >= 0.75:
            colors.append('orange')
        else:
            colors.append('red')

    ax.barh(y_pos, pds, color=colors, alpha=0.7)

    ax.axvline(0.95, color='green', linestyle='--', alpha=0.5, label='95% threshold')
    ax.axvline(0.75, color='orange', linestyle='--', alpha=0.5, label='75% threshold')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(display_names)
    ax.set_xlabel("Probability of Direction (positive)")
    ax.set_title(title)
    ax.set_xlim(0.5, 1.0)
    ax.legend(loc='lower right')

    plt.tight_layout()
    return fig


def create_prior_vs_posterior_plot(
    sensitivity_results: List[PriorSensitivityResult],
    channel_display_names: Optional[Dict[str, str]] = None,
    figsize: Tuple[int, int] = (10, 10),
    title: str = "Prior vs Posterior ROI"
) -> plt.Figure:
    """Create a scatter plot comparing prior and posterior ROI.

    Args:
        sensitivity_results: List of prior sensitivity results
        channel_display_names: Optional mapping of channel names to display names
        figsize: Figure size
        title: Plot title

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    channels = [r.channel for r in sensitivity_results]
    prior_rois = [r.prior_roi for r in sensitivity_results]
    posterior_rois = [r.posterior_roi for r in sensitivity_results]

    # Create display names
    if channel_display_names:
        display_names = [channel_display_names.get(ch, ch) for ch in channels]
    else:
        display_names = [
            ch.replace("PaidMedia_", "")
              .replace("_spend", "")
              .replace("_", " ")
            for ch in channels
        ]

    # Color based on data influence
    colors = []
    for r in sensitivity_results:
        if r.data_influence == "Strong":
            colors.append('green')
        elif r.data_influence == "Moderate":
            colors.append('orange')
        else:
            colors.append('red')

    ax.scatter(prior_rois, posterior_rois, s=100, c=colors, alpha=0.7)

    # Add diagonal line (no change from prior)
    max_val = max(max(prior_rois), max(posterior_rois)) * 1.1
    min_val = min(min(prior_rois), min(posterior_rois)) * 0.9
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='No change from prior')

    # Add labels for each point
    for i, (x, y, name) in enumerate(zip(prior_rois, posterior_rois, display_names)):
        ax.annotate(name, (x, y), fontsize=8, ha='left', va='bottom')

    ax.set_xlabel("Prior ROI")
    ax.set_ylabel("Posterior ROI")
    ax.set_title(title)
    ax.legend()

    # Add legend for colors
    strong_patch = mpatches.Patch(color='green', alpha=0.7, label='Strong data influence')
    moderate_patch = mpatches.Patch(color='orange', alpha=0.7, label='Moderate data influence')
    weak_patch = mpatches.Patch(color='red', alpha=0.7, label='Weak (prior-driven)')
    ax.legend(handles=[strong_patch, moderate_patch, weak_patch], loc='upper left')

    plt.tight_layout()
    return fig


def create_significance_dashboard(
    report: BayesianSignificanceReport,
    channel_display_names: Optional[Dict[str, str]] = None,
    figsize: Tuple[int, int] = (16, 12)
) -> plt.Figure:
    """Create a 2x2 dashboard of Bayesian significance visualizations.

    Args:
        report: BayesianSignificanceReport with all analysis results
        channel_display_names: Optional mapping of channel names to display names
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Get data from report
    channels = [r.channel for r in report.roi_posteriors]

    # Create display names
    if channel_display_names:
        display_names = [channel_display_names.get(ch, ch) for ch in channels]
    else:
        display_names = [
            ch.replace("PaidMedia_", "")
              .replace("_spend", "")
              .replace("_", " ")
            for ch in channels
        ]

    y_pos = np.arange(len(channels))

    # Plot 1: ROI posteriors with credible intervals
    ax1 = axes[0, 0]
    roi_means = [r.roi_mean for r in report.roi_posteriors]
    roi_5s = [r.roi_5pct for r in report.roi_posteriors]
    roi_95s = [r.roi_95pct for r in report.roi_posteriors]
    colors_roi = ['steelblue' if r.significant else 'lightcoral' for r in report.roi_posteriors]

    errors_low = np.array(roi_means) - np.array(roi_5s)
    errors_high = np.array(roi_95s) - np.array(roi_means)

    ax1.barh(y_pos, roi_means,
             xerr=[errors_low, errors_high],
             capsize=3, color=colors_roi, alpha=0.7)
    ax1.axvline(0, color='red', linestyle='--', alpha=0.5)
    ax1.axvline(1, color='green', linestyle='--', alpha=0.5)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(display_names)
    ax1.set_xlabel("ROI")
    ax1.set_title("ROI Posteriors with 90% CI")

    # Plot 2: Beta posteriors with HDI
    ax2 = axes[0, 1]
    beta_means = [r.mean for r in report.credible_intervals]
    beta_lows = [r.hdi_low for r in report.credible_intervals]
    beta_highs = [r.hdi_high for r in report.credible_intervals]
    colors_beta = ['steelblue' if r.excludes_zero else 'lightcoral' for r in report.credible_intervals]

    errors_low_beta = np.array(beta_means) - np.array(beta_lows)
    errors_high_beta = np.array(beta_highs) - np.array(beta_means)

    ax2.barh(y_pos, beta_means,
             xerr=[errors_low_beta, errors_high_beta],
             capsize=3, color=colors_beta, alpha=0.7)
    ax2.axvline(0, color='red', linestyle='--', alpha=0.5)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(display_names)
    ax2.set_xlabel("Beta (saturation_beta)")
    ax2.set_title("Beta Posteriors with 95% HDI")

    # Plot 3: Probability of Direction
    ax3 = axes[1, 0]
    pds = [r.pd for r in report.probability_of_direction]
    colors_pd = []
    for pd_val in pds:
        if pd_val >= 0.95:
            colors_pd.append('green')
        elif pd_val >= 0.75:
            colors_pd.append('orange')
        else:
            colors_pd.append('red')

    ax3.barh(y_pos, pds, color=colors_pd, alpha=0.7)
    ax3.axvline(0.95, color='green', linestyle='--', alpha=0.5)
    ax3.axvline(0.75, color='orange', linestyle='--', alpha=0.5)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(display_names)
    ax3.set_xlabel("Probability of Direction (positive)")
    ax3.set_title("Probability Effect is Positive")
    ax3.set_xlim(0.5, 1.0)

    # Plot 4: Prior vs Posterior
    ax4 = axes[1, 1]
    prior_rois = [r.prior_roi for r in report.prior_sensitivity]
    post_rois = [r.posterior_roi for r in report.prior_sensitivity]

    colors_sens = []
    for r in report.prior_sensitivity:
        if r.data_influence == "Strong":
            colors_sens.append('green')
        elif r.data_influence == "Moderate":
            colors_sens.append('orange')
        else:
            colors_sens.append('red')

    ax4.scatter(prior_rois, post_rois, s=100, c=colors_sens, alpha=0.7)
    max_val = max(max(prior_rois), max(post_rois)) * 1.1
    min_val = min(min(prior_rois) if min(prior_rois) > 0 else 0, min(post_rois) if min(post_rois) > 0 else 0)
    ax4.plot([min_val, max_val], [min_val, max_val], 'r--', label='No change')

    for i, name in enumerate(display_names):
        ax4.annotate(name, (prior_rois[i], post_rois[i]), fontsize=7)

    ax4.set_xlabel("Prior ROI")
    ax4.set_ylabel("Posterior ROI")
    ax4.set_title("Prior vs Posterior ROI")

    plt.tight_layout()
    return fig


def create_model_decomposition_plot(
    dates: pd.DatetimeIndex,
    actual: pd.Series,
    channel_contrib: pd.Series,
    control_contrib: pd.Series,
    baseline_contrib: pd.Series,
    target_col: str = "KPI",
    figsize: Tuple[int, int] = (14, 6)
) -> plt.Figure:
    """Create a model decomposition time series plot.

    Args:
        dates: DatetimeIndex for x-axis
        actual: Actual target values
        channel_contrib: Total channel contributions
        control_contrib: Total control contributions
        baseline_contrib: Baseline (intercept) contributions
        target_col: Name of target column for labeling
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(dates, actual, label=f"Actual {target_col}", color="black", linewidth=2)
    ax.plot(dates, channel_contrib, label="Channels (media)", linewidth=2)
    ax.plot(dates, control_contrib + baseline_contrib, label="Controls + Baseline", linewidth=2)

    ax.set_xlabel("Date")
    ax.set_ylabel("Contribution")
    ax.set_title("Model Decomposition")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def create_residual_analysis_plots(
    dates: pd.DatetimeIndex,
    actual: pd.Series,
    fitted: pd.Series,
    figsize: Tuple[int, int] = (14, 12)
) -> plt.Figure:
    """Create a 2x2 panel of residual analysis plots.

    Args:
        dates: DatetimeIndex for x-axis
        actual: Actual target values
        fitted: Fitted values
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    residuals = actual - fitted

    # Calculate R-squared
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((actual - actual.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Plot 1: Actual vs Fitted over time
    ax1 = axes[0, 0]
    ax1.plot(dates, actual, label="Actual", linewidth=2)
    ax1.plot(dates, fitted, label="Fitted", linewidth=2)
    ax1.set_title(f"Actual vs Fitted (R² = {r2:.3f})")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Value")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Residuals over time
    ax2 = axes[0, 1]
    ax2.plot(dates, residuals, label="Residuals")
    ax2.axhline(0, color="black", linewidth=1)
    ax2.set_title("Residuals over Time")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Actual - Fitted")
    ax2.grid(True, alpha=0.3)

    # Plot 3: Residual histogram
    ax3 = axes[1, 0]
    ax3.hist(residuals, bins=20, edgecolor="k", alpha=0.7)
    ax3.set_title("Residuals Distribution")
    ax3.set_xlabel("Residual")
    ax3.set_ylabel("Frequency")

    # Plot 4: Actual vs Fitted scatter
    ax4 = axes[1, 1]
    ax4.scatter(actual, fitted, alpha=0.7)
    max_val = max(actual.max(), fitted.max())
    min_val = min(actual.min(), fitted.min())
    ax4.plot([min_val, max_val], [min_val, max_val], "r--", label="y = x")
    ax4.set_title("Actual vs Fitted")
    ax4.set_xlabel("Actual")
    ax4.set_ylabel("Fitted")
    ax4.legend()

    plt.tight_layout()
    return fig


def create_channel_roi_bar_chart(
    channels: List[str],
    roi_values: List[float],
    channel_display_names: Optional[Dict[str, str]] = None,
    figsize: Tuple[int, int] = (12, 6),
    title: str = "Channel ROI (Contribution / Spend)"
) -> plt.Figure:
    """Create a bar chart of channel ROI values.

    Args:
        channels: List of channel names
        roi_values: List of ROI values
        channel_display_names: Optional mapping of channel names to display names
        figsize: Figure size
        title: Plot title

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Create display names
    if channel_display_names:
        display_names = [channel_display_names.get(ch, ch) for ch in channels]
    else:
        display_names = [
            ch.replace("PaidMedia_", "")
              .replace("_spend", "")
              .replace("_", " ")
            for ch in channels
        ]

    # Sort by ROI
    sorted_indices = np.argsort(roi_values)[::-1]
    sorted_names = [display_names[i] for i in sorted_indices]
    sorted_rois = [roi_values[i] for i in sorted_indices]

    # Color based on ROI value
    colors = ['green' if r > 1 else 'orange' if r > 0 else 'red' for r in sorted_rois]

    bars = ax.bar(range(len(sorted_names)), sorted_rois, color=colors, alpha=0.7)

    ax.axhline(1, color='green', linestyle='--', alpha=0.5, label='ROI = 1 (breakeven)')
    ax.axhline(0, color='red', linestyle='--', alpha=0.5)

    ax.set_xticks(range(len(sorted_names)))
    ax.set_xticklabels(sorted_names, rotation=45, ha='right')
    ax.set_ylabel("ROI")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    plt.tight_layout()
    return fig


def create_contribution_waterfall_chart(
    contributions: Dict[str, float],
    figsize: Tuple[int, int] = (14, 8),
    title: str = "Contribution Breakdown (Waterfall)"
) -> plt.Figure:
    """Create a true waterfall chart showing contribution breakdown.

    The chart shows baseline at the top, then each channel contribution
    cascading down to build the total. Each bar starts where the previous
    one ended, creating a waterfall effect.

    Args:
        contributions: Dictionary of component names to contribution values.
                      Should include 'Baseline' key for the baseline contribution.
        figsize: Figure size
        title: Plot title

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Separate baseline from channels
    baseline_val = contributions.get('Baseline', 0)
    channel_contribs = {k: v for k, v in contributions.items() if k != 'Baseline'}

    # Sort channels by contribution (descending)
    sorted_channels = sorted(channel_contribs.items(), key=lambda x: -x[1])

    # Build waterfall data: Baseline first, then channels, then Total
    names = ['Baseline'] + [ch[0] for ch in sorted_channels] + ['Total']
    values = [baseline_val] + [ch[1] for ch in sorted_channels]
    total = sum(values)

    # Calculate running totals for bar positions
    # Each bar starts at the cumulative sum up to that point
    running_total = 0
    bar_starts = []
    bar_values = []

    for val in values:
        bar_starts.append(running_total)
        bar_values.append(val)
        running_total += val

    # Add total bar (starts at 0, goes to total)
    bar_starts.append(0)
    bar_values.append(total)

    # Y positions (reversed so Baseline is at top)
    y_pos = np.arange(len(names))[::-1]

    # Colors: baseline=orange, positive=blue, negative=red, total=green
    colors = []
    for i, (name, val) in enumerate(zip(names, bar_values)):
        if name == 'Baseline':
            colors.append('#E59400')  # Orange for baseline
        elif name == 'Total':
            colors.append('#2E8B57')  # Green for total
        elif val >= 0:
            colors.append('#4A90D9')  # Blue for positive
        else:
            colors.append('#DC143C')  # Red for negative

    # Draw bars
    bars = ax.barh(y_pos, bar_values, left=bar_starts, color=colors, alpha=0.8, height=0.6)

    # Draw connector lines between bars (waterfall effect)
    for i in range(len(names) - 2):  # Don't connect to Total
        end_of_current = bar_starts[i] + bar_values[i]
        # Draw line from end of current bar to start of next bar
        ax.plot([end_of_current, end_of_current],
                [y_pos[i] - 0.3, y_pos[i + 1] + 0.3],
                color='gray', linestyle='--', linewidth=1, alpha=0.5)

    # Labels and formatting
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel("Contribution Value")
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(axis="x", alpha=0.3)

    # Add value labels on bars
    for i, (bar, val, start) in enumerate(zip(bars, bar_values, bar_starts)):
        # Position label at end of bar
        label_x = start + val
        label_text = f'{val:,.0f}'

        # Adjust label position based on bar direction
        if val >= 0:
            ax.text(label_x + total * 0.01, y_pos[i], label_text,
                    va='center', ha='left', fontsize=9, fontweight='bold')
        else:
            ax.text(start - total * 0.01, y_pos[i], label_text,
                    va='center', ha='right', fontsize=9, fontweight='bold')

    # Add percentage labels
    for i, (name, val) in enumerate(zip(names, bar_values)):
        if name != 'Total' and total > 0:
            pct = (val / total) * 100
            ax.text(bar_starts[i] + val / 2, y_pos[i] - 0.35,
                    f'({pct:.1f}%)', va='top', ha='center', fontsize=8, color='gray')

    # Format x-axis
    def format_axis(x, p):
        if abs(x) >= 1e6:
            return f'{x/1e6:.1f}M'
        elif abs(x) >= 1e3:
            return f'{x/1e3:.0f}K'
        else:
            return f'{x:.0f}'

    ax.xaxis.set_major_formatter(plt.FuncFormatter(format_axis))

    # Set x-axis limits with padding
    ax.set_xlim(-total * 0.05, total * 1.15)

    plt.tight_layout()
    return fig


def create_baseline_channels_donut(
    baseline_contrib: float,
    channels_contrib: float,
    figsize: Tuple[int, int] = (8, 8),
    title: str = "Contribution: Baseline vs All Channels"
) -> plt.Figure:
    """Create a donut chart showing baseline vs all channels contribution.

    Args:
        baseline_contrib: Total baseline contribution (intercept + trend + seasonality)
        channels_contrib: Total contribution from all marketing channels
        figsize: Figure size
        title: Plot title

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    sizes = [baseline_contrib, channels_contrib]
    labels = ['Baseline', 'All Channels']
    colors = ['#E59400', '#4A90D9']

    # Create donut
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=None,
        autopct='%1.0f%%',
        colors=colors,
        startangle=90,
        pctdistance=0.75,
        wedgeprops=dict(width=0.5)
    )

    for autotext in autotexts:
        autotext.set_fontsize(16)
        autotext.set_fontweight('bold')

    ax.legend(wedges, labels, loc='lower center', ncol=2, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    plt.tight_layout()
    return fig


def create_contribution_rank_over_time(
    contrib_ts: pd.DataFrame,
    resample_freq: str = 'Q',
    figsize: Tuple[int, int] = (14, 7),
    title: str = "Contribution Rank Over Time"
) -> plt.Figure:
    """Create a line chart showing contribution rank changes over time.

    Args:
        contrib_ts: DataFrame with columns as channels and DatetimeIndex
        resample_freq: Resampling frequency ('Q' for quarterly, 'M' for monthly)
        figsize: Figure size
        title: Plot title

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Resample and calculate ranks
    contrib_resampled = contrib_ts.resample(resample_freq).sum()
    ranks = contrib_resampled.rank(axis=1, ascending=False)

    # Color palette
    n_cols = len(ranks.columns)
    colors = plt.cm.tab10(np.linspace(0, 1, min(n_cols, 10)))

    for i, col in enumerate(ranks.columns):
        color = colors[i % len(colors)]
        ax.plot(ranks.index, ranks[col], 'o-', label=col,
                color=color, linewidth=2, markersize=6)

    ax.set_xlabel('Time Period', fontsize=12)
    ax.set_ylabel('Contribution Rank', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.invert_yaxis()  # Rank 1 at top
    ax.set_yticks(range(1, len(ranks.columns) + 1))
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def create_roi_effectiveness_bubble(
    channel_metrics: List[Dict],
    figsize: Tuple[int, int] = (10, 8),
    title: str = "ROI vs. Effectiveness"
) -> plt.Figure:
    """Create a bubble chart showing ROI vs effectiveness.

    Args:
        channel_metrics: List of dicts with keys: name, spend, roi, effectiveness
        figsize: Figure size
        title: Plot title

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Normalize bubble sizes
    spends = [m['spend'] for m in channel_metrics]
    max_spend = max(spends) if spends else 1
    min_size = 200
    max_size = 2000

    # Color palette
    n_channels = len(channel_metrics)
    colors = plt.cm.tab10(np.linspace(0, 1, min(n_channels, 10)))

    for i, m in enumerate(channel_metrics):
        bubble_size = min_size + (m['spend'] / max_spend) * (max_size - min_size)
        color = colors[i % len(colors)]
        ax.scatter(m['roi'], m['effectiveness'], s=bubble_size, color=color,
                   alpha=0.6, edgecolors='white', linewidth=2, label=m['name'])

    ax.set_xlabel('ROI', fontsize=12)
    ax.set_ylabel('Effectiveness (Incremental Outcome)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Add note
    ax.text(0.02, 0.98,
            "Note: Bubble size = Media Spend\nEffectiveness = Incremental Outcome",
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    return fig


def create_response_curves(
    channel_data: List[Dict],
    figsize: Tuple[int, int] = (12, 8),
    title: str = "Response Curves by Marketing Channel"
) -> plt.Figure:
    """Create response curves showing spend vs contribution for each channel.

    Args:
        channel_data: List of dicts with keys:
            - name: channel display name
            - spend_range: array of spend values
            - contribution: array of contribution values at each spend level
            - current_spend: current total spend
            - current_contrib: current contribution
        figsize: Figure size
        title: Plot title

    Returns:
        Matplotlib figure
    """
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=figsize)

    # Color palette
    n_channels = len(channel_data)
    colors = plt.cm.tab10(np.linspace(0, 1, min(n_channels, 10)))

    for i, ch in enumerate(channel_data):
        color = colors[i % len(colors)]
        spend_range = ch['spend_range']
        contribution = ch['contribution']
        current_spend = ch['current_spend']

        # Find index of current spend
        current_idx = np.argmin(np.abs(spend_range - current_spend))

        # Plot solid line below current spend, dashed above
        ax.plot(spend_range[:current_idx+1], contribution[:current_idx+1],
                color=color, linewidth=2, label=ch['name'])
        ax.plot(spend_range[current_idx:], contribution[current_idx:],
                color=color, linewidth=2, linestyle='--')

        # Mark current spend point
        ax.scatter([current_spend], [contribution[current_idx]],
                   color=color, s=100, zorder=5, edgecolors='black', linewidth=2)

    ax.set_xlabel('Spend', fontsize=12)
    ax.set_ylabel('Incremental Outcome', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Format axes
    def format_axis(x, p):
        if x >= 1e6:
            return f'{x/1e6:.0f}M'
        elif x >= 1e3:
            return f'{x/1e3:.0f}K'
        else:
            return f'{x:.0f}'

    ax.xaxis.set_major_formatter(plt.FuncFormatter(format_axis))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(format_axis))

    # Build legend
    legend_elements = [
        Line2D([0], [0], color='gray', linestyle='-', label='Below current spend'),
        Line2D([0], [0], color='gray', linestyle='--', label='Above current spend'),
        Line2D([0], [0], marker='o', color='gray', label='Current spend',
               markersize=8, linestyle='None')
    ]

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + legend_elements, loc='upper left', fontsize=9)

    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def create_current_vs_marginal_roi(
    channels: List[str],
    current_rois: List[float],
    marginal_rois: List[float],
    channel_display_names: Optional[Dict[str, str]] = None,
    figsize: Tuple[int, int] = (12, 7),
    title: str = "Current ROI vs Marginal ROI by Channel"
) -> plt.Figure:
    """Create a grouped bar chart comparing current and marginal ROI.

    Args:
        channels: List of channel names
        current_rois: List of current (average) ROI values
        marginal_rois: List of marginal ROI values
        channel_display_names: Optional mapping of channel names to display names
        figsize: Figure size
        title: Plot title

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Create display names
    if channel_display_names:
        display_names = [channel_display_names.get(ch, ch) for ch in channels]
    else:
        display_names = [
            ch.replace("PaidMedia_", "")
              .replace("_spend", "")
              .replace("_", " ")
            for ch in channels
        ]

    # Sort by current spend (approximated by keeping original order)
    x = np.arange(len(display_names))
    width = 0.35

    bars1 = ax.bar(x - width/2, current_rois, width, label='Current (Avg) ROI', color='#4A90D9')
    bars2 = ax.bar(x + width/2, marginal_rois, width, label='Marginal ROI', color='#E59400')

    ax.axhline(1, color='red', linestyle='--', alpha=0.7, label='Breakeven (ROI = 1)')
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, rotation=45, ha='right')
    ax.set_ylabel('ROI ($)')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    return fig


def create_spend_vs_breakeven(
    channels: List[str],
    current_spends: List[float],
    breakeven_spends: List[float],
    channel_display_names: Optional[Dict[str, str]] = None,
    figsize: Tuple[int, int] = (12, 7),
    title: str = "Current Spend vs Breakeven Spend (Headroom)"
) -> plt.Figure:
    """Create a horizontal bar chart comparing current spend to breakeven spend.

    Args:
        channels: List of channel names
        current_spends: List of current spend values
        breakeven_spends: List of breakeven spend values
        channel_display_names: Optional mapping of channel names to display names
        figsize: Figure size
        title: Plot title

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Filter to channels with valid breakeven
    valid_indices = [i for i, b in enumerate(breakeven_spends) if b is not None]
    channels = [channels[i] for i in valid_indices]
    current_spends = [current_spends[i] for i in valid_indices]
    breakeven_spends = [breakeven_spends[i] for i in valid_indices]

    # Create display names
    if channel_display_names:
        display_names = [channel_display_names.get(ch, ch) for ch in channels]
    else:
        display_names = [
            ch.replace("PaidMedia_", "")
              .replace("_spend", "")
              .replace("_", " ")
            for ch in channels
        ]

    y = np.arange(len(display_names))
    height = 0.35

    bars1 = ax.barh(y - height/2, current_spends, height, label='Current Spend', color='#4A90D9')
    bars2 = ax.barh(y + height/2, breakeven_spends, height, label='Breakeven Spend', color='#90D94A', alpha=0.7)

    ax.set_yticks(y)
    ax.set_yticklabels(display_names)
    ax.set_xlabel('Spend ($)')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()

    # Format x-axis
    def format_axis(x, p):
        if x >= 1e6:
            return f'{x/1e6:.1f}M'
        elif x >= 1e3:
            return f'{x/1e3:.0f}K'
        else:
            return f'{x:.0f}'

    ax.xaxis.set_major_formatter(plt.FuncFormatter(format_axis))
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    return fig


def create_stacked_contributions_area(
    dates: pd.DatetimeIndex,
    contributions: pd.DataFrame,
    figsize: Tuple[int, int] = (14, 7),
    title: str = "Contribution Over Time by Channel"
) -> plt.Figure:
    """Create a stacked area chart of contributions over time.

    Args:
        dates: DatetimeIndex for x-axis
        contributions: DataFrame with columns as components, values as contributions
        figsize: Figure size
        title: Plot title

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Ensure all values are positive for stacking
    contrib_positive = contributions.clip(lower=0)

    # Color palette
    n_cols = len(contrib_positive.columns)
    colors = plt.cm.tab20(np.linspace(0, 1, n_cols))

    # Stack plot
    ax.stackplot(dates,
                 [contrib_positive[col].values for col in contrib_positive.columns],
                 labels=contrib_positive.columns,
                 colors=colors)

    ax.set_xlabel('Time Period', fontsize=12)
    ax.set_ylabel('Contribution', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', bbox_to_anchor=(0, -0.15), ncol=5, fontsize=9)

    # Format y-axis
    def format_axis(x, p):
        if x >= 1e6:
            return f'{x/1e6:.1f}M'
        elif x >= 1e3:
            return f'{x/1e3:.0f}K'
        else:
            return f'{x:.0f}'

    ax.yaxis.set_major_formatter(plt.FuncFormatter(format_axis))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    return fig
