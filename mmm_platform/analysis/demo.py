"""
Demo module for testing MMM Platform features without running a model.

Provides mock data and objects that simulate a fitted MMM model,
allowing testing of all analysis features:
- Marginal ROI analysis
- Executive summary generation
- Combined model analysis
- All visualizations

Usage:
    from mmm_platform.analysis.demo import create_demo_scenario
    demo = create_demo_scenario()

    # Test marginal ROI
    demo.marginal_analyzer.print_priority_table()

    # Test executive summary
    demo.exec_generator.print_summary()

    # Test combined models
    demo.combined_analyzer.print_summary(demo.combined_result)

    # Test visualizations
    demo.show_all_visualizations()
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
import xarray as xr
import arviz as az
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# MOCK INFERENCE DATA
# =============================================================================

def create_mock_inference_data(
    channels: List[str],
    n_chains: int = 4,
    n_samples: int = 1000,
    seed: int = 42
) -> az.InferenceData:
    """
    Create a proper ArviZ InferenceData object with mock posterior samples.

    This creates real xarray-based inference data that works with all
    arviz functions (az.summary, az.hdi, etc.)

    Parameters
    ----------
    channels : list[str]
        Channel names
    n_chains : int
        Number of MCMC chains
    n_samples : int
        Number of samples per chain
    seed : int
        Random seed for reproducibility

    Returns
    -------
    az.InferenceData
        Properly structured inference data
    """
    np.random.seed(seed)
    n_channels = len(channels)

    # Beta (saturation effect strength) - varies by channel
    beta_means = {
        "google_search_brand": 0.15,
        "google_search_nonbrand": 0.08,
        "google_pmax": 0.12,
        "meta_prospecting": 0.10,
        "meta_retargeting": 0.18,
        "tiktok": 0.05,
        "programmatic_display": 0.04,
        "youtube": 0.07,
        "affiliate": 0.09,
        "email": 0.20,
    }

    # Generate posterior samples
    beta_data = np.zeros((n_chains, n_samples, n_channels))
    for i, ch in enumerate(channels):
        mean = beta_means.get(ch, 0.10)
        std = mean * 0.3  # 30% coefficient of variation
        beta_data[:, :, i] = np.random.normal(mean, std, (n_chains, n_samples))
        beta_data[:, :, i] = np.clip(beta_data[:, :, i], 0.01, None)

    # Alpha (adstock decay) - typically 0.3-0.8
    alpha_data = np.random.beta(5, 3, (n_chains, n_samples, n_channels)) * 0.5 + 0.3

    # Lambda (saturation rate) - typically 1-5
    lam_data = np.random.gamma(3, 1, (n_chains, n_samples, n_channels)) + 1

    # Create xarray DataArrays with proper coordinates
    coords = {
        "chain": np.arange(n_chains),
        "draw": np.arange(n_samples),
        "channel": channels,
    }

    posterior_dict = {
        "saturation_beta": xr.DataArray(
            beta_data,
            dims=["chain", "draw", "channel"],
            coords=coords
        ),
        "adstock_alpha": xr.DataArray(
            alpha_data,
            dims=["chain", "draw", "channel"],
            coords=coords
        ),
        "saturation_lam": xr.DataArray(
            lam_data,
            dims=["chain", "draw", "channel"],
            coords=coords
        ),
    }

    # Create xarray Dataset
    posterior_ds = xr.Dataset(posterior_dict)

    # Create InferenceData object
    idata = az.InferenceData(posterior=posterior_ds)

    return idata


# =============================================================================
# MOCK DATA GENERATOR
# =============================================================================

def generate_mock_data(
    n_weeks: int = 104,
    channels: Optional[List[str]] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Generate realistic mock MMM data.

    Parameters
    ----------
    n_weeks : int
        Number of weeks of data
    channels : list[str], optional
        Channel names. Uses defaults if not provided.
    seed : int
        Random seed for reproducibility

    Returns
    -------
    dict
        Contains df_scaled, contribs, dates, channel_cols, target_col
    """
    np.random.seed(seed)

    if channels is None:
        channels = [
            "google_search_brand",
            "google_search_nonbrand",
            "google_pmax",
            "meta_prospecting",
            "meta_retargeting",
            "tiktok",
            "programmatic_display",
            "youtube",
            "affiliate",
            "email",
        ]

    # Generate dates
    start_date = datetime(2022, 1, 3)  # First Monday of 2022
    dates = pd.date_range(start=start_date, periods=n_weeks, freq='W-MON')

    # Generate spend data with realistic patterns
    spend_data = {}
    spend_levels = {
        "google_search_brand": 50000,
        "google_search_nonbrand": 80000,
        "google_pmax": 60000,
        "meta_prospecting": 100000,
        "meta_retargeting": 40000,
        "tiktok": 30000,
        "programmatic_display": 25000,
        "youtube": 45000,
        "affiliate": 20000,
        "email": 5000,
    }

    for ch in channels:
        base_level = spend_levels.get(ch, 30000)
        # Add seasonality and trend
        seasonality = 1 + 0.2 * np.sin(2 * np.pi * np.arange(n_weeks) / 52)
        trend = 1 + 0.001 * np.arange(n_weeks)
        noise = np.random.normal(1, 0.1, n_weeks)
        spend_data[ch] = base_level * seasonality * trend * noise
        spend_data[ch] = np.clip(spend_data[ch], 0, None)

    # Generate target (revenue) with baseline + channel contributions
    baseline = 2000000  # $2M weekly baseline
    seasonality = 1 + 0.15 * np.sin(2 * np.pi * np.arange(n_weeks) / 52)
    trend = 1 + 0.002 * np.arange(n_weeks)

    target = baseline * seasonality * trend

    # Add channel contributions (with diminishing returns)
    roi_multipliers = {
        "google_search_brand": 3.5,
        "google_search_nonbrand": 2.2,
        "google_pmax": 2.8,
        "meta_prospecting": 1.8,
        "meta_retargeting": 4.0,
        "tiktok": 1.2,
        "programmatic_display": 0.9,
        "youtube": 1.5,
        "affiliate": 2.5,
        "email": 5.0,
    }

    contrib_data = {}
    for ch in channels:
        roi = roi_multipliers.get(ch, 1.5)
        # Apply saturation (diminishing returns)
        normalized_spend = spend_data[ch] / spend_data[ch].max()
        saturation = 1 - np.exp(-2 * normalized_spend)  # Simplified saturation
        contrib_data[ch] = spend_data[ch] * roi * saturation * 0.5  # Scale down
        target += contrib_data[ch]

    # Add noise to target
    target += np.random.normal(0, target * 0.05)

    # Create DataFrames
    df_scaled = pd.DataFrame(spend_data, index=dates)
    df_scaled['revenue'] = target
    df_scaled['intercept'] = baseline * seasonality * trend

    # Scale to normalized units (as MMM would use)
    spend_scale = 1000
    revenue_scale = 1000

    for ch in channels:
        df_scaled[ch] = df_scaled[ch] / spend_scale
    df_scaled['revenue'] = df_scaled['revenue'] / revenue_scale
    df_scaled['intercept'] = df_scaled['intercept'] / revenue_scale

    # Create contributions DataFrame
    contribs = pd.DataFrame(index=dates)
    contribs['intercept'] = df_scaled['intercept']
    for ch in channels:
        contribs[ch] = contrib_data[ch] / revenue_scale
    contribs['revenue'] = df_scaled['revenue']

    return {
        'df_scaled': df_scaled,
        'contribs': contribs,
        'dates': dates,
        'channel_cols': channels,
        'target_col': 'revenue',
        'spend_scale': spend_scale,
        'revenue_scale': revenue_scale,
    }


# =============================================================================
# DEMO SCENARIO
# =============================================================================

@dataclass
class DemoScenario:
    """Container for all demo objects and data."""

    # Raw data
    df_scaled: pd.DataFrame
    contribs: pd.DataFrame
    dates: pd.DatetimeIndex
    channel_cols: List[str]
    target_col: str
    spend_scale: float
    revenue_scale: float

    # Mock inference data (proper arviz InferenceData objects)
    idata: az.InferenceData
    idata_offline: az.InferenceData  # For combined model demo

    # Analyzers
    marginal_analyzer: Any
    exec_generator: Any
    combined_analyzer: Any
    combined_result: Any

    # Offline data (for combined model)
    contribs_offline: pd.DataFrame

    def show_all_visualizations(self, save_path: Optional[str] = None):
        """
        Generate and display all visualizations.

        Parameters
        ----------
        save_path : str, optional
            Directory to save figures. If None, displays interactively.
        """
        import matplotlib.pyplot as plt
        from . import visualizations as viz

        print("Generating visualizations...")

        # 1. Baseline vs Channels Donut
        breakdown = self._get_contribution_breakdown()
        fig1 = viz.create_baseline_channels_donut(
            baseline_contrib=breakdown['baseline'],
            channels_contrib=breakdown['channels'],
            title="Demo: Baseline vs All Channels"
        )
        self._handle_figure(fig1, "01_donut", save_path)

        # 2. Contribution Rank Over Time
        contrib_ts = self.contribs[self.channel_cols].copy()
        contrib_ts.index = self.dates
        fig2 = viz.create_contribution_rank_over_time(
            contrib_ts=contrib_ts,
            resample_freq='Q',
            title="Demo: Contribution Rank Over Time"
        )
        self._handle_figure(fig2, "02_rank_over_time", save_path)

        # 3. ROI vs Effectiveness Bubble
        metrics = self._get_channel_metrics()
        fig3 = viz.create_roi_effectiveness_bubble(
            channel_metrics=metrics,
            title="Demo: ROI vs Effectiveness"
        )
        self._handle_figure(fig3, "03_bubble", save_path)

        # 4. Response Curves
        response_data = self._get_response_curves()
        fig4 = viz.create_response_curves(
            channel_data=response_data[:7],  # Top 7
            title="Demo: Response Curves (Top 7 Channels)"
        )
        self._handle_figure(fig4, "04_response_curves", save_path)

        # 5. Current vs Marginal ROI
        priority = self.marginal_analyzer.run_full_analysis()
        channels = [ch.channel for ch in priority.channel_analysis]
        current_rois = [ch.current_roi for ch in priority.channel_analysis]
        marginal_rois = [ch.marginal_roi for ch in priority.channel_analysis]

        fig5 = viz.create_current_vs_marginal_roi(
            channels=channels,
            current_rois=current_rois,
            marginal_rois=marginal_rois,
            title="Demo: Current vs Marginal ROI"
        )
        self._handle_figure(fig5, "05_current_vs_marginal", save_path)

        # 6. Spend vs Breakeven
        current_spends = [ch.current_spend for ch in priority.channel_analysis]
        breakeven_spends = [ch.breakeven_spend for ch in priority.channel_analysis]

        fig6 = viz.create_spend_vs_breakeven(
            channels=channels,
            current_spends=current_spends,
            breakeven_spends=breakeven_spends,
            title="Demo: Current vs Breakeven Spend"
        )
        self._handle_figure(fig6, "06_spend_vs_breakeven", save_path)

        # 7. Stacked Contributions Area
        fig7 = viz.create_stacked_contributions_area(
            dates=self.dates,
            contributions=self.contribs[self.channel_cols],
            title="Demo: Contributions Over Time"
        )
        self._handle_figure(fig7, "07_stacked_area", save_path)

        # 8. Contribution Waterfall
        contrib_dict = {ch: float(self.contribs[ch].sum()) for ch in self.channel_cols}
        contrib_dict['Baseline'] = float(self.contribs['intercept'].sum())

        fig8 = viz.create_contribution_waterfall_chart(
            contributions=contrib_dict,
            title="Demo: Contribution Waterfall"
        )
        self._handle_figure(fig8, "08_waterfall", save_path)

        print(f"Generated 8 visualizations" + (f" saved to {save_path}" if save_path else ""))

        if save_path is None:
            plt.show()

    def _handle_figure(self, fig, name: str, save_path: Optional[str]):
        """Save or display a figure."""
        import matplotlib.pyplot as plt
        from pathlib import Path

        if save_path:
            path = Path(save_path)
            path.mkdir(parents=True, exist_ok=True)
            fig.savefig(path / f"{name}.png", dpi=150, bbox_inches='tight')
            plt.close(fig)
        # If not saving, figures will be shown at the end

    def _get_contribution_breakdown(self) -> Dict[str, float]:
        """Get baseline vs channels contribution."""
        baseline = float(self.contribs['intercept'].sum()) * self.revenue_scale
        channels = float(self.contribs[self.channel_cols].sum().sum()) * self.revenue_scale
        return {'baseline': baseline, 'channels': channels}

    def _get_channel_metrics(self) -> List[Dict]:
        """Get channel metrics for bubble chart."""
        metrics = []
        for ch in self.channel_cols:
            spend = float(self.df_scaled[ch].sum()) * self.spend_scale
            contrib = float(self.contribs[ch].sum()) * self.revenue_scale
            roi = contrib / (spend + 1e-9)
            metrics.append({
                'name': ch.replace('_', ' ').title(),
                'spend': spend,
                'roi': roi,
                'effectiveness': contrib,
            })
        return metrics

    def _get_response_curves(self) -> List[Dict]:
        """Generate response curve data for each channel."""
        curves = []

        for ch in self.channel_cols:
            current_spend = float(self.df_scaled[ch].sum()) * self.spend_scale
            current_contrib = float(self.contribs[ch].sum()) * self.revenue_scale

            # Generate spend range (0 to 3x current)
            spend_range = np.linspace(0, current_spend * 3, 200)

            # Simulate saturation curve
            max_contrib = current_contrib * 2.5  # Asymptote
            lam = 2 / current_spend  # Calibrate saturation rate
            contribution = max_contrib * (1 - np.exp(-lam * spend_range))

            curves.append({
                'name': ch.replace('_', ' ').title(),
                'spend_range': spend_range,
                'contribution': contribution,
                'current_spend': current_spend,
                'current_contrib': current_contrib,
            })

        # Sort by current contribution
        curves = sorted(curves, key=lambda x: -x['current_contrib'])
        return curves

    def print_all_summaries(self):
        """Print all analysis summaries."""
        print("\n" + "=" * 80)
        print("DEMO: MARGINAL ROI PRIORITY TABLE")
        print("=" * 80)
        self.marginal_analyzer.print_priority_table()

        print("\n" + "=" * 80)
        print("DEMO: EXECUTIVE SUMMARY")
        print("=" * 80)
        self.exec_generator.print_summary()

        print("\n" + "=" * 80)
        print("DEMO: COMBINED MODEL ANALYSIS")
        print("=" * 80)
        self.combined_analyzer.print_summary(self.combined_result)

    def get_all_dataframes(self) -> Dict[str, pd.DataFrame]:
        """Get all analysis results as DataFrames."""
        priority_df = self.marginal_analyzer.get_priority_table()
        combined_df = self.combined_analyzer.get_summary_table(
            self.combined_result.combined_analysis
        )

        return {
            'priority_table': priority_df,
            'combined_analysis': combined_df,
            'contributions': self.contribs,
            'spend_data': self.df_scaled[self.channel_cols],
        }


def create_demo_scenario(
    n_weeks: int = 104,
    seed: int = 42,
    online_margin: float = 0.35,
    offline_margin: float = 0.25,
) -> DemoScenario:
    """
    Create a complete demo scenario with mock data.

    Parameters
    ----------
    n_weeks : int
        Number of weeks of data
    seed : int
        Random seed
    online_margin : float
        Online profit margin for combined model
    offline_margin : float
        Offline profit margin for combined model

    Returns
    -------
    DemoScenario
        Complete demo with all analyzers ready to use

    Example
    -------
    >>> from mmm_platform.analysis.demo import create_demo_scenario
    >>> demo = create_demo_scenario()
    >>>
    >>> # Print all summaries
    >>> demo.print_all_summaries()
    >>>
    >>> # Show visualizations
    >>> demo.show_all_visualizations()
    >>>
    >>> # Get DataFrames
    >>> dfs = demo.get_all_dataframes()
    >>> print(dfs['priority_table'])
    """
    from .marginal_roi import MarginalROIAnalyzer
    from .executive_summary import ExecutiveSummaryGenerator
    from .combined_models import CombinedModelAnalyzer

    print("Creating demo scenario...")

    # Generate mock data
    data = generate_mock_data(n_weeks=n_weeks, seed=seed)

    # Create proper arviz-compatible inference data
    idata = create_mock_inference_data(data['channel_cols'], seed=seed)

    # Create marginal ROI analyzer
    marginal_analyzer = MarginalROIAnalyzer(
        idata=idata,
        df_scaled=data['df_scaled'],
        contribs=data['contribs'],
        channel_cols=data['channel_cols'],
        target_col=data['target_col'],
        spend_scale=data['spend_scale'],
        revenue_scale=data['revenue_scale'],
    )

    # Create executive summary generator
    exec_generator = ExecutiveSummaryGenerator(marginal_analyzer)

    # Create offline data (different ROI profile for combined model demo)
    np.random.seed(seed + 1)
    contribs_offline = data['contribs'].copy()
    for ch in data['channel_cols']:
        # Offline has different contribution pattern
        multiplier = np.random.uniform(0.3, 1.5)
        contribs_offline[ch] = contribs_offline[ch] * multiplier

    # Create offline inference data (proper arviz-compatible)
    idata_offline = create_mock_inference_data(data['channel_cols'], seed=seed + 1)

    # Create offline marginal analyzer
    marginal_analyzer_offline = MarginalROIAnalyzer(
        idata=idata_offline,
        df_scaled=data['df_scaled'],
        contribs=contribs_offline,
        channel_cols=data['channel_cols'],
        target_col=data['target_col'],
        spend_scale=data['spend_scale'],
        revenue_scale=data['revenue_scale'],
    )

    # Create combined model analyzer
    combined_analyzer = CombinedModelAnalyzer(
        online_margin=online_margin,
        offline_margin=offline_margin,
    )

    # Run analyses
    online_result = marginal_analyzer.run_full_analysis()
    offline_result = marginal_analyzer_offline.run_full_analysis()

    combined_result = combined_analyzer.run_full_analysis(
        channel_analysis_online=online_result.channel_analysis,
        channel_analysis_offline=offline_result.channel_analysis,
    )

    print("Demo scenario ready!")
    print(f"  - {len(data['channel_cols'])} channels")
    print(f"  - {n_weeks} weeks of data")
    print(f"  - Online margin: {online_margin*100:.0f}%")
    print(f"  - Offline margin: {offline_margin*100:.0f}%")

    return DemoScenario(
        df_scaled=data['df_scaled'],
        contribs=data['contribs'],
        dates=data['dates'],
        channel_cols=data['channel_cols'],
        target_col=data['target_col'],
        spend_scale=data['spend_scale'],
        revenue_scale=data['revenue_scale'],
        idata=idata,
        idata_offline=idata_offline,
        marginal_analyzer=marginal_analyzer,
        exec_generator=exec_generator,
        combined_analyzer=combined_analyzer,
        combined_result=combined_result,
        contribs_offline=contribs_offline,
    )


# =============================================================================
# QUICK TEST FUNCTIONS
# =============================================================================

def test_marginal_roi():
    """Quick test of marginal ROI analysis."""
    demo = create_demo_scenario()
    demo.marginal_analyzer.print_priority_table()
    return demo.marginal_analyzer.get_priority_table()


def test_executive_summary():
    """Quick test of executive summary."""
    demo = create_demo_scenario()
    demo.exec_generator.print_summary()
    return demo.exec_generator.get_summary_dict()


def test_combined_models():
    """Quick test of combined model analysis."""
    demo = create_demo_scenario()
    demo.combined_analyzer.print_summary(demo.combined_result)
    return demo.combined_analyzer.get_summary_dict(demo.combined_result)


def test_visualizations(save_path: Optional[str] = None):
    """Quick test of all visualizations."""
    demo = create_demo_scenario()
    demo.show_all_visualizations(save_path=save_path)


def run_full_demo():
    """Run complete demo with all features."""
    demo = create_demo_scenario()

    print("\n" + "=" * 80)
    print("MMM PLATFORM DEMO - ALL FEATURES")
    print("=" * 80)

    # Print summaries
    demo.print_all_summaries()

    # Get DataFrames
    dfs = demo.get_all_dataframes()
    print("\n" + "=" * 80)
    print("AVAILABLE DATAFRAMES")
    print("=" * 80)
    for name, df in dfs.items():
        print(f"\n{name}: {df.shape[0]} rows x {df.shape[1]} columns")
        print(df.head(3).to_string())

    # Show visualizations
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS...")
    print("=" * 80)
    demo.show_all_visualizations()

    return demo


if __name__ == "__main__":
    run_full_demo()
