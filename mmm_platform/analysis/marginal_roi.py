"""
Marginal ROI Analysis and Investment Prioritization.

Provides:
- Marginal ROI calculation using saturation curve derivatives
- Breakeven spend analysis
- Channel investment prioritization (INCREASE/HOLD/REDUCE)
- Headroom calculations
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from scipy.optimize import brentq
import arviz as az
from pymc_marketing.mmm.transformers import geometric_adstock, logistic_saturation
import logging

logger = logging.getLogger(__name__)


def logistic_saturation_derivative(x: np.ndarray, lam: float) -> np.ndarray:
    """
    Derivative of logistic saturation function.

    saturation(x) = (1 - exp(-lam*x)) / (1 + exp(-lam*x))
    derivative = 2*lam*exp(-lam*x) / (1 + exp(-lam*x))^2

    Parameters
    ----------
    x : np.ndarray
        Input values (normalized spend)
    lam : float
        Saturation parameter

    Returns
    -------
    np.ndarray
        Derivative values at each x
    """
    exp_term = np.exp(-lam * x)
    return 2 * lam * exp_term / (1 + exp_term)**2


def calculate_marginal_roi(
    x_normalized: float,
    beta: float,
    lam: float,
    target_scale: float,
    x_max: float
) -> float:
    """
    Calculate marginal ROI at a given spend level.

    Uses chain rule: d(contribution)/d(spend) = d(contribution)/d(x_norm) * d(x_norm)/d(spend)
                                              = beta * target_scale * sat_deriv * (1/x_max)

    Parameters
    ----------
    x_normalized : float
        Normalized spend level (spend / x_max)
    beta : float
        Saturation beta coefficient
    lam : float
        Saturation lambda parameter
    target_scale : float
        Scale factor for target variable
    x_max : float
        Maximum spend value (for denormalization)

    Returns
    -------
    float
        Marginal ROI at the given spend level
    """
    sat_deriv = logistic_saturation_derivative(np.array([x_normalized]), lam)[0]
    # Divide by x_max to convert from normalized to actual spend units
    return beta * target_scale * sat_deriv / (x_max + 1e-9)


def find_breakeven_spend(
    beta: float,
    lam: float,
    target_scale: float,
    x_max: float,
    max_spend_normalized: float = 100.0
) -> Optional[float]:
    """
    Find spend level where marginal ROI = 1 (breakeven).

    Parameters
    ----------
    beta : float
        Saturation beta coefficient
    lam : float
        Saturation lambda parameter
    target_scale : float
        Scale factor for target variable
    x_max : float
        Maximum spend value
    max_spend_normalized : float
        Maximum normalized spend to search

    Returns
    -------
    float or None
        Normalized spend level at breakeven, or None if not found
    """
    # Check if marginal ROI at x=0 is already below 1
    marginal_at_zero = calculate_marginal_roi(1e-6, beta, lam, target_scale, x_max)
    if marginal_at_zero < 1:
        return 0.0  # Already below breakeven

    # Check if marginal ROI at max is still above 1
    marginal_at_max = calculate_marginal_roi(max_spend_normalized, beta, lam, target_scale, x_max)
    if marginal_at_max > 1:
        return max_spend_normalized  # Still above breakeven at max

    # Find the crossover point
    try:
        breakeven = brentq(
            lambda x: calculate_marginal_roi(x, beta, lam, target_scale, x_max) - 1,
            1e-6,
            max_spend_normalized
        )
        return float(breakeven)
    except Exception:
        return None


@dataclass
class ChannelMarginalROI:
    """Marginal ROI analysis results for a single channel."""
    channel: str
    channel_name: str  # Display name
    current_spend: float  # In real units
    current_roi: float  # Average ROI
    marginal_roi: float  # Marginal ROI at current spend
    breakeven_spend: Optional[float]  # Spend level where marginal ROI = 1
    headroom: bool  # Is current spend below breakeven?
    headroom_amount: float  # How much more can be spent profitably
    priority_rank: int = 0  # Rank by marginal ROI (1 = highest)

    # Uncertainty metrics
    roi_5pct: float = 0.0
    roi_95pct: float = 0.0
    roi_uncertainty: float = 0.0
    prob_profitable: float = 0.0
    needs_test: bool = False


@dataclass
class InvestmentPriorityResult:
    """Complete investment priority analysis."""
    channel_analysis: List[ChannelMarginalROI]
    increase_channels: List[ChannelMarginalROI]
    hold_channels: List[ChannelMarginalROI]
    reduce_channels: List[ChannelMarginalROI]
    channels_needing_test: List[ChannelMarginalROI]
    total_spend: float
    total_contribution: float
    portfolio_roi: float
    reallocation_potential: float  # Funds from reduce channels
    headroom_available: float  # Total headroom in increase channels


class MarginalROIAnalyzer:
    """
    Analyzer for marginal ROI and investment prioritization.

    Calculates true marginal ROI using saturation curve derivatives,
    finds breakeven spend levels, and categorizes channels into
    INCREASE/HOLD/REDUCE recommendations.
    """

    def __init__(
        self,
        idata: Any,
        df_scaled: pd.DataFrame,
        contribs: pd.DataFrame,
        channel_cols: List[str],
        target_col: str,
        spend_scale: float = 1.0,
        revenue_scale: float = 1.0,
        l_max: int = 8,
        uncertainty_threshold: float = 5.0,  # ROI CI width threshold for "needs test"
        increase_threshold: float = 1.5,  # Marginal ROI > this AND headroom -> INCREASE
        reduce_threshold: float = 1.0,  # Marginal ROI < this -> REDUCE
        display_names: Optional[Dict[str, str]] = None,  # Column name -> display name
    ):
        """
        Initialize MarginalROIAnalyzer.

        Parameters
        ----------
        idata : InferenceData
            ArviZ inference data with posterior samples
        df_scaled : pd.DataFrame
            Scaled data used for modeling
        contribs : pd.DataFrame
            Contribution dataframe from model
        channel_cols : list[str]
            Channel column names
        target_col : str
            Target column name
        spend_scale : float
            Scale factor for spend (to convert back to real units)
        revenue_scale : float
            Scale factor for revenue
        l_max : int
            Maximum lag for adstock
        uncertainty_threshold : float
            ROI CI width above which channel needs validation test
        increase_threshold : float
            Marginal ROI threshold for INCREASE recommendation
        reduce_threshold : float
            Marginal ROI threshold for REDUCE recommendation
        display_names : dict, optional
            Mapping from column names to display names
        """
        self.idata = idata
        self.df_scaled = df_scaled
        self.contribs = contribs
        self.channel_cols = channel_cols
        self.target_col = target_col
        self.spend_scale = spend_scale
        self.revenue_scale = revenue_scale
        self.l_max = l_max
        self.uncertainty_threshold = uncertainty_threshold
        self.increase_threshold = increase_threshold
        self.reduce_threshold = reduce_threshold
        self.display_names = display_names or {}

        # Get posterior summaries
        self._alpha_summary = None
        self._lam_summary = None
        self._beta_summary = None

    @property
    def alpha_summary(self):
        if self._alpha_summary is None:
            self._alpha_summary = az.summary(self.idata, var_names=["adstock_alpha"])
        return self._alpha_summary

    @property
    def lam_summary(self):
        if self._lam_summary is None:
            self._lam_summary = az.summary(self.idata, var_names=["saturation_lam"])
        return self._lam_summary

    @property
    def beta_summary(self):
        if self._beta_summary is None:
            self._beta_summary = az.summary(self.idata, var_names=["saturation_beta"])
        return self._beta_summary

    def _get_channel_params(self, channel: str) -> Tuple[float, float, float]:
        """Get posterior mean parameters for a channel."""
        try:
            alpha = float(self.alpha_summary.loc[f"adstock_alpha[{channel}]", "mean"])
            lam = float(self.lam_summary.loc[f"saturation_lam[{channel}]", "mean"])
            beta = float(self.beta_summary.loc[f"saturation_beta[{channel}]", "mean"])
        except KeyError:
            # Try alternative indexing
            idx = self.channel_cols.index(channel)
            alpha = float(self.alpha_summary.iloc[idx]["mean"])
            lam = float(self.lam_summary.iloc[idx]["mean"])
            beta = float(self.beta_summary.iloc[idx]["mean"])
        return alpha, lam, beta

    def _get_channel_display_name(self, channel: str) -> str:
        """Convert channel column name to display name."""
        # Use provided display name if available
        if channel in self.display_names:
            return self.display_names[channel]
        # Fallback to formatted column name
        return (channel
                .replace("PaidMedia_", "")
                .replace("_spend", "")
                .replace("_", " "))

    def analyze_channel(self, channel: str) -> ChannelMarginalROI:
        """
        Analyze marginal ROI for a single channel.

        Parameters
        ----------
        channel : str
            Channel column name

        Returns
        -------
        ChannelMarginalROI
            Complete marginal ROI analysis for the channel
        """
        # Get parameters
        alpha, lam, beta = self._get_channel_params(channel)

        # Get spend and contribution data
        total_spend = float(self.df_scaled[channel].sum())
        weekly_avg_spend = total_spend / len(self.df_scaled)
        x_max = float(self.df_scaled[channel].max())

        current_contribution = float(self.contribs[channel].sum())
        current_roi = current_contribution / (total_spend + 1e-9)

        # Get target scale
        target_scale = float(self.df_scaled[self.target_col].max())

        # Calculate marginal ROI at current spend
        current_spend_normalized = weekly_avg_spend / (x_max + 1e-9)
        marginal_roi = calculate_marginal_roi(
            current_spend_normalized, beta, lam, target_scale, x_max
        )

        # Find breakeven spend
        breakeven_normalized = find_breakeven_spend(beta, lam, target_scale, x_max)
        if breakeven_normalized is not None:
            # Scale back to total spend over period
            breakeven_spend = breakeven_normalized * (x_max + 1e-9) * len(self.df_scaled)
            breakeven_spend_real = breakeven_spend * self.spend_scale
        else:
            breakeven_spend_real = None

        # Calculate headroom
        total_spend_real = total_spend * self.spend_scale
        if breakeven_spend_real is not None and breakeven_spend_real > total_spend_real:
            headroom = True
            headroom_amount = breakeven_spend_real - total_spend_real
        else:
            headroom = False
            headroom_amount = 0.0

        # Get uncertainty from posterior
        try:
            beta_posterior = self.idata.posterior["saturation_beta"]
            beta_samples = beta_posterior.sel(channel=channel).values.flatten()
        except (KeyError, ValueError):
            idx = self.channel_cols.index(channel)
            beta_samples = beta_posterior.isel(channel=idx).values.flatten()

        # Calculate ROI uncertainty
        # Simplified: use beta variation as proxy for ROI variation
        roi_samples = beta_samples * current_contribution / (total_spend * beta_samples.mean() + 1e-9)
        roi_5pct = float(np.percentile(roi_samples, 5))
        roi_95pct = float(np.percentile(roi_samples, 95))
        roi_uncertainty = roi_95pct - roi_5pct
        prob_profitable = float((roi_samples > 1.0).mean())
        needs_test = roi_uncertainty > self.uncertainty_threshold

        return ChannelMarginalROI(
            channel=channel,
            channel_name=self._get_channel_display_name(channel),
            current_spend=total_spend_real,
            current_roi=current_roi,
            marginal_roi=marginal_roi,
            breakeven_spend=breakeven_spend_real,
            headroom=headroom,
            headroom_amount=headroom_amount,
            roi_5pct=roi_5pct,
            roi_95pct=roi_95pct,
            roi_uncertainty=roi_uncertainty,
            prob_profitable=prob_profitable,
            needs_test=needs_test,
        )

    def run_full_analysis(self) -> InvestmentPriorityResult:
        """
        Run complete investment priority analysis for all channels.

        Returns
        -------
        InvestmentPriorityResult
            Complete analysis with categorized recommendations
        """
        # Analyze all channels
        channel_results = []
        for ch in self.channel_cols:
            try:
                result = self.analyze_channel(ch)
                channel_results.append(result)
            except Exception as e:
                logger.warning(f"Could not analyze channel {ch}: {e}")

        # Rank by marginal ROI
        channel_results = sorted(channel_results, key=lambda x: -x.marginal_roi)
        for i, ch in enumerate(channel_results):
            ch.priority_rank = i + 1

        # Categorize channels
        increase_channels = []
        hold_channels = []
        reduce_channels = []

        for ch in channel_results:
            if ch.marginal_roi > self.increase_threshold and ch.headroom:
                increase_channels.append(ch)
            elif ch.marginal_roi < self.reduce_threshold:
                reduce_channels.append(ch)
            else:
                hold_channels.append(ch)

        # Channels needing validation
        channels_needing_test = [ch for ch in channel_results if ch.needs_test]

        # Portfolio metrics
        total_spend = sum(ch.current_spend for ch in channel_results)
        total_contribution = sum(
            float(self.contribs[ch.channel].sum()) * self.revenue_scale
            for ch in channel_results
        )
        portfolio_roi = total_contribution / (total_spend + 1e-9)

        # Reallocation potential
        reallocation_potential = sum(ch.current_spend for ch in reduce_channels)
        headroom_available = sum(ch.headroom_amount for ch in increase_channels)

        return InvestmentPriorityResult(
            channel_analysis=channel_results,
            increase_channels=increase_channels,
            hold_channels=hold_channels,
            reduce_channels=reduce_channels,
            channels_needing_test=channels_needing_test,
            total_spend=total_spend,
            total_contribution=total_contribution,
            portfolio_roi=portfolio_roi,
            reallocation_potential=reallocation_potential,
            headroom_available=headroom_available,
        )

    def get_priority_table(self) -> pd.DataFrame:
        """
        Get investment priority as a DataFrame.

        Returns
        -------
        pd.DataFrame
            Priority table with all metrics
        """
        result = self.run_full_analysis()

        data = []
        for ch in result.channel_analysis:
            # Determine action
            if ch in result.increase_channels:
                action = "INCREASE"
            elif ch in result.reduce_channels:
                action = "REDUCE"
            else:
                action = "HOLD"

            data.append({
                "channel": ch.channel_name,
                "current_spend": ch.current_spend,
                "current_roi": ch.current_roi,
                "marginal_roi": ch.marginal_roi,
                "priority_rank": ch.priority_rank,
                "breakeven_spend": ch.breakeven_spend,
                "headroom_available": ch.headroom,
                "headroom_amount": ch.headroom_amount,
                "action": action,
                "needs_test": ch.needs_test,
                "roi_5pct": ch.roi_5pct,
                "roi_95pct": ch.roi_95pct,
            })

        return pd.DataFrame(data)

    def print_priority_table(self) -> None:
        """Print formatted priority table."""
        result = self.run_full_analysis()

        print("=" * 100)
        print("MARGINAL ROI & INVESTMENT PRIORITY TABLE")
        print("=" * 100)

        print(f"\n{'Channel':<25} {'Current':>12} {'Current':>10} {'Marginal':>10} {'Priority':>10} {'Breakeven':>12} {'Headroom':>10}")
        print(f"{'':<25} {'Spend':>12} {'ROI':>10} {'ROI':>10} {'Rank':>10} {'Spend':>12} {'Available':>10}")
        print("-" * 105)

        for ch in result.channel_analysis:
            spend_str = f"${ch.current_spend:,.0f}"
            roi_str = f"${ch.current_roi:.2f}"
            marginal_str = f"${ch.marginal_roi:.2f}"
            rank_str = str(ch.priority_rank)

            if ch.breakeven_spend:
                breakeven_str = f"${ch.breakeven_spend:,.0f}"
            else:
                breakeven_str = "N/A"

            headroom_str = "Yes" if ch.headroom else "No"

            # Indicators
            if ch.marginal_roi >= 2:
                indicator = " *"
            elif ch.marginal_roi < 1:
                indicator = " !"
            else:
                indicator = ""

            print(f"{ch.channel_name:<25} {spend_str:>12} {roi_str:>10} {marginal_str:>10}{indicator} {rank_str:>10} {breakeven_str:>12} {headroom_str:>10}")

        print("-" * 105)
        print("\n* = High priority (Marginal ROI >= $2)")
        print("! = Below breakeven (Marginal ROI < $1)")

    @classmethod
    def from_mmm_wrapper(cls, mmm_wrapper: Any) -> "MarginalROIAnalyzer":
        """
        Create analyzer from an MMMWrapper instance.

        Parameters
        ----------
        mmm_wrapper : MMMWrapper
            Fitted model wrapper

        Returns
        -------
        MarginalROIAnalyzer
            Analyzer instance
        """
        # Build display names dict from config (paid media only)
        display_names = {}
        for ch_config in mmm_wrapper.config.channels:
            display_names[ch_config.name] = ch_config.get_display_name()

        return cls(
            idata=mmm_wrapper.idata,
            df_scaled=mmm_wrapper.df_scaled,
            contribs=mmm_wrapper.get_contributions(),
            channel_cols=mmm_wrapper.config.get_channel_columns(),  # Paid media only
            target_col=mmm_wrapper.config.data.target_column,
            spend_scale=mmm_wrapper.config.data.spend_scale,
            revenue_scale=mmm_wrapper.config.data.revenue_scale,
            display_names=display_names,
        )
