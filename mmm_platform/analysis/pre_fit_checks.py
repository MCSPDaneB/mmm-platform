"""
Pre-Fit Configuration Checks

Analyzes data and configuration before model fitting to warn about
potential issues:
- Low spend channels that may have unstable ROI
- Wide prior ranges that may cause drift
- Channels with high spend variance
"""

from dataclasses import dataclass
from typing import List
import numpy as np
import pandas as pd
import logging

from mmm_platform.config.schema import ModelConfig

logger = logging.getLogger(__name__)


@dataclass
class PreFitWarning:
    """A single pre-fit warning."""
    channel: str
    severity: str  # "info", "warning", "critical"
    issue: str
    recommendation: str


class PreFitChecker:
    """Check configuration and data before fitting to identify potential issues."""

    def __init__(self, config: ModelConfig, df: pd.DataFrame):
        """
        Initialize the pre-fit checker.

        Parameters
        ----------
        config : ModelConfig
            Model configuration.
        df : pd.DataFrame
            Data to be used for fitting.
        """
        self.config = config
        self.df = df

    def check_all(self) -> List[PreFitWarning]:
        """
        Run all pre-fit checks.

        Returns
        -------
        List[PreFitWarning]
            List of warnings found.
        """
        warnings = []
        warnings.extend(self._check_low_spend_channels())
        warnings.extend(self._check_wide_roi_priors())
        warnings.extend(self._check_spend_variance())
        warnings.extend(self._check_zero_spend_periods())
        return warnings

    def _check_low_spend_channels(self) -> List[PreFitWarning]:
        """Flag channels with <5% of total spend - ROI may be unstable."""
        warnings = []
        channel_cols = self.config.get_channel_columns()

        # Calculate total spend across all channels
        total_spend = 0
        for ch in channel_cols:
            if ch in self.df.columns:
                total_spend += self.df[ch].sum()

        if total_spend == 0:
            return warnings

        for ch in channel_cols:
            if ch in self.df.columns:
                ch_spend = self.df[ch].sum()
                pct = ch_spend / total_spend

                if pct < 0.03:  # Less than 3% of total spend
                    warnings.append(PreFitWarning(
                        channel=ch,
                        severity="warning",
                        issue=f"Very low spend channel ({pct:.1%} of total)",
                        recommendation="Consider tighter ROI priors (lower beta_sigma_multiplier) or narrower prior range"
                    ))
                elif pct < 0.05:  # Less than 5% of total spend
                    warnings.append(PreFitWarning(
                        channel=ch,
                        severity="info",
                        issue=f"Low spend channel ({pct:.1%} of total)",
                        recommendation="ROI estimates may have higher uncertainty"
                    ))

        return warnings

    def _check_wide_roi_priors(self) -> List[PreFitWarning]:
        """Flag channels with very wide ROI prior ranges (high/low ratio > 20x)."""
        warnings = []

        for ch in self.config.channels:
            if ch.roi_prior_low > 0:
                ratio = ch.roi_prior_high / ch.roi_prior_low

                if ratio > 50:  # Extremely wide
                    warnings.append(PreFitWarning(
                        channel=ch.name,
                        severity="warning",
                        issue=f"Very wide ROI prior range ({ch.roi_prior_low:.1f} - {ch.roi_prior_high:.1f}, {ratio:.0f}x ratio)",
                        recommendation="Consider narrowing range or using beta_sigma_multiplier < 1.0 for tighter priors"
                    ))
                elif ratio > 20:  # Wide
                    warnings.append(PreFitWarning(
                        channel=ch.name,
                        severity="info",
                        issue=f"Wide ROI prior range ({ch.roi_prior_low:.1f} - {ch.roi_prior_high:.1f}, {ratio:.0f}x ratio)",
                        recommendation="Posterior may drift significantly from prior mid-point"
                    ))

        return warnings

    def _check_spend_variance(self) -> List[PreFitWarning]:
        """Flag channels with very high spend variance (coefficient of variation > 1.5)."""
        warnings = []
        channel_cols = self.config.get_channel_columns()

        for ch in channel_cols:
            if ch in self.df.columns:
                spend = self.df[ch].values
                mean_spend = np.mean(spend)
                std_spend = np.std(spend)

                if mean_spend > 0:
                    cv = std_spend / mean_spend  # Coefficient of variation

                    if cv > 2.0:  # Very high variance
                        warnings.append(PreFitWarning(
                            channel=ch,
                            severity="info",
                            issue=f"High spend variance (CV={cv:.2f})",
                            recommendation="Sporadic spending may lead to less stable ROI estimates"
                        ))

        return warnings

    def _check_zero_spend_periods(self) -> List[PreFitWarning]:
        """Flag channels with many zero-spend periods."""
        warnings = []
        channel_cols = self.config.get_channel_columns()

        for ch in channel_cols:
            if ch in self.df.columns:
                spend = self.df[ch].values
                zero_pct = np.mean(spend == 0)

                if zero_pct > 0.5:  # More than 50% zeros
                    warnings.append(PreFitWarning(
                        channel=ch,
                        severity="warning",
                        issue=f"High zero-spend rate ({zero_pct:.0%} of periods)",
                        recommendation="Channel effect may be harder to estimate reliably"
                    ))
                elif zero_pct > 0.3:  # More than 30% zeros
                    warnings.append(PreFitWarning(
                        channel=ch,
                        severity="info",
                        issue=f"Many zero-spend periods ({zero_pct:.0%})",
                        recommendation="Consider if this channel has sufficient data for reliable estimation"
                    ))

        return warnings


def run_pre_fit_checks(config: ModelConfig, df: pd.DataFrame) -> List[PreFitWarning]:
    """
    Run all pre-fit checks and return warnings.

    Parameters
    ----------
    config : ModelConfig
        Model configuration.
    df : pd.DataFrame
        Data to be used for fitting.

    Returns
    -------
    List[PreFitWarning]
        List of warnings found.

    Example
    -------
    >>> from mmm_platform.analysis.pre_fit_checks import run_pre_fit_checks
    >>> warnings = run_pre_fit_checks(config, df)
    >>> for w in warnings:
    ...     print(f"[{w.severity}] {w.channel}: {w.issue}")
    """
    checker = PreFitChecker(config, df)
    return checker.check_all()
