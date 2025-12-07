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
        warnings.extend(self._check_channel_data_quality())
        warnings.extend(self._check_wide_roi_priors())
        return warnings

    def _check_channel_data_quality(self) -> List[PreFitWarning]:
        """
        Single consolidated check for channel data quality.

        Combines spend %, zero periods, and variance into one warning per channel.
        """
        warnings = []
        channel_cols = self.config.get_channel_columns()

        # Calculate total spend across all channels
        total_spend = sum(
            self.df[ch].sum() for ch in channel_cols if ch in self.df.columns
        )

        if total_spend == 0:
            return warnings

        for ch in channel_cols:
            if ch not in self.df.columns:
                continue

            spend = self.df[ch].values
            ch_spend = spend.sum()
            spend_pct = ch_spend / total_spend
            zero_pct = np.mean(spend == 0)
            mean_spend = np.mean(spend)
            cv = np.std(spend) / mean_spend if mean_spend > 0 else 0

            # Build issue description with all relevant metrics
            issues = []
            if spend_pct < 0.03:
                issues.append(f"{spend_pct:.1%} of spend")
            if zero_pct > 0.5:
                issues.append(f"{zero_pct:.0%} zeros")
            if cv > 2.0:
                issues.append(f"CV={cv:.1f}")

            if issues:
                severity = "warning" if spend_pct < 0.03 or zero_pct > 0.5 else "info"
                warnings.append(PreFitWarning(
                    channel=ch,
                    severity=severity,
                    issue=", ".join(issues),
                    recommendation="Prior auto-tightened; ROI estimate may have higher uncertainty"
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
                        issue=f"Wide ROI prior ({ch.roi_prior_low:.1f}-{ch.roi_prior_high:.1f}, {ratio:.0f}x)",
                        recommendation="Consider narrowing the ROI range"
                    ))
                elif ratio > 20:  # Wide
                    warnings.append(PreFitWarning(
                        channel=ch.name,
                        severity="info",
                        issue=f"ROI range {ratio:.0f}x ({ch.roi_prior_low:.1f}-{ch.roi_prior_high:.1f})",
                        recommendation="Posterior may drift from prior"
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
