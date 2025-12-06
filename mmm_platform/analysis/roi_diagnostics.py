"""
ROI Prior Diagnostics Module

Validates consistency between ROI beliefs and model behavior:
- Post-fit: Did posterior ROI match prior beliefs?
- Tracks λ (saturation) shifts that explain ROI changes

Addresses the limitation that β priors assume fixed λ, but λ is learned.

Usage:
    from mmm_platform.analysis.roi_diagnostics import quick_roi_diagnostics

    # After fitting
    report = quick_roi_diagnostics(wrapper)
    print(report.overall_health)
    print(report.to_dataframe())
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, TYPE_CHECKING
import numpy as np
import pandas as pd
import arviz as az
import logging

if TYPE_CHECKING:
    from mmm_platform.model.mmm import MMMWrapper

logger = logging.getLogger(__name__)


@dataclass
class ChannelROIResult:
    """ROI validation results for a single channel."""
    channel_name: str

    # Prior beliefs (from config)
    prior_roi_low: float
    prior_roi_mid: float
    prior_roi_high: float

    # Posterior results (after fitting)
    posterior_roi_mean: Optional[float] = None
    posterior_roi_std: Optional[float] = None
    posterior_roi_hdi_low: Optional[float] = None
    posterior_roi_hdi_high: Optional[float] = None

    # Diagnostics
    prior_belief_in_posterior_hdi: Optional[bool] = None
    posterior_vs_prior_shift: Optional[float] = None  # (posterior - prior) / prior
    lambda_shift: Optional[float] = None  # (posterior_λ - prior_λ) / prior_λ

    # Warnings and recommendations
    warnings: List[str] = field(default_factory=list)

    def summary_dict(self) -> dict:
        """Return summary as dictionary."""
        return {
            "channel": self.channel_name,
            "prior_roi_low": self.prior_roi_low,
            "prior_roi_mid": self.prior_roi_mid,
            "prior_roi_high": self.prior_roi_high,
            "posterior_roi_mean": self.posterior_roi_mean,
            "posterior_roi_hdi_low": self.posterior_roi_hdi_low,
            "posterior_roi_hdi_high": self.posterior_roi_hdi_high,
            "prior_in_hdi": self.prior_belief_in_posterior_hdi,
            "roi_shift_pct": self.posterior_vs_prior_shift,
            "lambda_shift_pct": self.lambda_shift,
            "warnings": self.warnings,
        }


@dataclass
class ROIDiagnosticReport:
    """Full diagnostic report across all channels."""
    channel_results: Dict[str, ChannelROIResult]
    channels_with_prior_tension: List[str] = field(default_factory=list)
    channels_with_large_shift: List[str] = field(default_factory=list)
    overall_health: str = "Not assessed"
    recommendations: List[str] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to a summary DataFrame."""
        rows = []
        for name, result in self.channel_results.items():
            rows.append({
                "channel": name,
                "prior_roi_low": result.prior_roi_low,
                "prior_roi_mid": result.prior_roi_mid,
                "prior_roi_high": result.prior_roi_high,
                "posterior_roi_mean": result.posterior_roi_mean,
                "posterior_roi_std": result.posterior_roi_std,
                "posterior_roi_hdi_low": result.posterior_roi_hdi_low,
                "posterior_roi_hdi_high": result.posterior_roi_hdi_high,
                "prior_in_hdi": result.prior_belief_in_posterior_hdi,
                "roi_shift_pct": result.posterior_vs_prior_shift,
                "lambda_shift_pct": result.lambda_shift,
            })
        return pd.DataFrame(rows)

    def print_summary(self):
        """Print a human-readable summary."""
        print("\n" + "=" * 70)
        print("ROI PRIOR DIAGNOSTICS REPORT")
        print("=" * 70)

        print(f"\nOverall Health: {self.overall_health}")

        if self.channels_with_prior_tension:
            print(f"\nChannels with prior tension (belief outside posterior HDI):")
            for ch in self.channels_with_prior_tension:
                print(f"   - {ch}")

        if self.channels_with_large_shift:
            print(f"\nChannels with large ROI shift (>50% from prior):")
            for ch in self.channels_with_large_shift:
                print(f"   - {ch}")

        print("\n" + "-" * 70)
        print("Channel Details:")
        print("-" * 70)

        for name, result in self.channel_results.items():
            print(f"\n{name}:")
            print(f"  Prior belief:     {result.prior_roi_low:.2f} - "
                  f"{result.prior_roi_mid:.2f} - {result.prior_roi_high:.2f}")

            if result.posterior_roi_mean is not None:
                print(f"  Posterior:        {result.posterior_roi_mean:.2f} "
                      f"[{result.posterior_roi_hdi_low:.2f}, {result.posterior_roi_hdi_high:.2f}]")
                print(f"  Shift from prior: {result.posterior_vs_prior_shift:+.1%}")
                print(f"  Lambda shift:     {result.lambda_shift:+.1%}")
                in_hdi = "Yes" if result.prior_belief_in_posterior_hdi else "No"
                print(f"  Belief in HDI:    {in_hdi}")

            if result.warnings:
                for w in result.warnings:
                    print(f"  WARNING: {w}")

        if self.recommendations:
            print("\n" + "-" * 70)
            print("Recommendations:")
            print("-" * 70)
            for rec in self.recommendations:
                print(f"  - {rec}")

        print("\n" + "=" * 70)


class ROIDiagnostics:
    """
    Validates ROI priors against posterior model behavior.

    This class addresses the fundamental challenge that:
    1. We set priors based on ROI beliefs
    2. Those get translated to beta priors (assuming fixed λ)
    3. But λ is also learned, so the implied ROI can drift

    It provides:
    - Posterior validation: Did the learned ROI match our prior beliefs?
    - Lambda tracking: Did the saturation curve change significantly?
    - Recommendations: Should we adjust priors for future models?

    Usage:
        diagnostics = ROIDiagnostics(wrapper)
        report = diagnostics.validate_posterior()
        report.to_dataframe()  # Summary table
        report.print_summary()  # Human-readable output
    """

    def __init__(self, wrapper: "MMMWrapper", hdi_prob: float = 0.9):
        """
        Initialize diagnostics.

        Parameters
        ----------
        wrapper : MMMWrapper
            Fitted model wrapper with idata containing posterior samples.
        hdi_prob : float
            Probability mass for HDI intervals (default 0.9 = 90% HDI).
        """
        self.wrapper = wrapper
        self.config = wrapper.config
        self.hdi_prob = hdi_prob

        # Extract ROI beliefs from config
        self.roi_beliefs = self._extract_roi_beliefs()

    def _extract_roi_beliefs(self) -> Dict[str, Dict[str, float]]:
        """Extract ROI beliefs from config for all channels with ROI priors."""
        beliefs = {}

        # Paid media channels (always have ROI)
        for ch in self.config.channels:
            beliefs[ch.name] = {
                "low": ch.roi_prior_low,
                "mid": ch.roi_prior_mid,
                "high": ch.roi_prior_high,
            }

        # Owned media (only if include_roi=True)
        for om in self.config.owned_media:
            if om.include_roi and om.roi_prior_mid is not None:
                beliefs[om.name] = {
                    "low": om.roi_prior_low,
                    "mid": om.roi_prior_mid,
                    "high": om.roi_prior_high,
                }

        return beliefs

    def _compute_posterior_roi_samples(self, channel: str) -> np.ndarray:
        """
        Compute ROI samples from posterior using PyMC-Marketing's contribution calculation.

        Uses compute_channel_contribution_original_scale() to get contribution samples
        that properly account for PyMC-Marketing's internal scaling.

        ROI = Σ contribution / Σ spend (per posterior sample)

        Parameters
        ----------
        channel : str
            Channel name.

        Returns
        -------
        np.ndarray
            Array of ROI samples, one per posterior draw.
        """
        df = self.wrapper.df_scaled

        # Get total spend for this channel
        total_spend = df[channel].sum()

        # Get contribution samples from PyMC-Marketing (posterior=True by default)
        # This returns a DataArray with dims (chain, draw, date, channel)
        contrib_samples = self.wrapper.mmm.compute_channel_contribution_original_scale(prior=False)

        # Get channel index in PyMC's channel ordering
        pymc_channels = list(self.wrapper.mmm.channel_columns)
        if channel not in pymc_channels:
            logger.warning(f"Channel {channel} not found in PyMC channel_columns")
            return np.array([])

        ch_idx = pymc_channels.index(channel)

        # Select this channel and sum over time dimension
        # Result has dims (chain, draw) after summing over date
        channel_contribs = contrib_samples.isel(channel=ch_idx).sum(dim="date")

        # Flatten to 1D array of samples
        contrib_flat = channel_contribs.values.flatten()

        # Compute ROI for each sample
        roi_samples = contrib_flat / (total_spend + 1e-9)

        return roi_samples

    def validate_posterior(self) -> ROIDiagnosticReport:
        """
        Validate posterior ROI against prior beliefs.

        This compares the posterior distribution of ROI (computed by sampling
        over β, λ, α jointly) against the prior ROI beliefs from the config.

        Returns
        -------
        ROIDiagnosticReport
            Diagnostic report with per-channel results and recommendations.
        """
        if self.wrapper.idata is None:
            raise ValueError("Model must be fitted before validating posterior ROI")

        results = {}

        for channel, beliefs in self.roi_beliefs.items():
            # Compute ROI samples from posterior
            try:
                roi_samples = self._compute_posterior_roi_samples(channel)
            except Exception as e:
                logger.warning(f"Could not compute ROI for {channel}: {e}")
                continue

            # Filter extreme values (numerical issues)
            roi_samples = roi_samples[np.isfinite(roi_samples)]
            roi_samples = roi_samples[(roi_samples > 0) & (roi_samples < 100)]

            if len(roi_samples) < 50:
                logger.warning(f"Too few valid ROI samples for {channel} "
                              f"({len(roi_samples)} samples)")
                continue

            # Compute HDI
            hdi = az.hdi(roi_samples, hdi_prob=self.hdi_prob)

            # Get λ shift (prior vs posterior)
            # Use PyMC's channel ordering for posterior access
            pymc_channels = list(self.wrapper.mmm.channel_columns)
            pymc_ch_idx = pymc_channels.index(channel)

            # Use our effective_channels ordering for prior (lam_vec)
            effective_channels = self.wrapper.transform_engine.get_effective_channel_columns()
            our_ch_idx = effective_channels.index(channel)
            prior_lam = self.wrapper.lam_vec[our_ch_idx]

            posterior_lam = float(
                self.wrapper.idata.posterior["saturation_lam"]
                .isel(channel=pymc_ch_idx).mean()
            )
            lam_shift = (posterior_lam - prior_lam) / (prior_lam + 1e-9)

            # Build result
            result = ChannelROIResult(
                channel_name=channel,
                prior_roi_low=beliefs["low"],
                prior_roi_mid=beliefs["mid"],
                prior_roi_high=beliefs["high"],
                posterior_roi_mean=float(np.mean(roi_samples)),
                posterior_roi_std=float(np.std(roi_samples)),
                posterior_roi_hdi_low=float(hdi[0]),
                posterior_roi_hdi_high=float(hdi[1]),
                prior_belief_in_posterior_hdi=(
                    beliefs["mid"] >= hdi[0] and beliefs["mid"] <= hdi[1]
                ),
                posterior_vs_prior_shift=(
                    (np.mean(roi_samples) - beliefs["mid"]) / (beliefs["mid"] + 1e-9)
                ),
                lambda_shift=lam_shift,
            )

            # Add warnings based on diagnostics
            if not result.prior_belief_in_posterior_hdi:
                result.warnings.append(
                    f"Prior ROI belief ({beliefs['mid']:.2f}) outside posterior HDI "
                    f"[{hdi[0]:.2f}, {hdi[1]:.2f}]"
                )

            if abs(result.posterior_vs_prior_shift) > 0.5:
                result.warnings.append(
                    f"Large ROI shift ({result.posterior_vs_prior_shift:+.1%}) from prior"
                )

            if abs(lam_shift) > 0.3:
                result.warnings.append(
                    f"Saturation curve changed significantly "
                    f"(λ shifted {lam_shift:+.1%} from prior)"
                )

            results[channel] = result

        # Build report
        report = ROIDiagnosticReport(channel_results=results)

        # Identify problem channels
        for name, result in results.items():
            if not result.prior_belief_in_posterior_hdi:
                report.channels_with_prior_tension.append(name)
            if (result.posterior_vs_prior_shift is not None and
                    abs(result.posterior_vs_prior_shift) > 0.5):
                report.channels_with_large_shift.append(name)

        # Assess overall health
        n_channels = len(results)
        n_tension = len(report.channels_with_prior_tension)

        if n_channels == 0:
            report.overall_health = "No channels to analyze"
        elif n_tension == 0:
            report.overall_health = "Good - priors consistent with posterior"
        elif n_tension <= n_channels * 0.3:
            report.overall_health = "Moderate - some prior/data tension"
        else:
            report.overall_health = "Review needed - significant disagreement"

        # Generate recommendations
        if report.channels_with_prior_tension:
            report.recommendations.append(
                f"Review ROI beliefs for: {', '.join(report.channels_with_prior_tension)}. "
                f"Either update priors based on posterior learnings, or investigate data quality."
            )

        # Check for λ shifts
        lam_shifted = [
            name for name, r in results.items()
            if r.lambda_shift and abs(r.lambda_shift) > 0.3
        ]
        if lam_shifted:
            report.recommendations.append(
                f"Saturation curves shifted significantly for: {', '.join(lam_shifted)}. "
                f"Consider updating curve_sharpness settings for future models."
            )

        if report.channels_with_large_shift:
            report.recommendations.append(
                f"Consider using posterior ROI means as starting point for "
                f"future model priors: {', '.join(report.channels_with_large_shift)}"
            )

        return report


def quick_roi_diagnostics(wrapper: "MMMWrapper", hdi_prob: float = 0.9) -> ROIDiagnosticReport:
    """
    One-liner for ROI diagnostics after fitting.

    Parameters
    ----------
    wrapper : MMMWrapper
        Fitted model wrapper.
    hdi_prob : float
        Probability mass for HDI intervals (default 0.9).

    Returns
    -------
    ROIDiagnosticReport
        Diagnostic report with per-channel results.

    Example
    -------
    >>> from mmm_platform.analysis.roi_diagnostics import quick_roi_diagnostics
    >>> report = quick_roi_diagnostics(wrapper)
    >>> print(report.overall_health)
    >>> report.print_summary()
    >>> df = report.to_dataframe()
    """
    diagnostics = ROIDiagnostics(wrapper, hdi_prob=hdi_prob)
    return diagnostics.validate_posterior()
