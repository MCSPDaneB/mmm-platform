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
        Compute ROI samples from posterior by properly applying transforms.

        ROI = Σ contribution / Σ spend
            = target_scale × β × Σ sat(adstock(x_norm)) / Σ spend

        We sample over (β, λ, α) jointly to capture correlations.
        This is more accurate than using mean λ/α as in bayesian_significance.py.

        Parameters
        ----------
        channel : str
            Channel name.

        Returns
        -------
        np.ndarray
            Array of ROI samples, one per posterior draw.
        """
        idata = self.wrapper.idata
        df = self.wrapper.df_scaled
        config = self.config

        # Get channel index in effective channels list
        effective_channels = self.wrapper.transform_engine.get_effective_channel_columns()
        ch_idx = effective_channels.index(channel)

        # DEBUG: Log channel ordering to diagnose mismatches (using print for visibility)
        pymc_channels = list(self.wrapper.mmm.channel_columns) if hasattr(self.wrapper.mmm, 'channel_columns') else []
        print(f"DEBUG ROI - Channel: {channel}")
        print(f"DEBUG ROI - Our effective_channels: {effective_channels}")
        print(f"DEBUG ROI - PyMC channel_columns: {pymc_channels}")
        print(f"DEBUG ROI - Our ch_idx for {channel}: {ch_idx}")
        if pymc_channels and channel in pymc_channels:
            pymc_idx = pymc_channels.index(channel)
            print(f"DEBUG ROI - PyMC idx for {channel}: {pymc_idx}")
            if ch_idx != pymc_idx:
                print(f"DEBUG ROI - INDEX MISMATCH! Our idx={ch_idx}, PyMC idx={pymc_idx}")

        # Get posterior samples for all three parameters
        beta_samples = idata.posterior["saturation_beta"].isel(channel=ch_idx).values
        lam_samples = idata.posterior["saturation_lam"].isel(channel=ch_idx).values
        alpha_samples = idata.posterior["adstock_alpha"].isel(channel=ch_idx).values

        # Flatten chains × draws to single dimension
        beta_flat = beta_samples.flatten()
        lam_flat = lam_samples.flatten()
        alpha_flat = alpha_samples.flatten()

        n_samples = len(beta_flat)

        # Get data for this channel
        x = df[channel].values.astype(float)
        x_max = x.max()
        x_norm = x / (x_max + 1e-9)
        total_spend = x.sum()
        target_scale = df[config.data.target_column].max()
        l_max = config.adstock.l_max

        # DEBUG: Log intermediate values for TikTok
        if "tiktok" in channel.lower():
            print(f"DEBUG ROI CALC - {channel}:")
            print(f"  x_max (spend max): {x_max}")
            print(f"  total_spend: {total_spend}")
            print(f"  target_scale (target max): {target_scale}")
            print(f"  l_max: {l_max}")
            print(f"  n_samples: {n_samples}")
            print(f"  beta mean: {beta_flat.mean():.6f}, std: {beta_flat.std():.6f}")
            print(f"  lam mean: {lam_flat.mean():.6f}")
            print(f"  alpha mean: {alpha_flat.mean():.6f}")

        # Compute ROI for each posterior sample
        roi_samples = np.zeros(n_samples)

        for i in range(n_samples):
            alpha = alpha_flat[i]
            lam = lam_flat[i]
            beta = beta_flat[i]

            # Apply adstock transformation
            x_ad = self._geometric_adstock(x_norm, alpha, l_max)

            # Apply saturation transformation
            x_sat = self._logistic_saturation(x_ad, lam)

            # Compute contribution and ROI
            contribution = target_scale * beta * x_sat.sum()
            roi_samples[i] = contribution / (total_spend + 1e-9)

            # DEBUG: Log first sample details for TikTok
            if i == 0 and "tiktok" in channel.lower():
                print(f"  Sample 0 details:")
                print(f"    x_ad.sum(): {x_ad.sum():.6f}")
                print(f"    x_sat.sum(): {x_sat.sum():.6f}")
                print(f"    beta: {beta:.6f}")
                print(f"    contribution: {contribution:.2f}")
                print(f"    ROI: {roi_samples[i]:.6f}")

        # DEBUG: Log final ROI stats for TikTok
        if "tiktok" in channel.lower():
            print(f"  Final ROI mean: {roi_samples.mean():.6f}")
            print(f"  Final ROI std: {roi_samples.std():.6f}")

        return roi_samples

    def _geometric_adstock(self, x: np.ndarray, alpha: float, l_max: int) -> np.ndarray:
        """
        Apply geometric adstock transformation.

        Matches the implementation in transforms.py with backward-looking weights.

        Parameters
        ----------
        x : np.ndarray
            Input time series (normalized).
        alpha : float
            Decay rate.
        l_max : int
            Maximum lag.

        Returns
        -------
        np.ndarray
            Adstocked time series.
        """
        weights = np.array([alpha ** i for i in range(l_max)])
        weights = weights / weights.sum()
        weights = weights[::-1]  # Reverse for backward-looking
        return np.convolve(x, weights, mode='full')[:len(x)]

    def _logistic_saturation(self, x: np.ndarray, lam: float) -> np.ndarray:
        """
        Apply logistic saturation transformation.

        Parameters
        ----------
        x : np.ndarray
            Input time series.
        lam : float
            Saturation parameter.

        Returns
        -------
        np.ndarray
            Saturated time series.
        """
        return (1 - np.exp(-lam * x)) / (1 + np.exp(-lam * x))

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
            effective_channels = self.wrapper.transform_engine.get_effective_channel_columns()
            ch_idx = effective_channels.index(channel)
            prior_lam = self.wrapper.lam_vec[ch_idx]
            posterior_lam = float(
                self.wrapper.idata.posterior["saturation_lam"]
                .isel(channel=ch_idx).mean()
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
