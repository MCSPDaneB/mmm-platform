"""Bayesian Significance Analysis for MMM Platform.

Provides comprehensive Bayesian analysis of model results including:
- Credible intervals (HDI)
- Probability of Direction (pd)
- ROPE Analysis (Region of Practical Equivalence)
- ROI posterior distributions with uncertainty quantification
- Prior vs Posterior sensitivity analysis
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import arviz as az
from pymc_marketing.mmm.transformers import geometric_adstock, logistic_saturation


@dataclass
class CredibleIntervalResult:
    """Results for a single channel's credible interval analysis."""
    channel: str
    mean: float
    hdi_low: float
    hdi_high: float
    excludes_zero: bool

    @property
    def interval_width(self) -> float:
        """Width of the credible interval."""
        return self.hdi_high - self.hdi_low


@dataclass
class ProbabilityOfDirectionResult:
    """Results for probability of direction analysis."""
    channel: str
    pd: float  # Probability of direction (positive)
    interpretation: str

    @classmethod
    def interpret_pd(cls, pd_value: float) -> str:
        """Interpret the probability of direction value."""
        if pd_value >= 0.99:
            return "Very strong"
        elif pd_value >= 0.95:
            return "Strong"
        elif pd_value >= 0.90:
            return "Moderate"
        elif pd_value >= 0.75:
            return "Weak"
        else:
            return "Inconclusive"


@dataclass
class ROPEResult:
    """Results for ROPE (Region of Practical Equivalence) analysis."""
    channel: str
    pct_in_rope: float
    pct_below_rope: float
    pct_above_rope: float
    conclusion: str
    rope_low: float
    rope_high: float


@dataclass
class ROICredibleIntervalResult:
    """Results for ROI posterior analysis with credible intervals."""
    channel: str
    roi_mean: float
    roi_median: float
    roi_5pct: float
    roi_95pct: float
    roi_hdi_low: float
    roi_hdi_high: float
    significant: bool
    posterior_samples: np.ndarray = field(repr=False)


@dataclass
class PriorSensitivityResult:
    """Results for prior vs posterior sensitivity analysis."""
    channel: str
    prior_roi: float
    posterior_roi: float
    shift: float
    relative_shift: float
    data_influence: str


@dataclass
class BayesianSignificanceReport:
    """Complete Bayesian significance analysis report."""
    credible_intervals: List[CredibleIntervalResult]
    probability_of_direction: List[ProbabilityOfDirectionResult]
    rope_analysis: List[ROPEResult]
    roi_posteriors: List[ROICredibleIntervalResult]
    prior_sensitivity: List[PriorSensitivityResult]

    def to_dataframe(self) -> pd.DataFrame:
        """Convert report to a summary DataFrame."""
        data = []
        for ci, pd_result, rope, roi, sens in zip(
            self.credible_intervals,
            self.probability_of_direction,
            self.rope_analysis,
            self.roi_posteriors,
            self.prior_sensitivity
        ):
            data.append({
                'channel': ci.channel,
                'beta_mean': ci.mean,
                'beta_hdi_low': ci.hdi_low,
                'beta_hdi_high': ci.hdi_high,
                'excludes_zero': ci.excludes_zero,
                'prob_direction': pd_result.pd,
                'pd_interpretation': pd_result.interpretation,
                'pct_in_rope': rope.pct_in_rope,
                'pct_above_rope': rope.pct_above_rope,
                'rope_conclusion': rope.conclusion,
                'roi_mean': roi.roi_mean,
                'roi_5pct': roi.roi_5pct,
                'roi_95pct': roi.roi_95pct,
                'roi_significant': roi.significant,
                'prior_roi': sens.prior_roi,
                'posterior_roi': sens.posterior_roi,
                'prior_shift': sens.shift,
                'data_influence': sens.data_influence
            })
        return pd.DataFrame(data)


class BayesianSignificanceAnalyzer:
    """Analyzer for Bayesian significance of MMM results."""

    def __init__(
        self,
        idata: Any,  # arviz.InferenceData
        df_scaled: pd.DataFrame,
        channel_cols: List[str],
        target_col: str,
        prior_rois: Optional[Dict[str, float]] = None,
        rope_low: float = -0.05,
        rope_high: float = 0.05,
        hdi_prob: float = 0.95,
        l_max: int = 8
    ):
        """Initialize the Bayesian significance analyzer.

        Args:
            idata: ArviZ InferenceData object with posterior samples
            df_scaled: Scaled dataframe used for modeling
            channel_cols: List of channel column names
            target_col: Name of the target column
            prior_rois: Dictionary mapping channel names to prior ROI expectations
            rope_low: Lower bound for ROPE (default -0.05)
            rope_high: Upper bound for ROPE (default 0.05)
            hdi_prob: Probability for HDI calculation (default 0.95)
            l_max: Maximum lag for adstock (default 8)
        """
        self.idata = idata
        self.df_scaled = df_scaled
        self.channel_cols = channel_cols
        self.target_col = target_col
        self.prior_rois = prior_rois or {ch: 1.0 for ch in channel_cols}
        self.rope_low = rope_low
        self.rope_high = rope_high
        self.hdi_prob = hdi_prob
        self.l_max = l_max

        # Cache posterior samples
        self._beta_posterior = None
        self._alpha_posterior = None
        self._lam_posterior = None
        self._roi_posteriors: Dict[str, np.ndarray] = {}

    @property
    def beta_posterior(self):
        """Get beta (saturation_beta) posterior samples."""
        if self._beta_posterior is None:
            self._beta_posterior = self.idata.posterior["saturation_beta"]
        return self._beta_posterior

    @property
    def alpha_posterior(self):
        """Get alpha (adstock_alpha) posterior samples."""
        if self._alpha_posterior is None:
            self._alpha_posterior = self.idata.posterior["adstock_alpha"]
        return self._alpha_posterior

    @property
    def lam_posterior(self):
        """Get lambda (saturation_lam) posterior samples."""
        if self._lam_posterior is None:
            self._lam_posterior = self.idata.posterior["saturation_lam"]
        return self._lam_posterior

    def _probability_of_direction(self, samples: np.ndarray) -> float:
        """Calculate probability that effect is positive.

        Args:
            samples: Array of posterior samples

        Returns:
            Probability that the effect is positive (0 to 1)
        """
        samples_flat = np.asarray(samples).flatten()
        return float((samples_flat > 0).mean())

    def _rope_analysis(
        self,
        samples: np.ndarray,
        rope_low: Optional[float] = None,
        rope_high: Optional[float] = None
    ) -> Tuple[float, float, float]:
        """Calculate percentage of posterior in, below, and above ROPE.

        Args:
            samples: Array of posterior samples
            rope_low: Lower bound for ROPE (uses instance default if None)
            rope_high: Upper bound for ROPE (uses instance default if None)

        Returns:
            Tuple of (pct_in_rope, pct_below_rope, pct_above_rope)
        """
        rope_low = rope_low if rope_low is not None else self.rope_low
        rope_high = rope_high if rope_high is not None else self.rope_high

        samples_flat = np.asarray(samples).flatten()
        in_rope = ((samples_flat >= rope_low) & (samples_flat <= rope_high)).mean()
        above_rope = (samples_flat > rope_high).mean()
        below_rope = (samples_flat < rope_low).mean()

        return float(in_rope), float(below_rope), float(above_rope)

    def _interpret_rope(self, in_rope: float, above_rope: float) -> str:
        """Interpret ROPE analysis results.

        Args:
            in_rope: Percentage of posterior in ROPE
            above_rope: Percentage of posterior above ROPE

        Returns:
            Interpretation string
        """
        if above_rope >= 0.95:
            return "Practically significant"
        elif in_rope >= 0.95:
            return "Practically zero"
        else:
            return "Uncertain"

    def compute_credible_intervals(self) -> List[CredibleIntervalResult]:
        """Compute credible intervals for all channels.

        Returns:
            List of CredibleIntervalResult for each channel
        """
        # Get HDI summary from ArviZ
        beta_summary = az.summary(
            self.idata,
            var_names=["saturation_beta"],
            hdi_prob=self.hdi_prob
        )

        results = []
        hdi_low_col = f"hdi_{(1-self.hdi_prob)/2*100:.1f}%"
        hdi_high_col = f"hdi_{(1-(1-self.hdi_prob)/2)*100:.1f}%"

        # Handle different ArviZ summary column naming conventions
        if hdi_low_col not in beta_summary.columns:
            hdi_low_col = "hdi_2.5%"
            hdi_high_col = "hdi_97.5%"

        for ch in self.channel_cols:
            try:
                row = beta_summary.loc[f"saturation_beta[{ch}]"]
            except KeyError:
                # Try alternative indexing
                idx = self.channel_cols.index(ch)
                row = beta_summary.iloc[idx]

            mean = row["mean"]
            hdi_low = row[hdi_low_col]
            hdi_high = row[hdi_high_col]
            excludes_zero = hdi_low > 0 or hdi_high < 0

            results.append(CredibleIntervalResult(
                channel=ch,
                mean=mean,
                hdi_low=hdi_low,
                hdi_high=hdi_high,
                excludes_zero=excludes_zero
            ))

        return results

    def compute_probability_of_direction(self) -> List[ProbabilityOfDirectionResult]:
        """Compute probability of direction for all channels.

        Returns:
            List of ProbabilityOfDirectionResult for each channel
        """
        results = []

        for ch in self.channel_cols:
            try:
                samples = self.beta_posterior.sel(channel=ch)
            except (KeyError, ValueError):
                # Fallback to index-based selection
                idx = self.channel_cols.index(ch)
                samples = self.beta_posterior.isel(channel=idx)

            pd_value = self._probability_of_direction(samples.values)
            interpretation = ProbabilityOfDirectionResult.interpret_pd(pd_value)

            results.append(ProbabilityOfDirectionResult(
                channel=ch,
                pd=pd_value,
                interpretation=interpretation
            ))

        return results

    def compute_rope_analysis(
        self,
        rope_low: Optional[float] = None,
        rope_high: Optional[float] = None
    ) -> List[ROPEResult]:
        """Compute ROPE analysis for all channels.

        Args:
            rope_low: Lower bound for ROPE (uses instance default if None)
            rope_high: Upper bound for ROPE (uses instance default if None)

        Returns:
            List of ROPEResult for each channel
        """
        rope_low = rope_low if rope_low is not None else self.rope_low
        rope_high = rope_high if rope_high is not None else self.rope_high

        results = []

        for ch in self.channel_cols:
            try:
                samples = self.beta_posterior.sel(channel=ch)
            except (KeyError, ValueError):
                idx = self.channel_cols.index(ch)
                samples = self.beta_posterior.isel(channel=idx)

            in_rope, below_rope, above_rope = self._rope_analysis(
                samples.values, rope_low, rope_high
            )
            conclusion = self._interpret_rope(in_rope, above_rope)

            results.append(ROPEResult(
                channel=ch,
                pct_in_rope=in_rope,
                pct_below_rope=below_rope,
                pct_above_rope=above_rope,
                conclusion=conclusion,
                rope_low=rope_low,
                rope_high=rope_high
            ))

        return results

    def compute_roi_posteriors(self) -> List[ROICredibleIntervalResult]:
        """Compute ROI posterior distributions for all channels.

        Returns:
            List of ROICredibleIntervalResult for each channel
        """
        target_scale = self.df_scaled[self.target_col].max()
        results = []

        for ch in self.channel_cols:
            # Get posterior samples
            try:
                beta_samples = self.beta_posterior.sel(channel=ch).values.flatten()
                alpha_samples = self.alpha_posterior.sel(channel=ch).values.flatten()
                lam_samples = self.lam_posterior.sel(channel=ch).values.flatten()
            except (KeyError, ValueError):
                idx = self.channel_cols.index(ch)
                beta_samples = self.beta_posterior.isel(channel=idx).values.flatten()
                alpha_samples = self.alpha_posterior.isel(channel=idx).values.flatten()
                lam_samples = self.lam_posterior.isel(channel=idx).values.flatten()

            # Get spend data
            x = self.df_scaled[ch].values.astype(float)
            x_max = x.max()
            x_normalized = x / (x_max + 1e-9)
            total_spend = x.sum()

            # Calculate ROI using mean transforms (simplified approach)
            alpha_mean = alpha_samples.mean()
            lam_mean = lam_samples.mean()

            # Apply transforms
            x_ad = geometric_adstock(x_normalized, alpha=alpha_mean, l_max=self.l_max).eval()
            x_sat = logistic_saturation(x_ad, lam=lam_mean).eval()
            sat_sum = x_sat.sum()

            # ROI posterior = target_scale * beta * sat_sum / spend
            roi_samples = target_scale * beta_samples * sat_sum / (total_spend + 1e-9)

            roi_mean = float(roi_samples.mean())
            roi_median = float(np.median(roi_samples))
            roi_5pct = float(np.percentile(roi_samples, 5))
            roi_95pct = float(np.percentile(roi_samples, 95))

            # Calculate HDI for ROI
            roi_hdi = az.hdi(roi_samples, hdi_prob=self.hdi_prob)
            roi_hdi_low = float(roi_hdi[0])
            roi_hdi_high = float(roi_hdi[1])

            # Significant if 90% interval excludes zero (or some practical threshold)
            significant = roi_5pct > 0.1

            # Cache for later use
            self._roi_posteriors[ch] = roi_samples

            results.append(ROICredibleIntervalResult(
                channel=ch,
                roi_mean=roi_mean,
                roi_median=roi_median,
                roi_5pct=roi_5pct,
                roi_95pct=roi_95pct,
                roi_hdi_low=roi_hdi_low,
                roi_hdi_high=roi_hdi_high,
                significant=significant,
                posterior_samples=roi_samples
            ))

        return results

    def compute_prior_sensitivity(
        self,
        roi_results: Optional[List[ROICredibleIntervalResult]] = None
    ) -> List[PriorSensitivityResult]:
        """Compute prior vs posterior sensitivity analysis.

        Args:
            roi_results: Pre-computed ROI results (will compute if None)

        Returns:
            List of PriorSensitivityResult for each channel
        """
        if roi_results is None:
            roi_results = self.compute_roi_posteriors()

        roi_dict = {r.channel: r.roi_mean for r in roi_results}
        results = []

        for ch in self.channel_cols:
            prior_roi = self.prior_rois.get(ch, 1.0)
            posterior_roi = roi_dict.get(ch, 1.0)
            shift = posterior_roi - prior_roi

            # Calculate relative shift
            rel_shift = abs(shift) / (prior_roi + 0.1)

            if rel_shift > 0.5:
                influence = "Strong"
            elif rel_shift > 0.2:
                influence = "Moderate"
            else:
                influence = "Weak (prior-driven)"

            results.append(PriorSensitivityResult(
                channel=ch,
                prior_roi=prior_roi,
                posterior_roi=posterior_roi,
                shift=shift,
                relative_shift=rel_shift,
                data_influence=influence
            ))

        return results

    def run_full_analysis(self) -> BayesianSignificanceReport:
        """Run complete Bayesian significance analysis.

        Returns:
            BayesianSignificanceReport with all analysis results
        """
        credible_intervals = self.compute_credible_intervals()
        probability_of_direction = self.compute_probability_of_direction()
        rope_analysis = self.compute_rope_analysis()
        roi_posteriors = self.compute_roi_posteriors()
        prior_sensitivity = self.compute_prior_sensitivity(roi_posteriors)

        return BayesianSignificanceReport(
            credible_intervals=credible_intervals,
            probability_of_direction=probability_of_direction,
            rope_analysis=rope_analysis,
            roi_posteriors=roi_posteriors,
            prior_sensitivity=prior_sensitivity
        )

    def print_report(self, report: Optional[BayesianSignificanceReport] = None) -> None:
        """Print formatted Bayesian significance report.

        Args:
            report: Pre-computed report (will compute if None)
        """
        if report is None:
            report = self.run_full_analysis()

        print("=" * 80)
        print("BAYESIAN SIGNIFICANCE ANALYSIS")
        print("=" * 80)

        # 1. Credible Intervals
        print("\n1. CREDIBLE INTERVALS (95% HDI)")
        print("-" * 80)
        print(f"{'Channel':<40} {'Mean':>8} {'HDI Low':>10} {'HDI High':>10} {'Excludes 0':>12}")
        print("-" * 85)

        for ci in report.credible_intervals:
            excludes_str = "Yes" if ci.excludes_zero else "No"
            print(f"{ci.channel:<40} {ci.mean:>8.3f} {ci.hdi_low:>10.3f} {ci.hdi_high:>10.3f} {excludes_str:>12}")

        # 2. Probability of Direction
        print("\n\n2. PROBABILITY OF DIRECTION")
        print("-" * 80)
        print(f"{'Channel':<40} {'pd':>10} {'Interpretation':>20}")
        print("-" * 75)

        for pd_result in report.probability_of_direction:
            print(f"{pd_result.channel:<40} {pd_result.pd:>10.1%} {pd_result.interpretation:>20}")

        # 3. ROPE Analysis
        print("\n\n3. ROPE ANALYSIS")
        print("-" * 80)
        print(f"ROPE: effects between {self.rope_low} and {self.rope_high} are considered 'practically zero'")
        print(f"\n{'Channel':<40} {'% in ROPE':>12} {'% > ROPE':>12} {'Conclusion':>20}")
        print("-" * 90)

        for rope in report.rope_analysis:
            print(f"{rope.channel:<40} {rope.pct_in_rope:>12.1%} {rope.pct_above_rope:>12.1%} {rope.conclusion:>20}")

        # 4. ROI Credible Intervals
        print("\n\n4. ROI CREDIBLE INTERVALS")
        print("-" * 80)
        print(f"{'Channel':<40} {'ROI Mean':>10} {'ROI 5%':>10} {'ROI 95%':>10} {'Significant':>12}")
        print("-" * 90)

        for roi in report.roi_posteriors:
            sig_str = "Yes" if roi.significant else "No"
            print(f"{roi.channel:<40} {roi.roi_mean:>10.2f} {roi.roi_5pct:>10.2f} {roi.roi_95pct:>10.2f} {sig_str:>12}")

        # 5. Prior Sensitivity
        print("\n\n5. PRIOR SENSITIVITY CHECK")
        print("-" * 80)
        print(f"{'Channel':<40} {'Prior ROI':>10} {'Post ROI':>10} {'Shift':>10} {'Data Influence':>15}")
        print("-" * 90)

        for sens in report.prior_sensitivity:
            print(f"{sens.channel:<40} {sens.prior_roi:>10.2f} {sens.posterior_roi:>10.2f} {sens.shift:>+10.2f} {sens.data_influence:>15}")

        print("\n" + "=" * 80)


def get_interpretation_guide() -> str:
    """Get the interpretation guide text for Bayesian vs Frequentist significance."""
    return """
BAYESIAN vs FREQUENTIST SIGNIFICANCE:

+-----------------------------------------------------------------------------+
| Frequentist (OLS)              | Bayesian                                   |
+-----------------------------------------------------------------------------+
| p < 0.05 -> "significant"      | 95% HDI excludes 0 -> "credibly non-zero" |
| p > 0.05 -> "not significant"  | pd > 95% -> "confident in direction"      |
| Binary yes/no decision         | Probability of being meaningful            |
| "Reject null hypothesis"       | "95% probability effect is positive"      |
+-----------------------------------------------------------------------------+

WHAT TO REPORT:

For each channel, report:
1. ROI estimate with credible interval: "ROI = 2.1 [1.3, 3.0]"
2. Probability of positive effect: "98% probability ROI > 0"
3. Practical significance: "95% probability ROI > 0.5 (meaningful threshold)"

DECISION FRAMEWORK:

+-----------------------------------------------------------------------------+
| Metric              | Threshold    | Interpretation                        |
+-----------------------------------------------------------------------------+
| pd (prob direction) | > 95%        | Confident the effect is positive      |
|                     | > 99%        | Very confident                        |
|                     | < 75%        | Weak evidence, interpret cautiously   |
+-----------------------------------------------------------------------------+
| 95% HDI             | Excludes 0   | "Credibly non-zero"                   |
|                     | Above 0.5    | "Credibly positive and meaningful"    |
+-----------------------------------------------------------------------------+
| ROPE                | > 95% above  | Practically significant               |
|                     | > 95% inside | Practically zero (no real effect)     |
+-----------------------------------------------------------------------------+
| Prior shift         | Large shift  | Data informing posterior (good)       |
|                     | No shift     | Prior-driven (may need more data)     |
+-----------------------------------------------------------------------------+

EXAMPLE REPORTING:

"Google Search Brand shows an ROI of 2.1 with a 95% credible interval of
[1.3, 3.0]. There is a 99% probability the effect is positive. The posterior
moved from our prior expectation of 1.5, indicating the data provided
meaningful information beyond our initial assumptions."

vs

"TikTok shows an ROI of 1.0 with a wide credible interval of [-0.5, 2.5].
The probability of a positive effect is 78%. The posterior closely matches
our prior, suggesting limited data signal for this low-spend channel.
Results should be interpreted with caution."
"""
