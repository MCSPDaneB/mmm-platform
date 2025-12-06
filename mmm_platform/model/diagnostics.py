"""
Diagnostics and recommendations for model convergence issues.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class Recommendation:
    """A single setting change recommendation."""
    setting: str
    current: Any
    suggested: Any
    reason: str


@dataclass
class DiagnosticResult:
    """Result from a single diagnostic check."""
    issue: str  # e.g., "Divergent Transitions", "High R-hat"
    severity: str  # "warning" or "critical"
    details: str  # Human-readable description
    recommendations: List[Recommendation] = field(default_factory=list)


class DiagnosticsAdvisor:
    """
    Analyze model convergence issues and provide actionable recommendations.

    This class examines convergence diagnostics (divergences, R-hat, ESS) and
    suggests specific settings changes to improve model fit.
    """

    def __init__(self):
        """Initialize the advisor."""
        pass

    def analyze_from_convergence_dict(
        self,
        convergence: Dict[str, Any],
        sampling_config: Optional[Dict[str, Any]] = None
    ) -> List[DiagnosticResult]:
        """
        Analyze convergence data returned from EC2 and provide recommendations.

        Parameters
        ----------
        convergence : dict
            Convergence data from EC2 response with keys:
            - converged: bool
            - divergences: int
            - high_rhat_params: list
            - warnings: list
            - ess_bulk_min: float
            - ess_tail_min: float
            - ess_sufficient: bool
        sampling_config : dict, optional
            Current sampling settings (draws, tune, chains, target_accept)

        Returns
        -------
        list[DiagnosticResult]
            List of diagnostic results with recommendations
        """
        if convergence is None:
            return []

        results = []

        # Default sampling config if not provided
        if sampling_config is None:
            sampling_config = {
                "draws": 1500,
                "tune": 1500,
                "chains": 4,
                "target_accept": 0.9,
            }

        # Check divergences
        divergence_result = self._check_divergences(convergence, sampling_config)
        if divergence_result:
            results.append(divergence_result)

        # Check R-hat
        rhat_result = self._check_rhat(convergence, sampling_config)
        if rhat_result:
            results.append(rhat_result)

        # Check ESS
        ess_result = self._check_ess(convergence, sampling_config)
        if ess_result:
            results.append(ess_result)

        return results

    def _check_divergences(
        self,
        convergence: Dict[str, Any],
        sampling_config: Dict[str, Any]
    ) -> Optional[DiagnosticResult]:
        """Check for divergent transitions and recommend fixes."""
        divergences = convergence.get("divergences", 0)

        if divergences == 0:
            return None

        current_target_accept = sampling_config.get("target_accept", 0.9)
        current_tune = sampling_config.get("tune", 1500)

        recommendations = []

        if divergences <= 50:
            severity = "warning"
            details = f"{divergences} divergent transitions detected. This may indicate sampling difficulties."

            if current_target_accept < 0.95:
                recommendations.append(Recommendation(
                    setting="target_accept",
                    current=current_target_accept,
                    suggested=0.95,
                    reason="Higher acceptance rate reduces divergences"
                ))

        elif divergences <= 100:
            severity = "warning"
            details = f"{divergences} divergent transitions detected. Consider adjusting sampling settings."

            if current_target_accept < 0.95:
                recommendations.append(Recommendation(
                    setting="target_accept",
                    current=current_target_accept,
                    suggested=0.95,
                    reason="Higher acceptance rate reduces divergences"
                ))

            if current_tune < 2000:
                recommendations.append(Recommendation(
                    setting="tune",
                    current=current_tune,
                    suggested=2000,
                    reason="More warmup iterations help the sampler adapt"
                ))

        else:  # > 100 divergences
            severity = "critical"
            details = f"{divergences} divergent transitions detected. This is a significant number - results may be unreliable."

            if current_target_accept < 0.95:
                recommendations.append(Recommendation(
                    setting="target_accept",
                    current=current_target_accept,
                    suggested=0.95,
                    reason="Higher acceptance rate reduces divergences"
                ))

            if current_tune < 2500:
                recommendations.append(Recommendation(
                    setting="tune",
                    current=current_tune,
                    suggested=2500,
                    reason="Significantly more warmup may be needed"
                ))

            recommendations.append(Recommendation(
                setting="priors",
                current="current",
                suggested="review",
                reason="Check if ROI priors are too narrow or conflict with data"
            ))

        return DiagnosticResult(
            issue="Divergent Transitions",
            severity=severity,
            details=details,
            recommendations=recommendations
        )

    def _check_rhat(
        self,
        convergence: Dict[str, Any],
        sampling_config: Dict[str, Any]
    ) -> Optional[DiagnosticResult]:
        """Check R-hat values and recommend fixes."""
        high_rhat_params = convergence.get("high_rhat_params", [])

        if not high_rhat_params:
            return None

        current_draws = sampling_config.get("draws", 1500)
        current_tune = sampling_config.get("tune", 1500)
        current_chains = sampling_config.get("chains", 4)

        recommendations = []
        n_bad_params = len(high_rhat_params)

        if n_bad_params <= 3:
            severity = "warning"
            details = f"{n_bad_params} parameter(s) have R-hat > 1.01: {', '.join(high_rhat_params[:3])}. Chains may not have converged."

            if current_draws < 2000:
                recommendations.append(Recommendation(
                    setting="draws",
                    current=current_draws,
                    suggested=2000,
                    reason="More samples improve chain mixing"
                ))

        else:
            severity = "critical"
            details = f"{n_bad_params} parameters have R-hat > 1.01. Chains have not converged - results may be unreliable."

            if current_draws < 2500:
                recommendations.append(Recommendation(
                    setting="draws",
                    current=current_draws,
                    suggested=2500,
                    reason="Significantly more samples needed for convergence"
                ))

            if current_tune < 2000:
                recommendations.append(Recommendation(
                    setting="tune",
                    current=current_tune,
                    suggested=2000,
                    reason="More warmup helps chains find the target distribution"
                ))

            if current_chains < 4:
                recommendations.append(Recommendation(
                    setting="chains",
                    current=current_chains,
                    suggested=4,
                    reason="More chains provide better convergence diagnostics"
                ))

        return DiagnosticResult(
            issue="High R-hat Values",
            severity=severity,
            details=details,
            recommendations=recommendations
        )

    def _check_ess(
        self,
        convergence: Dict[str, Any],
        sampling_config: Dict[str, Any]
    ) -> Optional[DiagnosticResult]:
        """Check effective sample size and recommend fixes."""
        ess_bulk_min = convergence.get("ess_bulk_min")
        ess_tail_min = convergence.get("ess_tail_min")
        ess_sufficient = convergence.get("ess_sufficient", True)

        if ess_sufficient and (ess_bulk_min is None or ess_bulk_min >= 400):
            return None

        current_draws = sampling_config.get("draws", 1500)
        current_tune = sampling_config.get("tune", 1500)

        recommendations = []

        min_ess = min(
            ess_bulk_min if ess_bulk_min is not None else float('inf'),
            ess_tail_min if ess_tail_min is not None else float('inf')
        )

        if min_ess < 100:
            severity = "critical"
            details = f"Effective sample size is very low (min ESS = {min_ess:.0f}). Posterior estimates may be unreliable."

            recommendations.append(Recommendation(
                setting="draws",
                current=current_draws,
                suggested=max(current_draws, 2500),
                reason="Significantly more samples needed"
            ))

            recommendations.append(Recommendation(
                setting="tune",
                current=current_tune,
                suggested=max(current_tune, 2000),
                reason="More warmup improves sampling efficiency"
            ))

        elif min_ess < 400:
            severity = "warning"
            details = f"Effective sample size is low (min ESS = {min_ess:.0f}). Consider increasing samples."

            if current_draws < 2000:
                recommendations.append(Recommendation(
                    setting="draws",
                    current=current_draws,
                    suggested=2000,
                    reason="More samples increase effective sample size"
                ))

        else:
            return None

        return DiagnosticResult(
            issue="Low Effective Sample Size",
            severity=severity,
            details=details,
            recommendations=recommendations
        )

    def format_recommendations_text(self, results: List[DiagnosticResult]) -> str:
        """Format diagnostic results as human-readable text."""
        if not results:
            return "No convergence issues detected."

        lines = []
        for result in results:
            icon = "🔴" if result.severity == "critical" else "⚠️"
            lines.append(f"{icon} **{result.issue}**")
            lines.append(f"   {result.details}")

            if result.recommendations:
                lines.append("   Suggested changes:")
                for rec in result.recommendations:
                    lines.append(f"   - {rec.setting}: {rec.current} → {rec.suggested} ({rec.reason})")

            lines.append("")

        return "\n".join(lines)
