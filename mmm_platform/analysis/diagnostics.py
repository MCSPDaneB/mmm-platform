"""
Model diagnostics and fit analysis.
"""

import numpy as np
import pandas as pd
from typing import Any, Optional
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class DiagnosticResult:
    """Result of a diagnostic check."""
    name: str
    passed: bool
    value: Any
    threshold: Optional[Any] = None
    message: str = ""
    severity: str = "info"  # info, warning, error


@dataclass
class DiagnosticReport:
    """Complete diagnostic report."""
    results: list[DiagnosticResult] = field(default_factory=list)
    overall_passed: bool = True
    issues: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    def add_result(self, result: DiagnosticResult) -> None:
        """Add a diagnostic result."""
        self.results.append(result)
        if not result.passed:
            self.overall_passed = False
            if result.severity == "error":
                self.issues.append(result.message)


class ModelDiagnostics:
    """
    Comprehensive model diagnostics.

    Analyzes:
    - Overall fit quality (R², MAPE, residuals)
    - Residual patterns (autocorrelation, heteroscedasticity)
    - Systematic prediction errors
    - Variable contributions
    """

    def __init__(
        self,
        actual: pd.Series,
        fitted: pd.Series,
        dates: pd.DatetimeIndex,
        contribs: pd.DataFrame,
        channel_cols: list[str],
        control_cols: list[str],
        target_col: str,
    ):
        """
        Initialize ModelDiagnostics.

        Parameters
        ----------
        actual : pd.Series
            Actual target values.
        fitted : pd.Series
            Fitted (predicted) values.
        dates : pd.DatetimeIndex
            Date index.
        contribs : pd.DataFrame
            Contribution dataframe.
        channel_cols : list[str]
            Channel column names.
        control_cols : list[str]
            Control column names.
        target_col : str
            Target column name.
        """
        self.actual = actual
        self.fitted = fitted
        self.dates = dates
        self.residuals = actual - fitted
        self.contribs = contribs
        self.channel_cols = channel_cols
        self.control_cols = control_cols
        self.target_col = target_col

    def run_all_diagnostics(
        self,
        r2_threshold: float = 0.85,
        mape_threshold: float = 15.0,
        autocorr_threshold: float = 0.3,
    ) -> DiagnosticReport:
        """
        Run all diagnostic checks.

        Parameters
        ----------
        r2_threshold : float
            Minimum acceptable R².
        mape_threshold : float
            Maximum acceptable MAPE (%).
        autocorr_threshold : float
            Maximum acceptable autocorrelation.

        Returns
        -------
        DiagnosticReport
            Complete diagnostic report.
        """
        report = DiagnosticReport()

        # Fit quality
        report.add_result(self._check_r2(r2_threshold))
        report.add_result(self._check_mape(mape_threshold))

        # Residual analysis
        report.add_result(self._check_autocorrelation(autocorr_threshold))
        report.add_result(self._check_heteroscedasticity())
        report.add_result(self._check_residual_trend())

        # Systematic errors
        run_result = self._check_prediction_runs()
        report.add_result(run_result)

        # Generate recommendations
        report.recommendations = self._generate_recommendations(report)

        return report

    def _check_r2(self, threshold: float) -> DiagnosticResult:
        """Check R-squared."""
        ss_res = np.sum(self.residuals ** 2)
        ss_tot = np.sum((self.actual - self.actual.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot

        passed = r2 >= threshold
        return DiagnosticResult(
            name="R-squared",
            passed=passed,
            value=float(r2),
            threshold=threshold,
            message=f"R² = {r2:.3f}" + ("" if passed else f" (below threshold {threshold})"),
            severity="warning" if not passed else "info",
        )

    def _check_mape(self, threshold: float) -> DiagnosticResult:
        """Check Mean Absolute Percentage Error."""
        mape = np.mean(np.abs(self.residuals / self.actual)) * 100

        passed = mape <= threshold
        return DiagnosticResult(
            name="MAPE",
            passed=passed,
            value=float(mape),
            threshold=threshold,
            message=f"MAPE = {mape:.1f}%" + ("" if passed else f" (above threshold {threshold}%)"),
            severity="warning" if not passed else "info",
        )

    def _check_autocorrelation(self, threshold: float) -> DiagnosticResult:
        """Check residual autocorrelation."""
        autocorr_1 = self.residuals.autocorr(lag=1)

        passed = abs(autocorr_1) <= threshold
        return DiagnosticResult(
            name="Autocorrelation (lag 1)",
            passed=passed,
            value=float(autocorr_1),
            threshold=threshold,
            message=f"Autocorr = {autocorr_1:.3f}" + ("" if passed else " - model missing temporal patterns"),
            severity="warning" if not passed else "info",
        )

    def _check_heteroscedasticity(self) -> DiagnosticResult:
        """Check for heteroscedasticity (variance changing with fitted values)."""
        het_corr = np.corrcoef(self.fitted.values, np.abs(self.residuals.values))[0, 1]

        passed = abs(het_corr) <= 0.3
        return DiagnosticResult(
            name="Heteroscedasticity",
            passed=passed,
            value=float(het_corr),
            threshold=0.3,
            message=f"Correlation = {het_corr:.3f}" + ("" if passed else " - residual variance changes with predictions"),
            severity="warning" if not passed else "info",
        )

    def _check_residual_trend(self) -> DiagnosticResult:
        """Check for trend in residuals over time."""
        residuals_reset = self.residuals.reset_index(drop=True)
        time_idx = np.arange(len(residuals_reset))
        trend_corr = np.corrcoef(time_idx, residuals_reset.values)[0, 1]

        passed = abs(trend_corr) <= 0.2
        return DiagnosticResult(
            name="Residual Trend",
            passed=passed,
            value=float(trend_corr),
            threshold=0.2,
            message=f"Trend correlation = {trend_corr:.3f}" + ("" if passed else " - systematic drift in predictions"),
            severity="warning" if not passed else "info",
        )

    def _check_prediction_runs(self) -> DiagnosticResult:
        """Check for runs of same-sign residuals (systematic under/over prediction)."""
        signs = np.sign(self.residuals.values)
        runs = []
        current_run_start = 0
        current_sign = signs[0]

        for i in range(1, len(signs)):
            if signs[i] != current_sign:
                runs.append({
                    "start_idx": current_run_start,
                    "end_idx": i - 1,
                    "length": i - current_run_start,
                    "sign": current_sign,
                })
                current_run_start = i
                current_sign = signs[i]

        # Last run
        runs.append({
            "start_idx": current_run_start,
            "end_idx": len(signs) - 1,
            "length": len(signs) - current_run_start,
            "sign": current_sign,
        })

        longest_run = max(runs, key=lambda x: x["length"])
        max_run_length = longest_run["length"]

        passed = max_run_length <= 6

        message = f"Longest run = {max_run_length} periods"
        if not passed:
            start_date = self.dates[longest_run["start_idx"]]
            end_date = self.dates[longest_run["end_idx"]]
            direction = "under" if longest_run["sign"] > 0 else "over"
            message += f" ({direction}-predicting from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})"

        return DiagnosticResult(
            name="Prediction Runs",
            passed=passed,
            value={
                "max_run_length": max_run_length,
                "longest_run": longest_run,
                "all_runs": runs,
            },
            threshold=6,
            message=message,
            severity="warning" if not passed else "info",
        )

    def get_worst_predictions(self, n: int = 10) -> pd.DataFrame:
        """
        Get the periods with worst prediction errors.

        Parameters
        ----------
        n : int
            Number of periods to return.

        Returns
        -------
        pd.DataFrame
            Worst predictions.
        """
        error_df = pd.DataFrame({
            "date": self.dates,
            "actual": self.actual.values,
            "fitted": self.fitted.values,
            "error": self.residuals.values,
            "error_pct": (self.residuals.values / self.actual.values * 100),
            "abs_error": np.abs(self.residuals.values),
        })

        return error_df.nlargest(n, "abs_error")

    def get_monthly_residuals(self) -> pd.DataFrame:
        """
        Get residual statistics by month.

        Returns
        -------
        pd.DataFrame
            Monthly residual summary.
        """
        resid_df = pd.DataFrame({
            "residual": self.residuals.values,
            "month": self.dates.month,
        })

        return resid_df.groupby("month").agg({
            "residual": ["mean", "std", "count"]
        }).round(2)

    def get_contribution_summary(self) -> pd.DataFrame:
        """
        Get summary of variable contributions.

        Returns
        -------
        pd.DataFrame
            Contribution summary.
        """
        component_cols = [c for c in self.contribs.columns if c != self.target_col]
        total_abs = self.contribs[component_cols].abs().sum()
        total_signed = self.contribs[component_cols].sum()

        summary = pd.DataFrame({
            "total_contribution": total_signed,
            "abs_contribution": total_abs,
            "pct_of_total": (total_abs / total_abs.sum() * 100).round(1),
        })

        return summary.sort_values("abs_contribution", ascending=False)

    def _generate_recommendations(self, report: DiagnosticReport) -> list[str]:
        """Generate recommendations based on diagnostic results."""
        recommendations = []

        for result in report.results:
            if not result.passed:
                if result.name == "R-squared":
                    recommendations.append(
                        "Consider adding more control variables or adjusting priors"
                    )
                elif result.name == "Autocorrelation (lag 1)":
                    recommendations.append(
                        "Consider using time_varying_intercept=True or adding AR(1) errors"
                    )
                elif result.name == "Residual Trend":
                    recommendations.append(
                        "Check if trend variable is properly specified"
                    )
                elif result.name == "Prediction Runs":
                    run_data = result.value
                    if run_data["max_run_length"] > 6:
                        longest = run_data["longest_run"]
                        start_date = self.dates[longest["start_idx"]].strftime("%Y-%m-%d")
                        end_date = self.dates[longest["end_idx"]].strftime("%Y-%m-%d")
                        recommendations.append(
                            f"Consider adding a dummy variable for period {start_date} to {end_date}"
                        )

        return recommendations

    @classmethod
    def from_mmm_wrapper(cls, mmm_wrapper: Any) -> "ModelDiagnostics":
        """
        Create diagnostics from an MMMWrapper instance.

        Parameters
        ----------
        mmm_wrapper : MMMWrapper
            Fitted model wrapper.

        Returns
        -------
        ModelDiagnostics
            Diagnostics instance.
        """
        contribs = mmm_wrapper.get_contributions()
        target_col = mmm_wrapper.config.data.target_column

        # Get actual values aligned with contributions
        df_indexed = mmm_wrapper.df_scaled.set_index(mmm_wrapper.config.data.date_column)
        actual = df_indexed[target_col].reindex(contribs.index)

        # Compute fitted values
        component_cols = [c for c in contribs.columns if c != target_col]
        fitted = contribs[component_cols].sum(axis=1)

        return cls(
            actual=actual,
            fitted=fitted,
            dates=contribs.index,
            contribs=contribs,
            channel_cols=mmm_wrapper.config.get_channel_columns(),
            control_cols=mmm_wrapper.control_cols,
            target_col=target_col,
        )
