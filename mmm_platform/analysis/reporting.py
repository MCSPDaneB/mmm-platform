"""
Report generation for MMM results.
"""

import json
from typing import Any, Optional
from datetime import datetime
from pathlib import Path
import logging

from .diagnostics import ModelDiagnostics
from .contributions import ContributionAnalyzer
from .marginal_roi import MarginalROIAnalyzer
from .executive_summary import ExecutiveSummaryGenerator

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Generate reports from MMM results.

    Supports:
    - JSON reports for programmatic access
    - HTML reports for viewing
    - Summary dictionaries for UI display
    """

    def __init__(self, mmm_wrapper: Any):
        """
        Initialize ReportGenerator.

        Parameters
        ----------
        mmm_wrapper : MMMWrapper
            Fitted model wrapper.
        """
        self.wrapper = mmm_wrapper
        self.diagnostics = ModelDiagnostics.from_mmm_wrapper(mmm_wrapper)
        self.contributions = ContributionAnalyzer.from_mmm_wrapper(mmm_wrapper)

    def generate_summary(self) -> dict:
        """
        Generate a summary dictionary suitable for UI display.

        Returns
        -------
        dict
            Summary of model results.
        """
        # Run diagnostics
        diag_report = self.diagnostics.run_all_diagnostics()

        # Get fit statistics
        fit_stats = self.wrapper.get_fit_statistics()

        # Get contribution breakdown
        breakdown = self.contributions.get_contribution_breakdown()

        # Get channel ROI
        channel_roi = self.contributions.get_channel_roi()

        return {
            "metadata": {
                "model_name": self.wrapper.config.name,
                "generated_at": datetime.now().isoformat(),
                "fitted_at": self.wrapper.fitted_at.isoformat() if self.wrapper.fitted_at else None,
                "fit_duration_seconds": self.wrapper.fit_duration_seconds,
            },
            "data_summary": self.wrapper.get_data_summary(),
            "fit_statistics": fit_stats,
            "diagnostics": {
                "overall_passed": diag_report.overall_passed,
                "issues": diag_report.issues,
                "recommendations": diag_report.recommendations,
                "results": [
                    {
                        "name": r.name,
                        "passed": r.passed,
                        "value": r.value if not isinstance(r.value, dict) else None,
                        "message": r.message,
                    }
                    for r in diag_report.results
                ],
            },
            "contributions": {
                "breakdown": breakdown,
                "channel_roi": channel_roi.to_dict(orient="records") if len(channel_roi) > 0 else [],
            },
        }

    def generate_json_report(self, output_path: Optional[Path] = None) -> str:
        """
        Generate a JSON report.

        Parameters
        ----------
        output_path : Path, optional
            Path to save the report.

        Returns
        -------
        str
            JSON string.
        """
        summary = self.generate_summary()

        # Add detailed tables
        summary["channel_roi_detail"] = self.contributions.get_channel_roi().to_dict(orient="records")
        summary["control_contributions"] = self.contributions.get_control_contributions().to_dict(orient="records")
        summary["grouped_contributions"] = self.contributions.get_grouped_contributions().to_dict(orient="records")

        json_str = json.dumps(summary, indent=2, default=str)

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                f.write(json_str)
            logger.info(f"Report saved to {output_path}")

        return json_str

    def generate_html_report(self, output_path: Optional[Path] = None) -> str:
        """
        Generate an HTML report.

        Parameters
        ----------
        output_path : Path, optional
            Path to save the report.

        Returns
        -------
        str
            HTML string.
        """
        summary = self.generate_summary()
        channel_roi = self.contributions.get_channel_roi()
        grouped = self.contributions.get_grouped_contributions()

        # Build channel ROI rows
        channel_rows = []
        for _, row in channel_roi.iterrows():
            channel_rows.append(
                f"<tr><td>{row['channel']}</td>"
                f"<td>${row['spend_real']:,.0f}</td>"
                f"<td>${row['contribution_real']:,.0f}</td>"
                f"<td>{row['roi']:.2f}</td></tr>"
            )
        channel_rows_html = "\n            ".join(channel_rows)

        # Build grouped contribution rows
        grouped_rows = []
        for _, row in grouped.iterrows():
            grouped_rows.append(
                f"<tr><td>{row['group']}</td>"
                f"<td>${row['contribution_real']:,.0f}</td>"
                f"<td>{row['pct_of_total']:.1f}%</td></tr>"
            )
        grouped_rows_html = "\n            ".join(grouped_rows)

        # Build warning divs
        warning_html = "".join(
            f'<div class="warning">{issue}</div>'
            for issue in summary['diagnostics']['issues']
        )

        # Build recommendation divs
        rec_header = '<h3>Recommendations</h3>' if summary['diagnostics']['recommendations'] else ''
        rec_html = "".join(
            f'<div class="recommendation">{rec}</div>'
            for rec in summary['diagnostics']['recommendations']
        )

        # Diagnostics status
        diag_class = 'pass' if summary['diagnostics']['overall_passed'] else 'fail'
        diag_text = 'PASSED' if summary['diagnostics']['overall_passed'] else 'ISSUES DETECTED'

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>MMM Report - {summary['metadata']['model_name']}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        .metric-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin: 20px 0; }}
        .metric-card {{ background: #f0f7f0; padding: 20px; border-radius: 8px; text-align: center; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #4CAF50; }}
        .metric-label {{ color: #666; margin-top: 5px; }}
        .pass {{ color: #4CAF50; }}
        .fail {{ color: #f44336; }}
        .warning {{ background: #fff3cd; padding: 10px; border-radius: 4px; margin: 10px 0; }}
        .recommendation {{ background: #e3f2fd; padding: 10px; border-radius: 4px; margin: 5px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Marketing Mix Model Report</h1>
        <p><strong>Model:</strong> {summary['metadata']['model_name']}</p>
        <p><strong>Generated:</strong> {summary['metadata']['generated_at']}</p>

        <h2>Model Fit Statistics</h2>
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-value">{summary['fit_statistics']['r2']:.3f}</div>
                <div class="metric-label">R-squared</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{summary['fit_statistics']['mape']:.1f}%</div>
                <div class="metric-label">MAPE</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{summary['fit_statistics']['rmse']:.1f}</div>
                <div class="metric-label">RMSE</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{summary['fit_statistics']['n_observations']}</div>
                <div class="metric-label">Observations</div>
            </div>
        </div>

        <h2>Diagnostics</h2>
        <p>Overall: <span class="{diag_class}">{diag_text}</span></p>

        {warning_html}

        {rec_header}
        {rec_html}

        <h2>Channel ROI</h2>
        <table>
            <tr>
                <th>Channel</th>
                <th>Spend</th>
                <th>Contribution</th>
                <th>ROI</th>
            </tr>
            {channel_rows_html}
        </table>

        <h2>Contribution Breakdown</h2>
        <table>
            <tr>
                <th>Category</th>
                <th>Contribution</th>
                <th>% of Total</th>
            </tr>
            {grouped_rows_html}
        </table>
    </div>
</body>
</html>
"""

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                f.write(html)
            logger.info(f"HTML report saved to {output_path}")

        return html

    def export_contributions_csv(self, output_path: Path) -> None:
        """
        Export contributions to CSV.

        Parameters
        ----------
        output_path : Path
            Path to save the CSV.
        """
        contribs = self.contributions.get_time_series_contributions()
        contribs = contribs.reset_index()

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        contribs.to_csv(output_path, index=False)

        logger.info(f"Contributions exported to {output_path}")

    def generate_executive_summary(self) -> dict:
        """
        Generate executive summary with investment recommendations.

        Returns
        -------
        dict
            Executive summary data including:
            - Portfolio metrics
            - Channel recommendations (INCREASE/HOLD/REDUCE)
            - Marginal ROI analysis
            - Reallocation opportunities
        """
        try:
            exec_gen = ExecutiveSummaryGenerator.from_mmm_wrapper(self.wrapper)
            return exec_gen.get_summary_dict()
        except Exception as e:
            logger.warning(f"Could not generate executive summary: {e}")
            return {
                "error": str(e),
                "message": "Executive summary requires inference data from model fitting."
            }

    def print_executive_summary(self) -> None:
        """Print formatted executive summary to console."""
        try:
            exec_gen = ExecutiveSummaryGenerator.from_mmm_wrapper(self.wrapper)
            exec_gen.print_summary()
        except Exception as e:
            logger.warning(f"Could not generate executive summary: {e}")
            print(f"Error generating executive summary: {e}")

    def export_priority_table_csv(self, output_path: Path) -> None:
        """
        Export investment priority table to CSV.

        Parameters
        ----------
        output_path : Path
            Path to save the CSV.
        """
        try:
            marginal_analyzer = MarginalROIAnalyzer.from_mmm_wrapper(self.wrapper)
            priority_df = marginal_analyzer.get_priority_table()

            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            priority_df.to_csv(output_path, index=False)

            logger.info(f"Priority table exported to {output_path}")
        except Exception as e:
            logger.warning(f"Could not export priority table: {e}")
