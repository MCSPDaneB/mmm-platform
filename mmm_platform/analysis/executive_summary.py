"""
Executive Summary Generator for MMM Platform.

Provides:
- Channel categorization (INCREASE/HOLD/REDUCE)
- Investment recommendations based on marginal ROI
- Reallocation opportunity analysis
- Portfolio summary with actionable insights
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
import logging

from .marginal_roi import MarginalROIAnalyzer, InvestmentPriorityResult, ChannelMarginalROI

logger = logging.getLogger(__name__)


@dataclass
class ReallocationRecommendation:
    """A specific reallocation recommendation."""
    from_channel: str
    to_channel: str
    amount: float
    expected_return: float
    needs_validation: bool


@dataclass
class ExecutiveSummaryResult:
    """Complete executive summary result."""
    priority_result: InvestmentPriorityResult
    reallocation_recommendations: List[ReallocationRecommendation]
    summary_text: str


class ExecutiveSummaryGenerator:
    """
    Generate executive summaries with investment recommendations.

    Based on marginal ROI analysis, provides actionable recommendations
    for budget allocation across marketing channels.
    """

    def __init__(
        self,
        marginal_analyzer: MarginalROIAnalyzer,
        priority_result: Optional[InvestmentPriorityResult] = None,
    ):
        """
        Initialize ExecutiveSummaryGenerator.

        Parameters
        ----------
        marginal_analyzer : MarginalROIAnalyzer
            Analyzer with marginal ROI calculations
        priority_result : InvestmentPriorityResult, optional
            Pre-computed priority result
        """
        self.analyzer = marginal_analyzer
        self._priority_result = priority_result

    @property
    def priority_result(self) -> InvestmentPriorityResult:
        if self._priority_result is None:
            self._priority_result = self.analyzer.run_full_analysis()
        return self._priority_result

    def generate_reallocation_recommendations(
        self,
        max_recommendations: int = 5
    ) -> List[ReallocationRecommendation]:
        """
        Generate specific reallocation recommendations.

        Parameters
        ----------
        max_recommendations : int
            Maximum number of recommendations to generate

        Returns
        -------
        list[ReallocationRecommendation]
            Ordered list of reallocation recommendations
        """
        result = self.priority_result
        recommendations = []

        # Calculate total funds available from reduce channels
        remaining_funds = result.reallocation_potential

        # Allocate to increase channels in priority order
        for to_ch in result.increase_channels[:max_recommendations]:
            if remaining_funds <= 0:
                break

            amount = min(remaining_funds, to_ch.headroom_amount)
            if amount <= 0:
                continue

            remaining_funds -= amount
            expected_return = amount * to_ch.marginal_roi

            # Determine source (simplification: proportional from reduce channels)
            recommendations.append(ReallocationRecommendation(
                from_channel="Reduce channels (combined)",
                to_channel=to_ch.channel_name,
                amount=amount,
                expected_return=expected_return,
                needs_validation=to_ch.needs_test,
            ))

        return recommendations

    def generate_summary(self) -> ExecutiveSummaryResult:
        """
        Generate complete executive summary.

        Returns
        -------
        ExecutiveSummaryResult
            Complete summary with recommendations
        """
        result = self.priority_result
        recommendations = self.generate_reallocation_recommendations()
        summary_text = self._build_summary_text(result, recommendations)

        return ExecutiveSummaryResult(
            priority_result=result,
            reallocation_recommendations=recommendations,
            summary_text=summary_text,
        )

    def _build_summary_text(
        self,
        result: InvestmentPriorityResult,
        recommendations: List[ReallocationRecommendation]
    ) -> str:
        """Build formatted summary text."""
        lines = []
        lines.append("=" * 80)
        lines.append("EXECUTIVE SUMMARY: INVESTMENT PRIORITIES")
        lines.append("=" * 80)

        # Section 1: INCREASE
        lines.append("")
        lines.append("-" * 80)
        lines.append("1. INCREASE INVESTMENT")
        lines.append("-" * 80)
        lines.append("   Marginal ROI > $1.50 with headroom to grow")
        lines.append("")

        if result.increase_channels:
            for ch in result.increase_channels:
                test_flag = " [VALIDATE WITH TEST]" if ch.needs_test else ""
                lines.append(f"   + {ch.channel_name}{test_flag}")
                lines.append(f"     Current Spend:    ${ch.current_spend:>12,.0f}")
                lines.append(f"     Marginal ROI:     ${ch.marginal_roi:>12.2f}  (next dollar returns this)")
                lines.append(f"     Current ROI:      ${ch.current_roi:>12.2f}  (average)")
                lines.append(f"     Headroom:         ${ch.headroom_amount:>12,.0f}  (spend before breakeven)")
                if ch.breakeven_spend:
                    lines.append(f"     Breakeven Spend:  ${ch.breakeven_spend:>12,.0f}")
                if ch.needs_test:
                    lines.append(f"     ROI Range:        ${ch.roi_5pct:>6.2f} - ${ch.roi_95pct:.2f}  (wide CI - run test)")
                lines.append("")
        else:
            lines.append("   No channels qualify. All channels near saturation or low marginal ROI.")
            lines.append("")

        # Section 2: HOLD
        lines.append("-" * 80)
        lines.append("2. HOLD STEADY")
        lines.append("-" * 80)
        lines.append("   Marginal ROI $1.00-$1.50 - profitable but limited upside")
        lines.append("")

        if result.hold_channels:
            for ch in result.hold_channels:
                test_flag = " [VALIDATE WITH TEST]" if ch.needs_test else ""
                lines.append(f"   = {ch.channel_name}{test_flag}")
                lines.append(f"     Current Spend:    ${ch.current_spend:>12,.0f}")
                lines.append(f"     Marginal ROI:     ${ch.marginal_roi:>12.2f}")
                lines.append(f"     Status:           Maintain current investment")
                lines.append("")
        else:
            lines.append("   No channels in this range.")
            lines.append("")

        # Section 3: REDUCE
        lines.append("-" * 80)
        lines.append("3. REDUCE OR REALLOCATE")
        lines.append("-" * 80)
        lines.append("   Marginal ROI < $1.00 - next dollar loses money")
        lines.append("")

        if result.reduce_channels:
            for ch in result.reduce_channels:
                lines.append(f"   - {ch.channel_name}")
                lines.append(f"     Current Spend:    ${ch.current_spend:>12,.0f}")
                lines.append(f"     Marginal ROI:     ${ch.marginal_roi:>12.2f}  [Below breakeven]")
                lines.append(f"     Current ROI:      ${ch.current_roi:>12.2f}")
                lines.append(f"     Action:           Reduce spend or reallocate to higher-priority channels")
                lines.append("")
        else:
            lines.append("   No channels have marginal ROI below $1.00. Portfolio is efficient.")
            lines.append("")

        # Section 4: Channels needing validation
        if result.channels_needing_test:
            lines.append("-" * 80)
            lines.append("4. CHANNELS NEEDING VALIDATION")
            lines.append("-" * 80)
            lines.append("   High uncertainty in ROI estimates - run incrementality test")
            lines.append("")

            for ch in sorted(result.channels_needing_test, key=lambda x: -x.roi_uncertainty):
                if ch in result.increase_channels:
                    action = "INCREASE"
                elif ch in result.reduce_channels:
                    action = "REDUCE"
                else:
                    action = "HOLD"

                lines.append(f"   ? {ch.channel_name} (Recommended Action: {action})")
                lines.append(f"     ROI Range:        ${ch.roi_5pct:>6.2f} - ${ch.roi_95pct:.2f}")
                lines.append(f"     Uncertainty:      ${ch.roi_uncertainty:>12.2f}")
                lines.append(f"     Recommendation:   Run geo holdout or conversion lift test")
                lines.append("")
        else:
            lines.append("-" * 80)
            lines.append("4. CHANNELS NEEDING VALIDATION")
            lines.append("-" * 80)
            lines.append("   All channels have acceptable uncertainty. No testing required.")
            lines.append("")

        # Priority ranking table
        lines.append("=" * 80)
        lines.append("PRIORITY RANKING TABLE")
        lines.append("=" * 80)
        lines.append("")
        lines.append(f"{'Rank':<6} {'Channel':<25} {'Marginal ROI':>12} {'Headroom':>15} {'Action':>12} {'Test?':>8}")
        lines.append("-" * 82)

        for ch in result.channel_analysis:
            if ch in result.increase_channels:
                action = "INCREASE"
            elif ch in result.reduce_channels:
                action = "REDUCE"
            else:
                action = "HOLD"

            if ch.headroom:
                headroom_str = f"${ch.headroom_amount:,.0f}"
            else:
                headroom_str = "Saturated"

            test_str = "Yes" if ch.needs_test else "No"
            lines.append(f"{ch.priority_rank:<6} {ch.channel_name:<25} ${ch.marginal_roi:>11.2f} {headroom_str:>15} {action:>12} {test_str:>8}")

        # Reallocation opportunity
        lines.append("")
        lines.append("=" * 80)
        lines.append("REALLOCATION OPPORTUNITY")
        lines.append("=" * 80)
        lines.append("")
        lines.append(f"    Funds available from REDUCE channels:    ${result.reallocation_potential:>15,.0f}")
        lines.append(f"    Headroom in INCREASE channels:           ${result.headroom_available:>15,.0f}")
        lines.append("")
        lines.append("    RECOMMENDATION:")
        realloc_amount = min(result.reallocation_potential, result.headroom_available)
        lines.append(f"    Reallocate ${realloc_amount:,.0f} from underperforming channels")
        lines.append("    to high marginal ROI channels.")
        lines.append("")

        if recommendations:
            lines.append("   TOP REALLOCATION MOVES:")
            lines.append("   " + "-" * 60)

            for i, rec in enumerate(recommendations[:3]):
                test_note = " (validate with test)" if rec.needs_validation else ""
                lines.append(f"   {i+1}. Allocate ${rec.amount:,.0f} -> {rec.to_channel}{test_note}")
                lines.append(f"      Expected marginal return: ${rec.expected_return:,.0f}")
                lines.append("")

        # Portfolio summary
        lines.append("=" * 80)
        lines.append("PORTFOLIO SUMMARY")
        lines.append("=" * 80)

        n_increase_need_test = sum(1 for c in result.increase_channels if c.needs_test)

        lines.append(f"""
+----------------------------------------------------------------------------+
|  CURRENT PORTFOLIO                                                         |
+----------------------------------------------------------------------------+
|  Total Media Spend:              ${result.total_spend:>15,.0f}                     |
|  Total Media Contribution:       ${result.total_contribution:>15,.0f}                     |
|  Portfolio ROI:                  ${result.portfolio_roi:>15.2f}                     |
+----------------------------------------------------------------------------+
|  RECOMMENDATIONS                                                           |
|    + Increase:      {len(result.increase_channels):>3} channels  ({n_increase_need_test} need validation)              |
|    = Hold:          {len(result.hold_channels):>3} channels                                              |
|    - Reduce:        {len(result.reduce_channels):>3} channels                                              |
+----------------------------------------------------------------------------+
|  REALLOCATION POTENTIAL                                                    |
|    Move ${result.reallocation_potential:>12,.0f} from underperformers                            |
|    To channels with ${result.headroom_available:>12,.0f} headroom                           |
+----------------------------------------------------------------------------+
""")

        return "\n".join(lines)

    def print_summary(self) -> None:
        """Print formatted executive summary."""
        summary = self.generate_summary()
        print(summary.summary_text)

    def get_summary_dict(self) -> Dict[str, Any]:
        """
        Get executive summary as a dictionary for JSON export or UI.

        Returns
        -------
        dict
            Summary data suitable for JSON or UI rendering
        """
        result = self.priority_result
        recommendations = self.generate_reallocation_recommendations()

        def channel_to_dict(ch: ChannelMarginalROI) -> dict:
            return {
                "channel": ch.channel,
                "channel_name": ch.channel_name,
                "current_spend": ch.current_spend,
                "current_roi": ch.current_roi,
                "marginal_roi": ch.marginal_roi,
                "breakeven_spend": ch.breakeven_spend,
                "headroom": ch.headroom,
                "headroom_amount": ch.headroom_amount,
                "priority_rank": ch.priority_rank,
                "roi_5pct": ch.roi_5pct,
                "roi_95pct": ch.roi_95pct,
                "roi_uncertainty": ch.roi_uncertainty,
                "prob_profitable": ch.prob_profitable,
                "needs_test": ch.needs_test,
            }

        return {
            "portfolio": {
                "total_spend": result.total_spend,
                "total_contribution": result.total_contribution,
                "portfolio_roi": result.portfolio_roi,
                "reallocation_potential": result.reallocation_potential,
                "headroom_available": result.headroom_available,
            },
            "recommendations": {
                "increase": [channel_to_dict(ch) for ch in result.increase_channels],
                "hold": [channel_to_dict(ch) for ch in result.hold_channels],
                "reduce": [channel_to_dict(ch) for ch in result.reduce_channels],
                "needs_validation": [channel_to_dict(ch) for ch in result.channels_needing_test],
            },
            "priority_table": [channel_to_dict(ch) for ch in result.channel_analysis],
            "reallocation_moves": [
                {
                    "from_channel": r.from_channel,
                    "to_channel": r.to_channel,
                    "amount": r.amount,
                    "expected_return": r.expected_return,
                    "needs_validation": r.needs_validation,
                }
                for r in recommendations
            ],
            "counts": {
                "increase": len(result.increase_channels),
                "hold": len(result.hold_channels),
                "reduce": len(result.reduce_channels),
                "needs_validation": len(result.channels_needing_test),
            },
        }

    @classmethod
    def from_mmm_wrapper(cls, mmm_wrapper: Any) -> "ExecutiveSummaryGenerator":
        """
        Create generator from an MMMWrapper instance.

        Parameters
        ----------
        mmm_wrapper : MMMWrapper
            Fitted model wrapper

        Returns
        -------
        ExecutiveSummaryGenerator
            Generator instance
        """
        analyzer = MarginalROIAnalyzer.from_mmm_wrapper(mmm_wrapper)
        return cls(analyzer)
