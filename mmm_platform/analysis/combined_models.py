"""
Combined Model Analysis for Multi-Outcome MMM.

Provides functionality to combine multiple MMM models (e.g., online + offline)
and generate unified investment recommendations across different views:
- Online Revenue
- Offline Revenue
- Total Revenue
- Profit (margin-adjusted)

This enables decision-making when a single marketing channel drives
multiple outcomes with different margins.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import logging

from .marginal_roi import (
    MarginalROIAnalyzer,
    ChannelMarginalROI,
    InvestmentPriorityResult,
    calculate_marginal_roi,
)

logger = logging.getLogger(__name__)


@dataclass
class CombinedChannelAnalysis:
    """Combined analysis for a single channel across multiple models."""
    channel: str
    channel_name: str
    current_spend: float

    # Marginal ROIs from each model
    marginal_online: float
    marginal_offline: float
    marginal_total: float  # online + offline
    marginal_profit: float  # margin-weighted

    # Current (average) ROIs
    current_roi_online: float
    current_roi_offline: float
    current_roi_total: float
    current_roi_profit: float

    # Headroom from each model
    headroom_online: float
    headroom_offline: float

    # Model parameters (for breakeven calculation)
    beta_online: float = 0.0
    lam_online: float = 1.0
    beta_offline: float = 0.0
    lam_offline: float = 1.0
    x_max: float = 1.0


@dataclass
class ViewRecommendation:
    """Recommendations for a specific view (online, offline, total, profit)."""
    view: str
    view_title: str
    threshold_high: float
    threshold_low: float
    increase_channels: List[CombinedChannelAnalysis]
    hold_channels: List[CombinedChannelAnalysis]
    reduce_channels: List[CombinedChannelAnalysis]
    ranked_channels: List[Tuple[int, CombinedChannelAnalysis, str]]  # rank, channel, action


@dataclass
class CombinedModelResult:
    """Complete combined model analysis result."""
    combined_analysis: List[CombinedChannelAnalysis]
    view_recommendations: Dict[str, ViewRecommendation]
    conflicting_channels: List[str]  # Channels with different recommendations across views
    online_margin: float
    offline_margin: float


class CombinedModelAnalyzer:
    """
    Analyzer for combining multiple MMM models.

    Merges channel-level analysis from two models (e.g., online and offline)
    and provides unified recommendations across multiple views:
    - Online: Optimize for online revenue
    - Offline: Optimize for offline revenue
    - Total: Optimize for total revenue (sum)
    - Profit: Optimize for profit (margin-weighted)
    """

    # Default thresholds for each view
    VIEW_CONFIG = {
        "online": {
            "title": "ONLINE REVENUE",
            "threshold_high": 1.5,
            "threshold_low": 1.0,
        },
        "offline": {
            "title": "OFFLINE REVENUE",
            "threshold_high": 1.5,
            "threshold_low": 1.0,
        },
        "total": {
            "title": "TOTAL REVENUE",
            "threshold_high": 1.5,
            "threshold_low": 1.0,
        },
        "profit": {
            "title": "PROFIT",
            "threshold_high": 0.5,  # $0.50 profit per $1 spent
            "threshold_low": 0.3,
        },
    }

    def __init__(
        self,
        online_margin: float = 0.35,
        offline_margin: float = 0.25,
    ):
        """
        Initialize CombinedModelAnalyzer.

        Parameters
        ----------
        online_margin : float
            Profit margin for online revenue (e.g., 0.35 = 35%)
        offline_margin : float
            Profit margin for offline revenue (e.g., 0.25 = 25%)
        """
        self.online_margin = online_margin
        self.offline_margin = offline_margin
        self._combined_analysis: Optional[List[CombinedChannelAnalysis]] = None

    def merge_channel_analysis(
        self,
        channel_analysis_online: List[ChannelMarginalROI],
        channel_analysis_offline: List[ChannelMarginalROI],
    ) -> List[CombinedChannelAnalysis]:
        """
        Combine channel analysis from online and offline models.

        Parameters
        ----------
        channel_analysis_online : list[ChannelMarginalROI]
            Channel analysis from online revenue model
        channel_analysis_offline : list[ChannelMarginalROI]
            Channel analysis from offline revenue model

        Returns
        -------
        list[CombinedChannelAnalysis]
            Combined analysis for each channel
        """
        # Create lookup for offline channels
        offline_lookup = {ch.channel: ch for ch in channel_analysis_offline}

        combined = []
        for ch_on in channel_analysis_online:
            ch_off = offline_lookup.get(ch_on.channel)

            if ch_off is None:
                logger.warning(f"{ch_on.channel} not found in offline model, skipping")
                continue

            # Current spend should be same in both (same input data)
            current_spend = ch_on.current_spend

            # Marginal ROIs from each model
            marginal_online = ch_on.marginal_roi
            marginal_offline = ch_off.marginal_roi

            # Combined metrics
            marginal_total = marginal_online + marginal_offline
            marginal_profit = (
                marginal_online * self.online_margin +
                marginal_offline * self.offline_margin
            )

            # Current ROIs
            current_roi_online = ch_on.current_roi
            current_roi_offline = ch_off.current_roi
            current_roi_total = current_roi_online + current_roi_offline
            current_roi_profit = (
                current_roi_online * self.online_margin +
                current_roi_offline * self.offline_margin
            )

            combined.append(CombinedChannelAnalysis(
                channel=ch_on.channel,
                channel_name=ch_on.channel_name,
                current_spend=current_spend,
                marginal_online=marginal_online,
                marginal_offline=marginal_offline,
                marginal_total=marginal_total,
                marginal_profit=marginal_profit,
                current_roi_online=current_roi_online,
                current_roi_offline=current_roi_offline,
                current_roi_total=current_roi_total,
                current_roi_profit=current_roi_profit,
                headroom_online=ch_on.headroom_amount,
                headroom_offline=ch_off.headroom_amount,
            ))

        self._combined_analysis = combined
        return combined

    def categorize_channels(
        self,
        combined_analysis: List[CombinedChannelAnalysis],
        view: str = "total",
    ) -> Tuple[List[CombinedChannelAnalysis], List[CombinedChannelAnalysis], List[CombinedChannelAnalysis]]:
        """
        Categorize channels into INCREASE/HOLD/REDUCE based on a specific view.

        Parameters
        ----------
        combined_analysis : list[CombinedChannelAnalysis]
            Combined channel analysis
        view : str
            View to use: "online", "offline", "total", or "profit"

        Returns
        -------
        tuple
            (increase_channels, hold_channels, reduce_channels)
        """
        config = self.VIEW_CONFIG.get(view, self.VIEW_CONFIG["total"])
        threshold_high = config["threshold_high"]
        threshold_low = config["threshold_low"]

        increase = []
        hold = []
        reduce = []

        for ch in combined_analysis:
            # Get marginal ROI for this view
            marginal = getattr(ch, f"marginal_{view}")

            if marginal > threshold_high:
                increase.append(ch)
            elif marginal < threshold_low:
                reduce.append(ch)
            else:
                hold.append(ch)

        # Sort each category
        increase = sorted(increase, key=lambda x: -getattr(x, f"marginal_{view}"))
        hold = sorted(hold, key=lambda x: -getattr(x, f"marginal_{view}"))
        reduce = sorted(reduce, key=lambda x: getattr(x, f"marginal_{view}"))

        return increase, hold, reduce

    def get_view_recommendation(
        self,
        combined_analysis: List[CombinedChannelAnalysis],
        view: str,
    ) -> ViewRecommendation:
        """
        Get recommendations for a specific view.

        Parameters
        ----------
        combined_analysis : list[CombinedChannelAnalysis]
            Combined channel analysis
        view : str
            View: "online", "offline", "total", or "profit"

        Returns
        -------
        ViewRecommendation
            Recommendations for the view
        """
        config = self.VIEW_CONFIG.get(view, self.VIEW_CONFIG["total"])
        increase, hold, reduce = self.categorize_channels(combined_analysis, view)

        # Create ranked list with actions
        ranked = sorted(
            combined_analysis,
            key=lambda x: -getattr(x, f"marginal_{view}")
        )

        ranked_with_action = []
        for i, ch in enumerate(ranked):
            if ch in increase:
                action = "INCREASE"
            elif ch in reduce:
                action = "REDUCE"
            else:
                action = "HOLD"
            ranked_with_action.append((i + 1, ch, action))

        return ViewRecommendation(
            view=view,
            view_title=config["title"],
            threshold_high=config["threshold_high"],
            threshold_low=config["threshold_low"],
            increase_channels=increase,
            hold_channels=hold,
            reduce_channels=reduce,
            ranked_channels=ranked_with_action,
        )

    def find_conflicting_channels(
        self,
        combined_analysis: List[CombinedChannelAnalysis],
    ) -> List[str]:
        """
        Find channels with conflicting recommendations across views.

        Parameters
        ----------
        combined_analysis : list[CombinedChannelAnalysis]
            Combined channel analysis

        Returns
        -------
        list[str]
            Channel names with conflicts
        """
        conflicts = []

        for ch in combined_analysis:
            actions = set()
            for view in ["online", "offline", "total", "profit"]:
                increase, hold, reduce = self.categorize_channels(combined_analysis, view)
                if ch in increase:
                    actions.add("INCREASE")
                elif ch in reduce:
                    actions.add("REDUCE")
                else:
                    actions.add("HOLD")

            if len(actions) > 1:
                conflicts.append(ch.channel_name)

        return conflicts

    def run_full_analysis(
        self,
        channel_analysis_online: List[ChannelMarginalROI],
        channel_analysis_offline: List[ChannelMarginalROI],
    ) -> CombinedModelResult:
        """
        Run complete combined model analysis.

        Parameters
        ----------
        channel_analysis_online : list[ChannelMarginalROI]
            From online model's MarginalROIAnalyzer
        channel_analysis_offline : list[ChannelMarginalROI]
            From offline model's MarginalROIAnalyzer

        Returns
        -------
        CombinedModelResult
            Complete analysis result
        """
        # Merge channel analysis
        combined = self.merge_channel_analysis(
            channel_analysis_online,
            channel_analysis_offline,
        )

        # Get recommendations for each view
        view_recommendations = {}
        for view in ["online", "offline", "total", "profit"]:
            view_recommendations[view] = self.get_view_recommendation(combined, view)

        # Find conflicts
        conflicts = self.find_conflicting_channels(combined)

        return CombinedModelResult(
            combined_analysis=combined,
            view_recommendations=view_recommendations,
            conflicting_channels=conflicts,
            online_margin=self.online_margin,
            offline_margin=self.offline_margin,
        )

    def get_summary_table(
        self,
        combined_analysis: List[CombinedChannelAnalysis],
    ) -> pd.DataFrame:
        """
        Get combined analysis as a DataFrame.

        Parameters
        ----------
        combined_analysis : list[CombinedChannelAnalysis]
            Combined channel analysis

        Returns
        -------
        pd.DataFrame
            Summary table with all views
        """
        data = []
        for ch in combined_analysis:
            # Get actions for each view
            actions = {}
            for view in ["online", "offline", "total", "profit"]:
                increase, hold, reduce = self.categorize_channels(combined_analysis, view)
                if ch in increase:
                    actions[f"action_{view}"] = "INCREASE"
                elif ch in reduce:
                    actions[f"action_{view}"] = "REDUCE"
                else:
                    actions[f"action_{view}"] = "HOLD"

            data.append({
                "channel": ch.channel_name,
                "current_spend": ch.current_spend,
                "marginal_online": ch.marginal_online,
                "marginal_offline": ch.marginal_offline,
                "marginal_total": ch.marginal_total,
                "marginal_profit": ch.marginal_profit,
                "current_roi_online": ch.current_roi_online,
                "current_roi_offline": ch.current_roi_offline,
                "current_roi_total": ch.current_roi_total,
                "current_roi_profit": ch.current_roi_profit,
                **actions,
            })

        df = pd.DataFrame(data)
        return df.sort_values("marginal_total", ascending=False)

    def print_summary(
        self,
        result: CombinedModelResult,
    ) -> None:
        """
        Print formatted combined model summary.

        Parameters
        ----------
        result : CombinedModelResult
            Result from run_full_analysis()
        """
        print("=" * 100)
        print("COMBINED MODEL ANALYSIS: ONLINE + OFFLINE")
        print("=" * 100)
        print(f"\nMargins: Online = {self.online_margin*100:.0f}%, Offline = {self.offline_margin*100:.0f}%")

        # Master table
        print("\n" + "=" * 100)
        print("MARGINAL ROI BY VIEW")
        print("=" * 100)

        print(f"\n{'Channel':<25} {'Spend':>12} {'Online':>10} {'Offline':>10} {'Total':>10} {'Profit':>10}")
        print("-" * 82)

        for ch in sorted(result.combined_analysis, key=lambda x: -x.marginal_total):
            print(f"{ch.channel_name:<25} ${ch.current_spend:>11,.0f} "
                  f"${ch.marginal_online:>9.2f} ${ch.marginal_offline:>9.2f} "
                  f"${ch.marginal_total:>9.2f} ${ch.marginal_profit:>9.2f}")

        # Recommendations by view
        for view in ["online", "offline", "total", "profit"]:
            rec = result.view_recommendations[view]
            print("\n" + "=" * 100)
            print(f"VIEW: {rec.view_title}")
            print(f"Thresholds: INCREASE > ${rec.threshold_high:.2f}, REDUCE < ${rec.threshold_low:.2f}")
            print("=" * 100)

            print(f"\n{'Rank':<6} {'Channel':<25} {'Marginal ROI':>12} {'Action':>12}")
            print("-" * 60)

            for rank, ch, action in rec.ranked_channels:
                marginal = getattr(ch, f"marginal_{view}")
                action_str = f"{'+'if action=='INCREASE' else '-' if action=='REDUCE' else '='} {action}"
                print(f"{rank:<6} {ch.channel_name:<25} ${marginal:>11.2f} {action_str:>12}")

            print(f"\n   + Increase: {len(rec.increase_channels)} channels")
            print(f"   = Hold:     {len(rec.hold_channels)} channels")
            print(f"   - Reduce:   {len(rec.reduce_channels)} channels")

        # Recommendation comparison
        print("\n" + "=" * 100)
        print("RECOMMENDATION COMPARISON ACROSS VIEWS")
        print("=" * 100)

        print(f"\n{'Channel':<25} {'Online':>12} {'Offline':>12} {'Total':>12} {'Profit':>12}")
        print("-" * 78)

        for ch in sorted(result.combined_analysis, key=lambda x: -x.marginal_total):
            actions = []
            for view in ["online", "offline", "total", "profit"]:
                rec = result.view_recommendations[view]
                for rank, c, action in rec.ranked_channels:
                    if c.channel == ch.channel:
                        actions.append(f"{'+'if action=='INCREASE' else '-' if action=='REDUCE' else '='} {action}")
                        break

            print(f"{ch.channel_name:<25} {actions[0]:>12} {actions[1]:>12} {actions[2]:>12} {actions[3]:>12}")

        # Conflicts
        print("\n" + "-" * 78)
        print("CONFLICTING RECOMMENDATIONS:")

        if result.conflicting_channels:
            for c in result.conflicting_channels:
                print(f"   - {c}")
            print("\n   -> Decision depends on business objective!")
        else:
            print("   None - all views agree on recommendations")

    def get_summary_dict(
        self,
        result: CombinedModelResult,
    ) -> Dict[str, Any]:
        """
        Get combined analysis as a dictionary for JSON/UI.

        Parameters
        ----------
        result : CombinedModelResult
            Result from run_full_analysis()

        Returns
        -------
        dict
            Summary data
        """
        def channel_to_dict(ch: CombinedChannelAnalysis) -> dict:
            return {
                "channel": ch.channel,
                "channel_name": ch.channel_name,
                "current_spend": ch.current_spend,
                "marginal_online": ch.marginal_online,
                "marginal_offline": ch.marginal_offline,
                "marginal_total": ch.marginal_total,
                "marginal_profit": ch.marginal_profit,
                "current_roi_online": ch.current_roi_online,
                "current_roi_offline": ch.current_roi_offline,
                "current_roi_total": ch.current_roi_total,
                "current_roi_profit": ch.current_roi_profit,
                "headroom_online": ch.headroom_online,
                "headroom_offline": ch.headroom_offline,
            }

        views = {}
        for view, rec in result.view_recommendations.items():
            views[view] = {
                "title": rec.view_title,
                "threshold_high": rec.threshold_high,
                "threshold_low": rec.threshold_low,
                "increase": [channel_to_dict(ch) for ch in rec.increase_channels],
                "hold": [channel_to_dict(ch) for ch in rec.hold_channels],
                "reduce": [channel_to_dict(ch) for ch in rec.reduce_channels],
                "counts": {
                    "increase": len(rec.increase_channels),
                    "hold": len(rec.hold_channels),
                    "reduce": len(rec.reduce_channels),
                },
            }

        return {
            "margins": {
                "online": self.online_margin,
                "offline": self.offline_margin,
            },
            "channels": [channel_to_dict(ch) for ch in result.combined_analysis],
            "views": views,
            "conflicts": result.conflicting_channels,
        }

    @classmethod
    def from_mmm_wrappers(
        cls,
        mmm_wrapper_online: Any,
        mmm_wrapper_offline: Any,
        online_margin: float = 0.35,
        offline_margin: float = 0.25,
    ) -> Tuple["CombinedModelAnalyzer", CombinedModelResult]:
        """
        Create analyzer and run analysis from two MMMWrapper instances.

        Parameters
        ----------
        mmm_wrapper_online : MMMWrapper
            Fitted online model wrapper
        mmm_wrapper_offline : MMMWrapper
            Fitted offline model wrapper
        online_margin : float
            Online profit margin
        offline_margin : float
            Offline profit margin

        Returns
        -------
        tuple
            (CombinedModelAnalyzer, CombinedModelResult)
        """
        # Create marginal ROI analyzers for each model
        analyzer_online = MarginalROIAnalyzer.from_mmm_wrapper(mmm_wrapper_online)
        analyzer_offline = MarginalROIAnalyzer.from_mmm_wrapper(mmm_wrapper_offline)

        # Get channel analysis from each
        result_online = analyzer_online.run_full_analysis()
        result_offline = analyzer_offline.run_full_analysis()

        # Create combined analyzer
        combined_analyzer = cls(
            online_margin=online_margin,
            offline_margin=offline_margin,
        )

        # Run combined analysis
        result = combined_analyzer.run_full_analysis(
            channel_analysis_online=result_online.channel_analysis,
            channel_analysis_offline=result_offline.channel_analysis,
        )

        return combined_analyzer, result
