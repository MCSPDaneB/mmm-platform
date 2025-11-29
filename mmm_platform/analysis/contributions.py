"""
Contribution analysis and decomposition.
"""

import numpy as np
import pandas as pd
from typing import Any, Optional
import logging

logger = logging.getLogger(__name__)


class ContributionAnalyzer:
    """
    Analyze and decompose model contributions.

    Provides:
    - Channel ROI calculations
    - Contribution breakdowns
    - Time-based analysis
    - Grouped contributions
    """

    def __init__(
        self,
        contribs: pd.DataFrame,
        df_scaled: pd.DataFrame,
        channel_cols: list[str],
        control_cols: list[str],
        target_col: str,
        date_col: str,
        revenue_scale: float = 1000.0,
        spend_scale: float = 1000.0,
    ):
        """
        Initialize ContributionAnalyzer.

        Parameters
        ----------
        contribs : pd.DataFrame
            Contribution dataframe from model.
        df_scaled : pd.DataFrame
            Scaled data used for modeling.
        channel_cols : list[str]
            Channel column names.
        control_cols : list[str]
            Control column names.
        target_col : str
            Target column name.
        date_col : str
            Date column name.
        revenue_scale : float
            Scale factor for revenue.
        spend_scale : float
            Scale factor for spend.
        """
        self.contribs = contribs
        self.df_scaled = df_scaled
        self.channel_cols = channel_cols
        self.control_cols = control_cols
        self.target_col = target_col
        self.date_col = date_col
        self.revenue_scale = revenue_scale
        self.spend_scale = spend_scale

    def get_channel_roi(self) -> pd.DataFrame:
        """
        Calculate ROI for each channel.

        Returns
        -------
        pd.DataFrame
            ROI by channel with spend and contribution details.
        """
        results = []

        for ch in self.channel_cols:
            if ch not in self.contribs.columns or ch not in self.df_scaled.columns:
                continue

            contrib = self.contribs[ch].sum()
            spend = self.df_scaled[ch].sum()
            roi = contrib / (spend + 1e-9)

            # Real units
            contrib_real = contrib * self.revenue_scale
            spend_real = spend * self.spend_scale

            results.append({
                "channel": ch,
                "contribution_scaled": float(contrib),
                "spend_scaled": float(spend),
                "roi": float(roi),
                "contribution_real": float(contrib_real),
                "spend_real": float(spend_real),
            })

        df = pd.DataFrame(results)
        if len(df) > 0:
            df = df.sort_values("roi", ascending=False)

        return df

    def get_contribution_breakdown(self) -> dict:
        """
        Get contribution breakdown by category.

        Returns
        -------
        dict
            Contribution totals by category.
        """
        component_cols = [c for c in self.contribs.columns if c != self.target_col]

        # Identify categories
        intercept_cols = [c for c in component_cols if "intercept" in c.lower()]
        channel_cols = [c for c in self.channel_cols if c in component_cols]
        control_cols = [c for c in self.control_cols if c in component_cols]
        seasonality_cols = [c for c in component_cols if "season" in c.lower()]

        # Calculate totals
        intercept_contrib = self.contribs[intercept_cols].sum().sum() if intercept_cols else 0
        channel_contrib = self.contribs[channel_cols].sum().sum() if channel_cols else 0
        control_contrib = self.contribs[control_cols].sum().sum() if control_cols else 0
        seasonality_contrib = self.contribs[seasonality_cols].sum().sum() if seasonality_cols else 0

        total = intercept_contrib + channel_contrib + control_contrib + seasonality_contrib

        return {
            "intercept": {
                "value": float(intercept_contrib),
                "real_value": float(intercept_contrib * self.revenue_scale),
                "pct": float(intercept_contrib / total * 100) if total != 0 else 0,
            },
            "channels": {
                "value": float(channel_contrib),
                "real_value": float(channel_contrib * self.revenue_scale),
                "pct": float(channel_contrib / total * 100) if total != 0 else 0,
            },
            "controls": {
                "value": float(control_contrib),
                "real_value": float(control_contrib * self.revenue_scale),
                "pct": float(control_contrib / total * 100) if total != 0 else 0,
            },
            "seasonality": {
                "value": float(seasonality_contrib),
                "real_value": float(seasonality_contrib * self.revenue_scale),
                "pct": float(seasonality_contrib / total * 100) if total != 0 else 0,
            },
            "total": {
                "value": float(total),
                "real_value": float(total * self.revenue_scale),
            },
        }

    def get_control_contributions(self) -> pd.DataFrame:
        """
        Get detailed control variable contributions.

        Returns
        -------
        pd.DataFrame
            Control contributions with sign validation.
        """
        results = []

        for ctrl in self.control_cols:
            if ctrl not in self.contribs.columns:
                continue

            contrib = self.contribs[ctrl].sum()
            is_inverted = "_inv" in ctrl

            # Determine expected sign
            if is_inverted:
                expected_sign = "negative"
                sign_valid = contrib < 0
            else:
                expected_sign = "positive"
                sign_valid = contrib > 0

            results.append({
                "control": ctrl,
                "contribution_scaled": float(contrib),
                "contribution_real": float(contrib * self.revenue_scale),
                "expected_sign": expected_sign,
                "actual_sign": "positive" if contrib > 0 else "negative",
                "sign_valid": sign_valid,
            })

        return pd.DataFrame(results)

    def get_grouped_contributions(self) -> pd.DataFrame:
        """
        Get contributions grouped by category.

        Groups:
        - Intercept/Baseline
        - Paid Media
        - Promotions
        - Email/DM
        - Seasonality
        - Trend
        - Other
        """
        component_cols = [c for c in self.contribs.columns if c != self.target_col]

        # Define groups by pattern matching
        groups = {
            "BASELINE": [c for c in component_cols if "intercept" in c.lower()],
            "PAID MEDIA": [c for c in self.channel_cols if c in component_cols],
            "PROMOTIONS": [c for c in component_cols if "promo" in c.lower()],
            "EMAIL/DM": [c for c in component_cols if "email" in c.lower() or "_dm" in c.lower()],
            "SEASONALITY": [c for c in component_cols if "season" in c.lower()],
            "TREND": [c for c in component_cols if c == "t"],
            "MONTH EFFECTS": [c for c in component_cols if "month_" in c.lower()],
            "DUMMIES/EVENTS": [c for c in component_cols if "dummy_" in c.lower() or "shock" in c.lower()],
        }

        # Calculate what's left as "OTHER"
        assigned = set()
        for cols in groups.values():
            assigned.update(cols)
        groups["OTHER"] = [c for c in component_cols if c not in assigned]

        # Calculate contributions using absolute values for percentage
        total_abs = self.contribs[component_cols].abs().sum().sum()

        results = []
        for group_name, cols in groups.items():
            if not cols:
                continue

            contrib = self.contribs[cols].sum().sum()
            abs_contrib = self.contribs[cols].abs().sum().sum()

            results.append({
                "group": group_name,
                "contribution_scaled": float(contrib),
                "contribution_real": float(contrib * self.revenue_scale),
                "abs_contribution": float(abs_contrib),
                "pct_of_total": float(abs_contrib / total_abs * 100) if total_abs > 0 else 0,
                "n_variables": len(cols),
            })

        df = pd.DataFrame(results)
        if len(df) > 0:
            df = df.sort_values("pct_of_total", ascending=False)

        return df

    def get_time_series_contributions(self) -> pd.DataFrame:
        """
        Get contributions as a time series.

        Returns
        -------
        pd.DataFrame
            Time series of contributions in real units.
        """
        contribs_real = self.contribs.copy()

        # Scale to real units
        for col in contribs_real.columns:
            if col != self.target_col:
                contribs_real[col] = contribs_real[col] * self.revenue_scale

        return contribs_real

    def get_channel_marginal_roi(
        self,
        idata: Optional[Any] = None,
    ) -> pd.DataFrame:
        """
        Get marginal ROI for each channel using saturation curve derivatives.

        For accurate marginal ROI calculation, use the MarginalROIAnalyzer class
        which requires the inference data (idata) from model fitting.

        Parameters
        ----------
        idata : InferenceData, optional
            ArviZ inference data. If not provided, returns average ROI with a warning.

        Returns
        -------
        pd.DataFrame
            Marginal ROI estimates with breakeven and headroom.
        """
        if idata is None:
            logger.warning(
                "Marginal ROI requires inference data. "
                "Use MarginalROIAnalyzer for full marginal ROI analysis. "
                "Returning average ROI as fallback."
            )
            return self.get_channel_roi()

        # Use the MarginalROIAnalyzer for proper calculation
        from .marginal_roi import MarginalROIAnalyzer

        analyzer = MarginalROIAnalyzer(
            idata=idata,
            df_scaled=self.df_scaled,
            contribs=self.contribs,
            channel_cols=self.channel_cols,
            target_col=self.target_col,
            spend_scale=self.spend_scale,
            revenue_scale=self.revenue_scale,
        )

        return analyzer.get_priority_table()

    @classmethod
    def from_mmm_wrapper(cls, mmm_wrapper: Any) -> "ContributionAnalyzer":
        """
        Create analyzer from an MMMWrapper instance.

        Parameters
        ----------
        mmm_wrapper : MMMWrapper
            Fitted model wrapper.

        Returns
        -------
        ContributionAnalyzer
            Analyzer instance.
        """
        return cls(
            contribs=mmm_wrapper.get_contributions(),
            df_scaled=mmm_wrapper.df_scaled,
            channel_cols=mmm_wrapper.config.get_channel_columns(),
            control_cols=mmm_wrapper.control_cols,
            target_col=mmm_wrapper.config.data.target_column,
            date_col=mmm_wrapper.config.data.date_column,
            revenue_scale=mmm_wrapper.config.data.revenue_scale,
            spend_scale=mmm_wrapper.config.data.spend_scale,
        )
