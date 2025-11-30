"""
Contribution analysis and decomposition.
"""

import numpy as np
import pandas as pd
from typing import Any, Optional, Dict
import logging

logger = logging.getLogger(__name__)


# Category color mapping for visualizations
CATEGORY_COLORS = {
    "BASELINE": "#808080",
    "Baseline": "#808080",
    "SEASONALITY": "#9370DB",
    "Seasonality": "#9370DB",
    "TREND": "#4B0082",
    "Trend": "#4B0082",
    "PAID MEDIA": "#4A90D9",
    "Paid Media": "#4A90D9",
    "Display": "#5DADE2",
    "Search": "#2ECC71",
    "Social": "#E74C3C",
    "Video": "#F39C12",
    "Email/DM": "#8E44AD",
    "Affiliate": "#1ABC9C",
    "Brand": "#3498DB",
    "Performance": "#E67E22",
    "PROMOTIONS": "#27AE60",
    "Promotions": "#27AE60",
    "MONTH EFFECTS": "#3498DB",
    "Month Effects": "#3498DB",
    "EVENTS/DUMMIES": "#E67E22",
    "Events/Dummies": "#E67E22",
    "Economic": "#95A5A6",
    "Competitor": "#C0392B",
    "Weather": "#16A085",
    "OTHER": "#7F8C8D",
    "Other": "#7F8C8D",
}


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

    def get_grouped_contributions(
        self,
        channel_categories: Optional[Dict[str, str]] = None,
        control_categories: Optional[Dict[str, str]] = None,
    ) -> pd.DataFrame:
        """
        Get contributions grouped by category.

        Parameters
        ----------
        channel_categories : dict, optional
            Mapping of channel column names to categories.
            If not provided, groups all channels under "PAID MEDIA".
        control_categories : dict, optional
            Mapping of control column names to categories.
            If not provided, uses pattern matching as fallback.

        Returns
        -------
        pd.DataFrame
            Contributions grouped by category.
        """
        component_cols = [c for c in self.contribs.columns if c != self.target_col]

        groups: Dict[str, list] = {}
        assigned = set()

        # 1. Baseline/intercept (always hardcoded - model structure)
        intercept_cols = [c for c in component_cols if "intercept" in c.lower()]
        if intercept_cols:
            groups["Baseline"] = intercept_cols
            assigned.update(intercept_cols)

        # 2. Seasonality (always hardcoded - model structure)
        seasonality_cols = [c for c in component_cols if "season" in c.lower() or "fourier" in c.lower()]
        if seasonality_cols:
            groups["Seasonality"] = seasonality_cols
            assigned.update(seasonality_cols)

        # 3. Trend (always hardcoded - model structure)
        trend_cols = [c for c in component_cols if c == "t"]
        if trend_cols:
            groups["Trend"] = trend_cols
            assigned.update(trend_cols)

        # 4. Group channels by provided categories
        if channel_categories:
            for col in self.channel_cols:
                if col in component_cols and col not in assigned:
                    category = channel_categories.get(col, "Paid Media")
                    if category not in groups:
                        groups[category] = []
                    groups[category].append(col)
                    assigned.add(col)
        else:
            # Fallback: all channels under "Paid Media"
            channel_cols_present = [c for c in self.channel_cols if c in component_cols and c not in assigned]
            if channel_cols_present:
                groups["Paid Media"] = channel_cols_present
                assigned.update(channel_cols_present)

        # 5. Group controls by provided categories
        if control_categories:
            for col in self.control_cols:
                if col in component_cols and col not in assigned:
                    category = control_categories.get(col, "Other")
                    if category not in groups:
                        groups[category] = []
                    groups[category].append(col)
                    assigned.add(col)
        else:
            # Fallback: pattern matching for controls
            for col in self.control_cols:
                if col in component_cols and col not in assigned:
                    col_lower = col.lower()
                    if "promo" in col_lower:
                        category = "Promotions"
                    elif "email" in col_lower or "_dm" in col_lower:
                        category = "Email/DM"
                    elif "month_" in col_lower:
                        category = "Month Effects"
                    elif "dummy_" in col_lower or "shock" in col_lower:
                        category = "Events/Dummies"
                    else:
                        category = "Other"

                    if category not in groups:
                        groups[category] = []
                    groups[category].append(col)
                    assigned.add(col)

        # 6. Anything remaining goes to "Other"
        remaining_cols = [c for c in component_cols if c not in assigned]
        if remaining_cols:
            if "Other" not in groups:
                groups["Other"] = []
            groups["Other"].extend(remaining_cols)

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
                "color": CATEGORY_COLORS.get(group_name, "#7F8C8D"),
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
