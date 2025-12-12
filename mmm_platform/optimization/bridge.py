"""
Bridge between MMMWrapper and PyMC-Marketing's budget optimization.

This module provides the OptimizationBridge class that translates
the MMMWrapper to work with PyMC-Marketing's BudgetOptimizer.
"""

from typing import Any
import numpy as np
import pandas as pd
import xarray as xr
import logging

from pymc_marketing.mmm.utility import (
    average_response,
    value_at_risk,
    conditional_value_at_risk,
    sharpe_ratio,
)

logger = logging.getLogger(__name__)


# Mapping from user-friendly names to utility functions
UTILITY_FUNCTIONS = {
    "mean": average_response,
    "average": average_response,
    "var": value_at_risk,
    "value_at_risk": value_at_risk,
    "cvar": conditional_value_at_risk,
    "expected_shortfall": conditional_value_at_risk,
    "sharpe": sharpe_ratio,
    "sharpe_ratio": sharpe_ratio,
}


class OptimizationBridge:
    """
    Bridge between MMMWrapper and PyMC-Marketing BudgetOptimizer.

    This class extracts the necessary information from MMMWrapper
    and provides helper methods for budget optimization.

    Parameters
    ----------
    wrapper : MMMWrapper
        A fitted MMMWrapper instance.

    Raises
    ------
    ValueError
        If the wrapper is not fitted (no idata).
    """

    def __init__(self, wrapper: Any):
        """
        Initialize the optimization bridge.

        Parameters
        ----------
        wrapper : MMMWrapper
            A fitted MMMWrapper instance with idata.
        """
        if wrapper.idata is None:
            raise ValueError(
                "MMMWrapper is not fitted. Call wrapper.fit() before optimization."
            )
        if wrapper.mmm is None:
            raise ValueError(
                "MMMWrapper has no MMM model. Build the model first."
            )

        self.wrapper = wrapper
        self._validate_model()

    def _validate_model(self):
        """Validate that the model is ready for optimization."""
        # Check that the MMM object has the optimization method
        if not hasattr(self.wrapper.mmm, "optimize_budget"):
            raise ValueError(
                "The MMM model does not support optimize_budget(). "
                "Ensure you're using a compatible PyMC-Marketing version."
            )

        logger.info("OptimizationBridge initialized successfully")

    @property
    def mmm(self):
        """Get the underlying PyMC-Marketing MMM model."""
        return self.wrapper.mmm

    @property
    def config(self):
        """Get the model configuration."""
        return self.wrapper.config

    @property
    def channel_columns(self) -> list[str]:
        """Get the effective channel columns used in the model."""
        return self.wrapper.transform_engine.get_effective_channel_columns()

    @property
    def channel_display_names(self) -> dict[str, str]:
        """Get mapping from column names to display names."""
        display_names = {}

        # Paid media channels
        for ch in self.config.channels:
            display_names[ch.name] = ch.display_name or ch.name

        # Owned media
        for om in self.config.owned_media:
            display_names[om.name] = om.display_name or om.name

        return display_names

    def get_optimizable_channels(self) -> list[str]:
        """
        Get channels that can be optimized.

        Returns paid media channels + owned media with include_roi=True.
        Channels without ROI tracking (like DM, email with include_roi=False)
        are excluded from optimization.

        Returns
        -------
        list[str]
            List of channel column names that can be optimized.
        """
        # All paid media channels are always optimizable
        channels = list(self.config.get_channel_columns())

        # Only add owned media with include_roi=True
        for om in self.config.owned_media:
            if om.include_roi:
                channels.append(om.name)

        return channels

    def get_historical_spend(self) -> dict[str, float]:
        """
        Get total historical spend per channel.

        Returns
        -------
        dict[str, float]
            {channel_name: total_spend} in original units.
        """
        spend = {}
        spend_scale = self.config.data.spend_scale

        for ch in self.channel_columns:
            if ch in self.wrapper.df_scaled.columns:
                # Sum of scaled spend, converted to original units
                spend[ch] = float(self.wrapper.df_scaled[ch].sum() * spend_scale)

        return spend

    def get_average_period_spend(self) -> dict[str, float]:
        """
        Get average spend per period (week/day) per channel.

        Returns
        -------
        dict[str, float]
            {channel_name: avg_spend_per_period} in original units.
        """
        spend = {}
        spend_scale = self.config.data.spend_scale
        n_periods = len(self.wrapper.df_scaled)

        for ch in self.channel_columns:
            if ch in self.wrapper.df_scaled.columns:
                spend[ch] = float(
                    self.wrapper.df_scaled[ch].sum() * spend_scale / n_periods
                )

        return spend

    def get_channel_bounds(
        self,
        min_multiplier: float = 0.0,
        max_multiplier: float = 2.0,
        reference: str = "average",
    ) -> dict[str, tuple[float, float]]:
        """
        Generate channel bounds based on historical spend.

        Parameters
        ----------
        min_multiplier : float
            Minimum spend as multiplier of reference (0 = allow zero spend).
        max_multiplier : float
            Maximum spend as multiplier of reference.
        reference : str
            Reference for bounds: "average" (per period) or "total" (historical).

        Returns
        -------
        dict[str, tuple[float, float]]
            {channel_name: (min_bound, max_bound)} in original units.
        """
        if reference == "average":
            ref_spend = self.get_average_period_spend()
        else:
            ref_spend = self.get_historical_spend()

        bounds = {}
        for ch, spend in ref_spend.items():
            min_val = spend * min_multiplier
            max_val = spend * max_multiplier
            # Ensure max > min
            if max_val <= min_val:
                max_val = min_val + 1.0
            bounds[ch] = (min_val, max_val)

        return bounds

    def get_current_allocation(
        self,
        num_periods: int,
        comparison_mode: str = "average",
        n_weeks: int | None = None,
    ) -> dict[str, float]:
        """
        Get the implied current allocation for comparison.

        Parameters
        ----------
        num_periods : int
            Number of periods in the optimization horizon.
        comparison_mode : str
            How to calculate baseline:
            - "average": Average spend per period × num_periods (default)
            - "last_n_weeks": Actual spend from last N weeks, extrapolated to num_periods
            - "most_recent_period": Actual spend from most recent num_periods weeks
        n_weeks : int, optional
            Number of weeks for "last_n_weeks" mode.

        Returns
        -------
        dict[str, float]
            {channel_name: projected_spend} in original units.
        """
        if comparison_mode == "average":
            avg_spend = self.get_average_period_spend()
            return {ch: spend * num_periods for ch, spend in avg_spend.items()}

        elif comparison_mode == "last_n_weeks":
            if n_weeks is None:
                raise ValueError("n_weeks required for 'last_n_weeks' comparison mode")
            spend, _, _ = self.get_last_n_weeks_spend(n_weeks)  # No extrapolation - use raw actual values
            return spend

        elif comparison_mode == "most_recent_period":
            spend, _, _, _ = self.get_most_recent_matching_period_spend(num_periods)
            return spend

        else:
            raise ValueError(f"Unknown comparison_mode: {comparison_mode}")

    def get_available_date_range(self) -> tuple[pd.Timestamp, pd.Timestamp, int]:
        """
        Get the available date range in the historical data.

        Returns
        -------
        tuple[pd.Timestamp, pd.Timestamp, int]
            (min_date, max_date, num_periods) from the data.
        """
        # Prefer df_original (full unfiltered data), fallback to df_scaled
        df = getattr(self.wrapper, 'df_original', None)
        if df is None:
            df = self.wrapper.df_scaled

        date_col = self.config.data.date_column
        min_date = pd.to_datetime(df[date_col].min())
        max_date = pd.to_datetime(df[date_col].max())
        num_periods = len(df)

        return min_date, max_date, num_periods

    def get_spend_for_date_range(
        self,
        start_date: pd.Timestamp | str | None = None,
        end_date: pd.Timestamp | str | None = None,
        num_periods: int | None = None,
    ) -> dict[str, float]:
        """
        Get actual spend per channel for a specific date range.

        Parameters
        ----------
        start_date : pd.Timestamp or str, optional
            Start date (inclusive). If None, uses earliest date.
        end_date : pd.Timestamp or str, optional
            End date (inclusive). If None, uses latest date.
        num_periods : int, optional
            If provided and differs from actual periods in range,
            extrapolates the spend proportionally.

        Returns
        -------
        dict[str, float]
            {channel_name: total_spend} in original units for the date range.
        """
        # Prefer df_original (unfiltered, original units), fallback to df_scaled
        df = getattr(self.wrapper, 'df_original', None)
        use_original = df is not None
        if df is None:
            df = self.wrapper.df_scaled

        date_col = self.config.data.date_column
        spend_scale = self.config.data.spend_scale

        # Convert dates if strings
        if start_date is not None:
            start_date = pd.to_datetime(start_date)
        if end_date is not None:
            end_date = pd.to_datetime(end_date)

        # Filter by date range
        mask = pd.Series([True] * len(df), index=df.index)
        if start_date is not None:
            mask &= pd.to_datetime(df[date_col]) >= start_date
        if end_date is not None:
            mask &= pd.to_datetime(df[date_col]) <= end_date

        df_filtered = df[mask]

        if len(df_filtered) == 0:
            logger.warning(f"No data found for date range {start_date} to {end_date}")
            return {ch: 0.0 for ch in self.channel_columns}

        # Sum spend per channel
        spend = {}
        for ch in self.channel_columns:
            if ch in df_filtered.columns:
                raw_sum = float(df_filtered[ch].sum())
                # If using df_original, values are already in original units
                # If using df_scaled, multiply by spend_scale
                if use_original:
                    spend[ch] = raw_sum
                else:
                    spend[ch] = raw_sum * spend_scale
            else:
                spend[ch] = 0.0

        # Extrapolate if num_periods specified and different from actual
        if num_periods is not None:
            actual_periods = len(df_filtered)
            if actual_periods > 0 and actual_periods != num_periods:
                scale_factor = num_periods / actual_periods
                spend = {ch: val * scale_factor for ch, val in spend.items()}
                logger.info(
                    f"Extrapolated spend from {actual_periods} to {num_periods} periods "
                    f"(scale factor: {scale_factor:.2f})"
                )

        return spend

    def get_last_n_weeks_spend(
        self,
        n_weeks: int,
        num_periods: int | None = None,
    ) -> tuple[dict[str, float], pd.Timestamp, pd.Timestamp]:
        """
        Get actual spend from the last N weeks of historical data.

        Parameters
        ----------
        n_weeks : int
            Number of weeks to look back from the most recent date.
        num_periods : int, optional
            If provided, extrapolates total to match optimization horizon.

        Returns
        -------
        tuple[dict[str, float], pd.Timestamp, pd.Timestamp]
            (spend_dict, start_date, end_date) - spend per channel and
            the actual date range used.
        """
        df = getattr(self.wrapper, 'df_original', None)
        if df is None:
            df = self.wrapper.df_scaled

        date_col = self.config.data.date_column

        # Get date range
        max_date = pd.to_datetime(df[date_col].max())
        min_date = pd.to_datetime(df[date_col].min())

        # Calculate start date
        start_date = max_date - pd.Timedelta(weeks=n_weeks)

        # Warn if requested range exceeds available data
        available_weeks = (max_date - min_date).days // 7 + 1
        if n_weeks > available_weeks:
            logger.warning(
                f"Requested {n_weeks} weeks but only {available_weeks} weeks available. "
                f"Using all available data."
            )
            start_date = min_date

        # Get spend for this range
        spend = self.get_spend_for_date_range(
            start_date=start_date,
            end_date=max_date,
            num_periods=num_periods,
        )

        return spend, start_date, max_date

    def get_most_recent_matching_period_spend(
        self,
        num_periods: int,
    ) -> tuple[dict[str, float], pd.Timestamp, pd.Timestamp, int]:
        """
        Get actual spend from the most recent period matching the optimization horizon.

        Takes the last `num_periods` rows of data and returns actual spend,
        without extrapolation.

        Parameters
        ----------
        num_periods : int
            Number of periods (weeks) matching the optimization horizon.

        Returns
        -------
        tuple[dict[str, float], pd.Timestamp, pd.Timestamp, int]
            (spend_dict, start_date, end_date, actual_periods) - spend per channel,
            the actual date range used, and number of periods found.
        """
        df = getattr(self.wrapper, 'df_original', None)
        use_original = df is not None
        if df is None:
            df = self.wrapper.df_scaled

        date_col = self.config.data.date_column
        spend_scale = self.config.data.spend_scale

        # Get the most recent N rows
        df_sorted = df.sort_values(date_col, ascending=False)
        df_recent = df_sorted.head(num_periods)

        actual_periods = len(df_recent)
        start_date = pd.to_datetime(df_recent[date_col].min())
        end_date = pd.to_datetime(df_recent[date_col].max())

        # Sum spend (no extrapolation - actual values only)
        spend = {}
        for ch in self.channel_columns:
            if ch in df_recent.columns:
                raw_sum = float(df_recent[ch].sum())
                if use_original:
                    spend[ch] = raw_sum
                else:
                    spend[ch] = raw_sum * spend_scale
            else:
                spend[ch] = 0.0

        if actual_periods < num_periods:
            logger.warning(
                f"Requested {num_periods} periods but only {actual_periods} available. "
                f"Returning actual spend without extrapolation."
            )

        return spend, start_date, end_date, actual_periods

    def get_utility_function(self, utility_name: str, **kwargs):
        """
        Get a utility function by name.

        Parameters
        ----------
        utility_name : str
            Name of utility function: "mean", "var", "cvar", "sharpe".
        **kwargs
            Additional arguments for the utility function.

        Returns
        -------
        callable
            The utility function.

        Raises
        ------
        ValueError
            If utility name is not recognized.
        """
        utility_name_lower = utility_name.lower()

        if utility_name_lower not in UTILITY_FUNCTIONS:
            valid = list(UTILITY_FUNCTIONS.keys())
            raise ValueError(
                f"Unknown utility function: {utility_name}. "
                f"Valid options: {valid}"
            )

        utility_fn = UTILITY_FUNCTIONS[utility_name_lower]

        # Some utility functions take parameters
        if utility_name_lower in ["var", "value_at_risk"]:
            confidence_level = kwargs.get("confidence_level", 0.95)
            return value_at_risk(confidence_level=confidence_level)
        elif utility_name_lower in ["cvar", "expected_shortfall"]:
            confidence_level = kwargs.get("confidence_level", 0.95)
            return conditional_value_at_risk(confidence_level=confidence_level)
        elif utility_name_lower in ["sharpe", "sharpe_ratio"]:
            risk_free_rate = kwargs.get("risk_free_rate", 0.0)
            return sharpe_ratio(risk_free_rate=risk_free_rate)
        else:
            return utility_fn

    def estimate_response_at_allocation(
        self,
        allocation: dict[str, float],
        num_periods: int,
    ) -> tuple[float, float, float]:
        """
        Estimate expected response at a given allocation.

        This is a simplified estimate using the ROI from the model.
        For accurate posterior-based estimates, use the full optimizer.

        Parameters
        ----------
        allocation : dict[str, float]
            {channel_name: spend} allocation to evaluate.
        num_periods : int
            Number of periods for the allocation.

        Returns
        -------
        tuple[float, float, float]
            (expected_response, ci_low, ci_high) estimates.
        """
        # Get ROI estimates from the model
        roi_df = self.wrapper.get_channel_roi()

        total_response = 0.0
        for ch, spend in allocation.items():
            ch_roi = roi_df[roi_df["channel"] == ch]["roi"].values
            if len(ch_roi) > 0:
                total_response += spend * ch_roi[0]

        # Simple CI estimate (±20% for now - real CI comes from optimizer)
        ci_low = total_response * 0.8
        ci_high = total_response * 1.2

        return total_response, ci_low, ci_high

    def get_contributions_for_period(
        self,
        num_periods: int,
        comparison_mode: str = "average",
        n_weeks: int | None = None,
    ) -> float:
        """
        Get actual total channel contributions for a historical period.

        Returns the sum of actual contributions (from model decomposition),
        not an estimate based on ROI. This provides a true apples-to-apples
        comparison with the optimizer's predicted response.

        Parameters
        ----------
        num_periods : int
            Number of periods in the optimization horizon.
        comparison_mode : str
            How to calculate baseline:
            - "average": Average contribution per period × num_periods
            - "last_n_weeks": Actual contributions from last N weeks
            - "most_recent_period": Actual contributions from most recent num_periods
        n_weeks : int, optional
            Number of weeks for "last_n_weeks" mode.

        Returns
        -------
        float
            Total channel contributions for the period in original units.
        """
        # Get contributions in real units
        contribs = self.wrapper.get_contributions_real_units()
        date_col = self.config.data.date_column
        df = self.wrapper.df_scaled

        # Get only channel columns (not controls/baseline)
        channel_cols = [c for c in self.channel_columns if c in contribs.columns]

        if comparison_mode == "average":
            # Average contribution per period × num_periods
            total_channel_contribs = contribs[channel_cols].sum().sum()
            avg_per_period = total_channel_contribs / len(df)
            return avg_per_period * num_periods

        elif comparison_mode == "last_n_weeks":
            if n_weeks is None:
                raise ValueError("n_weeks required for 'last_n_weeks' comparison mode")
            # Sum contributions from last n_weeks
            # Sort both df and contribs by date, then take tail
            sort_order = df[date_col].argsort()
            contribs_sorted = contribs.iloc[sort_order]
            return contribs_sorted.tail(n_weeks)[channel_cols].sum().sum()

        elif comparison_mode == "most_recent_period":
            # Sum contributions from most recent num_periods
            # Sort both df and contribs by date, then take tail
            sort_order = df[date_col].argsort()
            contribs_sorted = contribs.iloc[sort_order]
            return contribs_sorted.tail(num_periods)[channel_cols].sum().sum()

        else:
            raise ValueError(f"Unknown comparison_mode: {comparison_mode}")

    def get_scales(self) -> dict[str, float]:
        """
        Get the scaling factors used in the model.

        Returns
        -------
        dict[str, float]
            {"spend_scale": float, "revenue_scale": float}
        """
        return {
            "spend_scale": self.config.data.spend_scale,
            "revenue_scale": self.config.data.revenue_scale,
        }

    def to_display_allocation(
        self,
        allocation: dict[str, float] | xr.DataArray,
    ) -> dict[str, float]:
        """
        Convert allocation to display-friendly format with nice names.

        Parameters
        ----------
        allocation : dict or xr.DataArray
            Allocation from optimizer.

        Returns
        -------
        dict[str, float]
            {display_name: amount}
        """
        display_names = self.channel_display_names

        if isinstance(allocation, xr.DataArray):
            # Convert xarray to dict
            allocation_dict = {
                str(ch): float(allocation.sel(channel=ch).values)
                for ch in allocation.coords["channel"].values
            }
        else:
            allocation_dict = allocation

        result = {}
        for ch, amount in allocation_dict.items():
            display_name = display_names.get(ch, ch)
            result[display_name] = amount

        return result
