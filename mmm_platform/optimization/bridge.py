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

        Returns all effective channels (paid media + owned media with ROI).

        Returns
        -------
        list[str]
            List of channel column names.
        """
        return self.channel_columns

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

    def get_current_allocation(self, num_periods: int) -> dict[str, float]:
        """
        Get the implied current allocation for comparison.

        Calculates what the average spend per period would be,
        extrapolated to the optimization horizon.

        Parameters
        ----------
        num_periods : int
            Number of periods in the optimization horizon.

        Returns
        -------
        dict[str, float]
            {channel_name: projected_spend} in original units.
        """
        avg_spend = self.get_average_period_spend()
        return {ch: spend * num_periods for ch, spend in avg_spend.items()}

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
