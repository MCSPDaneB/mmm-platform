"""
Time distribution patterns for budget allocation.

This module provides factory methods for creating time-phased
budget distribution patterns across optimization periods.
"""

import numpy as np
import xarray as xr
from typing import Sequence
import logging

logger = logging.getLogger(__name__)


class TimeDistribution:
    """
    Factory for time-phased budget distribution patterns.

    Time distributions specify how each channel's budget should be
    spread across the optimization periods. All distributions
    must sum to 1.0 along the time dimension.

    Examples
    --------
    >>> # Uniform distribution (equal spend each period)
    >>> dist = TimeDistribution.uniform(8, ["search", "display"])
    >>>
    >>> # Front-loaded (more spend early)
    >>> dist = TimeDistribution.front_loaded(8, ["search", "display"])
    """

    @staticmethod
    def uniform(
        num_periods: int,
        channels: Sequence[str],
    ) -> xr.DataArray:
        """
        Create uniform distribution (equal allocation per period).

        Parameters
        ----------
        num_periods : int
            Number of time periods.
        channels : Sequence[str]
            Channel column names.

        Returns
        -------
        xr.DataArray
            Distribution array with dims (channel, date).
        """
        # Each period gets 1/num_periods of the budget
        values = np.ones((len(channels), num_periods)) / num_periods

        return xr.DataArray(
            values,
            dims=["channel", "date"],
            coords={
                "channel": list(channels),
                "date": np.arange(num_periods),
            },
        )

    @staticmethod
    def front_loaded(
        num_periods: int,
        channels: Sequence[str],
        decay: float = 0.9,
    ) -> xr.DataArray:
        """
        Create front-loaded distribution (more spend early, tapering).

        Parameters
        ----------
        num_periods : int
            Number of time periods.
        channels : Sequence[str]
            Channel column names.
        decay : float
            Decay factor per period (0.9 = 10% less each period).
            Lower values = more aggressive front-loading.

        Returns
        -------
        xr.DataArray
            Distribution array with dims (channel, date).
        """
        # Geometric decay pattern
        pattern = np.array([decay ** i for i in range(num_periods)])
        # Normalize to sum to 1
        pattern = pattern / pattern.sum()

        # Same pattern for all channels
        values = np.tile(pattern, (len(channels), 1))

        return xr.DataArray(
            values,
            dims=["channel", "date"],
            coords={
                "channel": list(channels),
                "date": np.arange(num_periods),
            },
        )

    @staticmethod
    def back_loaded(
        num_periods: int,
        channels: Sequence[str],
        growth: float = 1.1,
    ) -> xr.DataArray:
        """
        Create back-loaded distribution (ramp up spend over time).

        Parameters
        ----------
        num_periods : int
            Number of time periods.
        channels : Sequence[str]
            Channel column names.
        growth : float
            Growth factor per period (1.1 = 10% more each period).
            Higher values = more aggressive back-loading.

        Returns
        -------
        xr.DataArray
            Distribution array with dims (channel, date).
        """
        # Geometric growth pattern
        pattern = np.array([growth ** i for i in range(num_periods)])
        # Normalize to sum to 1
        pattern = pattern / pattern.sum()

        # Same pattern for all channels
        values = np.tile(pattern, (len(channels), 1))

        return xr.DataArray(
            values,
            dims=["channel", "date"],
            coords={
                "channel": list(channels),
                "date": np.arange(num_periods),
            },
        )

    @staticmethod
    def linear_ramp(
        num_periods: int,
        channels: Sequence[str],
        start_weight: float = 0.5,
        end_weight: float = 1.5,
    ) -> xr.DataArray:
        """
        Create linear ramp distribution.

        Parameters
        ----------
        num_periods : int
            Number of time periods.
        channels : Sequence[str]
            Channel column names.
        start_weight : float
            Relative weight for first period.
        end_weight : float
            Relative weight for last period.

        Returns
        -------
        xr.DataArray
            Distribution array with dims (channel, date).
        """
        # Linear interpolation from start to end weight
        pattern = np.linspace(start_weight, end_weight, num_periods)
        # Normalize to sum to 1
        pattern = pattern / pattern.sum()

        values = np.tile(pattern, (len(channels), 1))

        return xr.DataArray(
            values,
            dims=["channel", "date"],
            coords={
                "channel": list(channels),
                "date": np.arange(num_periods),
            },
        )

    @staticmethod
    def pulsed(
        num_periods: int,
        channels: Sequence[str],
        pulse_periods: Sequence[int],
        pulse_weight: float = 2.0,
    ) -> xr.DataArray:
        """
        Create pulsed distribution (higher spend in specific periods).

        Useful for campaigns with specific launch dates or events.

        Parameters
        ----------
        num_periods : int
            Number of time periods.
        channels : Sequence[str]
            Channel column names.
        pulse_periods : Sequence[int]
            List of period indices (0-based) with higher spend.
        pulse_weight : float
            Weight multiplier for pulse periods relative to base.

        Returns
        -------
        xr.DataArray
            Distribution array with dims (channel, date).
        """
        pattern = np.ones(num_periods)
        for i in pulse_periods:
            if 0 <= i < num_periods:
                pattern[i] = pulse_weight
        # Normalize to sum to 1
        pattern = pattern / pattern.sum()

        values = np.tile(pattern, (len(channels), 1))

        return xr.DataArray(
            values,
            dims=["channel", "date"],
            coords={
                "channel": list(channels),
                "date": np.arange(num_periods),
            },
        )

    @staticmethod
    def seasonal(
        num_periods: int,
        channels: Sequence[str],
        seasonal_weights: Sequence[float],
    ) -> xr.DataArray:
        """
        Create custom seasonal distribution.

        Parameters
        ----------
        num_periods : int
            Number of time periods.
        channels : Sequence[str]
            Channel column names.
        seasonal_weights : Sequence[float]
            Weights for each period (will be normalized to sum to 1).
            Must have length == num_periods.

        Returns
        -------
        xr.DataArray
            Distribution array with dims (channel, date).

        Raises
        ------
        ValueError
            If seasonal_weights length doesn't match num_periods.
        """
        if len(seasonal_weights) != num_periods:
            raise ValueError(
                f"seasonal_weights length ({len(seasonal_weights)}) "
                f"must match num_periods ({num_periods})"
            )

        pattern = np.array(seasonal_weights, dtype=float)
        # Ensure non-negative
        pattern = np.maximum(pattern, 0)
        # Normalize to sum to 1
        if pattern.sum() > 0:
            pattern = pattern / pattern.sum()
        else:
            # Fallback to uniform if all zeros
            pattern = np.ones(num_periods) / num_periods

        values = np.tile(pattern, (len(channels), 1))

        return xr.DataArray(
            values,
            dims=["channel", "date"],
            coords={
                "channel": list(channels),
                "date": np.arange(num_periods),
            },
        )

    @staticmethod
    def per_channel(
        num_periods: int,
        channel_patterns: dict[str, Sequence[float]],
    ) -> xr.DataArray:
        """
        Create different distribution patterns per channel.

        Parameters
        ----------
        num_periods : int
            Number of time periods.
        channel_patterns : dict
            {channel_name: weights} where weights is a sequence
            of length num_periods. Will be normalized.

        Returns
        -------
        xr.DataArray
            Distribution array with dims (channel, date).
        """
        channels = list(channel_patterns.keys())
        values = np.zeros((len(channels), num_periods))

        for i, (ch, weights) in enumerate(channel_patterns.items()):
            pattern = np.array(weights[:num_periods], dtype=float)
            # Pad if too short
            if len(pattern) < num_periods:
                pattern = np.pad(
                    pattern,
                    (0, num_periods - len(pattern)),
                    constant_values=pattern[-1] if len(pattern) > 0 else 1,
                )
            # Normalize
            pattern = np.maximum(pattern, 0)
            if pattern.sum() > 0:
                pattern = pattern / pattern.sum()
            else:
                pattern = np.ones(num_periods) / num_periods

            values[i, :] = pattern

        return xr.DataArray(
            values,
            dims=["channel", "date"],
            coords={
                "channel": channels,
                "date": np.arange(num_periods),
            },
        )


def validate_time_distribution(
    distribution: xr.DataArray,
    num_periods: int,
    channels: Sequence[str],
) -> tuple[bool, str]:
    """
    Validate a time distribution array.

    Parameters
    ----------
    distribution : xr.DataArray
        Distribution to validate.
    num_periods : int
        Expected number of periods.
    channels : Sequence[str]
        Expected channel names.

    Returns
    -------
    tuple[bool, str]
        (is_valid, error_message)
    """
    # Check dimensions
    if "channel" not in distribution.dims or "date" not in distribution.dims:
        return False, "Distribution must have 'channel' and 'date' dimensions"

    # Check date length
    if len(distribution.coords["date"]) != num_periods:
        return False, (
            f"Date dimension length ({len(distribution.coords['date'])}) "
            f"doesn't match num_periods ({num_periods})"
        )

    # Check channels
    dist_channels = set(distribution.coords["channel"].values)
    expected_channels = set(channels)
    if dist_channels != expected_channels:
        missing = expected_channels - dist_channels
        extra = dist_channels - expected_channels
        return False, f"Channel mismatch. Missing: {missing}, Extra: {extra}"

    # Check normalization (should sum to 1 along date axis)
    sums = distribution.sum(dim="date")
    if not np.allclose(sums.values, 1.0, rtol=1e-5):
        bad_channels = [
            str(ch) for ch in sums.coords["channel"].values
            if not np.isclose(sums.sel(channel=ch).values, 1.0, rtol=1e-5)
        ]
        return False, f"Distribution doesn't sum to 1 for channels: {bad_channels}"

    # Check non-negative
    if (distribution.values < 0).any():
        return False, "Distribution contains negative values"

    return True, ""
