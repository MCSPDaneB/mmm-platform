"""
Tests for the OptimizationBridge class.

This module tests the bridge between MMMWrapper and PyMC-Marketing's
budget optimization, including historical spend retrieval, date range
handling, and channel bounds calculation.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, MagicMock, patch, PropertyMock
from datetime import datetime, timedelta

from mmm_platform.optimization.bridge import OptimizationBridge, UTILITY_FUNCTIONS


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_config():
    """Create a mock configuration."""
    config = Mock()
    config.data = Mock()
    config.data.date_column = "date"
    config.data.target_column = "revenue"
    config.data.spend_scale = 1000.0
    config.data.target_scale = 1000.0

    # Channels
    ch1 = Mock()
    ch1.name = "tv_spend"
    ch1.display_name = "TV"
    ch2 = Mock()
    ch2.name = "search_spend"
    ch2.display_name = "Search"
    config.channels = [ch1, ch2]

    # Owned media
    om1 = Mock()
    om1.name = "email_sends"
    om1.display_name = "Email"
    om1.include_roi = True
    config.owned_media = [om1]

    # Helper methods
    config.get_channel_columns = Mock(return_value=["tv_spend", "search_spend"])

    return config


@pytest.fixture
def sample_scaled_df():
    """Create sample scaled DataFrame (100 weeks of data)."""
    np.random.seed(42)
    n_rows = 100
    dates = pd.date_range(start="2022-01-01", periods=n_rows, freq="W")

    return pd.DataFrame({
        "date": dates,
        "revenue": np.random.uniform(50, 150, n_rows),  # Scaled values
        "tv_spend": np.random.uniform(5, 20, n_rows),
        "search_spend": np.random.uniform(3, 15, n_rows),
        "email_sends": np.random.uniform(1, 5, n_rows),
    })


@pytest.fixture
def sample_original_df():
    """Create sample original DataFrame (unscaled, 100 weeks)."""
    np.random.seed(42)
    n_rows = 100
    dates = pd.date_range(start="2022-01-01", periods=n_rows, freq="W")

    return pd.DataFrame({
        "date": dates,
        "revenue": np.random.uniform(50000, 150000, n_rows),  # Original units
        "tv_spend": np.random.uniform(5000, 20000, n_rows),
        "search_spend": np.random.uniform(3000, 15000, n_rows),
        "email_sends": np.random.uniform(1000, 5000, n_rows),
    })


@pytest.fixture
def mock_mmm():
    """Create mock MMM object."""
    mmm = Mock()
    mmm.optimize_budget = Mock()
    return mmm


@pytest.fixture
def mock_transform_engine():
    """Create mock transform engine."""
    engine = Mock()
    engine.get_effective_channel_columns = Mock(
        return_value=["tv_spend", "search_spend", "email_sends"]
    )
    return engine


@pytest.fixture
def mock_fitted_wrapper(mock_config, sample_scaled_df, sample_original_df, mock_mmm, mock_transform_engine):
    """Create a mock fitted MMMWrapper."""
    wrapper = Mock()
    wrapper.config = mock_config
    wrapper.df_scaled = sample_scaled_df
    wrapper.df_original = sample_original_df
    wrapper.mmm = mock_mmm
    wrapper.idata = Mock()  # Non-None to indicate fitted
    wrapper.transform_engine = mock_transform_engine
    return wrapper


@pytest.fixture
def bridge(mock_fitted_wrapper):
    """Create OptimizationBridge instance."""
    return OptimizationBridge(mock_fitted_wrapper)


# =============================================================================
# Initialization Tests
# =============================================================================

class TestOptimizationBridgeInit:
    """Tests for OptimizationBridge initialization."""

    def test_requires_fitted_wrapper(self, mock_config, sample_scaled_df, mock_mmm, mock_transform_engine):
        """Bridge raises error for unfitted wrapper (no idata)."""
        wrapper = Mock()
        wrapper.config = mock_config
        wrapper.df_scaled = sample_scaled_df
        wrapper.mmm = mock_mmm
        wrapper.idata = None  # Not fitted
        wrapper.transform_engine = mock_transform_engine

        with pytest.raises(ValueError, match="not fitted"):
            OptimizationBridge(wrapper)

    def test_requires_mmm_object(self, mock_config, sample_scaled_df, mock_transform_engine):
        """Bridge raises error when no MMM model."""
        wrapper = Mock()
        wrapper.config = mock_config
        wrapper.df_scaled = sample_scaled_df
        wrapper.mmm = None  # No MMM
        wrapper.idata = Mock()
        wrapper.transform_engine = mock_transform_engine

        with pytest.raises(ValueError, match="has no MMM model"):
            OptimizationBridge(wrapper)

    def test_init_extracts_channels(self, bridge):
        """Bridge extracts channel columns from transform engine."""
        channels = bridge.channel_columns
        assert "tv_spend" in channels
        assert "search_spend" in channels
        assert "email_sends" in channels

    def test_init_extracts_display_names(self, bridge):
        """Bridge extracts display name mapping."""
        names = bridge.channel_display_names
        assert names["tv_spend"] == "TV"
        assert names["search_spend"] == "Search"
        assert names["email_sends"] == "Email"


# =============================================================================
# Date Range Tests
# =============================================================================

class TestDateRangeMethods:
    """Tests for date range retrieval methods."""

    def test_get_available_date_range_returns_tuple(self, bridge):
        """get_available_date_range returns (min, max, count) tuple."""
        result = bridge.get_available_date_range()
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_get_available_date_range_correct_bounds(self, bridge, sample_original_df):
        """get_available_date_range returns correct date bounds."""
        min_date, max_date, _ = bridge.get_available_date_range()

        expected_min = pd.to_datetime(sample_original_df["date"].min())
        expected_max = pd.to_datetime(sample_original_df["date"].max())

        assert min_date == expected_min
        assert max_date == expected_max

    def test_get_available_date_range_period_count(self, bridge, sample_original_df):
        """get_available_date_range returns correct period count."""
        _, _, num_periods = bridge.get_available_date_range()
        assert num_periods == len(sample_original_df)

    def test_get_available_date_range_uses_df_original(self, bridge):
        """Prefers df_original over df_scaled for date range."""
        # df_original has 100 rows
        _, _, num_periods = bridge.get_available_date_range()
        assert num_periods == 100


# =============================================================================
# Historical Spend Tests
# =============================================================================

class TestHistoricalSpendMethods:
    """Tests for historical spend retrieval."""

    def test_get_historical_spend_returns_dict(self, bridge):
        """get_historical_spend returns dictionary."""
        result = bridge.get_historical_spend()
        assert isinstance(result, dict)

    def test_get_historical_spend_all_channels(self, bridge):
        """get_historical_spend includes all channel columns."""
        result = bridge.get_historical_spend()
        assert "tv_spend" in result
        assert "search_spend" in result
        assert "email_sends" in result

    def test_get_historical_spend_positive_values(self, bridge):
        """Historical spend values are positive."""
        result = bridge.get_historical_spend()
        for ch, spend in result.items():
            assert spend >= 0, f"{ch} has negative spend"

    def test_get_average_period_spend_returns_dict(self, bridge):
        """get_average_period_spend returns dictionary."""
        result = bridge.get_average_period_spend()
        assert isinstance(result, dict)

    def test_get_average_period_spend_positive_values(self, bridge):
        """Average period spend values are positive."""
        result = bridge.get_average_period_spend()
        for ch, spend in result.items():
            assert spend >= 0, f"{ch} has negative average spend"

    def test_get_average_period_spend_divides_by_periods(self, bridge, sample_scaled_df):
        """Average spend should be total/periods."""
        total = bridge.get_historical_spend()
        avg = bridge.get_average_period_spend()
        n_periods = len(sample_scaled_df)

        for ch in total:
            expected = total[ch] / n_periods
            assert abs(avg[ch] - expected) < 0.01, f"{ch} avg doesn't match"


class TestLastNWeeksSpend:
    """Tests for get_last_n_weeks_spend method."""

    def test_get_last_n_weeks_spend_returns_tuple(self, bridge):
        """get_last_n_weeks_spend returns (dict, start, end) tuple."""
        result = bridge.get_last_n_weeks_spend(n_weeks=8)
        assert isinstance(result, tuple)
        assert len(result) == 3
        assert isinstance(result[0], dict)
        assert isinstance(result[1], pd.Timestamp)
        assert isinstance(result[2], pd.Timestamp)

    def test_get_last_n_weeks_spend_date_range_correct(self, bridge):
        """Start date is approximately n_weeks before end date."""
        n_weeks = 8
        spend, start_date, end_date = bridge.get_last_n_weeks_spend(n_weeks=n_weeks)

        # End date should be max date in data
        _, max_date, _ = bridge.get_available_date_range()
        assert end_date == max_date

        # Start date should be approximately n_weeks before
        expected_start = max_date - pd.Timedelta(weeks=n_weeks)
        assert start_date == expected_start

    def test_get_last_n_weeks_spend_extrapolation(self, bridge):
        """Spend is extrapolated when num_periods specified."""
        # Get 8 weeks of data, extrapolate to 16 periods
        spend_8, _, _ = bridge.get_last_n_weeks_spend(n_weeks=8, num_periods=None)
        spend_16, _, _ = bridge.get_last_n_weeks_spend(n_weeks=8, num_periods=16)

        # Extrapolated spend should be larger than base (ratio depends on actual weeks found)
        for ch in spend_8:
            if spend_8[ch] > 0:
                ratio = spend_16[ch] / spend_8[ch]
                # Should extrapolate to higher value (ratio > 1)
                assert ratio > 1.0, f"{ch} extrapolation should increase spend, ratio={ratio}"
                # Should be less than 3x (reasonable extrapolation)
                assert ratio < 3.0, f"{ch} extrapolation too high, ratio={ratio}"

    def test_get_last_n_weeks_spend_insufficient_data_uses_available(self, bridge):
        """When requested weeks > available, uses all available data."""
        # Request 200 weeks but only 100 available
        spend, start_date, end_date = bridge.get_last_n_weeks_spend(n_weeks=200)

        # Should use earliest available date
        min_date, _, _ = bridge.get_available_date_range()
        assert start_date == min_date

    def test_get_last_n_weeks_spend_all_channels_present(self, bridge):
        """All channels are present in result."""
        spend, _, _ = bridge.get_last_n_weeks_spend(n_weeks=8)
        assert "tv_spend" in spend
        assert "search_spend" in spend
        assert "email_sends" in spend


class TestMostRecentMatchingPeriodSpend:
    """Tests for get_most_recent_matching_period_spend method."""

    def test_returns_tuple(self, bridge):
        """Returns (dict, start, end, actual_periods) tuple."""
        result = bridge.get_most_recent_matching_period_spend(num_periods=8)
        assert isinstance(result, tuple)
        assert len(result) == 4
        assert isinstance(result[0], dict)
        assert isinstance(result[3], int)

    def test_actual_weeks_returned(self, bridge):
        """Actual periods count is returned."""
        spend, _, _, actual_periods = bridge.get_most_recent_matching_period_spend(num_periods=8)
        assert actual_periods == 8

    def test_no_extrapolation(self, bridge, sample_original_df):
        """Values are actual spend, not extrapolated."""
        num_periods = 8
        spend, start_date, end_date, actual = bridge.get_most_recent_matching_period_spend(
            num_periods=num_periods
        )

        # Should be exactly num_periods
        assert actual == num_periods

        # Calculate expected spend manually from last 8 rows
        df_recent = sample_original_df.sort_values("date", ascending=False).head(num_periods)
        expected_tv = df_recent["tv_spend"].sum()

        assert abs(spend["tv_spend"] - expected_tv) < 0.01

    def test_handles_insufficient_data(self, bridge):
        """Returns what's available when insufficient data."""
        # Request more periods than available
        spend, _, _, actual = bridge.get_most_recent_matching_period_spend(num_periods=200)

        # Should return all available (100)
        assert actual == 100


class TestSpendForDateRange:
    """Tests for get_spend_for_date_range method."""

    def test_full_range_returns_all_spend(self, bridge):
        """Full date range returns total historical spend."""
        min_date, max_date, _ = bridge.get_available_date_range()

        range_spend = bridge.get_spend_for_date_range(
            start_date=min_date,
            end_date=max_date
        )
        total_spend = bridge.get_historical_spend()

        for ch in total_spend:
            assert abs(range_spend[ch] - total_spend[ch]) < 1.0

    def test_partial_range_returns_subset(self, bridge, sample_original_df):
        """Partial date range returns subset of spend."""
        # First 50 weeks
        dates = sample_original_df["date"]
        start = pd.to_datetime(dates.iloc[0])
        end = pd.to_datetime(dates.iloc[49])

        range_spend = bridge.get_spend_for_date_range(
            start_date=start,
            end_date=end
        )

        # Should be less than total
        total_spend = bridge.get_historical_spend()
        for ch in total_spend:
            assert range_spend[ch] < total_spend[ch]

    def test_extrapolation_works(self, bridge):
        """Extrapolation scales spend correctly."""
        min_date, max_date, _ = bridge.get_available_date_range()

        # Get 50 weeks, extrapolate to 100
        mid_date = min_date + pd.Timedelta(weeks=50)

        spend_50 = bridge.get_spend_for_date_range(
            start_date=min_date,
            end_date=mid_date,
            num_periods=None
        )
        spend_100 = bridge.get_spend_for_date_range(
            start_date=min_date,
            end_date=mid_date,
            num_periods=100
        )

        # Extrapolated should be approximately 2x
        for ch in spend_50:
            if spend_50[ch] > 0:
                ratio = spend_100[ch] / spend_50[ch]
                # ~50 rows extrapolated to 100 = ~2x
                assert 1.5 < ratio < 2.5


# =============================================================================
# Current Allocation Tests
# =============================================================================

class TestCurrentAllocation:
    """Tests for get_current_allocation method."""

    def test_average_mode(self, bridge):
        """Average mode returns average * num_periods."""
        result = bridge.get_current_allocation(
            comparison_mode="average",
            num_periods=8
        )

        avg = bridge.get_average_period_spend()
        for ch in avg:
            expected = avg[ch] * 8
            assert abs(result[ch] - expected) < 0.01

    def test_last_n_weeks_mode(self, bridge):
        """Last N weeks mode calls get_last_n_weeks_spend."""
        result = bridge.get_current_allocation(
            comparison_mode="last_n_weeks",
            num_periods=8,
            n_weeks=52
        )

        # Should match get_last_n_weeks_spend
        expected, _, _ = bridge.get_last_n_weeks_spend(n_weeks=52, num_periods=8)
        for ch in expected:
            assert abs(result[ch] - expected[ch]) < 0.01

    def test_most_recent_mode(self, bridge):
        """Most recent mode calls get_most_recent_matching_period_spend."""
        result = bridge.get_current_allocation(
            comparison_mode="most_recent_period",
            num_periods=8
        )

        expected, _, _, _ = bridge.get_most_recent_matching_period_spend(num_periods=8)
        for ch in expected:
            assert abs(result[ch] - expected[ch]) < 0.01

    def test_invalid_mode_raises(self, bridge):
        """Invalid comparison mode raises ValueError."""
        with pytest.raises(ValueError, match="Unknown comparison_mode"):
            bridge.get_current_allocation(
                comparison_mode="invalid_mode",
                num_periods=8
            )


# =============================================================================
# Channel Bounds Tests
# =============================================================================

class TestChannelBounds:
    """Tests for get_channel_bounds method."""

    def test_default_bounds(self, bridge):
        """Default bounds use min=0, max=2x average."""
        bounds = bridge.get_channel_bounds(
            min_multiplier=0.0,
            max_multiplier=2.0,
            reference="average"
        )

        assert isinstance(bounds, dict)
        avg = bridge.get_average_period_spend()

        for ch in avg:
            min_bound, max_bound = bounds[ch]
            assert min_bound == 0.0
            assert abs(max_bound - avg[ch] * 2) < 0.01

    def test_with_min_spend(self, bridge):
        """Min multiplier creates minimum spend floor."""
        bounds = bridge.get_channel_bounds(
            min_multiplier=0.5,
            max_multiplier=2.0,
            reference="average"
        )

        avg = bridge.get_average_period_spend()
        for ch in avg:
            min_bound, _ = bounds[ch]
            assert abs(min_bound - avg[ch] * 0.5) < 0.01

    def test_with_max_spend(self, bridge):
        """Max multiplier limits spend ceiling."""
        bounds = bridge.get_channel_bounds(
            min_multiplier=0.0,
            max_multiplier=3.0,
            reference="average"
        )

        avg = bridge.get_average_period_spend()
        for ch in avg:
            _, max_bound = bounds[ch]
            assert abs(max_bound - avg[ch] * 3) < 0.01


# =============================================================================
# Utility Function Tests
# =============================================================================

class TestUtilityFunction:
    """Tests for utility function retrieval."""

    def test_mean_utility(self, bridge):
        """Can get mean utility function."""
        func = bridge.get_utility_function("mean")
        assert callable(func)

    def test_var_utility(self, bridge):
        """Can get VaR utility function."""
        func = bridge.get_utility_function("var")
        assert callable(func)

    def test_cvar_utility(self, bridge):
        """Can get CVaR utility function."""
        func = bridge.get_utility_function("cvar")
        assert callable(func)

    def test_sharpe_utility(self, bridge):
        """Can get Sharpe utility function."""
        func = bridge.get_utility_function("sharpe")
        assert callable(func)

    def test_invalid_utility_raises(self, bridge):
        """Invalid utility name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown utility function"):
            bridge.get_utility_function("invalid_utility")

    def test_case_insensitive(self, bridge):
        """Utility names are case-insensitive."""
        func1 = bridge.get_utility_function("MEAN")
        func2 = bridge.get_utility_function("Mean")
        func3 = bridge.get_utility_function("mean")
        # All should return the same function
        assert func1 == func2 == func3


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_zero_spend_channel_handling(self, mock_fitted_wrapper):
        """Handles channels with zero spend."""
        # Set one channel to all zeros
        mock_fitted_wrapper.df_scaled["tv_spend"] = 0.0
        mock_fitted_wrapper.df_original["tv_spend"] = 0.0

        bridge = OptimizationBridge(mock_fitted_wrapper)

        spend = bridge.get_historical_spend()
        assert spend["tv_spend"] == 0.0

        avg = bridge.get_average_period_spend()
        assert avg["tv_spend"] == 0.0

    def test_short_data_handling(self, mock_config, mock_mmm, mock_transform_engine):
        """Handles very short datasets (< 10 rows)."""
        np.random.seed(42)
        n_rows = 5
        dates = pd.date_range(start="2022-01-01", periods=n_rows, freq="W")

        short_df = pd.DataFrame({
            "date": dates,
            "revenue": np.random.uniform(50, 150, n_rows),
            "tv_spend": np.random.uniform(5, 20, n_rows),
            "search_spend": np.random.uniform(3, 15, n_rows),
            "email_sends": np.random.uniform(1, 5, n_rows),
        })

        wrapper = Mock()
        wrapper.config = mock_config
        wrapper.df_scaled = short_df
        wrapper.df_original = None  # No original
        wrapper.mmm = mock_mmm
        wrapper.idata = Mock()
        wrapper.transform_engine = mock_transform_engine

        bridge = OptimizationBridge(wrapper)

        _, _, num_periods = bridge.get_available_date_range()
        assert num_periods == 5

        # Get last N weeks should handle limited data
        spend, _, _ = bridge.get_last_n_weeks_spend(n_weeks=10)
        assert len(spend) > 0

    def test_single_period_data(self, mock_config, mock_mmm, mock_transform_engine):
        """Handles single-period dataset."""
        single_df = pd.DataFrame({
            "date": [pd.Timestamp("2022-01-01")],
            "revenue": [100.0],
            "tv_spend": [10.0],
            "search_spend": [5.0],
            "email_sends": [2.0],
        })

        wrapper = Mock()
        wrapper.config = mock_config
        wrapper.df_scaled = single_df
        wrapper.df_original = None
        wrapper.mmm = mock_mmm
        wrapper.idata = Mock()
        wrapper.transform_engine = mock_transform_engine

        bridge = OptimizationBridge(wrapper)

        _, _, num_periods = bridge.get_available_date_range()
        assert num_periods == 1

        # Average should work
        avg = bridge.get_average_period_spend()
        # With spend_scale=1000, 10 * 1000 / 1 = 10000
        assert avg["tv_spend"] == 10.0 * 1000.0


# =============================================================================
# Optimizable Channels Tests
# =============================================================================

class TestOptimizableChannels:
    """Tests for get_optimizable_channels method."""

    def test_includes_paid_media(self, bridge):
        """All paid media channels are included."""
        channels = bridge.get_optimizable_channels()
        assert "tv_spend" in channels
        assert "search_spend" in channels

    def test_includes_owned_with_roi(self, bridge):
        """Owned media with include_roi=True is included."""
        channels = bridge.get_optimizable_channels()
        assert "email_sends" in channels

    def test_excludes_owned_without_roi(self, mock_fitted_wrapper):
        """Owned media with include_roi=False is excluded."""
        # Add owned media without ROI
        om_no_roi = Mock()
        om_no_roi.name = "direct_mail"
        om_no_roi.display_name = "Direct Mail"
        om_no_roi.include_roi = False
        mock_fitted_wrapper.config.owned_media.append(om_no_roi)

        bridge = OptimizationBridge(mock_fitted_wrapper)
        channels = bridge.get_optimizable_channels()

        assert "direct_mail" not in channels
