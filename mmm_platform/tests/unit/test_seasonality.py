"""
Tests for the seasonal index calculator.

This module tests the SeasonalIndexCalculator class which computes
seasonal effectiveness indices from MMM results.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, MagicMock, PropertyMock
from datetime import datetime, timedelta

from mmm_platform.optimization.seasonality import SeasonalIndexCalculator


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_wrapper():
    """Create a mock MMMWrapper with required attributes for seasonality tests."""
    wrapper = Mock()

    # Set up config
    config = Mock()
    config.data.date_column = "date"
    config.data.spend_scale = 1000.0
    config.data.revenue_scale = 1000.0
    config.data.target_column = "revenue"
    config.channels = [
        Mock(name="tv_spend", get_display_name=Mock(return_value="TV")),
        Mock(name="search_spend", get_display_name=Mock(return_value="Search")),
    ]
    config.owned_media = []
    wrapper.config = config

    # Set up transform engine
    transform_engine = Mock()
    transform_engine.get_effective_channel_columns.return_value = ["tv_spend", "search_spend"]
    wrapper.transform_engine = transform_engine

    # Mock idata to indicate model is fitted
    wrapper.idata = Mock()

    return wrapper


@pytest.fixture
def sample_df_with_dates():
    """Create sample DataFrame with 2 years of weekly data."""
    np.random.seed(42)
    n_weeks = 104  # 2 years

    dates = pd.date_range(start="2022-01-01", periods=n_weeks, freq="W")

    # Create spend data with some seasonality
    tv_spend = 10000 + 3000 * np.sin(np.arange(n_weeks) * 2 * np.pi / 52) + np.random.normal(0, 500, n_weeks)
    search_spend = 8000 + 2000 * np.sin(np.arange(n_weeks) * 2 * np.pi / 52 + np.pi/4) + np.random.normal(0, 400, n_weeks)

    return pd.DataFrame({
        "date": dates,
        "tv_spend": np.maximum(tv_spend, 1000),  # Ensure positive spend
        "search_spend": np.maximum(search_spend, 500),
        "revenue": np.random.uniform(50000, 150000, n_weeks),
    })


@pytest.fixture
def sample_contributions(sample_df_with_dates):
    """Create sample contributions DataFrame that correlates with spend."""
    np.random.seed(42)
    n_rows = len(sample_df_with_dates)

    # Create contributions with seasonality (higher effectiveness in Q2/Q3 for TV)
    month = pd.to_datetime(sample_df_with_dates["date"]).dt.month
    tv_effectiveness = 1.0 + 0.3 * np.sin((month - 4) * 2 * np.pi / 12)  # Peaks in June
    search_effectiveness = 1.0 + 0.2 * np.sin((month - 10) * 2 * np.pi / 12)  # Peaks in October

    tv_contrib = sample_df_with_dates["tv_spend"] * tv_effectiveness * 1.5 / 1000  # scaled
    search_contrib = sample_df_with_dates["search_spend"] * search_effectiveness * 2.0 / 1000

    # Add small noise but ensure contributions stay positive
    tv_noise = np.random.normal(0, 100, n_rows)  # Small noise relative to signal
    search_noise = np.random.normal(0, 80, n_rows)

    return pd.DataFrame({
        "intercept": np.random.uniform(30000, 50000, n_rows),
        "tv_spend": np.maximum(tv_contrib + tv_noise, 100),  # Ensure positive
        "search_spend": np.maximum(search_contrib + search_noise, 50),  # Ensure positive
        "revenue": np.random.uniform(50000, 100000, n_rows),
    })


@pytest.fixture
def mock_wrapper_with_data(mock_wrapper, sample_df_with_dates, sample_contributions):
    """Mock wrapper with realistic data and contributions."""
    mock_wrapper.df_scaled = sample_df_with_dates.copy()
    mock_wrapper.df_original = sample_df_with_dates.copy()
    mock_wrapper.get_contributions.return_value = sample_contributions
    return mock_wrapper


@pytest.fixture
def seasonal_calculator(mock_wrapper_with_data):
    """Initialized SeasonalIndexCalculator."""
    return SeasonalIndexCalculator(mock_wrapper_with_data)


# =============================================================================
# Initialization Tests
# =============================================================================

class TestSeasonalIndexCalculatorInit:
    """Tests for SeasonalIndexCalculator initialization."""

    def test_init_with_fitted_wrapper(self, mock_wrapper_with_data):
        """Can initialize with a fitted wrapper."""
        calc = SeasonalIndexCalculator(mock_wrapper_with_data)
        assert calc.wrapper == mock_wrapper_with_data

    def test_init_fails_without_idata(self, mock_wrapper):
        """Raises error if wrapper is not fitted."""
        mock_wrapper.idata = None
        with pytest.raises(ValueError, match="not fitted"):
            SeasonalIndexCalculator(mock_wrapper)

    def test_channels_property(self, seasonal_calculator):
        """channels property returns correct channels."""
        channels = seasonal_calculator.channels
        assert "tv_spend" in channels
        assert "search_spend" in channels

    def test_date_column_property(self, seasonal_calculator):
        """date_column property returns correct column name."""
        assert seasonal_calculator.date_column == "date"

    def test_spend_scale_property(self, seasonal_calculator):
        """spend_scale property returns correct value."""
        assert seasonal_calculator.spend_scale == 1000.0


# =============================================================================
# Monthly Index Computation Tests
# =============================================================================

class TestMonthlyIndices:
    """Tests for monthly effectiveness index computation."""

    def test_compute_monthly_indices_shape(self, seasonal_calculator):
        """Returns DataFrame with channels × months."""
        indices = seasonal_calculator.compute_monthly_indices()

        assert isinstance(indices, pd.DataFrame)
        assert indices.shape == (2, 12)  # 2 channels, 12 months
        assert "tv_spend" in indices.index
        assert "search_spend" in indices.index

    def test_index_columns_are_months(self, seasonal_calculator):
        """Columns are month numbers 1-12."""
        indices = seasonal_calculator.compute_monthly_indices()
        assert list(indices.columns) == list(range(1, 13))

    def test_index_average_approximately_one(self, seasonal_calculator):
        """Average index across months should be approximately 1.0."""
        indices = seasonal_calculator.compute_monthly_indices()

        for channel in indices.index:
            avg_index = indices.loc[channel].mean()
            assert abs(avg_index - 1.0) < 0.1, f"{channel} average index is {avg_index}, expected ~1.0"

    def test_indices_are_positive(self, seasonal_calculator):
        """All indices should be positive."""
        indices = seasonal_calculator.compute_monthly_indices()
        assert (indices > 0).all().all()

    def test_indices_cached(self, seasonal_calculator):
        """Indices are cached after first computation."""
        indices1 = seasonal_calculator.compute_monthly_indices()
        indices2 = seasonal_calculator.compute_monthly_indices()

        pd.testing.assert_frame_equal(indices1, indices2)

    def test_min_observations_parameter(self, seasonal_calculator):
        """min_observations filters out months with insufficient data."""
        # With high min_observations, some months may default to 1.0
        indices_strict = seasonal_calculator._monthly_indices = None  # Clear cache
        indices = seasonal_calculator.compute_monthly_indices(min_observations=20)

        # Should still return valid indices
        assert indices.shape == (2, 12)


# =============================================================================
# Quarterly Index Computation Tests
# =============================================================================

class TestQuarterlyIndices:
    """Tests for quarterly effectiveness index computation."""

    def test_compute_quarterly_indices_shape(self, seasonal_calculator):
        """Returns DataFrame with channels × quarters."""
        indices = seasonal_calculator.compute_quarterly_indices()

        assert isinstance(indices, pd.DataFrame)
        assert indices.shape == (2, 4)  # 2 channels, 4 quarters

    def test_quarterly_columns_are_quarters(self, seasonal_calculator):
        """Columns are quarter numbers 1-4."""
        indices = seasonal_calculator.compute_quarterly_indices()
        assert list(indices.columns) == [1, 2, 3, 4]

    def test_quarterly_average_approximately_one(self, seasonal_calculator):
        """Average quarterly index should be approximately 1.0."""
        indices = seasonal_calculator.compute_quarterly_indices()

        for channel in indices.index:
            avg_index = indices.loc[channel].mean()
            assert abs(avg_index - 1.0) < 0.15, f"{channel} average index is {avg_index}"


# =============================================================================
# Period Selection Tests
# =============================================================================

class TestGetIndicesForPeriod:
    """Tests for get_indices_for_period method."""

    def test_single_month_returns_that_month(self, seasonal_calculator):
        """Single month returns that month's index."""
        indices = seasonal_calculator.get_indices_for_period(start_month=1, num_months=1)

        monthly = seasonal_calculator.compute_monthly_indices()

        for ch in indices:
            assert indices[ch] == monthly.loc[ch, 1]

    def test_multi_month_returns_average(self, seasonal_calculator):
        """Multi-month returns average of those months."""
        indices = seasonal_calculator.get_indices_for_period(start_month=1, num_months=3)

        monthly = seasonal_calculator.compute_monthly_indices()

        for ch in indices:
            expected = (monthly.loc[ch, 1] + monthly.loc[ch, 2] + monthly.loc[ch, 3]) / 3
            assert abs(indices[ch] - expected) < 0.001

    def test_wraps_around_year(self, seasonal_calculator):
        """Correctly wraps around Dec to Jan."""
        indices = seasonal_calculator.get_indices_for_period(start_month=11, num_months=4)

        monthly = seasonal_calculator.compute_monthly_indices()

        # Should include Nov(11), Dec(12), Jan(1), Feb(2)
        for ch in indices:
            expected = (monthly.loc[ch, 11] + monthly.loc[ch, 12] +
                       monthly.loc[ch, 1] + monthly.loc[ch, 2]) / 4
            assert abs(indices[ch] - expected) < 0.001

    def test_returns_dict(self, seasonal_calculator):
        """Returns dictionary with channel names as keys."""
        indices = seasonal_calculator.get_indices_for_period(start_month=6, num_months=2)

        assert isinstance(indices, dict)
        assert "tv_spend" in indices
        assert "search_spend" in indices

    def test_quarterly_mode(self, seasonal_calculator):
        """Can force quarterly indices."""
        indices = seasonal_calculator.get_indices_for_period(
            start_month=1, num_months=3, use_quarterly=True
        )

        # Should use Q1 index
        quarterly = seasonal_calculator.compute_quarterly_indices()
        for ch in indices:
            assert indices[ch] == quarterly.loc[ch, 1]


# =============================================================================
# Confidence Info Tests
# =============================================================================

class TestConfidenceInfo:
    """Tests for confidence information."""

    def test_get_confidence_info_returns_dict(self, seasonal_calculator):
        """get_confidence_info returns expected structure."""
        info = seasonal_calculator.get_confidence_info(start_month=1, num_months=3)

        assert "min_observations" in info
        assert "max_observations" in info
        assert "avg_observations" in info
        assert "using_quarterly" in info
        assert "confidence_level" in info

    def test_confidence_level_values(self, seasonal_calculator):
        """confidence_level is one of expected values."""
        info = seasonal_calculator.get_confidence_info(start_month=1, num_months=1)
        assert info["confidence_level"] in ["high", "medium", "low"]

    def test_get_observations_per_month(self, seasonal_calculator):
        """get_observations_per_month returns DataFrame."""
        # Compute indices first to populate observations
        seasonal_calculator.compute_monthly_indices()
        obs = seasonal_calculator.get_observations_per_month()

        assert isinstance(obs, pd.DataFrame)
        assert "n_observations" in obs.columns


# =============================================================================
# DataFrame Output Tests
# =============================================================================

class TestToDataFrame:
    """Tests for to_dataframe method."""

    def test_monthly_dataframe(self, seasonal_calculator):
        """to_dataframe returns formatted monthly table."""
        df = seasonal_calculator.to_dataframe(use_quarterly=False)

        assert "Display Name" in df.columns
        assert "Jan" in df.columns
        assert "Dec" in df.columns
        assert len(df) == 2  # 2 channels

    def test_quarterly_dataframe(self, seasonal_calculator):
        """to_dataframe returns formatted quarterly table."""
        df = seasonal_calculator.to_dataframe(use_quarterly=True)

        assert "Display Name" in df.columns
        assert "Q1" in df.columns
        assert "Q4" in df.columns


# =============================================================================
# Edge Cases Tests
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases."""

    def test_handles_zero_spend_months(self, mock_wrapper):
        """Handles months with zero spend gracefully."""
        np.random.seed(42)
        n_weeks = 52

        dates = pd.date_range(start="2022-01-01", periods=n_weeks, freq="W")

        # Create data where January has zero TV spend
        df = pd.DataFrame({
            "date": dates,
            "tv_spend": [0 if d.month == 1 else 10000 for d in dates],
            "search_spend": np.random.uniform(5000, 15000, n_weeks),
        })

        contribs = pd.DataFrame({
            "tv_spend": [0 if dates[i].month == 1 else 1000 for i in range(n_weeks)],
            "search_spend": np.random.uniform(500, 1500, n_weeks),
        })

        mock_wrapper.df_scaled = df
        mock_wrapper.df_original = df
        mock_wrapper.get_contributions.return_value = contribs

        calc = SeasonalIndexCalculator(mock_wrapper)
        indices = calc.compute_monthly_indices()

        # January should have index = 1.0 (default for zero spend)
        # or NaN handling should result in valid numbers
        assert not np.isnan(indices.loc["tv_spend", 1])

    def test_handles_missing_channels_in_contributions(self, mock_wrapper):
        """Handles channels not in contributions DataFrame."""
        np.random.seed(42)
        n_weeks = 52
        dates = pd.date_range(start="2022-01-01", periods=n_weeks, freq="W")

        df = pd.DataFrame({
            "date": dates,
            "tv_spend": np.random.uniform(5000, 15000, n_weeks),
            "search_spend": np.random.uniform(3000, 10000, n_weeks),
        })

        # Contributions missing search_spend
        contribs = pd.DataFrame({
            "tv_spend": np.random.uniform(500, 1500, n_weeks),
        })

        mock_wrapper.df_scaled = df
        mock_wrapper.df_original = df
        mock_wrapper.get_contributions.return_value = contribs

        calc = SeasonalIndexCalculator(mock_wrapper)
        indices = calc.compute_monthly_indices()

        # search_spend should have default index of 1.0
        assert indices.loc["search_spend"].mean() == 1.0

    def test_short_data_uses_quarterly(self, mock_wrapper):
        """With limited data (sparse observations), should recommend quarterly indices."""
        np.random.seed(42)
        n_months = 6  # Only 6 monthly data points

        # Monthly data over 6 months = ~1 obs per month, triggering quarterly mode
        dates = pd.date_range(start="2022-07-01", periods=n_months, freq="MS")

        df = pd.DataFrame({
            "date": dates,
            "tv_spend": np.random.uniform(5000, 15000, n_months),
            "search_spend": np.random.uniform(3000, 10000, n_months),
        })

        contribs = pd.DataFrame({
            "tv_spend": np.random.uniform(500, 1500, n_months),
            "search_spend": np.random.uniform(300, 1000, n_months),
        })

        mock_wrapper.df_scaled = df
        mock_wrapper.df_original = df
        mock_wrapper.get_contributions.return_value = contribs

        calc = SeasonalIndexCalculator(mock_wrapper)

        # Should use quarterly due to ~1 observation per month
        assert calc._should_use_quarterly() is True


# =============================================================================
# Summary Dict Tests
# =============================================================================

class TestSummaryDict:
    """Tests for get_summary_dict method."""

    def test_summary_dict_serializable(self, seasonal_calculator):
        """get_summary_dict returns JSON-serializable dict."""
        import json

        summary = seasonal_calculator.get_summary_dict()

        # Should not raise
        json_str = json.dumps(summary)
        assert isinstance(json_str, str)

    def test_summary_dict_contains_expected_keys(self, seasonal_calculator):
        """Summary dict has expected structure."""
        summary = seasonal_calculator.get_summary_dict()

        assert "monthly_indices" in summary
        assert "quarterly_indices" in summary
        assert "using_quarterly" in summary
        assert "n_channels" in summary
        assert "channels" in summary


# =============================================================================
# Demand Seasonality Tests
# =============================================================================

class TestDemandIndices:
    """Tests for demand seasonality calculations."""

    def test_compute_demand_indices_returns_dataframe(self, seasonal_calculator):
        """compute_demand_indices returns a DataFrame."""
        result = seasonal_calculator.compute_demand_indices()
        assert isinstance(result, pd.DataFrame)

    def test_demand_indices_has_12_months(self, seasonal_calculator):
        """Demand indices DataFrame covers all 12 months."""
        result = seasonal_calculator.compute_demand_indices()
        assert len(result) == 12
        assert list(result.index) == list(range(1, 13))

    def test_demand_indices_has_demand_index_column(self, seasonal_calculator):
        """DataFrame has 'demand_index' column."""
        result = seasonal_calculator.compute_demand_indices()
        assert "demand_index" in result.columns

    def test_demand_indices_average_approximately_one(self, seasonal_calculator):
        """Demand indices should average close to 1.0."""
        result = seasonal_calculator.compute_demand_indices()
        avg_index = result["demand_index"].mean()
        # Allow some tolerance due to missing months potentially filled with 1.0
        assert 0.9 < avg_index < 1.1

    def test_demand_indices_all_positive(self, seasonal_calculator):
        """All demand indices must be positive."""
        result = seasonal_calculator.compute_demand_indices()
        assert (result["demand_index"] > 0).all()

    def test_get_demand_index_for_period_returns_float(self, seasonal_calculator):
        """get_demand_index_for_period returns a float."""
        result = seasonal_calculator.get_demand_index_for_period(
            start_month=6,
            num_months=3,
        )
        assert isinstance(result, float)

    def test_get_demand_index_for_period_single_month(self, seasonal_calculator):
        """Single month returns that month's index."""
        full_indices = seasonal_calculator.compute_demand_indices()

        for month in range(1, 13):
            result = seasonal_calculator.get_demand_index_for_period(
                start_month=month,
                num_months=1,
            )
            expected = full_indices.loc[month, "demand_index"]
            assert abs(result - expected) < 0.001

    def test_get_demand_index_for_period_multi_month_average(self, seasonal_calculator):
        """Multi-month periods return weighted average."""
        full_indices = seasonal_calculator.compute_demand_indices()

        # Q2: April-June (months 4, 5, 6)
        result = seasonal_calculator.get_demand_index_for_period(
            start_month=4,
            num_months=3,
        )

        expected = (
            full_indices.loc[4, "demand_index"] +
            full_indices.loc[5, "demand_index"] +
            full_indices.loc[6, "demand_index"]
        ) / 3

        assert abs(result - expected) < 0.001

    def test_get_demand_index_for_period_wraps_around_year(self, seasonal_calculator):
        """Periods that span year boundary calculate correctly."""
        full_indices = seasonal_calculator.compute_demand_indices()

        # Nov-Jan (months 11, 12, 1)
        result = seasonal_calculator.get_demand_index_for_period(
            start_month=11,
            num_months=3,
        )

        expected = (
            full_indices.loc[11, "demand_index"] +
            full_indices.loc[12, "demand_index"] +
            full_indices.loc[1, "demand_index"]
        ) / 3

        assert abs(result - expected) < 0.001


class TestDemandIndicesEdgeCases:
    """Edge case tests for demand indices."""

    def test_missing_target_column_raises_error(self, mock_wrapper, sample_df_with_dates):
        """Error when target column not in data."""
        # Create wrapper with valid idata
        mock_wrapper.idata = Mock()
        mock_wrapper.df_original = sample_df_with_dates.drop(columns=["revenue"])
        mock_wrapper.config.data.target_column = "revenue"

        # Contributions mock
        mock_wrapper.get_contributions.return_value = pd.DataFrame()

        calc = SeasonalIndexCalculator(mock_wrapper)

        with pytest.raises(ValueError, match="Target column 'revenue' not found"):
            calc.compute_demand_indices()

    def test_handles_sparse_monthly_data(self, mock_wrapper):
        """Handles data that doesn't cover all months."""
        # Create sparse data with only a few months
        mock_wrapper.idata = Mock()
        dates = pd.date_range(start="2023-01-01", periods=20, freq="W")

        df = pd.DataFrame({
            "date": dates,
            "tv_spend": [10000] * 20,
            "search_spend": [8000] * 20,
            "revenue": [100000] * 20,
        })

        mock_wrapper.df_original = df
        mock_wrapper.get_contributions.return_value = pd.DataFrame()

        calc = SeasonalIndexCalculator(mock_wrapper)
        result = calc.compute_demand_indices()

        # Should have 12 rows (all months)
        assert len(result) == 12
        # Missing months should be filled with 1.0
        assert all(result["demand_index"].notna())
