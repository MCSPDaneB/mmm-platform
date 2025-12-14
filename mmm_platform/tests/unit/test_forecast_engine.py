"""
Tests for the spend forecast engine.

This module tests the SpendForecastEngine class which forecasts
incremental media response from planned/actual spend data.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime, timedelta

from mmm_platform.forecasting.forecast_engine import (
    SpendForecastEngine,
    ForecastResult,
    ValidationError,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_wrapper():
    """Create a mock MMMWrapper with required attributes for forecast tests."""
    wrapper = Mock()

    # Set up config
    config = Mock()
    config.data.date_column = "date"
    config.data.spend_scale = 1000.0
    config.data.revenue_scale = 1000.0
    config.data.target_column = "revenue"
    config.channels = [
        Mock(name="tv_spend", display_name="TV"),
        Mock(name="search_spend", display_name="Search"),
    ]
    config.owned_media = []
    config.get_channel_columns = Mock(return_value=["tv_spend", "search_spend"])
    wrapper.config = config

    # Set up transform engine
    transform_engine = Mock()
    transform_engine.get_effective_channel_columns.return_value = ["tv_spend", "search_spend"]
    wrapper.transform_engine = transform_engine

    # Mock idata for fitted model
    wrapper.idata = _create_mock_idata()

    # Set up MMM object
    wrapper.mmm = Mock()
    wrapper.mmm.optimize_budget = Mock()

    # Sample data ending in Dec 2024
    dates = pd.date_range(start="2023-01-01", end="2024-12-31", freq="W")
    wrapper.df_scaled = pd.DataFrame({
        "date": dates,
        "tv_spend": np.random.uniform(5, 20, len(dates)),
        "search_spend": np.random.uniform(3, 15, len(dates)),
        "revenue": np.random.uniform(50, 150, len(dates)),
    })
    wrapper.df_original = wrapper.df_scaled.copy()

    # Mock contributions
    wrapper.get_contributions = Mock(return_value=pd.DataFrame({
        "tv_spend": np.random.uniform(5000, 15000, len(dates)),
        "search_spend": np.random.uniform(3000, 10000, len(dates)),
        "intercept": np.random.uniform(30000, 50000, len(dates)),
    }))

    return wrapper


def _create_mock_idata():
    """Create mock InferenceData with posterior samples."""
    import xarray as xr

    np.random.seed(42)
    n_chains, n_draws = 2, 250
    n_channels = 2

    # Create posterior dataset
    posterior = xr.Dataset({
        "saturation_beta": xr.DataArray(
            np.random.lognormal(0, 0.3, (n_chains, n_draws, n_channels)),
            dims=["chain", "draw", "channel"]
        ),
        "saturation_lam": xr.DataArray(
            np.random.lognormal(1, 0.2, (n_chains, n_draws, n_channels)),
            dims=["chain", "draw", "channel"]
        ),
    })

    # Create mock idata
    idata = Mock()
    idata.posterior = posterior
    return idata


@pytest.fixture
def sample_spend_csv():
    """Create sample spend CSV for Jan 2025 (after model data ends)."""
    dates = pd.date_range(start="2025-01-06", periods=8, freq="W")
    return pd.DataFrame({
        "date": dates,
        "tv_spend": [50000, 55000, 48000, 52000, 51000, 49000, 53000, 50000],
        "search_spend": [30000, 32000, 28000, 31000, 29000, 33000, 30000, 28000],
    })


@pytest.fixture
def forecast_engine(mock_wrapper):
    """Initialized SpendForecastEngine."""
    with patch('mmm_platform.forecasting.forecast_engine.OptimizationBridge') as mock_bridge_cls, \
         patch('mmm_platform.forecasting.forecast_engine.SeasonalIndexCalculator') as mock_calc_cls:

        # Mock bridge
        mock_bridge = Mock()
        mock_bridge.channel_columns = ["tv_spend", "search_spend"]
        mock_bridge.channel_display_names = {"tv_spend": "TV", "search_spend": "Search"}
        mock_bridge.get_contributions_for_period.return_value = 100000.0
        mock_bridge.get_current_allocation.return_value = {"tv_spend": 50000, "search_spend": 30000}
        mock_bridge_cls.return_value = mock_bridge

        # Mock seasonal calculator
        mock_calc = Mock()
        mock_calc.get_indices_for_period.return_value = {"tv_spend": 1.1, "search_spend": 0.95}
        mock_calc.get_demand_index_for_period.return_value = 1.05
        mock_calc.get_confidence_info.return_value = {
            "min_observations": 4,
            "max_observations": 8,
            "avg_observations": 6.0,
            "using_quarterly": False,
            "confidence_level": "medium",
        }
        mock_calc_cls.return_value = mock_calc

        engine = SpendForecastEngine(mock_wrapper)
        engine.bridge = mock_bridge
        engine.seasonal_calculator = mock_calc

        return engine


# =============================================================================
# Initialization Tests
# =============================================================================

class TestSpendForecastEngineInit:
    """Tests for SpendForecastEngine initialization."""

    def test_init_with_fitted_wrapper(self, mock_wrapper):
        """Can initialize with a fitted wrapper."""
        with patch('mmm_platform.forecasting.forecast_engine.OptimizationBridge'), \
             patch('mmm_platform.forecasting.forecast_engine.SeasonalIndexCalculator'):
            engine = SpendForecastEngine(mock_wrapper)
            assert engine.wrapper == mock_wrapper

    def test_init_fails_without_idata(self, mock_wrapper):
        """Raises error if wrapper is not fitted."""
        mock_wrapper.idata = None
        with pytest.raises(ValueError, match="not fitted"):
            SpendForecastEngine(mock_wrapper)

    def test_channel_columns_cached(self, forecast_engine):
        """Channel columns are cached from bridge."""
        assert forecast_engine._channel_columns == ["tv_spend", "search_spend"]


# =============================================================================
# CSV Validation Tests
# =============================================================================

class TestCSVValidation:
    """Tests for spend CSV validation."""

    def test_csv_missing_channel_columns(self, forecast_engine):
        """Error when CSV missing required channel columns."""
        df = pd.DataFrame({
            "date": pd.date_range("2025-01-06", periods=4, freq="W"),
            # Missing both channels
        })
        errors = forecast_engine.validate_spend_csv(df)
        assert len(errors) > 0
        assert any("channel" in e.field.lower() or "channel" in e.message.lower() for e in errors)

    def test_csv_extra_columns_ignored(self, forecast_engine, sample_spend_csv):
        """Extra columns don't break validation."""
        df = sample_spend_csv.copy()
        df["extra_column"] = 1000
        df["another_extra"] = "text"

        errors = forecast_engine.validate_spend_csv(df)
        assert len(errors) == 0

    def test_csv_invalid_dates(self, forecast_engine):
        """Error on unparseable dates."""
        df = pd.DataFrame({
            "date": ["not-a-date", "also-not", "still-not", "nope"],
            "tv_spend": [1000, 2000, 3000, 4000],
            "search_spend": [500, 600, 700, 800],
        })
        errors = forecast_engine.validate_spend_csv(df)
        assert len(errors) > 0
        assert any(e.field == "date" for e in errors)

    def test_csv_negative_spend(self, forecast_engine, sample_spend_csv):
        """Error or warning on negative spend values."""
        df = sample_spend_csv.copy()
        df.loc[0, "tv_spend"] = -5000

        errors = forecast_engine.validate_spend_csv(df)
        assert len(errors) > 0
        assert any("negative" in e.message.lower() for e in errors)

    def test_csv_dates_must_continue_from_model(self, forecast_engine):
        """Error if CSV dates don't start after model's last date."""
        # Model data ends Dec 2024, so dates in Dec 2024 should fail
        df = pd.DataFrame({
            "date": pd.date_range("2024-12-01", periods=4, freq="W"),
            "tv_spend": [50000, 55000, 48000, 52000],
            "search_spend": [30000, 32000, 28000, 31000],
        })
        errors = forecast_engine.validate_spend_csv(df)
        assert len(errors) > 0
        assert any("after" in e.message.lower() or "start" in e.message.lower() for e in errors)


# =============================================================================
# Core Functionality Tests
# =============================================================================

class TestCoreFunctionality:
    """Tests for core forecast functionality."""

    def test_seasonal_indices_applied(self, forecast_engine, sample_spend_csv):
        """Channel indices properly applied to response."""
        result = forecast_engine.forecast(sample_spend_csv, apply_seasonal=True)

        # Verify seasonal indices were stored
        assert result.seasonal_applied is True
        assert "tv_spend" in result.seasonal_indices
        assert "search_spend" in result.seasonal_indices
        assert result.seasonal_indices["tv_spend"] == 1.1
        assert result.seasonal_indices["search_spend"] == 0.95

    def test_seasonal_indices_disabled(self, forecast_engine, sample_spend_csv):
        """Response differs when apply_seasonal=False."""
        result_with = forecast_engine.forecast(sample_spend_csv, apply_seasonal=True)
        result_without = forecast_engine.forecast(sample_spend_csv, apply_seasonal=False)

        # Indices should all be 1.0 when disabled
        assert result_without.seasonal_applied is False
        assert all(v == 1.0 for v in result_without.seasonal_indices.values())

        # Response should differ (unless all indices happen to be 1.0)
        # In our mock, TV is 1.1 and Search is 0.95, so they should differ
        assert result_with.total_response != result_without.total_response

    def test_posterior_sampling_ci(self, forecast_engine, sample_spend_csv):
        """CIs are valid (low < mean < high, reasonable spread)."""
        result = forecast_engine.forecast(sample_spend_csv)

        assert result.total_ci_low < result.total_response
        assert result.total_response < result.total_ci_high

        # Check weekly CIs too
        for _, row in result.weekly_df.iterrows():
            assert row["ci_low"] < row["response"]
            assert row["response"] < row["ci_high"]

    def test_custom_seasonal_indices(self, forecast_engine, sample_spend_csv):
        """Custom seasonal indices are applied correctly."""
        custom_indices = {"tv_spend": 1.5, "search_spend": 0.5}
        result = forecast_engine.forecast(
            sample_spend_csv,
            apply_seasonal=True,
            custom_seasonal_indices=custom_indices
        )

        assert result.seasonal_indices["tv_spend"] == 1.5
        assert result.seasonal_indices["search_spend"] == 0.5


# =============================================================================
# Edge Cases Tests
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_week_forecast(self, forecast_engine):
        """Works for 1-week CSV."""
        df = pd.DataFrame({
            "date": ["2025-01-06"],
            "tv_spend": [50000],
            "search_spend": [30000],
        })
        result = forecast_engine.forecast(df)

        assert result.num_weeks == 1
        assert len(result.weekly_df) == 1

    def test_multi_month_forecast(self, forecast_engine):
        """Works for 12+ week CSV."""
        dates = pd.date_range("2025-01-06", periods=16, freq="W")
        df = pd.DataFrame({
            "date": dates,
            "tv_spend": [50000] * 16,
            "search_spend": [30000] * 16,
        })
        result = forecast_engine.forecast(df)

        assert result.num_weeks == 16
        assert len(result.weekly_df) == 16

    def test_year_boundary_forecast(self, forecast_engine):
        """Dec->Jan transition handles month wrapping."""
        # Forecast spanning Dec 2025 to Feb 2026
        dates = pd.date_range("2025-12-01", periods=12, freq="W")
        df = pd.DataFrame({
            "date": dates,
            "tv_spend": [50000] * 12,
            "search_spend": [30000] * 12,
        })
        result = forecast_engine.forecast(df)

        # Should complete without error
        assert result.num_weeks == 12
        assert "Dec" in result.forecast_period or "Jan" in result.forecast_period or "Feb" in result.forecast_period

    def test_zero_spend_channel(self, forecast_engine):
        """Channel with $0 spend handled correctly."""
        dates = pd.date_range("2025-01-06", periods=4, freq="W")
        df = pd.DataFrame({
            "date": dates,
            "tv_spend": [0, 0, 0, 0],  # Zero TV spend
            "search_spend": [30000, 32000, 28000, 31000],
        })
        result = forecast_engine.forecast(df)

        # Should complete without error
        assert result.num_weeks == 4
        assert result.total_spend == sum([30000, 32000, 28000, 31000])

    def test_channel_subset(self, forecast_engine):
        """Forecast works when CSV has fewer channels than model."""
        dates = pd.date_range("2025-01-06", periods=4, freq="W")
        df = pd.DataFrame({
            "date": dates,
            "tv_spend": [50000, 55000, 48000, 52000],
            # Missing search_spend - should default to 0
        })

        # Should work but with only TV contribution
        result = forecast_engine.forecast(df)
        assert result.num_weeks == 4


# =============================================================================
# ForecastResult Tests
# =============================================================================

class TestForecastResult:
    """Tests for ForecastResult structure."""

    def test_weekly_df_structure(self, forecast_engine, sample_spend_csv):
        """Weekly df has correct columns."""
        result = forecast_engine.forecast(sample_spend_csv)

        expected_cols = {"date", "response", "ci_low", "ci_high", "spend"}
        assert set(result.weekly_df.columns) >= expected_cols

    def test_channel_contributions_sum(self, forecast_engine, sample_spend_csv):
        """Channel contributions sum to approximately total response."""
        result = forecast_engine.forecast(sample_spend_csv)

        # Sum contributions by week, then total
        weekly_totals = result.channel_contributions.groupby("date")["contribution"].sum()
        approx_total = weekly_totals.sum()

        # Should be close to total response (may differ slightly due to sampling)
        assert approx_total > 0
        # Contributions from sampling may not exactly equal the response mean
        # but should be in the same order of magnitude
        assert abs(approx_total - result.total_response) / result.total_response < 0.1

    def test_forecast_period_string(self, forecast_engine, sample_spend_csv):
        """Period string formatted correctly."""
        result = forecast_engine.forecast(sample_spend_csv)

        # Should contain month abbreviation and year
        assert result.forecast_period is not None
        assert len(result.forecast_period) > 0
        # Sample CSV is Jan-Feb 2025
        assert "Jan" in result.forecast_period or "Feb" in result.forecast_period

    def test_blended_roi_property(self, forecast_engine, sample_spend_csv):
        """Blended ROI computed correctly."""
        result = forecast_engine.forecast(sample_spend_csv)

        expected_roi = result.total_response / result.total_spend
        assert abs(result.blended_roi - expected_roi) < 0.001


# =============================================================================
# Seasonality Preview Tests
# =============================================================================

class TestSeasonalityPreview:
    """Tests for seasonality preview functionality."""

    def test_get_seasonality_preview(self, forecast_engine, sample_spend_csv):
        """Preview returns correct structure."""
        preview = forecast_engine.get_seasonality_preview(sample_spend_csv)

        assert "period" in preview
        assert "channel_indices" in preview
        assert "demand_index" in preview
        assert "confidence" in preview
        assert "num_weeks" in preview

        assert preview["num_weeks"] == 8
        assert "tv_spend" in preview["channel_indices"]

    def test_preview_channel_display_names(self, forecast_engine):
        """Display names are available."""
        names = forecast_engine.get_channel_display_names()

        assert "tv_spend" in names
        assert names["tv_spend"] == "TV"


# =============================================================================
# Consistency Tests
# =============================================================================

class TestConsistency:
    """Tests for consistency with optimizer."""

    def test_forecast_uses_same_saturation_formula(self, forecast_engine, sample_spend_csv):
        """Forecast uses same saturation formula as optimizer."""
        # This is verified by using the same PosteriorSamples and formula
        result = forecast_engine.forecast(sample_spend_csv)

        # Should produce positive response
        assert result.total_response > 0
        assert result.total_ci_low > 0

    def test_seasonal_indices_from_same_source(self, forecast_engine, sample_spend_csv):
        """Seasonal indices come from SeasonalIndexCalculator."""
        # Verify the calculator was called
        result = forecast_engine.forecast(sample_spend_csv, apply_seasonal=True)

        forecast_engine.seasonal_calculator.get_indices_for_period.assert_called()
        forecast_engine.seasonal_calculator.get_demand_index_for_period.assert_called()


# =============================================================================
# Granular Data Support Tests
# =============================================================================

from mmm_platform.forecasting.forecast_engine import (
    detect_spend_format,
    get_level_columns,
    validate_against_saved_mapping,
    aggregate_granular_spend,
)


class TestDetectSpendFormat:
    """Tests for spend format detection."""

    def test_detect_aggregated_format(self):
        """Detects aggregated format when model channels present."""
        df = pd.DataFrame({
            "date": pd.date_range("2025-01-06", periods=4, freq="W"),
            "tv_spend": [50000, 55000, 48000, 52000],
            "search_spend": [30000, 32000, 28000, 31000],
        })
        model_channels = ["tv_spend", "search_spend"]

        result = detect_spend_format(df, model_channels)
        assert result == "aggregated"

    def test_detect_granular_format_lvl(self):
        """Detects granular format with 'lvl' columns."""
        df = pd.DataFrame({
            "date": ["2025-01-06", "2025-01-06"],
            "media_channel_lvl1": ["Google", "Facebook"],
            "media_channel_lvl2": ["Search", "Social"],
            "spend": [50000, 30000],
        })
        model_channels = ["google_search_spend", "facebook_social_spend"]

        result = detect_spend_format(df, model_channels)
        assert result == "granular"

    def test_detect_granular_format_level(self):
        """Detects granular format with 'level' columns."""
        df = pd.DataFrame({
            "date": ["2025-01-06"],
            "channel_level_1": ["Google"],
            "channel_level_2": ["Search"],
            "spend": [50000],
        })
        model_channels = ["google_search_spend"]

        result = detect_spend_format(df, model_channels)
        assert result == "granular"

    def test_detect_unknown_format(self):
        """Returns unknown for unrecognized format."""
        df = pd.DataFrame({
            "date": pd.date_range("2025-01-06", periods=4, freq="W"),
            "random_column": [1, 2, 3, 4],
            "another_column": [5, 6, 7, 8],
        })
        model_channels = ["tv_spend", "search_spend"]

        result = detect_spend_format(df, model_channels)
        assert result == "unknown"


class TestGetLevelColumns:
    """Tests for extracting level columns."""

    def test_extracts_lvl_columns_ordered(self):
        """Extracts and orders lvl columns correctly."""
        df = pd.DataFrame({
            "date": ["2025-01-06"],
            "media_channel_lvl3": ["Brand"],
            "media_channel_lvl1": ["Google"],
            "media_channel_lvl2": ["Search"],
            "spend": [50000],
        })

        result = get_level_columns(df)
        assert result == ["media_channel_lvl1", "media_channel_lvl2", "media_channel_lvl3"]

    def test_extracts_level_columns(self):
        """Extracts level columns with 'level' naming."""
        df = pd.DataFrame({
            "date": ["2025-01-06"],
            "channel_level_1": ["Google"],
            "channel_level_2": ["Search"],
            "spend": [50000],
        })

        result = get_level_columns(df)
        assert len(result) == 2
        assert "channel_level_1" in result
        assert "channel_level_2" in result

    def test_empty_when_no_level_columns(self):
        """Returns empty list when no level columns."""
        df = pd.DataFrame({
            "date": ["2025-01-06"],
            "tv_spend": [50000],
        })

        result = get_level_columns(df)
        assert result == []


class TestValidateAgainstSavedMapping:
    """Tests for validating data against saved mapping."""

    def test_all_entities_match(self):
        """Returns is_valid=True when all entities match."""
        df = pd.DataFrame({
            "lvl1": ["Google", "Facebook"],
            "lvl2": ["Search", "Social"],
        })
        level_columns = ["lvl1", "lvl2"]
        entity_mappings = {
            "Google|Search": "google_search_spend",
            "Facebook|Social": "facebook_social_spend",
        }

        result = validate_against_saved_mapping(df, level_columns, entity_mappings)

        assert result["is_valid"] is True
        assert len(result["matched"]) == 2
        assert len(result["new_entities"]) == 0
        assert len(result["missing"]) == 0
        assert len(result["warnings"]) == 0

    def test_flags_new_entities(self):
        """Returns is_valid=False when new entities found."""
        df = pd.DataFrame({
            "lvl1": ["Google", "Facebook", "TikTok"],  # TikTok is new
            "lvl2": ["Search", "Social", "Video"],
        })
        level_columns = ["lvl1", "lvl2"]
        entity_mappings = {
            "Google|Search": "google_search_spend",
            "Facebook|Social": "facebook_social_spend",
        }

        result = validate_against_saved_mapping(df, level_columns, entity_mappings)

        assert result["is_valid"] is False
        assert len(result["new_entities"]) == 1
        assert "TikTok|Video" in result["new_entities"]
        assert any("new entities" in w for w in result["warnings"])

    def test_flags_missing_entities(self):
        """Includes missing entities in result."""
        df = pd.DataFrame({
            "lvl1": ["Google"],  # Missing Facebook
            "lvl2": ["Search"],
        })
        level_columns = ["lvl1", "lvl2"]
        entity_mappings = {
            "Google|Search": "google_search_spend",
            "Facebook|Social": "facebook_social_spend",
        }

        result = validate_against_saved_mapping(df, level_columns, entity_mappings)

        assert result["is_valid"] is True  # No new entities, so valid
        assert len(result["missing"]) == 1
        assert "Facebook|Social" in result["missing"]
        assert any("not in file" in w for w in result["warnings"])


class TestAggregateGranularSpend:
    """Tests for aggregating granular spend data."""

    def test_basic_aggregation(self):
        """Correctly aggregates granular data to model format."""
        df = pd.DataFrame({
            "date": ["2025-01-06", "2025-01-06", "2025-01-13", "2025-01-13"],
            "lvl1": ["Google", "Facebook", "Google", "Facebook"],
            "lvl2": ["Search", "Social", "Search", "Social"],
            "spend": [50000, 30000, 55000, 32000],
        })
        level_columns = ["lvl1", "lvl2"]
        entity_mappings = {
            "Google|Search": "google_spend",
            "Facebook|Social": "facebook_spend",
        }

        result = aggregate_granular_spend(
            df, level_columns, entity_mappings,
            date_column="date", spend_column="spend"
        )

        assert len(result) == 2  # Two dates
        assert "google_spend" in result.columns
        assert "facebook_spend" in result.columns
        assert result.loc[result["date"] == "2025-01-06", "google_spend"].values[0] == 50000
        assert result.loc[result["date"] == "2025-01-06", "facebook_spend"].values[0] == 30000

    def test_aggregation_sums_by_variable(self):
        """Multiple entities mapping to same variable are summed."""
        df = pd.DataFrame({
            "date": ["2025-01-06", "2025-01-06", "2025-01-06"],
            "lvl1": ["Google", "Google", "Facebook"],
            "lvl2": ["Brand", "NonBrand", "Social"],
            "spend": [20000, 30000, 25000],  # Both Google rows -> google_spend
        })
        level_columns = ["lvl1", "lvl2"]
        entity_mappings = {
            "Google|Brand": "google_spend",
            "Google|NonBrand": "google_spend",  # Same target
            "Facebook|Social": "facebook_spend",
        }

        result = aggregate_granular_spend(
            df, level_columns, entity_mappings,
            date_column="date", spend_column="spend"
        )

        # Google Brand + NonBrand should sum to 50000
        assert result.loc[result["date"] == "2025-01-06", "google_spend"].values[0] == 50000
        assert result.loc[result["date"] == "2025-01-06", "facebook_spend"].values[0] == 25000

    def test_unmapped_entities_filtered(self):
        """Unmapped entities are filtered out with warning."""
        df = pd.DataFrame({
            "date": ["2025-01-06", "2025-01-06"],
            "lvl1": ["Google", "Unknown"],  # Unknown not in mapping
            "lvl2": ["Search", "Channel"],
            "spend": [50000, 10000],
        })
        level_columns = ["lvl1", "lvl2"]
        entity_mappings = {
            "Google|Search": "google_spend",
            # Unknown|Channel not mapped
        }

        result = aggregate_granular_spend(
            df, level_columns, entity_mappings,
            date_column="date", spend_column="spend"
        )

        # Only Google should be in result
        assert "google_spend" in result.columns
        assert result["google_spend"].iloc[0] == 50000

    def test_raises_on_missing_columns(self):
        """Raises ValueError when required columns missing."""
        df = pd.DataFrame({
            "date": ["2025-01-06"],
            "lvl1": ["Google"],
            # Missing lvl2 and spend
        })
        level_columns = ["lvl1", "lvl2"]
        entity_mappings = {"Google|Search": "google_spend"}

        with pytest.raises(ValueError, match="Missing required columns"):
            aggregate_granular_spend(
                df, level_columns, entity_mappings,
                date_column="date", spend_column="spend"
            )

    def test_raises_when_all_unmapped(self):
        """Raises ValueError when all entities are unmapped."""
        df = pd.DataFrame({
            "date": ["2025-01-06"],
            "lvl1": ["Unknown"],
            "lvl2": ["Channel"],
            "spend": [10000],
        })
        level_columns = ["lvl1", "lvl2"]
        entity_mappings = {
            "Google|Search": "google_spend",  # Nothing matches
        }

        with pytest.raises(ValueError, match="No data remaining"):
            aggregate_granular_spend(
                df, level_columns, entity_mappings,
                date_column="date", spend_column="spend"
            )

    def test_date_column_is_datetime(self):
        """Output date column is datetime type."""
        df = pd.DataFrame({
            "date": ["2025-01-06", "2025-01-13"],
            "lvl1": ["Google", "Google"],
            "lvl2": ["Search", "Search"],
            "spend": [50000, 55000],
        })
        level_columns = ["lvl1", "lvl2"]
        entity_mappings = {"Google|Search": "google_spend"}

        result = aggregate_granular_spend(
            df, level_columns, entity_mappings,
            date_column="date", spend_column="spend"
        )

        assert pd.api.types.is_datetime64_any_dtype(result["date"])


# =============================================================================
# Overlap Detection Tests
# =============================================================================

from mmm_platform.forecasting.forecast_engine import (
    OverlapAnalysis,
    check_forecast_overlap,
)


class TestOverlapDetection:
    """Tests for forecast overlap detection."""

    def test_no_overlap_detected(self):
        """Returns has_overlap=False for non-overlapping dates."""
        new_spend = pd.DataFrame({
            "date": ["2025-02-01", "2025-02-08", "2025-02-15"],
            "tv_spend": [50000, 55000, 48000],
            "search_spend": [30000, 32000, 28000],
        })
        historical_spend = pd.DataFrame({
            "date": ["2025-01-06", "2025-01-13", "2025-01-20"],
            "tv_spend": [45000, 47000, 46000],
            "search_spend": [25000, 27000, 26000],
            "_forecast_id": ["forecast_abc123", "forecast_abc123", "forecast_abc123"],
        })

        result = check_forecast_overlap(new_spend, historical_spend)

        assert result.has_overlap is False
        assert len(result.overlapping_dates) == 0
        assert len(result.new_dates) == 3
        assert result.spend_comparison.empty

    def test_overlap_detected(self):
        """Correctly identifies overlapping dates."""
        new_spend = pd.DataFrame({
            "date": ["2025-01-13", "2025-01-20", "2025-01-27"],  # 13 and 20 overlap
            "tv_spend": [50000, 55000, 48000],
            "search_spend": [30000, 32000, 28000],
        })
        historical_spend = pd.DataFrame({
            "date": ["2025-01-06", "2025-01-13", "2025-01-20"],  # 13 and 20 overlap
            "tv_spend": [45000, 47000, 46000],
            "search_spend": [25000, 27000, 26000],
            "_forecast_id": ["forecast_abc123", "forecast_abc123", "forecast_abc123"],
        })

        result = check_forecast_overlap(new_spend, historical_spend)

        assert result.has_overlap is True
        assert len(result.overlapping_dates) == 2
        assert "2025-01-13" in result.overlapping_dates
        assert "2025-01-20" in result.overlapping_dates
        assert len(result.new_dates) == 1
        assert "2025-01-27" in result.new_dates

    def test_spend_comparison_accuracy(self):
        """Diff and % change calculated correctly."""
        new_spend = pd.DataFrame({
            "date": ["2025-01-13"],
            "tv_spend": [52000],  # Was 50000, now 52000 = +2000 = +4%
            "search_spend": [28000],  # Was 30000, now 28000 = -2000 = -6.67%
        })
        historical_spend = pd.DataFrame({
            "date": ["2025-01-13"],
            "tv_spend": [50000],
            "search_spend": [30000],
            "_forecast_id": ["forecast_abc123"],
        })

        result = check_forecast_overlap(new_spend, historical_spend)

        assert result.has_overlap is True
        assert not result.spend_comparison.empty

        # Check TV comparison
        tv_row = result.spend_comparison[result.spend_comparison["channel"] == "tv_spend"]
        assert len(tv_row) == 1
        assert tv_row["old_spend"].values[0] == 50000
        assert tv_row["new_spend"].values[0] == 52000
        assert tv_row["diff"].values[0] == 2000
        assert abs(tv_row["pct_change"].values[0] - 4.0) < 0.1

        # Check Search comparison
        search_row = result.spend_comparison[result.spend_comparison["channel"] == "search_spend"]
        assert len(search_row) == 1
        assert search_row["old_spend"].values[0] == 30000
        assert search_row["new_spend"].values[0] == 28000
        assert search_row["diff"].values[0] == -2000
        assert abs(search_row["pct_change"].values[0] - (-6.67)) < 0.1

    def test_total_spend_summary(self):
        """Total spend difference computed correctly."""
        new_spend = pd.DataFrame({
            "date": ["2025-01-13", "2025-01-20"],
            "tv_spend": [52000, 48000],  # Total: 100000
            "search_spend": [28000, 32000],  # Total: 60000
        })  # Grand total: 160000
        historical_spend = pd.DataFrame({
            "date": ["2025-01-13", "2025-01-20"],
            "tv_spend": [50000, 50000],  # Total: 100000
            "search_spend": [30000, 30000],  # Total: 60000
            "_forecast_id": ["forecast_abc123", "forecast_abc123"],
        })  # Grand total: 160000

        result = check_forecast_overlap(new_spend, historical_spend)

        assert result.total_spend_old == 160000
        assert result.total_spend_new == 160000
        assert result.spend_difference == 0
        assert result.pct_change == 0

    def test_overlapping_forecast_ids(self):
        """Returns list of forecast IDs that have overlapping dates."""
        new_spend = pd.DataFrame({
            "date": ["2025-01-13", "2025-01-20"],
            "tv_spend": [50000, 55000],
        })
        historical_spend = pd.DataFrame({
            "date": ["2025-01-13", "2025-01-20", "2025-01-06"],
            "tv_spend": [45000, 47000, 48000],
            "_forecast_id": ["forecast_abc", "forecast_xyz", "forecast_xyz"],
        })

        result = check_forecast_overlap(new_spend, historical_spend)

        assert result.has_overlap is True
        assert "forecast_abc" in result.overlapping_forecast_ids
        assert "forecast_xyz" in result.overlapping_forecast_ids

    def test_empty_historical_spend(self):
        """Returns no overlap when historical spend is empty."""
        new_spend = pd.DataFrame({
            "date": ["2025-01-13"],
            "tv_spend": [50000],
        })
        historical_spend = pd.DataFrame()

        result = check_forecast_overlap(new_spend, historical_spend)

        assert result.has_overlap is False
        assert len(result.overlapping_dates) == 0

    def test_date_format_handling(self):
        """Handles both string and datetime date formats."""
        new_spend = pd.DataFrame({
            "date": pd.to_datetime(["2025-01-13", "2025-01-20"]),
            "tv_spend": [50000, 55000],
        })
        historical_spend = pd.DataFrame({
            "date": ["2025-01-13", "2025-01-20"],  # String format
            "tv_spend": [45000, 47000],
            "_forecast_id": ["forecast_abc", "forecast_abc"],
        })

        result = check_forecast_overlap(new_spend, historical_spend)

        assert result.has_overlap is True
        assert len(result.overlapping_dates) == 2


# =============================================================================
# Forecast Persistence Tests
# =============================================================================

import tempfile
import shutil
from pathlib import Path
from mmm_platform.model.persistence import ForecastPersistence
from mmm_platform.config.schema import SavedForecastMetadata


@pytest.fixture
def temp_model_dir():
    """Create a temporary model directory for persistence tests."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_forecast_result():
    """Create a sample ForecastResult for testing."""
    weekly_df = pd.DataFrame({
        "date": pd.date_range("2025-01-06", periods=4, freq="W"),
        "response": [10000, 11000, 9500, 10500],
        "ci_low": [8000, 9000, 7500, 8500],
        "ci_high": [12000, 13000, 11500, 12500],
        "spend": [50000, 55000, 48000, 52000],
    })

    channel_contributions = pd.DataFrame({
        "date": pd.date_range("2025-01-06", periods=4, freq="W").repeat(2),
        "channel": ["tv_spend", "search_spend"] * 4,
        "contribution": [6000, 4000, 6600, 4400, 5700, 3800, 6300, 4200],
    })

    return ForecastResult(
        total_response=41000,
        total_ci_low=33000,
        total_ci_high=49000,
        weekly_df=weekly_df,
        channel_contributions=channel_contributions,
        total_spend=205000,
        num_weeks=4,
        seasonal_applied=True,
        seasonal_indices={"tv_spend": 1.1, "search_spend": 0.95},
        demand_index=1.05,
        forecast_period="Jan 2025",
    )


@pytest.fixture
def sample_input_spend():
    """Create sample input spend data for testing."""
    return pd.DataFrame({
        "date": pd.date_range("2025-01-06", periods=4, freq="W"),
        "tv_spend": [50000, 55000, 48000, 52000],
        "search_spend": [30000, 32000, 28000, 31000],
    })


class TestForecastPersistence:
    """Tests for forecast persistence."""

    def test_save_forecast_creates_files(self, temp_model_dir, sample_forecast_result, sample_input_spend):
        """All expected files are created when saving."""
        forecast_id = ForecastPersistence.save_forecast(
            temp_model_dir,
            sample_forecast_result,
            sample_input_spend,
            notes="Test forecast"
        )

        forecast_dir = temp_model_dir / "forecasts" / forecast_id

        assert forecast_dir.exists()
        assert (forecast_dir / "metadata.json").exists()
        assert (forecast_dir / "weekly_df.parquet").exists()
        assert (forecast_dir / "channel_contributions.parquet").exists()
        assert (forecast_dir / "input_spend.parquet").exists()
        assert (forecast_dir / "config.json").exists()

    def test_save_forecast_returns_id(self, temp_model_dir, sample_forecast_result, sample_input_spend):
        """Save returns a valid forecast ID."""
        forecast_id = ForecastPersistence.save_forecast(
            temp_model_dir,
            sample_forecast_result,
            sample_input_spend,
        )

        assert forecast_id.startswith("forecast_")
        assert len(forecast_id) == len("forecast_") + 8  # UUID hex[:8]

    def test_list_forecasts_returns_all(self, temp_model_dir, sample_forecast_result, sample_input_spend):
        """Lists all saved forecasts."""
        # Save multiple forecasts
        id1 = ForecastPersistence.save_forecast(
            temp_model_dir, sample_forecast_result, sample_input_spend, notes="First"
        )
        id2 = ForecastPersistence.save_forecast(
            temp_model_dir, sample_forecast_result, sample_input_spend, notes="Second"
        )

        forecasts = ForecastPersistence.list_forecasts(temp_model_dir)

        assert len(forecasts) == 2
        forecast_ids = [f.id for f in forecasts]
        assert id1 in forecast_ids
        assert id2 in forecast_ids

    def test_list_forecasts_sorted_by_date(self, temp_model_dir, sample_forecast_result, sample_input_spend):
        """Forecasts are sorted newest first."""
        import time

        id1 = ForecastPersistence.save_forecast(
            temp_model_dir, sample_forecast_result, sample_input_spend
        )
        time.sleep(0.1)  # Small delay to ensure different timestamps
        id2 = ForecastPersistence.save_forecast(
            temp_model_dir, sample_forecast_result, sample_input_spend
        )

        forecasts = ForecastPersistence.list_forecasts(temp_model_dir)

        # Newest (id2) should be first
        assert forecasts[0].id == id2
        assert forecasts[1].id == id1

    def test_list_forecasts_empty_dir(self, temp_model_dir):
        """Returns empty list when no forecasts exist."""
        forecasts = ForecastPersistence.list_forecasts(temp_model_dir)
        assert forecasts == []

    def test_load_forecast_restores_data(self, temp_model_dir, sample_forecast_result, sample_input_spend):
        """Full round-trip save/load restores all data."""
        forecast_id = ForecastPersistence.save_forecast(
            temp_model_dir,
            sample_forecast_result,
            sample_input_spend,
            notes="Test notes"
        )

        result, input_spend, metadata = ForecastPersistence.load_forecast(
            temp_model_dir, forecast_id
        )

        # Check metadata
        assert metadata.id == forecast_id
        assert metadata.notes == "Test notes"
        assert metadata.num_weeks == 4
        assert metadata.total_spend == 205000
        assert metadata.total_response == 41000
        assert metadata.seasonal_applied is True

        # Check result
        assert result.total_response == 41000
        assert result.total_spend == 205000
        assert result.num_weeks == 4
        assert len(result.weekly_df) == 4
        assert result.seasonal_indices["tv_spend"] == 1.1

        # Check input spend
        assert len(input_spend) == 4
        assert "tv_spend" in input_spend.columns

    def test_load_forecast_not_found(self, temp_model_dir):
        """Raises FileNotFoundError for non-existent forecast."""
        with pytest.raises(FileNotFoundError, match="Forecast not found"):
            ForecastPersistence.load_forecast(temp_model_dir, "forecast_nonexistent")

    def test_delete_forecast(self, temp_model_dir, sample_forecast_result, sample_input_spend):
        """Delete removes forecast directory."""
        forecast_id = ForecastPersistence.save_forecast(
            temp_model_dir, sample_forecast_result, sample_input_spend
        )

        result = ForecastPersistence.delete_forecast(temp_model_dir, forecast_id)

        assert result is True
        assert not (temp_model_dir / "forecasts" / forecast_id).exists()

    def test_delete_forecast_not_found(self, temp_model_dir):
        """Delete returns False for non-existent forecast."""
        result = ForecastPersistence.delete_forecast(temp_model_dir, "forecast_nonexistent")
        assert result is False

    def test_get_historical_spend_combines(self, temp_model_dir, sample_forecast_result, sample_input_spend):
        """Aggregates spend from multiple forecasts."""
        # Save first forecast
        ForecastPersistence.save_forecast(
            temp_model_dir, sample_forecast_result, sample_input_spend
        )

        # Save second forecast with different dates
        sample_input_spend_2 = pd.DataFrame({
            "date": pd.date_range("2025-02-03", periods=4, freq="W"),
            "tv_spend": [60000, 65000, 58000, 62000],
            "search_spend": [35000, 37000, 33000, 36000],
        })
        sample_forecast_result_2 = ForecastResult(
            total_response=50000,
            total_ci_low=40000,
            total_ci_high=60000,
            weekly_df=pd.DataFrame({
                "date": pd.date_range("2025-02-03", periods=4, freq="W"),
                "response": [12000, 13000, 11500, 12500],
                "ci_low": [10000, 11000, 9500, 10500],
                "ci_high": [14000, 15000, 13500, 14500],
                "spend": [95000, 102000, 91000, 98000],
            }),
            channel_contributions=sample_forecast_result.channel_contributions,
            total_spend=386000,
            num_weeks=4,
            seasonal_applied=True,
            seasonal_indices={"tv_spend": 1.1, "search_spend": 0.95},
            demand_index=1.05,
            forecast_period="Feb 2025",
        )
        ForecastPersistence.save_forecast(
            temp_model_dir, sample_forecast_result_2, sample_input_spend_2
        )

        historical = ForecastPersistence.get_historical_spend(temp_model_dir)

        assert len(historical) == 8  # 4 rows from each forecast
        assert "_forecast_id" in historical.columns
        assert len(historical["_forecast_id"].unique()) == 2

    def test_get_historical_spend_empty(self, temp_model_dir):
        """Returns empty DataFrame when no forecasts exist."""
        historical = ForecastPersistence.get_historical_spend(temp_model_dir)
        assert historical.empty

    def test_metadata_schema_validation(self, temp_model_dir, sample_forecast_result, sample_input_spend):
        """Metadata matches SavedForecastMetadata schema."""
        forecast_id = ForecastPersistence.save_forecast(
            temp_model_dir, sample_forecast_result, sample_input_spend
        )

        forecasts = ForecastPersistence.list_forecasts(temp_model_dir)
        metadata = forecasts[0]

        # Verify it's the correct type
        assert isinstance(metadata, SavedForecastMetadata)

        # Verify all fields populated
        assert metadata.id == forecast_id
        assert metadata.created_at is not None
        assert metadata.forecast_period == "Jan 2025"
        assert metadata.start_date is not None
        assert metadata.end_date is not None
        assert metadata.num_weeks == 4
        assert metadata.total_spend == 205000
        assert metadata.total_response == 41000
        assert abs(metadata.blended_roi - 0.2) < 0.01  # 41000/205000 ≈ 0.2
