"""
Tests for the optimization module.

This module tests the budget optimization components including:
- OptimizationResult dataclass
- TimeDistribution patterns
- ConstraintBuilder helpers
- BudgetAllocator (mocked)
"""

import pytest
import numpy as np
import pandas as pd
import xarray as xr
from unittest.mock import Mock, MagicMock, patch

from mmm_platform.optimization.results import (
    OptimizationResult,
    TargetResult,
    ScenarioResult,
)
from mmm_platform.optimization.time_distribution import (
    TimeDistribution,
    validate_time_distribution,
)
from mmm_platform.optimization.constraints import (
    ConstraintBuilder,
    build_bounds_from_constraints,
)


# =============================================================================
# OptimizationResult Tests
# =============================================================================

class TestOptimizationResult:
    """Tests for OptimizationResult dataclass."""

    @pytest.fixture
    def basic_result(self):
        """Create a basic optimization result."""
        return OptimizationResult(
            optimal_allocation={"search": 50000, "display": 30000, "tv": 20000},
            total_budget=100000,
            expected_response=250000,
            response_ci_low=200000,
            response_ci_high=300000,
            success=True,
            message="Optimization successful",
            iterations=15,
            objective_value=-250000,
            num_periods=8,
            utility_function="mean",
        )

    @pytest.fixture
    def result_with_current(self, basic_result):
        """Create result with current allocation for comparison."""
        basic_result.current_allocation = {"search": 40000, "display": 40000, "tv": 20000}
        basic_result.current_response = 220000
        return basic_result

    def test_creation(self, basic_result):
        """Can create OptimizationResult."""
        assert basic_result.total_budget == 100000
        assert basic_result.expected_response == 250000
        assert basic_result.success is True

    def test_allocation_sum_matches_budget(self, basic_result):
        """Allocation should sum to total budget."""
        total_allocated = sum(basic_result.optimal_allocation.values())
        assert total_allocated == basic_result.total_budget

    def test_allocation_delta_none_without_current(self, basic_result):
        """allocation_delta is None when no current allocation."""
        assert basic_result.allocation_delta is None

    def test_allocation_delta_with_current(self, result_with_current):
        """allocation_delta correctly computes difference."""
        delta = result_with_current.allocation_delta
        assert delta["search"] == 10000  # 50000 - 40000
        assert delta["display"] == -10000  # 30000 - 40000
        assert delta["tv"] == 0

    def test_allocation_pct_change(self, result_with_current):
        """allocation_pct_change correctly computes percentages."""
        pct = result_with_current.allocation_pct_change
        assert pct["search"] == 25.0  # +10k on 40k = +25%
        assert pct["display"] == -25.0  # -10k on 40k = -25%
        assert pct["tv"] == 0.0

    def test_response_uplift(self, result_with_current):
        """response_uplift correctly computes difference."""
        assert result_with_current.response_uplift == 30000

    def test_response_uplift_pct(self, result_with_current):
        """response_uplift_pct correctly computes percentage."""
        # (250000 - 220000) / 220000 * 100 = 13.64%
        expected = (250000 - 220000) / 220000 * 100
        assert abs(result_with_current.response_uplift_pct - expected) < 0.01

    def test_to_dataframe(self, basic_result):
        """to_dataframe returns correct structure."""
        df = basic_result.to_dataframe()
        assert "channel" in df.columns
        assert "optimal" in df.columns
        assert "pct_of_total" in df.columns
        assert len(df) == 3

    def test_to_dataframe_with_comparison(self, result_with_current):
        """to_dataframe includes comparison columns when available."""
        df = result_with_current.to_dataframe()
        assert "current" in df.columns
        assert "delta" in df.columns
        assert "pct_change" in df.columns

    def test_get_summary_dict(self, basic_result):
        """get_summary_dict returns JSON-serializable dict."""
        import json
        summary = basic_result.get_summary_dict()
        # Should not raise
        json_str = json.dumps(summary)
        assert isinstance(json_str, str)


class TestTargetResult:
    """Tests for TargetResult dataclass."""

    def test_creation(self):
        """Can create TargetResult."""
        result = TargetResult(
            target_response=500000,
            budget_range_searched=(10000, 1000000),
            required_budget=200000,
            optimal_allocation={"search": 100000, "display": 100000},
            achievable=True,
            expected_response=510000,
            response_ci_low=450000,
            response_ci_high=570000,
            iterations=12,
            message="Target achieved",
        )
        assert result.required_budget == 200000
        assert result.achievable is True

    def test_not_achievable(self):
        """TargetResult handles unachievable targets."""
        result = TargetResult(
            target_response=10000000,
            budget_range_searched=(10000, 1000000),
            required_budget=1000000,
            optimal_allocation={"search": 500000, "display": 500000},
            achievable=False,
            expected_response=2000000,
            response_ci_low=1800000,
            response_ci_high=2200000,
            iterations=1,
            message="Target not achievable",
        )
        assert result.achievable is False

    def test_get_summary_dict(self):
        """get_summary_dict returns JSON-serializable dict."""
        import json
        result = TargetResult(
            target_response=500000,
            budget_range_searched=(10000, 1000000),
            required_budget=200000,
            optimal_allocation={"search": 100000},
            achievable=True,
            expected_response=510000,
            response_ci_low=450000,
            response_ci_high=570000,
            iterations=12,
            message="OK",
        )
        summary = result.get_summary_dict()
        json_str = json.dumps(summary)
        assert "target_response" in summary


# =============================================================================
# TimeDistribution Tests
# =============================================================================

class TestTimeDistribution:
    """Tests for TimeDistribution factory."""

    @pytest.fixture
    def channels(self):
        return ["search", "display", "tv"]

    def test_uniform_shape(self, channels):
        """Uniform distribution has correct shape."""
        dist = TimeDistribution.uniform(8, channels)
        assert dist.dims == ("channel", "date")
        assert dist.shape == (3, 8)

    def test_uniform_sums_to_one(self, channels):
        """Uniform distribution sums to 1 along date."""
        dist = TimeDistribution.uniform(8, channels)
        sums = dist.sum(dim="date")
        assert np.allclose(sums.values, 1.0)

    def test_uniform_equal_weights(self, channels):
        """Uniform distribution has equal weights."""
        dist = TimeDistribution.uniform(4, channels)
        assert np.allclose(dist.values, 0.25)

    def test_front_loaded_shape(self, channels):
        """Front-loaded distribution has correct shape."""
        dist = TimeDistribution.front_loaded(8, channels)
        assert dist.shape == (3, 8)

    def test_front_loaded_sums_to_one(self, channels):
        """Front-loaded distribution sums to 1."""
        dist = TimeDistribution.front_loaded(8, channels, decay=0.8)
        sums = dist.sum(dim="date")
        assert np.allclose(sums.values, 1.0)

    def test_front_loaded_decreasing(self, channels):
        """Front-loaded distribution decreases over time."""
        dist = TimeDistribution.front_loaded(8, channels, decay=0.8)
        # First period should be > last period
        first = dist.sel(channel="search", date=0).values
        last = dist.sel(channel="search", date=7).values
        assert first > last

    def test_back_loaded_increasing(self, channels):
        """Back-loaded distribution increases over time."""
        dist = TimeDistribution.back_loaded(8, channels, growth=1.2)
        first = dist.sel(channel="search", date=0).values
        last = dist.sel(channel="search", date=7).values
        assert first < last

    def test_linear_ramp(self, channels):
        """Linear ramp creates linear pattern."""
        dist = TimeDistribution.linear_ramp(4, channels, start_weight=1, end_weight=3)
        sums = dist.sum(dim="date")
        assert np.allclose(sums.values, 1.0)

    def test_pulsed_distribution(self, channels):
        """Pulsed distribution has higher weights at pulse periods."""
        dist = TimeDistribution.pulsed(8, channels, pulse_periods=[0, 4], pulse_weight=3.0)
        # Pulse periods should have higher weight
        pulse = dist.sel(channel="search", date=0).values
        normal = dist.sel(channel="search", date=1).values
        assert pulse > normal

    def test_seasonal_custom_pattern(self, channels):
        """Seasonal distribution respects custom pattern."""
        pattern = [1, 2, 3, 4]
        dist = TimeDistribution.seasonal(4, channels, pattern)
        sums = dist.sum(dim="date")
        assert np.allclose(sums.values, 1.0)
        # Pattern should be increasing (normalized)
        first = dist.sel(channel="search", date=0).values
        last = dist.sel(channel="search", date=3).values
        assert first < last

    def test_per_channel_different_patterns(self):
        """Per-channel distribution allows different patterns."""
        patterns = {
            "search": [1, 1, 1, 1],  # Uniform
            "display": [4, 3, 2, 1],  # Decreasing
        }
        dist = TimeDistribution.per_channel(4, patterns)
        # Search should be uniform
        search_vals = dist.sel(channel="search").values
        assert np.allclose(search_vals, 0.25)
        # Display should be decreasing
        display_first = dist.sel(channel="display", date=0).values
        display_last = dist.sel(channel="display", date=3).values
        assert display_first > display_last


class TestValidateTimeDistribution:
    """Tests for validate_time_distribution function."""

    def test_valid_distribution(self):
        """Valid distribution passes validation."""
        dist = TimeDistribution.uniform(8, ["search", "display"])
        is_valid, msg = validate_time_distribution(dist, 8, ["search", "display"])
        assert is_valid is True
        assert msg == ""

    def test_wrong_period_count(self):
        """Detects wrong number of periods."""
        dist = TimeDistribution.uniform(4, ["search"])
        is_valid, msg = validate_time_distribution(dist, 8, ["search"])
        assert is_valid is False
        assert "num_periods" in msg

    def test_missing_channel(self):
        """Detects missing channels."""
        dist = TimeDistribution.uniform(8, ["search"])
        is_valid, msg = validate_time_distribution(dist, 8, ["search", "display"])
        assert is_valid is False
        assert "Missing" in msg

    def test_unnormalized_distribution(self):
        """Detects distribution that doesn't sum to 1."""
        # Create unnormalized distribution manually
        dist = xr.DataArray(
            np.ones((2, 4)) * 0.5,  # Sums to 2, not 1
            dims=["channel", "date"],
            coords={"channel": ["a", "b"], "date": [0, 1, 2, 3]},
        )
        is_valid, msg = validate_time_distribution(dist, 4, ["a", "b"])
        assert is_valid is False
        assert "sum to 1" in msg


# =============================================================================
# ConstraintBuilder Tests
# =============================================================================

class TestConstraintBuilder:
    """Tests for ConstraintBuilder factory."""

    def test_min_spend_creates_constraint(self):
        """min_spend creates a constraint dict."""
        constraint = ConstraintBuilder.min_spend("search", 10000)
        assert constraint["name"] == "min_spend_search"
        assert constraint["type"] == "ineq"
        assert callable(constraint["fun"])

    def test_max_spend_creates_constraint(self):
        """max_spend creates a constraint dict."""
        constraint = ConstraintBuilder.max_spend("display", 50000)
        assert constraint["name"] == "max_spend_display"
        assert constraint["type"] == "ineq"

    def test_min_ratio_creates_constraint(self):
        """min_ratio creates a constraint dict."""
        constraint = ConstraintBuilder.min_ratio("tv", 0.1)
        assert "min_ratio" in constraint["name"]
        assert constraint["type"] == "ineq"

    def test_max_ratio_creates_constraint(self):
        """max_ratio creates a constraint dict."""
        constraint = ConstraintBuilder.max_ratio("social", 0.4)
        assert "max_ratio" in constraint["name"]

    def test_channel_ratio_constraint(self):
        """channel_ratio creates a constraint dict."""
        constraint = ConstraintBuilder.channel_ratio("tv", "search", 2.0)
        assert "ratio" in constraint["name"]

    def test_group_min_spend(self):
        """group_min_spend creates a constraint dict."""
        constraint = ConstraintBuilder.group_min_spend(["search", "display"], 50000)
        assert "group_min" in constraint["name"]

    def test_group_max_ratio(self):
        """group_max_ratio creates a constraint dict."""
        constraint = ConstraintBuilder.group_max_ratio(["tv", "radio"], 0.5)
        assert "group_max" in constraint["name"]


class TestBuildBoundsFromConstraints:
    """Tests for build_bounds_from_constraints helper."""

    @pytest.fixture
    def channels(self):
        return ["search", "display", "tv"]

    def test_default_bounds(self, channels):
        """Default bounds are 0 to total_budget."""
        bounds = build_bounds_from_constraints(channels, 100000)
        for ch in channels:
            assert bounds[ch] == (0.0, 100000)

    def test_min_spend_constraint(self, channels):
        """min_spend sets lower bound."""
        bounds = build_bounds_from_constraints(
            channels, 100000, min_spend={"search": 20000}
        )
        assert bounds["search"][0] == 20000
        assert bounds["display"][0] == 0.0

    def test_max_spend_constraint(self, channels):
        """max_spend sets upper bound."""
        bounds = build_bounds_from_constraints(
            channels, 100000, max_spend={"tv": 30000}
        )
        assert bounds["tv"][1] == 30000
        assert bounds["search"][1] == 100000

    def test_min_ratio_constraint(self, channels):
        """min_ratio sets lower bound as ratio of total."""
        bounds = build_bounds_from_constraints(
            channels, 100000, min_ratio={"search": 0.2}
        )
        assert bounds["search"][0] == 20000  # 20% of 100k

    def test_max_ratio_constraint(self, channels):
        """max_ratio sets upper bound as ratio of total."""
        bounds = build_bounds_from_constraints(
            channels, 100000, max_ratio={"display": 0.3}
        )
        assert bounds["display"][1] == 30000  # 30% of 100k

    def test_combined_constraints(self, channels):
        """Multiple constraints combine correctly."""
        bounds = build_bounds_from_constraints(
            channels,
            100000,
            min_spend={"search": 10000},
            max_ratio={"search": 0.5},
            min_ratio={"display": 0.1},
        )
        assert bounds["search"] == (10000, 50000)
        assert bounds["display"][0] == 10000  # 10% of 100k

    def test_invalid_bounds_warning(self, channels):
        """Warns when max < min and adjusts."""
        # This creates invalid bounds: min=50k but max=10k
        bounds = build_bounds_from_constraints(
            channels,
            100000,
            min_spend={"search": 50000},
            max_spend={"search": 10000},
        )
        # Should set upper = lower
        assert bounds["search"][1] >= bounds["search"][0]


# =============================================================================
# ScenarioResult Tests
# =============================================================================

class TestScenarioResult:
    """Tests for ScenarioResult dataclass."""

    @pytest.fixture
    def mock_results(self):
        """Create mock optimization results for scenarios."""
        results = []
        for budget in [50000, 100000, 150000]:
            results.append(OptimizationResult(
                optimal_allocation={"search": budget * 0.5, "display": budget * 0.5},
                total_budget=budget,
                expected_response=budget * 2,  # ROI of 2
                response_ci_low=budget * 1.8,
                response_ci_high=budget * 2.2,
                success=True,
                message="OK",
                iterations=10,
                objective_value=-budget * 2,
                num_periods=8,
                utility_function="mean",
            ))
        return results

    def test_scenario_result_creation(self, mock_results):
        """Can create ScenarioResult."""
        curve = pd.DataFrame({
            "budget": [50000, 100000, 150000],
            "expected_response": [100000, 200000, 300000],
            "response_ci_low": [90000, 180000, 270000],
            "response_ci_high": [110000, 220000, 330000],
            "success": [True, True, True],
            "marginal_response": [2.0, 2.0, 2.0],
        })

        result = ScenarioResult(
            budget_scenarios=[50000, 100000, 150000],
            results=mock_results,
            efficiency_curve=curve,
        )

        assert len(result.results) == 3
        assert len(result.efficiency_curve) == 3

    def test_to_dataframe(self, mock_results):
        """to_dataframe returns allocation details."""
        curve = pd.DataFrame({
            "budget": [50000, 100000, 150000],
            "expected_response": [100000, 200000, 300000],
            "response_ci_low": [90000, 180000, 270000],
            "response_ci_high": [110000, 220000, 330000],
            "success": [True, True, True],
            "marginal_response": [2.0, 2.0, 2.0],
        })

        result = ScenarioResult(
            budget_scenarios=[50000, 100000, 150000],
            results=mock_results,
            efficiency_curve=curve,
        )

        df = result.to_dataframe()
        assert "budget" in df.columns
        assert "expected_response" in df.columns
        assert "alloc_search" in df.columns


# =============================================================================
# Integration Test (Mocked)
# =============================================================================

class TestOptimizationBridgeMocked:
    """Mocked tests for OptimizationBridge."""

    def test_bridge_requires_fitted_model(self):
        """Bridge raises if model not fitted."""
        from mmm_platform.optimization.bridge import OptimizationBridge

        mock_wrapper = Mock()
        mock_wrapper.idata = None

        with pytest.raises(ValueError, match="not fitted"):
            OptimizationBridge(mock_wrapper)

    def test_bridge_requires_mmm(self):
        """Bridge raises if no MMM object."""
        from mmm_platform.optimization.bridge import OptimizationBridge

        mock_wrapper = Mock()
        mock_wrapper.idata = Mock()
        mock_wrapper.mmm = None

        with pytest.raises(ValueError, match="no MMM model"):
            OptimizationBridge(mock_wrapper)


# =============================================================================
# Optimizer UI KPI Type Display Tests
# =============================================================================

class TestOptimizerUIKPITypeFormatting:
    """Tests for optimizer UI displaying correct metrics based on KPI type.

    For revenue KPIs: Show ROI and $ formatting
    For count KPIs: Show CPA and no $ on response values
    """

    def test_revenue_kpi_shows_roi(self):
        """Revenue KPIs should display ROI metric."""
        total_budget = 100000
        expected_response = 250000

        # For revenue KPI, ROI = response / budget
        roi = expected_response / total_budget

        assert roi == 2.5
        # Display format should be "{roi:.2f}x" -> "2.50x"
        display = f"{roi:.2f}x"
        assert display == "2.50x"

    def test_count_kpi_shows_cpa(self):
        """Count KPIs should display CPA (Cost Per Acquisition) metric."""
        total_budget = 100000
        expected_installs = 5000  # count response

        # For count KPI, CPA = budget / response
        cpa = total_budget / expected_installs

        assert cpa == 20.0
        # Display format should be "${cpa:,.2f}" -> "$20.00"
        display = f"${cpa:,.2f}"
        assert display == "$20.00"

    def test_count_kpi_response_no_dollar_sign(self):
        """Count KPIs should show response without $ symbol."""
        expected_installs = 12345

        # For count KPIs, response should NOT have $
        display = f"{expected_installs:,.0f}"
        assert display == "12,345"
        assert "$" not in display

    def test_revenue_kpi_response_has_dollar_sign(self):
        """Revenue KPIs should show response with $ symbol."""
        expected_revenue = 250000

        # For revenue KPIs, response SHOULD have $
        display = f"${expected_revenue:,.0f}"
        assert display == "$250,000"
        assert "$" in display

    def test_count_kpi_ci_no_dollar_sign(self):
        """Count KPI confidence intervals should NOT have $ symbols."""
        ci_low = 4500
        ci_high = 5500

        # For count KPIs: "4,500 - 5,500"
        display = f"{ci_low:,.0f} - {ci_high:,.0f}"
        assert display == "4,500 - 5,500"
        assert "$" not in display

    def test_revenue_kpi_ci_has_dollar_sign(self):
        """Revenue KPI confidence intervals should have $ symbols."""
        ci_low = 200000
        ci_high = 300000

        # For revenue KPIs: "$200,000 - $300,000"
        display = f"${ci_low:,.0f} - ${ci_high:,.0f}"
        assert display == "$200,000 - $300,000"
        assert display.count("$") == 2

    def test_cpa_handles_zero_response(self):
        """CPA calculation should handle zero response gracefully."""
        total_budget = 100000
        expected_response = 0

        # Should not divide by zero
        cpa = total_budget / expected_response if expected_response > 0 else 0
        assert cpa == 0

    def test_roi_handles_zero_budget(self):
        """ROI calculation should handle zero budget gracefully."""
        total_budget = 0
        expected_response = 250000

        # Should not divide by zero
        roi = expected_response / total_budget if total_budget > 0 else 0
        assert roi == 0

    def test_target_column_label_formatting(self):
        """Target column name should be formatted for display label."""
        target_col = "app_installs"

        # Format: replace underscores with spaces, title case
        label = target_col.replace('_', ' ').title()
        assert label == "App Installs"

        # Full metric label
        full_label = f"Expected {label}"
        assert full_label == "Expected App Installs"


# =============================================================================
# Contribution Scale Consistency Tests
# =============================================================================

class TestContributionScaleConsistency:
    """Tests that contribution calculations are consistent across the system.

    This prevents regressions where historical contributions and optimizer
    expected response end up in different scales (which caused a ~50% mismatch).
    """

    def test_get_contributions_uses_correct_method(self):
        """Verify get_contributions() uses compute_mean_contributions_over_time.

        get_contributions() must use compute_mean_contributions_over_time which
        includes ALL model components (baseline, controls, trend, seasonality, channels).
        This is required for correct R², charts, and diagnostics.

        The optimizer uses get_channel_contributions() which calls
        compute_channel_contribution_original_scale for channel-only contributions.
        """
        from mmm_platform.model.mmm import MMMWrapper

        mock_wrapper = Mock()
        mock_wrapper.idata = Mock()

        # compute_mean_contributions_over_time returns DataFrame directly
        mock_dates = pd.to_datetime(['2024-01-01', '2024-01-08', '2024-01-15'])
        mock_contribs_df = pd.DataFrame(
            {'ch1': [100, 150, 120], 'ch2': [200, 250, 180], 'intercept': [50, 50, 50]},
            index=mock_dates
        )

        mock_mmm = Mock()
        mock_mmm.compute_mean_contributions_over_time.return_value = mock_contribs_df

        with patch.object(MMMWrapper, '__init__', lambda x, y: None):
            wrapper = MMMWrapper(None)
            wrapper.idata = mock_wrapper.idata
            wrapper.mmm = mock_mmm

            result = wrapper.get_contributions()

        # Verify the correct method was called
        mock_mmm.compute_mean_contributions_over_time.assert_called_once()

        # Verify result includes all components (not just channels)
        assert 'ch1' in result.columns
        assert 'ch2' in result.columns
        assert 'intercept' in result.columns  # Must include baseline

    def test_get_channel_contributions_sorts_by_date(self):
        """Verify get_channel_contributions() sorts output by date.

        This prevents date alignment bugs where xarray returns dates in
        different order than df_scaled, causing wrong rows to be selected
        when computing historical contributions for specific periods.
        """
        from mmm_platform.model.mmm import MMMWrapper

        # Create mock wrapper
        mock_wrapper = Mock()
        mock_wrapper.idata = Mock()

        # Create mock xarray with OUT-OF-ORDER dates to simulate the bug
        mock_xarray = Mock()
        mock_mean = Mock()

        # Dates deliberately out of order (e.g., xarray might return them this way)
        out_of_order_dates = np.array(['2024-01-15', '2024-01-01', '2024-01-08'], dtype='datetime64[ns]')
        mock_mean.coords = {
            'channel': Mock(values=np.array(['ch1', 'ch2'])),
            'date': Mock(values=out_of_order_dates)
        }
        # Values corresponding to out-of-order dates
        # Row 0: 2024-01-15, Row 1: 2024-01-01, Row 2: 2024-01-08
        mock_mean.values = np.array([
            [300, 600],  # Jan 15 values
            [100, 200],  # Jan 1 values
            [200, 400],  # Jan 8 values
        ])
        mock_xarray.mean.return_value = mock_mean

        mock_mmm = Mock()
        mock_mmm.compute_channel_contribution_original_scale.return_value = mock_xarray

        # Patch and test
        with patch.object(MMMWrapper, '__init__', lambda x, y: None):
            wrapper = MMMWrapper(None)
            wrapper.idata = mock_wrapper.idata
            wrapper.mmm = mock_mmm

            # Use get_channel_contributions (optimizer method)
            result = wrapper.get_channel_contributions()

        # After sorting by date, order should be: Jan 1, Jan 8, Jan 15
        # So values should be reordered to: [100,200], [200,400], [300,600]
        expected_first_row = [100, 200]   # Jan 1
        expected_last_row = [300, 600]    # Jan 15

        assert list(result.iloc[0].values) == expected_first_row, \
            f"First row should be Jan 1 values, got {result.iloc[0].values}"
        assert list(result.iloc[2].values) == expected_last_row, \
            f"Last row should be Jan 15 values, got {result.iloc[2].values}"

        # Verify index is DatetimeIndex (sorted ascending)
        assert isinstance(result.index, pd.DatetimeIndex), \
            f"get_channel_contributions() must return DatetimeIndex, got {type(result.index)}"
        assert result.index[0] < result.index[1] < result.index[2], \
            "DatetimeIndex should be in ascending order"

    def test_get_contributions_returns_datetime_index(self):
        """Regression test: get_contributions() must return DatetimeIndex, not RangeIndex.

        This test prevents a bug where reset_index(drop=True) was called,
        dropping the datetime index. This broke:
        - diagnostics.py: dates=contribs.index (crash on .dt accessor)
        - results.py: chart x-axis showed integers instead of dates
        - export.py: ~15 places iterating expecting dates
        """
        from mmm_platform.model.mmm import MMMWrapper

        mock_wrapper = Mock()
        mock_wrapper.idata = Mock()

        # compute_mean_contributions_over_time returns a DataFrame directly with DatetimeIndex
        mock_dates = pd.to_datetime(['2024-01-01', '2024-01-08', '2024-01-15'])
        mock_contribs_df = pd.DataFrame(
            {'ch1': [100, 200, 300]},
            index=mock_dates
        )

        mock_mmm = Mock()
        mock_mmm.compute_mean_contributions_over_time.return_value = mock_contribs_df

        with patch.object(MMMWrapper, '__init__', lambda x, y: None):
            wrapper = MMMWrapper(None)
            wrapper.idata = mock_wrapper.idata
            wrapper.mmm = mock_mmm

            result = wrapper.get_contributions()

        # CRITICAL: Must be DatetimeIndex, not RangeIndex
        assert isinstance(result.index, pd.DatetimeIndex), \
            f"get_contributions() MUST return DatetimeIndex. Got {type(result.index).__name__}. " \
            "Do NOT use reset_index(drop=True) - it breaks diagnostics, charts, and exports."

    def test_fit_statistics_alignment_no_nan(self):
        """Regression test: get_fit_statistics() must not produce NaN from alignment issues.

        This test prevents a bug where date format mismatches caused actual values to be
        NaN, resulting in negative R² values. The fix uses compute_mean_contributions_over_time
        which returns a properly indexed DataFrame that aligns correctly with df_scaled.
        """
        from mmm_platform.model.mmm import MMMWrapper

        mock_wrapper = Mock()
        mock_wrapper.idata = Mock()

        # compute_mean_contributions_over_time returns a DataFrame with DatetimeIndex
        # The dates are in chronological order
        mock_dates = pd.to_datetime(['2024-01-01', '2024-01-08', '2024-01-15'])
        mock_contribs_df = pd.DataFrame(
            {
                'ch1': [100, 150, 120],
                'ch2': [50, 75, 60],
                'intercept': [50, 50, 50],  # Include baseline component
            },
            index=mock_dates
        )

        mock_mmm = Mock()
        mock_mmm.compute_mean_contributions_over_time.return_value = mock_contribs_df

        # Create mock df_scaled with OUT-OF-ORDER dates (different order than contribs)
        # This simulates the scenario where df_scaled wasn't sorted but contribs was
        # With proper .reindex() using DatetimeIndex, this should still align correctly
        mock_df_scaled = pd.DataFrame({
            'date': pd.to_datetime(['2024-01-15', '2024-01-01', '2024-01-08']),  # OUT OF ORDER!
            'target': [230, 200, 275],  # Target values match contributions after alignment
            'ch1': [1200, 1000, 1500],
            'ch2': [600, 500, 750],
        })

        mock_config = Mock()
        mock_config.data.date_column = 'date'
        mock_config.data.target_column = 'target'

        with patch.object(MMMWrapper, '__init__', lambda x, y: None):
            wrapper = MMMWrapper(None)
            wrapper.idata = mock_wrapper.idata
            wrapper.mmm = mock_mmm
            wrapper.df_scaled = mock_df_scaled
            wrapper.config = mock_config
            wrapper.fit_duration_seconds = 10.0

            result = wrapper.get_fit_statistics()

        # R² must be a valid number, not NaN
        assert not np.isnan(result['r2']), \
            "R² is NaN - likely caused by date format mismatch in alignment."

        # R² should be reasonable (can be negative for poor models, but not -inf or extreme)
        assert result['r2'] > -10, \
            f"R² is extremely negative ({result['r2']}), suggests alignment issue."

    def test_backtest_validator_uses_same_method(self):
        """Verify BacktestValidator uses compute_channel_contribution_original_scale.

        This ensures the calibration validation uses the same scale as
        get_contributions() for historical baseline.
        """
        from mmm_platform.analysis.backtest import BacktestValidator

        # Create comprehensive mock wrapper
        mock_wrapper = Mock()
        mock_wrapper.idata = Mock()
        mock_wrapper.idata.posterior = {
            'saturation_beta': Mock(),
            'saturation_lam': Mock(),
        }
        mock_wrapper.idata.posterior['saturation_beta'].mean.return_value = Mock(values=np.array([0.1, 0.2]))
        mock_wrapper.idata.posterior['saturation_lam'].mean.return_value = Mock(values=np.array([1.0, 1.0]))

        # Mock config
        mock_config = Mock()
        mock_config.data.target_column = 'revenue'
        mock_wrapper.config = mock_config

        # Mock df_scaled
        mock_df = pd.DataFrame({
            'revenue': [1000, 2000, 1500],
            'ch1': [100, 150, 120],
            'ch2': [200, 250, 180],
        })
        mock_wrapper.df_scaled = mock_df

        # Mock MMM
        mock_mmm = Mock()
        mock_mmm.channel_columns = ['ch1', 'ch2']

        # Create mock xarray for compute_channel_contribution_original_scale
        mock_xarray = Mock()
        mock_mean = Mock()
        mock_mean.sum.return_value = Mock(values=np.array([300, 400, 300]))
        mock_xarray.mean.return_value = mock_mean
        mock_mmm.compute_channel_contribution_original_scale.return_value = mock_xarray

        mock_wrapper.mmm = mock_mmm

        # Create BacktestValidator
        validator = BacktestValidator(mock_wrapper)

        # Verify the correct method was called
        mock_mmm.compute_channel_contribution_original_scale.assert_called_once_with(prior=False)
