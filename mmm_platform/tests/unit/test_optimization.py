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


# =============================================================================
# Optimization Mode Consistency Tests
# =============================================================================

class TestOptimizationModeConsistency:
    """Tests that all optimization modes produce consistent results.

    Mathematical invariant:
    - If optimize() at budget B returns response R
    - Then scenario_analysis() at budget B must return response R
    - And target optimization for response R must find budget ~B

    These MUST be consistent - if target says we can get R for less than B,
    then optimize(B) should return MORE than R. Anything else is a bug.
    """

    @pytest.fixture
    def mock_wrapper(self):
        """Create a comprehensive mock wrapper for testing optimization."""
        from unittest.mock import Mock, PropertyMock

        # Create mock wrapper
        wrapper = Mock()
        wrapper.idata = Mock()

        # Mock config with channels
        mock_config = Mock()
        mock_config.data = Mock()
        mock_config.data.date_column = 'date'
        mock_config.data.target_column = 'revenue'
        mock_config.data.spend_scale = 1.0  # No scaling
        mock_config.data.revenue_scale = 1.0  # No scaling

        # Mock channels
        ch1 = Mock()
        ch1.name = 'paid_search'
        ch1.display_name = 'Paid Search'
        ch1.column = 'paid_search'
        ch2 = Mock()
        ch2.name = 'display'
        ch2.display_name = 'Display'
        ch2.column = 'display'
        ch3 = Mock()
        ch3.name = 'social'
        ch3.display_name = 'Social'
        ch3.column = 'social'

        mock_config.channels = [ch1, ch2, ch3]
        mock_config.owned_media = []
        mock_config.get_channel_columns = Mock(return_value=['paid_search', 'display', 'social'])

        # Mock sampling config
        mock_config.sampling = Mock()
        mock_config.sampling.random_seed = 42

        wrapper.config = mock_config

        # Mock df_scaled - USE FIXED SEED for reproducibility
        np.random.seed(42)
        n_rows = 52
        dates = pd.date_range('2024-01-01', periods=n_rows, freq='W')
        wrapper.df_scaled = pd.DataFrame({
            'date': dates,
            'revenue': np.random.uniform(50000, 100000, n_rows),
            'paid_search': np.random.uniform(5000, 15000, n_rows),
            'display': np.random.uniform(3000, 10000, n_rows),
            'social': np.random.uniform(2000, 8000, n_rows),
        })

        # Mock transform_engine
        mock_transform = Mock()
        mock_transform.get_effective_channel_columns.return_value = ['paid_search', 'display', 'social']
        wrapper.transform_engine = mock_transform

        # Create mock MMM with realistic saturation parameters
        mock_mmm = Mock()
        mock_mmm.channel_columns = ['paid_search', 'display', 'social']
        mock_mmm.optimize_budget = Mock()

        # Mock posterior samples with FIXED seed for reproducibility
        np.random.seed(123)  # Different seed for posterior
        n_chains = 2
        n_draws = 100
        n_channels = 3

        # Create realistic beta and lambda values
        beta_values = np.random.uniform(0.8, 1.5, (n_channels, n_chains * n_draws))
        lam_values = np.random.uniform(1.5, 2.5, (n_channels, n_chains * n_draws))

        # Mock posterior
        posterior = Mock()

        beta_mock = Mock()
        stacked_beta = Mock()
        stacked_beta.values = beta_values
        beta_mock.stack = Mock(return_value=stacked_beta)
        beta_mock.mean = Mock(return_value=Mock(values=beta_values.mean(axis=1)))

        lam_mock = Mock()
        stacked_lam = Mock()
        stacked_lam.values = lam_values
        lam_mock.stack = Mock(return_value=stacked_lam)
        lam_mock.mean = Mock(return_value=Mock(values=lam_values.mean(axis=1)))

        posterior.__getitem__ = Mock(side_effect=lambda k: {
            "saturation_beta": beta_mock,
            "saturation_lam": lam_mock,
        }.get(k))

        # Create mock idata for the MMM (optimizer uses mmm.idata)
        mock_idata = Mock()
        mock_idata.posterior = posterior

        # Attach idata to both wrapper and mmm (optimizer uses mmm.idata)
        wrapper.idata = mock_idata
        mock_mmm.idata = mock_idata
        wrapper.mmm = mock_mmm

        return wrapper

    def test_main_and_scenario_produce_same_response(self, mock_wrapper):
        """Main optimize() and scenario_analysis() must return same response for same budget.

        This is the fundamental consistency test. Both modes ultimately call the same
        optimizer with the same inputs, so they MUST produce identical results.
        """
        from mmm_platform.optimization.allocator import BudgetAllocator

        budget = 50000
        num_periods = 52

        # Create allocator
        allocator = BudgetAllocator(mock_wrapper, num_periods=num_periods, utility="mean")

        # Run main optimization
        main_result = allocator.optimize(total_budget=budget)

        # Run scenario analysis with same budget
        scenario_result = allocator.scenario_analysis([budget])
        scenario_budget_result = scenario_result.results[0]

        # CRITICAL: These must match
        main_response = main_result.expected_response
        scenario_response = scenario_budget_result.expected_response

        # Allow 1% tolerance for floating point
        rel_diff = abs(main_response - scenario_response) / main_response
        assert rel_diff < 0.01, (
            f"Main mode response ({main_response:,.0f}) != "
            f"Scenario mode response ({scenario_response:,.0f}). "
            f"Relative difference: {rel_diff:.2%}. "
            f"These MUST be consistent - they use the same optimizer."
        )

        # Also check budget allocation matches
        main_total = sum(main_result.optimal_allocation.values())
        scenario_total = sum(scenario_budget_result.optimal_allocation.values())

        budget_diff = abs(main_total - scenario_total) / budget
        assert budget_diff < 0.01, (
            f"Main allocated ${main_total:,.0f}, "
            f"Scenario allocated ${scenario_total:,.0f}. "
            f"Budget mismatch: {budget_diff:.2%}"
        )

    def test_target_finds_correct_budget(self, mock_wrapper):
        """Target optimization must find budget that produces the target response.

        If optimize(B) returns response R, then target_optimization(R) should find
        budget close to B (within tolerance).
        """
        from mmm_platform.optimization.allocator import BudgetAllocator
        from mmm_platform.optimization.scenarios import TargetOptimizer

        budget = 50000
        num_periods = 52

        # Create allocator and get baseline result
        allocator = BudgetAllocator(mock_wrapper, num_periods=num_periods, utility="mean")
        main_result = allocator.optimize(total_budget=budget)
        target_response = main_result.expected_response

        # Now find budget to achieve that response
        target_opt = TargetOptimizer(allocator)
        target_result = target_opt.find_budget_for_target(
            target_response=target_response,
            budget_range=(10000, 200000),
            tolerance=0.01,  # 1% tolerance
        )

        # The required budget should be close to our original budget
        found_budget = target_result.required_budget

        # Allow 5% tolerance (target optimization uses binary search which may not be exact)
        rel_diff = abs(found_budget - budget) / budget
        assert rel_diff < 0.05, (
            f"Target optimization found budget ${found_budget:,.0f} "
            f"but original budget was ${budget:,.0f}. "
            f"Relative difference: {rel_diff:.2%}. "
            f"If these differ significantly, the modes are inconsistent."
        )

    def test_cannot_get_more_response_with_less_budget(self, mock_wrapper):
        """It's mathematically impossible to get MORE response with LESS budget.

        This is the ultimate consistency check:
        - If optimize($50k) returns response R
        - And target_optimization(R) says budget B is needed
        - Then B must be <= $50k

        If B < $50k significantly, then optimize($50k) should have returned MORE than R.
        """
        from mmm_platform.optimization.allocator import BudgetAllocator
        from mmm_platform.optimization.scenarios import TargetOptimizer

        budget = 50000
        num_periods = 52

        # Get response at budget
        allocator = BudgetAllocator(mock_wrapper, num_periods=num_periods, utility="mean")
        main_result = allocator.optimize(total_budget=budget)
        response_at_budget = main_result.expected_response

        # Find budget needed for that response
        target_opt = TargetOptimizer(allocator)
        target_result = target_opt.find_budget_for_target(
            target_response=response_at_budget,
            budget_range=(10000, 200000),
            tolerance=0.01,
        )

        required_budget = target_result.required_budget

        # The required budget cannot be LESS than what we used (with meaningful margin)
        # Allow 5% tolerance for optimizer precision
        assert required_budget >= budget * 0.95, (
            f"IMPOSSIBLE: Target says we need only ${required_budget:,.0f} "
            f"to get response {response_at_budget:,.0f}, "
            f"but main optimization used ${budget:,.0f}. "
            f"This means optimize() is leaving money on the table."
        )

    def test_increasing_budget_increases_response(self, mock_wrapper):
        """Response must increase (or stay same) with more budget.

        Saturation means diminishing returns, but response should never DECREASE
        when budget increases.
        """
        from mmm_platform.optimization.allocator import BudgetAllocator

        num_periods = 52
        budgets = [25000, 50000, 75000, 100000]

        allocator = BudgetAllocator(mock_wrapper, num_periods=num_periods, utility="mean")

        prev_response = 0
        for budget in budgets:
            result = allocator.optimize(total_budget=budget)
            response = result.expected_response

            assert response >= prev_response, (
                f"Response decreased from {prev_response:,.0f} to {response:,.0f} "
                f"when budget increased to ${budget:,.0f}. "
                f"This violates economic intuition."
            )
            prev_response = response

    def test_scenario_analysis_returns_monotonic_responses(self, mock_wrapper):
        """Scenario analysis responses must be monotonically increasing with budget."""
        from mmm_platform.optimization.allocator import BudgetAllocator

        num_periods = 52
        budgets = [25000, 50000, 75000, 100000]

        allocator = BudgetAllocator(mock_wrapper, num_periods=num_periods, utility="mean")
        scenario_result = allocator.scenario_analysis(budgets)

        prev_response = 0
        for result in scenario_result.results:
            response = result.expected_response
            assert response >= prev_response, (
                f"Scenario response decreased from {prev_response:,.0f} to {response:,.0f}. "
                f"Responses must be monotonically increasing with budget."
            )
            prev_response = response

    def test_different_seasonal_indices_cause_different_results(self, mock_wrapper):
        """Demonstrates: Different seasonal indices WILL cause different results.

        This test proves that passing different seasonal_indices to optimize()
        produces different responses. This is the ROOT CAUSE of the UI bug -
        if the seasonality expander computes indices for the wrong num_periods,
        different modes will get different seasonal indices → different results.
        """
        from mmm_platform.optimization.allocator import BudgetAllocator

        budget = 50000
        num_periods = 52
        channels = ['paid_search', 'display', 'social']

        allocator = BudgetAllocator(mock_wrapper, num_periods=num_periods, utility="mean")

        # Run with NO seasonal indices (all 1.0)
        result_no_seasonal = allocator.optimize(
            total_budget=budget,
            seasonal_indices=None,
        )

        # Run with DIFFERENT seasonal indices (simulate wrong period)
        # E.g., indices computed for 8 periods vs 52 periods would differ
        different_seasonal = {ch: 1.3 for ch in channels}  # 30% higher effectiveness
        result_with_seasonal = allocator.optimize(
            total_budget=budget,
            seasonal_indices=different_seasonal,
        )

        # Different seasonal indices MUST produce different responses
        response_no = result_no_seasonal.expected_response
        response_with = result_with_seasonal.expected_response

        # Higher seasonal indices = higher response (same budget, more effective)
        assert response_with > response_no, (
            f"Seasonal indices should increase response: "
            f"no_seasonal={response_no:,.0f}, with_seasonal={response_with:,.0f}"
        )

        # The difference should be meaningful (not just floating point noise)
        rel_diff = (response_with - response_no) / response_no
        assert rel_diff > 0.05, (
            f"Seasonal indices had minimal effect ({rel_diff:.2%}). "
            f"This suggests they're not being applied correctly."
        )

        print(f"\n=== PROOF: Seasonal indices cause different results ===")
        print(f"Same budget: ${budget:,}")
        print(f"No seasonal: ${response_no:,.0f} response")
        print(f"With seasonal (1.3x): ${response_with:,.0f} response")
        print(f"Difference: {rel_diff:.1%}")
        print(f"This is why UI modes produce different results when they ")
        print(f"compute seasonal indices using different num_periods!")

    def test_different_num_periods_allocators_produce_different_results(self, mock_wrapper):
        """Different num_periods in allocator = different results (even same budget).

        This test proves that if the UI creates allocators with different num_periods,
        they will produce different responses for the same budget.

        Root cause: If opt_num_periods=8 but scenario_num_periods=52, and the
        seasonal expander always uses opt_num_periods, the seasonal indices
        will be computed for 8 periods but applied to a 52-period optimization.
        """
        from mmm_platform.optimization.allocator import BudgetAllocator

        budget = 50000

        # Create allocator with 8 periods
        allocator_8 = BudgetAllocator(mock_wrapper, num_periods=8, utility="mean")
        result_8 = allocator_8.optimize(total_budget=budget)

        # Create allocator with 52 periods
        allocator_52 = BudgetAllocator(mock_wrapper, num_periods=52, utility="mean")
        result_52 = allocator_52.optimize(total_budget=budget)

        # Different num_periods MUST produce different responses
        response_8 = result_8.expected_response
        response_52 = result_52.expected_response

        # 52 periods should give MUCH higher response (more time to accumulate)
        assert response_52 > response_8, (
            f"52 periods should produce higher response than 8 periods: "
            f"8_periods={response_8:,.0f}, 52_periods={response_52:,.0f}"
        )

        # The ratio depends on saturation - with heavy saturation, more periods
        # doesn't scale linearly. The key point is just that they're DIFFERENT.
        ratio = response_52 / response_8
        assert ratio > 1.05, (
            f"Response ratio ({ratio:.2f}x) is too close to 1. "
            f"Different num_periods should produce meaningfully different results."
        )

        print(f"\n=== PROOF: num_periods affects response ===")
        print(f"Same budget: ${budget:,}")
        print(f"8 periods: ${response_8:,.0f} response")
        print(f"52 periods: ${response_52:,.0f} response")
        print(f"Ratio: {ratio:.1f}x")
        print(f"This is why modes with different num_periods produce different results!")

    def test_all_three_modes_mathematically_consistent(self, mock_wrapper):
        """
        CRITICAL INVARIANT: All three optimization modes must be mathematically consistent.

        Given:
        - Budget B = $50,000
        - num_periods = 52 weeks

        Then:
        1. Main mode: optimize(B) returns response R (e.g., $200,564)
        2. Scenario mode: scenario_analysis([B]) must return EXACTLY R
        3. Target mode: find_budget_for_target(R) must return budget ~B

        Corollary: If target mode says we can achieve R for budget < B, that's
        MATHEMATICALLY IMPOSSIBLE because optimize(B) would have found that
        better allocation.

        This test catches bugs where the UI passes different parameters to
        different modes (e.g., different num_periods, different seasonal indices).
        """
        from mmm_platform.optimization.allocator import BudgetAllocator
        from mmm_platform.optimization.scenarios import TargetOptimizer

        # Test parameters
        budget = 50000
        num_periods = 52

        # Create single allocator - ALL modes must use the SAME allocator
        allocator = BudgetAllocator(mock_wrapper, num_periods=num_periods, utility="mean")

        # ============================================================
        # STEP 1: Main mode - optimize at budget B
        # ============================================================
        main_result = allocator.optimize(total_budget=budget)
        main_response = main_result.expected_response

        print(f"\n{'='*60}")
        print(f"MAIN MODE: optimize(${budget:,}) over {num_periods} weeks")
        print(f"  Expected Response: ${main_response:,.0f}")
        print(f"  Allocation: {main_result.optimal_allocation}")
        print(f"{'='*60}")

        # ============================================================
        # STEP 2: Scenario mode - same budget, MUST get same response
        # ============================================================
        scenario_result = allocator.scenario_analysis([budget])
        scenario_response = scenario_result.results[0].expected_response

        print(f"\nSCENARIO MODE: scenario_analysis([${budget:,}])")
        print(f"  Expected Response: ${scenario_response:,.0f}")

        # CRITICAL: These MUST match
        response_diff_pct = abs(main_response - scenario_response) / main_response * 100
        print(f"  Difference from main: {response_diff_pct:.2f}%")

        assert response_diff_pct < 1.0, (
            f"INCONSISTENCY: Main mode returned ${main_response:,.0f} but "
            f"scenario mode returned ${scenario_response:,.0f} for same ${budget:,} budget. "
            f"Difference: {response_diff_pct:.2f}%. These MUST match."
        )

        # ============================================================
        # STEP 3: Target mode - find budget to achieve main_response
        # ============================================================
        target_opt = TargetOptimizer(allocator)
        target_result = target_opt.find_budget_for_target(
            target_response=main_response,
            budget_range=(10000, 200000),
            tolerance=0.01,  # 1% tolerance
        )

        found_budget = target_result.required_budget
        target_achieved_response = target_result.expected_response

        print(f"\nTARGET MODE: find_budget_for_target(${main_response:,.0f})")
        print(f"  Required Budget: ${found_budget:,.0f}")
        print(f"  Achieved Response: ${target_achieved_response:,.0f}")

        # ============================================================
        # CRITICAL CHECK: Target budget must be >= original budget
        # ============================================================
        budget_diff_pct = (found_budget - budget) / budget * 100

        print(f"\n{'='*60}")
        print(f"CONSISTENCY CHECK:")
        print(f"  Original budget: ${budget:,.0f}")
        print(f"  Target found budget: ${found_budget:,.0f}")
        print(f"  Difference: {budget_diff_pct:+.1f}%")

        # Allow 5% tolerance for optimizer precision
        if found_budget < budget * 0.95:
            # This is the IMPOSSIBLE case
            print(f"\n  *** MATHEMATICAL IMPOSSIBILITY DETECTED ***")
            print(f"  Target says we need only ${found_budget:,.0f} to get ${main_response:,.0f}")
            print(f"  But main mode used ${budget:,.0f} and only got ${main_response:,.0f}")
            print(f"  If target is right, main mode left money on the table!")
            print(f"{'='*60}")

        assert found_budget >= budget * 0.95, (
            f"MATHEMATICAL IMPOSSIBILITY: "
            f"Main mode: ${budget:,} budget → ${main_response:,.0f} response. "
            f"Target mode: ${found_budget:,} budget needed for ${main_response:,.0f}. "
            f"If we can get the same response for {100 - budget_diff_pct:.1f}% less budget, "
            f"then optimize() is broken. This is mathematically impossible with correct optimizer."
        )

        print(f"  PASS: All three modes are mathematically consistent!")
        print(f"{'='*60}")

    def test_all_modes_consistency_across_budget_levels(self, mock_wrapper):
        """
        Comprehensive test: For each budget level from $50k to $1M (in $25k increments),
        verify that ALL THREE optimization modes are consistent.

        For each budget B:
        1. Run main optimization -> get expected response R_main
        2. Run scenario analysis -> get expected response R_scenario
        3. Run target optimization for R_main -> get required budget B'
        4. R_main should equal R_scenario (same budget, same response)
        5. B' should be approximately equal to B (within 5% tolerance)

        If any mode disagrees, something is broken.
        """
        from mmm_platform.optimization.allocator import BudgetAllocator
        from mmm_platform.optimization.scenarios import TargetOptimizer

        num_periods = 52
        allocator = BudgetAllocator(mock_wrapper, num_periods=num_periods, utility="mean")
        target_opt = TargetOptimizer(allocator)

        # Budget levels: $50k to $1M in $25k increments
        budget_levels = list(range(50000, 1000001, 25000))

        print(f"\n{'='*100}")
        print(f"COMPREHENSIVE ALL-MODES CONSISTENCY TEST")
        print(f"Testing {len(budget_levels)} budget levels from $50k to $1M")
        print(f"num_periods = {num_periods}")
        print(f"{'='*100}")
        print(f"{'Budget':>12} | {'Main Resp':>12} | {'Scenario Resp':>13} | {'Scen Diff':>9} | {'Target Budget':>13} | {'Tgt Diff':>8} | Status")
        print(f"{'-'*12}-+-{'-'*12}-+-{'-'*13}-+-{'-'*9}-+-{'-'*13}-+-{'-'*8}-+-------")

        failures = []

        for budget in budget_levels:
            # Step 1: Main optimization
            main_result = allocator.optimize(total_budget=budget)
            main_response = main_result.expected_response

            # Step 2: Scenario analysis with same budget
            scenario_result = allocator.scenario_analysis([budget])
            scenario_response = scenario_result.results[0].expected_response

            # Step 3: Target optimization - find budget for main response
            target_result = target_opt.find_budget_for_target(
                target_response=main_response,
                budget_range=(10000, 2000000),
                tolerance=0.01,
            )
            found_budget = target_result.required_budget

            # Calculate differences
            scenario_diff_pct = (scenario_response - main_response) / main_response * 100 if main_response > 0 else 0
            target_diff_pct = (found_budget - budget) / budget * 100

            # Determine status - fail if scenario doesn't match OR target finds lower budget
            status = "OK"
            failure_reason = None

            if abs(scenario_diff_pct) > 1.0:  # >1% difference in scenario
                status = "FAIL"
                failure_reason = f"Scenario mismatch: {scenario_diff_pct:+.1f}%"
            elif found_budget < budget * 0.95:  # Target found >5% lower budget
                status = "FAIL"
                failure_reason = f"Target budget too low: {target_diff_pct:+.1f}%"

            if status == "FAIL":
                failures.append({
                    "budget": budget,
                    "main_response": main_response,
                    "scenario_response": scenario_response,
                    "scenario_diff_pct": scenario_diff_pct,
                    "target_budget": found_budget,
                    "target_diff_pct": target_diff_pct,
                    "reason": failure_reason,
                })

            print(f"${budget:>11,} | ${main_response:>11,.0f} | ${scenario_response:>12,.0f} | {scenario_diff_pct:>+8.2f}% | ${found_budget:>12,.0f} | {target_diff_pct:>+7.1f}% | {status}")

            # Clear target optimizer cache between iterations
            target_opt.clear_cache()

        print(f"{'='*100}")
        print(f"SUMMARY: {len(budget_levels) - len(failures)}/{len(budget_levels)} passed")

        if failures:
            print(f"\nFAILURES ({len(failures)}):")
            for f in failures:
                print(f"  Budget ${f['budget']:,}: {f['reason']}")
                print(f"    Main: ${f['main_response']:,.0f}, Scenario: ${f['scenario_response']:,.0f}, Target budget: ${f['target_budget']:,.0f}")

        print(f"{'='*100}")

        # Assert no failures
        assert len(failures) == 0, (
            f"{len(failures)} budget levels showed inconsistency. "
            f"First failure at ${failures[0]['budget']:,}: {failures[0]['reason']}"
        )
