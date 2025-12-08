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
