"""
Tests for scenario analysis and target optimization.

Tests cover:
- TargetOptimizer binary search
- ResponseCurveAnalyzer
- Efficiency frontier computation
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, MagicMock


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_optimization_result():
    """Factory for creating mock OptimizationResult."""
    def _create(
        budget: float,
        response: float,
        success: bool = True,
    ):
        result = Mock()
        result.total_budget = budget
        result.expected_response = response
        result.response_ci_low = response * 0.9
        result.response_ci_high = response * 1.1
        result.success = success
        result.optimal_allocation = {
            "tv_spend": budget * 0.5,
            "search_spend": budget * 0.3,
            "email_sends": budget * 0.2,
        }
        return result
    return _create


@pytest.fixture
def mock_allocator(mock_optimization_result):
    """Create mock BudgetAllocator with predictable responses."""
    allocator = Mock()

    # Response curve: diminishing returns (sqrt-like)
    # response = 1000 * sqrt(budget / 1000)
    def mock_optimize(total_budget, **kwargs):
        response = 1000 * np.sqrt(total_budget / 1000)
        return mock_optimization_result(total_budget, response)

    allocator.optimize = Mock(side_effect=mock_optimize)
    allocator.channels = ["tv_spend", "search_spend", "email_sends"]

    return allocator


# ============================================================================
# Test Classes
# ============================================================================

class TestTargetOptimizer:
    """Tests for TargetOptimizer class."""

    def test_init(self, mock_allocator):
        """Initializes with allocator."""
        from mmm_platform.optimization.scenarios import TargetOptimizer

        optimizer = TargetOptimizer(mock_allocator)

        assert optimizer.allocator == mock_allocator
        assert optimizer._cache == {}

    def test_find_budget_for_achievable_target(self, mock_allocator):
        """Finds budget for achievable target."""
        from mmm_platform.optimization.scenarios import TargetOptimizer

        optimizer = TargetOptimizer(mock_allocator)

        # Target 5000 response - budget should be around 25000 (5000^2 / 1000)
        result = optimizer.find_budget_for_target(
            target_response=5000,
            budget_range=(1000, 100000),
            tolerance=0.05,
        )

        assert result.achievable is True
        assert abs(result.expected_response - 5000) / 5000 < 0.05
        # Budget should be close to 25000
        assert 20000 < result.required_budget < 30000

    def test_find_budget_for_unachievable_target(self, mock_allocator):
        """Returns max budget result for unachievable target."""
        from mmm_platform.optimization.scenarios import TargetOptimizer

        optimizer = TargetOptimizer(mock_allocator)

        # Target 50000 response - impossible with max budget 100000
        # Max response at 100000 = 1000 * sqrt(100) = 10000
        result = optimizer.find_budget_for_target(
            target_response=50000,
            budget_range=(1000, 100000),
        )

        assert result.achievable is False
        assert result.required_budget == 100000
        assert "not achievable" in result.message.lower()

    def test_find_budget_achievable_at_min(self, mock_allocator):
        """Returns min budget if target already met."""
        from mmm_platform.optimization.scenarios import TargetOptimizer

        optimizer = TargetOptimizer(mock_allocator)

        # Target 100 response - achievable at min budget
        # Response at 1000 = 1000 * sqrt(1) = 1000 > 100
        result = optimizer.find_budget_for_target(
            target_response=100,
            budget_range=(1000, 100000),
        )

        assert result.achievable is True
        assert result.required_budget == 1000
        assert "minimum budget" in result.message.lower()

    def test_binary_search_converges(self, mock_allocator):
        """Binary search converges within max iterations."""
        from mmm_platform.optimization.scenarios import TargetOptimizer

        optimizer = TargetOptimizer(mock_allocator)

        result = optimizer.find_budget_for_target(
            target_response=3000,
            budget_range=(1000, 50000),
            tolerance=0.01,
            max_iterations=20,
        )

        assert result.achievable is True
        assert result.iterations <= 20
        # Should converge well within tolerance
        relative_error = abs(result.expected_response - 3000) / 3000
        assert relative_error <= 0.05

    def test_caching_works(self, mock_allocator):
        """Caches optimization results."""
        from mmm_platform.optimization.scenarios import TargetOptimizer

        optimizer = TargetOptimizer(mock_allocator)

        # First call
        optimizer._get_response_at_budget(50000)
        initial_calls = mock_allocator.optimize.call_count

        # Second call with same budget
        optimizer._get_response_at_budget(50000)

        # Should use cache
        assert mock_allocator.optimize.call_count == initial_calls

    def test_clear_cache(self, mock_allocator):
        """clear_cache removes cached results."""
        from mmm_platform.optimization.scenarios import TargetOptimizer

        optimizer = TargetOptimizer(mock_allocator)

        optimizer._get_response_at_budget(50000)
        assert len(optimizer._cache) > 0

        optimizer.clear_cache()
        assert len(optimizer._cache) == 0

    def test_returns_optimal_allocation(self, mock_allocator):
        """Result includes optimal allocation."""
        from mmm_platform.optimization.scenarios import TargetOptimizer

        optimizer = TargetOptimizer(mock_allocator)

        result = optimizer.find_budget_for_target(
            target_response=3000,
            budget_range=(1000, 50000),
        )

        assert result.optimal_allocation is not None
        assert "tv_spend" in result.optimal_allocation


class TestResponseCurveAnalyzer:
    """Tests for ResponseCurveAnalyzer class."""

    def test_init(self, mock_allocator):
        """Initializes with allocator."""
        from mmm_platform.optimization.scenarios import ResponseCurveAnalyzer

        analyzer = ResponseCurveAnalyzer(mock_allocator)

        assert analyzer.allocator == mock_allocator

    def test_compute_response_curve_returns_dataframe(self, mock_allocator):
        """compute_response_curve returns DataFrame."""
        from mmm_platform.optimization.scenarios import ResponseCurveAnalyzer

        analyzer = ResponseCurveAnalyzer(mock_allocator)

        curve = analyzer.compute_response_curve(
            budget_range=(10000, 50000),
            num_points=5,
        )

        assert isinstance(curve, pd.DataFrame)
        assert len(curve) == 5

    def test_curve_has_required_columns(self, mock_allocator):
        """Curve has all required columns."""
        from mmm_platform.optimization.scenarios import ResponseCurveAnalyzer

        analyzer = ResponseCurveAnalyzer(mock_allocator)

        curve = analyzer.compute_response_curve(
            budget_range=(10000, 50000),
            num_points=5,
        )

        required_columns = [
            "budget",
            "expected_response",
            "response_ci_low",
            "response_ci_high",
            "success",
            "roi",
            "marginal_response",
            "marginal_roi",
        ]

        for col in required_columns:
            assert col in curve.columns

    def test_marginal_roi_computed(self, mock_allocator):
        """Marginal ROI is computed between points."""
        from mmm_platform.optimization.scenarios import ResponseCurveAnalyzer

        analyzer = ResponseCurveAnalyzer(mock_allocator)

        curve = analyzer.compute_response_curve(
            budget_range=(10000, 50000),
            num_points=5,
        )

        # All marginal ROI should be positive (increasing response)
        assert all(curve["marginal_roi"] > 0)

        # Marginal ROI should decrease (diminishing returns)
        marginal_rois = curve["marginal_roi"].values[1:]
        for i in range(len(marginal_rois) - 1):
            assert marginal_rois[i + 1] <= marginal_rois[i] * 1.1  # Allow small variance

    def test_find_efficient_budget(self, mock_allocator):
        """Finds budget where marginal ROI = target."""
        from mmm_platform.optimization.scenarios import ResponseCurveAnalyzer

        analyzer = ResponseCurveAnalyzer(mock_allocator)

        # With sqrt response curve, marginal ROI = 500 / sqrt(budget/1000)
        # At budget=25000 (sqrt(25)=5), marginal ROI = 500/5 = 100
        # At budget=10000 (sqrt(10)~3.16), marginal ROI = 500/3.16 ~ 158
        efficient_budget = analyzer.find_efficient_budget(
            budget_range=(10000, 100000),
            target_marginal_roi=0.1,  # Low target - high budget
        )

        assert efficient_budget > 10000


class TestEfficiencyFrontier:
    """Tests for compute_efficiency_frontier function."""

    def test_returns_dataframe(self, mock_optimization_result):
        """Returns DataFrame with efficiency metrics."""
        from mmm_platform.optimization.scenarios import compute_efficiency_frontier
        from mmm_platform.optimization.results import ScenarioResult

        # Create mock scenario result
        results = [
            mock_optimization_result(10000, 5000),
            mock_optimization_result(20000, 8000),
            mock_optimization_result(30000, 10000),
        ]

        efficiency_curve = pd.DataFrame({
            "budget": [10000, 20000, 30000],
            "expected_response": [5000, 8000, 10000],
            "response_ci_low": [4500, 7200, 9000],
            "response_ci_high": [5500, 8800, 11000],
            "success": [True, True, True],
            "marginal_response": [5000, 3000, 2000],
        })

        scenario_result = ScenarioResult(
            budget_scenarios=[10000, 20000, 30000],
            results=results,
            efficiency_curve=efficiency_curve,
        )

        frontier = compute_efficiency_frontier(scenario_result)

        assert isinstance(frontier, pd.DataFrame)
        assert "efficiency" in frontier.columns
        assert "diminishing_returns" in frontier.columns
        assert "is_optimal" in frontier.columns

    def test_marks_diminishing_returns(self, mock_optimization_result):
        """Correctly marks diminishing returns points."""
        from mmm_platform.optimization.scenarios import compute_efficiency_frontier
        from mmm_platform.optimization.results import ScenarioResult

        results = [
            mock_optimization_result(10000, 5000),
            mock_optimization_result(20000, 7000),
            mock_optimization_result(30000, 8000),
        ]

        efficiency_curve = pd.DataFrame({
            "budget": [10000, 20000, 30000],
            "expected_response": [5000, 7000, 8000],
            "response_ci_low": [4500, 6300, 7200],
            "response_ci_high": [5500, 7700, 8800],
            "success": [True, True, True],
            "marginal_response": [0.5, 0.2, 0.1],  # ROI = 0.5, 0.35, 0.27
        })

        scenario_result = ScenarioResult(
            budget_scenarios=[10000, 20000, 30000],
            results=results,
            efficiency_curve=efficiency_curve,
        )

        frontier = compute_efficiency_frontier(scenario_result)

        # Should have diminishing returns marked
        assert "diminishing_returns" in frontier.columns
        assert frontier["diminishing_returns"].dtype == bool
