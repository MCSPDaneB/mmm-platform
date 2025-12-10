"""
Tests for ROI/CPA floor optimization mode.

This module tests the optimize_with_efficiency_floor method in BudgetAllocator.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, MagicMock, patch

from mmm_platform.optimization.results import OptimizationResult


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_allocator():
    """Create a mock BudgetAllocator for testing efficiency floor."""
    from unittest.mock import Mock

    allocator = Mock()
    allocator.channels = ["tv_spend", "search_spend", "display_spend"]
    allocator.num_periods = 8
    allocator.utility_name = "mean"

    return allocator


@pytest.fixture
def high_roi_result():
    """Mock optimization result with high ROI (2.5x)."""
    return OptimizationResult(
        optimal_allocation={"tv_spend": 30000, "search_spend": 50000, "display_spend": 20000},
        total_budget=100000,
        expected_response=250000,  # ROI = 2.5x
        response_ci_low=200000,
        response_ci_high=300000,
        success=True,
        message="Success",
        iterations=10,
        objective_value=-250000,
        num_periods=8,
        utility_function="mean",
    )


@pytest.fixture
def low_roi_result():
    """Mock optimization result with low ROI (1.2x)."""
    return OptimizationResult(
        optimal_allocation={"tv_spend": 40000, "search_spend": 40000, "display_spend": 20000},
        total_budget=100000,
        expected_response=120000,  # ROI = 1.2x
        response_ci_low=100000,
        response_ci_high=140000,
        success=True,
        message="Success",
        iterations=10,
        objective_value=-120000,
        num_periods=8,
        utility_function="mean",
    )


@pytest.fixture
def medium_cpa_result():
    """Mock optimization result with medium CPA ($8)."""
    return OptimizationResult(
        optimal_allocation={"tv_spend": 25000, "search_spend": 40000, "display_spend": 15000},
        total_budget=80000,
        expected_response=10000,  # CPA = $8
        response_ci_low=8000,
        response_ci_high=12000,
        success=True,
        message="Success",
        iterations=10,
        objective_value=-10000,
        num_periods=8,
        utility_function="mean",
    )


# =============================================================================
# OptimizationResult Efficiency Fields Tests
# =============================================================================

class TestOptimizationResultEfficiencyFields:
    """Tests for efficiency floor fields in OptimizationResult."""

    def test_result_has_efficiency_fields(self):
        """OptimizationResult includes efficiency floor fields."""
        result = OptimizationResult(
            optimal_allocation={"search": 50000},
            total_budget=100000,
            expected_response=200000,
            response_ci_low=180000,
            response_ci_high=220000,
            success=True,
            message="Success",
            iterations=10,
            objective_value=-200000,
            num_periods=8,
            utility_function="mean",
        )

        # Check fields exist and default to None
        assert hasattr(result, 'unallocated_budget')
        assert hasattr(result, 'efficiency_target')
        assert hasattr(result, 'efficiency_metric')
        assert hasattr(result, 'achieved_efficiency')

        assert result.unallocated_budget is None
        assert result.efficiency_target is None
        assert result.efficiency_metric is None
        assert result.achieved_efficiency is None

    def test_result_with_efficiency_floor_fields(self):
        """OptimizationResult can have efficiency floor fields populated."""
        result = OptimizationResult(
            optimal_allocation={"search": 50000},
            total_budget=50000,
            expected_response=100000,
            response_ci_low=90000,
            response_ci_high=110000,
            success=True,
            message="Success",
            iterations=10,
            objective_value=-100000,
            num_periods=8,
            utility_function="mean",
            unallocated_budget=50000,
            efficiency_target=2.0,
            efficiency_metric="roi",
            achieved_efficiency=2.0,
        )

        assert result.unallocated_budget == 50000
        assert result.efficiency_target == 2.0
        assert result.efficiency_metric == "roi"
        assert result.achieved_efficiency == 2.0


# =============================================================================
# ROI Floor Tests
# =============================================================================

class TestROIFloorOptimization:
    """Tests for ROI floor optimization mode."""

    def test_roi_floor_met_at_full_budget(self, mock_allocator, high_roi_result):
        """If ROI target achievable at full budget, spend it all."""
        # Mock optimize to return high ROI result
        mock_allocator.optimize.return_value = high_roi_result

        # Import the real method to test
        from mmm_platform.optimization.allocator import BudgetAllocator

        # Bind the real method to mock
        mock_allocator.optimize_with_efficiency_floor = (
            lambda *args, **kwargs: BudgetAllocator.optimize_with_efficiency_floor(
                mock_allocator, *args, **kwargs
            )
        )

        result = mock_allocator.optimize_with_efficiency_floor(
            total_budget=100000,
            efficiency_metric="roi",
            efficiency_target=2.0,  # Require 2x ROI
        )

        # Full budget meets target (2.5x > 2.0x)
        assert result.unallocated_budget == 0.0
        assert result.efficiency_metric == "roi"
        assert result.efficiency_target == 2.0
        assert result.achieved_efficiency == 2.5

    def test_roi_floor_not_met_returns_unspent(self, mock_allocator, low_roi_result, high_roi_result):
        """If ROI target requires less spend, return unallocated."""
        # Create a sequence: first call returns low ROI, subsequent calls return high ROI
        # (simulating binary search finding a lower budget that works)
        call_count = [0]

        def mock_optimize(*args, **kwargs):
            call_count[0] += 1
            budget = kwargs.get('total_budget', args[0] if args else 100000)

            if budget >= 100000:
                # Full budget doesn't meet target
                result = low_roi_result
                result.total_budget = budget
                return result
            else:
                # Lower budget meets target
                result = high_roi_result
                result.total_budget = budget
                result.expected_response = budget * 2.5  # Scale response
                return result

        mock_allocator.optimize = mock_optimize

        from mmm_platform.optimization.allocator import BudgetAllocator

        mock_allocator.optimize_with_efficiency_floor = (
            lambda *args, **kwargs: BudgetAllocator.optimize_with_efficiency_floor(
                mock_allocator, *args, **kwargs
            )
        )

        result = mock_allocator.optimize_with_efficiency_floor(
            total_budget=100000,
            efficiency_metric="roi",
            efficiency_target=2.0,
        )

        # Should have found a budget level that works
        assert result.unallocated_budget is not None
        assert result.efficiency_metric == "roi"
        assert result.total_budget < 100000


# =============================================================================
# CPA Floor Tests
# =============================================================================

class TestCPAFloorOptimization:
    """Tests for CPA floor optimization mode."""

    def test_cpa_floor_met_at_full_budget(self, mock_allocator, medium_cpa_result):
        """If CPA target achievable at full budget, spend it all."""
        mock_allocator.optimize.return_value = medium_cpa_result

        from mmm_platform.optimization.allocator import BudgetAllocator

        mock_allocator.optimize_with_efficiency_floor = (
            lambda *args, **kwargs: BudgetAllocator.optimize_with_efficiency_floor(
                mock_allocator, *args, **kwargs
            )
        )

        result = mock_allocator.optimize_with_efficiency_floor(
            total_budget=80000,
            efficiency_metric="cpa",
            efficiency_target=10.0,  # Max CPA of $10
        )

        # Full budget meets target ($8 < $10)
        assert result.unallocated_budget == 0.0
        assert result.efficiency_metric == "cpa"
        assert result.achieved_efficiency == 8.0  # CPA = 80000/10000


# =============================================================================
# Edge Cases Tests
# =============================================================================

class TestEfficiencyFloorEdgeCases:
    """Tests for edge cases in efficiency floor optimization."""

    def test_zero_response_handling(self, mock_allocator):
        """Handles zero response gracefully."""
        zero_result = OptimizationResult(
            optimal_allocation={"tv_spend": 0, "search_spend": 0, "display_spend": 0},
            total_budget=0,
            expected_response=0,
            response_ci_low=0,
            response_ci_high=0,
            success=True,
            message="No allocation",
            iterations=0,
            objective_value=0,
            num_periods=8,
            utility_function="mean",
        )

        mock_allocator.optimize.return_value = zero_result

        from mmm_platform.optimization.allocator import BudgetAllocator

        mock_allocator.optimize_with_efficiency_floor = (
            lambda *args, **kwargs: BudgetAllocator.optimize_with_efficiency_floor(
                mock_allocator, *args, **kwargs
            )
        )

        # Should not crash with division by zero
        result = mock_allocator.optimize_with_efficiency_floor(
            total_budget=100000,
            efficiency_metric="roi",
            efficiency_target=2.0,
        )

        assert result is not None
        assert result.total_budget == 0

    def test_result_includes_all_fields(self, mock_allocator, high_roi_result):
        """Result from efficiency floor includes all required fields."""
        mock_allocator.optimize.return_value = high_roi_result

        from mmm_platform.optimization.allocator import BudgetAllocator

        mock_allocator.optimize_with_efficiency_floor = (
            lambda *args, **kwargs: BudgetAllocator.optimize_with_efficiency_floor(
                mock_allocator, *args, **kwargs
            )
        )

        result = mock_allocator.optimize_with_efficiency_floor(
            total_budget=100000,
            efficiency_metric="roi",
            efficiency_target=2.0,
        )

        # Standard fields
        assert result.optimal_allocation is not None
        assert result.total_budget is not None
        assert result.expected_response is not None
        assert result.success is not None

        # Efficiency floor fields
        assert result.unallocated_budget is not None
        assert result.efficiency_target is not None
        assert result.efficiency_metric is not None
        assert result.achieved_efficiency is not None
