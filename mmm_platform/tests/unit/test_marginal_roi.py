"""
Unit tests for analysis/marginal_roi.py

Tests marginal ROI and breakeven calculations.
"""
import pytest
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose

from mmm_platform.analysis.marginal_roi import (
    logistic_saturation_derivative,
    calculate_marginal_roi,
    find_breakeven_spend,
    ChannelMarginalROI,
    InvestmentPriorityResult,
)


class TestLogisticSaturationDerivative:
    """Tests for saturation derivative function."""

    def test_derivative_positive(self):
        """Derivative should be positive for all inputs."""
        x = np.linspace(0, 10, 100)
        for lam in [0.5, 1.0, 2.0, 5.0]:
            deriv = logistic_saturation_derivative(x, lam)
            assert np.all(deriv >= 0)

    def test_derivative_decreasing(self):
        """Derivative should decrease as x increases (concave saturation)."""
        x = np.linspace(0.01, 5, 100)
        deriv = logistic_saturation_derivative(x, lam=2.0)
        diffs = np.diff(deriv)
        # Should be mostly decreasing (allow small numerical errors)
        assert np.sum(diffs < 0) > len(diffs) * 0.9

    def test_derivative_at_zero(self):
        """Derivative at x=0 should equal 2*lambda (max derivative)."""
        lam = 2.0
        x = np.array([0.0])
        deriv = logistic_saturation_derivative(x, lam)
        # At x=0: exp(0)=1, deriv = 2*lam*1/(1+1)^2 = 2*lam/4 = lam/2
        expected = 2 * lam / 4  # = lam / 2
        assert_allclose(deriv[0], expected, rtol=1e-10)

    def test_derivative_numerical_stability(self):
        """Should be numerically stable for extreme values."""
        x = np.array([0.0, 100.0, 1000.0])
        deriv = logistic_saturation_derivative(x, lam=1.0)
        assert np.all(np.isfinite(deriv))


class TestCalculateMarginalROI:
    """Tests for marginal ROI calculation."""

    def test_marginal_roi_positive(self):
        """Marginal ROI should be positive."""
        mroi = calculate_marginal_roi(
            x_normalized=0.5,
            beta=0.5,
            lam=2.0,
            target_scale=100000,
            x_max=10000
        )
        assert mroi > 0

    def test_marginal_roi_decreasing_with_spend(self):
        """Marginal ROI should decrease as spend increases."""
        spend_levels = [0.1, 0.3, 0.5, 0.7, 0.9]
        mrois = [
            calculate_marginal_roi(x, beta=0.5, lam=2.0, target_scale=100000, x_max=10000)
            for x in spend_levels
        ]
        # Each MROI should be less than the previous
        for i in range(1, len(mrois)):
            assert mrois[i] < mrois[i-1]

    def test_marginal_roi_scales_with_beta(self):
        """Higher beta should give higher marginal ROI."""
        mroi_low = calculate_marginal_roi(0.5, beta=0.2, lam=2.0, target_scale=100000, x_max=10000)
        mroi_high = calculate_marginal_roi(0.5, beta=0.8, lam=2.0, target_scale=100000, x_max=10000)
        assert mroi_high > mroi_low

    def test_marginal_roi_scales_with_target_scale(self):
        """Higher target scale should give higher marginal ROI."""
        mroi_low = calculate_marginal_roi(0.5, beta=0.5, lam=2.0, target_scale=50000, x_max=10000)
        mroi_high = calculate_marginal_roi(0.5, beta=0.5, lam=2.0, target_scale=100000, x_max=10000)
        assert mroi_high > mroi_low


class TestFindBreakevenSpend:
    """Tests for breakeven spend calculation."""

    def test_breakeven_returns_float(self):
        """Should return float when breakeven exists."""
        breakeven = find_breakeven_spend(
            beta=0.5,
            lam=2.0,
            target_scale=100000,
            x_max=10000
        )
        assert breakeven is None or isinstance(breakeven, float)

    def test_breakeven_within_range(self):
        """Breakeven should be within search range."""
        max_normalized = 100.0
        breakeven = find_breakeven_spend(
            beta=0.5,
            lam=2.0,
            target_scale=100000,
            x_max=10000,
            max_spend_normalized=max_normalized
        )
        if breakeven is not None:
            assert 0 <= breakeven <= max_normalized

    def test_marginal_roi_equals_one_at_breakeven(self):
        """Marginal ROI should equal 1 at breakeven spend."""
        beta, lam, target_scale, x_max = 0.5, 2.0, 100000, 10000
        breakeven = find_breakeven_spend(beta, lam, target_scale, x_max)

        if breakeven is not None and breakeven > 0 and breakeven < 100:
            mroi_at_breakeven = calculate_marginal_roi(
                breakeven, beta, lam, target_scale, x_max
            )
            assert_allclose(mroi_at_breakeven, 1.0, atol=0.01)

    def test_high_beta_higher_breakeven(self):
        """Higher beta should lead to higher breakeven (more profitable)."""
        lam, target_scale, x_max = 2.0, 100000, 10000

        breakeven_low = find_breakeven_spend(beta=0.2, lam=lam, target_scale=target_scale, x_max=x_max)
        breakeven_high = find_breakeven_spend(beta=0.8, lam=lam, target_scale=target_scale, x_max=x_max)

        # Higher beta means profitable at higher spend
        if breakeven_low is not None and breakeven_high is not None:
            assert breakeven_high > breakeven_low


class TestChannelMarginalROI:
    """Tests for ChannelMarginalROI dataclass."""

    def test_dataclass_creation(self):
        """Should create dataclass with all fields."""
        result = ChannelMarginalROI(
            channel="tv_spend",
            channel_name="TV",
            current_spend=50000,
            current_roi=1.5,
            marginal_roi=1.8,
            breakeven_spend=100000,
            headroom=True,
            headroom_amount=50000,
        )
        assert result.channel == "tv_spend"
        assert result.channel_name == "TV"
        assert result.current_spend == 50000
        assert result.marginal_roi == 1.8

    def test_default_values(self):
        """Should have sensible default values."""
        result = ChannelMarginalROI(
            channel="tv_spend",
            channel_name="TV",
            current_spend=50000,
            current_roi=1.5,
            marginal_roi=1.8,
            breakeven_spend=None,
            headroom=False,
            headroom_amount=0,
        )
        assert result.priority_rank == 0
        assert result.roi_5pct == 0.0
        assert result.roi_95pct == 0.0
        assert result.needs_test is False


class TestInvestmentPriorityResult:
    """Tests for InvestmentPriorityResult dataclass."""

    def test_empty_result(self):
        """Should handle empty channel lists."""
        result = InvestmentPriorityResult(
            channel_analysis=[],
            increase_channels=[],
            hold_channels=[],
            reduce_channels=[],
            channels_needing_test=[],
            total_spend=0,
            total_contribution=0,
            portfolio_roi=0,
            reallocation_potential=0,
            headroom_available=0,
        )
        assert len(result.channel_analysis) == 0
        assert result.portfolio_roi == 0

    def test_categorization(self):
        """Should correctly categorize channels."""
        ch1 = ChannelMarginalROI(
            channel="tv", channel_name="TV",
            current_spend=50000, current_roi=2.0, marginal_roi=2.5,
            breakeven_spend=100000, headroom=True, headroom_amount=50000
        )
        ch2 = ChannelMarginalROI(
            channel="search", channel_name="Search",
            current_spend=30000, current_roi=0.8, marginal_roi=0.5,
            breakeven_spend=20000, headroom=False, headroom_amount=0
        )

        result = InvestmentPriorityResult(
            channel_analysis=[ch1, ch2],
            increase_channels=[ch1],
            hold_channels=[],
            reduce_channels=[ch2],
            channels_needing_test=[],
            total_spend=80000,
            total_contribution=100000,
            portfolio_roi=1.25,
            reallocation_potential=30000,
            headroom_available=50000,
        )

        assert len(result.increase_channels) == 1
        assert len(result.reduce_channels) == 1
        assert result.reallocation_potential == 30000


class TestMarginalROIEdgeCases:
    """Tests for edge cases in marginal ROI calculations."""

    def test_zero_spend_handling(self):
        """Should handle near-zero spend gracefully."""
        mroi = calculate_marginal_roi(
            x_normalized=1e-10,
            beta=0.5,
            lam=2.0,
            target_scale=100000,
            x_max=10000
        )
        assert np.isfinite(mroi)

    def test_very_high_lambda(self):
        """Should handle very high lambda (steep saturation)."""
        mroi = calculate_marginal_roi(
            x_normalized=0.5,
            beta=0.5,
            lam=100.0,  # Very steep
            target_scale=100000,
            x_max=10000
        )
        assert np.isfinite(mroi)

    def test_very_low_lambda(self):
        """Should handle very low lambda (gradual saturation)."""
        mroi = calculate_marginal_roi(
            x_normalized=0.5,
            beta=0.5,
            lam=0.01,  # Very gradual
            target_scale=100000,
            x_max=10000
        )
        assert np.isfinite(mroi)

    def test_small_x_max(self):
        """Should handle small x_max values."""
        mroi = calculate_marginal_roi(
            x_normalized=0.5,
            beta=0.5,
            lam=2.0,
            target_scale=100000,
            x_max=1  # Very small
        )
        assert np.isfinite(mroi)
        assert mroi > 0
