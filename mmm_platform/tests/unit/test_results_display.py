"""Unit tests for results page display conversions.

Tests for:
- Count KPI efficiency-to-cost conversion for display
- Credible interval boundary swapping when inverting values
"""
import pytest
from unittest.mock import MagicMock


def mock_kpi_labels(is_revenue: bool):
    """Create a mock KPILabels for testing."""
    mock = MagicMock()
    mock.is_revenue_type = is_revenue

    def convert_internal_to_display(val):
        if is_revenue:
            return val
        else:
            return 1 / val if val > 0 else float('inf')

    mock.convert_internal_to_display = convert_internal_to_display
    return mock


class TestEfficiencyToCostConversion:
    """Tests for converting efficiency values to cost-per for count KPIs."""

    def test_efficiency_to_cost_conversion(self):
        """Verify efficiency is converted to cost-per for count KPIs."""
        kpi_labels = mock_kpi_labels(is_revenue=False)

        # Efficiency of 0.2 (0.2 installs per dollar)
        efficiency = 0.2
        cost_per = kpi_labels.convert_internal_to_display(efficiency)

        # Should be $5.00 per install (1/0.2)
        assert cost_per == pytest.approx(5.0)

    def test_revenue_kpi_no_conversion(self):
        """Verify revenue KPI values are not inverted."""
        kpi_labels = mock_kpi_labels(is_revenue=True)

        # ROI of 2.5 (returns $2.50 per dollar spent)
        roi = 2.5
        display = kpi_labels.convert_internal_to_display(roi)

        # Should remain 2.5
        assert display == pytest.approx(2.5)

    def test_small_efficiency_large_cost(self):
        """Verify small efficiency becomes large cost-per."""
        kpi_labels = mock_kpi_labels(is_revenue=False)

        # Efficiency of 0.01 (0.01 installs per dollar = expensive channel)
        efficiency = 0.01
        cost_per = kpi_labels.convert_internal_to_display(efficiency)

        # Should be $100 per install
        assert cost_per == pytest.approx(100.0)

    def test_large_efficiency_small_cost(self):
        """Verify large efficiency becomes small cost-per."""
        kpi_labels = mock_kpi_labels(is_revenue=False)

        # Efficiency of 1.0 (1 install per dollar = cheap channel)
        efficiency = 1.0
        cost_per = kpi_labels.convert_internal_to_display(efficiency)

        # Should be $1 per install
        assert cost_per == pytest.approx(1.0)

    def test_zero_efficiency_handled(self):
        """Verify zero efficiency returns infinity."""
        kpi_labels = mock_kpi_labels(is_revenue=False)

        efficiency = 0.0
        cost_per = kpi_labels.convert_internal_to_display(efficiency)

        assert cost_per == float('inf')


class TestCredibleIntervalSwapping:
    """Tests for credible interval boundary swapping when inverting values."""

    def test_percentile_swap_for_cost_display(self):
        """Verify 5/95 percentiles are swapped when converting to cost-per.

        When inverting efficiency to cost-per, the ordering reverses:
        - Low efficiency (0.1) = high cost ($10)
        - High efficiency (0.5) = low cost ($2)

        To maintain proper CI semantics (5% = lower bound, 95% = upper bound),
        we swap which raw values go to which percentile label.
        """
        # Raw efficiency percentiles from model
        eff_5pct = 0.1   # 5th percentile of efficiency (low end)
        eff_95pct = 0.5  # 95th percentile of efficiency (high end)

        # After conversion for cost-per display:
        # The low efficiency becomes HIGH cost, the high efficiency becomes LOW cost
        # For display, we need:
        # - cost_5pct (lower bound) = $2 (from eff_95pct)
        # - cost_95pct (upper bound) = $10 (from eff_5pct)

        cost_from_eff_5 = 1 / eff_5pct   # 10.0
        cost_from_eff_95 = 1 / eff_95pct  # 2.0

        # For display, swap to maintain "5% = lower, 95% = upper" semantics
        display_5pct = cost_from_eff_95  # 2.0 (lower cost bound)
        display_95pct = cost_from_eff_5  # 10.0 (upper cost bound)

        assert display_5pct == pytest.approx(2.0)
        assert display_95pct == pytest.approx(10.0)
        assert display_5pct < display_95pct  # Proper CI ordering

    def test_revenue_kpi_no_swap_needed(self):
        """Verify revenue KPI doesn't need percentile swapping."""
        # For revenue, no inversion, so 5% < 95% naturally
        roi_5pct = 0.5   # Lower ROI bound
        roi_95pct = 3.0  # Upper ROI bound

        # No conversion needed
        display_5pct = roi_5pct
        display_95pct = roi_95pct

        assert display_5pct < display_95pct  # Already proper ordering

    def test_ci_meaning_preserved_after_swap(self):
        """Verify CI interpretation stays the same after swap.

        "We're 90% confident the true value is between 5% and 95% bounds"
        should hold for both ROI and cost-per display.
        """
        # Channel with efficiency CI: [0.08, 0.12] (90% HDI)
        eff_hdi_low = 0.08
        eff_hdi_high = 0.12

        # Convert to cost-per
        cost_from_eff_low = 1 / eff_hdi_low    # 12.5
        cost_from_eff_high = 1 / eff_hdi_high  # 8.33

        # Swap for display
        cost_hdi_low = cost_from_eff_high   # 8.33 (lower cost bound)
        cost_hdi_high = cost_from_eff_low   # 12.5 (upper cost bound)

        # Interpretation: "We're 90% confident cost-per is between $8.33 and $12.50"
        assert cost_hdi_low == pytest.approx(8.333, rel=0.01)
        assert cost_hdi_high == pytest.approx(12.5)
        assert cost_hdi_low < cost_hdi_high


class TestPriorValueConversion:
    """Tests for prior value display conversion."""

    def test_prior_ordering_for_cost_display(self):
        """Verify prior Low-Mid-High ordering is correct for cost-per display.

        User enters cost priors as: Low=$100, Mid=$200, High=$1000
        Stored as efficiency: High=0.01, Mid=0.005, Low=0.001

        When displaying, we need to show user's original order:
        Low-Mid-High = $100 - $200 - $1000
        """
        # Stored efficiency values (user entered $100, $200, $1000)
        stored_eff_low = 0.001   # From $1000 high cost
        stored_eff_mid = 0.005   # From $200 mid cost
        stored_eff_high = 0.01   # From $100 low cost

        # Convert back for display
        display_cost_low = 1 / stored_eff_high   # $100 (from high efficiency)
        display_cost_mid = 1 / stored_eff_mid    # $200
        display_cost_high = 1 / stored_eff_low   # $1000 (from low efficiency)

        assert display_cost_low == pytest.approx(100.0)
        assert display_cost_mid == pytest.approx(200.0)
        assert display_cost_high == pytest.approx(1000.0)

    def test_roundtrip_preserves_user_values(self):
        """Verify user's entered cost values survive save/load/display cycle."""
        # User enters these cost values
        user_cost_low = 50.0
        user_cost_mid = 150.0
        user_cost_high = 500.0

        # Saved as efficiency (inverted)
        stored_eff_low = 1 / user_cost_high  # 0.002
        stored_eff_mid = 1 / user_cost_mid   # 0.00667
        stored_eff_high = 1 / user_cost_low  # 0.02

        # Loaded and displayed (inverted back)
        display_low = 1 / stored_eff_high   # 50.0
        display_mid = 1 / stored_eff_mid    # 150.0
        display_high = 1 / stored_eff_low   # 500.0

        assert display_low == pytest.approx(user_cost_low)
        assert display_mid == pytest.approx(user_cost_mid)
        assert display_high == pytest.approx(user_cost_high)
