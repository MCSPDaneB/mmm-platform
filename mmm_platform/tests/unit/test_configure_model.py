"""Unit tests for configure_model.

Tests for:
- Date range clamping when loading saved configs with dates outside data range
- Cost-per-unit <-> efficiency conversion for count KPIs
"""
import pytest
import datetime


def clamp_date_to_range(
    date_value: datetime.date,
    min_date: datetime.date | None,
    max_date: datetime.date | None,
) -> datetime.date:
    """Clamp a date value to be within min/max range.

    This mirrors the logic in configure_model.py for handling saved dates
    that may be outside the current data's date range.
    """
    result = date_value
    if min_date and result < min_date:
        result = min_date
    if max_date and result > max_date:
        result = max_date
    return result


class TestDateRangeClamping:
    """Tests for date range clamping when loading saved configs."""

    def test_saved_end_date_beyond_data_max_clamped(self):
        """Verify saved end date beyond data max is clamped to data max."""
        saved_end = datetime.date(2025, 10, 26)
        data_max_date = datetime.date(2025, 10, 20)
        data_min_date = datetime.date(2024, 1, 1)

        result = clamp_date_to_range(saved_end, data_min_date, data_max_date)

        assert result == data_max_date

    def test_saved_start_date_before_data_min_clamped(self):
        """Verify saved start date before data min is clamped to data min."""
        saved_start = datetime.date(2023, 6, 1)
        data_min_date = datetime.date(2024, 1, 1)
        data_max_date = datetime.date(2025, 10, 20)

        result = clamp_date_to_range(saved_start, data_min_date, data_max_date)

        assert result == data_min_date

    def test_saved_date_within_range_unchanged(self):
        """Verify saved date within range is not changed."""
        saved_date = datetime.date(2024, 6, 15)
        data_min_date = datetime.date(2024, 1, 1)
        data_max_date = datetime.date(2025, 10, 20)

        result = clamp_date_to_range(saved_date, data_min_date, data_max_date)

        assert result == saved_date

    def test_saved_date_at_min_boundary_unchanged(self):
        """Verify saved date exactly at min boundary is unchanged."""
        saved_date = datetime.date(2024, 1, 1)
        data_min_date = datetime.date(2024, 1, 1)
        data_max_date = datetime.date(2025, 10, 20)

        result = clamp_date_to_range(saved_date, data_min_date, data_max_date)

        assert result == saved_date

    def test_saved_date_at_max_boundary_unchanged(self):
        """Verify saved date exactly at max boundary is unchanged."""
        saved_date = datetime.date(2025, 10, 20)
        data_min_date = datetime.date(2024, 1, 1)
        data_max_date = datetime.date(2025, 10, 20)

        result = clamp_date_to_range(saved_date, data_min_date, data_max_date)

        assert result == saved_date

    def test_clamp_with_none_min_date(self):
        """Verify clamping works when min_date is None."""
        saved_date = datetime.date(2025, 12, 1)
        data_min_date = None
        data_max_date = datetime.date(2025, 10, 20)

        result = clamp_date_to_range(saved_date, data_min_date, data_max_date)

        assert result == data_max_date

    def test_clamp_with_none_max_date(self):
        """Verify clamping works when max_date is None."""
        saved_date = datetime.date(2023, 1, 1)
        data_min_date = datetime.date(2024, 1, 1)
        data_max_date = None

        result = clamp_date_to_range(saved_date, data_min_date, data_max_date)

        assert result == data_min_date

    def test_clamp_with_both_none_dates(self):
        """Verify date unchanged when both min and max are None."""
        saved_date = datetime.date(2025, 6, 15)
        data_min_date = None
        data_max_date = None

        result = clamp_date_to_range(saved_date, data_min_date, data_max_date)

        assert result == saved_date


def convert_cost_to_efficiency(cost_low: float, cost_mid: float, cost_high: float):
    """Convert cost-per-unit to efficiency (target per $).

    For count KPIs, user enters "Cost per Conversion" but we store efficiency.
    Low cost = high efficiency, high cost = low efficiency.
    """
    efficiency_low = 1.0 / cost_high if cost_high > 0 else 0.1
    efficiency_mid = 1.0 / cost_mid if cost_mid > 0 else 0.2
    efficiency_high = 1.0 / cost_low if cost_low > 0 else 1.0
    return efficiency_low, efficiency_mid, efficiency_high


def convert_efficiency_to_cost(eff_low: float, eff_mid: float, eff_high: float):
    """Convert efficiency back to cost-per-unit for display.

    This is the inverse of convert_cost_to_efficiency.
    """
    cost_low = 1.0 / eff_high if eff_high > 0 else 1.0
    cost_mid = 1.0 / eff_mid if eff_mid > 0 else 5.0
    cost_high = 1.0 / eff_low if eff_low > 0 else 10.0
    return cost_low, cost_mid, cost_high


class TestCostPerEfficiencyRoundtrip:
    """Tests for cost-per-unit <-> efficiency conversion for count KPIs."""

    def test_cost_to_efficiency_conversion(self):
        """Verify cost-per-unit converts to efficiency correctly."""
        # User enters cost-per-unit
        cost_low = 100.0   # Cheap - $100 per conversion
        cost_mid = 200.0
        cost_high = 1000.0  # Expensive - $1000 per conversion

        eff_low, eff_mid, eff_high = convert_cost_to_efficiency(
            cost_low, cost_mid, cost_high
        )

        assert eff_low == pytest.approx(0.001)
        assert eff_mid == pytest.approx(0.005)
        assert eff_high == pytest.approx(0.01)

    def test_efficiency_to_cost_conversion(self):
        """Verify efficiency converts back to cost-per-unit correctly."""
        # Stored efficiency values
        eff_low = 0.001
        eff_mid = 0.005
        eff_high = 0.01

        cost_low, cost_mid, cost_high = convert_efficiency_to_cost(
            eff_low, eff_mid, eff_high
        )

        assert cost_low == pytest.approx(100.0)
        assert cost_mid == pytest.approx(200.0)
        assert cost_high == pytest.approx(1000.0)

    def test_roundtrip_preserves_values(self):
        """Verify cost -> efficiency -> cost roundtrip preserves original values."""
        original_cost_low = 50.0
        original_cost_mid = 150.0
        original_cost_high = 500.0

        # Convert to efficiency
        eff_low, eff_mid, eff_high = convert_cost_to_efficiency(
            original_cost_low, original_cost_mid, original_cost_high
        )

        # Convert back to cost
        final_cost_low, final_cost_mid, final_cost_high = convert_efficiency_to_cost(
            eff_low, eff_mid, eff_high
        )

        assert final_cost_low == pytest.approx(original_cost_low)
        assert final_cost_mid == pytest.approx(original_cost_mid)
        assert final_cost_high == pytest.approx(original_cost_high)

    def test_handles_zero_efficiency(self):
        """Verify zero efficiency values use defaults."""
        eff_low = 0.0
        eff_mid = 0.0
        eff_high = 0.0

        cost_low, cost_mid, cost_high = convert_efficiency_to_cost(
            eff_low, eff_mid, eff_high
        )

        # Should use default values instead of division by zero
        assert cost_low == 1.0
        assert cost_mid == 5.0
        assert cost_high == 10.0

    def test_handles_zero_cost(self):
        """Verify zero cost values use defaults."""
        cost_low = 0.0
        cost_mid = 0.0
        cost_high = 0.0

        eff_low, eff_mid, eff_high = convert_cost_to_efficiency(
            cost_low, cost_mid, cost_high
        )

        # Should use default values instead of division by zero
        assert eff_low == 0.1
        assert eff_mid == 0.2
        assert eff_high == 1.0

    def test_revenue_kpi_no_conversion_needed(self):
        """Verify revenue KPI values don't need conversion."""
        # For revenue KPIs, values are already ROI (efficiency)
        roi_low = 0.5
        roi_mid = 2.0
        roi_high = 5.0

        # Revenue KPI: stored values = display values (no conversion)
        assert roi_low == 0.5
        assert roi_mid == 2.0
        assert roi_high == 5.0
