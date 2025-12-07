"""
Tests for disaggregation reconciliation validation functions.
"""

import pytest
import pandas as pd
import numpy as np

from mmm_platform.analysis.export import (
    get_reconcilable_columns,
    validate_disaggregation_reconciliation,
    get_reconciliation_summary,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def base_df():
    """Base DataFrame for testing."""
    return pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=4, freq="W"),
        "brand": ["TestBrand"] * 4,
        "decomp": ["channel_a", "channel_a", "channel_b", "channel_b"],
        "kpi_revenue": [100.0, 200.0, 150.0, 250.0],
        "spend": [50.0, 100.0, 75.0, 125.0],
    })


@pytest.fixture
def matching_disagg_df():
    """Disaggregated DataFrame that matches base sums."""
    return pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=8, freq="W").repeat(1)[:8],
        "brand": ["TestBrand"] * 8,
        "decomp": ["channel_a"] * 4 + ["channel_b"] * 4,
        "granular_name": ["entity_1", "entity_2", "entity_1", "entity_2"] * 2,
        "kpi_revenue": [50.0, 50.0, 100.0, 100.0, 75.0, 75.0, 125.0, 125.0],
        "spend": [25.0, 25.0, 50.0, 50.0, 37.5, 37.5, 62.5, 62.5],
    })


@pytest.fixture
def mismatched_disagg_df():
    """Disaggregated DataFrame with sums that don't match base."""
    return pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=8, freq="W").repeat(1)[:8],
        "brand": ["TestBrand"] * 8,
        "decomp": ["channel_a"] * 4 + ["channel_b"] * 4,
        "granular_name": ["entity_1", "entity_2", "entity_1", "entity_2"] * 2,
        # kpi_revenue sums to 650 instead of 700 (base)
        "kpi_revenue": [40.0, 50.0, 90.0, 100.0, 70.0, 75.0, 100.0, 125.0],
        "spend": [25.0, 25.0, 50.0, 50.0, 37.5, 37.5, 62.5, 62.5],
    })


# =============================================================================
# get_reconcilable_columns Tests
# =============================================================================

class TestGetReconcilableColumns:
    """Tests for get_reconcilable_columns function."""

    def test_finds_numeric_columns(self, base_df, matching_disagg_df):
        """Finds numeric columns present in both DataFrames."""
        cols = get_reconcilable_columns(base_df, matching_disagg_df)

        assert "kpi_revenue" in cols
        assert "spend" in cols

    def test_excludes_non_numeric(self, base_df, matching_disagg_df):
        """Excludes non-numeric columns."""
        cols = get_reconcilable_columns(base_df, matching_disagg_df)

        assert "brand" not in cols
        assert "decomp" not in cols

    def test_excludes_date_columns(self):
        """Excludes date-like columns."""
        base = pd.DataFrame({
            "date": [1, 2, 3],
            "week_number": [1, 2, 3],
            "month": [1, 1, 2],
            "year": [2023, 2023, 2023],
            "kpi_value": [100, 200, 300],
        })
        disagg = base.copy()

        cols = get_reconcilable_columns(base, disagg)

        assert "kpi_value" in cols
        assert "date" not in cols
        assert "week_number" not in cols
        assert "month" not in cols
        assert "year" not in cols

    def test_only_columns_in_both(self):
        """Only returns columns present in both DataFrames."""
        base = pd.DataFrame({"kpi_a": [1, 2], "kpi_b": [3, 4]})
        disagg = pd.DataFrame({"kpi_a": [0.5, 0.5, 1, 1], "kpi_c": [1, 1, 2, 2]})

        cols = get_reconcilable_columns(base, disagg)

        assert cols == ["kpi_a"]


# =============================================================================
# validate_disaggregation_reconciliation Tests
# =============================================================================

class TestValidateDisaggregationReconciliation:
    """Tests for validate_disaggregation_reconciliation function."""

    def test_matching_sums_ok(self, base_df, matching_disagg_df):
        """Returns ok status when sums match."""
        results = validate_disaggregation_reconciliation(
            base_df, matching_disagg_df
        )

        for result in results:
            assert result["status"] == "ok"
            assert result["diff_pct"] < 0.01

    def test_mismatched_sums_detected(self, base_df, mismatched_disagg_df):
        """Detects mismatched sums."""
        results = validate_disaggregation_reconciliation(
            base_df, mismatched_disagg_df
        )

        # Find the kpi_revenue result
        kpi_result = next(r for r in results if r["column"] == "kpi_revenue")

        assert kpi_result["status"] != "ok"
        assert kpi_result["diff_pct"] > 0.01

    def test_result_structure(self, base_df, matching_disagg_df):
        """Results have expected structure."""
        results = validate_disaggregation_reconciliation(
            base_df, matching_disagg_df
        )

        for result in results:
            assert "column" in result
            assert "base_sum" in result
            assert "disagg_sum" in result
            assert "diff_pct" in result
            assert "status" in result
            assert result["status"] in ["ok", "warning", "error"]

    def test_custom_tolerance(self, base_df, mismatched_disagg_df):
        """Respects custom tolerance."""
        # With very high tolerance, should be ok
        results = validate_disaggregation_reconciliation(
            base_df, mismatched_disagg_df, tolerance=0.50
        )

        for result in results:
            assert result["status"] == "ok"

    def test_custom_columns(self, base_df, matching_disagg_df):
        """Checks only specified columns."""
        results = validate_disaggregation_reconciliation(
            base_df, matching_disagg_df, numeric_cols=["kpi_revenue"]
        )

        assert len(results) == 1
        assert results[0]["column"] == "kpi_revenue"

    def test_handles_zero_base_sum(self):
        """Handles zero base sum correctly."""
        base = pd.DataFrame({"kpi_zero": [0.0, 0.0]})
        disagg = pd.DataFrame({"kpi_zero": [0.0, 0.0, 0.0, 0.0]})

        results = validate_disaggregation_reconciliation(
            base, disagg, numeric_cols=["kpi_zero"]
        )

        assert results[0]["status"] == "ok"
        assert results[0]["diff_pct"] == 0.0

    def test_handles_zero_base_nonzero_disagg(self):
        """Detects when base is zero but disagg is not."""
        base = pd.DataFrame({"kpi_value": [0.0, 0.0]})
        disagg = pd.DataFrame({"kpi_value": [10.0, 10.0]})

        results = validate_disaggregation_reconciliation(
            base, disagg, numeric_cols=["kpi_value"]
        )

        assert results[0]["status"] == "error"
        assert results[0]["diff_pct"] == float('inf')


# =============================================================================
# get_reconciliation_summary Tests
# =============================================================================

class TestGetReconciliationSummary:
    """Tests for get_reconciliation_summary function."""

    def test_all_ok_summary(self):
        """Summary when all columns reconcile."""
        results = [
            {"column": "kpi_a", "status": "ok", "base_sum": 100, "disagg_sum": 100, "diff_pct": 0},
            {"column": "kpi_b", "status": "ok", "base_sum": 200, "disagg_sum": 200, "diff_pct": 0},
        ]

        summary = get_reconciliation_summary(results)

        assert summary["total_columns"] == 2
        assert summary["ok_count"] == 2
        assert summary["warning_count"] == 0
        assert summary["error_count"] == 0
        assert summary["all_ok"] is True
        assert len(summary["issues"]) == 0

    def test_mixed_status_summary(self):
        """Summary with mixed statuses."""
        results = [
            {"column": "kpi_a", "status": "ok", "base_sum": 100, "disagg_sum": 100, "diff_pct": 0},
            {"column": "kpi_b", "status": "warning", "base_sum": 200, "disagg_sum": 195, "diff_pct": 0.025},
            {"column": "kpi_c", "status": "error", "base_sum": 300, "disagg_sum": 250, "diff_pct": 0.167},
        ]

        summary = get_reconciliation_summary(results)

        assert summary["total_columns"] == 3
        assert summary["ok_count"] == 1
        assert summary["warning_count"] == 1
        assert summary["error_count"] == 1
        assert summary["all_ok"] is False
        assert len(summary["issues"]) == 2

    def test_empty_results(self):
        """Handles empty results."""
        summary = get_reconciliation_summary([])

        assert summary["total_columns"] == 0
        assert summary["all_ok"] is True
        assert len(summary["issues"]) == 0
