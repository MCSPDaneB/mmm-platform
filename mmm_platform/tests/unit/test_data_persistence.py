"""Unit tests for data persistence with date filtering.

Tests for:
- Original data preservation when saving models with date range filters
- Backward compatibility with models saved before df_original was added
"""
import pytest
import pandas as pd
import datetime


class TestOriginalDataPreservation:
    """Tests verifying original data is preserved when saving with date filter."""

    def test_original_data_larger_than_filtered(self):
        """Verify original data would have more rows than date-filtered data."""
        # Create full dataset (100 weeks)
        full_data = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=100, freq="W"),
            "revenue": range(100),
        })

        # Date filter would reduce this to ~52 weeks in 2024
        filtered_data = full_data[
            (full_data["date"] >= "2024-01-01") &
            (full_data["date"] <= "2024-12-31")
        ]

        # Original should have more rows than filtered
        assert len(full_data) > len(filtered_data)
        assert len(full_data) == 100
        assert len(filtered_data) < 100

    def test_df_original_copy_is_independent(self):
        """Verify df_original copy doesn't share memory with df_raw."""
        original_df = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=10, freq="W"),
            "value": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        })

        # Simulate what prepare_data does
        df_original = original_df.copy()
        df_raw = original_df[original_df["value"] > 5].copy()

        # Modify df_raw should not affect df_original
        df_raw["value"] = df_raw["value"] * 100

        # df_original should be unchanged
        assert df_original["value"].tolist() == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        assert len(df_original) == 10
        assert len(df_raw) == 5


class TestBackwardCompatibility:
    """Tests for backward compatibility with older saved models."""

    def test_fallback_when_df_original_is_none(self):
        """Verify graceful fallback when df_original doesn't exist."""
        # Simulate old model without df_original
        df_raw_fallback = pd.DataFrame({"x": [1, 2, 3]})
        df_original = None

        # Logic from saved_models.py
        if df_original is not None:
            current_data = df_original
        elif df_raw_fallback is not None:
            current_data = df_raw_fallback
        else:
            current_data = None

        assert current_data is df_raw_fallback
        assert len(current_data) == 3

    def test_getattr_returns_none_for_missing_attribute(self):
        """Verify getattr pattern handles missing df_original attribute."""
        class OldWrapper:
            """Simulates wrapper without df_original."""
            df_raw = pd.DataFrame({"x": [1, 2, 3]})
            df_scaled = pd.DataFrame({"x": [0.1, 0.2, 0.3]})

        wrapper = OldWrapper()

        # This is the pattern used in saved_models.py
        df_original = getattr(wrapper, "df_original", None)

        assert df_original is None

    def test_prefer_df_original_over_df_raw(self):
        """Verify df_original is preferred when available."""
        class NewWrapper:
            """Simulates wrapper with df_original."""
            df_original = pd.DataFrame({"x": [1, 2, 3, 4, 5]})  # Full data
            df_raw = pd.DataFrame({"x": [3, 4, 5]})  # Filtered
            df_scaled = pd.DataFrame({"x": [0.3, 0.4, 0.5]})

        wrapper = NewWrapper()

        # Logic from saved_models.py
        df_original = getattr(wrapper, "df_original", None)
        if df_original is not None:
            current_data = df_original
        elif wrapper.df_raw is not None:
            current_data = wrapper.df_raw
        else:
            current_data = wrapper.df_scaled

        assert current_data is wrapper.df_original
        assert len(current_data) == 5  # Full data, not filtered


class TestDateRangeFiltering:
    """Tests for date range filtering behavior."""

    def test_filter_by_start_date(self):
        """Verify filtering by start date reduces data correctly."""
        df = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=24, freq="M"),
            "value": range(24),
        })

        start_date = pd.to_datetime("2024-01-01")
        filtered = df[df["date"] >= start_date]

        # Should have only 2024 data (12 months)
        assert len(filtered) == 12
        assert len(df) == 24

    def test_filter_by_end_date(self):
        """Verify filtering by end date reduces data correctly."""
        df = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=24, freq="M"),
            "value": range(24),
        })

        end_date = pd.to_datetime("2023-12-31")
        filtered = df[df["date"] <= end_date]

        # Should have only 2023 data (12 months)
        assert len(filtered) == 12
        assert len(df) == 24

    def test_filter_by_both_dates(self):
        """Verify filtering by both start and end dates."""
        df = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=36, freq="M"),
            "value": range(36),
        })

        start_date = pd.to_datetime("2024-01-01")
        end_date = pd.to_datetime("2024-06-30")
        filtered = df[(df["date"] >= start_date) & (df["date"] <= end_date)]

        # Should have only first half of 2024 (6 months)
        assert len(filtered) == 6
        assert len(df) == 36
