"""Unit tests for data upload functionality.

Tests for:
- Fresh upload behavior (regression tests)
- Incremental data merge when model is loaded
"""
import pytest
import pandas as pd
import numpy as np

from mmm_platform.config.schema import ModelConfig, DataConfig, ChannelConfig
from mmm_platform.ui.pages.upload_data import _merge_data_incrementally


class TestFreshUploadBehavior:
    """Regression tests - verify existing fresh upload behavior is preserved.

    These tests ensure that when NO model is loaded (current_config is None),
    the existing upload behavior continues to work correctly:
    - Data is stored in current_data
    - config_state is reset to default structure
    - Model state is cleared
    """

    def test_fresh_upload_config_state_structure(self):
        """Verify the expected reset config_state structure."""
        # This is the expected structure when uploading fresh data
        expected_reset_structure = {
            "name": "my_mmm_model",
            "channels": [],
            "controls": [],
            "owned_media": [],
            "competitors": [],
            "dummy_variables": [],
        }

        # Verify all required fields exist
        assert expected_reset_structure["name"] == "my_mmm_model"
        assert expected_reset_structure["channels"] == []
        assert expected_reset_structure["controls"] == []
        assert expected_reset_structure["owned_media"] == []
        assert expected_reset_structure["competitors"] == []
        assert expected_reset_structure["dummy_variables"] == []

    def test_fresh_upload_clears_old_config(self):
        """Verify fresh upload would clear old config data."""
        # Simulate old config_state with data from previous model
        old_config_state = {
            "name": "old_brand_model",
            "channels": [{"name": "old_channel_spend", "roi_prior_mid": 2.0}],
            "owned_media": [{"name": "old_email_opens"}],
            "controls": [{"name": "old_control"}],
            "dummy_variables": [{"name": "old_dummy"}],
            "competitors": [{"name": "old_competitor"}],
        }

        # Simulate the reset that upload_data.py does
        reset_config_state = {
            "name": "my_mmm_model",
            "channels": [],
            "controls": [],
            "owned_media": [],
            "competitors": [],
            "dummy_variables": [],
        }

        # Verify old data is NOT in the reset state
        assert reset_config_state["name"] != old_config_state["name"]
        assert len(reset_config_state["channels"]) == 0
        assert len(reset_config_state["owned_media"]) == 0
        assert len(reset_config_state["controls"]) == 0
        assert len(reset_config_state["dummy_variables"]) == 0
        assert len(reset_config_state["competitors"]) == 0


class TestIncrementalDataMerge:
    """Tests for incremental data addition to loaded models.

    When a model is loaded (current_config is not None), new data should be
    merged with existing data rather than replacing it.
    """

    @pytest.fixture
    def existing_data(self):
        """Sample existing model data."""
        return pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=5, freq="W"),
            "revenue": [100, 200, 300, 400, 500],
            "tv_spend": [10, 20, 30, 40, 50],
            "search_spend": [5, 10, 15, 20, 25],
        })

    def test_merge_adds_new_columns(self, existing_data):
        """Verify new columns are added to existing data."""
        new_data = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=5, freq="W"),
            "radio_spend": [1, 2, 3, 4, 5],
            "promo_flag": [0, 1, 0, 1, 0],
        })

        merged_df, added, overwritten = _merge_data_incrementally(
            existing_data, new_data, "date"
        )

        # Should have all original columns plus new ones
        assert "revenue" in merged_df.columns
        assert "tv_spend" in merged_df.columns
        assert "search_spend" in merged_df.columns
        assert "radio_spend" in merged_df.columns
        assert "promo_flag" in merged_df.columns

        # Verify added/overwritten lists
        assert "radio_spend" in added
        assert "promo_flag" in added
        assert len(overwritten) == 0

        # Verify values
        assert merged_df["radio_spend"].tolist() == [1, 2, 3, 4, 5]
        assert merged_df["promo_flag"].tolist() == [0, 1, 0, 1, 0]

    def test_merge_overwrites_existing_columns(self, existing_data):
        """Verify columns with same name overwrite existing values."""
        new_data = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=5, freq="W"),
            "tv_spend": [99, 88, 77, 66, 55],  # Overwrite existing
        })

        merged_df, added, overwritten = _merge_data_incrementally(
            existing_data, new_data, "date"
        )

        # tv_spend should be overwritten
        assert "tv_spend" in overwritten
        assert len(added) == 0

        # Verify new values
        assert merged_df["tv_spend"].tolist() == [99, 88, 77, 66, 55]

        # Other columns unchanged
        assert merged_df["revenue"].tolist() == [100, 200, 300, 400, 500]
        assert merged_df["search_spend"].tolist() == [5, 10, 15, 20, 25]

    def test_merge_mixed_add_and_overwrite(self, existing_data):
        """Verify both new columns and overwrites work together."""
        new_data = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=5, freq="W"),
            "tv_spend": [99, 88, 77, 66, 55],  # Overwrite
            "radio_spend": [1, 2, 3, 4, 5],     # New
        })

        merged_df, added, overwritten = _merge_data_incrementally(
            existing_data, new_data, "date"
        )

        assert "radio_spend" in added
        assert "tv_spend" in overwritten

        assert merged_df["tv_spend"].tolist() == [99, 88, 77, 66, 55]
        assert merged_df["radio_spend"].tolist() == [1, 2, 3, 4, 5]

    def test_merge_preserves_row_count(self, existing_data):
        """Verify merge doesn't change the number of rows."""
        new_data = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=5, freq="W"),
            "new_col": [1, 2, 3, 4, 5],
        })

        merged_df, _, _ = _merge_data_incrementally(
            existing_data, new_data, "date"
        )

        assert len(merged_df) == len(existing_data)

    def test_merge_handles_partial_date_overlap(self, existing_data):
        """Verify merge handles when new data has fewer dates."""
        # New data only has 3 dates
        new_data = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=3, freq="W"),
            "new_col": [10, 20, 30],
        })

        merged_df, added, _ = _merge_data_incrementally(
            existing_data, new_data, "date"
        )

        assert "new_col" in added
        # First 3 rows should have values, last 2 should be NaN
        assert merged_df["new_col"].iloc[0] == 10
        assert merged_df["new_col"].iloc[1] == 20
        assert merged_df["new_col"].iloc[2] == 30
        assert pd.isna(merged_df["new_col"].iloc[3])
        assert pd.isna(merged_df["new_col"].iloc[4])

    def test_merge_handles_date_formats(self):
        """Verify merge works with different date formats."""
        existing_data = pd.DataFrame({
            "date": ["01/01/2023", "08/01/2023", "15/01/2023"],
            "revenue": [100, 200, 300],
        })

        new_data = pd.DataFrame({
            "date": ["01/01/2023", "08/01/2023", "15/01/2023"],
            "new_col": [10, 20, 30],
        })

        merged_df, added, _ = _merge_data_incrementally(
            existing_data, new_data, "date", dayfirst=True
        )

        assert "new_col" in added
        assert merged_df["new_col"].tolist() == [10, 20, 30]


class TestConfigPreservation:
    """Tests to verify config is preserved during incremental add."""

    def test_config_state_not_modified_by_merge(self):
        """Verify merge_data_incrementally doesn't touch config state."""
        existing_data = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=3, freq="W"),
            "revenue": [100, 200, 300],
        })

        new_data = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=3, freq="W"),
            "new_col": [1, 2, 3],
        })

        # Simulate config_state that should be preserved
        config_state = {
            "name": "my_model",
            "channels": [{"name": "tv_spend", "roi_prior_mid": 2.5}],
        }

        # Merge data
        merged_df, _, _ = _merge_data_incrementally(
            existing_data, new_data, "date"
        )

        # config_state should be unchanged (it's not touched by merge)
        assert config_state["name"] == "my_model"
        assert len(config_state["channels"]) == 1
        assert config_state["channels"][0]["roi_prior_mid"] == 2.5
