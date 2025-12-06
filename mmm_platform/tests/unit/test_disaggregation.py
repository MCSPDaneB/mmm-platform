"""Unit tests for disaggregation config and weights persistence."""
import pytest
import pandas as pd
from pathlib import Path
import json

from mmm_platform.model.persistence import (
    save_disaggregation_config,
    load_disaggregation_configs,
    load_disaggregation_weights,
    delete_disaggregation_config,
)


class TestDisaggregationWeightsPersistence:
    """Tests for saving/loading disaggregation weights with config."""

    def test_save_config_with_weights(self, tmp_path):
        """Verify weights DataFrame is saved alongside config."""
        config = {
            "id": "test_config_1",
            "name": "Test Config",
            "granular_name_cols": ["campaign", "region"],
            "date_column": "date",
            "weight_column": "impressions",
            "entity_to_channel_mapping": {"campaign_a": "paid_search"},
        }
        weights_df = pd.DataFrame({
            "date": ["2024-01-01", "2024-01-08"],
            "campaign": ["campaign_a", "campaign_a"],
            "region": ["US", "UK"],
            "impressions": [1000, 2000],
        })

        save_disaggregation_config(tmp_path, config, weighting_df=weights_df)

        # Check weights file exists
        weights_file = tmp_path / "disagg_weights_test_config_1.parquet"
        assert weights_file.exists()

        # Check weights content
        loaded_weights = pd.read_parquet(weights_file)
        assert len(loaded_weights) == 2
        assert "impressions" in loaded_weights.columns

    def test_load_config_with_weights(self, tmp_path):
        """Verify loading config also loads associated weights."""
        # Setup: save config with weights
        config = {
            "id": "test_config_2",
            "name": "Test Config 2",
            "granular_name_cols": ["campaign"],
            "date_column": "date",
            "weight_column": "clicks",
            "entity_to_channel_mapping": {},
        }
        weights_df = pd.DataFrame({
            "date": ["2024-01-01"],
            "campaign": ["test"],
            "clicks": [500],
        })
        save_disaggregation_config(tmp_path, config, weighting_df=weights_df)

        # Load weights
        loaded_weights = load_disaggregation_weights(tmp_path, "test_config_2")

        assert loaded_weights is not None
        assert len(loaded_weights) == 1
        assert loaded_weights["clicks"].iloc[0] == 500

    def test_delete_config_removes_weights_file(self, tmp_path):
        """Verify deleting config also deletes weights parquet."""
        config = {
            "id": "test_config_3",
            "name": "To Delete",
            "granular_name_cols": ["campaign"],
            "date_column": "date",
            "weight_column": "spend",
            "entity_to_channel_mapping": {},
        }
        weights_df = pd.DataFrame({"date": ["2024-01-01"], "campaign": ["x"], "spend": [100]})
        save_disaggregation_config(tmp_path, config, weighting_df=weights_df)

        weights_file = tmp_path / "disagg_weights_test_config_3.parquet"
        assert weights_file.exists()

        # Delete config
        delete_disaggregation_config(tmp_path, "test_config_3")

        # Weights file should be gone
        assert not weights_file.exists()

    def test_load_weights_returns_none_when_no_file(self, tmp_path):
        """Verify graceful handling when weights file doesn't exist."""
        # Create session_state.json but no weights file (backward compat)
        session_file = tmp_path / "session_state.json"
        session_file.write_text(json.dumps({
            "disaggregation": {
                "saved_configs": [{
                    "id": "old_config",
                    "name": "Old Config Without Weights",
                }]
            }
        }))

        loaded_weights = load_disaggregation_weights(tmp_path, "old_config")
        assert loaded_weights is None

    def test_save_config_without_weights_still_works(self, tmp_path):
        """Verify saving config without weights doesn't break."""
        config = {
            "id": "no_weights_config",
            "name": "No Weights",
            "granular_name_cols": ["campaign"],
            "date_column": "date",
            "weight_column": "impressions",
            "entity_to_channel_mapping": {},
        }

        # Save without weights (weighting_df=None)
        save_disaggregation_config(tmp_path, config, weighting_df=None)

        # Config should be saved
        configs = load_disaggregation_configs(tmp_path)
        assert len(configs) == 1
        assert configs[0]["name"] == "No Weights"

        # No weights file should exist
        weights_file = tmp_path / "disagg_weights_no_weights_config.parquet"
        assert not weights_file.exists()

    def test_update_config_overwrites_weights(self, tmp_path):
        """Verify updating a config also updates the weights file."""
        config = {
            "id": "update_test",
            "name": "Original",
            "granular_name_cols": ["campaign"],
            "date_column": "date",
            "weight_column": "spend",
            "entity_to_channel_mapping": {},
        }
        weights_v1 = pd.DataFrame({"date": ["2024-01-01"], "campaign": ["a"], "spend": [100]})
        save_disaggregation_config(tmp_path, config, weighting_df=weights_v1)

        # Update with new weights
        config["name"] = "Updated"
        weights_v2 = pd.DataFrame({"date": ["2024-01-01"], "campaign": ["b"], "spend": [999]})
        save_disaggregation_config(tmp_path, config, weighting_df=weights_v2)

        # Load and verify updated
        loaded_weights = load_disaggregation_weights(tmp_path, "update_test")
        assert loaded_weights is not None
        assert loaded_weights["spend"].iloc[0] == 999
        assert loaded_weights["campaign"].iloc[0] == "b"

    def test_weights_preserve_dtypes(self, tmp_path):
        """Verify parquet preserves DataFrame dtypes."""
        config = {
            "id": "dtype_test",
            "name": "DType Test",
            "granular_name_cols": ["campaign"],
            "date_column": "date",
            "weight_column": "impressions",
            "entity_to_channel_mapping": {},
        }
        weights_df = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01", "2024-01-08"]),
            "campaign": ["a", "b"],
            "impressions": [1000, 2000],
            "cost": [10.5, 20.75],
        })

        save_disaggregation_config(tmp_path, config, weighting_df=weights_df)
        loaded_weights = load_disaggregation_weights(tmp_path, "dtype_test")

        assert loaded_weights is not None
        assert pd.api.types.is_datetime64_any_dtype(loaded_weights["date"])
        assert pd.api.types.is_integer_dtype(loaded_weights["impressions"])
        assert pd.api.types.is_float_dtype(loaded_weights["cost"])
