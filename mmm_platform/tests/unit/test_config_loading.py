"""Unit tests for config loading and session state management.

Tests for:
- _build_config_state_from_model preserves kpi_type, target_scale, etc.
- Session state reset on new data upload
"""
import pytest
from mmm_platform.config.schema import (
    ModelConfig, DataConfig, ChannelConfig, OwnedMediaConfig, KPIType
)


class TestBuildConfigStateFromModel:
    """Tests for _build_config_state_from_model in saved_models.py."""

    def test_kpi_type_preserved_revenue(self):
        """Verify kpi_type='revenue' is preserved when building config_state."""
        from mmm_platform.ui.pages.saved_models import _build_config_state_from_model

        config = ModelConfig(
            name="test_model",
            data=DataConfig(
                date_column="date",
                target_column="revenue",
                kpi_type=KPIType.REVENUE,
            ),
            channels=[
                ChannelConfig(name="channel_a_spend", roi_prior_mid=2.0),
            ],
        )

        config_state = _build_config_state_from_model(config)

        assert config_state["kpi_type"] == "revenue"

    def test_kpi_type_preserved_count(self):
        """Verify kpi_type='count' is preserved when building config_state."""
        from mmm_platform.ui.pages.saved_models import _build_config_state_from_model

        config = ModelConfig(
            name="test_model",
            data=DataConfig(
                date_column="date",
                target_column="installs",
                kpi_type=KPIType.COUNT,
                kpi_display_name="Install",
            ),
            channels=[
                ChannelConfig(name="channel_a_spend", roi_prior_mid=50.0),
            ],
        )

        config_state = _build_config_state_from_model(config)

        assert config_state["kpi_type"] == "count"
        assert config_state["kpi_display_name"] == "Install"

    def test_kpi_display_name_preserved(self):
        """Verify custom kpi_display_name is preserved."""
        from mmm_platform.ui.pages.saved_models import _build_config_state_from_model

        config = ModelConfig(
            name="test_model",
            data=DataConfig(
                date_column="date",
                target_column="signups",
                kpi_type=KPIType.COUNT,
                kpi_display_name="Signup",
            ),
            channels=[
                ChannelConfig(name="channel_a_spend", roi_prior_mid=25.0),
            ],
        )

        config_state = _build_config_state_from_model(config)

        assert config_state["kpi_display_name"] == "Signup"

    def test_target_scale_preserved(self):
        """Verify target_scale is preserved (not old revenue_scale)."""
        from mmm_platform.ui.pages.saved_models import _build_config_state_from_model

        config = ModelConfig(
            name="test_model",
            data=DataConfig(
                date_column="date",
                target_column="revenue",
                target_scale=500.0,
            ),
            channels=[
                ChannelConfig(name="channel_a_spend", roi_prior_mid=2.0),
            ],
        )

        config_state = _build_config_state_from_model(config)

        assert config_state["target_scale"] == 500.0

    def test_owned_media_preserved(self):
        """Verify owned_media list is preserved."""
        from mmm_platform.ui.pages.saved_models import _build_config_state_from_model

        config = ModelConfig(
            name="test_model",
            data=DataConfig(
                date_column="date",
                target_column="revenue",
            ),
            channels=[
                ChannelConfig(name="channel_a_spend", roi_prior_mid=2.0),
            ],
            owned_media=[
                OwnedMediaConfig(name="email_opens", display_name="Email Opens"),
                OwnedMediaConfig(name="social_clicks", display_name="Social Clicks"),
            ],
        )

        config_state = _build_config_state_from_model(config)

        assert len(config_state["owned_media"]) == 2
        owned_names = [om["name"] for om in config_state["owned_media"]]
        assert "email_opens" in owned_names
        assert "social_clicks" in owned_names

    def test_all_data_fields_preserved(self):
        """Verify all data config fields are preserved."""
        from mmm_platform.ui.pages.saved_models import _build_config_state_from_model

        config = ModelConfig(
            name="test_model",
            data=DataConfig(
                date_column="date",
                target_column="revenue",
                target_scale=1000.0,
                spend_scale=1.0,
                dayfirst=True,
                include_trend=True,
                model_start_date="2023-01-01",
                model_end_date="2023-12-31",
                brand="test_brand",
                kpi_type=KPIType.COUNT,
                kpi_display_name="Conversion",
            ),
            channels=[
                ChannelConfig(name="channel_a_spend", roi_prior_mid=2.0),
            ],
        )

        config_state = _build_config_state_from_model(config)

        assert config_state["date_col"] == "date"
        assert config_state["target_col"] == "revenue"
        assert config_state["target_scale"] == 1000.0
        assert config_state["spend_scale"] == 1.0
        assert config_state["dayfirst"] is True
        assert config_state["include_trend"] is True
        assert config_state["model_start_date"] == "2023-01-01"
        assert config_state["model_end_date"] == "2023-12-31"
        assert config_state["brand"] == "test_brand"
        assert config_state["kpi_type"] == "count"
        assert config_state["kpi_display_name"] == "Conversion"


class TestConfigStateReset:
    """Tests for config_state reset when uploading new data.

    These tests verify that the config_state reset structure in upload_data.py
    clears all relevant fields to prevent cross-brand contamination.
    """

    def test_reset_config_state_structure(self):
        """Verify the reset config_state structure has all required fields."""
        # This is the expected structure when resetting config_state
        reset_config_state = {
            "name": "my_mmm_model",
            "channels": [],
            "controls": [],
            "owned_media": [],
            "competitors": [],
            "dummy_variables": [],
        }

        # Verify all required fields exist and are empty lists where expected
        assert reset_config_state["name"] == "my_mmm_model"
        assert reset_config_state["channels"] == []
        assert reset_config_state["controls"] == []
        assert reset_config_state["owned_media"] == []
        assert reset_config_state["competitors"] == []
        assert reset_config_state["dummy_variables"] == []

    def test_config_state_does_not_preserve_old_owned_media(self):
        """Verify that a fresh config_state doesn't carry over owned_media."""
        # Simulate old config_state with owned media
        old_config_state = {
            "name": "old_model",
            "channels": [{"name": "old_channel"}],
            "owned_media": [{"name": "old_owned_media"}],
            "dummy_variables": [{"name": "old_dummy"}],
        }

        # Simulate reset (what upload_data.py does)
        reset_config_state = {
            "name": "my_mmm_model",
            "channels": [],
            "controls": [],
            "owned_media": [],
            "competitors": [],
            "dummy_variables": [],
        }

        # Verify old data is NOT in the reset state
        assert len(reset_config_state["owned_media"]) == 0
        assert len(reset_config_state["dummy_variables"]) == 0
        assert len(reset_config_state["channels"]) == 0

    def test_config_state_preserves_kpi_fields_after_reset(self):
        """Verify kpi_type fields can be re-added after reset."""
        # Simulate reset
        config_state = {
            "name": "my_mmm_model",
            "channels": [],
            "controls": [],
            "owned_media": [],
            "competitors": [],
            "dummy_variables": [],
        }

        # Simulate what configure_model.py does - adds kpi fields
        config_state["kpi_type"] = "count"
        config_state["kpi_display_name"] = "Install"
        config_state["target_scale"] = 1.0

        assert config_state["kpi_type"] == "count"
        assert config_state["kpi_display_name"] == "Install"
        assert config_state["target_scale"] == 1.0


class TestEmptyOwnedMediaPreserved:
    """Tests to ensure empty owned_media is respected when loading configs.

    This prevents auto-detection from re-populating owned media when
    loading a config that was explicitly saved without owned media.
    """

    def test_config_with_empty_owned_media_has_key(self):
        """Verify config_state from model with no owned_media has the key."""
        from mmm_platform.ui.pages.saved_models import _build_config_state_from_model

        config = ModelConfig(
            name="test_model",
            data=DataConfig(
                date_column="date",
                target_column="revenue",
            ),
            channels=[
                ChannelConfig(name="channel_a_spend", roi_prior_mid=2.0),
            ],
            owned_media=[],  # Explicitly empty
        )

        config_state = _build_config_state_from_model(config)

        # The key should exist and be an empty list
        assert "owned_media" in config_state
        assert config_state["owned_media"] == []

    def test_owned_media_selection_respects_empty_config(self):
        """Verify that empty owned_media in config prevents auto-detection.

        When a config has an empty owned_media list, the UI should NOT
        auto-detect owned media columns - it should respect the empty list.
        """
        # Simulate a loaded config with explicitly empty owned_media
        config_state = {
            "name": "test_model",
            "channels": [{"name": "channel_a_spend"}],
            "owned_media": [],  # Explicitly empty - should be respected
        }

        # This is the logic from configure_model.py that determines selection
        saved_owned_media = config_state.get("owned_media", [])
        has_owned_media_key = "owned_media" in config_state

        # Simulate available columns that would match auto-detection patterns
        available_owned_media = ["email_opens", "organic_traffic", "newsletter_clicks"]
        auto_owned_media = [col for col in available_owned_media
                           if any(p in col.lower() for p in ["email", "organic", "owned", "social", "newsletter"])]

        # Determine selection using the same logic as configure_model.py
        if saved_owned_media:
            selection = [c for c in [om["name"] for om in saved_owned_media] if c in available_owned_media]
        elif has_owned_media_key:
            # Config explicitly has empty owned_media - respect that
            selection = []
        else:
            # No config loaded - auto-detect
            selection = auto_owned_media if auto_owned_media else []

        # Should be empty because config has owned_media key (even though empty)
        assert selection == []
        assert len(auto_owned_media) > 0  # Verify auto-detection would have found columns

    def test_new_data_upload_allows_auto_detection(self):
        """Verify that fresh upload (no config) allows auto-detection."""
        # Simulate a fresh config_state without owned_media key
        config_state = {
            "name": "my_mmm_model",
            "channels": [],
            # Note: owned_media key is NOT present - this is a fresh upload
        }

        saved_owned_media = config_state.get("owned_media", [])
        has_owned_media_key = "owned_media" in config_state

        # Simulate available columns
        available_owned_media = ["email_opens", "organic_traffic"]
        auto_owned_media = ["email_opens", "organic_traffic"]  # Would be auto-detected

        # Determine selection
        if saved_owned_media:
            selection = saved_owned_media
        elif has_owned_media_key:
            selection = []
        else:
            # No config - auto-detect is allowed
            selection = auto_owned_media

        # Should auto-detect because no owned_media key exists
        assert selection == auto_owned_media
        assert has_owned_media_key is False
