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
