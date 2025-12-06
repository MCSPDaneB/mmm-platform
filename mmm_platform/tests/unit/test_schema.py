"""
Unit tests for config/schema.py

Tests Pydantic model validation and config methods.
"""
import pytest
from pydantic import ValidationError

from mmm_platform.config.schema import (
    ModelConfig, DataConfig, ChannelConfig, ControlConfig,
    OwnedMediaConfig, AdstockConfig, SaturationConfig, SignConstraint,
    sharpness_to_percentile, sharpness_label_to_value
)


class TestSharpnessConversion:
    """Tests for sharpness conversion functions."""

    def test_sharpness_to_percentile_range(self):
        """Percentile should be in range [20, 80]."""
        for sharpness in range(0, 101, 10):
            percentile = sharpness_to_percentile(sharpness)
            assert 20 <= percentile <= 80

    def test_sharpness_zero_is_gradual(self):
        """Sharpness=0 should give low percentile (gradual curve)."""
        percentile = sharpness_to_percentile(0)
        assert percentile == 20

    def test_sharpness_100_is_sharp(self):
        """Sharpness=100 should give high percentile (sharp curve)."""
        percentile = sharpness_to_percentile(100)
        assert percentile == 80

    def test_sharpness_50_is_middle(self):
        """Sharpness=50 should give middle percentile."""
        percentile = sharpness_to_percentile(50)
        assert percentile == 50

    def test_sharpness_label_to_value(self):
        """String labels should convert to correct values."""
        # Actual values from schema: gradual=25, balanced=50, sharp=75
        assert sharpness_label_to_value("sharp") == 75
        assert sharpness_label_to_value("balanced") == 50
        assert sharpness_label_to_value("gradual") == 25

    def test_sharpness_label_case_insensitive(self):
        """String labels should be case insensitive."""
        assert sharpness_label_to_value("Sharp") == 75
        assert sharpness_label_to_value("BALANCED") == 50
        assert sharpness_label_to_value("Gradual") == 25


class TestChannelConfig:
    """Tests for ChannelConfig validation."""

    def test_valid_channel(self):
        """Valid channel config should not raise."""
        channel = ChannelConfig(
            name="tv_spend",
            roi_prior_low=0.5,
            roi_prior_mid=1.5,
            roi_prior_high=3.0,
        )
        assert channel.name == "tv_spend"

    def test_channel_with_defaults(self):
        """Channel with just name should use defaults."""
        channel = ChannelConfig(name="tv_spend")
        # Actual defaults from schema
        assert channel.roi_prior_low == 0.1
        assert channel.roi_prior_mid == 1.0
        assert channel.roi_prior_high == 5.0
        assert channel.adstock_type == "medium"

    def test_roi_high_must_exceed_low(self):
        """ROI high must be greater than low."""
        with pytest.raises(ValidationError):
            ChannelConfig(
                name="tv_spend",
                roi_prior_low=3.0,
                roi_prior_mid=2.0,
                roi_prior_high=1.0,  # Less than low!
            )

    def test_display_name_defaults_to_formatted(self):
        """Display name should default to formatted column name."""
        channel = ChannelConfig(name="tv_spend")
        # get_display_name strips _spend and PaidMedia_ prefixes
        assert channel.get_display_name() == "tv"

    def test_display_name_override(self):
        """Display name should be used when provided."""
        channel = ChannelConfig(name="tv_spend", display_name="Television")
        assert channel.get_display_name() == "Television"

    def test_adstock_type_validation(self):
        """Adstock type should be one of short/medium/long."""
        channel = ChannelConfig(name="tv_spend", adstock_type="short")
        assert channel.adstock_type == "short"

        channel = ChannelConfig(name="tv_spend", adstock_type="long")
        assert channel.adstock_type == "long"


class TestControlConfig:
    """Tests for ControlConfig validation."""

    def test_valid_control(self):
        """Valid control config should not raise."""
        control = ControlConfig(
            name="promo_flag",
            sign_constraint=SignConstraint.POSITIVE,
        )
        assert control.name == "promo_flag"
        assert control.sign_constraint == SignConstraint.POSITIVE

    def test_control_with_defaults(self):
        """Control with just name should use defaults."""
        control = ControlConfig(name="holiday")
        assert control.sign_constraint == SignConstraint.UNCONSTRAINED

    def test_sign_constraint_validation(self):
        """Sign constraint should be positive/negative/unconstrained."""
        control = ControlConfig(name="x", sign_constraint=SignConstraint.POSITIVE)
        assert control.sign_constraint == SignConstraint.POSITIVE

        control = ControlConfig(name="x", sign_constraint=SignConstraint.NEGATIVE)
        assert control.sign_constraint == SignConstraint.NEGATIVE

        control = ControlConfig(name="x", sign_constraint=SignConstraint.UNCONSTRAINED)
        assert control.sign_constraint == SignConstraint.UNCONSTRAINED


class TestOwnedMediaConfig:
    """Tests for OwnedMediaConfig validation."""

    def test_owned_media_without_roi(self):
        """Owned media with include_roi=False should work."""
        om = OwnedMediaConfig(name="email", include_roi=False)
        assert om.include_roi is False

    def test_valid_owned_media_with_roi(self):
        """Valid owned media with ROI should not raise."""
        om = OwnedMediaConfig(
            name="email_sends",
            include_roi=True,
            roi_prior_low=0.1,
            roi_prior_mid=0.5,
            roi_prior_high=1.0,
        )
        assert om.include_roi is True
        assert om.roi_prior_mid == 0.5

    def test_owned_media_adstock_type(self):
        """Owned media should support adstock type."""
        om = OwnedMediaConfig(
            name="email_sends",
            adstock_type="short",
        )
        assert om.adstock_type == "short"


class TestModelConfig:
    """Tests for ModelConfig methods."""

    def test_get_channel_columns(self, basic_config):
        """get_channel_columns should return channel names."""
        channels = basic_config.get_channel_columns()
        assert "tv_spend" in channels
        assert "search_spend" in channels

    def test_get_roi_dicts(self, basic_config):
        """get_roi_dicts should return (low, mid, high) tuples."""
        low, mid, high = basic_config.get_roi_dicts()

        assert "tv_spend" in mid
        assert mid["tv_spend"] == 1.5
        assert low["tv_spend"] == 0.5
        assert high["tv_spend"] == 3.0

    def test_get_channel_by_name(self, basic_config):
        """Should retrieve correct channel config."""
        channel = basic_config.get_channel_by_name("tv_spend")
        assert channel is not None
        assert channel.name == "tv_spend"

    def test_get_channel_by_name_not_found(self, basic_config):
        """Should return None for unknown channel."""
        channel = basic_config.get_channel_by_name("nonexistent")
        assert channel is None

    def test_get_control_by_name(self, basic_config):
        """Should retrieve correct control config."""
        control = basic_config.get_control_by_name("promo_flag")
        assert control is not None
        assert control.name == "promo_flag"

    def test_get_control_by_name_not_found(self, basic_config):
        """Should return None for unknown control."""
        control = basic_config.get_control_by_name("nonexistent")
        assert control is None


class TestDataConfig:
    """Tests for DataConfig validation."""

    def test_valid_data_config(self):
        """Valid data config should not raise."""
        data = DataConfig(
            date_column="date",
            target_column="revenue",
            spend_scale=1000.0,
            revenue_scale=1000.0,
        )
        assert data.date_column == "date"
        assert data.target_column == "revenue"

    def test_data_config_requires_target_column(self):
        """DataConfig should require target_column."""
        with pytest.raises(ValidationError):
            DataConfig()  # Missing required target_column

    def test_data_config_defaults(self):
        """Data config should have sensible defaults for optional fields."""
        data = DataConfig(target_column="revenue")
        assert data.date_column == "time"  # Default from schema
        # Scale defaults are now 1.0 - PyMC-Marketing handles scaling internally
        assert data.spend_scale == 1.0
        assert data.revenue_scale == 1.0


class TestAdstockConfig:
    """Tests for AdstockConfig validation."""

    def test_valid_adstock_config(self):
        """Valid adstock config should not raise."""
        adstock = AdstockConfig(
            l_max=8,
            short_decay=0.3,
            medium_decay=0.5,
            long_decay=0.7,
        )
        assert adstock.l_max == 8

    def test_adstock_defaults(self):
        """Adstock config should have sensible defaults."""
        adstock = AdstockConfig()
        assert adstock.l_max == 8
        assert 0 < adstock.short_decay < adstock.medium_decay < adstock.long_decay < 1


class TestSaturationConfig:
    """Tests for SaturationConfig validation."""

    def test_valid_saturation_config(self):
        """Valid saturation config should not raise."""
        sat = SaturationConfig(curve_sharpness=50)
        assert sat.curve_sharpness == 50

    def test_saturation_defaults(self):
        """Saturation config should have sensible defaults."""
        sat = SaturationConfig()
        assert 0 <= sat.curve_sharpness <= 100


class TestBackwardCompatibility:
    """Tests for backward compatibility with old config formats."""

    def test_old_category_field_migrates(self):
        """Old 'category' field should migrate to 'categories' dict."""
        # Simulate old config format
        channel = ChannelConfig(
            name="tv_spend",
            category="Brand",  # Old format
        )
        # Should have migrated to categories dict with "Category" key (capital C)
        assert channel.categories.get("Category") == "Brand"

    def test_new_categories_format_works(self):
        """New 'categories' dict format should work."""
        channel = ChannelConfig(
            name="tv_spend",
            categories={"region": "North", "type": "Brand"},
        )
        assert channel.categories["region"] == "North"
        assert channel.categories["type"] == "Brand"

    def test_both_category_and_categories(self):
        """When both provided, categories should take precedence."""
        channel = ChannelConfig(
            name="tv_spend",
            category="OldBrand",
            categories={"type": "NewBrand"},
        )
        # categories dict should be preserved, old category also migrated
        assert "type" in channel.categories


class TestModelConfigSerialization:
    """Tests for config serialization/deserialization."""

    def test_model_dump(self, basic_config):
        """Should serialize to dict."""
        data = basic_config.model_dump()
        assert isinstance(data, dict)
        assert "channels" in data
        assert "controls" in data

    def test_model_dump_json(self, basic_config):
        """Should serialize to JSON string."""
        json_str = basic_config.model_dump_json()
        assert isinstance(json_str, str)
        assert "tv_spend" in json_str

    def test_roundtrip_serialization(self, basic_config):
        """Config should survive serialization roundtrip."""
        data = basic_config.model_dump()
        restored = ModelConfig(**data)

        assert restored.name == basic_config.name
        assert len(restored.channels) == len(basic_config.channels)
        assert restored.channels[0].name == basic_config.channels[0].name
