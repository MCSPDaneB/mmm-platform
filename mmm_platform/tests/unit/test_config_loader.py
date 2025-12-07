"""
Tests for config/loader.py - Configuration loading and saving.
"""

import pytest
import tempfile
from pathlib import Path
import yaml

from mmm_platform.config.loader import ConfigLoader
from mmm_platform.config.schema import (
    ModelConfig, DataConfig, ChannelConfig, ControlConfig,
    AdstockConfig, SaturationConfig, SamplingConfig
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_config():
    """Sample ModelConfig for testing."""
    return ModelConfig(
        name="test_config",
        description="Test configuration",
        data=DataConfig(
            date_column="date",
            target_column="revenue",
            spend_scale=1000.0,
            target_scale=1000.0,
        ),
        channels=[
            ChannelConfig(
                name="tv_spend",
                display_name="TV Advertising",
                roi_prior_low=0.5,
                roi_prior_mid=2.0,
                roi_prior_high=5.0,
            ),
            ChannelConfig(
                name="search_spend",
                display_name="Paid Search",
                roi_prior_low=1.0,
                roi_prior_mid=3.0,
                roi_prior_high=6.0,
            ),
        ],
        controls=[
            ControlConfig(name="promo_flag", is_dummy=True),
        ],
        adstock=AdstockConfig(l_max=8),
        saturation=SaturationConfig(curve_sharpness=50),
        sampling=SamplingConfig(draws=1000, tune=1000, chains=4),
    )


@pytest.fixture
def sample_yaml_content():
    """Sample YAML content for testing."""
    return """
name: yaml_test_config
description: Config loaded from YAML

data:
  date_column: time
  target_column: sales
  spend_scale: 1.0
  target_scale: 1.0

channels:
  - name: facebook_spend
    display_name: Facebook
    roi_prior_low: 0.5
    roi_prior_mid: 1.5
    roi_prior_high: 3.0
    adstock_type: short

controls:
  - name: holiday_flag
    is_dummy: true

sampling:
  draws: 500
  tune: 500
  chains: 2
"""


@pytest.fixture
def temp_yaml_file(sample_yaml_content):
    """Temporary YAML file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(sample_yaml_content)
        temp_path = Path(f.name)

    yield temp_path

    # Cleanup
    if temp_path.exists():
        temp_path.unlink()


# =============================================================================
# from_yaml Tests
# =============================================================================

class TestFromYaml:
    """Tests for ConfigLoader.from_yaml method."""

    def test_loads_valid_yaml(self, temp_yaml_file):
        """Successfully loads valid YAML file."""
        config = ConfigLoader.from_yaml(temp_yaml_file)

        assert isinstance(config, ModelConfig)
        assert config.name == "yaml_test_config"

    def test_loads_correct_values(self, temp_yaml_file):
        """Loads correct values from YAML."""
        config = ConfigLoader.from_yaml(temp_yaml_file)

        assert config.data.date_column == "time"
        assert config.data.target_column == "sales"
        assert len(config.channels) == 1
        assert config.channels[0].name == "facebook_spend"
        assert config.sampling.draws == 500

    def test_file_not_found_raises(self):
        """Raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            ConfigLoader.from_yaml("nonexistent_file.yaml")

    def test_invalid_yaml_raises_value_error(self):
        """Raises ValueError for invalid YAML content."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
name: invalid_config
data:
  date_column: time
  # Missing required target_column
channels: "not a list"  # Invalid type
""")
            temp_path = Path(f.name)

        try:
            with pytest.raises(ValueError):
                ConfigLoader.from_yaml(temp_path)
        finally:
            temp_path.unlink()

    def test_accepts_path_string(self, temp_yaml_file):
        """Accepts string path."""
        config = ConfigLoader.from_yaml(str(temp_yaml_file))
        assert isinstance(config, ModelConfig)

    def test_accepts_path_object(self, temp_yaml_file):
        """Accepts Path object."""
        config = ConfigLoader.from_yaml(temp_yaml_file)
        assert isinstance(config, ModelConfig)


# =============================================================================
# to_yaml Tests
# =============================================================================

class TestToYaml:
    """Tests for ConfigLoader.to_yaml method."""

    def test_saves_to_yaml(self, sample_config):
        """Successfully saves config to YAML file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yaml_path = Path(temp_dir) / "output.yaml"
            ConfigLoader.to_yaml(sample_config, yaml_path)

            assert yaml_path.exists()

    def test_saved_yaml_is_valid(self, sample_config):
        """Saved YAML file is valid YAML."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yaml_path = Path(temp_dir) / "output.yaml"
            ConfigLoader.to_yaml(sample_config, yaml_path)

            with open(yaml_path, 'r') as f:
                data = yaml.safe_load(f)

            assert data["name"] == "test_config"

    def test_creates_parent_directories(self, sample_config):
        """Creates parent directories if they don't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yaml_path = Path(temp_dir) / "subdir" / "nested" / "output.yaml"
            ConfigLoader.to_yaml(sample_config, yaml_path)

            assert yaml_path.exists()

    def test_roundtrip_preserves_config(self, sample_config):
        """Saving and loading preserves config values."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yaml_path = Path(temp_dir) / "output.yaml"

            # Save
            ConfigLoader.to_yaml(sample_config, yaml_path)

            # Load
            loaded_config = ConfigLoader.from_yaml(yaml_path)

            # Verify key fields
            assert loaded_config.name == sample_config.name
            assert loaded_config.data.date_column == sample_config.data.date_column
            assert len(loaded_config.channels) == len(sample_config.channels)
            assert loaded_config.channels[0].name == sample_config.channels[0].name
            assert loaded_config.sampling.draws == sample_config.sampling.draws


# =============================================================================
# from_dict Tests
# =============================================================================

class TestFromDict:
    """Tests for ConfigLoader.from_dict method."""

    def test_loads_from_dict(self):
        """Successfully loads config from dictionary."""
        config_dict = {
            "name": "dict_config",
            "data": {
                "date_column": "date",
                "target_column": "revenue",
            },
            "channels": [
                {"name": "tv_spend", "roi_prior_mid": 2.0},
            ],
        }

        config = ConfigLoader.from_dict(config_dict)

        assert isinstance(config, ModelConfig)
        assert config.name == "dict_config"

    def test_validates_dict(self):
        """Validates dictionary against schema."""
        invalid_dict = {
            "name": "invalid",
            # Missing required 'data' field
        }

        with pytest.raises(Exception):  # Pydantic validation error
            ConfigLoader.from_dict(invalid_dict)


# =============================================================================
# get_template Tests
# =============================================================================

class TestGetTemplate:
    """Tests for ConfigLoader.get_template method."""

    def test_returns_dict(self):
        """Returns dictionary."""
        template = ConfigLoader.get_template()

        assert isinstance(template, dict)

    def test_has_required_sections(self):
        """Template has all required sections."""
        template = ConfigLoader.get_template()

        assert "name" in template
        assert "data" in template
        assert "channels" in template
        assert "sampling" in template

    def test_data_section_complete(self):
        """Data section has required fields."""
        template = ConfigLoader.get_template()

        assert "date_column" in template["data"]
        assert "target_column" in template["data"]

    def test_channel_section_complete(self):
        """Channel section has example with ROI priors."""
        template = ConfigLoader.get_template()

        assert len(template["channels"]) > 0
        channel = template["channels"][0]
        assert "name" in channel
        assert "roi_prior_low" in channel
        assert "roi_prior_mid" in channel
        assert "roi_prior_high" in channel

    def test_sampling_section_complete(self):
        """Sampling section has MCMC parameters."""
        template = ConfigLoader.get_template()

        assert "draws" in template["sampling"]
        assert "tune" in template["sampling"]
        assert "chains" in template["sampling"]

    def test_template_is_valid_config(self):
        """Template can be used to create a valid config."""
        template = ConfigLoader.get_template()

        # Should not raise
        config = ConfigLoader.from_dict(template)
        assert isinstance(config, ModelConfig)
