"""
Tests for model/persistence.py - Model and config persistence.
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np

from mmm_platform.model.persistence import (
    get_workspace_dir,
    set_workspace_dir,
    get_configs_dir,
    get_models_dir,
    get_clients_dir,
    list_clients,
    get_client_configs_dir,
    get_client_models_dir,
    ModelPersistence,
    ConfigPersistence,
)
from mmm_platform.config.schema import (
    ModelConfig, DataConfig, ChannelConfig, ControlConfig,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_workspace():
    """Temporary workspace directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        workspace = Path(temp_dir) / "test_workspace"
        workspace.mkdir()
        yield workspace


@pytest.fixture
def sample_config():
    """Sample ModelConfig for testing."""
    return ModelConfig(
        name="persistence_test",
        data=DataConfig(
            date_column="date",
            target_column="revenue",
        ),
        channels=[
            ChannelConfig(name="tv_spend"),
            ChannelConfig(name="search_spend"),
        ],
        controls=[
            ControlConfig(name="promo_flag"),
        ],
    )


@pytest.fixture
def sample_df():
    """Sample DataFrame for testing."""
    np.random.seed(42)
    n_rows = 50

    return pd.DataFrame({
        "date": pd.date_range("2022-01-01", periods=n_rows, freq="W"),
        "revenue": np.random.uniform(50000, 150000, n_rows),
        "tv_spend": np.random.uniform(5000, 20000, n_rows),
        "search_spend": np.random.uniform(3000, 15000, n_rows),
        "promo_flag": np.random.choice([0, 1], n_rows),
    })


@pytest.fixture
def mock_mmm_wrapper(sample_config, sample_df):
    """Mock MMMWrapper for testing."""
    wrapper = Mock()
    wrapper.config = sample_config
    wrapper.df_scaled = sample_df
    wrapper.df_raw = sample_df.copy()
    wrapper.control_cols = ["promo_flag"]
    wrapper.lam_vec = np.array([1.0, 1.5])
    wrapper.beta_mu = np.array([0.5, 0.8])
    wrapper.beta_sigma = np.array([0.1, 0.15])
    wrapper.mmm = None
    wrapper.idata = None
    wrapper.fitted_at = None
    wrapper.fit_duration_seconds = None
    wrapper.get_fit_statistics.return_value = {"r2": 0.85, "mape": 0.12}

    return wrapper


# =============================================================================
# Workspace Directory Tests
# =============================================================================

class TestWorkspaceDirectory:
    """Tests for workspace directory functions."""

    def test_get_workspace_dir_returns_path(self):
        """get_workspace_dir returns a Path object."""
        workspace = get_workspace_dir()
        assert isinstance(workspace, Path)

    def test_set_workspace_dir_creates_directory(self, temp_workspace):
        """set_workspace_dir creates the directory."""
        new_workspace = temp_workspace / "new_workspace"

        with patch('mmm_platform.model.persistence.SETTINGS_FILE', temp_workspace / ".settings.json"):
            set_workspace_dir(new_workspace)

        assert new_workspace.exists()

    def test_get_configs_dir_creates_if_missing(self, temp_workspace):
        """get_configs_dir creates configs directory."""
        with patch('mmm_platform.model.persistence.get_workspace_dir', return_value=temp_workspace):
            configs_dir = get_configs_dir()

        assert configs_dir.exists()
        assert configs_dir.name == "configs"

    def test_get_models_dir_creates_if_missing(self, temp_workspace):
        """get_models_dir creates models directory."""
        with patch('mmm_platform.model.persistence.get_workspace_dir', return_value=temp_workspace):
            models_dir = get_models_dir()

        assert models_dir.exists()
        assert models_dir.name == "models"


# =============================================================================
# Client Directory Tests
# =============================================================================

class TestClientDirectories:
    """Tests for client-specific directory functions."""

    def test_get_clients_dir_creates_if_missing(self, temp_workspace):
        """get_clients_dir creates clients directory."""
        with patch('mmm_platform.model.persistence.get_workspace_dir', return_value=temp_workspace):
            clients_dir = get_clients_dir()

        assert clients_dir.exists()
        assert clients_dir.name == "clients"

    def test_list_clients_empty(self, temp_workspace):
        """list_clients returns empty list when no clients."""
        with patch('mmm_platform.model.persistence.get_workspace_dir', return_value=temp_workspace):
            clients = list_clients()

        assert clients == []

    def test_list_clients_finds_folders(self, temp_workspace):
        """list_clients finds client folders."""
        clients_dir = temp_workspace / "clients"
        clients_dir.mkdir()
        (clients_dir / "client_a").mkdir()
        (clients_dir / "client_b").mkdir()

        with patch('mmm_platform.model.persistence.get_workspace_dir', return_value=temp_workspace):
            clients = list_clients()

        assert "client_a" in clients
        assert "client_b" in clients

    def test_get_client_configs_dir(self, temp_workspace):
        """get_client_configs_dir creates client configs directory."""
        with patch('mmm_platform.model.persistence.get_workspace_dir', return_value=temp_workspace):
            configs_dir = get_client_configs_dir("test_client")

        assert configs_dir.exists()
        assert "test_client" in str(configs_dir)
        assert configs_dir.name == "configs"

    def test_get_client_models_dir(self, temp_workspace):
        """get_client_models_dir creates client models directory."""
        with patch('mmm_platform.model.persistence.get_workspace_dir', return_value=temp_workspace):
            models_dir = get_client_models_dir("test_client")

        assert models_dir.exists()
        assert "test_client" in str(models_dir)
        assert models_dir.name == "models"


# =============================================================================
# ModelPersistence Tests
# =============================================================================

class TestModelPersistenceSave:
    """Tests for ModelPersistence.save method."""

    def test_save_creates_directory(self, mock_mmm_wrapper, temp_workspace):
        """save creates model directory."""
        model_path = temp_workspace / "test_model"

        ModelPersistence.save(mock_mmm_wrapper, model_path)

        assert model_path.exists()

    def test_save_creates_metadata(self, mock_mmm_wrapper, temp_workspace):
        """save creates metadata.json."""
        model_path = temp_workspace / "test_model"

        ModelPersistence.save(mock_mmm_wrapper, model_path)

        metadata_file = model_path / "metadata.json"
        assert metadata_file.exists()

        with open(metadata_file) as f:
            metadata = json.load(f)

        assert "version" in metadata
        assert "created_at" in metadata
        assert metadata["config_name"] == "persistence_test"

    def test_save_creates_config(self, mock_mmm_wrapper, temp_workspace):
        """save creates config.json."""
        model_path = temp_workspace / "test_model"

        ModelPersistence.save(mock_mmm_wrapper, model_path)

        config_file = model_path / "config.json"
        assert config_file.exists()

    def test_save_creates_model_pkl(self, mock_mmm_wrapper, temp_workspace):
        """save creates model.pkl."""
        model_path = temp_workspace / "test_model"

        ModelPersistence.save(mock_mmm_wrapper, model_path)

        model_file = model_path / "model.pkl"
        assert model_file.exists()

    def test_save_with_data(self, mock_mmm_wrapper, temp_workspace):
        """save includes data when include_data=True."""
        model_path = temp_workspace / "test_model"

        ModelPersistence.save(mock_mmm_wrapper, model_path, include_data=True)

        data_file = model_path / "data_scaled.parquet"
        assert data_file.exists()


class TestModelPersistenceLoad:
    """Tests for ModelPersistence.load method."""

    def test_load_returns_wrapper(self, mock_mmm_wrapper, temp_workspace):
        """load returns an MMMWrapper instance."""
        model_path = temp_workspace / "test_model"
        ModelPersistence.save(mock_mmm_wrapper, model_path, include_data=True)

        loaded = ModelPersistence.load(model_path)

        assert loaded is not None
        assert loaded.config.name == "persistence_test"

    def test_load_restores_state(self, mock_mmm_wrapper, temp_workspace):
        """load restores model state."""
        model_path = temp_workspace / "test_model"
        ModelPersistence.save(mock_mmm_wrapper, model_path, include_data=True)

        loaded = ModelPersistence.load(model_path)

        assert loaded.control_cols == ["promo_flag"]
        np.testing.assert_array_almost_equal(loaded.lam_vec, [1.0, 1.5])

    def test_load_not_found_raises(self, temp_workspace):
        """load raises FileNotFoundError for missing directory."""
        with pytest.raises(FileNotFoundError):
            ModelPersistence.load(temp_workspace / "nonexistent")


class TestModelPersistenceListModels:
    """Tests for ModelPersistence.list_saved_models method."""

    def test_list_empty_directory(self, temp_workspace):
        """list_saved_models returns empty list for empty directory."""
        models = ModelPersistence.list_saved_models(temp_workspace)

        assert models == []

    def test_list_finds_models(self, mock_mmm_wrapper, temp_workspace):
        """list_saved_models finds saved models."""
        # Save two models
        ModelPersistence.save(mock_mmm_wrapper, temp_workspace / "model1")
        ModelPersistence.save(mock_mmm_wrapper, temp_workspace / "model2")

        models = ModelPersistence.list_saved_models(temp_workspace)

        assert len(models) == 2

    def test_list_returns_metadata(self, mock_mmm_wrapper, temp_workspace):
        """list_saved_models returns metadata for each model."""
        ModelPersistence.save(mock_mmm_wrapper, temp_workspace / "model1")

        models = ModelPersistence.list_saved_models(temp_workspace)

        assert len(models) == 1
        assert "config_name" in models[0]
        assert "created_at" in models[0]
        assert "path" in models[0]


# =============================================================================
# ConfigPersistence Tests
# =============================================================================

class TestConfigPersistenceSave:
    """Tests for ConfigPersistence.save method."""

    def test_save_creates_directory(self, sample_config, sample_df, temp_workspace):
        """save creates config directory."""
        config_path = ConfigPersistence.save(
            name="test_config",
            config=sample_config,
            data=sample_df,
            session_state={},
            workspace=temp_workspace,
        )

        assert config_path.exists()

    def test_save_creates_metadata(self, sample_config, sample_df, temp_workspace):
        """save creates metadata.json with type='config'."""
        config_path = ConfigPersistence.save(
            name="test_config",
            config=sample_config,
            data=sample_df,
            session_state={},
            workspace=temp_workspace,
        )

        metadata_file = config_path / "metadata.json"
        assert metadata_file.exists()

        with open(metadata_file) as f:
            metadata = json.load(f)

        assert metadata["type"] == "config"

    def test_save_saves_data(self, sample_config, sample_df, temp_workspace):
        """save saves data as parquet."""
        config_path = ConfigPersistence.save(
            name="test_config",
            config=sample_config,
            data=sample_df,
            session_state={},
            workspace=temp_workspace,
        )

        data_file = config_path / "data.parquet"
        assert data_file.exists()


class TestConfigPersistenceLoad:
    """Tests for ConfigPersistence.load method."""

    def test_load_returns_tuple(self, sample_config, sample_df, temp_workspace):
        """load returns (config, data, session_state) tuple."""
        config_path = ConfigPersistence.save(
            name="test_config",
            config=sample_config,
            data=sample_df,
            session_state={"test_key": "test_value"},
            workspace=temp_workspace,
        )

        config, data, session = ConfigPersistence.load(config_path)

        assert isinstance(config, ModelConfig)
        assert isinstance(data, pd.DataFrame)
        assert isinstance(session, dict)

    def test_load_restores_config(self, sample_config, sample_df, temp_workspace):
        """load restores config values."""
        config_path = ConfigPersistence.save(
            name="test_config",
            config=sample_config,
            data=sample_df,
            session_state={},
            workspace=temp_workspace,
        )

        config, _, _ = ConfigPersistence.load(config_path)

        assert config.name == "persistence_test"
        assert len(config.channels) == 2


class TestConfigPersistenceList:
    """Tests for ConfigPersistence.list_saved_configs method."""

    def test_list_finds_configs(self, sample_config, sample_df, temp_workspace):
        """list_saved_configs finds saved configs."""
        # Save two configs
        ConfigPersistence.save(
            name="config1",
            config=sample_config,
            data=sample_df,
            session_state={},
            workspace=temp_workspace,
        )
        ConfigPersistence.save(
            name="config2",
            config=sample_config,
            data=sample_df,
            session_state={},
            workspace=temp_workspace,
        )

        # Need to scan the correct directory
        configs_dir = temp_workspace / "configs"
        configs = ConfigPersistence._scan_configs_dir(configs_dir)

        assert len(configs) == 2

    def test_list_returns_only_configs(self, mock_mmm_wrapper, sample_config, sample_df, temp_workspace):
        """list_saved_configs only returns type='config', not models."""
        # Save a config
        config_path = ConfigPersistence.save(
            name="my_config",
            config=sample_config,
            data=sample_df,
            session_state={},
            workspace=temp_workspace,
        )

        # Save a model (different type)
        models_dir = temp_workspace / "models"
        models_dir.mkdir(parents=True)
        ModelPersistence.save(mock_mmm_wrapper, models_dir / "my_model")

        # List should only find configs
        configs_dir = temp_workspace / "configs"
        configs = ConfigPersistence._scan_configs_dir(configs_dir)

        for config in configs:
            assert config.get("type") == "config"
