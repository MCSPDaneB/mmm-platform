"""
Model persistence - save and load fitted models and configurations.
"""

import pickle
import cloudpickle
import json
from pathlib import Path
from typing import Union, Optional, Any
from datetime import datetime
import logging
import hashlib
import pandas as pd

logger = logging.getLogger(__name__)


# Default workspace directory
DEFAULT_WORKSPACE = Path("mmm_workspace")
SETTINGS_FILE = Path(".mmm_settings.json")


def get_workspace_dir() -> Path:
    """Get the current workspace directory from settings."""
    if SETTINGS_FILE.exists():
        try:
            with open(SETTINGS_FILE, "r") as f:
                settings = json.load(f)
                return Path(settings.get("workspace", str(DEFAULT_WORKSPACE)))
        except Exception:
            pass
    return DEFAULT_WORKSPACE


def set_workspace_dir(workspace: Union[str, Path]) -> None:
    """Set the workspace directory in settings."""
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    settings = {}
    if SETTINGS_FILE.exists():
        try:
            with open(SETTINGS_FILE, "r") as f:
                settings = json.load(f)
        except Exception:
            pass

    settings["workspace"] = str(workspace)
    with open(SETTINGS_FILE, "w") as f:
        json.dump(settings, f, indent=2)


def get_configs_dir() -> Path:
    """Get the configs subdirectory within workspace (legacy, non-client)."""
    configs_dir = get_workspace_dir() / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    return configs_dir


def get_models_dir() -> Path:
    """Get the models subdirectory within workspace (legacy, non-client)."""
    models_dir = get_workspace_dir() / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


# =============================================================================
# Client-aware directory functions
# =============================================================================

def get_clients_dir() -> Path:
    """Get the clients subdirectory within workspace."""
    clients_dir = get_workspace_dir() / "clients"
    clients_dir.mkdir(parents=True, exist_ok=True)
    return clients_dir


def list_clients() -> list[str]:
    """List all client folders."""
    clients_dir = get_workspace_dir() / "clients"
    if not clients_dir.exists():
        return []
    return sorted([d.name for d in clients_dir.iterdir() if d.is_dir()])


def get_client_configs_dir(client: str) -> Path:
    """Get configs directory for a specific client."""
    configs_dir = get_clients_dir() / client / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    return configs_dir


def get_client_models_dir(client: str) -> Path:
    """Get models directory for a specific client."""
    models_dir = get_clients_dir() / client / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


class ModelPersistence:
    """
    Save and load fitted MMM models.

    Models are saved as a bundle containing:
    - The fitted PyMC model (pickled)
    - InferenceData (ArviZ format)
    - Configuration
    - Metadata
    """

    METADATA_FILE = "metadata.json"
    MODEL_FILE = "model.pkl"
    IDATA_FILE = "idata.nc"
    CONFIG_FILE = "config.json"

    @classmethod
    def save(
        cls,
        mmm_wrapper: Any,  # MMMWrapper
        path: Union[str, Path],
        include_data: bool = False
    ) -> Path:
        """
        Save a fitted model to disk.

        Parameters
        ----------
        mmm_wrapper : MMMWrapper
            Fitted model wrapper.
        path : Union[str, Path]
            Directory to save the model.
        include_data : bool
            Whether to include the training data.

        Returns
        -------
        Path
            Path to the saved model directory.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Get fit statistics if available
        fit_stats = {}
        try:
            fit_stats = mmm_wrapper.get_fit_statistics()
        except Exception:
            pass

        # Get client from config
        client = getattr(mmm_wrapper.config, 'client', None)

        # Save metadata
        metadata = {
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "fitted_at": mmm_wrapper.fitted_at.isoformat() if mmm_wrapper.fitted_at else None,
            "fit_duration_seconds": mmm_wrapper.fit_duration_seconds,
            "config_name": mmm_wrapper.config.name,
            "client": client,
            "n_channels": len(mmm_wrapper.config.channels),
            "n_controls": len(mmm_wrapper.control_cols) if mmm_wrapper.control_cols else 0,
            "include_data": include_data,
            "r2": fit_stats.get("r2"),
            "mape": fit_stats.get("mape"),
            "rmse": fit_stats.get("rmse"),
        }

        with open(path / cls.METADATA_FILE, "w") as f:
            json.dump(metadata, f, indent=2)

        # Save config
        config_dict = mmm_wrapper.config.model_dump(mode="json")
        with open(path / cls.CONFIG_FILE, "w") as f:
            json.dump(config_dict, f, indent=2)

        # Save model state (excluding large objects)
        model_state = {
            "control_cols": mmm_wrapper.control_cols,
            "lam_vec": mmm_wrapper.lam_vec,
            "beta_mu": mmm_wrapper.beta_mu,
            "beta_sigma": mmm_wrapper.beta_sigma,
        }

        # Save the PyMC model using cloudpickle
        if mmm_wrapper.mmm is not None:
            model_state["mmm"] = mmm_wrapper.mmm

        with open(path / cls.MODEL_FILE, "wb") as f:
            cloudpickle.dump(model_state, f)

        # Save InferenceData
        if mmm_wrapper.idata is not None:
            mmm_wrapper.idata.to_netcdf(str(path / cls.IDATA_FILE))

        # Optionally save data
        if include_data and mmm_wrapper.df_scaled is not None:
            mmm_wrapper.df_scaled.to_parquet(path / "data_scaled.parquet")
            if mmm_wrapper.df_raw is not None:
                mmm_wrapper.df_raw.to_parquet(path / "data_raw.parquet")

        logger.info(f"Model saved to {path}")

        return path

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        mmm_wrapper_class: Any = None  # MMMWrapper class
    ) -> Any:
        """
        Load a saved model.

        Parameters
        ----------
        path : Union[str, Path]
            Path to the saved model directory.
        mmm_wrapper_class : class, optional
            MMMWrapper class to instantiate.

        Returns
        -------
        MMMWrapper
            Loaded model wrapper.
        """
        import arviz as az
        import pandas as pd
        from ..config.schema import ModelConfig

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Model directory not found: {path}")

        # Load metadata
        with open(path / cls.METADATA_FILE, "r") as f:
            metadata = json.load(f)

        logger.info(f"Loading model from {path} (created: {metadata['created_at']})")

        # Load config
        with open(path / cls.CONFIG_FILE, "r") as f:
            config_dict = json.load(f)
        config = ModelConfig(**config_dict)

        # Import MMMWrapper if not provided
        if mmm_wrapper_class is None:
            from .mmm import MMMWrapper
            mmm_wrapper_class = MMMWrapper

        # Create wrapper instance
        wrapper = mmm_wrapper_class(config)

        # Load model state
        with open(path / cls.MODEL_FILE, "rb") as f:
            model_state = pickle.load(f)

        wrapper.control_cols = model_state.get("control_cols")
        wrapper.lam_vec = model_state.get("lam_vec")
        wrapper.beta_mu = model_state.get("beta_mu")
        wrapper.beta_sigma = model_state.get("beta_sigma")

        if "mmm" in model_state:
            wrapper.mmm = model_state["mmm"]

        # Load InferenceData
        idata_path = path / cls.IDATA_FILE
        if idata_path.exists():
            wrapper.idata = az.from_netcdf(str(idata_path))

        # Load data if available
        data_scaled_path = path / "data_scaled.parquet"
        if data_scaled_path.exists():
            wrapper.df_scaled = pd.read_parquet(data_scaled_path)

        data_raw_path = path / "data_raw.parquet"
        if data_raw_path.exists():
            wrapper.df_raw = pd.read_parquet(data_raw_path)

        # Restore metadata
        if metadata.get("fitted_at"):
            wrapper.fitted_at = datetime.fromisoformat(metadata["fitted_at"])
        wrapper.fit_duration_seconds = metadata.get("fit_duration_seconds")

        logger.info("Model loaded successfully")

        return wrapper

    @classmethod
    def list_saved_models(
        cls,
        directory: Optional[Union[str, Path]] = None,
        client: Optional[str] = None
    ) -> list[dict]:
        """
        List all saved models in a directory.

        Parameters
        ----------
        directory : Union[str, Path], optional
            Directory to search. If not provided, uses default or client directory.
        client : str, optional
            If provided, only list models for this client.
            If "all", list from all clients plus legacy models.

        Returns
        -------
        list[dict]
            List of model metadata.
        """
        models = []

        if directory is not None:
            # Use specified directory
            models.extend(cls._scan_models_dir(Path(directory)))
        elif client == "all" or client is None:
            # Include legacy models (non-client)
            legacy_dir = get_models_dir()
            if legacy_dir.exists():
                models.extend(cls._scan_models_dir(legacy_dir))

            # Include all client models
            for client_name in list_clients():
                client_dir = get_client_models_dir(client_name)
                if client_dir.exists():
                    models.extend(cls._scan_models_dir(client_dir))
        elif client:
            # Only search specific client directory
            client_dir = get_client_models_dir(client)
            if client_dir.exists():
                models.extend(cls._scan_models_dir(client_dir))

        # Sort by creation date
        models.sort(key=lambda x: x.get("created_at", ""), reverse=True)

        return models

    @classmethod
    def _scan_models_dir(cls, directory: Path) -> list[dict]:
        """Scan a directory for model metadata files."""
        models = []

        if not directory.exists():
            return models

        for subdir in directory.iterdir():
            if subdir.is_dir():
                metadata_path = subdir / cls.METADATA_FILE
                if metadata_path.exists():
                    try:
                        with open(metadata_path, "r") as f:
                            metadata = json.load(f)
                        metadata["path"] = str(subdir)
                        models.append(metadata)
                    except Exception as e:
                        logger.warning(f"Could not read metadata from {subdir}: {e}")

        return models

    @classmethod
    def update_config(cls, path: Union[str, Path], config: Any) -> None:
        """
        Update the config.json file for a saved model.

        This is useful for updating metadata-only fields (like categories)
        without re-running the model.

        Parameters
        ----------
        path : Union[str, Path]
            Path to the saved model directory.
        config : ModelConfig
            Updated model configuration.
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Model directory not found: {path}")

        config_file = path / cls.CONFIG_FILE
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")

        # Save updated config
        config_dict = config.model_dump(mode="json")
        with open(config_file, "w") as f:
            json.dump(config_dict, f, indent=2)

        # Also update metadata with new counts if channels/controls changed
        metadata_file = path / cls.METADATA_FILE
        if metadata_file.exists():
            with open(metadata_file, "r") as f:
                metadata = json.load(f)

            # Update channel/control counts
            metadata["n_channels"] = len(config.channels)
            metadata["n_controls"] = len(config.controls)
            metadata["config_name"] = config.name
            metadata["client"] = getattr(config, 'client', None)

            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)

        logger.info(f"Config updated for model at {path}")

    @classmethod
    def get_model_hash(cls, mmm_wrapper: Any) -> str:
        """
        Generate a hash for the model configuration.

        Useful for detecting if a model needs to be re-fitted.

        Parameters
        ----------
        mmm_wrapper : MMMWrapper
            Model wrapper.

        Returns
        -------
        str
            Hash string.
        """
        config_str = json.dumps(
            mmm_wrapper.config.model_dump(mode="json"),
            sort_keys=True
        )
        return hashlib.md5(config_str.encode()).hexdigest()[:12]


class ConfigPersistence:
    """
    Save and load model configurations (without fitted model).

    Configs are saved as a bundle containing:
    - Configuration JSON
    - Data file (parquet)
    - Metadata
    - Session state (category columns, UI state)
    """

    METADATA_FILE = "metadata.json"
    CONFIG_FILE = "config.json"
    DATA_FILE = "data.parquet"
    SESSION_FILE = "session_state.json"

    @classmethod
    def save(
        cls,
        name: str,
        config: Any,  # ModelConfig
        data: pd.DataFrame,
        session_state: dict,
        workspace: Optional[Union[str, Path]] = None
    ) -> Path:
        """
        Save a configuration to disk.

        Parameters
        ----------
        name : str
            Name for the configuration.
        config : ModelConfig
            Model configuration.
        data : pd.DataFrame
            Data to save with config.
        session_state : dict
            Session state containing category_columns, config_state, etc.
        workspace : Path, optional
            Workspace directory. Uses default if not provided.
            If config has a client set, saves to client-specific directory.

        Returns
        -------
        Path
            Path to the saved config directory.
        """
        # Determine save directory based on client
        client = getattr(config, 'client', None) if config else None

        if workspace is None:
            if client:
                workspace = get_client_configs_dir(client)
            else:
                workspace = get_configs_dir()
        else:
            workspace = Path(workspace) / "configs"

        workspace.mkdir(parents=True, exist_ok=True)

        # Create timestamped directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
        config_dir = workspace / f"{safe_name}_{timestamp}"
        config_dir.mkdir(parents=True, exist_ok=True)

        # Save metadata
        metadata = {
            "version": "1.0",
            "type": "config",
            "name": name,
            "client": client,
            "created_at": datetime.now().isoformat(),
            "config_name": config.name if config else name,
            "n_channels": len(config.channels) if config and config.channels else 0,
            "n_controls": len(config.controls) if config and config.controls else 0,
            "n_rows": len(data) if data is not None else 0,
        }

        with open(config_dir / cls.METADATA_FILE, "w") as f:
            json.dump(metadata, f, indent=2)

        # Save config
        if config is not None:
            config_dict = config.model_dump(mode="json")
            with open(config_dir / cls.CONFIG_FILE, "w") as f:
                json.dump(config_dict, f, indent=2)

        # Save data
        if data is not None:
            data.to_parquet(config_dir / cls.DATA_FILE)

        # Save relevant session state
        session_to_save = {
            "category_columns": session_state.get("category_columns", []),
            "config_state": session_state.get("config_state", {}),
            "date_column": session_state.get("date_column"),
            "target_column": session_state.get("target_column"),
            "detected_channels": session_state.get("detected_channels", []),
            "dayfirst": session_state.get("dayfirst", False),
            # Multiselect states for channels and controls
            "channel_multiselect": session_state.get("channel_multiselect", []),
            "control_multiselect": session_state.get("control_multiselect", []),
            # Also save include_trend explicitly for backwards compatibility
            "include_trend": session_state.get("config_state", {}).get("include_trend", True),
        }

        with open(config_dir / cls.SESSION_FILE, "w") as f:
            json.dump(session_to_save, f, indent=2)

        logger.info(f"Configuration saved to {config_dir}")

        return config_dir

    @classmethod
    def load(cls, path: Union[str, Path]) -> tuple[Any, pd.DataFrame, dict]:
        """
        Load a saved configuration.

        Parameters
        ----------
        path : Union[str, Path]
            Path to the saved config directory.

        Returns
        -------
        tuple
            (ModelConfig, DataFrame, session_state_dict)
        """
        from ..config.schema import ModelConfig

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Config directory not found: {path}")

        # Load metadata
        with open(path / cls.METADATA_FILE, "r") as f:
            metadata = json.load(f)

        logger.info(f"Loading config from {path} (created: {metadata['created_at']})")

        # Load config
        config = None
        config_path = path / cls.CONFIG_FILE
        if config_path.exists():
            with open(config_path, "r") as f:
                config_dict = json.load(f)
            config = ModelConfig(**config_dict)

        # Load data
        data = None
        data_path = path / cls.DATA_FILE
        if data_path.exists():
            data = pd.read_parquet(data_path)

        # Load session state
        session_state = {}
        session_path = path / cls.SESSION_FILE
        if session_path.exists():
            with open(session_path, "r") as f:
                session_state = json.load(f)

        return config, data, session_state

    @classmethod
    def list_saved_configs(
        cls,
        directory: Optional[Union[str, Path]] = None,
        client: Optional[str] = None
    ) -> list[dict]:
        """
        List all saved configs in a directory.

        Parameters
        ----------
        directory : Path, optional
            Directory to search. Uses default configs dir if not provided.
        client : str, optional
            If provided, only list configs for this client.
            If "all", list from all clients plus legacy configs.

        Returns
        -------
        list[dict]
            List of config metadata.
        """
        configs = []

        if client == "all" or client is None:
            # Include legacy configs (non-client)
            legacy_dir = get_configs_dir()
            if legacy_dir.exists():
                configs.extend(cls._scan_configs_dir(legacy_dir))

            # Include all client configs
            for client_name in list_clients():
                client_dir = get_client_configs_dir(client_name)
                if client_dir.exists():
                    configs.extend(cls._scan_configs_dir(client_dir))
        elif client:
            # Only search specific client directory
            client_dir = get_client_configs_dir(client)
            if client_dir.exists():
                configs.extend(cls._scan_configs_dir(client_dir))
        elif directory is not None:
            # Use specified directory
            configs.extend(cls._scan_configs_dir(Path(directory)))

        # Sort by creation date
        configs.sort(key=lambda x: x.get("created_at", ""), reverse=True)

        return configs

    @classmethod
    def _scan_configs_dir(cls, directory: Path) -> list[dict]:
        """Scan a directory for config metadata files."""
        configs = []

        if not directory.exists():
            return configs

        for subdir in directory.iterdir():
            if subdir.is_dir():
                metadata_path = subdir / cls.METADATA_FILE
                if metadata_path.exists():
                    try:
                        with open(metadata_path, "r") as f:
                            metadata = json.load(f)
                        # Only include config type (not models)
                        if metadata.get("type") == "config":
                            metadata["path"] = str(subdir)
                            configs.append(metadata)
                    except Exception as e:
                        logger.warning(f"Could not read metadata from {subdir}: {e}")

        return configs


def restore_config_to_session(config: Any, data: pd.DataFrame, session_state: dict) -> dict:
    """
    Build a dictionary of session state updates from a loaded config.

    Parameters
    ----------
    config : ModelConfig
        Loaded model configuration.
    data : pd.DataFrame
        Loaded data.
    session_state : dict
        Loaded session state (category_columns, config_state, etc.)

    Returns
    -------
    dict
        Dictionary of session state keys to update.
    """
    updates = {
        "current_data": data,
        "current_config": config,
        "model_fitted": False,
    }

    # Restore category columns
    if "category_columns" in session_state:
        updates["category_columns"] = session_state["category_columns"]

    # Restore config_state: start with saved values, then MERGE authoritative values from config
    # This ensures config.json is the source of truth while preserving other session fields
    config_state = session_state.get("config_state", {}).copy() if session_state.get("config_state") else {}

    # ALWAYS merge/override with authoritative values from loaded ModelConfig
    if config is not None:
        config_state["name"] = config.name
        config_state["client"] = config.client
        config_state["description"] = config.description
        if config.data:
            config_state["date_col"] = config.data.date_column
            config_state["target_col"] = config.data.target_column
            config_state["revenue_scale"] = config.data.revenue_scale
            config_state["spend_scale"] = config.data.spend_scale
            config_state["dayfirst"] = config.data.dayfirst
            config_state["include_trend"] = config.data.include_trend
            config_state["model_start_date"] = config.data.model_start_date
            config_state["model_end_date"] = config.data.model_end_date

    updates["config_state"] = config_state

    # Restore data settings - ALWAYS prefer config values (authoritative source)
    if config is not None and config.data and config.data.date_column:
        updates["date_column"] = config.data.date_column
    elif session_state.get("date_column"):
        updates["date_column"] = session_state["date_column"]

    if config is not None and config.data and config.data.target_column:
        updates["target_column"] = config.data.target_column
    elif session_state.get("target_column"):
        updates["target_column"] = session_state["target_column"]

    if session_state.get("detected_channels"):
        updates["detected_channels"] = session_state["detected_channels"]
    if "dayfirst" in session_state:
        updates["dayfirst"] = session_state["dayfirst"]
    elif config is not None and config.data:
        updates["dayfirst"] = config.data.dayfirst

    # Restore multiselect states for channels and controls
    if session_state.get("channel_multiselect"):
        updates["channel_multiselect"] = session_state["channel_multiselect"]
    if session_state.get("control_multiselect"):
        updates["control_multiselect"] = session_state["control_multiselect"]

    return updates
