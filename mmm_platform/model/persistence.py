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
            "is_archived": False,
            "is_favorite": False,
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
            # Save original unfiltered data (preserves full timeseries)
            if getattr(mmm_wrapper, "df_original", None) is not None:
                mmm_wrapper.df_original.to_parquet(path / "data_original.parquet")

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

            # Reconstruct MMM from idata to restore internal PyMC model state
            # This is required for optimize_budget to work correctly
            try:
                from pymc_marketing.mmm import MMM
                wrapper.mmm = MMM.load_from_idata(wrapper.idata)
                logger.info("Reconstructed MMM from idata for optimization support")
            except Exception as e:
                logger.warning(f"Could not reconstruct MMM from idata: {e}. "
                              "Budget optimization may not work correctly.")
                # Fall back to pickled version (works for everything except optimizer)

        # Load data if available
        data_scaled_path = path / "data_scaled.parquet"
        if data_scaled_path.exists():
            wrapper.df_scaled = pd.read_parquet(data_scaled_path)

        data_raw_path = path / "data_raw.parquet"
        if data_raw_path.exists():
            wrapper.df_raw = pd.read_parquet(data_raw_path)

        # Load original unfiltered data if available (for full timeseries in UI)
        data_original_path = path / "data_original.parquet"
        if data_original_path.exists():
            wrapper.df_original = pd.read_parquet(data_original_path)

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

    @classmethod
    def set_archived(cls, path: Union[str, Path], archived: bool) -> None:
        """
        Set the archived status of a saved model.

        Parameters
        ----------
        path : Union[str, Path]
            Path to the saved model directory.
        archived : bool
            Whether the model is archived.
        """
        path = Path(path)
        metadata_file = path / cls.METADATA_FILE

        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

        with open(metadata_file, "r") as f:
            metadata = json.load(f)

        metadata["is_archived"] = archived

        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

    @classmethod
    def set_favorite(cls, path: Union[str, Path], favorite: bool) -> None:
        """
        Set the favorite status of a saved model.

        Parameters
        ----------
        path : Union[str, Path]
            Path to the saved model directory.
        favorite : bool
            Whether the model is a favorite.
        """
        path = Path(path)
        metadata_file = path / cls.METADATA_FILE

        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

        with open(metadata_file, "r") as f:
            metadata = json.load(f)

        metadata["is_favorite"] = favorite

        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)


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
            "is_archived": False,
            "is_favorite": False,
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
            # Optimization settings
            "bounds_config": session_state.get("bounds_config"),
            "seasonal_indices": session_state.get("seasonal_indices"),
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

    @classmethod
    def set_archived(cls, path: Union[str, Path], archived: bool) -> None:
        """
        Set the archived status of a saved config.

        Parameters
        ----------
        path : Union[str, Path]
            Path to the saved config directory.
        archived : bool
            Whether the config is archived.
        """
        path = Path(path)
        metadata_file = path / cls.METADATA_FILE

        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

        with open(metadata_file, "r") as f:
            metadata = json.load(f)

        metadata["is_archived"] = archived

        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

    @classmethod
    def set_favorite(cls, path: Union[str, Path], favorite: bool) -> None:
        """
        Set the favorite status of a saved config.

        Parameters
        ----------
        path : Union[str, Path]
            Path to the saved config directory.
        favorite : bool
            Whether the config is a favorite.
        """
        path = Path(path)
        metadata_file = path / cls.METADATA_FILE

        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

        with open(metadata_file, "r") as f:
            metadata = json.load(f)

        metadata["is_favorite"] = favorite

        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)


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

        # CRITICAL: Populate config_state with channels, owned_media, controls from loaded config
        # Without this, build_config_from_state() will have empty lists when user clicks "Build Configuration"
        if config.channels:
            config_state["channels"] = [
                {
                    "name": ch.name,
                    "display_name": ch.display_name,
                    "categories": ch.categories,
                    "adstock_type": ch.adstock_type.value if hasattr(ch.adstock_type, 'value') else str(ch.adstock_type),
                    "roi_prior_low": ch.roi_prior_low,
                    "roi_prior_mid": ch.roi_prior_mid,
                    "roi_prior_high": ch.roi_prior_high,
                    "curve_sharpness_override": ch.curve_sharpness_override,
                }
                for ch in config.channels
            ]

        if config.owned_media:
            config_state["owned_media"] = [
                {
                    "name": om.name,
                    "display_name": om.display_name,
                    "categories": om.categories,
                    "adstock_type": om.adstock_type.value if hasattr(om.adstock_type, 'value') else str(om.adstock_type),
                    "curve_sharpness_override": om.curve_sharpness_override,
                    "include_roi": om.include_roi,
                    "roi_prior_low": om.roi_prior_low,
                    "roi_prior_mid": om.roi_prior_mid,
                    "roi_prior_high": om.roi_prior_high,
                }
                for om in config.owned_media
            ]

        if config.controls:
            config_state["controls"] = [
                {
                    "name": ctrl.name,
                    "display_name": ctrl.display_name,
                    "categories": ctrl.categories,
                    "sign_constraint": ctrl.sign_constraint.value if hasattr(ctrl.sign_constraint, 'value') else str(ctrl.sign_constraint),
                    "is_dummy": ctrl.is_dummy,
                    "scale": ctrl.scale,
                }
                for ctrl in config.controls
            ]

        if config.competitors:
            config_state["competitors"] = [
                {
                    "name": comp.name,
                    "display_name": comp.display_name,
                    "categories": comp.categories,
                    "adstock_type": comp.adstock_type.value if hasattr(comp.adstock_type, 'value') else str(comp.adstock_type),
                }
                for comp in config.competitors
            ]

        if config.dummy_variables:
            config_state["dummy_variables"] = [
                {
                    "name": dv.name,
                    "start_date": dv.start_date,
                    "end_date": dv.end_date,
                    "categories": dv.categories,
                    "sign_constraint": dv.sign_constraint.value if hasattr(dv.sign_constraint, 'value') else str(dv.sign_constraint),
                }
                for dv in config.dummy_variables
            ]

        # Also populate adstock/saturation/seasonality/sampling settings
        if config.adstock:
            config_state["l_max"] = config.adstock.l_max
            config_state["short_decay"] = config.adstock.short_decay
            config_state["medium_decay"] = config.adstock.medium_decay
            config_state["long_decay"] = config.adstock.long_decay

        if config.saturation:
            config_state["curve_sharpness"] = config.saturation.curve_sharpness

        if config.seasonality:
            config_state["yearly_seasonality"] = config.seasonality.yearly_seasonality

        if config.sampling:
            config_state["draws"] = config.sampling.draws
            config_state["tune"] = config.sampling.tune
            config_state["chains"] = config.sampling.chains
            config_state["target_accept"] = config.sampling.target_accept
            config_state["random_seed"] = config.sampling.random_seed

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

    # Restore multiselect states from loaded config (for Configure Model page widgets)
    # Prefer config values (source of truth) over session_state values
    if config is not None and config.channels:
        updates["channel_multiselect"] = [ch.name for ch in config.channels]
    elif session_state.get("channel_multiselect"):
        updates["channel_multiselect"] = session_state["channel_multiselect"]

    if config is not None and config.owned_media:
        updates["owned_media_multiselect"] = [om.name for om in config.owned_media]

    if config is not None and config.controls:
        # Include non-dummy controls only (dummies are auto-added from dummy_variables)
        updates["control_multiselect"] = [ctrl.name for ctrl in config.controls if not ctrl.is_dummy]
    elif session_state.get("control_multiselect"):
        updates["control_multiselect"] = session_state["control_multiselect"]

    # Restore optimization settings
    if session_state.get("bounds_config"):
        updates["bounds_config"] = session_state["bounds_config"]
    if session_state.get("seasonal_indices"):
        updates["seasonal_indices"] = session_state["seasonal_indices"]

    return updates


# =============================================================================
# Disaggregation Configuration Persistence
# =============================================================================

def load_disaggregation_configs(model_path: Union[str, Path]) -> list[dict]:
    """
    Load saved disaggregation configurations from a model's session_state.json.

    Parameters
    ----------
    model_path : Union[str, Path]
        Path to the model directory.

    Returns
    -------
    list[dict]
        List of saved disaggregation configurations.
    """
    model_path = Path(model_path)
    session_file = model_path / "session_state.json"

    if not session_file.exists():
        return []

    try:
        with open(session_file, "r") as f:
            session_state = json.load(f)

        disagg_data = session_state.get("disaggregation", {})
        return disagg_data.get("saved_configs", [])
    except Exception as e:
        logger.warning(f"Could not load disaggregation configs from {model_path}: {e}")
        return []


def load_disaggregation_weights(model_path: Union[str, Path], config_id: str) -> pd.DataFrame | None:
    """
    Load saved weighting DataFrame for a disaggregation config.

    Parameters
    ----------
    model_path : Union[str, Path]
        Path to the model directory.
    config_id : str
        ID of the disaggregation config.

    Returns
    -------
    pd.DataFrame | None
        The weighting DataFrame if found, None otherwise.
    """
    model_path = Path(model_path)
    weights_file = model_path / f"disagg_weights_{config_id}.parquet"

    if weights_file.exists():
        try:
            return pd.read_parquet(weights_file)
        except Exception as e:
            logger.warning(f"Could not load disaggregation weights from {weights_file}: {e}")
            return None
    return None


def save_disaggregation_config(
    model_path: Union[str, Path],
    config: dict,
    weighting_df: pd.DataFrame = None,
    set_active: bool = True
) -> None:
    """
    Save a disaggregation configuration to a model's session_state.json.

    Parameters
    ----------
    model_path : Union[str, Path]
        Path to the model directory.
    config : dict
        Disaggregation configuration dict with keys:
        - id, name, created_at, granular_name_cols, date_column, weight_column,
        - entity_to_channel_mapping, notes (optional)
    weighting_df : pd.DataFrame, optional
        The weighting DataFrame to save alongside the config.
    set_active : bool
        If True, set this config as active and deactivate others.
    """
    model_path = Path(model_path)
    session_file = model_path / "session_state.json"

    # Load existing session state
    if session_file.exists():
        with open(session_file, "r") as f:
            session_state = json.load(f)
    else:
        session_state = {}

    # Initialize disaggregation section if needed
    if "disaggregation" not in session_state:
        session_state["disaggregation"] = {"saved_configs": []}

    # Generate ID if not present
    if "id" not in config or not config["id"]:
        import uuid
        config["id"] = f"disagg_{uuid.uuid4().hex[:8]}"

    # Set is_active
    config["is_active"] = set_active

    # If setting active, deactivate others
    if set_active:
        for existing in session_state["disaggregation"]["saved_configs"]:
            existing["is_active"] = False

    # Check if updating existing config (by id)
    existing_ids = [c["id"] for c in session_state["disaggregation"]["saved_configs"]]
    if config["id"] in existing_ids:
        # Update existing
        for i, existing in enumerate(session_state["disaggregation"]["saved_configs"]):
            if existing["id"] == config["id"]:
                session_state["disaggregation"]["saved_configs"][i] = config
                break
    else:
        # Add new
        session_state["disaggregation"]["saved_configs"].append(config)

    # Save back
    with open(session_file, "w") as f:
        json.dump(session_state, f, indent=2)

    # Save weighting DataFrame if provided
    if weighting_df is not None:
        weights_file = model_path / f"disagg_weights_{config['id']}.parquet"
        weighting_df.to_parquet(weights_file)
        logger.info(f"Saved disaggregation weights to {weights_file}")

    logger.info(f"Saved disaggregation config '{config['name']}' to {model_path}")


def delete_disaggregation_config(model_path: Union[str, Path], config_id: str) -> bool:
    """
    Delete a disaggregation configuration from a model.

    Parameters
    ----------
    model_path : Union[str, Path]
        Path to the model directory.
    config_id : str
        ID of the config to delete.

    Returns
    -------
    bool
        True if deleted, False if not found.
    """
    model_path = Path(model_path)
    session_file = model_path / "session_state.json"

    if not session_file.exists():
        return False

    with open(session_file, "r") as f:
        session_state = json.load(f)

    if "disaggregation" not in session_state:
        return False

    configs = session_state["disaggregation"]["saved_configs"]
    original_count = len(configs)
    session_state["disaggregation"]["saved_configs"] = [
        c for c in configs if c["id"] != config_id
    ]

    if len(session_state["disaggregation"]["saved_configs"]) < original_count:
        with open(session_file, "w") as f:
            json.dump(session_state, f, indent=2)

        # Also delete the weights file if it exists
        weights_file = model_path / f"disagg_weights_{config_id}.parquet"
        if weights_file.exists():
            weights_file.unlink()
            logger.info(f"Deleted disaggregation weights file {weights_file}")

        logger.info(f"Deleted disaggregation config {config_id} from {model_path}")
        return True

    return False


def validate_disaggregation_config(config: dict, model_channels: list[str]) -> tuple[bool, str]:
    """
    Validate a saved disaggregation config against current model channels.

    Parameters
    ----------
    config : dict
        Saved disaggregation configuration.
    model_channels : list[str]
        Current model channel names.

    Returns
    -------
    tuple[bool, str]
        (is_valid, error_message). If valid, error_message is empty.
    """
    mapped_channels = set(config.get("entity_to_channel_mapping", {}).values())
    # Remove "-- Not Mapped --" from validation
    mapped_channels.discard("-- Not Mapped --")

    current_channels = set(model_channels)

    missing = mapped_channels - current_channels
    if missing:
        return False, f"Mapping references channels not in model: {', '.join(sorted(missing))}"

    return True, ""


# =============================================================================
# Export Column Schema Persistence
# =============================================================================

def get_client_schemas_dir(client: str) -> Path:
    """
    Get the schemas directory for a specific client.

    Parameters
    ----------
    client : str
        Client name.

    Returns
    -------
    Path
        Path to schemas directory (created if doesn't exist).
    """
    schemas_dir = get_clients_dir() / client / "schemas"
    schemas_dir.mkdir(parents=True, exist_ok=True)
    return schemas_dir


def list_export_schemas(client: str) -> list[dict]:
    """
    List all export column schemas for a client.

    Parameters
    ----------
    client : str
        Client name.

    Returns
    -------
    list[dict]
        List of schema metadata (id, name, created_at, description, path).
    """
    schemas_dir = get_client_schemas_dir(client)
    schemas = []

    for schema_file in schemas_dir.glob("*.json"):
        try:
            with open(schema_file, "r") as f:
                schema = json.load(f)
            schemas.append({
                "id": schema.get("id"),
                "name": schema.get("name"),
                "description": schema.get("description"),
                "created_at": schema.get("created_at"),
                "updated_at": schema.get("updated_at"),
                "path": str(schema_file)
            })
        except Exception as e:
            logger.warning(f"Could not read schema from {schema_file}: {e}")

    return sorted(schemas, key=lambda x: x.get("created_at", ""), reverse=True)


def load_export_schema(schema_path: Union[str, Path]) -> dict:
    """
    Load an export column schema from file.

    Parameters
    ----------
    schema_path : Union[str, Path]
        Path to the schema JSON file.

    Returns
    -------
    dict
        Schema dictionary.
    """
    with open(schema_path, "r") as f:
        return json.load(f)


def save_export_schema(client: str, schema: dict) -> Path:
    """
    Save an export column schema for a client.

    Parameters
    ----------
    client : str
        Client name.
    schema : dict
        Schema dict (ExportColumnSchema.model_dump()).

    Returns
    -------
    Path
        Path to saved schema file.
    """
    import uuid

    schemas_dir = get_client_schemas_dir(client)

    # Generate ID if not present
    if "id" not in schema or not schema["id"]:
        schema["id"] = f"schema_{uuid.uuid4().hex[:8]}"

    # Set timestamps
    if "created_at" not in schema or not schema["created_at"]:
        schema["created_at"] = datetime.now().isoformat()
    schema["updated_at"] = datetime.now().isoformat()

    # Save to file
    schema_file = schemas_dir / f"{schema['id']}.json"
    with open(schema_file, "w") as f:
        json.dump(schema, f, indent=2)

    logger.info(f"Saved export schema '{schema['name']}' to {schema_file}")
    return schema_file


def delete_export_schema(schema_path: Union[str, Path]) -> bool:
    """
    Delete an export column schema file.

    Parameters
    ----------
    schema_path : Union[str, Path]
        Path to the schema file.

    Returns
    -------
    bool
        True if deleted, False if file didn't exist.
    """
    schema_path = Path(schema_path)
    if schema_path.exists():
        schema_path.unlink()
        logger.info(f"Deleted export schema at {schema_path}")
        return True
    return False


def load_model_schema_override(model_path: Union[str, Path]) -> Optional[dict]:
    """
    Load model-level schema override from model's session_state.json.

    Parameters
    ----------
    model_path : Union[str, Path]
        Path to model directory.

    Returns
    -------
    Optional[dict]
        Schema override dict if exists, None otherwise.
    """
    model_path = Path(model_path)
    session_file = model_path / "session_state.json"

    if not session_file.exists():
        return None

    try:
        with open(session_file, "r") as f:
            session_state = json.load(f)
        return session_state.get("export_schema_override")
    except Exception as e:
        logger.warning(f"Could not load model schema override from {model_path}: {e}")
        return None


def save_model_schema_override(model_path: Union[str, Path], schema: dict) -> None:
    """
    Save a model-level schema override to model's session_state.json.

    Parameters
    ----------
    model_path : Union[str, Path]
        Path to model directory.
    schema : dict
        Schema override dict.
    """
    import uuid

    model_path = Path(model_path)
    session_file = model_path / "session_state.json"

    # Load existing session state
    if session_file.exists():
        with open(session_file, "r") as f:
            session_state = json.load(f)
    else:
        session_state = {}

    # Generate ID if not present
    if "id" not in schema or not schema["id"]:
        schema["id"] = f"schema_{uuid.uuid4().hex[:8]}"

    # Set timestamps
    if "created_at" not in schema or not schema["created_at"]:
        schema["created_at"] = datetime.now().isoformat()
    schema["updated_at"] = datetime.now().isoformat()

    # Mark as override
    schema["is_model_override"] = True
    session_state["export_schema_override"] = schema

    with open(session_file, "w") as f:
        json.dump(session_state, f, indent=2)

    logger.info(f"Saved model schema override to {model_path}")


# =============================================================================
# Forecast Persistence
# =============================================================================


class ForecastPersistence:
    """
    Persistence layer for saved forecasts.

    Stores forecasts as subdirectories within the model's folder:
    saved_models/{client}/{model_name}/forecasts/{forecast_id}/
        - metadata.json - SavedForecastMetadata
        - weekly_df.parquet - Weekly response data
        - channel_contributions.parquet - Channel breakdown
        - input_spend.parquet - Original uploaded spend data
        - config.json - Settings used (seasonal_indices, etc.)
    """

    FORECASTS_DIR = "forecasts"
    METADATA_FILE = "metadata.json"
    WEEKLY_FILE = "weekly_df.parquet"
    CONTRIBUTIONS_FILE = "channel_contributions.parquet"
    INPUT_SPEND_FILE = "input_spend.parquet"
    CONFIG_FILE = "config.json"

    @classmethod
    def save_forecast(
        cls,
        model_path: Union[str, Path],
        result: "ForecastResult",
        input_spend: pd.DataFrame,
        notes: Optional[str] = None,
    ) -> str:
        """
        Save a forecast to the model's forecast history.

        Parameters
        ----------
        model_path : Union[str, Path]
            Path to the saved model directory.
        result : ForecastResult
            The forecast result to save.
        input_spend : pd.DataFrame
            The original spend CSV uploaded by the user.
        notes : str, optional
            User notes about this forecast.

        Returns
        -------
        str
            The forecast ID (UUID).
        """
        import uuid
        from mmm_platform.config.schema import SavedForecastMetadata

        model_path = Path(model_path)
        forecasts_dir = model_path / cls.FORECASTS_DIR
        forecasts_dir.mkdir(exist_ok=True)

        # Generate forecast ID
        forecast_id = f"forecast_{uuid.uuid4().hex[:8]}"
        forecast_dir = forecasts_dir / forecast_id
        forecast_dir.mkdir()

        # Extract dates from weekly_df
        dates = pd.to_datetime(result.weekly_df["date"])
        start_date = dates.min().strftime("%Y-%m-%d")
        end_date = dates.max().strftime("%Y-%m-%d")

        # Create metadata
        metadata = SavedForecastMetadata(
            id=forecast_id,
            created_at=datetime.now().isoformat(),
            forecast_period=result.forecast_period,
            start_date=start_date,
            end_date=end_date,
            num_weeks=result.num_weeks,
            total_spend=result.total_spend,
            total_response=result.total_response,
            blended_roi=result.blended_roi,
            seasonal_applied=result.seasonal_applied,
            notes=notes,
        )

        # Save metadata
        with open(forecast_dir / cls.METADATA_FILE, "w") as f:
            json.dump(metadata.model_dump(), f, indent=2)

        # Save dataframes
        result.weekly_df.to_parquet(forecast_dir / cls.WEEKLY_FILE)
        result.channel_contributions.to_parquet(forecast_dir / cls.CONTRIBUTIONS_FILE)
        input_spend.to_parquet(forecast_dir / cls.INPUT_SPEND_FILE)

        # Save config (seasonal indices, etc.)
        config = {
            "seasonal_indices": result.seasonal_indices,
            "demand_index": result.demand_index,
            "sanity_check": result.sanity_check,
        }
        with open(forecast_dir / cls.CONFIG_FILE, "w") as f:
            json.dump(config, f, indent=2)

        logger.info(f"Saved forecast {forecast_id} to {forecast_dir}")
        return forecast_id

    @classmethod
    def list_forecasts(
        cls,
        model_path: Union[str, Path],
    ) -> list["SavedForecastMetadata"]:
        """
        List all saved forecasts for a model.

        Parameters
        ----------
        model_path : Union[str, Path]
            Path to the saved model directory.

        Returns
        -------
        list[SavedForecastMetadata]
            List of forecast metadata, sorted by created_at (newest first).
        """
        from mmm_platform.config.schema import SavedForecastMetadata

        model_path = Path(model_path)
        forecasts_dir = model_path / cls.FORECASTS_DIR

        if not forecasts_dir.exists():
            return []

        forecasts = []
        for forecast_dir in forecasts_dir.iterdir():
            if not forecast_dir.is_dir():
                continue

            metadata_file = forecast_dir / cls.METADATA_FILE
            if not metadata_file.exists():
                continue

            try:
                with open(metadata_file, "r") as f:
                    data = json.load(f)
                forecasts.append(SavedForecastMetadata(**data))
            except Exception as e:
                logger.warning(f"Could not load forecast metadata from {forecast_dir}: {e}")

        # Sort by created_at (newest first)
        forecasts.sort(key=lambda x: x.created_at, reverse=True)
        return forecasts

    @classmethod
    def load_forecast(
        cls,
        model_path: Union[str, Path],
        forecast_id: str,
    ) -> tuple["ForecastResult", pd.DataFrame, "SavedForecastMetadata"]:
        """
        Load a specific forecast.

        Parameters
        ----------
        model_path : Union[str, Path]
            Path to the saved model directory.
        forecast_id : str
            The forecast ID to load.

        Returns
        -------
        tuple[ForecastResult, pd.DataFrame, SavedForecastMetadata]
            The forecast result, input spend DataFrame, and metadata.

        Raises
        ------
        FileNotFoundError
            If the forecast does not exist.
        """
        from mmm_platform.config.schema import SavedForecastMetadata
        from mmm_platform.forecasting.forecast_engine import ForecastResult

        model_path = Path(model_path)
        forecast_dir = model_path / cls.FORECASTS_DIR / forecast_id

        if not forecast_dir.exists():
            raise FileNotFoundError(f"Forecast not found: {forecast_id}")

        # Load metadata
        with open(forecast_dir / cls.METADATA_FILE, "r") as f:
            metadata = SavedForecastMetadata(**json.load(f))

        # Load dataframes
        weekly_df = pd.read_parquet(forecast_dir / cls.WEEKLY_FILE)
        channel_contributions = pd.read_parquet(forecast_dir / cls.CONTRIBUTIONS_FILE)
        input_spend = pd.read_parquet(forecast_dir / cls.INPUT_SPEND_FILE)

        # Load config
        with open(forecast_dir / cls.CONFIG_FILE, "r") as f:
            config = json.load(f)

        # Reconstruct ForecastResult
        result = ForecastResult(
            total_response=metadata.total_response,
            total_ci_low=weekly_df["ci_low"].sum(),
            total_ci_high=weekly_df["ci_high"].sum(),
            weekly_df=weekly_df,
            channel_contributions=channel_contributions,
            total_spend=metadata.total_spend,
            num_weeks=metadata.num_weeks,
            seasonal_applied=metadata.seasonal_applied,
            seasonal_indices=config.get("seasonal_indices", {}),
            demand_index=config.get("demand_index", 1.0),
            forecast_period=metadata.forecast_period,
            sanity_check=config.get("sanity_check", {}),
        )

        return result, input_spend, metadata

    @classmethod
    def delete_forecast(
        cls,
        model_path: Union[str, Path],
        forecast_id: str,
    ) -> bool:
        """
        Delete a forecast.

        Parameters
        ----------
        model_path : Union[str, Path]
            Path to the saved model directory.
        forecast_id : str
            The forecast ID to delete.

        Returns
        -------
        bool
            True if deleted, False if not found.
        """
        import shutil

        model_path = Path(model_path)
        forecast_dir = model_path / cls.FORECASTS_DIR / forecast_id

        if not forecast_dir.exists():
            return False

        shutil.rmtree(forecast_dir)
        logger.info(f"Deleted forecast {forecast_id}")
        return True

    @classmethod
    def get_historical_spend(
        cls,
        model_path: Union[str, Path],
    ) -> pd.DataFrame:
        """
        Get combined spend data from all historical forecasts.

        Used for overlap detection when uploading new forecast data.

        Parameters
        ----------
        model_path : Union[str, Path]
            Path to the saved model directory.

        Returns
        -------
        pd.DataFrame
            Combined input_spend data from all forecasts with forecast_id column.
            Returns empty DataFrame if no forecasts exist.
        """
        model_path = Path(model_path)
        forecasts_dir = model_path / cls.FORECASTS_DIR

        if not forecasts_dir.exists():
            return pd.DataFrame()

        all_spend = []
        for forecast_dir in forecasts_dir.iterdir():
            if not forecast_dir.is_dir():
                continue

            input_file = forecast_dir / cls.INPUT_SPEND_FILE
            if not input_file.exists():
                continue

            try:
                df = pd.read_parquet(input_file)
                df["_forecast_id"] = forecast_dir.name
                all_spend.append(df)
            except Exception as e:
                logger.warning(f"Could not load spend from {forecast_dir}: {e}")

        if not all_spend:
            return pd.DataFrame()

        combined = pd.concat(all_spend, ignore_index=True)
        return combined
