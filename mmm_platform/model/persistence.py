"""
Model persistence - save and load fitted models.
"""

import pickle
import cloudpickle
import json
from pathlib import Path
from typing import Union, Optional, Any
from datetime import datetime
import logging
import hashlib

logger = logging.getLogger(__name__)


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

        # Save metadata
        metadata = {
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "fitted_at": mmm_wrapper.fitted_at.isoformat() if mmm_wrapper.fitted_at else None,
            "fit_duration_seconds": mmm_wrapper.fit_duration_seconds,
            "config_name": mmm_wrapper.config.name,
            "n_channels": len(mmm_wrapper.config.channels),
            "n_controls": len(mmm_wrapper.control_cols) if mmm_wrapper.control_cols else 0,
            "include_data": include_data,
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
    def list_saved_models(cls, directory: Union[str, Path]) -> list[dict]:
        """
        List all saved models in a directory.

        Parameters
        ----------
        directory : Union[str, Path]
            Directory to search.

        Returns
        -------
        list[dict]
            List of model metadata.
        """
        directory = Path(directory)
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

        # Sort by creation date
        models.sort(key=lambda x: x.get("created_at", ""), reverse=True)

        return models

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
