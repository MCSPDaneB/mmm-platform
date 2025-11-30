"""
Configuration loader for YAML files and database.
"""

import yaml
from pathlib import Path
from typing import Optional, Union
import logging

from .schema import ModelConfig

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Load and save model configurations from various sources."""

    @staticmethod
    def from_yaml(path: Union[str, Path]) -> ModelConfig:
        """
        Load configuration from a YAML file.

        Parameters
        ----------
        path : Union[str, Path]
            Path to the YAML configuration file.

        Returns
        -------
        ModelConfig
            Validated model configuration.

        Raises
        ------
        FileNotFoundError
            If the configuration file doesn't exist.
        ValueError
            If the configuration is invalid.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)

        try:
            config = ModelConfig(**config_dict)
            logger.info(f"Loaded configuration '{config.name}' from {path}")
            return config
        except Exception as e:
            raise ValueError(f"Invalid configuration in {path}: {e}")

    @staticmethod
    def to_yaml(config: ModelConfig, path: Union[str, Path]) -> None:
        """
        Save configuration to a YAML file.

        Parameters
        ----------
        config : ModelConfig
            Model configuration to save.
        path : Union[str, Path]
            Path to save the YAML file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        config_dict = config.model_dump(mode="json")

        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Saved configuration '{config.name}' to {path}")

    @staticmethod
    def from_dict(config_dict: dict) -> ModelConfig:
        """
        Load configuration from a dictionary.

        Parameters
        ----------
        config_dict : dict
            Configuration as a dictionary.

        Returns
        -------
        ModelConfig
            Validated model configuration.
        """
        return ModelConfig(**config_dict)

    @staticmethod
    def get_template() -> dict:
        """
        Get a template configuration dictionary with all options documented.

        Returns
        -------
        dict
            Template configuration with default values.
        """
        return {
            "name": "my_mmm_model",
            "description": "Marketing Mix Model configuration",
            "data": {
                "date_column": "time",
                "target_column": "revenue",
                "dayfirst": True,
                "revenue_scale": 1000.0,
                "spend_scale": 1000.0,
            },
            "category_columns": [
                {
                    "name": "Channel Type",
                    "options": ["Paid Media", "Organic", "Direct"],
                },
                {
                    "name": "Funnel Stage",
                    "options": ["Awareness", "Consideration", "Conversion"],
                },
            ],
            "channels": [
                {
                    "name": "channel_1_spend",
                    "display_name": "Channel 1",
                    "categories": {"Channel Type": "Paid Media", "Funnel Stage": "Awareness"},
                    "adstock_type": "medium",
                    "roi_prior_low": 0.5,
                    "roi_prior_mid": 2.0,
                    "roi_prior_high": 5.0,
                },
            ],
            "controls": [
                {
                    "name": "promo_variable",
                    "display_name": "Promotion",
                    "categories": {"Channel Type": "Promotions"},
                    "sign_constraint": "positive",
                    "is_dummy": True,
                    "scale": False,
                },
            ],
            "dummy_variables": [
                {
                    "name": "holiday_period",
                    "start_date": "2024-12-20",
                    "end_date": "2024-12-31",
                    "sign_constraint": "positive",
                },
            ],
            "month_dummies": {
                "months": [1, 2],
                "sign_constraints": {1: "negative", 2: "negative"},
            },
            "adstock": {
                "l_max": 8,
                "short_decay": 0.15,
                "medium_decay": 0.40,
                "long_decay": 0.70,
                "prior_concentration": 20.0,
            },
            "saturation": {
                "saturation_percentile": 50,
                "lam_sigma": 0.3,
            },
            "seasonality": {
                "yearly_seasonality": 2,
            },
            "sampling": {
                "draws": 1500,
                "tune": 1500,
                "chains": 4,
                "target_accept": 0.9,
                "random_seed": 42,
            },
            "control_prior": {
                "distribution": "HalfNormal",
                "sigma": 1.0,
            },
        }
