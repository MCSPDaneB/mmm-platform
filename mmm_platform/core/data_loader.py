"""
Data loading functionality for various sources.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Optional, BinaryIO
import logging

from ..config.schema import ModelConfig, DataConfig

logger = logging.getLogger(__name__)


class DataLoader:
    """Load and prepare data from various sources."""

    def __init__(self, config: Union[ModelConfig, DataConfig]):
        """
        Initialize DataLoader.

        Parameters
        ----------
        config : Union[ModelConfig, DataConfig]
            Configuration containing data settings.
        """
        if isinstance(config, ModelConfig):
            self.config = config.data
            self.model_config = config
        else:
            self.config = config
            self.model_config = None

    def load_csv(
        self,
        source: Union[str, Path, BinaryIO],
        **kwargs
    ) -> pd.DataFrame:
        """
        Load data from a CSV file or file-like object.

        Parameters
        ----------
        source : Union[str, Path, BinaryIO]
            File path or file-like object (e.g., uploaded file).
        **kwargs
            Additional arguments passed to pd.read_csv.

        Returns
        -------
        pd.DataFrame
            Loaded and prepared dataframe.
        """
        # Set default read options
        read_options = {
            "parse_dates": [self.config.date_column],
            "dayfirst": self.config.dayfirst,
        }
        if self.config.date_format:
            read_options["date_format"] = self.config.date_format

        # Override with any provided kwargs
        read_options.update(kwargs)

        try:
            df = pd.read_csv(source, **read_options)
            logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        except Exception as e:
            raise ValueError(f"Error loading CSV: {e}")

        return self._prepare_dataframe(df)

    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare dataframe after loading.

        Parameters
        ----------
        df : pd.DataFrame
            Raw dataframe.

        Returns
        -------
        pd.DataFrame
            Prepared dataframe.
        """
        date_col = self.config.date_column

        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            df[date_col] = pd.to_datetime(
                df[date_col],
                dayfirst=self.config.dayfirst,
                format=self.config.date_format
            )

        # Sort by date
        df = df.sort_values(date_col).reset_index(drop=True)

        # Create time index
        df["t"] = np.arange(1, len(df) + 1)

        logger.info(f"Data prepared: {df[date_col].min()} to {df[date_col].max()}")

        return df

    def create_dummy_variables(
        self,
        df: pd.DataFrame,
        config: ModelConfig
    ) -> pd.DataFrame:
        """
        Create dummy variables based on configuration.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe.
        config : ModelConfig
            Model configuration with dummy variable settings.

        Returns
        -------
        pd.DataFrame
            Dataframe with dummy variables added.
        """
        df = df.copy()
        date_col = config.data.date_column

        # Create configured dummy variables
        for dummy in config.dummy_variables:
            # Convert dates to datetime for proper comparison
            start = pd.to_datetime(dummy.start_date)
            end = pd.to_datetime(dummy.end_date)
            df[dummy.name] = (
                (df[date_col] >= start) &
                (df[date_col] <= end)
            ).astype(int)
            # Log count of active periods to verify it worked
            count = df[dummy.name].sum()
            logger.info(f"Created dummy variable: {dummy.name} ({count} periods active)")

        # Create month dummies if configured
        if config.month_dummies and config.month_dummies.months:
            for month in config.month_dummies.months:
                col_name = f"month_{month:02d}"
                df[col_name] = (df[date_col].dt.month == month).astype(int)
                logger.info(f"Created month dummy: {col_name}")

        return df

    def apply_sign_adjustments(
        self,
        df: pd.DataFrame,
        config: ModelConfig
    ) -> tuple[pd.DataFrame, list[str]]:
        """
        Apply sign adjustments to control variables.

        For controls with negative sign constraints, we invert the data
        so that a positive coefficient (from HalfNormal prior) results
        in a negative effect.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe.
        config : ModelConfig
            Model configuration.

        Returns
        -------
        tuple[pd.DataFrame, list[str]]
            Adjusted dataframe and list of final control column names.
        """
        df = df.copy()
        final_control_cols = []

        for ctrl in config.controls:
            if ctrl.name not in df.columns:
                logger.warning(f"Control column not found: {ctrl.name}")
                continue

            if ctrl.sign_constraint == "negative":
                # Invert the data for negative constraints
                new_col = f"{ctrl.name}_inv"
                df[new_col] = -1 * df[ctrl.name]
                final_control_cols.append(new_col)
                logger.info(f"Inverted control: {ctrl.name} -> {new_col}")
            else:
                final_control_cols.append(ctrl.name)

        # Handle month dummies with sign constraints
        if config.month_dummies:
            for month, constraint in config.month_dummies.sign_constraints.items():
                col_name = f"month_{month:02d}"
                if col_name in df.columns:
                    if constraint == "negative":
                        new_col = f"{col_name}_inv"
                        df[new_col] = -1 * df[col_name]
                        final_control_cols.append(new_col)
                    else:
                        final_control_cols.append(col_name)

            # Add unconstrained month dummies
            for month in config.month_dummies.months:
                if month not in config.month_dummies.sign_constraints:
                    col_name = f"month_{month:02d}"
                    if col_name in df.columns and col_name not in final_control_cols:
                        final_control_cols.append(col_name)

        # Handle dummy variables with sign constraints
        for dummy in config.dummy_variables:
            if dummy.name in df.columns:
                if dummy.sign_constraint == "negative":
                    new_col = f"{dummy.name}_inv"
                    df[new_col] = -1 * df[dummy.name]
                    final_control_cols.append(new_col)
                else:
                    final_control_cols.append(dummy.name)

        # Remove duplicates while preserving order
        final_control_cols = list(dict.fromkeys(final_control_cols))

        logger.info(f"Final control columns ({len(final_control_cols)}): {final_control_cols}")

        return df, final_control_cols

    def scale_data(
        self,
        df: pd.DataFrame,
        config: ModelConfig
    ) -> pd.DataFrame:
        """
        Scale numeric columns according to configuration.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe.
        config : ModelConfig
            Model configuration.

        Returns
        -------
        pd.DataFrame
            Scaled dataframe.
        """
        df = df.copy()
        scale_factor = config.data.spend_scale

        # Columns to scale: target + channels + scalable controls
        scale_cols = [config.data.target_column] + config.get_channel_columns()

        # Add scalable control columns
        for ctrl in config.controls:
            if ctrl.scale and ctrl.name in df.columns:
                scale_cols.append(ctrl.name)

        # Apply scaling
        for col in scale_cols:
            if col in df.columns:
                df[col] = df[col] / scale_factor

        logger.info(f"Scaled {len(scale_cols)} columns by factor {scale_factor}")

        return df

    def prepare_model_data(
        self,
        df: pd.DataFrame,
        config: ModelConfig
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, list[str]]:
        """
        Complete data preparation pipeline for model fitting.

        Parameters
        ----------
        df : pd.DataFrame
            Raw input dataframe.
        config : ModelConfig
            Model configuration.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame, pd.Series, list[str]]
            - Original dataframe with dummies added
            - Scaled dataframe ready for modeling
            - Target series (scaled)
            - List of control column names (with sign adjustments)
        """
        # Create dummy variables
        df = self.create_dummy_variables(df, config)

        # Apply sign adjustments
        df, control_cols = self.apply_sign_adjustments(df, config)

        # Scale data
        df_scaled = self.scale_data(df, config)

        # Prepare X and y
        feature_cols = (
            [config.data.date_column] +
            config.get_channel_columns() +
            control_cols
        )

        # Remove any columns that don't exist
        feature_cols = [c for c in feature_cols if c in df_scaled.columns]

        X = df_scaled[feature_cols]
        y = df_scaled[config.data.target_column]

        return df, df_scaled, y, control_cols
