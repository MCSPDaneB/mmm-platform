"""
Data transformation functions for adstock and saturation.
"""

import numpy as np
import pandas as pd
from typing import Optional
import logging

from ..config.schema import ModelConfig, AdstockType, sharpness_to_percentile, sharpness_label_to_value

logger = logging.getLogger(__name__)


class TransformEngine:
    """
    Engine for applying adstock and saturation transformations.

    Used for:
    - Prior calibration (computing expected transforms)
    - Visualization (showing transform curves)
    - Post-hoc analysis
    """

    def __init__(self, config: ModelConfig):
        """
        Initialize TransformEngine.

        Parameters
        ----------
        config : ModelConfig
            Model configuration.
        """
        self.config = config
        self._adstock_cache = {}

    def get_adstock_decay(self, channel: str) -> float:
        """
        Get the adstock decay rate for a channel.

        Parameters
        ----------
        channel : str
            Channel name.

        Returns
        -------
        float
            Decay rate (alpha parameter).
        """
        channel_config = self.config.get_channel_by_name(channel)

        if channel_config is None:
            # Default to medium
            return self.config.adstock.medium_decay

        adstock_type = channel_config.adstock_type

        if adstock_type == AdstockType.SHORT or adstock_type == "short":
            return self.config.adstock.short_decay
        elif adstock_type == AdstockType.LONG or adstock_type == "long":
            return self.config.adstock.long_decay
        else:
            return self.config.adstock.medium_decay

    def get_adstock_means(self) -> np.ndarray:
        """
        Get adstock decay means for all channels.

        Returns
        -------
        np.ndarray
            Array of decay rates.
        """
        return np.array([
            self.get_adstock_decay(ch)
            for ch in self.config.get_channel_columns()
        ])

    def compute_geometric_adstock(
        self,
        x: np.ndarray,
        alpha: float,
        l_max: Optional[int] = None
    ) -> np.ndarray:
        """
        Apply geometric adstock transformation.

        Parameters
        ----------
        x : np.ndarray
            Input time series (spend).
        alpha : float
            Decay rate (0 = no carryover, 1 = full carryover).
        l_max : int, optional
            Maximum lag. Defaults to config value.

        Returns
        -------
        np.ndarray
            Adstocked time series.
        """
        if l_max is None:
            l_max = self.config.adstock.l_max

        # Geometric weights
        weights = np.array([alpha ** i for i in range(l_max)])
        weights = weights / weights.sum()  # Normalize

        # Apply convolution
        x_adstocked = np.convolve(x, weights, mode='full')[:len(x)]

        return x_adstocked

    def compute_logistic_saturation(
        self,
        x: np.ndarray,
        lam: float
    ) -> np.ndarray:
        """
        Apply logistic saturation transformation.

        Parameters
        ----------
        x : np.ndarray
            Input time series (typically adstocked spend).
        lam : float
            Saturation parameter (higher = faster saturation).

        Returns
        -------
        np.ndarray
            Saturated time series (values in [0, 1]).
        """
        return (1 - np.exp(-lam * x)) / (1 + np.exp(-lam * x))

    def compute_channel_lam(
        self,
        df: pd.DataFrame,
        channel: str,
        percentile: int = 50
    ) -> float:
        """
        Compute lambda parameter for a channel based on data.

        Lambda is set so that half-saturation occurs at the specified
        percentile of non-zero normalized spend.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe with channel data.
        channel : str
            Channel column name.
        percentile : int
            Percentile for half-saturation point.

        Returns
        -------
        float
            Lambda parameter.
        """
        x = df[channel].values.astype(float)
        x_max = x.max()

        if x_max == 0:
            return 1.0

        # Normalize (as PyMC-Marketing does internally)
        x_normalized = x / (x_max + 1e-9)

        # Get non-zero values
        x_pos = x_normalized[x_normalized > 0]

        if len(x_pos) == 0:
            return 1.0

        # Target spend at percentile
        target_spend = np.percentile(x_pos, percentile)

        # Solve for lambda: at target_spend, we want ~50% saturation
        # logistic_saturation(target_spend, lam) ≈ 0.5
        # This gives: lam ≈ log(3) / target_spend
        lam = np.log(3) / (target_spend + 0.01)

        return lam

    def get_channel_percentile(self, channel: str) -> int:
        """
        Get the effective saturation percentile for a channel.

        Uses per-channel override if set, otherwise uses global curve_sharpness.

        Parameters
        ----------
        channel : str
            Channel name.

        Returns
        -------
        int
            Effective percentile for half-saturation.
        """
        channel_config = self.config.get_channel_by_name(channel)

        # Check for per-channel override
        if channel_config and channel_config.curve_sharpness_override:
            override_value = sharpness_label_to_value(channel_config.curve_sharpness_override)
            if override_value is not None:
                return sharpness_to_percentile(override_value)

        # Use global curve_sharpness setting
        global_sharpness = self.config.saturation.curve_sharpness
        return sharpness_to_percentile(global_sharpness)

    def compute_all_channel_lams(
        self,
        df: pd.DataFrame,
        percentile: Optional[int] = None
    ) -> np.ndarray:
        """
        Compute lambda parameters for all channels.

        Uses per-channel curve sharpness overrides if set, otherwise uses
        the global curve_sharpness setting.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe with channel data.
        percentile : int, optional
            Override percentile for all channels. If None, uses config settings.

        Returns
        -------
        np.ndarray
            Array of lambda parameters.
        """
        lams = []
        for ch in self.config.get_channel_columns():
            if percentile is not None:
                # Use explicit percentile override
                ch_percentile = percentile
            else:
                # Use per-channel or global sharpness setting
                ch_percentile = self.get_channel_percentile(ch)

            lams.append(self.compute_channel_lam(df, ch, ch_percentile))

        return np.array(lams)

    def visualize_adstock_curve(
        self,
        alpha: float,
        l_max: Optional[int] = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate adstock decay curve for visualization.

        Parameters
        ----------
        alpha : float
            Decay rate.
        l_max : int, optional
            Maximum lag.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (lags, weights) for plotting.
        """
        if l_max is None:
            l_max = self.config.adstock.l_max

        lags = np.arange(l_max)
        weights = np.array([alpha ** i for i in range(l_max)])
        weights = weights / weights.sum()

        return lags, weights

    def visualize_saturation_curve(
        self,
        lam: float,
        x_max: float = 1.0,
        n_points: int = 100
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate saturation curve for visualization.

        Parameters
        ----------
        lam : float
            Saturation parameter.
        x_max : float
            Maximum x value.
        n_points : int
            Number of points.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (x, y) for plotting.
        """
        x = np.linspace(0, x_max, n_points)
        y = self.compute_logistic_saturation(x, lam)
        return x, y

    def apply_full_transform(
        self,
        x: np.ndarray,
        alpha: float,
        lam: float,
        l_max: Optional[int] = None
    ) -> np.ndarray:
        """
        Apply full adstock + saturation transformation.

        Parameters
        ----------
        x : np.ndarray
            Input time series.
        alpha : float
            Adstock decay rate.
        lam : float
            Saturation parameter.
        l_max : int, optional
            Maximum lag.

        Returns
        -------
        np.ndarray
            Transformed time series.
        """
        x_adstocked = self.compute_geometric_adstock(x, alpha, l_max)
        x_saturated = self.compute_logistic_saturation(x_adstocked, lam)
        return x_saturated
