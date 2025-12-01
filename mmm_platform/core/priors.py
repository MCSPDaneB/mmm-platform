"""
Prior calibration functions for ROI-informed beta priors.
"""

import numpy as np
import pandas as pd
from typing import Optional
import logging

from ..config.schema import ModelConfig
from .transforms import TransformEngine

logger = logging.getLogger(__name__)


class PriorCalibrator:
    """
    Calibrate model priors based on expected ROI values.

    This class implements the ROI-to-beta calibration that ensures
    the model's prior beliefs about channel effectiveness match
    business expectations.
    """

    def __init__(self, config: ModelConfig):
        """
        Initialize PriorCalibrator.

        Parameters
        ----------
        config : ModelConfig
            Model configuration.
        """
        self.config = config
        self.transform_engine = TransformEngine(config)

    def calibrate_beta_priors(
        self,
        df_scaled: pd.DataFrame,
        lam_vec: np.ndarray,
        l_max: Optional[int] = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Calibrate beta priors from ROI expectations.

        This accounts for PyMC-Marketing's internal scaling:
        - Normalizes each channel by its max: x_norm = x / x_max
        - Normalizes target by its max: y_norm = y / y_max
        - Computes: contribution = target_scale * beta * saturation(adstock(x_norm))

        So: ROI = contribution / spend
                = target_scale * beta * sum(f(g(x_norm))) / sum(x)

        Solving: beta = ROI * sum(x) / (target_scale * sum(f(g(x_norm))))

        This method handles all effective channels:
        - Paid media channels (always have ROI)
        - Owned media with adstock+saturation (ROI if include_roi=True)

        Parameters
        ----------
        df_scaled : pd.DataFrame
            Scaled dataframe.
        lam_vec : np.ndarray
            Lambda values for each effective channel.
        l_max : int, optional
            Maximum adstock lag.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (mu, sigma) for LogNormal beta prior.
        """
        if l_max is None:
            l_max = self.config.adstock.l_max

        target_col = self.config.data.target_column
        target_scale = df_scaled[target_col].max()

        roi_low, roi_mid, roi_high = self.config.get_roi_dicts()

        # Get effective channels and their adstock means
        effective_channels = self.transform_engine.get_effective_channel_columns()
        adstock_means = self.transform_engine.get_all_channel_adstock_means()

        beta_low_list = []
        beta_mid_list = []
        beta_high_list = []

        for i, ch in enumerate(effective_channels):
            x = df_scaled[ch].values.astype(float)
            x_max = x.max()

            if x_max == 0:
                # No spend in this channel - use default priors
                beta_low_list.append(0.01)
                beta_mid_list.append(0.1)
                beta_high_list.append(1.0)
                continue

            # Normalize as PyMC-Marketing does internally
            x_normalized = x / (x_max + 1e-9)

            alpha = adstock_means[i]
            lam = lam_vec[i]

            # Apply transforms to normalized data
            x_ad = self.transform_engine.compute_geometric_adstock(
                x_normalized, alpha, l_max
            )
            x_sat = self.transform_engine.compute_logistic_saturation(x_ad, lam)

            denom = x_sat.sum() + 1e-9
            total_spend = x.sum() + 1e-9

            # Get ROI priors (with defaults)
            # Owned media without include_roi=True won't be in roi_dicts
            r_low = roi_low.get(ch, 0.2)
            r_mid = roi_mid.get(ch, 1.0)
            r_high = roi_high.get(ch, 5.0)

            # Calculate beta values that produce expected ROI
            beta_low_list.append(r_low * total_spend / (target_scale * denom))
            beta_mid_list.append(r_mid * total_spend / (target_scale * denom))
            beta_high_list.append(r_high * total_spend / (target_scale * denom))

        beta_low = np.array(beta_low_list)
        beta_mid = np.array(beta_mid_list)
        beta_high = np.array(beta_high_list)

        # Convert to LogNormal parameters
        mu = np.log(beta_mid)
        sigma = np.clip((np.log(beta_high) - np.log(beta_low)) / 4.0, 0.1, 1.5)

        logger.info(f"Calibrated beta priors: mu range [{mu.min():.2f}, {mu.max():.2f}]")

        return mu, sigma

    def validate_prior_roi(
        self,
        df_scaled: pd.DataFrame,
        beta_mu: np.ndarray,
        lam_vec: np.ndarray,
        l_max: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Validate that beta priors translate back to intended ROI values.

        Parameters
        ----------
        df_scaled : pd.DataFrame
            Scaled dataframe.
        beta_mu : np.ndarray
            Log of beta center values.
        lam_vec : np.ndarray
            Lambda values for each channel.
        l_max : int, optional
            Maximum adstock lag.

        Returns
        -------
        pd.DataFrame
            Validation results with target vs prior ROI.
        """
        if l_max is None:
            l_max = self.config.adstock.l_max

        target_col = self.config.data.target_column
        target_scale = df_scaled[target_col].max()

        _, roi_mid, _ = self.config.get_roi_dicts()
        channel_cols = self.config.get_channel_columns()

        results = []

        for i, ch in enumerate(channel_cols):
            x = df_scaled[ch].values.astype(float)
            x_max = x.max()

            if x_max == 0:
                results.append({
                    "channel": ch,
                    "target_roi": roi_mid.get(ch, 1.0),
                    "prior_roi": 0.0,
                    "match": False,
                    "error_pct": 100.0,
                })
                continue

            x_normalized = x / (x_max + 1e-9)

            alpha = self.transform_engine.get_adstock_decay(ch)
            lam = lam_vec[i]
            beta_med = np.exp(beta_mu[i])

            x_ad = self.transform_engine.compute_geometric_adstock(
                x_normalized, alpha, l_max
            )
            x_sat = self.transform_engine.compute_logistic_saturation(x_ad, lam)

            # Contribution on original scale
            total_contrib = target_scale * beta_med * x_sat.sum()
            total_spend = x.sum() + 1e-9
            prior_roi = total_contrib / total_spend

            target_roi = roi_mid.get(ch, 1.0)
            error_pct = abs(prior_roi - target_roi) / (target_roi + 1e-9) * 100
            match = error_pct < 5  # Within 5%

            results.append({
                "channel": ch,
                "target_roi": target_roi,
                "prior_roi": prior_roi,
                "match": match,
                "error_pct": error_pct,
            })

        return pd.DataFrame(results)

    def get_adstock_prior_params(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Get Beta distribution parameters for adstock priors.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (alpha, beta) parameters for Beta distribution.
        """
        K = self.config.adstock.prior_concentration
        # Use effective channel adstock means (includes owned media with adstock+saturation)
        adstock_means = self.transform_engine.get_all_channel_adstock_means()

        alpha_params = adstock_means * K
        beta_params = (1 - adstock_means) * K

        return alpha_params, beta_params

    def get_saturation_prior_params(
        self,
        lam_vec: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """
        Get LogNormal parameters for saturation lambda priors.

        Parameters
        ----------
        lam_vec : np.ndarray
            Computed lambda values.

        Returns
        -------
        tuple[np.ndarray, float]
            (mu, sigma) for LogNormal distribution.
        """
        mu = np.log(lam_vec)
        sigma = self.config.saturation.lam_sigma

        return mu, sigma

    def get_all_priors_summary(
        self,
        df_scaled: pd.DataFrame
    ) -> dict:
        """
        Get a summary of all calibrated priors.

        Parameters
        ----------
        df_scaled : pd.DataFrame
            Scaled dataframe.

        Returns
        -------
        dict
            Summary of all prior configurations.
        """
        lam_vec = self.transform_engine.compute_all_channel_lams(df_scaled)
        beta_mu, beta_sigma = self.calibrate_beta_priors(df_scaled, lam_vec)
        adstock_alpha, adstock_beta = self.get_adstock_prior_params()
        lam_mu, lam_sigma = self.get_saturation_prior_params(lam_vec)

        channel_cols = self.config.get_channel_columns()

        summary = {
            "channels": {},
            "control_prior": {
                "distribution": self.config.control_prior.distribution,
                "sigma": self.config.control_prior.sigma,
            },
        }

        for i, ch in enumerate(channel_cols):
            summary["channels"][ch] = {
                "beta": {
                    "distribution": "LogNormal",
                    "mu": float(beta_mu[i]),
                    "sigma": float(beta_sigma[i]),
                    "median": float(np.exp(beta_mu[i])),
                },
                "adstock": {
                    "distribution": "Beta",
                    "alpha": float(adstock_alpha[i]),
                    "beta": float(adstock_beta[i]),
                    "mean": float(adstock_alpha[i] / (adstock_alpha[i] + adstock_beta[i])),
                },
                "saturation_lam": {
                    "distribution": "LogNormal",
                    "mu": float(lam_mu[i]),
                    "sigma": float(lam_sigma),
                    "median": float(np.exp(lam_mu[i])),
                },
            }

        return summary
