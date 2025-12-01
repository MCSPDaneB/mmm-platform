"""
Main MMM wrapper class that orchestrates the modeling pipeline.
"""

import numpy as np
import pandas as pd
from typing import Optional, Any
from datetime import datetime
import logging

from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation
from pymc_extras.prior import Prior

from ..config.schema import ModelConfig
from ..core.data_loader import DataLoader
from ..core.validation import DataValidator, ValidationResult
from ..core.transforms import TransformEngine
from ..core.priors import PriorCalibrator

logger = logging.getLogger(__name__)


class MMMWrapper:
    """
    High-level wrapper for PyMC-Marketing MMM.

    This class orchestrates the entire modeling pipeline:
    - Data loading and preparation
    - Validation
    - Prior calibration
    - Model building and fitting
    - Results extraction
    """

    def __init__(self, config: ModelConfig):
        """
        Initialize MMMWrapper.

        Parameters
        ----------
        config : ModelConfig
            Model configuration.
        """
        self.config = config
        self.data_loader = DataLoader(config)
        self.validator = DataValidator(config)
        self.transform_engine = TransformEngine(config)
        self.prior_calibrator = PriorCalibrator(config)

        # State
        self.df_raw: Optional[pd.DataFrame] = None
        self.df_scaled: Optional[pd.DataFrame] = None
        self.control_cols: Optional[list[str]] = None
        self.lam_vec: Optional[np.ndarray] = None
        self.beta_mu: Optional[np.ndarray] = None
        self.beta_sigma: Optional[np.ndarray] = None

        # PyMC model
        self.mmm: Optional[MMM] = None
        self.idata: Optional[Any] = None  # InferenceData

        # Metadata
        self.created_at: datetime = datetime.now()
        self.fitted_at: Optional[datetime] = None
        self.fit_duration_seconds: Optional[float] = None

    def load_data(self, source, **kwargs) -> ValidationResult:
        """
        Load and validate data.

        Parameters
        ----------
        source : Union[str, Path, BinaryIO]
            Data source (file path or file-like object).
        **kwargs
            Additional arguments for data loading.

        Returns
        -------
        ValidationResult
            Validation results.
        """
        # Load data
        self.df_raw = self.data_loader.load_csv(source, **kwargs)

        # Validate
        validation = self.validator.validate_all(self.df_raw)

        if validation.valid:
            logger.info("Data loaded and validated successfully")
        else:
            logger.warning(f"Data validation issues: {validation}")

        return validation

    def prepare_data(self) -> None:
        """
        Prepare data for modeling.

        This includes:
        - Creating dummy variables
        - Applying sign adjustments
        - Scaling data
        """
        if self.df_raw is None:
            raise ValueError("No data loaded. Call load_data() first.")

        # Full preparation pipeline
        self.df_raw, self.df_scaled, _, self.control_cols = \
            self.data_loader.prepare_model_data(self.df_raw, self.config)

        # Compute transform parameters
        self.lam_vec = self.transform_engine.compute_all_channel_lams(self.df_scaled)

        # Calibrate priors
        self.beta_mu, self.beta_sigma = self.prior_calibrator.calibrate_beta_priors(
            self.df_scaled, self.lam_vec
        )

        logger.info("Data prepared for modeling")

    def build_model(self) -> MMM:
        """
        Build the PyMC-Marketing MMM model.

        Returns
        -------
        MMM
            Built (but not fitted) model.
        """
        if self.df_scaled is None:
            raise ValueError("Data not prepared. Call prepare_data() first.")

        # Add time trend if configured
        if self.config.data.include_trend:
            if "t" not in self.df_scaled.columns:
                self.df_scaled["t"] = np.arange(1, len(self.df_scaled) + 1)
            if "t" not in self.control_cols:
                self.control_cols.append("t")
                logger.info("Added time trend 't' as control variable")

        # Ensure at least one control column (pymc-marketing 0.16+ requirement)
        if not self.control_cols:
            # Add time trend as fallback if no controls configured
            if "t" not in self.df_scaled.columns:
                self.df_scaled["t"] = np.arange(1, len(self.df_scaled) + 1)
            self.control_cols = ["t"]
            logger.warning("No control columns configured. Using time trend 't' as fallback.")

        # Build priors
        model_config = self._build_model_config()
        sampler_config = {"target_accept": self.config.sampling.target_accept}

        # Create model
        self.mmm = MMM(
            model_config=model_config,
            sampler_config=sampler_config,
            date_column=self.config.data.date_column,
            channel_columns=self.config.get_channel_columns(),
            control_columns=self.control_cols,
            adstock=GeometricAdstock(l_max=self.config.adstock.l_max),
            saturation=LogisticSaturation(),
            yearly_seasonality=self.config.seasonality.yearly_seasonality,
        )

        # Build model graph
        X = self._get_feature_dataframe()
        y = self.df_scaled[self.config.data.target_column]
        self.mmm.build_model(X, y)

        logger.info("Model built successfully")

        return self.mmm

    def fit(self, **kwargs) -> Any:
        """
        Fit the model.

        Parameters
        ----------
        **kwargs
            Override sampling parameters.

        Returns
        -------
        InferenceData
            ArviZ InferenceData object with posterior samples.
        """
        if self.mmm is None:
            self.build_model()

        X = self._get_feature_dataframe()
        y = self.df_scaled[self.config.data.target_column]

        # Sampling parameters
        sampling = self.config.sampling
        fit_kwargs = {
            "draws": kwargs.get("draws", sampling.draws),
            "tune": kwargs.get("tune", sampling.tune),
            "chains": kwargs.get("chains", sampling.chains),
            "target_accept": kwargs.get("target_accept", sampling.target_accept),
            "random_seed": kwargs.get("random_seed", sampling.random_seed),
        }
        if sampling.cores:
            fit_kwargs["cores"] = sampling.cores

        # Use nutpie sampler if available (much faster)
        use_nutpie = kwargs.get("use_nutpie", True)
        if use_nutpie:
            try:
                import nutpie
                fit_kwargs["nuts_sampler"] = "nutpie"
                logger.info("Using nutpie sampler (faster)")
            except ImportError:
                logger.info("nutpie not installed, using default sampler. Install with: pip install nutpie")

        logger.info(f"Fitting model: {fit_kwargs}")

        import time
        start_time = time.time()

        try:
            self.idata = self.mmm.fit(X, y, **fit_kwargs)
            self.fitted_at = datetime.now()
            self.fit_duration_seconds = time.time() - start_time

            logger.info(f"Model fitted in {self.fit_duration_seconds:.1f} seconds")

        except Exception as e:
            logger.error(f"Model fitting failed: {e}")
            raise

        return self.idata

    def _build_model_config(self) -> dict:
        """Build the model_config dictionary for PyMC-Marketing."""
        # Beta prior (channel effects)
        beta_prior = Prior(
            "LogNormal",
            mu=self.beta_mu,
            sigma=self.beta_sigma,
            dims="channel",
        )

        # Adstock prior
        adstock_alpha, adstock_beta = self.prior_calibrator.get_adstock_prior_params()
        adstock_prior = Prior(
            "Beta",
            alpha=adstock_alpha,
            beta=adstock_beta,
            dims="channel",
        )

        # Saturation lambda prior
        lam_mu, lam_sigma = self.prior_calibrator.get_saturation_prior_params(
            self.lam_vec
        )
        saturation_lam_prior = Prior(
            "LogNormal",
            mu=lam_mu,
            sigma=lam_sigma,
            dims="channel",
        )

        # Control prior
        ctrl_prior_config = self.config.control_prior
        gamma_control_prior = Prior(
            ctrl_prior_config.distribution,
            sigma=ctrl_prior_config.sigma,
            dims="control",
        )

        return {
            "intercept": Prior("Normal", mu=0, sigma=2),
            "saturation_beta": beta_prior,
            "adstock_alpha": adstock_prior,
            "saturation_lam": saturation_lam_prior,
            "gamma_control": gamma_control_prior,
            "likelihood": Prior(
                "Normal",
                sigma=Prior("HalfNormal", sigma=2),
            ),
        }

    def _get_feature_dataframe(self) -> pd.DataFrame:
        """Get the feature dataframe for model fitting."""
        feature_cols = (
            [self.config.data.date_column] +
            self.config.get_channel_columns() +
            self.control_cols
        )
        # Filter to columns that exist
        feature_cols = [c for c in feature_cols if c in self.df_scaled.columns]
        return self.df_scaled[feature_cols]

    def get_contributions(self) -> pd.DataFrame:
        """
        Get channel and control contributions over time.

        Returns
        -------
        pd.DataFrame
            Contributions dataframe.
        """
        if self.idata is None:
            raise ValueError("Model not fitted. Call fit() first.")

        contribs = self.mmm.compute_mean_contributions_over_time(self.idata)
        return contribs

    def get_contributions_real_units(self) -> pd.DataFrame:
        """
        Get contributions rescaled to original units.

        Returns
        -------
        pd.DataFrame
            Contributions in original revenue units.
        """
        contribs = self.get_contributions()
        contribs_real = contribs * self.config.data.revenue_scale
        return contribs_real

    def get_channel_roi(self) -> pd.DataFrame:
        """
        Calculate ROI for each channel.

        Returns
        -------
        pd.DataFrame
            ROI by channel.
        """
        contribs = self.get_contributions()
        channel_cols = self.config.get_channel_columns()

        results = []
        for ch in channel_cols:
            if ch in contribs.columns and ch in self.df_scaled.columns:
                contrib = contribs[ch].sum()
                spend = self.df_scaled[ch].sum()
                roi = contrib / (spend + 1e-9)

                results.append({
                    "channel": ch,
                    "contribution": contrib * self.config.data.revenue_scale,
                    "spend": spend * self.config.data.spend_scale,
                    "roi": roi,
                })

        return pd.DataFrame(results)

    def get_fit_statistics(self) -> dict:
        """
        Calculate model fit statistics.

        Returns
        -------
        dict
            R², RMSE, MAE, etc.
        """
        if self.idata is None:
            raise ValueError("Model not fitted. Call fit() first.")

        contribs = self.get_contributions()
        target_col = self.config.data.target_column

        # Get actual values aligned with contributions index
        df_indexed = self.df_scaled.set_index(self.config.data.date_column)
        actual = df_indexed[target_col].reindex(contribs.index)

        # Fitted = sum of all components
        component_cols = [c for c in contribs.columns if c != target_col]
        fitted = contribs[component_cols].sum(axis=1)

        residuals = actual - fitted

        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((actual - actual.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot

        rmse = np.sqrt(ss_res / len(actual))
        mae = np.mean(np.abs(residuals))
        mape = np.mean(np.abs(residuals / actual)) * 100

        return {
            "r2": float(r2),
            "rmse": float(rmse),
            "mae": float(mae),
            "mape": float(mape),
            "n_observations": len(actual),
            "fit_duration_seconds": self.fit_duration_seconds,
        }

    def get_data_summary(self) -> dict:
        """Get summary of loaded data."""
        if self.df_raw is None:
            return {}
        return self.validator.get_data_summary(self.df_raw)

    def get_state(self) -> dict:
        """
        Get current state of the wrapper for persistence.

        Returns
        -------
        dict
            State dictionary.
        """
        return {
            "config": self.config.model_dump(),
            "control_cols": self.control_cols,
            "lam_vec": self.lam_vec.tolist() if self.lam_vec is not None else None,
            "beta_mu": self.beta_mu.tolist() if self.beta_mu is not None else None,
            "beta_sigma": self.beta_sigma.tolist() if self.beta_sigma is not None else None,
            "created_at": self.created_at.isoformat(),
            "fitted_at": self.fitted_at.isoformat() if self.fitted_at else None,
            "fit_duration_seconds": self.fit_duration_seconds,
        }
