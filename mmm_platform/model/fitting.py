"""
Model fitting utilities and helpers.
"""

import numpy as np
import arviz as az
from typing import Any, Optional
import logging

logger = logging.getLogger(__name__)


class ModelFitter:
    """
    Utilities for model fitting and convergence diagnostics.
    """

    @staticmethod
    def check_convergence(idata: Any, r_hat_threshold: float = 1.01) -> dict:
        """
        Check model convergence using R-hat and divergences.

        Parameters
        ----------
        idata : InferenceData
            ArviZ InferenceData object.
        r_hat_threshold : float
            Maximum acceptable R-hat value.

        Returns
        -------
        dict
            Convergence diagnostics.
        """
        result = {
            "converged": True,
            "high_rhat_params": [],
            "divergences": 0,
            "warnings": [],
        }

        # Check R-hat
        try:
            summary = az.summary(
                idata,
                var_names=["intercept", "saturation_beta", "adstock_alpha", "saturation_lam"]
            )

            if "r_hat" in summary.columns:
                high_rhat = summary[summary["r_hat"] > r_hat_threshold]
                if len(high_rhat) > 0:
                    result["converged"] = False
                    result["high_rhat_params"] = high_rhat.index.tolist()
                    result["warnings"].append(
                        f"{len(high_rhat)} parameters have r_hat > {r_hat_threshold}"
                    )

        except Exception as e:
            logger.warning(f"Could not compute R-hat: {e}")

        # Check divergences
        try:
            if hasattr(idata, "sample_stats") and hasattr(idata.sample_stats, "diverging"):
                divergences = int(idata.sample_stats.diverging.sum().values)
                result["divergences"] = divergences
                if divergences > 0:
                    result["warnings"].append(f"{divergences} divergent transitions")
                    if divergences > 100:
                        result["converged"] = False

        except Exception as e:
            logger.warning(f"Could not check divergences: {e}")

        return result

    @staticmethod
    def get_parameter_summary(
        idata: Any,
        var_names: Optional[list[str]] = None,
        hdi_prob: float = 0.95
    ) -> dict:
        """
        Get summary statistics for model parameters.

        Parameters
        ----------
        idata : InferenceData
            ArviZ InferenceData object.
        var_names : list[str], optional
            Variables to summarize.
        hdi_prob : float
            HDI probability.

        Returns
        -------
        dict
            Parameter summaries.
        """
        if var_names is None:
            var_names = [
                "intercept",
                "saturation_beta",
                "adstock_alpha",
                "saturation_lam",
                "gamma_control",
            ]

        summaries = {}

        for var in var_names:
            try:
                summary = az.summary(idata, var_names=[var], hdi_prob=hdi_prob)
                summaries[var] = summary.to_dict()
            except Exception as e:
                logger.warning(f"Could not summarize {var}: {e}")

        return summaries

    @staticmethod
    def get_effective_sample_size(idata: Any) -> dict:
        """
        Get effective sample size for key parameters.

        Parameters
        ----------
        idata : InferenceData
            ArviZ InferenceData object.

        Returns
        -------
        dict
            ESS statistics.
        """
        try:
            summary = az.summary(
                idata,
                var_names=["saturation_beta", "adstock_alpha"]
            )

            ess_bulk_min = summary["ess_bulk"].min()
            ess_tail_min = summary["ess_tail"].min()

            return {
                "ess_bulk_min": float(ess_bulk_min),
                "ess_tail_min": float(ess_tail_min),
                "sufficient": ess_bulk_min > 400 and ess_tail_min > 400,
            }

        except Exception as e:
            logger.warning(f"Could not compute ESS: {e}")
            return {"error": str(e)}

    @staticmethod
    def compute_waic(idata: Any) -> Optional[float]:
        """
        Compute WAIC for model comparison.

        Parameters
        ----------
        idata : InferenceData
            ArviZ InferenceData object.

        Returns
        -------
        float or None
            WAIC value.
        """
        try:
            waic = az.waic(idata)
            return float(waic.waic)
        except Exception as e:
            logger.warning(f"Could not compute WAIC: {e}")
            return None

    @staticmethod
    def compute_loo(idata: Any) -> Optional[float]:
        """
        Compute LOO-CV for model comparison.

        Parameters
        ----------
        idata : InferenceData
            ArviZ InferenceData object.

        Returns
        -------
        float or None
            LOO value.
        """
        try:
            loo = az.loo(idata)
            return float(loo.loo)
        except Exception as e:
            logger.warning(f"Could not compute LOO: {e}")
            return None
