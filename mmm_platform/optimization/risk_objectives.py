"""
Risk-aware objective functions for the fallback optimizer.

Provides VaR, CVaR, and Sharpe ratio objectives using posterior samples
to capture uncertainty in response estimates.
"""

from typing import Literal
import numpy as np
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class PosteriorSamples:
    """
    Container for posterior samples needed for risk computation.

    Attributes
    ----------
    beta_samples : np.ndarray
        Saturation beta values, shape (n_samples, n_channels)
    lam_samples : np.ndarray
        Saturation lambda values, shape (n_samples, n_channels)
    n_samples : int
        Number of posterior samples
    n_channels : int
        Number of channels
    """

    beta_samples: np.ndarray
    lam_samples: np.ndarray
    n_samples: int
    n_channels: int

    @classmethod
    def from_idata(cls, idata, n_samples: int = 500) -> "PosteriorSamples":
        """
        Extract posterior samples from InferenceData.

        Parameters
        ----------
        idata : arviz.InferenceData
            Fitted model inference data
        n_samples : int
            Number of samples to use (subsampled if posterior has more)

        Returns
        -------
        PosteriorSamples
            Container with beta and lam samples
        """
        posterior = idata.posterior

        # Stack chains and draws into single sample dimension
        beta_all = posterior["saturation_beta"].stack(sample=("chain", "draw")).values
        lam_all = posterior["saturation_lam"].stack(sample=("chain", "draw")).values

        # Shape is (n_channels, n_samples_total) - transpose to (n_samples_total, n_channels)
        beta_all = beta_all.T
        lam_all = lam_all.T

        total_samples = beta_all.shape[0]
        n_channels = beta_all.shape[1]

        # Subsample if we have more than needed
        if total_samples > n_samples:
            indices = np.random.choice(total_samples, n_samples, replace=False)
            beta_samples = beta_all[indices]
            lam_samples = lam_all[indices]
        else:
            beta_samples = beta_all
            lam_samples = lam_all
            n_samples = total_samples

        logger.debug(
            f"Extracted {n_samples} posterior samples for {n_channels} channels"
        )

        return cls(
            beta_samples=beta_samples,
            lam_samples=lam_samples,
            n_samples=n_samples,
            n_channels=n_channels,
        )


RiskProfile = Literal["mean", "var", "cvar", "sharpe"]


class RiskAwareObjective:
    """
    Computes risk-aware objectives for budget optimization.

    Supports:
    - mean: Expected response (risk neutral) - uses fast path with posterior means
    - var: Value at Risk (5th percentile - conservative)
    - cvar: Conditional VaR (mean of bottom 5% - very conservative)
    - sharpe: Sharpe ratio (mean / std - risk-adjusted)

    Parameters
    ----------
    posterior_samples : PosteriorSamples
        Beta and lambda samples from posterior
    x_maxes : np.ndarray
        Maximum spend values per channel (for normalization)
    target_scale : float
        Scale factor for target variable
    num_periods : int
        Number of periods for optimization horizon
    risk_profile : str
        Which risk metric to optimize ("mean", "var", "cvar", "sharpe")
    confidence_level : float
        Confidence level for VaR/CVaR (default 0.95 = 5th percentile worst case)
    risk_free_rate : float
        Risk-free rate for Sharpe ratio (default 0.0)
    seasonal_indices : np.ndarray, optional
        Per-channel seasonal effectiveness multipliers, shape (n_channels,).
        Index > 1 = more effective during the optimization period.
        If None, no seasonal adjustment is applied (all indices = 1.0).
    """

    def __init__(
        self,
        posterior_samples: PosteriorSamples,
        x_maxes: np.ndarray,
        target_scale: float,
        num_periods: int,
        risk_profile: RiskProfile = "mean",
        confidence_level: float = 0.95,
        risk_free_rate: float = 0.0,
        seasonal_indices: np.ndarray | None = None,
    ):
        self.samples = posterior_samples
        self.x_maxes = np.maximum(x_maxes, 1e-9)  # Avoid division by zero
        self.target_scale = target_scale
        self.num_periods = num_periods
        self.risk_profile = risk_profile
        self.confidence_level = confidence_level
        self.risk_free_rate = risk_free_rate

        # Seasonal indices: default to 1.0 (no adjustment) if not provided
        if seasonal_indices is not None:
            self.seasonal_indices = np.array(seasonal_indices)
        else:
            self.seasonal_indices = np.ones(posterior_samples.n_channels)

        # Precompute posterior means for fast path (mean profile) and gradient
        self.beta_mean = posterior_samples.beta_samples.mean(axis=0)
        self.lam_mean = posterior_samples.lam_samples.mean(axis=0)

        logger.debug(
            f"RiskAwareObjective initialized: profile={risk_profile}, "
            f"confidence={confidence_level}, n_samples={posterior_samples.n_samples}, "
            f"seasonal_indices={'custom' if seasonal_indices is not None else 'none'}"
        )

    def compute_response_distribution(self, x: np.ndarray) -> np.ndarray:
        """
        Compute response for all posterior samples at allocation x.

        Parameters
        ----------
        x : np.ndarray
            Per-period spend in scaled units, shape (n_channels,)

        Returns
        -------
        np.ndarray
            Response values for each posterior sample, shape (n_samples,)
        """
        # Normalize by x_max (same as current implementation)
        x_normalized = x / self.x_maxes  # Shape: (n_channels,)

        # Broadcast for all samples: (n_samples, n_channels)
        x_norm_broadcast = x_normalized[np.newaxis, :]

        # Compute saturation for all samples
        # saturation(x) = (1 - exp(-lam*x)) / (1 + exp(-lam*x))
        exp_term = np.exp(-self.samples.lam_samples * x_norm_broadcast)
        saturation = (1 - exp_term) / (1 + exp_term)  # Shape: (n_samples, n_channels)

        # Apply seasonal indices to adjust effectiveness per channel
        # seasonal_indices shape: (n_channels,) -> broadcast to (n_samples, n_channels)
        seasonal_adjusted = saturation * self.seasonal_indices[np.newaxis, :]

        # Compute response for each sample
        # response = sum(beta * target_scale * adjusted_saturation) * num_periods
        response_per_sample = (
            np.sum(self.samples.beta_samples * self.target_scale * seasonal_adjusted, axis=1)
            * self.num_periods
        )  # Shape: (n_samples,)

        return response_per_sample

    def objective(self, x: np.ndarray) -> float:
        """
        Compute objective value based on risk profile.

        Parameters
        ----------
        x : np.ndarray
            Allocation in scaled units per period

        Returns
        -------
        float
            Negative of risk-adjusted response (for minimization)
        """
        if self.risk_profile == "mean":
            return self._objective_mean(x)
        elif self.risk_profile == "var":
            return self._objective_var(x)
        elif self.risk_profile == "cvar":
            return self._objective_cvar(x)
        elif self.risk_profile == "sharpe":
            return self._objective_sharpe(x)
        else:
            raise ValueError(f"Unknown risk profile: {self.risk_profile}")

    def _objective_mean(self, x: np.ndarray) -> float:
        """Mean response (original behavior - fast path using means only)."""
        x_normalized = x / self.x_maxes
        exp_term = np.exp(-self.lam_mean * x_normalized)
        saturation = (1 - exp_term) / (1 + exp_term)
        # Apply seasonal indices
        seasonal_adjusted = saturation * self.seasonal_indices
        response = (
            np.sum(self.beta_mean * self.target_scale * seasonal_adjusted) * self.num_periods
        )
        return -response

    def _objective_var(self, x: np.ndarray) -> float:
        """Value at Risk: percentile of response distribution."""
        responses = self.compute_response_distribution(x)
        # VaR at 95% confidence = 5th percentile (worst 5% of outcomes)
        percentile = (1 - self.confidence_level) * 100
        var = np.percentile(responses, percentile)
        return -var  # Maximize the worst-case response

    def _objective_cvar(self, x: np.ndarray) -> float:
        """Conditional VaR: Expected response below VaR threshold."""
        responses = self.compute_response_distribution(x)
        percentile = (1 - self.confidence_level) * 100
        var_threshold = np.percentile(responses, percentile)
        # CVaR = mean of responses below VaR threshold
        below_var = responses[responses <= var_threshold]
        if len(below_var) == 0:
            cvar = var_threshold
        else:
            cvar = np.mean(below_var)
        return -cvar  # Maximize expected shortfall

    def _objective_sharpe(self, x: np.ndarray) -> float:
        """Sharpe ratio: (mean - risk_free) / std."""
        responses = self.compute_response_distribution(x)
        mean_response = np.mean(responses)
        std_response = np.std(responses)

        if std_response < 1e-9:
            # No variance - just return mean
            return -(mean_response - self.risk_free_rate)

        sharpe = (mean_response - self.risk_free_rate) / std_response
        return -sharpe  # Maximize Sharpe ratio

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """
        Compute gradient of objective.

        For mean: uses analytical gradient (fast)
        For VaR/CVaR/Sharpe: uses numerical gradient (finite differences)

        Parameters
        ----------
        x : np.ndarray
            Allocation in scaled units per period

        Returns
        -------
        np.ndarray
            Gradient of objective w.r.t. each channel
        """
        if self.risk_profile == "mean":
            return self._gradient_mean(x)
        else:
            return self._gradient_numerical(x)

    def _gradient_mean(self, x: np.ndarray) -> np.ndarray:
        """Analytical gradient for mean objective (fast)."""
        x_normalized = x / self.x_maxes
        exp_term = np.exp(-self.lam_mean * x_normalized)
        sat_deriv = 2 * self.lam_mean * exp_term / (1 + exp_term) ** 2
        # Apply seasonal indices to gradient
        grad = (
            -self.beta_mean
            * self.target_scale
            * sat_deriv
            * self.seasonal_indices
            / self.x_maxes
            * self.num_periods
        )
        return grad

    def _gradient_numerical(self, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """Numerical gradient using central differences."""
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += eps
            x_minus[i] -= eps
            grad[i] = (self.objective(x_plus) - self.objective(x_minus)) / (2 * eps)
        return grad

    def compute_all_risk_metrics(self, x: np.ndarray) -> dict:
        """
        Compute all risk metrics at a given allocation.

        Useful for populating OptimizationResult with full risk analysis.

        Parameters
        ----------
        x : np.ndarray
            Allocation in scaled units per period

        Returns
        -------
        dict
            All risk metrics: mean, std, var, cvar, sharpe
        """
        responses = self.compute_response_distribution(x)

        mean_response = float(np.mean(responses))
        std_response = float(np.std(responses))

        percentile = (1 - self.confidence_level) * 100
        var_response = float(np.percentile(responses, percentile))

        below_var = responses[responses <= var_response]
        cvar_response = float(np.mean(below_var)) if len(below_var) > 0 else var_response

        sharpe = (
            (mean_response - self.risk_free_rate) / std_response
            if std_response > 1e-9
            else 0.0
        )

        return {
            "expected_response": mean_response,
            "response_std": std_response,
            "response_var": var_response,
            "response_cvar": cvar_response,
            "response_sharpe": float(sharpe),
            "response_ci_low": float(np.percentile(responses, 5)),
            "response_ci_high": float(np.percentile(responses, 95)),
        }
