"""
Tests for risk-aware optimization objectives.

Tests cover:
- PosteriorSamples creation and extraction
- RiskAwareObjective with different risk profiles
- Gradient computation
- Risk metrics calculation
- Seasonal adjustments
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_idata():
    """Create mock InferenceData with posterior samples."""
    idata = Mock()

    # Create mock posterior with realistic shapes
    n_chains = 2
    n_draws = 100
    n_channels = 3

    # Create mock xarray DataArrays
    posterior = Mock()

    # saturation_beta: shape (n_channels, n_chains*n_draws) after stacking
    beta_values = np.random.uniform(0.5, 2.0, (n_channels, n_chains * n_draws))
    beta_mock = Mock()
    stacked_beta = Mock()
    stacked_beta.values = beta_values
    beta_mock.stack = Mock(return_value=stacked_beta)

    # saturation_lam: shape (n_channels, n_chains*n_draws) after stacking
    lam_values = np.random.uniform(1.0, 3.0, (n_channels, n_chains * n_draws))
    lam_mock = Mock()
    stacked_lam = Mock()
    stacked_lam.values = lam_values
    lam_mock.stack = Mock(return_value=stacked_lam)

    posterior.__getitem__ = Mock(side_effect=lambda k: {
        "saturation_beta": beta_mock,
        "saturation_lam": lam_mock,
    }.get(k))

    idata.posterior = posterior

    return idata


@pytest.fixture
def posterior_samples():
    """Create PosteriorSamples directly."""
    from mmm_platform.optimization.risk_objectives import PosteriorSamples

    np.random.seed(42)
    n_samples = 100
    n_channels = 3

    return PosteriorSamples(
        beta_samples=np.random.uniform(0.5, 2.0, (n_samples, n_channels)),
        lam_samples=np.random.uniform(1.0, 3.0, (n_samples, n_channels)),
        n_samples=n_samples,
        n_channels=n_channels,
    )


@pytest.fixture
def risk_objective(posterior_samples):
    """Create RiskAwareObjective with default settings."""
    from mmm_platform.optimization.risk_objectives import RiskAwareObjective

    return RiskAwareObjective(
        posterior_samples=posterior_samples,
        x_maxes=np.array([10.0, 8.0, 5.0]),
        target_scale=1000.0,
        num_periods=8,
        risk_profile="mean",
    )


# ============================================================================
# Test Classes
# ============================================================================

class TestPosteriorSamples:
    """Tests for PosteriorSamples dataclass."""

    def test_from_idata_returns_samples(self, mock_idata):
        """from_idata extracts samples from InferenceData."""
        from mmm_platform.optimization.risk_objectives import PosteriorSamples

        samples = PosteriorSamples.from_idata(mock_idata, n_samples=50)

        assert samples.n_samples == 50
        assert samples.n_channels == 3
        assert samples.beta_samples.shape == (50, 3)
        assert samples.lam_samples.shape == (50, 3)

    def test_from_idata_subsamples_when_needed(self, mock_idata):
        """Subsamples when more samples available than requested."""
        from mmm_platform.optimization.risk_objectives import PosteriorSamples

        samples = PosteriorSamples.from_idata(mock_idata, n_samples=25)

        # Should subsample to 25
        assert samples.n_samples == 25

    def test_from_idata_uses_all_if_fewer(self, mock_idata):
        """Uses all available if fewer than requested."""
        from mmm_platform.optimization.risk_objectives import PosteriorSamples

        # Request more than available (200 draws total)
        samples = PosteriorSamples.from_idata(mock_idata, n_samples=500)

        # Should use all 200
        assert samples.n_samples == 200

    def test_direct_initialization(self):
        """Direct initialization works."""
        from mmm_platform.optimization.risk_objectives import PosteriorSamples

        beta = np.ones((10, 2))
        lam = np.ones((10, 2))

        samples = PosteriorSamples(
            beta_samples=beta,
            lam_samples=lam,
            n_samples=10,
            n_channels=2,
        )

        assert samples.n_samples == 10
        assert samples.n_channels == 2


class TestRiskAwareObjectiveInit:
    """Tests for RiskAwareObjective initialization."""

    def test_init_with_defaults(self, posterior_samples):
        """Initializes with default parameters."""
        from mmm_platform.optimization.risk_objectives import RiskAwareObjective

        obj = RiskAwareObjective(
            posterior_samples=posterior_samples,
            x_maxes=np.array([10.0, 8.0, 5.0]),
            target_scale=1000.0,
            num_periods=8,
        )

        assert obj.risk_profile == "mean"
        assert obj.confidence_level == 0.95
        assert obj.num_periods == 8

    def test_init_with_custom_risk_profile(self, posterior_samples):
        """Accepts custom risk profile."""
        from mmm_platform.optimization.risk_objectives import RiskAwareObjective

        obj = RiskAwareObjective(
            posterior_samples=posterior_samples,
            x_maxes=np.array([10.0, 8.0, 5.0]),
            target_scale=1000.0,
            num_periods=8,
            risk_profile="var",
        )

        assert obj.risk_profile == "var"

    def test_init_with_seasonal_indices(self, posterior_samples):
        """Accepts seasonal indices."""
        from mmm_platform.optimization.risk_objectives import RiskAwareObjective

        seasonal = np.array([1.2, 0.9, 1.0])

        obj = RiskAwareObjective(
            posterior_samples=posterior_samples,
            x_maxes=np.array([10.0, 8.0, 5.0]),
            target_scale=1000.0,
            num_periods=8,
            seasonal_indices=seasonal,
        )

        np.testing.assert_array_equal(obj.seasonal_indices, seasonal)

    def test_init_defaults_seasonal_to_ones(self, posterior_samples):
        """Defaults seasonal indices to 1.0."""
        from mmm_platform.optimization.risk_objectives import RiskAwareObjective

        obj = RiskAwareObjective(
            posterior_samples=posterior_samples,
            x_maxes=np.array([10.0, 8.0, 5.0]),
            target_scale=1000.0,
            num_periods=8,
        )

        expected = np.ones(3)
        np.testing.assert_array_equal(obj.seasonal_indices, expected)


class TestResponseDistribution:
    """Tests for compute_response_distribution method."""

    def test_returns_array_of_responses(self, risk_objective):
        """Returns response for each posterior sample."""
        x = np.array([5.0, 4.0, 2.5])

        responses = risk_objective.compute_response_distribution(x)

        assert responses.shape == (100,)  # n_samples
        assert all(r > 0 for r in responses)

    def test_response_increases_with_spend(self, risk_objective):
        """Response increases with higher spend."""
        x_low = np.array([1.0, 1.0, 1.0])
        x_high = np.array([5.0, 4.0, 2.5])

        resp_low = risk_objective.compute_response_distribution(x_low)
        resp_high = risk_objective.compute_response_distribution(x_high)

        assert np.mean(resp_high) > np.mean(resp_low)

    def test_response_affected_by_seasonal_indices(self, posterior_samples):
        """Seasonal indices affect response."""
        from mmm_platform.optimization.risk_objectives import RiskAwareObjective

        x = np.array([5.0, 4.0, 2.5])

        # No seasonal adjustment
        obj_no_seasonal = RiskAwareObjective(
            posterior_samples=posterior_samples,
            x_maxes=np.array([10.0, 8.0, 5.0]),
            target_scale=1000.0,
            num_periods=8,
            seasonal_indices=np.array([1.0, 1.0, 1.0]),
        )

        # High seasonal adjustment
        obj_high_seasonal = RiskAwareObjective(
            posterior_samples=posterior_samples,
            x_maxes=np.array([10.0, 8.0, 5.0]),
            target_scale=1000.0,
            num_periods=8,
            seasonal_indices=np.array([1.5, 1.5, 1.5]),
        )

        resp_no = obj_no_seasonal.compute_response_distribution(x)
        resp_high = obj_high_seasonal.compute_response_distribution(x)

        # Higher seasonal indices should give higher response
        assert np.mean(resp_high) > np.mean(resp_no)


class TestObjectiveFunctions:
    """Tests for different objective functions."""

    def test_mean_objective_returns_float(self, risk_objective):
        """Mean objective returns negative float."""
        x = np.array([5.0, 4.0, 2.5])

        result = risk_objective.objective(x)

        assert isinstance(result, float)
        assert result < 0  # Negative for minimization

    def test_var_objective(self, posterior_samples):
        """VaR objective uses percentile."""
        from mmm_platform.optimization.risk_objectives import RiskAwareObjective

        obj = RiskAwareObjective(
            posterior_samples=posterior_samples,
            x_maxes=np.array([10.0, 8.0, 5.0]),
            target_scale=1000.0,
            num_periods=8,
            risk_profile="var",
            confidence_level=0.95,
        )

        x = np.array([5.0, 4.0, 2.5])
        result = obj.objective(x)

        assert isinstance(result, float)
        assert result < 0

    def test_cvar_objective(self, posterior_samples):
        """CVaR objective uses mean of tail."""
        from mmm_platform.optimization.risk_objectives import RiskAwareObjective

        obj = RiskAwareObjective(
            posterior_samples=posterior_samples,
            x_maxes=np.array([10.0, 8.0, 5.0]),
            target_scale=1000.0,
            num_periods=8,
            risk_profile="cvar",
            confidence_level=0.95,
        )

        x = np.array([5.0, 4.0, 2.5])
        result = obj.objective(x)

        assert isinstance(result, float)
        assert result < 0

    def test_sharpe_objective(self, posterior_samples):
        """Sharpe objective uses mean/std."""
        from mmm_platform.optimization.risk_objectives import RiskAwareObjective

        obj = RiskAwareObjective(
            posterior_samples=posterior_samples,
            x_maxes=np.array([10.0, 8.0, 5.0]),
            target_scale=1000.0,
            num_periods=8,
            risk_profile="sharpe",
        )

        x = np.array([5.0, 4.0, 2.5])
        result = obj.objective(x)

        assert isinstance(result, float)
        assert result < 0

    def test_invalid_risk_profile_raises(self, posterior_samples):
        """Invalid risk profile raises error."""
        from mmm_platform.optimization.risk_objectives import RiskAwareObjective

        obj = RiskAwareObjective(
            posterior_samples=posterior_samples,
            x_maxes=np.array([10.0, 8.0, 5.0]),
            target_scale=1000.0,
            num_periods=8,
            risk_profile="mean",  # Valid for init
        )

        # Change to invalid
        obj.risk_profile = "invalid"
        x = np.array([5.0, 4.0, 2.5])

        with pytest.raises(ValueError, match="Unknown risk profile"):
            obj.objective(x)

    def test_var_more_conservative_than_mean(self, posterior_samples):
        """VaR is more conservative (lower) than mean."""
        from mmm_platform.optimization.risk_objectives import RiskAwareObjective

        x = np.array([5.0, 4.0, 2.5])

        obj_mean = RiskAwareObjective(
            posterior_samples=posterior_samples,
            x_maxes=np.array([10.0, 8.0, 5.0]),
            target_scale=1000.0,
            num_periods=8,
            risk_profile="mean",
        )

        obj_var = RiskAwareObjective(
            posterior_samples=posterior_samples,
            x_maxes=np.array([10.0, 8.0, 5.0]),
            target_scale=1000.0,
            num_periods=8,
            risk_profile="var",
            confidence_level=0.95,
        )

        # Mean gives higher expected value, so more negative objective
        # VaR is more conservative, so less negative (higher) objective
        mean_obj = obj_mean.objective(x)
        var_obj = obj_var.objective(x)

        # VaR objective should be less negative (worse) since 5th percentile < mean
        assert var_obj > mean_obj


class TestGradients:
    """Tests for gradient computation."""

    def test_mean_gradient_shape(self, risk_objective):
        """Mean gradient has correct shape."""
        x = np.array([5.0, 4.0, 2.5])

        grad = risk_objective.gradient(x)

        assert grad.shape == (3,)

    def test_mean_gradient_negative(self, risk_objective):
        """Mean gradient is negative (increasing spend increases response)."""
        x = np.array([5.0, 4.0, 2.5])

        grad = risk_objective.gradient(x)

        # Gradient should be negative (objective decreases with more spend)
        assert all(g < 0 for g in grad)

    def test_numerical_gradient_for_var(self, posterior_samples):
        """VaR uses numerical gradient."""
        from mmm_platform.optimization.risk_objectives import RiskAwareObjective

        obj = RiskAwareObjective(
            posterior_samples=posterior_samples,
            x_maxes=np.array([10.0, 8.0, 5.0]),
            target_scale=1000.0,
            num_periods=8,
            risk_profile="var",
        )

        x = np.array([5.0, 4.0, 2.5])
        grad = obj.gradient(x)

        assert grad.shape == (3,)

    def test_gradient_approximately_correct(self, risk_objective):
        """Analytical gradient matches numerical gradient."""
        x = np.array([5.0, 4.0, 2.5])

        # Analytical gradient
        analytical = risk_objective._gradient_mean(x)

        # Numerical gradient
        numerical = risk_objective._gradient_numerical(x, eps=1e-5)

        # Should be close
        np.testing.assert_allclose(analytical, numerical, rtol=1e-3)


class TestAllRiskMetrics:
    """Tests for compute_all_risk_metrics method."""

    def test_returns_dict_with_all_metrics(self, risk_objective):
        """Returns dict with all risk metrics."""
        x = np.array([5.0, 4.0, 2.5])

        metrics = risk_objective.compute_all_risk_metrics(x)

        assert "expected_response" in metrics
        assert "response_std" in metrics
        assert "response_var" in metrics
        assert "response_cvar" in metrics
        assert "response_sharpe" in metrics
        assert "response_ci_low" in metrics
        assert "response_ci_high" in metrics

    def test_ci_contains_mean(self, risk_objective):
        """Confidence interval contains expected response."""
        x = np.array([5.0, 4.0, 2.5])

        metrics = risk_objective.compute_all_risk_metrics(x)

        assert metrics["response_ci_low"] < metrics["expected_response"]
        assert metrics["expected_response"] < metrics["response_ci_high"]

    def test_var_less_than_mean(self, risk_objective):
        """VaR (5th percentile) is less than mean."""
        x = np.array([5.0, 4.0, 2.5])

        metrics = risk_objective.compute_all_risk_metrics(x)

        assert metrics["response_var"] < metrics["expected_response"]

    def test_cvar_less_than_var(self, risk_objective):
        """CVaR (mean of tail) is less than or equal to VaR."""
        x = np.array([5.0, 4.0, 2.5])

        metrics = risk_objective.compute_all_risk_metrics(x)

        # CVaR is mean of responses below VaR, so should be <= VaR
        assert metrics["response_cvar"] <= metrics["response_var"]

    def test_sharpe_positive(self, risk_objective):
        """Sharpe ratio is positive for positive response."""
        x = np.array([5.0, 4.0, 2.5])

        metrics = risk_objective.compute_all_risk_metrics(x)

        assert metrics["response_sharpe"] > 0


class TestEdgeCases:
    """Tests for edge cases."""

    def test_zero_allocation_gives_low_response(self, risk_objective):
        """Zero allocation gives near-zero response."""
        x = np.array([0.0, 0.0, 0.0])

        responses = risk_objective.compute_response_distribution(x)

        # With zero spend, saturation should be 0
        assert all(r == 0 or r < 1e-6 for r in responses)

    def test_very_high_allocation(self, risk_objective):
        """Very high allocation saturates response."""
        x_moderate = np.array([5.0, 4.0, 2.5])
        x_high = np.array([100.0, 100.0, 100.0])

        resp_mod = risk_objective.compute_response_distribution(x_moderate)
        resp_high = risk_objective.compute_response_distribution(x_high)

        # Saturation means diminishing returns - high spend should give more
        # but not linearly more
        assert np.mean(resp_high) > np.mean(resp_mod)
        # Response ratio should be much less than spend ratio
        spend_ratio = 100 / 5
        response_ratio = np.mean(resp_high) / np.mean(resp_mod)
        assert response_ratio < spend_ratio

    def test_handles_small_x_max(self, posterior_samples):
        """Handles small x_max values safely."""
        from mmm_platform.optimization.risk_objectives import RiskAwareObjective

        # Very small x_max (should be protected from division by zero)
        obj = RiskAwareObjective(
            posterior_samples=posterior_samples,
            x_maxes=np.array([0.0, 0.0, 0.0]),  # Will be clamped to 1e-9
            target_scale=1000.0,
            num_periods=8,
        )

        x = np.array([1.0, 1.0, 1.0])

        # Should not raise
        responses = obj.compute_response_distribution(x)
        assert len(responses) == posterior_samples.n_samples
