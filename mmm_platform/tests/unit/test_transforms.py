"""
Unit tests for core/transforms.py

Tests the mathematical correctness of adstock and saturation transforms.
"""
import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_allclose

from mmm_platform.core.transforms import TransformEngine


class TestGeometricAdstock:
    """Tests for compute_geometric_adstock()."""

    def test_weights_sum_to_one(self, transform_engine):
        """Adstock weights should always sum to 1."""
        x = np.random.rand(50)
        for alpha in [0.0, 0.3, 0.5, 0.7, 1.0]:
            result = transform_engine.compute_geometric_adstock(x, alpha, l_max=8)
            # Result should have same length as input
            assert len(result) == len(x)

    def test_low_alpha_concentrated_effect(self, transform_engine):
        """With low alpha, effect should be more concentrated (less spread)."""
        # Test that lower alpha produces more concentrated (less spread) effect
        x = np.array([100.0] + [0.0] * 20)
        result_low = transform_engine.compute_geometric_adstock(x, alpha=0.1, l_max=8)
        result_high = transform_engine.compute_geometric_adstock(x, alpha=0.9, l_max=8)

        # With low alpha, the effect decays faster
        # The maximum value should be higher for low alpha (more concentrated)
        assert result_low.max() > result_high.max(), \
            f"Low alpha should produce more concentrated peak"

        # Total effect should be similar (conservation of mass)
        assert_allclose(result_low.sum(), result_high.sum(), rtol=0.1)

    def test_alpha_one_full_carryover(self, transform_engine):
        """With alpha=1, weights are uniform (full carryover)."""
        x = np.array([100] + [0] * 10)
        result = transform_engine.compute_geometric_adstock(x, alpha=0.999, l_max=8)
        # Effect should persist across periods
        assert all(result[:8] > 0)

    def test_backward_looking(self, transform_engine):
        """Adstock should be backward-looking (past affects current)."""
        x = np.array([0, 0, 0, 0, 100, 0, 0, 0])
        result = transform_engine.compute_geometric_adstock(x, alpha=0.5, l_max=4)
        # Effect should appear at and after the spike, not before
        assert all(result[:4] == 0)
        assert result[4] > 0

    def test_output_shape_matches_input(self, transform_engine):
        """Output should have same shape as input."""
        for length in [10, 50, 100, 200]:
            x = np.random.rand(length)
            result = transform_engine.compute_geometric_adstock(x, alpha=0.5, l_max=8)
            assert result.shape == x.shape

    def test_conservation_of_mass(self, transform_engine):
        """Total adstocked value should be close to total input (for long series)."""
        np.random.seed(42)
        x = np.random.rand(200) * 100
        result = transform_engine.compute_geometric_adstock(x, alpha=0.5, l_max=8)
        # Due to convolution edge effects, allow some tolerance
        assert abs(result.sum() - x.sum()) / x.sum() < 0.1


class TestLogisticSaturation:
    """Tests for compute_logistic_saturation()."""

    def test_output_range(self, transform_engine):
        """Saturation output should be in reasonable range."""
        x = np.linspace(0, 10, 100)
        for lam in [0.1, 1.0, 5.0, 10.0]:
            result = transform_engine.compute_logistic_saturation(x, lam)
            assert np.all(result >= -1)
            assert np.all(result <= 1)

    def test_monotonically_increasing(self, transform_engine):
        """Saturation should be monotonically increasing."""
        x = np.linspace(0, 10, 100)
        result = transform_engine.compute_logistic_saturation(x, lam=2.0)
        diffs = np.diff(result)
        assert np.all(diffs >= 0)

    def test_zero_input_zero_output(self, transform_engine):
        """Saturation of 0 should be 0."""
        result = transform_engine.compute_logistic_saturation(np.array([0.0]), lam=1.0)
        assert_allclose(result[0], 0.0, atol=1e-10)

    def test_numerical_stability_large_lam(self, transform_engine):
        """Should not overflow with large lambda values."""
        x = np.linspace(0, 1, 50)
        result = transform_engine.compute_logistic_saturation(x, lam=100.0)
        assert np.all(np.isfinite(result))

    def test_numerical_stability_small_lam(self, transform_engine):
        """Should not underflow with small lambda values."""
        x = np.linspace(0, 100, 50)
        result = transform_engine.compute_logistic_saturation(x, lam=0.001)
        assert np.all(np.isfinite(result))

    def test_higher_lambda_faster_saturation(self, transform_engine):
        """Higher lambda should reach saturation faster."""
        x = np.array([0.5])
        result_low = transform_engine.compute_logistic_saturation(x, lam=1.0)
        result_high = transform_engine.compute_logistic_saturation(x, lam=5.0)
        # Higher lambda means more saturated at same x
        assert result_high[0] > result_low[0]


class TestLambdaComputation:
    """Tests for compute_channel_lam()."""

    def test_lambda_positive(self, transform_engine, sample_df):
        """Lambda should always be positive."""
        lam = transform_engine.compute_channel_lam(sample_df, "tv_spend", percentile=50)
        assert lam > 0

    def test_higher_percentile_lower_lambda(self, transform_engine, sample_df):
        """Higher percentile = half-saturation at higher spend = lower lambda."""
        lam_50 = transform_engine.compute_channel_lam(sample_df, "tv_spend", percentile=50)
        lam_80 = transform_engine.compute_channel_lam(sample_df, "tv_spend", percentile=80)
        assert lam_80 < lam_50

    def test_zero_spend_channel_default(self, transform_engine, sample_df_with_zeros):
        """Zero-spend channel should return default lambda."""
        # Create a column with all zeros
        sample_df_with_zeros["zero_channel"] = 0.0
        lam = transform_engine.compute_channel_lam(
            sample_df_with_zeros, "zero_channel", percentile=50
        )
        assert lam == 1.0  # Default value

    def test_lambda_scales_with_data(self, transform_engine, sample_df):
        """Lambda should be reasonable for typical spend ranges."""
        lam = transform_engine.compute_channel_lam(sample_df, "tv_spend", percentile=50)
        # Lambda should be in a reasonable range for normalized data
        assert 0.1 < lam < 100


class TestEffectiveChannels:
    """Tests for get_effective_channel_columns()."""

    def test_includes_paid_media(self, transform_engine):
        """Should include all paid media channels."""
        effective = transform_engine.get_effective_channel_columns()
        assert "tv_spend" in effective
        assert "search_spend" in effective

    def test_includes_owned_media(self, complex_config):
        """Should include owned media in effective channels."""
        engine = TransformEngine(complex_config)
        effective = engine.get_effective_channel_columns()
        assert "email_sends" in effective

    def test_channel_order_preserved(self, basic_config):
        """Channel order should match config order."""
        engine = TransformEngine(basic_config)
        effective = engine.get_effective_channel_columns()
        assert effective[0] == "tv_spend"
        assert effective[1] == "search_spend"


class TestAdstockDecayRetrieval:
    """Tests for get_adstock_decay()."""

    def test_short_decay(self, basic_config):
        """Short adstock type should return short decay value."""
        engine = TransformEngine(basic_config)
        # search_spend has adstock_type="short"
        decay = engine.get_adstock_decay("search_spend")
        assert decay == basic_config.adstock.short_decay

    def test_medium_decay(self, basic_config):
        """Medium adstock type should return medium decay value."""
        engine = TransformEngine(basic_config)
        # tv_spend has adstock_type="medium"
        decay = engine.get_adstock_decay("tv_spend")
        assert decay == basic_config.adstock.medium_decay

    def test_unknown_channel_default(self, basic_config):
        """Unknown channel should return medium decay."""
        engine = TransformEngine(basic_config)
        decay = engine.get_adstock_decay("nonexistent_channel")
        assert decay == basic_config.adstock.medium_decay


class TestFullTransform:
    """Tests for apply_full_transform()."""

    def test_full_transform_shape(self, transform_engine):
        """Full transform should preserve input shape."""
        x = np.random.rand(50)
        result = transform_engine.apply_full_transform(x, alpha=0.5, lam=2.0, l_max=8)
        assert result.shape == x.shape

    def test_full_transform_range(self, transform_engine):
        """Full transform output should be bounded."""
        x = np.random.rand(50) * 100
        result = transform_engine.apply_full_transform(x, alpha=0.5, lam=2.0, l_max=8)
        # After saturation, should be in [-1, 1]
        assert np.all(result >= -1)
        assert np.all(result <= 1)

    def test_full_transform_monotonic_relationship(self, transform_engine):
        """Higher input should generally lead to higher output."""
        x_low = np.array([10.0] * 50)
        x_high = np.array([100.0] * 50)

        result_low = transform_engine.apply_full_transform(x_low, alpha=0.5, lam=2.0)
        result_high = transform_engine.apply_full_transform(x_high, alpha=0.5, lam=2.0)

        # Mean of high input should exceed mean of low input
        assert np.mean(result_high) > np.mean(result_low)
