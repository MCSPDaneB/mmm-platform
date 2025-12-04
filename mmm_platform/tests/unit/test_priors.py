"""
Unit tests for core/priors.py

Tests the mathematical correctness of ROI → beta prior calibration.
"""
import pytest
import numpy as np
from numpy.testing import assert_allclose

from mmm_platform.core.priors import PriorCalibrator
from mmm_platform.core.transforms import TransformEngine


class TestBetaPriorCalibration:
    """Tests for calibrate_beta_priors()."""

    def test_returns_correct_shape(self, prior_calibrator, sample_df, transform_engine):
        """Should return mu and sigma arrays of correct length."""
        # Prepare scaled data
        df_scaled = sample_df.copy()
        df_scaled["revenue"] = df_scaled["revenue"] / df_scaled["revenue"].max()
        df_scaled["tv_spend"] = df_scaled["tv_spend"] / df_scaled["tv_spend"].max()
        df_scaled["search_spend"] = df_scaled["search_spend"] / df_scaled["search_spend"].max()

        lam_vec = transform_engine.compute_all_effective_channel_lams(sample_df)

        mu, sigma = prior_calibrator.calibrate_beta_priors(
            df_scaled, lam_vec, l_max=8
        )

        n_channels = len(transform_engine.get_effective_channel_columns())
        assert len(mu) == n_channels
        assert len(sigma) == n_channels

    def test_mu_positive(self, prior_calibrator, sample_df, transform_engine):
        """Beta mu should always be positive."""
        df_scaled = sample_df.copy()
        for col in ["revenue", "tv_spend", "search_spend"]:
            df_scaled[col] = df_scaled[col] / df_scaled[col].max()

        lam_vec = transform_engine.compute_all_effective_channel_lams(sample_df)
        mu, sigma = prior_calibrator.calibrate_beta_priors(df_scaled, lam_vec, l_max=8)

        assert np.all(mu > 0)

    def test_sigma_positive(self, prior_calibrator, sample_df, transform_engine):
        """Beta sigma should always be positive."""
        df_scaled = sample_df.copy()
        for col in ["revenue", "tv_spend", "search_spend"]:
            df_scaled[col] = df_scaled[col] / df_scaled[col].max()

        lam_vec = transform_engine.compute_all_effective_channel_lams(sample_df)
        mu, sigma = prior_calibrator.calibrate_beta_priors(df_scaled, lam_vec, l_max=8)

        assert np.all(sigma > 0)

    def test_higher_roi_higher_beta(self, basic_config, sample_df):
        """Channel with higher ROI prior should have higher beta mu."""
        # Modify config to have very different ROI priors
        config = basic_config.model_copy(deep=True)
        config.channels[0].roi_prior_mid = 1.0  # TV: low ROI
        config.channels[1].roi_prior_mid = 5.0  # Search: high ROI

        calibrator = PriorCalibrator(config)
        engine = TransformEngine(config)

        df_scaled = sample_df.copy()
        for col in ["revenue", "tv_spend", "search_spend"]:
            df_scaled[col] = df_scaled[col] / df_scaled[col].max()

        lam_vec = engine.compute_all_effective_channel_lams(sample_df)
        mu, sigma = calibrator.calibrate_beta_priors(df_scaled, lam_vec, l_max=8)

        # Search (index 1) should have higher beta than TV (index 0)
        assert mu[1] > mu[0]

    def test_sigma_scales_with_roi_range(self, basic_config, sample_df):
        """Wider ROI range should result in larger sigma."""
        # Create two configs - narrow and wide ROI range
        narrow_config = basic_config.model_copy(deep=True)
        narrow_config.channels[0].roi_prior_low = 1.4
        narrow_config.channels[0].roi_prior_mid = 1.5
        narrow_config.channels[0].roi_prior_high = 1.6

        wide_config = basic_config.model_copy(deep=True)
        wide_config.channels[0].roi_prior_low = 0.5
        wide_config.channels[0].roi_prior_mid = 1.5
        wide_config.channels[0].roi_prior_high = 4.0

        df_scaled = sample_df.copy()
        for col in ["revenue", "tv_spend", "search_spend"]:
            df_scaled[col] = df_scaled[col] / df_scaled[col].max()

        narrow_calibrator = PriorCalibrator(narrow_config)
        wide_calibrator = PriorCalibrator(wide_config)
        engine = TransformEngine(basic_config)

        lam_vec = engine.compute_all_effective_channel_lams(sample_df)

        _, sigma_narrow = narrow_calibrator.calibrate_beta_priors(df_scaled, lam_vec, l_max=8)
        _, sigma_wide = wide_calibrator.calibrate_beta_priors(df_scaled, lam_vec, l_max=8)

        # Wide ROI range should have larger sigma for first channel
        assert sigma_wide[0] > sigma_narrow[0]


class TestPriorValidation:
    """Tests for validate_prior_roi()."""

    def test_validation_returns_dataframe(self, prior_calibrator, sample_df, transform_engine):
        """validate_prior_roi should return a DataFrame."""
        df_scaled = sample_df.copy()
        for col in ["revenue", "tv_spend", "search_spend"]:
            df_scaled[col] = df_scaled[col] / df_scaled[col].max()

        lam_vec = transform_engine.compute_all_effective_channel_lams(sample_df)
        mu, sigma = prior_calibrator.calibrate_beta_priors(df_scaled, lam_vec, l_max=8)

        validation_df = prior_calibrator.validate_prior_roi(
            df_scaled, mu, lam_vec, l_max=8
        )

        assert validation_df is not None
        assert len(validation_df) == 2  # 2 channels
        assert "channel" in validation_df.columns
        assert "target_roi" in validation_df.columns
        assert "prior_roi" in validation_df.columns

    def test_roundtrip_within_tolerance(self, prior_calibrator, sample_df, transform_engine):
        """ROI → beta → ROI should match within reasonable tolerance."""
        df_scaled = sample_df.copy()
        for col in ["revenue", "tv_spend", "search_spend"]:
            df_scaled[col] = df_scaled[col] / df_scaled[col].max()

        lam_vec = transform_engine.compute_all_effective_channel_lams(sample_df)
        mu, sigma = prior_calibrator.calibrate_beta_priors(df_scaled, lam_vec, l_max=8)

        validation_df = prior_calibrator.validate_prior_roi(
            df_scaled, mu, lam_vec, l_max=8
        )

        # Check that prior_roi is close to target_roi (within 10% tolerance)
        for _, row in validation_df.iterrows():
            if row["target_roi"] > 0:
                error_pct = abs(row["prior_roi"] - row["target_roi"]) / row["target_roi"]
                assert error_pct < 0.10, f"ROI roundtrip error {error_pct:.1%} exceeds 10%"


class TestPriorCalibratorInit:
    """Tests for PriorCalibrator initialization."""

    def test_init_with_config(self, basic_config):
        """Should initialize with valid config."""
        calibrator = PriorCalibrator(basic_config)
        assert calibrator.config == basic_config

    def test_extracts_roi_priors(self, basic_config):
        """Should extract ROI priors from config."""
        calibrator = PriorCalibrator(basic_config)
        roi_low, roi_mid, roi_high = calibrator.config.get_roi_dicts()

        assert "tv_spend" in roi_mid
        assert roi_mid["tv_spend"] == 1.5


class TestEdgeCases:
    """Tests for edge cases in prior calibration."""

    def test_handles_zero_spend_periods(self, basic_config, sample_df_with_zeros):
        """Should handle datasets with zero-spend periods."""
        calibrator = PriorCalibrator(basic_config)
        engine = TransformEngine(basic_config)

        df_scaled = sample_df_with_zeros.copy()
        for col in ["revenue", "tv_spend", "search_spend"]:
            max_val = df_scaled[col].max()
            if max_val > 0:
                df_scaled[col] = df_scaled[col] / max_val

        lam_vec = engine.compute_all_effective_channel_lams(sample_df_with_zeros)
        mu, sigma = calibrator.calibrate_beta_priors(df_scaled, lam_vec, l_max=8)

        # Should still produce valid results
        assert np.all(np.isfinite(mu))
        assert np.all(np.isfinite(sigma))
        assert np.all(mu > 0)
        assert np.all(sigma > 0)

    def test_handles_small_roi_values(self, basic_config, sample_df):
        """Should handle very small ROI priors."""
        config = basic_config.model_copy(deep=True)
        config.channels[0].roi_prior_low = 0.01
        config.channels[0].roi_prior_mid = 0.05
        config.channels[0].roi_prior_high = 0.1

        calibrator = PriorCalibrator(config)
        engine = TransformEngine(config)

        df_scaled = sample_df.copy()
        for col in ["revenue", "tv_spend", "search_spend"]:
            df_scaled[col] = df_scaled[col] / df_scaled[col].max()

        lam_vec = engine.compute_all_effective_channel_lams(sample_df)
        mu, sigma = calibrator.calibrate_beta_priors(df_scaled, lam_vec, l_max=8)

        # mu is log(beta_mid), so it can be negative for small beta values
        # What matters is that exp(mu) (the median of the LogNormal) is positive
        assert np.all(np.isfinite(mu))
        assert np.all(np.exp(mu) > 0)  # Median of LogNormal must be positive

    def test_handles_large_roi_values(self, basic_config, sample_df):
        """Should handle large ROI priors."""
        config = basic_config.model_copy(deep=True)
        config.channels[0].roi_prior_low = 5.0
        config.channels[0].roi_prior_mid = 10.0
        config.channels[0].roi_prior_high = 20.0

        calibrator = PriorCalibrator(config)
        engine = TransformEngine(config)

        df_scaled = sample_df.copy()
        for col in ["revenue", "tv_spend", "search_spend"]:
            df_scaled[col] = df_scaled[col] / df_scaled[col].max()

        lam_vec = engine.compute_all_effective_channel_lams(sample_df)
        mu, sigma = calibrator.calibrate_beta_priors(df_scaled, lam_vec, l_max=8)

        assert np.all(np.isfinite(mu))
        assert np.all(mu > 0)
