"""
Integration tests for the full model pipeline.

Tests Load → Prepare → Build workflows (without actual MCMC fitting).
"""
import pytest
import pandas as pd
import numpy as np

from mmm_platform.model.mmm import MMMWrapper
from mmm_platform.config.schema import ModelConfig


class TestModelPipeline:
    """Tests for MMMWrapper pipeline."""

    def test_wrapper_initialization(self, basic_config):
        """Should initialize wrapper with config."""
        wrapper = MMMWrapper(basic_config)
        assert wrapper.config == basic_config
        assert wrapper.idata is None
        assert wrapper.df_raw is None

    def test_load_data_from_dataframe(self, basic_config, sample_df):
        """Should load data directly from DataFrame."""
        wrapper = MMMWrapper(basic_config)
        wrapper.df_raw = sample_df

        assert wrapper.df_raw is not None
        assert len(wrapper.df_raw) == 100

    def test_load_data_validates(self, basic_config, sample_df, tmp_path):
        """load_data should validate against config."""
        # Save sample data to temp file
        csv_path = tmp_path / "test_data.csv"
        sample_df.to_csv(csv_path, index=False)

        wrapper = MMMWrapper(basic_config)
        # Pass date_format to match ISO8601 dates from pd.date_range
        result = wrapper.load_data(str(csv_path), date_format="%Y-%m-%d")

        assert result.valid

    def test_load_data_detects_missing_columns(self, basic_config, sample_df, tmp_path):
        """load_data should detect missing required columns."""
        # Remove a required column
        df_missing = sample_df.drop(columns=["tv_spend"])
        csv_path = tmp_path / "test_data_missing.csv"
        df_missing.to_csv(csv_path, index=False)

        wrapper = MMMWrapper(basic_config)
        # Pass date_format to match ISO8601 dates from pd.date_range
        result = wrapper.load_data(str(csv_path), date_format="%Y-%m-%d")

        assert not result.valid

    def test_prepare_data_creates_arrays(self, basic_config, sample_df, tmp_path):
        """prepare_data should create lam_vec and beta priors."""
        csv_path = tmp_path / "test_data.csv"
        sample_df.to_csv(csv_path, index=False)

        wrapper = MMMWrapper(basic_config)
        wrapper.load_data(str(csv_path), date_format="%Y-%m-%d")
        wrapper.prepare_data()

        assert wrapper.lam_vec is not None
        assert wrapper.beta_mu is not None
        assert wrapper.beta_sigma is not None
        assert len(wrapper.lam_vec) == 2  # 2 channels

    def test_prepare_data_lam_positive(self, basic_config, sample_df, tmp_path):
        """Lambda values should be positive."""
        csv_path = tmp_path / "test_data.csv"
        sample_df.to_csv(csv_path, index=False)

        wrapper = MMMWrapper(basic_config)
        wrapper.load_data(str(csv_path), date_format="%Y-%m-%d")
        wrapper.prepare_data()

        assert np.all(wrapper.lam_vec > 0)

    def test_prepare_data_beta_positive(self, basic_config, sample_df, tmp_path):
        """Beta mu (log-scale) should be finite, sigma should be positive."""
        csv_path = tmp_path / "test_data.csv"
        sample_df.to_csv(csv_path, index=False)

        wrapper = MMMWrapper(basic_config)
        wrapper.load_data(str(csv_path), date_format="%Y-%m-%d")
        wrapper.prepare_data()

        # beta_mu is log(beta_mid), so it can be negative for small beta values
        # What matters is that exp(mu) (the median of the LogNormal) is positive
        assert np.all(np.isfinite(wrapper.beta_mu))
        assert np.all(np.exp(wrapper.beta_mu) > 0)
        assert np.all(wrapper.beta_sigma > 0)

    def test_build_model_creates_mmm(self, basic_config, sample_df, tmp_path):
        """build_model should create PyMC MMM object."""
        csv_path = tmp_path / "test_data.csv"
        sample_df.to_csv(csv_path, index=False)

        wrapper = MMMWrapper(basic_config)
        wrapper.load_data(str(csv_path), date_format="%Y-%m-%d")
        wrapper.prepare_data()
        mmm = wrapper.build_model()

        assert mmm is not None
        assert hasattr(mmm, 'fit')

    def test_df_scaled_created(self, basic_config, sample_df, tmp_path):
        """Scaled dataframe should be created during preparation."""
        csv_path = tmp_path / "test_data.csv"
        sample_df.to_csv(csv_path, index=False)

        wrapper = MMMWrapper(basic_config)
        wrapper.load_data(str(csv_path), date_format="%Y-%m-%d")
        wrapper.prepare_data()

        assert wrapper.df_scaled is not None
        assert len(wrapper.df_scaled) == len(sample_df)


class TestTransformEngineIntegration:
    """Tests for TransformEngine integration with MMMWrapper."""

    def test_transform_engine_created(self, basic_config, sample_df, tmp_path):
        """TransformEngine should be created during preparation."""
        csv_path = tmp_path / "test_data.csv"
        sample_df.to_csv(csv_path, index=False)

        wrapper = MMMWrapper(basic_config)
        wrapper.load_data(str(csv_path), date_format="%Y-%m-%d")
        wrapper.prepare_data()

        assert wrapper.transform_engine is not None

    def test_effective_channels_match(self, basic_config, sample_df, tmp_path):
        """Effective channels should match config channels."""
        csv_path = tmp_path / "test_data.csv"
        sample_df.to_csv(csv_path, index=False)

        wrapper = MMMWrapper(basic_config)
        wrapper.load_data(str(csv_path), date_format="%Y-%m-%d")
        wrapper.prepare_data()

        effective = wrapper.transform_engine.get_effective_channel_columns()
        config_channels = basic_config.get_channel_columns()

        # All config channels should be in effective channels
        for ch in config_channels:
            assert ch in effective


class TestPriorCalibratorIntegration:
    """Tests for PriorCalibrator integration with MMMWrapper."""

    def test_priors_calibrated(self, basic_config, sample_df, tmp_path):
        """Priors should be calibrated based on ROI beliefs."""
        csv_path = tmp_path / "test_data.csv"
        sample_df.to_csv(csv_path, index=False)

        wrapper = MMMWrapper(basic_config)
        wrapper.load_data(str(csv_path), date_format="%Y-%m-%d")
        wrapper.prepare_data()

        # Check that different ROI priors give different beta values
        # TV has ROI 1.5, Search has ROI 2.5
        # Search should have higher beta
        assert wrapper.beta_mu[1] > wrapper.beta_mu[0]


class TestDataValidation:
    """Tests for data validation during loading."""

    def test_date_column_parsed(self, basic_config, sample_df, tmp_path):
        """Date column should be parsed as datetime."""
        csv_path = tmp_path / "test_data.csv"
        sample_df.to_csv(csv_path, index=False)

        wrapper = MMMWrapper(basic_config)
        wrapper.load_data(str(csv_path), date_format="%Y-%m-%d")

        assert pd.api.types.is_datetime64_any_dtype(wrapper.df_raw["date"])

    def test_numeric_columns_numeric(self, basic_config, sample_df, tmp_path):
        """Numeric columns should be numeric type."""
        csv_path = tmp_path / "test_data.csv"
        sample_df.to_csv(csv_path, index=False)

        wrapper = MMMWrapper(basic_config)
        wrapper.load_data(str(csv_path), date_format="%Y-%m-%d")

        assert pd.api.types.is_numeric_dtype(wrapper.df_raw["revenue"])
        assert pd.api.types.is_numeric_dtype(wrapper.df_raw["tv_spend"])
        assert pd.api.types.is_numeric_dtype(wrapper.df_raw["search_spend"])


class TestComplexConfig:
    """Tests with complex configuration."""

    def test_owned_media_handling(self, complex_config, tmp_path):
        """Should handle owned media in config."""
        np.random.seed(42)
        n_rows = 100
        dates = pd.date_range(start="2022-01-01", periods=n_rows, freq="W")

        # Create data matching complex config
        df = pd.DataFrame({
            "date": dates,
            "revenue": np.random.uniform(50000, 150000, n_rows),
            "channel_0_spend": np.random.uniform(5000, 20000, n_rows),
            "channel_1_spend": np.random.uniform(5000, 20000, n_rows),
            "channel_2_spend": np.random.uniform(5000, 20000, n_rows),
            "channel_3_spend": np.random.uniform(5000, 20000, n_rows),
            "channel_4_spend": np.random.uniform(5000, 20000, n_rows),
            "email_sends": np.random.uniform(1000, 10000, n_rows),
            "holiday": np.random.choice([0, 1], n_rows, p=[0.9, 0.1]),
            "competitor_price": np.random.uniform(80, 120, n_rows),
        })

        csv_path = tmp_path / "complex_data.csv"
        df.to_csv(csv_path, index=False)

        wrapper = MMMWrapper(complex_config)
        result = wrapper.load_data(str(csv_path), date_format="%Y-%m-%d")

        assert result.valid

    def test_multiple_controls_handling(self, complex_config, tmp_path):
        """Should handle multiple control variables."""
        np.random.seed(42)
        n_rows = 100
        dates = pd.date_range(start="2022-01-01", periods=n_rows, freq="W")

        df = pd.DataFrame({
            "date": dates,
            "revenue": np.random.uniform(50000, 150000, n_rows),
            "channel_0_spend": np.random.uniform(5000, 20000, n_rows),
            "channel_1_spend": np.random.uniform(5000, 20000, n_rows),
            "channel_2_spend": np.random.uniform(5000, 20000, n_rows),
            "channel_3_spend": np.random.uniform(5000, 20000, n_rows),
            "channel_4_spend": np.random.uniform(5000, 20000, n_rows),
            "email_sends": np.random.uniform(1000, 10000, n_rows),
            "holiday": np.random.choice([0, 1], n_rows, p=[0.9, 0.1]),
            "competitor_price": np.random.uniform(80, 120, n_rows),
        })

        csv_path = tmp_path / "complex_data.csv"
        df.to_csv(csv_path, index=False)

        wrapper = MMMWrapper(complex_config)
        wrapper.load_data(str(csv_path), date_format="%Y-%m-%d")
        wrapper.prepare_data()

        # Should have 5 paid channels + 1 owned media = 6 effective channels
        effective = wrapper.transform_engine.get_effective_channel_columns()
        assert len(effective) == 6
