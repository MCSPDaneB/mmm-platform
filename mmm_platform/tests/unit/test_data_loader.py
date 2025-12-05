"""Unit tests for data_loader.py date filtering."""
import pytest
import pandas as pd
import numpy as np

from mmm_platform.core.data_loader import DataLoader
from mmm_platform.config.schema import (
    ModelConfig, DataConfig, ChannelConfig
)


@pytest.fixture
def sample_df_with_dates():
    """Create sample dataframe with 100 weeks of data."""
    np.random.seed(42)
    n_rows = 100
    dates = pd.date_range(start="2022-01-01", periods=n_rows, freq="W")
    return pd.DataFrame({
        "date": dates,
        "revenue": np.random.uniform(50000, 150000, n_rows),
        "tv_spend": np.random.uniform(5000, 20000, n_rows),
    })


@pytest.fixture
def basic_config_with_dates():
    """Config with date filtering."""
    return ModelConfig(
        name="test_model",
        data=DataConfig(
            date_column="date",
            target_column="revenue",
            model_start_date="2022-06-01",
            model_end_date="2022-12-31",
        ),
        channels=[
            ChannelConfig(name="tv_spend", roi_prior_mid=1.5),
        ],
    )


class TestDateFiltering:
    """Tests for date filtering in prepare_model_data."""

    def test_date_filtering_reduces_rows(self, sample_df_with_dates, basic_config_with_dates):
        """Verify date filtering reduces dataset size."""
        loader = DataLoader(basic_config_with_dates)
        df, df_scaled, y, controls = loader.prepare_model_data(
            sample_df_with_dates, basic_config_with_dates
        )

        # Original has 100 rows, filtered should have fewer
        assert len(df) < 100
        assert len(df_scaled) < 100

        # Check date range
        min_date = df["date"].min()
        max_date = df["date"].max()
        assert min_date >= pd.Timestamp("2022-06-01")
        assert max_date <= pd.Timestamp("2022-12-31")

    def test_no_filtering_when_dates_none(self, sample_df_with_dates):
        """Verify full data when no dates specified."""
        config = ModelConfig(
            name="test_model",
            data=DataConfig(
                date_column="date",
                target_column="revenue",
                model_start_date=None,
                model_end_date=None,
            ),
            channels=[
                ChannelConfig(name="tv_spend", roi_prior_mid=1.5),
            ],
        )
        loader = DataLoader(config)
        df, df_scaled, y, controls = loader.prepare_model_data(
            sample_df_with_dates, config
        )

        # Should keep all 100 rows
        assert len(df) == 100
        assert len(df_scaled) == 100

    def test_start_date_only_filtering(self, sample_df_with_dates):
        """Test filtering with only start date."""
        config = ModelConfig(
            name="test_model",
            data=DataConfig(
                date_column="date",
                target_column="revenue",
                model_start_date="2022-07-01",
                model_end_date=None,
            ),
            channels=[
                ChannelConfig(name="tv_spend", roi_prior_mid=1.5),
            ],
        )
        loader = DataLoader(config)
        df, df_scaled, y, controls = loader.prepare_model_data(
            sample_df_with_dates, config
        )

        assert len(df) < 100
        assert df["date"].min() >= pd.Timestamp("2022-07-01")

    def test_end_date_only_filtering(self, sample_df_with_dates):
        """Test filtering with only end date."""
        config = ModelConfig(
            name="test_model",
            data=DataConfig(
                date_column="date",
                target_column="revenue",
                model_start_date=None,
                model_end_date="2022-06-30",
            ),
            channels=[
                ChannelConfig(name="tv_spend", roi_prior_mid=1.5),
            ],
        )
        loader = DataLoader(config)
        df, df_scaled, y, controls = loader.prepare_model_data(
            sample_df_with_dates, config
        )

        assert len(df) < 100
        assert df["date"].max() <= pd.Timestamp("2022-06-30")

    def test_time_index_reset_after_filtering(self, sample_df_with_dates, basic_config_with_dates):
        """Verify time index 't' is reset after filtering."""
        loader = DataLoader(basic_config_with_dates)
        df, df_scaled, y, controls = loader.prepare_model_data(
            sample_df_with_dates, basic_config_with_dates
        )

        # Time index should start at 1 and be consecutive
        if "t" in df.columns:
            assert df["t"].iloc[0] == 1
            assert df["t"].iloc[-1] == len(df)
            assert (df["t"].diff().dropna() == 1).all()
