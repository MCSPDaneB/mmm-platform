"""
Global pytest fixtures for MMM Platform tests.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from mmm_platform.config.schema import (
    ModelConfig, DataConfig, AdstockConfig, SaturationConfig,
    SamplingConfig, ChannelConfig, ControlConfig, OwnedMediaConfig
)
from mmm_platform.core.transforms import TransformEngine
from mmm_platform.core.priors import PriorCalibrator


# =============================================================================
# Sample Configurations
# =============================================================================

@pytest.fixture
def basic_config() -> ModelConfig:
    """Minimal 2-channel, 1-control configuration."""
    return ModelConfig(
        name="test_model",
        data=DataConfig(
            date_column="date",
            target_column="revenue",
            spend_scale=1.0,  # PyMC-Marketing handles scaling internally
            target_scale=1.0,  # PyMC-Marketing handles scaling internally
            dayfirst=False,  # Test data uses YYYY-MM-DD format
        ),
        adstock=AdstockConfig(
            l_max=8,
            short_decay=0.3,
            medium_decay=0.5,
            long_decay=0.7,
        ),
        saturation=SaturationConfig(
            curve_sharpness=50,
        ),
        sampling=SamplingConfig(
            draws=100,
            tune=100,
            chains=2,
        ),
        channels=[
            ChannelConfig(
                name="tv_spend",
                display_name="TV",
                roi_prior_low=0.5,
                roi_prior_mid=1.5,
                roi_prior_high=3.0,
                adstock_type="medium",
            ),
            ChannelConfig(
                name="search_spend",
                display_name="Search",
                roi_prior_low=1.0,
                roi_prior_mid=2.5,
                roi_prior_high=5.0,
                adstock_type="short",
            ),
        ],
        controls=[
            ControlConfig(
                name="promo_flag",
                display_name="Promotions",
            ),
        ],
    )


@pytest.fixture
def complex_config() -> ModelConfig:
    """Complex config with owned media, competitors, multiple controls."""
    return ModelConfig(
        name="complex_test_model",
        data=DataConfig(
            date_column="date",
            target_column="revenue",
            spend_scale=1.0,  # PyMC-Marketing handles scaling internally
            target_scale=1.0,  # PyMC-Marketing handles scaling internally
            dayfirst=False,  # Test data uses YYYY-MM-DD format
        ),
        adstock=AdstockConfig(l_max=12),
        saturation=SaturationConfig(curve_sharpness=60),
        sampling=SamplingConfig(draws=100, tune=100, chains=2),
        channels=[
            ChannelConfig(name=f"channel_{i}_spend", roi_prior_mid=1.5 + i*0.5)
            for i in range(5)
        ],
        controls=[
            ControlConfig(name="holiday"),
            ControlConfig(name="competitor_price"),
        ],
        owned_media=[
            OwnedMediaConfig(
                name="email_sends",
                display_name="Email",
                include_roi=True,
                roi_prior_low=0.1,
                roi_prior_mid=0.5,
                roi_prior_high=1.0,
            ),
        ],
    )


# =============================================================================
# Sample DataFrames
# =============================================================================

@pytest.fixture
def sample_df() -> pd.DataFrame:
    """100-row sample dataset with 2 channels."""
    np.random.seed(42)
    n_rows = 100

    dates = pd.date_range(start="2022-01-01", periods=n_rows, freq="W")

    return pd.DataFrame({
        "date": dates,
        "revenue": np.random.uniform(50000, 150000, n_rows),
        "tv_spend": np.random.uniform(5000, 20000, n_rows),
        "search_spend": np.random.uniform(3000, 15000, n_rows),
        "promo_flag": np.random.choice([0, 1], n_rows, p=[0.7, 0.3]),
    })


@pytest.fixture
def sample_df_with_zeros() -> pd.DataFrame:
    """Dataset with some zero-spend periods (edge case)."""
    np.random.seed(42)
    n_rows = 100

    dates = pd.date_range(start="2022-01-01", periods=n_rows, freq="W")
    tv_spend = np.random.uniform(5000, 20000, n_rows)
    tv_spend[:10] = 0  # First 10 weeks have zero TV spend

    return pd.DataFrame({
        "date": dates,
        "revenue": np.random.uniform(50000, 150000, n_rows),
        "tv_spend": tv_spend,
        "search_spend": np.random.uniform(3000, 15000, n_rows),
        "promo_flag": np.random.choice([0, 1], n_rows),
    })


# =============================================================================
# Transform & Prior Fixtures
# =============================================================================

@pytest.fixture
def transform_engine(basic_config) -> TransformEngine:
    """Initialized TransformEngine."""
    return TransformEngine(basic_config)


@pytest.fixture
def prior_calibrator(basic_config) -> PriorCalibrator:
    """Initialized PriorCalibrator."""
    return PriorCalibrator(basic_config)


# =============================================================================
# Mock Inference Data (for analysis tests)
# =============================================================================

@pytest.fixture
def mock_posterior_samples():
    """Mock posterior samples for 2-channel model."""
    np.random.seed(42)
    n_chains, n_draws = 2, 100
    n_channels = 2

    return {
        "saturation_beta": np.random.lognormal(0, 0.5, (n_chains, n_draws, n_channels)),
        "saturation_lam": np.random.lognormal(1, 0.3, (n_chains, n_draws, n_channels)),
        "adstock_alpha": np.random.beta(2, 3, (n_chains, n_draws, n_channels)),
    }


# =============================================================================
# Contribution Analysis Fixtures
# =============================================================================

@pytest.fixture
def sample_contributions(sample_df) -> pd.DataFrame:
    """Sample contribution DataFrame for analysis tests."""
    np.random.seed(42)
    n_rows = len(sample_df)

    return pd.DataFrame({
        "intercept": np.random.uniform(30000, 50000, n_rows),
        "tv_spend": np.random.uniform(5000, 15000, n_rows),
        "search_spend": np.random.uniform(3000, 10000, n_rows),
        "promo_flag": np.random.uniform(1000, 5000, n_rows),
        "revenue": np.random.uniform(50000, 100000, n_rows),
    })


@pytest.fixture
def contribution_analyzer(sample_contributions, sample_df):
    """Initialized ContributionAnalyzer for testing."""
    from mmm_platform.analysis.contributions import ContributionAnalyzer

    return ContributionAnalyzer(
        contribs=sample_contributions,
        df_scaled=sample_df,
        channel_cols=["tv_spend", "search_spend"],
        control_cols=["promo_flag"],
        target_col="revenue",
        date_col="date",
        revenue_scale=1000.0,
        spend_scale=1000.0,
        display_names={
            "tv_spend": "TV Advertising",
            "search_spend": "Paid Search",
            "promo_flag": "Promotions",
        }
    )


# =============================================================================
# Validation Fixtures
# =============================================================================

@pytest.fixture
def sample_validation_df() -> pd.DataFrame:
    """DataFrame with various validation issues for testing."""
    np.random.seed(42)
    n_rows = 50

    dates = pd.date_range(start="2022-01-01", periods=n_rows, freq="W")

    return pd.DataFrame({
        "date": dates,
        "revenue": np.concatenate([
            np.random.uniform(50000, 150000, n_rows - 5),
            [np.nan, -1000, 0, 0, 1000000],  # Issues: NaN, negative, zeros, outlier
        ]),
        "tv_spend": np.concatenate([
            np.random.uniform(5000, 20000, n_rows - 10),
            np.zeros(10),  # Low activity
        ]),
        "search_spend": np.random.uniform(3000, 15000, n_rows),
        "promo_flag": np.concatenate([
            np.random.choice([0, 1], n_rows - 5),
            [0, 1, 2, 3, 4],  # Non-binary values for dummy
        ]),
    })
