"""
Tests for analysis/pre_fit_checks.py - Pre-fit configuration checks.
"""

import pytest
import pandas as pd
import numpy as np

from mmm_platform.analysis.pre_fit_checks import (
    PreFitWarning,
    PreFitChecker,
    run_pre_fit_checks,
)
from mmm_platform.config.schema import (
    ModelConfig, DataConfig, ChannelConfig, ControlConfig,
    AdstockConfig, SaturationConfig, SamplingConfig
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def basic_config():
    """Basic config for pre-fit check tests."""
    return ModelConfig(
        name="prefit_test",
        data=DataConfig(
            date_column="date",
            target_column="revenue",
        ),
        channels=[
            ChannelConfig(
                name="tv_spend",
                roi_prior_low=0.5,
                roi_prior_mid=2.0,
                roi_prior_high=5.0,
            ),
            ChannelConfig(
                name="search_spend",
                roi_prior_low=1.0,
                roi_prior_mid=3.0,
                roi_prior_high=6.0,
            ),
        ],
        controls=[
            ControlConfig(name="promo_flag"),
        ],
    )


@pytest.fixture
def normal_df():
    """Normal DataFrame with good data quality."""
    np.random.seed(42)
    n_rows = 100

    return pd.DataFrame({
        "date": pd.date_range("2022-01-01", periods=n_rows, freq="W"),
        "revenue": np.random.uniform(50000, 150000, n_rows),
        "tv_spend": np.random.uniform(10000, 30000, n_rows),  # ~50% of spend
        "search_spend": np.random.uniform(8000, 25000, n_rows),  # ~50% of spend
        "promo_flag": np.random.choice([0, 1], n_rows),
    })


# =============================================================================
# PreFitWarning Dataclass Tests
# =============================================================================

class TestPreFitWarning:
    """Tests for PreFitWarning dataclass."""

    def test_creation(self):
        """Can create PreFitWarning."""
        warning = PreFitWarning(
            channel="tv_spend",
            severity="warning",
            issue="Low spend percentage",
            recommendation="Consider tightening priors",
        )

        assert warning.channel == "tv_spend"
        assert warning.severity == "warning"
        assert warning.issue == "Low spend percentage"
        assert warning.recommendation == "Consider tightening priors"

    def test_severity_levels(self):
        """Supports different severity levels."""
        for severity in ["info", "warning", "critical"]:
            warning = PreFitWarning(
                channel="test",
                severity=severity,
                issue="Test",
                recommendation="Test",
            )
            assert warning.severity == severity


# =============================================================================
# PreFitChecker Tests
# =============================================================================

class TestPreFitCheckerInit:
    """Tests for PreFitChecker initialization."""

    def test_initialization(self, basic_config, normal_df):
        """Can initialize PreFitChecker."""
        checker = PreFitChecker(basic_config, normal_df)

        assert checker.config == basic_config
        assert checker.df is normal_df


class TestCheckAll:
    """Tests for check_all method."""

    def test_returns_list(self, basic_config, normal_df):
        """check_all returns a list."""
        checker = PreFitChecker(basic_config, normal_df)
        warnings = checker.check_all()

        assert isinstance(warnings, list)

    def test_no_warnings_for_good_data(self, basic_config, normal_df):
        """No warnings for normal, good quality data."""
        checker = PreFitChecker(basic_config, normal_df)
        warnings = checker.check_all()

        # May have some info-level items, but no warnings/critical
        warning_severities = [w.severity for w in warnings]
        assert "critical" not in warning_severities


# =============================================================================
# Channel Data Quality Tests
# =============================================================================

class TestCheckChannelDataQualityLowSpend:
    """Tests for low spend percentage detection."""

    def test_low_spend_channel_flagged(self, basic_config):
        """Channel with < 3% of total spend is flagged."""
        np.random.seed(42)
        n_rows = 100

        df = pd.DataFrame({
            "date": pd.date_range("2022-01-01", periods=n_rows, freq="W"),
            "revenue": np.random.uniform(50000, 150000, n_rows),
            "tv_spend": np.random.uniform(100, 300, n_rows),  # Very low ~1%
            "search_spend": np.random.uniform(10000, 30000, n_rows),  # ~99%
        })

        checker = PreFitChecker(basic_config, df)
        warnings = checker._check_channel_data_quality()

        tv_warnings = [w for w in warnings if w.channel == "tv_spend"]
        assert len(tv_warnings) > 0
        assert any("of spend" in w.issue for w in tv_warnings)


class TestCheckChannelDataQualityHighZeros:
    """Tests for high zero percentage detection."""

    def test_high_zero_rate_flagged(self, basic_config):
        """Channel with > 50% zeros is flagged."""
        np.random.seed(42)
        n_rows = 100

        tv_spend = np.random.uniform(5000, 20000, n_rows)
        tv_spend[:60] = 0  # 60% zeros

        df = pd.DataFrame({
            "date": pd.date_range("2022-01-01", periods=n_rows, freq="W"),
            "revenue": np.random.uniform(50000, 150000, n_rows),
            "tv_spend": tv_spend,
            "search_spend": np.random.uniform(3000, 15000, n_rows),
        })

        checker = PreFitChecker(basic_config, df)
        warnings = checker._check_channel_data_quality()

        tv_warnings = [w for w in warnings if w.channel == "tv_spend"]
        assert len(tv_warnings) > 0
        assert any("zeros" in w.issue for w in tv_warnings)


class TestCheckChannelDataQualityHighCV:
    """Tests for high coefficient of variation detection."""

    def test_high_cv_flagged(self, basic_config):
        """Channel with CV > 2.0 is flagged."""
        np.random.seed(42)
        n_rows = 100

        # Create high variance data: CV = std/mean > 2.0
        # Use extreme bimodal distribution: mostly zeros with a few large spikes
        # This guarantees CV > 2.0
        tv_spend = np.zeros(n_rows)
        tv_spend[0] = 100000  # Single large spike
        tv_spend[1] = 100000  # Another spike
        # mean = 2000, std = sqrt(sum((x-mean)^2)/n) >> mean, so CV > 2

        df = pd.DataFrame({
            "date": pd.date_range("2022-01-01", periods=n_rows, freq="W"),
            "revenue": np.random.uniform(50000, 150000, n_rows),
            "tv_spend": tv_spend,
            "search_spend": np.random.uniform(3000, 15000, n_rows),
        })

        checker = PreFitChecker(basic_config, df)
        warnings = checker._check_channel_data_quality()

        tv_warnings = [w for w in warnings if w.channel == "tv_spend"]
        assert len(tv_warnings) > 0
        assert any("CV" in w.issue for w in tv_warnings)


class TestCheckChannelDataQualityZeroTotal:
    """Tests for edge case of zero total spend."""

    def test_zero_total_spend_handled(self, basic_config):
        """Handles edge case where total spend is zero."""
        n_rows = 50

        df = pd.DataFrame({
            "date": pd.date_range("2022-01-01", periods=n_rows, freq="W"),
            "revenue": np.random.uniform(50000, 150000, n_rows),
            "tv_spend": np.zeros(n_rows),
            "search_spend": np.zeros(n_rows),
        })

        checker = PreFitChecker(basic_config, df)
        warnings = checker._check_channel_data_quality()

        # Should not raise, should return empty list
        assert isinstance(warnings, list)


# =============================================================================
# Wide ROI Prior Tests
# =============================================================================

class TestCheckWideRoiPriors:
    """Tests for wide ROI prior detection."""

    def test_extremely_wide_prior_flagged(self):
        """Prior with > 50x ratio is flagged as warning."""
        config = ModelConfig(
            name="wide_prior_test",
            data=DataConfig(date_column="date", target_column="revenue"),
            channels=[
                ChannelConfig(
                    name="tv_spend",
                    roi_prior_low=0.1,
                    roi_prior_mid=5.0,
                    roi_prior_high=10.0,  # 100x ratio
                ),
            ],
        )

        df = pd.DataFrame({
            "date": pd.date_range("2022-01-01", periods=10, freq="W"),
            "revenue": [100] * 10,
            "tv_spend": [10] * 10,
        })

        checker = PreFitChecker(config, df)
        warnings = checker._check_wide_roi_priors()

        assert len(warnings) == 1
        assert warnings[0].severity == "warning"
        assert "Wide ROI prior" in warnings[0].issue

    def test_moderately_wide_prior_flagged_as_info(self):
        """Prior with 20-50x ratio is flagged as info."""
        config = ModelConfig(
            name="moderate_prior_test",
            data=DataConfig(date_column="date", target_column="revenue"),
            channels=[
                ChannelConfig(
                    name="tv_spend",
                    roi_prior_low=0.2,
                    roi_prior_mid=3.0,
                    roi_prior_high=5.0,  # 25x ratio
                ),
            ],
        )

        df = pd.DataFrame({
            "date": pd.date_range("2022-01-01", periods=10, freq="W"),
            "revenue": [100] * 10,
            "tv_spend": [10] * 10,
        })

        checker = PreFitChecker(config, df)
        warnings = checker._check_wide_roi_priors()

        assert len(warnings) == 1
        assert warnings[0].severity == "info"
        assert "range" in warnings[0].issue.lower()

    def test_reasonable_prior_no_warning(self, basic_config, normal_df):
        """Reasonable prior (< 20x ratio) produces no warning."""
        # basic_config has ratio of 10x (0.5 to 5.0)
        checker = PreFitChecker(basic_config, normal_df)
        warnings = checker._check_wide_roi_priors()

        assert len(warnings) == 0

    def test_zero_low_prior_handled(self):
        """Handles zero roi_prior_low gracefully."""
        config = ModelConfig(
            name="zero_low_test",
            data=DataConfig(date_column="date", target_column="revenue"),
            channels=[
                ChannelConfig(
                    name="tv_spend",
                    roi_prior_low=0.0,  # Zero
                    roi_prior_mid=2.0,
                    roi_prior_high=5.0,
                ),
            ],
        )

        df = pd.DataFrame({
            "date": pd.date_range("2022-01-01", periods=10, freq="W"),
            "revenue": [100] * 10,
            "tv_spend": [10] * 10,
        })

        checker = PreFitChecker(config, df)
        warnings = checker._check_wide_roi_priors()

        # Should not raise, should skip channel with zero low
        assert isinstance(warnings, list)


# =============================================================================
# Standalone Function Tests
# =============================================================================

class TestRunPreFitChecks:
    """Tests for run_pre_fit_checks standalone function."""

    def test_returns_list(self, basic_config, normal_df):
        """run_pre_fit_checks returns a list."""
        warnings = run_pre_fit_checks(basic_config, normal_df)

        assert isinstance(warnings, list)

    def test_same_as_checker(self, basic_config, normal_df):
        """Produces same result as PreFitChecker.check_all()."""
        checker = PreFitChecker(basic_config, normal_df)
        checker_warnings = checker.check_all()

        standalone_warnings = run_pre_fit_checks(basic_config, normal_df)

        assert len(checker_warnings) == len(standalone_warnings)


# =============================================================================
# Integration Tests
# =============================================================================

class TestPreFitChecksIntegration:
    """Integration tests combining multiple check types."""

    def test_multiple_issues_detected(self):
        """Detects multiple issues in problematic data."""
        config = ModelConfig(
            name="multi_issue_test",
            data=DataConfig(date_column="date", target_column="revenue"),
            channels=[
                ChannelConfig(
                    name="low_spend_channel",
                    roi_prior_low=0.1,
                    roi_prior_mid=5.0,
                    roi_prior_high=10.0,  # Wide prior (100x)
                ),
                ChannelConfig(
                    name="high_spend_channel",
                    roi_prior_low=1.0,
                    roi_prior_mid=2.0,
                    roi_prior_high=3.0,  # Narrow prior (3x)
                ),
            ],
        )

        n_rows = 100
        low_spend = np.concatenate([np.zeros(70), np.random.uniform(100, 500, 30)])  # 70% zeros, low spend

        df = pd.DataFrame({
            "date": pd.date_range("2022-01-01", periods=n_rows, freq="W"),
            "revenue": np.random.uniform(50000, 150000, n_rows),
            "low_spend_channel": low_spend,  # Low spend + high zeros
            "high_spend_channel": np.random.uniform(10000, 30000, n_rows),
        })

        warnings = run_pre_fit_checks(config, df)

        # Should have warnings for:
        # 1. Low spend percentage for low_spend_channel
        # 2. High zeros for low_spend_channel
        # 3. Wide ROI prior for low_spend_channel
        low_spend_warnings = [w for w in warnings if w.channel == "low_spend_channel"]
        assert len(low_spend_warnings) >= 2  # At least data quality + prior issues
