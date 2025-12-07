"""
Tests for analysis/bayesian_significance.py - Bayesian significance analysis.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, MagicMock, patch
import xarray as xr

from mmm_platform.analysis.bayesian_significance import (
    CredibleIntervalResult,
    ProbabilityOfDirectionResult,
    ROPEResult,
    ROICredibleIntervalResult,
    PriorSensitivityResult,
    BayesianSignificanceReport,
    BayesianSignificanceAnalyzer,
    get_interpretation_guide,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def channel_cols():
    """Channel column names."""
    return ["tv_spend", "search_spend"]


@pytest.fixture
def sample_df_scaled(channel_cols):
    """Sample scaled DataFrame."""
    np.random.seed(42)
    n_rows = 50

    return pd.DataFrame({
        "date": pd.date_range("2022-01-01", periods=n_rows, freq="W"),
        "revenue": np.random.uniform(50000, 150000, n_rows),
        "tv_spend": np.random.uniform(5000, 20000, n_rows),
        "search_spend": np.random.uniform(3000, 15000, n_rows),
    })


@pytest.fixture
def mock_idata(channel_cols):
    """Mock ArviZ InferenceData with posterior samples."""
    np.random.seed(42)
    n_chains, n_draws = 2, 100
    n_channels = len(channel_cols)

    # Create xarray DataArrays for posterior samples
    beta_samples = xr.DataArray(
        np.random.lognormal(0, 0.5, (n_chains, n_draws, n_channels)),
        dims=["chain", "draw", "channel"],
        coords={"channel": channel_cols},
    )

    alpha_samples = xr.DataArray(
        np.random.beta(2, 3, (n_chains, n_draws, n_channels)),
        dims=["chain", "draw", "channel"],
        coords={"channel": channel_cols},
    )

    lam_samples = xr.DataArray(
        np.random.lognormal(1, 0.3, (n_chains, n_draws, n_channels)),
        dims=["chain", "draw", "channel"],
        coords={"channel": channel_cols},
    )

    # Create mock posterior dataset
    posterior = xr.Dataset({
        "saturation_beta": beta_samples,
        "adstock_alpha": alpha_samples,
        "saturation_lam": lam_samples,
    })

    # Create mock InferenceData
    idata = Mock()
    idata.posterior = posterior

    return idata


@pytest.fixture
def analyzer(mock_idata, sample_df_scaled, channel_cols):
    """Initialized BayesianSignificanceAnalyzer."""
    return BayesianSignificanceAnalyzer(
        idata=mock_idata,
        df_scaled=sample_df_scaled,
        channel_cols=channel_cols,
        target_col="revenue",
        prior_rois={"tv_spend": 1.5, "search_spend": 2.5},
    )


# =============================================================================
# Dataclass Tests
# =============================================================================

class TestCredibleIntervalResult:
    """Tests for CredibleIntervalResult dataclass."""

    def test_creation(self):
        """Can create CredibleIntervalResult."""
        result = CredibleIntervalResult(
            channel="tv_spend",
            mean=1.5,
            hdi_low=0.8,
            hdi_high=2.2,
            excludes_zero=True,
        )

        assert result.channel == "tv_spend"
        assert result.mean == 1.5
        assert result.excludes_zero is True

    def test_interval_width_property(self):
        """interval_width property works correctly."""
        result = CredibleIntervalResult(
            channel="test",
            mean=1.0,
            hdi_low=0.5,
            hdi_high=1.5,
            excludes_zero=True,
        )

        assert result.interval_width == 1.0


class TestProbabilityOfDirectionResult:
    """Tests for ProbabilityOfDirectionResult dataclass."""

    def test_creation(self):
        """Can create ProbabilityOfDirectionResult."""
        result = ProbabilityOfDirectionResult(
            channel="tv_spend",
            pd=0.98,
            interpretation="Very strong",
        )

        assert result.channel == "tv_spend"
        assert result.pd == 0.98

    def test_interpret_pd_very_strong(self):
        """Interprets pd >= 0.99 as 'Very strong'."""
        assert ProbabilityOfDirectionResult.interpret_pd(0.99) == "Very strong"
        assert ProbabilityOfDirectionResult.interpret_pd(1.0) == "Very strong"

    def test_interpret_pd_strong(self):
        """Interprets pd >= 0.95 as 'Strong'."""
        assert ProbabilityOfDirectionResult.interpret_pd(0.95) == "Strong"
        assert ProbabilityOfDirectionResult.interpret_pd(0.98) == "Strong"

    def test_interpret_pd_moderate(self):
        """Interprets pd >= 0.90 as 'Moderate'."""
        assert ProbabilityOfDirectionResult.interpret_pd(0.90) == "Moderate"
        assert ProbabilityOfDirectionResult.interpret_pd(0.94) == "Moderate"

    def test_interpret_pd_weak(self):
        """Interprets pd >= 0.75 as 'Weak'."""
        assert ProbabilityOfDirectionResult.interpret_pd(0.75) == "Weak"
        assert ProbabilityOfDirectionResult.interpret_pd(0.89) == "Weak"

    def test_interpret_pd_inconclusive(self):
        """Interprets pd < 0.75 as 'Inconclusive'."""
        assert ProbabilityOfDirectionResult.interpret_pd(0.50) == "Inconclusive"
        assert ProbabilityOfDirectionResult.interpret_pd(0.74) == "Inconclusive"


class TestROPEResult:
    """Tests for ROPEResult dataclass."""

    def test_creation(self):
        """Can create ROPEResult."""
        result = ROPEResult(
            channel="tv_spend",
            pct_in_rope=0.05,
            pct_below_rope=0.02,
            pct_above_rope=0.93,
            conclusion="Practically significant",
            rope_low=-0.05,
            rope_high=0.05,
        )

        assert result.channel == "tv_spend"
        assert result.pct_above_rope == 0.93
        assert result.conclusion == "Practically significant"


class TestROICredibleIntervalResult:
    """Tests for ROICredibleIntervalResult dataclass."""

    def test_creation(self):
        """Can create ROICredibleIntervalResult."""
        samples = np.array([1.0, 1.5, 2.0, 2.5, 3.0])
        result = ROICredibleIntervalResult(
            channel="tv_spend",
            roi_mean=2.0,
            roi_median=2.0,
            roi_5pct=1.1,
            roi_95pct=2.9,
            roi_hdi_low=1.0,
            roi_hdi_high=2.8,
            significant=True,
            posterior_samples=samples,
        )

        assert result.channel == "tv_spend"
        assert result.significant is True
        assert len(result.posterior_samples) == 5


class TestPriorSensitivityResult:
    """Tests for PriorSensitivityResult dataclass."""

    def test_creation(self):
        """Can create PriorSensitivityResult."""
        result = PriorSensitivityResult(
            channel="tv_spend",
            prior_roi=1.5,
            posterior_roi=2.3,
            shift=0.8,
            relative_shift=0.5,
            data_influence="Strong",
        )

        assert result.channel == "tv_spend"
        assert result.shift == 0.8
        assert result.data_influence == "Strong"


class TestBayesianSignificanceReport:
    """Tests for BayesianSignificanceReport dataclass."""

    def test_to_dataframe(self):
        """to_dataframe returns expected columns."""
        ci = CredibleIntervalResult("tv", 1.0, 0.5, 1.5, True)
        pd_res = ProbabilityOfDirectionResult("tv", 0.98, "Strong")
        rope = ROPEResult("tv", 0.05, 0.02, 0.93, "Significant", -0.05, 0.05)
        roi = ROICredibleIntervalResult("tv", 2.0, 2.0, 1.0, 3.0, 1.0, 3.0, True, np.array([2.0]))
        sens = PriorSensitivityResult("tv", 1.5, 2.0, 0.5, 0.3, "Moderate")

        report = BayesianSignificanceReport(
            credible_intervals=[ci],
            probability_of_direction=[pd_res],
            rope_analysis=[rope],
            roi_posteriors=[roi],
            prior_sensitivity=[sens],
        )

        df = report.to_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert "channel" in df.columns
        assert "beta_mean" in df.columns
        assert "prob_direction" in df.columns
        assert "roi_mean" in df.columns


# =============================================================================
# BayesianSignificanceAnalyzer Tests
# =============================================================================

class TestAnalyzerInit:
    """Tests for analyzer initialization."""

    def test_initialization(self, mock_idata, sample_df_scaled, channel_cols):
        """Can initialize analyzer."""
        analyzer = BayesianSignificanceAnalyzer(
            idata=mock_idata,
            df_scaled=sample_df_scaled,
            channel_cols=channel_cols,
            target_col="revenue",
        )

        assert analyzer.channel_cols == channel_cols
        assert analyzer.hdi_prob == 0.95  # Default

    def test_default_prior_rois(self, mock_idata, sample_df_scaled, channel_cols):
        """Default prior ROIs are 1.0 for all channels."""
        analyzer = BayesianSignificanceAnalyzer(
            idata=mock_idata,
            df_scaled=sample_df_scaled,
            channel_cols=channel_cols,
            target_col="revenue",
        )

        for ch in channel_cols:
            assert analyzer.prior_rois[ch] == 1.0

    def test_custom_rope_bounds(self, mock_idata, sample_df_scaled, channel_cols):
        """Can set custom ROPE bounds."""
        analyzer = BayesianSignificanceAnalyzer(
            idata=mock_idata,
            df_scaled=sample_df_scaled,
            channel_cols=channel_cols,
            target_col="revenue",
            rope_low=-0.1,
            rope_high=0.1,
        )

        assert analyzer.rope_low == -0.1
        assert analyzer.rope_high == 0.1


class TestProbabilityOfDirection:
    """Tests for _probability_of_direction method."""

    def test_all_positive_samples(self, analyzer):
        """Returns 1.0 when all samples are positive."""
        samples = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        pd_value = analyzer._probability_of_direction(samples)

        assert pd_value == 1.0

    def test_all_negative_samples(self, analyzer):
        """Returns 0.0 when all samples are negative."""
        samples = np.array([-1.0, -2.0, -3.0, -4.0, -5.0])
        pd_value = analyzer._probability_of_direction(samples)

        assert pd_value == 0.0

    def test_mixed_samples(self, analyzer):
        """Returns proportion of positive samples."""
        samples = np.array([1.0, 2.0, -1.0, -2.0, 3.0])  # 3/5 = 0.6
        pd_value = analyzer._probability_of_direction(samples)

        assert pd_value == 0.6


class TestRopeAnalysis:
    """Tests for _rope_analysis method."""

    def test_all_above_rope(self, analyzer):
        """Returns correct percentages when all above ROPE."""
        samples = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        in_rope, below_rope, above_rope = analyzer._rope_analysis(samples)

        assert above_rope == 1.0
        assert in_rope == 0.0
        assert below_rope == 0.0

    def test_all_in_rope(self, analyzer):
        """Returns correct percentages when all in ROPE."""
        samples = np.array([0.0, 0.01, -0.01, 0.02, -0.02])
        in_rope, below_rope, above_rope = analyzer._rope_analysis(samples)

        assert in_rope == 1.0

    def test_custom_rope_bounds(self, analyzer):
        """Respects custom ROPE bounds."""
        samples = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        in_rope, below_rope, above_rope = analyzer._rope_analysis(
            samples, rope_low=-0.5, rope_high=0.5
        )

        # 0.0 and 0.5 are in [-0.5, 0.5]
        assert in_rope == 0.4


class TestInterpretRope:
    """Tests for _interpret_rope method."""

    def test_practically_significant(self, analyzer):
        """Above ROPE >= 95% is practically significant."""
        result = analyzer._interpret_rope(in_rope=0.02, above_rope=0.98)
        assert result == "Practically significant"

    def test_practically_zero(self, analyzer):
        """In ROPE >= 95% is practically zero."""
        result = analyzer._interpret_rope(in_rope=0.98, above_rope=0.01)
        assert result == "Practically zero"

    def test_uncertain(self, analyzer):
        """Otherwise is uncertain."""
        result = analyzer._interpret_rope(in_rope=0.50, above_rope=0.40)
        assert result == "Uncertain"


class TestComputeProbabilityOfDirection:
    """Tests for compute_probability_of_direction method."""

    def test_returns_list(self, analyzer):
        """Returns list of results."""
        results = analyzer.compute_probability_of_direction()

        assert isinstance(results, list)
        assert len(results) == 2  # Two channels

    def test_result_structure(self, analyzer):
        """Results have expected structure."""
        results = analyzer.compute_probability_of_direction()

        for result in results:
            assert isinstance(result, ProbabilityOfDirectionResult)
            assert result.channel in analyzer.channel_cols
            assert 0 <= result.pd <= 1


class TestComputeRopeAnalysis:
    """Tests for compute_rope_analysis method."""

    def test_returns_list(self, analyzer):
        """Returns list of results."""
        results = analyzer.compute_rope_analysis()

        assert isinstance(results, list)
        assert len(results) == 2

    def test_result_structure(self, analyzer):
        """Results have expected structure."""
        results = analyzer.compute_rope_analysis()

        for result in results:
            assert isinstance(result, ROPEResult)
            assert result.pct_in_rope + result.pct_below_rope + result.pct_above_rope == pytest.approx(1.0)


class TestComputePriorSensitivity:
    """Tests for compute_prior_sensitivity method."""

    def test_returns_list(self, analyzer):
        """Returns list of results."""
        # First compute ROI posteriors
        roi_results = analyzer.compute_roi_posteriors()
        results = analyzer.compute_prior_sensitivity(roi_results)

        assert isinstance(results, list)
        assert len(results) == 2

    def test_data_influence_interpretation(self, analyzer):
        """Data influence is correctly interpreted."""
        roi_results = analyzer.compute_roi_posteriors()
        results = analyzer.compute_prior_sensitivity(roi_results)

        for result in results:
            assert result.data_influence in ["Strong", "Moderate", "Weak (prior-driven)"]


class TestRunFullAnalysis:
    """Tests for run_full_analysis method."""

    @patch('mmm_platform.analysis.bayesian_significance.az.summary')
    def test_returns_report(self, mock_summary, analyzer):
        """Returns BayesianSignificanceReport."""
        # Mock ArviZ summary
        mock_summary.return_value = pd.DataFrame({
            "mean": [1.0, 1.5],
            "hdi_2.5%": [0.5, 0.8],
            "hdi_97.5%": [1.5, 2.2],
        }, index=["saturation_beta[tv_spend]", "saturation_beta[search_spend]"])

        report = analyzer.run_full_analysis()

        assert isinstance(report, BayesianSignificanceReport)
        assert len(report.credible_intervals) == 2
        assert len(report.probability_of_direction) == 2
        assert len(report.rope_analysis) == 2


# =============================================================================
# Interpretation Guide Tests
# =============================================================================

class TestGetInterpretationGuide:
    """Tests for get_interpretation_guide function."""

    def test_returns_string(self):
        """Returns string."""
        guide = get_interpretation_guide()

        assert isinstance(guide, str)

    def test_contains_key_concepts(self):
        """Contains key Bayesian concepts."""
        guide = get_interpretation_guide()

        assert "Bayesian" in guide or "bayesian" in guide
        assert "HDI" in guide or "credible" in guide.lower()
        assert "ROPE" in guide
