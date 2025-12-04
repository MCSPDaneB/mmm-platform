"""
Unit tests for analysis/roi_diagnostics.py

Tests ROI prior validation against posterior results.
"""
import pytest
import numpy as np
from numpy.testing import assert_allclose

from mmm_platform.analysis.roi_diagnostics import (
    ROIDiagnostics, ROIDiagnosticReport, ChannelROIResult
)


class TestChannelROIResult:
    """Tests for ChannelROIResult dataclass."""

    def test_summary_dict_keys(self):
        """summary_dict should return expected keys."""
        result = ChannelROIResult(
            channel_name="tv_spend",
            prior_roi_low=0.5,
            prior_roi_mid=1.5,
            prior_roi_high=3.0,
            posterior_roi_mean=1.8,
            posterior_roi_hdi_low=1.2,
            posterior_roi_hdi_high=2.5,
        )
        summary = result.summary_dict()

        assert "channel" in summary
        assert "prior_roi_mid" in summary
        assert "posterior_roi_mean" in summary
        assert summary["channel"] == "tv_spend"

    def test_warnings_default_empty(self):
        """Warnings should default to empty list."""
        result = ChannelROIResult(
            channel_name="tv",
            prior_roi_low=0.5,
            prior_roi_mid=1.5,
            prior_roi_high=3.0,
        )
        assert result.warnings == []

    def test_all_fields_in_summary(self):
        """summary_dict should include all relevant fields."""
        result = ChannelROIResult(
            channel_name="tv_spend",
            prior_roi_low=0.5,
            prior_roi_mid=1.5,
            prior_roi_high=3.0,
            posterior_roi_mean=1.8,
            posterior_roi_std=0.3,
            posterior_roi_hdi_low=1.2,
            posterior_roi_hdi_high=2.5,
            prior_belief_in_posterior_hdi=True,
            posterior_vs_prior_shift=0.2,
            lambda_shift=0.1,
        )
        summary = result.summary_dict()

        assert summary["prior_roi_low"] == 0.5
        assert summary["prior_roi_mid"] == 1.5
        assert summary["prior_roi_high"] == 3.0
        assert summary["posterior_roi_mean"] == 1.8
        assert summary["prior_in_hdi"] is True
        assert summary["roi_shift_pct"] == 0.2


class TestROIDiagnosticReport:
    """Tests for ROIDiagnosticReport dataclass."""

    def test_to_dataframe_columns(self):
        """to_dataframe should return expected columns."""
        result = ChannelROIResult(
            channel_name="tv_spend",
            prior_roi_low=0.5,
            prior_roi_mid=1.5,
            prior_roi_high=3.0,
            posterior_roi_mean=1.8,
            posterior_roi_std=0.3,
            posterior_roi_hdi_low=1.2,
            posterior_roi_hdi_high=2.5,
            prior_belief_in_posterior_hdi=True,
            posterior_vs_prior_shift=0.2,
            lambda_shift=0.1,
        )
        report = ROIDiagnosticReport(channel_results={"tv_spend": result})
        df = report.to_dataframe()

        assert "channel" in df.columns
        assert "prior_roi_mid" in df.columns
        assert "posterior_roi_mean" in df.columns
        assert len(df) == 1

    def test_empty_report(self):
        """Empty report should have sensible defaults."""
        report = ROIDiagnosticReport(channel_results={})
        assert report.overall_health == "Not assessed"
        assert report.recommendations == []

    def test_multiple_channels(self):
        """Report should handle multiple channels."""
        results = {
            "tv_spend": ChannelROIResult(
                channel_name="tv_spend",
                prior_roi_low=0.5, prior_roi_mid=1.5, prior_roi_high=3.0,
            ),
            "search_spend": ChannelROIResult(
                channel_name="search_spend",
                prior_roi_low=1.0, prior_roi_mid=2.5, prior_roi_high=5.0,
            ),
        }
        report = ROIDiagnosticReport(channel_results=results)
        df = report.to_dataframe()

        assert len(df) == 2
        assert set(df["channel"].tolist()) == {"tv_spend", "search_spend"}

    def test_channels_with_tension_tracking(self):
        """Should track channels with prior tension."""
        report = ROIDiagnosticReport(
            channel_results={},
            channels_with_prior_tension=["tv_spend", "display_spend"],
        )
        assert len(report.channels_with_prior_tension) == 2
        assert "tv_spend" in report.channels_with_prior_tension

    def test_channels_with_large_shift_tracking(self):
        """Should track channels with large ROI shift."""
        report = ROIDiagnosticReport(
            channel_results={},
            channels_with_large_shift=["search_spend"],
        )
        assert len(report.channels_with_large_shift) == 1


class TestROIDiagnostics:
    """Tests for ROIDiagnostics class."""

    def test_extract_roi_beliefs(self, basic_config):
        """Should extract ROI beliefs from config."""
        # Create a mock wrapper-like object
        class MockWrapper:
            def __init__(self, config):
                self.config = config
                self.idata = None

        wrapper = MockWrapper(basic_config)
        diagnostics = ROIDiagnostics(wrapper)

        beliefs = diagnostics.roi_beliefs
        assert "tv_spend" in beliefs
        assert beliefs["tv_spend"]["mid"] == 1.5
        assert beliefs["tv_spend"]["low"] == 0.5
        assert beliefs["tv_spend"]["high"] == 3.0

    def test_extract_multiple_channel_beliefs(self, basic_config):
        """Should extract beliefs for all channels."""
        class MockWrapper:
            def __init__(self, config):
                self.config = config
                self.idata = None

        wrapper = MockWrapper(basic_config)
        diagnostics = ROIDiagnostics(wrapper)

        beliefs = diagnostics.roi_beliefs
        assert len(beliefs) == 2  # tv_spend and search_spend
        assert "search_spend" in beliefs
        assert beliefs["search_spend"]["mid"] == 2.5

    def test_geometric_adstock_matches_transforms(self, basic_config):
        """Adstock implementation should match TransformEngine."""
        from mmm_platform.core.transforms import TransformEngine

        class MockWrapper:
            def __init__(self, config):
                self.config = config
                self.idata = None

        wrapper = MockWrapper(basic_config)
        diagnostics = ROIDiagnostics(wrapper)
        engine = TransformEngine(basic_config)

        np.random.seed(42)
        x = np.random.rand(50)
        alpha = 0.5
        l_max = 8

        result_diag = diagnostics._geometric_adstock(x, alpha, l_max)
        result_engine = engine.compute_geometric_adstock(x, alpha, l_max)

        assert_allclose(result_diag, result_engine, rtol=1e-10)

    def test_logistic_saturation_matches_transforms(self, basic_config):
        """Saturation implementation should match TransformEngine."""
        from mmm_platform.core.transforms import TransformEngine

        class MockWrapper:
            def __init__(self, config):
                self.config = config
                self.idata = None

        wrapper = MockWrapper(basic_config)
        diagnostics = ROIDiagnostics(wrapper)
        engine = TransformEngine(basic_config)

        x = np.linspace(0, 1, 50)
        lam = 3.0

        result_diag = diagnostics._logistic_saturation(x, lam)
        result_engine = engine.compute_logistic_saturation(x, lam)

        assert_allclose(result_diag, result_engine, rtol=1e-10)

    def test_validate_posterior_requires_fitted_model(self, basic_config):
        """validate_posterior should raise if model not fitted."""
        class MockWrapper:
            def __init__(self, config):
                self.config = config
                self.idata = None

        wrapper = MockWrapper(basic_config)
        diagnostics = ROIDiagnostics(wrapper)

        with pytest.raises(ValueError, match="must be fitted"):
            diagnostics.validate_posterior()

    def test_hdi_prob_parameter(self, basic_config):
        """Should accept custom HDI probability."""
        class MockWrapper:
            def __init__(self, config):
                self.config = config
                self.idata = None

        wrapper = MockWrapper(basic_config)
        diagnostics = ROIDiagnostics(wrapper, hdi_prob=0.95)

        assert diagnostics.hdi_prob == 0.95


class TestROIDiagnosticsTransformConsistency:
    """Tests to ensure transform implementations are consistent."""

    def test_adstock_output_shape(self, basic_config):
        """Adstock should preserve input shape."""
        class MockWrapper:
            def __init__(self, config):
                self.config = config
                self.idata = None

        wrapper = MockWrapper(basic_config)
        diagnostics = ROIDiagnostics(wrapper)

        x = np.random.rand(100)
        result = diagnostics._geometric_adstock(x, alpha=0.5, l_max=8)

        assert result.shape == x.shape

    def test_saturation_output_range(self, basic_config):
        """Saturation output should be bounded."""
        class MockWrapper:
            def __init__(self, config):
                self.config = config
                self.idata = None

        wrapper = MockWrapper(basic_config)
        diagnostics = ROIDiagnostics(wrapper)

        x = np.linspace(0, 10, 100)
        result = diagnostics._logistic_saturation(x, lam=2.0)

        assert np.all(result >= -1)
        assert np.all(result <= 1)

    def test_saturation_monotonic(self, basic_config):
        """Saturation should be monotonically increasing."""
        class MockWrapper:
            def __init__(self, config):
                self.config = config
                self.idata = None

        wrapper = MockWrapper(basic_config)
        diagnostics = ROIDiagnostics(wrapper)

        x = np.linspace(0, 5, 100)
        result = diagnostics._logistic_saturation(x, lam=2.0)

        diffs = np.diff(result)
        assert np.all(diffs >= 0)
