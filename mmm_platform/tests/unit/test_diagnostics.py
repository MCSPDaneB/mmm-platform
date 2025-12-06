"""Unit tests for model diagnostics and recommendations."""

import pytest
from mmm_platform.model.diagnostics import (
    DiagnosticsAdvisor,
    DiagnosticResult,
    Recommendation,
)


class TestDiagnosticsAdvisor:
    """Tests for DiagnosticsAdvisor class."""

    @pytest.fixture
    def advisor(self):
        """Create a DiagnosticsAdvisor instance."""
        return DiagnosticsAdvisor()

    @pytest.fixture
    def default_sampling_config(self):
        """Default sampling configuration."""
        return {
            "draws": 1500,
            "tune": 1500,
            "chains": 4,
            "target_accept": 0.9,
        }

    def test_no_issues_returns_empty_list(self, advisor, default_sampling_config):
        """Verify no diagnostics returned when model converged without issues."""
        convergence = {
            "converged": True,
            "divergences": 0,
            "high_rhat_params": [],
            "warnings": [],
            "ess_bulk_min": 1000.0,
            "ess_tail_min": 800.0,
            "ess_sufficient": True,
        }

        results = advisor.analyze_from_convergence_dict(convergence, default_sampling_config)

        assert results == []

    def test_none_convergence_returns_empty(self, advisor, default_sampling_config):
        """Verify None convergence data returns empty list."""
        results = advisor.analyze_from_convergence_dict(None, default_sampling_config)
        assert results == []


class TestDivergenceChecks:
    """Tests for divergence detection and recommendations."""

    @pytest.fixture
    def advisor(self):
        return DiagnosticsAdvisor()

    def test_few_divergences_warning(self, advisor):
        """Verify 1-50 divergences triggers warning severity."""
        convergence = {
            "divergences": 25,
            "high_rhat_params": [],
            "ess_sufficient": True,
        }
        sampling_config = {"target_accept": 0.9, "tune": 1500}

        results = advisor.analyze_from_convergence_dict(convergence, sampling_config)

        assert len(results) == 1
        assert results[0].issue == "Divergent Transitions"
        assert results[0].severity == "warning"
        assert "25 divergent" in results[0].details

    def test_moderate_divergences_warning_with_tune(self, advisor):
        """Verify 50-100 divergences suggests target_accept and tune."""
        convergence = {
            "divergences": 75,
            "high_rhat_params": [],
            "ess_sufficient": True,
        }
        sampling_config = {"target_accept": 0.9, "tune": 1500}

        results = advisor.analyze_from_convergence_dict(convergence, sampling_config)

        assert len(results) == 1
        assert results[0].severity == "warning"

        # Should have both target_accept and tune recommendations
        settings = [rec.setting for rec in results[0].recommendations]
        assert "target_accept" in settings
        assert "tune" in settings

    def test_many_divergences_critical(self, advisor):
        """Verify >100 divergences triggers critical severity."""
        convergence = {
            "divergences": 150,
            "high_rhat_params": [],
            "ess_sufficient": True,
        }
        sampling_config = {"target_accept": 0.9, "tune": 1500}

        results = advisor.analyze_from_convergence_dict(convergence, sampling_config)

        assert len(results) == 1
        assert results[0].issue == "Divergent Transitions"
        assert results[0].severity == "critical"
        assert "unreliable" in results[0].details

        # Should recommend priors review for critical divergences
        settings = [rec.setting for rec in results[0].recommendations]
        assert "priors" in settings

    def test_no_recommendation_if_target_accept_already_high(self, advisor):
        """Verify no target_accept recommendation if already at 0.95."""
        convergence = {
            "divergences": 25,
            "high_rhat_params": [],
            "ess_sufficient": True,
        }
        sampling_config = {"target_accept": 0.95, "tune": 1500}

        results = advisor.analyze_from_convergence_dict(convergence, sampling_config)

        assert len(results) == 1
        settings = [rec.setting for rec in results[0].recommendations]
        assert "target_accept" not in settings

    def test_zero_divergences_no_result(self, advisor):
        """Verify zero divergences produces no diagnostic."""
        convergence = {
            "divergences": 0,
            "high_rhat_params": [],
            "ess_sufficient": True,
        }

        results = advisor.analyze_from_convergence_dict(convergence)

        assert len(results) == 0


class TestRhatChecks:
    """Tests for R-hat value detection and recommendations."""

    @pytest.fixture
    def advisor(self):
        return DiagnosticsAdvisor()

    def test_few_high_rhat_warning(self, advisor):
        """Verify 1-3 high R-hat params triggers warning."""
        convergence = {
            "divergences": 0,
            "high_rhat_params": ["alpha", "beta"],
            "ess_sufficient": True,
        }
        sampling_config = {"draws": 1500}

        results = advisor.analyze_from_convergence_dict(convergence, sampling_config)

        assert len(results) == 1
        assert results[0].issue == "High R-hat Values"
        assert results[0].severity == "warning"
        assert "alpha" in results[0].details

    def test_many_high_rhat_critical(self, advisor):
        """Verify >3 high R-hat params triggers critical severity."""
        convergence = {
            "divergences": 0,
            "high_rhat_params": ["alpha", "beta", "gamma", "delta", "epsilon"],
            "ess_sufficient": True,
        }
        sampling_config = {"draws": 1500, "tune": 1500, "chains": 4}

        results = advisor.analyze_from_convergence_dict(convergence, sampling_config)

        assert len(results) == 1
        assert results[0].severity == "critical"
        assert "5 parameters" in results[0].details

        # Should recommend more draws and tune
        settings = [rec.setting for rec in results[0].recommendations]
        assert "draws" in settings
        assert "tune" in settings

    def test_rhat_recommends_more_chains_if_low(self, advisor):
        """Verify chains recommendation if using fewer than 4."""
        convergence = {
            "divergences": 0,
            "high_rhat_params": ["a", "b", "c", "d"],  # 4 params = critical
            "ess_sufficient": True,
        }
        sampling_config = {"draws": 1500, "tune": 1500, "chains": 2}

        results = advisor.analyze_from_convergence_dict(convergence, sampling_config)

        settings = [rec.setting for rec in results[0].recommendations]
        assert "chains" in settings

        chains_rec = [r for r in results[0].recommendations if r.setting == "chains"][0]
        assert chains_rec.current == 2
        assert chains_rec.suggested == 4

    def test_no_rhat_issue_if_empty_list(self, advisor):
        """Verify no R-hat diagnostic if high_rhat_params is empty."""
        convergence = {
            "divergences": 0,
            "high_rhat_params": [],
            "ess_sufficient": True,
        }

        results = advisor.analyze_from_convergence_dict(convergence)

        assert len(results) == 0


class TestESSChecks:
    """Tests for Effective Sample Size detection and recommendations."""

    @pytest.fixture
    def advisor(self):
        return DiagnosticsAdvisor()

    def test_low_ess_warning(self, advisor):
        """Verify ESS 100-400 triggers warning."""
        convergence = {
            "divergences": 0,
            "high_rhat_params": [],
            "ess_bulk_min": 250.0,
            "ess_tail_min": 300.0,
            "ess_sufficient": False,
        }
        sampling_config = {"draws": 1500, "tune": 1500}

        results = advisor.analyze_from_convergence_dict(convergence, sampling_config)

        assert len(results) == 1
        assert results[0].issue == "Low Effective Sample Size"
        assert results[0].severity == "warning"
        assert "250" in results[0].details

    def test_very_low_ess_critical(self, advisor):
        """Verify ESS < 100 triggers critical severity."""
        convergence = {
            "divergences": 0,
            "high_rhat_params": [],
            "ess_bulk_min": 50.0,
            "ess_tail_min": 75.0,
            "ess_sufficient": False,
        }
        sampling_config = {"draws": 1500, "tune": 1500}

        results = advisor.analyze_from_convergence_dict(convergence, sampling_config)

        assert len(results) == 1
        assert results[0].severity == "critical"
        assert "unreliable" in results[0].details

        # Should recommend both draws and tune
        settings = [rec.setting for rec in results[0].recommendations]
        assert "draws" in settings
        assert "tune" in settings

    def test_ess_sufficient_no_result(self, advisor):
        """Verify no ESS diagnostic if sufficient."""
        convergence = {
            "divergences": 0,
            "high_rhat_params": [],
            "ess_bulk_min": 1000.0,
            "ess_tail_min": 800.0,
            "ess_sufficient": True,
        }

        results = advisor.analyze_from_convergence_dict(convergence)

        assert len(results) == 0

    def test_ess_uses_min_of_bulk_and_tail(self, advisor):
        """Verify ESS check uses the minimum of bulk and tail."""
        convergence = {
            "divergences": 0,
            "high_rhat_params": [],
            "ess_bulk_min": 500.0,
            "ess_tail_min": 80.0,  # Lower tail ESS should trigger
            "ess_sufficient": False,
        }

        results = advisor.analyze_from_convergence_dict(convergence)

        assert len(results) == 1
        assert results[0].severity == "critical"  # 80 < 100
        assert "80" in results[0].details


class TestMultipleIssues:
    """Tests for handling multiple convergence issues."""

    @pytest.fixture
    def advisor(self):
        return DiagnosticsAdvisor()

    def test_multiple_issues_all_reported(self, advisor):
        """Verify all issues are reported when multiple problems exist."""
        convergence = {
            "divergences": 60,  # Warning
            "high_rhat_params": ["alpha", "beta"],  # Warning
            "ess_bulk_min": 200.0,  # Warning
            "ess_tail_min": 150.0,
            "ess_sufficient": False,
        }
        sampling_config = {"draws": 1500, "tune": 1500, "chains": 4, "target_accept": 0.9}

        results = advisor.analyze_from_convergence_dict(convergence, sampling_config)

        assert len(results) == 3
        issues = [r.issue for r in results]
        assert "Divergent Transitions" in issues
        assert "High R-hat Values" in issues
        assert "Low Effective Sample Size" in issues


class TestRecommendationClass:
    """Tests for Recommendation dataclass."""

    def test_recommendation_fields(self):
        """Verify Recommendation has all required fields."""
        rec = Recommendation(
            setting="draws",
            current=1500,
            suggested=2500,
            reason="More samples needed"
        )

        assert rec.setting == "draws"
        assert rec.current == 1500
        assert rec.suggested == 2500
        assert rec.reason == "More samples needed"


class TestDiagnosticResultClass:
    """Tests for DiagnosticResult dataclass."""

    def test_diagnostic_result_fields(self):
        """Verify DiagnosticResult has all required fields."""
        result = DiagnosticResult(
            issue="Test Issue",
            severity="warning",
            details="Test details",
            recommendations=[
                Recommendation("setting", 1, 2, "reason")
            ]
        )

        assert result.issue == "Test Issue"
        assert result.severity == "warning"
        assert result.details == "Test details"
        assert len(result.recommendations) == 1

    def test_diagnostic_result_default_recommendations(self):
        """Verify recommendations defaults to empty list."""
        result = DiagnosticResult(
            issue="Test",
            severity="warning",
            details="Details"
        )

        assert result.recommendations == []


class TestFormatRecommendationsText:
    """Tests for text formatting of recommendations."""

    @pytest.fixture
    def advisor(self):
        return DiagnosticsAdvisor()

    def test_format_empty_results(self, advisor):
        """Verify empty results produces appropriate message."""
        text = advisor.format_recommendations_text([])
        assert "No convergence issues detected" in text

    def test_format_warning(self, advisor):
        """Verify warning is formatted with correct icon."""
        results = [
            DiagnosticResult(
                issue="Test Warning",
                severity="warning",
                details="Warning details"
            )
        ]

        text = advisor.format_recommendations_text(results)

        assert "⚠️" in text
        assert "Test Warning" in text
        assert "Warning details" in text

    def test_format_critical(self, advisor):
        """Verify critical is formatted with correct icon."""
        results = [
            DiagnosticResult(
                issue="Test Critical",
                severity="critical",
                details="Critical details"
            )
        ]

        text = advisor.format_recommendations_text(results)

        assert "🔴" in text
        assert "Test Critical" in text

    def test_format_with_recommendations(self, advisor):
        """Verify recommendations are included in formatted text."""
        results = [
            DiagnosticResult(
                issue="Test",
                severity="warning",
                details="Details",
                recommendations=[
                    Recommendation("draws", 1500, 2500, "More samples")
                ]
            )
        ]

        text = advisor.format_recommendations_text(results)

        assert "draws" in text
        assert "1500" in text
        assert "2500" in text
        assert "More samples" in text
