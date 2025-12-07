"""
Tests for analysis/executive_summary.py - Executive summary generation.
"""

import pytest
from unittest.mock import Mock, MagicMock
import json

from mmm_platform.analysis.executive_summary import (
    ReallocationRecommendation,
    ExecutiveSummaryResult,
    ExecutiveSummaryGenerator,
)
from mmm_platform.analysis.marginal_roi import (
    ChannelMarginalROI,
    InvestmentPriorityResult,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_channel_increase():
    """Channel recommended for increase investment."""
    return ChannelMarginalROI(
        channel="search_spend",
        channel_name="Paid Search",
        current_spend=50000.0,
        current_roi=3.5,
        marginal_roi=2.5,  # > 1.50 threshold
        breakeven_spend=100000.0,
        headroom=True,
        headroom_amount=50000.0,
        priority_rank=1,
        roi_5pct=1.8,
        roi_95pct=4.2,
        roi_uncertainty=2.4,
        prob_profitable=0.98,
        needs_test=False,
    )


@pytest.fixture
def sample_channel_hold():
    """Channel recommended for hold steady."""
    return ChannelMarginalROI(
        channel="display_spend",
        channel_name="Display Ads",
        current_spend=30000.0,
        current_roi=1.8,
        marginal_roi=1.25,  # Between 1.00 and 1.50
        breakeven_spend=40000.0,
        headroom=True,
        headroom_amount=10000.0,
        priority_rank=2,
        roi_5pct=0.8,
        roi_95pct=2.1,
        roi_uncertainty=1.3,
        prob_profitable=0.85,
        needs_test=True,
    )


@pytest.fixture
def sample_channel_reduce():
    """Channel recommended for reduce investment."""
    return ChannelMarginalROI(
        channel="tv_spend",
        channel_name="TV Advertising",
        current_spend=80000.0,
        current_roi=1.2,
        marginal_roi=0.7,  # < 1.00 threshold
        breakeven_spend=60000.0,
        headroom=False,
        headroom_amount=0.0,
        priority_rank=3,
        roi_5pct=0.4,
        roi_95pct=1.5,
        roi_uncertainty=1.1,
        prob_profitable=0.65,
        needs_test=False,
    )


@pytest.fixture
def sample_priority_result(sample_channel_increase, sample_channel_hold, sample_channel_reduce):
    """Sample InvestmentPriorityResult with all channel types."""
    return InvestmentPriorityResult(
        channel_analysis=[sample_channel_increase, sample_channel_hold, sample_channel_reduce],
        increase_channels=[sample_channel_increase],
        hold_channels=[sample_channel_hold],
        reduce_channels=[sample_channel_reduce],
        channels_needing_test=[sample_channel_hold],
        total_spend=160000.0,
        total_contribution=280000.0,
        portfolio_roi=1.75,
        reallocation_potential=20000.0,  # Amount to move from reduce channels
        headroom_available=50000.0,  # Headroom in increase channels
    )


@pytest.fixture
def mock_marginal_analyzer(sample_priority_result):
    """Mock MarginalROIAnalyzer with preset results."""
    analyzer = Mock()
    analyzer.run_full_analysis.return_value = sample_priority_result
    return analyzer


@pytest.fixture
def generator(mock_marginal_analyzer, sample_priority_result):
    """ExecutiveSummaryGenerator for testing."""
    return ExecutiveSummaryGenerator(
        marginal_analyzer=mock_marginal_analyzer,
        priority_result=sample_priority_result,
    )


# =============================================================================
# Dataclass Tests
# =============================================================================

class TestReallocationRecommendation:
    """Tests for ReallocationRecommendation dataclass."""

    def test_creation(self):
        """Can create ReallocationRecommendation."""
        rec = ReallocationRecommendation(
            from_channel="TV",
            to_channel="Search",
            amount=10000.0,
            expected_return=25000.0,
            needs_validation=False,
        )

        assert rec.from_channel == "TV"
        assert rec.to_channel == "Search"
        assert rec.amount == 10000.0
        assert rec.expected_return == 25000.0
        assert rec.needs_validation is False

    def test_with_validation_needed(self):
        """Can mark recommendation as needing validation."""
        rec = ReallocationRecommendation(
            from_channel="Reduce channels (combined)",
            to_channel="Display",
            amount=5000.0,
            expected_return=6000.0,
            needs_validation=True,
        )

        assert rec.needs_validation is True


class TestExecutiveSummaryResult:
    """Tests for ExecutiveSummaryResult dataclass."""

    def test_creation(self, sample_priority_result):
        """Can create ExecutiveSummaryResult."""
        rec = ReallocationRecommendation(
            from_channel="X",
            to_channel="Y",
            amount=1000.0,
            expected_return=2000.0,
            needs_validation=False,
        )

        result = ExecutiveSummaryResult(
            priority_result=sample_priority_result,
            reallocation_recommendations=[rec],
            summary_text="Test summary",
        )

        assert result.priority_result == sample_priority_result
        assert len(result.reallocation_recommendations) == 1
        assert result.summary_text == "Test summary"


# =============================================================================
# ExecutiveSummaryGenerator Tests
# =============================================================================

class TestExecutiveSummaryGenerator:
    """Tests for ExecutiveSummaryGenerator class."""

    def test_initialization(self, mock_marginal_analyzer):
        """Can initialize with marginal analyzer."""
        gen = ExecutiveSummaryGenerator(marginal_analyzer=mock_marginal_analyzer)

        assert gen.analyzer == mock_marginal_analyzer
        assert gen._priority_result is None

    def test_initialization_with_priority_result(self, mock_marginal_analyzer, sample_priority_result):
        """Can initialize with pre-computed priority result."""
        gen = ExecutiveSummaryGenerator(
            marginal_analyzer=mock_marginal_analyzer,
            priority_result=sample_priority_result,
        )

        assert gen._priority_result == sample_priority_result

    def test_priority_result_property_caches(self, mock_marginal_analyzer):
        """priority_result property caches the result."""
        gen = ExecutiveSummaryGenerator(marginal_analyzer=mock_marginal_analyzer)

        # First access should call run_full_analysis
        _ = gen.priority_result
        mock_marginal_analyzer.run_full_analysis.assert_called_once()

        # Second access should use cached value
        _ = gen.priority_result
        mock_marginal_analyzer.run_full_analysis.assert_called_once()


class TestGenerateReallocationRecommendations:
    """Tests for generate_reallocation_recommendations method."""

    def test_returns_list(self, generator):
        """Returns list of recommendations."""
        recs = generator.generate_reallocation_recommendations()

        assert isinstance(recs, list)

    def test_recommendations_target_increase_channels(self, generator):
        """Recommendations target increase channels."""
        recs = generator.generate_reallocation_recommendations()

        for rec in recs:
            assert rec.to_channel == "Paid Search"  # Only increase channel

    def test_respects_max_recommendations(self, generator):
        """Respects max_recommendations parameter."""
        recs = generator.generate_reallocation_recommendations(max_recommendations=1)

        assert len(recs) <= 1

    def test_expected_return_calculation(self, generator):
        """Expected return is calculated correctly."""
        recs = generator.generate_reallocation_recommendations()

        if recs:
            rec = recs[0]
            # Expected return = amount * marginal_roi
            expected = rec.amount * 2.5  # search_spend marginal_roi
            assert abs(rec.expected_return - expected) < 0.01


class TestGenerateSummary:
    """Tests for generate_summary method."""

    def test_returns_summary_result(self, generator):
        """Returns ExecutiveSummaryResult."""
        result = generator.generate_summary()

        assert isinstance(result, ExecutiveSummaryResult)

    def test_summary_has_priority_result(self, generator, sample_priority_result):
        """Summary includes priority result."""
        result = generator.generate_summary()

        assert result.priority_result == sample_priority_result

    def test_summary_has_recommendations(self, generator):
        """Summary includes recommendations."""
        result = generator.generate_summary()

        assert isinstance(result.reallocation_recommendations, list)

    def test_summary_has_text(self, generator):
        """Summary includes formatted text."""
        result = generator.generate_summary()

        assert isinstance(result.summary_text, str)
        assert len(result.summary_text) > 0


class TestBuildSummaryText:
    """Tests for _build_summary_text method."""

    def test_includes_executive_summary_header(self, generator):
        """Summary text includes header."""
        result = generator.generate_summary()

        assert "EXECUTIVE SUMMARY" in result.summary_text

    def test_includes_increase_section(self, generator):
        """Summary text includes increase section."""
        result = generator.generate_summary()

        assert "INCREASE INVESTMENT" in result.summary_text
        assert "Paid Search" in result.summary_text

    def test_includes_hold_section(self, generator):
        """Summary text includes hold section."""
        result = generator.generate_summary()

        assert "HOLD STEADY" in result.summary_text
        assert "Display Ads" in result.summary_text

    def test_includes_reduce_section(self, generator):
        """Summary text includes reduce section."""
        result = generator.generate_summary()

        assert "REDUCE OR REALLOCATE" in result.summary_text
        assert "TV Advertising" in result.summary_text

    def test_includes_validation_section(self, generator):
        """Summary text includes validation section for channels needing test."""
        result = generator.generate_summary()

        assert "VALIDATION" in result.summary_text

    def test_includes_priority_ranking_table(self, generator):
        """Summary text includes priority ranking table."""
        result = generator.generate_summary()

        assert "PRIORITY RANKING" in result.summary_text

    def test_includes_portfolio_summary(self, generator):
        """Summary text includes portfolio summary."""
        result = generator.generate_summary()

        assert "PORTFOLIO SUMMARY" in result.summary_text


class TestGetSummaryDict:
    """Tests for get_summary_dict method."""

    def test_returns_dict(self, generator):
        """Returns dictionary."""
        summary_dict = generator.get_summary_dict()

        assert isinstance(summary_dict, dict)

    def test_json_serializable(self, generator):
        """Result is JSON serializable."""
        summary_dict = generator.get_summary_dict()

        # Should not raise
        json_str = json.dumps(summary_dict)
        assert isinstance(json_str, str)

    def test_has_portfolio_section(self, generator):
        """Has portfolio section with key metrics."""
        summary_dict = generator.get_summary_dict()

        assert "portfolio" in summary_dict
        assert "total_spend" in summary_dict["portfolio"]
        assert "total_contribution" in summary_dict["portfolio"]
        assert "portfolio_roi" in summary_dict["portfolio"]

    def test_has_recommendations_section(self, generator):
        """Has recommendations section with channel lists."""
        summary_dict = generator.get_summary_dict()

        assert "recommendations" in summary_dict
        assert "increase" in summary_dict["recommendations"]
        assert "hold" in summary_dict["recommendations"]
        assert "reduce" in summary_dict["recommendations"]

    def test_has_priority_table(self, generator):
        """Has priority table with all channels."""
        summary_dict = generator.get_summary_dict()

        assert "priority_table" in summary_dict
        assert len(summary_dict["priority_table"]) == 3  # 3 channels

    def test_has_counts(self, generator):
        """Has counts of channels in each category."""
        summary_dict = generator.get_summary_dict()

        assert "counts" in summary_dict
        assert summary_dict["counts"]["increase"] == 1
        assert summary_dict["counts"]["hold"] == 1
        assert summary_dict["counts"]["reduce"] == 1

    def test_channel_dict_has_required_fields(self, generator):
        """Channel dicts have all required fields."""
        summary_dict = generator.get_summary_dict()

        required_fields = [
            "channel", "channel_name", "current_spend", "current_roi",
            "marginal_roi", "headroom", "priority_rank", "needs_test"
        ]

        for ch_dict in summary_dict["priority_table"]:
            for field in required_fields:
                assert field in ch_dict


class TestChannelCategorization:
    """Tests for channel categorization logic."""

    def test_increase_channels_have_high_marginal_roi(self, sample_priority_result):
        """Increase channels have marginal ROI > 1.50."""
        for ch in sample_priority_result.increase_channels:
            assert ch.marginal_roi > 1.50

    def test_hold_channels_have_medium_marginal_roi(self, sample_priority_result):
        """Hold channels have marginal ROI between 1.00 and 1.50."""
        for ch in sample_priority_result.hold_channels:
            assert 1.00 <= ch.marginal_roi <= 1.50

    def test_reduce_channels_have_low_marginal_roi(self, sample_priority_result):
        """Reduce channels have marginal ROI < 1.00."""
        for ch in sample_priority_result.reduce_channels:
            assert ch.marginal_roi < 1.00


class TestEmptyChannelLists:
    """Tests for edge cases with empty channel lists."""

    def test_no_increase_channels(self, mock_marginal_analyzer, sample_channel_reduce):
        """Handles case with no increase channels."""
        priority_result = InvestmentPriorityResult(
            channel_analysis=[sample_channel_reduce],
            increase_channels=[],
            hold_channels=[],
            reduce_channels=[sample_channel_reduce],
            channels_needing_test=[],
            total_spend=80000.0,
            total_contribution=96000.0,
            portfolio_roi=1.2,
            reallocation_potential=20000.0,
            headroom_available=0.0,
        )

        gen = ExecutiveSummaryGenerator(
            marginal_analyzer=mock_marginal_analyzer,
            priority_result=priority_result,
        )

        result = gen.generate_summary()
        # Should not raise, should handle gracefully

    def test_no_reduce_channels(self, mock_marginal_analyzer, sample_channel_increase):
        """Handles case with no reduce channels."""
        priority_result = InvestmentPriorityResult(
            channel_analysis=[sample_channel_increase],
            increase_channels=[sample_channel_increase],
            hold_channels=[],
            reduce_channels=[],
            channels_needing_test=[],
            total_spend=50000.0,
            total_contribution=175000.0,
            portfolio_roi=3.5,
            reallocation_potential=0.0,
            headroom_available=50000.0,
        )

        gen = ExecutiveSummaryGenerator(
            marginal_analyzer=mock_marginal_analyzer,
            priority_result=priority_result,
        )

        result = gen.generate_summary()
        assert "efficient" in result.summary_text.lower() or "No channels" in result.summary_text
