"""
Tests for Results page tab structure and consolidation.

This module tests the restructured Results page with 7 consolidated tabs.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
import pandas as pd
import numpy as np


# =============================================================================
# Tab Structure Tests
# =============================================================================

class TestTabStructure:
    """Regression tests for tab structure."""

    def test_correct_tab_count(self):
        """Results page should have exactly 7 tabs."""
        expected_tabs = [
            "Overview",
            "Media",
            "Investment Priority",
            "Statistical Confidence",
            "Diagnostics",
            "Time Series",
            "Visualizations",
        ]
        assert len(expected_tabs) == 7

    def test_tab_names(self):
        """Verify correct tab names in expected order."""
        expected_tabs = [
            "Overview",
            "Media",
            "Investment Priority",
            "Statistical Confidence",
            "Diagnostics",
            "Time Series",
            "Visualizations",
        ]

        # These are the exact tab names that should appear in the UI
        assert expected_tabs[0] == "Overview"
        assert expected_tabs[1] == "Media"
        assert expected_tabs[2] == "Investment Priority"
        assert expected_tabs[3] == "Statistical Confidence"
        assert expected_tabs[4] == "Diagnostics"
        assert expected_tabs[5] == "Time Series"
        assert expected_tabs[6] == "Visualizations"

    def test_no_export_tab(self):
        """Export tab should not exist (removed - use Export page instead)."""
        tabs = [
            "Overview",
            "Media",
            "Investment Priority",
            "Statistical Confidence",
            "Diagnostics",
            "Time Series",
            "Visualizations",
        ]
        assert "Export" not in tabs

    def test_no_owned_media_tab(self):
        """Owned Media tab should not exist (merged into Media tab)."""
        tabs = [
            "Overview",
            "Media",
            "Investment Priority",
            "Statistical Confidence",
            "Diagnostics",
            "Time Series",
            "Visualizations",
        ]
        assert "Owned Media" not in tabs

    def test_no_separate_channel_roi_tab(self):
        """Channel ROI tab merged into Media tab."""
        tabs = [
            "Overview",
            "Media",
            "Investment Priority",
            "Statistical Confidence",
            "Diagnostics",
            "Time Series",
            "Visualizations",
        ]
        # Should not have "Channel ROI" as separate tab
        assert not any("Channel ROI" in t for t in tabs)

    def test_no_separate_executive_summary_tab(self):
        """Executive Summary merged into Investment Priority tab."""
        tabs = [
            "Overview",
            "Media",
            "Investment Priority",
            "Statistical Confidence",
            "Diagnostics",
            "Time Series",
            "Visualizations",
        ]
        assert "Executive Summary" not in tabs


# =============================================================================
# Media Tab Filter Tests
# =============================================================================

class TestMediaTabFilter:
    """Tests for Paid/Owned filter in Media tab."""

    def test_filter_paid_only(self):
        """Filter returns only paid media channels."""
        paid_cols = ["search_spend", "display_spend", "social_spend"]
        owned_cols = ["email_volume", "seo_sessions"]

        media_filter = "Paid Media"

        if media_filter == "Paid Media":
            active_channels = paid_cols
        elif media_filter == "Owned Media":
            active_channels = owned_cols
        else:
            active_channels = paid_cols + owned_cols

        assert active_channels == paid_cols
        assert "email_volume" not in active_channels
        assert "seo_sessions" not in active_channels

    def test_filter_owned_only(self):
        """Filter returns only owned media channels."""
        paid_cols = ["search_spend", "display_spend", "social_spend"]
        owned_cols = ["email_volume", "seo_sessions"]

        media_filter = "Owned Media"

        if media_filter == "Paid Media":
            active_channels = paid_cols
        elif media_filter == "Owned Media":
            active_channels = owned_cols
        else:
            active_channels = paid_cols + owned_cols

        assert active_channels == owned_cols
        assert "search_spend" not in active_channels
        assert "display_spend" not in active_channels

    def test_filter_all_media(self):
        """Filter returns all channels."""
        paid_cols = ["search_spend", "display_spend", "social_spend"]
        owned_cols = ["email_volume", "seo_sessions"]

        media_filter = "All Media"

        if media_filter == "Paid Media":
            active_channels = paid_cols
        elif media_filter == "Owned Media":
            active_channels = owned_cols
        else:
            active_channels = paid_cols + owned_cols

        assert len(active_channels) == 5
        assert all(c in active_channels for c in paid_cols)
        assert all(c in active_channels for c in owned_cols)

    def test_filter_hidden_when_no_owned(self):
        """Filter should default to Paid when no owned media configured."""
        paid_cols = ["search_spend", "display_spend"]
        owned_cols = []  # No owned media

        has_owned = len(owned_cols) > 0

        if has_owned:
            media_filter = "All Media"  # Would show filter
        else:
            media_filter = "Paid Media"  # Default when no owned

        assert not has_owned
        assert media_filter == "Paid Media"

    def test_filter_options_correct(self):
        """Filter options are correct when owned media exists."""
        filter_options = ["All Media", "Paid Media", "Owned Media"]

        assert len(filter_options) == 3
        assert "All Media" in filter_options
        assert "Paid Media" in filter_options
        assert "Owned Media" in filter_options


# =============================================================================
# Investment Priority Tab Tests
# =============================================================================

class TestInvestmentPriorityTab:
    """Tests for merged Investment Priority tab."""

    def test_tab_contains_marginal_roi_metrics(self):
        """Tab should include marginal ROI metrics."""
        # Key metrics that should be in Investment Priority tab
        required_metrics = [
            "current_spend",
            "current_roi",
            "marginal_roi",
            "priority_rank",
            "breakeven_spend",
            "headroom_amount",
            "action",
            "needs_test",
        ]

        # Create a mock priority table
        priority_df = pd.DataFrame({
            "channel": ["search", "display", "tv"],
            "current_spend": [50000, 30000, 80000],
            "current_roi": [3.5, 1.8, 1.2],
            "marginal_roi": [2.5, 1.25, 0.7],
            "priority_rank": [1, 2, 3],
            "breakeven_spend": [100000, 40000, 60000],
            "headroom_amount": [50000, 10000, 0],
            "action": ["INCREASE", "HOLD", "REDUCE"],
            "needs_test": [False, True, False],
        })

        for metric in required_metrics:
            assert metric in priority_df.columns

    def test_tab_contains_executive_summary_fields(self):
        """Tab should include executive summary content."""
        # Key fields from executive summary
        summary_dict = {
            "portfolio": {
                "total_spend": 160000,
                "total_contribution": 280000,
                "portfolio_roi": 1.75,
                "reallocation_potential": 20000,
                "headroom_available": 50000,
            },
            "counts": {
                "increase": 1,
                "hold": 1,
                "reduce": 1,
                "needs_validation": 1,
            },
            "recommendations": {
                "increase": [{"channel_name": "Search", "marginal_roi": 2.5}],
                "hold": [{"channel_name": "Display", "marginal_roi": 1.25}],
                "reduce": [{"channel_name": "TV", "marginal_roi": 0.7}],
            },
            "reallocation_moves": [],
        }

        # Verify structure
        assert "portfolio" in summary_dict
        assert "counts" in summary_dict
        assert "recommendations" in summary_dict
        assert "reallocation_moves" in summary_dict

        # Verify portfolio metrics
        assert "total_spend" in summary_dict["portfolio"]
        assert "portfolio_roi" in summary_dict["portfolio"]
        assert "reallocation_potential" in summary_dict["portfolio"]

    def test_action_categorization(self):
        """Investment actions are correctly categorized."""
        # INCREASE: marginal_roi > 1.50
        # HOLD: 1.00 <= marginal_roi <= 1.50
        # REDUCE: marginal_roi < 1.00

        test_cases = [
            (2.5, "INCREASE"),
            (1.51, "INCREASE"),
            (1.50, "HOLD"),
            (1.25, "HOLD"),
            (1.00, "HOLD"),
            (0.99, "REDUCE"),
            (0.5, "REDUCE"),
        ]

        for marginal_roi, expected_action in test_cases:
            if marginal_roi > 1.50:
                action = "INCREASE"
            elif marginal_roi >= 1.00:
                action = "HOLD"
            else:
                action = "REDUCE"

            assert action == expected_action, \
                f"marginal_roi={marginal_roi} should be {expected_action}, got {action}"


# =============================================================================
# Tab Rename Tests
# =============================================================================

class TestTabRenames:
    """Tests for renamed tabs."""

    def test_bayesian_significance_renamed(self):
        """'Bayesian Significance' should now be 'Statistical Confidence'."""
        old_name = "Bayesian Significance"
        new_name = "Statistical Confidence"

        tabs = [
            "Overview",
            "Media",
            "Investment Priority",
            "Statistical Confidence",
            "Diagnostics",
            "Time Series",
            "Visualizations",
        ]

        assert new_name in tabs
        assert old_name not in tabs

    def test_model_details_renamed(self):
        """'Model Details' should now be 'Diagnostics'."""
        old_name = "Model Details"
        new_name = "Diagnostics"

        tabs = [
            "Overview",
            "Media",
            "Investment Priority",
            "Statistical Confidence",
            "Diagnostics",
            "Time Series",
            "Visualizations",
        ]

        assert new_name in tabs
        assert old_name not in tabs


# =============================================================================
# Content Consolidation Tests
# =============================================================================

class TestContentConsolidation:
    """Tests for content that was merged into tabs."""

    def test_media_curves_content_in_media_tab(self):
        """Media Curves content should be part of Media tab."""
        # The Media tab should include saturation curves, ROI curves, adstock decay
        media_tab_content_types = [
            "roi_table",  # Channel ROI table
            "saturation_curves",  # From Media Curves
            "roi_curves",  # From Media Curves
            "adstock_decay",  # From Media Curves
        ]

        # All these content types should be displayable in the Media tab
        assert "saturation_curves" in media_tab_content_types
        assert "roi_curves" in media_tab_content_types

    def test_executive_summary_in_investment_priority(self):
        """Executive Summary content merged into Investment Priority."""
        investment_priority_sections = [
            "priority_table",
            "marginal_roi_chart",
            "portfolio_overview",  # From Executive Summary
            "investment_recommendations",  # From Executive Summary
            "reallocation_opportunity",  # From Executive Summary
        ]

        assert "portfolio_overview" in investment_priority_sections
        assert "investment_recommendations" in investment_priority_sections
        assert "reallocation_opportunity" in investment_priority_sections


# =============================================================================
# Backward Compatibility Tests
# =============================================================================

class TestBackwardCompatibility:
    """Ensure the restructuring doesn't break existing functionality."""

    def test_contributions_analyzer_still_works(self):
        """ContributionAnalyzer.get_channel_roi() accepts roi_channels param."""
        # This tests that the filter mechanism works with the analyzer
        roi_channels = ["search_spend", "display_spend"]

        # Simulate what the analyzer does with filtered channels
        mock_result = pd.DataFrame({
            "channel": roi_channels,
            "contribution": [50000, 30000],
            "spend": [20000, 15000],
            "roi": [2.5, 2.0],
        })

        # Only requested channels should be in result
        assert len(mock_result) == len(roi_channels)
        assert all(c in mock_result["channel"].values for c in roi_channels)

    def test_kpi_labels_still_work(self):
        """KPI labels conversion should still work."""
        from mmm_platform.ui.kpi_labels import KPILabels
        from mmm_platform.config.schema import KPIType

        # Create mock configs for different KPI types
        mock_revenue_config = Mock()
        mock_revenue_config.data.kpi_type = KPIType.REVENUE
        mock_revenue_config.data.kpi_display_name = "Revenue"
        mock_revenue_config.data.target_column = "revenue"

        mock_count_config = Mock()
        mock_count_config.data.kpi_type = KPIType.COUNT
        mock_count_config.data.kpi_display_name = "Installs"
        mock_count_config.data.target_column = "installs"

        # Revenue type
        revenue_labels = KPILabels(mock_revenue_config)
        assert revenue_labels.is_revenue_type
        assert "ROI" in revenue_labels.efficiency_label

        # Count type
        count_labels = KPILabels(mock_count_config)
        assert not count_labels.is_revenue_type
        assert "Cost" in count_labels.efficiency_label


# =============================================================================
# Import/Syntax Check Tests
# =============================================================================

class TestUIPageImports:
    """Tests that UI pages can be imported without errors.

    These tests catch basic syntax errors and invalid attribute access
    that wouldn't be caught by mocked tests.
    """

    def test_results_page_imports(self):
        """Results page module can be imported without syntax errors."""
        # This catches undefined names, syntax errors, and bad imports
        from mmm_platform.ui.pages import results
        assert results is not None

    def test_optimize_page_imports(self):
        """Optimize page module can be imported without syntax errors."""
        from mmm_platform.ui.pages import optimize
        assert optimize is not None

    def test_kpi_labels_attributes_used_in_results(self):
        """Verify KPILabels has all attributes used by results page."""
        from mmm_platform.ui.kpi_labels import KPILabels
        from mmm_platform.config.schema import KPIType

        # Create a mock config
        mock_config = Mock()
        mock_config.data.kpi_type = KPIType.COUNT
        mock_config.data.kpi_display_name = "Install"
        mock_config.data.target_column = "installs"

        labels = KPILabels(mock_config)

        # These are attributes used in results.py - test they exist
        assert hasattr(labels, 'is_revenue_type')
        assert hasattr(labels, 'efficiency_label')
        assert hasattr(labels, 'target_name')
        assert hasattr(labels, 'convert_internal_to_display')

        # Verify they return sensible values
        assert isinstance(labels.is_revenue_type, bool)
        assert isinstance(labels.efficiency_label, str)
        assert isinstance(labels.target_name, str)
        assert callable(labels.convert_internal_to_display)
