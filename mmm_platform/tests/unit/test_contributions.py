"""
Tests for analysis/contributions.py - Contribution analysis and decomposition.
"""

import pytest
import pandas as pd
import numpy as np

from mmm_platform.analysis.contributions import ContributionAnalyzer, CATEGORY_COLORS


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_contributions():
    """Sample contribution DataFrame mimicking model output."""
    np.random.seed(42)
    n_rows = 50

    return pd.DataFrame({
        "intercept": np.random.uniform(30000, 50000, n_rows),
        "tv_spend": np.random.uniform(5000, 15000, n_rows),
        "search_spend": np.random.uniform(3000, 10000, n_rows),
        "promo_flag": np.random.uniform(1000, 5000, n_rows),
        "revenue": np.random.uniform(50000, 100000, n_rows),  # Target
    })


@pytest.fixture
def sample_df_scaled():
    """Sample scaled DataFrame used for modeling."""
    np.random.seed(42)
    n_rows = 50

    return pd.DataFrame({
        "date": pd.date_range("2022-01-01", periods=n_rows, freq="W"),
        "revenue": np.random.uniform(50000, 100000, n_rows),
        "tv_spend": np.random.uniform(5000, 20000, n_rows),
        "search_spend": np.random.uniform(3000, 15000, n_rows),
        "promo_flag": np.random.choice([0, 1], n_rows),
    })


@pytest.fixture
def analyzer(sample_contributions, sample_df_scaled):
    """Initialized ContributionAnalyzer for testing."""
    return ContributionAnalyzer(
        contribs=sample_contributions,
        df_scaled=sample_df_scaled,
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
# Display Name Tests
# =============================================================================

class TestGetDisplayName:
    """Tests for get_display_name method."""

    def test_with_mapping_returns_mapped_name(self, analyzer):
        """Returns mapped display name when available."""
        assert analyzer.get_display_name("tv_spend") == "TV Advertising"
        assert analyzer.get_display_name("search_spend") == "Paid Search"

    def test_fallback_formatting(self, analyzer):
        """Formats column name when no mapping exists."""
        # Unknown column should get fallback formatting
        # Note: get_display_name removes _spend suffix, so "facebook_spend" -> "Facebook"
        name = analyzer.get_display_name("facebook_spend")
        assert name == "Facebook"

    def test_removes_paidmedia_prefix(self):
        """Removes PaidMedia_ prefix in fallback formatting."""
        analyzer = ContributionAnalyzer(
            contribs=pd.DataFrame(),
            df_scaled=pd.DataFrame(),
            channel_cols=[],
            control_cols=[],
            target_col="revenue",
            date_col="date",
        )
        name = analyzer.get_display_name("PaidMedia_google_spend")
        assert "PaidMedia" not in name

    def test_removes_spend_suffix(self):
        """Removes _spend suffix in fallback formatting."""
        analyzer = ContributionAnalyzer(
            contribs=pd.DataFrame(),
            df_scaled=pd.DataFrame(),
            channel_cols=[],
            control_cols=[],
            target_col="revenue",
            date_col="date",
        )
        name = analyzer.get_display_name("google_ads_spend")
        assert "_spend" not in name


# =============================================================================
# Channel ROI Tests
# =============================================================================

class TestGetChannelRoi:
    """Tests for get_channel_roi method."""

    def test_basic_roi_calculation(self, analyzer):
        """Returns DataFrame with ROI for each channel."""
        roi_df = analyzer.get_channel_roi()

        assert isinstance(roi_df, pd.DataFrame)
        assert len(roi_df) == 2  # Two channels
        assert "channel" in roi_df.columns
        assert "roi" in roi_df.columns
        assert "contribution_scaled" in roi_df.columns
        assert "spend_scaled" in roi_df.columns

    def test_filtered_channels(self, analyzer):
        """Can filter to specific channels."""
        roi_df = analyzer.get_channel_roi(roi_channels=["tv_spend"])

        assert len(roi_df) == 1
        assert roi_df.iloc[0]["channel"] == "tv_spend"

    def test_missing_channel_skipped(self, analyzer):
        """Channels not in data are skipped."""
        roi_df = analyzer.get_channel_roi(roi_channels=["tv_spend", "nonexistent_channel"])

        assert len(roi_df) == 1
        assert "nonexistent_channel" not in roi_df["channel"].values

    def test_roi_positive(self, analyzer):
        """ROI values should be positive for positive contributions."""
        roi_df = analyzer.get_channel_roi()

        for _, row in roi_df.iterrows():
            assert row["roi"] > 0

    def test_real_units_scaled(self, analyzer):
        """Real units are properly scaled."""
        roi_df = analyzer.get_channel_roi()

        for _, row in roi_df.iterrows():
            # Real values should be scaled by revenue_scale/spend_scale
            assert row["contribution_real"] == row["contribution_scaled"] * analyzer.revenue_scale
            assert row["spend_real"] == row["spend_scaled"] * analyzer.spend_scale

    def test_sorted_by_roi_descending(self, analyzer):
        """Results are sorted by ROI descending."""
        roi_df = analyzer.get_channel_roi()

        if len(roi_df) > 1:
            rois = roi_df["roi"].tolist()
            assert rois == sorted(rois, reverse=True)

    def test_includes_display_name(self, analyzer):
        """Results include display names."""
        roi_df = analyzer.get_channel_roi()

        assert "display_name" in roi_df.columns
        tv_row = roi_df[roi_df["channel"] == "tv_spend"].iloc[0]
        assert tv_row["display_name"] == "TV Advertising"


# =============================================================================
# Contribution Breakdown Tests
# =============================================================================

class TestGetContributionBreakdown:
    """Tests for get_contribution_breakdown method."""

    def test_breakdown_categories(self, analyzer):
        """Returns expected category breakdown."""
        breakdown = analyzer.get_contribution_breakdown()

        assert "intercept" in breakdown
        assert "channels" in breakdown
        assert "controls" in breakdown
        assert "total" in breakdown

    def test_breakdown_percentages_sum_to_100(self, analyzer):
        """Category percentages should sum to approximately 100."""
        breakdown = analyzer.get_contribution_breakdown()

        total_pct = (
            breakdown["intercept"]["pct"] +
            breakdown["channels"]["pct"] +
            breakdown["controls"]["pct"] +
            breakdown["seasonality"]["pct"]
        )
        assert abs(total_pct - 100) < 0.1

    def test_breakdown_includes_real_values(self, analyzer):
        """Breakdown includes real (scaled) values."""
        breakdown = analyzer.get_contribution_breakdown()

        for category in ["intercept", "channels", "controls"]:
            assert "real_value" in breakdown[category]
            assert "value" in breakdown[category]


# =============================================================================
# Control Contributions Tests
# =============================================================================

class TestGetControlContributions:
    """Tests for get_control_contributions method."""

    def test_returns_dataframe(self, analyzer):
        """Returns DataFrame with control contributions."""
        ctrl_df = analyzer.get_control_contributions()

        assert isinstance(ctrl_df, pd.DataFrame)
        assert len(ctrl_df) == 1  # One control
        assert "control" in ctrl_df.columns
        assert "contribution_scaled" in ctrl_df.columns

    def test_sign_validation(self, analyzer):
        """Includes sign validation information."""
        ctrl_df = analyzer.get_control_contributions()

        assert "expected_sign" in ctrl_df.columns
        assert "actual_sign" in ctrl_df.columns
        assert "sign_valid" in ctrl_df.columns

    def test_inverted_control_expected_negative(self):
        """Controls with _inv suffix expect negative contribution."""
        contribs = pd.DataFrame({
            "competitor_inv": [-500, -600, -700],
            "revenue": [1000, 1100, 1200],
        })
        df_scaled = pd.DataFrame({
            "competitor_inv": [100, 200, 300],
        })

        analyzer = ContributionAnalyzer(
            contribs=contribs,
            df_scaled=df_scaled,
            channel_cols=[],
            control_cols=["competitor_inv"],
            target_col="revenue",
            date_col="date",
        )

        ctrl_df = analyzer.get_control_contributions()
        row = ctrl_df[ctrl_df["control"] == "competitor_inv"].iloc[0]

        assert row["expected_sign"] == "negative"
        assert row["sign_valid"] == True  # Contribution is negative as expected (use == for numpy bool)


# =============================================================================
# Grouped Contributions Tests
# =============================================================================

class TestGetGroupedContributions:
    """Tests for get_grouped_contributions method."""

    def test_with_channel_categories(self, analyzer):
        """Groups channels by provided categories."""
        channel_categories = {
            "tv_spend": "Broadcast",
            "search_spend": "Digital",
        }
        grouped = analyzer.get_grouped_contributions(channel_categories=channel_categories)

        assert isinstance(grouped, pd.DataFrame)
        groups = grouped["group"].tolist()
        assert "Broadcast" in groups or "Digital" in groups

    def test_default_fallback(self, analyzer):
        """Uses default grouping when no categories provided."""
        grouped = analyzer.get_grouped_contributions()

        groups = grouped["group"].tolist()
        # Should have at least Base and some channel group
        assert any("Base" in g or "Paid Media" in g for g in groups)

    def test_includes_color_mapping(self, analyzer):
        """Includes color mapping for visualization."""
        grouped = analyzer.get_grouped_contributions()

        assert "color" in grouped.columns
        # All colors should be valid hex codes
        for color in grouped["color"]:
            assert color.startswith("#")
            assert len(color) == 7

    def test_percentage_of_total(self, analyzer):
        """Includes percentage of total contribution."""
        grouped = analyzer.get_grouped_contributions()

        assert "pct_of_total" in grouped.columns
        # Percentages should sum to approximately 100
        total_pct = grouped["pct_of_total"].sum()
        assert 99 < total_pct < 101


# =============================================================================
# Time Series Contributions Tests
# =============================================================================

class TestGetTimeSeriesContributions:
    """Tests for get_time_series_contributions method."""

    def test_returns_scaled_contributions(self, analyzer):
        """Returns contributions in real units."""
        ts_contribs = analyzer.get_time_series_contributions()

        assert isinstance(ts_contribs, pd.DataFrame)
        # Should have same shape as original contributions
        assert len(ts_contribs) == len(analyzer.contribs)

    def test_scales_by_revenue_scale(self, analyzer):
        """Values are scaled by revenue_scale."""
        ts_contribs = analyzer.get_time_series_contributions()

        # Check a channel column
        original_sum = analyzer.contribs["tv_spend"].sum()
        scaled_sum = ts_contribs["tv_spend"].sum()

        assert abs(scaled_sum - original_sum * analyzer.revenue_scale) < 0.01


# =============================================================================
# Category Colors Tests
# =============================================================================

class TestCategoryColors:
    """Tests for CATEGORY_COLORS constant."""

    def test_common_categories_have_colors(self):
        """Common categories have assigned colors."""
        common_categories = [
            "Base", "BASELINE", "Seasonality", "PAID MEDIA", "Paid Media",
            "OWNED MEDIA", "Promotions", "Other"
        ]

        for category in common_categories:
            assert category in CATEGORY_COLORS
            assert CATEGORY_COLORS[category].startswith("#")

    def test_colors_are_valid_hex(self):
        """All colors are valid hex codes."""
        for color in CATEGORY_COLORS.values():
            assert color.startswith("#")
            assert len(color) == 7
            # Should be valid hex
            int(color[1:], 16)
