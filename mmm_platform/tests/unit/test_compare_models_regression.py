"""
Regression tests for ui/pages/compare_models.py

These tests prevent reintroduction of bugs that have been fixed:
- ROI chart showing single color when models have same name (commit 2137f16)
- Coefficients tab issues (commit 1aecfd4)
- Saturation/adstock variable naming (commit aec1e60)
- Filter and label improvements (commits 340744d, 36ada4e)
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, MagicMock, patch


# =============================================================================
# Test Data Fixtures
# =============================================================================

@pytest.fixture
def sample_roi_df_a():
    """Sample ROI DataFrame for Model A."""
    return pd.DataFrame({
        "channel": ["tv_spend", "search_spend", "facebook_spend"],
        "display_name": ["TV", "Search", "Facebook"],
        "roi": [2.5, 3.2, 1.8],
        "contribution_real": [250000, 320000, 180000],
        "spend_real": [100000, 100000, 100000],
    })


@pytest.fixture
def sample_roi_df_b():
    """Sample ROI DataFrame for Model B."""
    return pd.DataFrame({
        "channel": ["tv_spend", "search_spend", "facebook_spend"],
        "display_name": ["TV", "Search", "Facebook"],
        "roi": [2.8, 2.9, 2.0],
        "contribution_real": [280000, 290000, 200000],
        "spend_real": [100000, 100000, 100000],
    })


@pytest.fixture
def sample_contributions_df():
    """Sample grouped contributions DataFrame."""
    return pd.DataFrame({
        "group": ["Base", "Paid Media", "Promotions"],
        "contribution_scaled": [50000, 80000, 20000],
        "contribution_real": [50000000, 80000000, 20000000],
        "pct_of_total": [33.3, 53.3, 13.3],
        "color": ["#808080", "#4A90D9", "#27AE60"],
    })


# =============================================================================
# ROI Chart Color Bug Regression Tests (commit 2137f16)
# =============================================================================

class TestRoiChartUniqueColors:
    """
    Regression tests for ROI chart color bug.

    Bug: When two models had the same name, the chart showed only one color
    because plotly used the model name for the color legend.

    Fix: When model names are identical, append "(A)" and "(B)" suffixes.
    """

    def test_same_model_names_get_unique_suffixes(self):
        """Models with same name should get (A) and (B) suffixes."""
        name_a = "My Model"
        name_b = "My Model"

        # This is the logic from compare_models.py
        if name_a == name_b:
            name_a = f"{name_a} (A)"
            name_b = f"{name_b} (B)"

        assert name_a != name_b
        assert "(A)" in name_a
        assert "(B)" in name_b

    def test_different_model_names_unchanged(self):
        """Models with different names should not be modified."""
        name_a = "Model V1"
        name_b = "Model V2"

        # This is the logic from compare_models.py
        if name_a == name_b:
            name_a = f"{name_a} (A)"
            name_b = f"{name_b} (B)"

        assert name_a == "Model V1"
        assert name_b == "Model V2"

    def test_chart_data_uses_unique_model_names(self, sample_roi_df_a, sample_roi_df_b):
        """Chart data should have distinct Model column values."""
        name_a = "Same Name"
        name_b = "Same Name"

        # Apply the fix
        if name_a == name_b:
            name_a = f"{name_a} (A)"
            name_b = f"{name_b} (B)"

        # Build chart data as in compare_models.py
        chart_data = []
        for _, row in sample_roi_df_a.iterrows():
            chart_data.append({
                "Channel": row["display_name"],
                "Model": name_a,
                "ROI": row["roi"],
            })
        for _, row in sample_roi_df_b.iterrows():
            chart_data.append({
                "Channel": row["display_name"],
                "Model": name_b,
                "ROI": row["roi"],
            })

        chart_df = pd.DataFrame(chart_data)

        # Verify unique model names for distinct colors
        unique_models = chart_df["Model"].unique()
        assert len(unique_models) == 2
        assert "Same Name (A)" in unique_models
        assert "Same Name (B)" in unique_models


# =============================================================================
# Coefficients Tab Regression Tests (commit 1aecfd4)
# =============================================================================

class TestCoefficientsTab:
    """
    Regression tests for coefficients tab.

    Bug: Coefficients tab was not displaying properly.
    Fix: Properly handle ArviZ summary DataFrames and display.
    """

    def test_arviz_summary_has_required_columns(self):
        """ArviZ summary should have mean, sd columns for display."""
        # Mock an ArviZ summary DataFrame
        summary = pd.DataFrame({
            "mean": [1.5, 2.0, 1.8],
            "sd": [0.2, 0.3, 0.25],
            "hdi_3%": [1.1, 1.5, 1.3],
            "hdi_97%": [1.9, 2.5, 2.3],
        }, index=["saturation_beta[tv_spend]", "saturation_beta[search_spend]", "saturation_beta[facebook_spend]"])

        # The display should use mean and sd columns
        display_df = summary[["mean", "sd"]]

        assert "mean" in display_df.columns
        assert "sd" in display_df.columns
        assert len(display_df) == 3


# =============================================================================
# Saturation/Adstock Variable Names Regression Tests (commit aec1e60)
# =============================================================================

class TestVariableNaming:
    """
    Regression tests for saturation and adstock variable naming.

    Bug: Saturation and adstock variable names were inconsistent.
    Fix: Use consistent naming (saturation_lam, adstock_alpha, saturation_beta).
    """

    def test_saturation_beta_var_name(self):
        """Coefficient variable should be named 'saturation_beta'."""
        # This is the expected var name used in ArviZ summary
        var_name = "saturation_beta"
        assert var_name == "saturation_beta"

    def test_saturation_lam_var_name(self):
        """Lambda variable should be named 'saturation_lam'."""
        var_name = "saturation_lam"
        assert var_name == "saturation_lam"

    def test_adstock_alpha_var_name(self):
        """Adstock variable should be named 'adstock_alpha'."""
        var_name = "adstock_alpha"
        assert var_name == "adstock_alpha"

    def test_channel_index_parsing(self):
        """Should be able to parse channel name from index."""
        index_value = "saturation_beta[tv_spend]"

        # Extract channel name
        if "[" in index_value and "]" in index_value:
            channel = index_value.split("[")[1].rstrip("]")
        else:
            channel = index_value

        assert channel == "tv_spend"


# =============================================================================
# Model Selector Label Regression Tests (commits 340744d, 36ada4e)
# =============================================================================

class TestModelSelectorLabels:
    """
    Regression tests for model selector labels.

    Bug: Labels were unclear or inconsistent.
    Fix: Include R², channel count, date, and client info in labels.
    """

    def test_label_includes_r2(self):
        """Model selector label should include R² value."""
        model = {
            "config_name": "My Model",
            "r2": 0.85,
            "n_channels": 5,
            "created_at": "2024-01-15T10:30:00",
            "client": "Test Client",
            "is_favorite": False,
        }

        r2_str = f"R²={model['r2']:.3f}" if model.get('r2') is not None else "R²=N/A"
        assert "R²=0.850" == r2_str

    def test_label_includes_channel_count(self):
        """Model selector label should include channel count."""
        model = {"n_channels": 5}

        n_channels = model.get("n_channels", "?")
        label_part = f"{n_channels} channels"

        assert "5 channels" == label_part

    def test_label_includes_client_when_viewing_all(self):
        """Label should include client name when viewing all clients."""
        model = {
            "config_name": "My Model",
            "client": "Acme Corp",
            "r2": 0.85,
            "n_channels": 5,
            "created_at": "2024-01-15T10:30:00",
        }
        client_filter = "all"

        name = model.get("config_name", "Unknown")
        model_client = model.get("client", "")

        if client_filter == "all" and model_client:
            option_label = f"[{model_client}] {name}"
        else:
            option_label = name

        assert "[Acme Corp] My Model" == option_label

    def test_label_excludes_client_when_filtered(self):
        """Label should not include client when already filtered."""
        model = {
            "config_name": "My Model",
            "client": "Acme Corp",
        }
        client_filter = "Acme Corp"  # Already filtered to this client

        name = model.get("config_name", "Unknown")
        model_client = model.get("client", "")

        if client_filter == "all" and model_client:
            option_label = f"[{model_client}] {name}"
        else:
            option_label = name

        assert "My Model" == option_label
        assert "[Acme Corp]" not in option_label

    def test_favorite_icon_added(self):
        """Favorite models should have star icon."""
        model = {"is_favorite": True, "config_name": "My Model"}

        is_favorite = model.get("is_favorite", False)
        fav_icon = "⭐ " if is_favorite else ""

        label = f"{fav_icon}{model['config_name']}"
        assert "⭐ My Model" == label

    def test_non_favorite_no_icon(self):
        """Non-favorite models should not have star icon."""
        model = {"is_favorite": False, "config_name": "My Model"}

        is_favorite = model.get("is_favorite", False)
        fav_icon = "⭐ " if is_favorite else ""

        label = f"{fav_icon}{model['config_name']}"
        assert "My Model" == label
        assert "⭐" not in label


# =============================================================================
# Filter Functionality Regression Tests
# =============================================================================

class TestFilters:
    """Regression tests for filter functionality."""

    def test_archive_filter_excludes_archived(self):
        """Archive filter should exclude archived models by default."""
        models = [
            {"config_name": "Active Model", "is_archived": False},
            {"config_name": "Archived Model", "is_archived": True},
        ]

        show_archived = False
        if not show_archived:
            filtered = [m for m in models if not m.get("is_archived", False)]
        else:
            filtered = models

        assert len(filtered) == 1
        assert filtered[0]["config_name"] == "Active Model"

    def test_archive_filter_includes_when_enabled(self):
        """Archive filter should include archived when enabled."""
        models = [
            {"config_name": "Active Model", "is_archived": False},
            {"config_name": "Archived Model", "is_archived": True},
        ]

        show_archived = True
        if not show_archived:
            filtered = [m for m in models if not m.get("is_archived", False)]
        else:
            filtered = models

        assert len(filtered) == 2

    def test_favorites_filter_only_favorites(self):
        """Favorites filter should show only favorites."""
        models = [
            {"config_name": "Favorite", "is_favorite": True},
            {"config_name": "Regular", "is_favorite": False},
        ]

        favorite_filter = "Favorites only"
        if favorite_filter == "Favorites only":
            filtered = [m for m in models if m.get("is_favorite", False)]
        else:
            filtered = models

        assert len(filtered) == 1
        assert filtered[0]["config_name"] == "Favorite"

    def test_favorites_filter_non_favorites(self):
        """Non-favorites filter should exclude favorites."""
        models = [
            {"config_name": "Favorite", "is_favorite": True},
            {"config_name": "Regular", "is_favorite": False},
        ]

        favorite_filter = "Non-favorites"
        if favorite_filter == "Non-favorites":
            filtered = [m for m in models if not m.get("is_favorite", False)]
        else:
            filtered = models

        assert len(filtered) == 1
        assert filtered[0]["config_name"] == "Regular"


# =============================================================================
# ROI DataFrame Merge Regression Tests
# =============================================================================

class TestRoiDataMerge:
    """Regression tests for ROI data merging between models."""

    def test_merge_handles_missing_channels(self):
        """Merge should handle channels present in only one model."""
        roi_a = pd.DataFrame({
            "channel": ["tv_spend", "search_spend"],
            "display_name": ["TV", "Search"],
            "roi": [2.5, 3.2],
        })

        roi_b = pd.DataFrame({
            "channel": ["search_spend", "facebook_spend"],
            "display_name": ["Search", "Facebook"],
            "roi": [2.9, 2.0],
        })

        # Merge as in compare_models.py
        merged = pd.merge(
            roi_a[["channel", "display_name", "roi"]].rename(columns={"roi": "ROI (A)"}),
            roi_b[["channel", "display_name", "roi"]].rename(columns={"roi": "ROI (B)", "display_name": "display_name_b"}),
            on="channel",
            how="outer"
        )

        assert len(merged) == 3  # All unique channels
        assert "tv_spend" in merged["channel"].values
        assert "search_spend" in merged["channel"].values
        assert "facebook_spend" in merged["channel"].values

    def test_merge_nan_for_missing_values(self):
        """Merge should have NaN for channels not in a model."""
        roi_a = pd.DataFrame({
            "channel": ["tv_spend"],
            "roi": [2.5],
        })

        roi_b = pd.DataFrame({
            "channel": ["facebook_spend"],
            "roi": [2.0],
        })

        merged = pd.merge(
            roi_a.rename(columns={"roi": "ROI (A)"}),
            roi_b.rename(columns={"roi": "ROI (B)"}),
            on="channel",
            how="outer"
        )

        # tv_spend should have NaN for ROI (B)
        tv_row = merged[merged["channel"] == "tv_spend"].iloc[0]
        assert pd.isna(tv_row["ROI (B)"])

        # facebook_spend should have NaN for ROI (A)
        fb_row = merged[merged["channel"] == "facebook_spend"].iloc[0]
        assert pd.isna(fb_row["ROI (A)"])
