"""
Export functions for generating CSV files in platform-specific formats.

These formats are designed for upload to external visualization platforms.
"""

import pandas as pd
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mmm_platform.model.mmm import MMMWrapper
    from mmm_platform.config.schema import ModelConfig


def generate_decomps_stacked(
    wrapper: "MMMWrapper",
    config: "ModelConfig",
    brand: str
) -> pd.DataFrame:
    """
    Generate stacked decomposition export with all components.

    Creates a long-format DataFrame with one row per date per decomposition component.
    Includes channels, controls, and base components (intercept, trend, seasonality).

    Parameters
    ----------
    wrapper : MMMWrapper
        Fitted model wrapper
    config : ModelConfig
        Model configuration
    brand : str
        Brand name for the export

    Returns
    -------
    pd.DataFrame
        Stacked decomposition data with columns:
        date, wc_mon, brand, decomp, decomp_lvl1, decomp_lvl2, ..., kpi_{target}
    """
    contribs = wrapper.get_contributions()
    date_col = config.data.date_column
    target_col = config.data.target_column
    revenue_scale = config.data.revenue_scale

    # Get category column names
    cat_col_names = config.get_category_column_names()

    rows = []
    for date_val in contribs.index:
        for col in contribs.columns:
            # Determine if channel, control, or base component
            channel_cfg = config.get_channel_by_name(col)
            control_cfg = config.get_control_by_name(col)

            # Get display name and categories
            if channel_cfg:
                display_name = channel_cfg.get_display_name()
                categories = [channel_cfg.get_category(cat_name) for cat_name in cat_col_names]
            elif control_cfg:
                display_name = control_cfg.get_display_name()
                categories = [control_cfg.get_category(cat_name) for cat_name in cat_col_names]
            else:
                # Base component (intercept, trend, seasonality, etc.)
                display_name = col.replace("_", " ").title()
                # Classify base components
                if col in ['intercept', 'trend']:
                    base_category = "Base"
                elif 'season' in col.lower() or 'fourier' in col.lower():
                    base_category = "Seasonality"
                else:
                    base_category = "Other"
                categories = [base_category] * len(cat_col_names)

            # If no category columns defined, use display_name as default
            if not categories:
                categories = [display_name, display_name]

            # Build row
            row = {
                'date': date_val,
                'wc_mon': date_val,
                'brand': brand,
                'decomp': col,
            }

            # Add category levels (decomp_lvl1, decomp_lvl2, etc.)
            for i, cat_val in enumerate(categories):
                row[f'decomp_lvl{i + 1}'] = cat_val if cat_val else display_name

            # If fewer than 2 category columns, ensure at least lvl1 and lvl2
            if len(categories) < 1:
                row['decomp_lvl1'] = display_name
            if len(categories) < 2:
                row['decomp_lvl2'] = row.get('decomp_lvl1', display_name)

            # Add KPI value
            row[f'kpi_{target_col}'] = contribs.loc[date_val, col] * revenue_scale

            rows.append(row)

    return pd.DataFrame(rows)


def generate_media_results(
    wrapper: "MMMWrapper",
    config: "ModelConfig",
    brand: str
) -> pd.DataFrame:
    """
    Generate media results export with spend data (channels only).

    Creates a long-format DataFrame with one row per date per media channel.
    Includes spend data and contribution values. Impressions/clicks are placeholders.

    Parameters
    ----------
    wrapper : MMMWrapper
        Fitted model wrapper
    config : ModelConfig
        Model configuration
    brand : str
        Brand name for the export

    Returns
    -------
    pd.DataFrame
        Media results data with columns:
        date, wc_mon, brand, decomp_lvl1, decomp_lvl2, spend, impressions, clicks, decomp, kpi_{target}
    """
    contribs = wrapper.get_contributions()
    date_col = config.data.date_column
    target_col = config.data.target_column
    revenue_scale = config.data.revenue_scale
    spend_scale = config.data.spend_scale

    # Get category column names
    cat_col_names = config.get_category_column_names()

    # Get date values from the original data
    df_with_dates = wrapper.df_scaled.copy()
    if date_col in df_with_dates.columns:
        dates_list = df_with_dates[date_col].tolist()
    else:
        dates_list = contribs.index.tolist()

    rows = []
    for date_idx, date_val in enumerate(contribs.index):
        for channel_cfg in config.channels:
            col = channel_cfg.name

            # Skip if channel not in contributions (shouldn't happen but be safe)
            if col not in contribs.columns:
                continue

            # Get spend from scaled data
            if date_idx < len(wrapper.df_scaled):
                spend = wrapper.df_scaled.iloc[date_idx][col] * spend_scale
            else:
                spend = 0

            # Get display name and categories
            display_name = channel_cfg.get_display_name()
            categories = [channel_cfg.get_category(cat_name) for cat_name in cat_col_names]

            # If no category columns defined, use display_name
            if not categories:
                categories = [display_name, display_name]

            # Build row
            row = {
                'date': date_val,
                'wc_mon': date_val,
                'brand': brand,
            }

            # Add category levels
            for i, cat_val in enumerate(categories):
                row[f'decomp_lvl{i + 1}'] = cat_val if cat_val else display_name

            # If fewer than 2 category columns, ensure at least lvl1 and lvl2
            if len(categories) < 1:
                row['decomp_lvl1'] = display_name
            if len(categories) < 2:
                row['decomp_lvl2'] = row.get('decomp_lvl1', display_name)

            # Add spend and placeholder metrics
            row['spend'] = spend
            row['impressions'] = 0  # Placeholder - to be added later
            row['clicks'] = 0  # Placeholder - to be added later
            row['decomp'] = col

            # Add KPI value
            row[f'kpi_{target_col}'] = contribs.loc[date_val, col] * revenue_scale

            rows.append(row)

    return pd.DataFrame(rows)


def generate_actual_vs_fitted(
    wrapper: "MMMWrapper",
    config: "ModelConfig",
    brand: str
) -> pd.DataFrame:
    """
    Generate actual vs fitted export in long format.

    Creates a long-format DataFrame with two rows per date (Actual and Fitted).

    Parameters
    ----------
    wrapper : MMMWrapper
        Fitted model wrapper
    config : ModelConfig
        Model configuration
    brand : str
        Brand name for the export

    Returns
    -------
    pd.DataFrame
        Actual vs fitted data with columns:
        date, wc_mon, brand, kpi_label, actual_fitted, value
    """
    contribs = wrapper.get_contributions()
    date_col = config.data.date_column
    target_col = config.data.target_column
    revenue_scale = config.data.revenue_scale

    # Get actual values from scaled data
    df_indexed = wrapper.df_scaled.set_index(date_col)
    actual = df_indexed[target_col].reindex(contribs.index) * revenue_scale

    # Calculate fitted as sum of all contribution components
    fitted = contribs.sum(axis=1) * revenue_scale

    rows = []
    for date_val in contribs.index:
        # Actual row
        rows.append({
            'date': date_val,
            'wc_mon': date_val,
            'brand': brand,
            'kpi_label': f'kpi_{target_col}',
            'actual_fitted': 'Actual',
            'value': actual.loc[date_val] if date_val in actual.index else np.nan
        })

        # Fitted row
        rows.append({
            'date': date_val,
            'wc_mon': date_val,
            'brand': brand,
            'kpi_label': f'kpi_{target_col}',
            'actual_fitted': 'Fitted',
            'value': fitted.loc[date_val]
        })

    return pd.DataFrame(rows)
