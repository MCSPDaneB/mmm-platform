"""
Export functions for generating CSV files in platform-specific formats.

These formats are designed for upload to external visualization platforms.
"""

import pandas as pd
import numpy as np
from typing import TYPE_CHECKING, List, Tuple

if TYPE_CHECKING:
    from mmm_platform.model.mmm import MMMWrapper
    from mmm_platform.config.schema import ModelConfig


def generate_decomps_stacked(
    wrapper: "MMMWrapper",
    config: "ModelConfig",
    brand: str,
    force_to_actuals: bool = False
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
    force_to_actuals : bool, optional
        If True, absorb residuals into intercept so decomposition sums to actuals

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

    # Calculate residuals per date if forcing to actuals
    residuals = {}
    if force_to_actuals:
        df_indexed = wrapper.df_scaled.set_index(date_col)
        for date_val in contribs.index:
            if date_val in df_indexed.index:
                actual = df_indexed.loc[date_val, target_col] * revenue_scale
                fitted = contribs.loc[date_val].sum() * revenue_scale
                residuals[date_val] = actual - fitted
            else:
                residuals[date_val] = 0

    rows = []
    for date_val in contribs.index:
        for col in contribs.columns:
            # Determine if channel, owned media, competitor, control, or base component
            channel_cfg = config.get_channel_by_name(col)
            owned_media_cfg = config.get_owned_media_by_name(col)
            competitor_cfg = config.get_competitor_by_name(col)
            control_cfg = config.get_control_by_name(col)

            # Get display name and categories
            if channel_cfg:
                display_name = channel_cfg.get_display_name()
                categories = [channel_cfg.get_category(cat_name) for cat_name in cat_col_names]
            elif owned_media_cfg:
                display_name = owned_media_cfg.get_display_name()
                categories = [owned_media_cfg.get_category(cat_name) for cat_name in cat_col_names]
            elif competitor_cfg:
                display_name = competitor_cfg.get_display_name()
                categories = [competitor_cfg.get_category(cat_name) for cat_name in cat_col_names]
            elif control_cfg:
                display_name = control_cfg.get_display_name()
                categories = [control_cfg.get_category(cat_name) for cat_name in cat_col_names]
            else:
                # Base component (intercept, trend, seasonality, etc.)
                # Also handle adstock-transformed columns
                base_col = col.replace("_adstock", "").replace("_inv", "")
                display_name = col.replace("_", " ").title()

                # Check if this is a transformed version of a known variable
                if config.get_competitor_by_name(base_col):
                    comp_cfg = config.get_competitor_by_name(base_col)
                    display_name = comp_cfg.get_display_name()
                    categories = [comp_cfg.get_category(cat_name) for cat_name in cat_col_names]
                elif config.get_control_by_name(base_col):
                    ctrl_cfg = config.get_control_by_name(base_col)
                    display_name = ctrl_cfg.get_display_name()
                    categories = [ctrl_cfg.get_category(cat_name) for cat_name in cat_col_names]
                else:
                    # Classify base components
                    if col in ['intercept', 'trend', 't']:
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

            # Add KPI value (with residual adjustment for intercept if forcing to actuals)
            kpi_value = contribs.loc[date_val, col] * revenue_scale
            if force_to_actuals and col == 'intercept':
                kpi_value += residuals.get(date_val, 0)
            row[f'kpi_{target_col}'] = kpi_value

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


# =============================================================================
# Combined Model Export Functions
# =============================================================================


def generate_combined_decomps_stacked(
    wrappers_with_labels: List[Tuple["MMMWrapper", str]],
    brand: str,
    force_to_actuals: bool = False
) -> pd.DataFrame:
    """
    Generate stacked decomposition export combining multiple models.

    Creates a long-format DataFrame with one row per date per decomposition component,
    with separate KPI columns for each model plus a total column.

    Parameters
    ----------
    wrappers_with_labels : List[Tuple[MMMWrapper, str]]
        List of (wrapper, label) tuples. Labels are used for column names (e.g., "online", "offline")
    brand : str
        Brand name for the export
    force_to_actuals : bool, optional
        If True, absorb residuals into intercept so decomposition sums to actuals

    Returns
    -------
    pd.DataFrame
        Stacked decomposition data with columns:
        date, wc_mon, brand, decomp, decomp_lvl1, decomp_lvl2, kpi_{label1}, kpi_{label2}, kpi_total
    """
    if len(wrappers_with_labels) < 2:
        raise ValueError("At least 2 models required for combined export")

    # Get contributions and configs from all wrappers
    all_contribs = []
    all_configs = []
    all_wrappers = []
    labels = []
    revenue_scales = []

    for wrapper, label in wrappers_with_labels:
        contribs = wrapper.get_contributions()
        all_contribs.append(contribs)
        all_configs.append(wrapper.config)
        all_wrappers.append(wrapper)
        labels.append(label)
        revenue_scales.append(wrapper.config.data.revenue_scale)

    # Find common dates (inner join)
    common_dates = all_contribs[0].index
    for contribs in all_contribs[1:]:
        common_dates = common_dates.intersection(contribs.index)

    # Get all unique decomp components across all models
    all_components = set()
    for contribs in all_contribs:
        all_components.update(contribs.columns)

    # Use first config for category columns (assume same structure)
    config = all_configs[0]
    cat_col_names = config.get_category_column_names()

    # Calculate residuals per date per model if forcing to actuals
    all_residuals = []
    if force_to_actuals:
        for idx, (wrapper, cfg, contribs, scale) in enumerate(
            zip(all_wrappers, all_configs, all_contribs, revenue_scales)
        ):
            date_col = cfg.data.date_column
            target_col = cfg.data.target_column
            df_indexed = wrapper.df_scaled.set_index(date_col)
            residuals = {}
            for date_val in common_dates:
                if date_val in df_indexed.index:
                    actual = df_indexed.loc[date_val, target_col] * scale
                    fitted = contribs.loc[date_val].sum() * scale
                    residuals[date_val] = actual - fitted
                else:
                    residuals[date_val] = 0
            all_residuals.append(residuals)

    rows = []
    for date_val in common_dates:
        for col in sorted(all_components):
            # Determine display name and categories from first config that has this component
            display_name = col.replace("_", " ").title()
            categories = []

            for cfg in all_configs:
                channel_cfg = cfg.get_channel_by_name(col)
                owned_media_cfg = cfg.get_owned_media_by_name(col)
                competitor_cfg = cfg.get_competitor_by_name(col)
                control_cfg = cfg.get_control_by_name(col)

                if channel_cfg:
                    display_name = channel_cfg.get_display_name()
                    categories = [channel_cfg.get_category(cat_name) for cat_name in cat_col_names]
                    break
                elif owned_media_cfg:
                    display_name = owned_media_cfg.get_display_name()
                    categories = [owned_media_cfg.get_category(cat_name) for cat_name in cat_col_names]
                    break
                elif competitor_cfg:
                    display_name = competitor_cfg.get_display_name()
                    categories = [competitor_cfg.get_category(cat_name) for cat_name in cat_col_names]
                    break
                elif control_cfg:
                    display_name = control_cfg.get_display_name()
                    categories = [control_cfg.get_category(cat_name) for cat_name in cat_col_names]
                    break
                else:
                    # Check for transformed versions of known variables
                    base_col = col.replace("_adstock", "").replace("_inv", "")
                    comp_cfg = cfg.get_competitor_by_name(base_col)
                    ctrl_cfg = cfg.get_control_by_name(base_col)

                    if comp_cfg:
                        display_name = comp_cfg.get_display_name()
                        categories = [comp_cfg.get_category(cat_name) for cat_name in cat_col_names]
                        break
                    elif ctrl_cfg:
                        display_name = ctrl_cfg.get_display_name()
                        categories = [ctrl_cfg.get_category(cat_name) for cat_name in cat_col_names]
                        break
                    else:
                        # Base component
                        if col in ['intercept', 'trend', 't']:
                            base_category = "Base"
                        elif 'season' in col.lower() or 'fourier' in col.lower():
                            base_category = "Seasonality"
                        else:
                            base_category = "Other"
                        categories = [base_category] * len(cat_col_names)

            # If no category columns defined, use display_name
            if not categories:
                categories = [display_name, display_name]

            # Build row
            row = {
                'date': date_val,
                'wc_mon': date_val,
                'brand': brand,
                'decomp': col,
            }

            # Add category levels
            for i, cat_val in enumerate(categories):
                row[f'decomp_lvl{i + 1}'] = cat_val if cat_val else display_name

            # Ensure at least lvl1 and lvl2
            if len(categories) < 1:
                row['decomp_lvl1'] = display_name
            if len(categories) < 2:
                row['decomp_lvl2'] = row.get('decomp_lvl1', display_name)

            # Add KPI values for each model
            total = 0
            for idx, (contribs, label, scale) in enumerate(zip(all_contribs, labels, revenue_scales)):
                if col in contribs.columns:
                    value = contribs.loc[date_val, col] * scale
                else:
                    value = 0
                # Add residual to intercept if forcing to actuals
                if force_to_actuals and col == 'intercept':
                    value += all_residuals[idx].get(date_val, 0)
                row[f'kpi_{label}'] = value
                total += value

            row['kpi_total'] = total

            rows.append(row)

    return pd.DataFrame(rows)


def generate_combined_media_results(
    wrappers_with_labels: List[Tuple["MMMWrapper", str]],
    brand: str
) -> pd.DataFrame:
    """
    Generate media results export combining multiple models (channels only).

    Creates a long-format DataFrame with one row per date per media channel,
    with separate KPI columns for each model plus a total column.
    Spend is a single column (same business spend regardless of KPI).

    Parameters
    ----------
    wrappers_with_labels : List[Tuple[MMMWrapper, str]]
        List of (wrapper, label) tuples
    brand : str
        Brand name for the export

    Returns
    -------
    pd.DataFrame
        Media results data with columns:
        date, wc_mon, brand, decomp_lvl1, decomp_lvl2, spend, kpi_{label}, kpi_total, decomp
    """
    if len(wrappers_with_labels) < 2:
        raise ValueError("At least 2 models required for combined export")

    # Get contributions and configs from all wrappers
    all_contribs = []
    all_configs = []
    all_scaled_dfs = []
    labels = []
    revenue_scales = []
    spend_scales = []

    for wrapper, label in wrappers_with_labels:
        all_contribs.append(wrapper.get_contributions())
        all_configs.append(wrapper.config)
        all_scaled_dfs.append(wrapper.df_scaled)
        labels.append(label)
        revenue_scales.append(wrapper.config.data.revenue_scale)
        spend_scales.append(wrapper.config.data.spend_scale)

    # Find common dates
    common_dates = all_contribs[0].index
    for contribs in all_contribs[1:]:
        common_dates = common_dates.intersection(contribs.index)

    # Get all unique channels across all models
    all_channels = set()
    channel_configs = {}  # col_name -> first config that has it
    for cfg in all_configs:
        for channel_cfg in cfg.channels:
            if channel_cfg.name not in channel_configs:
                channel_configs[channel_cfg.name] = channel_cfg
            all_channels.add(channel_cfg.name)

    # Use first config for category columns and spend
    config = all_configs[0]
    cat_col_names = config.get_category_column_names()
    spend_scale = spend_scales[0]  # Use first model's spend scale (should be same)

    rows = []
    for date_idx, date_val in enumerate(common_dates):
        for col in sorted(all_channels):
            channel_cfg = channel_configs.get(col)
            if not channel_cfg:
                continue

            # Get display name and categories
            display_name = channel_cfg.get_display_name()
            categories = [channel_cfg.get_category(cat_name) for cat_name in cat_col_names]

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

            if len(categories) < 1:
                row['decomp_lvl1'] = display_name
            if len(categories) < 2:
                row['decomp_lvl2'] = row.get('decomp_lvl1', display_name)

            # Get spend from first model that has this channel (spend is same for all KPIs)
            spend = 0
            for idx, df_scaled in enumerate(all_scaled_dfs):
                if col in df_scaled.columns:
                    date_col = all_configs[idx].data.date_column
                    if date_col in df_scaled.columns:
                        mask = df_scaled[date_col] == date_val
                        if mask.any():
                            spend = df_scaled.loc[mask, col].iloc[0] * spend_scale
                            break  # Found spend, no need to check other models
            row['spend'] = spend

            # Add KPI contribution for each model
            kpi_total = 0
            for idx, (contribs, label, rev_scale) in enumerate(
                zip(all_contribs, labels, revenue_scales)
            ):
                if col in contribs.columns and date_val in contribs.index:
                    kpi_value = contribs.loc[date_val, col] * rev_scale
                else:
                    kpi_value = 0
                row[f'kpi_{label}'] = kpi_value
                kpi_total += kpi_value

            row['kpi_total'] = kpi_total
            row['decomp'] = col

            # Placeholders
            row['impressions'] = 0
            row['clicks'] = 0

            rows.append(row)

    return pd.DataFrame(rows)


def generate_combined_actual_vs_fitted(
    wrappers_with_labels: List[Tuple["MMMWrapper", str]],
    brand: str
) -> pd.DataFrame:
    """
    Generate actual vs fitted export combining multiple models.

    Creates a long-format DataFrame with two rows per date (Actual and Fitted),
    with separate value columns for each model plus a total column.

    Parameters
    ----------
    wrappers_with_labels : List[Tuple[MMMWrapper, str]]
        List of (wrapper, label) tuples
    brand : str
        Brand name for the export

    Returns
    -------
    pd.DataFrame
        Actual vs fitted data with columns:
        date, wc_mon, brand, actual_fitted, value_{label1}, value_{label2}, value_total
    """
    if len(wrappers_with_labels) < 2:
        raise ValueError("At least 2 models required for combined export")

    # Get contributions and configs
    all_contribs = []
    all_configs = []
    all_scaled_dfs = []
    labels = []
    revenue_scales = []

    for wrapper, label in wrappers_with_labels:
        all_contribs.append(wrapper.get_contributions())
        all_configs.append(wrapper.config)
        all_scaled_dfs.append(wrapper.df_scaled)
        labels.append(label)
        revenue_scales.append(wrapper.config.data.revenue_scale)

    # Find common dates
    common_dates = all_contribs[0].index
    for contribs in all_contribs[1:]:
        common_dates = common_dates.intersection(contribs.index)

    rows = []
    for date_val in common_dates:
        # Actual row
        actual_row = {
            'date': date_val,
            'wc_mon': date_val,
            'brand': brand,
            'actual_fitted': 'Actual',
        }

        actual_total = 0
        for idx, (df_scaled, cfg, label, scale) in enumerate(
            zip(all_scaled_dfs, all_configs, labels, revenue_scales)
        ):
            date_col = cfg.data.date_column
            target_col = cfg.data.target_column

            # Get actual value
            df_indexed = df_scaled.set_index(date_col)
            if date_val in df_indexed.index:
                actual = df_indexed.loc[date_val, target_col] * scale
            else:
                actual = np.nan
            actual_row[f'value_{label}'] = actual
            if not np.isnan(actual):
                actual_total += actual

        actual_row['value_total'] = actual_total
        rows.append(actual_row)

        # Fitted row
        fitted_row = {
            'date': date_val,
            'wc_mon': date_val,
            'brand': brand,
            'actual_fitted': 'Fitted',
        }

        fitted_total = 0
        for idx, (contribs, label, scale) in enumerate(
            zip(all_contribs, labels, revenue_scales)
        ):
            fitted = contribs.loc[date_val].sum() * scale
            fitted_row[f'value_{label}'] = fitted
            fitted_total += fitted

        fitted_row['value_total'] = fitted_total
        rows.append(fitted_row)

    return pd.DataFrame(rows)
