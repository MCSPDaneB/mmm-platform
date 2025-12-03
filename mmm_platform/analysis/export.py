"""
Export functions for generating CSV files in platform-specific formats.

These formats are designed for upload to external visualization platforms.
"""

import pandas as pd
import numpy as np
from typing import TYPE_CHECKING, List, Tuple, Dict

if TYPE_CHECKING:
    from mmm_platform.model.mmm import MMMWrapper
    from mmm_platform.config.schema import ModelConfig


def _reorder_decomp_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure decomp_lvl columns are in correct order (lvl1, lvl2, lvl3, etc.).

    This handles cases where category columns might be stored in a different order
    in the config, ensuring the output always has decomp_lvl1 before decomp_lvl2, etc.
    """
    cols = list(df.columns)
    # Find all decomp_lvl columns and sort by numeric suffix
    decomp_cols = sorted(
        [c for c in cols if c.startswith('decomp_lvl')],
        key=lambda x: int(x.replace('decomp_lvl', ''))
    )
    # Get non-decomp columns in their original order
    other_cols = [c for c in cols if not c.startswith('decomp_lvl')]

    # Rebuild column order: insert decomp_lvl columns after 'brand' if it exists
    if 'brand' in other_cols:
        brand_idx = other_cols.index('brand') + 1
        new_order = other_cols[:brand_idx] + decomp_cols + other_cols[brand_idx:]
    else:
        new_order = other_cols + decomp_cols

    return df[new_order]


def get_channel_adstock_alphas(wrapper: "MMMWrapper") -> Dict[str, float]:
    """
    Get posterior mean adstock alpha for each channel from fitted model.

    Parameters
    ----------
    wrapper : MMMWrapper
        Fitted model wrapper with idata containing posterior samples

    Returns
    -------
    Dict[str, float]
        Dictionary mapping channel names to posterior mean alpha values
    """
    idata = wrapper.idata
    adstock_posterior = idata.posterior["adstock_alpha"]

    # Get channel names in order (paid media + owned media)
    channel_cols = wrapper.transform_engine.get_effective_channel_columns()

    alphas = {}
    for idx, ch in enumerate(channel_cols):
        samples = adstock_posterior.isel(channel=idx).values.flatten()
        alphas[ch] = float(samples.mean())

    return alphas


def apply_adstock_to_weights(
    granular_df: pd.DataFrame,
    weight_col: str,
    date_col: str,
    entity_col: str,
    channel_col: str,
    channel_alphas: Dict[str, float],
    l_max: int = 8
) -> pd.DataFrame:
    """
    Apply adstock transformation to weight column per entity.

    CRITICAL: Data is sorted chronologically before applying adstock.
    The carryover effect flows forward in time - past spend affects current period.

    Parameters
    ----------
    granular_df : pd.DataFrame
        DataFrame with granular-level data
    weight_col : str
        Column name containing weight values (e.g., spend)
    date_col : str
        Column name containing dates
    entity_col : str
        Column name containing entity identifiers (composite key)
    channel_col : str
        Column name containing model channel mappings
    channel_alphas : Dict[str, float]
        Dictionary mapping channel names to adstock alpha values
    l_max : int, optional
        Maximum lag for adstock effect (default 8)

    Returns
    -------
    pd.DataFrame
        DataFrame with new column '{weight_col}_adstocked'
    """
    df = granular_df.copy()

    # Ensure date column is datetime for proper sorting
    df[date_col] = pd.to_datetime(df[date_col], dayfirst=True)

    adstocked_col = f"{weight_col}_adstocked"
    df[adstocked_col] = 0.0

    # Group by entity (each entity gets its own adstock)
    for entity in df[entity_col].unique():
        entity_mask = df[entity_col] == entity

        # CRITICAL: Sort by date ascending (oldest first) before adstock
        entity_data = df.loc[entity_mask].sort_values(date_col, ascending=True)

        if len(entity_data) == 0:
            continue

        # Get the channel this entity maps to
        channel = entity_data[channel_col].iloc[0]
        if channel not in channel_alphas:
            # No adstock alpha for this channel - use raw weights
            df.loc[entity_data.index, adstocked_col] = entity_data[weight_col].values
            continue

        alpha = channel_alphas[channel]

        # Apply geometric adstock to this entity's TIME-ORDERED series
        # Formula: weights = [alpha^0, alpha^1, ..., alpha^(l_max-1)] normalized
        weights = np.array([alpha ** i for i in range(l_max)])
        weights = weights / weights.sum()

        x = entity_data[weight_col].values
        x_adstocked = np.convolve(x, weights, mode='full')[:len(x)]

        # Assign back using the sorted index (preserves time order mapping)
        df.loc[entity_data.index, adstocked_col] = x_adstocked

    return df


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
            # Determine if channel, owned media, competitor, control, dummy, or base component
            channel_cfg = config.get_channel_by_name(col)
            owned_media_cfg = config.get_owned_media_by_name(col)
            competitor_cfg = config.get_competitor_by_name(col)
            control_cfg = config.get_control_by_name(col)
            dummy_cfg = next((d for d in config.dummy_variables if d.name == col), None)

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
            elif dummy_cfg:
                display_name = dummy_cfg.name  # DummyVariableConfig doesn't have display_name
                categories = [dummy_cfg.categories.get(cat_name, "") for cat_name in cat_col_names]
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
                    # Base component - read from config.base_component_categories if available
                    base_cats = config.base_component_categories.get(col, {}) if hasattr(config, 'base_component_categories') else {}
                    if base_cats:
                        categories = [base_cats.get(cat_name, "") for cat_name in cat_col_names]
                    else:
                        # Fallback to defaults if no saved categories
                        if col in ['intercept', 'trend', 't']:
                            base_category = "Base"
                        elif 'season' in col.lower() or 'fourier' in col.lower() or col.startswith('sin_order') or col.startswith('cos_order'):
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

            # Add category levels using actual column names from config
            for i, cat_val in enumerate(categories):
                col_name = cat_col_names[i] if i < len(cat_col_names) else f'decomp_lvl{i + 1}'
                row[col_name] = cat_val if cat_val else display_name

            # If fewer than 2 category columns, ensure at least lvl1 and lvl2
            if len(cat_col_names) < 1:
                row['decomp_lvl1'] = display_name
            if len(cat_col_names) < 2:
                row['decomp_lvl2'] = row.get(cat_col_names[0] if cat_col_names else 'decomp_lvl1', display_name)

            # Add KPI value (with residual adjustment for intercept if forcing to actuals)
            kpi_value = contribs.loc[date_val, col] * revenue_scale
            if force_to_actuals and col == 'intercept':
                kpi_value += residuals.get(date_val, 0)
            row[f'kpi_{target_col}'] = kpi_value

            rows.append(row)

    return _reorder_decomp_columns(pd.DataFrame(rows))


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

            # Add category levels using actual column names from config
            for i, cat_val in enumerate(categories):
                col_name = cat_col_names[i] if i < len(cat_col_names) else f'decomp_lvl{i + 1}'
                row[col_name] = cat_val if cat_val else display_name

            # If fewer than 2 category columns, ensure at least lvl1 and lvl2
            if len(cat_col_names) < 1:
                row['decomp_lvl1'] = display_name
            if len(cat_col_names) < 2:
                row['decomp_lvl2'] = row.get(cat_col_names[0] if cat_col_names else 'decomp_lvl1', display_name)

            # Add spend and derived metrics
            row['spend'] = spend

            # Derive impressions/clicks column names from spend column
            impr_col = col.replace('_spend', '_impr').replace('_Spend', '_impr')
            clicks_col = col.replace('_spend', '_clicks').replace('_Spend', '_clicks')

            # Get impressions value (if column exists)
            if impr_col in wrapper.df_scaled.columns and date_idx < len(wrapper.df_scaled):
                row['impressions'] = wrapper.df_scaled.iloc[date_idx][impr_col]
            else:
                row['impressions'] = 0

            # Get clicks value (if column exists)
            if clicks_col in wrapper.df_scaled.columns and date_idx < len(wrapper.df_scaled):
                row['clicks'] = wrapper.df_scaled.iloc[date_idx][clicks_col]
            else:
                row['clicks'] = 0

            row['decomp'] = col

            # Add KPI value
            row[f'kpi_{target_col}'] = contribs.loc[date_val, col] * revenue_scale

            rows.append(row)

    return _reorder_decomp_columns(pd.DataFrame(rows))


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

    return _reorder_decomp_columns(pd.DataFrame(rows))


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
                dummy_cfg = next((d for d in cfg.dummy_variables if d.name == col), None)

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
                elif dummy_cfg:
                    display_name = dummy_cfg.name  # DummyVariableConfig doesn't have display_name
                    categories = [dummy_cfg.categories.get(cat_name, "") for cat_name in cat_col_names]
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
                        # Base component - read from config.base_component_categories if available
                        base_cats = cfg.base_component_categories.get(col, {}) if hasattr(cfg, 'base_component_categories') else {}
                        if base_cats:
                            categories = [base_cats.get(cat_name, "") for cat_name in cat_col_names]
                            break
                        else:
                            # Fallback to defaults if no saved categories
                            if col in ['intercept', 'trend', 't']:
                                base_category = "Base"
                            elif 'season' in col.lower() or 'fourier' in col.lower() or col.startswith('sin_order') or col.startswith('cos_order'):
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

            # Add category levels using actual column names from config
            for i, cat_val in enumerate(categories):
                col_name = cat_col_names[i] if i < len(cat_col_names) else f'decomp_lvl{i + 1}'
                row[col_name] = cat_val if cat_val else display_name

            # Ensure at least lvl1 and lvl2
            if len(cat_col_names) < 1:
                row['decomp_lvl1'] = display_name
            if len(cat_col_names) < 2:
                row['decomp_lvl2'] = row.get(cat_col_names[0] if cat_col_names else 'decomp_lvl1', display_name)

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

    return _reorder_decomp_columns(pd.DataFrame(rows))


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

            # Add category levels using actual column names from config
            for i, cat_val in enumerate(categories):
                col_name = cat_col_names[i] if i < len(cat_col_names) else f'decomp_lvl{i + 1}'
                row[col_name] = cat_val if cat_val else display_name

            if len(cat_col_names) < 1:
                row['decomp_lvl1'] = display_name
            if len(cat_col_names) < 2:
                row['decomp_lvl2'] = row.get(cat_col_names[0] if cat_col_names else 'decomp_lvl1', display_name)

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

            # Derive impressions/clicks column names from spend column
            impr_col = col.replace('_spend', '_impr').replace('_Spend', '_impr')
            clicks_col = col.replace('_spend', '_clicks').replace('_Spend', '_clicks')

            # Get impressions and clicks from first model that has these columns
            row['impressions'] = 0
            row['clicks'] = 0
            for df_scaled in all_scaled_dfs:
                if impr_col in df_scaled.columns:
                    date_col_name = all_configs[all_scaled_dfs.index(df_scaled)].data.date_column
                    if date_col_name in df_scaled.columns:
                        mask = df_scaled[date_col_name] == date_val
                        if mask.any():
                            row['impressions'] = df_scaled.loc[mask, impr_col].iloc[0]
                            break
            for df_scaled in all_scaled_dfs:
                if clicks_col in df_scaled.columns:
                    date_col_name = all_configs[all_scaled_dfs.index(df_scaled)].data.date_column
                    if date_col_name in df_scaled.columns:
                        mask = df_scaled[date_col_name] == date_val
                        if mask.any():
                            row['clicks'] = df_scaled.loc[mask, clicks_col].iloc[0]
                            break

            rows.append(row)

    return _reorder_decomp_columns(pd.DataFrame(rows))


def generate_disaggregated_results(
    wrapper: "MMMWrapper",
    config: "ModelConfig",
    granular_df: pd.DataFrame,
    granular_name_cols: list[str],
    date_col: str,
    model_channel_col: str,
    weight_col: str,
    brand: str
) -> pd.DataFrame:
    """
    Generate disaggregated results by splitting model contributions to granular level.

    Takes aggregated model results and splits them proportionally based on weights
    from a granular mapping file.

    Parameters
    ----------
    wrapper : MMMWrapper
        Fitted model wrapper
    config : ModelConfig
        Model configuration
    granular_df : pd.DataFrame
        DataFrame with granular-level data containing:
        - granular name column(s) (e.g., placement names, or multiple columns for composite key)
        - date column
        - model channel mapping column (maps to existing model channels)
        - weight column (numeric values for proportional splitting)
    granular_name_cols : list[str]
        List of column names forming the entity identifier (composite key).
        Multiple columns are joined with " | " to create a single identifier.
    date_col : str
        Column name containing dates
    model_channel_col : str
        Column name containing model channel mappings
    weight_col : str
        Column name containing weight values for proportional allocation
    brand : str
        Brand name for the export

    Returns
    -------
    pd.DataFrame
        Disaggregated results with columns:
        date, wc_mon, brand, model_channel, granular_name, weight, weight_pct,
        contribution, spend (if weight_col is spend-like), roi (if applicable)
    """
    contribs = wrapper.get_contributions()
    revenue_scale = config.data.revenue_scale
    target_col = config.data.target_column
    l_max = config.adstock.l_max

    # Get category column names (may be empty)
    cat_col_names = config.get_category_column_names()

    # Convert granular date column to datetime and NORMALIZE (remove time component)
    granular_df = granular_df.copy()
    granular_df[date_col] = pd.to_datetime(granular_df[date_col], dayfirst=True).dt.normalize()

    # Helper to create composite key from multiple columns
    def make_composite_key(row):
        return " | ".join(str(row[col]) for col in granular_name_cols)

    # Create composite key column for adstock grouping
    granular_df["_entity_key"] = granular_df.apply(make_composite_key, axis=1)

    # Get posterior adstock alphas from fitted model
    channel_alphas = get_channel_adstock_alphas(wrapper)

    # Apply adstock transformation to weights (CRITICAL: time-ordered per entity)
    granular_df = apply_adstock_to_weights(
        granular_df=granular_df,
        weight_col=weight_col,
        date_col=date_col,
        entity_col="_entity_key",
        channel_col=model_channel_col,
        channel_alphas=channel_alphas,
        l_max=l_max
    )

    # Use adstocked weights for proportional allocation
    adstocked_weight_col = f"{weight_col}_adstocked"

    # Get model channel names
    model_channels = [ch.name for ch in config.channels]

    rows = []

    # Helper to build row with decomp_lvl columns
    def build_row(date_val, channel_name, granular_name, kpi_value, raw_weight, adstocked_weight, weight_pct, is_mapped):
        # Get channel config for categories
        channel_cfg = config.get_channel_by_name(channel_name)
        if channel_cfg:
            categories = [channel_cfg.get_category(cat_name) for cat_name in cat_col_names]
            display_name = channel_cfg.get_display_name()
        else:
            categories = []
            display_name = channel_name

        row = {
            'date': date_val,
            'wc_mon': date_val,
            'brand': brand,
            'decomp': channel_name,
        }

        # Add existing category levels using actual column names from config
        for i, cat_val in enumerate(categories):
            col_name = cat_col_names[i] if i < len(cat_col_names) else f'decomp_lvl{i + 1}'
            row[col_name] = cat_val if cat_val else display_name

        # Add granular_name as the NEXT level after existing categories
        next_lvl = len(cat_col_names) + 1
        row[f'decomp_lvl{next_lvl}'] = granular_name

        # Add KPI and weight columns
        row[f'kpi_{target_col}'] = kpi_value
        row['weight'] = raw_weight
        row['weight_adstocked'] = adstocked_weight
        row['weight_pct'] = weight_pct
        row['is_mapped'] = is_mapped

        return row

    # Process each date in the model contributions
    for date_val in contribs.index:
        # Normalize date for matching
        date_dt = pd.Timestamp(date_val).normalize()

        # Get granular data for this date
        date_mask = granular_df[date_col] == date_dt
        date_granular = granular_df[date_mask]

        # Process each model channel
        for channel_name in model_channels:
            if channel_name not in contribs.columns:
                continue

            # Get channel contribution for this date
            channel_contrib = contribs.loc[date_val, channel_name] * revenue_scale

            # Get granular rows mapped to this channel
            channel_mask = date_granular[model_channel_col] == channel_name
            channel_granular = date_granular[channel_mask]

            if len(channel_granular) == 0:
                # No granular mapping for this channel/date - output at channel level
                rows.append(build_row(
                    date_val=date_val,
                    channel_name=channel_name,
                    granular_name=channel_name,  # Use channel name as granular
                    kpi_value=channel_contrib,
                    raw_weight=0,
                    adstocked_weight=0,
                    weight_pct=1.0,
                    is_mapped=False
                ))
                continue

            # Calculate total ADSTOCKED weight for this channel/date
            total_weight = channel_granular[adstocked_weight_col].sum()

            if total_weight == 0:
                # Zero total weight - distribute equally
                n_granular = len(channel_granular)
                for _, g_row in channel_granular.iterrows():
                    rows.append(build_row(
                        date_val=date_val,
                        channel_name=channel_name,
                        granular_name=make_composite_key(g_row),
                        kpi_value=channel_contrib / n_granular,
                        raw_weight=g_row[weight_col],
                        adstocked_weight=g_row[adstocked_weight_col],
                        weight_pct=1.0 / n_granular,
                        is_mapped=True
                    ))
            else:
                # Split proportionally by ADSTOCKED weight
                for _, g_row in channel_granular.iterrows():
                    adstocked_wt = g_row[adstocked_weight_col]
                    weight_pct = adstocked_wt / total_weight
                    granular_contrib = channel_contrib * weight_pct

                    rows.append(build_row(
                        date_val=date_val,
                        channel_name=channel_name,
                        granular_name=make_composite_key(g_row),
                        kpi_value=granular_contrib,
                        raw_weight=g_row[weight_col],
                        adstocked_weight=adstocked_wt,
                        weight_pct=weight_pct,
                        is_mapped=True
                    ))

    return _reorder_decomp_columns(pd.DataFrame(rows))


def generate_decomps_stacked_disaggregated(
    wrapper: "MMMWrapper",
    config: "ModelConfig",
    granular_df: pd.DataFrame,
    granular_name_cols: list[str],
    date_col: str,
    model_channel_col: str,
    weight_col: str,
    brand: str,
    force_to_actuals: bool = False
) -> pd.DataFrame:
    """
    Generate disaggregated decomps_stacked export.

    Same format as generate_decomps_stacked but with mapped channels expanded
    to granular rows. Non-channel components (intercept, trend, etc.) and
    unmapped channels remain at original level.

    Parameters
    ----------
    wrapper : MMMWrapper
        Fitted model wrapper
    config : ModelConfig
        Model configuration
    granular_df : pd.DataFrame
        DataFrame with granular-level data containing entity mappings
    granular_name_cols : list[str]
        List of column names forming the entity identifier
    date_col : str
        Column name containing dates
    model_channel_col : str
        Column name containing model channel mappings
    weight_col : str
        Column name containing weight values for proportional allocation
    brand : str
        Brand name for the export
    force_to_actuals : bool, optional
        If True, absorb residuals into intercept

    Returns
    -------
    pd.DataFrame
        Disaggregated decomps stacked with same format as generate_decomps_stacked
        but granular_name as the next decomp_lvl for mapped channels
    """
    contribs = wrapper.get_contributions()
    revenue_scale = config.data.revenue_scale
    target_col = config.data.target_column
    l_max = config.adstock.l_max

    # Get category column names
    cat_col_names = config.get_category_column_names()

    # Prepare granular data with adstock
    granular_df = granular_df.copy()
    granular_df[date_col] = pd.to_datetime(granular_df[date_col], dayfirst=True).dt.normalize()

    def make_composite_key(row):
        return " | ".join(str(row[col]) for col in granular_name_cols)

    granular_df["_entity_key"] = granular_df.apply(make_composite_key, axis=1)

    # Get posterior adstock alphas and apply
    channel_alphas = get_channel_adstock_alphas(wrapper)
    granular_df = apply_adstock_to_weights(
        granular_df=granular_df,
        weight_col=weight_col,
        date_col=date_col,
        entity_col="_entity_key",
        channel_col=model_channel_col,
        channel_alphas=channel_alphas,
        l_max=l_max
    )
    adstocked_weight_col = f"{weight_col}_adstocked"

    # Get model channels that have granular mapping
    mapped_channels = set(granular_df[model_channel_col].unique())

    # Calculate residuals if forcing to actuals
    residuals = {}
    if force_to_actuals:
        date_col_cfg = config.data.date_column
        df_indexed = wrapper.df_scaled.set_index(date_col_cfg)
        for date_val in contribs.index:
            if date_val in df_indexed.index:
                actual = df_indexed.loc[date_val, target_col] * revenue_scale
                fitted = contribs.loc[date_val].sum() * revenue_scale
                residuals[date_val] = actual - fitted
            else:
                residuals[date_val] = 0

    rows = []

    # Helper to get categories for a component
    def get_component_info(col):
        channel_cfg = config.get_channel_by_name(col)
        owned_media_cfg = config.get_owned_media_by_name(col)
        competitor_cfg = config.get_competitor_by_name(col)
        control_cfg = config.get_control_by_name(col)
        dummy_cfg = next((d for d in config.dummy_variables if d.name == col), None)

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
        elif dummy_cfg:
            display_name = dummy_cfg.name
            categories = [dummy_cfg.categories.get(cat_name, "") for cat_name in cat_col_names]
        else:
            # Base component
            base_col = col.replace("_adstock", "").replace("_inv", "")
            display_name = col.replace("_", " ").title()

            if config.get_competitor_by_name(base_col):
                comp_cfg = config.get_competitor_by_name(base_col)
                display_name = comp_cfg.get_display_name()
                categories = [comp_cfg.get_category(cat_name) for cat_name in cat_col_names]
            elif config.get_control_by_name(base_col):
                ctrl_cfg = config.get_control_by_name(base_col)
                display_name = ctrl_cfg.get_display_name()
                categories = [ctrl_cfg.get_category(cat_name) for cat_name in cat_col_names]
            else:
                base_cats = config.base_component_categories.get(col, {}) if hasattr(config, 'base_component_categories') else {}
                if base_cats:
                    categories = [base_cats.get(cat_name, "") for cat_name in cat_col_names]
                else:
                    if col in ['intercept', 'trend', 't']:
                        base_category = "Base"
                    elif 'season' in col.lower() or 'fourier' in col.lower() or col.startswith('sin_order') or col.startswith('cos_order'):
                        base_category = "Seasonality"
                    else:
                        base_category = "Other"
                    categories = [base_category] * len(cat_col_names)

        if not categories:
            categories = [display_name, display_name]

        return display_name, categories

    # Process each date and component
    for date_val in contribs.index:
        date_dt = pd.Timestamp(date_val).normalize()
        date_mask = granular_df[date_col] == date_dt
        date_granular = granular_df[date_mask]

        for col in contribs.columns:
            display_name, categories = get_component_info(col)
            kpi_value = contribs.loc[date_val, col] * revenue_scale

            if force_to_actuals and col == 'intercept':
                kpi_value += residuals.get(date_val, 0)

            # Check if this is a mapped channel
            is_channel = config.get_channel_by_name(col) is not None
            is_mapped = is_channel and col in mapped_channels

            if is_mapped:
                # Get granular rows for this channel/date
                channel_mask = date_granular[model_channel_col] == col
                channel_granular = date_granular[channel_mask]

                if len(channel_granular) == 0:
                    # No granular data for this date - output at channel level
                    row = {
                        'date': date_val,
                        'wc_mon': date_val,
                        'brand': brand,
                        'decomp': col,
                    }
                    for i, cat_val in enumerate(categories):
                        col_name = cat_col_names[i] if i < len(cat_col_names) else f'decomp_lvl{i + 1}'
                        row[col_name] = cat_val if cat_val else display_name
                    # Add granular_name as next level (channel name as placeholder)
                    next_lvl = len(cat_col_names) + 1
                    row[f'decomp_lvl{next_lvl}'] = col
                    if len(cat_col_names) < 1:
                        row['decomp_lvl1'] = display_name
                    if len(cat_col_names) < 2:
                        row['decomp_lvl2'] = row.get(cat_col_names[0] if cat_col_names else 'decomp_lvl1', display_name)
                    row[f'kpi_{target_col}'] = kpi_value
                    rows.append(row)
                else:
                    # Split proportionally
                    total_weight = channel_granular[adstocked_weight_col].sum()

                    for _, g_row in channel_granular.iterrows():
                        if total_weight > 0:
                            weight_pct = g_row[adstocked_weight_col] / total_weight
                        else:
                            weight_pct = 1.0 / len(channel_granular)
                        granular_contrib = kpi_value * weight_pct
                        granular_name = make_composite_key(g_row)

                        row = {
                            'date': date_val,
                            'wc_mon': date_val,
                            'brand': brand,
                            'decomp': col,
                        }
                        for i, cat_val in enumerate(categories):
                            col_name = cat_col_names[i] if i < len(cat_col_names) else f'decomp_lvl{i + 1}'
                            row[col_name] = cat_val if cat_val else display_name
                        # Add granular_name as next level
                        next_lvl = len(cat_col_names) + 1
                        row[f'decomp_lvl{next_lvl}'] = granular_name
                        if len(cat_col_names) < 1:
                            row['decomp_lvl1'] = display_name
                        if len(cat_col_names) < 2:
                            row['decomp_lvl2'] = row.get(cat_col_names[0] if cat_col_names else 'decomp_lvl1', display_name)
                        row[f'kpi_{target_col}'] = granular_contrib
                        rows.append(row)
            else:
                # Non-mapped: output at original level
                row = {
                    'date': date_val,
                    'wc_mon': date_val,
                    'brand': brand,
                    'decomp': col,
                }
                for i, cat_val in enumerate(categories):
                    col_name = cat_col_names[i] if i < len(cat_col_names) else f'decomp_lvl{i + 1}'
                    row[col_name] = cat_val if cat_val else display_name
                if len(cat_col_names) < 1:
                    row['decomp_lvl1'] = display_name
                if len(cat_col_names) < 2:
                    row['decomp_lvl2'] = row.get(cat_col_names[0] if cat_col_names else 'decomp_lvl1', display_name)
                row[f'kpi_{target_col}'] = kpi_value
                rows.append(row)

    return _reorder_decomp_columns(pd.DataFrame(rows))


def generate_media_results_disaggregated(
    wrapper: "MMMWrapper",
    config: "ModelConfig",
    granular_df: pd.DataFrame,
    granular_name_cols: list[str],
    date_col: str,
    model_channel_col: str,
    weight_col: str,
    include_cols: list[str],
    brand: str
) -> pd.DataFrame:
    """
    Generate disaggregated media results export.

    Same format as generate_media_results but with mapped channels expanded
    to granular rows. Includes actual spend/impressions/clicks from granular file.

    Parameters
    ----------
    wrapper : MMMWrapper
        Fitted model wrapper
    config : ModelConfig
        Model configuration
    granular_df : pd.DataFrame
        DataFrame with granular-level data containing entity mappings
    granular_name_cols : list[str]
        List of column names forming the entity identifier
    date_col : str
        Column name containing dates
    model_channel_col : str
        Column name containing model channel mappings
    weight_col : str
        Column name containing weight values for proportional allocation
    include_cols : list[str]
        Additional columns to include from granular file (e.g., impressions, clicks)
    brand : str
        Brand name for the export

    Returns
    -------
    pd.DataFrame
        Disaggregated media results with same format as generate_media_results
        but granular_name as the next decomp_lvl for mapped channels
    """
    contribs = wrapper.get_contributions()
    revenue_scale = config.data.revenue_scale
    spend_scale = config.data.spend_scale
    target_col = config.data.target_column
    l_max = config.adstock.l_max

    # Get category column names
    cat_col_names = config.get_category_column_names()

    # Prepare granular data with adstock
    granular_df = granular_df.copy()
    granular_df[date_col] = pd.to_datetime(granular_df[date_col], dayfirst=True).dt.normalize()

    def make_composite_key(row):
        return " | ".join(str(row[col]) for col in granular_name_cols)

    granular_df["_entity_key"] = granular_df.apply(make_composite_key, axis=1)

    # Get posterior adstock alphas and apply
    channel_alphas = get_channel_adstock_alphas(wrapper)
    granular_df = apply_adstock_to_weights(
        granular_df=granular_df,
        weight_col=weight_col,
        date_col=date_col,
        entity_col="_entity_key",
        channel_col=model_channel_col,
        channel_alphas=channel_alphas,
        l_max=l_max
    )
    adstocked_weight_col = f"{weight_col}_adstocked"

    # Get model channels that have granular mapping
    mapped_channels = set(granular_df[model_channel_col].unique())

    rows = []

    # Process each date and channel
    for date_idx, date_val in enumerate(contribs.index):
        date_dt = pd.Timestamp(date_val).normalize()
        date_mask = granular_df[date_col] == date_dt
        date_granular = granular_df[date_mask]

        for channel_cfg in config.channels:
            col = channel_cfg.name

            if col not in contribs.columns:
                continue

            display_name = channel_cfg.get_display_name()
            categories = [channel_cfg.get_category(cat_name) for cat_name in cat_col_names]
            if not categories:
                categories = [display_name, display_name]

            kpi_value = contribs.loc[date_val, col] * revenue_scale

            # Check if this channel is mapped
            is_mapped = col in mapped_channels

            if is_mapped:
                channel_mask = date_granular[model_channel_col] == col
                channel_granular = date_granular[channel_mask]

                if len(channel_granular) == 0:
                    # No granular data for this date - output at channel level
                    # Get spend from model data
                    if date_idx < len(wrapper.df_scaled):
                        spend = wrapper.df_scaled.iloc[date_idx][col] * spend_scale
                    else:
                        spend = 0

                    row = {
                        'date': date_val,
                        'wc_mon': date_val,
                        'brand': brand,
                    }
                    for i, cat_val in enumerate(categories):
                        col_name = cat_col_names[i] if i < len(cat_col_names) else f'decomp_lvl{i + 1}'
                        row[col_name] = cat_val if cat_val else display_name
                    next_lvl = len(cat_col_names) + 1
                    row[f'decomp_lvl{next_lvl}'] = col
                    if len(cat_col_names) < 1:
                        row['decomp_lvl1'] = display_name
                    if len(cat_col_names) < 2:
                        row['decomp_lvl2'] = row.get(cat_col_names[0] if cat_col_names else 'decomp_lvl1', display_name)
                    row['spend'] = spend
                    # For no-granular-data case, set all include_cols to 0
                    for inc_col in include_cols:
                        row[inc_col] = 0
                    row['decomp'] = col
                    row[f'kpi_{target_col}'] = kpi_value
                    rows.append(row)
                else:
                    # Split proportionally
                    total_weight = channel_granular[adstocked_weight_col].sum()

                    for _, g_row in channel_granular.iterrows():
                        if total_weight > 0:
                            weight_pct = g_row[adstocked_weight_col] / total_weight
                        else:
                            weight_pct = 1.0 / len(channel_granular)
                        granular_contrib = kpi_value * weight_pct
                        granular_name = make_composite_key(g_row)

                        row = {
                            'date': date_val,
                            'wc_mon': date_val,
                            'brand': brand,
                        }
                        for i, cat_val in enumerate(categories):
                            col_name = cat_col_names[i] if i < len(cat_col_names) else f'decomp_lvl{i + 1}'
                            row[col_name] = cat_val if cat_val else display_name
                        next_lvl = len(cat_col_names) + 1
                        row[f'decomp_lvl{next_lvl}'] = granular_name
                        if len(cat_col_names) < 1:
                            row['decomp_lvl1'] = display_name
                        if len(cat_col_names) < 2:
                            row['decomp_lvl2'] = row.get(cat_col_names[0] if cat_col_names else 'decomp_lvl1', display_name)

                        # Use actual values from granular file for spend column
                        row['spend'] = g_row[weight_col] if weight_col else 0

                        # Get include_cols values from granular file (actual values, not split)
                        for inc_col in include_cols:
                            if inc_col in g_row.index:
                                row[inc_col] = g_row[inc_col]
                            else:
                                row[inc_col] = 0  # Default if column not found

                        row['decomp'] = col
                        row[f'kpi_{target_col}'] = granular_contrib
                        rows.append(row)
            else:
                # Unmapped channel - output at channel level
                if date_idx < len(wrapper.df_scaled):
                    spend = wrapper.df_scaled.iloc[date_idx][col] * spend_scale
                else:
                    spend = 0

                row = {
                    'date': date_val,
                    'wc_mon': date_val,
                    'brand': brand,
                }
                for i, cat_val in enumerate(categories):
                    col_name = cat_col_names[i] if i < len(cat_col_names) else f'decomp_lvl{i + 1}'
                    row[col_name] = cat_val if cat_val else display_name
                if len(cat_col_names) < 1:
                    row['decomp_lvl1'] = display_name
                if len(cat_col_names) < 2:
                    row['decomp_lvl2'] = row.get(cat_col_names[0] if cat_col_names else 'decomp_lvl1', display_name)
                row['spend'] = spend

                # For unmapped channels, set include_cols to 0 (no granular data available)
                for inc_col in include_cols:
                    row[inc_col] = 0

                row['decomp'] = col
                row[f'kpi_{target_col}'] = kpi_value
                rows.append(row)

    return _reorder_decomp_columns(pd.DataFrame(rows))


def generate_combined_decomps_stacked_disaggregated(
    wrappers_with_disagg: List[Tuple["MMMWrapper", str, tuple]],
    brand: str,
    force_to_actuals: bool = False
) -> pd.DataFrame:
    """
    Generate combined disaggregated decomps_stacked export from multiple models.

    Each model's disaggregated decomps are merged into a single DataFrame with
    separate KPI columns per model plus a total column.

    Parameters
    ----------
    wrappers_with_disagg : List[Tuple[MMMWrapper, str, tuple]]
        List of (wrapper, label, disagg_config) tuples where disagg_config is
        (mapped_df, granular_name_cols, date_col, weight_col, include_cols)
    brand : str
        Brand name for the export
    force_to_actuals : bool, optional
        If True, absorb residuals into intercept

    Returns
    -------
    pd.DataFrame
        Combined disaggregated decomps with columns:
        date, wc_mon, brand, decomp, decomp_lvl1..N, kpi_{label1}, kpi_{label2}, kpi_total
    """
    if len(wrappers_with_disagg) < 2:
        raise ValueError("At least 2 models required for combined export")

    # Generate disaggregated decomps for each model
    all_disagg_dfs = []
    labels = []
    all_include_cols = set()  # Collect include_cols from all models dynamically

    for wrapper, label, disagg_config in wrappers_with_disagg:
        mapped_df, granular_name_cols, date_col, weight_col, include_cols = disagg_config

        # Track all include_cols across models for merge key exclusion
        if include_cols:
            all_include_cols.update(include_cols)

        df_disagg = generate_decomps_stacked_disaggregated(
            wrapper=wrapper,
            config=wrapper.config,
            granular_df=mapped_df,
            granular_name_cols=granular_name_cols,
            date_col=date_col,
            model_channel_col="_model_channel",
            weight_col=weight_col,
            brand=brand,
            force_to_actuals=force_to_actuals
        )

        # Find the KPI column name
        target_col = wrapper.config.data.target_column
        kpi_col = f"kpi_{target_col}"

        # Rename KPI column to kpi_{label}
        df_disagg = df_disagg.rename(columns={kpi_col: f"kpi_{label}"})

        all_disagg_dfs.append(df_disagg)
        labels.append(label)

    # Identify key columns - structural columns only, exclude value columns
    # Value columns can have floating-point differences across models causing duplicate rows
    first_df = all_disagg_dfs[0]
    # Base value columns that are always numeric
    base_value_cols = {'spend', 'impressions', 'clicks', 'weight_pct', 'raw_weight', 'adstocked_weight'}
    # Combine with dynamically collected include_cols (e.g., instore_revenue, online_revenue)
    value_cols = base_value_cols | all_include_cols
    key_cols = [c for c in first_df.columns
                if not c.startswith('kpi_') and c not in value_cols]

    # Merge all DataFrames on key columns
    result = all_disagg_dfs[0]
    for i, df in enumerate(all_disagg_dfs[1:], 1):
        label = labels[i]
        # Only keep key columns and the kpi column from subsequent dfs
        df_subset = df[key_cols + [f"kpi_{label}"]]
        result = result.merge(df_subset, on=key_cols, how='outer')

    # Fill NaN with 0 for KPI columns (when a model doesn't have a particular row)
    for label in labels:
        kpi_col = f"kpi_{label}"
        if kpi_col in result.columns:
            result[kpi_col] = result[kpi_col].fillna(0)

    # Calculate kpi_total
    kpi_cols = [f"kpi_{label}" for label in labels]
    result['kpi_total'] = result[kpi_cols].sum(axis=1)

    # Sort by date and decomp
    if 'date' in result.columns:
        result = result.sort_values(['date', 'decomp'])

    return _reorder_decomp_columns(result)


def generate_combined_media_results_disaggregated(
    wrappers_with_disagg: List[Tuple["MMMWrapper", str, tuple]],
    brand: str
) -> pd.DataFrame:
    """
    Generate combined disaggregated media results export from multiple models.

    Each model's disaggregated media results are merged into a single DataFrame with
    separate KPI columns per model plus a total column.

    Parameters
    ----------
    wrappers_with_disagg : List[Tuple[MMMWrapper, str, tuple]]
        List of (wrapper, label, disagg_config) tuples where disagg_config is
        (mapped_df, granular_name_cols, date_col, weight_col, include_cols)
    brand : str
        Brand name for the export

    Returns
    -------
    pd.DataFrame
        Combined disaggregated media results with columns:
        date, wc_mon, brand, decomp_lvl1..N, spend, impressions, clicks, decomp, kpi_{label1}, kpi_{label2}, kpi_total
    """
    if len(wrappers_with_disagg) < 2:
        raise ValueError("At least 2 models required for combined export")

    # Generate disaggregated media results for each model
    all_disagg_dfs = []
    labels = []
    all_include_cols = set()  # Collect include_cols from all models dynamically

    for wrapper, label, disagg_config in wrappers_with_disagg:
        mapped_df, granular_name_cols, date_col, weight_col, include_cols = disagg_config

        # Track all include_cols across models for merge key exclusion
        if include_cols:
            all_include_cols.update(include_cols)

        df_disagg = generate_media_results_disaggregated(
            wrapper=wrapper,
            config=wrapper.config,
            granular_df=mapped_df,
            granular_name_cols=granular_name_cols,
            date_col=date_col,
            model_channel_col="_model_channel",
            weight_col=weight_col,
            include_cols=include_cols,
            brand=brand
        )

        # Find the KPI column name
        target_col = wrapper.config.data.target_column
        kpi_col = f"kpi_{target_col}"

        # Rename KPI column to kpi_{label}
        df_disagg = df_disagg.rename(columns={kpi_col: f"kpi_{label}"})

        all_disagg_dfs.append(df_disagg)
        labels.append(label)

    # Identify key columns for merging (exclude value columns that should be taken from first df only)
    first_df = all_disagg_dfs[0]
    # Base value columns that are always numeric
    base_value_cols = {'spend', 'impressions', 'clicks', 'weight_pct', 'raw_weight', 'adstocked_weight'}
    # Combine with dynamically collected include_cols (e.g., instore_revenue, online_revenue)
    value_cols = base_value_cols | all_include_cols
    # Also exclude any kpi columns
    all_cols_to_exclude = value_cols | {c for c in first_df.columns if c.startswith('kpi_')}
    key_cols = [c for c in first_df.columns if c not in all_cols_to_exclude]

    # Merge all DataFrames on key columns
    result = all_disagg_dfs[0]
    for i, df in enumerate(all_disagg_dfs[1:], 1):
        label = labels[i]
        # Only keep key columns and the kpi column from subsequent dfs
        # Don't include spend/impressions/clicks - those are identical and come from first df
        df_subset = df[key_cols + [f"kpi_{label}"]]
        result = result.merge(df_subset, on=key_cols, how='outer')

    # Fill NaN with 0 for KPI columns (when a model doesn't have a particular row)
    for label in labels:
        kpi_col = f"kpi_{label}"
        if kpi_col in result.columns:
            result[kpi_col] = result[kpi_col].fillna(0)

    # Calculate kpi_total
    kpi_cols = [f"kpi_{label}" for label in labels]
    result['kpi_total'] = result[kpi_cols].sum(axis=1)

    # Sort by date and decomp
    if 'date' in result.columns:
        result = result.sort_values(['date', 'decomp'])

    return _reorder_decomp_columns(result)


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

    return _reorder_decomp_columns(pd.DataFrame(rows))
