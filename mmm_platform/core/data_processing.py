"""
Data processing module for two-file upload system.

Handles parsing media data (long format) and merging with other data (wide format).
"""

import pandas as pd
import numpy as np
from typing import Optional


def parse_media_file(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Parse media file and detect channel level columns.

    Args:
        df: Raw media data DataFrame

    Returns:
        Tuple of (cleaned DataFrame, list of level column names)
    """
    df = df.copy()

    # Detect level columns (media_channel_lvl1, media_channel_lvl2, etc.)
    level_cols = []
    for i in range(1, 6):  # Support up to 5 levels
        for col in df.columns:
            if f"lvl{i}" in col.lower() or f"level{i}" in col.lower() or f"level_{i}" in col.lower():
                level_cols.append(col)
                break

    # Clean spend values (remove commas, handle dashes)
    spend_cols = [col for col in df.columns if 'spend' in col.lower()]
    for col in spend_cols:
        if df[col].dtype == 'object':
            # Remove commas and convert dashes to 0
            df[col] = df[col].astype(str).str.replace(',', '', regex=False)
            df[col] = df[col].replace('-', '0')
            df[col] = df[col].replace('', '0')
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Clean impressions values
    impr_cols = [col for col in df.columns if 'impression' in col.lower() or 'impr' in col.lower()]
    for col in impr_cols:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.replace(',', '', regex=False)
            df[col] = df[col].replace('-', '0')
            df[col] = df[col].replace('', '0')
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Clean clicks values
    click_cols = [col for col in df.columns if 'click' in col.lower()]
    for col in click_cols:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.replace(',', '', regex=False)
            df[col] = df[col].replace('-', '0')
            df[col] = df[col].replace('', '0')
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    return df, level_cols


def get_unique_channel_combinations(df: pd.DataFrame, level_cols: list[str]) -> pd.DataFrame:
    """Get unique combinations of channel levels for mapping UI.

    Args:
        df: Media data DataFrame
        level_cols: List of level column names

    Returns:
        DataFrame with unique combinations and suggested variable names
    """
    if not level_cols:
        return pd.DataFrame()

    # Get unique combinations
    unique_combos = df[level_cols].drop_duplicates().reset_index(drop=True)

    # Sort by level columns
    unique_combos = unique_combos.sort_values(level_cols).reset_index(drop=True)

    # Add suggested variable names based on lvl1 (sanitized)
    def suggest_name(row):
        lvl1 = str(row[level_cols[0]]).lower()
        # Sanitize: remove spaces, special chars
        name = lvl1.replace(' ', '_').replace('-', '_')
        name = ''.join(c for c in name if c.isalnum() or c == '_')
        return f"{name}_spend"

    unique_combos['variable_name'] = unique_combos.apply(suggest_name, axis=1)

    return unique_combos


def aggregate_media_data(
    df: pd.DataFrame,
    level_cols: list[str],
    mapping: dict[tuple, str],
    date_col: str = 'date',
    dayfirst: bool = True
) -> pd.DataFrame:
    """Aggregate media data based on user mapping.

    Args:
        df: Media data DataFrame (cleaned)
        level_cols: List of level column names
        mapping: Dict mapping channel tuples to variable names
                 e.g., {('Google Search', 'Brand', 'Prospecting'): 'gs_brand_spend', ...}
        date_col: Name of date column
        dayfirst: Whether dates are day-first format

    Returns:
        Wide-format DataFrame with one column per variable name (for spend, impressions, clicks)
    """
    df = df.copy()

    # Parse dates
    df[date_col] = pd.to_datetime(df[date_col], dayfirst=dayfirst)

    # Create tuple key for each row
    def get_combo_key(row):
        return tuple(row[col] for col in level_cols)

    df['_combo_key'] = df.apply(get_combo_key, axis=1)

    # Map to variable names (exclude empty mappings)
    df['_var_name'] = df['_combo_key'].map(mapping)

    # Filter out unmapped rows (empty string or None)
    df = df[df['_var_name'].notna() & (df['_var_name'] != '')]

    if df.empty:
        return pd.DataFrame()

    # Detect metric columns
    spend_col = next((col for col in df.columns if 'spend' in col.lower()), None)
    impr_col = next((col for col in df.columns if 'impression' in col.lower() or 'impr' in col.lower()), None)
    click_col = next((col for col in df.columns if 'click' in col.lower()), None)

    # Aggregate by date and variable name
    agg_cols = {}
    if spend_col:
        agg_cols[spend_col] = 'sum'
    if impr_col:
        agg_cols[impr_col] = 'sum'
    if click_col:
        agg_cols[click_col] = 'sum'

    if not agg_cols:
        return pd.DataFrame()

    grouped = df.groupby([date_col, '_var_name']).agg(agg_cols).reset_index()

    # Pivot to wide format
    result_dfs = []

    if spend_col:
        spend_wide = grouped.pivot(index=date_col, columns='_var_name', values=spend_col)
        spend_wide.columns = [f"{col}" for col in spend_wide.columns]  # Already has _spend suffix
        result_dfs.append(spend_wide)

    if impr_col:
        impr_wide = grouped.pivot(index=date_col, columns='_var_name', values=impr_col)
        # Replace _spend with _impr in column names
        impr_wide.columns = [col.replace('_spend', '_impr').replace('_Spend', '_impr')
                            if '_spend' in col.lower() else f"{col}_impr"
                            for col in impr_wide.columns]
        result_dfs.append(impr_wide)

    if click_col:
        click_wide = grouped.pivot(index=date_col, columns='_var_name', values=click_col)
        # Replace _spend with _clicks in column names
        click_wide.columns = [col.replace('_spend', '_clicks').replace('_Spend', '_clicks')
                             if '_spend' in col.lower() else f"{col}_clicks"
                             for col in click_wide.columns]
        result_dfs.append(click_wide)

    # Combine all metrics
    if result_dfs:
        result = pd.concat(result_dfs, axis=1)
        result = result.reset_index()
        result = result.fillna(0)
        return result

    return pd.DataFrame()


def merge_datasets(
    media_df: pd.DataFrame,
    other_df: pd.DataFrame,
    date_col: str = 'date',
    dayfirst: bool = True
) -> tuple[pd.DataFrame, dict]:
    """Merge aggregated media data with other data on date.

    Args:
        media_df: Aggregated media data (wide format)
        other_df: Other data (KPIs, promos, etc.)
        date_col: Name of date column
        dayfirst: Whether dates are day-first format

    Returns:
        Tuple of (merged DataFrame, merge stats dict)
    """
    media_df = media_df.copy()
    other_df = other_df.copy()

    # Detect date column in other_df
    other_date_col = None
    for col in other_df.columns:
        if 'date' in col.lower() or 'time' in col.lower():
            other_date_col = col
            break

    if other_date_col is None:
        other_date_col = other_df.columns[0]  # Assume first column is date

    # Parse dates
    media_df[date_col] = pd.to_datetime(media_df[date_col], dayfirst=dayfirst)
    other_df[other_date_col] = pd.to_datetime(other_df[other_date_col], dayfirst=dayfirst)

    # Rename other date column to match media date column
    if other_date_col != date_col:
        other_df = other_df.rename(columns={other_date_col: date_col})

    # Get date ranges for stats
    media_dates = set(media_df[date_col])
    other_dates = set(other_df[date_col])

    stats = {
        'media_date_range': (media_df[date_col].min(), media_df[date_col].max()),
        'other_date_range': (other_df[date_col].min(), other_df[date_col].max()),
        'media_only_dates': len(media_dates - other_dates),
        'other_only_dates': len(other_dates - media_dates),
        'common_dates': len(media_dates & other_dates),
    }

    # Merge on date (outer join to keep all dates)
    merged = pd.merge(media_df, other_df, on=date_col, how='outer')
    merged = merged.sort_values(date_col).reset_index(drop=True)

    # Fill missing values with 0 for numeric columns
    numeric_cols = merged.select_dtypes(include=[np.number]).columns
    merged[numeric_cols] = merged[numeric_cols].fillna(0)

    return merged, stats


def detect_date_column(df: pd.DataFrame) -> Optional[str]:
    """Detect the date column in a DataFrame.

    Args:
        df: DataFrame to search

    Returns:
        Name of detected date column, or None
    """
    for col in df.columns:
        if 'date' in col.lower() or 'time' in col.lower():
            return col
    return None


def detect_metric_columns(df: pd.DataFrame) -> dict[str, Optional[str]]:
    """Detect spend, impressions, and clicks columns.

    Args:
        df: DataFrame to search

    Returns:
        Dict with keys 'spend', 'impressions', 'clicks' and column name values
    """
    result = {'spend': None, 'impressions': None, 'clicks': None}

    for col in df.columns:
        col_lower = col.lower()
        if 'spend' in col_lower and result['spend'] is None:
            result['spend'] = col
        elif ('impression' in col_lower or 'impr' in col_lower) and result['impressions'] is None:
            result['impressions'] = col
        elif 'click' in col_lower and result['clicks'] is None:
            result['clicks'] = col

    return result
