"""
Schema validation for export column schemas.

Handles detecting drift between saved schemas and actual data columns,
and applying schemas to DataFrames for export.
"""

import copy
from typing import List, Tuple, Optional
import pandas as pd


def validate_schema_against_data(
    schema_columns: List[str],
    data_columns: List[str]
) -> Tuple[List[str], List[str], List[str]]:
    """
    Validate schema columns against actual data columns.

    Parameters
    ----------
    schema_columns : List[str]
        Column names defined in schema (original_name values).
    data_columns : List[str]
        Actual column names in the data.

    Returns
    -------
    Tuple[List[str], List[str], List[str]]
        (matched, new_in_data, removed_from_data)
        - matched: columns present in both schema and data
        - new_in_data: columns in data but not in schema
        - removed_from_data: columns in schema but not in data
    """
    schema_set = set(schema_columns)
    data_set = set(data_columns)

    matched = list(schema_set & data_set)
    new_in_data = list(data_set - schema_set)
    removed = list(schema_set - data_set)

    return matched, new_in_data, removed


def determine_drift_severity(
    matched: List[str],
    new_columns: List[str],
    removed_columns: List[str]
) -> str:
    """
    Determine severity of schema drift.

    Parameters
    ----------
    matched : List[str]
        Columns that match between schema and data.
    new_columns : List[str]
        Columns in data but not in schema.
    removed_columns : List[str]
        Columns in schema but not in data.

    Returns
    -------
    str
        "none", "minor", or "major"
        - none: all columns match
        - minor: some new or removed columns but mostly matching
        - major: schema is completely different or most columns don't match
    """
    if not new_columns and not removed_columns:
        return "none"

    # Major drift: no matches or most columns don't match
    if len(matched) == 0:
        return "major"

    total_in_schema = len(matched) + len(removed_columns)
    if total_in_schema > 0 and len(removed_columns) > len(matched):
        return "major"

    # Minor drift: some new or removed columns but mostly matching
    return "minor"


def apply_schema_to_dataframe(
    df: pd.DataFrame,
    schema_columns: List[dict]
) -> pd.DataFrame:
    """
    Apply a column schema to a DataFrame.

    Filters, reorders, and renames columns based on the schema configuration.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    schema_columns : List[dict]
        List of column schema entries, each with keys:
        - original_name: str
        - display_name: Optional[str] (None means use original)
        - visible: bool
        - order: int

    Returns
    -------
    pd.DataFrame
        DataFrame with columns renamed, filtered, and reordered.
    """
    if not schema_columns:
        return df

    # Build visible columns sorted by order
    visible = sorted(
        [c for c in schema_columns if c.get("visible", True)],
        key=lambda x: x.get("order", 0)
    )

    # Filter to columns that exist in the DataFrame
    result_columns = []
    rename_map = {}

    for col_config in visible:
        orig = col_config["original_name"]
        if orig in df.columns:
            result_columns.append(orig)
            display = col_config.get("display_name")
            if display and display != orig:
                rename_map[orig] = display

    # Apply filter and order
    result_df = df[result_columns].copy()

    # Apply renaming
    if rename_map:
        result_df = result_df.rename(columns=rename_map)

    return result_df


def generate_default_schema_from_dataframe(
    df: pd.DataFrame,
    dataset_name: str
) -> dict:
    """
    Generate a default schema from a DataFrame's columns.

    Creates a schema with all columns visible in their original order.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to generate schema from.
    dataset_name : str
        Name of the dataset (for reference, not used in schema).

    Returns
    -------
    dict
        DatasetColumnSchema dict with all columns visible in original order.
    """
    columns = []
    for i, col in enumerate(df.columns):
        columns.append({
            "original_name": col,
            "display_name": None,  # None means use original
            "visible": True,
            "order": i
        })

    return {"columns": columns}


def auto_fix_schema_drift(
    schema: dict,
    validation_results: dict
) -> dict:
    """
    Auto-fix schema drift by adding new columns and removing missing ones.

    Parameters
    ----------
    schema : dict
        Original ExportColumnSchema dict.
    validation_results : dict
        Dict keyed by dataset_name with validation results containing:
        - matched: list of matched column names
        - new: list of new column names
        - removed: list of removed column names

    Returns
    -------
    dict
        Updated schema with drift fixed.
    """
    updated_schema = schema.copy()

    for dataset_name, result in validation_results.items():
        if dataset_name not in updated_schema:
            continue

        dataset_schema = updated_schema.get(dataset_name, {})
        if not dataset_schema:
            continue

        columns = dataset_schema.get("columns", [])
        if not columns:
            continue

        # Remove columns that no longer exist in data
        removed_set = set(result.get("removed", []))
        columns = [c for c in columns if c["original_name"] not in removed_set]

        # Add new columns at the end
        new_cols = result.get("new", [])
        if new_cols:
            max_order = max((c.get("order", 0) for c in columns), default=-1)
            for i, new_col in enumerate(new_cols):
                columns.append({
                    "original_name": new_col,
                    "display_name": None,
                    "visible": True,
                    "order": max_order + 1 + i
                })

        updated_schema[dataset_name] = {"columns": columns}

    return updated_schema


def validate_full_schema(
    schema: dict,
    df_decomps: Optional[pd.DataFrame],
    df_media: Optional[pd.DataFrame],
    df_fit: Optional[pd.DataFrame]
) -> dict:
    """
    Validate a complete export schema against all three datasets.

    Parameters
    ----------
    schema : dict
        ExportColumnSchema dict.
    df_decomps : Optional[pd.DataFrame]
        Decomps stacked DataFrame.
    df_media : Optional[pd.DataFrame]
        Media results DataFrame.
    df_fit : Optional[pd.DataFrame]
        Actual vs fitted DataFrame.

    Returns
    -------
    dict
        Validation results keyed by dataset name, each containing:
        - matched: list of matched columns
        - new: list of new columns
        - removed: list of removed columns
        - severity: "none", "minor", or "major"
    """
    validation_results = {}

    datasets = [
        ("decomps_stacked", df_decomps),
        ("media_results", df_media),
        ("actual_vs_fitted", df_fit)
    ]

    for name, df in datasets:
        if df is None:
            continue

        dataset_schema = schema.get(name, {})
        if not dataset_schema or not dataset_schema.get("columns"):
            # No schema for this dataset - treat as fully compatible
            validation_results[name] = {
                "matched": list(df.columns),
                "new": [],
                "removed": [],
                "severity": "none"
            }
            continue

        schema_cols = [c["original_name"] for c in dataset_schema["columns"]]
        data_cols = list(df.columns)

        matched, new_cols, removed_cols = validate_schema_against_data(
            schema_cols, data_cols
        )
        severity = determine_drift_severity(matched, new_cols, removed_cols)

        validation_results[name] = {
            "matched": matched,
            "new": new_cols,
            "removed": removed_cols,
            "severity": severity
        }

    return validation_results


def has_any_drift(validation_results: dict) -> bool:
    """
    Check if any dataset has schema drift.

    Parameters
    ----------
    validation_results : dict
        Results from validate_full_schema.

    Returns
    -------
    bool
        True if any dataset has drift (severity != "none").
    """
    return any(
        result.get("severity", "none") != "none"
        for result in validation_results.values()
    )


def has_major_drift(validation_results: dict) -> bool:
    """
    Check if any dataset has major schema drift.

    Parameters
    ----------
    validation_results : dict
        Results from validate_full_schema.

    Returns
    -------
    bool
        True if any dataset has major drift.
    """
    return any(
        result.get("severity") == "major"
        for result in validation_results.values()
    )


def resolve_disagg_schema(
    base_schema: Optional[dict],
    disagg_schema: Optional[dict],
    disagg_df: pd.DataFrame
) -> dict:
    """
    Resolve disaggregated schema with inheritance from base schema.

    If disagg_schema is explicitly set, use it (override mode).
    If disagg_schema is None, inherit from base_schema and auto-extend
    with any extra columns present in the disaggregated DataFrame.

    Parameters
    ----------
    base_schema : Optional[dict]
        Base schema dict (e.g., decomps_stacked schema).
    disagg_schema : Optional[dict]
        Explicit disagg schema dict, or None to inherit.
    disagg_df : pd.DataFrame
        The disaggregated DataFrame (used to find extra columns).

    Returns
    -------
    dict
        Resolved schema dict with columns list.
    """
    # If explicit disagg schema is set, use it directly (override)
    if disagg_schema and disagg_schema.get("columns"):
        return disagg_schema

    # Inherit from base schema
    if base_schema and base_schema.get("columns"):
        inherited = copy.deepcopy(base_schema)
    else:
        inherited = {"columns": []}

    # Find columns in disagg DataFrame that aren't in the base schema
    base_cols = {c["original_name"] for c in inherited.get("columns", [])}
    max_order = max(
        (c.get("order", 0) for c in inherited.get("columns", [])),
        default=-1
    )

    # Add extra columns from disagg DataFrame (preserving their order)
    extra_col_index = 0
    for col in disagg_df.columns:
        if col not in base_cols:
            inherited["columns"].append({
                "original_name": col,
                "display_name": None,
                "visible": True,
                "order": max_order + 1 + extra_col_index,
                "is_disagg_only": True  # Mark as disagg-specific for UI
            })
            extra_col_index += 1

    return inherited


def get_disagg_only_columns(schema: dict) -> List[str]:
    """
    Get list of column names that are disagg-only (not in base schema).

    Parameters
    ----------
    schema : dict
        Schema dict with columns list.

    Returns
    -------
    List[str]
        List of column names marked as disagg-only.
    """
    return [
        c["original_name"]
        for c in schema.get("columns", [])
        if c.get("is_disagg_only", False)
    ]
