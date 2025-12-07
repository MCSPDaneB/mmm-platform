"""
Export page for MMM Platform.

Generates CSV files in specific formats for upload to external visualization platforms.
"""

import io
import zipfile
import streamlit as st
import pandas as pd
from datetime import datetime
from typing import Optional


# =============================================================================
# Column Schema UI Components
# =============================================================================

def _show_schema_selector(client: str, model_path: str = None, key_suffix: str = "") -> Optional[dict]:
    """
    Show schema selection dropdown with client and model-level schemas.

    Parameters
    ----------
    client : str
        Client name.
    model_path : str, optional
        Path to model (for model-level overrides).
    key_suffix : str
        Suffix for widget keys to avoid conflicts.

    Returns
    -------
    Optional[dict]
        Selected schema dict, or None for no schema (original columns).
    """
    from mmm_platform.model.persistence import (
        list_export_schemas,
        load_export_schema,
        load_model_schema_override
    )

    # Get client-level schemas
    client_schemas = list_export_schemas(client) if client else []

    # Get model-level override if exists
    model_override = None
    if model_path:
        model_override = load_model_schema_override(model_path)

    # Build options
    options = ["None (Original Columns)"]
    schema_map = {}

    if model_override:
        label = f"[Model] {model_override.get('name', 'Custom Override')}"
        options.append(label)
        schema_map[label] = ("model", model_override)

    for schema_meta in client_schemas:
        label = f"[Client] {schema_meta['name']}"
        if schema_meta.get("description"):
            label += f" - {schema_meta['description'][:30]}..."
        options.append(label)
        schema_map[label] = ("client", schema_meta["path"])

    # Show selector
    selected = st.selectbox(
        "Column Schema",
        options=options,
        key=f"export_schema_selector{key_suffix}",
        help="Select a schema to rename/reorder/filter columns, or use original columns"
    )

    if selected == "None (Original Columns)":
        return None

    # Load selected schema
    schema_type, schema_ref = schema_map[selected]
    if schema_type == "model":
        return schema_ref
    else:
        return load_export_schema(schema_ref)


def _show_schema_validation_ui(
    schema: dict,
    df_decomps: pd.DataFrame,
    df_media: pd.DataFrame,
    df_fit: pd.DataFrame,
    key_suffix: str = "",
    session_key: str = "export_selected_schema"
) -> tuple[bool, Optional[dict]]:
    """
    Validate schema against actual data and show drift warnings.

    Parameters
    ----------
    schema : dict
        Schema to validate.
    df_decomps : pd.DataFrame
        Decomps DataFrame.
    df_media : pd.DataFrame
        Media results DataFrame.
    df_fit : pd.DataFrame
        Actual vs fitted DataFrame.
    key_suffix : str
        Suffix for widget keys.
    session_key : str
        Session state key for storing the schema.

    Returns
    -------
    tuple[bool, Optional[dict]]
        (should_proceed, updated_schema)
        - should_proceed: True if user wants to continue with schema
        - updated_schema: The schema to use (may be auto-fixed or None)
    """
    from mmm_platform.analysis.schema_validation import (
        validate_full_schema,
        has_any_drift,
        has_major_drift,
        auto_fix_schema_drift
    )

    validation_results = validate_full_schema(schema, df_decomps, df_media, df_fit)

    if not has_any_drift(validation_results):
        st.success("Schema validated successfully - all columns match")
        return True, schema

    # Show drift warnings
    if has_major_drift(validation_results):
        st.error("Major schema drift detected - most columns don't match")
    else:
        st.warning("Schema drift detected - some columns have changed")

    for name, result in validation_results.items():
        if result["severity"] == "none":
            continue

        severity_icon = "X" if result["severity"] == "major" else "!"
        with st.expander(f"[{severity_icon}] {name.replace('_', ' ').title()}: {result['severity'].upper()} drift"):
            if result["new"]:
                st.write("**New columns** (in data but not in schema):")
                for col in sorted(result["new"]):
                    st.write(f"  + {col}")
            if result["removed"]:
                st.write("**Removed columns** (in schema but not in data):")
                for col in sorted(result["removed"]):
                    st.write(f"  - {col}")

            if result["severity"] == "major":
                st.error("Major drift - schema may not be compatible. Consider creating a new schema.")

    # Options for user
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Auto-fix Schema", key=f"schema_auto_fix{key_suffix}", help="Add new columns, remove missing ones"):
            fixed_schema = auto_fix_schema_drift(schema, validation_results)
            st.session_state[session_key] = fixed_schema
            st.success("Schema auto-fixed!")
            st.rerun()
    with col2:
        if st.button("Proceed Anyway", key=f"schema_proceed_anyway{key_suffix}"):
            return True, schema
    with col3:
        if st.button("Use Original Columns", key=f"schema_use_original{key_suffix}"):
            st.session_state[session_key] = None
            return True, None

    return False, schema


def _show_column_editor(
    dataset_name: str,
    df: pd.DataFrame,
    current_schema: Optional[dict],
    key_suffix: str = ""
) -> Optional[dict]:
    """
    Show column editor for a single dataset.

    Parameters
    ----------
    dataset_name : str
        Name of dataset (decomps_stacked, media_results, actual_vs_fitted).
    df : pd.DataFrame
        DataFrame with columns to edit.
    current_schema : Optional[dict]
        Current DatasetColumnSchema dict, or None.
    key_suffix : str
        Suffix for widget keys.

    Returns
    -------
    Optional[dict]
        Updated DatasetColumnSchema dict if changes applied, None otherwise.
    """
    # Use session state to persist schema changes between renders
    schema_key = f"working_schema_{dataset_name}{key_suffix}"

    # Initialize from current_schema if not in session state
    if schema_key not in st.session_state:
        st.session_state[schema_key] = current_schema

    working_schema = st.session_state[schema_key]

    # Build editor data from working schema (which has normalized order)
    if working_schema and working_schema.get("columns"):
        # Use existing schema, adding any new columns
        schema_col_names = {c["original_name"] for c in working_schema["columns"]}
        editor_data = []

        for col in working_schema["columns"]:
            if col["original_name"] in df.columns:
                editor_data.append({
                    "Original": col["original_name"],
                    "Display Name": col.get("display_name") or col["original_name"],
                    "Visible": col.get("visible", True),
                    "Order": col.get("order", len(editor_data))
                })

        # Add new columns not in schema
        for col in df.columns:
            if col not in schema_col_names:
                editor_data.append({
                    "Original": col,
                    "Display Name": col,
                    "Visible": True,
                    "Order": len(editor_data)
                })
    else:
        # Generate from DataFrame columns
        editor_data = [
            {
                "Original": col,
                "Display Name": col,
                "Visible": True,
                "Order": i
            }
            for i, col in enumerate(df.columns)
        ]

    # Sort editor data by order before displaying (so rows appear in correct order)
    editor_data_sorted = sorted(editor_data, key=lambda x: x["Order"])
    editor_df = pd.DataFrame(editor_data_sorted)

    # Track original order for each row (by Original column name) to detect changes
    original_orders = {row["Original"]: row["Order"] for row in editor_data_sorted}

    # Info text about ordering
    st.caption("*Change order numbers, then click 'Apply Reorder' to see changes.*")

    # Show data editor
    edited_df = st.data_editor(
        editor_df,
        column_config={
            "Original": st.column_config.TextColumn("Original Column", disabled=True),
            "Display Name": st.column_config.TextColumn("Display Name"),
            "Visible": st.column_config.CheckboxColumn("Include"),
            "Order": st.column_config.NumberColumn(
                "Order",
                min_value=0,
                step=1,
                help="Lower number = earlier position"
            )
        },
        hide_index=True,
        height=min(400, 50 + len(editor_data) * 35),
        key=f"column_editor_{dataset_name}{key_suffix}"
    )

    # Reorder button - normalizes order, saves, and applies changes
    if st.button("Apply Reorder", key=f"reorder_{dataset_name}{key_suffix}"):
        # Normalize order: handle insertions correctly
        # When user changes order (e.g., 6→3), that item goes BEFORE existing items at position 3
        rows_with_priority = []
        for _, row in edited_df.iterrows():
            orig_order = original_orders.get(row["Original"], row["Order"])
            new_order = row["Order"]
            # If order changed, use fractional value to insert BEFORE existing items at that position
            if new_order != orig_order:
                sort_key = new_order - 0.5  # Insert before, not after
            else:
                sort_key = new_order
            rows_with_priority.append((sort_key, row))

        # Sort by priority key
        rows_with_priority.sort(key=lambda x: x[0])

        # Convert back to schema format with sequential order values
        columns = []
        for i, (_, row) in enumerate(rows_with_priority):
            columns.append({
                "original_name": row["Original"],
                "display_name": row["Display Name"] if row["Display Name"] != row["Original"] else None,
                "visible": row["Visible"],
                "order": i
            })

        # Save to session state
        result_schema = {"columns": columns}
        st.session_state[schema_key] = result_schema

        # Also update parent schema and apply changes based on context
        if "_single" in key_suffix:
            parent_schema = st.session_state.get("export_selected_schema") or {}
            parent_schema[dataset_name] = result_schema
            st.session_state["export_selected_schema"] = parent_schema
            _apply_schema_to_exports()
        elif "_combined" in key_suffix:
            parent_schema = st.session_state.get("combined_selected_schema") or {}
            parent_schema[dataset_name] = result_schema
            st.session_state["combined_selected_schema"] = parent_schema
            _apply_schema_to_combined_exports()

        st.rerun()

    # Return current edited state (not saved to session state on every render)
    # This allows "Save & Apply Schema" to capture current edits
    columns = []
    for _, row in edited_df.iterrows():
        columns.append({
            "original_name": row["Original"],
            "display_name": row["Display Name"] if row["Display Name"] != row["Original"] else None,
            "visible": row["Visible"],
            "order": int(row["Order"])
        })

    return {"columns": columns}


def _show_column_editor_section_single(brand: str, model_path: str):
    """Expander section for single model column schema editing."""
    from mmm_platform.analysis.schema_validation import resolve_disagg_schema
    from mmm_platform.model.persistence import save_export_schema, save_model_schema_override

    # Get working schema from session state
    working_schema = st.session_state.get("export_selected_schema") or {}

    # Check if disaggregated files are available
    has_disagg = (
        st.session_state.get("export_df_decomps_disagg_original") is not None or
        st.session_state.get("export_df_media_disagg_original") is not None
    )

    # Toggle between base and disaggregated files
    if has_disagg:
        schema_mode = st.radio(
            "Edit schema for",
            options=["Base Files", "Disaggregated Files"],
            horizontal=True,
            key="schema_edit_mode_single"
        )
    else:
        schema_mode = "Base Files"

    if schema_mode == "Base Files":
        # Tabs for base datasets
        tab1, tab2, tab3 = st.tabs(["Decomps Stacked", "Media Results", "Actual vs Fitted"])

        with tab1:
            df_for_editor = st.session_state.get("export_df_decomps_original")
            if df_for_editor is not None:
                current_decomps_schema = working_schema.get("decomps_stacked", {})
                updated_decomps = _show_column_editor("decomps_stacked", df_for_editor, current_decomps_schema, "_single")
                if updated_decomps:
                    # Check if schema actually changed
                    if updated_decomps != working_schema.get("decomps_stacked"):
                        working_schema["decomps_stacked"] = updated_decomps
                        st.session_state["export_selected_schema"] = working_schema
                        _apply_schema_to_exports()  # Auto-apply changes
                        st.rerun()

        with tab2:
            df_for_editor = st.session_state.get("export_df_media_original")
            if df_for_editor is not None:
                current_media_schema = working_schema.get("media_results", {})
                updated_media = _show_column_editor("media_results", df_for_editor, current_media_schema, "_single")
                if updated_media:
                    # Check if schema actually changed
                    if updated_media != working_schema.get("media_results"):
                        working_schema["media_results"] = updated_media
                        st.session_state["export_selected_schema"] = working_schema
                        _apply_schema_to_exports()  # Auto-apply changes
                        st.rerun()

        with tab3:
            df_for_editor = st.session_state.get("export_df_fit_original")
            if df_for_editor is not None:
                current_fit_schema = working_schema.get("actual_vs_fitted", {})
                updated_fit = _show_column_editor("actual_vs_fitted", df_for_editor, current_fit_schema, "_single")
                if updated_fit:
                    # Check if schema actually changed
                    if updated_fit != working_schema.get("actual_vs_fitted"):
                        working_schema["actual_vs_fitted"] = updated_fit
                        st.session_state["export_selected_schema"] = working_schema
                        _apply_schema_to_exports()  # Auto-apply changes
                        st.rerun()

    else:
        # Tabs for disaggregated datasets
        tab1, tab2 = st.tabs(["Decomps Stacked (Disagg)", "Media Results (Disagg)"])

        with tab1:
            df_disagg = st.session_state.get("export_df_decomps_disagg_original")
            if df_disagg is not None:
                explicit_disagg_schema = working_schema.get("decomps_stacked_disagg")
                if not explicit_disagg_schema:
                    st.info("Inheriting from base schema. Edit below to override.")
                resolved_schema = resolve_disagg_schema(
                    working_schema.get("decomps_stacked", {}),
                    explicit_disagg_schema,
                    df_disagg
                )
                updated_disagg = _show_column_editor("decomps_stacked_disagg", df_disagg, resolved_schema, "_single_disagg")
                if updated_disagg:
                    # Check if schema actually changed
                    if updated_disagg != working_schema.get("decomps_stacked_disagg"):
                        working_schema["decomps_stacked_disagg"] = updated_disagg
                        st.session_state["export_selected_schema"] = working_schema
                        _apply_schema_to_exports()  # Auto-apply changes
                        st.rerun()
            else:
                st.info("No disaggregated decomps data available")

        with tab2:
            df_disagg = st.session_state.get("export_df_media_disagg_original")
            if df_disagg is not None:
                explicit_disagg_schema = working_schema.get("media_results_disagg")
                if not explicit_disagg_schema:
                    st.info("Inheriting from base schema. Edit below to override.")
                resolved_schema = resolve_disagg_schema(
                    working_schema.get("media_results", {}),
                    explicit_disagg_schema,
                    df_disagg
                )
                updated_disagg = _show_column_editor("media_results_disagg", df_disagg, resolved_schema, "_single_disagg")
                if updated_disagg:
                    # Check if schema actually changed
                    if updated_disagg != working_schema.get("media_results_disagg"):
                        working_schema["media_results_disagg"] = updated_disagg
                        st.session_state["export_selected_schema"] = working_schema
                        _apply_schema_to_exports()  # Auto-apply changes
                        st.rerun()
            else:
                st.info("No disaggregated media data available")

    # Save & Apply Section
    st.markdown("---")

    # Generate default schema name with timestamp if not already set
    from datetime import datetime as dt
    default_schema_name = working_schema.get("name", "")
    if not default_schema_name:
        default_schema_name = f"Export Schema {dt.now().strftime('%Y%m%d_%H%M%S')}"

    # Schema name and save options
    save_col1, save_col2 = st.columns([3, 1])
    with save_col1:
        schema_name = st.text_input(
            "Schema Name",
            value=default_schema_name,
            key="save_schema_name_single",
            placeholder="e.g., Standard BI Export"
        )
    with save_col2:
        save_level = st.radio(
            "Save to",
            options=["Client", "Model"],
            key="save_schema_level_single",
            horizontal=True
        )

    schema_desc = st.text_input(
        "Description (optional)",
        value=working_schema.get("description", ""),
        key="save_schema_desc_single",
        placeholder="Brief description of this schema"
    )

    # Buttons
    col_save, col_reset = st.columns(2)
    with col_save:
        if st.button("Save & Apply Schema", key="save_apply_schema_single", type="primary"):
            if not schema_name:
                st.warning("Please enter a schema name")
            else:
                # Auto-resolve disaggregated schemas before saving
                # If decomps disagg data exists but no explicit schema, resolve and include it
                df_decomps_disagg = st.session_state.get("export_df_decomps_disagg_original")
                if df_decomps_disagg is not None and not working_schema.get("decomps_stacked_disagg"):
                    resolved = resolve_disagg_schema(
                        working_schema.get("decomps_stacked", {}),
                        None,
                        df_decomps_disagg
                    )
                    if resolved and resolved.get("columns"):
                        working_schema["decomps_stacked_disagg"] = resolved

                # If media disagg data exists but no explicit schema, resolve and include it
                df_media_disagg = st.session_state.get("export_df_media_disagg_original")
                if df_media_disagg is not None and not working_schema.get("media_results_disagg"):
                    resolved = resolve_disagg_schema(
                        working_schema.get("media_results", {}),
                        None,
                        df_media_disagg
                    )
                    if resolved and resolved.get("columns"):
                        working_schema["media_results_disagg"] = resolved

                # Build schema to save
                schema_to_save = working_schema.copy()
                schema_to_save["name"] = schema_name
                schema_to_save["description"] = schema_desc if schema_desc else None

                # Save to database
                if save_level == "Client" and brand:
                    save_export_schema(brand, schema_to_save)
                    st.success(f"Saved schema '{schema_name}' at client level")
                elif model_path:
                    save_model_schema_override(model_path, schema_to_save)
                    st.success(f"Saved schema '{schema_name}' as model override")
                else:
                    st.error("Unable to save: no client or model path available")

                # Apply to exports
                st.session_state["export_selected_schema"] = schema_to_save
                _apply_schema_to_exports()

    with col_reset:
        if st.button("Reset to Original", key="reset_columns_single"):
            st.session_state["export_selected_schema"] = None
            # Clear working schemas from session state
            for key in list(st.session_state.keys()):
                if key.startswith("working_schema_"):
                    del st.session_state[key]
            # Restore original DataFrames
            if "export_df_decomps_original" in st.session_state:
                st.session_state["export_df_decomps"] = st.session_state["export_df_decomps_original"]
            if "export_df_media_original" in st.session_state:
                st.session_state["export_df_media"] = st.session_state["export_df_media_original"]
            if "export_df_fit_original" in st.session_state:
                st.session_state["export_df_fit"] = st.session_state["export_df_fit_original"]
            st.success("Reset to original columns!")
            st.rerun()


def _show_column_editor_section_combined(brand: str, model_path: str):
    """Expander section for combined model column schema editing."""
    from mmm_platform.analysis.schema_validation import resolve_disagg_schema
    from mmm_platform.model.persistence import save_export_schema, save_model_schema_override

    # Get working schema from session state
    working_schema = st.session_state.get("combined_selected_schema") or {}

    # Check if disaggregated files are available
    has_disagg = (
        st.session_state.get("combined_df_decomps_disagg_original") is not None or
        st.session_state.get("combined_df_media_disagg_original") is not None
    )

    # Toggle between base and disaggregated files
    if has_disagg:
        schema_mode = st.radio(
            "Edit schema for",
            options=["Base Files", "Disaggregated Files"],
            horizontal=True,
            key="schema_edit_mode_combined"
        )
    else:
        schema_mode = "Base Files"

    if schema_mode == "Base Files":
        # Tabs for base datasets
        tab1, tab2, tab3 = st.tabs(["Decomps Stacked", "Media Results", "Actual vs Fitted"])

        with tab1:
            df_for_editor = st.session_state.get("combined_df_decomps_original")
            if df_for_editor is not None:
                current_decomps_schema = working_schema.get("decomps_stacked", {})
                updated_decomps = _show_column_editor("decomps_stacked", df_for_editor, current_decomps_schema, "_combined")
                if updated_decomps:
                    # Check if schema actually changed
                    if updated_decomps != working_schema.get("decomps_stacked"):
                        working_schema["decomps_stacked"] = updated_decomps
                        st.session_state["combined_selected_schema"] = working_schema
                        _apply_schema_to_combined_exports()  # Auto-apply changes
                        st.rerun()

        with tab2:
            df_for_editor = st.session_state.get("combined_df_media_original")
            if df_for_editor is not None:
                current_media_schema = working_schema.get("media_results", {})
                updated_media = _show_column_editor("media_results", df_for_editor, current_media_schema, "_combined")
                if updated_media:
                    # Check if schema actually changed
                    if updated_media != working_schema.get("media_results"):
                        working_schema["media_results"] = updated_media
                        st.session_state["combined_selected_schema"] = working_schema
                        _apply_schema_to_combined_exports()  # Auto-apply changes
                        st.rerun()

        with tab3:
            df_for_editor = st.session_state.get("combined_df_fit_original")
            if df_for_editor is not None:
                current_fit_schema = working_schema.get("actual_vs_fitted", {})
                updated_fit = _show_column_editor("actual_vs_fitted", df_for_editor, current_fit_schema, "_combined")
                if updated_fit:
                    # Check if schema actually changed
                    if updated_fit != working_schema.get("actual_vs_fitted"):
                        working_schema["actual_vs_fitted"] = updated_fit
                        st.session_state["combined_selected_schema"] = working_schema
                        _apply_schema_to_combined_exports()  # Auto-apply changes
                        st.rerun()

    else:
        # Tabs for disaggregated datasets
        tab1, tab2 = st.tabs(["Decomps Stacked (Disagg)", "Media Results (Disagg)"])

        with tab1:
            df_disagg = st.session_state.get("combined_df_decomps_disagg_original")
            if df_disagg is not None:
                explicit_disagg_schema = working_schema.get("decomps_stacked_disagg")
                if not explicit_disagg_schema:
                    st.info("Inheriting from base schema. Edit below to override.")
                resolved_schema = resolve_disagg_schema(
                    working_schema.get("decomps_stacked", {}),
                    explicit_disagg_schema,
                    df_disagg
                )
                updated_disagg = _show_column_editor("decomps_stacked_disagg", df_disagg, resolved_schema, "_combined_disagg")
                if updated_disagg:
                    # Check if schema actually changed
                    if updated_disagg != working_schema.get("decomps_stacked_disagg"):
                        working_schema["decomps_stacked_disagg"] = updated_disagg
                        st.session_state["combined_selected_schema"] = working_schema
                        _apply_schema_to_combined_exports()  # Auto-apply changes
                        st.rerun()
            else:
                st.info("No disaggregated decomps data available")

        with tab2:
            df_disagg = st.session_state.get("combined_df_media_disagg_original")
            if df_disagg is not None:
                explicit_disagg_schema = working_schema.get("media_results_disagg")
                if not explicit_disagg_schema:
                    st.info("Inheriting from base schema. Edit below to override.")
                resolved_schema = resolve_disagg_schema(
                    working_schema.get("media_results", {}),
                    explicit_disagg_schema,
                    df_disagg
                )
                updated_disagg = _show_column_editor("media_results_disagg", df_disagg, resolved_schema, "_combined_disagg")
                if updated_disagg:
                    # Check if schema actually changed
                    if updated_disagg != working_schema.get("media_results_disagg"):
                        working_schema["media_results_disagg"] = updated_disagg
                        st.session_state["combined_selected_schema"] = working_schema
                        _apply_schema_to_combined_exports()  # Auto-apply changes
                        st.rerun()
            else:
                st.info("No disaggregated media data available")

    # Save & Apply Section
    st.markdown("---")

    # Generate default schema name with timestamp if not already set
    from datetime import datetime as dt
    default_schema_name = working_schema.get("name", "")
    if not default_schema_name:
        default_schema_name = f"Export Schema {dt.now().strftime('%Y%m%d_%H%M%S')}"

    # Schema name and save options
    save_col1, save_col2 = st.columns([3, 1])
    with save_col1:
        schema_name = st.text_input(
            "Schema Name",
            value=default_schema_name,
            key="save_schema_name_combined",
            placeholder="e.g., Standard BI Export"
        )
    with save_col2:
        save_level = st.radio(
            "Save to",
            options=["Client", "Model"],
            key="save_schema_level_combined",
            horizontal=True
        )

    schema_desc = st.text_input(
        "Description (optional)",
        value=working_schema.get("description", ""),
        key="save_schema_desc_combined",
        placeholder="Brief description of this schema"
    )

    # Buttons
    col_save, col_reset = st.columns(2)
    with col_save:
        if st.button("Save & Apply Schema", key="save_apply_schema_combined", type="primary"):
            if not schema_name:
                st.warning("Please enter a schema name")
            else:
                # Auto-resolve disaggregated schemas before saving
                # If decomps disagg data exists but no explicit schema, resolve and include it
                df_decomps_disagg = st.session_state.get("combined_df_decomps_disagg_original")
                if df_decomps_disagg is not None and not working_schema.get("decomps_stacked_disagg"):
                    resolved = resolve_disagg_schema(
                        working_schema.get("decomps_stacked", {}),
                        None,
                        df_decomps_disagg
                    )
                    if resolved and resolved.get("columns"):
                        working_schema["decomps_stacked_disagg"] = resolved

                # If media disagg data exists but no explicit schema, resolve and include it
                df_media_disagg = st.session_state.get("combined_df_media_disagg_original")
                if df_media_disagg is not None and not working_schema.get("media_results_disagg"):
                    resolved = resolve_disagg_schema(
                        working_schema.get("media_results", {}),
                        None,
                        df_media_disagg
                    )
                    if resolved and resolved.get("columns"):
                        working_schema["media_results_disagg"] = resolved

                # Build schema to save
                schema_to_save = working_schema.copy()
                schema_to_save["name"] = schema_name
                schema_to_save["description"] = schema_desc if schema_desc else None

                # Save to database
                if save_level == "Client" and brand:
                    save_export_schema(brand, schema_to_save)
                    st.success(f"Saved schema '{schema_name}' at client level")
                elif model_path:
                    save_model_schema_override(model_path, schema_to_save)
                    st.success(f"Saved schema '{schema_name}' as model override")
                else:
                    st.error("Unable to save: no client or model path available")

                # Apply to exports
                st.session_state["combined_selected_schema"] = schema_to_save
                _apply_schema_to_combined_exports()

    with col_reset:
        if st.button("Reset to Original", key="reset_columns_combined"):
            st.session_state["combined_selected_schema"] = None
            # Clear working schemas from session state
            for key in list(st.session_state.keys()):
                if key.startswith("working_schema_"):
                    del st.session_state[key]
            # Restore original DataFrames
            if "combined_df_decomps_original" in st.session_state:
                st.session_state["combined_df_decomps"] = st.session_state["combined_df_decomps_original"]
            if "combined_df_media_original" in st.session_state:
                st.session_state["combined_df_media"] = st.session_state["combined_df_media_original"]
            if "combined_df_fit_original" in st.session_state:
                st.session_state["combined_df_fit"] = st.session_state["combined_df_fit_original"]
            st.success("Reset to original columns!")
            st.rerun()


def _apply_schema_to_exports():
    """
    Apply the selected schema to the export DataFrames in session state.

    Always starts from the ORIGINAL DataFrames to ensure correct application
    even when schema is applied multiple times.
    """
    from mmm_platform.analysis.schema_validation import (
        apply_schema_to_dataframe,
        resolve_disagg_schema
    )

    schema = st.session_state.get("export_selected_schema")
    if not schema:
        return

    # Apply to base datasets - always start from ORIGINAL
    datasets = [
        ("export_df_decomps", "export_df_decomps_original", "decomps_stacked"),
        ("export_df_media", "export_df_media_original", "media_results"),
        ("export_df_fit", "export_df_fit_original", "actual_vs_fitted"),
    ]

    for session_key, original_key, schema_key in datasets:
        original_df = st.session_state.get(original_key)
        if original_df is not None:
            dataset_schema = schema.get(schema_key, {})
            if dataset_schema and dataset_schema.get("columns"):
                st.session_state[session_key] = apply_schema_to_dataframe(
                    original_df,
                    dataset_schema["columns"]
                )
            else:
                # No schema for this dataset - use original as-is
                st.session_state[session_key] = original_df.copy()

    # Apply to disaggregated versions with inheritance - always start from ORIGINAL
    disagg_mappings = [
        ("export_df_decomps_disagg", "export_df_decomps_disagg_original", "decomps_stacked", "decomps_stacked_disagg"),
        ("export_df_media_disagg", "export_df_media_disagg_original", "media_results", "media_results_disagg"),
    ]

    for session_key, original_key, base_schema_key, disagg_schema_key in disagg_mappings:
        original_df = st.session_state.get(original_key)
        if original_df is not None:
            # Resolve schema with inheritance
            resolved_schema = resolve_disagg_schema(
                base_schema=schema.get(base_schema_key, {}),
                disagg_schema=schema.get(disagg_schema_key),
                disagg_df=original_df
            )
            if resolved_schema and resolved_schema.get("columns"):
                st.session_state[session_key] = apply_schema_to_dataframe(
                    original_df,
                    resolved_schema["columns"]
                )
            else:
                # No schema - use original as-is
                st.session_state[session_key] = original_df.copy()


def _apply_schema_to_combined_exports():
    """
    Apply the selected schema to the combined export DataFrames in session state.

    Always starts from the ORIGINAL DataFrames to ensure correct application
    even when schema is applied multiple times.
    """
    from mmm_platform.analysis.schema_validation import (
        apply_schema_to_dataframe,
        resolve_disagg_schema
    )

    schema = st.session_state.get("combined_selected_schema")
    if not schema:
        return

    # Apply to base datasets - always start from ORIGINAL
    datasets = [
        ("combined_df_decomps", "combined_df_decomps_original", "decomps_stacked"),
        ("combined_df_media", "combined_df_media_original", "media_results"),
        ("combined_df_fit", "combined_df_fit_original", "actual_vs_fitted"),
    ]

    for session_key, original_key, schema_key in datasets:
        original_df = st.session_state.get(original_key)
        if original_df is not None:
            dataset_schema = schema.get(schema_key, {})
            if dataset_schema and dataset_schema.get("columns"):
                st.session_state[session_key] = apply_schema_to_dataframe(
                    original_df,
                    dataset_schema["columns"]
                )
            else:
                # No schema for this dataset - use original as-is
                st.session_state[session_key] = original_df.copy()

    # Apply to disaggregated versions with inheritance - always start from ORIGINAL
    disagg_mappings = [
        ("combined_df_decomps_disagg", "combined_df_decomps_disagg_original", "decomps_stacked", "decomps_stacked_disagg"),
        ("combined_df_media_disagg", "combined_df_media_disagg_original", "media_results", "media_results_disagg"),
    ]

    for session_key, original_key, base_schema_key, disagg_schema_key in disagg_mappings:
        original_df = st.session_state.get(original_key)
        if original_df is not None:
            # Resolve schema with inheritance
            resolved_schema = resolve_disagg_schema(
                base_schema=schema.get(base_schema_key, {}),
                disagg_schema=schema.get(disagg_schema_key),
                disagg_df=original_df
            )
            if resolved_schema and resolved_schema.get("columns"):
                st.session_state[session_key] = apply_schema_to_dataframe(
                    original_df,
                    resolved_schema["columns"]
                )
            else:
                # No schema - use original as-is
                st.session_state[session_key] = original_df.copy()


def show():
    """Show the export page."""
    # Lazy import to avoid circular import deadlock
    from mmm_platform.analysis.export import (
        generate_decomps_stacked,
        generate_media_results,
        generate_actual_vs_fitted,
        generate_combined_decomps_stacked,
        generate_combined_media_results,
        generate_combined_actual_vs_fitted,
    )
    from mmm_platform.model.persistence import ModelPersistence, list_clients
    from mmm_platform.model.mmm import MMMWrapper

    st.title("Export Data")
    st.caption("Generate CSV files for upload to external visualization platforms")

    # Reset button in a small column to keep it unobtrusive
    reset_col, _ = st.columns([1, 5])
    with reset_col:
        if st.button("Reset Export", help="Clear all export settings and start fresh"):
            # Clear export-related session state keys
            keys_to_clear = [k for k in list(st.session_state.keys())
                             if k.startswith(('export_', 'combined_', 'granular_'))]
            for key in keys_to_clear:
                del st.session_state[key]
            st.rerun()

    # Model Selection Section
    st.subheader("Select Models")

    # Filter controls
    filter_col1, filter_col2 = st.columns(2)

    # Client filter
    clients = list_clients()
    with filter_col1:
        if clients:
            client_options = ["All Clients"] + clients

            # Sync widget state FROM active_client before render
            active = st.session_state.get("active_client")
            widget_value = active if active and active in clients else "All Clients"
            st.session_state.export_client_filter = widget_value

            def on_export_client_change():
                val = st.session_state.export_client_filter
                st.session_state.active_client = None if val == "All Clients" else val

            selected_client = st.selectbox(
                "Filter by Client",
                options=client_options,
                key="export_client_filter",
                on_change=on_export_client_change,
                help="Show models for a specific client"
            )

            client_filter = "all" if selected_client == "All Clients" else selected_client
        else:
            st.info("No clients found. Please run some models first.")
            st.stop()

    with filter_col2:
        export_fav_filter = st.selectbox(
            "Filter",
            options=["All", "Favorites only", "Non-favorites"],
            key="export_favorite_filter",
            help="Filter by favorite status"
        )

    # Get saved models (client-aware)
    saved_models = ModelPersistence.list_saved_models(client=client_filter)

    # Apply favorites filter
    if export_fav_filter == "Favorites only":
        saved_models = [m for m in saved_models if m.get("is_favorite", False)]
    elif export_fav_filter == "Non-favorites":
        saved_models = [m for m in saved_models if not m.get("is_favorite", False)]

    # Sort favorites to the top, then by created_at descending (most recent first)
    saved_models.sort(key=lambda m: (not m.get("is_favorite", False), m.get("created_at", "")), reverse=True)

    if not saved_models:
        st.warning("No fitted models found. Please run and save at least 1 model first.")
        st.stop()

    # Create options for multiselect with rich labels
    model_options = {}
    for model in saved_models:
        r2 = model.get("r2")
        r2_str = f"R²={r2:.3f}" if r2 is not None else "R²=N/A"
        n_channels = model.get("n_channels", "?")
        created = model.get("created_at", "")[:10]
        name = model.get("config_name", "Unknown")
        model_client = model.get("client", "")
        is_favorite = model.get("is_favorite", False)

        # Add star icon for favorites
        fav_icon = "⭐ " if is_favorite else ""

        # Show client in label when viewing all clients
        if client_filter == "all" and model_client:
            option_label = f"{fav_icon}[{model_client}] {name} ({created}) - {n_channels} channels, {r2_str}"
        else:
            option_label = f"{fav_icon}{name} ({created}) - {n_channels} channels, {r2_str}"
        model_options[option_label] = {
            "path": model["path"],
            "name": name,
            "client": model_client
        }

    selected_options = st.multiselect(
        "Select models to export",
        options=list(model_options.keys()),
        default=[],
        help="Select 1 model for single export, or 2+ models for combined export",
        key="export_model_multiselect"
    )

    # Handle based on selection count
    if len(selected_options) == 0:
        st.info("Please select 1 or more models to export.")
        st.stop()

    st.markdown("---")

    if len(selected_options) == 1:
        # Single model export
        model_info = model_options[selected_options[0]]
        try:
            with st.spinner("Loading model..."):
                wrapper = ModelPersistence.load(model_info["path"], MMMWrapper)
        except Exception as e:
            st.error(f"Error loading model: {e}")
            st.stop()

        _show_single_model_export(
            wrapper=wrapper,
            config=wrapper.config,
            brand=model_info["client"] or "unknown",
            model_path=model_info["path"],
            generate_decomps_stacked=generate_decomps_stacked,
            generate_media_results=generate_media_results,
            generate_actual_vs_fitted=generate_actual_vs_fitted
        )
    else:
        # Combined model export (2+ models)
        st.warning(
            "⚠️ **KPIs must be in the same units to be summable** (e.g., both in revenue £). "
            "The kpi_total column is the sum of all model contributions."
        )

        _show_combined_model_export(
            selected_options=selected_options,
            model_options=model_options,
            client_filter=client_filter,
            ModelPersistence=ModelPersistence,
            MMMWrapper=MMMWrapper,
            generate_combined_decomps_stacked=generate_combined_decomps_stacked,
            generate_combined_media_results=generate_combined_media_results,
            generate_combined_actual_vs_fitted=generate_combined_actual_vs_fitted
        )


def _show_disaggregation_ui(wrapper, config, brand: str, model_path: str = None, key_suffix: str = ""):
    """
    Show disaggregation UI for a single model.

    Parameters
    ----------
    wrapper : MMMWrapper
        The fitted model wrapper
    config : ModelConfig
        The model configuration
    brand : str
        Brand name for exports
    model_path : str, optional
        Path to model directory (for loading/saving disaggregation configs)
    key_suffix : str
        Suffix to add to all widget keys (for multiple instances)

    Returns
    -------
    tuple or None
        (mapped_df, granular_name_cols, date_col, weight_col, include_cols) if ready, None otherwise
        granular_name_cols is a list of column names forming the entity identifier
        include_cols is a list of additional column names to include from granular file
    """
    from mmm_platform.model.persistence import (
        load_disaggregation_configs,
        load_disaggregation_weights,
        save_disaggregation_config,
        validate_disaggregation_config
    )

    model_channels = [ch.name for ch in config.channels]

    # Load saved disaggregation configs if model_path provided
    saved_configs = []
    if model_path:
        saved_configs = load_disaggregation_configs(model_path)

    # Show saved configs selector if any exist
    selected_saved_config = None
    if saved_configs:
        st.markdown("#### Saved Configurations")

        config_options = ["+ Create New"] + [
            f"{c['name']} ({c['created_at'][:16].replace('T', ' ')})" for c in saved_configs
        ]

        selected_option = st.selectbox(
            "Use saved configuration",
            options=config_options,
            key=f"disagg_saved_config{key_suffix}",
            help="Select a saved configuration or create a new one"
        )

        if selected_option != "+ Create New":
            # Find the selected config
            idx = config_options.index(selected_option) - 1  # -1 for "Create New"
            selected_saved_config = saved_configs[idx]

            # Validate against current model channels
            is_valid, error_msg = validate_disaggregation_config(selected_saved_config, model_channels)

            if not is_valid:
                st.error(f"⚠️ Saved config is invalid: {error_msg}")
                st.info("Please create a new configuration or edit this one.")
                selected_saved_config = None
            else:
                st.success(f"Using saved config: **{selected_saved_config['name']}**")

                # Load saved weights DataFrame if available
                granular_df_key = f"granular_df{key_suffix}"
                if granular_df_key not in st.session_state:
                    saved_weights = load_disaggregation_weights(model_path, selected_saved_config['id'])
                    if saved_weights is not None:
                        st.session_state[granular_df_key] = saved_weights
                        st.info(f"Loaded saved weights: {len(saved_weights):,} rows")

                # Explicitly set widget session state values from saved config
                # This is needed because Streamlit ignores `default` after first render
                include_cols_key = f"granular_include_cols{key_suffix}"
                saved_include_cols = selected_saved_config.get('include_columns', [])
                if saved_include_cols:
                    st.session_state[include_cols_key] = saved_include_cols

                with st.expander("Config details"):
                    st.write(f"**Entity columns:** {', '.join(selected_saved_config['granular_name_cols'])}")
                    st.write(f"**Date column:** {selected_saved_config['date_column']}")
                    st.write(f"**Weight column:** {selected_saved_config['weight_column']}")
                    include_cols_saved = selected_saved_config.get('include_columns', [])
                    if include_cols_saved:
                        st.write(f"**Include columns:** {', '.join(include_cols_saved)}")
                    st.write(f"**Mappings:** {len(selected_saved_config['entity_to_channel_mapping'])} entities")

    # File upload
    granular_file = st.file_uploader(
        "Upload granular mapping file",
        type=["csv", "xlsx"],
        help="CSV or Excel file with granular-level data (e.g., placement-level spend/attribution)",
        key=f"granular_file_uploader{key_suffix}"
    )

    # Session state key for caching the DataFrame
    granular_df_key = f"granular_df{key_suffix}"

    if granular_file is None:
        # Check if we have a previously loaded DataFrame in session state
        if granular_df_key not in st.session_state:
            return None
        granular_df = st.session_state[granular_df_key]
        st.info(f"Using cached data: {len(granular_df):,} rows × {len(granular_df.columns)} columns")
    else:
        # Load the file
        try:
            if granular_file.name.endswith('.csv'):
                granular_df = pd.read_csv(granular_file)
            else:
                granular_df = pd.read_excel(granular_file)

            # Store in session state for persistence across button clicks
            st.session_state[granular_df_key] = granular_df
            st.success(f"Loaded {len(granular_df):,} rows × {len(granular_df.columns)} columns")
        except Exception as e:
            st.error(f"Error loading file: {e}")
            return None

    # Get all columns for mapping (works with both fresh upload and cached data)
    all_columns = granular_df.columns.tolist()
    numeric_columns = granular_df.select_dtypes(include=['number']).columns.tolist()

    # Column mapping UI
    st.markdown("#### Column Mapping")

    col1, col2 = st.columns(2)

    # Pre-populate from saved config if available
    default_entity_cols = selected_saved_config['granular_name_cols'] if selected_saved_config else []
    default_entity_cols = [c for c in default_entity_cols if c in all_columns]  # Filter to valid columns

    default_date_col = selected_saved_config['date_column'] if selected_saved_config else None
    default_date_idx = all_columns.index(default_date_col) if default_date_col and default_date_col in all_columns else 0

    default_weight_col = selected_saved_config['weight_column'] if selected_saved_config else None
    weight_options = numeric_columns if numeric_columns else all_columns
    default_weight_idx = weight_options.index(default_weight_col) if default_weight_col and default_weight_col in weight_options else 0

    with col1:
        granular_name_cols = st.multiselect(
            "Entity Identifier Column(s)",
            options=all_columns,
            default=default_entity_cols,
            help="Select one or more columns to form the unique entity identifier (e.g., Region + Store_ID)",
            key=f"granular_name_cols{key_suffix}"
        )

        date_col_granular = st.selectbox(
            "Date Column",
            options=all_columns,
            index=default_date_idx,
            help="Column containing dates",
            key=f"granular_date_col{key_suffix}"
        )

    with col2:
        weight_col = st.selectbox(
            "Weight Column",
            options=weight_options,
            index=default_weight_idx,
            help="Numeric column to use for proportional allocation (e.g., spend, impressions, attribution)",
            key=f"granular_weight_col{key_suffix}"
        )

    # Columns to Include multiselect
    # Exclude already selected columns from options
    used_cols = set(granular_name_cols + [date_col_granular, weight_col])
    available_include_cols = [c for c in all_columns if c not in used_cols]

    # Initialize session state for include_cols if not already set
    # (saved config values are set earlier when config is selected)
    include_cols_key = f"granular_include_cols{key_suffix}"
    if include_cols_key not in st.session_state:
        # Use saved config values or empty list as initial value
        default_include_cols = selected_saved_config.get('include_columns', []) if selected_saved_config else []
        default_include_cols = [c for c in default_include_cols if c in available_include_cols]
        st.session_state[include_cols_key] = default_include_cols

    include_cols = st.multiselect(
        "Columns to Include",
        options=available_include_cols,
        # No default parameter - use session state exclusively to avoid warning
        help="Additional columns to carry through to output (e.g., impressions, clicks). These will appear in the disaggregated media results.",
        key=include_cols_key
    )

    # Validate entity columns selected
    if not granular_name_cols:
        st.warning("Please select at least one entity identifier column.")
        return None

    # Model channel mapping
    st.markdown("#### Map Granular Entities to Model Channels")
    st.caption(
        "For each unique entity in your granular file, select which model channel it maps to. "
        "Leave as '-- Not Mapped --' to exclude from disaggregation."
    )

    # Get unique entity combinations (composite key from multiple columns)
    unique_entities_df = granular_df[granular_name_cols].drop_duplicates()

    # Create composite key for display and mapping
    def make_composite_key(row):
        return " | ".join(str(row[col]) for col in granular_name_cols)

    unique_entities_df["_composite_key"] = unique_entities_df.apply(make_composite_key, axis=1)
    unique_granular = unique_entities_df["_composite_key"].tolist()

    # Create mapping DataFrame
    mapping_options = ["-- Not Mapped --"] + model_channels

    # Initialize mapping from saved config or session state
    mapping_key = f"granular_mapping{key_suffix}"
    if mapping_key not in st.session_state:
        if selected_saved_config:
            # Use saved config mapping
            st.session_state[mapping_key] = selected_saved_config.get('entity_to_channel_mapping', {}).copy()
        else:
            st.session_state[mapping_key] = {}

    # CSV Download/Upload for bulk mapping
    st.markdown("**Bulk Mapping Options**")
    csv_col1, csv_col2 = st.columns(2)

    with csv_col1:
        # Build template with ALL entities (not just first 50)
        template_data = []
        for granular_val in unique_granular:
            current_mapping = st.session_state[mapping_key].get(str(granular_val), "-- Not Mapped --")
            # Auto-match if not already mapped
            if current_mapping == "-- Not Mapped --" or current_mapping not in mapping_options:
                for ch in model_channels:
                    if ch.lower() in str(granular_val).lower() or str(granular_val).lower() in ch.lower():
                        current_mapping = ch
                        break
                else:
                    current_mapping = "-- Not Mapped --"
            template_data.append({
                "entity": str(granular_val),
                "model_channel": current_mapping
            })

        template_df = pd.DataFrame(template_data)

        # Build valid channels reference CSV
        channels_df = pd.DataFrame({
            "model_channel": mapping_options
        })

        # Create ZIP with both files
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("mapping_template.csv", template_df.to_csv(index=False))
            zf.writestr("valid_channels.csv", channels_df.to_csv(index=False))
        zip_buffer.seek(0)

        st.download_button(
            "Download Mapping Template",
            data=zip_buffer.getvalue(),
            file_name="mapping_template.zip",
            mime="application/zip",
            key=f"download_mapping_template{key_suffix}",
            help=f"ZIP with mapping template ({len(unique_granular)} entities) + valid channels reference"
        )

    with csv_col2:
        uploaded_mapping = st.file_uploader(
            "Upload Mapping CSV",
            type=["csv"],
            key=f"mapping_upload{key_suffix}",
            help="Upload completed mapping CSV (columns: entity, model_channel)"
        )

    if uploaded_mapping is not None:
        try:
            mapping_csv = pd.read_csv(uploaded_mapping)
            if "entity" in mapping_csv.columns and "model_channel" in mapping_csv.columns:
                imported_count = 0
                invalid_channels = []
                for _, row in mapping_csv.iterrows():
                    entity = str(row["entity"])
                    channel = str(row["model_channel"])
                    if channel in mapping_options:
                        st.session_state[mapping_key][entity] = channel
                        imported_count += 1
                    elif channel and channel != "nan":
                        invalid_channels.append(channel)

                if imported_count > 0:
                    st.success(f"Imported {imported_count} mappings from CSV")
                if invalid_channels:
                    st.warning(f"Skipped invalid channels: {set(invalid_channels)}")
            else:
                st.error("CSV must have 'entity' and 'model_channel' columns")
        except Exception as e:
            st.error(f"Error reading CSV: {e}")

    # Build mapping table data for all entities
    mapping_data = []
    for granular_val in unique_granular:
        # Check saved mapping first
        if str(granular_val) in st.session_state[mapping_key]:
            saved_mapping = st.session_state[mapping_key][str(granular_val)]
            # Validate the saved mapping is still valid
            if saved_mapping not in mapping_options:
                saved_mapping = "-- Not Mapped --"
        else:
            # Try to auto-match by checking if entity name contains channel name
            saved_mapping = "-- Not Mapped --"
            for ch in model_channels:
                if ch.lower() in str(granular_val).lower() or str(granular_val).lower() in ch.lower():
                    saved_mapping = ch
                    break

        mapping_data.append({
            "Entity": str(granular_val),
            "Model Channel": saved_mapping,
        })

    mapping_df = pd.DataFrame(mapping_data)

    # Wrap data_editor in form to prevent rerender on each change
    with st.form(key=f"mapping_form{key_suffix}"):
        edited_mapping = st.data_editor(
            mapping_df,
            column_config={
                "Entity": st.column_config.TextColumn("Entity", disabled=True),
                "Model Channel": st.column_config.SelectboxColumn(
                    "Model Channel",
                    options=mapping_options,
                    help="Select which model channel this maps to"
                ),
            },
            hide_index=True,
            width="stretch",
            height=400,
            key=f"granular_mapping_editor{key_suffix}",
        )

        apply_btn = st.form_submit_button("Apply Mappings", type="primary")

    st.caption("*Click Apply Mappings to update the summary below. Mappings are automatically used when you Prepare Export Files.*")

    # Always sync editor state to session state (ensures Prepare Export uses latest)
    for _, row in edited_mapping.iterrows():
        st.session_state[mapping_key][row["Entity"]] = row["Model Channel"]

    # Rerun only on Apply button to refresh the summary display below
    if apply_btn:
        st.rerun()

    # Create composite key column in granular_df for mapping
    granular_df["_composite_key"] = granular_df.apply(make_composite_key, axis=1)

    # Apply mapping to granular DataFrame using composite key
    granular_df["_model_channel"] = granular_df["_composite_key"].map(
        lambda x: st.session_state[mapping_key].get(str(x), "-- Not Mapped --")
    )

    # Filter out unmapped rows
    mapped_df = granular_df[granular_df["_model_channel"] != "-- Not Mapped --"].copy()

    # Deduplicate: aggregate by key columns to prevent row duplication in disaggregation
    # Key columns are entity identifiers + date + channel mapping
    dedup_key_cols = granular_name_cols + [date_col_granular, "_model_channel"]
    # Identify numeric columns to sum (spend, impressions, revenue, etc.)
    numeric_cols = mapped_df.select_dtypes(include=['number']).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in dedup_key_cols]

    if numeric_cols and len(mapped_df) > 0:
        # Aggregate: sum numeric values for duplicate key combinations
        agg_dict = {col: 'sum' for col in numeric_cols}
        # Ensure include_cols are numeric and in the aggregation
        for ic in include_cols:
            if ic in mapped_df.columns:
                # Convert to numeric in case they're stored as strings
                mapped_df[ic] = pd.to_numeric(mapped_df[ic], errors='coerce').fillna(0)
                if ic not in agg_dict and ic not in dedup_key_cols:
                    agg_dict[ic] = 'sum'
        mapped_df = mapped_df.groupby(dedup_key_cols, as_index=False).agg(agg_dict)
    elif len(mapped_df) > 0:
        # No numeric columns - just drop duplicates
        mapped_df = mapped_df.drop_duplicates(subset=dedup_key_cols)

    # Show mapping summary
    st.markdown("#### Mapping Summary")
    mapped_count = len(mapped_df)
    total_count = len(granular_df)
    mapped_channels_count = mapped_df["_model_channel"].nunique()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Mapped Rows", f"{mapped_count:,} / {total_count:,}")
    with col2:
        st.metric("Mapped Channels", f"{mapped_channels_count} / {len(model_channels)}")
    with col3:
        unmapped_channels = set(model_channels) - set(mapped_df["_model_channel"].unique())
        st.metric("Unmapped Channels", len(unmapped_channels))

    if unmapped_channels:
        with st.expander("Unmapped Model Channels"):
            st.write("These channels have no granular mapping and will be output at channel level:")
            for ch in sorted(unmapped_channels):
                st.write(f"- {ch}")

    # Save configuration button
    if model_path and mapped_count > 0:
        st.markdown("---")
        st.markdown("#### Save Configuration")
        save_col1, save_col2 = st.columns([3, 1])

        with save_col1:
            config_name = st.text_input(
                "Configuration name",
                value=selected_saved_config['name'] if selected_saved_config else f"Disagg by {weight_col}",
                key=f"disagg_config_name{key_suffix}",
                placeholder="e.g., Placements by Spend"
            )

        with save_col2:
            if st.button("💾 Save Config", key=f"save_disagg_config{key_suffix}"):
                if config_name:
                    new_config = {
                        "id": selected_saved_config['id'] if selected_saved_config else None,
                        "name": config_name,
                        "created_at": datetime.now().isoformat(),
                        "granular_name_cols": granular_name_cols,
                        "date_column": date_col_granular,
                        "weight_column": weight_col,
                        "include_columns": include_cols,
                        "entity_to_channel_mapping": dict(st.session_state[mapping_key]),
                    }
                    save_disaggregation_config(model_path, new_config, weighting_df=mapped_df)
                    st.success(f"Saved configuration: {config_name}")
                    st.rerun()
                else:
                    st.warning("Please enter a configuration name")

    if mapped_count > 0:
        return (mapped_df, granular_name_cols, date_col_granular, weight_col, include_cols)
    else:
        st.warning("No rows are mapped to model channels. Please map at least one entity.")
        return None


def _show_single_model_export(
    wrapper,
    config,
    brand: str,
    model_path: str,
    generate_decomps_stacked,
    generate_media_results,
    generate_actual_vs_fitted
):
    """Show single model export UI."""
    from mmm_platform.analysis.export import (
        generate_disaggregated_results,
        generate_decomps_stacked_disaggregated,
        generate_media_results_disaggregated
    )

    # Brand input section
    st.subheader("Export Settings")

    col1, col2 = st.columns([1, 2])
    with col1:
        # Get brand from config or use provided default
        default_brand = config.data.brand if config.data.brand else brand
        brand = st.text_input(
            "Brand Name",
            value=default_brand,
            help="Brand name to include in all export files",
            placeholder="Enter brand name (e.g., bevmo)"
        )

    if not brand:
        st.warning("Please enter a brand name to enable exports.")
        st.stop()

    # Export options
    force_to_actuals = st.checkbox(
        "Force decomposition to actuals",
        value=False,
        help="Absorb residuals into intercept so decomposition sums to actual values instead of fitted",
        key="single_force_to_actuals"
    )

    enable_disagg = st.checkbox(
        "Enable disaggregation",
        value=False,
        help="Split model results to granular level (e.g., placements) using proportional weighting",
        key="enable_disaggregation"
    )

    # Disaggregation settings (if enabled) - BEFORE prepare button
    disagg_config = None
    if enable_disagg:
        st.markdown("---")
        st.subheader("Disaggregation Settings")
        st.caption(
            "Split model results to a more granular level (e.g., placements, campaigns) "
            "using proportional weighting from an uploaded file."
        )
        disagg_config = _show_disaggregation_ui(wrapper, config, brand, model_path=model_path, key_suffix="")

    # Column Schema Selection
    st.markdown("---")
    st.subheader("Column Schema")
    st.caption("Optionally apply a schema to rename, reorder, or filter columns in exports")

    selected_schema = _show_schema_selector(client=brand, model_path=model_path, key_suffix="_single")
    st.session_state["export_selected_schema"] = selected_schema

    st.markdown("---")

    # Prepare button
    if st.button("Prepare Export Files", type="primary", key="single_prepare_btn"):
        with st.spinner("Generating export files..."):
            try:
                # Generate main exports
                st.session_state["export_df_decomps"] = generate_decomps_stacked(wrapper, config, brand, force_to_actuals)
                st.session_state["export_df_media"] = generate_media_results(wrapper, config, brand)
                st.session_state["export_df_fit"] = generate_actual_vs_fitted(wrapper, config, brand)

                # Generate disaggregated results if configured
                if enable_disagg and disagg_config is not None:
                    mapped_df, granular_name_cols, date_col, weight_col, include_cols = disagg_config

                    # Generate disaggregated decomps_stacked
                    st.session_state["export_df_decomps_disagg"] = generate_decomps_stacked_disaggregated(
                        wrapper=wrapper,
                        config=config,
                        granular_df=mapped_df,
                        granular_name_cols=granular_name_cols,
                        date_col=date_col,
                        model_channel_col="_model_channel",
                        weight_col=weight_col,
                        brand=brand,
                        force_to_actuals=force_to_actuals
                    )

                    # Generate disaggregated media_results
                    st.session_state["export_df_media_disagg"] = generate_media_results_disaggregated(
                        wrapper=wrapper,
                        config=config,
                        granular_df=mapped_df,
                        granular_name_cols=granular_name_cols,
                        date_col=date_col,
                        model_channel_col="_model_channel",
                        weight_col=weight_col,
                        include_cols=include_cols,
                        brand=brand
                    )

                    # Legacy: also generate the old disaggregated results format
                    st.session_state["export_df_disagg"] = generate_disaggregated_results(
                        wrapper=wrapper,
                        config=config,
                        granular_df=mapped_df,
                        granular_name_cols=granular_name_cols,
                        date_col=date_col,
                        model_channel_col="_model_channel",
                        weight_col=weight_col,
                        brand=brand
                    )

                    # Validate totals match
                    df_decomps = st.session_state["export_df_decomps"]
                    df_decomps_disagg = st.session_state["export_df_decomps_disagg"]
                    target_col = config.data.target_column
                    kpi_col = f"kpi_{target_col}"

                    # Sum by channel in both
                    channel_names = [ch.name for ch in config.channels]
                    for ch in channel_names:
                        orig_total = df_decomps[df_decomps["decomp"] == ch][kpi_col].sum()
                        disagg_total = df_decomps_disagg[df_decomps_disagg["decomp"] == ch][kpi_col].sum()
                        diff = abs(orig_total - disagg_total)
                        if diff > 0.01:
                            st.warning(f"Validation warning: {ch} totals differ by {diff:.2f}")
                else:
                    st.session_state["export_df_decomps_disagg"] = None
                    st.session_state["export_df_media_disagg"] = None
                    st.session_state["export_df_disagg"] = None

                st.session_state["export_files_ready"] = True
                st.session_state["export_brand"] = brand

                # Validate schema if one is selected
                if st.session_state.get("export_selected_schema"):
                    st.session_state["export_schema_validated"] = False
                else:
                    st.session_state["export_schema_validated"] = True

                st.success("Export files ready!")

            except Exception as e:
                st.error(f"Error generating exports: {e}")
                st.session_state["export_files_ready"] = False

    # Schema validation step (if schema selected and not yet validated)
    if (st.session_state.get("export_files_ready", False)
        and st.session_state.get("export_selected_schema")
        and not st.session_state.get("export_schema_validated", False)):

        st.markdown("---")
        st.subheader("Schema Validation")

        should_proceed, updated_schema = _show_schema_validation_ui(
            st.session_state["export_selected_schema"],
            st.session_state["export_df_decomps"],
            st.session_state["export_df_media"],
            st.session_state["export_df_fit"],
            key_suffix="_single",
            session_key="export_selected_schema"
        )

        if should_proceed:
            st.session_state["export_selected_schema"] = updated_schema
            st.session_state["export_schema_validated"] = True
            # Apply schema to exports
            if updated_schema:
                _apply_schema_to_exports()
            st.rerun()
        else:
            st.info("Please resolve schema validation issues above to proceed.")
            st.stop()

    # Show downloads only if files are ready
    if st.session_state.get("export_files_ready", False):
        st.markdown("---")
        st.subheader("Platform Export Files")

        # Store original DataFrames for column editing (before schema was applied)
        if "export_df_decomps_original" not in st.session_state:
            st.session_state["export_df_decomps_original"] = st.session_state.get("export_df_decomps")
            st.session_state["export_df_media_original"] = st.session_state.get("export_df_media")
            st.session_state["export_df_fit_original"] = st.session_state.get("export_df_fit")
            st.session_state["export_df_decomps_disagg_original"] = st.session_state.get("export_df_decomps_disagg")
            st.session_state["export_df_media_disagg_original"] = st.session_state.get("export_df_media_disagg")

        # Column schema editor expander
        # Track if user has opened the expander (stays open once opened until page refresh)
        if "column_editor_opened_single" not in st.session_state:
            st.session_state["column_editor_opened_single"] = False

        # Check if any working schema exists (indicates user was editing)
        has_working_schema = any(
            k.startswith("working_schema_") and k.endswith("_single")
            for k in st.session_state.keys()
        )
        if has_working_schema:
            st.session_state["column_editor_opened_single"] = True

        with st.expander("Configure Column Schema", expanded=st.session_state["column_editor_opened_single"]):
            st.session_state["column_editor_opened_single"] = True  # Mark as opened once user sees it
            st.caption("Customize column names, order, and visibility before download")
            _show_column_editor_section_single(brand, model_path)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Three columns for the three export types
        col1, col2, col3 = st.columns(3)

        # 1. Decomps Stacked
        with col1:
            st.markdown("### Decomps Stacked")
            df_decomps = st.session_state["export_df_decomps"]
            csv_decomps = df_decomps.to_csv(index=False)

            st.download_button(
                label="Download decomps_stacked.csv",
                data=csv_decomps,
                file_name=f"decomps_stacked_{timestamp}.csv",
                mime="text/csv",
                key="decomps_download"
            )

            # Disaggregated version (if available)
            df_decomps_disagg = st.session_state.get("export_df_decomps_disagg")
            if df_decomps_disagg is not None:
                csv_decomps_disagg = df_decomps_disagg.to_csv(index=False)
                st.download_button(
                    label="Download decomps_stacked_disagg.csv",
                    data=csv_decomps_disagg,
                    file_name=f"decomps_stacked_disagg_{timestamp}.csv",
                    mime="text/csv",
                    key="decomps_disagg_download",
                    help="Disaggregated version with granular-level rows"
                )

            with st.expander("Preview (first 10 rows)"):
                st.dataframe(df_decomps.head(10), width="stretch")

            st.caption(f"{len(df_decomps):,} rows × {len(df_decomps.columns)} columns")
            if df_decomps_disagg is not None:
                st.caption(f"Disagg: {len(df_decomps_disagg):,} rows")

        # 2. Media Results
        with col2:
            st.markdown("### Media Results")
            df_media = st.session_state["export_df_media"]
            csv_media = df_media.to_csv(index=False)

            st.download_button(
                label="Download mmm_media_results.csv",
                data=csv_media,
                file_name=f"mmm_media_results_{timestamp}.csv",
                mime="text/csv",
                key="media_download"
            )

            # Disaggregated version (if available)
            df_media_disagg = st.session_state.get("export_df_media_disagg")
            if df_media_disagg is not None:
                csv_media_disagg = df_media_disagg.to_csv(index=False)
                st.download_button(
                    label="Download media_results_disagg.csv",
                    data=csv_media_disagg,
                    file_name=f"mmm_media_results_disagg_{timestamp}.csv",
                    mime="text/csv",
                    key="media_disagg_download",
                    help="Disaggregated version with granular-level rows and actual spend/impressions/clicks"
                )

            with st.expander("Preview (first 10 rows)"):
                st.dataframe(df_media.head(10), width="stretch")

            st.caption(f"{len(df_media):,} rows × {len(df_media.columns)} columns")
            if df_media_disagg is not None:
                st.caption(f"Disagg: {len(df_media_disagg):,} rows")

        # 3. Actual vs Fitted
        with col3:
            st.markdown("### Actual vs Fitted")
            df_fit = st.session_state["export_df_fit"]
            csv_fit = df_fit.to_csv(index=False)

            st.download_button(
                label="Download actual_vs_fitted.csv",
                data=csv_fit,
                file_name=f"actual_vs_fitted_{timestamp}.csv",
                mime="text/csv",
                key="fit_download"
            )

            with st.expander("Preview (first 10 rows)"):
                st.dataframe(df_fit.head(10), width="stretch")

            st.caption(f"{len(df_fit):,} rows × {len(df_fit.columns)} columns")

        # Disaggregation workings (diagnostic file with weight calculations)
        df_disagg = st.session_state.get("export_df_disagg")
        if df_disagg is not None:
            with st.expander("Show Disaggregation Workings"):
                st.caption("Diagnostic file with weight calculations (weight, weight_adstocked, weight_pct, is_mapped)")
                csv_disagg = df_disagg.to_csv(index=False)

                st.download_button(
                    label="Download disaggregation_workings.csv",
                    data=csv_disagg,
                    file_name=f"disaggregation_workings_{timestamp}.csv",
                    mime="text/csv",
                    key="workings_download"
                )

                st.dataframe(df_disagg.head(20), width="stretch")
                st.caption(f"{len(df_disagg):,} rows × {len(df_disagg.columns)} columns")

        # Download All button
        st.markdown("---")
        st.subheader("Download All Files")

        export_brand = st.session_state.get("export_brand", brand)
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(f"decomps_stacked_{timestamp}.csv", df_decomps.to_csv(index=False))
            zf.writestr(f"mmm_media_results_{timestamp}.csv", df_media.to_csv(index=False))
            zf.writestr(f"actual_vs_fitted_{timestamp}.csv", df_fit.to_csv(index=False))
            # Include disaggregated files if available
            if df_decomps_disagg is not None:
                zf.writestr(f"decomps_stacked_disagg_{timestamp}.csv", df_decomps_disagg.to_csv(index=False))
            if df_media_disagg is not None:
                zf.writestr(f"mmm_media_results_disagg_{timestamp}.csv", df_media_disagg.to_csv(index=False))
            if df_disagg is not None:
                zf.writestr(f"disaggregation_workings_{timestamp}.csv", df_disagg.to_csv(index=False))

        zip_buffer.seek(0)

        st.download_button(
            label="Download All (ZIP)",
            data=zip_buffer.getvalue(),
            file_name=f"{export_brand}_exports_{timestamp}.zip",
            mime="application/zip",
            type="primary",
            key="single_download_all"
        )
        st.caption("Downloads all CSV files in a single ZIP archive")

    # File format documentation
    st.markdown("---")
    with st.expander("File Format Documentation"):
        st.markdown("""
        ### decomps_stacked.csv

        | Column | Description |
        |--------|-------------|
        | date | Date value |
        | wc_mon | Week commencing Monday |
        | brand | Brand name |
        | decomp | Variable name (e.g., "Google_PMAX_spend", "trend") |
        | decomp_lvl1 | Category level 1 |
        | decomp_lvl2 | Category level 2 |
        | kpi_{target} | Contribution value in real units |

        ### mmm_media_results.csv

        | Column | Description |
        |--------|-------------|
        | date | Date value |
        | wc_mon | Week commencing Monday |
        | brand | Brand name |
        | decomp_lvl1 | Category level 1 |
        | decomp_lvl2 | Category level 2 |
        | spend | Spend in real units |
        | impressions | Impressions (placeholder = 0) |
        | clicks | Clicks (placeholder = 0) |
        | decomp | Channel name |
        | kpi_{target} | Contribution value in real units |

        ### actual_vs_fitted.csv

        | Column | Description |
        |--------|-------------|
        | date | Date value |
        | wc_mon | Week commencing Monday |
        | brand | Brand name |
        | kpi_label | KPI column name |
        | actual_fitted | "Actual" or "Fitted" |
        | value | Value in real units |
        """)

    # Category columns info
    if config.category_columns:
        st.markdown("---")
        st.subheader("Category Columns")
        st.markdown("The following category columns are used for grouping (decomp_lvl1, decomp_lvl2, etc.):")

        for i, cat_col in enumerate(config.category_columns):
            st.markdown(f"**decomp_lvl{i + 1}**: {cat_col.name}")
            if cat_col.options:
                st.caption(f"Options: {', '.join(cat_col.options)}")
    else:
        st.info(
            "No category columns configured. Using display names for decomp_lvl1 and decomp_lvl2. "
            "You can add category columns in the Configure Model page."
        )


def _show_combined_model_export(
    selected_options,
    model_options,
    client_filter,
    ModelPersistence,
    MMMWrapper,
    generate_combined_decomps_stacked,
    generate_combined_media_results,
    generate_combined_actual_vs_fitted
):
    """Show combined model export UI for merging 2+ models."""
    from mmm_platform.analysis.export import (
        generate_disaggregated_results,
        generate_decomps_stacked_disaggregated,
        generate_media_results_disaggregated,
        generate_combined_decomps_stacked_disaggregated,
        generate_combined_media_results_disaggregated
    )

    # Model Configuration Section
    st.subheader("Configure Labels")
    st.caption("Set custom labels for each model (used in column names like kpi_online, kpi_offline)")

    # Load models first (so we can get KPI names for default labels)
    try:
        with st.spinner("Loading models..."):
            loaded_wrappers = []
            for option in selected_options:
                model_info = model_options[option]
                wrapper = ModelPersistence.load(model_info["path"], MMMWrapper)
                loaded_wrappers.append((wrapper, model_info["name"], model_info["path"]))
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.stop()

    # Build configuration data using KPI name from loaded models as default label
    config_data = []
    for wrapper, name, path in loaded_wrappers:
        # Use the model's target column (KPI name) as the default label
        default_label = wrapper.config.data.target_column

        config_data.append({
            "Model": name,
            "Path": path,
            "Label": default_label,
        })

    # Display editable configuration
    config_df = pd.DataFrame(config_data)

    edited_config = st.data_editor(
        config_df,
        column_config={
            "Model": st.column_config.TextColumn("Model", disabled=True),
            "Path": None,  # Hide path column
            "Label": st.column_config.TextColumn(
                "Label",
                help="Used in column names (e.g., kpi_online_revenue, kpi_instore_revenue)",
                max_chars=30,
            ),
        },
        hide_index=True,
        width="stretch",
        key="export_config_editor",
    )

    # Validate labels
    labels = edited_config["Label"].tolist()
    if len(labels) != len(set(labels)):
        st.error("Labels must be unique for each model.")
        st.stop()

    if any(not label or not label.strip() for label in labels):
        st.warning("Please enter labels for all models.")
        st.stop()

    # Determine brand from client
    # If filtered by a specific client, use that; otherwise use the first model's client
    if client_filter != "all":
        brand = client_filter
    else:
        # Use client from first selected model
        brand = model_options[selected_options[0]]["client"] or "unknown"

    # Build wrappers_with_labels using already-loaded wrappers
    wrappers_with_labels = []
    for i, row in edited_config.iterrows():
        # Find the wrapper that matches this path
        wrapper = next(w for w, n, p in loaded_wrappers if p == row["Path"])
        wrappers_with_labels.append((wrapper, row["Label"]))

    # Export options
    st.subheader("Export Settings")

    force_to_actuals = st.checkbox(
        "Force decomposition to actuals",
        value=False,
        help="Absorb residuals into intercept so decomposition sums to actual values instead of fitted",
        key="combined_force_to_actuals"
    )

    enable_disagg = st.checkbox(
        "Enable disaggregation",
        value=False,
        help="Split each model's results to granular level (e.g., placements) using proportional weighting",
        key="combined_enable_disaggregation"
    )

    # Disaggregation settings per model (if enabled) - BEFORE prepare button
    disagg_configs = {}
    if enable_disagg:
        st.markdown("---")
        st.subheader("Disaggregation Settings")
        st.caption(
            "Split each model's results independently to granular level (e.g., placements, campaigns) "
            "using proportional weighting from uploaded files."
        )

        for idx, row in edited_config.iterrows():
            wrapper = next(w for w, n, p in loaded_wrappers if p == row["Path"])
            label = row["Label"]
            model_path = row["Path"]
            with st.expander(f"Disaggregation for {label}", expanded=False):
                disagg_config = _show_disaggregation_ui(
                    wrapper=wrapper,
                    config=wrapper.config,
                    brand=brand,
                    model_path=model_path,
                    key_suffix=f"_{idx}_{label}"
                )
                if disagg_config is not None:
                    disagg_configs[label] = (wrapper, disagg_config)

        # Store in session state so it persists when button is clicked
        if disagg_configs:
            st.session_state["combined_disagg_configs"] = disagg_configs

    # Column Schema Selection
    st.markdown("---")
    st.subheader("Column Schema")
    st.caption("Optionally apply a schema to rename, reorder, or filter columns in exports")

    # For combined export, use the first model's path for model-level override lookup
    first_model_path = loaded_wrappers[0][2] if loaded_wrappers else None
    combined_selected_schema = _show_schema_selector(client=brand, model_path=first_model_path, key_suffix="_combined")
    st.session_state["combined_selected_schema"] = combined_selected_schema

    st.markdown("---")

    # Prepare button
    if st.button("Prepare Export Files", type="primary", key="combined_prepare_btn"):
        with st.spinner("Generating export files..."):
            try:
                # Generate main exports
                st.session_state["combined_df_decomps"] = generate_combined_decomps_stacked(wrappers_with_labels, brand, force_to_actuals)
                st.session_state["combined_df_media"] = generate_combined_media_results(wrappers_with_labels, brand)
                st.session_state["combined_df_fit"] = generate_combined_actual_vs_fitted(wrappers_with_labels, brand)

                # Generate combined disaggregated results if any models have disagg configs
                # Read from session state since UI state may be lost on button click
                disagg_configs_to_use = st.session_state.get("combined_disagg_configs", {})
                if disagg_configs_to_use:
                    # Build list of (wrapper, label, disagg_config) tuples
                    wrappers_with_disagg = [
                        (wrapper, label, disagg_config)
                        for label, (wrapper, disagg_config) in disagg_configs_to_use.items()
                    ]

                    # Generate combined disaggregated files
                    st.session_state["combined_df_decomps_disagg"] = generate_combined_decomps_stacked_disaggregated(
                        wrappers_with_disagg, brand, force_to_actuals
                    )
                    st.session_state["combined_df_media_disagg"] = generate_combined_media_results_disaggregated(
                        wrappers_with_disagg, brand
                    )

                    # Generate workings files (per-model for diagnostic purposes)
                    combined_disagg_results = {}
                    for label, (wrapper, disagg_config) in disagg_configs_to_use.items():
                        mapped_df, granular_name_cols, date_col, weight_col, include_cols = disagg_config
                        combined_disagg_results[label] = generate_disaggregated_results(
                            wrapper=wrapper,
                            config=wrapper.config,
                            granular_df=mapped_df,
                            granular_name_cols=granular_name_cols,
                            date_col=date_col,
                            model_channel_col="_model_channel",
                            weight_col=weight_col,
                            brand=brand
                        )
                    st.session_state["combined_disagg_results"] = combined_disagg_results
                else:
                    st.session_state["combined_df_decomps_disagg"] = None
                    st.session_state["combined_df_media_disagg"] = None
                    st.session_state["combined_disagg_results"] = None

                st.session_state["combined_files_ready"] = True
                st.session_state["combined_brand"] = brand
                st.session_state["combined_labels"] = [label for _, label in wrappers_with_labels]
                st.session_state["combined_first_model_path"] = loaded_wrappers[0][2] if loaded_wrappers else None

                # Validate schema if one is selected
                if st.session_state.get("combined_selected_schema"):
                    st.session_state["combined_schema_validated"] = False
                else:
                    st.session_state["combined_schema_validated"] = True

                st.success("Export files ready!")

            except Exception as e:
                st.error(f"Error generating exports: {e}")
                st.session_state["combined_files_ready"] = False

    # Schema validation step (if schema selected and not yet validated)
    if (st.session_state.get("combined_files_ready", False)
        and st.session_state.get("combined_selected_schema")
        and not st.session_state.get("combined_schema_validated", False)):

        st.markdown("---")
        st.subheader("Schema Validation")

        should_proceed, updated_schema = _show_schema_validation_ui(
            st.session_state["combined_selected_schema"],
            st.session_state["combined_df_decomps"],
            st.session_state["combined_df_media"],
            st.session_state["combined_df_fit"],
            key_suffix="_combined",
            session_key="combined_selected_schema"
        )

        if should_proceed:
            st.session_state["combined_selected_schema"] = updated_schema
            st.session_state["combined_schema_validated"] = True
            # Apply schema to combined exports
            if updated_schema:
                _apply_schema_to_combined_exports()
            st.rerun()
        else:
            st.info("Please resolve schema validation issues above to proceed.")
            st.stop()

    # Show downloads only if files are ready
    if st.session_state.get("combined_files_ready", False):
        st.markdown("---")

        label_list = st.session_state.get("combined_labels", [])
        label_display = ", ".join([f"kpi_{l}" for l in label_list])

        st.subheader("Platform Export Files")
        st.info(
            f"These CSV files combine data from all models with separate columns: "
            f"{label_display}, kpi_total"
        )

        # Store original DataFrames for column editing (before schema was applied)
        if "combined_df_decomps_original" not in st.session_state:
            st.session_state["combined_df_decomps_original"] = st.session_state.get("combined_df_decomps")
            st.session_state["combined_df_media_original"] = st.session_state.get("combined_df_media")
            st.session_state["combined_df_fit_original"] = st.session_state.get("combined_df_fit")
            st.session_state["combined_df_decomps_disagg_original"] = st.session_state.get("combined_df_decomps_disagg")
            st.session_state["combined_df_media_disagg_original"] = st.session_state.get("combined_df_media_disagg")

        # Column schema editor expander
        combined_brand = st.session_state.get("combined_brand", brand)
        combined_model_path = st.session_state.get("combined_first_model_path", "")

        # Track if user has opened the expander (stays open once opened until page refresh)
        if "column_editor_opened_combined" not in st.session_state:
            st.session_state["column_editor_opened_combined"] = False

        # Check if any working schema exists (indicates user was editing)
        has_working_schema = any(
            k.startswith("working_schema_") and k.endswith("_combined")
            for k in st.session_state.keys()
        )
        if has_working_schema:
            st.session_state["column_editor_opened_combined"] = True

        with st.expander("Configure Column Schema", expanded=st.session_state["column_editor_opened_combined"]):
            st.session_state["column_editor_opened_combined"] = True  # Mark as opened once user sees it
            st.caption("Customize column names, order, and visibility before download")
            _show_column_editor_section_combined(combined_brand, combined_model_path)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Three columns for the three export types
        col1, col2, col3 = st.columns(3)

        # Get disaggregated data (now single combined files)
        df_decomps_disagg = st.session_state.get("combined_df_decomps_disagg")
        df_media_disagg = st.session_state.get("combined_df_media_disagg")
        combined_disagg = st.session_state.get("combined_disagg_results")

        # 1. Decomps Stacked
        with col1:
            st.markdown("### Decomps Stacked")
            df_decomps = st.session_state["combined_df_decomps"]
            csv_decomps = df_decomps.to_csv(index=False)

            st.download_button(
                label="Download decomps_stacked.csv",
                data=csv_decomps,
                file_name=f"decomps_stacked_combined_{timestamp}.csv",
                mime="text/csv",
                key="combined_decomps_download",
                type="primary"
            )

            # Single combined disaggregated decomps
            if df_decomps_disagg is not None:
                csv_decomps_disagg = df_decomps_disagg.to_csv(index=False)
                st.download_button(
                    label="Download decomps_stacked_disagg.csv",
                    data=csv_decomps_disagg,
                    file_name=f"decomps_stacked_disagg_combined_{timestamp}.csv",
                    mime="text/csv",
                    key="combined_decomps_disagg_download",
                    help="Combined disaggregated decomps with kpi columns per model"
                )

            with st.expander("Preview (first 10 rows)"):
                st.dataframe(df_decomps.head(10), width="stretch")

            st.caption(f"{len(df_decomps):,} rows × {len(df_decomps.columns)} columns")
            if df_decomps_disagg is not None:
                st.caption(f"Disagg: {len(df_decomps_disagg):,} rows")

        # 2. Media Results
        with col2:
            st.markdown("### Media Results")
            df_media = st.session_state["combined_df_media"]
            csv_media = df_media.to_csv(index=False)

            st.download_button(
                label="Download mmm_media_results.csv",
                data=csv_media,
                file_name=f"mmm_media_results_combined_{timestamp}.csv",
                mime="text/csv",
                key="combined_media_download",
                type="primary"
            )

            # Single combined disaggregated media
            if df_media_disagg is not None:
                csv_media_disagg = df_media_disagg.to_csv(index=False)
                st.download_button(
                    label="Download media_results_disagg.csv",
                    data=csv_media_disagg,
                    file_name=f"mmm_media_results_disagg_combined_{timestamp}.csv",
                    mime="text/csv",
                    key="combined_media_disagg_download",
                    help="Combined disaggregated media results with kpi columns per model"
                )

            with st.expander("Preview (first 10 rows)"):
                st.dataframe(df_media.head(10), width="stretch")

            st.caption(f"{len(df_media):,} rows × {len(df_media.columns)} columns")
            if df_media_disagg is not None:
                st.caption(f"Disagg: {len(df_media_disagg):,} rows")

        # 3. Actual vs Fitted
        with col3:
            st.markdown("### Actual vs Fitted")
            df_fit = st.session_state["combined_df_fit"]
            csv_fit = df_fit.to_csv(index=False)

            st.download_button(
                label="Download actual_vs_fitted.csv",
                data=csv_fit,
                file_name=f"actual_vs_fitted_combined_{timestamp}.csv",
                mime="text/csv",
                key="combined_fit_download",
                type="primary"
            )

            with st.expander("Preview (first 10 rows)"):
                st.dataframe(df_fit.head(10), width="stretch")

            st.caption(f"{len(df_fit):,} rows × {len(df_fit.columns)} columns")

        # Disaggregation workings (diagnostic files with weight calculations)
        if combined_disagg:
            with st.expander("Show Disaggregation Workings"):
                st.caption("Diagnostic files with weight calculations (weight, weight_adstocked, weight_pct, is_mapped)")

                for label, df_disagg in combined_disagg.items():
                    st.markdown(f"**{label}**")
                    csv_disagg = df_disagg.to_csv(index=False)

                    st.download_button(
                        label=f"Download workings_{label}.csv",
                        data=csv_disagg,
                        file_name=f"workings_{label}_{timestamp}.csv",
                        mime="text/csv",
                        key=f"combined_workings_download_{label}"
                    )

                    st.dataframe(df_disagg.head(20), width="stretch")
                    st.caption(f"{len(df_disagg):,} rows × {len(df_disagg.columns)} columns")

        # Download All button
        st.markdown("---")
        st.subheader("Download All Files")

        export_brand = st.session_state.get("combined_brand", brand)
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(f"decomps_stacked_combined_{timestamp}.csv", df_decomps.to_csv(index=False))
            zf.writestr(f"mmm_media_results_combined_{timestamp}.csv", df_media.to_csv(index=False))
            zf.writestr(f"actual_vs_fitted_combined_{timestamp}.csv", df_fit.to_csv(index=False))
            # Include combined disaggregated files
            if df_decomps_disagg is not None:
                zf.writestr(f"decomps_stacked_disagg_combined_{timestamp}.csv", df_decomps_disagg.to_csv(index=False))
            if df_media_disagg is not None:
                zf.writestr(f"mmm_media_results_disagg_combined_{timestamp}.csv", df_media_disagg.to_csv(index=False))
            # Include workings files (per-model diagnostic)
            if combined_disagg:
                for label, df_disagg in combined_disagg.items():
                    zf.writestr(f"workings_{label}_{timestamp}.csv", df_disagg.to_csv(index=False))

        zip_buffer.seek(0)

        st.download_button(
            label="Download All (ZIP)",
            data=zip_buffer.getvalue(),
            file_name=f"{export_brand}_combined_exports_{timestamp}.zip",
            mime="application/zip",
            type="primary",
            key="combined_download_all"
        )
        st.caption("Downloads all CSV files in a single ZIP archive")

    # File format documentation for combined exports
    st.markdown("---")
    with st.expander("Combined Export File Format Documentation"):
        st.markdown(f"""
        ### decomps_stacked.csv (Combined)

        | Column | Description |
        |--------|-------------|
        | date | Date value |
        | wc_mon | Week commencing Monday |
        | brand | Brand name |
        | decomp | Variable name |
        | decomp_lvl1 | Category level 1 |
        | decomp_lvl2 | Category level 2 |
        | kpi_{{label}} | Contribution from each model |
        | kpi_total | Sum of all model contributions |

        ### mmm_media_results.csv (Combined)

        | Column | Description |
        |--------|-------------|
        | date | Date value |
        | wc_mon | Week commencing Monday |
        | brand | Brand name |
        | decomp_lvl1 | Category level 1 |
        | decomp_lvl2 | Category level 2 |
        | spend | Business spend (same for all KPIs) |
        | kpi_{{label}} | Contribution from each model |
        | kpi_total | Sum of all model contributions |
        | decomp | Channel name |

        ### actual_vs_fitted.csv (Combined)

        | Column | Description |
        |--------|-------------|
        | date | Date value |
        | wc_mon | Week commencing Monday |
        | brand | Brand name |
        | actual_fitted | "Actual" or "Fitted" |
        | value_{{label}} | Value from each model |
        | value_total | Sum of all model values |
        """)
