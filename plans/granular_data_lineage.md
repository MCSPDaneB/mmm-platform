# Granular Data Lineage and Auto-Category Inference

## Summary

Preserve the granular media data (with level columns) from upload, automatically infer category values using "highest common level" logic, and reuse this data for disaggregated exports without re-upload.

## User Requirements

1. **Preserve level columns**: Store pre-aggregation granular data with the model
2. **Highest common level**: When rows aggregate into one variable, category values use the shared ancestor values
3. **User selects category levels**: UI to choose which level columns become category columns
4. **Parent fallback**: If entities differ at a selected level, fall back to highest matching parent level
5. **Reuse for disaggregation**: Stored granular data serves as default for exports
6. **Validate new files**: Cross-check new disaggregation uploads against stored data

---

## Implementation Plan

### Phase 1: Schema Changes

**File: `mmm_platform/config/schema.py`**

Add new models after `DisaggregationMappingConfig` (~line 124):

```python
class GranularEntityMapping(BaseModel):
    """Single entity's level values and its mapped variable."""
    level_values: dict[str, str]  # {"lvl1": "Google", "lvl2": "Search", "lvl3": "Branded"}
    variable_name: str            # "google_search_spend"

class GranularDataLineage(BaseModel):
    """Lineage from granular upload to aggregated variables."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())

    # Structure
    level_columns: list[str]      # ["media_channel_lvl1", "lvl2", "lvl3"]
    date_column: str              # "date"
    weight_column: str            # "spend"
    include_columns: list[str] = Field(default_factory=list)  # ["impressions", "clicks"]

    # Mappings
    entity_mappings: list[GranularEntityMapping] = Field(default_factory=list)

    # Category inference config
    level_to_category: dict[str, str] = Field(default_factory=dict)
    # e.g., {"media_channel_lvl1": "Platform", "media_channel_lvl2": "Channel Type"}
```

Extend `ModelConfig` to include:
```python
granular_lineage: Optional[GranularDataLineage] = None
```

---

### Phase 2: Category Inference Logic

**File: `mmm_platform/core/data_processing.py`**

Add new function `compute_highest_common_categories()`:

```python
def compute_highest_common_categories(
    entity_mappings: list[GranularEntityMapping],
    level_columns: list[str],
    level_to_category: dict[str, str]
) -> dict[str, dict[str, str]]:
    """
    Compute category values for each variable using highest common level logic.

    Returns: {variable_name: {category_name: value}}

    Logic:
    1. Group entity_mappings by variable_name
    2. For each selected category level:
       - If all entities for this variable share the same value -> use it
       - If values differ -> walk UP the hierarchy to find shared ancestor
    """
```

Example:
- Entities: `Google|Search|Branded`, `Google|Search|NonBranded` → `google_search_spend`
- User selects: lvl1→"Platform", lvl2→"Type", lvl3→"Segment"
- Result: `{"google_search_spend": {"Platform": "Google", "Type": "Search", "Segment": "Search"}}`
  - (lvl3 differs, so falls back to lvl2's shared value)

---

### Phase 3: Upload Flow Changes

**File: `mmm_platform/ui/pages/upload_data.py`**

#### 3.1 Modify channel mapping section

After the mapping table where users assign variable names, add new section:

```python
st.markdown("---")
st.subheader("Category Columns from Hierarchy")
st.caption("Select which levels should become category columns for grouping results.")

level_to_category = {}
for level_col in level_cols:
    col1, col2 = st.columns([1, 2])
    with col1:
        use_as_category = st.checkbox(f"Use {level_col}", key=f"use_cat_{level_col}")
    with col2:
        if use_as_category:
            cat_name = st.text_input(
                "Category name",
                value=level_col.replace("media_channel_", "").replace("_", " ").title(),
                key=f"cat_name_{level_col}"
            )
            level_to_category[level_col] = cat_name
```

#### 3.2 Build lineage after mapping confirmed

When user confirms mapping (before merge):

```python
# Build entity mappings from the mapping table
entity_mappings = []
for combo_tuple, var_name in mapping.items():
    if var_name:  # Skip unmapped
        entity_mappings.append(GranularEntityMapping(
            level_values=dict(zip(level_cols, combo_tuple)),
            variable_name=var_name
        ))

lineage = GranularDataLineage(
    level_columns=level_cols,
    date_column=date_col,
    weight_column=spend_col,
    include_columns=[c for c in [impr_col, click_col] if c],
    entity_mappings=entity_mappings,
    level_to_category=level_to_category
)

st.session_state["granular_lineage"] = lineage
st.session_state["granular_df"] = media_df.copy()  # Store pre-aggregation data
```

#### 3.3 Auto-populate category columns in config

When building ModelConfig after upload:

```python
# Create CategoryColumnConfigs from lineage
if lineage.level_to_category:
    category_columns = []
    variable_categories = compute_highest_common_categories(
        lineage.entity_mappings,
        lineage.level_columns,
        lineage.level_to_category
    )

    for level_col, cat_name in lineage.level_to_category.items():
        # Collect unique values for this category
        options = set()
        for var, cats in variable_categories.items():
            if cat_name in cats:
                options.add(cats[cat_name])

        category_columns.append(CategoryColumnConfig(
            name=cat_name,
            options=sorted(options)
        ))

    # Store for later use in configure_model.py
    st.session_state["inferred_category_columns"] = category_columns
    st.session_state["inferred_variable_categories"] = variable_categories
```

---

### Phase 4: Persistence Changes

**File: `mmm_platform/model/persistence.py`**

#### 4.1 Save granular data

In `save()` method, after saving other data files:

```python
# Save granular data if available
if hasattr(wrapper, 'df_granular') and wrapper.df_granular is not None:
    wrapper.df_granular.to_parquet(model_dir / "data_granular.parquet")
```

The `granular_lineage` is already part of `ModelConfig`, so it saves with `config.json`.

#### 4.2 Load granular data

In `load()` method:

```python
granular_path = model_dir / "data_granular.parquet"
if granular_path.exists():
    wrapper.df_granular = pd.read_parquet(granular_path)
else:
    wrapper.df_granular = None
```

**File: `mmm_platform/model/mmm.py`**

Add attribute to `MMMWrapper.__init__`:

```python
self.df_granular: Optional[pd.DataFrame] = None
```

---

### Phase 5: Export Integration

**File: `mmm_platform/ui/pages/export.py`**

#### 5.1 Modify `_show_disaggregation_ui()`

At the start, check for stored granular data:

```python
has_stored = (
    hasattr(wrapper, 'df_granular') and
    wrapper.df_granular is not None and
    config.granular_lineage is not None
)

if has_stored:
    st.success(f"Stored granular data available ({len(wrapper.df_granular):,} rows)")

    data_source = st.radio(
        "Data source",
        ["Use stored data", "Upload new file"],
        key=f"disagg_source{key_suffix}"
    )

    if data_source == "Use stored data":
        granular_df = wrapper.df_granular
        lineage = config.granular_lineage

        # Pre-populate all fields from lineage
        # (skip the file upload and column selection UI)
        # Build mapping dict from lineage.entity_mappings
        ...
```

#### 5.2 New file validation

When user uploads a new file, validate against stored lineage:

```python
def validate_against_lineage(new_df: pd.DataFrame, lineage: GranularDataLineage) -> tuple[bool, list[str]]:
    """Check if new file is compatible with stored lineage."""
    warnings = []

    # Check columns exist
    missing = [c for c in lineage.level_columns if c not in new_df.columns]
    if missing:
        warnings.append(f"Missing level columns: {missing}")

    if lineage.date_column not in new_df.columns:
        warnings.append(f"Missing date column: {lineage.date_column}")

    # Check entity overlap
    if not missing:
        new_entities = set(tuple(row) for _, row in new_df[lineage.level_columns].drop_duplicates().iterrows())
        stored_entities = set(
            tuple(em.level_values[c] for c in lineage.level_columns)
            for em in lineage.entity_mappings
        )

        only_in_new = new_entities - stored_entities
        only_in_stored = stored_entities - new_entities

        if only_in_new:
            warnings.append(f"{len(only_in_new)} new entities not in stored mapping (will need manual mapping)")
        if only_in_stored:
            warnings.append(f"{len(only_in_stored)} stored entities missing from new file")

    is_compatible = len([w for w in warnings if "Missing" in w]) == 0
    return is_compatible, warnings
```

Display validation results in UI, allowing user to proceed with warnings or use stored data instead.

---

### Phase 6: Config Integration

**File: `mmm_platform/ui/pages/configure_model.py`**

When initializing channel configs, check for inferred categories:

```python
inferred = st.session_state.get("inferred_variable_categories", {})

# When building ChannelConfig for each channel
for channel_name in selected_channels:
    categories = inferred.get(channel_name, {})
    # Pre-populate channel.categories with inferred values
```

Show indicator in UI when categories were auto-inferred:
```python
if channel_name in inferred:
    st.caption("Categories auto-inferred from upload hierarchy")
```

---

## Files to Modify

| File | Changes |
|------|---------|
| `config/schema.py` | Add `GranularEntityMapping`, `GranularDataLineage`; extend `ModelConfig` |
| `core/data_processing.py` | Add `compute_highest_common_categories()` |
| `ui/pages/upload_data.py` | Add category selection UI, build lineage, store granular df |
| `model/mmm.py` | Add `df_granular` attribute |
| `model/persistence.py` | Save/load `data_granular.parquet` |
| `ui/pages/export.py` | Add stored data option, validation function |
| `ui/pages/configure_model.py` | Pre-populate categories from inferred values |

---

## Edge Cases

| Scenario | Behavior |
|----------|----------|
| Variable aggregates entities with different values at selected level | Fall back to highest common parent level value |
| All levels differ for a variable | That category left empty for manual assignment |
| Legacy model without lineage | Graceful degradation - show normal file upload |
| New file has extra entities | Show warning, allow mapping new entities |
| New file missing entities | Show warning, suggest using stored data |
| Different column structure | Show validation errors, require manual mapping |

---

## Test Coverage Status

| Area | Current Tests | Coverage |
|------|--------------|----------|
| Data Processing | **0** | **NONE - Critical gap** |
| Upload Flow | 9 | Good |
| Disaggregation | 32 | Excellent |
| Schema | 38 | Excellent |
| Persistence | 28 | Excellent |

**Critical**: `data_processing.py` has NO tests. Must establish baseline before changes.

---

## Implementation Order (Test-First Approach)

### Phase 0: Establish Test Baseline (BEFORE any changes)

1. **Create `test_data_processing.py`** with tests for existing functions:
   - `test_parse_media_file()` - level column detection, numeric cleaning
   - `test_get_unique_channel_combinations()` - unique combo extraction, suggested names
   - `test_aggregate_media_data()` - grouping, pivoting, metric handling

2. **Run full test suite** to establish baseline:
   ```bash
   .\venv\Scripts\python.exe -m pytest mmm_platform/tests/ -v --tb=short
   ```

3. **Document any existing failures** before proceeding

### Phase 1: Schema Changes
- Add `GranularEntityMapping`, `GranularDataLineage` to schema.py
- Add tests to `test_schema.py` for new models
- Run tests

### Phase 2: Category Inference Logic
- Add `compute_highest_common_categories()` to data_processing.py
- Add comprehensive tests to `test_data_processing.py`
- Run tests

### Phase 3: Persistence Changes
- Add `df_granular` to MMMWrapper
- Modify persistence.py save/load
- Add tests to `test_model_persistence.py` for granular data
- Run tests

### Phase 4: Upload UI Changes
- Modify upload_data.py for category selection and lineage storage
- Add/update tests in `test_data_upload.py`
- Manual UI testing

### Phase 5: Export Integration
- Modify export.py for stored data option and validation
- Add tests to `test_disaggregation.py` for validation function
- Run tests

### Phase 6: Configure Model Integration
- Pre-populate categories from inferred values
- Manual UI testing

### Phase 7: Regression Testing
- Run full test suite
- Manual end-to-end testing of upload → configure → fit → export flow
- Verify legacy models (without lineage) still work
