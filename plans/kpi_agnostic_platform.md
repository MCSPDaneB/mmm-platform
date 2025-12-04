# KPI-Agnostic Platform Plan

## Summary

Make the MMM platform support any KPI (revenue, installs, volume, leads, etc.) with dynamic labeling:
- **Revenue-type KPIs** → "ROI" terminology (e.g., "ROI: $3.50")
- **Count-type KPIs** → "Cost Per X" terminology (e.g., "Cost Per Install: $5.00")

The core math already works with any target - this is primarily a naming/UX refactor.

---

## Safety Guarantees

**Tests will pass throughout implementation:**
1. Migration validators ensure old field names (`revenue_scale`, `roi_prior_*`) still work
2. Backward-compatible aliases maintain existing API (`get_roi_dicts()` calls `get_efficiency_dicts()`)
3. Each phase is independently testable - run tests after each phase
4. No changes to core mathematical logic (priors.py, transforms.py calculations unchanged)

**Existing test coverage protects:**
- `test_schema.py`: Config validation, ROI prior bounds, serialization roundtrip
- `test_priors.py`: Beta calibration math, ROI→beta→ROI roundtrip within 10% tolerance
- `test_roi_diagnostics.py`: Prior vs posterior validation
- `test_marginal_roi.py`: Marginal efficiency calculations

**Implementation order minimizes risk:**
1. Schema changes WITH migration validators (tests pass)
2. Add new helper class (additive, tests still pass)
3. Update UI labels (visual only, tests still pass)
4. Update tests to use new naming (confirms everything works)

---

## Phase 1: Schema Foundation

**File:** `mmm_platform/config/schema.py`

1. Add `KPIType` enum:
```python
class KPIType(str, Enum):
    REVENUE = "revenue"  # Currency KPIs (revenue, sales)
    COUNT = "count"      # Count KPIs (installs, leads, volume)
```

2. Update `DataConfig`:
   - Rename `revenue_scale` → `target_scale`
   - Add `kpi_type: KPIType = KPIType.REVENUE`
   - Add `kpi_display_name: Optional[str]` (e.g., "Install" for labeling)

3. Update `ChannelConfig` and `OwnedMediaConfig`:
   - Rename `roi_prior_low/mid/high` → `efficiency_prior_low/mid/high`
   - Add migration validator for old field names

4. Rename `get_roi_dicts()` → `get_efficiency_dicts()`

5. Add helper methods to `ModelConfig`:
   - `get_efficiency_label()` → returns "ROI" or "Cost Per {X}"
   - `is_revenue_type()` → boolean check

---

## Phase 2: UI Label System

**New file:** `mmm_platform/ui/kpi_labels.py`

Create `KPILabels` class with:
- `efficiency_label` → "ROI" or "Cost Per Install"
- `prior_input_label` → "Expected ROI" or "Expected Cost Per Install"
- `format_efficiency(value)` → "$3.50" or "$5.00" (inverted for cost-per)
- `convert_input_to_internal(value)` → handles cost-per inversion
- `convert_internal_to_display(value)` → reverse conversion

---

## Phase 3: Configure Model Page

**File:** `mmm_platform/ui/pages/configure_model.py`

1. Add KPI type selector in Data Settings:
```python
kpi_type = st.selectbox("KPI Type", ["revenue", "count"])
if kpi_type == "count":
    kpi_display_name = st.text_input("KPI Name", value="Install")
```

2. Rename "Revenue Scale" → "Target Scale"

3. Update channel prior table:
   - Dynamic column headers based on KPI type
   - For count KPIs: input as "Cost Per X", convert internally to efficiency

---

## Phase 4: Results Page

**File:** `mmm_platform/ui/pages/results.py`

1. Update tab names dynamically:
   - "Channel ROI" → f"Channel {labels.efficiency_label}"
   - "Marginal ROI" → f"Marginal {labels.efficiency_label}"
   - "ROI Prior Validation" → f"{labels.efficiency_label} Prior Validation"

2. Update all chart titles, axis labels, and formatted values

3. For count KPIs, display inverted values (cost-per instead of efficiency)

---

## Phase 5: Analysis Modules

**File:** `mmm_platform/analysis/contributions.py`
- Rename `revenue_scale` param → `target_scale`
- Rename `get_channel_roi()` → `get_channel_efficiency()`

**File:** `mmm_platform/analysis/marginal_roi.py`
- Rename `MarginalROIAnalyzer` → `MarginalEfficiencyAnalyzer` (keep alias)
- Rename `ChannelMarginalROI` → `ChannelMarginalEfficiency`
- Rename all `*_roi` fields → `*_efficiency`

**File:** `mmm_platform/analysis/roi_diagnostics.py`
- Rename `ROIDiagnostics` → `EfficiencyDiagnostics` (keep alias)
- Rename `ChannelROIResult` → `ChannelEfficiencyResult`

**File:** `mmm_platform/core/priors.py`
- Update to use `get_efficiency_dicts()` instead of `get_roi_dicts()`

---

## Phase 6: Other Files

**File:** `mmm_platform/config/loader.py`
- Update template defaults to use new field names

**File:** `mmm_platform/ui/pages/run_model.py`
- Update `revenue_scale` → `target_scale` in state passing

**File:** `mmm_platform/ui/pages/saved_models.py`
- Update config extraction to use new field names

**File:** `configs/default_config.yaml`
- Update example to use new field names

---

## Phase 7: Tests

Update all test files to use new field names:
- `mmm_platform/tests/conftest.py`
- `mmm_platform/tests/unit/test_schema.py`
- `mmm_platform/tests/unit/test_priors.py`
- `mmm_platform/tests/unit/test_roi_diagnostics.py`
- `mmm_platform/tests/unit/test_marginal_roi.py`

Add new tests for count-type KPIs.

---

## Key Files (Priority Order)

1. `mmm_platform/config/schema.py` - Foundation
2. `mmm_platform/ui/kpi_labels.py` - New helper (create)
3. `mmm_platform/ui/pages/configure_model.py` - KPI selector + labels
4. `mmm_platform/ui/pages/results.py` - Dynamic display
5. `mmm_platform/analysis/contributions.py` - Core rename
6. `mmm_platform/analysis/marginal_roi.py` - Core rename
7. `mmm_platform/analysis/roi_diagnostics.py` - Core rename
8. `mmm_platform/core/priors.py` - Use new method names
9. Tests - Update fixtures and assertions

---

## Migration Strategy

Add Pydantic `@model_validator(mode="before")` to automatically migrate:
- `revenue_scale` → `target_scale`
- `roi_prior_*` → `efficiency_prior_*`
- `include_roi` → `include_efficiency_priors`

This allows existing saved configs to load without manual changes.

---

## Implementation Checkpoints

After each phase, run: `.\venv\Scripts\python.exe -m pytest mmm_platform/tests/ -v --tb=short`

| Phase | Expected Test Result |
|-------|---------------------|
| Phase 1 (Schema) | All 50+ tests pass (migration validators active) |
| Phase 2 (KPILabels) | All tests pass (additive module) |
| Phase 3 (Configure Model) | All tests pass (UI only) |
| Phase 4 (Results) | All tests pass (UI only) |
| Phase 5 (Analysis) | All tests pass (aliases maintain API) |
| Phase 6 (Other Files) | All tests pass |
| Phase 7 (Tests) | All tests pass with new naming |

---

## What Does NOT Change

- **Core math in `priors.py`**: Beta calibration formula unchanged
- **Core math in `transforms.py`**: Adstock/saturation transforms unchanged
- **Model fitting in `mmm.py`**: PyMC model structure unchanged
- **Export column naming**: Already uses `kpi_{target_col}` pattern
- **Database models**: No schema changes needed
