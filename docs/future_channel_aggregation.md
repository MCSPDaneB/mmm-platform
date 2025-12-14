# Future Enhancement: Channel Aggregation for Optimization

## Status: Not Implemented

This document captures requirements for allowing optimization at different channel granularities than what was modeled.

## Current State

**Optimization is locked to modeled channels.** If you model Facebook and Instagram separately, you must optimize them separately. If you model them as a combined "Meta" channel, you optimize Meta as a single unit.

### What Works Today

| Feature | Status | Notes |
|---------|--------|-------|
| Optimize exact modeled channels | ✅ | Default behavior |
| Group constraints (min/max spend for channel groups) | ✅ | Channels stay separate, just constrained together |
| Post-optimization disaggregation | ✅ | `DisaggregationMappingConfig` splits results proportionally for export |

### What Doesn't Work

| Feature | Status | Notes |
|---------|--------|-------|
| Aggregate channels at optimization time | ❌ | Can't combine FB + IG → "Meta" as single decision variable |
| Disaggregate at optimization time | ❌ | Can't split "Meta" → FB/IG during optimization |

## Existing Workarounds

### Group Constraints

Use `ConstraintBuilder` to constrain groups together:

```python
from mmm_platform.optimization.constraints import ConstraintBuilder

# Ensure Meta channels together get at least 30% of budget
constraint = ConstraintBuilder.group_min_spend(
    ["facebook_spend", "instagram_spend"],
    budget * 0.3
)

# Or cap a group at 40%
constraint = ConstraintBuilder.group_max_ratio(
    ["facebook_spend", "instagram_spend"],
    0.4
)
```

**Limitation:** Channels remain separate decision variables. The optimizer can still allocate differently within the group (e.g., 25% FB, 5% IG) as long as the constraint is satisfied.

### Model at Desired Granularity

The most reliable approach: aggregate your data before modeling.

- If you want to optimize "Meta" as a unit, combine FB + IG spend in your input data
- Use `DisaggregationMappingConfig` to split results back to granular level for reporting

## What True Aggregation Would Require

To support runtime channel grouping for optimization:

### 1. Channel Group Configuration

```python
# New config structure (conceptual)
channel_groups = {
    "Meta": ["facebook_spend", "instagram_spend"],
    "Google": ["search_spend", "youtube_spend"],
}
```

### 2. Aggregation Adapter

Modify `OptimizationBridge` or `BudgetAllocator` to:
- Accept channel grouping config
- Aggregate response curves before optimization
- Run optimizer on aggregated channels
- Disaggregate results back to original channels

### 3. Response Curve Aggregation

Challenge: How to combine response curves for FB + IG into a single "Meta" curve?

Options:
- Sum the curves (assumes additive effects)
- Weighted average based on historical spend ratio
- Re-estimate combined curve from model posteriors

### 4. Result Disaggregation

After optimizer allocates $X to "Meta", split back:
- Proportional to historical spend
- Proportional to marginal ROI
- User-defined ratios

## Implementation Considerations

### Complexity
- Medium-high: Requires changes to optimizer pipeline, not just UI
- PyMC-Marketing's `BudgetOptimizer` expects specific channel indices

### Risk
- Aggregating response curves may introduce errors
- Need to validate that aggregated optimization matches sum of individual optimizations

### Alternative Approaches
1. **Constraint-only** (already implemented): Keep channels separate, just constrain groups
2. **Multiple models**: Fit one model at aggregated level, another at granular level
3. **Post-hoc allocation**: Optimize aggregated, then allocate within groups by marginal ROI

## Related Code

- `mmm_platform/optimization/bridge.py` - `get_optimizable_channels()`
- `mmm_platform/optimization/constraints.py` - `ConstraintBuilder.group_min_spend()`, `group_max_ratio()`
- `mmm_platform/config/schema.py` - `DisaggregationMappingConfig`
- `mmm_platform/analysis/export.py` - Uses disaggregation mapping for results
