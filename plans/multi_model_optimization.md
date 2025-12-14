# Multi-Model Budget Optimization Extension

## Status: PENDING (Updated Dec 2025)

The single-model optimization module is complete. This extension adds support for optimizing across multiple models (e.g., in-store revenue + online revenue) where the **same spend drives both outcomes** (shared channels approach).

## Design Decisions (Dec 2025)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Channel handling** | Union | Include all channels; channels missing from a model contribute zero to that model's response |
| **UI location** | Settings tab toggle | Not a separate tab; reuse existing optimization functions |
| **Default weighting** | Equal (1.0) | Simple starting point; user can customize |
| **Weighting options** | Equal / Profit Margin / Custom | Flexibility for different use cases |

## User Requirements

From conversation:
- **Budget Model**: "Shared channels" - same spend drives both outcomes
- **UI Location**: Settings tab toggle (enables multi-model for all existing optimization tabs)
- **Channel Mismatch**: Use union of channels; missing channels contribute zero

## Problem Statement

When a business has:
- **Model A**: In-store Revenue (different margins, different response curves)
- **Model B**: Online Revenue (different margins, different response curves)
- **Same Channels**: A $40k search campaign drives BOTH outcomes

We need to optimize total spend to maximize **combined value** (weighted by margins):
```
maximize: response_A(spend) * margin_A + response_B(spend) * margin_B
```

---

## Architecture

### New Files

| File | Purpose |
|------|---------|
| `mmm_platform/optimization/multi_model.py` | MultiModelBudgetAllocator class |
| `mmm_platform/tests/unit/test_multi_model_optimization.py` | Unit tests |

### Modified Files

| File | Changes |
|------|---------|
| `mmm_platform/optimization/__init__.py` | Export new classes |
| `mmm_platform/optimization/results.py` | Add MultiModelOptimizationResult |
| `mmm_platform/ui/pages/optimize.py` | Add "Multi-Model" tab |

---

## Core Component: `multi_model.py`

```python
@dataclass
class ModelWeight:
    """Configuration for a model in multi-model optimization."""
    wrapper: Any  # MMMWrapper
    label: str  # e.g., "Online Revenue"
    weight: float  # profit margin or relative importance (0-1)


@dataclass
class MultiModelOptimizationResult:
    """Result from multi-model optimization."""
    # Core results
    optimal_allocation: dict[str, float]  # {channel: total_amount}
    total_budget: float

    # Per-model expected responses
    response_by_model: dict[str, float]  # {label: expected_response}
    combined_response: float  # weighted sum

    # Per-model response CIs
    response_ci_by_model: dict[str, tuple[float, float]]

    # Breakdown showing how allocation drives each model
    allocation_contribution: pd.DataFrame  # channel x model response contribution

    # Standard optimization metadata
    success: bool
    message: str
    iterations: int


class MultiModelBudgetAllocator:
    """
    Optimize budget across multiple models with shared channels.

    Approach: Build a combined objective function that evaluates
    the same allocation against all models, then sums the weighted responses.

    Parameters
    ----------
    model_configs : list[ModelWeight]
        List of models with their weights/margins
    num_periods : int
        Forecast horizon in periods
    utility : str
        Utility function for optimization
    """

    def __init__(
        self,
        model_configs: list[ModelWeight],
        num_periods: int = 8,
        utility: str = "mean",
    ):
        self.model_configs = model_configs
        self.num_periods = num_periods
        self.utility = utility

        # Get union of all channels
        self.channels = self._get_union_channels()

        # Create bridges for each model
        self.bridges = [
            OptimizationBridge(mc.wrapper) for mc in model_configs
        ]

    def _get_union_channels(self) -> list[str]:
        """Get union of all channels across models."""
        all_channels = set()
        for mc in self.model_configs:
            bridge = OptimizationBridge(mc.wrapper)
            all_channels.update(bridge.get_optimizable_channels())
        return sorted(all_channels)

    def optimize(
        self,
        total_budget: float,
        channel_bounds: dict[str, tuple[float, float]] | None = None,
        compare_to_current: bool = False,
    ) -> MultiModelOptimizationResult:
        """
        Optimize budget to maximize combined weighted response.

        The optimization uses scipy.optimize with a custom objective that:
        1. Takes a candidate allocation
        2. Evaluates expected response in each model
        3. Returns negative of weighted sum (for minimization)
        """
        # Implementation using scipy.optimize.minimize with SLSQP
        # Similar to single-model but with combined objective
        pass

    def _combined_objective(self, allocation_vector: np.ndarray) -> float:
        """
        Evaluate combined objective for an allocation.

        For each model:
        1. Filter allocation to only channels that exist in that model
        2. Estimate response for that model
        3. Sum responses weighted by model weight
        """
        # Convert vector to dict
        allocation = dict(zip(self.channels, allocation_vector))

        total_weighted_response = 0.0
        for mc, bridge in zip(self.model_configs, self.bridges):
            # Filter allocation to only this model's channels
            model_channels = bridge.get_optimizable_channels()
            model_alloc = {ch: allocation.get(ch, 0) for ch in model_channels}

            # Estimate response for this model at this allocation
            response = bridge.estimate_response_at_allocation(
                model_alloc, self.num_periods
            )[0]  # Just mean, ignoring CI for objective

            total_weighted_response += response * mc.weight

        return -total_weighted_response  # Negative for minimization

    @property
    def model_labels(self) -> list[str]:
        """Get labels for all models."""
        return [mc.label for mc in self.model_configs]

    @classmethod
    def from_combined_analysis_session(
        cls,
        session_state,
        num_periods: int = 8,
    ) -> "MultiModelBudgetAllocator":
        """
        Create allocator from Combined Analysis session state.

        Uses the models already loaded in combined_models session state.
        """
        # Reuse models from st.session_state.combined_models
        pass
```

---

## UI Changes: `optimize.py`

### Settings Tab Addition (Not a New Tab)

Add multi-model toggle to the existing Settings tab:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Settings Tab                                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ☑ Multi-Model Optimization                                                 │
│                                                                             │
│  ┌─ Additional Models ─────────────────────────────────────────────────────┐│
│  │                                                                          ││
│  │  Current model: Online Revenue (loaded)                                  ││
│  │                                                                          ││
│  │  Add models:                                                             ││
│  │  [Select saved model...  ▼] [+ Add]                                      ││
│  │                                                                          ││
│  │  ☑ Offline Revenue    Weight: [1.0]  [×]                                 ││
│  │  ☑ Subscription       Weight: [0.5]  [×]                                 ││
│  │                                                                          ││
│  │  Weighting Preset: [Equal ▼] [Profit Margin] [Custom]                    ││
│  │                                                                          ││
│  └──────────────────────────────────────────────────────────────────────────┘│
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

When multi-model is enabled, **all existing tabs** (Optimize Budget, Find Target, Scenarios) use the combined objective. Results show per-model breakdown.

### Results Display (All Tabs)

When multi-model is enabled, results include:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Combined Response: $185,420                                                │
│  ├── Online Revenue:    $120,500  (weight: 1.0)                            │
│  ├── Offline Revenue:   $64,920   (weight: 1.0)                            │
│                                                                             │
│  📊 Allocation by Channel (with per-model contribution)                     │
│  📋 Table showing: Channel | Spend | Online Response | Offline Response    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key UI Features

1. **Settings Tab Toggle**
   - Enable/disable multi-model optimization
   - Select additional saved models
   - Configure weights (default: equal at 1.0)
   - Weighting presets: Equal, Profit Margin, Custom

2. **Reuse Existing Tabs**
   - All optimization tabs work with combined objective
   - No need for separate "Multi-Model" tab
   - Channel bounds apply to union of channels

3. **Results Display**
   - Show combined weighted response
   - Expandable per-model breakdown
   - Table with per-channel, per-model response

---

## Implementation Plan

### Step 1: Create `multi_model.py`
1. Create `ModelConfig` dataclass (wrapper, label, weight)
2. Create `MultiModelObjective` class
3. Implement `_get_union_channels()` - union of all model channels
4. Implement `evaluate()` - combined weighted response

### Step 2: Modify `allocator.py`
1. Add optional `multi_model_objective` parameter to `BudgetAllocator`
2. If provided, use combined objective instead of single-model
3. `get_optimizable_channels()` returns union when multi-model enabled

### Step 3: Modify `results.py`
Add optional fields to `OptimizationResult`:
- `response_by_model: dict[str, float] | None`
- `model_weights: dict[str, float] | None`

### Step 4: Modify `optimize.py` (Settings Tab)
1. Add "Multi-Model Optimization" toggle to Settings tab
2. Model selector (reuse saved models pattern)
3. Weight configuration with presets (Equal, Profit Margin, Custom)
4. Store in session state: `multi_model_enabled`, `multi_model_configs`

### Step 5: Update Results Display
1. When multi-model enabled, show combined response
2. Add expandable per-model breakdown
3. Update tables to show per-model response columns

### Step 6: Unit Tests
1. Test union channel calculation
2. Test combined objective evaluation
3. Test per-model response breakdown
4. Test equal vs custom weighting

### Step 7: Integration
1. Update `__init__.py` exports
2. Test end-to-end in UI

---

## Technical Approach

### Option A: Scipy Custom Optimizer (Recommended)
Build custom scipy.optimize objective that sums weighted responses.

**Pros:**
- Full control over objective function
- Can leverage existing response estimation code
- Works with any number of models

**Cons:**
- More code to write
- May be slower than native PyMC optimizer

### Option B: PyMC Optimizer with Custom Utility
Extend PyMC's utility function concept to handle multiple models.

**Pros:**
- Closer to PyMC-Marketing patterns
- May get uncertainty estimates for free

**Cons:**
- PyMC's optimizer is designed for single model
- Would require significant modification

**Decision: Option A** - Build scipy-based custom optimizer for flexibility.

---

## Edge Cases

1. **Channel Mismatch**: Use union of channels; channels missing from a model contribute zero (not an error)
2. **No Additional Models**: Multi-model disabled, use single-model optimization
3. **Single Model Selected**: Falls back to single-model behavior (weight=1.0)
4. **Optimization Failure**: Return partial results with warning message
5. **Zero Weights**: Warn user that model will be ignored in combined objective

---

## Files Summary

| File | Action | Description |
|------|--------|-------------|
| `optimization/multi_model.py` | CREATE | ModelConfig, MultiModelObjective |
| `optimization/allocator.py` | EDIT | Add multi_model_objective parameter |
| `optimization/results.py` | EDIT | Add response_by_model fields |
| `optimization/__init__.py` | EDIT | Export new classes |
| `ui/pages/optimize.py` | EDIT | Add Settings tab toggle + results display |
| `tests/unit/test_multi_model_optimization.py` | CREATE | Unit tests |
