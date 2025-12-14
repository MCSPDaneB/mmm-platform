# Spend Forecast Feature - Media Incrementality Checker

## Status: PENDING

## Summary

A media incrementality forecasting tool that predicts channel-driven response from planned/actual spend. Uses the same channel effectiveness indices as the optimizer to ensure consistency. Provides a forward-looking read on media performance between quarterly model refreshes.

## Use Case
- Model is fitted on historical data and refreshed quarterly
- User has new media spend data (from media agency, planned spend, etc.)
- They don't have all variables for future periods (controls, trends, etc.)
- They want: "Given THIS media spend, what incremental response do we expect?"

**Key Insight**: This is an **incrementality checker**, not a full model forecast. We're predicting media-driven contribution only.

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Input method** | CSV upload | Standard format, easy from Excel/agencies |
| **Output granularity** | Total + weekly | Both summary and detailed breakdown |
| **Adstock handling** | Use model history | CSV continues from model's data, warm start |
| **Uncertainty** | Proper Bayesian CIs | Sample from posterior for accurate intervals |
| **Channel indices** | Applied to calculation | Same as optimizer for consistency |
| **Demand index** | Shown as context only | Not applied (matches optimizer behavior) |

---

## Seasonal Index Usage

### Design Decision: Align with Optimizer

| Index Type | Applied? | Displayed? | Rationale |
|------------|----------|------------|-----------|
| **Channel effectiveness indices** | **YES** | YES | Directly measures media incrementality variation by period |
| **Demand index** | **NO** | YES (context) | Represents base demand seasonality, not media incrementality |

### How Channel Indices Work

```python
# Per channel, per month (from optimization/seasonality.py):
Effectiveness[channel, month] = Contribution[channel, month] / Spend[channel, month]
Index[channel, month] = Effectiveness[channel, month] / Avg_Effectiveness[channel]
```

- Index > 1.0 = channel more effective than average
- Index < 1.0 = channel less effective than average

### Application (Same as Optimizer)

```python
# After saturation transform, BEFORE beta weighting (from risk_objectives.py):
seasonal_adjusted = saturation * seasonal_indices
response = np.sum(beta * target_scale * seasonal_adjusted) * num_periods
```

### Why Not Apply Demand Index?

1. **Consistency**: Optimizer doesn't apply it to response calculation
2. **Avoid double-counting**: Channel indices already capture "contribution per spend by month"
3. **Conceptual clarity**: We're forecasting media incrementality, not full KPI
4. **Transparency**: User sees demand context without baking uncertain assumptions into numbers

---

## CSV Format

```csv
date,search_spend,social_spend,tv_spend
2025-01-06,50000,30000,100000
2025-01-13,55000,32000,95000
2025-01-20,48000,28000,110000
...
```
- Weekly dates (must continue from model's last date)
- Channel columns matching model's paid media channels
- Spend in same units as model training data

---

## DRY Principle - Reuse Existing Components

| Component | Source File | What we reuse |
|-----------|-------------|---------------|
| `SeasonalIndexCalculator` | optimization/seasonality.py | Compute channel effectiveness indices |
| `RiskAwareObjective` | optimization/risk_objectives.py | Posterior sampling for CIs |
| `OptimizationBridge` | optimization/bridge.py | Historical spend, contributions, channel info |
| Adstock/saturation params | Already extracted by RiskAwareObjective | x_max, lam_samples, beta_samples |

**NO modifications to existing files** - forecast engine wraps existing functionality.

---

## Files to Create/Modify

| File | Action | Changes |
|------|--------|---------|
| `forecasting/forecast_engine.py` | CREATE | SpendForecastEngine class, wraps existing components |
| `forecasting/__init__.py` | CREATE | Export SpendForecastEngine, ForecastResult |
| `ui/pages/forecast.py` | CREATE | Forecast UI page with sanity check |
| `ui/app.py` | MODIFY | Add Forecast to navigation |
| `tests/unit/test_forecast_engine.py` | CREATE | 21 unit tests |

---

## Core Implementation

### `forecasting/forecast_engine.py`

```python
from dataclasses import dataclass
import pandas as pd
import numpy as np
from mmm_platform.optimization.seasonality import SeasonalIndexCalculator
from mmm_platform.optimization.bridge import OptimizationBridge

@dataclass
class ForecastResult:
    # Response
    total_response: float
    total_ci_low: float
    total_ci_high: float
    weekly_df: pd.DataFrame  # date, response, ci_low, ci_high
    channel_contributions: pd.DataFrame  # date, channel, contribution

    # Seasonality info
    seasonal_applied: bool
    seasonal_indices: dict[str, float]  # channel -> index used
    demand_index: float
    forecast_period: str  # e.g., "Jan-Mar 2025"

    # Sanity check
    sanity_check: dict  # recent_roi, yoy_roi, warnings, is_reasonable

class SpendForecastEngine:
    """Forecast incremental media response from planned/actual spend."""

    def __init__(self, wrapper):
        self.wrapper = wrapper
        self.bridge = OptimizationBridge(wrapper)
        self.seasonal_calculator = SeasonalIndexCalculator(wrapper)

    def forecast(
        self,
        df_spend: pd.DataFrame,
        apply_seasonal: bool = True,
    ) -> ForecastResult:
        """
        Forecast response for weekly spend data.

        Steps:
        1. Validate spend columns match model channels
        2. Determine forecast period (start_month, num_months from dates)
        3. Get seasonal indices for that period (if apply_seasonal=True)
        4. Append to model's historical data (for adstock continuity)
        5. Apply transforms (adstock, saturation)
        6. Apply seasonal indices (like optimizer does)
        7. Sample from posterior for response + CIs
        8. Compute sanity check
        9. Return total + weekly breakdown
        """
        # ... implementation
```

---

## UI Design

### `ui/pages/forecast.py`

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Media Incrementality Forecast                                              │
│  Forecast incremental response from planned/actual media spend              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Upload spend CSV:  [Browse...] [Upload]                                    │
│                                                                             │
│  ☑ Apply seasonal adjustment                                                │
│                                                                             │
│  ┌─ Seasonality for Jan-Mar 2025 ─────────────────────────────────────────┐│
│  │                                                                         ││
│  │ Channel Effectiveness (applied to forecast):                            ││
│  │   search_spend: 1.25 (25% more effective)                              ││
│  │   social_spend: 0.92 (8% less effective)                               ││
│  │   tv_spend: 1.08 (8% more effective)                                   ││
│  │                                                                         ││
│  │ Context: Demand Index 1.12 (above average base demand this period)     ││
│  │                                                                         ││
│  │ [Use Custom Indices ▼]  (optional override)                            ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                             │
│  [Generate Forecast]                                                        │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│  Results: Incremental Media Contribution                                    │
│                                                                             │
│  Forecast Incremental Response: $1,234,567 (95% CI: $1,100,000 - $1,380,000)│
│  Total Media Spend: $500,000 over 8 weeks                                   │
│  Blended Incremental ROI: 2.47x                                             │
│                                                                             │
│  ℹ️ Includes channel effectiveness adjustment (seasonally adjusted)         │
│  ℹ️ Context: Demand index 1.12 for this period (above average base demand)  │
│                                                                             │
│  📊 Weekly Breakdown Chart                                                  │
│  📊 Channel Contribution Chart                                              │
│                                                                             │
│  ┌─ Sanity Check ─────────────────────────────────────────────────────────┐│
│  │ Your forecast: $1.2M from $500K spend (ROI: 2.4x)                      ││
│  │                                                                         ││
│  │ Historical comparison:                                                  ││
│  │   • Recent (last 8 weeks): $490K → $1.15M (ROI: 2.35x) ✓               ││
│  │   • Same period last year: $520K → $1.28M (ROI: 2.46x) ✓               ││
│  │                                                                         ││
│  │ ✓ Forecast appears reasonable (within ±20% of historical ROI)          ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                             │
│  [Download Report]                                                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Sanity Check Feature

Compare forecast to historical actuals to catch unreasonable predictions.

```python
def compute_sanity_check(
    forecast_spend: float,
    forecast_response: float,
    bridge: OptimizationBridge,
    forecast_num_periods: int,
    forecast_start_month: int,
) -> dict:
    """Compare forecast to historical actuals."""

    forecast_roi = forecast_response / forecast_spend

    # 1. Recent history (same number of periods)
    recent_contribution = bridge.get_contributions_for_period(
        num_periods=forecast_num_periods,
        comparison_mode="most_recent_period"
    )
    recent_spend = bridge.get_recent_spend(num_periods=forecast_num_periods)
    recent_roi = recent_contribution / recent_spend if recent_spend > 0 else None

    # 2. Same period last year (if data available)
    yoy_roi = get_yoy_roi_for_period(bridge, forecast_start_month, forecast_num_periods)

    # 3. Flag if forecast ROI differs significantly from historical
    warnings = []
    if recent_roi and abs(forecast_roi - recent_roi) / recent_roi > 0.20:
        warnings.append(f"Forecast ROI ({forecast_roi:.2f}x) differs >20% from recent ({recent_roi:.2f}x)")

    return {
        "recent_spend": recent_spend,
        "recent_contribution": recent_contribution,
        "recent_roi": recent_roi,
        "yoy_roi": yoy_roi,
        "forecast_roi": forecast_roi,
        "is_reasonable": len(warnings) == 0,
        "warnings": warnings,
    }
```

---

## Test Strategy

### Existing Test Coverage (no changes needed)
| Component | Test File | Tests |
|-----------|-----------|-------|
| SeasonalIndexCalculator | test_seasonality.py | 32+ tests |
| RiskAwareObjective | test_risk_objectives.py | Full coverage |
| OptimizationBridge | test_optimization.py | Full coverage |

### New Tests Required (21 total)

**Core Functionality (5):**
| Test | What it verifies |
|------|------------------|
| `test_forecast_matches_optimizer` | Forecast response ≈ optimizer response for same spend/period |
| `test_seasonal_indices_applied` | Channel indices properly applied to response |
| `test_seasonal_indices_disabled` | Response differs appropriately when `apply_seasonal=False` |
| `test_adstock_continuity` | Historical spend properly warm-starts adstock |
| `test_posterior_sampling_ci` | CIs are valid (low < mean < high, reasonable spread) |

**CSV Validation (5):**
| Test | What it verifies |
|------|------------------|
| `test_csv_missing_channel_columns` | Error when CSV missing required channel columns |
| `test_csv_extra_columns_ignored` | Extra columns don't break forecast |
| `test_csv_invalid_dates` | Error on unparseable dates |
| `test_csv_negative_spend` | Error or warning on negative spend values |
| `test_csv_dates_must_continue_from_model` | Error if CSV dates don't start after model's last date |

**Sanity Check (3):**
| Test | What it verifies |
|------|------------------|
| `test_sanity_check_flags_outliers` | Warnings triggered when forecast ROI deviates >20% |
| `test_sanity_check_reasonable_forecast` | No warnings when forecast aligns with history |
| `test_sanity_check_insufficient_history` | Graceful handling when model has <1 year data |

**Edge Cases (5):**
| Test | What it verifies |
|------|------------------|
| `test_single_week_forecast` | Works for 1-week CSV |
| `test_multi_month_forecast` | Works for 12+ week CSV |
| `test_year_boundary_forecast` | Dec→Jan transition handles month wrapping |
| `test_zero_spend_channel` | Channel with $0 spend handled correctly |
| `test_channel_subset` | Forecast works when CSV has fewer channels than model |

**ForecastResult (3):**
| Test | What it verifies |
|------|------------------|
| `test_weekly_df_structure` | Has correct columns (date, response, ci_low, ci_high) |
| `test_channel_contributions_sum` | Channel contributions sum to total response |
| `test_forecast_period_string` | Period string formatted correctly |

### Critical Consistency Test

```python
def test_forecast_matches_optimizer_response():
    """Forecast and optimizer must produce same response for same inputs."""
    # Run optimizer for Jan with $100k budget
    optimizer_result = allocator.optimize(budget=100000, num_periods=4, ...)

    # Create CSV with optimizer's allocation
    df_spend = create_spend_csv(optimizer_result.allocation, dates=jan_dates)

    # Forecast with same period/indices
    forecast_result = engine.forecast(df_spend, apply_seasonal=True)

    # Should match within sampling variance (~5%)
    assert abs(forecast_result.total_response - optimizer_result.response) / optimizer_result.response < 0.05
```

---

## Implementation Steps

### Step 1: Create `forecasting/forecast_engine.py`
1. Import SeasonalIndexCalculator, OptimizationBridge
2. ForecastResult dataclass with seasonality + sanity check fields
3. SpendForecastEngine class
4. CSV validation (channels, dates, values)
5. Seasonal index computation from dates
6. Adstock application with model history
7. Posterior sampling for response + CIs (reuse RiskAwareObjective pattern)
8. Sanity check computation

### Step 2: Create `ui/pages/forecast.py`
1. CSV upload component
2. Seasonal toggle checkbox
3. Seasonality preview (after CSV upload, before forecast)
4. Optional custom index override
5. Results display with annotations
6. Sanity check display
7. Charts (weekly response, channel contribution)
8. Download report button

### Step 3: Integration
1. Add to navigation in `app.py`
2. Export from `forecasting/__init__.py`

### Step 4: Tests
1. Create test_forecast_engine.py with 21 tests
2. Run consistency test with optimizer
3. Verify all edge cases

---

## Technical Details

### Adstock Continuity

```python
# Get model's last N periods (for adstock warmup)
historical_spend = wrapper.df_raw[channel_columns].tail(adstock_max_lag)

# Concatenate with forecast spend
full_spend = pd.concat([historical_spend, df_forecast_spend])

# Apply adstock transforms
adstocked = apply_adstock(full_spend, wrapper.config)

# Extract just the forecast period (after warmup)
forecast_adstocked = adstocked.iloc[adstock_max_lag:]
```

### Posterior Sampling for CIs

```python
# Get posterior samples from model (reuse RiskAwareObjective pattern)
# Apply saturation, seasonal indices, beta weighting
# Compute response distribution
# Extract percentiles for CIs
```
