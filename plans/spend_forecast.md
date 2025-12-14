# Spend Forecast Feature

## Status: PENDING

## Summary
Upload actual/planned media spend (CSV) and get forecasted KPI response with proper Bayesian credible intervals. Shows both total and week-by-week breakdown.

## Use Case
- Model is fitted on historical data
- User has new media spend data (from media agency, planned spend, etc.)
- They don't have API feeds for all base variables (trends, controls, etc.)
- They just want to quickly see: "If we spend this, what response do we expect?"

## Design Decisions
| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Input method** | CSV upload | Standard format, easy from Excel/agencies |
| **Output granularity** | Total + weekly | Both summary and detailed breakdown |
| **Adstock handling** | Use model history | CSV continues from model's data, warm start |
| **Uncertainty** | Proper Bayesian CIs | Sample from posterior for accurate intervals |

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

## Files to Create/Modify

### 1. `mmm_platform/forecasting/forecast_engine.py` (CREATE)
```python
@dataclass
class ForecastResult:
    total_response: float
    total_ci_low: float
    total_ci_high: float
    weekly_df: pd.DataFrame  # date, response, ci_low, ci_high
    channel_contributions: pd.DataFrame  # date, channel, contribution

class SpendForecastEngine:
    """Forecast KPI response from actual/planned media spend."""

    def __init__(self, wrapper: MMMWrapper):
        self.wrapper = wrapper
        self.bridge = OptimizationBridge(wrapper)

    def forecast(
        self,
        df_spend: pd.DataFrame,
        apply_seasonal: bool = True,
    ) -> ForecastResult:
        """
        Forecast response for weekly spend data.

        1. Validate spend columns match model channels
        2. Append to model's historical data (for adstock continuity)
        3. Apply transforms (adstock, saturation)
        4. Sample from posterior for response + CIs
        5. Return total + weekly breakdown
        """
        pass
```

### 2. `mmm_platform/ui/pages/forecast.py` (CREATE)
New page for spend forecasting:
```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Spend Forecast                                                             │
│  Forecast KPI response from planned media spend                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Upload spend CSV:  [Browse...] [Upload]                                    │
│                                                                             │
│  ☑ Apply seasonal adjustment                                                │
│                                                                             │
│  [Generate Forecast]                                                        │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│  Results:                                                                   │
│                                                                             │
│  Total Forecast: $1,234,567 (95% CI: $1,100,000 - $1,380,000)              │
│  Total Spend: $500,000 over 8 weeks                                        │
│  Blended ROI: 2.47x                                                         │
│                                                                             │
│  📊 Weekly Breakdown Chart (response over time)                             │
│  📋 Weekly Table: Date | Spend | Response | CI Low | CI High               │
│                                                                             │
│  📊 Channel Contribution Chart (stacked area)                               │
│  📋 Channel Table: Channel | Spend | Contribution | %                       │
│                                                                             │
│  [Download Report]                                                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3. `mmm_platform/forecasting/__init__.py` (CREATE)
Export `SpendForecastEngine`, `ForecastResult`

### 4. `mmm_platform/ui/app.py` (MODIFY)
Add "Forecast" page to navigation

### 5. `mmm_platform/tests/unit/test_forecast_engine.py` (CREATE)
- Test CSV validation
- Test adstock continuity from model history
- Test posterior sampling for CIs
- Test weekly vs total aggregation

## Implementation Steps

### Step 1: Create `forecasting/forecast_engine.py`
1. `ForecastResult` dataclass
2. `SpendForecastEngine` class
3. CSV validation (channels, dates, values)
4. Adstock application with model history
5. Posterior sampling for response + CIs

### Step 2: Create `ui/pages/forecast.py`
1. CSV upload component
2. Options (seasonal adjustment)
3. Results display (total + weekly)
4. Charts (weekly response, channel contribution)
5. Download report button

### Step 3: Integration
1. Add to navigation in `app.py`
2. Export from `forecasting/__init__.py`

### Step 4: Tests
1. Unit tests for engine
2. Validation tests for CSV format

## Technical Details

### Adstock Continuity
The CSV spend continues from model's training data:
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
# Get posterior samples from model
idata = wrapper.idata
beta_samples = idata.posterior['beta_channel'].values  # (chain, draw, channel)

# For each posterior sample, compute response
responses = []
for sample in beta_samples.reshape(-1, n_channels):
    response = compute_response(forecast_spend, sample, lam_values)
    responses.append(response)

# Compute credible intervals
ci_low = np.percentile(responses, 2.5)
ci_high = np.percentile(responses, 97.5)
mean_response = np.mean(responses)
```

## Files Summary

| File | Action | Description |
|------|--------|-------------|
| `forecasting/forecast_engine.py` | CREATE | SpendForecastEngine class |
| `forecasting/__init__.py` | CREATE | Module exports |
| `ui/pages/forecast.py` | CREATE | Forecast UI page |
| `ui/app.py` | MODIFY | Add Forecast to navigation |
| `tests/unit/test_forecast_engine.py` | CREATE | Unit tests |
