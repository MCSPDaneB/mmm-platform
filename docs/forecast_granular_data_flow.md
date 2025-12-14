# Forecast Granular Data Flow

This document describes how the Forecast feature handles granular spend data uploads.

## Overview

The forecast feature supports two data formats:
1. **Aggregated format** - CSV with model variable names as columns (e.g., `google_search_spend`)
2. **Granular format** - CSV with level columns (e.g., `media_channel_lvl1`, `lvl2`, `lvl3`)

When granular data is uploaded, it must be mapped to model variables before forecasting.

## Data Formats

### Aggregated Format (Direct)
```csv
date,google_search_spend,facebook_spend,tv_spend
2025-01-06,80000,25000,50000
2025-01-13,85000,30000,50000
```

### Granular Format (Requires Mapping)
```csv
date,media_channel_lvl1,media_channel_lvl2,media_channel_lvl3,spend
2025-01-06,Google,Search,Brand,50000
2025-01-06,Google,Search,NonBrand,30000
2025-01-06,Facebook,Social,Brand,25000
```

## Mapping Storage

The mapping from granular entities to model variables is stored in `ModelConfig.forecast_spend_mapping`:

```python
class ForecastSpendMapping(BaseModel):
    level_columns: list[str]      # ["media_channel_lvl1", "lvl2", "lvl3"]
    date_column: str = "date"
    spend_column: str = "spend"
    entity_mappings: dict[str, str]  # {"Google|Search|Brand": "google_search_spend", ...}
```

This persists with the model via `config.json`, so subsequent uploads auto-apply the saved mapping.

## UI Flow Diagram

```
User uploads spend CSV
        |
        v
+-------------------+
| Detect format     |
+--------+----------+
         |
         +-- Aggregated format ------> Proceed to forecast (current flow)
         |   (model variable names)
         |
         +-- Granular format --------+
             (lvl columns)           |
                                     v
                        +-------------------------+
                        | Check for saved mapping |
                        +------------+------------+
                                     |
                     +---------------+---------------+
                     |                               |
                     v                               v
         +-------------------+         +-------------------+
         | Has saved mapping |         | No saved mapping  |
         +---------+---------+         +---------+---------+
                   |                             |
                   v                             v
         +-------------------+         +-------------------+
         | Validate entities |         | Show mapping UI   |
         | against saved     |         | (entity -> var)   |
         +---------+---------+         +---------+---------+
                   |                             |
         +---------+---------+                   |
         |                   |                   |
         v                   v                   v
     All match         Has new           Save mapping
         |             entities              |
         v                   |               v
   Auto-apply          Map new        Store in config
   mapping             entities              |
         |                   |               |
         +---------+---------+---------------+
                   |
                   v
         +-------------------+
         | Aggregate to      |
         | model variables   |
         +---------+---------+
                   |
                   v
         +-------------------+
         | Run forecast      |
         | (existing flow)   |
         +-------------------+
```

## Validation Rules

When a saved mapping exists, the system validates uploaded data:

| Status | Description | Action |
|--------|-------------|--------|
| All match | All file entities exist in saved mapping | Auto-apply mapping |
| New entities | File contains entities not in mapping | Prompt user to map new entities |
| Missing entities | Saved mapping has entities not in file | Warning only (data may be partial) |

## Implementation Files

| File | Purpose |
|------|---------|
| `config/schema.py` | `ForecastSpendMapping` schema definition |
| `forecasting/forecast_engine.py` | Aggregation and validation methods |
| `ui/pages/forecast.py` | Format detection and mapping UI |

## Related Documentation

- [Future Channel Aggregation](future_channel_aggregation.md) - Related granular data lineage planning
