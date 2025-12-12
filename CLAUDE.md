# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MMM Platform is a Marketing Mix Modeling application built on PyMC-Marketing. It provides Bayesian inference for marketing channel effectiveness with a Streamlit UI for interactive analysis.

## Common Commands

```bash
# Run the Streamlit UI
streamlit run mmm_platform/ui/app.py

# Run all tests
.\venv\Scripts\python.exe -m pytest mmm_platform/tests/ -v --tb=short

# Run single test file
.\venv\Scripts\python.exe -m pytest mmm_platform/tests/unit/test_schema.py -v --tb=short

# Run tests matching pattern
.\venv\Scripts\python.exe -m pytest mmm_platform/tests/ -v -k "test_roi"

# CLI commands
python cli.py run --config config.yaml --data data.csv
python cli.py validate --data data.csv --config config.yaml
python cli.py ui --port 8501
```

## Architecture

### Core Modeling Pipeline (`mmm_platform/`)

**MMMWrapper** (`model/mmm.py`) orchestrates the full pipeline:
1. Data loading via `DataLoader` (`core/data_loader.py`)
2. Validation via `DataValidator` (`core/validation.py`)
3. Prior calibration via `PriorCalibrator` (`core/priors.py`)
4. Transform computation via `TransformEngine` (`core/transforms.py`)
5. Model building/fitting using PyMC-Marketing's `MMM` class
6. Results extraction and persistence

**ModelConfig** (`config/schema.py`) is the central Pydantic schema defining:
- `ChannelConfig` - Paid media with ROI priors and adstock settings
- `OwnedMediaConfig` - Owned media (email, organic) with optional ROI tracking
- `CompetitorConfig` - Competitor activity (adstock only, negative constraint)
- `ControlConfig` - Control variables with optional adstock
- `KPIType` enum - Determines "ROI" vs "Cost Per X" terminology

### Variable Types and Transform Rules

| Type | Adstock | Saturation | ROI Priors | Coefficient |
|------|---------|------------|------------|-------------|
| Paid Media (channels) | Yes | Yes | Required | Positive |
| Owned Media | Yes | Yes | Optional | Positive |
| Competitors | Yes | No | No | Negative |
| Controls | Optional | No | No | Configurable |

### Budget Optimization (`optimization/`)

- `BudgetAllocator` (`allocator.py`) - High-level optimization interface
- `OptimizationBridge` (`bridge.py`) - Wraps PyMC-Marketing's optimizer
- Supports utility functions: mean, VaR, CVaR, Sharpe
- `ScenarioManager` (`scenarios.py`) - Save/compare optimization scenarios

### Analysis (`analysis/`)

- `ContributionAnalyzer` (`contributions.py`) - ROI and decomposition analysis
- `marginal_roi.py` - Breakeven spend and headroom calculations
- `executive_summary.py` - Investment recommendations (INCREASE/HOLD/REDUCE)
- `combined_models.py` - Multi-outcome optimization across models

### Streamlit UI (`ui/`)

Pages follow workflow order:
1. `upload_data.py` - CSV upload with validation
2. `configure_model.py` - Channel/control setup, prior configuration, model fitting
3. `results.py` - ROI analysis, contributions, diagnostics
4. `optimize.py` - Budget allocation optimization
5. `compare_models.py` - Side-by-side model comparison
6. `export.py` - CSV/JSON/HTML report generation

Session state keys: `current_data`, `current_config`, `current_model`, `model_fitted`, `active_client`

### Model Persistence

Models saved via `ModelPersistence` (`model/persistence.py`) to `saved_models/{client}/{model_name}/`:
- `wrapper_state.json` - Configuration and metadata
- `mmm_model.pkl` - Fitted PyMC-Marketing model (cloudpickle)
- `idata.nc` - ArviZ InferenceData with posterior samples
- `df_raw.parquet`, `df_scaled.parquet` - Data snapshots

### EC2 Remote Execution

The platform supports running model fitting on EC2 for faster computation:
- `core/ec2_runner.py` - Remote execution coordinator
- `api/server.py` - FastAPI server for EC2 instance
- `api/client.py` - Client for communicating with EC2

## Key Dependencies

- **pymc-marketing**: Core MMM implementation (GeometricAdstock, LogisticSaturation)
- **nutpie**: Fast CPU sampler (auto-detected, falls back to default PyMC sampler)
- **streamlit**: UI framework
- **pydantic**: Configuration validation

## Testing

Tests in `mmm_platform/tests/unit/` cover:
- Config schema validation and migration
- Data upload and persistence
- Transform calculations
- ROI/contribution analysis
- Optimization functionality

Use `pytest -x` to stop on first failure during debugging.
