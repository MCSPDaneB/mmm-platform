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

#### SLSQP Optimizer Constraint Handling

The optimizer uses scipy's SLSQP with an equality constraint to enforce exact budget allocation. Two key implementation details prevent constraint violations:

1. **Analytical Constraint Jacobian**: The budget constraint must include an explicit Jacobian for numerical stability:
   ```python
   constraints = {
       'type': 'eq',
       'fun': lambda x: budget - x.sum(),
       'jac': lambda x: -np.ones(n_channels)  # Critical for convergence
   }
   ```
   Without the Jacobian, SLSQP uses numerical differentiation which can fail at "corners" where channel bounds intersect with the budget constraint.

2. **trust-constr Fallback**: When SLSQP fails to satisfy the constraint (>1% violation), the optimizer automatically falls back to scipy's `trust-constr` method which handles constrained optimization more robustly.

**Why this matters**: The logistic saturation curves create a non-convex optimization landscape. When the budget falls between a single channel's upper bound and the sum of multiple channels' optimal allocations, SLSQP can get stuck. The combination of analytical Jacobian + fallback optimizer ensures the budget constraint is always satisfied.

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

## Debugging Guidelines

### Validating Feature Concepts

Before implementing features borrowed from other domains (finance, portfolio theory, etc.):

1. **Question the analogy** - Does the concept actually apply to MMM/marketing? (e.g., VaR optimization doesn't create intuitive risk/return trade-offs in MMM due to saturation curves)
2. **Check for meaningful trade-offs** - Will different options produce meaningfully different results with the expected ordering?
3. **Validate with simple tests** - Run quick tests to confirm expected behavior before full implementation
4. **Ask the user** - If unsure whether a concept translates, ask before implementing

### Debugging with User Input

When the user has already performed debugging and shares observations:

1. **Trust their validated findings** - If they say something works (e.g., "the validator matches perfectly"), don't re-investigate that component
2. **Ask what they've already ruled out** before starting a fresh investigation
3. **Focus on the gap** between what works and what doesn't, rather than re-examining everything from scratch
4. **Leverage their insights** - The user often has domain knowledge and debugging context that can shortcut investigation

### UI Bug Investigation

When investigating UI-related issues in the Streamlit app:

1. **Ask for the exact workflow** - Get the user to describe their click-by-click steps
2. **Identify related state variables** - Check `st.session_state` keys that should update together
3. **Look for sync issues** - Common bug pattern: one state variable updates but related variables don't
4. **Check callbacks** - Streamlit callbacks (like `_fill_budget_callback`) may not update all dependent state

### Debugging Different Behavior in Similar Code Paths

When investigating why two code paths that should behave identically produce different results:

1. **FIRST**: Trace and compare the EXACT function calls with ALL parameters for both paths
2. **SECOND**: Identify which parameters differ (don't hypothesize - verify with debug output)
3. **THIRD**: Only after finding the actual difference, propose a fix
4. Do NOT propose fixes before understanding the root cause
5. Do NOT add incremental debug statements in circles - add comprehensive comparison logging once

### Root Cause Before Fix

Never propose fix options (Option A, B, C) until you have identified the ACTUAL difference causing the problem. "The inputs look the same but outputs differ" means you haven't found the real input difference yet - keep looking, don't start proposing workarounds.

### When to Skip Plan Mode for Debugging

Skip plan mode when:
- Adding debug/print statements to understand behavior
- Comparing two code paths to find differences
- Running quick experiments to isolate issues

Just make the debug changes directly and iterate quickly.

### Debugging Scipy Optimization Issues

When scipy optimizers (SLSQP, trust-constr, etc.) fail or return incorrect results:

1. **Always provide analytical Jacobians** - Missing Jacobians cause numerical instability at constraint boundaries
2. **Check constraint satisfaction** - Verify `result.x` actually satisfies constraints, don't trust `result.success`
3. **Use fallback optimizers** - If SLSQP fails, try trust-constr; different methods handle non-convex landscapes differently
4. **Watch for "corner" cases** - When bounds intersect with equality constraints, optimizers can get stuck
5. **Non-convex objectives** - Saturation/logistic functions create non-convex landscapes that local optimizers struggle with

### Streamlit Session State Consistency

When building multi-mode UIs (e.g., optimize/target/scenario modes):

1. **Share session state keys** - All modes using the same parameter must use the same key (e.g., `opt_num_periods`, not `target_num_periods` + `scenario_num_periods`)
2. **Don't override user settings in callbacks** - Fill/autopopulate callbacks should not change user-configured parameters like forecast periods
3. **Centralize shared settings** - Put parameters used by multiple modes in a dedicated Settings tab, not duplicated per mode
4. **Test mode consistency** - If mode A and mode B use the same inputs, they must produce the same outputs

### Plan Mode Discussion Flow

When in plan mode, stay in discussion/refinement mode until the user explicitly signals readiness:

1. **Continue discussion if** - User asks questions, raises concerns, or says "what about X"
2. **Exit plan mode if** - User says "looks good", "let's do it", "ready to implement"
3. **When uncertain** - Ask: "Does this address your concerns, or is there more to discuss?"

Don't prematurely attempt to exit plan mode while the user is still exploring the design space.

### Designing Features That Extend Existing Functionality

When building features that should align with existing code (like making a new feature match optimizer behavior):

1. **Trace the existing code path first** - Find exactly which functions/parameters are used
2. **Document what IS and ISN'T applied** - Don't assume; verify with code search (e.g., "demand_index is computed and displayed but NOT applied to response calculation")
3. **Explain current behavior to user** - Before proposing changes, confirm understanding of existing implementation
4. **Design for consistency** - New feature should use same functions/logic, not recreate them

### Before Implementing New Features

Before writing new code, proactively verify:

1. **Reusable components** - Search for existing functions that do similar work (use Grep/Glob to find candidates)
2. **Test coverage** - Check if areas being modified have existing tests; note gaps
3. **Consistency** - New code should use same patterns as existing codebase
4. **No breaking changes** - Plan to wrap/extend existing tested functions, not modify them
5. **Sanity checks** - Consider adding validation that compares new outputs to historical/expected values
