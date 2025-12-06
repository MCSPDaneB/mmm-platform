# Running Tests

## Quick Start

From the project root directory (`C:\localProjects\pymc_testing\mmm_platform`):

```bash
# Run all tests
.\venv\Scripts\python.exe -m pytest mmm_platform/tests/ -v --tb=short

# Run all tests (stop on first failure)
.\venv\Scripts\python.exe -m pytest mmm_platform/tests/ -v --tb=short -x
```

## Common Options

| Option | Description |
|--------|-------------|
| `-v` | Verbose - shows each test name |
| `--tb=short` | Short traceback on failures |
| `-x` | Stop on first failure |
| `-k "pattern"` | Run tests matching pattern |

## Run Specific Test Files

```bash
# Run a specific test file
.\venv\Scripts\python.exe -m pytest mmm_platform/tests/unit/test_schema.py -v --tb=short

# Run tests matching a pattern
.\venv\Scripts\python.exe -m pytest mmm_platform/tests/ -v -k "test_roi"
```

## Test Structure

```
mmm_platform/tests/
├── unit/
│   ├── test_config_loading.py    # Config save/load tests
│   ├── test_configure_model.py   # Date clamping, cost/efficiency conversion
│   ├── test_data_persistence.py  # Original data preservation
│   ├── test_data_upload.py       # Data upload validation
│   ├── test_deprecations.py      # Deprecated parameter checks
│   ├── test_diagnostics.py       # Convergence diagnostics
│   ├── test_results_display.py   # Cost-per display conversion
│   ├── test_schema.py            # Model config schema validation
│   └── test_transforms.py        # Adstock/saturation transforms
└── README.md
```
