# MMM Platform

A comprehensive Marketing Mix Modeling platform built with PyMC-Marketing, featuring a Streamlit UI for interactive analysis.

## Features

- **Bayesian MMM**: Full posterior distributions using PyMC-Marketing
- **ROI-Informed Priors**: Calibrate priors based on expected channel performance
- **Adstock & Saturation**: Geometric adstock and logistic saturation transforms
- **Interactive UI**: Streamlit-based interface for data upload, configuration, and analysis
- **Comprehensive Diagnostics**: Convergence checks, residual analysis, fit statistics
- **Flexible Configuration**: YAML configs with database storage option
- **Export Options**: CSV, JSON, and HTML reports
- **Docker Ready**: Containerized deployment with PostgreSQL

## Quick Start

### Local Development

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the UI**:
   ```bash
   streamlit run mmm_platform/ui/app.py
   ```

3. **Or use the CLI**:
   ```bash
   python cli.py run --config configs/my_config.yaml --data data/my_data.csv
   ```

### Docker Deployment

1. **Build and run with Docker Compose**:
   ```bash
   cd docker
   docker-compose up --build
   ```

2. **Access the UI** at http://localhost:8501

## Project Structure

```
mmm_platform/
├── docker/                    # Docker configuration
│   ├── Dockerfile
│   └── docker-compose.yml
├── configs/                   # YAML configuration templates
│   └── default_config.yaml
├── mmm_platform/              # Main package
│   ├── config/                # Configuration management
│   │   ├── schema.py          # Pydantic schemas
│   │   └── loader.py          # YAML/DB loading
│   ├── core/                  # Core functionality
│   │   ├── data_loader.py     # Data loading & preparation
│   │   ├── validation.py      # Data validation
│   │   ├── transforms.py      # Adstock & saturation
│   │   └── priors.py          # Prior calibration
│   ├── model/                 # Model management
│   │   ├── mmm.py             # Main MMM wrapper
│   │   ├── fitting.py         # Fitting utilities
│   │   └── persistence.py     # Save/load models
│   ├── analysis/              # Results analysis
│   │   ├── diagnostics.py     # Model diagnostics
│   │   ├── contributions.py   # Contribution analysis
│   │   └── reporting.py       # Report generation
│   ├── database/              # Database layer
│   │   ├── models.py          # SQLAlchemy models
│   │   ├── connection.py      # DB connection
│   │   └── repository.py      # CRUD operations
│   └── ui/                    # Streamlit UI
│       ├── app.py             # Main app
│       └── pages/             # Page modules
├── cli.py                     # Command-line interface
├── requirements.txt
└── README.md
```

## Configuration

Create a YAML configuration file based on `configs/default_config.yaml`:

```yaml
name: "my_mmm_model"

data:
  date_column: "time"
  target_column: "revenue"
  revenue_scale: 1000.0

channels:
  - name: "google_spend"
    adstock_type: "medium"
    roi_prior_low: 1.0
    roi_prior_mid: 3.0
    roi_prior_high: 6.0

controls:
  - name: "promo_event"
    sign_constraint: "positive"
    is_dummy: true

sampling:
  draws: 1500
  tune: 1500
  chains: 4
```

## CLI Commands

```bash
# Run a model
python cli.py run --config config.yaml --data data.csv --output ./results

# Quick run (fewer samples)
python cli.py run --config config.yaml --data data.csv --quick

# Validate data
python cli.py validate --data data.csv --config config.yaml

# Export results from saved model
python cli.py export --model saved_models/my_model --output ./results --format all

# Launch UI
python cli.py ui --port 8501
```

## Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
DATABASE_URL=postgresql://user:pass@localhost:5432/mmm_db
STREAMLIT_SERVER_PORT=8501
```

## UI Workflow

1. **Upload Data**: CSV with date, target, and channel columns
2. **Configure Model**: Select channels, set ROI priors, define controls
3. **Run Model**: Fit the Bayesian model
4. **View Results**: Analyze ROI, contributions, diagnostics
5. **Export**: Download reports and contribution data

## Requirements

- Python 3.11+
- PyMC 5.10+
- PyMC-Marketing 0.4+
- PostgreSQL (optional, for production)

## License

MIT
