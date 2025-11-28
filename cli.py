#!/usr/bin/env python
"""
Command-line interface for MMM Platform.

Usage:
    python cli.py run --config config.yaml --data data.csv
    python cli.py validate --data data.csv
    python cli.py export --model saved_models/my_model --output results/
"""

import argparse
import sys
import logging
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="MMM Platform - Marketing Mix Modeling CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run a model")
    run_parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="Path to YAML configuration file"
    )
    run_parser.add_argument(
        "--data", "-d",
        type=str,
        required=True,
        help="Path to CSV data file"
    )
    run_parser.add_argument(
        "--output", "-o",
        type=str,
        default="./output",
        help="Output directory for results"
    )
    run_parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick run with fewer samples"
    )

    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate data")
    validate_parser.add_argument(
        "--data", "-d",
        type=str,
        required=True,
        help="Path to CSV data file"
    )
    validate_parser.add_argument(
        "--config", "-c",
        type=str,
        help="Optional: Path to YAML configuration file"
    )

    # Export command
    export_parser = subparsers.add_parser("export", help="Export results from saved model")
    export_parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        help="Path to saved model directory"
    )
    export_parser.add_argument(
        "--output", "-o",
        type=str,
        default="./output",
        help="Output directory"
    )
    export_parser.add_argument(
        "--format",
        type=str,
        choices=["csv", "json", "html", "all"],
        default="all",
        help="Export format"
    )

    # UI command
    ui_parser = subparsers.add_parser("ui", help="Launch Streamlit UI")
    ui_parser.add_argument(
        "--port", "-p",
        type=int,
        default=8501,
        help="Port to run Streamlit on"
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Route to command handlers
    if args.command == "run":
        cmd_run(args)
    elif args.command == "validate":
        cmd_validate(args)
    elif args.command == "export":
        cmd_export(args)
    elif args.command == "ui":
        cmd_ui(args)


def cmd_run(args):
    """Run a model from command line."""
    from mmm_platform.config.loader import ConfigLoader
    from mmm_platform.model.mmm import MMMWrapper
    from mmm_platform.model.persistence import ModelPersistence
    from mmm_platform.analysis.reporting import ReportGenerator

    logger.info("=" * 60)
    logger.info("MMM Platform - Model Run")
    logger.info("=" * 60)

    # Load configuration
    logger.info(f"Loading configuration from: {args.config}")
    config = ConfigLoader.from_yaml(args.config)

    # Create wrapper and load data
    logger.info(f"Loading data from: {args.data}")
    wrapper = MMMWrapper(config)
    validation = wrapper.load_data(args.data)

    if not validation.valid:
        logger.error("Data validation failed:")
        for error in validation.errors:
            logger.error(f"  - {error}")
        sys.exit(1)

    logger.info("Data validation passed")

    # Prepare data
    logger.info("Preparing data...")
    wrapper.prepare_data()

    # Build and fit model
    logger.info("Building model...")
    wrapper.build_model()

    if args.quick:
        logger.info("Quick run: using reduced samples")
        wrapper.fit(draws=500, tune=500, chains=2)
    else:
        logger.info("Fitting model...")
        wrapper.fit()

    # Get results
    fit_stats = wrapper.get_fit_statistics()
    logger.info(f"Fit complete - R²: {fit_stats['r2']:.3f}, MAPE: {fit_stats['mape']:.1f}%")

    # Save outputs
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save model
    model_dir = output_dir / f"model_{timestamp}"
    ModelPersistence.save(wrapper, model_dir)
    logger.info(f"Model saved to: {model_dir}")

    # Generate reports
    report_gen = ReportGenerator(wrapper)

    report_gen.generate_json_report(output_dir / f"report_{timestamp}.json")
    report_gen.generate_html_report(output_dir / f"report_{timestamp}.html")
    report_gen.export_contributions_csv(output_dir / f"contributions_{timestamp}.csv")

    logger.info(f"Reports saved to: {output_dir}")
    logger.info("=" * 60)
    logger.info("Complete!")


def cmd_validate(args):
    """Validate data from command line."""
    import pandas as pd
    from mmm_platform.config.loader import ConfigLoader
    from mmm_platform.core.validation import DataValidator
    from mmm_platform.config.schema import ModelConfig, DataConfig

    logger.info("=" * 60)
    logger.info("MMM Platform - Data Validation")
    logger.info("=" * 60)

    # Load data
    logger.info(f"Loading data from: {args.data}")
    df = pd.read_csv(args.data)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    # Load or create minimal config
    if args.config:
        config = ConfigLoader.from_yaml(args.config)
    else:
        # Create minimal config for validation
        logger.warning("No config provided - using minimal validation")
        print("\nColumns in data:")
        for col in df.columns:
            print(f"  - {col}")
        return

    # Validate
    validator = DataValidator(config)
    result = validator.validate_all(df)

    print("\n" + str(result))

    # Show summary
    summary = validator.get_data_summary(df)
    print("\nData Summary:")
    print(f"  Observations: {summary['n_observations']}")
    print(f"  Date range: {summary['date_range']['start']} to {summary['date_range']['end']}")
    print(f"  Total spend: ${summary['total_spend']:,.2f}")
    print(f"  Total revenue: ${summary['target']['total']:,.2f}")

    if not result.valid:
        sys.exit(1)


def cmd_export(args):
    """Export results from a saved model."""
    from mmm_platform.model.persistence import ModelPersistence
    from mmm_platform.model.mmm import MMMWrapper
    from mmm_platform.analysis.reporting import ReportGenerator

    logger.info("=" * 60)
    logger.info("MMM Platform - Export Results")
    logger.info("=" * 60)

    # Load model
    logger.info(f"Loading model from: {args.model}")
    wrapper = ModelPersistence.load(args.model, MMMWrapper)

    if wrapper.idata is None:
        logger.error("Model was not fitted - cannot export results")
        sys.exit(1)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate reports
    report_gen = ReportGenerator(wrapper)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.format in ("csv", "all"):
        report_gen.export_contributions_csv(output_dir / f"contributions_{timestamp}.csv")
        logger.info("Exported CSV")

    if args.format in ("json", "all"):
        report_gen.generate_json_report(output_dir / f"report_{timestamp}.json")
        logger.info("Exported JSON")

    if args.format in ("html", "all"):
        report_gen.generate_html_report(output_dir / f"report_{timestamp}.html")
        logger.info("Exported HTML")

    logger.info(f"Results exported to: {output_dir}")


def cmd_ui(args):
    """Launch the Streamlit UI."""
    import subprocess

    logger.info("Launching Streamlit UI...")
    logger.info(f"Access at: http://localhost:{args.port}")

    subprocess.run([
        "streamlit", "run",
        "mmm_platform/ui/app.py",
        f"--server.port={args.port}",
        "--server.address=localhost",
    ])


if __name__ == "__main__":
    main()
