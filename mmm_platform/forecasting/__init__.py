"""
Forecasting module for spend-to-response predictions.

Provides tools for forecasting incremental media response from
planned or actual spend data.
"""

from mmm_platform.forecasting.forecast_engine import (
    SpendForecastEngine,
    ForecastResult,
    ValidationError,
    # Granular data functions
    detect_spend_format,
    get_level_columns,
    validate_against_saved_mapping,
    aggregate_granular_spend,
    # Overlap detection
    OverlapAnalysis,
    check_forecast_overlap,
)

__all__ = [
    "SpendForecastEngine",
    "ForecastResult",
    "ValidationError",
    # Granular data functions
    "detect_spend_format",
    "get_level_columns",
    "validate_against_saved_mapping",
    "aggregate_granular_spend",
    # Overlap detection
    "OverlapAnalysis",
    "check_forecast_overlap",
]
