"""
Data validation functionality.
"""

import pandas as pd
import numpy as np
from typing import Optional
from dataclasses import dataclass, field
import logging

from ..config.schema import ModelConfig

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of a validation check."""
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add_error(self, message: str) -> None:
        """Add an error message."""
        self.errors.append(message)
        self.valid = False

    def add_warning(self, message: str) -> None:
        """Add a warning message."""
        self.warnings.append(message)

    def merge(self, other: "ValidationResult") -> None:
        """Merge another validation result into this one."""
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        if not other.valid:
            self.valid = False

    def __str__(self) -> str:
        """String representation of validation result."""
        lines = []
        if self.valid:
            lines.append("Validation PASSED")
        else:
            lines.append("Validation FAILED")

        if self.errors:
            lines.append(f"  Errors ({len(self.errors)}):")
            for err in self.errors:
                lines.append(f"    - {err}")

        if self.warnings:
            lines.append(f"  Warnings ({len(self.warnings)}):")
            for warn in self.warnings:
                lines.append(f"    - {warn}")

        return "\n".join(lines)


class DataValidator:
    """Validate data for MMM modeling."""

    def __init__(self, config: ModelConfig):
        """
        Initialize DataValidator.

        Parameters
        ----------
        config : ModelConfig
            Model configuration.
        """
        self.config = config

    def validate_all(self, df: pd.DataFrame) -> ValidationResult:
        """
        Run all validation checks on the dataframe.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe to validate.

        Returns
        -------
        ValidationResult
            Combined validation result.
        """
        result = ValidationResult(valid=True)

        # Run all validation checks
        result.merge(self.validate_structure(df))
        result.merge(self.validate_date_column(df))
        result.merge(self.validate_target_column(df))
        result.merge(self.validate_channel_columns(df))
        result.merge(self.validate_control_columns(df))
        result.merge(self.validate_data_quality(df))

        return result

    def validate_structure(self, df: pd.DataFrame) -> ValidationResult:
        """Validate basic dataframe structure."""
        result = ValidationResult(valid=True)

        if df.empty:
            result.add_error("Dataframe is empty")
            return result

        if len(df) < 20:
            result.add_warning(f"Only {len(df)} observations - model may be unstable")

        return result

    def validate_date_column(self, df: pd.DataFrame) -> ValidationResult:
        """Validate the date column."""
        result = ValidationResult(valid=True)
        date_col = self.config.data.date_column

        if date_col not in df.columns:
            result.add_error(f"Date column '{date_col}' not found in data")
            return result

        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            result.add_error(f"Date column '{date_col}' is not datetime type")
            return result

        # Check for gaps in time series
        df_sorted = df.sort_values(date_col)
        date_diffs = df_sorted[date_col].diff().dropna()

        if len(date_diffs) > 0:
            median_diff = date_diffs.median()
            gaps = date_diffs[date_diffs > median_diff * 2]
            if len(gaps) > 0:
                result.add_warning(
                    f"Found {len(gaps)} potential gaps in time series"
                )

        # Check for duplicates
        duplicates = df[date_col].duplicated().sum()
        if duplicates > 0:
            result.add_error(f"Found {duplicates} duplicate dates")

        return result

    def validate_target_column(self, df: pd.DataFrame) -> ValidationResult:
        """Validate the target (KPI) column."""
        result = ValidationResult(valid=True)
        target_col = self.config.data.target_column

        if target_col not in df.columns:
            result.add_error(f"Target column '{target_col}' not found in data")
            return result

        target = df[target_col]

        # Check for missing values
        missing = target.isna().sum()
        if missing > 0:
            result.add_error(
                f"Target column has {missing} missing values ({missing/len(df)*100:.1f}%)"
            )

        # Check for negative values
        if (target < 0).any():
            neg_count = (target < 0).sum()
            result.add_warning(
                f"Target column has {neg_count} negative values"
            )

        # Check for zeros
        zero_pct = (target == 0).sum() / len(target) * 100
        if zero_pct > 10:
            result.add_warning(
                f"Target column has {zero_pct:.1f}% zero values"
            )

        # Check for outliers
        q1, q3 = target.quantile([0.25, 0.75])
        iqr = q3 - q1
        outliers = ((target < q1 - 3*iqr) | (target > q3 + 3*iqr)).sum()
        if outliers > 0:
            result.add_warning(
                f"Target column has {outliers} potential outliers"
            )

        return result

    def validate_channel_columns(self, df: pd.DataFrame) -> ValidationResult:
        """Validate media channel columns."""
        result = ValidationResult(valid=True)
        channel_cols = self.config.get_channel_columns()

        for col in channel_cols:
            if col not in df.columns:
                result.add_error(f"Channel column '{col}' not found in data")
                continue

            channel_data = df[col]

            # Check for missing values
            missing = channel_data.isna().sum()
            if missing > 0:
                result.add_error(
                    f"Channel '{col}' has {missing} missing values"
                )

            # Check for negative values
            if (channel_data < 0).any():
                neg_count = (channel_data < 0).sum()
                result.add_error(
                    f"Channel '{col}' has {neg_count} negative values (spend can't be negative)"
                )

            # Check for all zeros
            if (channel_data == 0).all():
                result.add_warning(
                    f"Channel '{col}' contains only zeros"
                )

            # Check for very low activity
            non_zero_pct = (channel_data > 0).sum() / len(channel_data) * 100
            if non_zero_pct < 20:
                result.add_warning(
                    f"Channel '{col}' has spend in only {non_zero_pct:.1f}% of periods"
                )

        return result

    def validate_control_columns(self, df: pd.DataFrame) -> ValidationResult:
        """Validate control variable columns."""
        result = ValidationResult(valid=True)

        for ctrl in self.config.controls:
            if ctrl.name not in df.columns:
                result.add_warning(
                    f"Control column '{ctrl.name}' not found in data"
                )
                continue

            ctrl_data = df[ctrl.name]

            # Check for missing values
            missing = ctrl_data.isna().sum()
            if missing > 0:
                result.add_error(
                    f"Control '{ctrl.name}' has {missing} missing values"
                )

            # For dummy variables, check they're 0/1
            if ctrl.is_dummy:
                unique_vals = ctrl_data.dropna().unique()
                if not set(unique_vals).issubset({0, 1, 0.0, 1.0}):
                    result.add_warning(
                        f"Dummy variable '{ctrl.name}' contains values other than 0/1: {unique_vals}"
                    )

        return result

    def validate_data_quality(self, df: pd.DataFrame) -> ValidationResult:
        """General data quality checks."""
        result = ValidationResult(valid=True)

        # Check total spend vs revenue ratio
        channel_cols = [c for c in self.config.get_channel_columns() if c in df.columns]
        target_col = self.config.data.target_column

        if channel_cols and target_col in df.columns:
            total_spend = df[channel_cols].sum().sum()
            total_revenue = df[target_col].sum()

            if total_spend > 0:
                overall_roi = total_revenue / total_spend
                if overall_roi < 0.1:
                    result.add_warning(
                        f"Very low overall ROI ({overall_roi:.2f}) - check data scaling"
                    )
                elif overall_roi > 100:
                    result.add_warning(
                        f"Very high overall ROI ({overall_roi:.2f}) - check data scaling"
                    )

        return result

    def get_data_summary(self, df: pd.DataFrame) -> dict:
        """
        Get a summary of the data.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe to summarize.

        Returns
        -------
        dict
            Summary statistics.
        """
        date_col = self.config.data.date_column
        target_col = self.config.data.target_column
        channel_cols = [c for c in self.config.get_channel_columns() if c in df.columns]

        # Handle date range - convert to datetime if needed
        date_start = None
        date_end = None
        if date_col in df.columns:
            dates = pd.to_datetime(df[date_col], errors='coerce')
            if not dates.isna().all():
                date_start = dates.min().isoformat()
                date_end = dates.max().isoformat()

        summary = {
            "n_observations": len(df),
            "date_range": {
                "start": date_start,
                "end": date_end,
            },
            "target": {
                "column": target_col,
                "mean": float(df[target_col].mean()) if target_col in df.columns else None,
                "std": float(df[target_col].std()) if target_col in df.columns else None,
                "min": float(df[target_col].min()) if target_col in df.columns else None,
                "max": float(df[target_col].max()) if target_col in df.columns else None,
                "total": float(df[target_col].sum()) if target_col in df.columns else None,
            },
            "channels": {},
            "total_spend": 0.0,
        }

        for col in channel_cols:
            spend = float(df[col].sum())
            summary["channels"][col] = {
                "total_spend": spend,
                "mean_spend": float(df[col].mean()),
                "non_zero_periods": int((df[col] > 0).sum()),
                "pct_active": float((df[col] > 0).sum() / len(df) * 100),
            }
            summary["total_spend"] += spend

        if summary["total_spend"] > 0 and summary["target"]["total"]:
            summary["overall_roi"] = summary["target"]["total"] / summary["total_spend"]

        return summary
