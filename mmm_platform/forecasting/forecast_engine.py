"""
Spend forecast engine for media incrementality predictions.

Forecasts incremental response from planned/actual media spend using
the same channel effectiveness indices as the optimizer to ensure consistency.
"""

from dataclasses import dataclass, field
from typing import Any
import numpy as np
import pandas as pd
import logging
from datetime import datetime

from mmm_platform.optimization.seasonality import SeasonalIndexCalculator
from mmm_platform.optimization.bridge import OptimizationBridge
from mmm_platform.optimization.risk_objectives import PosteriorSamples

logger = logging.getLogger(__name__)


@dataclass
class ForecastResult:
    """
    Result of a spend forecast.

    Contains both the forecasted response and metadata about how
    the forecast was computed.
    """

    # Response metrics
    total_response: float
    total_ci_low: float
    total_ci_high: float
    weekly_df: pd.DataFrame  # date, response, ci_low, ci_high, spend
    channel_contributions: pd.DataFrame  # date, channel, contribution

    # Input summary
    total_spend: float
    num_weeks: int

    # Seasonality info
    seasonal_applied: bool
    seasonal_indices: dict[str, float]  # channel -> index used
    demand_index: float
    forecast_period: str  # e.g., "Jan-Mar 2025"


    @property
    def blended_roi(self) -> float:
        """Compute blended ROI across all channels."""
        if self.total_spend <= 0:
            return 0.0
        return self.total_response / self.total_spend


@dataclass
class ValidationError:
    """Validation error details."""
    field: str
    message: str


class SpendForecastEngine:
    """
    Forecast incremental media response from planned/actual spend.

    This is an incrementality checker - it forecasts media-driven
    contribution only, not full KPI. Uses the same channel effectiveness
    indices as the optimizer to ensure consistency.

    Parameters
    ----------
    wrapper : MMMWrapper
        A fitted MMMWrapper instance.

    Examples
    --------
    >>> engine = SpendForecastEngine(wrapper)
    >>> df_spend = pd.read_csv("planned_spend.csv")
    >>> result = engine.forecast(df_spend, apply_seasonal=True)
    >>> print(f"Expected response: ${result.total_response:,.0f}")
    """

    def __init__(self, wrapper: Any):
        """
        Initialize the forecast engine.

        Parameters
        ----------
        wrapper : MMMWrapper
            A fitted MMMWrapper instance with idata.
        """
        if wrapper.idata is None:
            raise ValueError(
                "MMMWrapper is not fitted. Call wrapper.fit() before forecasting."
            )

        self.wrapper = wrapper
        self.bridge = OptimizationBridge(wrapper)
        self.seasonal_calculator = SeasonalIndexCalculator(wrapper)

        # Extract posterior samples for uncertainty quantification
        self.posterior_samples = PosteriorSamples.from_idata(
            wrapper.idata, n_samples=500
        )

        # Cache channel info
        self._channel_columns = self.bridge.channel_columns
        self._x_maxes = self._compute_x_maxes()

        logger.info(
            f"SpendForecastEngine initialized for {len(self._channel_columns)} channels"
        )

    def _compute_x_maxes(self) -> np.ndarray:
        """Compute max spend per channel for normalization."""
        x_maxes = []
        spend_scale = self.wrapper.config.data.spend_scale

        for ch in self._channel_columns:
            if ch in self.wrapper.df_scaled.columns:
                max_val = self.wrapper.df_scaled[ch].max() * spend_scale
                x_maxes.append(max(max_val, 1e-9))
            else:
                x_maxes.append(1e-9)

        return np.array(x_maxes)

    def validate_spend_csv(
        self,
        df_spend: pd.DataFrame,
        date_column: str = "date",
    ) -> list[ValidationError]:
        """
        Validate the spend CSV format and contents.

        Parameters
        ----------
        df_spend : pd.DataFrame
            Spend data with date and channel columns.
        date_column : str
            Name of the date column.

        Returns
        -------
        list[ValidationError]
            List of validation errors (empty if valid).
        """
        errors = []

        # Check date column exists
        if date_column not in df_spend.columns:
            errors.append(ValidationError(
                field="date",
                message=f"Date column '{date_column}' not found. Available: {list(df_spend.columns)}"
            ))
            return errors  # Can't continue without dates

        # Try to parse dates - try multiple formats to handle DD/MM/YYYY
        try:
            raw_dates = df_spend[date_column]
            sample_raw = str(raw_dates.iloc[0])

            # Try explicit DD/MM/YYYY format first if it looks like that format
            if '/' in sample_raw:
                try:
                    dates = pd.to_datetime(raw_dates, format='%d/%m/%Y')
                    logger.debug(f"Date parsed with DD/MM/YYYY format: '{sample_raw}' -> '{dates.iloc[0]}'")
                except (ValueError, TypeError):
                    # Fall back to dayfirst=True
                    dates = pd.to_datetime(raw_dates, dayfirst=True)
                    logger.debug(f"Date parsed with dayfirst=True: '{sample_raw}' -> '{dates.iloc[0]}'")
            else:
                # No slash - try standard parsing
                dates = pd.to_datetime(raw_dates, dayfirst=True)
                logger.debug(f"Date parsed with dayfirst=True: '{sample_raw}' -> '{dates.iloc[0]}'")
        except Exception as e:
            errors.append(ValidationError(
                field="date",
                message=f"Could not parse dates: {e}. Expected format: DD/MM/YYYY or YYYY-MM-DD"
            ))
            return errors

        # Check for channel columns
        spend_cols = [c for c in df_spend.columns if c != date_column]
        missing_channels = []
        found_channels = []

        for ch in self._channel_columns:
            if ch in spend_cols:
                found_channels.append(ch)
            else:
                missing_channels.append(ch)

        if not found_channels:
            errors.append(ValidationError(
                field="channels",
                message=f"No matching channel columns found. Expected: {self._channel_columns}"
            ))

        # Check for negative spend
        for ch in found_channels:
            if (df_spend[ch] < 0).any():
                errors.append(ValidationError(
                    field=ch,
                    message=f"Negative spend values found in '{ch}'"
                ))

        # Check dates continue from model's data
        model_last_date = pd.to_datetime(
            self.wrapper.df_scaled[self.wrapper.config.data.date_column].max()
        )
        forecast_first_date = dates.min()

        if forecast_first_date <= model_last_date:
            # Show raw value to help debug date parsing issues
            # Use iloc to get first value without triggering comparison
            raw_first_date = str(df_spend[date_column].iloc[0])
            errors.append(ValidationError(
                field="date",
                message=(
                    f"Forecast dates must start after model's last date. "
                    f"Model ends: {model_last_date.strftime('%Y-%m-%d')}, "
                    f"Forecast starts: {forecast_first_date.strftime('%Y-%m-%d')} "
                    f"(raw value: '{raw_first_date}')"
                )
            ))

        return errors

    def forecast(
        self,
        df_spend: pd.DataFrame,
        apply_seasonal: bool = True,
        date_column: str = "date",
        custom_seasonal_indices: dict[str, float] | None = None,
    ) -> ForecastResult:
        """
        Forecast response for weekly spend data.

        Parameters
        ----------
        df_spend : pd.DataFrame
            Spend data with date column and channel columns.
            Channel columns should match model's channel names.
        apply_seasonal : bool
            Whether to apply seasonal effectiveness indices.
        date_column : str
            Name of the date column in df_spend.
        custom_seasonal_indices : dict, optional
            Override seasonal indices. If None, uses computed indices.

        Returns
        -------
        ForecastResult
            Forecast results with response, CIs, and breakdown.

        Raises
        ------
        ValueError
            If the spend CSV is invalid.
        """
        # Validate input
        errors = self.validate_spend_csv(df_spend, date_column)
        if errors:
            error_msgs = [f"{e.field}: {e.message}" for e in errors]
            raise ValueError(f"Invalid spend data:\n" + "\n".join(error_msgs))

        # Parse dates and sort
        df = df_spend.copy()
        df[date_column] = pd.to_datetime(df[date_column])
        df = df.sort_values(date_column).reset_index(drop=True)

        # Determine forecast period
        start_date = df[date_column].min()
        end_date = df[date_column].max()
        start_month = start_date.month
        num_weeks = len(df)

        # Calculate number of months spanned
        num_months = max(1, ((end_date.year - start_date.year) * 12 +
                             end_date.month - start_date.month + 1))

        # Get seasonal indices for the period
        if apply_seasonal:
            if custom_seasonal_indices is not None:
                seasonal_indices = custom_seasonal_indices
            else:
                seasonal_indices = self.seasonal_calculator.get_indices_for_period(
                    start_month=start_month,
                    num_months=num_months
                )
            demand_index = self.seasonal_calculator.get_demand_index_for_period(
                start_month=start_month,
                num_months=num_months
            )
        else:
            seasonal_indices = {ch: 1.0 for ch in self._channel_columns}
            demand_index = 1.0

        # Format period string
        forecast_period = self._format_period_string(start_date, end_date)

        # Compute response for each week with uncertainty
        weekly_results = []
        channel_contribs = []

        for idx, row in df.iterrows():
            week_date = row[date_column]

            # Build spend array for this week
            week_spend = np.array([
                row.get(ch, 0.0) for ch in self._channel_columns
            ])

            # Compute response distribution for this week
            responses, ch_contrib = self._compute_week_response(
                week_spend, seasonal_indices
            )

            # Store results
            weekly_results.append({
                "date": week_date,
                "spend": float(week_spend.sum()),
                "response": float(np.mean(responses)),
                "ci_low": float(np.percentile(responses, 5)),
                "ci_high": float(np.percentile(responses, 95)),
            })

            # Store channel contributions
            for i, ch in enumerate(self._channel_columns):
                channel_contribs.append({
                    "date": week_date,
                    "channel": ch,
                    "contribution": float(ch_contrib[i]),
                })

        # Build DataFrames
        weekly_df = pd.DataFrame(weekly_results)
        channel_df = pd.DataFrame(channel_contribs)

        # Compute total response with proper uncertainty
        total_spend_array = np.array([
            df[ch].sum() if ch in df.columns else 0.0
            for ch in self._channel_columns
        ])

        # For total, we need to account for adstock carryover between weeks
        # For now, sum the weekly responses (conservative - doesn't capture cross-week adstock)
        total_response = weekly_df["response"].sum()
        total_ci_low = weekly_df["ci_low"].sum()
        total_ci_high = weekly_df["ci_high"].sum()

        return ForecastResult(
            total_response=total_response,
            total_ci_low=total_ci_low,
            total_ci_high=total_ci_high,
            weekly_df=weekly_df,
            channel_contributions=channel_df,
            total_spend=float(total_spend_array.sum()),
            num_weeks=num_weeks,
            seasonal_applied=apply_seasonal,
            seasonal_indices=seasonal_indices,
            demand_index=demand_index,
            forecast_period=forecast_period,
        )

    def _compute_week_response(
        self,
        week_spend: np.ndarray,
        seasonal_indices: dict[str, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute response distribution for a single week's spend.

        Uses posterior sampling for uncertainty quantification.
        Applies seasonal indices the same way as the optimizer.

        Parameters
        ----------
        week_spend : np.ndarray
            Spend per channel for this week, shape (n_channels,).
        seasonal_indices : dict
            Channel effectiveness indices.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (response_samples, mean_channel_contributions)
            - response_samples: shape (n_samples,)
            - mean_channel_contributions: shape (n_channels,)
        """
        # Build seasonal index array
        seasonal_array = np.array([
            seasonal_indices.get(ch, 1.0) for ch in self._channel_columns
        ])

        # Normalize spend by x_max
        x_normalized = week_spend / self._x_maxes

        # Broadcast for all posterior samples: (n_samples, n_channels)
        x_norm_broadcast = x_normalized[np.newaxis, :]

        # Compute saturation for all samples
        # saturation(x) = (1 - exp(-lam*x)) / (1 + exp(-lam*x))
        exp_term = np.exp(-self.posterior_samples.lam_samples * x_norm_broadcast)
        saturation = (1 - exp_term) / (1 + exp_term)

        # Apply seasonal indices (same as risk_objectives.py line 189-191)
        seasonal_adjusted = saturation * seasonal_array[np.newaxis, :]

        # Get target scale (same formula as allocator.py line 566)
        # target_scale = max(target) × revenue_scale
        target_col = self.wrapper.config.data.target_column
        revenue_scale = self.wrapper.config.data.revenue_scale
        target_scale = float(self.wrapper.df_scaled[target_col].max()) * revenue_scale

        # Compute response per sample
        # response = sum(beta * target_scale * adjusted_saturation)
        channel_response = (
            self.posterior_samples.beta_samples * target_scale * seasonal_adjusted
        )
        response_per_sample = np.sum(channel_response, axis=1)

        # Mean channel contributions (for breakdown)
        mean_channel_contrib = np.mean(channel_response, axis=0)

        return response_per_sample, mean_channel_contrib

    def _format_period_string(
        self,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
    ) -> str:
        """Format a human-readable period string."""
        month_names = [
            "Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
        ]

        start_month = month_names[start_date.month - 1]
        end_month = month_names[end_date.month - 1]

        if start_date.year == end_date.year:
            if start_date.month == end_date.month:
                return f"{start_month} {start_date.year}"
            else:
                return f"{start_month}-{end_month} {start_date.year}"
        else:
            return f"{start_month} {start_date.year} - {end_month} {end_date.year}"

    def get_seasonality_preview(
        self,
        df_spend: pd.DataFrame,
        date_column: str = "date",
    ) -> dict:
        """
        Get seasonality information for the forecast period.

        Call this after CSV upload but before running forecast
        to show users what indices will be applied.

        Parameters
        ----------
        df_spend : pd.DataFrame
            Spend data with date column.
        date_column : str
            Name of the date column.

        Returns
        -------
        dict
            Preview of seasonal indices and period info.
        """
        df = df_spend.copy()
        df[date_column] = pd.to_datetime(df[date_column])

        start_date = df[date_column].min()
        end_date = df[date_column].max()
        start_month = start_date.month
        num_months = max(1, ((end_date.year - start_date.year) * 12 +
                             end_date.month - start_date.month + 1))

        # Get indices
        channel_indices = self.seasonal_calculator.get_indices_for_period(
            start_month=start_month,
            num_months=num_months
        )
        demand_index = self.seasonal_calculator.get_demand_index_for_period(
            start_month=start_month,
            num_months=num_months
        )

        # Get confidence info
        confidence = self.seasonal_calculator.get_confidence_info(
            start_month=start_month,
            num_months=num_months
        )

        return {
            "period": self._format_period_string(start_date, end_date),
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "num_weeks": len(df),
            "channel_indices": channel_indices,
            "demand_index": demand_index,
            "confidence": confidence,
        }

    def get_channel_display_names(self) -> dict[str, str]:
        """Get mapping from channel column names to display names."""
        return self.bridge.channel_display_names


# =============================================================================
# Granular Data Functions
# =============================================================================

def detect_spend_format(
    df: pd.DataFrame,
    model_channels: list[str],
) -> str:
    """
    Detect if uploaded spend data is granular or aggregated format.

    Parameters
    ----------
    df : pd.DataFrame
        Uploaded spend data.
    model_channels : list[str]
        List of channel column names expected by the model.

    Returns
    -------
    str
        One of: "granular", "aggregated", "unknown"
    """
    # Check for level columns (granular format)
    level_cols = [c for c in df.columns if 'lvl' in c.lower() or 'level' in c.lower()]

    # Check for a "spend" column (indicator of granular format)
    has_spend_col = any(c.lower() == 'spend' for c in df.columns)

    # Granular format: has level columns OR has "spend" column without model channels
    if level_cols:
        return "granular"
    elif has_spend_col and not any(ch in df.columns for ch in model_channels):
        # Has "spend" column but no model channel columns - likely granular
        return "granular"
    elif all(ch in df.columns for ch in model_channels):
        return "aggregated"
    else:
        # Partial match - could be either format with missing columns
        return "unknown"


def get_level_columns(df: pd.DataFrame) -> list[str]:
    """
    Extract level columns from a granular DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Granular spend data.

    Returns
    -------
    list[str]
        Level column names in order.
    """
    # First try: columns with "lvl" or "level" in the name
    level_cols = [c for c in df.columns if 'lvl' in c.lower() or 'level' in c.lower()]

    if level_cols:
        # Try to sort by suffix number (lvl1, lvl2, lvl3)
        def sort_key(col):
            import re
            match = re.search(r'(\d+)$', col.lower())
            return int(match.group(1)) if match else 0
        return sorted(level_cols, key=sort_key)

    # Second try: if there's a "spend" column, treat non-date/non-numeric columns as levels
    has_spend_col = any(c.lower() == 'spend' for c in df.columns)
    if has_spend_col:
        # Exclude date-like columns and the spend column itself
        date_keywords = ['date', 'time', 'week', 'month', 'year', 'day']
        level_cols = []
        for col in df.columns:
            col_lower = col.lower()
            # Skip spend column
            if col_lower == 'spend':
                continue
            # Skip date-like columns
            if any(kw in col_lower for kw in date_keywords):
                continue
            # Skip numeric columns (these are likely spend values)
            if df[col].dtype in ['int64', 'float64']:
                # Check if it's actually categorical by looking at unique values
                if df[col].nunique() > 50:  # Likely a numeric spend column
                    continue
            level_cols.append(col)
        return level_cols

    return level_cols


def validate_against_saved_mapping(
    df_granular: pd.DataFrame,
    level_columns: list[str],
    entity_mappings: dict[str, str],
    skipped_entities: list[str] | None = None,
) -> dict:
    """
    Validate granular data against a saved mapping.

    Compares the entities in the uploaded file against the saved mapping
    to identify matches, new entities, and missing entities.

    Parameters
    ----------
    df_granular : pd.DataFrame
        Granular spend data with level columns.
    level_columns : list[str]
        Column names forming the entity key.
    entity_mappings : dict[str, str]
        Saved mapping from entity keys to model variable names.
    skipped_entities : list[str], optional
        Entity keys that user explicitly chose to skip (not map to any channel).
        These won't be flagged as "new" entities.

    Returns
    -------
    dict
        Validation result with:
        - is_valid: bool (True if no new entities)
        - matched: list of matched entity keys
        - skipped: list of skipped entity keys present in file
        - new_entities: list of entity keys not in saved mapping or skipped
        - missing: list of saved entities not in file
        - warnings: list of warning messages
    """
    # Get entities from file
    file_entities = set()
    for _, row in df_granular[level_columns].drop_duplicates().iterrows():
        entity_key = "|".join(str(row[c]) for c in level_columns)
        file_entities.add(entity_key)

    # Get entities from saved mapping and skipped list
    saved_entities = set(entity_mappings.keys())
    skipped_set = set(skipped_entities) if skipped_entities else set()

    # Entities that match the saved mapping (will be aggregated)
    matched = file_entities & saved_entities
    # Entities that were explicitly skipped (will be excluded from aggregation)
    skipped_in_file = file_entities & skipped_set
    # Truly new entities (not mapped and not skipped)
    new_entities = file_entities - saved_entities - skipped_set
    # Saved entities not in the file
    missing = saved_entities - file_entities

    warnings = []
    if new_entities:
        warnings.append(f"{len(new_entities)} new entities need mapping")
    if missing:
        warnings.append(f"{len(missing)} saved entities not in file")

    return {
        "is_valid": len(new_entities) == 0,
        "matched": sorted(list(matched)),
        "skipped": sorted(list(skipped_in_file)),
        "new_entities": sorted(list(new_entities)),
        "missing": sorted(list(missing)),
        "warnings": [w for w in warnings if w],
    }


def aggregate_granular_spend(
    df_granular: pd.DataFrame,
    level_columns: list[str],
    entity_mappings: dict[str, str],
    date_column: str = "date",
    spend_column: str = "spend",
) -> pd.DataFrame:
    """
    Aggregate granular spend data to model variable format.

    Converts granular spend data (with level columns) to the wide format
    expected by the forecast engine (date + one column per model variable).

    Parameters
    ----------
    df_granular : pd.DataFrame
        Granular spend data with level columns.
    level_columns : list[str]
        Column names forming the entity key.
    entity_mappings : dict[str, str]
        Mapping from entity keys (pipe-separated) to model variable names.
    date_column : str
        Name of date column in input.
    spend_column : str
        Name of spend column in input.

    Returns
    -------
    pd.DataFrame
        Aggregated data with date column and one column per model variable.

    Raises
    ------
    ValueError
        If required columns are missing.
    """
    # Validate required columns exist
    missing_cols = []
    if date_column not in df_granular.columns:
        missing_cols.append(date_column)
    if spend_column not in df_granular.columns:
        missing_cols.append(spend_column)
    for col in level_columns:
        if col not in df_granular.columns:
            missing_cols.append(col)

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df = df_granular.copy()

    # Create entity key column
    df['_entity'] = df[level_columns].apply(
        lambda row: "|".join(str(v) for v in row), axis=1
    )

    # Map to model variable
    df['_variable'] = df['_entity'].map(entity_mappings)

    # Log unmapped rows
    unmapped_mask = df['_variable'].isna()
    if unmapped_mask.any():
        unmapped_count = unmapped_mask.sum()
        unmapped_entities = df.loc[unmapped_mask, '_entity'].unique()[:5]  # First 5
        logger.warning(
            f"{unmapped_count} rows with unmapped entities (e.g., {list(unmapped_entities)})"
        )
        # Filter out unmapped rows
        df = df[~unmapped_mask]

    if df.empty:
        raise ValueError("No data remaining after filtering unmapped entities")

    # Aggregate: group by date + variable, sum spend
    aggregated = df.groupby([date_column, '_variable'])[spend_column].sum()

    # Pivot to wide format
    result = aggregated.unstack(fill_value=0).reset_index()
    result.columns.name = None
    result = result.rename(columns={date_column: 'date'})

    # Ensure date is datetime
    result['date'] = pd.to_datetime(result['date'])

    return result


# =============================================================================
# Overlap Detection
# =============================================================================

@dataclass
class OverlapAnalysis:
    """Result of checking for date overlap with historical forecasts."""
    has_overlap: bool
    overlapping_dates: list[str]           # Dates that exist in both
    new_dates: list[str]                   # Dates only in new upload

    # Per-date comparison for overlapping dates
    spend_comparison: pd.DataFrame         # date, channel, old_spend, new_spend, diff, pct_change

    # Summary stats
    total_spend_old: float
    total_spend_new: float
    spend_difference: float
    pct_change: float

    # Which forecasts overlap
    overlapping_forecast_ids: list[str]


def check_forecast_overlap(
    new_spend: pd.DataFrame,
    historical_spend: pd.DataFrame,
    date_column: str = "date",
) -> OverlapAnalysis:
    """
    Compare new spend data against historical forecasts.

    Detects overlapping dates and calculates spend differences for each
    date and channel.

    Parameters
    ----------
    new_spend : pd.DataFrame
        New spend data being uploaded (wide format with date + channel columns).
    historical_spend : pd.DataFrame
        Combined historical spend data (from ForecastPersistence.get_historical_spend()).
        Should have _forecast_id column identifying which forecast each row came from.
    date_column : str
        Name of the date column.

    Returns
    -------
    OverlapAnalysis
        Analysis of overlap between new and historical data.
    """
    # Handle empty historical data
    if historical_spend.empty:
        new_dates = new_spend[date_column].astype(str).tolist()
        return OverlapAnalysis(
            has_overlap=False,
            overlapping_dates=[],
            new_dates=new_dates,
            spend_comparison=pd.DataFrame(),
            total_spend_old=0.0,
            total_spend_new=float(new_spend.drop(columns=[date_column], errors='ignore').sum().sum()),
            spend_difference=0.0,
            pct_change=0.0,
            overlapping_forecast_ids=[],
        )

    # Normalize dates
    new_spend = new_spend.copy()
    historical_spend = historical_spend.copy()
    new_spend[date_column] = pd.to_datetime(new_spend[date_column])
    historical_spend[date_column] = pd.to_datetime(historical_spend[date_column])

    # Find overlapping dates
    new_dates_set = set(new_spend[date_column].dt.strftime("%Y-%m-%d"))
    historical_dates_set = set(historical_spend[date_column].dt.strftime("%Y-%m-%d"))

    overlapping_dates = sorted(new_dates_set & historical_dates_set)
    new_only_dates = sorted(new_dates_set - historical_dates_set)

    if not overlapping_dates:
        return OverlapAnalysis(
            has_overlap=False,
            overlapping_dates=[],
            new_dates=list(new_dates_set),
            spend_comparison=pd.DataFrame(),
            total_spend_old=0.0,
            total_spend_new=float(new_spend.drop(columns=[date_column], errors='ignore').sum().sum()),
            spend_difference=0.0,
            pct_change=0.0,
            overlapping_forecast_ids=[],
        )

    # Get channel columns (exclude date and metadata columns)
    channel_cols = [c for c in new_spend.columns if c != date_column]
    historical_channel_cols = [c for c in historical_spend.columns
                              if c not in [date_column, '_forecast_id']]

    # Melt new spend to long format
    new_long = new_spend.melt(
        id_vars=[date_column],
        value_vars=channel_cols,
        var_name="channel",
        value_name="new_spend"
    )
    new_long["date_str"] = new_long[date_column].dt.strftime("%Y-%m-%d")

    # Melt historical spend to long format
    hist_long = historical_spend.melt(
        id_vars=[date_column, '_forecast_id'],
        value_vars=historical_channel_cols,
        var_name="channel",
        value_name="old_spend"
    )
    hist_long["date_str"] = hist_long[date_column].dt.strftime("%Y-%m-%d")

    # For historical, take the most recent forecast per date/channel
    # (latest _forecast_id wins if multiple forecasts cover same date)
    hist_latest = hist_long.sort_values('_forecast_id').groupby(
        ['date_str', 'channel']
    ).last().reset_index()

    # Filter to overlapping dates only
    new_overlap = new_long[new_long["date_str"].isin(overlapping_dates)]
    hist_overlap = hist_latest[hist_latest["date_str"].isin(overlapping_dates)]

    # Merge for comparison
    comparison = pd.merge(
        new_overlap[["date_str", "channel", "new_spend"]],
        hist_overlap[["date_str", "channel", "old_spend", "_forecast_id"]],
        on=["date_str", "channel"],
        how="outer"
    ).fillna(0)

    comparison["diff"] = comparison["new_spend"] - comparison["old_spend"]
    comparison["pct_change"] = comparison.apply(
        lambda row: (row["diff"] / row["old_spend"] * 100) if row["old_spend"] != 0 else 0.0,
        axis=1
    )

    # Summary stats
    total_old = comparison["old_spend"].sum()
    total_new = comparison["new_spend"].sum()
    total_diff = total_new - total_old
    total_pct = (total_diff / total_old * 100) if total_old != 0 else 0.0

    # Which forecasts overlap
    overlapping_ids = hist_overlap["_forecast_id"].unique().tolist()

    # Rename for output
    comparison = comparison.rename(columns={"date_str": "date"})

    return OverlapAnalysis(
        has_overlap=True,
        overlapping_dates=overlapping_dates,
        new_dates=new_only_dates,
        spend_comparison=comparison,
        total_spend_old=float(total_old),
        total_spend_new=float(total_new),
        spend_difference=float(total_diff),
        pct_change=float(total_pct),
        overlapping_forecast_ids=overlapping_ids,
    )
