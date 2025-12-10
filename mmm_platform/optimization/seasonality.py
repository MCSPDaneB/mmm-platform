"""
Seasonal effectiveness index calculator for budget optimization.

This module computes seasonal indices that measure how channel effectiveness
varies by time period, allowing the optimizer to account for seasonal patterns.
"""

from typing import Any
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class SeasonalIndexCalculator:
    """
    Compute seasonal effectiveness indices from MMM results.

    The effectiveness index measures how much incremental response a channel
    generates per dollar of spend, relative to its average. An index > 1.0
    means the channel is more effective than average during that period.

    Formula:
        Effectiveness[channel, month] = Contribution[channel, month] / Spend[channel, month]
        Index[channel, month] = Effectiveness[channel, month] / Avg_Effectiveness[channel]

    Parameters
    ----------
    wrapper : MMMWrapper
        A fitted MMMWrapper instance with contributions available.

    Examples
    --------
    >>> calculator = SeasonalIndexCalculator(wrapper)
    >>> monthly_indices = calculator.compute_monthly_indices()
    >>> jan_indices = calculator.get_indices_for_period(start_month=1, num_months=1)
    >>> print(jan_indices)
    {'search': 0.95, 'social': 1.12, 'display': 0.88}
    """

    def __init__(self, wrapper: Any):
        """
        Initialize the seasonal index calculator.

        Parameters
        ----------
        wrapper : MMMWrapper
            A fitted MMMWrapper instance.
        """
        if wrapper.idata is None:
            raise ValueError(
                "MMMWrapper is not fitted. Call wrapper.fit() before computing "
                "seasonal indices."
            )

        self.wrapper = wrapper
        self._monthly_indices: pd.DataFrame | None = None
        self._quarterly_indices: pd.DataFrame | None = None
        self._observations_per_month: pd.DataFrame | None = None

    @property
    def channels(self) -> list[str]:
        """Get the list of channels to compute indices for."""
        return self.wrapper.transform_engine.get_effective_channel_columns()

    @property
    def date_column(self) -> str:
        """Get the date column name."""
        return self.wrapper.config.data.date_column

    @property
    def spend_scale(self) -> float:
        """Get the spend scaling factor."""
        return self.wrapper.config.data.spend_scale

    @property
    def revenue_scale(self) -> float:
        """Get the revenue scaling factor."""
        return self.wrapper.config.data.revenue_scale

    def compute_monthly_indices(self, min_observations: int = 2) -> pd.DataFrame:
        """
        Compute monthly effectiveness indices for each channel.

        Returns a DataFrame where each row is a channel and each column is a month (1-12).
        Values are effectiveness indices relative to each channel's average.

        Parameters
        ----------
        min_observations : int
            Minimum observations required per month to compute index.
            Months with fewer observations will have index = 1.0 (average).

        Returns
        -------
        pd.DataFrame
            DataFrame with shape (n_channels, 12) containing effectiveness indices.
            Index = channel names, columns = month numbers (1-12).
            Values: index > 1 = more effective, index < 1 = less effective.
        """
        if self._monthly_indices is not None:
            return self._monthly_indices

        # Get contributions and spend data
        try:
            contribs = self.wrapper.get_contributions()
        except Exception as e:
            raise RuntimeError(
                f"Could not retrieve model contributions: {type(e).__name__}: {e}"
            ) from e

        if contribs is None or contribs.empty:
            raise ValueError("Model contributions are empty. Ensure model is fitted.")

        df = self._get_data_with_dates()

        # Validate data alignment
        if len(contribs) != len(df):
            raise ValueError(
                f"Contributions length ({len(contribs)}) != data length ({len(df)}). "
                "Data may be misaligned."
            )

        # Ensure date column is datetime
        df[self.date_column] = pd.to_datetime(df[self.date_column])
        contribs_with_date = contribs.copy()
        contribs_with_date[self.date_column] = df[self.date_column].values

        # Extract month
        df["_month"] = df[self.date_column].dt.month
        contribs_with_date["_month"] = df["_month"].values

        # Compute effectiveness per channel per month
        monthly_data = []
        observations_data = []

        for month in range(1, 13):
            month_mask = df["_month"] == month
            n_obs = month_mask.sum()
            observations_data.append({"month": month, "n_observations": n_obs})

            month_row = {"month": month}

            for ch in self.channels:
                if ch not in contribs.columns or ch not in df.columns:
                    month_row[ch] = 1.0
                    continue

                # Get contribution and spend for this month
                # Use .values to avoid pandas index alignment issues
                month_contrib = contribs_with_date.loc[month_mask.values, ch].sum()
                month_spend = df.loc[month_mask.values, ch].sum() * self.spend_scale

                if n_obs >= min_observations and month_spend > 0:
                    # Effectiveness = contribution / spend (scaled to real units)
                    effectiveness = (month_contrib * self.revenue_scale) / month_spend
                    month_row[ch] = effectiveness
                else:
                    # Not enough data - will be filled with average later
                    month_row[ch] = np.nan

            monthly_data.append(month_row)

        # Create effectiveness DataFrame
        effectiveness_df = pd.DataFrame(monthly_data).set_index("month")

        # Store observations count
        self._observations_per_month = pd.DataFrame(observations_data).set_index("month")

        # Compute indices: normalize each channel so average = 1.0
        indices_df = effectiveness_df.copy()
        for ch in self.channels:
            if ch not in indices_df.columns:
                continue

            avg_effectiveness = effectiveness_df[ch].mean()
            if avg_effectiveness > 0 and not np.isnan(avg_effectiveness):
                indices_df[ch] = effectiveness_df[ch] / avg_effectiveness
            else:
                indices_df[ch] = 1.0

            # Fill NaN with 1.0 (average)
            indices_df[ch] = indices_df[ch].fillna(1.0)

        # Transpose: channels as rows, months as columns
        self._monthly_indices = indices_df.T
        self._monthly_indices.index.name = "channel"

        logger.info(
            f"Computed monthly indices for {len(self.channels)} channels "
            f"from {len(df)} observations"
        )

        return self._monthly_indices

    def compute_quarterly_indices(self, min_observations: int = 4) -> pd.DataFrame:
        """
        Compute quarterly effectiveness indices for each channel.

        Use this when monthly data is sparse (<2 years of data).

        Parameters
        ----------
        min_observations : int
            Minimum observations required per quarter to compute index.

        Returns
        -------
        pd.DataFrame
            DataFrame with shape (n_channels, 4) containing quarterly indices.
            Index = channel names, columns = quarter numbers (1-4).
        """
        if self._quarterly_indices is not None:
            return self._quarterly_indices

        # Get contributions and spend data
        contribs = self.wrapper.get_contributions()
        df = self._get_data_with_dates()

        # Ensure date column is datetime
        df[self.date_column] = pd.to_datetime(df[self.date_column])
        contribs_with_date = contribs.copy()
        contribs_with_date[self.date_column] = df[self.date_column].values

        # Extract quarter
        df["_quarter"] = df[self.date_column].dt.quarter
        contribs_with_date["_quarter"] = df["_quarter"].values

        # Compute effectiveness per channel per quarter
        quarterly_data = []

        for quarter in range(1, 5):
            quarter_mask = df["_quarter"] == quarter
            n_obs = quarter_mask.sum()

            quarter_row = {"quarter": quarter}

            for ch in self.channels:
                if ch not in contribs.columns or ch not in df.columns:
                    quarter_row[ch] = 1.0
                    continue

                # Get contribution and spend for this quarter
                # Use .values to avoid pandas index alignment issues
                quarter_contrib = contribs_with_date.loc[quarter_mask.values, ch].sum()
                quarter_spend = df.loc[quarter_mask.values, ch].sum() * self.spend_scale

                if n_obs >= min_observations and quarter_spend > 0:
                    effectiveness = (quarter_contrib * self.revenue_scale) / quarter_spend
                    quarter_row[ch] = effectiveness
                else:
                    quarter_row[ch] = np.nan

            quarterly_data.append(quarter_row)

        # Create effectiveness DataFrame
        effectiveness_df = pd.DataFrame(quarterly_data).set_index("quarter")

        # Normalize to indices
        indices_df = effectiveness_df.copy()
        for ch in self.channels:
            if ch not in indices_df.columns:
                continue

            avg_effectiveness = effectiveness_df[ch].mean()
            if avg_effectiveness > 0 and not np.isnan(avg_effectiveness):
                indices_df[ch] = effectiveness_df[ch] / avg_effectiveness
            else:
                indices_df[ch] = 1.0

            indices_df[ch] = indices_df[ch].fillna(1.0)

        # Transpose: channels as rows, quarters as columns
        self._quarterly_indices = indices_df.T
        self._quarterly_indices.index.name = "channel"

        logger.info(
            f"Computed quarterly indices for {len(self.channels)} channels"
        )

        return self._quarterly_indices

    def get_indices_for_period(
        self,
        start_month: int,
        num_months: int = 1,
        use_quarterly: bool | None = None,
    ) -> dict[str, float]:
        """
        Get average seasonal index per channel for a specific date range.

        Parameters
        ----------
        start_month : int
            Starting month (1-12).
        num_months : int
            Number of months in the optimization period.
        use_quarterly : bool, optional
            If True, use quarterly indices. If None, auto-detect based on
            data availability.

        Returns
        -------
        dict[str, float]
            {channel_name: average_index} for the specified period.
            Index > 1 = more effective than average,
            Index < 1 = less effective than average.
        """
        # Auto-detect whether to use quarterly indices
        if use_quarterly is None:
            use_quarterly = self._should_use_quarterly()

        if use_quarterly:
            return self._get_quarterly_indices_for_period(start_month, num_months)
        else:
            return self._get_monthly_indices_for_period(start_month, num_months)

    def _get_monthly_indices_for_period(
        self,
        start_month: int,
        num_months: int,
    ) -> dict[str, float]:
        """Get monthly indices averaged over a period."""
        indices_df = self.compute_monthly_indices()

        # Calculate which months are covered
        months = [(start_month + i - 1) % 12 + 1 for i in range(num_months)]

        result = {}
        for ch in self.channels:
            if ch in indices_df.index:
                # Average the indices for the covered months
                month_indices = [indices_df.loc[ch, m] for m in months]
                result[ch] = float(np.mean(month_indices))
            else:
                result[ch] = 1.0

        return result

    def _get_quarterly_indices_for_period(
        self,
        start_month: int,
        num_months: int,
    ) -> dict[str, float]:
        """Get quarterly indices averaged over a period."""
        indices_df = self.compute_quarterly_indices()

        # Map months to quarters and count coverage
        quarter_weights = {1: 0, 2: 0, 3: 0, 4: 0}
        for i in range(num_months):
            month = (start_month + i - 1) % 12 + 1
            quarter = (month - 1) // 3 + 1
            quarter_weights[quarter] += 1

        total_months = sum(quarter_weights.values())

        result = {}
        for ch in self.channels:
            if ch in indices_df.index:
                # Weighted average by months spent in each quarter
                weighted_sum = 0.0
                for q, weight in quarter_weights.items():
                    if weight > 0:
                        weighted_sum += indices_df.loc[ch, q] * weight
                result[ch] = float(weighted_sum / total_months) if total_months > 0 else 1.0
            else:
                result[ch] = 1.0

        return result

    def _should_use_quarterly(self) -> bool:
        """
        Determine whether to use quarterly indices based on data availability.

        Uses quarterly if fewer than 2 observations per month on average.
        """
        df = self._get_data_with_dates()
        n_periods = len(df)

        # Estimate months of data
        df[self.date_column] = pd.to_datetime(df[self.date_column])
        date_range = (df[self.date_column].max() - df[self.date_column].min()).days
        months_of_data = max(date_range / 30, 1)

        avg_obs_per_month = n_periods / months_of_data

        use_quarterly = avg_obs_per_month < 2
        if use_quarterly:
            logger.info(
                f"Using quarterly indices due to limited data "
                f"({avg_obs_per_month:.1f} observations per month)"
            )

        return use_quarterly

    def _get_data_with_dates(self) -> pd.DataFrame:
        """Get the data DataFrame with dates."""
        # Prefer df_original (unscaled), fall back to df_scaled
        df = getattr(self.wrapper, 'df_original', None)
        if df is None:
            df = getattr(self.wrapper, 'df_scaled', None)

        if df is None:
            raise ValueError(
                "No data available. Wrapper must have df_original or df_scaled attribute."
            )

        # Validate date column exists
        if self.date_column not in df.columns:
            raise ValueError(
                f"Date column '{self.date_column}' not found in data. "
                f"Available columns: {list(df.columns)[:10]}..."
            )

        return df.copy()

    def get_observations_per_month(self) -> pd.DataFrame:
        """
        Get the number of observations per month in the data.

        Useful for understanding data coverage and confidence in indices.

        Returns
        -------
        pd.DataFrame
            DataFrame with month and n_observations columns.
        """
        if self._observations_per_month is None:
            self.compute_monthly_indices()
        return self._observations_per_month

    def get_confidence_info(self, start_month: int, num_months: int = 1) -> dict:
        """
        Get confidence information about the seasonal indices for a period.

        Parameters
        ----------
        start_month : int
            Starting month (1-12).
        num_months : int
            Number of months in the period.

        Returns
        -------
        dict
            {
                "min_observations": int,
                "max_observations": int,
                "avg_observations": float,
                "using_quarterly": bool,
                "confidence_level": str  # "high", "medium", "low"
            }
        """
        obs_df = self.get_observations_per_month()

        # Get months covered
        months = [(start_month + i - 1) % 12 + 1 for i in range(num_months)]
        obs_counts = [obs_df.loc[m, "n_observations"] for m in months]

        min_obs = min(obs_counts)
        max_obs = max(obs_counts)
        avg_obs = np.mean(obs_counts)

        using_quarterly = self._should_use_quarterly()

        # Determine confidence level
        if avg_obs >= 8:  # 2+ years of data per month
            confidence = "high"
        elif avg_obs >= 4:  # 1+ year of data
            confidence = "medium"
        else:
            confidence = "low"

        return {
            "min_observations": int(min_obs),
            "max_observations": int(max_obs),
            "avg_observations": float(avg_obs),
            "using_quarterly": using_quarterly,
            "confidence_level": confidence,
        }

    def to_dataframe(self, use_quarterly: bool = False) -> pd.DataFrame:
        """
        Get the full indices table as a formatted DataFrame.

        Parameters
        ----------
        use_quarterly : bool
            If True, return quarterly indices. Otherwise monthly.

        Returns
        -------
        pd.DataFrame
            Indices table with channels as rows and periods as columns.
        """
        if use_quarterly:
            df = self.compute_quarterly_indices().copy()
            df.columns = ["Q1", "Q2", "Q3", "Q4"]
        else:
            df = self.compute_monthly_indices().copy()
            df.columns = [
                "Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
            ]

        # Add display names
        display_names = {}
        for ch_config in self.wrapper.config.channels:
            display_names[ch_config.name] = ch_config.get_display_name()
        for om_config in self.wrapper.config.owned_media:
            display_names[om_config.name] = om_config.get_display_name()

        df["Display Name"] = df.index.map(lambda x: display_names.get(x, x))
        cols = ["Display Name"] + [c for c in df.columns if c != "Display Name"]
        df = df[cols]

        return df

    def get_summary_dict(self) -> dict:
        """
        Get a JSON-serializable summary of seasonal indices.

        Returns
        -------
        dict
            Summary with monthly and quarterly indices.
        """
        monthly = self.compute_monthly_indices()
        quarterly = self.compute_quarterly_indices()

        return {
            "monthly_indices": monthly.to_dict(),
            "quarterly_indices": quarterly.to_dict(),
            "using_quarterly": self._should_use_quarterly(),
            "n_channels": len(self.channels),
            "channels": self.channels,
        }

    def compute_demand_indices(self) -> pd.DataFrame:
        """
        Compute demand seasonality indices from the target KPI.

        This captures how baseline demand varies by month, independent of
        channel effectiveness. An index > 1.0 means higher demand than average.

        Formula:
            Demand_Index[month] = Avg_KPI[month] / Overall_Avg_KPI

        Returns
        -------
        pd.DataFrame
            DataFrame with month as index and 'demand_index' column.
            Index > 1 = higher demand than average,
            Index < 1 = lower demand than average.

        Examples
        --------
        >>> calculator = SeasonalIndexCalculator(wrapper)
        >>> demand = calculator.compute_demand_indices()
        >>> print(demand.loc[12, 'demand_index'])  # December demand index
        1.35  # 35% higher demand than average
        """
        df = self._get_data_with_dates()
        target_col = self.wrapper.config.data.target_column

        if target_col not in df.columns:
            raise ValueError(
                f"Target column '{target_col}' not found in data. "
                f"Available columns: {list(df.columns)[:10]}..."
            )

        # Ensure date column is datetime
        df[self.date_column] = pd.to_datetime(df[self.date_column])
        df["_month"] = df[self.date_column].dt.month

        # Calculate average KPI per month
        monthly_avg = df.groupby("_month")[target_col].mean()

        # Calculate overall average
        overall_avg = df[target_col].mean()

        if overall_avg == 0 or np.isnan(overall_avg):
            logger.warning("Overall average KPI is zero or NaN, returning index of 1.0 for all months")
            demand_indices = pd.Series([1.0] * 12, index=range(1, 13))
        else:
            # Compute indices (normalized so average = 1.0)
            demand_indices = monthly_avg / overall_avg

        # Ensure all 12 months are present (fill missing with 1.0)
        full_index = pd.Series(index=range(1, 13), dtype=float)
        for month in range(1, 13):
            if month in demand_indices.index:
                full_index[month] = demand_indices[month]
            else:
                full_index[month] = 1.0

        result = full_index.to_frame(name="demand_index")
        result.index.name = "month"

        logger.info(
            f"Computed demand indices from {len(df)} observations. "
            f"Range: {result['demand_index'].min():.2f} - {result['demand_index'].max():.2f}"
        )

        return result

    def get_demand_index_for_period(
        self,
        start_month: int,
        num_months: int = 1,
    ) -> float:
        """
        Get average demand index for a specific period.

        Parameters
        ----------
        start_month : int
            Starting month (1-12).
        num_months : int
            Number of months in the optimization period.

        Returns
        -------
        float
            Average demand index for the period.
            > 1.0 = higher demand than average,
            < 1.0 = lower demand than average.

        Examples
        --------
        >>> calculator = SeasonalIndexCalculator(wrapper)
        >>> # Get demand index for Q4 (Oct-Dec)
        >>> q4_demand = calculator.get_demand_index_for_period(10, 3)
        >>> print(f"Q4 demand is {q4_demand:.0%} of average")
        Q4 demand is 125% of average
        """
        demand_df = self.compute_demand_indices()

        # Calculate which months are covered (handles year wrap-around)
        months = [(start_month + i - 1) % 12 + 1 for i in range(num_months)]
        period_indices = [demand_df.loc[m, "demand_index"] for m in months]

        return float(np.mean(period_indices))
