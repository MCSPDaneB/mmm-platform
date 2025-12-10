"""
Backtesting validation for MMM optimizer.

Validates that the model's response curves accurately predict
actual outcomes by comparing historical spend to predicted response.
"""

from typing import Any
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class BacktestValidator:
    """
    Validate model response curves using historical data.

    Runs historical spend through the model's saturation curves
    and compares predicted response to actual outcomes.

    Parameters
    ----------
    wrapper : MMMWrapper
        A fitted MMMWrapper instance with posterior samples.

    Examples
    --------
    >>> validator = BacktestValidator(wrapper)
    >>> backtest_df = validator.run_backtest(n_periods=52)
    >>> metrics = validator.get_metrics(backtest_df)
    >>> print(f"R²: {metrics['r2']:.3f}, MAPE: {metrics['mape']:.1f}%")
    """

    def __init__(self, wrapper: Any):
        """Initialize the backtest validator."""
        self.wrapper = wrapper
        self.config = wrapper.config
        self._extract_posterior_params()

    def _extract_posterior_params(self):
        """Extract posterior mean parameters for prediction."""
        if self.wrapper.idata is None:
            raise ValueError("Wrapper must have fitted model (idata) for backtesting")

        posterior = self.wrapper.idata.posterior

        # Get saturation parameters
        self.beta = posterior['saturation_beta'].mean(dim=['chain', 'draw']).values
        self.lam = posterior['saturation_lam'].mean(dim=['chain', 'draw']).values

        # Get channel list
        self.channels = list(self.wrapper.mmm.channel_columns)

        # Get scaling factors (same as allocator._optimize_with_working_gradients)
        df = self.wrapper.df_scaled
        target_col = self.config.data.target_column
        self.target_scale = float(df[target_col].max())

        # Get x_max for each channel (in scaled units)
        self.x_maxes = np.array([
            float(df[ch].max()) for ch in self.channels
        ])
        # Avoid division by zero
        self.x_maxes = np.maximum(self.x_maxes, 1e-9)

        logger.info(
            f"BacktestValidator initialized: {len(self.channels)} channels, "
            f"target_scale={self.target_scale:.0f}"
        )

    def predict_response_at_spend(self, spend_per_channel: np.ndarray) -> float:
        """
        Predict response using saturation curves.

        Uses the exact same formula as allocator._optimize_with_working_gradients().

        Parameters
        ----------
        spend_per_channel : np.ndarray
            Array of spend values per channel (scaled units, per-period).

        Returns
        -------
        float
            Predicted response for this period.
        """
        x_normalized = spend_per_channel / self.x_maxes
        exp_term = np.exp(-self.lam * x_normalized)
        saturation = (1 - exp_term) / (1 + exp_term)
        response = np.sum(self.beta * self.target_scale * saturation)
        return float(response)

    def run_backtest(self, n_periods: int | None = None) -> pd.DataFrame:
        """
        Run backtest comparing predicted vs actual response.

        Parameters
        ----------
        n_periods : int, optional
            Number of periods to backtest. If None, uses all available data.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns:
            - date: Period date
            - actual: Actual response value
            - predicted: Predicted response from saturation curves
            - residual: actual - predicted
            - pct_error: residual / actual * 100
            - channel spend columns
        """
        df = self.wrapper.df_scaled
        target_col = self.config.data.target_column
        date_col = self.config.data.date_column

        # Use all data if n_periods not specified
        if n_periods is None:
            n_periods = len(df)
        else:
            n_periods = min(n_periods, len(df))

        # Get the periods to test (most recent N)
        df_test = df.tail(n_periods).copy()

        logger.info(f"Running backtest on {len(df_test)} periods")

        results = []
        for idx, row in df_test.iterrows():
            # Get spend for this period (already in scaled units)
            spend = np.array([row[ch] for ch in self.channels])

            # Predict response using saturation curves
            predicted = self.predict_response_at_spend(spend)

            # Get actual response (need to unscale - it's normalized by max)
            actual_scaled = row[target_col]
            actual = actual_scaled * self.target_scale

            # Build result row
            result_row = {
                'date': row[date_col],
                'actual': actual,
                'predicted': predicted,
                'residual': actual - predicted,
            }

            # Add per-channel spend (for analysis)
            for ch in self.channels:
                result_row[f'{ch}_spend'] = row[ch]

            results.append(result_row)

        result_df = pd.DataFrame(results)

        # Calculate percentage error (handle division by zero)
        result_df['pct_error'] = np.where(
            result_df['actual'] != 0,
            result_df['residual'] / result_df['actual'] * 100,
            0
        )

        return result_df

    def get_metrics(self, backtest_df: pd.DataFrame) -> dict:
        """
        Calculate validation metrics from backtest results.

        Parameters
        ----------
        backtest_df : pd.DataFrame
            Output from run_backtest().

        Returns
        -------
        dict
            Metrics including:
            - r2: R-squared (coefficient of determination)
            - mape: Mean Absolute Percentage Error
            - rmse: Root Mean Squared Error
            - correlation: Pearson correlation coefficient
            - mean_actual: Mean of actual values
            - mean_predicted: Mean of predicted values
            - n_periods: Number of periods in backtest
        """
        actual = backtest_df['actual'].values
        predicted = backtest_df['predicted'].values
        residuals = backtest_df['residual'].values

        # R-squared
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((actual - actual.mean()) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # MAPE (avoid division by zero)
        nonzero_mask = actual != 0
        if nonzero_mask.any():
            mape = np.mean(np.abs(residuals[nonzero_mask] / actual[nonzero_mask])) * 100
        else:
            mape = 0

        # RMSE
        rmse = np.sqrt(np.mean(residuals ** 2))

        # Correlation
        if len(actual) > 1:
            correlation = np.corrcoef(actual, predicted)[0, 1]
        else:
            correlation = 0

        return {
            'r2': float(r2),
            'mape': float(mape),
            'rmse': float(rmse),
            'correlation': float(correlation),
            'mean_actual': float(actual.mean()),
            'mean_predicted': float(predicted.mean()),
            'n_periods': len(backtest_df),
        }

    def get_channel_analysis(self, backtest_df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze prediction accuracy by channel contribution.

        For each channel, calculates correlation between spend
        and prediction error to identify problematic channels.

        Parameters
        ----------
        backtest_df : pd.DataFrame
            Output from run_backtest().

        Returns
        -------
        pd.DataFrame
            Per-channel analysis with:
            - channel: Channel name
            - mean_spend: Average spend
            - spend_correlation: Correlation between spend and residual
            - high_spend_error: Mean error when spend > median
            - low_spend_error: Mean error when spend < median
        """
        results = []

        for ch in self.channels:
            spend_col = f'{ch}_spend'
            if spend_col not in backtest_df.columns:
                continue

            spend = backtest_df[spend_col].values
            residuals = backtest_df['residual'].values

            # Correlation between spend and residual
            if len(spend) > 1 and np.std(spend) > 0:
                spend_corr = np.corrcoef(spend, residuals)[0, 1]
            else:
                spend_corr = 0

            # Split by median spend
            median_spend = np.median(spend)
            high_spend_mask = spend > median_spend
            low_spend_mask = spend <= median_spend

            high_spend_error = residuals[high_spend_mask].mean() if high_spend_mask.any() else 0
            low_spend_error = residuals[low_spend_mask].mean() if low_spend_mask.any() else 0

            results.append({
                'channel': ch,
                'mean_spend': float(spend.mean()),
                'spend_correlation': float(spend_corr),
                'high_spend_error': float(high_spend_error),
                'low_spend_error': float(low_spend_error),
            })

        return pd.DataFrame(results)

    def generate_report(self, n_periods: int | None = None) -> dict:
        """
        Generate a complete backtest validation report.

        Parameters
        ----------
        n_periods : int, optional
            Number of periods to backtest.

        Returns
        -------
        dict
            Complete report with:
            - metrics: Overall metrics (R², MAPE, etc.)
            - backtest_df: Period-by-period results
            - channel_analysis: Per-channel breakdown
            - is_valid: Whether model passes validation thresholds
            - validation_message: Human-readable assessment
        """
        backtest_df = self.run_backtest(n_periods)
        metrics = self.get_metrics(backtest_df)
        channel_analysis = self.get_channel_analysis(backtest_df)

        # Determine if model is valid (reasonable thresholds)
        is_valid = metrics['r2'] > 0.5 and metrics['mape'] < 30

        if metrics['r2'] > 0.7 and metrics['mape'] < 15:
            validation_message = "Excellent fit - model predictions closely match actual results"
        elif metrics['r2'] > 0.5 and metrics['mape'] < 25:
            validation_message = "Good fit - model predictions are reasonably accurate"
        elif metrics['r2'] > 0.3:
            validation_message = "Moderate fit - model captures some variance but has significant errors"
        else:
            validation_message = "Poor fit - model predictions do not match actual results well"

        return {
            'metrics': metrics,
            'backtest_df': backtest_df,
            'channel_analysis': channel_analysis,
            'is_valid': is_valid,
            'validation_message': validation_message,
        }
