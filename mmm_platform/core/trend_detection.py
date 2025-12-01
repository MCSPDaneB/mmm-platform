"""
Trend detection utilities for time series analysis.
"""

from scipy import stats
import numpy as np
import pandas as pd


def detect_trend(series: pd.Series) -> dict:
    """
    Test for significant linear trend in a time series.

    Uses linear regression to detect if there's a statistically significant
    trend (p < 0.05) in the data.

    Parameters
    ----------
    series : pd.Series
        Time series data to test for trend.

    Returns
    -------
    dict
        Dictionary containing:
        - has_trend: bool - Whether trend is significant (p < 0.05)
        - p_value: float - P-value from regression
        - slope: float - Slope per period
        - r_squared: float - R² of the trend line
        - direction: str - 'increasing', 'decreasing', or 'none'
        - trend_line: np.ndarray - Fitted trend values for plotting
        - intercept: float - Y-intercept of trend line
    """
    y = series.dropna().values
    x = np.arange(len(y))

    if len(y) < 3:
        # Not enough data for trend detection
        return {
            'has_trend': False,
            'p_value': 1.0,
            'slope': 0.0,
            'r_squared': 0.0,
            'direction': 'none',
            'trend_line': np.zeros(len(y)),
            'intercept': 0.0,
        }

    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    has_trend = p_value < 0.05

    if has_trend:
        direction = 'increasing' if slope > 0 else 'decreasing'
    else:
        direction = 'none'

    return {
        'has_trend': has_trend,
        'p_value': p_value,
        'slope': slope,
        'r_squared': r_value ** 2,
        'direction': direction,
        'trend_line': intercept + slope * x,
        'intercept': intercept,
    }
