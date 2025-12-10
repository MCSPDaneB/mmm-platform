"""
Budget allocation optimization using PyMC-Marketing.

This module provides the BudgetAllocator class that wraps PyMC-Marketing's
budget optimization capabilities with a user-friendly interface.
"""

from typing import Any, Callable
import numpy as np
import pandas as pd
import xarray as xr
import logging

from mmm_platform.optimization.bridge import OptimizationBridge, UTILITY_FUNCTIONS
from mmm_platform.optimization.results import OptimizationResult, TargetResult, ScenarioResult

logger = logging.getLogger(__name__)


class BudgetAllocator:
    """
    High-level interface for budget optimization.

    Wraps PyMC-Marketing's BudgetOptimizer with a user-friendly API
    for common optimization use cases.

    Parameters
    ----------
    wrapper : MMMWrapper
        A fitted MMMWrapper instance.
    num_periods : int
        Number of time periods for optimization (e.g., 8 weeks).
    utility : str
        Utility function for optimization:
        - "mean": Risk-neutral, maximize expected response
        - "var": Value at Risk (conservative)
        - "cvar": Conditional VaR / Expected Shortfall (very conservative)
        - "sharpe": Risk-adjusted return

    Examples
    --------
    >>> allocator = BudgetAllocator(wrapper, num_periods=8)
    >>> result = allocator.optimize(total_budget=100000)
    >>> print(result.optimal_allocation)
    """

    def __init__(
        self,
        wrapper: Any,
        num_periods: int = 8,
        utility: str = "mean",
    ):
        """
        Initialize the budget allocator.

        Parameters
        ----------
        wrapper : MMMWrapper
            A fitted MMMWrapper instance.
        num_periods : int
            Number of periods for the optimization horizon.
        utility : str
            Name of utility function to use.
        """
        self.bridge = OptimizationBridge(wrapper)
        self.num_periods = num_periods
        self.utility_name = utility
        self._utility_fn = self.bridge.get_utility_function(utility)

        logger.info(
            f"BudgetAllocator initialized: {num_periods} periods, "
            f"utility={utility}, channels={self.bridge.channel_columns}"
        )

    @property
    def channels(self) -> list[str]:
        """Get the list of optimizable channels."""
        return self.bridge.get_optimizable_channels()

    @property
    def mmm(self):
        """Get the underlying PyMC-Marketing MMM model."""
        return self.bridge.mmm

    def optimize(
        self,
        total_budget: float,
        channel_bounds: dict[str, tuple[float, float]] | None = None,
        constraints: list | None = None,
        compare_to_current: bool = False,
        comparison_mode: str = "average",
        comparison_n_weeks: int | None = None,
        **kwargs,
    ) -> OptimizationResult:
        """
        Optimize budget allocation across channels.

        Parameters
        ----------
        total_budget : float
            Total budget to allocate (in original units, e.g., dollars).
        channel_bounds : dict, optional
            Per-channel bounds: {channel_name: (min, max)}.
            If None, uses default bounds based on historical spend.
        constraints : list, optional
            Custom constraints (PyMC-Marketing Constraint objects).
        compare_to_current : bool
            If True, include comparison to current/historical allocation.
        comparison_mode : str
            How to calculate baseline for comparison:
            - "average": Average spend per period × num_periods (default)
            - "last_n_weeks": Actual spend from last N weeks
            - "most_recent_period": Actual spend from most recent matching period
        comparison_n_weeks : int, optional
            Number of weeks for "last_n_weeks" mode.
        **kwargs
            Additional arguments passed to optimize_budget().

        Returns
        -------
        OptimizationResult
            Complete optimization result with allocation and analysis.
        """
        logger.info(f"Starting optimization: budget=${total_budget:,.0f}")

        # Get default bounds if not provided
        if channel_bounds is None:
            # Default: allow 0 to 2x average spend per period * num_periods
            avg_spend = self.bridge.get_average_period_spend()
            channel_bounds = {}
            for ch, avg in avg_spend.items():
                max_bound = avg * self.num_periods * 2
                # Ensure max_bound is at least as large as proportional share
                proportional_share = total_budget / len(self.channels)
                max_bound = max(max_bound, proportional_share * 2)
                channel_bounds[ch] = (0.0, max_bound)

        # Build constraint list
        constraint_list = []
        if constraints:
            constraint_list.extend(constraints)

        try:
            # Call PyMC-Marketing's optimize_budget
            optimal_allocation, scipy_result = self.mmm.optimize_budget(
                budget=total_budget,
                num_periods=self.num_periods,
                budget_bounds=channel_bounds,
                utility_function=self._utility_fn,
                constraints=constraint_list if constraint_list else (),
                default_constraints=True,
                callback=False,
                **kwargs,
            )

            # Convert xarray to dict
            allocation_dict = self._xarray_to_dict(optimal_allocation)

            # Detect flat allocation (indicates PyMC-Marketing gradient bug)
            used_fallback = False
            if self._is_flat_allocation(allocation_dict):
                logger.warning(
                    f"PyMC-Marketing optimizer returned flat allocation after {scipy_result.nit} iteration(s) "
                    "(likely zero/near-zero gradients). Using custom optimizer with working gradients."
                )
                allocation_dict, scipy_result = self._optimize_with_working_gradients(
                    total_budget, channel_bounds
                )
                used_fallback = True

            # Get response estimates by evaluating at the optimal allocation
            # Note: scipy_result.fun is the optimizer's objective value, not the actual response
            expected_response, ci_low, ci_high = self.bridge.estimate_response_at_allocation(
                allocation_dict, self.num_periods
            )

            # Get current allocation for comparison
            current_allocation = None
            current_response = None
            if compare_to_current:
                current_allocation = self.bridge.get_current_allocation(
                    num_periods=self.num_periods,
                    comparison_mode=comparison_mode,
                    n_weeks=comparison_n_weeks,
                )
                current_response, _, _ = self.bridge.estimate_response_at_allocation(
                    current_allocation, self.num_periods
                )

            result = OptimizationResult(
                optimal_allocation=allocation_dict,
                total_budget=total_budget,
                expected_response=expected_response,
                response_ci_low=ci_low,
                response_ci_high=ci_high,
                success=scipy_result.success,
                message=scipy_result.message,
                iterations=scipy_result.nit,
                objective_value=scipy_result.fun,
                num_periods=self.num_periods,
                current_allocation=current_allocation,
                current_response=current_response,
                utility_function=self.utility_name,
                used_fallback=used_fallback,
                _raw_result=scipy_result,
            )

            logger.info(
                f"Optimization {'succeeded' if result.success else 'failed'}: "
                f"expected_response=${expected_response:,.0f}"
            )

            return result

        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            # Return a failed result
            return OptimizationResult(
                optimal_allocation={ch: 0.0 for ch in self.channels},
                total_budget=total_budget,
                expected_response=0.0,
                response_ci_low=0.0,
                response_ci_high=0.0,
                success=False,
                message=str(e),
                iterations=0,
                objective_value=0.0,
                num_periods=self.num_periods,
                utility_function=self.utility_name,
            )

    def optimize_incremental(
        self,
        base_allocation: dict[str, float],
        incremental_budget: float,
        **kwargs,
    ) -> OptimizationResult:
        """
        Optimize incremental budget on top of existing allocation.

        Parameters
        ----------
        base_allocation : dict
            Current/base allocation: {channel_name: current_spend}.
        incremental_budget : float
            Additional budget to allocate optimally.
        **kwargs
            Additional arguments passed to optimize().

        Returns
        -------
        OptimizationResult
            Optimization result with base + incremental allocation.
        """
        total_budget = sum(base_allocation.values()) + incremental_budget

        # Set bounds: minimum = base allocation per channel
        channel_bounds = {}
        for ch in self.channels:
            base = base_allocation.get(ch, 0.0)
            # Allow each channel to grow by up to the incremental amount
            max_bound = base + incremental_budget
            channel_bounds[ch] = (base, max_bound)

        result = self.optimize(
            total_budget=total_budget,
            channel_bounds=channel_bounds,
            compare_to_current=True,
            **kwargs,
        )

        # Override current_allocation with the base
        result.current_allocation = base_allocation
        result.current_response, _, _ = self.bridge.estimate_response_at_allocation(
            base_allocation, self.num_periods
        )

        return result

    def scenario_analysis(
        self,
        budget_scenarios: list[float],
        channel_bounds: dict[str, tuple[float, float]] | None = None,
        **kwargs,
    ) -> ScenarioResult:
        """
        Run optimization across multiple budget scenarios.

        Parameters
        ----------
        budget_scenarios : list[float]
            List of total budgets to analyze.
        channel_bounds : dict, optional
            Per-channel bounds (same for all scenarios).
        **kwargs
            Additional arguments passed to optimize().

        Returns
        -------
        ScenarioResult
            Complete scenario analysis with efficiency curve.
        """
        logger.info(f"Running scenario analysis for {len(budget_scenarios)} budgets")

        results = []
        for budget in sorted(budget_scenarios):
            # Adjust bounds proportionally for different budget levels
            adjusted_bounds = None
            if channel_bounds:
                # Scale max bounds proportionally
                base_budget = budget_scenarios[0]
                scale_factor = budget / base_budget if base_budget > 0 else 1.0
                adjusted_bounds = {
                    ch: (bounds[0], bounds[1] * scale_factor)
                    for ch, bounds in channel_bounds.items()
                }

            result = self.optimize(
                total_budget=budget,
                channel_bounds=adjusted_bounds,
                **kwargs,
            )
            results.append(result)

        # Build efficiency curve
        curve_data = []
        for i, result in enumerate(results):
            row = {
                "budget": result.total_budget,
                "expected_response": result.expected_response,
                "response_ci_low": result.response_ci_low,
                "response_ci_high": result.response_ci_high,
                "success": result.success,
            }

            # Calculate marginal response
            if i > 0:
                prev = results[i - 1]
                budget_delta = result.total_budget - prev.total_budget
                response_delta = result.expected_response - prev.expected_response
                row["marginal_response"] = (
                    response_delta / budget_delta if budget_delta > 0 else 0
                )
            else:
                row["marginal_response"] = (
                    result.expected_response / result.total_budget
                    if result.total_budget > 0
                    else 0
                )

            curve_data.append(row)

        efficiency_curve = pd.DataFrame(curve_data)

        return ScenarioResult(
            budget_scenarios=budget_scenarios,
            results=results,
            efficiency_curve=efficiency_curve,
        )

    def _xarray_to_dict(self, allocation: xr.DataArray) -> dict[str, float]:
        """Convert xarray DataArray to dict."""
        if "channel" in allocation.dims:
            return {
                str(ch): float(allocation.sel(channel=ch).values)
                for ch in allocation.coords["channel"].values
            }
        else:
            # Single dimension case
            return {
                str(idx): float(val)
                for idx, val in zip(allocation.coords.values(), allocation.values.flat)
            }

    def _is_flat_allocation(self, allocation: dict[str, float]) -> bool:
        """
        Check if allocation is flat (all channels get same amount).

        This indicates the optimizer failed due to zero gradients.
        """
        values = list(allocation.values())
        if not values:
            return True
        mean_val = np.mean(values)
        if mean_val == 0:
            return True
        # Flat if std is less than 1% of mean
        return np.std(values) < 0.01 * mean_val

    def _optimize_with_working_gradients(
        self,
        total_budget: float,
        channel_bounds: dict[str, tuple[float, float]],
    ) -> tuple[dict[str, float], Any]:
        """
        Custom optimization using the same calculation as MarginalROIAnalyzer.

        Uses beta * target_scale * saturation formula which is proven to work
        correctly in the Marginal Investment analysis page.

        Parameters
        ----------
        total_budget : float
            Total budget to allocate.
        channel_bounds : dict
            Per-channel bounds: {channel_name: (min, max)}.

        Returns
        -------
        tuple[dict, Any]
            (allocation_dict, scipy_result)
        """
        from scipy.optimize import minimize
        import time

        mmm = self.bridge.mmm
        channels = list(mmm.channel_columns)
        n_channels = len(channels)
        num_periods = self.num_periods

        # Get posterior means for saturation parameters
        posterior = mmm.idata.posterior
        lam_values = posterior['saturation_lam'].mean(dim=['chain', 'draw']).values
        beta_values = posterior['saturation_beta'].mean(dim=['chain', 'draw']).values

        # Get scales - SAME AS MARGINAL ROI PAGE (marginal_roi.py line 309)
        df_scaled = self.bridge.wrapper.df_scaled
        spend_scale = self.bridge.config.data.spend_scale
        target_col = self.bridge.config.data.target_column
        target_scale = float(df_scaled[target_col].max())

        # Get x_max for each channel (in scaled units, same as marginal_roi.py line 303)
        x_maxes = np.array([float(df_scaled[ch].max()) for ch in channels])
        # Avoid division by zero
        x_maxes = np.maximum(x_maxes, 1e-9)

        logger.info(f"Running optimizer with target_scale={target_scale:.0f}")
        logger.info(f"Beta values: min={beta_values.min():.4f}, max={beta_values.max():.4f}")

        # Build bounds (convert from real $ to scaled units, per-period)
        bounds_list = []
        for ch in channels:
            ch_bounds = channel_bounds.get(ch, (0, total_budget))
            bounds_list.append((
                ch_bounds[0] / spend_scale / num_periods,
                ch_bounds[1] / spend_scale / num_periods
            ))

        def objective(x):
            """
            Compute negative total response.
            x is per-period spend in SCALED units (same as df_scaled).
            Response = sum(beta * target_scale * saturation(x / x_max))
            """
            x_normalized = x / x_maxes
            exp_term = np.exp(-lam_values * x_normalized)
            saturation = (1 - exp_term) / (1 + exp_term)
            response = np.sum(beta_values * target_scale * saturation) * num_periods
            return -response

        def gradient(x):
            """
            Analytical gradient (same formula as marginal_roi.py line 77-79).
            d(response)/d(x) = beta * target_scale * sat_deriv / x_max
            """
            x_normalized = x / x_maxes
            exp_term = np.exp(-lam_values * x_normalized)
            sat_deriv = 2 * lam_values * exp_term / (1 + exp_term)**2
            grad = -beta_values * target_scale * sat_deriv / x_maxes * num_periods
            return grad

        # Budget constraint (in scaled units)
        budget_per_period_scaled = total_budget / spend_scale / num_periods
        constraints = {'type': 'eq', 'fun': lambda x: x.sum() - budget_per_period_scaled}

        # Initial guess: uniform allocation (in scaled units)
        x0 = np.ones(n_channels) * budget_per_period_scaled / n_channels

        # Progress tracking
        start_time = time.time()

        # Run SLSQP optimizer
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            jac=gradient,
            bounds=bounds_list,
            constraints=constraints,
            options={'maxiter': 200, 'ftol': 1e-6},
        )

        # Scale back to real dollars
        allocation = {
            ch: float(val * spend_scale * num_periods)
            for ch, val in zip(channels, result.x)
        }

        elapsed = time.time() - start_time
        logger.info(
            f"Optimization complete: {result.nit} iterations in {elapsed:.1f}s, "
            f"success={result.success}"
        )

        return allocation, result

    def get_channel_info(self) -> pd.DataFrame:
        """
        Get information about channels for the UI.

        Returns
        -------
        pd.DataFrame
            DataFrame with channel info: name, display_name, historical_spend, avg_spend.
        """
        display_names = self.bridge.channel_display_names
        historical = self.bridge.get_historical_spend()
        average = self.bridge.get_average_period_spend()

        data = []
        for ch in self.channels:
            data.append({
                "channel": ch,
                "display_name": display_names.get(ch, ch),
                "historical_spend": historical.get(ch, 0),
                "avg_period_spend": average.get(ch, 0),
            })

        return pd.DataFrame(data)


def create_allocator_from_session(session_state: Any) -> BudgetAllocator | None:
    """
    Create a BudgetAllocator from Streamlit session state.

    Parameters
    ----------
    session_state : st.session_state
        Streamlit session state containing current_model.

    Returns
    -------
    BudgetAllocator or None
        Allocator if model is fitted, None otherwise.
    """
    wrapper = session_state.get("current_model")

    if wrapper is None:
        return None

    if wrapper.idata is None:
        return None

    return BudgetAllocator(wrapper)
