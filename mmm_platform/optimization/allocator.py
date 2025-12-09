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
            if scipy_result.nit == 1 and self._is_flat_allocation(allocation_dict):
                logger.warning(
                    "PyMC-Marketing optimizer returned flat allocation in 1 iteration "
                    "(likely zero gradients). Using custom optimizer with working gradients."
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
                current_allocation = self.bridge.get_current_allocation(self.num_periods)
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
        Custom optimization using manual gradient computation.

        Bypasses PyMC-Marketing BudgetOptimizer's broken gradient flow
        for models loaded via load_from_idata().

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
        import pytensor.tensor as pt
        from pymc import do
        from pymc.model.transform.optimization import freeze_dims_and_data
        from pymc_marketing.pytensor_utils import extract_response_distribution
        from pymc_marketing.mmm.utility import average_response
        from pymc.pytensorf import rewrite_pregrad
        from pytensor import function
        from scipy.optimize import minimize

        mmm = self.bridge.mmm
        channels = list(mmm.channel_columns)
        n_channels = len(channels)
        channel_scales = mmm._channel_scales
        num_periods = self.num_periods

        # Get the model's actual date dimension (includes l_max for carryover)
        # This is critical - the budget tensor must match the model's channel_data shape
        model_date_dim = len(mmm.model.coords['date']) + mmm.adstock.l_max

        logger.info(
            f"Building custom gradient-based optimizer for {n_channels} channels, "
            f"model_date_dim={model_date_dim}, num_periods={num_periods}"
        )

        # Build bounds for per-period budget
        # The optimizer works with per-period values, bounds are total / num_periods
        bounds_list = []
        for ch in channels:
            ch_bounds = channel_bounds.get(ch, (0, total_budget))
            # Convert total bounds to per-period bounds
            bounds_list.append((ch_bounds[0] / num_periods, ch_bounds[1] / num_periods))

        # Create budget tensor variable (per-period allocation per channel)
        budgets_flat = pt.vector('budgets_flat')

        # Scale by channel_scales (same as BudgetOptimizer does)
        scaled_budgets = budgets_flat / pt.constant(channel_scales)

        # Broadcast to (model_date_dim, n_channels) shape
        # CRITICAL: Must use model's date dimension, not num_periods, for gradients to flow
        budget_tensor = pt.broadcast_to(
            pt.expand_dims(scaled_budgets, 0),
            shape=(model_date_dim, n_channels)
        )

        # Freeze model and substitute channel_data
        frozen_model = freeze_dims_and_data(mmm.model, data=[])
        new_model = do(frozen_model, {'channel_data': budget_tensor})

        # Extract response distribution with posterior samples
        response_dist = extract_response_distribution(
            pymc_model=new_model,
            idata=mmm.idata,
            response_variable='total_contribution',
        )

        # Objective: negative mean response (we're minimizing)
        objective = -average_response(samples=response_dist, budgets=budgets_flat)

        # Compute gradient
        gradient = pt.grad(rewrite_pregrad(objective), budgets_flat)

        # Compile the function
        logger.info("Compiling custom optimization function...")
        f = function([budgets_flat], [objective, gradient])

        # Wrapper for scipy
        def objective_and_grad(x):
            obj, grad = f(x.astype('float64'))
            return float(obj), grad.astype('float64')

        # Budget constraint: sum of per-period allocations = total_budget / num_periods
        budget_per_period = total_budget / num_periods
        constraints = {'type': 'eq', 'fun': lambda x: x.sum() - budget_per_period}

        # Initial guess: uniform allocation
        x0 = np.ones(n_channels) * budget_per_period / n_channels

        # Run SLSQP optimizer with our correct gradients
        logger.info("Running custom SLSQP optimization...")
        result = minimize(
            lambda x: objective_and_grad(x)[0],
            x0,
            method='SLSQP',
            jac=lambda x: objective_and_grad(x)[1],
            bounds=bounds_list,
            constraints=constraints,
            options={'maxiter': 100, 'ftol': 1e-6},
        )

        # Scale back to total budget (multiply per-period by num_periods)
        allocation = {ch: float(val * num_periods) for ch, val in zip(channels, result.x)}

        logger.info(
            f"Custom optimization complete: {result.nit} iterations, "
            f"success={result.success}, message={result.message}"
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
