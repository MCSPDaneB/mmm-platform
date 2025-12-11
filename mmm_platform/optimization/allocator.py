"""
Budget allocation optimization using PyMC-Marketing.

This module provides the BudgetAllocator class that wraps PyMC-Marketing's
budget optimization capabilities with a user-friendly interface.
"""

from typing import Any
import numpy as np
import pandas as pd
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
        seasonal_indices: dict[str, float] | None = None,
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
        seasonal_indices : dict[str, float], optional
            Per-channel seasonal effectiveness multipliers.
            {channel_name: index} where index > 1 = more effective during period.
            If None, no seasonal adjustment is applied.
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
            # Use our custom optimizer directly (more reliable, supports all risk profiles)
            allocation_dict, scipy_result = self._optimize_with_working_gradients(
                total_budget, channel_bounds, seasonal_indices=seasonal_indices
            )

            # Get response estimates and risk metrics from our optimizer
            risk_metrics = scipy_result.risk_metrics
            expected_response = risk_metrics.get('expected_response', 0.0)
            ci_low = risk_metrics.get('response_ci_low', 0.0)
            ci_high = risk_metrics.get('response_ci_high', 0.0)
            response_var = risk_metrics.get('response_var')
            response_cvar = risk_metrics.get('response_cvar')
            response_sharpe = risk_metrics.get('response_sharpe')
            response_std = risk_metrics.get('response_std')

            # Get actual allocated amount and unallocated budget (from bounds constraints)
            actual_allocated = getattr(scipy_result, 'actual_allocated', total_budget)
            unallocated_budget = getattr(scipy_result, 'unallocated_budget', 0.0)

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
                total_budget=actual_allocated,  # Use actual allocated, not requested
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
                response_var=response_var,
                response_cvar=response_cvar,
                response_sharpe=response_sharpe,
                response_std=response_std,
                unallocated_budget=unallocated_budget if unallocated_budget > 1 else None,  # >$1 threshold for floating-point noise
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

    def optimize_with_efficiency_floor(
        self,
        total_budget: float,
        efficiency_metric: str,
        efficiency_target: float,
        channel_bounds: dict[str, tuple[float, float]] | None = None,
        seasonal_indices: dict[str, float] | None = None,
        **kwargs,
    ) -> OptimizationResult:
        """
        Optimize budget with an efficiency floor (minimum ROI or maximum CPA).

        If the efficiency target can be achieved at the full budget, spends all.
        If not, finds the maximum budget that achieves the target and returns
        the unallocated amount.

        Parameters
        ----------
        total_budget : float
            Maximum budget available to allocate.
        efficiency_metric : str
            Either "roi" (minimum ROI required) or "cpa" (maximum CPA allowed).
        efficiency_target : float
            Target value: ROI multiplier (e.g., 2.0 = 2x) or CPA in dollars.
        channel_bounds : dict, optional
            Per-channel bounds.
        seasonal_indices : dict, optional
            Per-channel seasonal effectiveness multipliers.
        **kwargs
            Additional arguments passed to optimize().

        Returns
        -------
        OptimizationResult
            Result with optimal allocation and unallocated_budget if applicable.
        """
        logger.info(
            f"Optimizing with efficiency floor: {efficiency_metric}={efficiency_target}, "
            f"max_budget=${total_budget:,.0f}"
        )

        def compute_efficiency(result: OptimizationResult) -> float:
            """Compute efficiency metric for a result."""
            if result.expected_response == 0:
                return float('inf') if efficiency_metric == "cpa" else 0.0

            if efficiency_metric == "roi":
                return result.expected_response / result.total_budget
            else:  # cpa
                return result.total_budget / result.expected_response

        def meets_target(result: OptimizationResult) -> bool:
            """Check if result meets efficiency target."""
            efficiency = compute_efficiency(result)
            if efficiency_metric == "roi":
                return efficiency >= efficiency_target
            else:  # cpa
                return efficiency <= efficiency_target

        # First, try full budget
        full_result = self.optimize(
            total_budget=total_budget,
            channel_bounds=channel_bounds,
            seasonal_indices=seasonal_indices,
            **kwargs,
        )

        full_efficiency = compute_efficiency(full_result)

        if meets_target(full_result):
            # Full budget meets target - spend it all
            full_result.unallocated_budget = 0.0
            full_result.efficiency_target = efficiency_target
            full_result.efficiency_metric = efficiency_metric
            full_result.achieved_efficiency = full_efficiency
            logger.info(
                f"Full budget meets {efficiency_metric} target: "
                f"achieved={full_efficiency:.2f}, target={efficiency_target:.2f}"
            )
            return full_result

        # Binary search to find max budget that meets target
        logger.info(
            f"Full budget doesn't meet target ({full_efficiency:.2f} vs {efficiency_target:.2f}), "
            f"searching for optimal budget level..."
        )

        low = 0.0
        high = total_budget
        best_result = None
        best_budget = 0.0

        # Binary search iterations
        for _ in range(15):  # ~0.01% precision on budget
            mid = (low + high) / 2

            if mid < total_budget * 0.01:
                # Budget too small to be meaningful
                break

            result = self.optimize(
                total_budget=mid,
                channel_bounds=channel_bounds,
                seasonal_indices=seasonal_indices,
                **kwargs,
            )

            if meets_target(result):
                # This budget level works, try higher
                best_result = result
                best_budget = mid
                low = mid
            else:
                # This budget level doesn't work, try lower
                high = mid

        if best_result is None:
            # Couldn't find any budget that meets target
            logger.warning(
                f"No budget level achieves {efficiency_metric} target of {efficiency_target}"
            )
            # Return zero allocation
            result = OptimizationResult(
                optimal_allocation={ch: 0.0 for ch in self.channels},
                total_budget=0.0,
                expected_response=0.0,
                response_ci_low=0.0,
                response_ci_high=0.0,
                success=True,
                message=f"No budget level achieves {efficiency_metric} target of {efficiency_target}",
                iterations=0,
                objective_value=0.0,
                num_periods=self.num_periods,
                utility_function=self.utility_name,
                unallocated_budget=total_budget,
                efficiency_target=efficiency_target,
                efficiency_metric=efficiency_metric,
                achieved_efficiency=None,
            )
            return result

        # Set efficiency floor fields
        best_result.unallocated_budget = total_budget - best_budget
        best_result.efficiency_target = efficiency_target
        best_result.efficiency_metric = efficiency_metric
        best_result.achieved_efficiency = compute_efficiency(best_result)

        logger.info(
            f"Found optimal budget: ${best_budget:,.0f} "
            f"({efficiency_metric}={best_result.achieved_efficiency:.2f}), "
            f"unallocated: ${best_result.unallocated_budget:,.0f}"
        )

        return best_result

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

    def _optimize_with_working_gradients(
        self,
        total_budget: float,
        channel_bounds: dict[str, tuple[float, float]],
        seasonal_indices: dict[str, float] | None = None,
    ) -> tuple[dict[str, float], Any]:
        """
        Custom optimization with risk-aware objectives.

        Supports all risk profiles:
        - mean: Fast path using posterior means (original behavior)
        - var: Value at Risk (5th percentile - conservative)
        - cvar: Conditional VaR (very conservative)
        - sharpe: Risk-adjusted returns

        Parameters
        ----------
        total_budget : float
            Total budget to allocate.
        channel_bounds : dict
            Per-channel bounds: {channel_name: (min, max)}.
        seasonal_indices : dict, optional
            Per-channel seasonal effectiveness multipliers.
            {channel_name: index} where index > 1 = more effective.

        Returns
        -------
        tuple[dict, Any]
            (allocation_dict, scipy_result)
        """
        from scipy.optimize import minimize
        from mmm_platform.optimization.risk_objectives import (
            RiskAwareObjective,
            PosteriorSamples,
        )
        import time

        mmm = self.bridge.mmm
        channels = list(mmm.channel_columns)
        n_channels = len(channels)
        num_periods = self.num_periods

        # Get scales
        df_scaled = self.bridge.wrapper.df_scaled
        spend_scale = self.bridge.config.data.spend_scale
        target_col = self.bridge.config.data.target_column
        target_scale = float(df_scaled[target_col].max())

        # Get x_max for each channel (in scaled units)
        x_maxes = np.array([float(df_scaled[ch].max()) for ch in channels])

        # Extract posterior samples
        # For mean: use minimal samples (fast path uses means anyway)
        # For VaR/CVaR/Sharpe: use 500 samples for uncertainty quantification
        n_samples = 500 if self.utility_name != "mean" else 100
        posterior_samples = PosteriorSamples.from_idata(mmm.idata, n_samples=n_samples)

        # Convert seasonal indices dict to numpy array matching channel order
        seasonal_indices_array = None
        if seasonal_indices is not None:
            seasonal_indices_array = np.array([
                seasonal_indices.get(ch, 1.0) for ch in channels
            ])
            logger.info(
                f"Applying seasonal indices: {dict(zip(channels, seasonal_indices_array))}"
            )

        # Create risk-aware objective
        risk_objective = RiskAwareObjective(
            posterior_samples=posterior_samples,
            x_maxes=x_maxes,
            target_scale=target_scale,
            num_periods=num_periods,
            risk_profile=self.utility_name,
            confidence_level=0.95,
            risk_free_rate=0.0,
            seasonal_indices=seasonal_indices_array,
        )

        logger.info(
            f"Running optimizer with risk_profile={self.utility_name}, "
            f"target_scale={target_scale:.0f}, n_samples={n_samples}, "
            f"seasonal={'yes' if seasonal_indices is not None else 'no'}"
        )

        # Build bounds (convert from real $ to scaled units, per-period)
        bounds_list = []
        for ch in channels:
            ch_bounds = channel_bounds.get(ch, (0, total_budget))
            bounds_list.append((
                ch_bounds[0] / spend_scale / num_periods,
                ch_bounds[1] / spend_scale / num_periods
            ))

        # Budget constraint (in scaled units) - use inequality to allow partial allocation
        # when bounds are tighter than the requested budget
        budget_per_period_scaled = total_budget / spend_scale / num_periods
        constraints = {'type': 'ineq', 'fun': lambda x: budget_per_period_scaled - x.sum()}

        # Initial guess: uniform allocation (in scaled units)
        x0 = np.ones(n_channels) * budget_per_period_scaled / n_channels

        # Progress tracking
        start_time = time.time()

        # Run SLSQP optimizer with risk-aware objective
        result = minimize(
            risk_objective.objective,
            x0,
            method='SLSQP',
            jac=risk_objective.gradient,
            bounds=bounds_list,
            constraints=constraints,
            options={'maxiter': 200, 'ftol': 1e-6},
        )

        # Scale back to real dollars
        allocation = {
            ch: float(val * spend_scale * num_periods)
            for ch, val in zip(channels, result.x)
        }

        # Calculate actual allocated vs requested budget
        actual_allocated = sum(allocation.values())
        unallocated_budget = total_budget - actual_allocated

        # Compute all risk metrics at optimal allocation for results
        risk_metrics = risk_objective.compute_all_risk_metrics(result.x)

        # Store risk metrics and unallocated budget in result for later use
        result.risk_metrics = risk_metrics
        result.actual_allocated = actual_allocated
        result.unallocated_budget = unallocated_budget

        elapsed = time.time() - start_time

        # Log with unallocated info if applicable
        if unallocated_budget > total_budget * 0.01:  # >1% unallocated
            logger.warning(
                f"Bounds prevented full budget allocation: "
                f"requested=${total_budget:,.0f}, allocated=${actual_allocated:,.0f}, "
                f"unallocated=${unallocated_budget:,.0f}"
            )

        logger.info(
            f"Optimization complete ({self.utility_name}): {result.nit} iterations "
            f"in {elapsed:.1f}s, success={result.success}, "
            f"allocated=${actual_allocated:,.0f}"
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
