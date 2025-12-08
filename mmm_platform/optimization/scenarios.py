"""
Scenario analysis and target-based optimization.

This module provides advanced optimization capabilities:
- Target-based optimization (find budget to achieve a target)
- Scenario comparison across budget levels
- Efficiency frontier analysis
"""

from typing import Any, Callable
import numpy as np
import pandas as pd
import logging
from scipy.optimize import brentq, minimize_scalar

from mmm_platform.optimization.results import TargetResult, ScenarioResult, OptimizationResult

logger = logging.getLogger(__name__)


class TargetOptimizer:
    """
    Find the budget required to achieve a target response.

    Uses binary search to find the minimum budget that achieves
    a specified target revenue/conversions.

    Parameters
    ----------
    allocator : BudgetAllocator
        Budget allocator instance.
    """

    def __init__(self, allocator: Any):
        """
        Initialize target optimizer.

        Parameters
        ----------
        allocator : BudgetAllocator
            A configured BudgetAllocator instance.
        """
        self.allocator = allocator
        self._cache: dict[float, OptimizationResult] = {}

    def find_budget_for_target(
        self,
        target_response: float,
        budget_range: tuple[float, float] = (10000, 1000000),
        tolerance: float = 0.01,
        max_iterations: int = 20,
        **optimize_kwargs,
    ) -> TargetResult:
        """
        Find minimum budget to achieve target response.

        Uses binary search to find the budget level where
        expected response equals the target.

        Parameters
        ----------
        target_response : float
            Target revenue/conversions to achieve.
        budget_range : tuple
            (min_budget, max_budget) to search within.
        tolerance : float
            Relative tolerance for target matching.
        max_iterations : int
            Maximum search iterations.
        **optimize_kwargs
            Additional arguments passed to allocator.optimize().

        Returns
        -------
        TargetResult
            Result with required budget and allocation.
        """
        min_budget, max_budget = budget_range
        logger.info(
            f"Searching for budget to achieve target={target_response:,.0f} "
            f"in range [{min_budget:,.0f}, {max_budget:,.0f}]"
        )

        # First check if target is achievable at max budget
        max_result = self._get_response_at_budget(max_budget, **optimize_kwargs)
        if max_result.expected_response < target_response:
            logger.warning(
                f"Target {target_response:,.0f} not achievable. "
                f"Max achievable at ${max_budget:,.0f}: {max_result.expected_response:,.0f}"
            )
            return TargetResult(
                target_response=target_response,
                budget_range_searched=budget_range,
                required_budget=max_budget,
                optimal_allocation=max_result.optimal_allocation,
                achievable=False,
                expected_response=max_result.expected_response,
                response_ci_low=max_result.response_ci_low,
                response_ci_high=max_result.response_ci_high,
                iterations=1,
                message=f"Target not achievable. Max response at ${max_budget:,.0f}: {max_result.expected_response:,.0f}",
                optimization_result=max_result,
            )

        # Check if we're already above target at min budget
        min_result = self._get_response_at_budget(min_budget, **optimize_kwargs)
        if min_result.expected_response >= target_response:
            logger.info(f"Target achievable at minimum budget ${min_budget:,.0f}")
            return TargetResult(
                target_response=target_response,
                budget_range_searched=budget_range,
                required_budget=min_budget,
                optimal_allocation=min_result.optimal_allocation,
                achievable=True,
                expected_response=min_result.expected_response,
                response_ci_low=min_result.response_ci_low,
                response_ci_high=min_result.response_ci_high,
                iterations=1,
                message=f"Target achievable at minimum budget",
                optimization_result=min_result,
            )

        # Binary search
        iterations = 0
        low, high = min_budget, max_budget

        while iterations < max_iterations:
            iterations += 1
            mid = (low + high) / 2

            result = self._get_response_at_budget(mid, **optimize_kwargs)
            response = result.expected_response

            # Check if we're close enough
            relative_error = abs(response - target_response) / target_response
            if relative_error <= tolerance:
                logger.info(
                    f"Found budget ${mid:,.0f} achieving response {response:,.0f} "
                    f"(target: {target_response:,.0f}, error: {relative_error:.2%})"
                )
                return TargetResult(
                    target_response=target_response,
                    budget_range_searched=budget_range,
                    required_budget=mid,
                    optimal_allocation=result.optimal_allocation,
                    achievable=True,
                    expected_response=response,
                    response_ci_low=result.response_ci_low,
                    response_ci_high=result.response_ci_high,
                    iterations=iterations,
                    message=f"Target achieved within {tolerance:.1%} tolerance",
                    optimization_result=result,
                )

            # Narrow the search
            if response < target_response:
                low = mid
            else:
                high = mid

            logger.debug(
                f"Iteration {iterations}: budget=${mid:,.0f}, "
                f"response={response:,.0f}, target={target_response:,.0f}"
            )

        # Return best result after max iterations
        final_result = self._get_response_at_budget((low + high) / 2, **optimize_kwargs)
        logger.warning(
            f"Max iterations reached. Best: budget=${(low + high) / 2:,.0f}, "
            f"response={final_result.expected_response:,.0f}"
        )

        return TargetResult(
            target_response=target_response,
            budget_range_searched=budget_range,
            required_budget=(low + high) / 2,
            optimal_allocation=final_result.optimal_allocation,
            achievable=True,
            expected_response=final_result.expected_response,
            response_ci_low=final_result.response_ci_low,
            response_ci_high=final_result.response_ci_high,
            iterations=iterations,
            message=f"Max iterations reached. Response within {abs(final_result.expected_response - target_response) / target_response:.1%}",
            optimization_result=final_result,
        )

    def _get_response_at_budget(
        self,
        budget: float,
        **optimize_kwargs,
    ) -> OptimizationResult:
        """Get optimization result at a specific budget (with caching)."""
        # Round budget to avoid floating point cache misses
        budget_key = round(budget, 2)

        if budget_key not in self._cache:
            self._cache[budget_key] = self.allocator.optimize(
                total_budget=budget,
                **optimize_kwargs,
            )

        return self._cache[budget_key]

    def clear_cache(self):
        """Clear the optimization result cache."""
        self._cache.clear()


class ResponseCurveAnalyzer:
    """
    Analyze the response curve across budget levels.

    Provides visualization data and efficiency analysis.
    """

    def __init__(self, allocator: Any):
        """
        Initialize response curve analyzer.

        Parameters
        ----------
        allocator : BudgetAllocator
            A configured BudgetAllocator instance.
        """
        self.allocator = allocator

    def compute_response_curve(
        self,
        budget_range: tuple[float, float],
        num_points: int = 20,
        **optimize_kwargs,
    ) -> pd.DataFrame:
        """
        Compute response curve across budget range.

        Parameters
        ----------
        budget_range : tuple
            (min_budget, max_budget) range to analyze.
        num_points : int
            Number of points to compute.
        **optimize_kwargs
            Additional arguments passed to optimizer.

        Returns
        -------
        pd.DataFrame
            Curve data with columns: budget, expected_response, marginal_roi, etc.
        """
        min_budget, max_budget = budget_range
        budgets = np.linspace(min_budget, max_budget, num_points)

        logger.info(
            f"Computing response curve: {num_points} points in "
            f"[${min_budget:,.0f}, ${max_budget:,.0f}]"
        )

        results = []
        prev_result = None

        for budget in budgets:
            result = self.allocator.optimize(
                total_budget=budget,
                **optimize_kwargs,
            )

            row = {
                "budget": budget,
                "expected_response": result.expected_response,
                "response_ci_low": result.response_ci_low,
                "response_ci_high": result.response_ci_high,
                "success": result.success,
                "roi": result.expected_response / budget if budget > 0 else 0,
            }

            # Marginal metrics
            if prev_result is not None:
                budget_delta = budget - prev_result.total_budget
                response_delta = result.expected_response - prev_result.expected_response
                row["marginal_response"] = response_delta
                row["marginal_roi"] = (
                    response_delta / budget_delta if budget_delta > 0 else 0
                )
            else:
                row["marginal_response"] = result.expected_response
                row["marginal_roi"] = row["roi"]

            results.append(row)
            prev_result = result

        return pd.DataFrame(results)

    def find_efficient_budget(
        self,
        budget_range: tuple[float, float],
        target_marginal_roi: float = 1.0,
        **optimize_kwargs,
    ) -> float:
        """
        Find the budget level where marginal ROI equals target.

        This is the "efficient" budget where incremental spend
        still generates positive returns.

        Parameters
        ----------
        budget_range : tuple
            (min_budget, max_budget) to search.
        target_marginal_roi : float
            Target marginal ROI (1.0 = breakeven).
        **optimize_kwargs
            Additional arguments passed to optimizer.

        Returns
        -------
        float
            Budget level where marginal ROI = target.
        """
        curve = self.compute_response_curve(
            budget_range,
            num_points=15,
            **optimize_kwargs,
        )

        # Find where marginal ROI crosses the target
        for i in range(1, len(curve)):
            if curve.iloc[i]["marginal_roi"] <= target_marginal_roi:
                # Linear interpolation
                prev = curve.iloc[i - 1]
                curr = curve.iloc[i]

                if prev["marginal_roi"] == curr["marginal_roi"]:
                    return curr["budget"]

                # Interpolate budget
                ratio = (target_marginal_roi - prev["marginal_roi"]) / (
                    curr["marginal_roi"] - prev["marginal_roi"]
                )
                budget = prev["budget"] + ratio * (curr["budget"] - prev["budget"])
                return budget

        # If never crosses, return max budget (still efficient)
        return budget_range[1]


def compute_efficiency_frontier(
    scenario_result: ScenarioResult,
) -> pd.DataFrame:
    """
    Compute the efficiency frontier from scenario results.

    The efficiency frontier shows the maximum achievable response
    at each budget level.

    Parameters
    ----------
    scenario_result : ScenarioResult
        Result from scenario_analysis().

    Returns
    -------
    pd.DataFrame
        Frontier data with budget, response, efficiency metrics.
    """
    curve = scenario_result.efficiency_curve.copy()

    # Calculate efficiency metrics
    curve["efficiency"] = curve["expected_response"] / curve["budget"]
    curve["cumulative_efficiency"] = (
        curve["expected_response"].cumsum() / curve["budget"].cumsum()
    )

    # Mark diminishing returns threshold (where marginal < average)
    curve["average_roi"] = curve["expected_response"] / curve["budget"]
    curve["diminishing_returns"] = curve["marginal_response"] < curve["average_roi"]

    # Find optimal budget (highest efficiency before diminishing returns)
    efficient_mask = ~curve["diminishing_returns"]
    if efficient_mask.any():
        optimal_idx = curve.loc[efficient_mask, "expected_response"].idxmax()
        curve["is_optimal"] = curve.index == optimal_idx
    else:
        curve["is_optimal"] = False

    return curve
