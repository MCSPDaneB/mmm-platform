"""
Result dataclasses for budget optimization.

This module defines the data structures returned by optimization operations,
including allocation results, target search results, and scenario comparisons.
"""

from dataclasses import dataclass, field
from typing import Any
import pandas as pd
import numpy as np


@dataclass
class OptimizationResult:
    """
    Complete result from a budget optimization run.

    Contains the optimal allocation, expected response, uncertainty bounds,
    and optional comparison to current/historical spend.
    """

    # Core results
    optimal_allocation: dict[str, float]  # {channel_name: amount}
    total_budget: float
    expected_response: float

    # Uncertainty quantification
    response_ci_low: float  # 5th percentile
    response_ci_high: float  # 95th percentile

    # Optimization metadata
    success: bool
    message: str
    iterations: int
    objective_value: float  # Final objective (negative expected response)

    # Time-phased allocation (if applicable)
    allocation_by_period: pd.DataFrame | None = None  # (period x channel)
    num_periods: int = 1

    # Comparison to current (optional)
    current_allocation: dict[str, float] | None = None
    current_response: float | None = None

    # Utility function used
    utility_function: str = "mean"

    # Risk metrics (populated by optimizer)
    response_var: float | None = None  # Value at Risk (5th percentile)
    response_cvar: float | None = None  # Conditional VaR (expected shortfall)
    response_sharpe: float | None = None  # Sharpe ratio (mean/std)
    response_std: float | None = None  # Standard deviation of response

    # Efficiency floor results (for ROI/CPA floor mode)
    unallocated_budget: float | None = None  # Budget not allocated due to efficiency floor
    efficiency_target: float | None = None  # Target ROI or CPA
    efficiency_metric: str | None = None  # "roi" or "cpa"
    achieved_efficiency: float | None = None  # Actual ROI or CPA achieved

    # Raw scipy result for debugging
    _raw_result: Any = field(default=None, repr=False)

    @property
    def allocation_delta(self) -> dict[str, float] | None:
        """Change from current to optimal allocation per channel."""
        if self.current_allocation is None:
            return None
        return {
            ch: self.optimal_allocation.get(ch, 0) - self.current_allocation.get(ch, 0)
            for ch in set(self.optimal_allocation) | set(self.current_allocation)
        }

    @property
    def allocation_pct_change(self) -> dict[str, float] | None:
        """Percentage change from current to optimal per channel."""
        if self.current_allocation is None:
            return None
        result = {}
        for ch in self.optimal_allocation:
            current = self.current_allocation.get(ch, 0)
            optimal = self.optimal_allocation[ch]
            if current > 0:
                result[ch] = (optimal - current) / current * 100
            elif optimal > 0:
                result[ch] = float("inf")
            else:
                result[ch] = 0.0
        return result

    @property
    def response_uplift(self) -> float | None:
        """Expected response improvement over current."""
        if self.current_response is None:
            return None
        return self.expected_response - self.current_response

    @property
    def response_uplift_pct(self) -> float | None:
        """Percentage response improvement over current."""
        if self.current_response is None or self.current_response == 0:
            return None
        return (self.expected_response - self.current_response) / self.current_response * 100

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert allocation to a DataFrame with analysis columns.

        Returns:
            DataFrame with columns: channel, optimal, current (if available),
            delta, pct_change
        """
        data = {
            "channel": list(self.optimal_allocation.keys()),
            "optimal": list(self.optimal_allocation.values()),
        }

        if self.current_allocation:
            data["current"] = [
                self.current_allocation.get(ch, 0)
                for ch in self.optimal_allocation.keys()
            ]
            data["delta"] = [
                self.optimal_allocation[ch] - self.current_allocation.get(ch, 0)
                for ch in self.optimal_allocation.keys()
            ]
            data["pct_change"] = [
                (
                    (self.optimal_allocation[ch] - self.current_allocation.get(ch, 0))
                    / self.current_allocation.get(ch, 1)
                    * 100
                    if self.current_allocation.get(ch, 0) > 0
                    else 0
                )
                for ch in self.optimal_allocation.keys()
            ]

        df = pd.DataFrame(data)

        # Add percentage of total (use actual allocated sum for consistency)
        allocated_total = sum(self.optimal_allocation.values())
        df["pct_of_total"] = (
            df["optimal"] / allocated_total * 100 if allocated_total > 0 else 0
        )

        return df.sort_values("optimal", ascending=False).reset_index(drop=True)

    def get_summary_dict(self) -> dict:
        """
        Get a JSON-serializable summary dictionary.

        Useful for export and API responses.
        """
        summary = {
            "total_budget": self.total_budget,
            "expected_response": self.expected_response,
            "response_ci": {
                "low": self.response_ci_low,
                "high": self.response_ci_high,
            },
            "success": self.success,
            "message": self.message,
            "iterations": self.iterations,
            "utility_function": self.utility_function,
            "num_periods": self.num_periods,
            "allocation": self.optimal_allocation,
        }

        if self.current_allocation:
            summary["current_allocation"] = self.current_allocation
            summary["allocation_delta"] = self.allocation_delta
            summary["response_uplift"] = self.response_uplift
            summary["response_uplift_pct"] = self.response_uplift_pct

        return summary


@dataclass
class TargetResult:
    """
    Result from target-based optimization (find budget to achieve target).
    """

    # Target specification
    target_response: float
    budget_range_searched: tuple[float, float]

    # Solution
    required_budget: float
    optimal_allocation: dict[str, float]
    achievable: bool

    # Actual expected response at required budget
    expected_response: float
    response_ci_low: float
    response_ci_high: float

    # Search metadata
    iterations: int
    message: str

    # Full optimization result at the solution
    optimization_result: OptimizationResult | None = None

    def get_summary_dict(self) -> dict:
        """Get a JSON-serializable summary dictionary."""
        return {
            "target_response": self.target_response,
            "required_budget": self.required_budget,
            "achievable": self.achievable,
            "expected_response": self.expected_response,
            "response_ci": {
                "low": self.response_ci_low,
                "high": self.response_ci_high,
            },
            "budget_range_searched": list(self.budget_range_searched),
            "iterations": self.iterations,
            "message": self.message,
            "allocation": self.optimal_allocation,
        }


@dataclass
class ScenarioResult:
    """
    Result from scenario analysis (multiple budget levels).
    """

    # Scenario definition
    budget_scenarios: list[float]

    # Results per scenario
    results: list[OptimizationResult]

    # Aggregated metrics
    efficiency_curve: pd.DataFrame  # budget, expected_response, marginal_response

    def __post_init__(self):
        """Build efficiency curve from results."""
        if self.efficiency_curve is None:
            self._build_efficiency_curve()

    def _build_efficiency_curve(self):
        """Calculate efficiency curve from scenario results."""
        data = []
        for i, result in enumerate(self.results):
            row = {
                "budget": result.total_budget,
                "expected_response": result.expected_response,
                "response_ci_low": result.response_ci_low,
                "response_ci_high": result.response_ci_high,
                "success": result.success,
            }

            # Calculate marginal response (incremental response per incremental budget)
            if i > 0:
                prev = self.results[i - 1]
                budget_delta = result.total_budget - prev.total_budget
                response_delta = result.expected_response - prev.expected_response
                if budget_delta > 0:
                    row["marginal_response"] = response_delta / budget_delta
                else:
                    row["marginal_response"] = 0
            else:
                row["marginal_response"] = (
                    result.expected_response / result.total_budget
                    if result.total_budget > 0
                    else 0
                )

            data.append(row)

        self.efficiency_curve = pd.DataFrame(data)

    def get_best_scenario(self, metric: str = "marginal_response") -> OptimizationResult:
        """
        Get the scenario with highest marginal efficiency.

        Args:
            metric: Which metric to optimize ("marginal_response", "expected_response")

        Returns:
            OptimizationResult for the best scenario
        """
        if metric == "marginal_response":
            best_idx = self.efficiency_curve["marginal_response"].idxmax()
        else:
            best_idx = self.efficiency_curve[metric].idxmax()

        return self.results[best_idx]

    def to_dataframe(self) -> pd.DataFrame:
        """
        Get full comparison DataFrame with allocations per scenario.

        Returns:
            DataFrame with budget, response, and allocation per channel
        """
        rows = []
        for result in self.results:
            row = {
                "budget": result.total_budget,
                "expected_response": result.expected_response,
                "response_ci_low": result.response_ci_low,
                "response_ci_high": result.response_ci_high,
                **{f"alloc_{ch}": val for ch, val in result.optimal_allocation.items()},
            }
            rows.append(row)

        return pd.DataFrame(rows)

    def get_summary_dict(self) -> dict:
        """Get a JSON-serializable summary dictionary."""
        return {
            "budget_scenarios": self.budget_scenarios,
            "efficiency_curve": self.efficiency_curve.to_dict(orient="records"),
            "scenario_count": len(self.results),
            "best_marginal_budget": float(
                self.efficiency_curve.loc[
                    self.efficiency_curve["marginal_response"].idxmax(), "budget"
                ]
            ),
        }


@dataclass
class MultiModelOptimizationResult:
    """
    Result from multi-model budget optimization.

    Contains the optimal allocation that maximizes combined weighted response
    across multiple models with shared channels.
    """

    # Core results
    optimal_allocation: dict[str, float]  # {channel_name: amount}
    total_budget: float

    # Per-model expected responses
    response_by_model: dict[str, float]  # {model_label: expected_response}
    combined_response: float  # weighted sum of all model responses

    # Per-model response CIs
    response_ci_by_model: dict[str, tuple[float, float]]  # {label: (low, high)}

    # Combined weighted response CI
    combined_ci_low: float
    combined_ci_high: float

    # Model weights used
    model_weights: dict[str, float]  # {model_label: weight}

    # Optimization metadata
    success: bool
    message: str
    iterations: int
    num_periods: int = 1

    # Comparison to current (optional)
    current_allocation: dict[str, float] | None = None
    current_response_by_model: dict[str, float] | None = None
    current_combined_response: float | None = None

    @property
    def response_uplift_by_model(self) -> dict[str, float] | None:
        """Response uplift per model vs current allocation."""
        if self.current_response_by_model is None:
            return None
        return {
            label: self.response_by_model[label] - self.current_response_by_model.get(label, 0)
            for label in self.response_by_model
        }

    @property
    def combined_uplift(self) -> float | None:
        """Combined response uplift vs current allocation."""
        if self.current_combined_response is None:
            return None
        return self.combined_response - self.current_combined_response

    @property
    def combined_uplift_pct(self) -> float | None:
        """Combined response uplift percentage."""
        if self.current_combined_response is None or self.current_combined_response == 0:
            return None
        return (self.combined_response - self.current_combined_response) / self.current_combined_response * 100

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert allocation to a DataFrame with per-model response breakdown.

        Returns:
            DataFrame with columns: channel, allocation, pct_of_total,
            plus response columns for each model
        """
        data = {
            "channel": list(self.optimal_allocation.keys()),
            "allocation": list(self.optimal_allocation.values()),
        }

        df = pd.DataFrame(data)
        df["pct_of_total"] = df["allocation"] / self.total_budget * 100

        return df.sort_values("allocation", ascending=False).reset_index(drop=True)

    def get_model_breakdown_df(self) -> pd.DataFrame:
        """
        Get per-model response breakdown.

        Returns:
            DataFrame with model, response, weight, weighted_response columns
        """
        data = []
        for label, response in self.response_by_model.items():
            weight = self.model_weights.get(label, 0)
            data.append({
                "model": label,
                "response": response,
                "weight": weight,
                "weighted_response": response * weight,
                "ci_low": self.response_ci_by_model.get(label, (0, 0))[0],
                "ci_high": self.response_ci_by_model.get(label, (0, 0))[1],
            })

        return pd.DataFrame(data)

    def get_summary_dict(self) -> dict:
        """Get a JSON-serializable summary dictionary."""
        summary = {
            "total_budget": self.total_budget,
            "combined_response": self.combined_response,
            "combined_ci": {
                "low": self.combined_ci_low,
                "high": self.combined_ci_high,
            },
            "response_by_model": self.response_by_model,
            "model_weights": self.model_weights,
            "success": self.success,
            "message": self.message,
            "iterations": self.iterations,
            "num_periods": self.num_periods,
            "allocation": self.optimal_allocation,
        }

        if self.current_combined_response is not None:
            summary["current_combined_response"] = self.current_combined_response
            summary["combined_uplift"] = self.combined_uplift
            summary["combined_uplift_pct"] = self.combined_uplift_pct

        return summary
