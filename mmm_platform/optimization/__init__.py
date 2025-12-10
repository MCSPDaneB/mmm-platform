"""
Budget Optimization Module for MMM Platform.

This module provides budget optimization capabilities using PyMC-Marketing's
BudgetOptimizer. It is designed to be encapsulated and reusable across
different tools and interfaces.

Key components:
- BudgetAllocator: Main interface for budget optimization
- OptimizationBridge: Translates MMMWrapper to PyMC-Marketing optimizer
- ConstraintBuilder: Factory for common optimization constraints
- TimeDistribution: Time-phased budget allocation patterns
- TargetOptimizer: Find budget to achieve target response
- OptimizationResult: Result dataclass with analysis methods

Usage:
    from mmm_platform.optimization import BudgetAllocator

    allocator = BudgetAllocator(wrapper, num_periods=8)
    result = allocator.optimize(total_budget=100000)
    print(result.optimal_allocation)

Target-based optimization:
    from mmm_platform.optimization import BudgetAllocator, TargetOptimizer

    allocator = BudgetAllocator(wrapper, num_periods=8)
    target_opt = TargetOptimizer(allocator)
    result = target_opt.find_budget_for_target(target_response=500000)
    print(f"Need ${result.required_budget:,.0f} to achieve ${result.target_response:,.0f}")
"""

from mmm_platform.optimization.results import (
    OptimizationResult,
    TargetResult,
    ScenarioResult,
)
from mmm_platform.optimization.bridge import OptimizationBridge, UTILITY_FUNCTIONS
from mmm_platform.optimization.allocator import BudgetAllocator, create_allocator_from_session
from mmm_platform.optimization.constraints import ConstraintBuilder, build_bounds_from_constraints
from mmm_platform.optimization.time_distribution import TimeDistribution, validate_time_distribution
from mmm_platform.optimization.scenarios import (
    TargetOptimizer,
    ResponseCurveAnalyzer,
    compute_efficiency_frontier,
)
from mmm_platform.optimization.seasonality import SeasonalIndexCalculator

__all__ = [
    # Main classes
    "BudgetAllocator",
    "OptimizationBridge",
    "ConstraintBuilder",
    "TimeDistribution",
    "TargetOptimizer",
    "ResponseCurveAnalyzer",
    "SeasonalIndexCalculator",
    # Result types
    "OptimizationResult",
    "TargetResult",
    "ScenarioResult",
    # Utility functions
    "create_allocator_from_session",
    "build_bounds_from_constraints",
    "validate_time_distribution",
    "compute_efficiency_frontier",
    "UTILITY_FUNCTIONS",
]
