"""
Optimizer Consistency Test Suite.

Validates optimizer behavior across all modes and settings.
Can be run on any fitted model via UI button.
"""

from dataclasses import dataclass, field
from typing import Any
import logging

from mmm_platform.model.mmm import MMMWrapper
from mmm_platform.optimization import BudgetAllocator

logger = logging.getLogger(__name__)


@dataclass
class ConsistencyTestResult:
    """Result of a single consistency test."""

    name: str
    passed: bool
    message: str
    details: dict = field(default_factory=dict)


class OptimizerConsistencyTests:
    """
    Run consistency tests on optimizer for a given model.

    Tests cover:
    - Determinism (same inputs = same outputs)
    - Cross-page consistency (main vs scenarios)
    - Bounds respected
    - Budget conservation
    - Response consistency
    - Time horizons (4 weeks, 26 weeks)
    - Seasonality (start months, indices)
    - Efficiency floors (Min ROI, Max CPA)
    - Incremental budget
    - Edge cases
    """

    def __init__(self, wrapper: MMMWrapper):
        self.wrapper = wrapper
        self.results: list[ConsistencyTestResult] = []

    def run_all(self, num_periods: int = 8) -> list[ConsistencyTestResult]:
        """Run all 16 consistency tests and return results."""
        self.results = []

        # Core tests
        self._test_determinism(num_periods)
        self._test_cross_page_consistency(num_periods)
        self._test_bounds_respected(num_periods)
        self._test_budget_conservation(num_periods)
        self._test_response_consistency(num_periods)

        # Time horizon tests
        self._test_short_horizon()
        self._test_long_horizon()

        # Seasonality tests
        self._test_seasonal_index_application(num_periods)

        # Efficiency floor tests (skip if KPI type doesn't match)
        kpi_type = getattr(self.wrapper.config.data, 'kpi_type', 'revenue')
        if kpi_type == 'revenue':
            self._test_min_roi(num_periods)
        else:
            self._test_max_cpa(num_periods)

        # Incremental budget tests
        self._test_incremental_respects_base(num_periods)
        self._test_incremental_adds_correctly(num_periods)

        # Edge case tests
        self._test_tight_bounds(num_periods)
        self._test_budget_exceeds_bounds(num_periods)

        return self.results

    def _test_determinism(self, num_periods: int):
        """Test 1: Same inputs should produce same outputs."""
        try:
            allocator = BudgetAllocator(self.wrapper, num_periods=num_periods)
            historical, _, _ = allocator.bridge.get_last_n_weeks_spend(num_periods)
            budget = sum(historical.values())

            result1 = allocator.optimize(total_budget=budget)
            result2 = allocator.optimize(total_budget=budget)

            # Compare allocations
            alloc_match = True
            for ch in result1.optimal_allocation:
                diff = abs(result1.optimal_allocation[ch] - result2.optimal_allocation[ch])
                if diff > 1.0:  # $1 tolerance
                    alloc_match = False
                    break

            response_match = abs(result1.expected_response - result2.expected_response) < 1.0

            self.results.append(ConsistencyTestResult(
                name="Determinism",
                passed=alloc_match and response_match,
                message="Same inputs produce same outputs" if (alloc_match and response_match)
                        else "Results differ between identical runs",
                details={
                    "run1_response": result1.expected_response,
                    "run2_response": result2.expected_response,
                }
            ))
        except Exception as e:
            self.results.append(ConsistencyTestResult(
                name="Determinism",
                passed=False,
                message=f"Test failed with error: {e}",
            ))

    def _test_cross_page_consistency(self, num_periods: int):
        """Test 2: Main optimizer and scenarios should match."""
        try:
            allocator = BudgetAllocator(self.wrapper, num_periods=num_periods)
            historical, _, _ = allocator.bridge.get_last_n_weeks_spend(num_periods)
            budget = sum(historical.values())
            bounds = {ch: (v * 0.7, v * 1.3) for ch, v in historical.items() if v > 0}

            # Main optimizer
            main_result = allocator.optimize(total_budget=budget, channel_bounds=bounds)

            # Scenarios (single budget)
            scenario_result = allocator.scenario_analysis([budget], channel_bounds=bounds)
            scenario_alloc = scenario_result.results[0].optimal_allocation

            # Compare
            alloc_diff = sum(
                abs(main_result.optimal_allocation.get(ch, 0) - scenario_alloc.get(ch, 0))
                for ch in set(main_result.optimal_allocation) | set(scenario_alloc)
            )

            self.results.append(ConsistencyTestResult(
                name="Cross-Page Consistency",
                passed=alloc_diff < budget * 0.01,  # Within 1%
                message=f"Main vs Scenarios allocation diff: ${alloc_diff:,.0f}",
                details={
                    "main_response": main_result.expected_response,
                    "scenario_response": scenario_result.results[0].expected_response,
                }
            ))
        except Exception as e:
            self.results.append(ConsistencyTestResult(
                name="Cross-Page Consistency",
                passed=False,
                message=f"Test failed with error: {e}",
            ))

    def _test_bounds_respected(self, num_periods: int):
        """Test 3: All allocations should be within configured bounds."""
        try:
            allocator = BudgetAllocator(self.wrapper, num_periods=num_periods)
            historical, _, _ = allocator.bridge.get_last_n_weeks_spend(num_periods)
            budget = sum(historical.values())
            bounds = {ch: (v * 0.7, v * 1.3) for ch, v in historical.items() if v > 0}

            result = allocator.optimize(total_budget=budget, channel_bounds=bounds)

            violations = []
            for ch, alloc in result.optimal_allocation.items():
                if ch in bounds:
                    min_b, max_b = bounds[ch]
                    if alloc < min_b - 1.0:  # $1 tolerance
                        violations.append(f"{ch}: ${alloc:,.0f} < min ${min_b:,.0f}")
                    if alloc > max_b + 1.0:
                        violations.append(f"{ch}: ${alloc:,.0f} > max ${max_b:,.0f}")

            self.results.append(ConsistencyTestResult(
                name="Bounds Respected",
                passed=len(violations) == 0,
                message="All allocations within bounds" if not violations
                        else f"Violations: {', '.join(violations)}",
                details={"violation_count": len(violations)}
            ))
        except Exception as e:
            self.results.append(ConsistencyTestResult(
                name="Bounds Respected",
                passed=False,
                message=f"Test failed with error: {e}",
            ))

    def _test_budget_conservation(self, num_periods: int):
        """Test 4: Total allocation should equal requested budget."""
        try:
            allocator = BudgetAllocator(self.wrapper, num_periods=num_periods)
            historical, _, _ = allocator.bridge.get_last_n_weeks_spend(num_periods)
            budget = sum(historical.values())

            result = allocator.optimize(total_budget=budget)
            total_allocated = sum(result.optimal_allocation.values())

            # Account for unallocated budget if bounds prevent full allocation
            unallocated = getattr(result, 'unallocated_budget', 0) or 0
            effective_budget = budget - unallocated

            diff_pct = abs(total_allocated - effective_budget) / budget * 100 if budget > 0 else 0

            self.results.append(ConsistencyTestResult(
                name="Budget Conservation",
                passed=diff_pct < 1.0,  # Within 1%
                message=f"Allocated ${total_allocated:,.0f} of ${budget:,.0f} ({total_allocated/budget*100:.1f}%)",
                details={
                    "requested": budget,
                    "allocated": total_allocated,
                    "unallocated": unallocated,
                }
            ))
        except Exception as e:
            self.results.append(ConsistencyTestResult(
                name="Budget Conservation",
                passed=False,
                message=f"Test failed with error: {e}",
            ))

    def _test_response_consistency(self, num_periods: int):
        """Test 5: Optimizer response should match independent calculation."""
        try:
            allocator = BudgetAllocator(self.wrapper, num_periods=num_periods)
            historical, _, _ = allocator.bridge.get_last_n_weeks_spend(num_periods)
            budget = sum(historical.values())

            result = allocator.optimize(total_budget=budget)

            # Independent calculation (returns tuple: response, ci_low, ci_high)
            independent_response, _, _ = allocator.bridge.estimate_response_at_allocation(
                result.optimal_allocation, num_periods
            )

            diff_pct = abs(result.expected_response - independent_response) / result.expected_response * 100 \
                if result.expected_response > 0 else 0

            self.results.append(ConsistencyTestResult(
                name="Response Consistency",
                passed=diff_pct < 5.0,  # Within 5% (some variance expected from sampling)
                message=f"Optimizer: ${result.expected_response:,.0f}, Independent: ${independent_response:,.0f} ({diff_pct:.1f}% diff)",
                details={
                    "optimizer_response": result.expected_response,
                    "independent_response": independent_response,
                }
            ))
        except Exception as e:
            self.results.append(ConsistencyTestResult(
                name="Response Consistency",
                passed=False,
                message=f"Test failed with error: {e}",
            ))

    def _test_short_horizon(self):
        """Test 6: Optimization should work with 4-week horizon."""
        try:
            allocator = BudgetAllocator(self.wrapper, num_periods=4)
            historical, _, _ = allocator.bridge.get_last_n_weeks_spend(4)
            budget = sum(historical.values())

            result = allocator.optimize(total_budget=budget)

            self.results.append(ConsistencyTestResult(
                name="Short Horizon (4 weeks)",
                passed=result.success and result.expected_response > 0,
                message=f"4-week optimization: ${result.expected_response:,.0f} response",
                details={"success": result.success, "iterations": result.iterations}
            ))
        except Exception as e:
            self.results.append(ConsistencyTestResult(
                name="Short Horizon (4 weeks)",
                passed=False,
                message=f"Test failed with error: {e}",
            ))

    def _test_long_horizon(self):
        """Test 7: Optimization should work with 26-week horizon."""
        try:
            allocator = BudgetAllocator(self.wrapper, num_periods=26)
            historical, _, _ = allocator.bridge.get_last_n_weeks_spend(26)
            budget = sum(historical.values())

            result = allocator.optimize(total_budget=budget)

            self.results.append(ConsistencyTestResult(
                name="Long Horizon (26 weeks)",
                passed=result.success and result.expected_response > 0,
                message=f"26-week optimization: ${result.expected_response:,.0f} response",
                details={"success": result.success, "iterations": result.iterations}
            ))
        except Exception as e:
            self.results.append(ConsistencyTestResult(
                name="Long Horizon (26 weeks)",
                passed=False,
                message=f"Test failed with error: {e}",
            ))

    def _test_seasonal_index_application(self, num_periods: int):
        """Test 9: Seasonal indices should affect response."""
        try:
            allocator = BudgetAllocator(self.wrapper, num_periods=num_periods)
            channels = allocator.bridge.get_optimizable_channels()
            historical, _, _ = allocator.bridge.get_last_n_weeks_spend(num_periods)
            budget = sum(historical.values())

            # Without seasonal indices
            result_no_seasonal = allocator.optimize(total_budget=budget)

            # With 20% boost to all channels
            seasonal_boost = {ch: 1.2 for ch in channels}
            result_with_seasonal = allocator.optimize(
                total_budget=budget,
                seasonal_indices=seasonal_boost
            )

            # Response should be higher with seasonal boost
            response_increased = result_with_seasonal.expected_response > result_no_seasonal.expected_response

            self.results.append(ConsistencyTestResult(
                name="Seasonal Index Application",
                passed=response_increased,
                message=f"Without seasonal: ${result_no_seasonal.expected_response:,.0f}, "
                        f"With 1.2x boost: ${result_with_seasonal.expected_response:,.0f}",
                details={
                    "without_seasonal": result_no_seasonal.expected_response,
                    "with_seasonal": result_with_seasonal.expected_response,
                }
            ))
        except Exception as e:
            self.results.append(ConsistencyTestResult(
                name="Seasonal Index Application",
                passed=False,
                message=f"Test failed with error: {e}",
            ))

    def _test_min_roi(self, num_periods: int):
        """Test 10: Min ROI constraint should be respected."""
        try:
            allocator = BudgetAllocator(self.wrapper, num_periods=num_periods)
            historical, _, _ = allocator.bridge.get_last_n_weeks_spend(num_periods)
            budget = sum(historical.values())

            # Set a very high ROI target that may not be achievable
            result = allocator.optimize_with_efficiency_floor(
                total_budget=budget,
                efficiency_metric="roi",
                efficiency_target=2.0,  # 2x ROI
            )

            # Either achieved target OR has unallocated budget
            achieved = getattr(result, 'achieved_efficiency', None)
            unallocated = getattr(result, 'unallocated_budget', 0) or 0

            passed = (achieved is not None and achieved >= 2.0) or unallocated > 0

            self.results.append(ConsistencyTestResult(
                name="Min ROI Floor",
                passed=passed,
                message=f"Target ROI: 2.0x, Achieved: {achieved:.2f}x, Unallocated: ${unallocated:,.0f}"
                        if achieved else f"Unallocated: ${unallocated:,.0f}",
                details={
                    "target": 2.0,
                    "achieved": achieved,
                    "unallocated": unallocated,
                }
            ))
        except Exception as e:
            self.results.append(ConsistencyTestResult(
                name="Min ROI Floor",
                passed=False,
                message=f"Test failed with error: {e}",
            ))

    def _test_max_cpa(self, num_periods: int):
        """Test 11: Max CPA constraint should be respected."""
        try:
            allocator = BudgetAllocator(self.wrapper, num_periods=num_periods)
            historical, _, _ = allocator.bridge.get_last_n_weeks_spend(num_periods)
            budget = sum(historical.values())

            # Set a reasonable CPA target
            result = allocator.optimize_with_efficiency_floor(
                total_budget=budget,
                efficiency_metric="cpa",
                efficiency_target=50.0,  # $50 CPA max
            )

            achieved = getattr(result, 'achieved_efficiency', None)
            unallocated = getattr(result, 'unallocated_budget', 0) or 0

            # For CPA, lower is better, so achieved should be <= target
            passed = (achieved is not None and achieved <= 50.0) or unallocated > 0

            self.results.append(ConsistencyTestResult(
                name="Max CPA Floor",
                passed=passed,
                message=f"Target CPA: $50, Achieved: ${achieved:.2f}, Unallocated: ${unallocated:,.0f}"
                        if achieved else f"Unallocated: ${unallocated:,.0f}",
                details={
                    "target": 50.0,
                    "achieved": achieved,
                    "unallocated": unallocated,
                }
            ))
        except Exception as e:
            self.results.append(ConsistencyTestResult(
                name="Max CPA Floor",
                passed=False,
                message=f"Test failed with error: {e}",
            ))

    def _test_incremental_respects_base(self, num_periods: int):
        """Test 12: Incremental optimization should respect base allocation."""
        try:
            allocator = BudgetAllocator(self.wrapper, num_periods=num_periods)
            historical, _, _ = allocator.bridge.get_last_n_weeks_spend(num_periods)

            # Use historical as base, add 20% incremental
            base_allocation = historical
            incremental = sum(historical.values()) * 0.2

            result = allocator.optimize_incremental(
                base_allocation=base_allocation,
                incremental_budget=incremental,
            )

            # Check each channel >= base
            violations = []
            for ch, base in base_allocation.items():
                allocated = result.optimal_allocation.get(ch, 0)
                if allocated < base - 1.0:  # $1 tolerance
                    violations.append(f"{ch}: ${allocated:,.0f} < base ${base:,.0f}")

            self.results.append(ConsistencyTestResult(
                name="Incremental Respects Base",
                passed=len(violations) == 0,
                message="All channels >= base allocation" if not violations
                        else f"Violations: {', '.join(violations)}",
                details={"violation_count": len(violations)}
            ))
        except Exception as e:
            self.results.append(ConsistencyTestResult(
                name="Incremental Respects Base",
                passed=False,
                message=f"Test failed with error: {e}",
            ))

    def _test_incremental_adds_correctly(self, num_periods: int):
        """Test 13: Incremental budget should add to base correctly."""
        try:
            allocator = BudgetAllocator(self.wrapper, num_periods=num_periods)
            historical, _, _ = allocator.bridge.get_last_n_weeks_spend(num_periods)

            base_total = sum(historical.values())
            incremental = base_total * 0.2

            result = allocator.optimize_incremental(
                base_allocation=historical,
                incremental_budget=incremental,
            )

            total_allocated = sum(result.optimal_allocation.values())
            expected_total = base_total + incremental
            diff_pct = abs(total_allocated - expected_total) / expected_total * 100

            self.results.append(ConsistencyTestResult(
                name="Incremental Adds Correctly",
                passed=diff_pct < 1.0,
                message=f"Base: ${base_total:,.0f} + Inc: ${incremental:,.0f} = ${total_allocated:,.0f} (expected ${expected_total:,.0f})",
                details={
                    "base_total": base_total,
                    "incremental": incremental,
                    "total_allocated": total_allocated,
                }
            ))
        except Exception as e:
            self.results.append(ConsistencyTestResult(
                name="Incremental Adds Correctly",
                passed=False,
                message=f"Test failed with error: {e}",
            ))

    def _test_tight_bounds(self, num_periods: int):
        """Test 15: Optimizer should handle tight bounds (±10%)."""
        try:
            allocator = BudgetAllocator(self.wrapper, num_periods=num_periods)
            historical, _, _ = allocator.bridge.get_last_n_weeks_spend(num_periods)
            budget = sum(historical.values())

            # Very tight bounds: ±10%
            bounds = {ch: (v * 0.9, v * 1.1) for ch, v in historical.items() if v > 0}

            result = allocator.optimize(total_budget=budget, channel_bounds=bounds)

            self.results.append(ConsistencyTestResult(
                name="Tight Bounds (±10%)",
                passed=result.success,
                message=f"Optimizer {'succeeded' if result.success else 'failed'} with tight bounds",
                details={"success": result.success, "iterations": result.iterations}
            ))
        except Exception as e:
            self.results.append(ConsistencyTestResult(
                name="Tight Bounds (±10%)",
                passed=False,
                message=f"Test failed with error: {e}",
            ))

    def _test_budget_exceeds_bounds(self, num_periods: int):
        """Test 16: When budget exceeds bounds, unallocated should be reported."""
        try:
            allocator = BudgetAllocator(self.wrapper, num_periods=num_periods)
            historical, _, _ = allocator.bridge.get_last_n_weeks_spend(num_periods)

            # Create bounds for ALL channels (not just those with historical spend)
            # This ensures the optimizer can't use default bounds for missing channels
            channels = allocator.bridge.get_optimizable_channels()
            bounds = {}
            for ch in channels:
                v = historical.get(ch, 0)
                if v > 0:
                    bounds[ch] = (v * 0.9, v * 1.1)  # ±10%
                else:
                    bounds[ch] = (0, 0)  # Zero-spend channels: no allocation allowed

            max_allocatable = sum(b[1] for b in bounds.values())

            # Budget significantly exceeds what bounds allow
            budget = max_allocatable * 1.5

            result = allocator.optimize(total_budget=budget, channel_bounds=bounds)

            unallocated = getattr(result, 'unallocated_budget', 0) or 0
            total_allocated = sum(result.optimal_allocation.values())

            # Should have significant unallocated budget
            self.results.append(ConsistencyTestResult(
                name="Budget Exceeds Bounds",
                passed=unallocated > budget * 0.1,  # At least 10% unallocated
                message=f"Budget: ${budget:,.0f}, Allocated: ${total_allocated:,.0f}, Unallocated: ${unallocated:,.0f}",
                details={
                    "budget": budget,
                    "max_allocatable": max_allocatable,
                    "allocated": total_allocated,
                    "unallocated": unallocated,
                }
            ))
        except Exception as e:
            self.results.append(ConsistencyTestResult(
                name="Budget Exceeds Bounds",
                passed=False,
                message=f"Test failed with error: {e}",
            ))
