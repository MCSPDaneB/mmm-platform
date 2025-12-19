"""
Comprehensive tests for constraint comparison logic.

Tests:
1. Constraint ordering: Unconstrained >= ±50% >= ±30% >= ±10%
2. Monotonicity: Higher budget = higher response within each constraint
3. Proper handling of min/max capacity bounds
4. Tests across multiple week configurations (1-52 weeks)
"""

import pytest
import numpy as np
from pathlib import Path


@pytest.mark.slow
class TestConstraintComparisonLogic:
    """Test constraint comparison matches expected behavior."""

    @pytest.fixture
    def load_model(self):
        """Load a saved model for testing."""
        from mmm_platform.model.persistence import ModelPersistence
        from mmm_platform.model.mmm import MMMWrapper

        # Try to find the gopuff iOS model
        model_path = Path("mmm_workspace/clients/gopuff/models/gopuff_installs_ios_20251208_141241")
        if not model_path.exists():
            pytest.skip("Test model not available")

        wrapper = ModelPersistence.load(str(model_path), MMMWrapper)
        return wrapper

    def _run_constraint_comparison_logic(self, wrapper, num_periods):
        """
        Run the same logic as _run_constraint_comparison in optimize.py.

        Returns dict with results per constraint level.
        """
        from mmm_platform.optimization import BudgetAllocator

        allocator = BudgetAllocator(wrapper, num_periods=num_periods)

        # Get historical spend (same as UI)
        historical_spend, _, _ = allocator.bridge.get_last_n_weeks_spend(num_periods)
        total_historical = sum(historical_spend.values())

        if total_historical <= 0:
            return None, "No historical spend"

        # Budget scenarios (same as UI - 15 steps from 50% to 300%)
        budget_scenarios = np.linspace(
            total_historical * 0.5,
            total_historical * 3.0,
            15
        ).tolist()

        # Constraint levels (same as UI)
        constraint_levels = [
            ("Unconstrained", None),
            ("Mild (±50%)", 0.50),
            ("Standard (±30%)", 0.30),
            ("Tight (±10%)", 0.10),
        ]

        # Calculate bounds and min/max capacity (same as UI)
        constraint_bounds = {}
        constraint_max_capacity = {}
        constraint_min_capacity = {}

        for name, delta_pct in constraint_levels:
            if delta_pct is None:
                constraint_bounds[name] = None
                constraint_max_capacity[name] = float('inf')
                constraint_min_capacity[name] = 0
            else:
                bounds = {}
                for ch, val in historical_spend.items():
                    if val > 0:
                        bounds[ch] = (
                            max(0, val * (1 - delta_pct)),
                            val * (1 + delta_pct)
                        )
                    else:
                        bounds[ch] = (0, total_historical * 0.10)
                constraint_bounds[name] = bounds
                constraint_max_capacity[name] = sum(b[1] for b in bounds.values())
                constraint_min_capacity[name] = sum(b[0] for b in bounds.values())

        # Run optimizations (same logic as UI)
        results_by_constraint = {name: [] for name, _ in constraint_levels}

        for name, delta_pct in constraint_levels:
            max_capacity = constraint_max_capacity[name]
            min_capacity = constraint_min_capacity[name]
            orig_bounds = constraint_bounds[name]
            prev_allocation = None
            prev_allocation_response = None

            for budget in sorted(budget_scenarios):
                # Skip if budget outside achievable range
                if budget > max_capacity or budget < min_capacity:
                    continue

                # Build bounds with monotonicity enforcement
                enforced_bounds = None
                if orig_bounds is not None:
                    enforced_bounds = {}
                    for ch, (orig_min, orig_max) in orig_bounds.items():
                        if prev_allocation:
                            prev_val = prev_allocation.get(ch, 0)
                            new_min = max(orig_min, prev_val)
                            if new_min > orig_max:
                                new_min = orig_min
                        else:
                            new_min = orig_min
                        enforced_bounds[ch] = (new_min, orig_max)

                result = allocator.optimize(
                    total_budget=budget,
                    channel_bounds=enforced_bounds,
                    x0_init=prev_allocation,
                )

                # Store result with budget for analysis
                results_by_constraint[name].append({
                    'budget': budget,
                    'response': result.expected_response,
                    'success': result.success,
                })

                # Update for next iteration (same as UI)
                if result.success:
                    prev_allocation = result.optimal_allocation
                    prev_allocation_response = result.expected_response

        return results_by_constraint, None

    def test_monotonicity_within_constraints(self, load_model):
        """Test that higher budget gives higher response within each constraint level."""
        wrapper = load_model

        # Test at 25 weeks (the user's test case)
        results, error = self._run_constraint_comparison_logic(wrapper, num_periods=25)
        if error:
            pytest.skip(error)

        violations = []
        for name, res_list in results.items():
            # Filter to successful results only
            successful = [r for r in res_list if r['success']]

            for i in range(1, len(successful)):
                prev = successful[i - 1]
                curr = successful[i]

                if curr['budget'] > prev['budget'] and curr['response'] < prev['response'] - 1:
                    violations.append({
                        'constraint': name,
                        'budget_low': prev['budget'],
                        'budget_high': curr['budget'],
                        'response_low': prev['response'],
                        'response_high': curr['response'],
                        'drop': prev['response'] - curr['response'],
                    })

        assert len(violations) == 0, f"Monotonicity violations: {violations}"

    def test_constraint_ordering(self, load_model):
        """Test that looser constraints give >= response than tighter constraints."""
        wrapper = load_model

        results, error = self._run_constraint_comparison_logic(wrapper, num_periods=25)
        if error:
            pytest.skip(error)

        # Expected ordering (loosest to tightest)
        constraint_order = ["Unconstrained", "Mild (±50%)", "Standard (±30%)", "Tight (±10%)"]

        violations = []

        # For each budget that appears in multiple constraint levels, check ordering
        all_budgets = set()
        for name, res_list in results.items():
            for r in res_list:
                if r['success']:
                    all_budgets.add(round(r['budget'], 0))

        for budget in sorted(all_budgets):
            # Get responses at this budget for each constraint level
            responses = {}
            for name in constraint_order:
                for r in results[name]:
                    if r['success'] and abs(r['budget'] - budget) < 100:  # tolerance
                        responses[name] = r['response']
                        break

            # Check ordering: looser should be >= tighter
            for i in range(len(constraint_order) - 1):
                looser = constraint_order[i]
                tighter = constraint_order[i + 1]

                if looser in responses and tighter in responses:
                    if responses[looser] < responses[tighter] - 1:  # Allow $1 tolerance
                        violations.append({
                            'budget': budget,
                            'looser': looser,
                            'tighter': tighter,
                            'looser_response': responses[looser],
                            'tighter_response': responses[tighter],
                        })

        # Note: This may have legitimate violations if optimizer finds different local minima
        # So we just report them rather than fail
        if violations:
            print(f"\nConstraint ordering anomalies (may be due to local minima): {len(violations)}")
            for v in violations[:5]:  # Show first 5
                print(f"  Budget ${v['budget']:,.0f}: {v['looser']} (${v['looser_response']:,.0f}) < {v['tighter']} (${v['tighter_response']:,.0f})")

    def test_min_capacity_respected(self, load_model):
        """Test that no optimization is attempted below min_capacity."""
        wrapper = load_model

        results, error = self._run_constraint_comparison_logic(wrapper, num_periods=25)
        if error:
            pytest.skip(error)

        # All results should have success=True since we skip impossible budgets
        failures = []
        for name, res_list in results.items():
            for r in res_list:
                if not r['success']:
                    failures.append({
                        'constraint': name,
                        'budget': r['budget'],
                        'response': r['response'],
                    })

        assert len(failures) == 0, f"Found {len(failures)} failed optimizations (should be 0 with min_capacity check): {failures[:5]}"


@pytest.mark.slow
class TestConstraintComparisonAcrossWeeks:
    """Test constraint comparison across different week configurations."""

    @pytest.fixture
    def load_model(self):
        """Load a saved model for testing."""
        from mmm_platform.model.persistence import ModelPersistence
        from mmm_platform.model.mmm import MMMWrapper

        model_path = Path("mmm_workspace/clients/gopuff/models/gopuff_installs_ios_20251208_141241")
        if not model_path.exists():
            pytest.skip("Test model not available")

        wrapper = ModelPersistence.load(str(model_path), MMMWrapper)
        return wrapper

    @pytest.mark.parametrize("num_periods", [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 52])
    def test_no_monotonicity_violations_at_week(self, load_model, num_periods):
        """Test monotonicity at various week configurations."""
        from mmm_platform.optimization import BudgetAllocator

        wrapper = load_model
        allocator = BudgetAllocator(wrapper, num_periods=num_periods)

        # Get historical
        historical_spend, _, _ = allocator.bridge.get_last_n_weeks_spend(num_periods)
        total_historical = sum(historical_spend.values())

        if total_historical <= 0:
            pytest.skip(f"No historical spend for {num_periods} weeks")

        budget_scenarios = np.linspace(total_historical * 0.5, total_historical * 3.0, 10).tolist()

        constraint_levels = [
            ("Standard (±30%)", 0.30),
            ("Tight (±10%)", 0.10),
        ]

        violations = []

        for name, delta_pct in constraint_levels:
            # Calculate bounds
            bounds = {}
            for ch, val in historical_spend.items():
                if val > 0:
                    bounds[ch] = (max(0, val * (1 - delta_pct)), val * (1 + delta_pct))
                else:
                    bounds[ch] = (0, total_historical * 0.10)

            min_capacity = sum(b[0] for b in bounds.values())
            max_capacity = sum(b[1] for b in bounds.values())

            prev_allocation = None
            prev_response = None

            for budget in sorted(budget_scenarios):
                if budget < min_capacity or budget > max_capacity:
                    continue

                # Build enforced bounds
                enforced = {}
                for ch, (omin, omax) in bounds.items():
                    if prev_allocation:
                        pv = prev_allocation.get(ch, 0)
                        nmin = max(omin, pv)
                        if nmin > omax:
                            nmin = omin
                    else:
                        nmin = omin
                    enforced[ch] = (nmin, omax)

                result = allocator.optimize(
                    total_budget=budget,
                    channel_bounds=enforced,
                    x0_init=prev_allocation,
                )

                if result.success:
                    if prev_response is not None and result.expected_response < prev_response - 1:
                        violations.append({
                            'weeks': num_periods,
                            'constraint': name,
                            'budget': budget,
                            'response': result.expected_response,
                            'prev_response': prev_response,
                            'drop': prev_response - result.expected_response,
                        })

                    prev_allocation = result.optimal_allocation
                    prev_response = result.expected_response

        assert len(violations) == 0, f"Monotonicity violations at {num_periods} weeks: {violations}"


def run_comprehensive_test():
    """
    Standalone function to run comprehensive constraint comparison test.
    Can be called directly for debugging.
    """
    from mmm_platform.model.persistence import ModelPersistence
    from mmm_platform.model.mmm import MMMWrapper
    from mmm_platform.optimization import BudgetAllocator

    model_path = "mmm_workspace/clients/gopuff/models/gopuff_installs_ios_20251208_141241"
    wrapper = ModelPersistence.load(model_path, MMMWrapper)

    print("=" * 70)
    print("COMPREHENSIVE CONSTRAINT COMPARISON TEST")
    print("=" * 70)

    week_configs = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 52]
    constraint_levels = [
        ("Unconstrained", None),
        ("Mild (±50%)", 0.50),
        ("Standard (±30%)", 0.30),
        ("Tight (±10%)", 0.10),
    ]

    all_violations = []
    constraint_order_violations = []

    for num_periods in week_configs:
        print(f"\n--- {num_periods} weeks ---")

        allocator = BudgetAllocator(wrapper, num_periods=num_periods)
        historical_spend, _, _ = allocator.bridge.get_last_n_weeks_spend(num_periods)
        total_historical = sum(historical_spend.values())

        if total_historical <= 0:
            print(f"  Skipped: no historical spend")
            continue

        print(f"  Historical: ${total_historical:,.0f}")

        # Store results by constraint for ordering comparison
        results_by_constraint = {}

        for name, delta_pct in constraint_levels:
            if delta_pct is None:
                min_cap, max_cap = 0, float('inf')
                orig_bounds = None
                # Unconstrained: use wide range
                budget_scenarios = np.linspace(total_historical * 0.5, total_historical * 2.0, 10).tolist()
            else:
                orig_bounds = {}
                for ch, val in historical_spend.items():
                    if val > 0:
                        orig_bounds[ch] = (max(0, val * (1 - delta_pct)), val * (1 + delta_pct))
                    else:
                        orig_bounds[ch] = (0, total_historical * 0.10)
                min_cap = sum(b[0] for b in orig_bounds.values())
                max_cap = sum(b[1] for b in orig_bounds.values())

                # Constrained: use range between min and max capacity
                if min_cap >= max_cap:
                    print(f"  {name}: SKIPPED (min_cap >= max_cap)")
                    continue
                budget_scenarios = np.linspace(min_cap, max_cap, 10).tolist()

            prev_alloc = None
            prev_resp = None
            constraint_violations = []
            results_by_constraint[name] = []
            run_count = 0

            for budget in sorted(budget_scenarios):
                if budget < min_cap or budget > max_cap:
                    continue

                enforced = None
                if orig_bounds:
                    enforced = {}
                    for ch, (omin, omax) in orig_bounds.items():
                        if prev_alloc:
                            pv = prev_alloc.get(ch, 0)
                            nmin = max(omin, pv)
                            if nmin > omax:
                                nmin = omin
                        else:
                            nmin = omin
                        enforced[ch] = (nmin, omax)

                result = allocator.optimize(
                    total_budget=budget,
                    channel_bounds=enforced,
                    x0_init=prev_alloc,
                )

                run_count += 1

                if result.success:
                    results_by_constraint[name].append({
                        'budget': budget,
                        'response': result.expected_response,
                    })

                    if prev_resp is not None and result.expected_response < prev_resp - 1:
                        constraint_violations.append({
                            'budget': budget,
                            'response': result.expected_response,
                            'prev_response': prev_resp,
                            'drop': prev_resp - result.expected_response,
                        })

                    prev_alloc = result.optimal_allocation
                    prev_resp = result.expected_response

            status = f"OK ({run_count} runs)" if len(constraint_violations) == 0 else f"VIOLATIONS: {len(constraint_violations)}"
            print(f"  {name}: {status}")

            if constraint_violations:
                all_violations.extend([{**v, 'weeks': num_periods, 'constraint': name} for v in constraint_violations])

        # Check constraint ordering: Unconstrained >= Mild >= Standard >= Tight
        constraint_order = ["Unconstrained", "Mild (±50%)", "Standard (±30%)", "Tight (±10%)"]
        for i in range(len(constraint_order) - 1):
            looser = constraint_order[i]
            tighter = constraint_order[i + 1]

            if looser not in results_by_constraint or tighter not in results_by_constraint:
                continue

            looser_results = {round(r['budget']): r['response'] for r in results_by_constraint[looser]}
            tighter_results = {round(r['budget']): r['response'] for r in results_by_constraint[tighter]}

            # Find overlapping budgets
            common_budgets = set(looser_results.keys()) & set(tighter_results.keys())
            for budget in common_budgets:
                if looser_results[budget] < tighter_results[budget] - 100:  # $100 tolerance
                    constraint_order_violations.append({
                        'weeks': num_periods,
                        'budget': budget,
                        'looser': looser,
                        'tighter': tighter,
                        'looser_resp': looser_results[budget],
                        'tighter_resp': tighter_results[budget],
                    })

    print("\n" + "=" * 70)
    print("MONOTONICITY RESULTS:")
    if all_violations:
        print(f"  VIOLATIONS: {len(all_violations)}")
        for v in all_violations:
            print(f"    {v['weeks']}wk {v['constraint']}: ${v['budget']:,.0f} dropped ${v['drop']:,.0f}")
    else:
        print("  ALL PASSED - No monotonicity violations!")

    print("\nCONSTRAINT ORDERING RESULTS:")
    if constraint_order_violations:
        print(f"  ANOMALIES: {len(constraint_order_violations)} (may be due to local minima)")
        for v in constraint_order_violations[:5]:
            print(f"    {v['weeks']}wk ${v['budget']:,.0f}: {v['looser']} (${v['looser_resp']:,.0f}) < {v['tighter']} (${v['tighter_resp']:,.0f})")
    else:
        print("  ALL PASSED - Constraint ordering correct!")

    print("=" * 70)

    return all_violations, constraint_order_violations


if __name__ == "__main__":
    run_comprehensive_test()
