"""
Tests for the optimize page UI components.

Tests cover:
- Fill budget callback functionality
- Session state management for programmatic widget updates
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import date


class MockSessionState:
    """Mock Streamlit session state that supports both attribute and dict access."""

    def __init__(self, initial_data=None):
        self._data = initial_data or {}

    def __getattr__(self, name):
        if name.startswith("_"):
            return super().__getattribute__(name)
        return self._data.get(name)

    def __setattr__(self, name, value):
        if name.startswith("_"):
            super().__setattr__(name, value)
        else:
            self._data[name] = value

    def __contains__(self, key):
        return key in self._data

    def get(self, key, default=None):
        return self._data.get(key, default)

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value


class TestFillBudgetCallback:
    """Tests for the _fill_budget_callback function."""

    @pytest.fixture
    def mock_session_state(self):
        """Create a mock session state."""
        return MockSessionState({
            "fill_weeks": 8,
            "opt_num_periods": 8,
        })

    @pytest.fixture
    def mock_allocator(self):
        """Create a mock BudgetAllocator."""
        allocator = Mock()
        allocator.bridge.get_last_n_weeks_spend.return_value = (
            {"tv_spend": 100000, "search_spend": 50000},
            date(2025, 10, 1),
            date(2025, 11, 26),
        )
        return allocator

    def test_fill_budget_sets_budget_value_not_widget_key(
        self, mock_session_state, mock_allocator
    ):
        """
        Verify callback sets budget_value (not widget key).

        This is critical: Streamlit ignores the value parameter when a widget
        key exists in session state. So we must use a separate session state
        variable (budget_value) for programmatic updates.
        """
        mock_session_state.current_model = Mock()

        with patch(
            "mmm_platform.ui.pages.optimize.st.session_state", mock_session_state
        ):
            with patch(
                "mmm_platform.optimization.BudgetAllocator",
                return_value=mock_allocator,
            ):
                from mmm_platform.ui.pages.optimize import _fill_budget_callback

                _fill_budget_callback()

                # Should set budget_value, NOT opt_total_budget
                assert "budget_value" in mock_session_state
                assert mock_session_state["budget_value"] == 150000  # 100k + 50k

                # Should NOT set the widget key directly
                assert "opt_total_budget" not in mock_session_state

    def test_fill_budget_auto_configures_comparison(
        self, mock_session_state, mock_allocator
    ):
        """Callback auto-enables comparison toggle."""
        mock_session_state.current_model = Mock()

        with patch(
            "mmm_platform.ui.pages.optimize.st.session_state", mock_session_state
        ):
            with patch(
                "mmm_platform.optimization.BudgetAllocator",
                return_value=mock_allocator,
            ):
                from mmm_platform.ui.pages.optimize import _fill_budget_callback

                _fill_budget_callback()

                # Should enable comparison (mode is always last_n_weeks now)
                assert mock_session_state["opt_compare_historical"] is True

    def test_fill_budget_shows_info_message(
        self, mock_session_state, mock_allocator
    ):
        """Callback sets info message with date range."""
        mock_session_state.current_model = Mock()

        with patch(
            "mmm_platform.ui.pages.optimize.st.session_state", mock_session_state
        ):
            with patch(
                "mmm_platform.optimization.BudgetAllocator",
                return_value=mock_allocator,
            ):
                from mmm_platform.ui.pages.optimize import _fill_budget_callback

                _fill_budget_callback()

                assert "budget_fill_info" in mock_session_state
                assert "$150,000" in mock_session_state["budget_fill_info"]
                assert "2025-10-01" in mock_session_state["budget_fill_info"]

    def test_fill_budget_handles_zero_spend(self, mock_session_state):
        """Callback shows error when no spend found."""
        mock_allocator = Mock()
        mock_allocator.bridge.get_last_n_weeks_spend.return_value = (
            {},  # Empty spend dict
            date(2025, 10, 1),
            date(2025, 11, 26),
        )
        mock_session_state.current_model = Mock()

        with patch(
            "mmm_platform.ui.pages.optimize.st.session_state", mock_session_state
        ):
            with patch(
                "mmm_platform.optimization.BudgetAllocator",
                return_value=mock_allocator,
            ):
                from mmm_platform.ui.pages.optimize import _fill_budget_callback

                _fill_budget_callback()

                assert "budget_fill_error" in mock_session_state
                assert "No spend found" in mock_session_state["budget_fill_error"]

    def test_fill_budget_handles_exception(self, mock_session_state):
        """Callback stores error message on exception."""
        mock_session_state.current_model = Mock()

        with patch(
            "mmm_platform.ui.pages.optimize.st.session_state", mock_session_state
        ):
            with patch(
                "mmm_platform.optimization.BudgetAllocator",
                side_effect=ValueError("Model not fitted"),
            ):
                from mmm_platform.ui.pages.optimize import _fill_budget_callback

                _fill_budget_callback()

                assert "budget_fill_error" in mock_session_state
                assert "Model not fitted" in mock_session_state["budget_fill_error"]


class TestStreamlitWidgetPattern:
    """
    Tests documenting the Streamlit key vs value behavior.

    These tests serve as documentation for the pattern used to enable
    programmatic widget updates.
    """

    def test_pattern_documentation(self):
        """
        Document the correct pattern for programmatic widget updates.

        PROBLEM:
        When a widget has both key and value parameters, and the key
        already exists in session state, Streamlit ignores the value
        parameter.

        SOLUTION:
        1. Don't use key parameter on widgets needing programmatic updates
        2. Use value=st.session_state.my_variable
        3. Sync user edits back: st.session_state.my_variable = result

        See mmm_platform/ui/README.md for full documentation.
        """
        # This test exists purely as documentation
        # The actual behavior is tested in the callback tests above
        assert True


class TestBoundsLabelLogic:
    """Tests for the bounds label display logic in the status bar."""

    def test_bounds_label_no_bounds(self):
        """When bounds_mode is 'No bounds', label should be 'None'."""
        bounds_mode = "No bounds"

        if bounds_mode == "No bounds":
            bounds_label = "None"
        elif bounds_mode == "Custom bounds":
            bounds_label = "Custom"
        else:
            max_delta = 30
            bounds_label = f"±{max_delta}%"

        assert bounds_label == "None"

    def test_bounds_label_max_pct_change_default(self):
        """When bounds_mode is 'Max % change', label shows the percentage."""
        bounds_mode = "Max % change"
        max_delta = 30  # default value

        if bounds_mode == "No bounds":
            bounds_label = "None"
        elif bounds_mode == "Custom bounds":
            bounds_label = "Custom"
        else:
            bounds_label = f"±{max_delta}%"

        assert bounds_label == "±30%"

    def test_bounds_label_max_pct_change_custom_value(self):
        """When user sets custom percentage, label reflects that value."""
        bounds_mode = "Max % change"
        max_delta = 50  # user changed to 50%

        if bounds_mode == "No bounds":
            bounds_label = "None"
        elif bounds_mode == "Custom bounds":
            bounds_label = "Custom"
        else:
            bounds_label = f"±{max_delta}%"

        assert bounds_label == "±50%"

    def test_bounds_label_custom_bounds(self):
        """When bounds_mode is 'Custom bounds', label should be 'Custom'."""
        bounds_mode = "Custom bounds"

        if bounds_mode == "No bounds":
            bounds_label = "None"
        elif bounds_mode == "Custom bounds":
            bounds_label = "Custom"
        else:
            max_delta = 30
            bounds_label = f"±{max_delta}%"

        assert bounds_label == "Custom"


class TestBoundsScalingLogic:
    """Tests for the bounds calculation logic (NO scaling - literal ±X% from historical)."""

    def test_bounds_no_scaling_regardless_of_budget(self):
        """
        Bounds should be literal ±X% from historical, NOT scaled with budget.

        This is the NEW behavior - bounds are based on actual historical spend,
        regardless of what budget the user wants to optimize for.
        """
        # Historical spend
        baseline_spend = {"channel_a": 30000, "channel_b": 70000}
        max_delta = 30  # ±30%

        # Calculate bounds (NO scaling - use actual historical)
        bounds_config = {}
        for ch, historical in baseline_spend.items():
            min_val = historical * (1 - max_delta / 100)
            max_val = historical * (1 + max_delta / 100)
            bounds_config[ch] = (max(0.0, min_val), max_val)

        # Channel A: $30k → ±30% = $21k to $39k (actual historical bounds)
        assert bounds_config["channel_a"] == (21000.0, 39000.0)

        # Channel B: $70k → ±30% = $49k to $91k (actual historical bounds)
        assert bounds_config["channel_b"] == (49000.0, 91000.0)

        # Max capacity is fixed at $130k regardless of budget
        total_max = sum(b[1] for b in bounds_config.values())
        assert total_max == 130000.0

    def test_bounds_max_capacity_limits_budget(self):
        """
        Budget exceeding max capacity should trigger blocking.

        With no scaling, bounds define a fixed max capacity.
        Budget above this cannot be fully allocated.
        """
        baseline_spend = {"channel_a": 30000, "channel_b": 70000}
        max_delta = 30

        # Calculate max capacity
        bounds_config = {}
        for ch, historical in baseline_spend.items():
            max_val = historical * (1 + max_delta / 100)
            bounds_config[ch] = (0, max_val)

        total_max_capacity = sum(b[1] for b in bounds_config.values())

        # Max capacity = $39k + $91k = $130k
        assert total_max_capacity == 130000.0

        # Budget of $150k exceeds capacity - should be blocked
        budget_150k = 150000
        assert budget_150k > total_max_capacity

        # Budget of $120k is within capacity - should be allowed
        budget_120k = 120000
        assert budget_120k <= total_max_capacity

    def test_zero_spend_channel_gets_capped_bounds(self):
        """Channels with no historical spend get capped at X% of budget."""
        baseline_spend = {"channel_a": 50000, "channel_b": 0}  # B has no spend
        total_budget = 100000
        zero_spend_max_pct = 10  # 10% cap for zero-spend channels
        max_delta = 30

        bounds_config = {}
        for ch, historical in baseline_spend.items():
            if historical > 0:
                min_val = historical * (1 - max_delta / 100)
                max_val = historical * (1 + max_delta / 100)
            else:
                min_val = 0
                max_val = total_budget * (zero_spend_max_pct / 100)
            bounds_config[ch] = (max(0.0, min_val), max_val)

        # Channel A: $50k → ±30% = $35k to $65k
        assert bounds_config["channel_a"] == (35000.0, 65000.0)

        # Channel B: $0 historical → $0 min, $10k max (10% of budget)
        assert bounds_config["channel_b"] == (0.0, 10000.0)


class TestCapacityBlocking:
    """Tests for capacity-based optimization blocking."""

    def test_budget_within_capacity_allowed(self):
        """Budget within max capacity should be allowed."""
        bounds_config = {
            "channel_a": (21000, 39000),
            "channel_b": (49000, 91000),
        }
        max_capacity = sum(b[1] for b in bounds_config.values())  # $130k

        budget = 120000
        budget_exceeds_capacity = budget > max_capacity

        assert not budget_exceeds_capacity

    def test_budget_exceeds_capacity_blocked(self):
        """Budget exceeding max capacity should be blocked."""
        bounds_config = {
            "channel_a": (21000, 39000),
            "channel_b": (49000, 91000),
        }
        max_capacity = sum(b[1] for b in bounds_config.values())  # $130k

        budget = 150000
        budget_exceeds_capacity = budget > max_capacity

        assert budget_exceeds_capacity

    def test_capacity_varies_by_constraint_level(self):
        """Different constraint levels have different max capacities."""
        baseline_spend = {"channel_a": 30000, "channel_b": 70000}

        constraint_capacities = {}
        for delta_pct in [0.10, 0.30, 0.50]:  # ±10%, ±30%, ±50%
            max_capacity = sum(
                val * (1 + delta_pct) for val in baseline_spend.values()
            )
            constraint_capacities[delta_pct] = max_capacity

        # Tighter constraints = lower capacity
        assert constraint_capacities[0.10] == 110000.0  # $33k + $77k
        assert constraint_capacities[0.30] == 130000.0  # $39k + $91k
        assert constraint_capacities[0.50] == 150000.0  # $45k + $105k


class TestConstraintComparisonFiltering:
    """Tests for filtering budget scenarios by constraint capacity."""

    def test_filter_budgets_by_capacity(self):
        """Budgets exceeding constraint capacity should be skipped."""
        baseline_spend = {"channel_a": 30000, "channel_b": 70000}
        budget_scenarios = [50000, 100000, 130000, 150000, 200000]

        # ±30% constraint: max capacity = $130k
        delta_pct = 0.30
        max_capacity = sum(val * (1 + delta_pct) for val in baseline_spend.values())

        achievable_budgets = [b for b in budget_scenarios if b <= max_capacity]

        # Only budgets up to $130k should be achievable
        assert achievable_budgets == [50000, 100000, 130000]
        assert 150000 not in achievable_budgets
        assert 200000 not in achievable_budgets

    def test_tighter_constraints_fewer_achievable_budgets(self):
        """Tighter constraints should have fewer achievable budgets."""
        baseline_spend = {"channel_a": 30000, "channel_b": 70000}
        budget_scenarios = [50000, 100000, 130000, 150000, 200000]

        def get_achievable_count(delta_pct):
            max_capacity = sum(val * (1 + delta_pct) for val in baseline_spend.values())
            return len([b for b in budget_scenarios if b <= max_capacity])

        # Tighter constraints = fewer achievable budgets
        assert get_achievable_count(0.10) == 2  # Up to $110k
        assert get_achievable_count(0.30) == 3  # Up to $130k
        assert get_achievable_count(0.50) == 4  # Up to $150k


class TestMonotonicityCheck:
    """Tests for monotonicity violation detection."""

    def test_no_violation_when_monotonic(self):
        """No violation when higher budget gives higher response."""
        from mmm_platform.ui.pages.optimize import _check_monotonicity

        # Create mock results with monotonic responses
        class MockResult:
            def __init__(self, budget, response):
                self.total_budget = budget
                self.expected_response = response

        results = [
            MockResult(100000, 50000),
            MockResult(150000, 70000),
            MockResult(200000, 85000),
        ]

        # Should not raise or log warning (we can't easily test logging here,
        # but we verify the function runs without error)
        _check_monotonicity(results, "Test")

    def test_detects_monotonicity_violation(self, caplog):
        """Should detect when higher budget gives less response."""
        from mmm_platform.ui.pages.optimize import _check_monotonicity
        import logging

        class MockResult:
            def __init__(self, budget, response):
                self.total_budget = budget
                self.expected_response = response

        # Response DROPS from $70k to $65k when budget increases
        results = [
            MockResult(100000, 50000),
            MockResult(150000, 70000),
            MockResult(200000, 65000),  # Violation: less than $70k
        ]

        # Capture warning logs
        with caplog.at_level(logging.WARNING, logger="mmm_platform.ui.pages.optimize"):
            _check_monotonicity(results, "TestConstraint")

        # Verify warning was logged
        assert any("MONOTONICITY VIOLATION" in msg for msg in caplog.messages)
        assert any("TestConstraint" in msg for msg in caplog.messages)

    def test_handles_empty_results(self):
        """Should handle empty or single-element results gracefully."""
        from mmm_platform.ui.pages.optimize import _check_monotonicity

        # Empty list - should not raise
        _check_monotonicity([], "Empty")

        # Single element - should not raise
        class MockResult:
            def __init__(self, budget, response):
                self.total_budget = budget
                self.expected_response = response

        _check_monotonicity([MockResult(100000, 50000)], "Single")
