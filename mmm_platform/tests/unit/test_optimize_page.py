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
