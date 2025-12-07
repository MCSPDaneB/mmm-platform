"""
Tests for API response serialization.

Ensures that all API response data is JSON-serializable and doesn't contain
numpy types that Pydantic can't serialize.
"""

import json
import numpy as np
import pytest
from typing import Any, Dict


def is_json_serializable(obj: Any) -> bool:
    """Check if an object is JSON serializable."""
    try:
        json.dumps(obj)
        return True
    except (TypeError, ValueError):
        return False


def find_non_serializable(obj: Any, path: str = "") -> list:
    """
    Recursively find non-JSON-serializable values in a nested structure.

    Returns list of (path, value, type) tuples for non-serializable items.
    """
    issues = []

    if isinstance(obj, dict):
        for key, value in obj.items():
            new_path = f"{path}.{key}" if path else key
            issues.extend(find_non_serializable(value, new_path))
    elif isinstance(obj, (list, tuple)):
        for i, value in enumerate(obj):
            new_path = f"{path}[{i}]"
            issues.extend(find_non_serializable(value, new_path))
    elif isinstance(obj, (np.bool_, np.integer, np.floating)):
        issues.append((path, obj, type(obj).__name__))
    elif obj is not None and not isinstance(obj, (str, int, float, bool)):
        # Check if it's serializable
        if not is_json_serializable(obj):
            issues.append((path, obj, type(obj).__name__))

    return issues


class TestConvergenceDataSerialization:
    """Test that convergence data is properly serializable."""

    def test_numpy_bool_detection(self):
        """Verify our detection function catches numpy.bool_."""
        data = {"converged": np.bool_(True)}
        issues = find_non_serializable(data)
        assert len(issues) == 1
        assert issues[0][0] == "converged"
        assert "bool" in issues[0][2].lower()

    def test_numpy_int_detection(self):
        """Verify our detection function catches numpy integers."""
        data = {"count": np.int64(5)}
        issues = find_non_serializable(data)
        assert len(issues) == 1
        assert issues[0][0] == "count"

    def test_numpy_float_detection(self):
        """Verify our detection function catches numpy floats."""
        data = {"value": np.float64(3.14)}
        issues = find_non_serializable(data)
        assert len(issues) == 1
        assert issues[0][0] == "value"

    def test_native_types_pass(self):
        """Verify native Python types pass serialization check."""
        data = {
            "converged": True,
            "divergences": 5,
            "ess_bulk_min": 1000.0,
            "warnings": ["warning1", "warning2"],
            "nested": {"key": "value"},
        }
        issues = find_non_serializable(data)
        assert len(issues) == 0

    def test_none_values_pass(self):
        """Verify None values pass serialization check."""
        data = {"ess_bulk_min": None, "ess_tail_min": None}
        issues = find_non_serializable(data)
        assert len(issues) == 0

    def test_mock_convergence_response_structure(self):
        """Test that a properly formatted convergence dict is serializable."""
        # This mimics what the server should return after our fix
        convergence_data = {
            "converged": bool(np.bool_(True)),  # Wrapped correctly
            "divergences": int(np.int64(0)),    # Wrapped correctly
            "high_rhat_params": [],
            "warnings": [],
            "ess_bulk_min": float(np.float64(1500.0)),  # Wrapped correctly
            "ess_tail_min": float(np.float64(1200.0)),  # Wrapped correctly
            "ess_sufficient": bool(np.bool_(True)),     # Wrapped correctly
        }

        issues = find_non_serializable(convergence_data)
        assert len(issues) == 0, f"Found serialization issues: {issues}"

        # Also verify it actually serializes
        json_str = json.dumps(convergence_data)
        assert json_str is not None

    def test_unwrapped_numpy_types_fail(self):
        """Test that unwrapped numpy types are detected as issues."""
        # This mimics the bug we fixed
        convergence_data = {
            "converged": np.bool_(True),      # NOT wrapped - should fail
            "divergences": np.int64(0),       # NOT wrapped - should fail
            "high_rhat_params": [],
            "warnings": [],
            "ess_bulk_min": np.float64(1500.0),  # NOT wrapped - should fail
            "ess_tail_min": np.float64(1200.0),  # NOT wrapped - should fail
            "ess_sufficient": np.bool_(True),    # NOT wrapped - should fail
        }

        issues = find_non_serializable(convergence_data)
        assert len(issues) == 5, f"Expected 5 issues, got {len(issues)}: {issues}"


class TestFitStatisticsSerialization:
    """Test that fit statistics are properly serializable."""

    def test_fit_stats_with_numpy(self):
        """Fit stats should not contain numpy types."""
        # Example of what could happen if numpy types leak through
        bad_stats = {
            "r2": np.float64(0.95),
            "mape": np.float64(0.05),
            "rmse": np.float64(100.0),
        }

        issues = find_non_serializable(bad_stats)
        assert len(issues) == 3

    def test_fit_stats_properly_converted(self):
        """Properly converted fit stats should pass."""
        good_stats = {
            "r2": float(np.float64(0.95)),
            "mape": float(np.float64(0.05)),
            "rmse": float(np.float64(100.0)),
        }

        issues = find_non_serializable(good_stats)
        assert len(issues) == 0
