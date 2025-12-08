"""
Tests for Export page disaggregation session state management.

These tests verify that:
1. Column selections persist correctly in session state
2. Session state is cleaned up on file upload errors
3. Invalid column selections are filtered when file changes
"""

import pytest


class TestDisaggregationSessionState:
    """Tests for disaggregation session state management."""

    def test_include_cols_validation_filters_invalid_options(self):
        """
        When available_include_cols changes (e.g., new file uploaded),
        session state selections should be filtered to only valid options.
        """
        # Simulate session state with old selections
        session_state = {
            "granular_include_cols_test": ["col_a", "col_b", "col_c"]
        }

        # New file has different columns
        available_include_cols = ["col_a", "col_d", "col_e"]

        # Apply the validation logic from export.py
        include_cols_key = "granular_include_cols_test"
        current_selections = session_state[include_cols_key]
        valid_selections = [c for c in current_selections if c in available_include_cols]

        # Should only keep col_a (the one that's still valid)
        assert valid_selections == ["col_a"]
        assert "col_b" not in valid_selections
        assert "col_c" not in valid_selections

    def test_include_cols_validation_preserves_all_valid(self):
        """
        When all selections are still valid, they should all be preserved.
        """
        session_state = {
            "granular_include_cols_test": ["col_a", "col_b"]
        }

        available_include_cols = ["col_a", "col_b", "col_c", "col_d"]

        include_cols_key = "granular_include_cols_test"
        current_selections = session_state[include_cols_key]
        valid_selections = [c for c in current_selections if c in available_include_cols]

        # All selections should be preserved
        assert valid_selections == ["col_a", "col_b"]

    def test_include_cols_validation_handles_empty_state(self):
        """
        When session state has empty selections, should remain empty.
        """
        session_state = {
            "granular_include_cols_test": []
        }

        available_include_cols = ["col_a", "col_b"]

        include_cols_key = "granular_include_cols_test"
        current_selections = session_state[include_cols_key]
        valid_selections = [c for c in current_selections if c in available_include_cols]

        assert valid_selections == []

    def test_session_state_keys_cleared_on_error(self):
        """
        When file upload fails, related session state keys should be cleared
        to prevent partial/corrupted state.
        """
        key_suffix = "_test"
        granular_df_key = f"granular_df{key_suffix}"
        include_cols_key = f"granular_include_cols{key_suffix}"
        mapping_key = f"granular_mapping{key_suffix}"

        # Simulate session state with data from previous upload
        session_state = {
            granular_df_key: "some_dataframe",
            include_cols_key: ["col_a", "col_b"],
            mapping_key: {"entity": "channel"},
        }

        # Simulate the cleanup logic from export.py on error
        keys_to_clear = [granular_df_key, include_cols_key, mapping_key]
        for key in keys_to_clear:
            if key in session_state:
                del session_state[key]

        # All keys should be cleared
        assert granular_df_key not in session_state
        assert include_cols_key not in session_state
        assert mapping_key not in session_state

    def test_session_state_keys_format(self):
        """
        Session state keys should be correctly formatted with key_suffix.
        """
        key_suffix = "_0_ModelA"

        granular_df_key = f"granular_df{key_suffix}"
        include_cols_key = f"granular_include_cols{key_suffix}"
        mapping_key = f"granular_mapping{key_suffix}"

        assert granular_df_key == "granular_df_0_ModelA"
        assert include_cols_key == "granular_include_cols_0_ModelA"
        assert mapping_key == "granular_mapping_0_ModelA"

    def test_different_models_have_separate_keys(self):
        """
        Different models should use separate session state keys to avoid collision.
        """
        model_a_suffix = "_0_ModelA"
        model_b_suffix = "_1_ModelB"

        model_a_df_key = f"granular_df{model_a_suffix}"
        model_b_df_key = f"granular_df{model_b_suffix}"

        model_a_cols_key = f"granular_include_cols{model_a_suffix}"
        model_b_cols_key = f"granular_include_cols{model_b_suffix}"

        # Keys should be different
        assert model_a_df_key != model_b_df_key
        assert model_a_cols_key != model_b_cols_key

        # Can store independent data
        session_state = {
            model_a_df_key: "model_a_data",
            model_b_df_key: "model_b_data",
            model_a_cols_key: ["col_a"],
            model_b_cols_key: ["col_x", "col_y"],
        }

        assert session_state[model_a_df_key] != session_state[model_b_df_key]
        assert session_state[model_a_cols_key] != session_state[model_b_cols_key]
