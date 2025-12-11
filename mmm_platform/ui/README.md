# MMM Platform UI Development Guide

## Streamlit Patterns

### Programmatic Widget Value Updates

When you need to programmatically update a widget's displayed value (e.g., via a button callback), you must understand how Streamlit's `key` and `value` parameters interact.

#### The Problem

When a widget has both `key` and `value` parameters, and the key already exists in session state, **Streamlit ignores the `value` parameter** and uses the session-state-bound value instead.

```python
# BROKEN: Streamlit ignores value when key exists in session state
st.session_state.my_value = 500  # Set programmatically
st.number_input("Amount", value=st.session_state.my_value, key="my_widget")
# Widget shows whatever is in st.session_state.my_widget, NOT 500
```

#### The Solution

For widgets that need programmatic updates, do NOT use the `key` parameter:

```python
# Initialize storage
if "budget_value" not in st.session_state:
    st.session_state.budget_value = 100000

# Widget without key - properly uses value parameter
result = st.number_input(
    "Total Budget ($)",
    min_value=1000,
    value=st.session_state.budget_value,  # This will be used!
)

# Sync user edits back to storage
st.session_state.budget_value = result
```

#### With Button Callbacks

```python
def fill_callback():
    """Callback sets the session state variable."""
    st.session_state.budget_value = calculated_value

# Button triggers callback
st.button("Fill", on_click=fill_callback)

# Widget reads from session state (no key parameter)
result = st.number_input("Budget", value=st.session_state.budget_value)
st.session_state.budget_value = result
```

### When to Use `key` vs No Key

| Use Case | Use `key`? | Notes |
|----------|-----------|-------|
| Simple input, no programmatic updates | Yes | Standard usage |
| Need to read value elsewhere via session state | Yes | Access via `st.session_state.key` |
| Need programmatic updates to displayed value | No | Use `value` param + manual sync |
| Callback needs to modify displayed value | No | Callbacks can't modify widget keys |

### Example: Quick Fill Budget

See `optimize.py` for a working example of this pattern with the "Quick Fill" budget button.
