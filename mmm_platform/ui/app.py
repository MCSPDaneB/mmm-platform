"""
Main Streamlit application for MMM Platform.
"""

import streamlit as st
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mmm_platform.database import init_db, get_db


def init_session_state():
    """Initialize session state variables."""
    if "initialized" not in st.session_state:
        st.session_state.initialized = True
        st.session_state.current_data = None
        st.session_state.current_config = None
        st.session_state.current_model = None
        st.session_state.model_fitted = False
        st.session_state.active_client = None  # Session-wide client selection


def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="MMM Platform",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Hide the automatic Streamlit pages navigation
    st.markdown(
        """
        <style>
            [data-testid="stSidebarNav"] {
                display: none;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Initialize
    init_session_state()

    # Initialize database
    db = get_db()
    db.create_tables()

    # Sidebar navigation
    st.sidebar.title("📊 MMM Platform")
    st.sidebar.markdown("---")

    # Client selector in sidebar
    from mmm_platform.model.persistence import list_clients
    clients = list_clients()

    if clients:
        client_options = ["All Clients"] + clients

        # Sync widget state FROM active_client before render
        active = st.session_state.get("active_client")
        widget_value = active if active and active in clients else "All Clients"
        st.session_state.sidebar_client_selector = widget_value

        def on_sidebar_client_change():
            val = st.session_state.sidebar_client_selector
            st.session_state.active_client = None if val == "All Clients" else val

        st.sidebar.selectbox(
            "🏢 Client",
            options=client_options,
            key="sidebar_client_selector",
            on_change=on_sidebar_client_change
        )

        st.sidebar.markdown("---")

    # Navigation
    page = st.sidebar.radio(
        "Navigation",
        [
            "🏠 Home",
            "📁 Upload Data",
            "⚙️ Configure Model",
            "🚀 Run Model",
            "📈 Results",
            "📤 Export",
            "📊 Combined Analysis",
            "🔍 Compare Models",
            "💾 Saved Configs & Models",
        ],
    )

    st.sidebar.markdown("---")

    # Status indicators
    st.sidebar.subheader("Status")
    data_status = "✅" if st.session_state.get("current_data") is not None else "⏳"
    config_status = "✅" if st.session_state.get("current_config") is not None else "⏳"
    model_status = "✅" if st.session_state.get("model_fitted", False) else "⏳"

    st.sidebar.write(f"{data_status} Data loaded")
    st.sidebar.write(f"{config_status} Model configured")
    st.sidebar.write(f"{model_status} Model fitted")

    # Route to pages
    if page == "🏠 Home":
        show_home()
    elif page == "📁 Upload Data":
        from mmm_platform.ui.pages import upload_data
        upload_data.show()
    elif page == "⚙️ Configure Model":
        from mmm_platform.ui.pages import configure_model
        configure_model.show()
    elif page == "🚀 Run Model":
        from mmm_platform.ui.pages import run_model
        run_model.show()
    elif page == "📈 Results":
        from mmm_platform.ui.pages import results
        results.show()
    elif page == "📤 Export":
        from mmm_platform.ui.pages import export
        export.show()
    elif page == "📊 Combined Analysis":
        from mmm_platform.ui.pages import combined_analysis
        combined_analysis.show()
    elif page == "🔍 Compare Models":
        from mmm_platform.ui.pages import compare_models
        compare_models.show()
    elif page == "💾 Saved Configs & Models":
        from mmm_platform.ui.pages import saved_models
        saved_models.show()


def show_home():
    """Show the home page."""
    from mmm_platform.model.persistence import list_clients, get_client_configs_dir

    st.title("Marketing Mix Modeling Platform")

    # Client selection section
    st.subheader("🏢 Select Client")
    st.caption("Which client are we working on today?")

    clients = list_clients()
    col1, col2 = st.columns([2, 1])

    with col1:
        client_options = ["-- Select a client --"] + clients if clients else ["-- No clients yet --"]

        # Sync widget state FROM active_client before render
        active = st.session_state.get("active_client")
        if active and active in clients:
            st.session_state.home_client_selector = active
        else:
            st.session_state.home_client_selector = "-- Select a client --"

        def on_home_client_change():
            val = st.session_state.home_client_selector
            if val not in ["-- Select a client --", "-- No clients yet --"]:
                st.session_state.active_client = val
            # Note: Don't set to None here - keep current selection if placeholder chosen

        st.selectbox(
            "Client",
            options=client_options,
            label_visibility="collapsed",
            key="home_client_selector",
            on_change=on_home_client_change
        )

    with col2:
        # Create new client option
        with st.expander("➕ Create New Client"):
            new_client = st.text_input(
                "New client name",
                placeholder="e.g., acme_corp",
                key="home_new_client"
            )
            if st.button("Create", key="home_create_client_btn"):
                if new_client and new_client.strip():
                    new_client = new_client.strip().lower().replace(" ", "_")
                    # Create the client directory
                    get_client_configs_dir(new_client)
                    st.session_state.active_client = new_client
                    st.success(f"Created client: {new_client}")
                    st.rerun()
                else:
                    st.warning("Please enter a client name")

    # Show active client status
    if st.session_state.get("active_client"):
        st.success(f"Working with: **{st.session_state.active_client}**")
    else:
        st.info("No client selected. Select a client above or choose 'All Clients' in the sidebar to view everything.")

    st.markdown("---")

    st.markdown("""
    Welcome to the **MMM Platform** - a comprehensive tool for building and analyzing
    Bayesian Marketing Mix Models using PyMC-Marketing.

    ## Getting Started

    Follow these steps to build your model:

    1. **📁 Upload Data** - Upload your marketing data (CSV format)
    2. **⚙️ Configure Model** - Define channels, controls, and priors
    3. **🚀 Run Model** - Fit the Bayesian model
    4. **📈 Results** - Analyze ROI, contributions, and diagnostics

    ## Features

    - **Bayesian Inference**: Full posterior distributions for all parameters
    - **ROI-Informed Priors**: Calibrate priors based on expected ROI
    - **Adstock & Saturation**: Geometric adstock and logistic saturation transforms
    - **Comprehensive Diagnostics**: Convergence checks, residual analysis
    - **Executive Summary**: Investment recommendations (INCREASE/HOLD/REDUCE)
    - **Marginal ROI Analysis**: Breakeven spend and headroom calculations
    - **Combined Model Analysis**: Multi-outcome optimization across models
    - **Model Comparison**: Side-by-side comparison of fitted models
    - **Export Results**: CSV, JSON, and HTML reports
    """)

    # Quick stats if data is loaded
    if st.session_state.current_data is not None:
        st.markdown("---")
        st.subheader("Current Session")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Data Rows", len(st.session_state.current_data))
        with col2:
            if st.session_state.current_config:
                st.metric("Channels", len(st.session_state.current_config.channels))
        with col3:
            if st.session_state.model_fitted:
                st.metric("Model Status", "Fitted ✅")
            else:
                st.metric("Model Status", "Not fitted")


if __name__ == "__main__":
    main()
