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

    # Navigation
    page = st.sidebar.radio(
        "Navigation",
        [
            "🏠 Home",
            "📁 Upload Data",
            "⚙️ Configure Model",
            "🚀 Run Model",
            "📈 Results",
            "💾 Saved Configs & Models",
        ],
    )

    st.sidebar.markdown("---")

    # Demo mode indicator and exit button
    if st.session_state.get("demo_mode", False):
        st.sidebar.warning("**Demo Mode Active**")
        if st.sidebar.button("Exit Demo Mode", type="secondary", use_container_width=True):
            # Clear demo state
            st.session_state.demo_mode = False
            st.session_state.demo = None
            st.session_state.current_data = None
            st.session_state.current_config = None
            st.session_state.current_model = None
            st.session_state.model_fitted = False
            st.rerun()
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
    elif page == "💾 Saved Configs & Models":
        from mmm_platform.ui.pages import saved_models
        saved_models.show()


def load_demo_data():
    """Load demo data into session state."""
    from mmm_platform.analysis.demo import create_demo_scenario

    with st.spinner("Loading demo scenario..."):
        demo = create_demo_scenario()

        # Store demo in session state
        st.session_state.demo_mode = True
        st.session_state.demo = demo
        st.session_state.current_data = demo.df_scaled.reset_index()
        st.session_state.model_fitted = True

        # Create a mock config for display purposes
        st.session_state.current_config = type('MockConfig', (), {
            'name': 'Demo Model',
            'channels': [type('Ch', (), {
                'name': ch,
                'roi_prior_low': 0.5,
                'roi_prior_mid': 1.5,
                'roi_prior_high': 3.0,
                'adstock_type': 'geometric',
            })() for ch in demo.channel_cols],
            'controls': [],
            'data': type('Data', (), {
                'target_column': demo.target_col,
                'date_column': 'date',
                'revenue_scale': demo.revenue_scale,
                'spend_scale': demo.spend_scale,
            })(),
            'sampling': type('Sampling', (), {
                'draws': 1000,
                'tune': 500,
                'chains': 4,
                'target_accept': 0.9,
            })(),
            'get_channel_columns': lambda: demo.channel_cols,
        })()

    st.success("Demo loaded! Go to **Results** to explore all features.")
    st.rerun()


def show_home():
    """Show the home page."""
    st.title("Marketing Mix Modeling Platform")

    # Demo Mode Button - Prominent at the top
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("### Quick Start: Try the Demo")
        st.markdown("Load a demo scenario with mock data to explore all features without running a model.")
        if st.button("Load Demo Data", type="primary", use_container_width=True):
            load_demo_data()

        if st.session_state.get("demo_mode"):
            st.success("Demo mode active! Navigate to **Results** to explore.")

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
    - **Combined Model Analysis**: Multi-outcome optimization (Online + Offline)
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
