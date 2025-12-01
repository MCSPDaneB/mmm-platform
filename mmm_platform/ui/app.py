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
    st.title("Marketing Mix Modeling Platform")

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
