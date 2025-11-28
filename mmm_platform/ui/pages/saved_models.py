"""
Saved models page for MMM Platform.
"""

import streamlit as st
from pathlib import Path
from datetime import datetime

from mmm_platform.model.persistence import ModelPersistence


def show():
    """Show the saved models page."""
    st.title("💾 Saved Models")

    # Default save directory
    save_dir = Path("saved_models")

    if not save_dir.exists():
        st.info("No saved models directory found. Models will be saved here after fitting.")
        save_dir.mkdir(exist_ok=True)
        return

    # List saved models
    models = ModelPersistence.list_saved_models(save_dir)

    if not models:
        st.info("No saved models found. Run a model and choose to save it.")
        return

    st.write(f"Found {len(models)} saved model(s)")

    # Display models
    for model in models:
        with st.expander(f"📦 {model.get('config_name', 'Unknown')} - {model.get('created_at', 'Unknown date')[:10]}"):
            col1, col2 = st.columns(2)

            with col1:
                st.write(f"**Created:** {model.get('created_at', 'N/A')}")
                st.write(f"**Fitted:** {model.get('fitted_at', 'N/A')}")
                st.write(f"**Duration:** {model.get('fit_duration_seconds', 0):.1f}s")

            with col2:
                st.write(f"**Channels:** {model.get('n_channels', 'N/A')}")
                st.write(f"**Controls:** {model.get('n_controls', 'N/A')}")
                st.write(f"**Path:** {model.get('path', 'N/A')}")

            # Actions
            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button("📂 Load Model", key=f"load_{model.get('path')}"):
                    load_model(model.get("path"))

            with col2:
                if st.button("📊 View Results", key=f"view_{model.get('path')}"):
                    st.info("Navigate to Results page after loading")

            with col3:
                if st.button("🗑️ Delete", key=f"delete_{model.get('path')}"):
                    delete_model(model.get("path"))

    # Upload saved model
    st.markdown("---")
    st.subheader("Import Model")

    st.info("To import a model, place the model folder in the 'saved_models' directory.")


def load_model(path: str):
    """Load a saved model."""
    try:
        from mmm_platform.model.mmm import MMMWrapper

        wrapper = ModelPersistence.load(path, MMMWrapper)

        st.session_state.current_model = wrapper
        st.session_state.current_config = wrapper.config
        st.session_state.current_data = wrapper.df_raw
        st.session_state.model_fitted = wrapper.idata is not None

        st.success(f"Model loaded successfully!")
        st.rerun()

    except Exception as e:
        st.error(f"Error loading model: {e}")


def delete_model(path: str):
    """Delete a saved model."""
    import shutil

    try:
        shutil.rmtree(path)
        st.success("Model deleted")
        st.rerun()
    except Exception as e:
        st.error(f"Error deleting model: {e}")
