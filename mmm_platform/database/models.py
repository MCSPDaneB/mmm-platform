"""
SQLAlchemy ORM models for MMM Platform.
"""

from datetime import datetime
from typing import Optional
from sqlalchemy import (
    Column, Integer, String, Text, DateTime, Float, Boolean,
    ForeignKey, JSON, LargeBinary
)
from sqlalchemy.orm import relationship, declarative_base

Base = declarative_base()


class Dataset(Base):
    """
    Uploaded datasets.
    """
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)

    # File info
    filename = Column(String(255), nullable=False)
    file_size_bytes = Column(Integer, nullable=True)
    file_hash = Column(String(64), nullable=True)  # SHA-256

    # Data info
    n_rows = Column(Integer, nullable=True)
    n_columns = Column(Integer, nullable=True)
    date_range_start = Column(DateTime, nullable=True)
    date_range_end = Column(DateTime, nullable=True)
    column_names = Column(JSON, nullable=True)  # List of column names

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Storage path (for file-based storage)
    storage_path = Column(String(500), nullable=True)

    # Relationships
    model_runs = relationship("ModelRun", back_populates="dataset")

    def __repr__(self):
        return f"<Dataset(id={self.id}, name='{self.name}')>"


class ModelConfig(Base):
    """
    Saved model configurations.
    """
    __tablename__ = "model_configs"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False, unique=True)
    description = Column(Text, nullable=True)

    # Full configuration as JSON
    config_json = Column(JSON, nullable=False)

    # Quick reference fields
    n_channels = Column(Integer, nullable=True)
    n_controls = Column(Integer, nullable=True)
    target_column = Column(String(255), nullable=True)

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_template = Column(Boolean, default=False)

    # Relationships
    model_runs = relationship("ModelRun", back_populates="config")

    def __repr__(self):
        return f"<ModelConfig(id={self.id}, name='{self.name}')>"


class ModelRun(Base):
    """
    Model fitting runs and results.
    """
    __tablename__ = "model_runs"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)

    # Foreign keys
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=True)
    config_id = Column(Integer, ForeignKey("model_configs.id"), nullable=True)

    # Status
    status = Column(
        String(50),
        default="pending"
    )  # pending, running, completed, failed
    error_message = Column(Text, nullable=True)

    # Timing
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    duration_seconds = Column(Float, nullable=True)

    # Fit statistics
    r_squared = Column(Float, nullable=True)
    mape = Column(Float, nullable=True)
    rmse = Column(Float, nullable=True)

    # Convergence
    converged = Column(Boolean, nullable=True)
    n_divergences = Column(Integer, nullable=True)

    # Storage paths
    model_path = Column(String(500), nullable=True)  # Path to saved model
    results_path = Column(String(500), nullable=True)  # Path to results

    # Results summary as JSON (for quick access without loading full model)
    results_summary = Column(JSON, nullable=True)

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    dataset = relationship("Dataset", back_populates="model_runs")
    config = relationship("ModelConfig", back_populates="model_runs")

    def __repr__(self):
        return f"<ModelRun(id={self.id}, name='{self.name}', status='{self.status}')>"


class ChannelResult(Base):
    """
    Per-channel results from a model run.
    """
    __tablename__ = "channel_results"

    id = Column(Integer, primary_key=True, index=True)
    model_run_id = Column(Integer, ForeignKey("model_runs.id"), nullable=False)

    channel_name = Column(String(255), nullable=False)
    display_name = Column(String(255), nullable=True)
    categories = Column(JSON, nullable=True)  # Category values keyed by column name

    # Spend and contribution
    total_spend = Column(Float, nullable=True)
    total_contribution = Column(Float, nullable=True)
    roi = Column(Float, nullable=True)

    # Posterior estimates
    beta_mean = Column(Float, nullable=True)
    beta_std = Column(Float, nullable=True)
    beta_hdi_low = Column(Float, nullable=True)
    beta_hdi_high = Column(Float, nullable=True)

    alpha_mean = Column(Float, nullable=True)  # Adstock
    lam_mean = Column(Float, nullable=True)  # Saturation

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<ChannelResult(channel='{self.channel_name}', roi={self.roi})>"


class ControlResult(Base):
    """
    Per-control variable results from a model run.
    """
    __tablename__ = "control_results"

    id = Column(Integer, primary_key=True, index=True)
    model_run_id = Column(Integer, ForeignKey("model_runs.id"), nullable=False)

    control_name = Column(String(255), nullable=False)
    display_name = Column(String(255), nullable=True)
    categories = Column(JSON, nullable=True)  # Category values keyed by column name

    # Contribution
    total_contribution = Column(Float, nullable=True)
    expected_sign = Column(String(20), nullable=True)
    actual_sign = Column(String(20), nullable=True)
    sign_valid = Column(Boolean, nullable=True)

    # Posterior estimates
    gamma_mean = Column(Float, nullable=True)
    gamma_std = Column(Float, nullable=True)

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<ControlResult(control='{self.control_name}')>"
