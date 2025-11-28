"""
Repository pattern for database operations.
"""

from typing import Optional, List
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import desc
import logging

from .models import Dataset, ModelConfig, ModelRun, ChannelResult, ControlResult
from ..config.schema import ModelConfig as ConfigSchema

logger = logging.getLogger(__name__)


class ModelRepository:
    """
    Repository for model-related database operations.
    """

    def __init__(self, session: Session):
        """
        Initialize repository.

        Parameters
        ----------
        session : Session
            SQLAlchemy session.
        """
        self.session = session

    # =========================================================================
    # Dataset operations
    # =========================================================================

    def create_dataset(
        self,
        name: str,
        filename: str,
        n_rows: int,
        n_columns: int,
        column_names: list,
        date_range_start: Optional[datetime] = None,
        date_range_end: Optional[datetime] = None,
        description: Optional[str] = None,
        storage_path: Optional[str] = None,
        file_size_bytes: Optional[int] = None,
        file_hash: Optional[str] = None,
    ) -> Dataset:
        """Create a new dataset record."""
        dataset = Dataset(
            name=name,
            filename=filename,
            n_rows=n_rows,
            n_columns=n_columns,
            column_names=column_names,
            date_range_start=date_range_start,
            date_range_end=date_range_end,
            description=description,
            storage_path=storage_path,
            file_size_bytes=file_size_bytes,
            file_hash=file_hash,
        )
        self.session.add(dataset)
        self.session.flush()
        logger.info(f"Created dataset: {name} (id={dataset.id})")
        return dataset

    def get_dataset(self, dataset_id: int) -> Optional[Dataset]:
        """Get a dataset by ID."""
        return self.session.query(Dataset).filter(Dataset.id == dataset_id).first()

    def list_datasets(self, limit: int = 100) -> List[Dataset]:
        """List all datasets."""
        return (
            self.session.query(Dataset)
            .order_by(desc(Dataset.created_at))
            .limit(limit)
            .all()
        )

    def delete_dataset(self, dataset_id: int) -> bool:
        """Delete a dataset."""
        dataset = self.get_dataset(dataset_id)
        if dataset:
            self.session.delete(dataset)
            logger.info(f"Deleted dataset: {dataset_id}")
            return True
        return False

    # =========================================================================
    # Config operations
    # =========================================================================

    def create_config(
        self,
        config: ConfigSchema,
        is_template: bool = False,
    ) -> ModelConfig:
        """Create a new config record."""
        db_config = ModelConfig(
            name=config.name,
            description=config.description,
            config_json=config.model_dump(mode="json"),
            n_channels=len(config.channels),
            n_controls=len(config.controls),
            target_column=config.data.target_column,
            is_template=is_template,
        )
        self.session.add(db_config)
        self.session.flush()
        logger.info(f"Created config: {config.name} (id={db_config.id})")
        return db_config

    def get_config(self, config_id: int) -> Optional[ModelConfig]:
        """Get a config by ID."""
        return self.session.query(ModelConfig).filter(ModelConfig.id == config_id).first()

    def get_config_by_name(self, name: str) -> Optional[ModelConfig]:
        """Get a config by name."""
        return self.session.query(ModelConfig).filter(ModelConfig.name == name).first()

    def list_configs(self, include_templates: bool = True, limit: int = 100) -> List[ModelConfig]:
        """List all configs."""
        query = self.session.query(ModelConfig)
        if not include_templates:
            query = query.filter(ModelConfig.is_template == False)
        return query.order_by(desc(ModelConfig.created_at)).limit(limit).all()

    def update_config(self, config_id: int, config: ConfigSchema) -> Optional[ModelConfig]:
        """Update an existing config."""
        db_config = self.get_config(config_id)
        if db_config:
            db_config.name = config.name
            db_config.description = config.description
            db_config.config_json = config.model_dump(mode="json")
            db_config.n_channels = len(config.channels)
            db_config.n_controls = len(config.controls)
            db_config.target_column = config.data.target_column
            logger.info(f"Updated config: {config_id}")
        return db_config

    def delete_config(self, config_id: int) -> bool:
        """Delete a config."""
        db_config = self.get_config(config_id)
        if db_config:
            self.session.delete(db_config)
            logger.info(f"Deleted config: {config_id}")
            return True
        return False

    # =========================================================================
    # Model run operations
    # =========================================================================

    def create_model_run(
        self,
        name: str,
        dataset_id: Optional[int] = None,
        config_id: Optional[int] = None,
        description: Optional[str] = None,
    ) -> ModelRun:
        """Create a new model run record."""
        run = ModelRun(
            name=name,
            dataset_id=dataset_id,
            config_id=config_id,
            description=description,
            status="pending",
        )
        self.session.add(run)
        self.session.flush()
        logger.info(f"Created model run: {name} (id={run.id})")
        return run

    def get_model_run(self, run_id: int) -> Optional[ModelRun]:
        """Get a model run by ID."""
        return self.session.query(ModelRun).filter(ModelRun.id == run_id).first()

    def list_model_runs(
        self,
        status: Optional[str] = None,
        limit: int = 100
    ) -> List[ModelRun]:
        """List model runs."""
        query = self.session.query(ModelRun)
        if status:
            query = query.filter(ModelRun.status == status)
        return query.order_by(desc(ModelRun.created_at)).limit(limit).all()

    def update_model_run_status(
        self,
        run_id: int,
        status: str,
        error_message: Optional[str] = None,
    ) -> Optional[ModelRun]:
        """Update model run status."""
        run = self.get_model_run(run_id)
        if run:
            run.status = status
            if status == "running":
                run.started_at = datetime.utcnow()
            elif status in ("completed", "failed"):
                run.completed_at = datetime.utcnow()
                if run.started_at:
                    run.duration_seconds = (
                        run.completed_at - run.started_at
                    ).total_seconds()
            if error_message:
                run.error_message = error_message
            logger.info(f"Updated run {run_id} status to {status}")
        return run

    def update_model_run_results(
        self,
        run_id: int,
        r_squared: float,
        mape: float,
        rmse: float,
        converged: bool,
        n_divergences: int,
        model_path: Optional[str] = None,
        results_path: Optional[str] = None,
        results_summary: Optional[dict] = None,
    ) -> Optional[ModelRun]:
        """Update model run with results."""
        run = self.get_model_run(run_id)
        if run:
            run.r_squared = r_squared
            run.mape = mape
            run.rmse = rmse
            run.converged = converged
            run.n_divergences = n_divergences
            run.model_path = model_path
            run.results_path = results_path
            run.results_summary = results_summary
            logger.info(f"Updated run {run_id} with results")
        return run

    def delete_model_run(self, run_id: int) -> bool:
        """Delete a model run."""
        run = self.get_model_run(run_id)
        if run:
            self.session.delete(run)
            logger.info(f"Deleted model run: {run_id}")
            return True
        return False

    # =========================================================================
    # Channel/Control results
    # =========================================================================

    def save_channel_results(
        self,
        run_id: int,
        results: List[dict],
    ) -> List[ChannelResult]:
        """Save channel results for a model run."""
        channel_results = []
        for r in results:
            result = ChannelResult(
                model_run_id=run_id,
                channel_name=r.get("channel_name"),
                display_name=r.get("display_name"),
                total_spend=r.get("total_spend"),
                total_contribution=r.get("total_contribution"),
                roi=r.get("roi"),
                beta_mean=r.get("beta_mean"),
                beta_std=r.get("beta_std"),
                beta_hdi_low=r.get("beta_hdi_low"),
                beta_hdi_high=r.get("beta_hdi_high"),
                alpha_mean=r.get("alpha_mean"),
                lam_mean=r.get("lam_mean"),
            )
            self.session.add(result)
            channel_results.append(result)
        self.session.flush()
        return channel_results

    def save_control_results(
        self,
        run_id: int,
        results: List[dict],
    ) -> List[ControlResult]:
        """Save control results for a model run."""
        control_results = []
        for r in results:
            result = ControlResult(
                model_run_id=run_id,
                control_name=r.get("control_name"),
                display_name=r.get("display_name"),
                total_contribution=r.get("total_contribution"),
                expected_sign=r.get("expected_sign"),
                actual_sign=r.get("actual_sign"),
                sign_valid=r.get("sign_valid"),
                gamma_mean=r.get("gamma_mean"),
                gamma_std=r.get("gamma_std"),
            )
            self.session.add(result)
            control_results.append(result)
        self.session.flush()
        return control_results

    def get_channel_results(self, run_id: int) -> List[ChannelResult]:
        """Get channel results for a model run."""
        return (
            self.session.query(ChannelResult)
            .filter(ChannelResult.model_run_id == run_id)
            .all()
        )

    def get_control_results(self, run_id: int) -> List[ControlResult]:
        """Get control results for a model run."""
        return (
            self.session.query(ControlResult)
            .filter(ControlResult.model_run_id == run_id)
            .all()
        )
