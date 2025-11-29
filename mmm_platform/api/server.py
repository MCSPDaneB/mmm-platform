"""
FastAPI server for MMM model execution on EC2.

Provides endpoints for:
- Starting model runs
- Checking job status
- Retrieving results
"""

import os
import uuid
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
from enum import Enum
import json
import traceback

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Pydantic Models for API
# ============================================================================

class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ChannelConfig(BaseModel):
    name: str
    roi_prior_low: float = 0.5
    roi_prior_mid: float = 1.5
    roi_prior_high: float = 3.0
    adstock_type: str = "medium"
    adstock_max_lag: Optional[int] = 8
    saturation_type: Optional[str] = "logistic"


class ControlConfig(BaseModel):
    name: str
    expected_sign: str = "positive"


class DataConfig(BaseModel):
    target_column: str
    date_column: str
    spend_scale: float = 1.0
    revenue_scale: float = 1.0


class SamplingConfig(BaseModel):
    draws: int = 1000
    tune: int = 500
    chains: int = 4
    target_accept: float = 0.9
    sampler: str = "nutpie"


class ModelRunRequest(BaseModel):
    """Request to start a model run."""
    model_name: str = Field(..., description="Name for this model run")
    data: List[Dict[str, Any]] = Field(..., description="Data as list of row dicts")
    channels: List[ChannelConfig]
    controls: List[ControlConfig] = []
    data_config: DataConfig
    sampling_config: SamplingConfig = SamplingConfig()


class JobStatusResponse(BaseModel):
    """Response for job status queries."""
    job_id: str
    status: JobStatus
    progress: float = 0.0
    message: str = ""
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None


class ModelResultResponse(BaseModel):
    """Response containing model results."""
    job_id: str
    status: JobStatus
    fit_statistics: Optional[Dict[str, Any]] = None
    channel_roi: Optional[List[Dict[str, Any]]] = None
    contributions: Optional[Dict[str, Any]] = None
    diagnostics: Optional[Dict[str, Any]] = None
    inference_data_path: Optional[str] = None


# ============================================================================
# Job Storage (in-memory for now, could be Redis/DB)
# ============================================================================

class JobStore:
    """Simple in-memory job storage."""

    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.results_dir = Path("/tmp/mmm_results")
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def create_job(self, job_id: str, request: ModelRunRequest) -> Dict:
        job = {
            "job_id": job_id,
            "status": JobStatus.PENDING,
            "progress": 0.0,
            "message": "Job created",
            "created_at": datetime.utcnow().isoformat(),
            "started_at": None,
            "completed_at": None,
            "error": None,
            "request": request.model_dump(),
            "result": None,
        }
        self.jobs[job_id] = job
        return job

    def get_job(self, job_id: str) -> Optional[Dict]:
        return self.jobs.get(job_id)

    def update_job(self, job_id: str, **kwargs):
        if job_id in self.jobs:
            self.jobs[job_id].update(kwargs)

    def list_jobs(self) -> List[Dict]:
        return list(self.jobs.values())


# Global job store
job_store = JobStore()

# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="MMM Model Runner API",
    description="API for running Marketing Mix Models on EC2",
    version="1.0.0",
)

# CORS middleware for Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Model Running Logic
# ============================================================================

async def run_model_task(job_id: str, request: ModelRunRequest):
    """Background task to run the MMM model."""
    try:
        job_store.update_job(
            job_id,
            status=JobStatus.RUNNING,
            started_at=datetime.utcnow().isoformat(),
            message="Starting model run...",
            progress=0.05
        )

        # Import here to avoid slow startup
        from mmm_platform.model.mmm import MMMWrapper
        from mmm_platform.config.schema import (
            ModelConfig, ChannelConfig as SchemaChannelConfig,
            ControlConfig as SchemaControlConfig, DataConfig as SchemaDataConfig,
            SamplingConfig as SchemaSamplingConfig
        )

        logger.info(f"Job {job_id}: Loading data...")
        job_store.update_job(job_id, message="Loading data...", progress=0.1)

        # Convert data to DataFrame
        df = pd.DataFrame(request.data)

        # Parse dates
        df[request.data_config.date_column] = pd.to_datetime(
            df[request.data_config.date_column]
        )

        logger.info(f"Job {job_id}: Data loaded - {len(df)} rows")
        job_store.update_job(job_id, message=f"Data loaded: {len(df)} rows", progress=0.15)

        # Build config
        logger.info(f"Job {job_id}: Building model config...")
        job_store.update_job(job_id, message="Building model config...", progress=0.2)

        channels = [
            SchemaChannelConfig(
                name=ch.name,
                roi_prior_low=ch.roi_prior_low,
                roi_prior_mid=ch.roi_prior_mid,
                roi_prior_high=ch.roi_prior_high,
                adstock_type=ch.adstock_type,
                adstock_max_lag=ch.adstock_max_lag,
                saturation_type=ch.saturation_type,
            )
            for ch in request.channels
        ]

        controls = [
            SchemaControlConfig(name=ctrl.name, expected_sign=ctrl.expected_sign)
            for ctrl in request.controls
        ]

        config = ModelConfig(
            name=request.model_name,
            channels=channels,
            controls=controls,
            data=SchemaDataConfig(
                target_column=request.data_config.target_column,
                date_column=request.data_config.date_column,
                spend_scale=request.data_config.spend_scale,
                revenue_scale=request.data_config.revenue_scale,
            ),
            sampling=SchemaSamplingConfig(
                draws=request.sampling_config.draws,
                tune=request.sampling_config.tune,
                chains=request.sampling_config.chains,
                target_accept=request.sampling_config.target_accept,
                sampler=request.sampling_config.sampler,
            ),
        )

        # Create wrapper and fit
        logger.info(f"Job {job_id}: Creating model wrapper...")
        job_store.update_job(job_id, message="Creating model...", progress=0.25)

        wrapper = MMMWrapper(config)
        wrapper.load_data(df)

        logger.info(f"Job {job_id}: Starting model fitting (this may take a while)...")
        job_store.update_job(
            job_id,
            message="Fitting model (MCMC sampling)...",
            progress=0.3
        )

        # Fit the model
        wrapper.fit()

        logger.info(f"Job {job_id}: Model fitting complete!")
        job_store.update_job(job_id, message="Model fitted, computing results...", progress=0.85)

        # Get results
        fit_stats = wrapper.get_fit_statistics()
        contributions = wrapper.get_contributions()

        # Get channel ROI
        from mmm_platform.analysis.contributions import ContributionAnalyzer
        contrib_analyzer = ContributionAnalyzer.from_mmm_wrapper(wrapper)
        channel_roi = contrib_analyzer.get_channel_roi()

        # Save inference data
        results_path = job_store.results_dir / f"{job_id}"
        results_path.mkdir(parents=True, exist_ok=True)

        idata_path = results_path / "inference_data.nc"
        wrapper.idata.to_netcdf(str(idata_path))

        # Save contributions
        contributions.to_csv(results_path / "contributions.csv")

        logger.info(f"Job {job_id}: Results saved to {results_path}")

        # Prepare result
        result = {
            "fit_statistics": fit_stats,
            "channel_roi": channel_roi.to_dict(orient="records"),
            "contributions": {
                "total_by_channel": contributions.sum().to_dict(),
            },
            "diagnostics": {
                "r2": fit_stats.get("r2"),
                "mape": fit_stats.get("mape"),
            },
            "inference_data_path": str(idata_path),
        }

        job_store.update_job(
            job_id,
            status=JobStatus.COMPLETED,
            completed_at=datetime.utcnow().isoformat(),
            message="Model completed successfully!",
            progress=1.0,
            result=result
        )

        logger.info(f"Job {job_id}: Completed successfully!")

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        logger.error(f"Job {job_id} failed: {error_msg}")
        job_store.update_job(
            job_id,
            status=JobStatus.FAILED,
            completed_at=datetime.utcnow().isoformat(),
            message="Model run failed",
            error=error_msg
        )


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "MMM Model Runner",
        "version": "1.0.0"
    }


@app.get("/health")
async def health():
    """Detailed health check."""
    return {
        "status": "healthy",
        "jobs_in_memory": len(job_store.jobs),
        "timestamp": datetime.utcnow().isoformat()
    }


@app.post("/run-model", response_model=JobStatusResponse)
async def run_model(request: ModelRunRequest, background_tasks: BackgroundTasks):
    """
    Start a new model run.

    Returns a job ID that can be used to check status and retrieve results.
    """
    job_id = str(uuid.uuid4())

    logger.info(f"Creating job {job_id} for model '{request.model_name}'")

    # Create job
    job = job_store.create_job(job_id, request)

    # Start background task
    background_tasks.add_task(run_model_task, job_id, request)

    return JobStatusResponse(
        job_id=job_id,
        status=JobStatus.PENDING,
        progress=0.0,
        message="Job created, starting model run...",
        created_at=job["created_at"]
    )


@app.get("/status/{job_id}", response_model=JobStatusResponse)
async def get_status(job_id: str):
    """Get the status of a model run job."""
    job = job_store.get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    return JobStatusResponse(
        job_id=job["job_id"],
        status=job["status"],
        progress=job["progress"],
        message=job["message"],
        created_at=job["created_at"],
        started_at=job["started_at"],
        completed_at=job["completed_at"],
        error=job["error"]
    )


@app.get("/results/{job_id}", response_model=ModelResultResponse)
async def get_results(job_id: str):
    """Get the results of a completed model run."""
    job = job_store.get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    if job["status"] == JobStatus.PENDING:
        raise HTTPException(status_code=400, detail="Job has not started yet")

    if job["status"] == JobStatus.RUNNING:
        raise HTTPException(status_code=400, detail="Job is still running")

    if job["status"] == JobStatus.FAILED:
        raise HTTPException(
            status_code=500,
            detail=f"Job failed: {job['error']}"
        )

    result = job.get("result", {})

    return ModelResultResponse(
        job_id=job_id,
        status=job["status"],
        fit_statistics=result.get("fit_statistics"),
        channel_roi=result.get("channel_roi"),
        contributions=result.get("contributions"),
        diagnostics=result.get("diagnostics"),
        inference_data_path=result.get("inference_data_path")
    )


@app.get("/jobs")
async def list_jobs():
    """List all jobs."""
    jobs = job_store.list_jobs()
    return {
        "jobs": [
            {
                "job_id": j["job_id"],
                "status": j["status"],
                "model_name": j["request"]["model_name"],
                "created_at": j["created_at"],
                "completed_at": j["completed_at"],
            }
            for j in jobs
        ]
    }


@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a job and its results."""
    job = job_store.get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    if job["status"] == JobStatus.RUNNING:
        raise HTTPException(status_code=400, detail="Cannot delete running job")

    # Remove from store
    del job_store.jobs[job_id]

    # Remove results files
    results_path = job_store.results_dir / job_id
    if results_path.exists():
        import shutil
        shutil.rmtree(results_path)

    return {"message": f"Job {job_id} deleted"}


# ============================================================================
# Run with: uvicorn mmm_platform.api.server:app --host 0.0.0.0 --port 8000
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
