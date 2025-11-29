"""
Client for communicating with the EC2 MMM API.

Used by Streamlit to submit jobs and retrieve results.
"""

import os
import time
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

import httpx
import pandas as pd

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class JobInfo:
    """Information about a model run job."""
    job_id: str
    status: JobStatus
    progress: float
    message: str
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None


@dataclass
class ModelResults:
    """Results from a completed model run."""
    job_id: str
    fit_statistics: Dict[str, Any]
    channel_roi: List[Dict[str, Any]]
    contributions: Dict[str, Any]
    diagnostics: Dict[str, Any]
    inference_data_path: Optional[str] = None


class EC2ModelClient:
    """
    Client for the EC2 MMM Model Runner API.

    Usage:
        client = EC2ModelClient("http://your-ec2-ip:8000")

        # Submit a job
        job_id = client.submit_model_run(
            model_name="My Model",
            data=df,
            channels=[...],
            data_config={...}
        )

        # Wait for completion
        results = client.wait_for_completion(job_id)
    """

    def __init__(self, base_url: Optional[str] = None, timeout: float = 120.0):
        """
        Initialize the EC2 client.

        Parameters
        ----------
        base_url : str, optional
            Base URL of the EC2 API. Defaults to EC2_API_URL env var or localhost.
        timeout : float
            Request timeout in seconds.
        """
        self.base_url = base_url or os.getenv("EC2_API_URL", "http://localhost:8000")
        self.timeout = timeout
        self._client = httpx.Client(base_url=self.base_url, timeout=httpx.Timeout(timeout))

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self._client.close()

    def close(self):
        """Close the client."""
        self._client.close()

    def health_check(self) -> bool:
        """
        Check if the EC2 API is healthy.

        Returns
        -------
        bool
            True if healthy, False otherwise.
        """
        try:
            response = self._client.get("/health")
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            return False

    def submit_model_run(
        self,
        model_name: str,
        data: pd.DataFrame,
        channels: List[Dict[str, Any]],
        data_config: Dict[str, Any],
        controls: List[Dict[str, Any]] = None,
        sampling_config: Dict[str, Any] = None,
    ) -> str:
        """
        Submit a model run to EC2.

        Parameters
        ----------
        model_name : str
            Name for this model run.
        data : pd.DataFrame
            The data to model.
        channels : list[dict]
            Channel configurations.
        data_config : dict
            Data configuration (target_column, date_column, etc.).
        controls : list[dict], optional
            Control variable configurations.
        sampling_config : dict, optional
            Sampling configuration (draws, tune, chains, etc.).

        Returns
        -------
        str
            Job ID for tracking the run.
        """
        # Convert DataFrame to list of dicts
        # Handle datetime serialization
        data_copy = data.copy()
        for col in data_copy.columns:
            if pd.api.types.is_datetime64_any_dtype(data_copy[col]):
                data_copy[col] = data_copy[col].dt.strftime("%Y-%m-%d")

        data_records = data_copy.to_dict(orient="records")

        # Build request
        request_body = {
            "model_name": model_name,
            "data": data_records,
            "channels": channels,
            "controls": controls or [],
            "data_config": data_config,
            "sampling_config": sampling_config or {},
        }

        logger.info(f"Submitting model run '{model_name}' to EC2...")

        response = self._client.post("/run-model", json=request_body)
        response.raise_for_status()

        result = response.json()
        job_id = result["job_id"]

        logger.info(f"Job submitted: {job_id}")
        return job_id

    def get_status(self, job_id: str) -> JobInfo:
        """
        Get the status of a job.

        Parameters
        ----------
        job_id : str
            The job ID.

        Returns
        -------
        JobInfo
            Job status information.
        """
        response = self._client.get(f"/status/{job_id}")
        response.raise_for_status()

        data = response.json()
        return JobInfo(
            job_id=data["job_id"],
            status=JobStatus(data["status"]),
            progress=data["progress"],
            message=data["message"],
            created_at=data["created_at"],
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            error=data.get("error"),
        )

    def get_results(self, job_id: str) -> ModelResults:
        """
        Get the results of a completed job.

        Parameters
        ----------
        job_id : str
            The job ID.

        Returns
        -------
        ModelResults
            The model results.

        Raises
        ------
        httpx.HTTPStatusError
            If the job is not complete or failed.
        """
        response = self._client.get(f"/results/{job_id}")
        response.raise_for_status()

        data = response.json()
        return ModelResults(
            job_id=data["job_id"],
            fit_statistics=data["fit_statistics"],
            channel_roi=data["channel_roi"],
            contributions=data["contributions"],
            diagnostics=data["diagnostics"],
            inference_data_path=data.get("inference_data_path"),
        )

    def wait_for_completion(
        self,
        job_id: str,
        poll_interval: float = 5.0,
        timeout: float = 3600.0,
        progress_callback=None,
    ) -> ModelResults:
        """
        Wait for a job to complete and return results.

        Parameters
        ----------
        job_id : str
            The job ID.
        poll_interval : float
            Seconds between status checks.
        timeout : float
            Maximum seconds to wait.
        progress_callback : callable, optional
            Function to call with (progress, message) on each poll.

        Returns
        -------
        ModelResults
            The model results.

        Raises
        ------
        TimeoutError
            If the job doesn't complete within timeout.
        RuntimeError
            If the job fails.
        """
        start_time = time.time()

        while True:
            elapsed = time.time() - start_time
            if elapsed > timeout:
                raise TimeoutError(f"Job {job_id} did not complete within {timeout}s")

            status = self.get_status(job_id)

            if progress_callback:
                progress_callback(status.progress, status.message)

            if status.status == JobStatus.COMPLETED:
                return self.get_results(job_id)

            if status.status == JobStatus.FAILED:
                raise RuntimeError(f"Job {job_id} failed: {status.error}")

            time.sleep(poll_interval)

    def list_jobs(self) -> List[Dict[str, Any]]:
        """
        List all jobs.

        Returns
        -------
        list[dict]
            List of job summaries.
        """
        response = self._client.get("/jobs")
        response.raise_for_status()
        return response.json()["jobs"]

    def delete_job(self, job_id: str) -> None:
        """
        Delete a job and its results.

        Parameters
        ----------
        job_id : str
            The job ID.
        """
        response = self._client.delete(f"/jobs/{job_id}")
        response.raise_for_status()


# Convenience function
def get_client(base_url: Optional[str] = None) -> EC2ModelClient:
    """
    Get an EC2 client instance.

    Parameters
    ----------
    base_url : str, optional
        EC2 API URL. Uses EC2_API_URL env var if not provided.

    Returns
    -------
    EC2ModelClient
        Client instance.
    """
    return EC2ModelClient(base_url)
