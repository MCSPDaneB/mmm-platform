"""
API module for MMM Platform.

- client.py: Client for Streamlit to communicate with EC2 API
- server.py: FastAPI server (only runs on EC2)
"""

from .client import EC2ModelClient, JobStatus, get_client

__all__ = ["EC2ModelClient", "JobStatus", "get_client"]
