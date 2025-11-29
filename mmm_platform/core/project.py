"""
Project save/load functionality for MMM Platform.

Saves and loads complete project state including:
- Data (CSV)
- Configuration (JSON)
- Session settings
"""

import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import zipfile
import tempfile
import shutil

import pandas as pd

logger = logging.getLogger(__name__)


class ProjectManager:
    """
    Manages saving and loading of complete MMM projects.

    A project includes:
    - data.csv: The uploaded data
    - config.json: Model configuration
    - metadata.json: Project metadata (created date, etc.)
    """

    PROJECTS_DIR = Path("projects")

    def __init__(self, projects_dir: Optional[Path] = None):
        """Initialize ProjectManager."""
        self.projects_dir = projects_dir or self.PROJECTS_DIR
        self.projects_dir.mkdir(parents=True, exist_ok=True)

    def save_project(
        self,
        name: str,
        data: pd.DataFrame,
        config: Any,
        session_state: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Save a complete project.

        Parameters
        ----------
        name : str
            Project name (used for filename).
        data : pd.DataFrame
            The data to save.
        config : ModelConfig
            The model configuration.
        session_state : dict, optional
            Additional session state to save.

        Returns
        -------
        Path
            Path to the saved project file.
        """
        # Sanitize name for filename
        safe_name = "".join(c if c.isalnum() or c in "._- " else "_" for c in name)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        project_name = f"{safe_name}_{timestamp}"

        # Create temp directory for project files
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Save data
            data_path = tmpdir / "data.csv"
            data.to_csv(data_path, index=False)
            logger.info(f"Saved data: {len(data)} rows")

            # Save config
            config_path = tmpdir / "config.json"
            if hasattr(config, 'model_dump'):
                config_dict = config.model_dump()
            elif hasattr(config, 'dict'):
                config_dict = config.dict()
            else:
                config_dict = dict(config) if config else {}

            with open(config_path, 'w') as f:
                json.dump(config_dict, f, indent=2, default=str)
            logger.info("Saved configuration")

            # Save metadata
            metadata = {
                "project_name": name,
                "created_at": datetime.now().isoformat(),
                "data_rows": len(data),
                "data_columns": list(data.columns),
                "channels": [ch.get('name') or ch.name for ch in (config_dict.get('channels') or [])] if config_dict else [],
            }

            # Add selected session state
            if session_state:
                metadata["session"] = {
                    "date_column": session_state.get("date_column"),
                    "target_column": session_state.get("target_column"),
                    "detected_channels": session_state.get("detected_channels"),
                    "dayfirst": session_state.get("dayfirst", True),
                }

            metadata_path = tmpdir / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            # Create zip file
            zip_path = self.projects_dir / f"{project_name}.mmm"
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                zf.write(data_path, "data.csv")
                zf.write(config_path, "config.json")
                zf.write(metadata_path, "metadata.json")

            logger.info(f"Project saved to: {zip_path}")
            return zip_path

    def load_project(self, project_path: Path) -> Dict[str, Any]:
        """
        Load a project from file.

        Parameters
        ----------
        project_path : Path
            Path to the .mmm project file.

        Returns
        -------
        dict
            Dictionary with keys: 'data', 'config', 'metadata'
        """
        project_path = Path(project_path)

        if not project_path.exists():
            raise FileNotFoundError(f"Project not found: {project_path}")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Extract zip
            with zipfile.ZipFile(project_path, 'r') as zf:
                zf.extractall(tmpdir)

            # Load data
            data_path = tmpdir / "data.csv"
            data = pd.read_csv(data_path)
            logger.info(f"Loaded data: {len(data)} rows")

            # Load config
            config_path = tmpdir / "config.json"
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            logger.info("Loaded configuration")

            # Load metadata
            metadata_path = tmpdir / "metadata.json"
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            return {
                "data": data,
                "config": config_dict,
                "metadata": metadata,
            }

    def list_projects(self) -> list:
        """
        List all saved projects.

        Returns
        -------
        list
            List of dicts with project info.
        """
        projects = []

        for project_file in self.projects_dir.glob("*.mmm"):
            try:
                # Quick read of metadata only
                with zipfile.ZipFile(project_file, 'r') as zf:
                    with zf.open("metadata.json") as f:
                        metadata = json.load(f)

                projects.append({
                    "path": project_file,
                    "name": metadata.get("project_name", project_file.stem),
                    "created_at": metadata.get("created_at"),
                    "data_rows": metadata.get("data_rows"),
                    "channels": metadata.get("channels", []),
                })
            except Exception as e:
                logger.warning(f"Could not read project {project_file}: {e}")
                projects.append({
                    "path": project_file,
                    "name": project_file.stem,
                    "created_at": None,
                    "error": str(e),
                })

        # Sort by creation date (newest first)
        projects.sort(key=lambda x: x.get("created_at") or "", reverse=True)

        return projects

    def delete_project(self, project_path: Path) -> None:
        """Delete a project file."""
        project_path = Path(project_path)
        if project_path.exists():
            project_path.unlink()
            logger.info(f"Deleted project: {project_path}")


def save_project_from_session(session_state: Dict[str, Any], name: str) -> Optional[Path]:
    """
    Convenience function to save project from Streamlit session state.

    Parameters
    ----------
    session_state : dict
        Streamlit session state.
    name : str
        Project name.

    Returns
    -------
    Path or None
        Path to saved project, or None if data/config missing.
    """
    data = session_state.get("current_data")
    config = session_state.get("current_config")

    if data is None:
        logger.warning("No data to save")
        return None

    pm = ProjectManager()
    return pm.save_project(name, data, config, dict(session_state))


def load_project_to_session(project_path: Path) -> Dict[str, Any]:
    """
    Load a project and return data for session state.

    Parameters
    ----------
    project_path : Path
        Path to project file.

    Returns
    -------
    dict
        Data to update session state with.
    """
    pm = ProjectManager()
    project = pm.load_project(project_path)

    # Convert config dict back to ModelConfig
    from mmm_platform.config.schema import ModelConfig

    config = None
    if project["config"]:
        try:
            config = ModelConfig(**project["config"])
        except Exception as e:
            logger.warning(f"Could not parse config: {e}")

    # Build session update
    session_update = {
        "current_data": project["data"],
        "current_config": config,
    }

    # Restore session settings from metadata
    if "session" in project["metadata"]:
        sess = project["metadata"]["session"]
        session_update["date_column"] = sess.get("date_column")
        session_update["target_column"] = sess.get("target_column")
        session_update["detected_channels"] = sess.get("detected_channels")
        session_update["dayfirst"] = sess.get("dayfirst", True)

    return session_update
