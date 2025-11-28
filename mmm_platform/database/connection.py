"""
Database connection management.
"""

import os
from typing import Generator, Optional
from contextlib import contextmanager
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
import logging

from .models import Base

logger = logging.getLogger(__name__)


class DatabaseConnection:
    """
    Manage database connections.

    Supports PostgreSQL (production) and SQLite (development/testing).
    """

    def __init__(
        self,
        database_url: Optional[str] = None,
        echo: bool = False,
    ):
        """
        Initialize database connection.

        Parameters
        ----------
        database_url : str, optional
            Database URL. If not provided, uses DATABASE_URL env var
            or defaults to SQLite.
        echo : bool
            Whether to echo SQL statements.
        """
        if database_url is None:
            database_url = os.getenv(
                "DATABASE_URL",
                "sqlite:///./mmm_platform.db"
            )

        # Handle postgres:// vs postgresql:// for compatibility
        if database_url.startswith("postgres://"):
            database_url = database_url.replace("postgres://", "postgresql://", 1)

        self.database_url = database_url
        self.is_sqlite = database_url.startswith("sqlite")

        # Create engine with appropriate settings
        engine_kwargs = {"echo": echo}

        if not self.is_sqlite:
            # PostgreSQL connection pooling
            engine_kwargs.update({
                "poolclass": QueuePool,
                "pool_size": 5,
                "max_overflow": 10,
                "pool_pre_ping": True,
            })

        self.engine = create_engine(database_url, **engine_kwargs)
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine,
        )

        logger.info(f"Database connection initialized: {self._safe_url()}")

    def _safe_url(self) -> str:
        """Get URL with password masked."""
        if "@" in self.database_url:
            # Mask password
            parts = self.database_url.split("@")
            prefix = parts[0].rsplit(":", 1)[0]
            return f"{prefix}:****@{parts[1]}"
        return self.database_url

    def create_tables(self) -> None:
        """Create all tables in the database."""
        Base.metadata.create_all(bind=self.engine)
        logger.info("Database tables created")

    def drop_tables(self) -> None:
        """Drop all tables (use with caution!)."""
        Base.metadata.drop_all(bind=self.engine)
        logger.warning("Database tables dropped")

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """
        Get a database session as a context manager.

        Yields
        ------
        Session
            SQLAlchemy session.
        """
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def get_session_dependency(self) -> Generator[Session, None, None]:
        """
        Get a database session for FastAPI/Streamlit dependency injection.

        Yields
        ------
        Session
            SQLAlchemy session.
        """
        session = self.SessionLocal()
        try:
            yield session
        finally:
            session.close()


# Global database instance
_db: Optional[DatabaseConnection] = None


def init_db(database_url: Optional[str] = None, echo: bool = False) -> DatabaseConnection:
    """
    Initialize the global database connection.

    Parameters
    ----------
    database_url : str, optional
        Database URL.
    echo : bool
        Whether to echo SQL.

    Returns
    -------
    DatabaseConnection
        Database connection instance.
    """
    global _db
    _db = DatabaseConnection(database_url, echo)
    return _db


def get_db() -> DatabaseConnection:
    """
    Get the global database connection.

    Returns
    -------
    DatabaseConnection
        Database connection instance.

    Raises
    ------
    RuntimeError
        If database not initialized.
    """
    global _db
    if _db is None:
        # Auto-initialize with defaults
        _db = DatabaseConnection()
    return _db
