"""
Database Connection & Session Governance for Kallvettu Research Platform.

Supports:
- Local zero-dependency SQLite storage (`datasets/kallvettu_research.db`).
- Production-grade PostgreSQL via `DATABASE_URL` environment variable.
- Connection health check and foreign key enforcement.
"""

from __future__ import annotations

import os
import logging
from contextlib import contextmanager
from typing import Generator, Dict, Any, Optional

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session

logger = logging.getLogger("kallvettu.database")

DEFAULT_DB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "datasets"))
DEFAULT_SQLITE_PATH = os.path.join(DEFAULT_DB_DIR, "kallvettu_research.db")

Base = declarative_base()

_ENGINE: Optional[Engine] = None
_SESSION_FACTORY: Optional[sessionmaker] = None


def get_database_url() -> str:
    """Returns the active database connection string from environment or SQLite fallback."""
    env_url = os.getenv("DATABASE_URL")
    if env_url and env_url.strip():
        # Handle Heroku/Vercel postgres:// vs postgresql://
        if env_url.startswith("postgres://"):
            env_url = env_url.replace("postgres://", "postgresql://", 1)
        return env_url
    os.makedirs(DEFAULT_DB_DIR, exist_ok=True)
    return f"sqlite:///{DEFAULT_SQLITE_PATH}"


def get_engine() -> Engine:
    """Returns singleton SQLAlchemy Engine with connection pool parameters."""
    global _ENGINE
    if _ENGINE is None:
        db_url = get_database_url()
        connect_args: Dict[str, Any] = {}
        if db_url.startswith("sqlite"):
            connect_args["check_same_thread"] = False
            _ENGINE = create_engine(
                db_url,
                connect_args=connect_args,
                echo=False,
                pool_pre_ping=True,
            )

            # Enable foreign keys for SQLite
            @event.listens_for(_ENGINE, "connect")
            def set_sqlite_pragma(dbapi_connection, connection_record):
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.close()
        else:
            _ENGINE = create_engine(
                db_url,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True,
                echo=False,
            )
        logger.info("Initialized database engine with URL: %s", db_url.split("@")[-1])
    return _ENGINE


def get_session_factory() -> sessionmaker:
    """Returns singleton SQLAlchemy sessionmaker."""
    global _SESSION_FACTORY
    if _SESSION_FACTORY is None:
        engine = get_engine()
        _SESSION_FACTORY = sessionmaker(
            bind=engine,
            autoflush=False,
            autocommit=False,
            expire_on_commit=False,
        )
    return _SESSION_FACTORY


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """Context manager providing an atomic database session with automatic commit/rollback."""
    factory = get_session_factory()
    session = factory()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error("Database session transaction rolled back: %s", e)
        raise
    finally:
        session.close()


def check_database_health() -> Dict[str, Any]:
    """Evaluates database connectivity and returns diagnostic metadata."""
    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {
            "status": "healthy",
            "dialect": engine.dialect.name,
            "url_masked": get_database_url().split("@")[-1],
            "error": None,
        }
    except Exception as exc:
        logger.exception("Database health check failed: %s", exc)
        return {
            "status": "unhealthy",
            "dialect": "unknown",
            "url_masked": "unknown",
            "error": str(exc),
        }
