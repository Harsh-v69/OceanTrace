"""
SQLAlchemy engine, session factory and the FastAPI ``get_db`` dependency.

SQLite is the default store. The design keeps a clean path to PostGIS:

* geometry is held in plain JSON columns (see ``models/jurisdiction.py``); the
  migration swaps those for ``geoalchemy2.Geometry`` and this URL for a
  ``postgresql+psycopg://`` one - nothing else changes.
* every session is request-scoped and closed in a ``finally``.
"""
from __future__ import annotations

from collections.abc import Iterator

from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from backend.core.config import settings


class Base(DeclarativeBase):
    """Declarative base shared by every ORM model."""


_connect_args: dict = {"check_same_thread": False} if settings.is_sqlite else {}

engine = create_engine(
    settings.DATABASE_URL,
    echo=settings.DATABASE_ECHO,
    connect_args=_connect_args,
    future=True,
)


if settings.is_sqlite:

    @event.listens_for(engine, "connect")
    def _enable_sqlite_fk(dbapi_connection, _connection_record):  # noqa: ANN001
        """SQLite ignores FOREIGN KEY constraints unless asked, per connection."""
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
    class_=Session,
)


def get_db() -> Iterator[Session]:
    """FastAPI dependency yielding a request-scoped session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create every table declared on ``Base``. Idempotent."""
    import backend.models  # noqa: F401  -- registers all mappers on Base.metadata

    Base.metadata.create_all(bind=engine)


def check_db() -> bool:
    """Cheap connectivity probe for the health endpoint."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:  # noqa: BLE001 -- health check must never raise
        return False
