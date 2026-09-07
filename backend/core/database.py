"""
SQLAlchemy engine, session factory and the FastAPI ``get_db`` dependency.

SQLite is the default store; **PostgreSQL + PostGIS** is supported by pointing
``DATABASE_URL`` at ``postgresql+psycopg://…`` (needs ``psycopg`` installed).

* Geometry is held in portable JSON columns (see ``models/jurisdiction.py``),
  which work identically on SQLite and Postgres. On Postgres the PostGIS
  extension is enabled at startup when ``geoalchemy2`` is installed, so a later
  swap of those columns for ``geoalchemy2.Geometry`` is a migration-only change.
* Alembic owns the schema (`alembic/`); ``init_db`` (create_all) stays as the
  zero-config path for SQLite dev / tests.
* every session is request-scoped and closed in a ``finally``.
"""
from __future__ import annotations

from collections.abc import Iterator

from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from backend.core.config import settings

from backend.core.logging import get_logger

log = get_logger("backend.core.database")


class Base(DeclarativeBase):
    """Declarative base shared by every ORM model."""


_is_sqlite = settings.is_sqlite
_is_postgres = settings.DATABASE_URL.startswith(("postgresql", "postgres"))

_connect_args: dict = {"check_same_thread": False} if _is_sqlite else {}
_engine_kwargs: dict = {"echo": settings.DATABASE_ECHO, "future": True,
                        "connect_args": _connect_args}
if _is_postgres:
    # sane server-side pool for a real deployment
    _engine_kwargs.update(pool_pre_ping=True, pool_size=5, max_overflow=10)

engine = create_engine(settings.DATABASE_URL, **_engine_kwargs)


if _is_sqlite:

    @event.listens_for(engine, "connect")
    def _enable_sqlite_fk(dbapi_connection, _connection_record):  # noqa: ANN001
        """SQLite ignores FOREIGN KEY constraints unless asked, per connection."""
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


def enable_postgis() -> bool:
    """On PostgreSQL, ``CREATE EXTENSION IF NOT EXISTS postgis``. Best-effort.

    Only attempted when ``geoalchemy2`` is importable (the marker that a PostGIS
    geometry backend is intended); returns True on success, False otherwise.
    """
    if not _is_postgres:
        return False
    try:
        import geoalchemy2  # noqa: F401
    except ImportError:
        log.info("Postgres URL but geoalchemy2 not installed; skipping PostGIS extension")
        return False
    try:
        with engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis"))
        log.info("PostGIS extension ensured")
        return True
    except Exception as exc:  # noqa: BLE001 - never block startup
        log.warning("could not enable the PostGIS extension: %s", exc)
        return False


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
