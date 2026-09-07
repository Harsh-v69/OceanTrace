"""
Shared test fixtures.

The environment is configured here, BEFORE any ``backend`` module is imported,
so the app builds its engine against a throwaway SQLite file.
"""
from __future__ import annotations

import os
import pathlib
import tempfile

_TEST_DB = pathlib.Path(tempfile.gettempdir()) / "sn_poseatsea_phase2_test.db"

os.environ["ENV"] = "test"
os.environ["DEBUG"] = "false"
os.environ["DATABASE_URL"] = f"sqlite:///{_TEST_DB.as_posix()}"
os.environ["JWT_SECRET_KEY"] = "unit-test-secret-not-for-production-use-0123456789"
os.environ["ALLOW_REGISTRATION_ROLE_SELECT"] = "true"
os.environ["ALLOW_OPEN_REGISTRATION"] = "true"   # keep the /auth/register helper usable
os.environ["SEED_DEFAULT_USERS"] = "false"        # deterministic user table per test
os.environ["SMS_PROVIDER"] = "mock"

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import delete  # noqa: E402

import backend.models  # noqa: E402,F401  -- register mappers
from backend.core.database import Base, SessionLocal, engine  # noqa: E402
from backend.main import app  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _schema():
    if _TEST_DB.exists():
        _TEST_DB.unlink()
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)
    # Windows keeps the SQLite file handle open until the engine is disposed.
    engine.dispose()
    try:
        _TEST_DB.unlink(missing_ok=True)
    except PermissionError:  # pragma: no cover - best-effort temp cleanup
        pass


@pytest.fixture(autouse=True)
def _clean_tables():
    """Wipe every table after each test so cases stay independent."""
    yield
    with SessionLocal() as session:
        for table in reversed(Base.metadata.sorted_tables):
            session.execute(delete(table))
        session.commit()


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def db():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
