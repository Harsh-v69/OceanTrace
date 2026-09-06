"""Database initialisation and basic persistence."""
from __future__ import annotations

import pytest
from sqlalchemy import inspect

from backend.core.database import Base, engine, init_db
from backend.models.investigation import Investigation
from backend.models.user import User, UserRole
from backend.services import users as user_svc

EXPECTED_TABLES = {
    "users",
    "jurisdictions",
    "investigations",
    "anomalies",
    "alerts",
    "vessels",
}


def test_init_db_is_idempotent():
    init_db()
    init_db()  # second call must not raise
    tables = set(inspect(engine).get_table_names())
    assert EXPECTED_TABLES.issubset(tables)


def test_every_model_table_registered():
    registered = set(Base.metadata.tables)
    assert EXPECTED_TABLES.issubset(registered)


def test_user_round_trips(db):
    created = user_svc.create_user(
        db,
        name="Ada Lovelace",
        email="Ada@Example.com",
        password="analytical-engine-1843",
        role=UserRole.REGIONAL,
        jurisdiction_ids=[1, 2, 3],
    )
    assert created.id is not None

    fetched = db.get(User, created.id)
    assert fetched is not None
    assert fetched.email == "ada@example.com"          # normalised to lower-case
    assert fetched.role is UserRole.REGIONAL
    assert fetched.jurisdiction_ids == [1, 2, 3]        # JSON column round-trips
    assert fetched.active is True
    assert fetched.created_at is not None


def test_email_unique_constraint(db):
    user_svc.create_user(db, name="A", email="dup@example.com", password="x" * 12)
    with pytest.raises(Exception):
        user_svc.create_user(db, name="B", email="dup@example.com", password="y" * 12)
        db.flush()


def test_sqlite_foreign_keys_enforced(db):
    """PRAGMA foreign_keys=ON should reject an orphan FK."""
    orphan = Investigation(reference="INV-ORPHAN", title="x", created_by_id=999999)
    db.add(orphan)
    with pytest.raises(Exception):
        db.commit()
    db.rollback()
