"""
Epic 1 - hierarchical user management + closed self-registration.
"""
from __future__ import annotations

import pytest

from backend.core.config import settings
from backend.models.user import UserRole
from backend.services import jurisdiction as juris
from backend.services import users as user_svc
from backend.tests._helpers import API, DEFAULT_PASSWORD, auth_header, make_user

PW = DEFAULT_PASSWORD


@pytest.fixture
def seeded(db):
    juris.seed_demo_jurisdictions(db)
    return db


def _code_id(db, code: str) -> int:
    return juris.resolve_codes_to_ids(db, [code])[0]


# --------------------------------------------------------------------------- #
# Closed self-registration
# --------------------------------------------------------------------------- #
def test_open_registration_disabled_returns_403(client, monkeypatch):
    monkeypatch.setattr(settings, "ALLOW_OPEN_REGISTRATION", False)
    r = client.post(f"{API}/auth/register", json={
        "name": "Nope", "email": "nope@example.com", "password": PW, "role": "PILOT",
    })
    assert r.status_code == 403
    assert "administrator" in r.json()["detail"].lower()


def test_open_registration_enabled_still_works_in_tests(client):
    # conftest sets ALLOW_OPEN_REGISTRATION=true so the existing helpers keep working
    r = client.post(f"{API}/auth/register", json={
        "name": "Yes", "email": "yes@example.com", "password": PW, "role": "PILOT",
    })
    assert r.status_code == 201


# --------------------------------------------------------------------------- #
# Role gate on /users
# --------------------------------------------------------------------------- #
def test_pilot_cannot_reach_user_management(client, seeded):
    mh = _code_id(seeded, "IN-MH")
    make_user(client, email="p@example.com", role="PILOT", jurisdiction_ids=[mh])
    h = auth_header(client, "p@example.com")
    assert client.get(f"{API}/users", headers=h).status_code == 403
    assert client.post(f"{API}/users", headers=h, json={
        "name": "x", "email": "x@example.com", "password": PW, "role": "PILOT",
        "jurisdiction_ids": [mh],
    }).status_code == 403


# --------------------------------------------------------------------------- #
# NATIONAL management
# --------------------------------------------------------------------------- #
def test_national_creates_regional_and_pilot(client, seeded):
    make_user(client, email="nat@example.com", role="NATIONAL")
    h = auth_header(client, "nat@example.com")

    r = client.post(f"{API}/users", headers=h, json={
        "name": "Reg", "email": "reg@example.com", "password": PW,
        "role": "REGIONAL", "jurisdiction_codes": ["IN-WEST"],
    })
    assert r.status_code == 201, r.text
    assert r.json()["role"] == "REGIONAL"
    assert r.json()["jurisdiction_ids"]

    r = client.post(f"{API}/users", headers=h, json={
        "name": "Pil", "email": "pil@example.com", "password": PW,
        "role": "PILOT", "jurisdiction_codes": ["IN-MH"],
    })
    assert r.status_code == 201, r.text

    r = client.post(f"{API}/users", headers=h, json={
        "name": "Nat2", "email": "nat2@example.com", "password": PW, "role": "NATIONAL",
    })
    assert r.status_code == 201, r.text

    # the new pilot can log in
    assert client.post(f"{API}/auth/login",
                       data={"username": "pil@example.com", "password": PW}).status_code == 200


def test_national_pilot_and_regional_need_a_jurisdiction(client, seeded):
    make_user(client, email="nat@example.com", role="NATIONAL")
    h = auth_header(client, "nat@example.com")
    r = client.post(f"{API}/users", headers=h, json={
        "name": "Pil", "email": "pil@example.com", "password": PW, "role": "PILOT",
        "jurisdiction_ids": [],
    })
    assert r.status_code == 422


# --------------------------------------------------------------------------- #
# REGIONAL management - strictly inside the region closure
# --------------------------------------------------------------------------- #
def test_regional_creates_pilot_only_inside_closure(client, seeded):
    west = _code_id(seeded, "IN-WEST")
    mh = _code_id(seeded, "IN-MH")
    kl = _code_id(seeded, "IN-KL")
    make_user(client, email="rw@example.com", role="REGIONAL", jurisdiction_ids=[west])
    h = auth_header(client, "rw@example.com")

    # inside IN-WEST -> ok
    assert client.post(f"{API}/users", headers=h, json={
        "name": "P-MH", "email": "pmh@example.com", "password": PW,
        "role": "PILOT", "jurisdiction_ids": [mh],
    }).status_code == 201

    # IN-KL is under IN-SOUTH, outside this REGIONAL's closure -> 403
    assert client.post(f"{API}/users", headers=h, json={
        "name": "P-KL", "email": "pkl@example.com", "password": PW,
        "role": "PILOT", "jurisdiction_ids": [kl],
    }).status_code == 403

    # a REGIONAL cannot mint another REGIONAL or a NATIONAL
    assert client.post(f"{API}/users", headers=h, json={
        "name": "R2", "email": "r2@example.com", "password": PW,
        "role": "REGIONAL", "jurisdiction_ids": [west],
    }).status_code == 403


def test_regional_list_is_scoped_to_its_pilots(client, seeded):
    west = _code_id(seeded, "IN-WEST")
    south = _code_id(seeded, "IN-SOUTH")
    mh = _code_id(seeded, "IN-MH")
    tn = _code_id(seeded, "IN-TN")
    make_user(client, email="rw@example.com", role="REGIONAL", jurisdiction_ids=[west])
    make_user(client, email="pmh@example.com", role="PILOT", jurisdiction_ids=[mh])
    make_user(client, email="ptn@example.com", role="PILOT", jurisdiction_ids=[tn])
    make_user(client, email="rs@example.com", role="REGIONAL", jurisdiction_ids=[south])

    rows = client.get(f"{API}/users", headers=auth_header(client, "rw@example.com")).json()
    emails = {u["email"] for u in rows}
    assert "pmh@example.com" in emails
    assert "ptn@example.com" not in emails      # different region
    assert "rw@example.com" not in emails       # not a PILOT
    assert "rs@example.com" not in emails


# --------------------------------------------------------------------------- #
# Disable / enable + guard rails
# --------------------------------------------------------------------------- #
def test_disable_blocks_login_and_enable_restores_it(client, seeded):
    mh = _code_id(seeded, "IN-MH")
    make_user(client, email="nat@example.com", role="NATIONAL")
    make_user(client, email="p@example.com", role="PILOT", jurisdiction_ids=[mh])
    h = auth_header(client, "nat@example.com")
    pid = client.get(f"{API}/users", headers=h).json()
    pid = next(u["id"] for u in pid if u["email"] == "p@example.com")

    assert client.post(f"{API}/users/{pid}/disable", headers=h).status_code == 200
    assert client.post(f"{API}/auth/login",
                       data={"username": "p@example.com", "password": PW}).status_code == 403

    assert client.post(f"{API}/users/{pid}/enable", headers=h).status_code == 200
    assert client.post(f"{API}/auth/login",
                       data={"username": "p@example.com", "password": PW}).status_code == 200


def test_cannot_manage_your_own_account_via_users(client, seeded):
    # self-service (name / phone / password) is on /auth/me, not /users
    make_user(client, email="nat@example.com", role="NATIONAL")
    h = auth_header(client, "nat@example.com")
    me = client.get(f"{API}/auth/me", headers=h).json()
    assert client.post(f"{API}/users/{me['id']}/disable", headers=h).status_code == 403
    assert client.patch(f"{API}/users/{me['id']}", headers=h,
                        json={"name": "Renamed"}).status_code == 403


def test_national_can_disable_another_national(client, seeded):
    make_user(client, email="nat1@example.com", role="NATIONAL")
    make_user(client, email="nat2@example.com", role="NATIONAL")
    h = auth_header(client, "nat1@example.com")
    nat2 = next(u["id"] for u in client.get(f"{API}/users", headers=h).json()
                if u["email"] == "nat2@example.com")
    assert client.post(f"{API}/users/{nat2}/disable", headers=h).status_code == 200
    assert client.post(f"{API}/auth/login",
                       data={"username": "nat2@example.com", "password": PW}).status_code == 403


# --------------------------------------------------------------------------- #
# Idempotent default-account seeding
# --------------------------------------------------------------------------- #
def test_seed_default_users_is_idempotent(seeded):
    n1 = user_svc.seed_default_users(seeded, password="Seeded!Password1")
    assert n1 == 3
    n2 = user_svc.seed_default_users(seeded, password="Seeded!Password1")
    assert n2 == 0
    emails = {u.email for u in user_svc.list_manageable_users(
        seeded, user_svc.get_user_by_email(seeded, "national@oceantrace.gov.in"))}
    assert {"regional@oceantrace.gov.in", "pilot@oceantrace.gov.in"} <= emails
    reg = user_svc.get_user_by_email(seeded, "regional@oceantrace.gov.in")
    assert reg.role == UserRole.REGIONAL and reg.jurisdiction_ids


def test_scope_endpoint_reports_creatable_roles(client, seeded):
    west = _code_id(seeded, "IN-WEST")
    make_user(client, email="nat@example.com", role="NATIONAL")
    make_user(client, email="rw@example.com", role="REGIONAL", jurisdiction_ids=[west])

    nat = client.get(f"{API}/users/me/scope", headers=auth_header(client, "nat@example.com")).json()
    assert set(nat["can_create_roles"]) == {"PILOT", "REGIONAL", "NATIONAL"}
    assert nat["jurisdiction_closure_ids"] is None

    reg = client.get(f"{API}/users/me/scope", headers=auth_header(client, "rw@example.com")).json()
    assert reg["can_create_roles"] == ["PILOT"]
    assert west in reg["jurisdiction_closure_ids"]
