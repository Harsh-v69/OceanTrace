"""Registration, login and the /auth/me protected route."""
from __future__ import annotations

from backend.tests._helpers import API, DEFAULT_PASSWORD, auth_header, login, register


def test_register_returns_user_without_secrets(client):
    resp = register(client, email="pilot1@example.com", name="Pilot One")
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["email"] == "pilot1@example.com"
    assert body["role"] == "PILOT"
    assert body["active"] is True
    assert "id" in body
    assert "password" not in body
    assert "password_hash" not in body


def test_register_rejects_duplicate_email(client):
    register(client, email="dupe@example.com")
    resp = register(client, email="dupe@example.com")
    assert resp.status_code == 409


def test_register_rejects_weak_password(client):
    resp = client.post(
        f"{API}/auth/register",
        json={"name": "x", "email": "weak@example.com", "password": "short"},
    )
    assert resp.status_code == 422


def test_register_can_set_role_when_enabled(client):
    resp = register(client, email="boss@example.com", role="NATIONAL")
    assert resp.status_code == 201
    assert resp.json()["role"] == "NATIONAL"


def test_login_success_returns_token_and_user(client):
    register(client, email="login-ok@example.com")
    resp = login(client, "login-ok@example.com")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["token_type"] == "bearer"
    assert body["access_token"]
    assert body["expires_in"] > 0
    assert body["user"]["email"] == "login-ok@example.com"


def test_login_wrong_password_is_401(client):
    register(client, email="login-bad@example.com")
    resp = login(client, "login-bad@example.com", password="not-the-password")
    assert resp.status_code == 401


def test_login_unknown_email_is_401(client):
    resp = login(client, "ghost@example.com")
    assert resp.status_code == 401


def test_me_requires_a_token(client):
    assert client.get(f"{API}/auth/me").status_code == 401


def test_me_rejects_garbage_token(client):
    resp = client.get(
        f"{API}/auth/me", headers={"Authorization": "Bearer not.a.jwt"}
    )
    assert resp.status_code == 401


def test_me_returns_the_authenticated_user(client):
    register(client, email="me@example.com", name="Self Test")
    resp = client.get(f"{API}/auth/me", headers=auth_header(client, "me@example.com"))
    assert resp.status_code == 200
    body = resp.json()
    assert body["email"] == "me@example.com"
    assert body["name"] == "Self Test"


def test_disabled_user_cannot_use_its_token(client, db):
    register(client, email="disabled@example.com")
    headers = auth_header(client, "disabled@example.com")

    from backend.models.user import User
    from sqlalchemy import select

    user = db.execute(
        select(User).where(User.email == "disabled@example.com")
    ).scalar_one()
    user.active = False
    db.commit()

    resp = client.get(f"{API}/auth/me", headers=headers)
    assert resp.status_code == 403
