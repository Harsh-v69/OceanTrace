"""Tiny helpers shared by the test modules."""
from __future__ import annotations

from fastapi.testclient import TestClient

API = "/api/v1"
DEFAULT_PASSWORD = "Passw0rd!secure"


def register(
    client: TestClient,
    *,
    email: str,
    password: str = DEFAULT_PASSWORD,
    name: str = "Test User",
    role: str = "PILOT",
    **extra: object,
):
    body = {"name": name, "email": email, "password": password, "role": role}
    body.update(extra)
    return client.post(f"{API}/auth/register", json=body)


def login(client: TestClient, email: str, password: str = DEFAULT_PASSWORD):
    return client.post(
        f"{API}/auth/login", data={"username": email, "password": password}
    )


def token_for(client: TestClient, email: str, password: str = DEFAULT_PASSWORD) -> str:
    resp = login(client, email, password)
    assert resp.status_code == 200, resp.text
    return resp.json()["access_token"]


def auth_header(client: TestClient, email: str, password: str = DEFAULT_PASSWORD) -> dict:
    return {"Authorization": f"Bearer {token_for(client, email, password)}"}


def make_user(
    client: TestClient,
    *,
    email: str,
    role: str = "PILOT",
    password: str = DEFAULT_PASSWORD,
    **extra: object,
) -> dict:
    resp = register(client, email=email, role=role, password=password, **extra)
    assert resp.status_code == 201, resp.text
    return resp.json()
