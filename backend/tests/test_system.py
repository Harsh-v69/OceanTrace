"""System endpoints and app wiring."""
from __future__ import annotations

from backend.tests._helpers import API


def test_health_reports_ok_and_db_up(client):
    resp = client.get(f"{API}/system/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["database"] == "up"
    assert body["environment"] == "test"
    assert body["version"]
    assert body["problem_statement_id"] == "26143"
    assert "time_utc" in body


def test_health_is_public(client):
    # No Authorization header at all.
    assert client.get(f"{API}/system/health").status_code == 200


def test_info_lists_roles(client):
    body = client.get(f"{API}/system/info").json()
    assert set(body["roles"]) == {"PILOT", "REGIONAL", "NATIONAL"}


def test_root_and_api_index(client):
    # "/" now redirects to the Operations Console; the API stays under /api/*
    root = client.get("/", follow_redirects=False)
    assert root.status_code in (307, 308)
    assert root.headers["location"] == "/app/"

    console = client.get("/app/")
    assert console.status_code == 200
    assert "Operations Console" in console.text

    index = client.get("/api")
    assert index.status_code == 200
    paths = index.json()["paths"]
    assert f"{API}/auth/login" in paths
    assert f"{API}/system/health" in paths


def test_openapi_schema_available(client):
    resp = client.get("/api/openapi.json")
    assert resp.status_code == 200
    assert resp.json()["info"]["title"]
