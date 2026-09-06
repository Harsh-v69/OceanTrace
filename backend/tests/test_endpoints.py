"""Skeleton resource routers: protected-route access and basic CRUD wiring."""
from __future__ import annotations

from backend.tests._helpers import API, auth_header, make_user


def test_investigations_list_requires_auth(client):
    assert client.get(f"{API}/investigations").status_code == 401


def test_investigations_crud_roundtrip(client):
    # PILOT scoped to the Maharashtra coastal zone; the slick below is inside it.
    make_user(client, email="inv@example.com", role="PILOT", jurisdiction_codes=["IN-MH"])
    headers = auth_header(client, "inv@example.com")

    empty = client.get(f"{API}/investigations", headers=headers)
    assert empty.status_code == 200
    assert empty.json() == []

    created = client.post(
        f"{API}/investigations",
        headers=headers,
        json={"title": "Arabian Sea slick", "centroid_lat": 18.7, "centroid_lon": 72.6},
    )
    assert created.status_code == 201, created.text
    body = created.json()
    assert body["reference"].startswith("INV-")
    assert body["status"] == "OPEN"
    assert body["title"] == "Arabian Sea slick"

    listed = client.get(f"{API}/investigations", headers=headers).json()
    assert len(listed) == 1

    one = client.get(f"{API}/investigations/{body['id']}", headers=headers)
    assert one.status_code == 200
    assert one.json()["id"] == body["id"]

    missing = client.get(f"{API}/investigations/999999", headers=headers)
    assert missing.status_code == 404


def test_vessels_list_requires_auth_and_is_empty(client):
    assert client.get(f"{API}/vessels").status_code == 401
    make_user(client, email="ves@example.com")
    headers = auth_header(client, "ves@example.com")
    resp = client.get(f"{API}/vessels", headers=headers)
    assert resp.status_code == 200
    assert resp.json() == []
