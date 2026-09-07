"""
Epic 1 - RBAC hardening pass.

Confirms the strict-403 guarantee on single-resource access, the REGIONAL
closure boundary, code-scoped alert visibility, and the empty-assignment case.
"""
from __future__ import annotations

import pytest

from backend.services import jurisdiction as juris
from backend.models.alert import Alert, AlertChannel, AlertStatus
from backend.tests._helpers import API, auth_header, make_user

# points inside distinct zones
MH = (18.7, 72.4)     # Maharashtra  -> IN-MH  -> IN-WEST
KL = (9.9, 75.8)      # Kerala       -> IN-KL  -> IN-SOUTH


@pytest.fixture
def world(client, db):
    juris.seed_demo_jurisdictions(db)
    mh_id = juris.resolve_codes_to_ids(db, ["IN-MH"])[0]
    west_id = juris.resolve_codes_to_ids(db, ["IN-WEST"])[0]

    make_user(client, email="nat@example.com", role="NATIONAL")
    make_user(client, email="pilot.mh@example.com", role="PILOT", jurisdiction_ids=[mh_id])
    make_user(client, email="reg.west@example.com", role="REGIONAL", jurisdiction_ids=[west_id])
    make_user(client, email="pilot.none@example.com", role="PILOT", jurisdiction_ids=[])

    nat = auth_header(client, "nat@example.com")
    inv_mh = client.post(f"{API}/investigations", headers=nat, json={
        "title": "MH case", "centroid_lat": MH[0], "centroid_lon": MH[1]}).json()
    inv_kl = client.post(f"{API}/investigations", headers=nat, json={
        "title": "KL case", "centroid_lat": KL[0], "centroid_lon": KL[1]}).json()
    return {"nat": nat, "inv_mh": inv_mh["id"], "inv_kl": inv_kl["id"]}


def test_pilot_single_resource_403_outside_zone(client, world):
    h = auth_header(client, "pilot.mh@example.com")
    assert client.get(f"{API}/investigations/{world['inv_mh']}", headers=h).status_code == 200
    assert client.get(f"{API}/investigations/{world['inv_kl']}", headers=h).status_code == 403


def test_regional_closure_boundary(client, world):
    h = auth_header(client, "reg.west@example.com")
    # IN-MH is a child of IN-WEST -> visible
    assert client.get(f"{API}/investigations/{world['inv_mh']}", headers=h).status_code == 200
    # IN-KL is under IN-SOUTH -> outside the closure
    assert client.get(f"{API}/investigations/{world['inv_kl']}", headers=h).status_code == 403


def test_pilot_with_no_assignment_sees_nothing_and_gets_403(client, world):
    h = auth_header(client, "pilot.none@example.com")
    assert client.get(f"{API}/investigations", headers=h).json() == []
    assert client.get(f"{API}/investigations/{world['inv_mh']}", headers=h).status_code == 403


def test_list_is_jurisdiction_scoped(client, world):
    mh = auth_header(client, "pilot.mh@example.com")
    ids = {i["id"] for i in client.get(f"{API}/investigations", headers=mh).json()}
    assert world["inv_mh"] in ids and world["inv_kl"] not in ids


def test_regional_sees_code_scoped_alert_without_coordinates(client, world, db):
    # an alert with NO lat/lon but a jurisdiction_codes chain touching IN-WEST
    db.add(Alert(
        channel=AlertChannel.SMS, status=AlertStatus.MOCKED,
        recipient="+10000000000", message="code-scoped",
        jurisdiction_codes=["IN-MH", "IN-WEST", "IN-NATIONAL"],
        lat=None, lon=None, attempts=1, error_log=[],
    ))
    db.commit()
    h = auth_header(client, "reg.west@example.com")
    msgs = [a["message"] for a in client.get(f"{API}/alerts", headers=h).json()]
    assert "code-scoped" in msgs

    # a REGIONAL from the south must NOT see it
    south_id = juris.resolve_codes_to_ids(db, ["IN-SOUTH"])[0]
    make_user(client, email="reg.south@example.com", role="REGIONAL", jurisdiction_ids=[south_id])
    hs = auth_header(client, "reg.south@example.com")
    assert "code-scoped" not in [a["message"] for a in client.get(f"{API}/alerts", headers=hs).json()]


def test_min_role_ladder_on_alerts(client, world):
    assert client.get(f"{API}/alerts", headers=auth_header(client, "pilot.mh@example.com")).status_code == 403
    assert client.get(f"{API}/alerts", headers=auth_header(client, "reg.west@example.com")).status_code == 200
    assert client.get(f"{API}/alerts", headers=world["nat"]).status_code == 200
