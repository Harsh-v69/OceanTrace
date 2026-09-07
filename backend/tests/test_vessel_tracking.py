"""
Epic 1.4 - vessel tracking engine.

A scenario run must now persist real Vessel rows, stash a map-ready track view
per candidate in ``summary_metrics.vessel_tracks``, and serve it through the
vessel endpoints - jurisdiction-scoped.
"""
from __future__ import annotations

import pytest

from backend.services import jurisdiction as juris
from backend.tests._helpers import API, auth_header, make_user

MUMBAI = "mumbai-high-confidence"
CULPRIT = "419810001"          # MT KONKAN PRIDE


@pytest.fixture
def run_mumbai(client, db):
    juris.seed_demo_jurisdictions(db)
    make_user(client, email="nat@example.com", role="NATIONAL")
    h = auth_header(client, "nat@example.com")
    r = client.post(f"{API}/scenarios/{MUMBAI}/run", headers=h)
    assert r.status_code == 201, r.text
    return {"h": h, "inv": r.json()["investigation_id"]}


def test_scenario_persists_vessel_rows(client, run_mumbai):
    rows = client.get(f"{API}/vessels", headers=run_mumbai["h"]).json()
    assert rows, "no Vessel rows created by the scenario run"
    by_mmsi = {v["mmsi"]: v for v in rows}
    assert CULPRIT in by_mmsi
    v = by_mmsi[CULPRIT]
    assert v["name"] and v["vessel_type"]          # identity populated


def test_vessel_detail_merges_track_and_attribution(client, run_mumbai):
    h, inv = run_mumbai["h"], run_mumbai["inv"]
    v = client.get(f"{API}/vessels/{CULPRIT}?investigation_id={inv}", headers=h).json()
    assert v["mmsi"] == CULPRIT
    assert v["track"] and v["track"]["pings"], "track view missing pings"
    assert v["metrics"]["n_points"] >= 2
    attr = v["attribution"]
    assert attr and attr["is_prime"] is True and attr["score"] is not None


def test_vessel_track_endpoint_shape(client, run_mumbai):
    h, inv = run_mumbai["h"], run_mumbai["inv"]
    t = client.get(f"{API}/vessels/{CULPRIT}/track?investigation_id={inv}", headers=h).json()
    for key in ("pings", "loiter", "blackouts", "metrics"):
        assert key in t, f"track view missing {key!r}"
    assert len(t["pings"]) >= 2
    p0 = t["pings"][0]
    assert {"lat", "lon", "t_h", "sog", "cog"} <= set(p0)
    m = t["metrics"]
    assert m["sog_max_kn"] >= m["sog_min_kn"]
    assert m["first_seen_h"] <= m["last_seen_h"]


def test_summary_metrics_carries_prime_track(client, run_mumbai):
    h, inv = run_mumbai["h"], run_mumbai["inv"]
    m = client.get(f"{API}/investigations/{inv}", headers=h).json()["summary_metrics"]
    vts = m.get("vessel_tracks") or {}
    assert CULPRIT in vts
    view = vts[CULPRIT]
    assert view["pings"] and view["attribution"]["is_prime"] is True
    # every scored candidate has a view
    for c in m["attribution"]["candidates"]:
        assert str(c["identity"]["mmsi"]) in vts


def test_vessel_is_jurisdiction_scoped(client, run_mumbai, db):
    kl = juris.resolve_codes_to_ids(db, ["IN-KL"])[0]
    make_user(client, email="pilot.kl@example.com", role="PILOT", jurisdiction_ids=[kl])
    hk = auth_header(client, "pilot.kl@example.com")
    # the culprit is linked to an anomaly in IN-MH -> a Kerala pilot cannot see it
    assert client.get(f"{API}/vessels/{CULPRIT}", headers=hk).status_code == 403
    assert client.get(f"{API}/vessels", headers=hk).json() == []

    mh = juris.resolve_codes_to_ids(db, ["IN-MH"])[0]
    make_user(client, email="pilot.mh@example.com", role="PILOT", jurisdiction_ids=[mh])
    hm = auth_header(client, "pilot.mh@example.com")
    assert client.get(f"{API}/vessels/{CULPRIT}", headers=hm).status_code == 200


def test_lookalike_scenario_creates_no_vessels(client, db):
    juris.seed_demo_jurisdictions(db)
    make_user(client, email="nat@example.com", role="NATIONAL")
    h = auth_header(client, "nat@example.com")
    r = client.post(f"{API}/scenarios/lookalike-darkpatch/run", headers=h)
    assert r.status_code == 201
    m = client.get(f"{API}/investigations/{r.json()['investigation_id']}",
                   headers=h).json()["summary_metrics"]
    assert not (m.get("vessel_tracks") or {})
