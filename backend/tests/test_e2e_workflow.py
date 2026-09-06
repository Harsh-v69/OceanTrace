"""
Phase 8 - end-to-end lifecycle.

One test walks the whole unified pipeline the way an operator would:

    register (PILOT)  ->  login  ->  Mission Control (list scenarios)
      ->  run a demo scenario  ->  SAR detection  ->  look-alike filter
      ->  drift hindcast + forecast  ->  AIS correlation + fusion scoring
      ->  jurisdiction determination  ->  Mock SMS alert
      ->  evidence dossier export (JSON / Markdown / HTML)

plus the RBAC guarantees (a PILOT cannot run a scenario outside their zone)
and the look-alike guarantee (a rejected scene raises no alert).
"""
from __future__ import annotations

import pytest

from backend.core.config import settings
from backend.services import sms as sms_svc
from backend.tests._helpers import API, auth_header, make_user

MUMBAI = "mumbai-high-confidence"
LOOKALIKE = "lookalike-darkpatch"
AMBIGUOUS = "ambiguous-drift"
WAKASHIO = "wakashio-mauritius"


@pytest.fixture
def sms_mock():
    """A clean MockSmsProvider that the service layer also hands out."""
    settings.SMS_PROVIDER = "mock"
    sms_svc.reset_sms_provider()
    prov = sms_svc.get_sms_provider()
    assert isinstance(prov, sms_svc.MockSmsProvider)
    yield prov
    sms_svc.reset_sms_provider()


# --------------------------------------------------------------------------- #
# The happy path
# --------------------------------------------------------------------------- #
def test_full_lifecycle_pilot_mumbai(client, sms_mock):
    # --- 1. registration (PILOT, assigned to Mumbai/Maharashtra) ----------
    make_user(
        client,
        email="pilot.mh@example.com",
        role="PILOT",
        name="MH Pilot",
        phone_number="+15005550111",
        jurisdiction_codes=["IN-MH"],
    )

    # --- 2. login -------------------------------------------------------
    hdr = auth_header(client, "pilot.mh@example.com")

    me = client.get(f"{API}/auth/me", headers=hdr)
    assert me.status_code == 200
    assert me.json()["role"] == "PILOT"

    # --- 3. Mission Control: the scenario catalogue -------------------
    cat = client.get(f"{API}/scenarios", headers=hdr)
    assert cat.status_code == 200
    keys = {s["key"] for s in cat.json()["scenarios"]}
    assert {MUMBAI, LOOKALIKE, AMBIGUOUS, WAKASHIO} <= keys

    # --- 4. trigger the investigation scenario ----------------------
    run = client.post(f"{API}/scenarios/{MUMBAI}/run", headers=hdr)
    assert run.status_code == 201, run.text
    body = run.json()
    inv_id = body["investigation_id"]

    # SAR detection produced an oil-like scene above the alert threshold
    assert body["classification"] == "Oil-like anomaly"
    assert body["confidence"] >= settings.ALERT_CONFIDENCE_THRESHOLD

    # jurisdiction determination -> Mumbai
    assert body["jurisdiction"]["primary_code"] == "IN-MH"
    assert body["jurisdiction"]["chain_codes"] == ["IN-MH", "IN-WEST", "IN-NATIONAL"]

    # AIS correlation + fusion produced a ranked prime suspect
    # (decoys are deliberately gated out as background traffic)
    assert body["n_candidates"] >= 1
    assert body["prime_suspect"] is not None

    # Mock SMS alert was dispatched
    assert body["alert_status"] in {"SENT", "MOCKED"}
    assert sms_mock.outbox, "expected the Mock SMS provider to have an outbound message"
    assert "Oil-like anomaly" in sms_mock.outbox[-1]["body"]

    # every pipeline step reported OK
    step_names = [s["step"] for s in body["pipeline"]]
    for expected in ("detect", "characterize", "trace", "correlate", "rank",
                     "jurisdiction", "alert"):
        assert expected in step_names

    # --- 5. read the persisted investigation ----------------------
    inv = client.get(f"{API}/investigations/{inv_id}", headers=hdr)
    assert inv.status_code == 200
    m = inv.json()["summary_metrics"]
    for section in ("sar", "hindcast", "forecast", "attribution",
                    "feedback_loop", "jurisdiction", "drift_frames"):
        assert section in m, f"summary_metrics missing {section!r}"

    # drift hindcast reconstructed an origin + a release-time window
    assert len(m["hindcast"]["best_estimate"]) == 2
    assert len(m["hindcast"]["release_window_h"]) == 2
    assert m["forecast"]["horizons_h"]  # forward projection present

    # fusion ranked the true culprit first
    gt = m["attribution"]["summary"]["ground_truth"]
    assert gt["correctly_ranked_first"] is True

    # the fusion score is transparent: components are exposed per candidate
    prime = m["attribution"]["candidates"][0]
    assert set(prime["components"]) >= {
        "spatiotemporal", "axis_alignment", "proximity",
        "blackout", "ais_anomaly", "route_deviation", "vessel_prior",
    }

    # --- 6. anomalies recorded for the case --------------------
    anoms = client.get(f"{API}/anomalies?investigation_id={inv_id}", headers=hdr)
    assert anoms.status_code == 200
    labels = {a["label"] for a in anoms.json()}
    assert "Oil-like anomaly" in labels

    # --- 7. evidence dossier export ---------------------------
    dj = client.get(f"{API}/investigations/{inv_id}/dossier", headers=hdr)
    assert dj.status_code == 200
    d = dj.json()
    for section in ("incident_summary", "satellite_metadata", "spill_geometry",
                    "met_ocean", "hindcast_origin", "release_time_window",
                    "ais_evidence", "candidate_ranking", "jurisdiction",
                    "audit_trail"):
        assert section in d, f"dossier missing {section!r}"
    assert d["candidate_ranking"]["prime_suspect"] is not None

    md = client.get(f"{API}/investigations/{inv_id}/dossier.md", headers=hdr)
    assert md.status_code == 200
    assert md.headers["content-type"].startswith("text/plain")
    assert body["reference"] in md.text

    html = client.get(f"{API}/investigations/{inv_id}/dossier.html", headers=hdr)
    assert html.status_code == 200
    assert html.headers["content-type"].startswith("text/html")
    assert "<html" in html.text.lower()


# --------------------------------------------------------------------------- #
# RBAC: a PILOT cannot reach outside their assigned zone
# --------------------------------------------------------------------------- #
def test_pilot_outside_zone_cannot_run_scenario(client):
    make_user(
        client,
        email="pilot.kl@example.com",
        role="PILOT",
        jurisdiction_codes=["IN-KL"],   # Kerala - not Mumbai
    )
    hdr = auth_header(client, "pilot.kl@example.com")

    run = client.post(f"{API}/scenarios/{MUMBAI}/run", headers=hdr)
    assert run.status_code == 403

    # and there is no investigation to see
    lst = client.get(f"{API}/investigations", headers=hdr)
    assert lst.status_code == 200
    assert lst.json() == []


# --------------------------------------------------------------------------- #
# Look-alike: a rejected dark patch raises no alert
# --------------------------------------------------------------------------- #
def test_lookalike_scene_is_filtered_and_raises_no_alert(client, sms_mock):
    make_user(client, email="nat@example.com", role="NATIONAL")
    hdr = auth_header(client, "nat@example.com")

    run = client.post(f"{API}/scenarios/{LOOKALIKE}/run", headers=hdr)
    assert run.status_code == 201, run.text
    body = run.json()

    assert body["classification"] != "Oil-like anomaly"
    assert body["status"] == "RESOLVED"
    assert body["alert_status"] is None
    assert body["prime_suspect"] is None
    assert sms_mock.outbox == []

    alert_step = next(s for s in body["pipeline"] if s["step"] == "alert")
    assert alert_step["dispatched"] is False

    # the dossier still builds for a rejected scene
    dj = client.get(f"{API}/investigations/{body['investigation_id']}/dossier", headers=hdr)
    assert dj.status_code == 200
    assert "incident_summary" in dj.json()


# --------------------------------------------------------------------------- #
# The remaining scenarios run end to end for a NATIONAL operator
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("key", [AMBIGUOUS, WAKASHIO])
def test_other_scenarios_run_and_rank_the_true_culprit(client, sms_mock, key):
    make_user(client, email=f"nat.{key}@example.com", role="NATIONAL")
    hdr = auth_header(client, f"nat.{key}@example.com")

    run = client.post(f"{API}/scenarios/{key}/run", headers=hdr)
    assert run.status_code == 201, run.text
    inv_id = run.json()["investigation_id"]

    m = client.get(f"{API}/investigations/{inv_id}", headers=hdr).json()["summary_metrics"]
    gt = m["attribution"]["summary"]["ground_truth"]
    assert gt["correctly_ranked_first"] is True
