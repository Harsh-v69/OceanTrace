"""
Phase 7 - geographical jurisdiction engine, strict RBAC + jurisdiction
filtering, and the real SMS alert engine (Twilio / Mock switching, fingerprint
deduplication, audit trail + retry, and the test-sms endpoint).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from backend.core.config import settings
from backend.models.alert import AlertStatus
from backend.models.jurisdiction import Jurisdiction, JurisdictionType
from backend.models.user import User, UserRole
from backend.services import jurisdiction as juris
from backend.services import sms as sms_svc
from backend.tests._helpers import API, auth_header, make_user

PW = "Passw0rd!secure"
NOW = datetime(2026, 3, 6, 12, 0, 0, tzinfo=timezone.utc)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def seeded_db(db):
    juris.seed_demo_jurisdictions(db)
    return db


@pytest.fixture
def sms_mock():
    """A clean MockSmsProvider that get_sms_provider() also returns."""
    settings.SMS_PROVIDER = "mock"
    sms_svc.reset_sms_provider()
    prov = sms_svc.get_sms_provider()
    assert isinstance(prov, sms_svc.MockSmsProvider)
    yield prov
    sms_svc.reset_sms_provider()


def _code_id(db, code: str) -> int:
    return db.query(Jurisdiction).filter_by(code=code).one().id


# =========================================================================== #
# Jurisdiction engine - point in polygon
# =========================================================================== #
def test_seeding_is_idempotent(seeded_db):
    n = juris.seed_demo_jurisdictions(seeded_db)
    assert n == 0
    zones = seeded_db.query(Jurisdiction).all()
    assert len(zones) == 14
    codes = {z.code for z in zones}
    assert "IN-NATIONAL" in codes and "IN-MH" in codes and "IN-WEST" in codes


def test_zone_hierarchy(seeded_db):
    mh = seeded_db.query(Jurisdiction).filter_by(code="IN-MH").one()
    west = seeded_db.query(Jurisdiction).filter_by(code="IN-WEST").one()
    national = seeded_db.query(Jurisdiction).filter_by(code="IN-NATIONAL").one()
    assert mh.type == JurisdictionType.COASTAL_STATE
    assert mh.parent_id == west.id
    assert west.parent_id == national.id
    assert national.parent_id is None


@pytest.mark.parametrize(
    "lat,lon,primary,chain",
    [
        (18.7, 72.4, "IN-MH", ["IN-MH", "IN-WEST", "IN-NATIONAL"]),
        (22.0, 69.5, "IN-GJ", ["IN-GJ", "IN-WEST", "IN-NATIONAL"]),
        (9.9, 75.8, "IN-KL", ["IN-KL", "IN-SOUTH", "IN-NATIONAL"]),
        (12.9, 80.5, "IN-TN", ["IN-TN", "IN-SOUTH", "IN-NATIONAL"]),
        (16.5, 82.5, "IN-AP", ["IN-AP", "IN-EAST", "IN-NATIONAL"]),
        (20.2, 86.8, "IN-OD", ["IN-OD", "IN-EAST", "IN-NATIONAL"]),
        (21.8, 88.5, "IN-WB", ["IN-WB", "IN-EAST", "IN-NATIONAL"]),
        (11.6, 92.8, "IN-AN", ["IN-AN", "IN-ISLANDS", "IN-NATIONAL"]),
    ],
)
def test_point_resolves_to_the_expected_jurisdiction_chain(seeded_db, lat, lon, primary, chain):
    r = juris.resolve_affected_jurisdictions(seeded_db, lat, lon)
    assert r["primary"].code == primary
    assert r["codes"] == chain
    # deterministic
    assert juris.resolve_affected_jurisdictions(seeded_db, lat, lon)["codes"] == chain


def test_state_zones_do_not_overlap(seeded_db):
    for lat, lon in [(22.0, 69.5), (18.7, 72.4), (14.5, 73.5), (9.9, 75.8),
                     (11.0, 79.5), (16.5, 82.5), (20.2, 86.8), (21.8, 88.5), (11.6, 92.8)]:
        hits = juris.jurisdictions_containing_point(seeded_db, lat, lon)
        states = [j for j in hits if j.type == JurisdictionType.COASTAL_STATE]
        assert len(states) == 1, f"({lat},{lon}) hit {[s.code for s in states]}"


def test_open_ocean_point_resolves_to_nothing(seeded_db):
    r = juris.resolve_affected_jurisdictions(seeded_db, 5.0, 60.0)
    assert r["primary"] is None and r["codes"] == []


def test_resolve_endpoint(client):
    make_user(client, email="nat_r@x.com", role="NATIONAL")
    h = auth_header(client, "nat_r@x.com")
    r = client.get(f"{API}/jurisdictions/resolve", headers=h, params={"lat": 18.7, "lon": 72.4})
    assert r.status_code == 200
    assert r.json()["primary_code"] == "IN-MH"
    assert r.json()["chain_codes"] == ["IN-MH", "IN-WEST", "IN-NATIONAL"]


def test_jurisdiction_list_endpoint_mine_only(client):
    make_user(client, email="pil_j@x.com", role="PILOT", jurisdiction_codes=["IN-MH"])
    h = auth_header(client, "pil_j@x.com")
    allz = client.get(f"{API}/jurisdictions", headers=h).json()
    assert len(allz) == 14
    mine = client.get(f"{API}/jurisdictions", headers=h, params={"mine_only": True}).json()
    assert [z["code"] for z in mine] == ["IN-MH"]


# =========================================================================== #
# Jurisdiction closure
# =========================================================================== #
def test_accessible_closure_by_role(seeded_db):
    mh_id = _code_id(seeded_db, "IN-MH")
    west_id = _code_id(seeded_db, "IN-WEST")

    pilot = User(name="p", email="p@c", password_hash="x", role=UserRole.PILOT,
                 jurisdiction_ids=[mh_id])
    regional = User(name="r", email="r@c", password_hash="x", role=UserRole.REGIONAL,
                    jurisdiction_ids=[west_id])
    national = User(name="n", email="n@c", password_hash="x", role=UserRole.NATIONAL,
                    jurisdiction_ids=[])
    seeded_db.add_all([pilot, regional, national])
    seeded_db.flush()

    codes = {j.id: j.code for j in seeded_db.query(Jurisdiction).all()}
    assert {codes[i] for i in juris.accessible_jurisdiction_ids(seeded_db, pilot)} == {"IN-MH"}
    assert {codes[i] for i in juris.accessible_jurisdiction_ids(seeded_db, regional)} == {
        "IN-WEST", "IN-GJ", "IN-MH", "IN-GA-KA"
    }
    assert juris.accessible_jurisdiction_ids(seeded_db, national) is None


def test_user_with_no_jurisdiction_sees_nothing(seeded_db):
    u = User(name="x", email="x@c", password_hash="x", role=UserRole.PILOT, jurisdiction_ids=[])
    seeded_db.add(u)
    seeded_db.flush()
    assert juris.accessible_jurisdiction_ids(seeded_db, u) == set()
    assert juris.user_can_access_point(seeded_db, u, 18.7, 72.4) is False


# =========================================================================== #
# RBAC + jurisdiction filtering over the API  (403 checks)
# =========================================================================== #
@pytest.fixture
def rbac_world(client):
    """A NATIONAL user plus MH / Kerala / Gujarat investigations + anomalies."""
    make_user(client, email="natl@x.com", role="NATIONAL")
    make_user(client, email="pilot.mh@x.com", role="PILOT", jurisdiction_codes=["IN-MH"],
              phone_number="+15005550006")
    make_user(client, email="regional.west@x.com", role="REGIONAL", jurisdiction_codes=["IN-WEST"])
    nh = auth_header(client, "natl@x.com")

    def inv(title, lat, lon):
        r = client.post(f"{API}/investigations", headers=nh,
                        json={"title": title, "centroid_lat": lat, "centroid_lon": lon})
        assert r.status_code == 201, r.text
        return r.json()

    def anom(lat, lon):
        r = client.post(f"{API}/anomalies", headers=nh,
                        json={"type": "OIL_LIKE", "confidence": 0.9, "lat": lat, "lon": lon})
        assert r.status_code == 201, r.text
        return r.json()

    return {
        "nh": nh,
        "ph": auth_header(client, "pilot.mh@x.com"),
        "rh": auth_header(client, "regional.west@x.com"),
        "mh_inv": inv("Mumbai slick", 18.7, 72.4),
        "kl_inv": inv("Kerala slick", 9.9, 75.8),
        "gj_inv": inv("Gujarat slick", 22.0, 69.5),
        "mh_anom": anom(18.7, 72.4),
        "kl_anom": anom(9.9, 75.8),
    }


def test_pilot_sees_only_their_zone(client, rbac_world):
    w = rbac_world
    assert client.get(f"{API}/investigations/{w['mh_inv']['id']}", headers=w["ph"]).status_code == 200
    assert client.get(f"{API}/investigations/{w['kl_inv']['id']}", headers=w["ph"]).status_code == 403
    listed = client.get(f"{API}/investigations", headers=w["ph"]).json()
    assert {i["id"] for i in listed} == {w["mh_inv"]["id"]}


def test_pilot_cannot_read_out_of_zone_anomaly(client, rbac_world):
    w = rbac_world
    assert client.get(f"{API}/anomalies/{w['mh_anom']['id']}", headers=w["ph"]).status_code == 200
    assert client.get(f"{API}/anomalies/{w['kl_anom']['id']}", headers=w["ph"]).status_code == 403
    listed = client.get(f"{API}/anomalies", headers=w["ph"]).json()
    assert {a["id"] for a in listed} == {w["mh_anom"]["id"]}


def test_regional_sees_its_whole_region(client, rbac_world):
    w = rbac_world
    assert client.get(f"{API}/investigations/{w['mh_inv']['id']}", headers=w["rh"]).status_code == 200
    assert client.get(f"{API}/investigations/{w['gj_inv']['id']}", headers=w["rh"]).status_code == 200
    assert client.get(f"{API}/investigations/{w['kl_inv']['id']}", headers=w["rh"]).status_code == 403
    listed = client.get(f"{API}/investigations", headers=w["rh"]).json()
    assert {i["id"] for i in listed} == {w["mh_inv"]["id"], w["gj_inv"]["id"]}


def test_national_sees_everything(client, rbac_world):
    w = rbac_world
    listed = client.get(f"{API}/investigations", headers=w["nh"]).json()
    assert {i["id"] for i in listed} == {w["mh_inv"]["id"], w["kl_inv"]["id"], w["gj_inv"]["id"]}
    assert client.get(f"{API}/anomalies/{w['kl_anom']['id']}", headers=w["nh"]).status_code == 200


def test_pilot_cannot_create_outside_its_zone(client, rbac_world):
    w = rbac_world
    out = client.post(f"{API}/investigations", headers=w["ph"],
                      json={"title": "x", "centroid_lat": 9.9, "centroid_lon": 75.8})
    assert out.status_code == 403
    ok = client.post(f"{API}/investigations", headers=w["ph"],
                     json={"title": "in zone", "centroid_lat": 18.9, "centroid_lon": 72.3})
    assert ok.status_code == 201
    # jurisdiction auto-resolved to the Maharashtra zone
    mh_id = None
    for z in client.get(f"{API}/jurisdictions", headers=w["nh"]).json():
        if z["code"] == "IN-MH":
            mh_id = z["id"]
    assert ok.json()["jurisdiction_id"] == mh_id


def test_anomaly_jurisdictions_endpoint(client, rbac_world):
    w = rbac_world
    r = client.get(f"{API}/anomalies/{w['mh_anom']['id']}/jurisdictions", headers=w["nh"])
    assert r.status_code == 200
    assert r.json()["primary_code"] == "IN-MH"
    assert r.json()["chain_codes"] == ["IN-MH", "IN-WEST", "IN-NATIONAL"]


def test_alert_feed_is_role_and_jurisdiction_scoped(client, rbac_world):
    w = rbac_world
    client.post(f"{API}/alerts/dispatch", headers=w["nh"],
                json={"recipient": "+15005550006", "message": "MH oil-like anomaly",
                      "lat": 18.7, "lon": 72.4})
    client.post(f"{API}/alerts/dispatch", headers=w["nh"],
                json={"recipient": "+15005550006", "message": "KL oil-like anomaly",
                      "lat": 9.9, "lon": 75.8})
    # PILOT is below the REGIONAL role floor for the alert feed
    assert client.get(f"{API}/alerts", headers=w["ph"]).status_code == 403
    reg_feed = client.get(f"{API}/alerts", headers=w["rh"]).json()
    assert [a["message"] for a in reg_feed] == ["MH oil-like anomaly"]
    nat_feed = client.get(f"{API}/alerts", headers=w["nh"]).json()
    assert len(nat_feed) == 2


# =========================================================================== #
# SMS provider switching
# =========================================================================== #
def test_default_provider_is_mock(sms_mock):
    prov = sms_svc.get_sms_provider()
    assert prov is sms_mock and prov.name == "mock"
    res = prov.send(to="+15005550006", body="hi")
    assert res.ok and res.status == "MOCKED"


def test_twilio_provider_selected_when_configured(monkeypatch):
    monkeypatch.setattr(settings, "SMS_PROVIDER", "twilio")
    monkeypatch.setattr(settings, "TWILIO_ACCOUNT_SID", "AC_test")
    monkeypatch.setattr(settings, "TWILIO_AUTH_TOKEN", "tok_test")
    monkeypatch.setattr(settings, "TWILIO_FROM_NUMBER", "+15005550000")
    sms_svc.reset_sms_provider()
    try:
        prov = sms_svc.get_sms_provider()
        assert isinstance(prov, sms_svc.TwilioSmsProvider)
        # twilio package is not installed -> send returns a FAILED result, does not raise
        res = prov.send(to="+15005550006", body="hi")
        assert res.ok is False and res.status == "FAILED" and res.error
    finally:
        sms_svc.reset_sms_provider()


def test_twilio_without_credentials_raises(monkeypatch):
    monkeypatch.setattr(settings, "SMS_PROVIDER", "twilio")
    monkeypatch.setattr(settings, "TWILIO_ACCOUNT_SID", None)
    sms_svc.reset_sms_provider()
    try:
        with pytest.raises(RuntimeError, match="TWILIO_ACCOUNT_SID"):
            sms_svc.get_sms_provider()
    finally:
        sms_svc.reset_sms_provider()


# =========================================================================== #
# Fingerprint deduplication
# =========================================================================== #
def test_fingerprint_is_deterministic_and_sensitive():
    fp1, _ = sms_svc.alert_fingerprint(lat=18.70, lon=72.40, occurred_at=NOW)
    fp2, _ = sms_svc.alert_fingerprint(lat=18.70, lon=72.40, occurred_at=NOW)
    assert fp1 == fp2

    fp_far, _ = sms_svc.alert_fingerprint(lat=19.20, lon=72.40, occurred_at=NOW)   # >grid
    assert fp_far != fp1

    fp_later, _ = sms_svc.alert_fingerprint(lat=18.70, lon=72.40,
                                            occurred_at=NOW + timedelta(hours=3))
    assert fp_later != fp1


def test_dispatch_suppresses_a_repeat_within_cooldown(db, sms_mock):
    kw = dict(recipient="+15005550006", lat=18.70, lon=72.40, occurred_at=NOW)
    a1 = sms_svc.dispatch_alert(db, message="Oil-like anomaly off Mumbai", **kw)
    a2 = sms_svc.dispatch_alert(db, message="Oil-like anomaly off Mumbai again", **kw)
    assert a1.status == AlertStatus.MOCKED
    assert a2.status == AlertStatus.SUPPRESSED
    assert a2.dedup_of_id == a1.id
    assert a2.provider_response["deduplicated"] is True
    assert len(sms_mock.outbox) == 1


def test_dispatch_different_location_is_not_deduped(db, sms_mock):
    a1 = sms_svc.dispatch_alert(db, message="m", recipient="+1", lat=18.7, lon=72.4, occurred_at=NOW)
    a2 = sms_svc.dispatch_alert(db, message="m", recipient="+1", lat=19.3, lon=72.4, occurred_at=NOW)
    assert a1.status == AlertStatus.MOCKED and a2.status == AlertStatus.MOCKED
    assert len(sms_mock.outbox) == 2


def test_dispatch_force_bypasses_dedup(db, sms_mock):
    kw = dict(recipient="+1", lat=18.7, lon=72.4, occurred_at=NOW, message="m")
    sms_svc.dispatch_alert(db, **kw)
    forced = sms_svc.dispatch_alert(db, force=True, **kw)
    assert forced.status == AlertStatus.MOCKED
    assert len(sms_mock.outbox) == 2


def test_dispatch_writes_a_full_audit_trail(db, sms_mock):
    a = sms_svc.dispatch_alert(db, message="m", recipient="+1", lat=18.7, lon=72.4, occurred_at=NOW)
    assert a.attempts == 1
    assert a.provider == "mock"
    assert a.sent_at is not None
    assert len(a.error_log) == 1
    assert a.error_log[0]["status"] == "MOCKED"
    assert a.fingerprint and a.time_bucket


# =========================================================================== #
# Failure + retry
# =========================================================================== #
def test_failed_alert_can_be_retried(db, sms_mock):
    sms_mock.fail_next = True
    a = sms_svc.dispatch_alert(db, message="m", recipient="+1", lat=18.7, lon=72.4, occurred_at=NOW)
    assert a.status == AlertStatus.FAILED
    assert a.attempts == 1
    assert a.last_error and len(a.error_log) == 1

    a2 = sms_svc.retry_alert(db, a.id)
    assert a2.status == AlertStatus.MOCKED
    assert a2.attempts == 2
    assert a2.last_error is None
    assert len(a2.error_log) == 2


def test_retry_respects_the_attempt_cap(db, sms_mock):
    sms_mock.fail_next = True
    a = sms_svc.dispatch_alert(db, message="m", recipient="+1", lat=18.7, lon=72.4, occurred_at=NOW)
    capped = sms_svc.retry_alert(db, a.id, max_attempts=1)   # already at 1 attempt
    assert capped.status == AlertStatus.FAILED
    assert "cap" in (capped.last_error or "")
    assert capped.error_log[-1]["status"] == "SKIPPED"


def test_retry_endpoint_requires_regional(client):
    make_user(client, email="p.retry@x.com", role="PILOT", jurisdiction_codes=["IN-MH"])
    make_user(client, email="r.retry@x.com", role="REGIONAL", jurisdiction_codes=["IN-WEST"])
    ph, rh = auth_header(client, "p.retry@x.com"), auth_header(client, "r.retry@x.com")
    assert client.post(f"{API}/alerts/1/retry", headers=ph).status_code == 403
    # nothing to retry -> 404, but the role check passed
    assert client.post(f"{API}/alerts/999999/retry", headers=rh).status_code == 404


# =========================================================================== #
# test-sms endpoint
# =========================================================================== #
def test_test_sms_delivers_to_the_users_own_number(client):
    user = make_user(client, email="phone.user@x.com", role="PILOT",
                     jurisdiction_codes=["IN-MH"], phone_number="+15005550123")
    h = auth_header(client, "phone.user@x.com")
    r = client.post(f"{API}/alerts/test-sms", headers=h)
    assert r.status_code == 201
    body = r.json()
    assert body["status"] == "MOCKED"
    assert body["recipient"] == "+15005550123"
    assert body["recipient_user_id"] == user["id"]
    assert body["attempts"] == 1


def test_test_sms_needs_a_phone_number_on_file(client):
    make_user(client, email="nophone@x.com", role="NATIONAL")
    h = auth_header(client, "nophone@x.com")
    r = client.post(f"{API}/alerts/test-sms", headers=h)
    assert r.status_code == 400
    assert "phone" in r.json()["detail"].lower()


def test_test_sms_requires_authentication(client):
    assert client.post(f"{API}/alerts/test-sms").status_code == 401


def test_test_sms_bypasses_dedup(client):
    make_user(client, email="twice@x.com", role="NATIONAL", phone_number="+15005559999")
    h = auth_header(client, "twice@x.com")
    a = client.post(f"{API}/alerts/test-sms", headers=h).json()
    b = client.post(f"{API}/alerts/test-sms", headers=h).json()
    assert a["status"] == "MOCKED" and b["status"] == "MOCKED"
    assert b["dedup_of_id"] is None


def test_test_sms_accepts_a_custom_message(client):
    make_user(client, email="custom@x.com", role="NATIONAL", phone_number="+15005558888")
    h = auth_header(client, "custom@x.com")
    r = client.post(f"{API}/alerts/test-sms", headers=h, json={"message": "delivery check 42"})
    assert r.status_code == 201
    assert r.json()["message"] == "delivery check 42"
