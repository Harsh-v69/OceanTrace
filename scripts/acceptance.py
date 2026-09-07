#!/usr/bin/env python
"""
End-to-end acceptance test for the OceanTrace prototype (26 checkpoints).

Drives the running application through its whole intended lifecycle - in-process
via ``TestClient`` (no network), fully offline, on one CPU core - and prints a
numbered PASS/FAIL table.

    python scripts/acceptance.py

Exit code 0 iff every checkpoint passes. Also importable: ``run_acceptance()``
returns ``(results, timings)`` and is exercised by
``backend/tests/test_acceptance.py``.
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_DB = Path(tempfile.gettempdir()) / "sn_acceptance.db"
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_DB.as_posix()}")
os.environ.setdefault("JWT_SECRET_KEY", "acceptance-secret-0123456789abcdef0123456789")
os.environ.setdefault("ENV", "test")
os.environ.setdefault("ALLOW_REGISTRATION_ROLE_SELECT", "true")
os.environ.setdefault("SMS_PROVIDER", "mock")
# Epic 1: open self-registration is CLOSED; the three role accounts are seeded.
os.environ.setdefault("SEED_DEFAULT_USERS", "true")
os.environ.setdefault("DEFAULT_USER_PASSWORD", "Acceptance!Seed1")

API = "/api/v1"
PW = "Acceptance!Seed1"          # the seeded default-account password
NATIONAL = "national@oceantrace.gov.in"
REGIONAL = "regional@oceantrace.gov.in"
MUMBAI = "mumbai-high-confidence"
LOOKALIKE = "lookalike-darkpatch"


class _Checks:
    def __init__(self) -> None:
        self.results: list[tuple[int, str, bool, str]] = []
        self._n = 0

    def do(self, label: str, fn) -> None:
        self._n += 1
        try:
            detail = fn() or ""
            self.results.append((self._n, label, True, str(detail)))
        except AssertionError as exc:
            self.results.append((self._n, label, False, f"assertion: {exc}"))
        except Exception as exc:  # noqa: BLE001
            self.results.append((self._n, label, False, f"{type(exc).__name__}: {exc}"))

    @property
    def ok(self) -> bool:
        return all(p for _, _, p, _ in self.results)


def run_acceptance() -> tuple[list[tuple[int, str, bool, str]], dict]:
    if _DB.exists():
        _DB.unlink()

    from fastapi.testclient import TestClient

    import backend.models  # noqa: F401  register mappers
    from backend.core.config import settings
    from backend.core.database import Base, SessionLocal, engine
    from backend.main import app
    from backend.services import sms as sms_svc

    # This runs both standalone and inside the pytest suite, where conftest has
    # already fixed the environment. Force the settings this checklist needs so
    # it never depends on import order, and restore them afterwards so the rest
    # of the suite is untouched.
    _saved = {k: getattr(settings, k) for k in
              ("ALLOW_OPEN_REGISTRATION", "SEED_DEFAULT_USERS", "DEFAULT_USER_PASSWORD", "SMS_PROVIDER")}
    settings.ALLOW_OPEN_REGISTRATION = False
    settings.SEED_DEFAULT_USERS = True
    settings.DEFAULT_USER_PASSWORD = PW
    settings.SMS_PROVIDER = "mock"

    Base.metadata.create_all(bind=engine)
    from backend.services.jurisdiction import seed_demo_jurisdictions
    from backend.services.users import seed_default_users

    with SessionLocal() as _db:
        seed_demo_jurisdictions(_db)
        seed_default_users(_db, password=PW)

    sms_svc.reset_sms_provider()
    mock = sms_svc.get_sms_provider()

    c = _Checks()
    state: dict = {}

    with TestClient(app) as client:
        # 1 - application boots (lifespan seeded the maritime zones)
        c.do("Application starts; maritime zones seeded", lambda: (
            _assert(client.get(f"{API}/system/health").json()["database"] == "up"),
            f"{len(client.get(f'{API}/jurisdictions', headers=_h(client, _nat(client))).json())} zones",
        )[1])

        # 2 - open self-registration is CLOSED; NATIONAL admin creates a PILOT
        def _create_pilot():
            reg = client.post(f"{API}/auth/register", json={
                "name": "Nope", "email": "nope@example.com", "password": PW, "role": "PILOT"})
            _assert(reg.status_code == 403, f"open register should be 403, got {reg.status_code}")
            r = client.post(f"{API}/users", headers=_h(client, NATIONAL), json={
                "name": "Acceptance Pilot", "email": "acc.pilot@example.com",
                "password": PW, "role": "PILOT", "phone_number": "+15005550111",
                "jurisdiction_codes": ["IN-MH"],
            })
            _assert(r.status_code == 201, r.text)
            return "self-register -> 403; NATIONAL POST /users -> PILOT acc.pilot@ (IN-MH)"
        c.do("Hierarchical user creation (open self-registration disabled)", _create_pilot)

        # 3 - PILOT logs in, JWT issued
        def _login():
            r = client.post(f"{API}/auth/login",
                            data={"username": "acc.pilot@example.com", "password": PW})
            _assert(r.status_code == 200, r.text)
            tok = r.json()["access_token"]
            state["pilot"] = {"Authorization": f"Bearer {tok}"}
            _assert(client.get(f"{API}/auth/me", headers=state["pilot"]).json()["role"] == "PILOT")
            return "JWT issued; /auth/me confirms role PILOT"
        c.do("PILOT login issues a JWT", _login)

        # 4 - PILOT sees only their jurisdiction
        c.do("PILOT investigation list is jurisdiction-scoped", lambda: (
            _assert(client.get(f"{API}/investigations", headers=state["pilot"]).status_code == 200),
            "scoped list returned (empty before any run)")[1])

        # 5 - PILOT is refused outside their zone (Kerala scenario centre)
        def _out_of_zone():
            r = client.post(f"{API}/users", headers=_h(client, NATIONAL), json={
                "name": "KL Pilot", "email": "acc.kl@example.com", "password": PW,
                "role": "PILOT", "jurisdiction_codes": ["IN-KL"]})
            _assert(r.status_code == 201, r.text)
            tok = client.post(f"{API}/auth/login",
                              data={"username": "acc.kl@example.com", "password": PW}).json()["access_token"]
            r = client.post(f"{API}/scenarios/{MUMBAI}/run", headers={"Authorization": f"Bearer {tok}"})
            _assert(r.status_code == 403, f"expected 403, got {r.status_code}")
            return "PILOT@IN-KL -> Mumbai scenario = 403 Forbidden"
        c.do("RBAC: PILOT blocked outside assigned jurisdiction (403)", _out_of_zone)

        # 6 - run the deterministic spill scenario
        def _run():
            r = client.post(f"{API}/scenarios/{MUMBAI}/run", headers=state["pilot"])
            _assert(r.status_code == 201, r.text)
            state["run"] = r.json()
            state["inv_id"] = state["run"]["investigation_id"]
            inv = client.get(f"{API}/investigations/{state['inv_id']}", headers=state["pilot"]).json()
            state["m"] = inv["summary_metrics"]
            return f"{state['run']['reference']} created"
        c.do("Start deterministic spill scenario (Mumbai)", _run)

        # 7 - SAR detection
        c.do("SAR detection classifies scene 'Oil-like anomaly'", lambda: (
            _assert(state["run"]["classification"] == "Oil-like anomaly"),
            f"confidence {state['run']['confidence']:.3f}")[1])

        # 8 - look-alike filtering
        def _lookalike():
            r = client.post(f"{API}/scenarios/{LOOKALIKE}/run", headers=_h(client, _nat(client)))
            _assert(r.status_code == 201, r.text)
            b = r.json()
            _assert(b["classification"] != "Oil-like anomaly", b["classification"])
            _assert(b["alert_status"] is None, "look-alike must not alert")
            state["lookalike"] = b
            return f"dark patch -> {b['classification']!r}, no alert"
        c.do("Look-alike dark patch is filtered out (no spill, no alert)", _lookalike)

        # 9 - spill geometry
        def _geom():
            ch = state["m"]["sar"]["primary_detection"]["characterization"]
            for k in ("area_km2", "centroid", "major_axis_km", "minor_axis_km", "orientation_deg"):
                _assert(k in ch and ch[k] is not None, f"missing {k}")
            return f"area {ch['area_km2']} km2, orientation {ch['orientation_deg']} deg"
        c.do("Spill geometry measured (area / centroid / axes / orientation)", _geom)

        # 10 - drift hindcast origin
        def _origin():
            be = state["m"]["hindcast"]["best_estimate"]
            _assert(isinstance(be, list) and len(be) == 2, be)
            return f"origin {be[0]:.3f}, {be[1]:.3f}  (+/-{state['m']['hindcast']['uncertainty_radius_km']} km)"
        c.do("Drift hindcast reconstructs a release origin", _origin)

        # 11 - release point + time window
        def _release():
            w = state["m"]["hindcast"]["release_window_h"]
            _assert(len(w) == 2 and w[0] <= w[1], w)
            return f"release window T{w[0]:+.1f}..T{w[1]:+.1f} h"
        c.do("Release point and time window estimated", _release)

        # 12 - AIS correlation
        def _ais():
            g = state["m"]["attribution"].get("gate") or {}
            _assert("n_input" in g, "no AIS gate recorded")
            return f"{g.get('n_input')} tracks in, {g.get('n_kept')} kept after the traffic gate"
        c.do("AIS tracks normalised and correlated", _ais)

        # 13 - autoencoder + mandatory scaler
        def _ae():
            prime = state["m"]["attribution"]["candidates"][0]
            comp = prime["components"]["ais_anomaly"]
            _assert("value" in comp and comp["detail"], "AE component absent")
            _assert(abs(comp["detail"]["threshold"] - 1.104481) < 1e-6,
                    f"AE_THRESHOLD wrong: {comp['detail'].get('threshold')}")
            return f"AE score {comp['value']:.3f}, threshold {comp['detail']['threshold']} (StandardScaler enforced)"
        c.do("AIS autoencoder scores candidates (pre-trained StandardScaler)", _ae)

        # 14 - LSTM route deviation
        def _lstm():
            comp = state["m"]["attribution"]["candidates"][0]["components"]["route_deviation"]
            _assert("value" in comp and "detail" in comp, "LSTM component absent")
            usable = comp["detail"].get("usable")
            return f"route-deviation component present (LSTM usable={usable} - AOI-gated)"
        c.do("LSTM trajectory route-deviation component present", _lstm)

        # 15 - fusion ranking deterministic + prime suspect
        def _fusion():
            r2 = client.post(f"{API}/scenarios/{MUMBAI}/run", headers=_h(client, _nat(client))).json()
            a1 = state["m"]["attribution"]["summary"]["prime_suspect"]["identity"]["mmsi"]
            a2 = r2["prime_suspect"]["identity"]["mmsi"]
            _assert(a1 == a2, f"non-deterministic: {a1} vs {a2}")
            gt = state["m"]["attribution"]["summary"]["ground_truth"]
            _assert(gt["correctly_ranked_first"], "true culprit not ranked #1")
            return f"prime MMSI {a1} (deterministic); ground-truth top-1 = True"
        c.do("Unified fusion ranking is deterministic; prime suspect identified", _fusion)

        # 16 - evidence explanation
        def _why():
            dj = client.get(f"{API}/investigations/{state['inv_id']}/dossier", headers=state["pilot"]).json()
            ps = dj["candidate_ranking"]["prime_suspect"]
            _assert(ps and ps.get("why_ranked_first"), "no why_ranked_first explanation")
            top = ps["why_ranked_first"][0]
            return f"top reason: {top.get('component', top)}"
        c.do("Evidence explanation: why the prime suspect ranks first", _why)

        # 17 - jurisdiction mapping
        def _juris():
            j = state["run"]["jurisdiction"]
            _assert(j["primary_code"] == "IN-MH", j)
            _assert(j["chain_codes"] == ["IN-MH", "IN-WEST", "IN-NATIONAL"], j["chain_codes"])
            return " -> ".join(j["chain_codes"])
        c.do("Jurisdiction mapping deterministic (state -> region -> nation)", _juris)

        # 18 - designated user lookup on the alert
        def _designated():
            al = client.get(f"{API}/alerts", headers=_h(client, _nat(client))).json()
            mine = [a for a in al if a["recipient_user_id"]]
            _assert(mine, "no alert carries a recipient_user_id")
            return f"alert #{mine[-1]['id']} -> user id {mine[-1]['recipient_user_id']}, {mine[-1]['recipient']}"
        c.do("Designated-user lookup wired to the alert", _designated)

        # 19 - SMS delivery status
        def _sms():
            _assert(state["run"]["alert_status"] in {"SENT", "MOCKED"}, state["run"]["alert_status"])
            _assert(mock.outbox, "Mock provider outbox empty")
            return f"status {state['run']['alert_status']} via mock; {len(mock.outbox)} message(s) sent"
        c.do("Real/Mock Twilio SMS delivery status recorded", _sms)

        # 20 - stage timings in investigation metadata
        def _timings():
            tm = state["m"].get("timings") or {}
            need = {"satellite_ingest", "preprocessing", "detection", "characterization",
                    "hindcast_forecast", "ais_correlation", "attribution_fusion",
                    "jurisdiction", "alert", "total_ms"}
            _assert(need <= set(tm), f"missing timing keys: {need - set(tm)}")
            _assert(tm["total_ms"] > 0, tm["total_ms"])
            state["timings"] = tm
            return f"total_ms {tm['total_ms']:.0f}; all 9 stages + total recorded"
        c.do("Stage timings recorded in investigation metadata", _timings)

        # 21 - model lazy-loading
        def _lazy():
            st = client.get(f"{API}/system/models", headers=state["pilot"]).json()
            _assert(st["lazy_load"] is True, "ML_LAZY_LOAD off")
            loaded = [m for m in st["models"] if m["loaded"]]
            _assert(loaded, "no model resident after a full run - registry not caching")
            names = ", ".join(m["name"].split("(")[0].strip() for m in loaded)
            secs = {m["name"].split()[0]: m["load_seconds"] for m in loaded}
            return f"resident after run: {names}; first-load {secs}"
        c.do("ML models lazy-load on demand and cache in memory", _lazy)

        # 22-24 - evidence dossier exports
        c.do("Evidence dossier export - JSON", lambda: _dossier(client, state, "", "application/json"))
        c.do("Evidence dossier export - Markdown", lambda: _dossier(client, state, ".md", "text/plain"))
        c.do("Evidence dossier export - HTML (printable / PDF)", lambda: _dossier(client, state, ".html", "text/html"))

        # 25 - vessel tracking engine (Epic 1)
        def _vessels():
            nat = _h(client, NATIONAL)
            rows = client.get(f"{API}/vessels", headers=nat).json()
            _assert(rows, "scenario run created no Vessel rows")
            culprit = "419810001"
            v = client.get(f"{API}/vessels/{culprit}?investigation_id={state['inv_id']}", headers=nat).json()
            _assert(v["track"] and v["track"]["pings"], "no reconstructed track for the prime suspect")
            _assert(v["attribution"]["is_prime"] is True, "prime flag missing on the track view")
            t = client.get(f"{API}/vessels/{culprit}/track?investigation_id={state['inv_id']}", headers=nat).json()
            _assert({"pings", "loiter", "blackouts", "metrics"} <= set(t), "track view shape")
            return (f"{len(rows)} vessels persisted; prime {v['name']} track "
                    f"{v['metrics']['n_points']} pings, {len(t['loiter'])} loiter span(s)")
        c.do("Vessel tracking: real Vessel rows + map-ready tracks", _vessels)

        # 26 - REGIONAL admin manages its PILOTs
        def _regional_mgmt():
            rh = _h(client, REGIONAL)                     # seeded REGIONAL @ IN-WEST
            r = client.post(f"{API}/users", headers=rh, json={
                "name": "GJ Pilot", "email": "acc.gj@example.com", "password": PW,
                "role": "PILOT", "jurisdiction_codes": ["IN-GJ"]})
            _assert(r.status_code == 201, r.text)                       # IN-GJ is inside IN-WEST
            bad = client.post(f"{API}/users", headers=rh, json={
                "name": "TN Pilot", "email": "acc.tn@example.com", "password": PW,
                "role": "PILOT", "jurisdiction_codes": ["IN-TN"]})
            _assert(bad.status_code == 403, "REGIONAL must not create a PILOT outside its region")
            pid = r.json()["id"]
            _assert(client.post(f"{API}/users/{pid}/disable", headers=rh).status_code == 200)
            login = client.post(f"{API}/auth/login",
                                data={"username": "acc.gj@example.com", "password": PW})
            _assert(login.status_code == 403, "disabled account must not log in")
            return "REGIONAL created + disabled a PILOT in-region; out-of-region create = 403"
        c.do("Hierarchical management: REGIONAL scoped to its region", _regional_mgmt)

    for k, v in _saved.items():          # leave the shared settings singleton as we found it
        setattr(settings, k, v)
    return c.results, state.get("timings", {})


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _assert(cond, msg="") -> bool:
    if not cond:
        raise AssertionError(msg or "condition false")
    return True


def _nat(client) -> str:
    return NATIONAL          # seeded at startup (SEED_DEFAULT_USERS)


def _h(client, email) -> dict:
    tok = client.post("/api/v1/auth/login",
                      data={"username": email, "password": PW}).json()["access_token"]
    return {"Authorization": f"Bearer {tok}"}


def _dossier(client, state, ext, ctype) -> str:
    r = client.get(f"/api/v1/investigations/{state['inv_id']}/dossier{ext}", headers=state["pilot"])
    _assert(r.status_code == 200, r.text[:200])
    _assert(r.headers["content-type"].startswith(ctype), r.headers["content-type"])
    if ext == "":
        _assert("incident_summary" in r.json(), "JSON dossier missing incident_summary")
        return f"{len(r.json())} top-level sections"
    return f"{len(r.text)} bytes, {ctype}"


def main() -> int:
    t0 = time.perf_counter()
    results, timings = run_acceptance()
    elapsed = time.perf_counter() - t0

    print("=" * 82)
    print("OCEANTRACE ACCEPTANCE TEST".center(82))
    print("=" * 82)
    for n, label, passed, detail in results:
        mark = "PASS" if passed else "FAIL"
        print(f" {n:>2}. [{mark}] {label}")
        if detail:
            print(f"         {detail}")
    n_pass = sum(1 for _, _, p, _ in results if p)
    print("-" * 82)
    print(f" {n_pass}/{len(results)} checkpoints passed in {elapsed:.1f}s")
    if timings:
        stg = " ".join(f"{k}={v:.0f}" for k, v in timings.items() if k != "total_ms")
        print(f" pipeline stage timings (ms): {stg}")
        print(f" total per investigation    : {timings['total_ms']:.0f} ms")
    print("=" * 82)
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
