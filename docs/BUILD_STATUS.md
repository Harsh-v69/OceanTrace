# BUILD STATUS — SIH26143 Unified Prototype

**Project:** OceanTrace — one FastAPI application for satellite
oil-spill detection + AIS vessel attribution.
**Problem statement:** SIH26143 (NTRO).
**Status date:** 2026-09-07
**Host:** Windows 11, Python 3.13.7, CPU only (no GPU), fully offline.

> Phases 1–9 delivered the prototype. A follow-on issues backlog is being worked
> as **Epics** (see §7). Epic 1 (RBAC, hierarchical user management, vessel
> tracking, OceanTrace rebrand) is **complete**.

---

## 1. Phase gate summary

| Phase | Scope | Tests | Status |
|---|---|---|---|
| **1 — Repository Audit** | `docs/MERGE_ARCHITECTURE.md`: component comparison, per-capability selection, dependency/model conflicts, runtime implications | n/a | **PASSED** |
| **2 — Foundation & Backend** | `backend/` FastAPI skeleton, SQLAlchemy models (User/Jurisdiction/Investigation/Anomaly/Vessel/Alert), JWT + bcrypt auth, RBAC (PILOT<REGIONAL<NATIONAL) | 38 | **PASSED** |
| **3 — SAR Detection** | `backend/ml/sar/` — Refined-Lee → adaptive dark-spot → 30 features → RF+GB ensemble → look-alike filter. Strict labels. CPU, no U-Net. | 28 (66 cum.) | **PASSED** |
| **4 — Drift / Hindcast / Forecast** | `backend/ml/drift/` — Lagrangian RK4, backward origin reconstruction + release window, 48 h forward + coastal contact. Deterministic met-ocean provider. | 30 (96 cum.) | **PASSED** |
| **5 — Baseline Attribution** | `backend/ml/attribution/` — spatiotemporal consistency, slick-axis alignment, CPA/proximity; pandas-free AIS track processing; transparent 0–1 weighted scorer. | 28 (124 cum.) | **PASSED** |
| **6 — AIS Intelligence + Fusion** | POSEatSea autoencoder (+ **mandatory StandardScaler**) and LSTM (AOI-gated) ported into `backend/ml/`; `fuse_attribution` (7 components); release-time feedback loop; lazy `backend/ml/registry.py`. | 27 (151 cum.) | **PASSED** |
| **7 — Jurisdiction + RBAC + SMS** | `services/jurisdiction.py` (14 GeoJSON maritime zones, point-in-polygon), server-side jurisdiction guards (PILOT out-of-zone → 403), `services/sms.py` (`SmsProvider` + Mock default + Twilio), fingerprint dedup, audit trail + retry, `/alerts/test-sms`. | 39 (190 cum.) | **PASSED** |
| **8 — Operations Console + Scenarios + Dossier** | `backend/simulator/` (4 deterministic scenarios), `services/orchestration.py` (full pipeline chain), `services/dossier.py` (15-section evidence report, JSON/MD/HTML), `api/v1/{scenarios,evidence}.py`, vanilla-JS SPA under `backend/static/` (12 role-filtered views), static mount at `/app`. | 5 E2E (195 cum.) | **PASSED** |
| **9 — Acceptance & Performance** | Full-suite validation, CPU profiling, lazy-load verification, 24-point acceptance test, `run.py`, this document. | +2 (197 cum.) | **PASSED** |

---

## 2. Phase 9 — test suite

```
$ python -m pytest -q
197 passed in ~130s
```

| Module | Tests | Area |
|---|---:|---|
| `test_database.py` | 5 | schema init, session lifecycle |
| `test_security.py` | 7 | bcrypt hashing, JWT round-trip |
| `test_auth.py` | 11 | register / login / me / protected routes |
| `test_rbac.py` | 7 | role hierarchy, min-role guards |
| `test_endpoints.py` | 3 | health / info / index |
| `test_system.py` | 5 | root redirect → `/app/`, OpenAPI |
| `test_sar_pipeline.py` | 28 | preprocess, dark-spot, ensemble, look-alike, labels, characterisation |
| `test_drift.py` | 30 | RK4 accuracy, hindcast convergence, forecast bounds, met-ocean fallback |
| `test_attribution.py` | 28 | spatiotemporal, axis, CPA, gating, weight normalisation, no-vessel edge cases |
| `test_attribution_ai.py` | 27 | **StandardScaler enforcement**, AE classification, LSTM AOI gate, fusion determinism, feedback loop |
| `test_jurisdiction_rbac_alerts.py` | 39 | point-in-polygon chains, PILOT zone isolation (403), Twilio/Mock switching, fingerprint dedup, test-sms |
| `test_e2e_workflow.py` | 5 | full lifecycle; PILOT-out-of-zone 403; look-alike raises no alert; every scenario ranks its true culprit |
| `test_acceptance.py` | 2 | the 24-point checklist + stage-timing contract |

Zero regressions. Two deprecation warnings only (`httpx`/`starlette` TestClient
shim — upstream, not our code).

---

## 3. Phase 9 — 24-point acceptance test

```
$ python scripts/acceptance.py
24/24 checkpoints passed
```

In-process (`TestClient`), fully offline, one CPU core:

| # | Checkpoint | Evidence |
|---:|---|---|
| 1 | Application starts; maritime zones seeded | `/system/health` → `database: up`; 14 zones |
| 2 | Register a PILOT user | `POST /auth/register` → 201, role PILOT, zone `IN-MH` |
| 3 | PILOT login issues a JWT | `POST /auth/login` → 200; `/auth/me` role PILOT |
| 4 | Investigation list is jurisdiction-scoped | `GET /investigations` scoped to closure |
| 5 | **RBAC — PILOT blocked outside zone (403)** | PILOT@`IN-KL` → Mumbai scenario = **403** |
| 6 | Start deterministic spill scenario | `POST /scenarios/mumbai-high-confidence/run` → 201 |
| 7 | SAR detection → "Oil-like anomaly" | confidence 0.998 |
| 8 | Look-alike dark patch filtered out | `lookalike-darkpatch` → `Likely look-alike`, **no alert** |
| 9 | Spill geometry measured | area 75.7 km², centroid, major/minor axis, orientation 125° |
| 10 | Drift hindcast reconstructs a release origin | origin 18.61, 72.11 (±15.7 km) |
| 11 | Release point + time window | window T−28.3 h … T−0.5 h |
| 12 | AIS tracks normalised + correlated | 3 in → 1 kept after the traffic gate |
| 13 | AIS autoencoder scores candidates | AE score reported; **threshold 1.104481, StandardScaler enforced** |
| 14 | LSTM route-deviation component present | component present; `usable=False` here (correctly AOI-gated outside Mauritius) |
| 15 | Fusion ranking deterministic; prime suspect | prime MMSI 419810001 identical across runs; ground-truth top-1 = **True** |
| 16 | Evidence explanation (why #1) | `why_ranked_first` sorted by points; top reason `spatiotemporal` |
| 17 | Jurisdiction mapping deterministic | `IN-MH → IN-WEST → IN-NATIONAL` |
| 18 | Designated-user lookup on the alert | alert → `recipient_user_id`, registered mobile |
| 19 | Real/Mock Twilio SMS delivery status | status `MOCKED`; message in the Mock outbox |
| 20 | Stage timings in investigation metadata | 9 stage keys + `total_ms` present |
| 21 | **ML models lazy-load and cache** | registry: 0 resident at boot → AE + LSTM + SAR resident after first run |
| 22 | Evidence dossier — JSON | 15 top-level sections |
| 23 | Evidence dossier — Markdown | `text/plain`, ~3.9 KB |
| 24 | Evidence dossier — HTML (printable → PDF) | `text/html`, ~5.6 KB |

For the look-alike (Goa/Karnataka) and ambiguous (Maharashtra offshore) and
Wakashio (Mauritius) scenarios the same checks were repeated via
`test_e2e_workflow.py`: every scenario with a known culprit ranks it #1, and the
look-alike raises no alert.

---

## 4. Phase 9 — CPU performance profile

```
$ python scripts/profile_pipeline.py --repeat 3
```

Single CPU core, no GPU, deterministic simulated met-ocean fields.

### 4.1 Model lazy-loading

| Model | Params | First-load (cold) | Resident at boot? |
|---|---:|---:|---|
| SAR Oil/Look-alike RF+GB ensemble | ensemble | ~2.5 s | **no** — loads on first SAR scene |
| AIS Anomaly Autoencoder (+ StandardScaler) | 735 | ~4–6 s | **no** — loads on first AIS fusion |
| Trajectory LSTM | 201,986 | ~0.02 s | **no** — loads on first fusion |

`python run.py --check` and `GET /system/models` both report **0 models
resident** until an investigation needs them; thereafter each stays cached in
memory for the life of the process. A session that only browses the map / lists
/ dossiers never imports torch.

### 4.2 Per-stage timings (warm mean, milliseconds)

| scenario | ingest | preproc | detect | charac | hindcast+fcst | AIS | fusion | juris | alert | **total** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mumbai-high-confidence | 20 | 90 | 397 | 84 | 1314 | 596 | 3376 | 2 | 9 | **5924** |
| lookalike-darkpatch | 19 | 87 | 334 | 65 | 0 | 0 | 0 | 3 | 0 | **512** |
| ambiguous-drift | 21 | 99 | 456 | 100 | 1436 | 839 | 4751 | 3 | 9 | **7751** |
| wakashio-mauritius | 21 | 83 | 249 | 25 | 1356 | 1896 | 10741 | 3 | 9 | **14432** |

Run-to-run variance on `attribution_fusion` is ±5–8 % (numpy threading).
Cold (first run of a process, model load folded in): mumbai ~14 s, lookalike
~0.55 s, ambiguous ~7.6 s, wakashio ~15.9 s.

### 4.3 Assessment against latency budgets

| Class of operation | Budget | Measured | Verdict |
|---|---|---|---|
| Interactive endpoints (auth, health, jurisdictions, lists, investigation read, dossier JSON/MD/HTML) | < 1.5 s | all sub-second (covered by the test suite) | **OK** |
| Look-alike rejection (short-circuit) | < 1 s | 0.57 s | **OK** |
| Typical full investigation (SAR + drift feedback + fusion) | ≤ 10 s | 5.9–7.8 s | **OK** |
| Heaviest forensic scenario (Wakashio) | ≤ 20 s | 14.4 s | **OK** |

**Where the time goes.** For a full investigation, `attribution_fusion`
dominates. It runs the unified 7-component scorer *and* the release-time
feedback loop, which re-runs the RK4 hindcast once per iteration
(`max_iterations=2`, `n_particles=280`). Wakashio is the outlier because its
culprit is a two-segment continuous-release track (approach + grounding) at 60 s
AIS cadence over ~16 h — the longest track and the longest back-tracking
interval of any scenario. The spatiotemporal volume search was vectorised in
Phase 9 (bit-identical results, ~10× faster on long tracks); the residual cost
is the repeated physics hindcasts, which is inherent to the feedback method and
is the price of moving the origin error from ~30 km (physics only) to ~3 km
(AIS-constrained).

A full investigation is a **triggered batch job with a live progress UI** (the
Live Monitoring view animates each stage as it completes), not a blocking
request, so single-digit-to-~15 s on one core with no GPU is within an
acceptable budget for this prototype. No GPU is required anywhere.

---

## 5. Constraint compliance (CLAUDE.md)

| Constraint | Status |
|---|---|
| One unified FastAPI app, no separate Streamlit | ✅ single `backend.main:app`; console served at `/app` by the same process |
| Explicitly label detections "Oil-like anomaly" | ✅ canonical labels enforced (`labels.assert_canonical`), invariant-guarded |
| Real ML metrics exposed (IoU, top-1 attribution) | ✅ SAR metrics on `/system/health`; per-component fusion breakdown in every ranking; ground-truth top-1 in scenario metadata |
| Offline-first local CPU; lazy loading; caching; deterministic met-ocean | ✅ 0 models resident until used; `CachedMetOceanProvider`; `DEMO_FIELD_LABEL`; no network in the core path |
| SQLite, PostGIS-ready | ✅ SQLAlchemy 2.0, geometry stored as GeoJSON, Shapely point-in-polygon |
| JWT/session auth, strict RBAC mapped to GeoJSON boundaries | ✅ PILOT/REGIONAL/NATIONAL; point-in-polygon jurisdiction guards; out-of-zone → 403 |
| Real Twilio SMS with Mock fallback, confidence-triggered | ✅ `SmsProvider` ABC; `MockSmsProvider` default; `TwilioSmsProvider` behind `SMS_PROVIDER=twilio`; dispatch gated on `ALERT_CONFIDENCE_THRESHOLD` |
| Must use POSEatSea's pre-trained StandardScaler | ✅ `_validate_scaler`, single reconstruction-error funnel, regression test proves bypass changes the answer |
| RK4 hindcast + 48 h forecast; AIS feedback refines release time | ✅ `run_hindcast`/`run_forecast`; `attribute_with_feedback` converges origin |

---

## 6. Known limitations carried forward

- Detection / attribution accuracy is validated on the **simulator**; real
  Sentinel-1 performance is lower and uses a separate decision threshold.
- Met-ocean fields are **synthesised**, not real ERA5/HYCOM (`_build_grids()` is
  the swap seam).
- The POSEatSea **LSTM is valid only inside the Mauritius AOI**; elsewhere the
  route-deviation component reports `usable=False` and the heuristic carries the
  signal.
- The AIS **autoencoder nominates** vessels for review (recall ~0.41); it never
  clears one.
- Attribution is an investigative lead, not proof — every dossier carries the
  caveat.
- Live OpenStreetMap basemap tiles need internet; the console falls back to a
  plain ocean canvas offline and stays fully usable.

---

## 7. Post-prototype Epics

### Epic 1 — RBAC, hierarchical user management, vessel tracking, rebrand — **COMPLETE** (2026-09-07)

| Item | What shipped |
|---|---|
| **Rebrand → OceanTrace** | Project-wide replace of "SAMUDRA NETRA" across `backend/` (config `APP_NAME`, ML docstrings, static console HTML/JS/CSS), `run.py`, `scripts/`, `docs/BUILD_STATUS.md`, `docs/DEPLOYMENT.md`, SMS template. `CLAUDE.md` and `docs/MERGE_ARCHITECTURE.md` left intact (they name the two upstream repos being merged). |
| **RBAC hardening** | The engine was already sound (39 Phase-7 tests). Added: alert visibility scoped by `jurisdiction_codes` even with null lat/lon (`user_can_access_any_code`, `filter_by_jurisdiction(code_attr=…)`); `test_rbac_hardening.py` (6 tests) — single-resource 403, REGIONAL closure boundary, empty-assignment → 403, code-scoped alerts, min-role ladder. |
| **Hierarchical user management** | `ALLOW_OPEN_REGISTRATION=False` by default → `POST /auth/register` returns 403. New `api/v1/users.py`: `GET/POST /users`, `GET/PATCH /users/{id}`, `/users/{id}/disable|enable`, `/users/me/scope`. NATIONAL manages any account; REGIONAL manages PILOTs whose zones ⊆ its closure; PILOT no access. Last-active-NATIONAL and self-management guards. `seed_default_users()` (idempotent) — `national@ / regional@ / pilot@oceantrace.gov.in`, password `DEFAULT_USER_PASSWORD`. Console: **User Management** view (REGIONAL+), Register tab hidden unless `open_registration`. `test_user_management.py` (12 tests). |
| **Vessel tracking** | Orchestrator now persists a `Vessel` row per candidate, links each to its FUSION `Anomaly` (so `/vessels` is jurisdiction-scoped), and stashes a map-ready `summary_metrics.vessel_tracks[mmsi]` — decimated pings, loiter spans, AIS-blackout gaps, per-track metrics, and the fused AE / route-deviation result. New `services/vessels.py` helpers (`persist_vessels`, `track_view`, `build_track_views`); `GET /vessels/{mmsi}?investigation_id=`, `GET /vessels/{mmsi}/track`. Console: Mission Control draws the prime suspect's track; Workstation draws every candidate track (prime red, loiter = amber ring, blackout = dashed red), a click populates a detail panel, and the T-48h…+48h slider walks each vessel to its interpolated position. `test_vessel_tracking.py` (6 tests). |
| **Console cache** | `/app` now served with `Cache-Control: no-cache, must-revalidate` + versioned module imports so an updated build is never masked by a stale browser copy. |

```
$ python -m pytest -q          →  221 passed
$ python scripts/acceptance.py  →  26/26 checkpoints passed
```

Acceptance grew from 24 → 26 (open-registration-closed + hierarchical creation;
vessel-tracking engine; REGIONAL-scoped management). `test_acceptance.py` updated.

**Epics 2–4** (drift accuracy + land collision + offline basemap; real met-ocean
+ Twilio + upload endpoints; PostGIS + Alembic + weathering) are queued; several
have external dependencies (coastline dataset, Copernicus CDS key, Twilio
account, a PostGIS instance) that must be supplied.

---

## 8. Sign-off

Phases 1–9 **PASSED** and Epic 1 **COMPLETE**. **221/221** automated tests green.
**26/26** acceptance checkpoints green. Lazy-loading verified. Performance within
tiered budgets on a single CPU core with no GPU and no network.

See `docs/DEPLOYMENT.md` for run and demo instructions.
