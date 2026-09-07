# BUILD STATUS — SIH26143 Unified Prototype

**Project:** OceanTrace — one FastAPI application for satellite
oil-spill detection + AIS vessel attribution.
**Problem statement:** SIH26143 (NTRO).
**Status date:** 2026-09-07
**Host:** Windows 11, Python 3.13.7, CPU only (no GPU), fully offline.

> Phases 1–9 delivered the prototype. A follow-on issues backlog is being worked
> as **Epics** (see §7). Epics 1–4 are **complete**: RBAC + hierarchical user
> management + vessel tracking + OceanTrace rebrand (1), drift/attribution
> accuracy + coastline (2), real feeds + custom ingestion (3, credential-gated:
> Twilio + real met-ocean fall back to Mock/Demo without keys), PostGIS/Alembic
> + oil weathering (4, PostGIS opt-in, verified on SQLite).

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

### 4.2 Per-stage timings (warm mean, milliseconds — after Epic 2)

| scenario | ingest | preproc | detect | charac | hindcast+fcst | AIS | fusion | juris | alert | **total** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mumbai-high-confidence | 12 | 52 | 214 | 41 | 687 | 169 | 959 | 1 | 4 | **2210** |
| lookalike-darkpatch | 11 | 46 | 187 | 34 | 0 | 0 | 0 | 1 | 0 | **286** |
| ambiguous-drift | 10 | 56 | 243 | 53 | 676 | 184 | 1045 | 1 | 4 | **2342** |
| wakashio-mauritius | 18 | 75 | 277 | 30 | 1012 | 313 | 1772 | 1 | 12 | **3658** |

Cold (first run of a process, model load folded in): mumbai ~4.8 s, lookalike
~0.3 s, ambiguous ~2.3 s, wakashio ~2.4 s.

### 4.3 Assessment against latency budgets

| Class of operation | Budget | Measured | Verdict |
|---|---|---|---|
| Interactive endpoints (auth, health, jurisdictions, lists, investigation read, dossier JSON/MD/HTML) | < 1.5 s | all sub-second (covered by the test suite) | **OK** |
| Look-alike rejection (short-circuit) | < 1 s | 0.29 s | **OK** |
| Typical full investigation (SAR + drift feedback + fusion) | ≤ 10 s | 2.2–2.3 s | **OK** |
| Heaviest forensic scenario (Wakashio) | ≤ 20 s | 3.7 s | **OK** |

**Epic 2 made the pipeline faster.** Rewriting `rolling_predictions` to push
every 8-ping LSTM window through the model in **one batched forward pass** (from
a per-window Python loop with per-window `assess_inputs` / pandas slicing) cut
`attribution_fusion` roughly 3×, more than paying for the newly-enabled
route-deviation signal on the Indian scenarios *and* the RK4 land-collision test
per particle-step (kept cheap with a `shapely.prepare`d coastline + a bounding-box
pre-filter). The Wakashio scenario — the heaviest, a two-segment continuous-release
grounding track at 60 s AIS cadence over ~16 h — dropped from ~14 s to ~3.7 s
warm. No GPU is required anywhere.

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

### Epic 2 — drift/attribution accuracy + coastline — **COMPLETE** (2026-09-07)

| Item | What shipped |
|---|---|
| **2.1 Coastline + RK4 land collision** | A supplied India admin boundary (6.6 MB, git-ignored) is simplified once (`shapely.simplify` tol 0.01° ≈ 1.1 km → `backend/static/data/coastline_in.geojson`, 167 KB) and loaded as a `LandMask` (`services/drift.load_indian_coastline()`, cached, `shapely.prepare`d + bbox pre-filter). The orchestrator passes it into both `run_hindcast` and `run_forecast`. `coastal_impact` now returns `first_contact_point`, `first_contact_eta_h`, and a decimated `contact_points` list alongside `fraction_beached`. The Mumbai scenario beaches on the Maharashtra coast in ~17 h; the offshore scenario does not. Shown on the Workstation & Drift maps (red stranding markers + a first-contact popup) and in a new dossier section `forward_forecast.shoreline_contact`. |
| **2.2 LSTM route deviation unlocked** | `AOI_HARD_GATE=False`. `normalize_features`/`denorm_latlon` take a `frame=(lat0,lon0,lat_span,lon_span)`; `frame_for(window)` keeps the fixed AOI frame inside the Mauritius AOI (bit-identical to before) and recentres the **same span** on the window centroid outside it, so the pre-trained LSTM extrapolates anywhere. Out-of-AOI results carry `confidence="degraded"`, `aoi=False`, and a caveat. `rolling_predictions` was rewritten to **batch every 8-ping window through the LSTM in one forward pass** — the Indian scenarios now get a real `route_deviation` component *and the pipeline got ~3× faster* (see §4.3). |
| **2.3 Dwell + exact fusion weights** | `FusionWeights` = the exact operating spec — spatiotemporal 0.30, axis 0.18, CPA 0.14, **dwell 0.10**, blackout 0.10, route‑deviation 0.09, vessel‑prior 0.09 (Σ 1.00). The POSEatSea `dwell` term is a first-class component. `ais_anomaly` is still **computed and shown** as evidence in every ranking/dossier but is **not weighted** (weight 0). |
| **2.4 Offline vector basemap** | When OSM tiles fail (no internet), `map.js` now fetches `data/coastline_in.geojson` and renders it as an `L.geoJSON` land layer so the map still shows a recognisable India outline instead of a blank canvas. |

All 4 scenarios still rank their true culprit #1. New: `test_coastline.py` (5 tests).
Updated: `test_attribution_ai.py`, `test_drift.py`, `test_e2e_workflow.py`.

```
$ python -m pytest -q          →  225 passed  (~108 s, faster than before)
$ python scripts/acceptance.py  →  26/26 checkpoints passed
```

### Epic 3 — real feeds + custom ingestion — **COMPLETE, credential-gated** (2026-09-07)

| Item | What shipped |
|---|---|
| **3.1 Twilio live wiring** | `twilio>=9.0` uncommented in `requirements.txt` and installed. `get_sms_provider()` uses `TwilioSmsProvider` **only when all three `TWILIO_*` credentials are set**; if any is missing (or the package is absent) it logs a warning and returns the Mock provider — the app never breaks over SMS. `GET /system/health` now reports `sms_effective`. A real send still needs a paid Twilio account. |
| **3.2 Real met-ocean** | New `services/metocean_real.py` — ERA5 10 m wind via `cdsapi` (needs `~/.cdsapirc`) + HYCOM GOFS 3.1 surface currents via an `xarray` OPeNDAP read, regridded onto the ERA5 grid. `RealMetOceanProvider`'s default `fetch_fn` is now `fetch_era5_hycom`; `available()` is a fast static probe (`cdsapi` + `xarray` + a netCDF reader + the key file all present) and `get_field` returns `MetOceanUnavailable` on any fetch failure → the deterministic Demo field. Optional deps are commented in `requirements.txt`. `GET /system/health` reports `metocean_effective` + `metocean_real_ready`. |
| **3.3 Upload endpoints** | `POST /api/v1/investigations/upload-scene` (multipart) — accepts a Sentinel-1 GeoTIFF/PNG (+ optional ground-truth mask), decodes it (`load_scene` / `tifffile` / `cv2`), runs the **full pipeline** (`orchestration.run_uploaded_scene`), and creates a jurisdiction-scoped Investigation. When a mask is supplied it computes a **real IoU** of the detected oil mask vs the mask and attaches it to `summary_metrics.iou`. `POST /api/v1/vessels/ingest-ais` (multipart) — parses a custom AIS CSV (MarineCadastre-style or generic headers via the schema aliases), reconstructs tracks with `tracks.py`, persists `Vessel` rows, and (default) **re-runs fusion** against the investigation's existing hindcast (`orchestration.reattribute_with_tracks`), refreshing `attribution` + `vessel_tracks` + the FUSION anomalies. Console: an *"Analyse an uploaded scene"* panel on Mission Control and an *"Ingest AIS CSV"* control on the Workstation. |

Fixed a latent bug surfaced by the zero-AIS upload path (`ranking["gate"]` could
be `None`). New: `test_uploads.py` (7 tests) + 2 real-met-ocean probe tests.
Acceptance 26 → 28 (upload + IoU; AIS ingestion + re-attribution).

```
$ python -m pytest -q          →  235 passed
$ python scripts/acceptance.py  →  28/28 checkpoints passed
```

### Epic 4 — database upgrade + advanced physics — **COMPLETE** (2026-09-07)

| Item | What shipped |
|---|---|
| **4.1 PostGIS + Alembic** | `backend/core/database.py` now detects a `postgresql*` `DATABASE_URL`, tunes a real connection pool for it (`pool_pre_ping`, `pool_size=5`, `max_overflow=10`), and exposes `enable_postgis()` — a best-effort `CREATE EXTENSION IF NOT EXISTS postgis`, a no-op on SQLite or when `geoalchemy2` is absent — called from the app lifespan. Full Alembic scaffold added: `alembic.ini`, `alembic/env.py` (wired to `settings.DATABASE_URL` + `Base.metadata`, `render_as_batch` on SQLite, `compare_type=True`, geoalchemy2-aware `render_item`), `alembic/script.py.mako`, and an autogenerated initial migration `2904cd6359f2_initial_schema` covering all 6 tables + indexes. `alembic upgrade head` then `alembic check` is clean on SQLite. Geometry columns stay portable JSON so the same schema runs on SQLite and PostgreSQL; PostGIS is opt-in. `alembic>=1.13` added to `requirements.txt`; `psycopg[binary]` / `geoalchemy2` listed commented with a `DATABASE_URL` example. |
| **4.2 Oil weathering physics** | `backend/ml/drift/aging.py` gained `evaporated_fraction()` (Fingas log-law `%Ev = (a + b·T)·ln(t_min)`, per-oil-class coefficients, clamped ≤ 75 %), `spread_area_km2()` (Fay gravity-viscous `A ∝ ((Δ·g·V²)/√ν)^{1/3}·t^{1/2}`), and `weather_slick()` which returns a `series` of `{t_h, evaporated_fraction, volume_remaining_m3, area_km2, mean_thickness_mm}` at t=0 and each forecast horizon. `WEATHERING` config block holds the constants. `orchestration.py` runs it from the characterised area and stashes `summary_metrics["weathering"]`; the Evidence Dossier gets a §4b weathering table (JSON + Markdown + HTML) and the Console *Spill Analysis* panel renders the curve + an "evaporated %" KPI. Order-of-magnitude by design — SAR gives area, not volume, so initial volume assumes a 1 mm film. |

New: `test_migrations.py` (4 tests — upgrade builds every model table, no
model/migration drift, downgrade drops them, `enable_postgis` no-op on SQLite)
and `test_weathering.py` (4 tests — evaporation monotone/bounded/zero-at-start +
warmer-is-faster, Fay area grows with time and volume, `weather_slick` conserves
mass and thins, scenario + dossier carry the block). Acceptance 28 → 30
(#29 weathering series, #30 Alembic migration parity vs ORM metadata).

```
$ python -m pytest -q          →  243 passed
$ python scripts/acceptance.py  →  30/30 checkpoints passed
```

---

## 8. Sign-off

Phases 1–9 **PASSED**; Epics 1–4 **COMPLETE** (Epic 3 credential-gated — Twilio
and real met-ocean fall back to Mock/Demo without keys; Epic 4 PostGIS opt-in,
verified on SQLite). **243/243** automated tests green. **30/30** acceptance
checkpoints green. Lazy-loading verified. Performance within tiered budgets on a
single CPU core with no GPU and no network (Epic 2 made the pipeline ~3× faster —
see §4.3). All four post-prototype Epics are done.

See `docs/DEPLOYMENT.md` for run and demo instructions.
