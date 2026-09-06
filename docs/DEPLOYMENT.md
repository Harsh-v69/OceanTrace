# DEPLOYMENT & DEMO — SIH26143 Unified Prototype

SAMUDRA NETRA × POSEatSea — one FastAPI application. No separate Streamlit
process. The Operations Console is served as static files by the same app.

---

## 1. Requirements

| | |
|---|---|
| OS | Windows 10/11, Linux, or macOS |
| Python | 3.11 – 3.13 (tested on 3.13.7) |
| CPU | any x86-64 core — **no GPU** |
| Internet | **not required** at run time (only for the one-time `pip install`, and for live basemap tiles, which degrade gracefully) |
| Disk | ~1.2 GB (mostly the CPU build of PyTorch) |
| RAM | ~1 GB resident once all three models are loaded |

> **Windows note:** install into a **short path** (e.g. the repo's own
> `C:\...\Oil_Spill_Prototype`) or enable `LongPathsEnabled`. scikit-learn and
> torch have internal filenames that overflow the 260-char `MAX_PATH` limit when
> the venv sits under a deep directory.

---

## 2. First-time setup

```bash
cd Oil_Spill_Prototype

python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# CPU-only torch keeps the install ~200 MB instead of ~2.5 GB of CUDA:
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

(If you skip the first line, `pip` may still pull a CUDA build of torch on
some platforms — harmless but large. `requirements.txt` notes this at the
torch line.)

The SAR classifier binary is **not** committed. If
`backend/ml/sar/models/oil_classifier.joblib` is missing it is trained
automatically on first use (~90–100 s, one time). The POSEatSea autoencoder,
scaler and LSTM (~820 KB total) **are** committed and load as-is.

### Configuration

Copy `.env.example` to `.env` and edit if needed. Everything has a working
default; the demo runs with **no `.env` at all**.

| Variable | Default | Purpose |
|---|---|---|
| `JWT_SECRET_KEY` | dev placeholder | **set a real secret for anything non-local** |
| `DATABASE_URL` | `sqlite:///data/app.db` | swap for PostGIS later — geometry is already GeoJSON |
| `SMS_PROVIDER` | `mock` | `twilio` to send real SMS |
| `TWILIO_ACCOUNT_SID` / `TWILIO_AUTH_TOKEN` / `TWILIO_FROM_NUMBER` | — | required only when `SMS_PROVIDER=twilio` |
| `ALERT_CONFIDENCE_THRESHOLD` | `0.75` | minimum SAR confidence that raises an alert |
| `ML_LAZY_LOAD` | `true` | keep models unloaded until first use |
| `SEED_DEMO_JURISDICTIONS` | `true` | seed the 14 Indian maritime zones at startup |

---

## 3. Pre-flight (offline, no server)

```bash
python run.py --check
```

```
Pre-flight checks (offline):
  [ OK ] Python imports - backend.main imports
  [ OK ] Configuration - env=development offline=True lazy_ml=True sms=mock
  [ OK ] Database - SQLite ready
  [ OK ] Jurisdiction engine - 14 maritime zones seeded
  [ OK ] ML registry (lazy) - 3 models registered, 0 resident (lazy)
  [ OK ] Demo scenarios - 4 demo scenarios
  [ OK ] Operations Console - console assets present

READY.
```

---

## 4. Run

```bash
python run.py                       # 127.0.0.1:8000
python run.py --host 0.0.0.0 --port 8080
python run.py --reload              # dev autoreload
```

| URL | What |
|---|---|
| `http://127.0.0.1:8000/` | redirects to the console |
| `http://127.0.0.1:8000/app/` | **Operations Console** |
| `http://127.0.0.1:8000/api/docs` | OpenAPI / Swagger UI |
| `http://127.0.0.1:8000/api/v1/system/health` | liveness + ML/SMS status |
| `http://127.0.0.1:8000/api/v1/system/models` | model registry (lazy-load state + load times) |

Equivalent without `run.py`: `uvicorn backend.main:app --port 8000`.

---

## 5. Guided demo (≈ 5 minutes)

1. **Open** `http://127.0.0.1:8000/app/`.
2. **Register** on the *Register* tab:
   - a **NATIONAL** user to drive every scenario, **or**
   - a **PILOT** with *Assigned zones* = `IN-MH` to show jurisdiction isolation.
3. **Mission Control** — the Indian-coastline map, KPI tiles, and one button per
   demo scenario.
4. Click **"Mumbai / Maharashtra – high-confidence spill"**. The pipeline runs
   (~6 s) and drops you in the **Investigation Workstation**:
   - *left* — scene + drift map with a **T−48 h … +48 h** timeline slider;
   - *centre* — spill mask (area / axes / orientation), reconstructed origin
     (±km, release-time window), forward forecast;
   - *right* — ranked candidate vessels with a per-component score breakdown;
     `MT KONKAN PRIDE` is #1 and carries a **ground truth** badge.
5. **Live Monitoring** — pick a feed, click **Replay observation**, and watch the
   seven stages (detect → characterise → trace → correlate → rank → jurisdiction
   → alert) light up with real per-stage timings.
6. **Run "Goa / Karnataka – look-alike dark patch"** — it is classified
   *Likely look-alike*, the investigation is auto-**RESOLVED**, and **no alert**
   is raised.
7. **Run "Mauritius – MV Wakashio grounding"** — the LSTM route-deviation and AIS
   autoencoder components are now *engaged* (inside the Mauritius AOI); the
   pipeline takes longer (~15 s) — the largest forensic scenario.
8. **Evidence** — pick an investigation → the dossier renders inline. Buttons:
   **Download JSON** (system of record), **Download Markdown**, **Open printable
   view** (Ctrl/Cmd-P → PDF).
9. **Alerts** *(REGIONAL/NATIONAL)* — the fingerprinted alert feed;
   **Send test SMS** delivers to your registered number via the Mock (or real
   Twilio) provider; failed alerts can be retried.
10. **System** — health, capabilities, and the **model registry**: models show
    `loaded: false` until step 4, then stay resident with their first-load time.
11. **RBAC** — sign in as the `IN-MH` PILOT and try the Mauritius scenario: the
    button returns **403 Forbidden** (outside the assigned jurisdiction).

### Fully-offline demo

Everything above works with the network cable pulled. The only degradation is
the basemap: OpenStreetMap tiles fail to load and the map falls back to a plain
ocean canvas — markers, tracks, polygons and the timeline all still work.

---

## 6. Deterministic scenarios (headless / API)

```bash
# list
curl -H "Authorization: Bearer $TOKEN" localhost:8000/api/v1/scenarios

# run one end to end (RBAC: the scenario centre must be in your jurisdiction)
curl -X POST -H "Authorization: Bearer $TOKEN" \
     localhost:8000/api/v1/scenarios/mumbai-high-confidence/run
```

| key | region | culprit | shows |
|---|---|---|---|
| `mumbai-high-confidence` | Maharashtra (`IN-MH`) | `MT KONKAN PRIDE` | clean high-confidence attribution |
| `lookalike-darkpatch` | Goa/Karnataka (`IN-GA-KA`) | — | look-alike rejection, no alert |
| `ambiguous-drift` | Maharashtra offshore | `MT DECCAN STAR` | drift + behaviour separate the culprit from a crossing decoy |
| `wakashio-mauritius` | Mauritius AOI (`MU-AOI`) | `MV WAKASHIO` | POSEatSea LSTM + autoencoder engaged |

Every run is byte-for-byte repeatable and each returns a
`dossier_url` plus the full per-stage `timings`.

---

## 7. Verification

```bash
python -m pytest -q                 # 197 passed
python scripts/acceptance.py        # 24/24 checkpoints passed
python scripts/profile_pipeline.py  # per-stage timings + lazy-load report
```

See `docs/BUILD_STATUS.md` for the recorded results.

---

## 8. Production notes (beyond the prototype)

- Set a strong `JWT_SECRET_KEY`; put the app behind TLS.
- Point `DATABASE_URL` at PostGIS — the ORM models and GeoJSON geometry are
  already compatible; swap Shapely point-in-polygon for `ST_Contains` if desired.
- Set `SMS_PROVIDER=twilio` with real credentials; the Mock provider stays as the
  automatic fallback if Twilio returns an error (the alert is recorded `FAILED`
  with the error, and is retryable).
- Replace the deterministic met-ocean provider with real ERA5/HYCOM behind
  `RealMetOceanProvider` (`_build_grids()` is the single seam).
- Run under a process manager (`uvicorn --workers N` behind nginx, or
  gunicorn+uvicorn workers). Model caches are per-process; lazy-loading means a
  fresh worker pays the load cost on its first investigation only.
