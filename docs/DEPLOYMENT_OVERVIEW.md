# Deployment Overview

A concise map of how to run OceanTrace. Full step-by-step lives in
[`DEPLOYMENT.md`](DEPLOYMENT.md); this page is the "which path and why".

---

## What the app is

One FastAPI process. It serves the JSON API under `/api/v1` **and** the
Operations Console (static SPA) under `/app`, from the same port. `/` redirects
to `/app/`. No separate frontend server, no database server required (SQLite by
default).

## What it needs

| Resource | Requirement |
|---|---|
| CPU | any x86-64 core, **no GPU** |
| RAM | ~1 GB resident once the 3 ML models lazy-load (PyTorch is the bulk) |
| Disk | ~1.2 GB (mostly the CPU build of PyTorch) |
| Network | **none at runtime** — models are committed, met-ocean is simulated, basemap tiles degrade to an offline vector map |
| Python | 3.11–3.13 |

The ~1 GB RAM floor is the single most important constraint: it rules out every
free 512 MB tier (Render free, Koyeb nano, most PaaS free plans).

---

## Three ways to run it

### 1. Local

```bash
python -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python run.py                 # http://127.0.0.1:8000/app/
python run.py --check         # offline preflight, no server
```

### 2. Temporary public demo — live-reload + tunnel  *(current setup)*

For showing the running app over the internet from your own machine, with code
edits reflecting instantly. No hosting account, no build.

```bash
# one-time: free authtoken from https://dashboard.ngrok.com
ngrok config add-authtoken <TOKEN>

python scripts/deploy_live.py
```

`scripts/deploy_live.py` starts `uvicorn ... --reload` on :8000, opens an ngrok
HTTPS tunnel, and prints the public URL. Ctrl+C stops both.

- **Windows Defender** flags `ngrok.exe` as a PUA and blocks it. Fix once, as
  Administrator:
  `Add-MpPreference -ExclusionPath "$env:LOCALAPPDATA\ngrok"`
- Free ngrok shows visitors a one-time "Visit Site" page; the URL is random and
  changes on every restart.
- The tunnel only lives as long as the script runs — use a dedicated terminal,
  not a transient shell.
- Ephemeral SQLite: jurisdictions + the 3 role accounts re-seed on boot; demo
  scenarios are deterministic, so a restart only drops ad-hoc investigations.

### 3. Persistent host — container

A real always-on URL. The repo ships a production [`Dockerfile`](../Dockerfile)
(CPU torch pinned, non-root, serves on `:7860`). It works on any container host;
pick by budget:

| Host | Cost for a demo week | Notes |
|---|---|---|
| **Google Cloud Run** | $0 within always-free tier | `gcloud run deploy --source .` (no local Docker). Needs a Google account with billing enabled. Cold start ~15–30 s when idle. Set `--memory 1Gi`. |
| **Fly.io** | ~$2–3 | `fly launch` uses the Dockerfile. 1 GB machine, no cold start, free volume keeps SQLite. Card required. |
| **Railway** | ~$5 trial credit | Nixpacks auto-detects Python (Dockerfile optional). Volume for SQLite. Credit runs down. |
| Hugging Face Spaces | **not free** | Docker/Gradio Spaces now require a paid PRO plan. Only Static Spaces are free — unusable here. |

The container `CMD` is `python run.py --host 0.0.0.0 --port 7860`. If your host
injects a `$PORT` (Cloud Run sends 8080), override the command to
`python run.py --host 0.0.0.0 --port $PORT`.

---

## Configuration

Every setting has a safe default (`backend/core/config.py`); the app boots with
zero config. Set these for anything public:

| Env var | Default | Set it when |
|---|---|---|
| `JWT_SECRET_KEY` | dev placeholder | **always**, for any non-local deployment |
| `DEFAULT_USER_PASSWORD` | `ChangeMe!OceanTrace1` | **always** — this is the sign-in password for the 3 seeded accounts |
| `SEED_DEFAULT_USERS` | `true` | leave on; seeds `national@`/`regional@`/`pilot@oceantrace.gov.in` idempotently |
| `ALLOW_OPEN_REGISTRATION` | `false` | keep false — accounts are admin-created via `POST /api/v1/users` |
| `CORS_ORIGINS` | `*` | tighten to your host if you expose the API to third-party clients |
| `DATABASE_URL` | `sqlite:///data/app.db` | point at PostgreSQL/PostGIS for a durable multi-instance deployment (see below) |
| `SMS_PROVIDER` + `TWILIO_*` | `mock` | only for live SMS; missing creds silently fall back to mock |
| met-ocean (`cdsapi`/HYCOM) | simulated | only for real ERA5/HYCOM feeds; failures fall back to the deterministic Demo field |

## Database

- **SQLite (default)** — a file at `data/app.db`. Fine for a single instance and
  every demo. `init_db()` creates the schema on boot.
- **PostgreSQL + PostGIS** — set `DATABASE_URL=postgresql+psycopg://user:pass@host/db`
  and `pip install "psycopg[binary]" geoalchemy2`. Geometry is stored as portable
  JSON, so the same models work on both. Manage the schema with Alembic instead
  of `init_db()`:
  ```bash
  alembic upgrade head                       # apply migrations
  alembic revision --autogenerate -m "..."   # after a model change
  alembic check                              # CI: fail on drift
  ```
  On boot `enable_postgis()` runs `CREATE EXTENSION IF NOT EXISTS postgis`
  (best-effort; create it once as a superuser if the app role lacks permission).

## Verify

```bash
python -m pytest -q                 # 243 passed
python scripts/acceptance.py        # 30/30 checkpoints
python scripts/profile_pipeline.py  # per-stage timings + lazy-load report
```

## First-request cost

Models load on first use, not at boot: ~3–5 s for the SAR ensemble, ~2 s each
for the AIS autoencoder and the LSTM. Every later request is warm. On a
scale-to-zero host, add that to the cold-start time — warm the app once before a
live demo.
