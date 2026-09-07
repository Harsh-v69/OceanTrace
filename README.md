---
title: OceanTrace
emoji: 🛢️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
short_description: Unified SAR oil-spill detection and vessel attribution
---

# OceanTrace

Unified oil-spill detection and vessel-attribution platform for **SIH problem
statement 26143**. One FastAPI application — SAR anomaly detection, Lagrangian
RK4 drift hindcast/forecast, AIS anomaly + trajectory ML, and a weighted
attribution-fusion engine — serving its own offline-first Operations Console.
No separate Streamlit process.

## Live demo

The Space boots the API and the Operations Console together. Open the Space URL
and sign in with one of the seeded accounts:

| Role | Email | Sees |
|---|---|---|
| National | `national@oceantrace.gov.in` | all maritime zones; manages every account |
| Regional | `regional@oceantrace.gov.in` | Western region closure; manages its pilots |
| Pilot | `pilot@oceantrace.gov.in` | Mumbai coastal zone only |

Password: the value of the `DEFAULT_USER_PASSWORD` secret (see below).

Try **Scenarios → “mumbai-high-confidence”** for a full detect → drift → attribute
run, or **“lookalike-darkpatch”** to see a false positive filtered out.

## Deploying your own copy

See [`docs/DEPLOY_HUGGINGFACE.md`](docs/DEPLOY_HUGGINGFACE.md) for the 4-step
runbook. In short: create a **Docker** Space, push this repo to it, and set two
secrets — `JWT_SECRET_KEY` (any long random string) and `DEFAULT_USER_PASSWORD`.

The free CPU tier has ~16 GB RAM (this app needs ~1 GB) and no persistent disk,
so the SQLite database is ephemeral: maritime jurisdictions and the three role
accounts re-seed on every boot, and the demo scenarios are deterministic.

## Running locally

```bash
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python run.py            # http://127.0.0.1:8000/app/
```

Full instructions, deterministic scenarios, PostgreSQL/PostGIS and Alembic notes:
[`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md). Build and test status:
[`docs/BUILD_STATUS.md`](docs/BUILD_STATUS.md).

## Verification

```bash
python -m pytest -q                 # 243 passed
python scripts/acceptance.py        # 30/30 checkpoints
```
