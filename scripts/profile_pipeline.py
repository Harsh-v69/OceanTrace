#!/usr/bin/env python
"""
CPU performance profile for the unified pipeline.

Runs every deterministic demo scenario end to end through
``orchestration.run_full_pipeline`` and reports:

  * per-stage wall-clock timings (ms) as recorded in investigation metadata
  * total_ms per scenario
  * ML model lazy-load behaviour: registry state before the first run,
    first-load (cold) cost, and that warm runs pay nothing further

    python scripts/profile_pipeline.py [--repeat 3]

No server, no network. Pure in-process timing on one CPU core.
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# isolated throwaway DB so a profile run never touches data/app.db
_DB = Path(tempfile.gettempdir()) / "sn_profile.db"
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_DB.as_posix()}")
os.environ.setdefault("JWT_SECRET_KEY", "profile-only-secret-0123456789abcdef0123456789")
os.environ.setdefault("ENV", "test")

from backend.core.database import Base, SessionLocal, engine, init_db  # noqa: E402
from backend.ml.registry import get_registry  # noqa: E402
from backend.models.user import User, UserRole  # noqa: E402
from backend.services import orchestration  # noqa: E402
from backend.services.jurisdiction import seed_demo_jurisdictions  # noqa: E402
from backend.simulator import SCENARIOS, build_scenario, seed_scenario_zones  # noqa: E402

STAGES = ["satellite_ingest", "preprocessing", "detection", "characterization",
          "hindcast_forecast", "ais_correlation", "attribution_fusion",
          "jurisdiction", "alert"]


def _fresh_db():
    if _DB.exists():
        _DB.unlink()
    Base.metadata.create_all(bind=engine)
    init_db()
    with SessionLocal() as db:
        seed_demo_jurisdictions(db)
        seed_scenario_zones(db)
        user = User(name="Profiler", email="profiler@local", role=UserRole.NATIONAL,
                    password_hash="x", phone_number="+10000000000", active=True)
        db.add(user)
        db.commit()
        db.refresh(user)
        return user.id


def _registry_line() -> str:
    st = get_registry().status()
    def _one(m):
        name = m["name"].split("(")[0].strip()
        if not m["loaded"]:
            return f"{name}: not loaded"
        p = f"/{m['parameters']:,}p" if m["parameters"] else ""
        return f"{name}: {m['load_seconds']}s{p}"
    return " | ".join(_one(m) for m in st["models"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=3, help="warm repeats per scenario")
    args = ap.parse_args()

    uid = _fresh_db()

    print("=" * 78)
    print("UNIFIED PIPELINE - CPU PERFORMANCE PROFILE")
    print("=" * 78)
    print(f"host cores      : {os.cpu_count()}")
    print(f"python          : {sys.version.split()[0]}")
    print(f"registry (start): {_registry_line()}")
    print()

    cold = {}
    warm = {s: [] for s in SCENARIOS}
    totals = {s: [] for s in SCENARIOS}

    for key in SCENARIOS:
        spec, scene = build_scenario(key)
        for i in range(args.repeat + 1):
            with SessionLocal() as db:
                user = db.get(User, uid)
                t0 = time.perf_counter()
                inv, _steps = orchestration.run_full_pipeline(db, spec, scene, user)
                wall = (time.perf_counter() - t0) * 1000.0
            tm = (inv.summary_metrics or {}).get("timings", {})
            if i == 0:
                cold[key] = (wall, tm)
            else:
                warm[key].append(tm)
                totals[key].append(wall)

    # ---- per-stage table (warm mean, ms) --------------------------------
    hdr = f"{'scenario':<24}" + "".join(f"{s[:9]:>10}" for s in STAGES) + f"{'TOTAL':>10}"
    print(hdr)
    print("-" * len(hdr))
    for key in SCENARIOS:
        rows = warm[key] or [cold[key][1]]
        means = {s: statistics.mean([r.get(s, 0.0) for r in rows]) for s in STAGES}
        tot = statistics.mean([r.get("total_ms", 0.0) for r in rows])
        line = f"{key:<24}" + "".join(f"{means[s]:>10.1f}" for s in STAGES) + f"{tot:>10.1f}"
        print(line)
    print()

    # ---- cold vs warm + lazy-load ------------------------------------
    print(f"{'scenario':<24}{'cold ms':>12}{'warm ms (mean)':>18}{'warm stdev':>14}")
    print("-" * 68)
    for key in SCENARIOS:
        cw = cold[key][0]
        ww = totals[key] or [cw]
        print(f"{key:<24}{cw:>12.1f}{statistics.mean(ww):>18.1f}"
              f"{(statistics.pstdev(ww) if len(ww) > 1 else 0.0):>14.1f}")
    print()
    print(f"registry (end)  : {_registry_line()}")
    print()

    # Tiered latency budgets (single CPU core, no GPU). Interactive endpoints
    # (auth / map / lists / dossier) are all sub-second and covered by the test
    # suite; a full investigation is a triggered batch job with a live progress
    # UI, so seconds are acceptable. The heaviest forensic scenario (Wakashio:
    # two-segment continuous-release culprit track + drift feedback loop) is
    # allowed more headroom.
    STD_BUDGET = 10_000.0     # typical investigation
    HARD_BUDGET = 20_000.0    # heaviest forensic scenario
    warm_means = {k: statistics.mean(totals[k] or [cold[k][0]]) for k in SCENARIOS}
    worst = max(warm_means.values())
    typical = statistics.median(sorted(warm_means.values())[:-1])  # drop the outlier
    print(f"latency budgets : typical <= {STD_BUDGET:.0f} ms, heaviest <= {HARD_BUDGET:.0f} ms")
    print(f"typical warm    : {typical:.1f} ms  ->  {'OK' if typical < STD_BUDGET else 'OVER'}")
    print(f"heaviest warm   : {worst:.1f} ms  ->  {'OK' if worst < HARD_BUDGET else 'OVER'}")
    print("=" * 78)
    return 0 if (typical < STD_BUDGET and worst < HARD_BUDGET) else 1


if __name__ == "__main__":
    raise SystemExit(main())
