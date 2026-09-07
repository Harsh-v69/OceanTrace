#!/usr/bin/env python
"""
Single entry point for the unified OceanTrace prototype.

    python run.py                 # start the API + Operations Console
    python run.py --check         # offline pre-flight, no server
    python run.py --host 0.0.0.0 --port 8080

One unified FastAPI app (no separate Streamlit process). The Operations Console
is served as static files at ``/app`` by the same process; ``/`` redirects there.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def preflight() -> int:
    """Verify the app can run fully offline on this host. Returns an exit code."""
    ok = True

    def check(label: str, fn) -> None:
        nonlocal ok
        try:
            detail = fn()
            print(f"  [ OK ] {label}" + (f" - {detail}" if detail else ""))
        except Exception as exc:  # noqa: BLE001 - preflight reports, never crashes
            ok = False
            print(f"  [FAIL] {label} - {type(exc).__name__}: {exc}")

    print("Pre-flight checks (offline):")

    def _imports():
        import backend.main  # noqa: F401
        return "backend.main imports"

    def _config():
        from backend.core.config import settings
        return (f"env={settings.ENV} offline={settings.OFFLINE_MODE} "
                f"lazy_ml={settings.ML_LAZY_LOAD} sms={settings.SMS_PROVIDER}")

    def _db():
        from backend.core.database import check_db, init_db
        init_db()
        assert check_db(), "database ping failed"
        return "SQLite ready"

    def _jurisdictions():
        from backend.core.database import SessionLocal
        from backend.services.jurisdiction import seed_demo_jurisdictions
        with SessionLocal() as db:
            seed_demo_jurisdictions(db)
            from backend.models.jurisdiction import Jurisdiction
            n = db.query(Jurisdiction).count()
        return f"{n} maritime zones seeded"

    def _registry():
        from backend.ml.registry import get_registry
        reg = get_registry()
        names = [m["name"] for m in reg.status()["models"]]
        resident = reg.status()["resident"] if "resident" in reg.status() else \
            sum(1 for m in reg.status()["models"] if m["loaded"])
        assert resident == 0, "a model loaded during preflight - lazy-load broken"
        return f"{len(names)} models registered, 0 resident (lazy)"

    def _scenarios():
        from backend.simulator import list_scenarios
        return f"{len(list_scenarios())} demo scenarios"

    def _static():
        idx = ROOT / "backend" / "static" / "index.html"
        assert idx.is_file(), "Operations Console index.html missing"
        return "console assets present"

    check("Python imports", _imports)
    check("Configuration", _config)
    check("Database", _db)
    check("Jurisdiction engine", _jurisdictions)
    check("ML registry (lazy)", _registry)
    check("Demo scenarios", _scenarios)
    check("Operations Console", _static)

    print()
    if ok:
        from backend.core.config import settings
        print(f"READY. Start with:  python run.py --host {settings.HOST} --port {settings.PORT}")
        print(f"Then open:          http://{settings.HOST}:{settings.PORT}/app/")
        return 0
    print("NOT READY - fix the failures above.")
    return 1


def main() -> int:
    from backend.core.config import settings

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="run offline pre-flight and exit")
    parser.add_argument("--host", default=settings.HOST)
    parser.add_argument("--port", type=int, default=settings.PORT)
    parser.add_argument("--reload", action="store_true", help="uvicorn autoreload (dev)")
    args = parser.parse_args()

    if args.check:
        return preflight()

    import uvicorn

    print(f"{settings.APP_NAME} v{settings.APP_VERSION}")
    print(f"  API   : http://{args.host}:{args.port}{settings.API_V1_PREFIX}")
    print(f"  Docs  : http://{args.host}:{args.port}/api/docs")
    print(f"  Console: http://{args.host}:{args.port}/app/")
    uvicorn.run("backend.main:app", host=args.host, port=args.port, reload=args.reload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
