"""
FastAPI application entry point.

    uvicorn backend.main:app --reload

One unified API (no separate Streamlit app). Routers are mounted under
``/api/v1``; interactive docs live at ``/api/docs``.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from backend.api.v1.routes import api_router
from backend.core.config import settings
from backend.core.database import init_db
from backend.core.logging import configure_logging, get_logger

configure_logging()
log = get_logger("backend.main")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    log.info(
        "Starting %s v%s (env=%s, offline=%s)",
        settings.APP_NAME,
        settings.APP_VERSION,
        settings.ENV,
        settings.OFFLINE_MODE,
    )
    init_db()
    log.info("Database ready: %s", settings.DATABASE_URL)

    if settings.SEED_DEMO_JURISDICTIONS or settings.SEED_DEFAULT_USERS:
        from backend.core.database import SessionLocal

        with SessionLocal() as db:
            if settings.SEED_DEMO_JURISDICTIONS:
                from backend.services.jurisdiction import seed_demo_jurisdictions

                n = seed_demo_jurisdictions(db)
                if n:
                    log.info("Seeded %d demo maritime jurisdiction(s)", n)
            if settings.SEED_DEFAULT_USERS:
                from backend.services.users import seed_default_users

                m = seed_default_users(db)
                if m:
                    log.info("Seeded %d default user account(s)", m)

    yield
    log.info("Shutdown complete")


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description=(
            "Unified oil-spill detection and vessel-attribution API for SIH "
            f"problem statement {settings.PROBLEM_STATEMENT_ID} "
            f"({settings.ORGANISATION})."
        ),
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router, prefix=settings.API_V1_PREFIX)

    # ---- Operations Console (single unified app - no separate Streamlit) ----
    static_dir = Path(__file__).parent / "static"
    if static_dir.is_dir():

        class _ConsoleStatic(StaticFiles):
            """Serve the SPA with revalidate-always caching so an updated build
            is never masked by a stale browser copy."""

            async def get_response(self, path, scope):  # noqa: ANN001
                resp = await super().get_response(path, scope)
                resp.headers["Cache-Control"] = "no-cache, must-revalidate"
                return resp

            def is_not_modified(self, response_headers, request_headers) -> bool:  # noqa: ANN001
                return False

        app.mount(
            "/app",
            _ConsoleStatic(directory=str(static_dir), html=True),
            name="console",
        )

    @app.get("/", include_in_schema=False, response_model=None)
    def root():
        if static_dir.is_dir():
            return RedirectResponse(url="/app/")
        return {
            "app": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "docs": "/api/docs",
            "health": f"{settings.API_V1_PREFIX}/system/health",
        }

    @app.get("/api", tags=["meta"], summary="API index")
    def api_index() -> dict:
        # FastAPI >=0.116 keeps included routers un-flattened in app.routes, so
        # enumerate from the OpenAPI schema instead (cached after first call).
        schema_paths = app.openapi().get("paths", {})
        paths = sorted(
            p for p in schema_paths if p.startswith(settings.API_V1_PREFIX)
        )
        return {"version": "v1", "prefix": settings.API_V1_PREFIX, "paths": paths}

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):  # noqa: ANN202
        log.exception("Unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content={
                "error": type(exc).__name__,
                "detail": str(exc),
                "path": request.url.path,
            },
        )

    return app


app = create_app()
