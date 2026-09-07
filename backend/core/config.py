"""
Application configuration.

Single source of truth for every tunable. Values come from environment
variables (optionally via a local ``.env`` file); every setting has a
development-safe default so the app boots with zero configuration.

Nothing here imports the rest of the backend, so this module is safe to import
from anywhere.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root = two levels up from this file (…/backend/core/config.py).
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


class Settings(BaseSettings):
    """Runtime settings, populated from the environment with safe defaults."""

    model_config = SettingsConfigDict(
        env_file=os.getenv("BACKEND_ENV_FILE", ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---------------------------------------------------------------- app ----
    APP_NAME: str = "OceanTrace - Unified API"
    APP_VERSION: str = "0.2.0"
    PROBLEM_STATEMENT_ID: str = "26143"
    ORGANISATION: str = "National Technical Research Organisation (NTRO)"
    ENV: str = "development"          # development | test | staging | production
    DEBUG: bool = True

    # ----------------------------------------------- offline-first / modes --
    # The prototype is designed to run fully offline on local CPU.
    OFFLINE_MODE: bool = True
    ML_LAZY_LOAD: bool = True         # load ML models only on first use
    SMS_PROVIDER: str = "mock"        # mock | twilio

    # ------------------------------------------------------------- server ----
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    API_V1_PREFIX: str = "/api/v1"
    # Comma-separated list, or "*" for any origin. Parsed in main.py.
    CORS_ORIGINS: str = "*"

    # ----------------------------------------------------------- database ----
    # SQLite by default; the schema is written to migrate cleanly to PostGIS
    # (swap this URL and the JSON geometry columns for geoalchemy2 types).
    DATABASE_URL: str = f"sqlite:///{(DATA_DIR / 'app.db').as_posix()}"
    DATABASE_ECHO: bool = False

    # -------------------------------------------------------- auth / jwt -----
    JWT_SECRET_KEY: str = "dev-insecure-secret-change-me-please-0000000000000000"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 12
    PASSWORD_MIN_LENGTH: int = 8
    # Phase 2: open registration. When False, every self-registration is forced
    # to the lowest role (PILOT) and elevation is an admin-only action.
    ALLOW_REGISTRATION_ROLE_SELECT: bool = True
    # Epic 1: hierarchical user management. When False (the default), the public
    # POST /auth/register endpoint is closed - accounts are created only by a
    # NATIONAL or REGIONAL admin through POST /api/v1/users.
    ALLOW_OPEN_REGISTRATION: bool = False
    # Idempotently seed one account per role at startup (see services/users.py).
    SEED_DEFAULT_USERS: bool = True
    DEFAULT_USER_PASSWORD: str = "ChangeMe!OceanTrace1"

    # ------------------------------------------------- twilio (SMS_PROVIDER) --
    TWILIO_ACCOUNT_SID: str | None = None
    TWILIO_AUTH_TOKEN: str | None = None
    TWILIO_FROM_NUMBER: str | None = None

    # ---------------------------------------------------------- alerting ----
    # Anomaly confidence at or above which an alert is raised.
    ALERT_CONFIDENCE_THRESHOLD: float = Field(default=0.75, ge=0.0, le=1.0)
    # Dedup: two alerts with the same fingerprint (location grid + time bucket +
    # geometry signature) inside the cooldown are treated as one.
    ALERT_DEDUP_COOLDOWN_MINUTES: int = 180
    ALERT_DEDUP_GRID_DEG: float = 0.05           # ~5.5 km location quantisation
    ALERT_DEDUP_TIME_BUCKET_MINUTES: int = 60
    ALERT_MAX_SEND_ATTEMPTS: int = 3
    ALERT_TEST_SMS_TEMPLATE: str = (
        "OceanTrace test alert for {name}. SMS delivery is working "
        "(sent {ts})."
    )

    # ----------------------------------------------------- jurisdictions ----
    SEED_DEMO_JURISDICTIONS: bool = True         # seed the Indian maritime zones at startup

    # ----------------------------------------------------------- helpers ----
    @property
    def is_sqlite(self) -> bool:
        return self.DATABASE_URL.startswith("sqlite")

    @property
    def cors_origin_list(self) -> list[str]:
        raw = self.CORS_ORIGINS.strip()
        if raw in ("", "*"):
            return ["*"]
        return [o.strip() for o in raw.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    """Cached accessor. Call ``get_settings.cache_clear()`` in tests to reload."""
    return Settings()


settings = get_settings()
