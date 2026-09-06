"""System / operational endpoints."""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends

from backend.core.config import settings
from backend.core.database import check_db
from backend.core.security import get_current_user, require_roles
from backend.models.user import User, UserRole

router = APIRouter(prefix="/system", tags=["system"])


@router.get("/health", summary="Liveness + dependency status")
def health() -> dict:
    return {
        "status": "ok",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.ENV,
        "problem_statement_id": settings.PROBLEM_STATEMENT_ID,
        "offline_mode": settings.OFFLINE_MODE,
        "sms_provider": settings.SMS_PROVIDER,
        "database": "up" if check_db() else "down",
        "time_utc": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/info", summary="Static capability descriptor")
def info() -> dict:
    return {
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "organisation": settings.ORGANISATION,
        "roles": [r.value for r in UserRole],
        "modes": {
            "offline": settings.OFFLINE_MODE,
            "ml_lazy_load": settings.ML_LAZY_LOAD,
            "sms_provider": settings.SMS_PROVIDER,
        },
        "alert_confidence_threshold": settings.ALERT_CONFIDENCE_THRESHOLD,
    }


@router.get("/models", summary="ML model registry - lazy-load status + timings")
def model_registry(_user: User = Depends(get_current_user)) -> dict:
    """
    Reports which models are resident, their first-load time and parameter
    count. Nothing here forces a load - a model shows ``loaded: false`` until an
    investigation actually needs it (lazy-loading proof).
    """
    from backend.ml.registry import get_registry

    status = get_registry().status()
    status["lazy_load"] = settings.ML_LAZY_LOAD
    status["resident"] = sum(1 for m in status["models"] if m["loaded"])
    return status


@router.get(
    "/settings",
    summary="Effective runtime settings (NATIONAL role only)",
    dependencies=[Depends(require_roles(UserRole.NATIONAL))],
)
def effective_settings() -> dict:
    """Deliberately role-gated: exercises RBAC end to end."""
    return {
        "env": settings.ENV,
        "debug": settings.DEBUG,
        "database_url": settings.DATABASE_URL,
        "jwt_algorithm": settings.JWT_ALGORITHM,
        "access_token_expire_minutes": settings.ACCESS_TOKEN_EXPIRE_MINUTES,
        "allow_registration_role_select": settings.ALLOW_REGISTRATION_ROLE_SELECT,
        "sms_provider": settings.SMS_PROVIDER,
        "alert_confidence_threshold": settings.ALERT_CONFIDENCE_THRESHOLD,
    }
