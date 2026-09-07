"""Aggregate every v1 router under one object for ``main`` to mount."""
from __future__ import annotations

from fastapi import APIRouter

from backend.api.v1 import (
    alerts,
    anomalies,
    auth,
    evidence,
    investigations,
    jurisdictions,
    scenarios,
    system,
    users,
    vessels,
)

api_router = APIRouter()
api_router.include_router(auth.router)
api_router.include_router(users.router)
api_router.include_router(system.router)
api_router.include_router(jurisdictions.router)
api_router.include_router(investigations.router)
api_router.include_router(evidence.router)
api_router.include_router(anomalies.router)
api_router.include_router(vessels.router)
api_router.include_router(alerts.router)
api_router.include_router(scenarios.router)

__all__ = ["api_router"]
