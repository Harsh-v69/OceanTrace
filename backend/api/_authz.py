"""
API-layer authorization helpers - role + jurisdiction scoping.

The role check lives in ``core/security.py`` (``require_roles`` /
``require_min_role``). This module adds the *geographical* dimension: a
PILOT/REGIONAL user may only touch data inside their jurisdiction closure;
anything outside is a 403.
"""
from __future__ import annotations

from typing import Iterable

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from backend.models.user import User
from backend.services import jurisdiction as juris


def require_point_access(db: Session, user: User, lat: float | None, lon: float | None) -> None:
    """403 if the user's jurisdiction does not cover ``(lat, lon)``."""
    if lat is None or lon is None:
        return
    if not juris.user_can_access_point(db, user, lat, lon):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=("This location is outside your assigned maritime jurisdiction."),
        )


def require_row_access(
    db: Session, user: User, *, lat: float | None, lon: float | None,
    jurisdiction_id: int | None,
) -> None:
    if juris.accessible_jurisdiction_ids(db, user) is None:
        return
    if juris.user_can_access_coords_or_jurisdiction(
        db, user, lat=lat, lon=lon, jurisdiction_id=jurisdiction_id
    ):
        return
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="This record is outside your assigned maritime jurisdiction.",
    )


def filter_by_jurisdiction(
    db: Session, user: User, rows: Iterable, *,
    lat_attr: str | None = None, lon_attr: str | None = None,
    jur_attr: str | None = None,
) -> list:
    """Drop rows the user's jurisdiction closure does not cover. NATIONAL keeps all."""
    if juris.accessible_jurisdiction_ids(db, user) is None:
        return list(rows)
    kept = []
    for r in rows:
        lat = getattr(r, lat_attr, None) if lat_attr else None
        lon = getattr(r, lon_attr, None) if lon_attr else None
        jid = getattr(r, jur_attr, None) if jur_attr else None
        if juris.user_can_access_coords_or_jurisdiction(
            db, user, lat=lat, lon=lon, jurisdiction_id=jid
        ):
            kept.append(r)
    return kept
