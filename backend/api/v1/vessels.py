"""
Vessel endpoints - jurisdiction-scoped by involvement.

A vessel is a global identity, but access to it here follows the anomalies it is
linked to: a PILOT sees a vessel only if it appears in an anomaly inside their
assigned zone. NATIONAL sees every vessel.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.core.database import get_db
from backend.core.security import get_current_user
from backend.models.anomaly import Anomaly
from backend.models.user import User
from backend.models.vessel import Vessel
from backend.schemas.vessel import VesselOut
from backend.services import jurisdiction as juris

router = APIRouter(prefix="/vessels", tags=["vessels"])


def _accessible_vessel_ids(db: Session, user: User) -> set[int] | None:
    """Vessel ids linked to an anomaly the user's jurisdiction covers. None = all."""
    if juris.accessible_jurisdiction_ids(db, user) is None:
        return None
    rows = db.execute(
        select(Anomaly.vessel_id, Anomaly.lat, Anomaly.lon).where(Anomaly.vessel_id.isnot(None))
    ).all()
    return {
        vid for vid, lat, lon in rows
        if lat is not None and lon is not None and juris.user_can_access_point(db, user, lat, lon)
    }


@router.get("", response_model=list[VesselOut], summary="List vessels (jurisdiction-scoped)")
def list_vessels(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    limit: int = 50,
    offset: int = 0,
) -> list[Vessel]:
    rows = list(db.execute(select(Vessel).order_by(Vessel.id.desc())).scalars().all())
    allowed = _accessible_vessel_ids(db, user)
    if allowed is not None:
        rows = [v for v in rows if v.id in allowed]
    return rows[offset: offset + min(limit, 200)]


@router.get("/{mmsi}", response_model=VesselOut, summary="Read one vessel by MMSI")
def get_vessel(
    mmsi: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> Vessel:
    vessel = db.execute(select(Vessel).where(Vessel.mmsi == mmsi)).scalar_one_or_none()
    if vessel is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Vessel not found")
    allowed = _accessible_vessel_ids(db, user)
    if allowed is not None and vessel.id not in allowed:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This vessel is not linked to any anomaly in your assigned jurisdiction.",
        )
    return vessel
