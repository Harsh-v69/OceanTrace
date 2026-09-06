"""
Anomaly endpoints - jurisdiction-scoped by the anomaly's coordinates.

A PILOT sees only anomalies inside their assigned coastal zone; a request for
one outside returns 403.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.api._authz import filter_by_jurisdiction, require_point_access, require_row_access
from backend.core.database import get_db
from backend.core.security import get_current_user
from backend.models.anomaly import Anomaly
from backend.models.user import User
from backend.schemas.anomaly import AnomalyCreate, AnomalyOut
from backend.services import jurisdiction as juris

router = APIRouter(prefix="/anomalies", tags=["anomalies"])


@router.get("", response_model=list[AnomalyOut], summary="List anomalies (jurisdiction-scoped)")
def list_anomalies(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    investigation_id: int | None = None,
    limit: int = 100,
    offset: int = 0,
) -> list[Anomaly]:
    stmt = select(Anomaly).order_by(Anomaly.id.desc())
    if investigation_id is not None:
        stmt = stmt.where(Anomaly.investigation_id == investigation_id)
    rows = list(db.execute(stmt).scalars().all())
    scoped = filter_by_jurisdiction(db, user, rows, lat_attr="lat", lon_attr="lon")
    return scoped[offset: offset + min(limit, 500)]


@router.post(
    "",
    response_model=AnomalyOut,
    status_code=status.HTTP_201_CREATED,
    summary="Record an anomaly",
)
def create_anomaly(
    payload: AnomalyCreate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> Anomaly:
    require_point_access(db, user, payload.lat, payload.lon)
    anomaly = Anomaly(
        investigation_id=payload.investigation_id,
        vessel_id=payload.vessel_id,
        type=payload.type,
        source=payload.source,
        label=payload.label,
        confidence=payload.confidence,
        score=payload.score,
        occurred_at=payload.occurred_at,
        lat=payload.lat,
        lon=payload.lon,
        details=payload.details,
    )
    db.add(anomaly)
    db.commit()
    db.refresh(anomaly)
    return anomaly


@router.get("/{anomaly_id}", response_model=AnomalyOut, summary="Read one anomaly")
def get_anomaly(
    anomaly_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> Anomaly:
    anomaly = db.get(Anomaly, anomaly_id)
    if anomaly is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Anomaly not found")
    require_row_access(db, user, lat=anomaly.lat, lon=anomaly.lon, jurisdiction_id=None)
    return anomaly


@router.get(
    "/{anomaly_id}/jurisdictions",
    summary="Which maritime jurisdictions this anomaly falls in",
)
def anomaly_jurisdictions(
    anomaly_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> dict:
    anomaly = db.get(Anomaly, anomaly_id)
    if anomaly is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Anomaly not found")
    require_row_access(db, user, lat=anomaly.lat, lon=anomaly.lon, jurisdiction_id=None)
    resolved = juris.resolve_affected_jurisdictions(db, anomaly.lat, anomaly.lon)
    return {
        "anomaly_id": anomaly_id,
        "lat": anomaly.lat,
        "lon": anomaly.lon,
        "primary_code": resolved["primary"].code if resolved["primary"] else None,
        "chain_codes": resolved["codes"],
        "jurisdiction_ids": resolved["ids"],
    }
