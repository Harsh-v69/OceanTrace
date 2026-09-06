"""
Investigation endpoints - jurisdiction-scoped.

PILOT / REGIONAL users see and touch only investigations inside their assigned
maritime jurisdiction closure; anything outside is a 403. NATIONAL sees all.
"""
from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.api._authz import filter_by_jurisdiction, require_point_access, require_row_access
from backend.core.database import get_db
from backend.core.security import get_current_user
from backend.models.investigation import Investigation
from backend.models.user import User
from backend.schemas.investigation import InvestigationCreate, InvestigationOut
from backend.services import jurisdiction as juris

router = APIRouter(prefix="/investigations", tags=["investigations"])


@router.get("", response_model=list[InvestigationOut], summary="List investigations (jurisdiction-scoped)")
def list_investigations(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    limit: int = 50,
    offset: int = 0,
) -> list[Investigation]:
    rows = list(db.execute(
        select(Investigation).order_by(Investigation.id.desc())
    ).scalars().all())
    scoped = filter_by_jurisdiction(
        db, user, rows,
        lat_attr="centroid_lat", lon_attr="centroid_lon", jur_attr="jurisdiction_id",
    )
    return scoped[offset: offset + min(limit, 200)]


@router.post(
    "",
    response_model=InvestigationOut,
    status_code=status.HTTP_201_CREATED,
    summary="Open a new investigation",
)
def create_investigation(
    payload: InvestigationCreate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> Investigation:
    # a PILOT/REGIONAL cannot open a case outside their zone
    require_point_access(db, user, payload.centroid_lat, payload.centroid_lon)

    jurisdiction_id = payload.jurisdiction_id
    if jurisdiction_id is None and payload.centroid_lat is not None and payload.centroid_lon is not None:
        resolved = juris.resolve_affected_jurisdictions(db, payload.centroid_lat, payload.centroid_lon)
        if resolved["primary"] is not None:
            jurisdiction_id = resolved["primary"].id
    else:
        require_row_access(db, user, lat=None, lon=None, jurisdiction_id=jurisdiction_id)

    inv = Investigation(
        reference=f"INV-{uuid.uuid4().hex[:8].upper()}",
        title=payload.title,
        description=payload.description,
        jurisdiction_id=jurisdiction_id,
        created_by_id=user.id,
        scene_ref=payload.scene_ref,
        detected_at=payload.detected_at,
        centroid_lat=payload.centroid_lat,
        centroid_lon=payload.centroid_lon,
    )
    db.add(inv)
    db.commit()
    db.refresh(inv)
    return inv


@router.get("/{investigation_id}", response_model=InvestigationOut, summary="Read one investigation")
def get_investigation(
    investigation_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> Investigation:
    inv = db.get(Investigation, investigation_id)
    if inv is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Investigation not found")
    require_row_access(
        db, user, lat=inv.centroid_lat, lon=inv.centroid_lon,
        jurisdiction_id=inv.jurisdiction_id,
    )
    return inv
