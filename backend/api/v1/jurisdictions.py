"""
Jurisdiction endpoints - the maritime-zone catalogue + coordinate resolver.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.core.database import get_db
from backend.core.security import get_current_user
from backend.models.jurisdiction import Jurisdiction
from backend.models.user import User
from backend.schemas.jurisdiction import JurisdictionOut, JurisdictionResolveOut
from backend.services import jurisdiction as juris

router = APIRouter(prefix="/jurisdictions", tags=["jurisdictions"])


@router.get("", response_model=list[JurisdictionOut], summary="List maritime jurisdictions")
def list_jurisdictions(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    mine_only: bool = False,
    with_geometry: bool = True,
) -> list[Jurisdiction]:
    rows = list(db.execute(select(Jurisdiction).order_by(Jurisdiction.id)).scalars().all())
    if mine_only:
        allowed = juris.accessible_jurisdiction_ids(db, user)
        if allowed is not None:
            rows = [j for j in rows if j.id in allowed]
    if not with_geometry:
        for j in rows:
            j.geometry = None
    return rows


@router.get("/resolve", response_model=JurisdictionResolveOut, summary="Map a coordinate to its jurisdictions")
def resolve(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
) -> JurisdictionResolveOut:
    r = juris.resolve_affected_jurisdictions(db, lat, lon)
    return JurisdictionResolveOut(
        lat=lat, lon=lon,
        primary_code=r["primary"].code if r["primary"] else None,
        chain_codes=r["codes"],
        jurisdiction_ids=r["ids"],
    )
