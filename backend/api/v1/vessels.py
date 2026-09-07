"""
Vessel endpoints - jurisdiction-scoped by involvement.

A vessel is a global identity, but access to it here follows the anomalies it is
linked to: a PILOT sees a vessel only if it appears in an anomaly inside their
assigned zone. NATIONAL sees every vessel.

``GET /vessels/{mmsi}?investigation_id=`` merges the reconstructed track view
(pings, loitering, blackouts) and the fusion result (score, AE anomaly, LSTM
route deviation) from that investigation's ``summary_metrics``.
``GET /vessels/{mmsi}/track?investigation_id=`` returns just the map-ready
track: decimated pings + loiter spans + blackout gaps.
"""
from __future__ import annotations

import csv
import io

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.api._authz import require_row_access
from backend.core.database import get_db
from backend.core.security import get_current_user
from backend.models.anomaly import Anomaly
from backend.models.investigation import Investigation
from backend.models.user import User
from backend.models.vessel import Vessel
from backend.schemas.vessel import VesselOut
from backend.services import jurisdiction as juris

router = APIRouter(prefix="/vessels", tags=["vessels"])


@router.post(
    "/ingest-ais",
    status_code=status.HTTP_201_CREATED,
    summary="Ingest a custom AIS CSV and attach the tracks to an investigation",
)
async def ingest_ais(
    investigation_id: int = Form(...),
    csv_file: UploadFile = File(..., description="AIS CSV (MarineCadastre-style or generic headers)"),
    reattribute: bool = Form(True, description="re-run fusion against the new tracks"),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> dict:
    from backend.services import orchestration

    inv = db.get(Investigation, investigation_id)
    if inv is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Investigation not found")
    require_row_access(db, user, lat=inv.centroid_lat, lon=inv.centroid_lon,
                       jurisdiction_id=inv.jurisdiction_id)

    raw = (await csv_file.read()).decode("utf-8-sig", errors="replace")
    rows = list(csv.DictReader(io.StringIO(raw)))
    if not rows:
        raise HTTPException(422, detail="the CSV has no data rows")

    if reattribute:
        result = orchestration.reattribute_with_tracks(db, inv, rows)
    else:
        import copy

        from backend.services import vessels as _v
        vproc = _v.process_ais_records(rows, inv.detected_at.isoformat())
        _v.persist_vessels(db, vproc["tracks"], last_seen_at=inv.detected_at)
        m = copy.deepcopy(inv.summary_metrics or {})
        vts = dict(m.get("vessel_tracks") or {})
        vts.update(_v.build_track_views(vproc["tracks"]))
        m["vessel_tracks"] = vts
        m.setdefault("caveats", []).append("Vessel tracks attached from operator-ingested AIS.")
        inv.summary_metrics = m
        db.commit()
        db.refresh(inv)
        result = {"reattributed": False, "vessels": len(vproc["tracks"]), "report": vproc["report"]}

    result["investigation_id"] = inv.id
    result["rows_ingested"] = len(rows)
    return result


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


def _load_vessel(db: Session, user: User, mmsi: str) -> Vessel:
    vessel = db.execute(select(Vessel).where(Vessel.mmsi == str(mmsi))).scalar_one_or_none()
    if vessel is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Vessel not found")
    allowed = _accessible_vessel_ids(db, user)
    if allowed is not None and vessel.id not in allowed:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This vessel is not linked to any anomaly in your assigned jurisdiction.",
        )
    return vessel


def _track_from_investigation(
    db: Session, user: User, mmsi: str, investigation_id: int
) -> dict | None:
    inv = db.get(Investigation, investigation_id)
    if inv is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Investigation not found")
    if juris.accessible_jurisdiction_ids(db, user) is not None:
        if not juris.user_can_access_coords_or_jurisdiction(
            db, user, lat=inv.centroid_lat, lon=inv.centroid_lon,
            jurisdiction_id=inv.jurisdiction_id,
        ):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                                detail="This investigation is outside your assigned jurisdiction.")
    return ((inv.summary_metrics or {}).get("vessel_tracks") or {}).get(str(mmsi))


@router.get("/{mmsi}", response_model=None, summary="Read one vessel by MMSI (+ optional track)")
def get_vessel(
    mmsi: str,
    investigation_id: int | None = None,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> dict:
    vessel = _load_vessel(db, user, mmsi)
    out = VesselOut.model_validate(vessel).model_dump()
    if investigation_id is not None:
        view = _track_from_investigation(db, user, mmsi, investigation_id)
        out["track"] = view
        out["attribution"] = (view or {}).get("attribution")
        out["metrics"] = (view or {}).get("metrics")
    return out


@router.get("/{mmsi}/track", summary="Map-ready track for a vessel in an investigation")
def get_vessel_track(
    mmsi: str,
    investigation_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> dict:
    _load_vessel(db, user, mmsi)
    view = _track_from_investigation(db, user, mmsi, investigation_id)
    if view is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail="No reconstructed track for this vessel in that investigation.")
    return view
