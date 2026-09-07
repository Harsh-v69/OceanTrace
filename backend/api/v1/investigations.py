"""
Investigation endpoints - jurisdiction-scoped.

PILOT / REGIONAL users see and touch only investigations inside their assigned
maritime jurisdiction closure; anything outside is a 403. NATIONAL sees all.
"""
from __future__ import annotations

import io
import uuid
from datetime import datetime

import numpy as np
from fastapi import (
    APIRouter, Depends, File, Form, HTTPException, UploadFile, status,
)
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


def _decode_mask(data: bytes) -> np.ndarray:
    """A boolean ground-truth mask from a PNG/JPEG or (Geo)TIFF upload."""
    head = data[:4]
    if head[:2] in (b"II", b"MM"):
        import tifffile
        arr = tifffile.imread(io.BytesIO(data))
    else:
        import cv2
        arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
    if arr is None:
        raise ValueError("could not decode the ground-truth mask")
    if arr.ndim == 3:
        arr = arr[..., 0]
    return np.asarray(arr) > 0


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


@router.post(
    "/upload-scene",
    response_model=InvestigationOut,
    status_code=status.HTTP_201_CREATED,
    summary="Analyse an uploaded Sentinel-1 GeoTIFF/PNG through the full pipeline",
)
async def upload_scene(
    scene: UploadFile = File(..., description="Sentinel-1 GeoTIFF or PNG quicklook"),
    ground_truth_mask: UploadFile | None = File(
        None, description="optional binary oil mask (GeoTIFF/PNG) - enables a real IoU"),
    title: str = Form("Uploaded SAR scene"),
    bbox: str | None = Form(None, description='"west,south,east,north" in degrees'),
    acquisition: str | None = Form(None, description="ISO 8601 acquisition time"),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> Investigation:
    from backend.ml.sar.ingest import load_scene
    from backend.services import orchestration

    box = None
    if bbox:
        try:
            box = [float(v) for v in bbox.replace(" ", "").split(",")]
            assert len(box) == 4
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(422, detail='bbox must be "west,south,east,north"') from exc

    acq = None
    if acquisition:
        try:
            acq = datetime.fromisoformat(acquisition.replace("Z", "+00:00"))
        except ValueError as exc:
            raise HTTPException(422, detail="acquisition must be ISO 8601") from exc

    raw = await scene.read()
    try:
        sar_scene = load_scene(raw, bbox=box, acquisition=acq.isoformat() if acq else None)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(422, detail=f"could not read the SAR scene: {exc}") from exc

    eff_box = box or list(sar_scene.meta.get("bbox") or [72.0, 18.0, 72.4, 18.4])
    centre = ((eff_box[1] + eff_box[3]) / 2.0, (eff_box[0] + eff_box[2]) / 2.0)
    # a PILOT/REGIONAL may only analyse a scene inside their zone
    require_point_access(db, user, centre[0], centre[1])

    truth = None
    if ground_truth_mask is not None:
        try:
            truth = _decode_mask(await ground_truth_mask.read())
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(422, detail=f"could not read the ground-truth mask: {exc}") from exc

    acq_final = sar_scene.acquisition if isinstance(sar_scene.acquisition, datetime) else acq
    return orchestration.run_uploaded_scene(
        db, user, sar_scene.sigma0_db,
        title=title, bbox=eff_box, acquisition=acq_final, truth_mask=truth,
    )


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
