"""
Alert endpoints.

* ``GET  /alerts``            - REGIONAL+, jurisdiction-scoped feed
* ``POST /alerts/dispatch``   - REGIONAL+, raise an alert (fingerprinted + deduped)
* ``POST /alerts/{id}/retry`` - REGIONAL+, re-attempt a FAILED alert
* ``POST /alerts/test-sms``   - any authenticated user, to their own number
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.api._authz import filter_by_jurisdiction, require_point_access
from backend.core.database import get_db
from backend.core.security import get_current_user, require_min_role
from backend.models.alert import Alert
from backend.models.user import User, UserRole
from backend.schemas.alert import AlertCreate, AlertOut, TestSmsRequest
from backend.services import sms as sms_svc

router = APIRouter(prefix="/alerts", tags=["alerts"])


@router.get("", response_model=list[AlertOut], summary="List alerts (REGIONAL or NATIONAL, jurisdiction-scoped)")
def list_alerts(
    db: Session = Depends(get_db),
    user: User = Depends(require_min_role(UserRole.REGIONAL)),
    limit: int = 50,
    offset: int = 0,
) -> list[Alert]:
    rows = list(db.execute(select(Alert).order_by(Alert.id.desc())).scalars().all())
    scoped = filter_by_jurisdiction(
        db, user, rows, lat_attr="lat", lon_attr="lon",
        code_attr="jurisdiction_codes",
    )
    return scoped[offset: offset + min(limit, 200)]


@router.post(
    "/dispatch",
    response_model=AlertOut,
    status_code=status.HTTP_201_CREATED,
    summary="Raise an alert (fingerprinted + deduplicated)",
)
def dispatch(
    payload: AlertCreate,
    db: Session = Depends(get_db),
    user: User = Depends(require_min_role(UserRole.REGIONAL)),
    force: bool = False,
) -> Alert:
    require_point_access(db, user, payload.lat, payload.lon)
    return sms_svc.dispatch_alert(
        db,
        message=payload.message,
        recipient=payload.recipient,
        lat=payload.lat,
        lon=payload.lon,
        investigation_id=payload.investigation_id,
        anomaly_id=payload.anomaly_id,
        triggered_by_confidence=payload.triggered_by_confidence,
        channel=payload.channel,
        force=force,
    )


@router.post("/{alert_id}/retry", response_model=AlertOut, summary="Retry a FAILED alert")
def retry(
    alert_id: int,
    db: Session = Depends(get_db),
    _user: User = Depends(require_min_role(UserRole.REGIONAL)),
) -> Alert:
    try:
        return sms_svc.retry_alert(db, alert_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.post(
    "/test-sms",
    response_model=AlertOut,
    status_code=status.HTTP_201_CREATED,
    summary="Send a test SMS to your own registered mobile number",
)
def test_sms(
    payload: TestSmsRequest | None = None,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> Alert:
    try:
        return sms_svc.send_test_sms(
            db, user, message=(payload.message if payload else None)
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
