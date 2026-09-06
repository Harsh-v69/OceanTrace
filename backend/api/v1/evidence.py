"""
Evidence dossier export endpoints.

``GET /investigations/{id}/dossier``       - structured JSON  (mandatory)
``GET /investigations/{id}/dossier.md``    - printable Markdown
``GET /investigations/{id}/dossier.html``  - printable structured view (-> PDF)
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import HTMLResponse, PlainTextResponse
from sqlalchemy.orm import Session

from backend.api._authz import require_row_access
from backend.core.database import get_db
from backend.core.security import get_current_user
from backend.models.investigation import Investigation
from backend.models.user import User
from backend.services.dossier import build_dossier, dossier_html, dossier_markdown

router = APIRouter(prefix="/investigations", tags=["evidence"])


def _load(db: Session, user: User, investigation_id: int) -> Investigation:
    inv = db.get(Investigation, investigation_id)
    if inv is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Investigation not found")
    require_row_access(db, user, lat=inv.centroid_lat, lon=inv.centroid_lon,
                       jurisdiction_id=inv.jurisdiction_id)
    return inv


@router.get("/{investigation_id}/dossier", summary="Evidence dossier (JSON)")
def dossier_json(
    investigation_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> dict:
    _load(db, user, investigation_id)
    return build_dossier(db, investigation_id)


@router.get("/{investigation_id}/dossier.md", response_class=PlainTextResponse,
            summary="Evidence dossier (Markdown)")
def dossier_md(
    investigation_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> str:
    _load(db, user, investigation_id)
    return dossier_markdown(build_dossier(db, investigation_id))


@router.get("/{investigation_id}/dossier.html", response_class=HTMLResponse,
            summary="Evidence dossier (printable HTML / PDF)")
def dossier_html_view(
    investigation_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> str:
    _load(db, user, investigation_id)
    return dossier_html(build_dossier(db, investigation_id))
