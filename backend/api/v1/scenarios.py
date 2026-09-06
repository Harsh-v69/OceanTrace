"""
Demo-scenario endpoints.

``GET  /scenarios``            - the four deterministic scenarios
``POST /scenarios/{key}/run``  - run one end to end, creating an investigation
                                 (RBAC: the scenario centre must be inside the
                                 caller's maritime jurisdiction)
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.api._authz import require_point_access
from backend.core.database import get_db
from backend.core.security import get_current_user
from backend.models.user import User
from backend.services import orchestration
from backend.simulator import build_scenario, list_scenarios, seed_scenario_zones

router = APIRouter(prefix="/scenarios", tags=["scenarios"])


@router.get("", summary="List the deterministic demo scenarios")
def scenarios(_user: User = Depends(get_current_user)) -> dict:
    return {"scenarios": list_scenarios()}


@router.post("/{key}/run", status_code=status.HTTP_201_CREATED,
             summary="Run a demo scenario end to end")
def run(
    key: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> dict:
    try:
        spec, scene = build_scenario(key)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    seed_scenario_zones(db)
    # a PILOT/REGIONAL may only run a scenario inside their assigned zone
    require_point_access(db, user, spec.center[0], spec.center[1])

    inv, steps = orchestration.run_full_pipeline(db, spec, scene, user)
    m = inv.summary_metrics or {}
    attr = m.get("attribution", {})
    alerts = [a for a in inv.alerts]
    return {
        "investigation_id": inv.id,
        "reference": inv.reference,
        "scenario": key,
        "status": inv.status.value,
        "classification": (m.get("sar") or {}).get("scene_classification"),
        "confidence": (m.get("sar") or {}).get("confidence"),
        "verdict": m.get("verdict"),
        "jurisdiction": m.get("jurisdiction"),
        "prime_suspect": (attr.get("summary") or {}).get("prime_suspect"),
        "n_candidates": len(attr.get("candidates", [])),
        "alert_status": alerts[-1].status.value if alerts else None,
        "pipeline": steps,
        "timings": m.get("timings"),
        "dossier_url": f"/api/v1/investigations/{inv.id}/dossier",
    }
