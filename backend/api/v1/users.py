"""
Hierarchical user management (Epic 1).

Open self-registration is disabled (``/auth/register`` -> 403 unless
``ALLOW_OPEN_REGISTRATION``). Accounts are created and maintained here:

* ``NATIONAL`` - manages every account (any role, any jurisdiction).
* ``REGIONAL`` - manages ``PILOT`` accounts whose assigned zones lie entirely
  inside the REGIONAL user's own jurisdiction closure.
* ``PILOT``    - no access (403).
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.core.database import get_db
from backend.core.security import get_current_user, require_min_role
from backend.models.user import User, UserRole
from backend.schemas.user import UserAdminOut, UserCreate, UserUpdate
from backend.services import jurisdiction as juris_svc
from backend.services import users as user_svc

router = APIRouter(prefix="/users", tags=["users"])

_MANAGER = require_min_role(UserRole.REGIONAL)   # PILOT never reaches these routes


def _resolve_zone_ids(db: Session, ids: list[int] | None, codes: list[str] | None) -> list[int]:
    out = list(ids or [])
    if codes:
        out += juris_svc.resolve_codes_to_ids(db, codes)
    return sorted(set(int(i) for i in out))


def _forbid(msg: str) -> HTTPException:
    return HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=msg)


@router.get("", response_model=list[UserAdminOut], summary="List manageable accounts")
def list_users(
    db: Session = Depends(get_db),
    actor: User = Depends(_MANAGER),
) -> list[User]:
    return user_svc.list_manageable_users(db, actor)


@router.post("", response_model=UserAdminOut, status_code=status.HTTP_201_CREATED,
             summary="Create an account")
def create_user(
    payload: UserCreate,
    db: Session = Depends(get_db),
    actor: User = Depends(_MANAGER),
) -> User:
    if user_svc.get_user_by_email(db, payload.email) is not None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT,
                            detail="Email already registered")

    jids = _resolve_zone_ids(db, payload.jurisdiction_ids, payload.jurisdiction_codes)
    if not user_svc.can_manage(db, actor, target_role=payload.role,
                               target_jurisdiction_ids=jids):
        raise _forbid(
            "Your role cannot create a "
            f"{payload.role.value} account with those jurisdictions."
        )
    if payload.role != UserRole.NATIONAL and not jids:
        raise HTTPException(status_code=422,
                            detail="PILOT and REGIONAL accounts need at least one jurisdiction.")

    return user_svc.create_user(
        db, name=payload.name, email=payload.email, password=payload.password,
        role=payload.role, phone_number=payload.phone_number,
        jurisdiction_ids=jids, active=payload.active,
    )


def _load_managed(db: Session, actor: User, user_id: int) -> User:
    target = db.get(User, user_id)
    if target is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    if not user_svc.can_manage(
        db, actor, target_role=target.role,
        target_jurisdiction_ids=target.jurisdiction_ids, target=target,
    ):
        raise _forbid("This account is outside your management scope.")
    return target


@router.get("/{user_id}", response_model=UserAdminOut, summary="Read one account")
def get_user(
    user_id: int,
    db: Session = Depends(get_db),
    actor: User = Depends(_MANAGER),
) -> User:
    return _load_managed(db, actor, user_id)


@router.patch("/{user_id}", response_model=UserAdminOut, summary="Update an account")
def update_user(
    user_id: int,
    payload: UserUpdate,
    db: Session = Depends(get_db),
    actor: User = Depends(_MANAGER),
) -> User:
    target = _load_managed(db, actor, user_id)

    new_role = payload.role or target.role
    new_zones = target.jurisdiction_ids
    if payload.jurisdiction_ids is not None or payload.jurisdiction_codes is not None:
        new_zones = _resolve_zone_ids(db, payload.jurisdiction_ids, payload.jurisdiction_codes)

    # the resulting account must still be inside the actor's management scope
    if not user_svc.can_manage(db, actor, target_role=new_role,
                               target_jurisdiction_ids=new_zones):
        raise _forbid("That change would move the account outside your management scope.")

    # protect the last active NATIONAL admin
    demoting = new_role != UserRole.NATIONAL and target.role == UserRole.NATIONAL
    disabling = payload.active is False and target.active
    if (demoting or disabling) and target.role == UserRole.NATIONAL:
        if user_svc.count_active_national(db, exclude_id=target.id) == 0:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT,
                                detail="Cannot disable or demote the last active NATIONAL admin.")

    return user_svc.update_user(
        db, target,
        name=payload.name, phone_number=payload.phone_number,
        role=payload.role,
        jurisdiction_ids=(new_zones if (payload.jurisdiction_ids is not None
                                        or payload.jurisdiction_codes is not None) else None),
        active=payload.active, password=payload.password,
    )


@router.post("/{user_id}/disable", response_model=UserAdminOut, summary="Disable an account")
def disable_user(
    user_id: int,
    db: Session = Depends(get_db),
    actor: User = Depends(_MANAGER),
) -> User:
    target = _load_managed(db, actor, user_id)
    if target.id == actor.id:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT,
                            detail="You cannot disable your own account.")
    if (target.role == UserRole.NATIONAL
            and user_svc.count_active_national(db, exclude_id=target.id) == 0):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT,
                            detail="Cannot disable the last active NATIONAL admin.")
    return user_svc.update_user(db, target, active=False)


@router.post("/{user_id}/enable", response_model=UserAdminOut, summary="Re-enable an account")
def enable_user(
    user_id: int,
    db: Session = Depends(get_db),
    actor: User = Depends(_MANAGER),
) -> User:
    target = _load_managed(db, actor, user_id)
    return user_svc.update_user(db, target, active=True)


@router.get("/me/scope", summary="What the caller may manage (for the UI)")
def my_scope(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> dict:
    can_create: list[str] = []
    if user.role == UserRole.NATIONAL:
        can_create = [r.value for r in UserRole]
    elif user.role == UserRole.REGIONAL:
        can_create = [UserRole.PILOT.value]
    closure = juris_svc.accessible_jurisdiction_ids(db, user)
    return {
        "role": user.role.value,
        "can_manage_users": user.role != UserRole.PILOT,
        "can_create_roles": can_create,
        "jurisdiction_closure_ids": (None if closure is None else sorted(closure)),
    }
