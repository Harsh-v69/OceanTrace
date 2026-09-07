"""User persistence helpers used by the auth layer + hierarchical management."""
from __future__ import annotations

from collections.abc import Iterable

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.core.logging import get_logger
from backend.core.security import hash_password, verify_password
from backend.models.user import User, UserRole

log = get_logger("backend.services.users")


def get_user_by_email(db: Session, email: str) -> User | None:
    return db.execute(
        select(User).where(User.email == email.strip().lower())
    ).scalar_one_or_none()


def get_user(db: Session, user_id: int) -> User | None:
    return db.get(User, user_id)


def create_user(
    db: Session,
    *,
    name: str,
    email: str,
    password: str,
    role: UserRole = UserRole.PILOT,
    phone_number: str | None = None,
    jurisdiction_ids: Iterable[int] | None = None,
    active: bool = True,
) -> User:
    user = User(
        name=name.strip(),
        email=email.strip().lower(),
        phone_number=phone_number,
        password_hash=hash_password(password),
        role=role,
        jurisdiction_ids=list(jurisdiction_ids or []),
        active=active,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def authenticate(db: Session, email: str, password: str) -> User | None:
    user = get_user_by_email(db, email)
    if user is None:
        return None
    if not verify_password(password, user.password_hash):
        return None
    return user


# --------------------------------------------------------------------------- #
# Hierarchical management (Epic 1)
#
#   NATIONAL - may manage every account (any role, any jurisdiction).
#   REGIONAL - may manage PILOT accounts whose jurisdiction_ids are entirely
#              inside the REGIONAL user's own jurisdiction closure.
#   PILOT    - may not manage any account.
# --------------------------------------------------------------------------- #
def _closure(db: Session, actor: User) -> set[int] | None:
    from backend.services import jurisdiction as juris

    return juris.accessible_jurisdiction_ids(db, actor)


def can_manage(
    db: Session,
    actor: User,
    *,
    target_role: UserRole,
    target_jurisdiction_ids: Iterable[int] | None,
    target: User | None = None,
) -> bool:
    """True if ``actor`` may create/modify a user with this role + zones."""
    if target is not None and target.id == actor.id:
        # self-service (name/phone/password) is handled on /auth/me, not here
        return False
    if actor.role == UserRole.NATIONAL:
        return True
    if actor.role != UserRole.REGIONAL:
        return False
    if target_role != UserRole.PILOT:
        return False
    zones = {int(i) for i in (target_jurisdiction_ids or [])}
    if not zones:
        return False
    closure = _closure(db, actor)
    if closure is None:               # defensive - REGIONAL is never "all"
        return True
    return zones.issubset(closure)


def list_manageable_users(db: Session, actor: User) -> list[User]:
    rows = db.execute(select(User).order_by(User.id)).scalars().all()
    if actor.role == UserRole.NATIONAL:
        return list(rows)
    if actor.role != UserRole.REGIONAL:
        return []
    closure = _closure(db, actor) or set()
    out = []
    for u in rows:
        if u.role != UserRole.PILOT:
            continue
        zones = {int(i) for i in (u.jurisdiction_ids or [])}
        if zones and zones.issubset(closure):
            out.append(u)
    return out


def count_active_national(db: Session, *, exclude_id: int | None = None) -> int:
    stmt = select(func.count()).select_from(User).where(
        User.role == UserRole.NATIONAL, User.active.is_(True)
    )
    if exclude_id is not None:
        stmt = stmt.where(User.id != exclude_id)
    return int(db.execute(stmt).scalar_one())


def update_user(
    db: Session,
    user: User,
    *,
    name: str | None = None,
    phone_number: str | None = None,
    role: UserRole | None = None,
    jurisdiction_ids: Iterable[int] | None = None,
    active: bool | None = None,
    password: str | None = None,
) -> User:
    if name is not None:
        user.name = name.strip()
    if phone_number is not None:
        user.phone_number = phone_number or None
    if role is not None:
        user.role = role
    if jurisdiction_ids is not None:
        user.jurisdiction_ids = [int(i) for i in jurisdiction_ids]
    if active is not None:
        user.active = bool(active)
    if password:
        user.password_hash = hash_password(password)
    db.commit()
    db.refresh(user)
    return user


# --------------------------------------------------------------------------- #
# Idempotent default-account seeding
# --------------------------------------------------------------------------- #
_DEFAULT_ACCOUNTS = [
    ("national@oceantrace.gov.in", "National Operations Admin", UserRole.NATIONAL, []),
    ("regional@oceantrace.gov.in", "Western Region Admin", UserRole.REGIONAL, ["IN-WEST"]),
    ("pilot@oceantrace.gov.in", "Mumbai Coastal Pilot", UserRole.PILOT, ["IN-MH"]),
]


def seed_default_users(db: Session, *, password: str | None = None) -> int:
    """
    Create one account per role if the email does not already exist. Idempotent -
    an existing account is never touched (no password reset, no zone change).
    Returns the number of accounts created.
    """
    from backend.core.config import settings
    from backend.services import jurisdiction as juris

    pw = password or settings.DEFAULT_USER_PASSWORD
    created = 0
    for email, name, role, codes in _DEFAULT_ACCOUNTS:
        if get_user_by_email(db, email) is not None:
            continue
        jids = juris.resolve_codes_to_ids(db, codes) if codes else []
        create_user(
            db, name=name, email=email, password=pw, role=role,
            jurisdiction_ids=jids, active=True,
        )
        created += 1
    if created:
        log.info("seeded %d default user account(s)", created)
    return created
