"""User persistence helpers used by the auth layer."""
from __future__ import annotations

from collections.abc import Iterable

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.core.security import hash_password, verify_password
from backend.models.user import User, UserRole


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
