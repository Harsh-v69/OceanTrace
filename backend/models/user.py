"""User account + role model."""
from __future__ import annotations

import enum

from sqlalchemy import JSON, Boolean, Enum as SAEnum, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from backend.core.database import Base
from backend.models.mixins import TimestampMixin


class UserRole(str, enum.Enum):
    """Strict RBAC ladder. NATIONAL > REGIONAL > PILOT."""

    PILOT = "PILOT"
    REGIONAL = "REGIONAL"
    NATIONAL = "NATIONAL"


#: Numeric rank for "at least this role" checks.
ROLE_LEVEL: dict[UserRole, int] = {
    UserRole.PILOT: 1,
    UserRole.REGIONAL: 2,
    UserRole.NATIONAL: 3,
}


class User(Base, TimestampMixin):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    email: Mapped[str] = mapped_column(
        String(320), unique=True, index=True, nullable=False
    )
    phone_number: Mapped[str | None] = mapped_column(String(32), nullable=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[UserRole] = mapped_column(
        SAEnum(UserRole, name="user_role"),
        default=UserRole.PILOT,
        nullable=False,
    )
    # List of Jurisdiction ids this user is scoped to. Held as JSON for Phase 2;
    # a join table is the natural next step but not required by the spec.
    jurisdiction_ids: Mapped[list[int]] = mapped_column(
        JSON, default=list, nullable=False
    )
    active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"<User id={self.id} email={self.email!r} role={self.role.value}>"
