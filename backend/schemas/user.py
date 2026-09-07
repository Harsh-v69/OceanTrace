"""Wire models for hierarchical user management (Epic 1)."""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field

from backend.models.user import UserRole


class UserCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    role: UserRole
    phone_number: str | None = Field(default=None, max_length=32)
    jurisdiction_ids: list[int] = Field(default_factory=list)
    # convenience: assign zones by stable code (e.g. ["IN-MH"]) - resolved to ids
    jurisdiction_codes: list[str] = Field(default_factory=list)
    active: bool = True


class UserUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=255)
    phone_number: str | None = Field(default=None, max_length=32)
    role: UserRole | None = None
    jurisdiction_ids: list[int] | None = None
    jurisdiction_codes: list[str] | None = None
    active: bool | None = None
    # optional password reset by an admin
    password: str | None = Field(default=None, min_length=8, max_length=128)


class UserAdminOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    email: EmailStr
    phone_number: str | None
    role: UserRole
    jurisdiction_ids: list[int]
    active: bool
    created_at: datetime
    updated_at: datetime
