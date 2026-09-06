"""Auth wire models: registration, login, current user, token."""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field

from backend.models.user import UserRole


class RegisterRequest(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    phone_number: str | None = Field(default=None, max_length=32)
    role: UserRole = UserRole.PILOT
    jurisdiction_ids: list[int] = Field(default_factory=list)
    # convenience: assign zones by stable code (e.g. ["IN-MH"]) - resolved to ids
    jurisdiction_codes: list[str] = Field(default_factory=list)


class LoginRequest(BaseModel):
    """JSON login body (the OAuth2 form at /auth/login is the primary path)."""

    email: EmailStr
    password: str


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    email: EmailStr
    phone_number: str | None
    role: UserRole
    jurisdiction_ids: list[int]
    active: bool
    created_at: datetime


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds
    user: UserOut
