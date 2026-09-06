"""Jurisdiction wire models."""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict

from backend.models.jurisdiction import JurisdictionType


class JurisdictionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    code: str | None
    name: str
    type: JurisdictionType
    parent_id: int | None
    geometry: dict | None
    created_at: datetime


class JurisdictionResolveOut(BaseModel):
    lat: float
    lon: float
    primary_code: str | None
    chain_codes: list[str]
    jurisdiction_ids: list[int]
