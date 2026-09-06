"""Vessel wire models."""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class VesselBase(BaseModel):
    mmsi: str = Field(min_length=1, max_length=16)
    imo: str | None = None
    name: str | None = None
    call_sign: str | None = None
    flag: str | None = None
    vessel_type: str | None = None
    length_m: float | None = Field(default=None, ge=0)
    width_m: float | None = Field(default=None, ge=0)


class VesselCreate(VesselBase):
    attributes: dict | None = None


class VesselOut(VesselBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    last_seen_at: datetime | None = None
    attributes: dict | None = None
    created_at: datetime
