"""Anomaly wire models."""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from backend.models.anomaly import AnomalySource, AnomalyType


class AnomalyCreate(BaseModel):
    investigation_id: int | None = None
    vessel_id: int | None = None
    type: AnomalyType
    source: AnomalySource = AnomalySource.HEURISTIC
    label: str = "Oil-like anomaly"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    score: float | None = None
    occurred_at: datetime | None = None
    lat: float | None = Field(default=None, ge=-90, le=90)
    lon: float | None = Field(default=None, ge=-180, le=180)
    details: dict | None = None


class AnomalyOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    investigation_id: int | None
    vessel_id: int | None
    type: AnomalyType
    source: AnomalySource
    label: str
    confidence: float
    score: float | None
    occurred_at: datetime | None
    lat: float | None
    lon: float | None
    details: dict | None
    reviewed: bool
    created_at: datetime
