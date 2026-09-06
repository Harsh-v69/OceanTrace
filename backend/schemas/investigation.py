"""Investigation wire models."""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from backend.models.investigation import InvestigationStatus


class InvestigationCreate(BaseModel):
    title: str = Field(min_length=1, max_length=255)
    description: str | None = None
    jurisdiction_id: int | None = None
    scene_ref: str | None = None
    detected_at: datetime | None = None
    centroid_lat: float | None = Field(default=None, ge=-90, le=90)
    centroid_lon: float | None = Field(default=None, ge=-180, le=180)


class InvestigationUpdate(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=255)
    description: str | None = None
    status: InvestigationStatus | None = None


class InvestigationOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    reference: str
    title: str
    description: str | None
    status: InvestigationStatus
    jurisdiction_id: int | None
    created_by_id: int | None
    scene_ref: str | None
    detected_at: datetime | None
    centroid_lat: float | None
    centroid_lon: float | None
    summary_metrics: dict | None
    created_at: datetime
    updated_at: datetime
