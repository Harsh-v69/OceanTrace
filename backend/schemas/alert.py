"""Alert wire models."""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from backend.models.alert import AlertChannel, AlertStatus


class AlertCreate(BaseModel):
    investigation_id: int | None = None
    anomaly_id: int | None = None
    channel: AlertChannel = AlertChannel.SMS
    recipient: str = Field(min_length=1, max_length=255)
    message: str = Field(min_length=1)
    triggered_by_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    lat: float | None = Field(default=None, ge=-90, le=90)
    lon: float | None = Field(default=None, ge=-180, le=180)


class TestSmsRequest(BaseModel):
    message: str | None = Field(default=None, max_length=480)


class AlertOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    investigation_id: int | None
    anomaly_id: int | None
    recipient_user_id: int | None
    channel: AlertChannel
    status: AlertStatus
    recipient: str
    message: str
    triggered_by_confidence: float | None
    lat: float | None
    lon: float | None
    jurisdiction_codes: list | None
    fingerprint: str | None
    time_bucket: str | None
    dedup_of_id: int | None
    provider: str | None
    provider_response: dict | None
    attempts: int
    last_error: str | None
    error_log: list | None
    sent_at: datetime | None
    created_at: datetime
