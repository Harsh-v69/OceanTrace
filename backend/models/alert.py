"""
Alert - an outbound notification raised when an anomaly clears the confidence
threshold. Delivery goes through the ``SmsProvider`` interface; the Mock
provider records a ``MOCKED`` status offline.

The row is a complete audit trail: ``fingerprint`` + ``dedup_of_id`` record
suppression during repeated runs; ``attempts`` / ``last_error`` / ``error_log``
record every delivery attempt for retry support.
"""
from __future__ import annotations

import enum
from datetime import datetime

from sqlalchemy import (
    JSON,
    DateTime,
    Enum as SAEnum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.core.database import Base
from backend.models.mixins import TimestampMixin


class AlertChannel(str, enum.Enum):
    SMS = "SMS"
    EMAIL = "EMAIL"
    WEBHOOK = "WEBHOOK"


class AlertStatus(str, enum.Enum):
    PENDING = "PENDING"
    SENT = "SENT"
    FAILED = "FAILED"
    MOCKED = "MOCKED"
    SUPPRESSED = "SUPPRESSED"     # a duplicate that was not dispatched


class Alert(Base, TimestampMixin):
    __tablename__ = "alerts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    investigation_id: Mapped[int | None] = mapped_column(
        ForeignKey("investigations.id", ondelete="CASCADE"), nullable=True, index=True
    )
    anomaly_id: Mapped[int | None] = mapped_column(
        ForeignKey("anomalies.id", ondelete="SET NULL"), nullable=True, index=True
    )
    recipient_user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True
    )

    channel: Mapped[AlertChannel] = mapped_column(
        SAEnum(AlertChannel, name="alert_channel"), default=AlertChannel.SMS, nullable=False
    )
    status: Mapped[AlertStatus] = mapped_column(
        SAEnum(AlertStatus, name="alert_status"), default=AlertStatus.PENDING, nullable=False
    )
    recipient: Mapped[str] = mapped_column(String(255), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    triggered_by_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)

    # -- geo context (for geometry-similarity dedup) --------------------
    lat: Mapped[float | None] = mapped_column(Float, nullable=True)
    lon: Mapped[float | None] = mapped_column(Float, nullable=True)
    geometry: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    jurisdiction_codes: Mapped[list | None] = mapped_column(JSON, nullable=True)

    # -- dedup ------------------------------------------------------
    fingerprint: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    time_bucket: Mapped[str | None] = mapped_column(String(32), nullable=True)
    dedup_of_id: Mapped[int | None] = mapped_column(
        ForeignKey("alerts.id", ondelete="SET NULL"), nullable=True, index=True
    )

    # -- delivery audit trail -----------------------------------------
    provider: Mapped[str | None] = mapped_column(String(32), nullable=True)
    provider_response: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    attempts: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_log: Mapped[list | None] = mapped_column(JSON, default=list, nullable=True)
    sent_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    investigation: Mapped["Investigation | None"] = relationship(
        "Investigation", back_populates="alerts"
    )
    anomaly: Mapped["Anomaly | None"] = relationship("Anomaly", back_populates="alerts")

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<Alert id={self.id} {self.channel.value} {self.status.value} "
            f"to={self.recipient!r} attempts={self.attempts}>"
        )
