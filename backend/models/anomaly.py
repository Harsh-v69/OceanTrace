"""
Anomaly - one detected event within an investigation.

Per the project rule, SAR detections are labelled an "Oil-like anomaly" rather
than asserted as oil; ``label`` carries that wording to every surface.
"""
from __future__ import annotations

import enum
from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Enum as SAEnum,
    Float,
    ForeignKey,
    Integer,
    String,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.core.database import Base
from backend.models.mixins import TimestampMixin


class AnomalyType(str, enum.Enum):
    OIL_LIKE = "OIL_LIKE"
    AIS_BLACKOUT = "AIS_BLACKOUT"
    SPEED_ANOMALY = "SPEED_ANOMALY"
    COURSE_ANOMALY = "COURSE_ANOMALY"
    ROUTE_DEVIATION = "ROUTE_DEVIATION"
    LOITERING = "LOITERING"
    OTHER = "OTHER"


class AnomalySource(str, enum.Enum):
    SAR = "SAR"
    AIS_AUTOENCODER = "AIS_AUTOENCODER"
    TRAJECTORY_LSTM = "TRAJECTORY_LSTM"
    HEURISTIC = "HEURISTIC"
    FUSION = "FUSION"


class Anomaly(Base, TimestampMixin):
    __tablename__ = "anomalies"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    investigation_id: Mapped[int | None] = mapped_column(
        ForeignKey("investigations.id", ondelete="CASCADE"), nullable=True, index=True
    )
    vessel_id: Mapped[int | None] = mapped_column(
        ForeignKey("vessels.id", ondelete="SET NULL"), nullable=True, index=True
    )
    type: Mapped[AnomalyType] = mapped_column(
        SAEnum(AnomalyType, name="anomaly_type"), nullable=False
    )
    source: Mapped[AnomalySource] = mapped_column(
        SAEnum(AnomalySource, name="anomaly_source"),
        default=AnomalySource.HEURISTIC,
        nullable=False,
    )
    label: Mapped[str] = mapped_column(
        String(64), default="Oil-like anomaly", nullable=False
    )
    confidence: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    occurred_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    lat: Mapped[float | None] = mapped_column(Float, nullable=True)
    lon: Mapped[float | None] = mapped_column(Float, nullable=True)
    details: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    reviewed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    investigation: Mapped["Investigation | None"] = relationship(
        "Investigation", back_populates="anomalies"
    )
    vessel: Mapped["Vessel | None"] = relationship("Vessel")
    alerts: Mapped[list["Alert"]] = relationship("Alert", back_populates="anomaly")

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<Anomaly id={self.id} type={self.type.value} "
            f"confidence={self.confidence:.2f}>"
        )
