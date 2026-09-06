"""
Maritime jurisdiction / boundary model.

``geometry`` holds a GeoJSON geometry object (Polygon / MultiPolygon) as JSON;
``code`` is a stable slug (``IN-MH``, ``IN-WEST``, ``IN-NATIONAL``) so users and
tests can reference a zone without depending on an auto-increment id.

On PostGIS the ``geometry`` column becomes
``geoalchemy2.Geometry('MULTIPOLYGON', srid=4326)`` and the point-in-polygon
lookup in ``services/jurisdiction.py`` moves into an ``ST_Covers`` query - the
call sites do not change.
"""
from __future__ import annotations

import enum

from sqlalchemy import JSON, Enum as SAEnum, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.core.database import Base
from backend.models.mixins import TimestampMixin


class JurisdictionType(str, enum.Enum):
    NATION = "NATION"
    MARITIME_REGION = "MARITIME_REGION"    # West / East / South / Islands
    COASTAL_STATE = "COASTAL_STATE"        # per-state offshore responsibility zone
    REGION = "REGION"                      # generic (kept for back-compat)
    PORT = "PORT"
    EEZ = "EEZ"
    ZONE = "ZONE"


class Jurisdiction(Base, TimestampMixin):
    __tablename__ = "jurisdictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    code: Mapped[str | None] = mapped_column(String(32), unique=True, index=True, nullable=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    type: Mapped[JurisdictionType] = mapped_column(
        SAEnum(JurisdictionType, name="jurisdiction_type"), nullable=False
    )
    # GeoJSON geometry, e.g. {"type": "Polygon", "coordinates": [...]}.
    geometry: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    parent_id: Mapped[int | None] = mapped_column(
        ForeignKey("jurisdictions.id", ondelete="SET NULL"), nullable=True, index=True
    )

    parent: Mapped["Jurisdiction | None"] = relationship(
        "Jurisdiction", remote_side="Jurisdiction.id", backref="children"
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Jurisdiction {self.code or self.id} {self.name!r} type={self.type.value}>"
