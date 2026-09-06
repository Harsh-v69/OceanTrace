"""
ORM models.

Importing this package registers every mapper on ``Base.metadata``; ``init_db``
relies on that. Import order matters only in that every class must be defined
before ``create_all`` / first query - which this module guarantees.
"""
from backend.core.database import Base
from backend.models.alert import Alert, AlertChannel, AlertStatus
from backend.models.anomaly import Anomaly, AnomalySource, AnomalyType
from backend.models.investigation import Investigation, InvestigationStatus
from backend.models.jurisdiction import Jurisdiction, JurisdictionType
from backend.models.mixins import TimestampMixin, utcnow
from backend.models.user import ROLE_LEVEL, User, UserRole
from backend.models.vessel import Vessel

__all__ = [
    "Base",
    "TimestampMixin",
    "utcnow",
    "User",
    "UserRole",
    "ROLE_LEVEL",
    "Jurisdiction",
    "JurisdictionType",
    "Investigation",
    "InvestigationStatus",
    "Anomaly",
    "AnomalyType",
    "AnomalySource",
    "Alert",
    "AlertChannel",
    "AlertStatus",
    "Vessel",
]
