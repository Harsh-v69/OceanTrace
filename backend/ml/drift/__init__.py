"""
Physics-based Lagrangian drift engine (Stage B).

Ported from SAMUDRA NETRA (drift physics = selected [SN], docs/MERGE_ARCHITECTURE.md).
Pure numpy / scipy / shapely, CPU-only, deterministic given a seed.

    from backend.ml.drift import hindcast, forecast
    from backend.ml.drift.metocean import SyntheticMetOcean

All physical constants live in ``backend.ml.drift.config``. The met-ocean
provider abstraction (real / demo / cached) is in ``backend.services.drift``.
"""
from backend.ml.drift.config import DRIFT, METOCEAN, DriftParams
from backend.ml.drift.forecast import forecast
from backend.ml.drift.hindcast import hindcast
from backend.ml.drift.particles import LandMask, advect, advect_core, drift_velocity, rk4_step

__all__ = [
    "DRIFT",
    "METOCEAN",
    "DriftParams",
    "hindcast",
    "forecast",
    "LandMask",
    "advect",
    "advect_core",
    "drift_velocity",
    "rk4_step",
]
