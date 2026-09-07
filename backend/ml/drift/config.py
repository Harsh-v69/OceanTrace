"""
Drift / hindcast / forecast physical constants - the single configuration layer.

Ported from OceanTrace ``config.DRIFT`` (drift physics = selected [SN] in
docs/MERGE_ARCHITECTURE.md). Every drift module imports its numbers from here;
nothing downstream hard-codes a coefficient. Each value keeps its citation.
"""
from __future__ import annotations

from dataclasses import dataclass


class DRIFT:
    """Lagrangian surface-drift coefficients and integrator settings."""

    # -- advection ------------------------------------------------------
    WIND_DRIFT_FACTOR = 0.030        # 3.0% of wind speed (ASCE 1996; Reed 1999)
    WIND_DEFLECTION_DEG = 17.0       # Coriolis/Ekman deflection (N hemisphere)
    CURRENT_FACTOR = 1.00            # full advection by the surface current
    STOKES_FACTOR = 0.012            # Stokes drift ~1.2% of wind (deep water)

    # -- turbulent diffusion ----------------------------------------
    HORIZONTAL_DIFFUSIVITY = 8.0     # m^2/s eddy diffusivity (coastal/shelf)

    # -- RK4 integrator -------------------------------------------
    TIMESTEP_S = 600                 # 10-minute step
    N_PARTICLES = 900               # Lagrangian cloud size
    MAX_BACKTRACK_H = 48            # hindcast horizon
    MAX_FORECAST_H = 48            # forecast horizon
    FORECAST_HORIZONS_H = (6, 12, 24, 48)

    # -- origin reconstruction --------------------------------------
    ORIGIN_GRID = 140               # origin-probability raster size (n x n)
    ORIGIN_GRID_PAD_KM = 8.0
    ORIGIN_GRID_SMOOTH = 2.2

    # -- diffusive age inversion (release-time window) ---------------
    AGE_INITIAL_WIDTH_M = 450.0     # effective initial filament width
    AGE_SPAN_FACTOR = 3.29          # 5-95 pct span = 3.29 * sigma (Gaussian)
    AGE_K_UNCERTAINTY = 2.2         # eddy diffusivity uncertain ~x2; age ~ 1/K_h
    RELEASE_WINDOW_NEAR_H = 0.5     # near edge of the search window stays open

    # -- seeding when only a centroid is known --------------------
    DEFAULT_SEED_SIGMA_M = 1000.0
    BEACH_MOVE_EPS_DEG = 1e-7       # "has stopped moving" threshold


class METOCEAN:
    """Defaults for a synthesised (demo) wind + current field."""

    MEAN_WIND_SPEED_MS = 7.5
    MEAN_WIND_DIR_FROM_DEG = 225.0      # meteorological convention (FROM)
    MEAN_CURRENT_SPEED_MS = 0.35
    MEAN_CURRENT_DIR_TO_DEG = 60.0      # oceanographic convention (TOWARDS)

    GRID_NX = 26
    GRID_NY = 26
    GRID_NT = 41
    FIELD_PAD_KM = 90.0
    SEED = 20261

    WINDOW_BEFORE_H = -54.0             # default field time span, hours vs t=0
    WINDOW_AFTER_H = 54.0


@dataclass
class DriftParams:
    """Per-run drift coefficients (defaults come from :class:`DRIFT`)."""

    wind_factor: float = DRIFT.WIND_DRIFT_FACTOR
    wind_deflection_deg: float = DRIFT.WIND_DEFLECTION_DEG
    current_factor: float = DRIFT.CURRENT_FACTOR
    stokes_factor: float = DRIFT.STOKES_FACTOR
    diffusivity: float = DRIFT.HORIZONTAL_DIFFUSIVITY
    timestep_s: float = DRIFT.TIMESTEP_S

    def as_dict(self) -> dict:
        return {
            "wind_drift_factor": self.wind_factor,
            "wind_deflection_deg": self.wind_deflection_deg,
            "current_factor": self.current_factor,
            "stokes_factor": self.stokes_factor,
            "horizontal_diffusivity_m2s": self.diffusivity,
            "timestep_s": self.timestep_s,
        }
