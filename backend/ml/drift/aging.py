"""
Slick age inversion + basic weathering.

*Age inversion* (release-time window): a drifting filament of passive tracer
spreads diffusively, so its cross-drift positions are Gaussian with variance
``sigma^2 = sigma0^2 + 2*K_h*t``. Inverting the observed width gives an age
that depends only on the eddy diffusivity, not on unknown released volume - the
most robust single age observable from one image.

*Weathering* (Epic 4.2): a coarse forward mass balance - Fingas log-law
evaporation plus Fay gravity-viscous spreading - giving evaporated fraction,
remaining volume, slick area and mean thickness over the forecast horizons.
Order-of-magnitude only; every number carries its assumptions.
"""
from __future__ import annotations

import math

from backend.ml.drift.config import DRIFT, WEATHERING


def age_from_width(width_m, k_h=DRIFT.HORIZONTAL_DIFFUSIVITY,
                   w0_m=DRIFT.AGE_INITIAL_WIDTH_M,
                   span_factor=DRIFT.AGE_SPAN_FACTOR) -> float:
    """Slick age in hours from its cross-drift width.

    ``width_m`` is a 5th-95th percentile SPAN, which for a Gaussian equals
    ``3.29 * sigma`` (``span_factor``). ``w0_m`` is the effective initial
    filament width: wake turbulence and the initial gravity-spreading phase
    widen a fresh discharge to a few hundred metres within minutes, so ~450 m
    is the realistic floor rather than the vessel's beam.
    """
    sigma = max(float(width_m), 1.0) / span_factor
    sigma0 = max(float(w0_m), 1.0) / span_factor
    t_s = max(sigma ** 2 - sigma0 ** 2, 0.0) / (2.0 * max(k_h, 1e-6))
    return float(t_s / 3600.0)


def age_uncertainty_window(age_h, k_factor=DRIFT.AGE_K_UNCERTAINTY) -> tuple[float, float]:
    """Plausible age range given the ~x2 uncertainty in eddy diffusivity.

    Age scales as ``1/K_h``, so the honest output is an interval. A single-valued
    age would be false precision and could let the AIS search window exclude the
    real culprit.
    """
    a = max(float(age_h), 0.1)
    return float(a / k_factor), float(a * k_factor)


# --------------------------------------------------------------------------- #
# Weathering (Epic 4.2)
# --------------------------------------------------------------------------- #
def evaporated_fraction(t_h: float, *, oil: str | None = None,
                        temp_c: float | None = None) -> float:
    """Fingas log-law evaporated mass fraction after ``t_h`` hours (0..ceiling)."""
    a, b = WEATHERING.EVAP_COEFFS.get(oil or WEATHERING.DEFAULT_OIL,
                                     WEATHERING.EVAP_COEFFS[WEATHERING.DEFAULT_OIL])
    T = WEATHERING.DEFAULT_WATER_TEMP_C if temp_c is None else float(temp_c)
    t_min = max(float(t_h), 0.0) * 60.0
    if t_min < 1.0:
        return 0.0
    pct = (a + b * T) * math.log(t_min)          # percent
    return float(min(max(pct / 100.0, 0.0), WEATHERING.MAX_EVAPORATED_FRACTION))


def spread_area_km2(volume_m3: float, t_h: float) -> float:
    """Fay gravity-viscous slick area (km^2) for ``volume_m3`` after ``t_h`` h."""
    V = max(float(volume_m3), 1e-6)
    delta = 1.0 - WEATHERING.OIL_DENSITY_KG_M3 / WEATHERING.WATER_DENSITY_KG_M3
    delta = max(delta, 1e-3)
    t_s = max(float(t_h), 0.0) * 3600.0
    if t_s <= 0.0:
        return 0.0
    term = (delta * WEATHERING.GRAVITY * V * V
            / math.sqrt(WEATHERING.WATER_KINEMATIC_VISCOSITY)) ** (1.0 / 3.0)
    area_m2 = (WEATHERING.FAY_K2 ** 2) * math.pi * term * math.sqrt(t_s)
    return float(area_m2 / 1.0e6)


def weather_slick(observed_area_km2: float, horizons_h,
                  *, oil: str | None = None, temp_c: float | None = None,
                  initial_thickness_m: float | None = None) -> dict:
    """Forward weathering time series over ``horizons_h``.

    The initial volume is inferred from the SAR-observed area and an assumed
    thin initial film (``INITIAL_SLICK_THICKNESS_M``) - SAR gives area, not
    volume, so this is an explicit, adjustable assumption.
    """
    thick0 = (WEATHERING.INITIAL_SLICK_THICKNESS_M if initial_thickness_m is None
              else float(initial_thickness_m))
    area0_m2 = max(float(observed_area_km2), 0.0) * 1.0e6
    v0_m3 = area0_m2 * thick0
    oil_name = oil or WEATHERING.DEFAULT_OIL

    series = []
    for h in sorted({0, *[float(x) for x in horizons_h]}):
        fe = evaporated_fraction(h, oil=oil_name, temp_c=temp_c)
        v_rem = v0_m3 * (1.0 - fe)
        # area is the larger of the observed footprint and the Fay estimate
        area_km2 = max(float(observed_area_km2), spread_area_km2(v0_m3, h))
        thickness_mm = (v_rem / (area_km2 * 1.0e6) * 1000.0) if area_km2 > 0 else 0.0
        series.append({
            "t_h": round(h, 1),
            "evaporated_fraction": round(fe, 4),
            "volume_remaining_m3": round(v_rem, 1),
            "area_km2": round(area_km2, 3),
            "mean_thickness_mm": round(thickness_mm, 4),
        })
    return {
        "oil_class": oil_name,
        "water_temp_c": (WEATHERING.DEFAULT_WATER_TEMP_C if temp_c is None else float(temp_c)),
        "assumed_initial_thickness_m": thick0,
        "initial_volume_m3": round(v0_m3, 1),
        "model": ("Fingas log-law evaporation + Fay gravity-viscous spreading "
                  "(order-of-magnitude; SAR gives area, initial volume is assumed)"),
        "series": series,
    }
