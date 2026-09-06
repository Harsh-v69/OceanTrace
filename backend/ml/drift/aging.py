"""
Slick age inversion  (release-time window).

Ported from SAMUDRA NETRA ``ml/drift/weathering.py`` - only the two pure
functions the hindcast needs. A drifting filament of passive tracer spreads
diffusively, so its cross-drift positions are Gaussian with variance
``sigma^2 = sigma0^2 + 2*K_h*t``. Inverting the observed width gives an age
that depends only on the eddy diffusivity, not on unknown released volume - the
most robust single age observable from one image.
"""
from __future__ import annotations

from backend.ml.drift.config import DRIFT


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
