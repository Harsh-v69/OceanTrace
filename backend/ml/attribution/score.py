"""
Baseline candidate scorer.

Combines the four normalised physical / spatiotemporal criteria into one
transparent 0-100 score:

    spatiotemporal   origin<->vessel space-time coincidence in the release window
    axis_alignment   vessel course vs the slick's reverse-drift axis
    proximity        closest point of approach to the reconstructed origin
    dwell            share of observed time spent inside the search radius

Every component is in [0, 1] (asserted here) before it is weighted. A proximity
gate scales the total toward 0 for a vessel that had no space-time signal at
all - so a decoy with, say, an AIS blackout on the far side of the domain
cannot outrank the real culprit once the behavioural criteria are added later.
"""
from __future__ import annotations

import numpy as np

from backend.ml.attribution.axis import axis_alignment, slick_axis_from_origin
from backend.ml.attribution.config import ATTRIB, AttributionWeights, assessment_band
from backend.ml.attribution.proximity import (
    closest_point_of_approach,
    cpa_score,
    dwell_fraction,
)
from backend.ml.attribution.spatiotemporal import spatiotemporal_consistency

_COMPONENT_KEYS = ("spatiotemporal", "axis_alignment", "proximity", "dwell")


def score_candidate(
    track,
    *,
    origin_point,
    window_h,
    observed_centroid=None,
    slick_axis_deg: float | None = None,
    origin_points=None,
    hindcast_track: dict | None = None,
    age_prior_window_h=None,
    weights: AttributionWeights | dict | None = None,
    radius_km: float | None = None,
    pad_h: float | None = None,
    proximity_gate: bool | None = None,
) -> dict:
    """Score one vessel track. Returns a fully transparent breakdown."""
    if isinstance(weights, AttributionWeights):
        w = weights.normalized()
    elif isinstance(weights, dict):
        total = sum(weights.values()) or 1.0
        w = {k: float(weights.get(k, 0.0)) / total for k in _COMPONENT_KEYS}
    else:
        w = AttributionWeights().normalized()

    radius_km = float(radius_km or ATTRIB.SEARCH_RADIUS_KM)
    gate_on = ATTRIB.PROXIMITY_GATE_ENABLED if proximity_gate is None else bool(proximity_gate)

    axis_deg = slick_axis_deg
    if axis_deg is None and observed_centroid is not None:
        axis_deg = slick_axis_from_origin(origin_point, observed_centroid)

    s_st, d_st = spatiotemporal_consistency(
        track, origin_point=origin_point, origin_points=origin_points,
        window_h=window_h, pad_h=pad_h, hindcast_track=hindcast_track,
        age_prior_window_h=age_prior_window_h,
    )
    if axis_deg is not None:
        s_ax, d_ax = axis_alignment(track, axis_deg, window_h, pad_h=pad_h,
                                    slick_centroid=observed_centroid)
    else:
        s_ax, d_ax = 0.0, {"finding": "no slick axis available (need observed_centroid or slick_axis_deg)"}

    cpa = closest_point_of_approach(track, origin_point[0], origin_point[1],
                                    window_h=window_h, pad_h=pad_h)
    s_pr = cpa_score(cpa["cpa_km"], radius_km)
    s_dw, d_dw = dwell_fraction(track, origin_point[0], origin_point[1],
                                radius_km=radius_km, window_h=window_h, pad_h=pad_h)

    components = {
        "spatiotemporal": float(s_st),
        "axis_alignment": float(s_ax),
        "proximity": float(s_pr),
        "dwell": float(s_dw),
    }
    for k, v in components.items():
        assert 0.0 - 1e-9 <= v <= 1.0 + 1e-9, f"component {k}={v} not in [0, 1]"

    raw_total = sum(w[k] * components[k] for k in _COMPONENT_KEYS) * 100.0

    gate = 1.0
    if gate_on:
        signal = max(components["spatiotemporal"], components["proximity"])
        gate = float(np.clip(signal / max(ATTRIB.PROXIMITY_GATE_KNEE, 1e-6), 0.0, 1.0))
    total = float(np.clip(raw_total * gate, 0.0, 100.0))

    return {
        "identity": track.identity(),
        "score": round(total, 2),
        "raw_score": round(raw_total, 2),
        "proximity_gate": round(gate, 3),
        "assessment": assessment_band(total),
        "components": {
            k: {
                "value": round(components[k], 4),
                "weight": round(w[k], 4),
                "points": round(w[k] * components[k] * 100.0 * gate, 2),
                "detail": {"spatiotemporal": d_st, "axis_alignment": d_ax,
                           "proximity": {**cpa, "cpa_km": round(float(cpa["cpa_km"]), 3)
                                         if np.isfinite(cpa["cpa_km"]) else None},
                           "dwell": d_dw}[k],
            }
            for k in _COMPONENT_KEYS
        },
        "slick_axis_deg": round(float(axis_deg) % 360.0, 1) if axis_deg is not None else None,
        "cpa_km": round(float(cpa["cpa_km"]), 3) if np.isfinite(cpa["cpa_km"]) else None,
        "cpa_time_h": round(cpa["cpa_time_h"], 3) if cpa["cpa_time_h"] is not None else None,
        "weights": {k: round(w[k], 4) for k in _COMPONENT_KEYS},
        "track": {
            "n_points": track.n_points,
            "first_seen_h": round(float(track.t_h[0]), 2),
            "last_seen_h": round(float(track.t_h[-1]), 2),
            "n_gaps": len(track.gaps),
            "n_segments": len(track.segments),
            "removed_impossible_speed": track.n_removed_speed,
        },
    }
