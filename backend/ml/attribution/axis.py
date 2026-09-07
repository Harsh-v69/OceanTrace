"""
Slick-axis alignment  (attribution criterion 2).

Ported from OceanTrace ``ml/ais/scoring.py::axis_alignment_score``.

A slick laid by a moving, continuously-discharging vessel lies ALONG that
vessel's course. The reverse-drift direction - from the reconstructed origin
toward the observed slick - is that same axis. So we compare the vessel's
heading through the release window against the slick's principal axis and score
the alignment in [0, 1]. This is what distinguishes the ship that laid the
slick from one that merely crossed the area.
"""
from __future__ import annotations

import numpy as np

from backend.ml.attribution.config import ATTRIB
from backend.ml.attribution.geo import (
    haversine_m,
    initial_bearing_deg,
    undirected_axis_difference_deg,
)


def slick_axis_from_origin(origin_point, observed_centroid) -> float:
    """Reverse-drift bearing: origin -> observed slick centroid. Directed, 0-360."""
    return float(initial_bearing_deg(
        origin_point[0], origin_point[1], observed_centroid[0], observed_centroid[1]
    ))


def axis_alignment(
    track,
    axis_deg: float,
    window_h,
    *,
    pad_h: float | None = None,
    tolerance_deg: float | None = None,
    slick_centroid=None,
) -> tuple[float, dict]:
    """Return ``(score in [0, 1], detail)``.

    ``axis_deg`` is treated as an UNDIRECTED axis (a slick has no head/tail from
    geometry alone), so the comparison folds into [0, 90] degrees.
    """
    tol = float(tolerance_deg or ATTRIB.AXIS_TOLERANCE_DEG)
    pad_h = float(pad_h if pad_h is not None else ATTRIB.TIME_PAD_H)
    t0, t1 = float(min(window_h)) - pad_h, float(max(window_h)) + pad_h

    idx = track.window_indices(t0, t1)
    if idx.size < 2:
        return 0.0, {"finding": "no track coverage inside the release window"}

    course = float(initial_bearing_deg(
        track.lat[idx[0]], track.lon[idx[0]], track.lat[idx[-1]], track.lon[idx[-1]]
    ))
    diff = undirected_axis_difference_deg(course, axis_deg)
    score = float(np.clip(1.0 - diff / tol, 0.0, 1.0))

    detail = {
        "vessel_course_deg": round(course, 1),
        "slick_axis_deg": round(float(axis_deg) % 360.0, 1),
        "angular_difference_deg": round(diff, 1),
        "finding": (f"Course {course:.0f} deg lies {diff:.0f} deg off the slick axis"
                    + (" - consistent with having laid it." if diff < 25.0
                       else " - not aligned with the slick.")),
    }
    if slick_centroid is not None:
        mid_lat = float(np.mean(track.lat[idx]))
        mid_lon = float(np.mean(track.lon[idx]))
        detail["distance_to_slick_centroid_km"] = round(
            float(haversine_m(mid_lat, mid_lon, slick_centroid[0], slick_centroid[1])) / 1000.0, 2
        )
    return score, detail
