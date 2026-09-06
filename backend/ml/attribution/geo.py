"""Geodesy helpers for attribution (re-exports + angle utilities)."""
from __future__ import annotations

import numpy as np

from backend.ml.drift.geo import (  # noqa: F401  (re-exported)
    LocalFrame,
    bbox_of_points,
    destination,
    haversine_m,
    initial_bearing_deg,
)


def angular_difference_deg(a, b):
    """Smallest signed difference ``b - a`` wrapped to [-180, 180]."""
    return (np.asarray(b, float) - np.asarray(a, float) + 180.0) % 360.0 - 180.0


def undirected_axis_difference_deg(course_deg, axis_deg) -> float:
    """Difference between a directed course and an UNDIRECTED axis, folded to [0, 90]."""
    diff = abs(float(angular_difference_deg(course_deg, axis_deg)))
    return min(diff, 180.0 - diff)


def circular_mean_deg(angles_deg) -> float:
    a = np.radians(np.asarray(angles_deg, float))
    return float((np.degrees(np.arctan2(np.sin(a).mean(), np.cos(a).mean())) + 360.0) % 360.0)
