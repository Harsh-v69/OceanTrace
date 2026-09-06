"""
Geodesy helpers for the drift engine.

Re-exports the shared primitives from ``backend.ml.sar.geo`` and adds the few
extras the Lagrangian model needs (bbox expansion, bearings, destination point).
"""
from __future__ import annotations

import numpy as np

from backend.ml.sar.geo import (  # noqa: F401  (re-exported)
    EARTH_R,
    GeoTransform,
    LocalFrame,
    bbox_center,
    bbox_size_m,
    feature,
    feature_collection,
    haversine_m,
    m_per_deg,
    ring_to_geojson,
)

_DEG = np.pi / 180.0


def initial_bearing_deg(lat1, lon1, lat2, lon2):
    """Forward azimuth (0 = North, clockwise) from point 1 to point 2."""
    lat1, lon1, lat2, lon2 = map(np.asarray, (lat1, lon1, lat2, lon2))
    p1, p2 = lat1 * _DEG, lat2 * _DEG
    dl = (lon2 - lon1) * _DEG
    y = np.sin(dl) * np.cos(p2)
    x = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)
    return (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0


def destination(lat, lon, bearing_deg, dist_m):
    """Point reached travelling ``dist_m`` along ``bearing_deg`` from (lat, lon)."""
    lat, lon, bearing_deg, dist_m = map(np.asarray, (lat, lon, bearing_deg, dist_m))
    p1, l1 = lat * _DEG, lon * _DEG
    th, d = bearing_deg * _DEG, np.asarray(dist_m) / EARTH_R
    p2 = np.arcsin(np.sin(p1) * np.cos(d) + np.cos(p1) * np.sin(d) * np.cos(th))
    l2 = l1 + np.arctan2(
        np.sin(th) * np.sin(d) * np.cos(p1),
        np.cos(d) - np.sin(p1) * np.sin(p2),
    )
    return np.degrees(p2), (np.degrees(l2) + 540.0) % 360.0 - 180.0


def bbox_expand_km(bbox, km):
    """Grow a ``[w, s, e, n]`` bbox by ``km`` on every side."""
    w, s, e, n = bbox
    m_lat, m_lon = m_per_deg((s + n) / 2.0)
    dlat, dlon = km * 1000.0 / m_lat, km * 1000.0 / m_lon
    return [w - dlon, s - dlat, e + dlon, n + dlat]


def bbox_of_points(lats, lons):
    lats = np.asarray(list(lats), float)
    lons = np.asarray(list(lons), float)
    return [float(lons.min()), float(lats.min()), float(lons.max()), float(lats.max())]
