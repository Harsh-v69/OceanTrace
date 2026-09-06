"""
Proximity and closest point of approach  (attribution criterion 3).

CPA is the minimum distance between a fixed target (the reconstructed origin)
and the vessel's track - computed on the true closest point of each track
SEGMENT, not just the reported fixes, plus the interpolated time at which it
occurred. ``dwell_fraction`` (ported from POSEatSea ``fusion.py`` per
docs/MERGE_ARCHITECTURE.md) measures how much of the vessel's OWN observed time
it spent inside the search radius.
"""
from __future__ import annotations

import numpy as np

from backend.ml.attribution.config import ATTRIB
from backend.ml.attribution.geo import LocalFrame, haversine_m, initial_bearing_deg


def closest_point_of_approach(track, target_lat: float, target_lon: float,
                              *, window_h=None, pad_h: float | None = None) -> dict:
    """True CPA between the vessel's polyline and the target point.

    Returns ``{cpa_km, cpa_time_h, cpa_point [lat, lon], approach_bearing_deg,
    n_segments}``.
    """
    lat, lon, t_h = track.lat, track.lon, track.t_h
    if window_h is not None:
        pad_h = float(pad_h if pad_h is not None else ATTRIB.TIME_PAD_H)
        t0, t1 = float(min(window_h)) - pad_h, float(max(window_h)) + pad_h
        idx = track.window_indices(t0, t1)
        # keep one fix of context on each side so a segment spanning the window is used
        if idx.size:
            i0 = max(int(idx[0]) - 1, 0)
            i1 = min(int(idx[-1]) + 2, t_h.size)
            lat, lon, t_h = lat[i0:i1], lon[i0:i1], t_h[i0:i1]

    if lat.size == 0:
        return {"cpa_km": float("inf"), "cpa_time_h": None, "cpa_point": None,
                "approach_bearing_deg": None, "n_segments": 0}
    if lat.size == 1:
        d = float(haversine_m(lat[0], lon[0], target_lat, target_lon)) / 1000.0
        return {"cpa_km": d, "cpa_time_h": float(t_h[0]),
                "cpa_point": [float(lat[0]), float(lon[0])],
                "approach_bearing_deg": float(initial_bearing_deg(target_lat, target_lon, lat[0], lon[0])),
                "n_segments": 0}

    frame = LocalFrame(target_lat, target_lon)
    x, y = frame.to_xy(lat, lon)
    x, y = np.asarray(x, float), np.asarray(y, float)   # target is the origin of this frame

    best = {"cpa_km": float("inf"), "cpa_time_h": None, "cpa_point": None,
            "approach_bearing_deg": None, "n_segments": int(x.size - 1)}
    for i in range(x.size - 1):
        ax, ay, bx, by = x[i], y[i], x[i + 1], y[i + 1]
        dx, dy = bx - ax, by - ay
        seg_len2 = dx * dx + dy * dy
        s = 0.0 if seg_len2 < 1e-9 else float(np.clip(-(ax * dx + ay * dy) / seg_len2, 0.0, 1.0))
        cx, cy = ax + s * dx, ay + s * dy
        d_km = float(np.hypot(cx, cy)) / 1000.0
        if d_km < best["cpa_km"]:
            cla, clo = frame.to_ll(cx, cy)
            best.update(
                cpa_km=d_km,
                cpa_time_h=float(t_h[i] + s * (t_h[i + 1] - t_h[i])),
                cpa_point=[float(cla), float(clo)],
                approach_bearing_deg=float(initial_bearing_deg(target_lat, target_lon, cla, clo)),
            )
    return best


def cpa_score(cpa_km: float, radius_km: float | None = None) -> float:
    """1 at the origin, decaying to 0 at the search radius (sqrt falloff)."""
    radius_km = float(radius_km or ATTRIB.SEARCH_RADIUS_KM)
    if not np.isfinite(cpa_km) or cpa_km >= radius_km:
        return 0.0
    return float(np.clip(1.0 - (cpa_km / radius_km) ** 0.5, 0.0, 1.0))


def dwell_fraction(track, target_lat: float, target_lon: float,
                   *, radius_km: float | None = None, window_h=None,
                   pad_h: float | None = None) -> tuple[float, dict]:
    """Fraction of the vessel's OBSERVED time spent within ``radius_km`` of the target.

    Normalised against the vessel's own observed span (POSEatSea's ``dwell``),
    not the nominal window - so coverage gaps in the feed are not charged
    against the vessel.
    """
    radius_km = float(radius_km or ATTRIB.SEARCH_RADIUS_KM)
    t_h, lat, lon = track.t_h, track.lat, track.lon
    if window_h is not None:
        pad_h = float(pad_h if pad_h is not None else ATTRIB.TIME_PAD_H)
        t0, t1 = float(min(window_h)) - pad_h, float(max(window_h)) + pad_h
        idx = track.window_indices(t0, t1)
        if idx.size < 2:
            return 0.0, {"finding": "vessel barely observed in the window"}
        t_h, lat, lon = t_h[idx], lat[idx], lon[idx]

    if t_h.size < 2:
        return 0.0, {"finding": "track too short to measure dwell"}

    d_km = haversine_m(lat, lon, target_lat, target_lon) / 1000.0
    inside = d_km <= radius_km
    # trapezoidal time in radius: a segment counts if either endpoint is inside
    seg_h = np.diff(t_h)
    seg_inside = inside[:-1] | inside[1:]
    time_in = float(np.sum(seg_h[seg_inside]))
    observed = float(t_h[-1] - t_h[0])
    frac = float(np.clip(time_in / observed, 0.0, 1.0)) if observed > 0 else 0.0
    return frac, {
        "minutes_in_radius": round(time_in * 60.0, 1),
        "observed_minutes": round(observed * 60.0, 1),
        "radius_km": round(radius_km, 1),
    }
