"""
Spatial-temporal consistency  (attribution criterion 1).

Ported from SAMUDRA NETRA ``ml/ais/scoring.py::spatiotemporal_score``.

The question: *during the reconstructed release window, was this vessel where
the oil is reconstructed to have come from?* A Gaussian kernel turns the
distance between the vessel's position and the reconstructed origin into an
overlap fraction in [0, 1].

Two modes:
  * **baseline** - distance from the vessel's position (sampled across the
    window) to the reconstructed origin point (or the nearest point of an
    origin cloud), best (max) kernel value over the window.
  * **volume search** - when the full backward-drift track of the origin cloud
    is supplied (Phase 4 ``run_hindcast(include_arrays=True)``), search every
    recorded time step: where was the vessel then, and how much of the oil
    cloud was around it? This is robust to the factor-of-two uncertainty in the
    slick's age because it never fixes a single release time.
"""
from __future__ import annotations

import numpy as np

from backend.ml.attribution.config import ATTRIB
from backend.ml.attribution.geo import haversine_m


def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def _cloud_overlap_ceiling(lat, lon, sigma_km, sample=160, seed=5) -> float:
    """Best overlap any single position could achieve against this cloud."""
    lat = np.asarray(lat, float)
    lon = np.asarray(lon, float)
    if lat.size == 0:
        return 1.0
    if lat.size > sample:
        idx = np.random.default_rng(seed).choice(lat.size, sample, replace=False)
        plat, plon = lat[idx], lon[idx]
    else:
        plat, plon = lat, lon
    best = 0.0
    for pla, plo in zip(plat, plon):
        d = np.asarray(haversine_m(pla, plo, lat, lon), float) / 1000.0
        best = max(best, float(np.mean(np.exp(-0.5 * (d / sigma_km) ** 2))))
    return max(best, 1e-6)


def spatiotemporal_consistency(
    track,
    *,
    origin_point=None,
    origin_points=None,
    window_h,
    pad_h: float | None = None,
    sigma_km: float | None = None,
    hindcast_track: dict | None = None,
    age_prior_window_h=None,
) -> tuple[float, dict]:
    """Return ``(score in [0, 1], detail)`` for one vessel track."""
    sigma_km = float(sigma_km or ATTRIB.ST_KERNEL_SIGMA_KM)
    pad_h = float(pad_h if pad_h is not None else ATTRIB.TIME_PAD_H)
    t0, t1 = float(min(window_h)) - pad_h, float(max(window_h)) + pad_h

    if track.n_points < 2 or track.t_h[-1] < t0 or track.t_h[0] > t1:
        return 0.0, {"finding": "vessel not under AIS observation during the release window"}

    # ---- volume search over the recorded backward-drift track ---------
    if hindcast_track is not None:
        return _volume_search(track, hindcast_track, sigma_km, age_prior_window_h)

    # ---- baseline: vessel-position-in-window vs the reconstructed origin
    grid = np.linspace(max(t0, float(track.t_h[0])), min(t1, float(track.t_h[-1])),
                       max(int((t1 - t0) * 6) + 2, 8))
    v_lat, v_lon = track.position_at(grid)

    if origin_points is not None and len(origin_points):
        o = np.asarray(origin_points, float)
        olat, olon = o[:, 0], o[:, 1]
        overlaps = np.array([
            float(np.mean(np.exp(-0.5 * (haversine_m(la, lo, olat, olon) / 1000.0 / sigma_km) ** 2)))
            for la, lo in zip(v_lat, v_lon)
        ])
        ceiling = _cloud_overlap_ceiling(olat, olon, sigma_km)
        nearest = np.array([
            float(np.min(haversine_m(la, lo, olat, olon)) / 1000.0)
            for la, lo in zip(v_lat, v_lon)
        ])
    else:
        if origin_point is None:
            raise ValueError("provide origin_point, origin_points, or hindcast_track")
        olat, olon = float(origin_point[0]), float(origin_point[1])
        d_km = haversine_m(v_lat, v_lon, olat, olon) / 1000.0
        overlaps = np.exp(-0.5 * (d_km / sigma_km) ** 2)
        ceiling = 1.0
        nearest = d_km

    best_i = int(np.argmax(overlaps))
    best = float(overlaps[best_i])
    best_t = float(grid[best_i])
    best_d = float(nearest[best_i])

    near = np.abs(grid - best_t) <= ATTRIB.ST_CONSISTENCY_HALF_WIN_H
    consistency = float(np.mean(overlaps[near])) if near.any() else best

    rel_overlap = _clip01(best / max(ceiling, 1e-6))
    rel_consist = _clip01(consistency / max(ceiling, 1e-6))
    prox = float(np.exp(-0.5 * (best_d / ATTRIB.ST_PROXIMITY_SIGMA_KM) ** 2))

    prior = 1.0
    if age_prior_window_h is not None:
        lo, hi = float(min(age_prior_window_h)), float(max(age_prior_window_h))
        if best_t < lo - 8.0 or best_t > hi + 8.0:
            prior = 0.80

    score = _clip01((0.42 * prox + 0.36 * rel_overlap + 0.22 * rel_consist) * prior)
    return score, {
        "best_match_time_h": round(best_t, 2),
        "min_distance_km": round(best_d, 2),
        "kernel_overlap": round(best, 4),
        "relative_overlap": round(rel_overlap, 4),
        "consistency": round(rel_consist, 4),
        "age_prior_applied": round(prior, 2),
        "mode": "cloud" if origin_points is not None else "point",
        "finding": (f"Closest approach to the reconstructed origin was {best_d:.1f} km, "
                    f"{abs(best_t):.1f} h before the observation."),
    }


def _volume_search(track, hindcast_track, sigma_km, age_prior_window_h) -> tuple[float, dict]:
    times = np.asarray(hindcast_track["times_h"], float)
    c_lat = np.asarray(hindcast_track["lats"], float)      # (T, N)
    c_lon = np.asarray(hindcast_track["lons"], float)
    if times.size == 0 or c_lat.size == 0:
        return 0.0, {"finding": "empty hindcast track volume"}

    usable = (times >= track.t_h[0] - 0.25) & (times <= track.t_h[-1] + 0.25)
    if not usable.any():
        return 0.0, {"finding": "vessel not observed during the back-tracked interval"}

    ks = np.nonzero(usable)[0]
    v_lat, v_lon = track.position_at(times[ks])
    # vectorised over (K time steps x N particles) - bit-identical to the old
    # per-step loop, ~10x faster on the long tracks (see scripts/profile_pipeline.py)
    d_km = np.asarray(
        haversine_m(v_lat[:, None], v_lon[:, None], c_lat[ks], c_lon[ks]), float
    ) / 1000.0
    overlap = np.mean(np.exp(-0.5 * (d_km / sigma_km) ** 2), axis=1)
    nearest = d_km.min(axis=1)

    best_i = int(np.argmax(overlap))
    best, best_t, best_d = float(overlap[best_i]), float(times[ks[best_i]]), float(nearest[best_i])

    near_best = np.abs(times[ks] - best_t) <= ATTRIB.ST_CONSISTENCY_HALF_WIN_H
    consistency = float(np.mean(overlap[near_best])) if near_best.any() else best

    ceiling = _cloud_overlap_ceiling(c_lat[ks[best_i]], c_lon[ks[best_i]], sigma_km)
    rel_overlap = _clip01(best / max(ceiling, 1e-6))
    rel_consist = _clip01(consistency / max(ceiling, 1e-6))
    prox = float(np.exp(-0.5 * (best_d / ATTRIB.ST_PROXIMITY_SIGMA_KM) ** 2))

    prior = 1.0
    if age_prior_window_h is not None:
        lo, hi = float(min(age_prior_window_h)), float(max(age_prior_window_h))
        if best_t < lo - 8.0 or best_t > hi + 8.0:
            prior = 0.80

    score = _clip01((0.42 * prox + 0.36 * rel_overlap + 0.22 * rel_consist) * prior)
    return score, {
        "best_match_time_h": round(best_t, 2),
        "min_distance_km": round(best_d, 2),
        "relative_overlap": round(rel_overlap, 4),
        "consistency": round(rel_consist, 4),
        "steps_searched": int(ks.size),
        "age_prior_applied": round(prior, 2),
        "mode": "volume",
        "finding": (f"Best space-time match {abs(best_t):.1f} h before the observation, "
                    f"closest approach {best_d:.1f} km, explaining "
                    f"{100 * rel_overlap:.0f}% of the reconstructable oil cloud."),
    }
