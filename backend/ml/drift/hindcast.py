"""
Backward hindcast - origin reconstruction  (Stage B).

Take the observed slick (centroid, polygon, or mask), apply reverse wind +
current advection plus turbulent diffusion, and produce:

  * a candidate release-origin PROBABILITY FIELD (kernel density on a grid)
  * a release-TIME WINDOW estimate (from the diffusive age and its ~x2
    uncertainty; the near edge stays open so a young spill is not mis-filtered)
  * a point best-estimate + an uncertainty radius + a 2-sigma error ellipse

Diffusion is irreversible: backward integration recovers a distribution, not a
point. Reporting that honestly is the whole design.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from backend.ml.drift.aging import age_from_width, age_uncertainty_window
from backend.ml.drift.config import DRIFT, DriftParams
from backend.ml.drift.geo import LocalFrame, bbox_expand_km, haversine_m
from backend.ml.drift.particles import (
    advect_core,
    seed_cloud,
    seed_from_mask,
    seed_in_polygon,
)


# --------------------------------------------------------------------------- #
# Seeding from whatever the caller has
# --------------------------------------------------------------------------- #
def _ring_latlon(polygon) -> tuple[np.ndarray, np.ndarray]:
    """GeoJSON ring ([lon, lat] pairs) -> (lats, lons) arrays."""
    coords = polygon["coordinates"][0] if isinstance(polygon, dict) else polygon
    arr = np.asarray(coords, float)
    return arr[:, 1], arr[:, 0]


def seed_observed(observed: dict, n: int, seed: int):
    """Return (lat0, lon0, weights, aux) for the observed slick.

    ``observed`` keys (all optional except one locator):
      centroid  [lat, lon]
      polygon   GeoJSON Polygon geometry or a bare [lon,lat] ring
      mask + transform (+ contrast)   pixel slick
      area_km2 / width_m              used to size a centroid-only cloud
    """
    aux: dict = {"seed_kind": None}

    if observed.get("mask") is not None and observed.get("transform") is not None:
        lat0, lon0, wts = seed_from_mask(
            observed["mask"], observed["transform"], observed.get("contrast"), n, seed
        )
        aux["seed_kind"] = "mask"
    elif observed.get("polygon") is not None:
        lats, lons = _ring_latlon(observed["polygon"])
        lat0, lon0 = seed_in_polygon(lats, lons, n, seed)
        wts = np.full(lat0.size, 1.0 / max(lat0.size, 1))
        aux["seed_kind"] = "polygon"
    elif observed.get("centroid") is not None:
        area = float(observed.get("area_km2") or 0.0)
        sigma_m = (
            np.sqrt(area * 1e6 / np.pi) if area > 0
            else float(observed.get("width_m") or DRIFT.DEFAULT_SEED_SIGMA_M) / 2.0
        )
        sigma_m = max(sigma_m, 150.0)
        lat0, lon0 = seed_cloud(observed["centroid"], n, sigma_m, seed)
        wts = np.full(n, 1.0 / n)
        aux["seed_kind"] = "centroid"
        aux["seed_sigma_m"] = float(sigma_m)
    else:
        raise ValueError(
            "observed must carry a 'centroid', a 'polygon', or ('mask' and 'transform')"
        )

    # along-axis coordinate of each seed (for a per-parcel age ramp)
    if lat0.size >= 3:
        frame = LocalFrame(float(lat0.mean()), float(lon0.mean()))
        x, y = frame.to_xy(lat0, lon0)
        pts = np.stack([np.asarray(x), np.asarray(y)], 1)
        pts = pts - pts.mean(0)
        cov = np.cov(pts.T)
        vals, vecs = np.linalg.eigh(cov)
        axis = vecs[:, int(np.argmax(vals))]
        aux["along_m"] = pts @ axis
    else:
        aux["along_m"] = np.zeros(lat0.size)
    return np.asarray(lat0, float), np.asarray(lon0, float), np.asarray(wts, float), aux


def _per_particle_age(aux: dict, age: float, age_lo: float, age_hi: float) -> np.ndarray:
    """Ramp age along the slick's principal axis (older away from the young end)."""
    along = np.asarray(aux.get("along_m"))
    if along.size == 0 or float(along.max() - along.min()) < 1.0:
        return np.full(max(along.size, 1), age)
    a = (along - along.min()) / (along.max() - along.min())      # 0..1
    return age_lo + a * (age_hi - age_lo)


# --------------------------------------------------------------------------- #
# Origin probability field
# --------------------------------------------------------------------------- #
def origin_probability_grid(lats, lons, weights=None, n=None,
                            pad_km=None, smooth=None) -> dict:
    """Kernel-density estimate of the release location on a regular grid."""
    n = int(n or DRIFT.ORIGIN_GRID)
    pad_km = float(pad_km if pad_km is not None else DRIFT.ORIGIN_GRID_PAD_KM)
    smooth = float(smooth if smooth is not None else DRIFT.ORIGIN_GRID_SMOOTH)

    lats = np.asarray(lats, float)
    lons = np.asarray(lons, float)
    w = np.ones(lats.size) if weights is None else np.asarray(weights, float)

    bbox = bbox_expand_km([lons.min(), lats.min(), lons.max(), lats.max()], pad_km)
    W, S, E, N = bbox
    grid_lats = np.linspace(S, N, n)
    grid_lons = np.linspace(W, E, n)

    ri = np.clip(((lats - S) / max(N - S, 1e-9) * (n - 1)).astype(int), 0, n - 1)
    ci = np.clip(((lons - W) / max(E - W, 1e-9) * (n - 1)).astype(int), 0, n - 1)
    acc = np.zeros((n, n), np.float64)
    np.add.at(acc, (ri, ci), w)
    prob = gaussian_filter(acc, sigma=smooth, mode="constant")
    tot = prob.sum()
    if tot > 0:
        prob = prob / tot
    return {"prob": prob, "lats": grid_lats, "lons": grid_lons, "bbox": bbox}


def grid_to_heatmap(grid, max_points=2600, min_rel=0.06) -> list[list[float]]:
    prob = grid["prob"]
    mx = float(prob.max())
    if mx <= 0:
        return []
    rel = prob / mx
    ys, xs = np.nonzero(rel >= min_rel)
    vals = rel[ys, xs]
    if vals.size > max_points:
        keep = np.argsort(vals)[::-1][:max_points]
        ys, xs, vals = ys[keep], xs[keep], vals[keep]
    return [
        [round(float(grid["lats"][y]), 5), round(float(grid["lons"][x]), 5), round(float(v), 4)]
        for y, x, v in zip(ys, xs, vals)
    ]


def confidence_ellipse(lats, lons, n_sigma=2.0, n_points=64) -> list[list[float]]:
    """n-sigma error ellipse of a point cloud, as a closed [lat, lon] ring."""
    lats = np.asarray(lats, float)
    lons = np.asarray(lons, float)
    if lats.size < 3:
        return []
    frame = LocalFrame(float(lats.mean()), float(lons.mean()))
    x, y = frame.to_xy(lats, lons)
    pts = np.stack([np.asarray(x), np.asarray(y)], 1)
    mu = pts.mean(0)
    cov = np.cov((pts - mu).T)
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 1e-6)
    t = np.linspace(0, 2 * np.pi, n_points)
    circle = np.stack([np.cos(t), np.sin(t)], 1)
    ell = circle * (n_sigma * np.sqrt(vals)) @ vecs.T + mu
    la, lo = frame.to_ll(ell[:, 0], ell[:, 1])
    ring = [[round(float(a), 5), round(float(b), 5)] for a, b in zip(la, lo)]
    if ring and ring[0] != ring[-1]:
        ring.append(ring[0])
    return ring


# --------------------------------------------------------------------------- #
# Hindcast
# --------------------------------------------------------------------------- #
def hindcast(
    field,
    observed: dict,
    *,
    params: DriftParams | None = None,
    max_backtrack_h: float | None = None,
    n_particles: int | None = None,
    seed: int = 7,
    land=None,
    record_every: int = 3,
    age_hours: float | None = None,
    age_window_h=None,
) -> dict:
    """Back-track the observed slick to a release-origin distribution + time window.

    ``age_window_h=(lo, hi)`` overrides the diffusive uncertainty window - used by
    the release-time feedback loop once a candidate vessel has pinned the release
    time far more tightly than the slick geometry can.
    """
    p = params or DriftParams()
    max_h = float(max_backtrack_h or DRIFT.MAX_BACKTRACK_H)
    n = int(n_particles or DRIFT.N_PARTICLES)

    lat0, lon0, wts, aux = seed_observed(observed, n, seed)
    if lat0.size == 0:
        raise ValueError("observed slick produced no seed particles")

    # ---- age estimate -> release-time window --------------------------
    age_source = "explicit"
    if age_hours is not None:
        age = float(age_hours)
    elif observed.get("width_m"):
        age = age_from_width(float(observed["width_m"]), p.diffusivity)
        age_source = "diffusive_width"
    else:
        age = max_h * 0.5
        age_source = "prior_midpoint"
    age = float(np.clip(age, DRIFT.RELEASE_WINDOW_NEAR_H, max_h))

    if age_window_h is not None:
        age_lo, age_hi = float(min(age_window_h)), float(max(age_window_h))
        age_source = "ais_feedback"
    else:
        age_lo, age_hi = age_uncertainty_window(age)
    age_lo = float(np.clip(age_lo, DRIFT.RELEASE_WINDOW_NEAR_H, max_h))
    age_hi = float(np.clip(age_hi, age_lo, max_h))

    # ---- integrate the cloud backwards, recording the track -----------
    times, lat_t, lon_t, _beached = advect_core(
        field, lat0, lon0, 0.0, -max_h, params=p, seed=seed,
        record_every=record_every, land=land,
    )
    t_axis = np.asarray(times, float)                     # 0 -> -max_h

    # ---- each parcel's origin = its position at ITS OWN age ----------
    part_age = np.clip(_per_particle_age(aux, age, age_lo, age_hi),
                       DRIFT.RELEASE_WINDOW_NEAR_H, max_h)
    order = np.argsort(t_axis)                            # ascending for np.interp
    ta = t_axis[order]
    org_lat = np.array([np.interp(-part_age[i], ta, lat_t[order, i]) for i in range(lat0.size)])
    org_lon = np.array([np.interp(-part_age[i], ta, lon_t[order, i]) for i in range(lat0.size)])

    grid = origin_probability_grid(org_lat, org_lon, wts)
    gi = np.unravel_index(int(np.argmax(grid["prob"])), grid["prob"].shape)
    best = [float(grid["lats"][gi[0]]), float(grid["lons"][gi[1]])]

    cen_lat, cen_lon = float(org_lat.mean()), float(org_lon.mean())
    spread_km = float(np.mean(haversine_m(org_lat, org_lon, cen_lat, cen_lon)) / 1000.0)

    return {
        "origin_lats": org_lat,
        "origin_lons": org_lon,
        "weights": wts,
        "grid": grid,
        "heatmap": grid_to_heatmap(grid),
        "confidence_ellipse": confidence_ellipse(org_lat, org_lon, n_sigma=2.0),
        "best_estimate": best,
        "centroid": [round(cen_lat, 5), round(cen_lon, 5)],
        "uncertainty_radius_km": round(spread_km, 2),
        "release_window_h": [-age_hi, -DRIFT.RELEASE_WINDOW_NEAR_H],
        "age_prior_window_h": [-age_hi, -age_lo],
        "age_point_estimate_h": round(age, 2),
        "age_source": age_source,
        "seed_kind": aux["seed_kind"],
        "n_particles": int(lat0.size),
        "max_backtrack_h": max_h,
        "track_times_h": t_axis,
        "track_lats": lat_t,
        "track_lons": lon_t,
        "params": p.as_dict(),
    }
