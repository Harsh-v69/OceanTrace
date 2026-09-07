"""
Forward forecast  (Stage B).

Propagate particles forward from the observed slick (or an estimated release
state) and snapshot the cloud at 6 h, 12 h, 24 h and 48 h: predicted centroid,
spread, footprint polygon, and - when a coastline is supplied - shoreline
contact (ETA, fraction beached, landfall point).
"""
from __future__ import annotations

import numpy as np

from backend.ml.drift.config import DRIFT, DriftParams
from backend.ml.drift.geo import bbox_of_points, haversine_m, ring_to_geojson
from backend.ml.drift.hindcast import seed_observed
from backend.ml.drift.particles import advect_core

try:
    import shapely
    from shapely import MultiPoint

    _HAVE_SHAPELY = True
except Exception:  # pragma: no cover
    _HAVE_SHAPELY = False


def _hull_ring(lats, lons) -> list[list[float]]:
    """Convex-hull footprint as a closed GeoJSON [lon, lat] ring."""
    lats = np.asarray(lats, float)
    lons = np.asarray(lons, float)
    if lats.size < 3:
        return []
    if _HAVE_SHAPELY:
        hull = MultiPoint(np.column_stack([lons, lats])).convex_hull
        if hull.geom_type != "Polygon":
            return []
        xy = np.asarray(hull.exterior.coords)
        return ring_to_geojson(xy[:, 1], xy[:, 0])
    # fallback: bbox ring
    w, s, e, n = bbox_of_points(lats, lons)
    return ring_to_geojson([s, s, n, n, s], [w, e, e, w, w])


def _snapshot(times, lat_t, lon_t, t_h: float, beach_time) -> dict:
    la = np.array([np.interp(t_h, times, lat_t[:, j]) for j in range(lat_t.shape[1])])
    lo = np.array([np.interp(t_h, times, lon_t[:, j]) for j in range(lon_t.shape[1])])
    cen_lat, cen_lon = float(la.mean()), float(lo.mean())
    d_km = haversine_m(la, lo, cen_lat, cen_lon) / 1000.0
    w, s, e, n = bbox_of_points(la, lo)
    frac_beached = (
        float(np.mean(np.isfinite(beach_time) & (beach_time <= t_h)))
        if beach_time is not None else 0.0
    )
    return {
        "t_h": round(float(t_h), 1),
        "centroid": [round(cen_lat, 5), round(cen_lon, 5)],
        "mean_radius_km": round(float(d_km.mean()), 2),
        "p90_radius_km": round(float(np.percentile(d_km, 90)), 2),
        "bbox": [round(v, 5) for v in (w, s, e, n)],
        "n_particles": int(la.size),
        "fraction_beached": round(frac_beached, 4),
        "footprint": {"type": "Polygon", "coordinates": [_hull_ring(la, lo)]},
    }


def _beach_times(times, lat_t, lon_t, beached) -> np.ndarray:
    """First recorded time after which a stranded parcel stops moving."""
    bt = np.full(lat_t.shape[1], np.nan)
    if not np.any(beached):
        return bt
    moved = (np.abs(np.diff(lat_t, axis=0)) + np.abs(np.diff(lon_t, axis=0))) > DRIFT.BEACH_MOVE_EPS_DEG
    for j in np.nonzero(beached)[0]:
        still = np.nonzero(~moved[:, j])[0]
        if still.size:
            bt[j] = float(times[still[0] + 1])
    return bt


def coastal_impact(times, lat_t, lon_t, beached, beach_time, land) -> dict:
    max_h = float(times[-1])
    if land is None or not getattr(land, "any", False):
        return {"will_beach": False, "note": "No coastline supplied to the forecast"}
    if beached is None or not np.any(beached):
        return {"will_beach": False,
                "note": f"No shoreline contact within {max_h:.0f} h"}

    bt = np.asarray(beach_time, float)
    valid = np.isfinite(bt)
    eta = float(np.nanmin(bt)) if valid.any() else max_h
    frac = float(np.mean(beached))
    j = np.nonzero(beached)[0]

    # the first parcel to strand = the earliest / most specific shoreline contact
    first_j = int(j[np.nanargmin(bt[j])]) if valid.any() else int(j[0])
    first_contact = [round(float(lat_t[-1][first_j]), 5), round(float(lon_t[-1][first_j]), 5)]

    # a decimated set of stranding coordinates for the map (<= 40 points)
    take = j if j.size <= 40 else j[np.linspace(0, j.size - 1, 40).astype(int)]
    contact_points = [
        [round(float(lat_t[-1][p]), 5), round(float(lon_t[-1][p]), 5)] for p in take
    ]
    return {
        "will_beach": True,
        "eta_hours": round(eta, 2),
        "fraction_beached": round(frac, 4),
        "n_parcels_beached": int(np.sum(beached)),
        "landfall_point": [
            round(float(np.mean(lat_t[-1][j])), 5),
            round(float(np.mean(lon_t[-1][j])), 5),
        ],
        "first_contact_point": first_contact,
        "first_contact_eta_h": round(float(np.nanmin(bt[j])), 2) if valid.any() else round(max_h, 2),
        "contact_points": contact_points,
        "note": (f"Shoreline contact in {eta:.1f} h at "
                 f"{first_contact[0]:.3f}, {first_contact[1]:.3f}; "
                 f"{100 * frac:.0f}% of the modelled oil strands within {max_h:.0f} h"),
    }


def forecast(
    field,
    observed: dict,
    *,
    params: DriftParams | None = None,
    horizons_h=None,
    n_particles: int | None = None,
    seed: int = 11,
    land=None,
    record_every: int = 2,
) -> dict:
    """Forward-drift the observed slick and snapshot it at each horizon."""
    p = params or DriftParams()
    horizons = tuple(sorted(float(h) for h in (horizons_h or DRIFT.FORECAST_HORIZONS_H)))
    max_h = float(max(horizons))
    n = int(n_particles or DRIFT.N_PARTICLES)

    lat0, lon0, wts, aux = seed_observed(observed, n, seed)
    times, lat_t, lon_t, beached = advect_core(
        field, lat0, lon0, 0.0, max_h, params=p, seed=seed,
        record_every=record_every, land=land,
    )
    times = np.asarray(times, float)
    beach_time = _beach_times(times, lat_t, lon_t, beached)

    snapshots = [
        _snapshot(times, lat_t, lon_t, h, beach_time) for h in horizons if h <= max_h
    ]
    start_lat, start_lon = float(lat0.mean()), float(lon0.mean())

    return {
        "start_centroid": [round(start_lat, 5), round(start_lon, 5)],
        "seed_kind": aux["seed_kind"],
        "n_particles": int(lat0.size),
        "horizons_h": list(horizons),
        "horizons": snapshots,
        "coastal_impact": coastal_impact(times, lat_t, lon_t, beached, beach_time, land),
        "track_times_h": times,
        "track_lats": lat_t,
        "track_lons": lon_t,
        "params": p.as_dict(),
    }
