"""
Lagrangian oil-drift engine  (Stage B core).

Ported from OceanTrace ``ml/drift/particles.py``.

Physics
-------
A surface oil parcel is advected by the vector sum of

    u_total = a_c * U_current
            + a_w * R(theta) * U_wind10        (wind-induced surface drift)
            + a_s * U_wind10                    (Stokes drift from the wind sea)
            + turbulent random walk

* ``a_w ~ 0.03`` - the classic 3% rule (ASCE 1996; Reed 1999). Oil floats in the
  top ~1 cm, so it feels the wind-drift current directly.
* ``R(theta)`` rotates the wind vector 15-25 deg to the RIGHT in the northern
  hemisphere (LEFT in the southern) - the Ekman/Coriolis deflection.
* ``a_s ~ 0.012`` - Stokes drift of a developed wind sea, downwind.
* random walk: ``sigma_step = sqrt(2 * K_h * dt)`` (Fickian equivalence).

Integration is 4th-order Runge-Kutta in a local East-North metric frame. The ODE
is autonomous only in time, so running it with a NEGATIVE timestep integrates
the trajectory BACKWARDS - the hindcast is the same code path, which is what
keeps the forward/backward pair consistent.
"""
from __future__ import annotations

import numpy as np

from backend.ml.drift.config import DRIFT, DriftParams
from backend.ml.drift.geo import LocalFrame

try:
    import shapely
    from shapely.geometry import MultiPolygon, Polygon, shape as _shape
    from shapely.ops import unary_union

    _HAVE_SHAPELY = True
except Exception:  # pragma: no cover - shapely is a hard dep, guard anyway
    _HAVE_SHAPELY = False


# --------------------------------------------------------------------------- #
# Drift velocity
# --------------------------------------------------------------------------- #
def drift_velocity(field, lat, lon, t_h, p: DriftParams):
    """Total surface drift velocity ``(u_east, v_north)`` in m/s."""
    cu, cv = field.current(lat, lon, t_h)
    wu, wv = field.wind(lat, lon, t_h)

    hemi = 1.0 if np.mean(np.atleast_1d(lat)) >= 0 else -1.0
    th = np.radians(p.wind_deflection_deg) * hemi
    wu_r = wu * np.cos(th) + wv * np.sin(th)
    wv_r = -wu * np.sin(th) + wv * np.cos(th)

    u = p.current_factor * cu + p.wind_factor * wu_r + p.stokes_factor * wu
    v = p.current_factor * cv + p.wind_factor * wv_r + p.stokes_factor * wv
    return u, v


def rk4_step(vel_fn, x, y, t, dt, *, dt_t=None):
    """One classic RK4 step of ``d[x, y]/ds = vel_fn(x, y, t)``.

    ``dt`` advances the state ``(x, y)`` (velocity units x this step). ``dt_t``
    advances the ``t`` argument passed to ``vel_fn`` and defaults to ``dt`` - it
    is separate only when the state step and the field's time axis use different
    units (metres/second state, hours of field time in the drift engine).

    ``vel_fn`` returns ``(u, v)``; ``x, y`` may be scalars or arrays. Exposed
    standalone so the integrator's order of accuracy can be tested directly.
    """
    h = dt if dt_t is None else dt_t
    k1u, k1v = vel_fn(x, y, t)
    k2u, k2v = vel_fn(x + 0.5 * dt * k1u, y + 0.5 * dt * k1v, t + 0.5 * h)
    k3u, k3v = vel_fn(x + 0.5 * dt * k2u, y + 0.5 * dt * k2v, t + 0.5 * h)
    k4u, k4v = vel_fn(x + dt * k3u, y + dt * k3v, t + h)
    nx = x + dt / 6.0 * (k1u + 2 * k2u + 2 * k3u + k4u)
    ny = y + dt / 6.0 * (k1v + 2 * k2v + 2 * k3v + k4v)
    return nx, ny


# --------------------------------------------------------------------------- #
# Coastline constraint
# --------------------------------------------------------------------------- #
class LandMask:
    """Coastline constraint for the drift model.

    Accepts land as GeoJSON polygon geometries and/or a boolean raster with a
    :class:`GeoTransform`. ``is_land`` answers, for a whole particle cloud at
    once, "is this position ashore?". With neither source, nothing is land and
    the forecast simply reports that no coastline was supplied.
    """

    def __init__(self, polygons=None, raster=None, transform=None, buffer_km=0.0):
        self._geom = None
        if polygons and _HAVE_SHAPELY:
            geoms = []
            for poly in polygons:
                if hasattr(poly, "geom_type"):
                    geoms.append(poly)
                elif isinstance(poly, dict):
                    geoms.append(_shape(poly))
                else:
                    geoms.append(Polygon(poly))
            merged = unary_union(geoms) if geoms else None
            if merged is not None and buffer_km:
                merged = merged.buffer(buffer_km / 111.0)  # deg approx
            self._geom = merged

        self._raster = None if raster is None else np.asarray(raster, bool)
        self._gt = transform

    @property
    def any(self) -> bool:
        return self._geom is not None or (
            self._raster is not None and bool(self._raster.any())
        )

    def is_land(self, lat, lon) -> np.ndarray:
        lat = np.atleast_1d(np.asarray(lat, float))
        lon = np.atleast_1d(np.asarray(lon, float))
        out = np.zeros(lat.shape, bool)

        if self._geom is not None and _HAVE_SHAPELY:
            pts = shapely.points(lon, lat)
            out |= shapely.contains(self._geom, pts)

        if self._raster is not None and self._gt is not None:
            col, row = self._gt.ll_to_pixel(lat, lon)
            col = np.round(np.asarray(col)).astype(int)
            row = np.round(np.asarray(row)).astype(int)
            h, w = self._raster.shape
            inside = (col >= 0) & (col < w) & (row >= 0) & (row < h)
            if inside.any():
                out[inside] |= self._raster[row[inside], col[inside]]
        return out


# --------------------------------------------------------------------------- #
# Integrator
# --------------------------------------------------------------------------- #
def advect_core(field, lat0, lon0, t_start_h, t_end_h, params: DriftParams | None = None,
                seed: int = 0, record_every: int = 1, diffusion: bool = True,
                land: "LandMask | None" = None):
    """RK4 integrate a particle cloud from ``t_start_h`` to ``t_end_h``.

    Works forwards (t_end > t_start) or backwards (t_end < t_start). When a
    parcel reaches the shore it STRANDS: it stops advecting and holds its last
    in-water position (oil does not sail over a headland; running backwards, oil
    cannot have originated on land).

    Returns ``(times_h, lats(T, N), lons(T, N), beached(N,))``.
    """
    p = params or DriftParams()
    rng = np.random.default_rng(seed)

    lat = np.atleast_1d(np.asarray(lat0, float)).copy()
    lon = np.atleast_1d(np.asarray(lon0, float)).copy()
    n = lat.size

    total_s = (t_end_h - t_start_h) * 3600.0
    direction = 1.0 if total_s >= 0 else -1.0
    n_steps = max(int(abs(total_s) / p.timestep_s), 1)
    dt = direction * abs(total_s) / n_steps          # signed seconds
    dt_h = dt / 3600.0

    frame = LocalFrame(float(np.mean(lat)), float(np.mean(lon)))
    x, y = frame.to_xy(lat, lon)
    x, y = np.asarray(x, float), np.asarray(y, float)

    sigma = np.sqrt(2.0 * p.diffusivity * abs(dt)) if diffusion else 0.0

    out_t = [t_start_h]
    out_lat = [lat.copy()]
    out_lon = [lon.copy()]

    t_h = t_start_h

    def vel(xx, yy, tt):
        la, lo = frame.to_ll(xx, yy)
        return drift_velocity(field, la, lo, np.full(n, tt), p)

    beached = np.zeros(n, bool)
    use_land = land is not None and land.any

    for step in range(n_steps):
        # dt advances the metric state (seconds); dt_h advances the field clock (hours)
        nx, ny = rk4_step(vel, x, y, t_h, dt, dt_t=dt_h)

        if sigma:
            nx = nx + rng.normal(0.0, sigma, n)
            ny = ny + rng.normal(0.0, sigma, n)

        if use_land:
            la_n, lo_n = frame.to_ll(nx, ny)
            hits = land.is_land(la_n, lo_n) & ~beached
            beached |= hits
            nx = np.where(beached, x, nx)
            ny = np.where(beached, y, ny)

        x, y = nx, ny
        t_h += dt_h
        if (step + 1) % record_every == 0 or step == n_steps - 1:
            la, lo = frame.to_ll(x, y)
            out_t.append(t_h)
            out_lat.append(np.asarray(la, float).copy())
            out_lon.append(np.asarray(lo, float).copy())

    return (np.array(out_t), np.array(out_lat), np.array(out_lon), beached)


def advect(field, lat0, lon0, t_start_h, t_end_h, params: DriftParams | None = None,
           seed: int = 0, record_every: int = 1, diffusion: bool = True,
           land: "LandMask | None" = None):
    """Integrate a particle cloud; returns ``(times, lats, lons)`` only."""
    return advect_core(field, lat0, lon0, t_start_h, t_end_h, params, seed,
                       record_every, diffusion, land)[:3]


# --------------------------------------------------------------------------- #
# Seeding
# --------------------------------------------------------------------------- #
def _point_in_poly(px, py, vx, vy):
    """Vectorised even-odd ray-casting test (px/py scalars or arrays)."""
    px, py = np.asarray(px), np.asarray(py)
    inside = np.zeros(px.shape, bool)
    n = len(vx)
    j = n - 1
    for i in range(n):
        cond = (vy[i] > py) != (vy[j] > py)
        with np.errstate(divide="ignore", invalid="ignore"):
            xin = (vx[j] - vx[i]) * (py - vy[i]) / (vy[j] - vy[i] + 1e-15) + vx[i]
        inside ^= cond & (px < xin)
        j = i
    return inside


def seed_in_polygon(lats, lons, n, seed=0):
    """Uniformly sample ``n`` points inside a polygon (rejection sampling)."""
    rng = np.random.default_rng(seed)
    lats = np.asarray(lats, float)
    lons = np.asarray(lons, float)
    lo_min, lo_max = lons.min(), lons.max()
    la_min, la_max = lats.min(), lats.max()

    out_la, out_lo, tries = [], [], 0
    while len(out_la) < n and tries < 200:
        tries += 1
        k = max(int((n - len(out_la)) * 3), 64)
        ca = rng.uniform(la_min, la_max, k)
        co = rng.uniform(lo_min, lo_max, k)
        inside = _point_in_poly(co, ca, lons, lats)
        out_la.extend(ca[inside].tolist())
        out_lo.extend(co[inside].tolist())
    if not out_la:
        return np.full(n, lats.mean()), np.full(n, lons.mean())
    return np.array(out_la[:n]), np.array(out_lo[:n])


def seed_cloud(centroid, n, sigma_m=DRIFT.DEFAULT_SEED_SIGMA_M, seed=0):
    """Isotropic Gaussian cloud of ``n`` particles around ``(lat, lon)``."""
    rng = np.random.default_rng(seed)
    lat_c, lon_c = float(centroid[0]), float(centroid[1])
    frame = LocalFrame(lat_c, lon_c)
    x = rng.normal(0.0, sigma_m, n)
    y = rng.normal(0.0, sigma_m, n)
    la, lo = frame.to_ll(x, y)
    return np.asarray(la, float), np.asarray(lo, float)


def seed_from_mask(mask, transform, contrast=None, n=None, seed=0):
    """Sample ``n`` particles inside a slick mask, weighted by damping contrast."""
    n = int(n or DRIFT.N_PARTICLES)
    rng = np.random.default_rng(seed)
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return np.zeros(0), np.zeros(0), np.zeros(0)

    if contrast is not None:
        w = np.clip(np.asarray(contrast)[ys, xs], 0.05, None).astype(np.float64)
    else:
        w = np.ones(ys.size)
    w = w / w.sum()

    idx = rng.choice(ys.size, size=min(n, max(ys.size, 1)), replace=True, p=w)
    r = ys[idx] + rng.uniform(-0.5, 0.5, idx.size)
    c = xs[idx] + rng.uniform(-0.5, 0.5, idx.size)
    lats, lons = transform.pixel_to_ll(c, r)
    wsel = w[idx] / max(w[idx].sum(), 1e-12)
    return np.asarray(lats), np.asarray(lons), wsel
