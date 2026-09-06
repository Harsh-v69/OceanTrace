"""
Met-ocean environment fields (Stage B input).

Ported from SAMUDRA NETRA ``ml/drift/metocean.py`` and refactored so the same
bilinear space-time interpolation serves three cases:

* ``SyntheticMetOcean`` - a deterministic field with the statistics and
  structure of ERA5 wind + HYCOM current (large-scale mean, mesoscale eddies,
  an inertial oscillation, a slow synoptic rotation). This is the offline demo
  data layer.
* a field reconstructed from a local cache (``GriddedMetOcean.from_npz``).
* a field built from real reanalysis grids by a future ``RealMetOceanProvider``
  (same array shapes; only the data source differs).

In an operational deployment the real sources are:
    wind    -> ECMWF ERA5 / ASCAT scatterometer (10 m neutral wind, u10/v10)
    current -> HYCOM GOFS 3.1 / Copernicus CMEMS GLORYS (surface u/v)
    waves   -> WAVEWATCH III (for the Stokes term)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from backend.ml.drift.config import METOCEAN
from backend.ml.drift.geo import LocalFrame, bbox_center, bbox_expand_km

_GRID_KEYS = ("wu", "wv", "cu", "cv")


def _douglas(ws_ms: float) -> int:
    for limit, state in (
        (0.5, 0), (1.5, 1), (3.3, 2), (5.4, 3), (7.9, 4),
        (10.7, 5), (13.8, 6), (17.1, 7), (20.7, 8),
    ):
        if ws_ms < limit:
            return state
    return 9


def _smooth_random_field(rng, ny, nx, nt, scale, sigma_xy=5.0, sigma_t=4.0):
    """Zero-mean Gaussian anomaly field with prescribed correlation lengths."""
    raw = rng.standard_normal((nt, ny, nx))
    out = gaussian_filter(raw, sigma=(sigma_t, sigma_xy, sigma_xy), mode="wrap")
    out -= out.mean()
    sd = out.std()
    return out / (sd if sd > 1e-12 else 1.0) * scale


class GriddedMetOcean:
    """A resolved (lats, lons, times) grid of wind + current, with interpolation.

    ``grids`` is a dict with keys ``wu, wv`` (10 m wind, m/s, east/north) and
    ``cu, cv`` (surface current, m/s, east/north), each shaped ``(nt, ny, nx)``.
    """

    def __init__(self, bbox, lats, lons, times, grids: dict, *, meta: dict | None = None):
        self.bbox = tuple(bbox)
        self.lats = np.asarray(lats, float)
        self.lons = np.asarray(lons, float)
        self.times = np.asarray(times, float)
        self.grids = {k: np.asarray(grids[k], np.float32) for k in _GRID_KEYS}
        self.ny, self.nx, self.nt = len(self.lats), len(self.lons), len(self.times)
        self.meta = dict(meta or {})
        lat_c, _ = bbox_center(self.bbox)
        f_inertial = 2 * 7.292e-5 * np.sin(np.radians(abs(lat_c) + 1e-6))
        self.inertial_period_h = max(2 * np.pi / max(f_inertial, 1e-9) / 3600.0, 12.0)

    # -- interpolation ------------------------------------------------
    def _interp(self, key, lat, lon, t_h):
        g = self.grids[key]
        lat = np.atleast_1d(np.asarray(lat, float))
        lon = np.atleast_1d(np.asarray(lon, float))
        t_h = np.broadcast_to(np.atleast_1d(np.asarray(t_h, float)), lat.shape)

        span_lat = max(self.lats[-1] - self.lats[0], 1e-9)
        span_lon = max(self.lons[-1] - self.lons[0], 1e-9)
        span_t = max(self.times[-1] - self.times[0], 1e-9)
        fy = np.clip((lat - self.lats[0]) / span_lat, 0, 1) * (self.ny - 1)
        fx = np.clip((lon - self.lons[0]) / span_lon, 0, 1) * (self.nx - 1)
        ft = np.clip((t_h - self.times[0]) / span_t, 0, 1) * (self.nt - 1)

        y0, x0, t0 = (np.floor(v).astype(int) for v in (fy, fx, ft))
        y1 = np.minimum(y0 + 1, self.ny - 1)
        x1 = np.minimum(x0 + 1, self.nx - 1)
        t1 = np.minimum(t0 + 1, self.nt - 1)
        wy, wx, wt = fy - y0, fx - x0, ft - t0

        def plane(ti):
            a = g[ti, y0, x0] * (1 - wx) + g[ti, y0, x1] * wx
            b = g[ti, y1, x0] * (1 - wx) + g[ti, y1, x1] * wx
            return a * (1 - wy) + b * wy

        return plane(t0) * (1 - wt) + plane(t1) * wt

    # -- public API -------------------------------------------------
    def wind(self, lat, lon, t_h):
        return self._interp("wu", lat, lon, t_h), self._interp("wv", lat, lon, t_h)

    def current(self, lat, lon, t_h):
        return self._interp("cu", lat, lon, t_h), self._interp("cv", lat, lon, t_h)

    def wind_speed_dir(self, lat, lon, t_h):
        u, v = self.wind(lat, lon, t_h)
        spd = np.hypot(u, v)
        to = (np.degrees(np.arctan2(u, v)) + 360.0) % 360.0
        return spd, (to + 180.0) % 360.0

    def current_speed_dir(self, lat, lon, t_h):
        u, v = self.current(lat, lon, t_h)
        return np.hypot(u, v), (np.degrees(np.arctan2(u, v)) + 360.0) % 360.0

    def summary_at(self, lat, lon, t_h=0.0) -> dict:
        ws, wd = self.wind_speed_dir(lat, lon, t_h)
        cs, cd = self.current_speed_dir(lat, lon, t_h)
        ws_f, cs_f = float(np.ravel(ws)[0]), float(np.ravel(cs)[0])
        return {
            "wind_speed_ms": round(ws_f, 2),
            "wind_speed_kn": round(ws_f * 1.94384, 1),
            "wind_from_deg": round(float(np.ravel(wd)[0]), 1),
            "current_speed_ms": round(cs_f, 3),
            "current_speed_kn": round(cs_f * 1.94384, 2),
            "current_towards_deg": round(float(np.ravel(cd)[0]), 1),
            "sea_state_douglas": _douglas(ws_f),
            "inertial_period_h": round(float(self.inertial_period_h), 1),
        }

    def vector_grid(self, n=12, t_h=0.0) -> list[dict]:
        w, s, e, nn = self.bbox
        la = np.linspace(s, nn, n)
        lo = np.linspace(w, e, n)
        LA, LO = np.meshgrid(la, lo, indexing="ij")
        fla, flo = LA.ravel(), LO.ravel()
        wu, wv = self.wind(fla, flo, t_h)
        cu, cv = self.current(fla, flo, t_h)
        return [
            {
                "lat": round(float(fla[i]), 4), "lon": round(float(flo[i]), 4),
                "wind_u": round(float(wu[i]), 2), "wind_v": round(float(wv[i]), 2),
                "cur_u": round(float(cu[i]), 3), "cur_v": round(float(cv[i]), 3),
            }
            for i in range(fla.size)
        ]

    # -- persistence ----------------------------------------------
    def to_npz(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            bbox=np.asarray(self.bbox, float),
            lats=self.lats, lons=self.lons, times=self.times,
            wu=self.grids["wu"], wv=self.grids["wv"],
            cu=self.grids["cu"], cv=self.grids["cv"],
            meta=np.asarray(self.meta, dtype=object),
        )
        return path

    @classmethod
    def from_npz(cls, path) -> "GriddedMetOcean":
        d = np.load(Path(path), allow_pickle=True)
        meta = {}
        if "meta" in d.files:
            try:
                meta = dict(d["meta"].item())
            except Exception:  # noqa: BLE001
                meta = {}
        return cls(
            list(d["bbox"]), d["lats"], d["lons"], d["times"],
            {k: d[k] for k in _GRID_KEYS}, meta=meta,
        )


class SyntheticMetOcean(GriddedMetOcean):
    """Deterministic ERA5/HYCOM-like field. Reproduces from (bbox, window, seed)."""

    def __init__(
        self,
        bbox,
        *,
        t0_h: float = METOCEAN.WINDOW_BEFORE_H,
        t1_h: float = METOCEAN.WINDOW_AFTER_H,
        seed: int = METOCEAN.SEED,
        mean_wind_speed: float = METOCEAN.MEAN_WIND_SPEED_MS,
        mean_wind_dir_from: float = METOCEAN.MEAN_WIND_DIR_FROM_DEG,
        mean_current_speed: float = METOCEAN.MEAN_CURRENT_SPEED_MS,
        mean_current_dir_to: float = METOCEAN.MEAN_CURRENT_DIR_TO_DEG,
        nx: int = METOCEAN.GRID_NX,
        ny: int = METOCEAN.GRID_NY,
        nt: int = METOCEAN.GRID_NT,
        pad_km: float = METOCEAN.FIELD_PAD_KM,
    ):
        field_bbox = bbox_expand_km(bbox, pad_km)
        w, s, e, n = field_bbox
        lats = np.linspace(s, n, ny)
        lons = np.linspace(w, e, nx)
        times = np.linspace(t0_h, t1_h, nt)
        grids = self._build_grids(
            lats, lons, times, seed,
            mean_wind_speed, mean_wind_dir_from,
            mean_current_speed, mean_current_dir_to,
            LocalFrame(*bbox_center(bbox)),
        )
        meta = {
            "kind": "synthetic",
            "seed": int(seed),
            "mean_wind_speed_ms": float(mean_wind_speed),
            "mean_wind_dir_from_deg": float(mean_wind_dir_from),
            "mean_current_speed_ms": float(mean_current_speed),
            "mean_current_dir_to_deg": float(mean_current_dir_to),
            "window_h": [float(t0_h), float(t1_h)],
        }
        super().__init__(bbox, lats, lons, times, grids, meta=meta)

    @staticmethod
    def _build_grids(lats, lons, times, seed, mws, mwd_from, mcs, mcd_to, frame):
        rng = np.random.default_rng(int(seed))
        ny, nx, nt = len(lats), len(lons), len(times)
        t0 = float(times[0])

        # wind: synoptic mean + slow rotation + gusty anomaly
        wdir_to = np.radians((mwd_from + 180.0) % 360.0)
        base_u = mws * np.sin(wdir_to)
        base_v = mws * np.cos(wdir_to)
        rot = np.radians(14.0 * np.sin(2 * np.pi * (times - t0) / 62.0))
        wu = base_u * np.cos(rot) - base_v * np.sin(rot)
        wv = base_u * np.sin(rot) + base_v * np.cos(rot)
        gust = 0.14 * mws
        grids = {
            "wu": (wu[:, None, None]
                   + _smooth_random_field(rng, ny, nx, nt, gust, 4.5, 3.5)),
            "wv": (wv[:, None, None]
                   + _smooth_random_field(rng, ny, nx, nt, gust, 4.5, 3.5)),
        }

        # current: mean + mesoscale eddies + inertial oscillation
        cdir = np.radians(mcd_to)
        cu0, cv0 = mcs * np.sin(cdir), mcs * np.cos(cdir)
        f_inertial = 2 * 7.292e-5 * np.sin(np.radians(abs(frame.lat0) + 1e-6))
        t_inertial_h = max(2 * np.pi / max(f_inertial, 1e-9) / 3600.0, 12.0)
        osc = 0.10 * mcs
        phase = 2 * np.pi * times / t_inertial_h
        eddy = 0.26 * mcs
        grids["cu"] = ((cu0 + osc * np.cos(phase))[:, None, None]
                       + _smooth_random_field(rng, ny, nx, nt, eddy, 6.0, 7.0))
        grids["cv"] = ((cv0 + osc * np.sin(phase))[:, None, None]
                       + _smooth_random_field(rng, ny, nx, nt, eddy, 6.0, 7.0))
        return grids
