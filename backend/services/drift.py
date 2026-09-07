"""
Drift service - met-ocean providers + hindcast / forecast orchestration.

The ML physics lives in ``backend.ml.drift`` (RK4 particle engine, ported from
OceanTrace). This module adds:

* the ``MetOceanProvider`` interface and three implementations - Real / Demo /
  Cached - with a clean fallback to the demo (simulated) field whenever real
  reanalysis is unavailable or the system is offline. Simulated fields are
  always tagged ``"Demo / simulated environmental field"``.
* ``run_hindcast`` / ``run_forecast`` / ``analyze_drift`` - JSON-serialisable
  wrappers the investigation orchestration will call later.
"""
from __future__ import annotations

import abc
import hashlib
import json
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from backend.core.config import DATA_DIR, settings
from backend.core.logging import get_logger
from backend.ml.drift.config import DRIFT, METOCEAN, DriftParams
from backend.ml.drift.forecast import _hull_ring, forecast as _forecast
from backend.ml.drift.geo import bbox_center, bbox_expand_km
from backend.ml.drift.hindcast import hindcast as _hindcast
from backend.ml.drift.metocean import GriddedMetOcean, SyntheticMetOcean
from backend.ml.drift.particles import LandMask

log = get_logger("backend.services.drift")

DEMO_FIELD_LABEL = "Demo / simulated environmental field"
REAL_FIELD_LABEL = "ERA5 (ECMWF) wind + HYCOM (GOFS 3.1) current - real reanalysis"
CACHE_DIR = Path(DATA_DIR) / "metocean_cache"


class MetOceanUnavailable(RuntimeError):
    """Raised by a provider that cannot supply a field for the request."""


# --------------------------------------------------------------------------- #
# Resolved field
# --------------------------------------------------------------------------- #
@dataclass
class MetOceanField:
    """A provider-agnostic wind + current field, ready for the drift engine."""

    grid: GriddedMetOcean
    source: str                 # demo | cached | era5_hycom
    simulated: bool
    label: str
    window_h: tuple[float, float]

    # delegate the physics interface the RK4 engine expects
    def wind(self, lat, lon, t_h):
        return self.grid.wind(lat, lon, t_h)

    def current(self, lat, lon, t_h):
        return self.grid.current(lat, lon, t_h)

    def summary_at(self, lat, lon, t_h=0.0) -> dict:
        return self.grid.summary_at(lat, lon, t_h)

    def vector_grid(self, n=12, t_h=0.0) -> list[dict]:
        return self.grid.vector_grid(n, t_h)

    @property
    def bbox(self) -> tuple:
        return self.grid.bbox

    def provenance(self) -> dict:
        lat_c, lon_c = bbox_center(self.grid.bbox)
        return {
            "source": self.source,
            "simulated": self.simulated,
            "label": self.label,
            "bbox": [round(float(v), 5) for v in self.grid.bbox],
            "window_h": [float(self.window_h[0]), float(self.window_h[1])],
            "grid_shape": [self.grid.ny, self.grid.nx, self.grid.nt],
            "mean_conditions": self.grid.summary_at(lat_c, lon_c, 0.0),
        }


# --------------------------------------------------------------------------- #
# Provider interface + implementations
# --------------------------------------------------------------------------- #
class MetOceanProvider(abc.ABC):
    name: str = "abstract"

    @abc.abstractmethod
    def available(self) -> bool:
        """Whether this provider can serve a field right now."""

    @abc.abstractmethod
    def get_field(self, bbox, *, t0_h: float, t1_h: float, **field_kwargs) -> MetOceanField:
        """Return a resolved field, or raise :class:`MetOceanUnavailable`."""


class DemoMetOceanProvider(MetOceanProvider):
    """Deterministic synthesised ERA5/HYCOM-like field. Always available, offline."""

    name = "demo"

    def available(self) -> bool:
        return True

    def get_field(self, bbox, *, t0_h: float, t1_h: float, **field_kwargs) -> MetOceanField:
        grid = SyntheticMetOcean(tuple(bbox), t0_h=t0_h, t1_h=t1_h, **field_kwargs)
        return MetOceanField(
            grid=grid, source="demo", simulated=True,
            label=DEMO_FIELD_LABEL, window_h=(t0_h, t1_h),
        )


class RealMetOceanProvider(MetOceanProvider):
    """Real ERA5 wind + HYCOM current.

    A ``fetch_fn(bbox, t0_h, t1_h, **kw) -> dict`` supplies the real grids
    (keys ``lats, lons, times, wu, wv, cu, cv``). Without one - the default,
    and always when the app is offline - this provider reports itself
    unavailable so callers fall back to the demo field. Wiring cdsapi (ERA5)
    and a HYCOM OPeNDAP reader into ``fetch_fn`` is the extension point.
    """

    name = "era5_hycom"

    def __init__(self, fetch_fn: Callable[..., dict] | None = None, *, offline: bool | None = None):
        self._fetch = fetch_fn
        self._offline = settings.OFFLINE_MODE if offline is None else bool(offline)

    def available(self) -> bool:
        return (not self._offline) and (self._fetch is not None)

    def get_field(self, bbox, *, t0_h: float, t1_h: float, **field_kwargs) -> MetOceanField:
        if not self.available():
            raise MetOceanUnavailable(
                "real ERA5/HYCOM backend not configured "
                f"(offline={self._offline}, fetch_fn={'set' if self._fetch else 'none'})"
            )
        raw = self._fetch(tuple(bbox), t0_h, t1_h, **field_kwargs)  # pragma: no cover
        grid = GriddedMetOcean(
            bbox, raw["lats"], raw["lons"], raw["times"],
            {k: raw[k] for k in ("wu", "wv", "cu", "cv")},
            meta={"kind": "real", **raw.get("meta", {})},
        )
        return MetOceanField(
            grid=grid, source=self.name, simulated=False,
            label=REAL_FIELD_LABEL, window_h=(t0_h, t1_h),
        )


class CachedMetOceanProvider(MetOceanProvider):
    """Local disk cache in front of an upstream provider.

    Cache hit -> load the stored grid. Miss -> ask ``upstream`` if it is
    available, otherwise generate a demo field; either way, persist it. Always
    available (it can always fall back to the demo generator).
    """

    name = "cached"

    def __init__(self, cache_dir: Path | str = CACHE_DIR, upstream: MetOceanProvider | None = None):
        self.cache_dir = Path(cache_dir)
        self.upstream = upstream or RealMetOceanProvider()
        self._demo = DemoMetOceanProvider()

    def available(self) -> bool:
        return True

    def _key(self, bbox, t0_h, t1_h, field_kwargs) -> str:
        payload = {
            "bbox": [round(float(v), 3) for v in bbox],
            "t0": round(float(t0_h), 1), "t1": round(float(t1_h), 1),
            "kw": {k: field_kwargs[k] for k in sorted(field_kwargs)},
        }
        digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
        return f"metocean_{digest}"

    def get_field(self, bbox, *, t0_h: float, t1_h: float, **field_kwargs) -> MetOceanField:
        key = self._key(bbox, t0_h, t1_h, field_kwargs)
        npz = self.cache_dir / f"{key}.npz"
        sidecar = self.cache_dir / f"{key}.json"

        if npz.exists() and sidecar.exists():
            grid = GriddedMetOcean.from_npz(npz)
            info = json.loads(sidecar.read_text())
            log.info("met-ocean cache hit: %s (%s)", key, info.get("source"))
            return MetOceanField(
                grid=grid, source="cached", simulated=bool(info.get("simulated", True)),
                label=info.get("label", DEMO_FIELD_LABEL),
                window_h=(t0_h, t1_h),
            )

        if self.upstream.available():
            field = self.upstream.get_field(bbox, t0_h=t0_h, t1_h=t1_h, **field_kwargs)
        else:
            field = self._demo.get_field(bbox, t0_h=t0_h, t1_h=t1_h, **field_kwargs)

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        field.grid.to_npz(npz)
        sidecar.write_text(json.dumps({
            "source": field.source, "simulated": field.simulated, "label": field.label,
        }))
        log.info("met-ocean cache store: %s (%s)", key, field.source)
        # served through the cache from here on
        return MetOceanField(
            grid=field.grid, source="cached", simulated=field.simulated,
            label=field.label, window_h=(t0_h, t1_h),
        )


# --------------------------------------------------------------------------- #
# Factory + resolver
# --------------------------------------------------------------------------- #
def get_metocean_provider(
    *,
    prefer_real: bool = True,
    cache: bool = True,
    fetch_fn: Callable[..., dict] | None = None,
    offline: bool | None = None,
) -> MetOceanProvider:
    """Assemble the provider stack for the current environment."""
    real = RealMetOceanProvider(fetch_fn=fetch_fn, offline=offline)
    upstream = real if prefer_real else DemoMetOceanProvider()
    if cache:
        return CachedMetOceanProvider(upstream=upstream)
    return upstream if upstream.available() else DemoMetOceanProvider()


def resolve_metocean_field(
    bbox,
    *,
    t0_h: float | None = None,
    t1_h: float | None = None,
    cache: bool = True,
    prefer_real: bool = True,
    fetch_fn: Callable[..., dict] | None = None,
    offline: bool | None = None,
    **field_kwargs,
) -> MetOceanField:
    """Always returns a usable field; simulated ones are tagged as such."""
    t0_h = METOCEAN.WINDOW_BEFORE_H if t0_h is None else float(t0_h)
    t1_h = METOCEAN.WINDOW_AFTER_H if t1_h is None else float(t1_h)
    provider = get_metocean_provider(
        prefer_real=prefer_real, cache=cache, fetch_fn=fetch_fn, offline=offline
    )
    try:
        return provider.get_field(bbox, t0_h=t0_h, t1_h=t1_h, **field_kwargs)
    except MetOceanUnavailable as exc:
        log.warning("met-ocean provider unavailable (%s); using demo field", exc)
        return DemoMetOceanProvider().get_field(bbox, t0_h=t0_h, t1_h=t1_h, **field_kwargs)


# --------------------------------------------------------------------------- #
# Observed-slick helpers
# --------------------------------------------------------------------------- #
def observed_from_sar(sar_result: dict) -> dict:
    """Adapt a ``services.satellite.analyze_scene`` result to a drift input."""
    dets = sar_result.get("detections") or []
    oil = [d for d in dets if d.get("classification") == "Oil-like anomaly"]
    chosen = (oil or dets or [None])[0]
    if chosen is None:
        raise ValueError("SAR result has no detections to hindcast/forecast")

    char = chosen.get("characterization", {})
    centroid = char.get("centroid")
    geom = (chosen.get("geojson") or {}).get("geometry") or {}
    polygon = None
    if geom.get("type") == "Polygon" and geom.get("coordinates"):
        polygon = {"type": "Polygon", "coordinates": [geom["coordinates"][0]]}
    elif geom.get("type") == "MultiPolygon" and geom.get("coordinates"):
        polygon = {"type": "Polygon", "coordinates": [geom["coordinates"][0][0]]}

    return {
        "centroid": centroid,
        "polygon": polygon,
        "area_km2": char.get("area_km2"),
        "width_m": (char.get("width_km") or 0.0) * 1000.0 or None,
        "acquisition": sar_result.get("acquisition"),
        "classification": chosen.get("classification"),
    }


def land_from_geojson(polygons: Any, *, buffer_km: float = 0.0) -> LandMask:
    """Build a :class:`LandMask` from GeoJSON polygons / features / geometries."""
    geoms: list = []
    if isinstance(polygons, dict) and polygons.get("type") == "FeatureCollection":
        polygons = [f.get("geometry") for f in polygons.get("features", [])]
    for item in polygons or []:
        if isinstance(item, dict) and item.get("type") == "Feature":
            geoms.append(item.get("geometry"))
        else:
            geoms.append(item)
    return LandMask(polygons=[g for g in geoms if g], buffer_km=buffer_km)


def _bbox_for_observed(observed: dict, pad_deg: float = 0.75) -> tuple:
    if observed.get("bbox"):
        return tuple(observed["bbox"])
    if observed.get("polygon"):
        coords = np.asarray(observed["polygon"]["coordinates"][0], float)
        lons, lats = coords[:, 0], coords[:, 1]
        return bbox_expand_km(
            [lons.min(), lats.min(), lons.max(), lats.max()], pad_deg * 111.0
        )
    lat, lon = observed["centroid"]
    return (lon - pad_deg, lat - pad_deg, lon + pad_deg, lat + pad_deg)


# --------------------------------------------------------------------------- #
# Serialisation of a raw ml/drift result
# --------------------------------------------------------------------------- #
_HINDCAST_ARRAY_KEYS = ("origin_lats", "origin_lons", "weights",
                        "track_times_h", "track_lats", "track_lons")
_FORECAST_ARRAY_KEYS = ("track_times_h", "track_lats", "track_lons")


def _clean_hindcast(raw: dict, include_arrays: bool) -> dict:
    grid = raw["grid"]
    origin_hull = _hull_ring(raw["origin_lats"], raw["origin_lons"])
    out = {
        "best_estimate": raw["best_estimate"],
        "centroid": raw["centroid"],
        "uncertainty_radius_km": raw["uncertainty_radius_km"],
        "release_window_h": [round(float(v), 3) for v in raw["release_window_h"]],
        "age_prior_window_h": [round(float(v), 3) for v in raw["age_prior_window_h"]],
        "age_point_estimate_h": raw["age_point_estimate_h"],
        "age_source": raw["age_source"],
        "seed_kind": raw["seed_kind"],
        "n_particles": raw["n_particles"],
        "max_backtrack_h": raw["max_backtrack_h"],
        "origin_probability_field": {
            "bbox": [round(float(v), 5) for v in grid["bbox"]],
            "shape": list(grid["prob"].shape),
            "lat_range": [float(grid["lats"][0]), float(grid["lats"][-1])],
            "lon_range": [float(grid["lons"][0]), float(grid["lons"][-1])],
            "heatmap": raw["heatmap"],
        },
        "release_polygon": {"type": "Polygon", "coordinates": [origin_hull]},
        "confidence_ellipse": raw["confidence_ellipse"],
        "params": raw["params"],
    }
    if include_arrays:
        out["arrays"] = {k: raw[k] for k in _HINDCAST_ARRAY_KEYS}
        out["arrays"]["origin_probability_grid"] = grid["prob"]
    return out


def _clean_forecast(raw: dict, include_arrays: bool) -> dict:
    out = {
        "start_centroid": raw["start_centroid"],
        "seed_kind": raw["seed_kind"],
        "n_particles": raw["n_particles"],
        "horizons_h": raw["horizons_h"],
        "horizons": raw["horizons"],
        "coastal_impact": raw["coastal_impact"],
        "params": raw["params"],
    }
    if include_arrays:
        out["arrays"] = {k: raw[k] for k in _FORECAST_ARRAY_KEYS}
    return out


# --------------------------------------------------------------------------- #
# Public service entry points
# --------------------------------------------------------------------------- #
def _resolve_field(observed, bbox, field, field_kwargs):
    if field is not None:
        return field
    bb = bbox or _bbox_for_observed(observed)
    return resolve_metocean_field(bb, **(field_kwargs or {}))


def run_hindcast(
    observed: dict,
    *,
    bbox=None,
    field: MetOceanField | None = None,
    params: DriftParams | None = None,
    max_backtrack_h: float | None = None,
    n_particles: int | None = None,
    seed: int = 7,
    land: LandMask | None = None,
    age_hours: float | None = None,
    age_window_h=None,
    include_arrays: bool = False,
    field_kwargs: dict | None = None,
) -> dict:
    """Reverse-drift the observed slick to a release-origin field + time window."""
    mo = _resolve_field(observed, bbox, field, field_kwargs)
    raw = _hindcast(
        mo, observed, params=params, max_backtrack_h=max_backtrack_h,
        n_particles=n_particles, seed=seed, land=land, age_hours=age_hours,
        age_window_h=age_window_h,
    )
    out = _clean_hindcast(raw, include_arrays)
    out["provenance"] = mo.provenance()
    out["environmental_field"] = mo.label
    log.info(
        "hindcast: origin ~%s +-%.1f km, release window %s h [%s]",
        out["best_estimate"], out["uncertainty_radius_km"],
        out["release_window_h"], mo.label,
    )
    return out


def run_forecast(
    observed: dict,
    *,
    bbox=None,
    field: MetOceanField | None = None,
    params: DriftParams | None = None,
    horizons_h=None,
    n_particles: int | None = None,
    seed: int = 11,
    land: LandMask | None = None,
    include_arrays: bool = False,
    field_kwargs: dict | None = None,
) -> dict:
    """Forward-drift the observed slick to 6/12/24/48 h snapshots + coastal contact."""
    mo = _resolve_field(observed, bbox, field, field_kwargs)
    raw = _forecast(
        mo, observed, params=params, horizons_h=horizons_h,
        n_particles=n_particles, seed=seed, land=land,
    )
    out = _clean_forecast(raw, include_arrays)
    out["provenance"] = mo.provenance()
    out["environmental_field"] = mo.label
    return out


def analyze_drift(
    observed: dict,
    *,
    bbox=None,
    params: DriftParams | None = None,
    land: LandMask | None = None,
    max_backtrack_h: float | None = None,
    horizons_h=None,
    n_particles: int | None = None,
    include_arrays: bool = False,
    field_kwargs: dict | None = None,
) -> dict:
    """Run hindcast + forecast against one shared met-ocean field."""
    bb = bbox or _bbox_for_observed(observed)
    mo = resolve_metocean_field(bb, **(field_kwargs or {}))
    hind = run_hindcast(
        observed, field=mo, params=params, land=land,
        max_backtrack_h=max_backtrack_h, n_particles=n_particles,
        include_arrays=include_arrays,
    )
    fore = run_forecast(
        observed, field=mo, params=params, land=land,
        horizons_h=horizons_h, n_particles=n_particles,
        include_arrays=include_arrays,
    )
    return {
        "observed": {k: observed.get(k) for k in
                     ("centroid", "area_km2", "width_m", "acquisition", "classification")},
        "environmental_field": mo.label,
        "provenance": mo.provenance(),
        "hindcast": hind,
        "forecast": fore,
    }


# --------------------------------------------------------------------------- #
# Demo fixture
# --------------------------------------------------------------------------- #
def demo_observed() -> dict:
    """A canned observed slick (matches the Phase 3 SAR demo incident)."""
    lat_c, lon_c = 15.63172, 73.25674
    # a short elongated ring around the centroid, ~26 km x 3 km, bearing ~125 deg
    half_len_deg = 0.117          # ~13 km each way
    half_wid_deg = 0.0135         # ~1.5 km each way
    br = np.radians(125.0)
    dl = np.array([-1, -1, 1, 1, -1]) * half_len_deg
    dw = np.array([-1, 1, 1, -1, -1]) * half_wid_deg
    lon = lon_c + dl * np.sin(br) + dw * np.cos(br)
    lat = lat_c + dl * np.cos(br) - dw * np.sin(br)
    return {
        "centroid": [lat_c, lon_c],
        "polygon": {"type": "Polygon", "coordinates": [
            [[float(lo), float(la)] for lo, la in zip(lon, lat)]
        ]},
        "area_km2": 74.9,
        "width_m": 2840.0,
        "acquisition": "2026-03-06T05:42:00+00:00",
        "classification": "Oil-like anomaly",
    }
