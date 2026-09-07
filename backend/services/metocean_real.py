"""
Real met-ocean fetch for :class:`RealMetOceanProvider` (Epic 3.2).

    ERA5 10 m wind  - Copernicus Climate Data Store, via ``cdsapi``
                      (needs the ``cdsapi`` package + a ``~/.cdsapirc`` key).
    HYCOM current   - GOFS 3.1 surface currents, via an OPeNDAP dataset read
                      with ``xarray`` (needs ``xarray`` + ``netCDF4``/``pydap``).

Everything is optional. ``fetch_era5_hycom`` returns ``None`` on *any* problem
(missing package, missing key, network failure, empty subset); the caller then
falls back to the deterministic Demo field. Nothing here runs at import time.
"""
from __future__ import annotations

import datetime as _dt
import os
import tempfile
from pathlib import Path

import numpy as np

from backend.core.logging import get_logger

log = get_logger("backend.services.metocean_real")

# GOFS 3.1 global analysis - surface currents. A THREDDS OPeNDAP endpoint.
HYCOM_OPENDAP = os.getenv(
    "HYCOM_OPENDAP_URL",
    "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0",
)
_CDSAPIRC = Path(os.getenv("CDSAPI_RC", str(Path.home() / ".cdsapirc")))


# --------------------------------------------------------------------------- #
# Capability probe (fast, no network)
# --------------------------------------------------------------------------- #
def real_metocean_status() -> dict:
    """What a real fetch would need, and whether it is present on this host."""
    def _importable(mod: str) -> bool:
        import importlib.util
        return importlib.util.find_spec(mod) is not None

    cdsapi_ok = _importable("cdsapi")
    xarray_ok = _importable("xarray")
    reader_ok = _importable("netCDF4") or _importable("pydap")
    key_ok = _CDSAPIRC.is_file()
    deps_ok = cdsapi_ok and xarray_ok and reader_ok and key_ok
    return {
        "cdsapi_installed": cdsapi_ok,
        "xarray_installed": xarray_ok,
        "netcdf_reader_installed": reader_ok,
        "cdsapirc_present": key_ok,
        "ready": deps_ok,
    }


def deps_present() -> bool:
    return real_metocean_status()["ready"]


# --------------------------------------------------------------------------- #
# Fetch
# --------------------------------------------------------------------------- #
def _window_datetimes(t0_h: float, t1_h: float, acquisition) -> tuple[_dt.datetime, _dt.datetime]:
    if isinstance(acquisition, str):
        try:
            base = _dt.datetime.fromisoformat(acquisition.replace("Z", "+00:00"))
        except ValueError:
            base = _dt.datetime.now(_dt.timezone.utc)
    elif isinstance(acquisition, _dt.datetime):
        base = acquisition
    else:
        base = _dt.datetime.now(_dt.timezone.utc)
    if base.tzinfo is None:
        base = base.replace(tzinfo=_dt.timezone.utc)
    return base + _dt.timedelta(hours=float(t0_h)), base + _dt.timedelta(hours=float(t1_h))


def _fetch_era5_wind(bbox, start: _dt.datetime, end: _dt.datetime):
    """(lats, lons, times_h, u10(T,Y,X), v10(T,Y,X)) from ERA5, or raise."""
    import cdsapi  # noqa: F401
    import xarray as xr

    w, s, e, n = (float(v) for v in bbox)
    days = sorted({(start + _dt.timedelta(days=k)).date()
                   for k in range((end.date() - start.date()).days + 1)})
    req = {
        "product_type": "reanalysis",
        "variable": ["10m_u_component_of_wind", "10m_v_component_of_wind"],
        "year": sorted({d.strftime("%Y") for d in days}),
        "month": sorted({d.strftime("%m") for d in days}),
        "day": sorted({d.strftime("%d") for d in days}),
        "time": [f"{h:02d}:00" for h in range(0, 24, 3)],
        "area": [n, w, s, e],          # N, W, S, E
        "format": "netcdf",
    }
    tmp = Path(tempfile.gettempdir()) / f"era5_{start:%Y%m%d}_{os.getpid()}.nc"
    cdsapi.Client(quiet=True, progress=False).retrieve(
        "reanalysis-era5-single-levels", req, str(tmp)
    )
    with xr.open_dataset(tmp) as ds:
        ds = ds.sortby("latitude")
        t = ds["valid_time" if "valid_time" in ds else "time"].values
        t0 = np.datetime64(start.replace(tzinfo=None))
        times_h = (t - t0) / np.timedelta64(1, "h")
        out = (ds["latitude"].values.astype(float), ds["longitude"].values.astype(float),
               times_h.astype(float), ds["u10"].values.astype(float), ds["v10"].values.astype(float))
    tmp.unlink(missing_ok=True)
    return out


def _fetch_hycom_current(bbox, start: _dt.datetime, end: _dt.datetime, lats, lons, times_h):
    """Surface (cu, cv) regridded onto the ERA5 (times_h, lats, lons) grid, or raise."""
    import xarray as xr

    w, s, e, n = (float(v) for v in bbox)
    with xr.open_dataset(HYCOM_OPENDAP, decode_times=True) as ds:
        depth0 = 0
        sub = ds[["water_u", "water_v"]].sel(
            lat=slice(s - 0.5, n + 0.5), lon=slice(w % 360 - 0.5, e % 360 + 0.5),
            time=slice(np.datetime64(start.replace(tzinfo=None)),
                       np.datetime64(end.replace(tzinfo=None))),
        ).isel(depth=depth0)
        base = np.datetime64(start.replace(tzinfo=None))
        target_t = base + (times_h * np.timedelta64(3600, "s")).astype("timedelta64[s]")
        sub = sub.interp(
            time=target_t,
            lat=xr.DataArray(lats, dims="y"),
            lon=xr.DataArray(np.where(lons < 0, lons + 360, lons), dims="x"),
            kwargs={"fill_value": 0.0},
        )
        return (np.nan_to_num(sub["water_u"].values.astype(float)),
                np.nan_to_num(sub["water_v"].values.astype(float)))


def fetch_era5_hycom(bbox, t0_h: float, t1_h: float, **field_kwargs) -> dict | None:
    """Real ERA5 wind + HYCOM current for ``bbox`` over the window. ``None`` on failure."""
    status = real_metocean_status()
    if not status["ready"]:
        log.warning("real met-ocean not available: %s", status)
        return None
    try:
        start, end = _window_datetimes(t0_h, t1_h, field_kwargs.get("acquisition"))
        lats, lons, times_h, u10, v10 = _fetch_era5_wind(bbox, start, end)
        cu, cv = _fetch_hycom_current(bbox, start, end, lats, lons, times_h)
        log.info("real met-ocean: ERA5 %s + HYCOM for %s..%s (%d steps)",
                 u10.shape, start, end, len(times_h))
        return {
            "lats": lats, "lons": lons, "times": times_h,
            "wu": u10, "wv": v10, "cu": cu, "cv": cv,
            "meta": {"wind": "ERA5", "current": "HYCOM GOFS 3.1",
                     "window_utc": [start.isoformat(), end.isoformat()]},
        }
    except Exception as exc:  # noqa: BLE001 - any failure -> demo fallback
        log.warning("real met-ocean fetch failed (%s); falling back to the demo field", exc)
        return None
