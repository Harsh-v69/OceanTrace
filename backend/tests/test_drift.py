"""
Phase 4 - Lagrangian drift, hindcast, forecast, and MetOceanProvider fallback.

Covers RK4 order-of-accuracy, backward-tracking convergence, forward-projection
bounds and coastal contact, the provider stack's clean fallback to the demo
(simulated) field, and the centralised drift configuration.
"""
from __future__ import annotations

import numpy as np
import pytest

from backend.ml.drift import DriftParams, advect, rk4_step
from backend.ml.drift.config import DRIFT, METOCEAN
from backend.ml.drift.geo import LocalFrame, haversine_m
from backend.ml.drift.metocean import GriddedMetOcean, SyntheticMetOcean
from backend.ml.drift.particles import LandMask, drift_velocity
from backend.services import drift as D
from backend.services import satellite


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def uniform_field(bbox=(72.0, 15.0, 74.5, 17.0), *, cu=0.0, cv=0.0, wu=0.0, wv=0.0,
                  t0_h=-60.0, t1_h=60.0) -> GriddedMetOcean:
    """A spatially/temporally constant wind+current field for analytic checks."""
    lats = np.linspace(bbox[1], bbox[3], 6)
    lons = np.linspace(bbox[0], bbox[2], 6)
    times = np.linspace(t0_h, t1_h, 4)
    shp = (len(times), len(lats), len(lons))
    grids = {
        "wu": np.full(shp, wu, np.float32), "wv": np.full(shp, wv, np.float32),
        "cu": np.full(shp, cu, np.float32), "cv": np.full(shp, cv, np.float32),
    }
    return GriddedMetOcean(bbox, lats, lons, times, grids, meta={"kind": "uniform"})


def fake_fetch(bbox, t0_h, t1_h, **kw):
    """Stand-in for a real ERA5/HYCOM fetch: constant SW wind + ENE current."""
    lats = np.linspace(bbox[1], bbox[3], 5)
    lons = np.linspace(bbox[0], bbox[2], 5)
    times = np.linspace(t0_h, t1_h, 3)
    shp = (len(times), len(lats), len(lons))
    return {
        "lats": lats, "lons": lons, "times": times,
        "wu": np.full(shp, 4.0), "wv": np.full(shp, 4.0),
        "cu": np.full(shp, 0.2), "cv": np.full(shp, 0.05),
        "meta": {"provider": "fake"},
    }


@pytest.fixture(scope="module")
def sar_observed() -> dict:
    return D.observed_from_sar(satellite.analyze_demo())


# --------------------------------------------------------------------------- #
# RK4 order of accuracy
# --------------------------------------------------------------------------- #
def test_rk4_exact_for_constant_velocity():
    vel = lambda x, y, t: (0.5, -0.25)          # noqa: E731
    x, y = rk4_step(vel, 0.0, 0.0, 0.0, 3600.0)
    assert x == pytest.approx(0.5 * 3600.0, abs=1e-9)
    assert y == pytest.approx(-0.25 * 3600.0, abs=1e-9)


def test_rk4_exact_for_polynomial_in_time():
    # dx/dt = a*t  ->  x(T) = a*T^2/2 ; RK4 is exact for cubics
    a = 3.0e-4
    T = 600.0
    x, _ = rk4_step(lambda x, y, t: (a * t, 0.0), 0.0, 0.0, 0.0, T)
    assert x == pytest.approx(a * T * T / 2.0, rel=1e-9)


def test_rk4_solid_body_rotation_conserves_radius():
    omega = 2 * np.pi / 4000.0                  # one revolution per 4000 s
    vel = lambda x, y, t: (-omega * y, omega * x)   # noqa: E731
    x, y = 1000.0, 0.0
    dt = 4000.0 / 2000.0
    for k in range(2000):
        x, y = rk4_step(vel, x, y, k * dt, dt)
    assert np.hypot(x, y) == pytest.approx(1000.0, rel=2e-4)
    assert x == pytest.approx(1000.0, abs=2.0)   # back to the start
    assert y == pytest.approx(0.0, abs=2.0)


def test_rk4_is_fourth_order():
    omega = 2 * np.pi / 4000.0
    vel = lambda x, y, t: (-omega * y, omega * x)   # noqa: E731

    def integrate(n):
        x, y = 1000.0, 0.0
        dt = 4000.0 / n
        for k in range(n):
            x, y = rk4_step(vel, x, y, k * dt, dt)
        return np.hypot(x - 1000.0, y - 0.0)

    e_coarse = integrate(200)
    e_fine = integrate(400)
    # halving dt should cut a 4th-order method's error ~16x
    assert e_fine > 0
    assert e_coarse / e_fine > 8.0


# --------------------------------------------------------------------------- #
# Advection physics against analytic displacement
# --------------------------------------------------------------------------- #
def test_advect_constant_current_matches_displacement():
    field = uniform_field(cu=0.3, cv=0.0)
    p = DriftParams(diffusivity=0.0)
    lat0, lon0 = 16.0, 73.0
    hours = 10.0
    _, la, lo = advect(field, [lat0], [lon0], 0.0, hours, params=p, diffusion=False)
    frame = LocalFrame(lat0, lon0)
    dx, dy = frame.to_xy(la[-1][0], lo[-1][0])
    expected = p.current_factor * 0.3 * hours * 3600.0
    assert dx == pytest.approx(expected, rel=0.02)
    assert abs(dy) < 0.02 * expected


def test_wind_deflection_is_to_the_right_in_northern_hemisphere():
    field = uniform_field(wu=10.0, wv=0.0)          # wind blowing due east
    p = DriftParams()
    u, v = drift_velocity(field, np.array([16.0]), np.array([73.0]), 0.0, p)
    assert float(np.ravel(u)[0]) > 0.0              # still mostly downwind
    assert float(np.ravel(v)[0]) < 0.0             # deflected right -> southward


# --------------------------------------------------------------------------- #
# Backward tracking convergence
# --------------------------------------------------------------------------- #
def test_forward_then_backward_returns_to_start():
    """Reversibility: forward 18 h then backward 18 h lands back on the start.

    The residual (~60 m over a ~54 km round trip, dt-independent) is the
    tangent-plane approximation limit - the forward and backward passes anchor
    their local metric frames ~50 km apart - not integrator error.
    """
    field = SyntheticMetOcean((72.0, 15.0, 74.5, 17.0), seed=1)
    p = DriftParams()
    lat0 = np.array([16.0, 16.05, 15.95])
    lon0 = np.array([73.0, 73.1, 72.9])
    _, laf, lof = advect(field, lat0, lon0, 0.0, 18.0, params=p,
                         seed=3, diffusion=False)
    _, lab, lob = advect(field, laf[-1], lof[-1], 18.0, 0.0, params=p,
                         seed=3, diffusion=False)
    back = haversine_m(lab[-1], lob[-1], lat0, lon0)
    assert float(np.max(back)) < 150.0             # < 0.3% of the ~54 km path

    # and it is dt-independent (a discretisation error would shrink with dt)
    p2 = DriftParams(timestep_s=150)
    _, l2f, o2f = advect(field, lat0, lon0, 0.0, 18.0, params=p2, seed=3, diffusion=False)
    _, l2b, o2b = advect(field, l2f[-1], o2f[-1], 18.0, 0.0, params=p2, seed=3, diffusion=False)
    assert float(np.max(haversine_m(l2b[-1], o2b[-1], lat0, lon0))) == pytest.approx(
        float(np.max(back)), abs=5.0
    )


def test_hindcast_origin_lies_upstream_of_the_slick():
    field = uniform_field(cu=0.25, cv=0.05, wu=3.0, wv=3.0)   # net drift toward NE
    observed = {"centroid": [16.0, 73.0], "area_km2": 20.0}
    out = D.run_hindcast(observed, field=_wrap(field), age_hours=12.0,
                         n_particles=300, seed=7)
    best = out["best_estimate"]
    # origin must sit south-west of the observed centroid (upstream of a NE drift)
    assert best[0] < 16.0
    assert best[1] < 73.0
    # and displacement is bounded by a plausible drift speed over 12 h
    d_km = haversine_m(best[0], best[1], 16.0, 73.0) / 1000.0
    assert 1.0 < d_km < (1.5 * 12 * 3.6)            # < |1.5 m/s| * 12 h


def test_hindcast_release_window_brackets_the_true_age():
    field = uniform_field(cu=0.2)
    out = D.run_hindcast({"centroid": [16.0, 73.0], "area_km2": 15.0},
                         field=_wrap(field), age_hours=10.0, n_particles=150)
    lo, hi = out["release_window_h"]
    assert lo < hi <= 0.0
    assert hi == pytest.approx(-DRIFT.RELEASE_WINDOW_NEAR_H)
    a_lo, a_hi = out["age_prior_window_h"]
    assert a_lo <= -10.0 <= a_hi                     # the true age is inside the prior


def test_hindcast_is_deterministic():
    field = _wrap(uniform_field(cu=0.2, wv=4.0))
    obs = {"centroid": [16.0, 73.0], "area_km2": 12.0}
    a = D.run_hindcast(obs, field=field, age_hours=8.0, n_particles=120, seed=7)
    b = D.run_hindcast(obs, field=field, age_hours=8.0, n_particles=120, seed=7)
    assert a["best_estimate"] == b["best_estimate"]
    assert a["centroid"] == b["centroid"]
    assert a["uncertainty_radius_km"] == b["uncertainty_radius_km"]


# --------------------------------------------------------------------------- #
# Forward projection bounds + coastal contact
# --------------------------------------------------------------------------- #
def _wrap(grid: GriddedMetOcean) -> D.MetOceanField:
    return D.MetOceanField(grid=grid, source="demo", simulated=True,
                           label=D.DEMO_FIELD_LABEL, window_h=(-60.0, 60.0))


@pytest.fixture(scope="module")
def forecast_uniform() -> dict:
    field = _wrap(uniform_field(cu=0.25, cv=0.10, wu=3.0, wv=2.0))
    obs = {"centroid": [16.0, 73.0], "area_km2": 20.0}
    return D.run_forecast(obs, field=field, n_particles=300, seed=11)


def test_forecast_reports_all_four_default_horizons(forecast_uniform):
    ts = [s["t_h"] for s in forecast_uniform["horizons"]]
    assert ts == [6.0, 12.0, 24.0, 48.0]
    assert forecast_uniform["horizons_h"] == [6.0, 12.0, 24.0, 48.0]


def test_forecast_displacement_grows_monotonically(forecast_uniform):
    start = forecast_uniform["start_centroid"]
    dists = [
        haversine_m(start[0], start[1], s["centroid"][0], s["centroid"][1]) / 1000.0
        for s in forecast_uniform["horizons"]
    ]
    assert all(b > a for a, b in zip(dists, dists[1:]))


def test_forecast_48h_displacement_is_physically_bounded(forecast_uniform):
    start = forecast_uniform["start_centroid"]
    last = forecast_uniform["horizons"][-1]["centroid"]
    d_km = haversine_m(start[0], start[1], last[0], last[1]) / 1000.0
    assert 5.0 < d_km < 1.5 * 48 * 3.6              # 0 < d < |1.5 m/s| * 48 h


def test_forecast_spread_does_not_shrink(forecast_uniform):
    p90 = [s["p90_radius_km"] for s in forecast_uniform["horizons"]]
    assert all(b >= a - 0.5 for a, b in zip(p90, p90[1:]))
    assert p90[-1] >= p90[0]


def test_forecast_footprint_polygons_are_valid(forecast_uniform):
    for s in forecast_uniform["horizons"]:
        ring = s["footprint"]["coordinates"][0]
        assert len(ring) >= 4
        assert ring[0] == ring[-1]                  # closed


def test_forecast_without_land_reports_no_beaching(forecast_uniform):
    ci = forecast_uniform["coastal_impact"]
    assert ci["will_beach"] is False
    assert "no coastline" in ci["note"].lower()


def test_forecast_with_land_predicts_shoreline_contact():
    field = _wrap(uniform_field(cu=0.4, cv=0.15, wu=4.0, wv=2.0))   # strong NE drift
    obs = {"centroid": [16.0, 73.0], "area_km2": 20.0}
    # a land mass straddling the drift path to the north-east
    land = D.land_from_geojson([{
        "type": "Polygon",
        "coordinates": [[[73.4, 16.2], [75.0, 16.2], [75.0, 18.0], [73.4, 18.0], [73.4, 16.2]]],
    }])
    out = D.run_forecast(obs, field=field, land=land, n_particles=400, seed=11)
    ci = out["coastal_impact"]
    assert ci["will_beach"] is True
    assert 0.0 < ci["eta_hours"] <= 48.0
    assert ci["fraction_beached"] > 0.0
    assert len(ci["landfall_point"]) == 2
    # Epic 2.1: a specific first-contact coordinate + a set of stranding points
    assert len(ci["first_contact_point"]) == 2
    assert 0.0 < ci["first_contact_eta_h"] <= 48.0
    assert ci["contact_points"] and all(len(p) == 2 for p in ci["contact_points"])
    # once stranded, the footprint stops advancing across the coast
    assert out["horizons"][-1]["fraction_beached"] >= out["horizons"][0]["fraction_beached"]


def test_forecast_is_deterministic():
    field = _wrap(uniform_field(cu=0.2, wv=3.0))
    obs = {"centroid": [16.0, 73.0], "area_km2": 10.0}
    a = D.run_forecast(obs, field=field, n_particles=150, seed=11)
    b = D.run_forecast(obs, field=field, n_particles=150, seed=11)
    assert a["horizons"][-1]["centroid"] == b["horizons"][-1]["centroid"]


# --------------------------------------------------------------------------- #
# MetOceanProvider fallback behaviour
# --------------------------------------------------------------------------- #
def test_demo_provider_is_always_available_and_tagged_simulated():
    prov = D.DemoMetOceanProvider()
    assert prov.available() is True
    fld = prov.get_field((72.0, 15.0, 74.0, 17.0), t0_h=-48, t1_h=48)
    assert fld.simulated is True
    assert fld.label == "Demo / simulated environmental field"
    assert fld.provenance()["simulated"] is True
    u, v = fld.wind(16.0, 73.0, 0.0)
    assert np.isfinite(u).all() and np.isfinite(v).all()


def test_real_provider_unavailable_when_offline_or_unconfigured():
    prov = D.RealMetOceanProvider()                 # offline default, no fetch_fn
    assert prov.available() is False
    with pytest.raises(D.MetOceanUnavailable):
        prov.get_field((72.0, 15.0, 74.0, 17.0), t0_h=-48, t1_h=48)


def test_real_provider_available_with_injected_fetcher():
    prov = D.RealMetOceanProvider(fetch_fn=fake_fetch, offline=False)
    assert prov.available() is True
    fld = prov.get_field((72.0, 15.0, 74.0, 17.0), t0_h=-24, t1_h=24)
    assert fld.simulated is False
    assert "real reanalysis" in fld.label.lower()


def test_resolve_field_falls_back_to_demo_when_offline():
    fld = D.resolve_metocean_field((72.0, 15.0, 74.0, 17.0), cache=False)
    assert fld.simulated is True
    assert fld.label == "Demo / simulated environmental field"
    cu, cv = fld.current(16.0, 73.0, 0.0)
    assert np.isfinite(cu).all() and np.isfinite(cv).all()


def test_cached_provider_round_trips_through_disk(tmp_path):
    prov = D.CachedMetOceanProvider(cache_dir=tmp_path)
    args = dict(t0_h=-48.0, t1_h=48.0)
    first = prov.get_field((72.0, 15.0, 74.0, 17.0), **args)
    files = list(tmp_path.glob("metocean_*"))
    assert any(f.suffix == ".npz" for f in files)
    assert any(f.suffix == ".json" for f in files)
    assert first.source == "cached"

    second = prov.get_field((72.0, 15.0, 74.0, 17.0), **args)   # cache hit
    assert second.source == "cached"
    u1, v1 = (float(np.ravel(a)[0]) for a in first.wind(16.0, 73.0, 3.0))
    u2, v2 = (float(np.ravel(a)[0]) for a in second.wind(16.0, 73.0, 3.0))
    assert u1 == pytest.approx(u2, abs=1e-4)
    assert v1 == pytest.approx(v2, abs=1e-4)


def test_cached_provider_stores_real_field_from_upstream(tmp_path):
    up = D.RealMetOceanProvider(fetch_fn=fake_fetch, offline=False)
    prov = D.CachedMetOceanProvider(cache_dir=tmp_path, upstream=up)
    fld = prov.get_field((72.0, 15.0, 74.0, 17.0), t0_h=-24.0, t1_h=24.0)
    assert fld.simulated is False
    sidecar = next(tmp_path.glob("metocean_*.json"))
    import json
    info = json.loads(sidecar.read_text())
    assert info["simulated"] is False
    assert "real reanalysis" in info["label"].lower()


def test_provider_factory_returns_cached_stack_by_default():
    assert isinstance(D.get_metocean_provider(), D.CachedMetOceanProvider)
    bare = D.get_metocean_provider(cache=False)
    assert isinstance(bare, D.DemoMetOceanProvider)   # real unavailable offline


# --------------------------------------------------------------------------- #
# Centralised configuration
# --------------------------------------------------------------------------- #
def test_driftparams_defaults_come_from_config():
    p = DriftParams()
    assert p.wind_factor == DRIFT.WIND_DRIFT_FACTOR == 0.030
    assert p.wind_deflection_deg == DRIFT.WIND_DEFLECTION_DEG == 17.0
    assert p.current_factor == DRIFT.CURRENT_FACTOR
    assert p.stokes_factor == DRIFT.STOKES_FACTOR
    assert p.diffusivity == DRIFT.HORIZONTAL_DIFFUSIVITY
    assert p.timestep_s == DRIFT.TIMESTEP_S


def test_config_holds_the_documented_constants():
    assert DRIFT.MAX_BACKTRACK_H == 48
    assert DRIFT.MAX_FORECAST_H == 48
    assert DRIFT.FORECAST_HORIZONS_H == (6, 12, 24, 48)
    assert METOCEAN.MEAN_WIND_SPEED_MS == 7.5


def test_changing_a_drift_coefficient_changes_the_result():
    field = _wrap(uniform_field(cu=0.2, cv=0.05, wu=3.0, wv=3.0))
    obs = {"centroid": [16.0, 73.0], "area_km2": 15.0}
    base = D.run_hindcast(obs, field=field, age_hours=12.0, n_particles=200, seed=7)
    strong_wind = D.run_hindcast(obs, field=field, age_hours=12.0, n_particles=200,
                                 seed=7, params=DriftParams(wind_factor=0.06))
    assert base["best_estimate"] != strong_wind["best_estimate"]


# --------------------------------------------------------------------------- #
# End-to-end wiring: SAR result -> drift
# --------------------------------------------------------------------------- #
def test_observed_from_sar_adapts_a_detection(sar_observed):
    assert sar_observed["classification"] == "Oil-like anomaly"
    lat, lon = sar_observed["centroid"]
    assert 14.0 < lat < 17.0 and 72.0 < lon < 75.0
    assert sar_observed["polygon"]["type"] == "Polygon"
    assert sar_observed["width_m"] and sar_observed["width_m"] > 0


def test_analyze_drift_end_to_end_from_sar(sar_observed):
    out = D.analyze_drift(sar_observed, n_particles=200)
    assert out["environmental_field"] == "Demo / simulated environmental field"
    assert out["provenance"]["simulated"] is True
    hind = out["hindcast"]
    assert len(hind["best_estimate"]) == 2
    lo, hi = hind["release_window_h"]
    assert lo < hi <= 0.0
    fore = out["forecast"]
    assert [s["t_h"] for s in fore["horizons"]] == [6.0, 12.0, 24.0, 48.0]
    assert fore["provenance"]["label"] == "Demo / simulated environmental field"


# =========================================================================== #
# Epic 3.2 - real met-ocean provider (credential-gated)
# =========================================================================== #
def test_real_metocean_status_probe_is_honest():
    from backend.services.metocean_real import real_metocean_status
    st = real_metocean_status()
    assert set(st) >= {"cdsapi_installed", "xarray_installed",
                       "netcdf_reader_installed", "cdsapirc_present", "ready"}
    # "ready" iff every dependency + the CDS key are present
    assert st["ready"] == all(st[k] for k in
                              ("cdsapi_installed", "xarray_installed",
                               "netcdf_reader_installed", "cdsapirc_present"))


def test_real_metocean_provider_unavailable_falls_back_to_demo():
    import backend.services.drift as D
    from backend.services.metocean_real import deps_present
    prov = D.RealMetOceanProvider(offline=False)      # default fetch_fn = fetch_era5_hycom
    # available() only when the real deps + key are actually present on this host
    assert prov.available() == deps_present()
    # resolve_metocean_field always returns a usable (here: demo) field
    field = D.resolve_metocean_field([72.0, 18.0, 72.6, 18.6], cache=False)
    assert field.grid is not None
