"""
Four deterministic demo scenarios.

    mumbai-high-confidence  - a clear oil slick off Maharashtra with one culprit
    lookalike-darkpatch     - a low-wind / biogenic dark patch that must be rejected
    ambiguous-drift         - a weak, aged slick where drift + behaviour separate
                              the culprit from decoys
    wakashio-mauritius      - the MV Wakashio grounding replay (Mauritius AOI), so
                              the POSEatSea LSTM route-deviation + AIS autoencoder
                              anomaly both engage

A scenario ships a synthetic SAR scene plus a ``make_tracks(origin, axis, t)``
callback. The orchestration service runs SAR detection + the hindcast first,
then asks the scenario to lay its AIS traffic on the *reconstructed* origin -
so the culprit is genuinely on the reverse-drift axis, deterministically, on
every run.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.ml.attribution.geo import destination
from backend.ml.sar.synth import synth_scene
from backend.models.jurisdiction import Jurisdiction, JurisdictionType

_KN = 0.514444
_T = datetime(2026, 3, 6, 5, 42, 0, tzinfo=timezone.utc)


# --------------------------------------------------------------------------- #
# AIS leg builder
# --------------------------------------------------------------------------- #
def _leg(mmsi, name, vtype, start, bearing_deg, speed_kn, t0_h, t1_h, obs_time,
         *, cadence_s=120.0, gap=None, decel_to=None, nav_status=0):
    n = max(int((t1_h - t0_h) * 3600.0 / cadence_s) + 1, 2)
    ts_h = np.linspace(t0_h, t1_h, n)
    lat, lon = float(start[0]), float(start[1])
    recs = []
    for i, th in enumerate(ts_h):
        if gap and gap[0] <= th <= gap[1]:
            continue
        frac = i / max(n - 1, 1)
        spd = speed_kn if decel_to is None else (
            speed_kn + (decel_to - speed_kn) * max(0.0, (frac - 0.55) / 0.45))
        st, rot = nav_status, 0.0
        if decel_to is not None and frac > 0.9:
            st, rot = 6, 70.0
        recs.append({
            "mmsi": mmsi, "name": name, "vessel_type": vtype,
            "timestamp": (obs_time + timedelta(hours=float(th))).isoformat(),
            "latitude": lat, "longitude": lon,
            "sog": round(max(spd, 0.1), 1), "cog": round(bearing_deg % 360.0, 1),
            "heading": round(bearing_deg % 360.0, 1), "rot": rot, "nav_status": st,
        })
        step = (ts_h[min(i + 1, n - 1)] - th) * 3600.0 * max(spd, 0.1) * _KN
        la, lo = destination(lat, lon, bearing_deg, step)
        lat, lon = float(la), float(lo)
    return recs


def _behind(point, bearing_deg, speed_kn, hours):
    d = hours * 3600.0 * speed_kn * _KN
    la, lo = destination(point[0], point[1], (bearing_deg + 180.0) % 360.0, d)
    return (float(la), float(lo))


# --------------------------------------------------------------------------- #
# Spec
# --------------------------------------------------------------------------- #
@dataclass
class ScenarioSpec:
    key: str
    name: str
    summary: str
    region_hint: str
    obs_time: datetime
    center: tuple[float, float]
    bbox: list
    truth_mmsi: int | None = None
    expected: dict = field(default_factory=dict)
    #: (origin[lat,lon], axis_deg, obs_time) -> [{"records": [...]}, ...]
    make_tracks: Callable | None = None


# --------------------------------------------------------------------------- #
# Track factories (called by orchestration with the reconstructed origin)
# --------------------------------------------------------------------------- #
def _mumbai_tracks(origin, axis, t):
    culprit = _leg(419810001, "MT KONKAN PRIDE", 80,
                   _behind(origin, axis, 11.0, 11.0), axis, 11.0, -22.0, -1.0, t,
                   cadence_s=120.0)
    early = _leg(636810002, "MV EARLY PASSAGE", 70,
                 _behind(origin, axis, 11.0, 11.0), axis, 11.0, -46.0, -26.0, t)
    far = _leg(477810003, "MV SOUTH TRADER", 70,
              destination(origin[0], origin[1], 180.0, 46_000.0), 90.0, 14.0, -30.0, 0.0, t)
    return [{"records": culprit}, {"records": early}, {"records": far}]


def _ambiguous_tracks(origin, axis, t):
    culprit = _leg(419810011, "MT DECCAN STAR", 80,
                   _behind(origin, axis, 10.0, 12.0), axis, 10.0, -24.0, -1.0, t,
                   cadence_s=150.0)
    cross_start = destination(origin[0], origin[1], (axis + 90.0) % 360.0, -9_000.0)
    crosser = _leg(419810012, "FV COASTAL NET", 30,
                   (float(cross_start[0]), float(cross_start[1])),
                   (axis + 90.0) % 360.0, 6.0, -13.0, -5.0, t)
    gap_far = _leg(636810013, "MV GULF HORIZON", 70,
                   destination(origin[0], origin[1], 45.0, 40_000.0),
                   120.0, 12.0, -18.0, -3.0, t, gap=(-11.0, -10.2))
    off_lane = _leg(477810014, "MV OFF LANE", 70,
                    destination(origin[0], origin[1], 200.0, 52_000.0),
                    90.0, 13.0, -28.0, 0.0, t)
    return [{"records": culprit}, {"records": crosser},
            {"records": gap_far}, {"records": off_lane}]


def _wakashio_tracks(origin, axis, t):
    # The Wakashio steams along the reverse-drift axis, laying the slick as it
    # goes (continuous-release model), passes the origin, then decelerates hard
    # over the final approach (the grounding). It tracks the drifting cloud, so
    # the space-time match is strong; the deceleration + hard rate-of-turn light
    # up the AIS autoencoder, and the manoeuvre lights up the LSTM
    # route-deviation (60 s cadence, NE-SW course -> inside the model envelope).
    start = _behind(origin, axis, 11.5, 10.0)
    approach = _leg(419990001, "MV WAKASHIO", 70, start, axis, 11.5, -16.0, -4.0, t,
                    cadence_s=60.0)
    hand = (approach[-1]["latitude"], approach[-1]["longitude"])
    grounding = _leg(419990001, "MV WAKASHIO", 70, hand, axis, 3.0, -4.0, -0.2, t,
                     cadence_s=60.0, decel_to=0.25)
    culprit = approach + grounding
    innocent_a = _leg(636990002, "MV KOTA SURIA", 70,
                      destination(origin[0], origin[1], (axis + 60.0) % 360.0, 17_000.0),
                      (axis + 25.0) % 360.0, 12.0, -20.0, 0.0, t, cadence_s=60.0)
    innocent_b = _leg(563990003, "MV VERY MARIA", 70,
                      destination(origin[0], origin[1], (axis + 205.0) % 360.0, 18_000.0),
                      (axis + 165.0) % 360.0, 13.0, -18.0, 0.0, t, cadence_s=60.0)
    return [{"records": culprit}, {"records": innocent_a}, {"records": innocent_b}]


# --------------------------------------------------------------------------- #
# Scenario builders
# --------------------------------------------------------------------------- #
def _mumbai_high_confidence():
    c = (18.72, 72.30)
    scene = synth_scene(center_lat=c[0], center_lon=c[1], seed=810001, wind_ms=7.5,
                        with_oil=True, with_lowwind=True, with_biogenic=True)
    spec = ScenarioSpec(
        key="mumbai-high-confidence",
        name="Mumbai / Maharashtra - high-confidence spill",
        summary=("A sharp-edged oil slick off the Maharashtra coast. One tanker "
                 "sits on the reconstructed reverse-drift axis; two decoys do not."),
        region_hint="IN-MH", obs_time=_T, center=c, bbox=list(scene.meta["bbox"]),
        truth_mmsi=419810001, make_tracks=_mumbai_tracks,
        expected={"classification": "Oil-like anomaly", "alert": True,
                  "prime_mmsi": 419810001, "jurisdiction": "IN-MH"},
    )
    return spec, scene


def _lookalike_darkpatch():
    c = (15.05, 73.60)
    scene = synth_scene(center_lat=c[0], center_lon=c[1], seed=810002, wind_ms=4.2,
                        with_oil=False, with_lowwind=True, with_biogenic=True)
    spec = ScenarioSpec(
        key="lookalike-darkpatch",
        name="Goa / Karnataka - look-alike dark patch",
        summary=("A low-wind cell plus a biogenic film - no mineral oil. The "
                 "look-alike filter must reject it and raise no alert."),
        region_hint="IN-GA-KA", obs_time=_T, center=c, bbox=list(scene.meta["bbox"]),
        truth_mmsi=None, make_tracks=None,
        expected={"classification_not": "Oil-like anomaly", "alert": False},
    )
    return spec, scene


def _ambiguous_drift():
    c = (17.05, 72.05)
    scene = synth_scene(center_lat=c[0], center_lon=c[1], seed=810003, wind_ms=6.0,
                        with_oil=True, with_lowwind=True, with_biogenic=True)
    spec = ScenarioSpec(
        key="ambiguous-drift",
        name="Maharashtra offshore - ambiguous spill",
        summary=("An aged, low-contrast slick. Proximity alone does not separate "
                 "the tanker from a fishing boat that crossed the origin - drift, "
                 "axis alignment and the release-time feedback loop do."),
        region_hint="IN-MH", obs_time=_T, center=c, bbox=list(scene.meta["bbox"]),
        truth_mmsi=419810011, make_tracks=_ambiguous_tracks,
        expected={"classification": "Oil-like anomaly", "prime_mmsi": 419810011,
                  "alert": True},
    )
    return spec, scene


def _wakashio_mauritius():
    c = (-20.35, 57.90)
    scene = synth_scene(center_lat=c[0], center_lon=c[1], seed=810004, wind_ms=6.5,
                        with_oil=True, with_lowwind=False, with_biogenic=True)
    spec = ScenarioSpec(
        key="wakashio-mauritius",
        name="Mauritius - MV Wakashio grounding",
        summary=("The MV Wakashio grounding near Pointe d'Esny. Inside the "
                 "Mauritius AOI the POSEatSea LSTM route-deviation and AIS "
                 "autoencoder anomaly both engage on the deceleration."),
        region_hint="MU-AOI", obs_time=_T, center=c, bbox=list(scene.meta["bbox"]),
        truth_mmsi=419990001, make_tracks=_wakashio_tracks,
        expected={"classification": "Oil-like anomaly", "prime_mmsi": 419990001,
                  "alert": True, "ai_components_engaged": True, "jurisdiction": "MU-AOI"},
    )
    return spec, scene


_BUILDERS = {
    "mumbai-high-confidence": _mumbai_high_confidence,
    "lookalike-darkpatch": _lookalike_darkpatch,
    "ambiguous-drift": _ambiguous_drift,
    "wakashio-mauritius": _wakashio_mauritius,
}

SCENARIOS = tuple(_BUILDERS)


def list_scenarios() -> list[dict]:
    out = []
    for key, fn in _BUILDERS.items():
        spec, _ = fn()
        out.append({
            "key": spec.key, "name": spec.name, "summary": spec.summary,
            "region_hint": spec.region_hint, "center": list(spec.center),
            "has_vessels": spec.make_tracks is not None, "expected": spec.expected,
        })
    return out


def build_scenario(key: str):
    if key not in _BUILDERS:
        raise KeyError(f"unknown scenario {key!r}; choose one of {list(_BUILDERS)}")
    return _BUILDERS[key]()


# --------------------------------------------------------------------------- #
# Mauritius AOI zone (seeded lazily - keeps the Phase-7 catalogue count intact)
# --------------------------------------------------------------------------- #
_MU_AOI = {
    "type": "Polygon",
    "coordinates": [[[57.45, -20.85], [58.20, -20.85], [58.20, -19.95],
                     [57.45, -19.95], [57.45, -20.85]]],
}


def seed_scenario_zones(db: Session) -> int:
    if db.execute(select(Jurisdiction).where(Jurisdiction.code == "MU-AOI")).scalars().first():
        return 0
    db.add(Jurisdiction(code="MU-AOI", name="Mauritius Area of Interest (Pointe d'Esny)",
                        type=JurisdictionType.ZONE, geometry=_MU_AOI))
    db.commit()
    return 1
