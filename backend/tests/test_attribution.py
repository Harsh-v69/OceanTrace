"""
Phase 5 - baseline vessel-attribution engine.

Covers AIS track processing (timestamp normalisation, invalid-coord + duplicate
+ impossible-speed removal, segmentation), the mathematical soundness of the
spatiotemporal / axis / CPA primitives, that a matching-trajectory culprit
outranks decoys, that every component is normalised to [0, 1] before weighting
and weights are centrally configurable, and that edge cases (no vessels, all
gated out, empty / single-point tracks, missing origin) are handled gracefully.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from backend.ml.ais.tracks import build_tracks, clean_records, spatial_temporal_gate
from backend.ml.attribution.axis import axis_alignment
from backend.ml.attribution.config import ATTRIB, AttributionWeights
from backend.ml.attribution.geo import destination, initial_bearing_deg
from backend.ml.attribution.proximity import closest_point_of_approach, cpa_score, dwell_fraction
from backend.ml.attribution.score import score_candidate
from backend.ml.attribution.spatiotemporal import spatiotemporal_consistency
from backend.services import attribution as A
from backend.services import drift as D
from backend.services import vessels as V

OBS = datetime(2026, 3, 6, 5, 42, 0, tzinfo=timezone.utc)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def leg(mmsi, start, bearing, speed_kn, t0_h, t1_h, *, cadence_s=120.0,
        vtype=80, name="TEST", gap=None):
    speed_ms = speed_kn * 0.514444
    n = max(int((t1_h - t0_h) * 3600.0 / cadence_s) + 1, 2)
    out = []
    for th in np.linspace(t0_h, t1_h, n):
        if gap and gap[0] <= th <= gap[1]:
            continue
        la, lo = destination(start[0], start[1], bearing, (th - t0_h) * 3600.0 * speed_ms)
        out.append({
            "mmsi": mmsi, "name": name, "vessel_type": vtype,
            "timestamp": (OBS + timedelta(hours=float(th))).isoformat(),
            "latitude": float(la), "longitude": float(lo),
            "sog": speed_kn, "cog": bearing % 360.0, "heading": bearing % 360.0,
            "nav_status": 0,
        })
    return out


@pytest.fixture(scope="module")
def scenario_result() -> dict:
    inv, tracks = A.synthetic_attribution_scenario()
    return A.rank_candidates(inv, tracks)


# --------------------------------------------------------------------------- #
# AIS track processing
# --------------------------------------------------------------------------- #
def test_clean_records_normalises_mixed_timestamps():
    recs = [
        {"mmsi": 1, "timestamp": "2026-03-06T05:00:00Z", "latitude": 15.0, "longitude": 73.0},
        {"mmsi": 1, "timestamp": "2026-03-06 05:02:00", "latitude": 15.01, "longitude": 73.0},
        {"mmsi": 1, "timestamp": OBS.timestamp(), "latitude": 15.02, "longitude": 73.0},
        {"mmsi": 1, "timestamp": "not-a-time", "latitude": 15.03, "longitude": 73.0},
    ]
    rows, report = clean_records(recs)
    assert report["input"] == 4
    assert report["dropped_unparseable_time"] == 1
    assert len(rows) == 3
    assert all(isinstance(r["timestamp"], datetime) and r["timestamp"].tzinfo for r in rows)


def test_clean_records_drops_invalid_coordinates_and_null_island():
    recs = [
        {"mmsi": 1, "timestamp": "2026-03-06T05:00:00Z", "latitude": 200.0, "longitude": 73.0},
        {"mmsi": 1, "timestamp": "2026-03-06T05:01:00Z", "latitude": 15.0, "longitude": "x"},
        {"mmsi": 1, "timestamp": "2026-03-06T05:02:00Z", "latitude": 0.0, "longitude": 0.0},
        {"mmsi": 1, "timestamp": "2026-03-06T05:03:00Z", "latitude": float("nan"), "longitude": 73.0},
        {"mmsi": 1, "timestamp": "2026-03-06T05:04:00Z", "latitude": 15.0, "longitude": 73.0},
    ]
    rows, report = clean_records(recs)
    assert report["dropped_bad_coords"] == 3
    assert report["dropped_null_island"] == 1
    assert len(rows) == 1


def test_clean_records_deduplicates():
    r = {"mmsi": 7, "timestamp": "2026-03-06T05:00:00Z", "latitude": 15.0, "longitude": 73.0}
    rows, report = clean_records([r, dict(r), dict(r)])
    assert len(rows) == 1
    assert report["dropped_duplicates"] == 2


def test_build_tracks_filters_impossible_speed():
    good = leg(419000001, (15.0, 73.0), 90.0, 10.0, -6.0, 0.0, cadence_s=300.0)
    # splice a jump ~600 km away, 1 minute after the 5th fix
    bad = dict(good[5])
    bad["timestamp"] = (datetime.fromisoformat(good[5]["timestamp"]) + timedelta(minutes=1)).isoformat()
    bad["latitude"] += 5.0
    records = good[:6] + [bad] + good[6:]
    tracks, report = build_tracks(records, OBS)
    assert len(tracks) == 1
    tr = tracks[0]
    assert tr.n_removed_speed >= 1
    assert report["dropped_impossible_speed"] >= 1
    assert np.all(np.diff(tr.t_h) > 0)                       # strictly time-ordered


def test_build_tracks_segments_on_long_gap():
    a = leg(419000002, (15.0, 73.0), 45.0, 10.0, -12.0, -8.0, cadence_s=300.0)
    b = leg(419000002, (15.3, 73.3), 45.0, 10.0, -1.0, 0.0, cadence_s=300.0)   # 7 h later
    tracks, _ = build_tracks(a + b, OBS)
    tr = tracks[0]
    assert len(tr.segments) == 2
    assert any(g[2] > ATTRIB.TIME_PAD_H * 60 for g in tr.gaps)   # a multi-hour gap recorded


def test_vesseltrack_interpolation_and_course():
    recs = leg(419000003, (15.0, 73.0), 45.0, 12.0, -10.0, -2.0, cadence_s=600.0)
    tr = build_tracks(recs, OBS)[0][0]
    la, lo = tr.position_at(-6.0)                            # midpoint in time (scalar in)
    assert 15.0 < float(la) < 15.6 and 73.0 < float(lo) < 73.6
    assert abs(((float(tr.course_over(-10.0, -2.0)) - 45.0) + 180) % 360 - 180) < 3.0


def test_single_point_vessel_produces_no_track():
    one = leg(419000004, (15.0, 73.0), 0.0, 10.0, -5.0, -5.0)[:1]
    tracks, _ = build_tracks(one, OBS)
    assert tracks == []


def test_process_ais_service_summary():
    recs = leg(419000005, (15.0, 73.0), 90.0, 10.0, -8.0, 0.0, cadence_s=300.0)
    out = V.process_ais_records(recs, OBS)
    assert out["report"]["vessels"] == 1
    s = out["summaries"][0]
    assert s["n_points"] > 5 and s["duration_h"] == pytest.approx(8.0, abs=0.2)
    assert "identity" in s and s["identity"]["mmsi"] == 419000005


# --------------------------------------------------------------------------- #
# Primitive math
# --------------------------------------------------------------------------- #
def test_cpa_true_closest_point_on_a_segment():
    # a straight west-east leg through (15.0, 73.0); target 3 km due north of the middle
    recs = leg(1, (15.0, 72.9), 90.0, 12.0, -10.0, -2.0, cadence_s=1200.0)
    tr = build_tracks(recs, OBS)[0][0]
    mid_lat = float(np.mean(tr.lat))
    mid_lon = float(np.mean(tr.lon))
    tgt_lat, tgt_lon = destination(mid_lat, mid_lon, 0.0, 3000.0)
    cpa = closest_point_of_approach(tr, float(tgt_lat), float(tgt_lon))
    assert cpa["cpa_km"] == pytest.approx(3.0, abs=0.15)
    assert cpa["cpa_point"][0] == pytest.approx(mid_lat, abs=0.02)
    assert -10.0 < cpa["cpa_time_h"] < -2.0


def test_cpa_clamps_to_the_nearest_endpoint():
    recs = leg(1, (15.0, 73.0), 90.0, 12.0, -10.0, -2.0, cadence_s=1200.0)
    tr = build_tracks(recs, OBS)[0][0]
    # target well west of the start -> CPA is the distance to the first fix
    tgt = destination(tr.lat[0], tr.lon[0], 270.0, 8000.0)
    cpa = closest_point_of_approach(tr, float(tgt[0]), float(tgt[1]))
    assert cpa["cpa_km"] == pytest.approx(8.0, abs=0.2)


def test_cpa_score_is_bounded_and_monotonic():
    assert cpa_score(0.0, 25.0) == pytest.approx(1.0)
    assert cpa_score(25.0, 25.0) == 0.0
    assert cpa_score(100.0, 25.0) == 0.0
    xs = [cpa_score(d, 25.0) for d in (0, 2, 5, 10, 20)]
    assert all(a >= b for a, b in zip(xs, xs[1:]))


def test_spatiotemporal_high_at_origin_low_when_far():
    origin = [15.5, 73.1]
    window = [-12.0, -6.0]
    near = build_tracks(leg(1, tuple(origin), 45.0, 0.0, -20.0, -1.0, cadence_s=600.0), OBS)[0][0]
    far_start = destination(origin[0], origin[1], 90.0, 60_000.0)
    far = build_tracks(leg(2, (float(far_start[0]), float(far_start[1])), 45.0, 10.0, -20.0, -1.0), OBS)[0][0]

    s_near, d_near = spatiotemporal_consistency(near, origin_point=origin, window_h=window)
    s_far, _ = spatiotemporal_consistency(far, origin_point=origin, window_h=window)
    assert s_near > 0.7
    assert s_far < 0.15
    assert d_near["min_distance_km"] < 1.0


def test_spatiotemporal_zero_outside_the_window():
    tr = build_tracks(leg(1, (15.5, 73.1), 45.0, 10.0, -40.0, -30.0), OBS)[0][0]
    score, detail = spatiotemporal_consistency(tr, origin_point=[15.5, 73.1], window_h=[-12.0, -6.0])
    assert score == 0.0
    assert "not under AIS observation" in detail["finding"]


def test_axis_alignment_aligned_perpendicular_and_undirected():
    origin, centroid = [15.5, 73.1], [15.7, 73.3]
    axis = float(initial_bearing_deg(*origin, *centroid))
    window = [-12.0, -6.0]

    def score_for(bearing):
        tr = build_tracks(leg(1, tuple(origin), bearing, 10.0, -16.0, -2.0), OBS)[0][0]
        return axis_alignment(tr, axis, window)[0]

    assert score_for(axis) > 0.99                              # ~great-circle bearing drift
    assert score_for((axis + 180.0) % 360.0) > 0.99            # undirected axis
    assert score_for((axis + 90.0) % 360.0) == 0.0
    assert score_for((axis + ATTRIB.AXIS_TOLERANCE_DEG / 2.0)) == pytest.approx(0.5, abs=0.05)


def test_dwell_fraction_matches_time_inside_radius():
    origin = [15.5, 73.1]
    # 4 h sitting on the origin, then 4 h steaming away fast
    inside = leg(1, tuple(origin), 0.0, 0.1, -12.0, -8.0, cadence_s=300.0)
    away_start = destination(origin[0], origin[1], 90.0, 500.0)
    outside = leg(1, (float(away_start[0]), float(away_start[1])), 90.0, 25.0, -8.0, -4.0, cadence_s=300.0)
    tr = build_tracks(inside + outside, OBS)[0][0]
    frac, detail = dwell_fraction(tr, origin[0], origin[1], radius_km=10.0)
    assert 0.35 < frac < 0.65
    assert detail["observed_minutes"] == pytest.approx(8 * 60, abs=15)


# --------------------------------------------------------------------------- #
# Ranking: culprit vs decoys
# --------------------------------------------------------------------------- #
def test_culprit_outranks_every_decoy(scenario_result):
    res = scenario_result
    assert res["summary"]["ground_truth"]["correctly_ranked_first"] is True
    ranked = res["candidates"]
    culprit = next(c for c in ranked if c["identity"]["mmsi"] == 419111111)
    assert culprit["rank"] == 1
    assert culprit["assessment"] == "PRIME_SUSPECT"
    decoy_scores = [c["score"] for c in ranked if c["identity"]["mmsi"] != 419111111]
    assert all(culprit["score"] > d for d in decoy_scores)


def test_gate_rejects_wrong_time_and_far_vessels(scenario_result):
    g = scenario_result["gate"]
    assert g["n_input"] == 5 and g["n_kept"] < 5 and g["n_rejected"] >= 2
    reasons = " ".join(r["reason"] for r in g["rejected"])
    assert "window" in reasons or "km" in reasons


def test_every_component_is_normalised_before_weighting(scenario_result):
    for c in scenario_result["candidates"]:
        comp = c["components"]
        assert set(comp) == {"spatiotemporal", "axis_alignment", "proximity", "dwell"}
        for k, v in comp.items():
            assert 0.0 <= v["value"] <= 1.0, f"{k} value {v['value']} not in [0,1]"
            assert 0.0 <= v["weight"] <= 1.0
        assert sum(v["weight"] for v in comp.values()) == pytest.approx(1.0, abs=1e-6)
        assert c["raw_score"] <= 100.0 + 1e-6
        assert 0.0 <= c["score"] <= 100.0


def test_score_breakdown_is_transparent(scenario_result):
    c = scenario_result["candidates"][0]
    for key in ("identity", "score", "raw_score", "proximity_gate", "assessment",
                "components", "slick_axis_deg", "cpa_km", "cpa_time_h", "weights", "track"):
        assert key in c
    for k, comp in c["components"].items():
        assert set(comp) >= {"value", "weight", "points", "detail"}
        assert isinstance(comp["detail"], dict)


def test_weights_are_configurable_centrally():
    inv, tracks = A.synthetic_attribution_scenario()
    # zero out axis alignment: the perpendicular crosser should close the gap
    flat = AttributionWeights(spatiotemporal=0.6, axis_alignment=0.0, proximity=0.3, dwell=0.1)
    res = A.rank_candidates(inv, tracks, weights=flat)
    assert res["weights"] == pytest.approx(flat.normalized(), abs=1e-9)
    culprit = next(c for c in res["candidates"] if c["identity"]["mmsi"] == 419111111)
    crosser = next(c for c in res["candidates"] if c["identity"]["mmsi"] == 419444444)
    # with axis weight removed the crosser is no longer clearly beaten
    assert crosser["score"] >= culprit["score"] - 25.0
    # and every component that fed the score was still 0-1
    assert all(0.0 <= comp["value"] <= 1.0
               for c in res["candidates"] for comp in c["components"].values())


def test_proximity_gate_downweights_a_weak_signal_vessel():
    origin = [15.5, 73.1]
    window = [-12.0, -6.0]
    # a vessel loitering ~20 km east of the origin: inside the gate, but with a
    # weak space-time / proximity signal
    start = destination(origin[0], origin[1], 90.0, 20_000.0)
    tr = build_tracks(leg(419999999, (float(start[0]), float(start[1])), 0.0, 0.3, -16.0, -2.0), OBS)[0][0]
    inv = {"origin": origin, "observed_centroid": [15.7, 73.3],
           "release_window_h": window, "acquisition": OBS.isoformat()}
    gated = A.rank_candidates(inv, [tr], proximity_gate=True)["candidates"][0]
    ungated = A.rank_candidates(inv, [tr], proximity_gate=False)["candidates"][0]
    assert gated["proximity_gate"] < 1.0
    assert gated["score"] < ungated["score"]


# --------------------------------------------------------------------------- #
# Edge cases - graceful, no crash
# --------------------------------------------------------------------------- #
def test_no_vessels_supplied_returns_empty_ranking():
    inv, _ = A.synthetic_attribution_scenario()
    res = A.rank_candidates(inv, [])
    assert res["candidates"] == []
    assert res["summary"]["n_candidates"] == 0
    assert "No AIS traffic" in res["summary"]["verdict"]


def test_all_vessels_gated_out_is_reported_not_crashed():
    _, tracks = A.synthetic_attribution_scenario()
    far_inv = {"origin": [10.0, 60.0], "observed_centroid": [10.1, 60.1],
               "release_window_h": [-12.0, -6.0], "acquisition": OBS.isoformat()}
    res = A.rank_candidates(far_inv, tracks)
    assert res["candidates"] == []
    assert res["gate"]["n_kept"] == 0
    assert "No vessel could have been" in res["summary"]["verdict"]


def test_empty_and_single_point_track_records_do_not_crash():
    inv, tracks = A.synthetic_attribution_scenario()
    noisy = tracks + [{"records": []}, {"records": leg(999, (15.5, 73.1), 0, 5, -9, -9)[:1]}]
    res = A.rank_candidates(inv, noisy)
    assert res["summary"]["ground_truth"]["correctly_ranked_first"] is True


def test_missing_origin_raises_clear_error():
    with pytest.raises(ValueError, match="origin"):
        A.rank_candidates({"release_window_h": [-10.0, -2.0]}, [])


def test_raw_records_without_reference_time_raise():
    inv = {"origin": [15.5, 73.1], "release_window_h": [-12.0, -6.0]}
    recs = [{"records": leg(1, (15.5, 73.1), 45, 10, -12, -6)}]
    with pytest.raises(ValueError, match="reference time"):
        A.rank_candidates(inv, recs)


def test_accepts_a_bare_hindcast_dict_as_the_investigation():
    hind_like = {
        "best_estimate": [15.5, 73.1],
        "centroid": [15.5, 73.1],
        "release_window_h": [-14.0, -6.0],
        "age_prior_window_h": [-16.0, -4.0],
    }
    _, tracks = A.synthetic_attribution_scenario()
    hind_like["acquisition"] = OBS.isoformat()
    res = A.rank_candidates(hind_like, tracks)
    assert res["candidates"], "should still rank the gated survivors"


# --------------------------------------------------------------------------- #
# Integration: Phase 4 hindcast -> Phase 5 volume search
# --------------------------------------------------------------------------- #
def test_end_to_end_hindcast_volume_search():
    observed = D.demo_observed()
    hind = D.run_hindcast(observed, n_particles=250, include_arrays=True)
    origin = hind["best_estimate"]
    centroid = observed["centroid"]
    axis = float(initial_bearing_deg(origin[0], origin[1], centroid[0], centroid[1]))

    # a culprit that steams along the reverse-drift axis, through the origin,
    # crossing it ~11 h before the observation
    back = destination(origin[0], origin[1], (axis + 180.0) % 360.0, 11 * 3600.0 * 11.0 * 0.514444)
    culprit = leg(419123123, (float(back[0]), float(back[1])), axis, 11.0, -22.0, -1.0)
    # an innocent vessel steaming the opposite way, well to the south
    south = destination(origin[0], origin[1], 180.0, 40_000.0)
    innocent = leg(636321321, (float(south[0]), float(south[1])), 270.0, 13.0, -30.0, 0.0)

    inv = {
        "hindcast": hind,
        "observed_centroid": centroid,
        "acquisition": observed["acquisition"],
        "truth_mmsi": 419123123,
    }
    res = A.rank_candidates(inv, [{"records": culprit}, {"records": innocent}])
    assert res["summary"]["ground_truth"]["correctly_ranked_first"] is True
    top = res["candidates"][0]
    assert top["components"]["spatiotemporal"]["detail"]["mode"] == "volume"
    assert top["assessment"] in ("PRIME_SUSPECT", "PERSON_OF_INTEREST")
