"""
Phase 6 - POSEatSea AIS intelligence + Unified Attribution Fusion.

Covers: StandardScaler ENFORCEMENT (never score raw features), autoencoder
anomaly classification, LSTM route-deviation, the 7-component fusion engine
(determinism, 0-1 normalisation, transparency, evidence families, AI-disengaged
fallback), and the release-time feedback loop (window narrowing, convergence,
re-ranking).
"""
from __future__ import annotations

import warnings
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

warnings.filterwarnings("ignore")

from backend.ml.ais.anomaly import (  # noqa: E402
    AE_THRESHOLD,
    _reconstruction_errors,
    _validate_scaler,
    anomaly_score_0_1,
    load_model,
    score_frame,
    score_row,
)
from backend.ml.ais.config import FEATURE_ORDER  # noqa: E402
from backend.ml.registry import get_ais_anomaly_model, get_trajectory_model  # noqa: E402
from backend.ml.trajectory import config as TCFG  # noqa: E402
from backend.ml.trajectory.lstm import (  # noqa: E402
    assess_inputs,
    predict_next_position,
    rolling_predictions,
    route_deviation_score,
)
from backend.services import attribution as A  # noqa: E402
from backend.services import drift as D  # noqa: E402

OBS = datetime(2026, 3, 6, 5, 42, 0, tzinfo=timezone.utc)

NORMAL_PING = {"speed": 11.2, "course": 225.0, "rot": 0.0, "msg_type": 1, "status": 0,
               "accuracy": 1, "course_diff": 0.4, "rot_diff": 0.0, "speed_diff": 0.1,
               "lat_diff": -0.0008, "long_diff": -0.0011}
GROUNDING_PING = {"speed": 0.3, "course": 246.0, "rot": 127.0, "msg_type": 1, "status": 6,
                  "accuracy": 1, "course_diff": 41.0, "rot_diff": 127.0, "speed_diff": -10.6,
                  "lat_diff": -0.0001, "long_diff": -0.0001}


@pytest.fixture(scope="module")
def ae_scaler():
    return get_ais_anomaly_model()


@pytest.fixture(scope="module")
def lstm():
    return get_trajectory_model()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def mauritius_history(n=TCFG.SEQ_LEN, *, course=225.0, speed=11.0, cadence_s=60.0,
                      lat0=-20.30, lon0=58.05):
    """An in-AOI, NE-SW-lane history the LSTM will accept."""
    import pandas as pd

    speed_ms = speed * 0.514444
    m_lat, m_lon = 111_132.0, 111_320.0 * np.cos(np.radians(lat0))
    rows = []
    for i in range(n):
        d = i * cadence_s * speed_ms
        rows.append({
            "latitude": lat0 + d * np.cos(np.radians(course)) / m_lat,
            "longitude": lon0 + d * np.sin(np.radians(course)) / m_lon,
            "speed": speed, "course": course, "rot": 0.0,
            "timestamp": OBS + timedelta(seconds=i * cadence_s),
        })
    return pd.DataFrame(rows)


def mauritius_track_df(n=24, *, turn_at=None, course=225.0, turn_course=180.0,
                       speed=11.0, cadence_s=60.0, lat0=-20.30, lon0=58.05):
    import pandas as pd

    speed_ms = speed * 0.514444
    m_lat, m_lon = 111_132.0, 111_320.0 * np.cos(np.radians(lat0))
    lat, lon = lat0, lon0
    rows = []
    for i in range(n):
        c = course if (turn_at is None or i < turn_at) else turn_course
        rows.append({"latitude": lat, "longitude": lon, "speed": speed, "course": c,
                     "rot": 0.0, "timestamp": OBS + timedelta(seconds=i * cadence_s)})
        d = cadence_s * speed_ms
        lat += d * np.cos(np.radians(c)) / m_lat
        lon += d * np.sin(np.radians(c)) / m_lon
    return pd.DataFrame(rows)


# =========================================================================== #
# StandardScaler enforcement  (CRITICAL)
# =========================================================================== #
def test_load_model_returns_a_fitted_11_feature_standard_scaler(ae_scaler):
    _, scaler = ae_scaler
    assert type(scaler).__name__ == "StandardScaler"
    assert int(scaler.n_features_in_) == 11
    assert hasattr(scaler, "mean_") and hasattr(scaler, "scale_")
    assert len(scaler.mean_) == 11


def test_registry_ais_handle_always_yields_the_scaler():
    model, scaler = get_ais_anomaly_model()
    assert scaler is not None
    _validate_scaler(scaler)          # must not raise


def test_scoring_without_a_scaler_raises(ae_scaler):
    ae, _ = ae_scaler
    with pytest.raises(ValueError, match="[Ss]caler"):
        score_row(ae, None, NORMAL_PING)
    with pytest.raises(ValueError, match="[Ss]caler"):
        _reconstruction_errors(ae, None, np.zeros((1, 11)))


def test_wrong_shape_scaler_is_rejected():
    from sklearn.preprocessing import StandardScaler

    bad = StandardScaler().fit(np.random.default_rng(0).normal(size=(10, 5)))
    with pytest.raises(ValueError, match="features"):
        _validate_scaler(bad)


def test_bypassing_the_scaler_changes_the_answer_by_orders_of_magnitude(ae_scaler):
    """The single most damaging silent bug: feeding raw, unscaled features."""
    import torch

    ae, scaler = ae_scaler
    x = np.array([[GROUNDING_PING[f] for f in FEATURE_ORDER]], dtype=np.float64)

    scaled_err, _ = _reconstruction_errors(ae, scaler, x)          # the correct path
    with torch.no_grad():
        recon = ae(torch.tensor(x, dtype=torch.float32)).numpy()
    raw_err = float(np.mean((x - recon) ** 2))                     # scaler bypassed

    assert raw_err > 100.0 * float(scaled_err[0])


# =========================================================================== #
# Autoencoder anomaly classification
# =========================================================================== #
def test_normal_transit_is_not_flagged(ae_scaler):
    ae, scaler = ae_scaler
    r = score_row(ae, scaler, NORMAL_PING)
    assert r.is_anomaly is False
    assert r.score < AE_THRESHOLD
    assert r.severity == "normal"


def test_grounding_ping_is_flagged_critical(ae_scaler):
    ae, scaler = ae_scaler
    r = score_row(ae, scaler, GROUNDING_PING)
    assert r.is_anomaly is True
    assert r.severity in {"high", "critical"}
    top = {f for f, _ in r.top_contributors()}
    assert top & {"speed_diff", "course_diff", "rot_diff", "course", "speed"}


def test_score_frame_derives_diffs_and_flags_a_sudden_stop(ae_scaler):
    import pandas as pd

    ae, scaler = ae_scaler
    base = dict(mmsi=1, latitude=-20.3, longitude=58.0, rot=0.0, msg_type=1, status=0, accuracy=1)
    rows = []
    for i in range(6):
        rows.append({**base, "timestamp": OBS + timedelta(seconds=60 * i),
                     "speed": 11.0, "course": 225.0,
                     "latitude": -20.3 - 0.001 * i, "longitude": 58.0 - 0.001 * i})
    rows.append({**base, "timestamp": OBS + timedelta(seconds=60 * 6),   # hard stop + turn
                 "speed": 0.2, "course": 130.0, "rot": 120.0,
                 "latitude": -20.306, "longitude": 58.006})
    scored = score_frame(ae, scaler, pd.DataFrame(rows))
    for col in ("anomaly_score", "is_anomaly", "severity", "top_feature"):
        assert col in scored.columns
    assert bool(scored["is_anomaly"].iloc[-1]) is True
    assert not bool(scored["is_anomaly"].iloc[0])


def test_anomaly_score_maps_onto_0_1_and_saturates():
    assert anomaly_score_0_1(0.0) == 0.0
    assert anomaly_score_0_1(-1.0) == 0.0
    xs = [anomaly_score_0_1(v) for v in (0.5, 1.0, 2.0, 8.0, 50.0)]
    assert all(a <= b for a, b in zip(xs, xs[1:]))
    assert xs[-1] <= 1.0
    assert anomaly_score_0_1(AE_THRESHOLD) == pytest.approx(np.log1p(1.0) / np.log1p(6.0), abs=1e-6)


# =========================================================================== #
# LSTM route deviation
# =========================================================================== #
def test_assess_inputs_gates_bad_windows(lstm):
    short = mauritius_history(n=5)
    assert not assess_inputs(short).usable

    # Epic 2.2: outside the Mauritius AOI the window is still USABLE (locally
    # renormalised) but its confidence is downgraded and a caveat is attached.
    off = mauritius_history()
    off["latitude"] = 19.0
    off["longitude"] = 70.0
    a = assess_inputs(off)
    assert a.usable is True
    assert a.confidence == "degraded"
    assert any("AOI" in w or "extrapolation" in w for w in a.warnings)
    assert not a.blockers

    ok = assess_inputs(mauritius_history())
    assert ok.usable and ok.confidence == "nominal"


def test_lstm_prediction_stays_inside_the_aoi(lstm):
    pred = predict_next_position(lstm, mauritius_history(), strict=True)
    assert TCFG.LAT_MIN - 0.05 <= pred.predicted_lat <= TCFG.LAT_MAX + 0.05
    assert TCFG.LON_MIN - 0.05 <= pred.predicted_lon <= TCFG.LON_MAX + 0.05
    assert pred.step_km >= 0.0


def test_rolling_predictions_produce_a_deviation_trace(lstm):
    trace = rolling_predictions(lstm, mauritius_track_df(n=24))
    assert not trace.empty
    assert "deviation_km" in trace.columns
    assert (trace["deviation_km"] >= 0).all()


def test_route_deviation_score_extrapolates_out_of_aoi(lstm):
    # Epic 2.2: a Konkan (Indian-coast) track now produces a real, bounded
    # route-deviation score - flagged as an unvalidated extrapolation.
    df = mauritius_track_df(n=24)
    df["latitude"] = 15.6            # Konkan, not Mauritius
    df["longitude"] = 73.2
    score, detail = route_deviation_score(lstm, df)
    assert 0.0 <= score <= 1.0
    assert detail["usable"] is True
    assert detail["aoi"] is False
    assert detail["confidence"] == "degraded"
    assert "caveat" in detail and "extrapolat" in detail["caveat"].lower()


def test_route_deviation_is_higher_after_a_sharp_manoeuvre(lstm):
    straight = route_deviation_score(lstm, mauritius_track_df(n=28, turn_at=None))[0]
    turned = route_deviation_score(lstm, mauritius_track_df(n=28, turn_at=14,
                                                           course=225.0, turn_course=180.0))[0]
    assert 0.0 <= straight < turned <= 1.0


# =========================================================================== #
# Unified fusion engine
# =========================================================================== #
@pytest.fixture(scope="module")
def fusion_result():
    inv, tracks = A.synthetic_attribution_scenario()
    return A.fuse_attribution(inv, tracks)


def test_fusion_components_all_normalised(fusion_result):
    families = {"spatiotemporal": "physical", "axis_alignment": "physical",
                "proximity": "ais", "dwell": "ais", "blackout": "ais",
                "ais_anomaly": "ais",
                "route_deviation": "behavioural", "vessel_prior": "behavioural"}
    for c in fusion_result["candidates"]:
        comp = c["components"]
        assert set(comp) == set(families)
        for k, v in comp.items():
            assert 0.0 <= v["value"] <= 1.0, f"{k}={v['value']} not in [0,1]"
            assert v["family"] == families[k]
        # Epic 2.3: ais_anomaly is computed + shown but NOT weighted
        assert comp["ais_anomaly"]["weight"] == 0.0
        assert comp["ais_anomaly"]["points"] == 0.0
        avail = [k for k in comp if comp[k]["available"] and comp[k]["weight"] > 0]
        assert sum(comp[k]["weight"] for k in avail) == pytest.approx(1.0, abs=3e-3)
        assert 0.0 <= c["score"] <= 100.0


def test_fusion_is_deterministic():
    inv, tracks = A.synthetic_attribution_scenario()
    a = A.fuse_attribution(inv, tracks)
    b = A.fuse_attribution(inv, tracks)
    assert [c["identity"]["mmsi"] for c in a["candidates"]] == [c["identity"]["mmsi"] for c in b["candidates"]]
    for x, y in zip(a["candidates"], b["candidates"]):
        assert x["score"] == y["score"]
        assert x["components"]["ais_anomaly"]["value"] == y["components"]["ais_anomaly"]["value"]


def test_fusion_culprit_still_outranks_decoys(fusion_result):
    gt = fusion_result["summary"]["ground_truth"]
    assert gt["correctly_ranked_first"] is True
    culprit = next(c for c in fusion_result["candidates"] if c["identity"]["mmsi"] == 419111111)
    others = [c["score"] for c in fusion_result["candidates"] if c["identity"]["mmsi"] != 419111111]
    assert all(culprit["score"] > s for s in others)


def test_fusion_vessel_prior_reflects_type(fusion_result):
    culprit = next(c for c in fusion_result["candidates"] if c["identity"]["mmsi"] == 419111111)
    assert culprit["components"]["vessel_prior"]["value"] == pytest.approx(1.0)   # tanker
    fisher = next((c for c in fusion_result["candidates"] if c["identity"]["mmsi"] == 419444444), None)
    if fisher:
        assert fisher["components"]["vessel_prior"]["value"] == pytest.approx(0.30)


def test_fusion_blackout_component_lights_up_for_a_dark_period():
    inv, _ = A.synthetic_attribution_scenario()
    origin, centroid = inv["origin"], inv["observed_centroid"]
    from backend.ml.attribution.geo import destination, initial_bearing_deg

    axis = float(initial_bearing_deg(*origin, *centroid))
    start, _ = A._back_along(origin, axis, 11.0, 10.0)
    culprit = A._leg(419111111, "MT DARK RUNNER", 80, start, axis, 11.0, -18.0, -2.0,
                     gap=(-11.0, -10.0))          # 60-minute AIS blackout over the window
    clean = A._leg(636000001, "MV STEADY", 70, start, axis, 11.0, -18.0, -2.0)
    res = A.fuse_attribution(inv, [{"records": culprit}, {"records": clean}])
    dark = next(c for c in res["candidates"] if c["identity"]["mmsi"] == 419111111)
    assert dark["components"]["blackout"]["value"] > 0.0
    assert dark["components"]["blackout"]["detail"]["has_blackout_over_window"] is True


def test_fusion_ais_anomaly_component_lights_up_near_the_slick():
    inv, _ = A.synthetic_attribution_scenario()
    origin, centroid = inv["origin"], inv["observed_centroid"]
    from backend.ml.attribution.geo import destination, initial_bearing_deg

    axis = float(initial_bearing_deg(*origin, *centroid))
    start, _ = A._back_along(origin, axis, 11.0, 10.0)
    # steady run in, then a hard slow-down + turn right at the origin (t ~ -10 h)
    recs = A._leg(419111111, "MT PUMP STOP", 80, start, axis, 11.0, -18.0, -10.2, cadence_s=90.0)
    slow = A._leg(419111111, "MT PUMP STOP", 80, origin, (axis + 70.0) % 360.0, 1.5,
                  -10.0, -6.0, cadence_s=90.0)
    for r in slow:
        r["rot"] = 90.0
    res = A.fuse_attribution(inv, [{"records": recs + slow}])
    top = res["candidates"][0]
    an = top["components"]["ais_anomaly"]
    assert an["available"] is True
    assert an["value"] > 0.0
    assert an["detail"]["flagged_pings"] >= 1


def test_fusion_runs_with_ai_disengaged():
    inv, tracks = A.synthetic_attribution_scenario()
    res = A.fuse_attribution(inv, tracks, engage_ai=False)
    assert res["ai_engaged"] is False
    for c in res["candidates"]:
        assert "ais_anomaly" in c["unavailable_components"]
        assert "route_deviation" in c["unavailable_components"]
        avail = [k for k in c["components"] if c["components"][k]["available"]]
        assert sum(c["components"][k]["weight"] for k in avail) == pytest.approx(1.0, abs=3e-3)
    assert res["summary"]["ground_truth"]["correctly_ranked_first"] is True


def test_fusion_breakdown_answers_why_ranked_first(fusion_result):
    top = fusion_result["candidates"][0]
    for k, comp in top["components"].items():
        assert set(comp) >= {"value", "weight", "points", "available", "family", "detail"}
        assert isinstance(comp["detail"], dict)
        assert "finding" in comp["detail"] or comp["available"] is False
    assert top["best_match_time_h"] is not None


# =========================================================================== #
# Release-time feedback loop
# =========================================================================== #
def _culprit_and_decoy_for_demo():
    from backend.ml.attribution.geo import destination, initial_bearing_deg

    observed = D.demo_observed()
    hind0 = D.run_hindcast(observed, n_particles=200, include_arrays=True)
    o, cen = hind0["best_estimate"], observed["centroid"]
    axis = float(initial_bearing_deg(o[0], o[1], cen[0], cen[1]))
    back = destination(o[0], o[1], (axis + 180.0) % 360.0, 11 * 3600.0 * 11.0 * 0.514444)
    culprit = A._leg(1, "CULPRIT", 80, (float(back[0]), float(back[1])), axis, 11.0, -24.0, -1.0)
    south = destination(o[0], o[1], 180.0, 42_000.0)
    decoy = A._leg(2, "DECOY", 70, (float(south[0]), float(south[1])), 270.0, 13.0, -30.0, 0.0)
    return {**observed, "truth_mmsi": 1}, [{"records": culprit}, {"records": decoy}]


@pytest.fixture(scope="module")
def feedback_result():
    obs, tracks = _culprit_and_decoy_for_demo()
    return A.attribute_with_feedback(obs, tracks, n_particles=200, max_iterations=3)


def _span(window):
    return abs(window[1] - window[0])


def test_feedback_loop_narrows_the_release_window(feedback_result):
    fb = feedback_result
    assert fb["n_iterations"] >= 2
    assert _span(fb["refined_release_window_h"]) < _span(fb["initial_release_window_h"])
    assert fb["refined_uncertainty_radius_km"] > 0


def test_feedback_loop_terminates(feedback_result):
    fb = feedback_result
    assert fb["converged"] is True or fb["n_iterations"] == 3
    assert fb["iterations"][-1]["feedback"] in {
        "converged", "max iterations reached",
        "top candidate below the confidence floor - not fed back",
    }


def test_feedback_loop_reranks_each_iteration(feedback_result):
    fb = feedback_result
    assert all("top_score" in it and "top_mmsi" in it for it in fb["iterations"])
    # the culprit stays on top throughout, and it feeds a release time back
    assert all(it["top_mmsi"] == 1 for it in fb["iterations"] if it["top_mmsi"] is not None)
    assert any("fed_back_release_time_h" in it for it in fb["iterations"])
    assert fb["final_ranking"]["candidates"][0]["identity"]["mmsi"] == 1


def test_feedback_loop_is_deterministic():
    obs, tracks = _culprit_and_decoy_for_demo()
    a = A.attribute_with_feedback(obs, tracks, n_particles=200, max_iterations=3)
    b = A.attribute_with_feedback(obs, tracks, n_particles=200, max_iterations=3)
    assert a["refined_origin"] == b["refined_origin"]
    assert a["n_iterations"] == b["n_iterations"]
    assert [it["top_score"] for it in a["iterations"]] == [it["top_score"] for it in b["iterations"]]


def test_feedback_loop_stops_without_a_confident_candidate():
    from backend.ml.attribution.geo import destination

    observed = D.demo_observed()
    hind0 = D.run_hindcast(observed, n_particles=150, include_arrays=True)
    o = hind0["best_estimate"]
    # only far, weak decoys - nothing clears the confidence floor
    far = destination(o[0], o[1], 90.0, 22_000.0)
    weak = A._leg(2, "FARAWAY", 30, (float(far[0]), float(far[1])), 0.0, 0.4, -18.0, -2.0)
    fb = A.attribute_with_feedback({**observed, "truth_mmsi": 2}, [{"records": weak}],
                                   n_particles=150, max_iterations=3, min_top_score=45.0)
    assert fb["converged"] is False
    assert fb["n_iterations"] == 1
    assert "confidence floor" in fb["iterations"][0]["feedback"]
