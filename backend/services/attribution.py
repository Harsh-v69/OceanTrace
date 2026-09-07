"""
Attribution service - rank candidate vessels against an investigation.

Ingests an investigation (its reconstructed origin + release-time window, from
the Phase 4 hindcast) and a set of vessel tracks (raw AIS or pre-built
:class:`VesselTrack`), gates the traffic, scores every survivor with the
baseline physical / spatiotemporal criteria, and returns a ranked list with a
fully transparent per-component breakdown.

Weights are configured centrally in ``backend.ml.attribution.config`` and every
score component is normalised to [0, 1] before weighting.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from backend.core.logging import get_logger
from backend.ml.ais.tracks import VesselTrack, build_tracks, spatial_temporal_gate
from backend.ml.attribution.axis import axis_alignment, slick_axis_from_origin
from backend.ml.attribution.config import (
    ATTRIB,
    AttributionWeights,
    FusionWeights,
    assessment_band,
)
from backend.ml.attribution.geo import haversine_m, initial_bearing_deg
from backend.ml.attribution.proximity import closest_point_of_approach, cpa_score, dwell_fraction
from backend.ml.attribution.score import score_candidate
from backend.ml.attribution.spatiotemporal import spatiotemporal_consistency

log = get_logger("backend.services.attribution")

# evidence family for each fusion component (shown in the UI breakdown)
_EVIDENCE_FAMILY = {
    "spatiotemporal": "physical",
    "axis_alignment": "physical",
    "proximity": "ais",
    "dwell": "ais",
    "blackout": "ais",
    "ais_anomaly": "ais",
    "route_deviation": "behavioural",
    "vessel_prior": "behavioural",
}

ATTRIBUTION_CAVEAT = (
    "Attribution is an investigative lead, not proof. The reconstructed origin "
    "carries the hindcast's uncertainty (about a factor of two in eddy "
    "diffusivity), so the release point and time are distributions rather than "
    "points. A vessel that never transmitted AIS cannot be attributed by AIS, "
    "and confirmation requires oil fingerprinting against a bunker sample."
)


# --------------------------------------------------------------------------- #
# Input resolution
# --------------------------------------------------------------------------- #
def _looks_like_hindcast(d: dict) -> bool:
    return isinstance(d, dict) and "best_estimate" in d and "release_window_h" in d


def _resolve_investigation(investigation: dict) -> dict:
    """Normalise the many shapes an investigation can arrive in."""
    inv = dict(investigation or {})
    hind = inv.get("hindcast")
    if hind is None and _looks_like_hindcast(inv):
        hind = inv                                   # a bare run_hindcast() result

    origin = inv.get("origin") or inv.get("best_estimate")
    window = inv.get("release_window_h")
    centroid = inv.get("observed_centroid") or inv.get("centroid")
    axis = inv.get("slick_axis_deg")
    age_prior = inv.get("age_prior_window_h")
    hindcast_track = None
    origin_points = None
    spread_km = float(inv.get("origin_spread_km") or 0.0)

    if hind is not None:
        origin = origin or hind.get("best_estimate")
        window = window or hind.get("release_window_h")
        centroid = centroid or hind.get("centroid")
        age_prior = age_prior or hind.get("age_prior_window_h")
        spread_km = spread_km or float(hind.get("uncertainty_radius_km") or 0.0)
        arrays = hind.get("arrays") or {}
        if {"track_times_h", "track_lats", "track_lons"} <= set(arrays):
            hindcast_track = {
                "times_h": np.asarray(arrays["track_times_h"], float),
                "lats": np.asarray(arrays["track_lats"], float),
                "lons": np.asarray(arrays["track_lons"], float),
            }
        if "origin_lats" in arrays and "origin_lons" in arrays:
            origin_points = np.column_stack(
                [np.asarray(arrays["origin_lats"], float), np.asarray(arrays["origin_lons"], float)]
            )

    if origin is None or window is None:
        raise ValueError(
            "investigation needs a reconstructed 'origin' [lat, lon] and a "
            "'release_window_h' [lo, hi] (or an embedded hindcast result)"
        )

    obs = inv.get("acquisition") or inv.get("observed_at") or inv.get("reference_time")
    return {
        "origin": [float(origin[0]), float(origin[1])],
        "window_h": [float(min(window)), float(max(window))],
        "observed_centroid": [float(centroid[0]), float(centroid[1])] if centroid else None,
        "slick_axis_deg": float(axis) if axis is not None else None,
        "age_prior_window_h": [float(min(age_prior)), float(max(age_prior))] if age_prior else None,
        "hindcast_track": hindcast_track,
        "origin_points": origin_points,
        "origin_spread_km": spread_km,
        "observed_at": obs,
        "truth_mmsi": inv.get("truth_mmsi"),
    }


def _coerce_tracks(vessel_tracks, reference_time) -> tuple[list[VesselTrack], dict]:
    """Accept VesselTracks, {mmsi, records}, {records,...}, or a flat AIS record list."""
    items = list(vessel_tracks or [])
    if not items:
        return [], {"vessels": 0, "input": 0, "kept": 0}
    if all(isinstance(x, VesselTrack) for x in items):
        return items, {"vessels": len(items), "prebuilt": True}

    records: list[dict] = []
    for x in items:
        if isinstance(x, VesselTrack):
            raise ValueError("mix of VesselTrack and raw records is not supported")
        if isinstance(x, dict) and "records" in x:
            records.extend(x["records"])
        elif isinstance(x, dict):
            records.append(x)                        # a flat AIS record
        else:
            raise ValueError(f"unsupported vessel-track item: {type(x)!r}")

    if reference_time is None:
        raise ValueError(
            "raw AIS records need a reference time - set investigation['acquisition']"
        )
    return build_tracks(records, reference_time)


# --------------------------------------------------------------------------- #
# Ranking
# --------------------------------------------------------------------------- #
def rank_candidates(
    investigation: dict,
    vessel_tracks,
    *,
    weights: AttributionWeights | dict | None = None,
    radius_km: float | None = None,
    pad_h: float | None = None,
    top_n: int | None = None,
    min_score: float | None = None,
    proximity_gate: bool | None = None,
) -> dict:
    """Return a ranked candidate list with transparent score components."""
    ctx = _resolve_investigation(investigation)
    radius_km = float(radius_km or ATTRIB.SEARCH_RADIUS_KM)
    pad_h = float(pad_h if pad_h is not None else ATTRIB.TIME_PAD_H)
    min_score = float(min_score if min_score is not None else ATTRIB.MIN_SCORE_TO_REPORT)
    weights_out = (weights.normalized() if isinstance(weights, AttributionWeights)
                   else weights) or AttributionWeights().normalized()

    tracks, track_report = _coerce_tracks(vessel_tracks, ctx["observed_at"])
    base = {
        "weights": weights_out,
        "search_radius_km": radius_km,
        "release_window_h": ctx["window_h"],
        "reconstructed_origin": ctx["origin"],
        "track_processing": track_report,
        "caveat": ATTRIBUTION_CAVEAT,
    }

    if not tracks:
        return {**base, "candidates": [], "gate": None,
                "summary": {"n_candidates": 0,
                            "verdict": "No AIS traffic supplied for this investigation."}}

    kept, rejected, gate = spatial_temporal_gate(
        tracks, ctx["origin"][0], ctx["origin"][1], ctx["window_h"],
        radius_km=radius_km, pad_h=pad_h, origin_spread_km=ctx["origin_spread_km"],
    )
    gate["rejected"] = [
        {"mmsi": int(tr.mmsi), "name": tr.name, "reason": reason} for tr, reason in rejected[:50]
    ]

    if not kept:
        return {**base, "candidates": [], "gate": gate,
                "summary": {"n_candidates": 0, "n_gated_out": len(rejected),
                            "verdict": ("No vessel could have been at the reconstructed "
                                        "origin during the release window.")}}

    scored = [
        score_candidate(
            tr,
            origin_point=ctx["origin"],
            window_h=ctx["window_h"],
            observed_centroid=ctx["observed_centroid"],
            slick_axis_deg=ctx["slick_axis_deg"],
            origin_points=ctx["origin_points"],
            hindcast_track=ctx["hindcast_track"],
            age_prior_window_h=ctx["age_prior_window_h"],
            weights=weights,
            radius_km=radius_km,
            pad_h=pad_h,
            proximity_gate=proximity_gate,
        )
        for tr in kept
    ]
    scored.sort(key=lambda r: -r["score"])

    for i, r in enumerate(scored):
        r["rank"] = i + 1
        r["reportable"] = r["score"] >= min_score
    if len(scored) > 1:
        scored[0]["margin_over_next"] = round(scored[0]["score"] - scored[1]["score"], 2)

    out = scored[: int(top_n)] if top_n else scored
    summary = _summarise(scored, ctx["truth_mmsi"], min_score)
    log.info(
        "attribution: %d gated -> %d scored; prime %s (%.1f, %s)",
        len(rejected), len(scored),
        summary["prime_suspect"]["identity"]["name"] if summary.get("prime_suspect") else "-",
        summary.get("prime_score", 0.0), summary.get("prime_assessment", "-"),
    )
    return {**base, "candidates": out, "gate": gate, "summary": summary}


def _summarise(results: list[dict], truth_mmsi, min_score: float) -> dict:
    top = results[0]
    runner = results[1] if len(results) > 1 else None
    margin = round(top["score"] - (runner["score"] if runner else 0.0), 2)

    if top["score"] < ATTRIB.BANDS["PERSON_OF_INTEREST"]:
        verdict = ("No vessel in the search window accounts for this slick with any "
                   "confidence. Widen the radius or the time window.")
    elif margin >= 15.0:
        verdict = (f"{top['identity']['name']} separates clearly from all other traffic in "
                   f"the window on space-time coincidence and axis alignment.")
    else:
        verdict = (f"{top['identity']['name']} ranks highest, but "
                   f"{runner['identity']['name'] if runner else 'the runner-up'} is close "
                   f"behind - treat this as unresolved between them.")

    summary = {
        "n_candidates": len(results),
        "n_reportable": sum(1 for r in results if r["score"] >= min_score),
        "prime_suspect": {"identity": top["identity"], "components": top["components"]},
        "prime_score": top["score"],
        "prime_assessment": top["assessment"],
        "margin_over_next": margin,
        "verdict": verdict,
    }
    if truth_mmsi is not None:
        pos = next((r["rank"] for r in results if r["identity"]["mmsi"] == int(truth_mmsi)), None)
        summary["ground_truth"] = {
            "culprit_mmsi": int(truth_mmsi),
            "rank_assigned": pos,
            "correctly_ranked_first": bool(pos == 1),
        }
    return summary


# --------------------------------------------------------------------------- #
# Synthetic scenario (orchestration demo + tests)
# --------------------------------------------------------------------------- #
_OBS_TIME = datetime(2026, 3, 6, 5, 42, 0, tzinfo=timezone.utc)


def _leg(mmsi, name, vessel_type, start, bearing_deg, speed_kn, t0_h, t1_h,
         *, cadence_s=120.0, gap=None, obs_time=_OBS_TIME):
    """Straight-line AIS leg: constant course + speed, one ping every ``cadence_s``."""
    from datetime import timedelta

    from backend.ml.attribution.geo import destination

    speed_ms = speed_kn * 0.514444
    n = max(int((t1_h - t0_h) * 3600.0 / cadence_s) + 1, 2)
    ts_h = np.linspace(t0_h, t1_h, n)
    recs = []
    for th in ts_h:
        if gap and gap[0] <= th <= gap[1]:
            continue
        dist_m = (th - t0_h) * 3600.0 * speed_ms
        la, lo = destination(start[0], start[1], bearing_deg, dist_m)
        recs.append({
            "mmsi": mmsi, "name": name, "vessel_type": vessel_type,
            "timestamp": (obs_time + timedelta(hours=float(th))).isoformat(),
            "latitude": float(la), "longitude": float(lo),
            "sog": round(speed_kn, 1), "cog": round(bearing_deg % 360.0, 1),
            "heading": round(bearing_deg % 360.0, 1), "nav_status": 0,
        })
    return recs


def synthetic_attribution_scenario() -> tuple[dict, list[dict]]:
    """A culprit + four decoys the baseline scorer must discriminate.

    Returns ``(investigation, vessel_track_records)``.
    """
    origin = [15.50, 73.10]
    centroid = [15.63, 73.26]
    axis = float(initial_bearing_deg(*origin, *centroid))        # reverse-drift bearing (~NE)
    window = [-14.0, -6.0]

    # culprit: steams along the axis, through the origin, inside the window
    culprit_start, _ = _back_along(origin, axis, 11.0, 10.0)     # start 10 h before origin pass
    culprit = _leg(419111111, "MT NORD MERIDIAN", 80, culprit_start, axis, 11.0, -20.0, -2.0)

    # decoy A: same path through the origin, but 20 h too early
    a_start, _ = _back_along(origin, axis, 11.0, 10.0)
    decoy_a = _leg(636222222, "MV EARLY TRADER", 70, a_start, axis, 11.0, -40.0, -24.0)

    # decoy B: has a 40-minute AIS gap, but transits 55 km NE of the origin
    from backend.ml.attribution.geo import destination
    b0 = destination(origin[0], origin[1], 40.0, 55_000.0)
    decoy_b = _leg(563333333, "MV GAP RUNNER", 70, (float(b0[0]), float(b0[1])),
                   120.0, 12.0, -18.0, -3.0, gap=(-11.0, -10.33))

    # decoy C: crosses near the origin inside the window, but perpendicular to the axis
    c_start = destination(origin[0], origin[1], (axis + 90.0) % 360.0, -9_000.0)
    decoy_c = _leg(419444444, "FV CROSS CURRENT", 30, (float(c_start[0]), float(c_start[1])),
                   (axis + 90.0) % 360.0, 6.0, -12.0, -6.0)

    # decoy D: never comes near - transits 45 km due south throughout
    d0 = destination(origin[0], origin[1], 180.0, 45_000.0)
    decoy_d = _leg(477555555, "MV SOUTHERN LANE", 70, (float(d0[0]), float(d0[1])),
                   90.0, 14.0, -30.0, 0.0)

    investigation = {
        "origin": origin,
        "observed_centroid": centroid,
        "release_window_h": window,
        "acquisition": _OBS_TIME.isoformat(),
        "age_prior_window_h": [-16.0, -4.0],
        "truth_mmsi": 419111111,
    }
    tracks = [
        {"records": culprit}, {"records": decoy_a}, {"records": decoy_b},
        {"records": decoy_c}, {"records": decoy_d},
    ]
    return investigation, tracks


def _back_along(point, bearing_deg, speed_kn, hours):
    """Point ``hours`` of travel BEHIND ``point`` along ``bearing_deg``."""
    from backend.ml.attribution.geo import destination

    dist_m = hours * 3600.0 * speed_kn * 0.514444
    la, lo = destination(point[0], point[1], (bearing_deg + 180.0) % 360.0, dist_m)
    return (float(la), float(lo)), dist_m


# =========================================================================== #
# Phase 6 - Unified Attribution Fusion Engine
# =========================================================================== #
_AI_STATUS: dict = {"loaded": False, "error": None}


def _load_ai_models():
    """Lazily load the AE (+ scaler) and the LSTM. Returns (ae, scaler, lstm) or Nones."""
    try:
        from backend.ml.registry import get_ais_anomaly_model, get_trajectory_model

        ae, scaler = get_ais_anomaly_model()          # scaler validated inside load_model
        lstm = get_trajectory_model()
        _AI_STATUS.update(loaded=True, error=None)
        return ae, scaler, lstm
    except Exception as exc:  # noqa: BLE001 - fusion still runs on physical evidence
        _AI_STATUS.update(loaded=False, error=f"{type(exc).__name__}: {exc}")
        log.warning("AI models unavailable (%s); fusion runs on physical evidence only", exc)
        return None, None, None


def _blackout_component(track, window_h) -> tuple[float, dict]:
    from backend.services import vessels as _v

    info = _v.detect_blackouts(track, window_h=window_h)
    mins = info["longest_blackout_over_window_min"]
    score = float(np.clip(mins / ATTRIB.BLACKOUT_FULL_SCORE_MIN, 0.0, 1.0))
    info["finding"] = (
        f"AIS silent for {mins:.0f} min over the release window."
        if info["has_blackout_over_window"] else
        "No AIS transmission gap over the release window."
    )
    return score, info


def _vessel_prior_component(track) -> tuple[float, dict]:
    grp = (track.vessel_type_group or "UNKNOWN").upper()
    val = float(ATTRIB.TYPE_PRIOR.get(grp, ATTRIB.VESSEL_PRIOR_DEFAULT))
    return val, {"vessel_type_group": grp, "prior": val,
                 "finding": f"A-priori discharge likelihood for a {grp.lower()} is {val:.2f}."}


def _ais_anomaly_component(track, ae, scaler, origin, window_h, radius_km) -> tuple[float, dict]:
    from backend.ml.ais.anomaly import AE_THRESHOLD, anomaly_score_0_1, score_frame
    from backend.services import vessels as _v

    if ae is None or scaler is None:
        return 0.0, {"available": False, "finding": "autoencoder not engaged"}

    frame = _v.ais_feature_frame(track)
    scored = score_frame(ae, scaler, frame)                # scaler.transform is mandatory here
    d_km = haversine_m(scored["latitude"].to_numpy(), scored["longitude"].to_numpy(),
                       origin[0], origin[1]) / 1000.0
    t0, t1 = float(min(window_h)) - ATTRIB.TIME_PAD_H, float(max(window_h)) + ATTRIB.TIME_PAD_H
    th = np.asarray(track.t_h, float)
    near = (d_km <= radius_km) & (th >= t0) & (th <= t1)
    sub = scored[near]
    if sub.empty:
        return 0.0, {"available": True, "pings_near_slick": 0,
                     "finding": "no AIS pings near the reconstructed origin in the window"}

    peak = float(sub["anomaly_score"].max())
    flagged = int(sub["is_anomaly"].sum())
    top_feats = sub.loc[sub["is_anomaly"], "top_feature"].value_counts().head(3).to_dict()
    score = anomaly_score_0_1(peak, AE_THRESHOLD)
    return score, {
        "available": True,
        "pings_near_slick": int(len(sub)),
        "flagged_pings": flagged,
        "peak_reconstruction_error": round(peak, 4),
        "threshold": AE_THRESHOLD,
        "top_error_features": {str(k): int(v) for k, v in top_feats.items()},
        "finding": (f"Autoencoder peak reconstruction error {peak:.2f} "
                    f"(threshold {AE_THRESHOLD:.2f}) on {flagged} flagged ping(s) near the slick."
                    if flagged else
                    f"Autoencoder saw nothing anomalous near the slick (peak {peak:.2f})."),
    }


def _route_deviation_component(track, lstm) -> tuple[float, dict]:
    from backend.ml.trajectory.lstm import route_deviation_score
    from backend.services import vessels as _v

    if lstm is None:
        return 0.0, {"available": False, "usable": False, "finding": "LSTM not engaged"}
    score, detail = route_deviation_score(lstm, _v.lstm_track_frame(track))
    detail["available"] = True
    return score, detail


def fuse_attribution(
    investigation: dict,
    vessel_tracks,
    *,
    weights: FusionWeights | dict | None = None,
    engage_ai: bool = True,
    radius_km: float | None = None,
    pad_h: float | None = None,
    top_n: int | None = None,
    min_score: float | None = None,
    proximity_gate: bool | None = None,
) -> dict:
    """
    Unified fusion: physical + AIS + behavioural evidence -> one transparent
    0-100 score per vessel. Weights are renormalised over whatever components
    are actually available for each vessel (e.g. the LSTM is Mauritius-AOI only).
    """
    ctx = _resolve_investigation(investigation)
    radius_km = float(radius_km or ATTRIB.SEARCH_RADIUS_KM)
    pad_h = float(pad_h if pad_h is not None else ATTRIB.TIME_PAD_H)
    min_score = float(min_score if min_score is not None else ATTRIB.MIN_SCORE_TO_REPORT)
    gate_on = ATTRIB.PROXIMITY_GATE_ENABLED if proximity_gate is None else bool(proximity_gate)

    if isinstance(weights, FusionWeights):
        base_w = weights.normalized()
    elif isinstance(weights, dict):
        tot = sum(weights.values()) or 1.0
        base_w = {k: float(v) / tot for k, v in weights.items()}
    else:
        base_w = FusionWeights().normalized()

    ae = scaler = lstm = None
    if engage_ai:
        ae, scaler, lstm = _load_ai_models()

    tracks, track_report = _coerce_tracks(vessel_tracks, ctx["observed_at"])
    header = {
        "engine": "unified_fusion",
        "base_weights": base_w,
        "evidence_families": _EVIDENCE_FAMILY,
        "ai_engaged": bool(ae is not None and lstm is not None),
        "ai_status": dict(_AI_STATUS),
        "search_radius_km": radius_km,
        "release_window_h": ctx["window_h"],
        "reconstructed_origin": ctx["origin"],
        "track_processing": track_report,
        "caveat": ATTRIBUTION_CAVEAT,
    }
    if not tracks:
        return {**header, "candidates": [], "gate": None,
                "summary": {"n_candidates": 0, "verdict": "No AIS traffic supplied."}}

    kept, rejected, gate = spatial_temporal_gate(
        tracks, ctx["origin"][0], ctx["origin"][1], ctx["window_h"],
        radius_km=radius_km, pad_h=pad_h, origin_spread_km=ctx["origin_spread_km"],
    )
    gate["rejected"] = [{"mmsi": int(t.mmsi), "name": t.name, "reason": r} for t, r in rejected[:50]]
    if not kept:
        return {**header, "candidates": [], "gate": gate,
                "summary": {"n_candidates": 0, "n_gated_out": len(rejected),
                            "verdict": ("No vessel could have been at the reconstructed "
                                        "origin during the release window.")}}

    axis_deg = ctx["slick_axis_deg"]
    if axis_deg is None and ctx["observed_centroid"] is not None:
        axis_deg = slick_axis_from_origin(ctx["origin"], ctx["observed_centroid"])

    results = [
        _fuse_one(tr, ctx, axis_deg, base_w, radius_km, pad_h, gate_on, ae, scaler, lstm)
        for tr in kept
    ]
    results.sort(key=lambda r: -r["score"])
    for i, r in enumerate(results):
        r["rank"] = i + 1
        r["reportable"] = r["score"] >= min_score
    if len(results) > 1:
        results[0]["margin_over_next"] = round(results[0]["score"] - results[1]["score"], 2)

    out = results[: int(top_n)] if top_n else results
    summary = _summarise(results, ctx["truth_mmsi"], min_score)
    log.info("fusion: %d gated -> %d scored; prime %s (%.1f, %s) ai=%s",
             len(rejected), len(results),
             summary["prime_suspect"]["identity"]["name"] if summary.get("prime_suspect") else "-",
             summary.get("prime_score", 0.0), summary.get("prime_assessment", "-"),
             header["ai_engaged"])
    return {**header, "candidates": out, "gate": gate, "summary": summary}


def _fuse_one(track, ctx, axis_deg, base_w, radius_km, pad_h, gate_on, ae, scaler, lstm) -> dict:
    window = ctx["window_h"]

    s_st, d_st = spatiotemporal_consistency(
        track, origin_point=ctx["origin"], origin_points=ctx["origin_points"],
        window_h=window, pad_h=pad_h, hindcast_track=ctx["hindcast_track"],
        age_prior_window_h=ctx["age_prior_window_h"],
    )
    if axis_deg is not None:
        s_ax, d_ax = axis_alignment(track, axis_deg, window, pad_h=pad_h,
                                    slick_centroid=ctx["observed_centroid"])
    else:
        s_ax, d_ax = 0.0, {"available": False, "finding": "no slick axis available"}

    cpa = closest_point_of_approach(track, ctx["origin"][0], ctx["origin"][1],
                                    window_h=window, pad_h=pad_h)
    s_pr = cpa_score(cpa["cpa_km"], radius_km)
    _cpa_km = float(cpa["cpa_km"])
    d_pr = {
        **cpa,
        "cpa_km": round(_cpa_km, 3) if np.isfinite(_cpa_km) else None,
        "finding": (f"Closest approach to the reconstructed origin was {_cpa_km:.1f} km"
                    + (f", {abs(cpa['cpa_time_h']):.1f} h before the observation."
                       if cpa["cpa_time_h"] is not None else ".")
                    if np.isfinite(_cpa_km) else
                    "Vessel never came within the search radius of the origin."),
    }

    s_dw, d_dw = dwell_fraction(track, ctx["origin"][0], ctx["origin"][1],
                                radius_km=radius_km, window_h=window, pad_h=pad_h)
    d_dw.setdefault(
        "finding",
        (f"Spent {d_dw['minutes_in_radius']:.0f} of {d_dw['observed_minutes']:.0f} "
         f"observed minutes within {d_dw['radius_km']:.0f} km of the origin."
         if "minutes_in_radius" in d_dw else "Too little coverage to measure dwell."),
    )

    s_bl, d_bl = _blackout_component(track, window)
    s_an, d_an = _ais_anomaly_component(track, ae, scaler, ctx["origin"], window, radius_km)
    s_rd, d_rd = _route_deviation_component(track, lstm)
    s_vp, d_vp = _vessel_prior_component(track)

    raw = {
        "spatiotemporal": (float(s_st), d_st, True),
        "axis_alignment": (float(s_ax), d_ax, axis_deg is not None),
        "proximity": (float(s_pr), d_pr, True),
        "dwell": (float(s_dw), d_dw, True),
        "blackout": (float(s_bl), d_bl, True),
        "ais_anomaly": (float(s_an), d_an, bool(d_an.get("available", True))),
        "route_deviation": (float(s_rd), d_rd, bool(d_rd.get("available") and d_rd.get("usable"))),
        "vessel_prior": (float(s_vp), d_vp, True),
    }
    for k, (v, _, _) in raw.items():
        assert -1e-9 <= v <= 1.0 + 1e-9, f"fusion component {k}={v} not in [0, 1]"

    available = [k for k, (_, _, ok) in raw.items() if ok and base_w.get(k, 0.0) > 0]
    denom = sum(base_w[k] for k in available) or 1.0
    eff_w = {k: (base_w[k] / denom if k in available else 0.0) for k in raw}

    raw_total = sum(eff_w[k] * raw[k][0] for k in raw) * 100.0
    gate = 1.0
    if gate_on:
        signal = max(raw["spatiotemporal"][0], raw["proximity"][0])
        gate = float(np.clip(signal / max(ATTRIB.PROXIMITY_GATE_KNEE, 1e-6), 0.0, 1.0))
    total = float(np.clip(raw_total * gate, 0.0, 100.0))

    components = {
        k: {
            "value": round(raw[k][0], 4),
            "weight": round(eff_w[k], 4),
            "points": round(eff_w[k] * raw[k][0] * 100.0 * gate, 2),
            "available": bool(raw[k][2]),
            "family": _EVIDENCE_FAMILY[k],
            "detail": raw[k][1],
        }
        for k in raw
    }
    return {
        "identity": track.identity(),
        "score": round(total, 2),
        "raw_score": round(raw_total, 2),
        "proximity_gate": round(gate, 3),
        "assessment": assessment_band(total),
        "components": components,
        "effective_weights": {k: round(eff_w[k], 4) for k in raw},
        "unavailable_components": [k for k in raw if not raw[k][2]],
        "slick_axis_deg": round(float(axis_deg) % 360.0, 1) if axis_deg is not None else None,
        "cpa_km": d_pr["cpa_km"],
        "cpa_time_h": round(cpa["cpa_time_h"], 3) if cpa["cpa_time_h"] is not None else None,
        "best_match_time_h": d_st.get("best_match_time_h"),
        "track": {
            "n_points": track.n_points,
            "first_seen_h": round(float(track.t_h[0]), 2),
            "last_seen_h": round(float(track.t_h[-1]), 2),
            "n_gaps": len(track.gaps),
            "n_segments": len(track.segments),
        },
    }


# =========================================================================== #
# Phase 6 - Release-Time Feedback Loop
# =========================================================================== #
def attribute_with_feedback(
    observed: dict,
    vessel_tracks,
    *,
    field=None,
    weights: FusionWeights | dict | None = None,
    engage_ai: bool = True,
    max_iterations: int = 3,
    min_top_score: float = 45.0,
    feedback_half_window_h: float = 2.5,
    converge_km: float = 1.5,
    converge_h: float = 0.75,
    radius_km: float | None = None,
    n_particles: int | None = None,
    field_kwargs: dict | None = None,
) -> dict:
    """
    Iterate: hindcast origin -> fuse/rank vessels -> take the top candidate's
    AIS space-time match time -> re-run the hindcast with that tighter release
    window -> re-rank. Stops on convergence or ``max_iterations``.
    """
    from backend.services import drift as D

    obs = dict(observed)
    bbox = obs.get("bbox") or D._bbox_for_observed(obs)
    mo = field or D.resolve_metocean_field(bbox, **(field_kwargs or {}))

    hind = D.run_hindcast(obs, field=mo, n_particles=n_particles, include_arrays=True)
    iterations: list[dict] = []
    ranking: dict = {}
    prev_origin = prev_age = None
    converged = False

    for it in range(max_iterations):
        investigation = {
            "hindcast": hind,
            "observed_centroid": obs.get("centroid"),
            "acquisition": obs.get("acquisition"),
            "truth_mmsi": obs.get("truth_mmsi"),
        }
        ranking = fuse_attribution(
            investigation, vessel_tracks, weights=weights, engage_ai=engage_ai,
            radius_km=radius_km,
        )
        top = ranking["candidates"][0] if ranking["candidates"] else None
        origin = hind["best_estimate"]
        rec: dict = {
            "iteration": it,
            "origin": [round(v, 5) for v in origin],
            "release_window_h": [round(v, 2) for v in hind["release_window_h"]],
            "age_point_estimate_h": hind["age_point_estimate_h"],
            "age_source": hind["age_source"],
            "top_mmsi": top["identity"]["mmsi"] if top else None,
            "top_name": top["identity"]["name"] if top else None,
            "top_score": top["score"] if top else None,
        }
        if prev_origin is not None:
            rec["origin_shift_km"] = round(
                float(haversine_m(origin[0], origin[1], prev_origin[0], prev_origin[1])) / 1000.0, 3
            )
        iterations.append(rec)

        if top is None or top["score"] < min_top_score:
            rec["feedback"] = "top candidate below the confidence floor - not fed back"
            break
        best_t = top.get("best_match_time_h")
        if best_t is None:
            rec["feedback"] = "no AIS space-time match time to feed back"
            break

        new_age = abs(float(best_t))
        rec["fed_back_release_time_h"] = round(float(best_t), 3)
        if (prev_age is not None
                and abs(new_age - prev_age) < converge_h
                and rec.get("origin_shift_km", 1e9) < converge_km):
            rec["feedback"] = "converged"
            converged = True
            break

        rec["feedback"] = f"re-running hindcast with release window centred on {best_t:.1f} h"
        hind = D.run_hindcast(
            obs, field=mo, n_particles=n_particles, include_arrays=True,
            age_hours=new_age,
            age_window_h=(max(new_age - feedback_half_window_h, 0.5),
                          new_age + feedback_half_window_h),
        )
        prev_origin, prev_age = origin, new_age
    else:
        iterations[-1]["feedback"] = "max iterations reached"

    return {
        "engine": "release_time_feedback_loop",
        "converged": converged,
        "n_iterations": len(iterations),
        "iterations": iterations,
        "environmental_field": mo.label,
        "initial_origin": iterations[0]["origin"],
        "initial_release_window_h": iterations[0]["release_window_h"],
        "refined_origin": [round(v, 5) for v in hind["best_estimate"]],
        "refined_release_window_h": [round(v, 2) for v in hind["release_window_h"]],
        "refined_uncertainty_radius_km": hind["uncertainty_radius_km"],
        "final_ranking": ranking,
        "caveat": ATTRIBUTION_CAVEAT,
    }
