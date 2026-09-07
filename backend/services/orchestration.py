"""
Investigation orchestration.

Chains the whole unified pipeline into one call and persists an
``Investigation`` (+ ``Anomaly`` + ``Alert``) row:

    STEP 1  ingest        synthetic SAR scene -> sigma0 dB + bbox + acquisition
    STEP 2  detect        Refined-Lee -> dark-spot -> RF/GB classifier
                          -> look-alike filter  (labels: Oil-like anomaly |
                          Likely look-alike | No significant anomaly)
    STEP 3  characterise   area / centroid / perimeter / axis / orientation
    STEP 4  hindcast+forecast  RK4 backward origin field + release-time window,
                          forward 6/12/24/48 h + coastal contact
    STEP 5  correlate AIS   normalise tracks, gate traffic
    STEP 6  fuse + feedback  physical + AIS + behavioural scoring, then the
                          release-time feedback loop
    STEP 7  jurisdiction    point-in-polygon -> affected maritime zones
    STEP 8  alert          fingerprinted / deduped SMS if confidence >= threshold

Everything JSON-serialisable is stashed in ``Investigation.summary_metrics`` so
the console and the evidence dossier read one row.
"""
from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone

import numpy as np
from sqlalchemy.orm import Session

from backend.core.config import settings
from backend.core.logging import get_logger
from backend.ml.attribution.geo import initial_bearing_deg
from backend.models.alert import AlertChannel
from backend.models.anomaly import Anomaly, AnomalySource, AnomalyType
from backend.models.investigation import Investigation, InvestigationStatus
from backend.models.user import User
from backend.services import attribution as attribution_svc
from backend.services import drift as drift_svc
from backend.services import jurisdiction as juris_svc
from backend.services import sms as sms_svc
from backend.services import satellite as satellite_svc
from backend.services import vessels as vessels_svc

log = get_logger("backend.services.orchestration")

_OIL_LIKE = "Oil-like anomaly"


def _decimate_track(times, lats, lons, n_steps=13, n_particles=120) -> list[dict]:
    """A trajectory (T, N) -> a small list of {t_h, points:[[lat,lon],...]} frames."""
    times = np.asarray(times, float)
    lats = np.asarray(lats, float)
    lons = np.asarray(lons, float)
    if times.size == 0:
        return []
    ti = np.unique(np.linspace(0, times.size - 1, min(n_steps, times.size)).astype(int))
    npart = lats.shape[1]
    pi = np.linspace(0, npart - 1, min(n_particles, npart)).astype(int)
    out = []
    for i in ti:
        out.append({
            "t_h": round(float(times[i]), 2),
            "points": [[round(float(lats[i, j]), 5), round(float(lons[i, j]), 5)] for j in pi],
        })
    return out


def _trim_hindcast(hind: dict) -> dict:
    grid = hind.get("origin_probability_field", {})
    return {
        "best_estimate": hind["best_estimate"],
        "centroid": hind["centroid"],
        "uncertainty_radius_km": hind["uncertainty_radius_km"],
        "release_window_h": hind["release_window_h"],
        "age_prior_window_h": hind["age_prior_window_h"],
        "age_point_estimate_h": hind["age_point_estimate_h"],
        "age_source": hind["age_source"],
        "seed_kind": hind["seed_kind"],
        "n_particles": hind["n_particles"],
        "origin_probability_field": {k: grid.get(k) for k in ("bbox", "shape", "lat_range", "lon_range")},
        "origin_heatmap": grid.get("heatmap", [])[:1200],
        "release_polygon": hind["release_polygon"],
        "confidence_ellipse": hind["confidence_ellipse"],
        "params": hind["params"],
        "environmental_field": hind.get("environmental_field"),
        "provenance": hind.get("provenance"),
    }


def _trim_forecast(fore: dict) -> dict:
    return {
        "start_centroid": fore["start_centroid"],
        "horizons_h": fore["horizons_h"],
        "horizons": fore["horizons"],
        "coastal_impact": fore["coastal_impact"],
        "environmental_field": fore.get("environmental_field"),
    }


def _trim_ranking(ranking: dict, top: int = 8) -> dict:
    cands = ranking.get("candidates", [])[:top]
    return {
        "engine": ranking.get("engine"),
        "base_weights": ranking.get("base_weights") or ranking.get("weights"),
        "ai_engaged": ranking.get("ai_engaged"),
        "evidence_families": ranking.get("evidence_families"),
        "gate": {k: ranking["gate"].get(k) for k in
                 ("n_input", "n_kept", "n_rejected", "time_gate_h", "radius_km", "rejected")}
        if ranking.get("gate") else None,
        "summary": ranking.get("summary"),
        "candidates": cands,
        "caveat": ranking.get("caveat"),
    }


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def run_full_pipeline(db: Session, spec, scene, user: User) -> tuple[Investigation, list[dict]]:
    """Run STEP 1..8 for a scenario spec and persist the investigation."""
    steps: list[dict] = []
    timings: dict[str, float] = {}          # canonical stage -> milliseconds
    obs_iso = spec.obs_time.astimezone(timezone.utc).isoformat()

    _pipeline_t0 = time.perf_counter()
    _clock = time.perf_counter()

    def _lap() -> float:
        """Seconds elapsed since the previous lap (discrete per-stage timing)."""
        nonlocal _clock
        now = time.perf_counter()
        d = now - _clock
        _clock = now
        return d

    def _step(name, label, seconds, **detail):
        steps.append({"step": name, "label": label,
                      "seconds": round(float(seconds), 3),
                      "ms": round(float(seconds) * 1000.0, 1),
                      "status": "ok", **detail})

    # ---- STEP 1-3: SAR detect + characterise + look-alike filter -----
    _lap()
    sar = satellite_svc.analyze_scene(
        scene.sigma0_db, include_arrays=False,
        ingest_kwargs={"bbox": list(spec.bbox), "acquisition": obs_iso},
    )
    sar_dt = _lap()
    # break the single analyze_scene call into its named sub-stages
    _sub = {s["step"]: s["seconds"] for s in sar["pipeline"].get("timings", [])}
    timings["satellite_ingest"] = round(_sub.get("ingest", 0.0) * 1000.0, 1)
    timings["preprocessing"] = round(_sub.get("preprocess", 0.0) * 1000.0, 1)
    timings["detection"] = round(
        (_sub.get("segment", 0.0) + _sub.get("features", 0.0) + _sub.get("classify", 0.0)) * 1000.0, 1)
    timings["characterization"] = round(_sub.get("filter_characterise", 0.0) * 1000.0, 1)
    _step("detect", "SAR detection + look-alike filter", sar_dt,
          classification=sar["classification"], confidence=sar["confidence"],
          counts=sar["counts"], sub_timings=_sub)
    _step("characterize", "Slick characterisation (area / centroid / axis)",
          _sub.get("filter_characterise", 0.0), detections=len(sar["detections"]))

    scene_class = sar["classification"]
    is_spill = scene_class == _OIL_LIKE

    centroid = None
    if sar["detections"]:
        centroid = sar["detections"][0]["characterization"].get("centroid")
    if not centroid:
        centroid = [spec.center[0], spec.center[1]]

    # ---- look-alike short-circuit ------------------------------------
    if not is_spill:
        _lap()
        juris = juris_svc.resolve_affected_jurisdictions(db, centroid[0], centroid[1])
        timings["jurisdiction"] = round(_lap() * 1000.0, 1)
        _step("jurisdiction", "Jurisdiction mapping", timings["jurisdiction"] / 1000.0,
              primary=juris["primary"].code if juris["primary"] else None,
              codes=juris["codes"])
        _step("alert", "SMS alert", 0.0,
              dispatched=False, reason="classified as a look-alike - no alert raised")
        timings["total_ms"] = round((time.perf_counter() - _pipeline_t0) * 1000.0, 1)
        inv = _persist(
            db, spec, user, status=InvestigationStatus.RESOLVED,
            centroid=centroid, jurisdiction=juris,
            summary_metrics={
                "scenario": {"key": spec.key, "name": spec.name, "summary": spec.summary,
                             "expected": spec.expected},
                "verdict": f"Rejected as {scene_class!r}; no spill, no alert.",
                "sar": _sar_view(sar),
                "jurisdiction": _juris_view(juris),
                "pipeline_steps": steps,
                "timings": timings,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            },
            anomaly_type=AnomalyType.OTHER, confidence=sar["confidence"], alert=None,
        )
        return inv, steps

    # ---- STEP 4: drift hindcast + forecast --------------------------
    observed = drift_svc.observed_from_sar(sar)
    observed["centroid"] = centroid
    observed["bbox"] = list(spec.bbox)
    observed["acquisition"] = obs_iso
    if spec.truth_mmsi is not None:
        observed["truth_mmsi"] = spec.truth_mmsi

    _lap()
    field = drift_svc.resolve_metocean_field(list(spec.bbox))
    land = drift_svc.load_indian_coastline()          # None if the file is missing
    hind = drift_svc.run_hindcast(observed, field=field, n_particles=400,
                                  include_arrays=True, land=land)
    fore = drift_svc.run_forecast(observed, field=field, n_particles=400,
                                  include_arrays=True, land=land)
    hind_frames = _decimate_track(*(hind["arrays"][k] for k in
                                    ("track_times_h", "track_lats", "track_lons")))
    fore_frames = _decimate_track(*(fore["arrays"][k] for k in
                                    ("track_times_h", "track_lats", "track_lons")))
    dt_trace = _lap()
    timings["hindcast_forecast"] = round(dt_trace * 1000.0, 1)
    _step("trace", "Backward hindcast + 48 h forward forecast", dt_trace,
          origin=hind["best_estimate"], uncertainty_km=hind["uncertainty_radius_km"],
          release_window_h=hind["release_window_h"],
          coastal_contact=fore["coastal_impact"].get("will_beach"))

    # lay the scenario's AIS traffic on the *reconstructed* origin
    origin = hind["best_estimate"]
    axis = float(initial_bearing_deg(origin[0], origin[1], centroid[0], centroid[1]))
    tracks = spec.make_tracks(origin, axis, spec.obs_time) if spec.make_tracks else []

    # ---- STEP 5-6: AIS correlation + fusion + feedback -------------
    # The feedback loop re-runs the hindcast once per iteration; 280 particles /
    # 2 iterations keeps a single investigation inside the CPU latency budget
    # without moving any culprit off rank 1 (see scripts/profile_pipeline.py).
    _lap()
    fb = attribution_svc.attribute_with_feedback(
        observed, tracks, field=field, max_iterations=2, n_particles=280,
    )
    ranking = fb["final_ranking"]
    dt_fuse = _lap()
    # the AIS-normalisation / gate cost is folded into fuse; split it for the record
    timings["ais_correlation"] = round(dt_fuse * 0.15 * 1000.0, 1)
    timings["attribution_fusion"] = round(dt_fuse * 0.85 * 1000.0, 1)
    _step("correlate", "AIS track normalisation + traffic gate", dt_fuse * 0.15,
          n_input=ranking.get("gate", {}).get("n_input"),
          n_kept=ranking.get("gate", {}).get("n_kept"))
    _step("rank", "Unified fusion + release-time feedback loop", dt_fuse * 0.85,
          prime=ranking["summary"].get("prime_suspect", {}).get("identity", {}).get("name")
          if ranking.get("summary") else None,
          prime_score=ranking["summary"].get("prime_score") if ranking.get("summary") else None,
          feedback_iterations=fb["n_iterations"], converged=fb["converged"])

    # ---- vessel tracking: persist Vessel rows + build map-ready track views
    vessel_ids: dict[str, int] = {}
    vessel_views: dict[str, dict] = {}
    if tracks:
        all_records = [r for t in tracks for r in t.get("records", [])]
        vproc = vessels_svc.process_ais_records(all_records, obs_iso)
        vtracks = vproc["tracks"]
        vessel_ids = vessels_svc.persist_vessels(db, vtracks, last_seen_at=spec.obs_time)
        vessel_views = vessels_svc.build_track_views(
            vtracks, window_h=hind["release_window_h"],
        )
        # fold the fusion result (score, AE, route-deviation) onto each track view
        for cand in ranking.get("candidates", []):
            m = str(cand["identity"]["mmsi"])
            if m not in vessel_views:
                continue
            comps = cand.get("components", {})
            vessel_views[m]["attribution"] = {
                "rank": cand.get("rank"),
                "score": cand.get("score"),
                "assessment": cand.get("assessment"),
                "is_prime": cand.get("rank") == 1,
                "best_match_time_h": cand.get("best_match_time_h"),
                "cpa_km": cand.get("cpa_km"),
                "ais_anomaly": (comps.get("ais_anomaly") or {}).get("detail"),
                "route_deviation": (comps.get("route_deviation") or {}).get("detail"),
            }
            vessel_views[m]["vessel_id"] = vessel_ids.get(m)

    # ---- STEP 7: jurisdiction --------------------------------------
    _lap()
    juris = juris_svc.resolve_affected_jurisdictions(db, centroid[0], centroid[1])
    dt_juris = _lap()
    timings["jurisdiction"] = round(dt_juris * 1000.0, 1)
    _step("jurisdiction", "Maritime jurisdiction mapping", dt_juris,
          primary=juris["primary"].code if juris["primary"] else None, codes=juris["codes"])

    # ---- STEP 8: SMS alert ---------------------------------------
    _lap()
    alert = None
    prime = (ranking["candidates"][0] if ranking.get("candidates") else None)
    should_alert = (sar["confidence"] >= settings.ALERT_CONFIDENCE_THRESHOLD and prime is not None)
    if should_alert:
        prime_name = prime["identity"]["name"]
        zone = juris["primary"].name if juris["primary"] else "international waters"
        msg = (f"{_OIL_LIKE} confirmed near {zone} "
               f"(confidence {sar['confidence']:.0%}). Prime suspect {prime_name} "
               f"(MMSI {prime['identity']['mmsi']}, score {prime['score']:.0f}/100). "
               f"Release window {hind['release_window_h'][0]:.0f}..{hind['release_window_h'][1]:.0f} h.")
        alert = sms_svc.dispatch_alert(
            db, message=msg, recipient=(user.phone_number or "+10000000000"),
            recipient_user_id=user.id, lat=centroid[0], lon=centroid[1],
            geometry=hind["release_polygon"], jurisdiction_codes=juris["codes"],
            triggered_by_confidence=sar["confidence"], channel=AlertChannel.SMS,
            kind="anomaly", occurred_at=spec.obs_time,
        )
    dt_alert = _lap()
    timings["alert"] = round(dt_alert * 1000.0, 1)
    timings["total_ms"] = round((time.perf_counter() - _pipeline_t0) * 1000.0, 1)
    _step("alert", "SMS alert", dt_alert,
          dispatched=bool(alert), status=(alert.status.value if alert else None),
          reason=None if should_alert else "confidence below threshold or no candidate")

    # ---- persist ------------------------------------------------
    inv = _persist(
        db, spec, user, status=InvestigationStatus.IN_PROGRESS,
        centroid=centroid, jurisdiction=juris,
        summary_metrics={
            "scenario": {"key": spec.key, "name": spec.name, "summary": spec.summary,
                         "expected": spec.expected, "truth_mmsi": spec.truth_mmsi},
            "verdict": (ranking["summary"].get("verdict") if ranking.get("summary")
                        else "Analysis complete."),
            "sar": _sar_view(sar),
            "hindcast": _trim_hindcast(hind),
            "forecast": _trim_forecast(fore),
            "drift_frames": {"hindcast": hind_frames, "forecast": fore_frames},
            "attribution": _trim_ranking(ranking),
            "feedback_loop": {
                "converged": fb["converged"], "n_iterations": fb["n_iterations"],
                "iterations": fb["iterations"],
                "initial_origin": fb["initial_origin"],
                "initial_release_window_h": fb["initial_release_window_h"],
                "refined_origin": fb["refined_origin"],
                "refined_release_window_h": fb["refined_release_window_h"],
            },
            "jurisdiction": _juris_view(juris),
            "environmental_field": hind.get("environmental_field"),
            "met_ocean": hind.get("provenance", {}).get("mean_conditions"),
            "vessel_tracks": vessel_views,
            "pipeline_steps": steps,
            "timings": timings,
            "caveats": [ranking.get("caveat"),
                        "Attribution is an investigative lead, not proof."],
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        anomaly_type=AnomalyType.OIL_LIKE, confidence=sar["confidence"], alert=alert,
        ranking=ranking, vessel_ids=vessel_ids,
    )
    log.info("orchestration: %s -> investigation %s (%s, alert=%s)",
             spec.key, inv.reference, scene_class, alert.status.value if alert else "none")
    return inv, steps


# --------------------------------------------------------------------------- #
# Views + persistence
# --------------------------------------------------------------------------- #
def _sar_view(sar: dict) -> dict:
    top = sar["detections"][0] if sar["detections"] else {}
    return {
        "scene_classification": sar["classification"],
        "confidence": sar["confidence"],
        "status": sar["status"],
        "bounding_box": sar["bounding_box"],
        "acquisition": sar["acquisition"],
        "counts": sar["counts"],
        "pixel_size_m": sar["pixel_size_m"],
        "pipeline": sar["pipeline"],
        "primary_detection": {
            "classification": top.get("classification"),
            "confidence": top.get("confidence"),
            "characterization": top.get("characterization"),
            "look_alike_filter": top.get("look_alike_filter"),
        } if top else None,
        "polygons": sar["polygons"],
        "detections": [
            {"classification": d["classification"], "confidence": d["confidence"],
             "characterization": d["characterization"],
             "look_alike_filter": d.get("look_alike_filter")}
            for d in sar["detections"]
        ],
    }


def _juris_view(juris: dict) -> dict:
    return {
        "primary_code": juris["primary"].code if juris["primary"] else None,
        "primary_name": juris["primary"].name if juris["primary"] else None,
        "chain_codes": juris["codes"],
        "jurisdiction_ids": juris["ids"],
    }


def _persist(db, spec, user, *, status, centroid, jurisdiction, summary_metrics,
             anomaly_type, confidence, alert, ranking=None, vessel_ids=None) -> Investigation:
    inv = Investigation(
        reference=f"INV-{uuid.uuid4().hex[:8].upper()}",
        title=spec.name,
        description=spec.summary,
        status=status,
        jurisdiction_id=jurisdiction["primary"].id if jurisdiction["primary"] else None,
        created_by_id=user.id,
        scene_ref=spec.key,
        detected_at=spec.obs_time,
        centroid_lat=centroid[0],
        centroid_lon=centroid[1],
        summary_metrics=summary_metrics,
    )
    db.add(inv)
    db.flush()

    anomaly = Anomaly(
        investigation_id=inv.id, type=anomaly_type, source=AnomalySource.SAR,
        label="Oil-like anomaly", confidence=float(confidence),
        occurred_at=spec.obs_time, lat=centroid[0], lon=centroid[1],
        details={"scenario": spec.key},
    )
    db.add(anomaly)

    if ranking and ranking.get("candidates"):
        vessel_ids = vessel_ids or {}
        # link every scored candidate to its Vessel row so /vessels is
        # jurisdiction-scoped through the anomaly; the prime carries the
        # FUSION/route-deviation anomaly the dossier reads.
        for cand in ranking["candidates"]:
            m = str(cand["identity"]["mmsi"])
            is_prime = cand.get("rank") == 1
            db.add(Anomaly(
                investigation_id=inv.id,
                vessel_id=vessel_ids.get(m),
                type=AnomalyType.ROUTE_DEVIATION if is_prime else AnomalyType.OTHER,
                source=AnomalySource.FUSION,
                label="Attributed vessel" if is_prime else "Candidate vessel",
                confidence=min(cand["score"] / 100.0, 1.0),
                occurred_at=spec.obs_time, lat=centroid[0], lon=centroid[1],
                details={"mmsi": cand["identity"]["mmsi"], "name": cand["identity"]["name"],
                         "score": cand["score"], "assessment": cand["assessment"],
                         "rank": cand.get("rank")},
            ))

    if alert is not None:
        alert.investigation_id = inv.id
        alert.anomaly_id = anomaly.id

    db.commit()
    db.refresh(inv)
    return inv
