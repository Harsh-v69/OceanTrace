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
from dataclasses import dataclass, field
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

    # weathering mass balance (Epic 4.2) over the forecast horizons
    from backend.ml.drift.aging import weather_slick
    _char0 = (sar["detections"][0]["characterization"] if sar["detections"] else {})
    _mo = hind.get("provenance", {}).get("mean_conditions") or {}
    weathering = weather_slick(
        float(_char0.get("area_km2") or 0.0),
        list(fore.get("horizons_h") or (6, 12, 24, 48)),
        temp_c=_mo.get("sea_surface_temp_c") or _mo.get("water_temp_c"),
    )

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
    _gate = ranking.get("gate") or {}
    _step("correlate", "AIS track normalisation + traffic gate", dt_fuse * 0.15,
          n_input=_gate.get("n_input"), n_kept=_gate.get("n_kept"))
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
            "weathering": weathering,
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


# --------------------------------------------------------------------------- #
# Epic 3.3 - arbitrary uploaded scenes + AIS ingestion
# --------------------------------------------------------------------------- #
@dataclass
class _UploadSpec:
    """A ``spec`` shaped like a demo scenario, for an operator-uploaded scene."""
    key: str
    name: str
    summary: str
    region_hint: str
    obs_time: datetime
    center: tuple
    bbox: list
    truth_mmsi: int | None = None
    expected: dict = field(default_factory=dict)
    make_tracks = None            # attribute, not a field - orchestration reads spec.make_tracks


@dataclass
class _UploadScene:
    sigma0_db: np.ndarray
    meta: dict = field(default_factory=dict)


def iou(mask_a, mask_b) -> float:
    """Intersection-over-union of two boolean masks (resized to match if needed)."""
    a = np.asarray(mask_a).astype(bool)
    b = np.asarray(mask_b).astype(bool)
    if a.shape != b.shape:
        import cv2
        b = cv2.resize(b.astype(np.uint8), (a.shape[1], a.shape[0]),
                       interpolation=cv2.INTER_NEAREST).astype(bool)
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def run_uploaded_scene(
    db: Session, user: User, sigma0_db: np.ndarray, *,
    title: str, bbox: list | None, acquisition: datetime | None,
    truth_mask=None,
) -> Investigation:
    """Full pipeline on an operator-uploaded SAR scene (no AIS traffic yet)."""
    obs_time = acquisition or datetime.now(timezone.utc)
    if bbox is None:
        # a 0.4 deg box centred on India's west coast as a neutral default
        bbox = [72.0, 18.0, 72.4, 18.4]
    center = ((bbox[1] + bbox[3]) / 2.0, (bbox[0] + bbox[2]) / 2.0)
    spec = _UploadSpec(
        key=f"upload-{uuid.uuid4().hex[:8]}",
        name=title or "Uploaded SAR scene",
        summary="Operator-uploaded Sentinel-1 scene analysed through the unified pipeline.",
        region_hint="", obs_time=obs_time, center=center, bbox=list(bbox),
    )
    scene = _UploadScene(sigma0_db=np.asarray(sigma0_db, dtype=np.float32))

    inv, _steps = run_full_pipeline(db, spec, scene, user)

    if truth_mask is not None:
        sar = satellite_svc.analyze_scene(
            scene.sigma0_db, include_arrays=True,
            ingest_kwargs={"bbox": list(bbox), "acquisition": obs_time.isoformat()},
        )
        detected = sar.get("mask")
        if detected is not None:
            score = iou(detected, truth_mask)
            m = dict(inv.summary_metrics or {})
            m["iou"] = {
                "value": round(score, 4),
                "against": "operator-supplied ground-truth mask",
                "detected_pixels": int(np.asarray(detected).sum()),
                "truth_pixels": int(np.asarray(truth_mask).astype(bool).sum()),
            }
            inv.summary_metrics = m
            db.commit()
            db.refresh(inv)
            log.info("uploaded scene %s: IoU vs ground truth = %.3f", inv.reference, score)
    return inv


def reattribute_with_tracks(db: Session, inv: Investigation, tracks: list) -> dict:
    """Re-run AIS correlation + fusion for an existing investigation against
    freshly-ingested vessel tracks; update ``summary_metrics`` in place."""
    import copy
    m = copy.deepcopy(inv.summary_metrics or {})
    hind_view = m.get("hindcast")
    obs_iso = (m.get("sar") or {}).get("acquisition") or inv.detected_at.isoformat()
    bbox = ((m.get("sar") or {}).get("bounding_box") or {})
    bbox_list = [bbox.get("west"), bbox.get("south"), bbox.get("east"), bbox.get("north")] \
        if isinstance(bbox, dict) else (bbox or None)

    from backend.services import vessels as _v

    vproc = _v.process_ais_records(
        [r for t in tracks for r in t.get("records", [])] if tracks and "records" in tracks[0] else tracks,
        obs_iso,
    )
    vtracks = vproc["tracks"]
    vessel_ids = _v.persist_vessels(db, vtracks, last_seen_at=inv.detected_at)

    if not hind_view:
        # look-alike / no origin: just attach the track views
        views = _v.build_track_views(vtracks)
        m.setdefault("vessel_tracks", {}).update(views)
        inv.summary_metrics = m
        db.commit(); db.refresh(inv)
        return {"reattributed": False, "vessels": len(vtracks), "report": vproc["report"]}

    observed = {
        "centroid": [inv.centroid_lat, inv.centroid_lon],
        "bbox": bbox_list,
        "acquisition": obs_iso,
        "area_km2": ((m.get("sar") or {}).get("primary_detection") or {})
                    .get("characterization", {}).get("area_km2"),
        "width_m": (((m.get("sar") or {}).get("primary_detection") or {})
                    .get("characterization", {}).get("width_km") or 0.0) * 1000.0 or None,
    }
    field_mo = drift_svc.resolve_metocean_field(bbox_list or list(drift_svc._bbox_for_observed(observed)))
    fb = attribution_svc.attribute_with_feedback(
        observed, [{"records": [r for t in tracks for r in t.get("records", [])]}]
        if tracks and isinstance(tracks[0], dict) and "records" in tracks[0] else tracks,
        field=field_mo, max_iterations=2, n_particles=280,
    )
    ranking = fb["final_ranking"]
    window_h = (m.get("hindcast") or {}).get("release_window_h")
    views = _v.build_track_views(vtracks, window_h=window_h)
    for cand in ranking.get("candidates", []):
        mm = str(cand["identity"]["mmsi"])
        if mm in views:
            comps = cand.get("components", {})
            views[mm]["attribution"] = {
                "rank": cand.get("rank"), "score": cand.get("score"),
                "assessment": cand.get("assessment"), "is_prime": cand.get("rank") == 1,
                "best_match_time_h": cand.get("best_match_time_h"), "cpa_km": cand.get("cpa_km"),
                "ais_anomaly": (comps.get("ais_anomaly") or {}).get("detail"),
                "route_deviation": (comps.get("route_deviation") or {}).get("detail"),
            }
            views[mm]["vessel_id"] = vessel_ids.get(mm)

    m["attribution"] = _trim_ranking(ranking)
    m["vessel_tracks"] = views
    m["verdict"] = (ranking["summary"].get("verdict") if ranking.get("summary")
                    else m.get("verdict"))
    m.setdefault("caveats", []).append("Attribution refreshed against operator-ingested AIS.")

    # replace the FUSION anomalies
    from backend.models.anomaly import Anomaly as _A
    for a in list(inv.anomalies):
        if a.source == AnomalySource.FUSION:
            db.delete(a)
    for cand in ranking.get("candidates", []):
        mm = str(cand["identity"]["mmsi"])
        is_prime = cand.get("rank") == 1
        db.add(_A(
            investigation_id=inv.id, vessel_id=vessel_ids.get(mm),
            type=AnomalyType.ROUTE_DEVIATION if is_prime else AnomalyType.OTHER,
            source=AnomalySource.FUSION,
            label="Attributed vessel" if is_prime else "Candidate vessel",
            confidence=min(cand["score"] / 100.0, 1.0),
            occurred_at=inv.detected_at, lat=inv.centroid_lat, lon=inv.centroid_lon,
            details={"mmsi": cand["identity"]["mmsi"], "name": cand["identity"]["name"],
                     "score": cand["score"], "assessment": cand["assessment"],
                     "rank": cand.get("rank"), "ingested_ais": True},
        ))
    inv.summary_metrics = m
    db.commit(); db.refresh(inv)
    return {
        "reattributed": True, "vessels": len(vtracks),
        "prime_suspect": (ranking["summary"].get("prime_suspect", {}).get("identity", {}).get("name")
                          if ranking.get("summary") else None),
        "n_candidates": len(ranking.get("candidates", [])),
        "report": vproc["report"],
    }
