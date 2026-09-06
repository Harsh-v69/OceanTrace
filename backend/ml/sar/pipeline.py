"""
End-to-end SAR oil-spill detection pipeline (STEP 1 -> STEP 5).

    STEP 1  ingest      GeoTIFF / PNG / npz / array  -> sigma0 dB + bbox + time
    STEP 2  preprocess  Refined-Lee speckle filter + adaptive local threshold
    STEP 3  detect      adaptive dark-spot + watershed split + 30 features
                        + RF/GB ensemble  (physics rule fallback if no model)
    STEP 4  filter      reject low-wind cells / coastal artefacts / biogenic
                        films by context gates
    STEP 5  characterise area (km2), centroid, perimeter, major/minor axis,
                        orientation, GeoJSON polygon

Every candidate is labelled strictly one of:
    "Oil-like anomaly" | "Likely look-alike" | "No significant anomaly"

CPU: this is a pure numpy/OpenCV/scikit chain. On a 1024x768 demo scene it runs
in well under a second on one core; large rasters are optionally decimated
before processing (``max_dim``) and the mask is mapped back to source pixels.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from backend.ml.sar import labels as L
from backend.ml.sar.characterize import characterise
from backend.ml.sar.classifier import (
    Prediction,
    confidence_band,
    get_classifier,
    rule_based_score,
)
from backend.ml.sar.config import SAR
from backend.ml.sar.darkspot import adaptive_offset, estimate_background, scales_for, segment
from backend.ml.sar.features import FEATURE_NAMES, extract_batch
from backend.ml.sar.geo import GeoTransform
from backend.ml.sar.ingest import SarScene, load_scene
from backend.ml.sar.preprocess import preprocess


def _classifier_for(sensor: str):
    """
    Obtain the oil classifier. The SAR model is pulled through the lazy model
    registry so ``/system/models`` reflects it; other sensors use the per-sensor
    cache directly. Both paths share the same process-wide singleton.
    """
    if str(sensor).upper() == "SAR":
        try:
            from backend.ml.registry import get_registry
            return get_registry()["sar_classifier"].get()
        except Exception:  # noqa: BLE001 - fall back to the direct loader
            return get_classifier("SAR")
    return get_classifier(sensor)


@dataclass
class _Timer:
    steps: list = field(default_factory=list)
    _t: float = field(default_factory=time.perf_counter)

    def mark(self, name: str, **info):
        now = time.perf_counter()
        self.steps.append({"step": name, "seconds": round(now - self._t, 4), **info})
        self._t = now


# --------------------------------------------------------------------------- #
# STEP 4 - look-alike context filter
# --------------------------------------------------------------------------- #
def _lookalike_gates(feats: dict) -> list[str]:
    """Reasons this candidate should be treated as a look-alike, not oil."""
    gates: list[str] = []
    if feats.get("local_wind_ms", 7.0) > SAR.LOOKALIKE_MAX_WIND_MS:
        gates.append("wind_too_strong")
    if feats.get("dist_to_land_km", 999.0) < SAR.LOOKALIKE_MIN_COAST_KM:
        gates.append("inside_surf_zone")
    if feats.get("mean_contrast_db", 0.0) < SAR.LOOKALIKE_MIN_CONTRAST_DB:
        gates.append("contrast_too_weak")
    diffuse_edge = feats.get("border_gradient_db_px", 0.0) < SAR.LOOKALIKE_MIN_BORDER_GRAD
    round_blob = feats.get("spreading", 0.0) > SAR.LOOKALIKE_MAX_SPREADING
    if diffuse_edge and round_blob:
        gates.append("diffuse_round_low_wind_cell")
    return gates


# --------------------------------------------------------------------------- #
# Large-raster guard
# --------------------------------------------------------------------------- #
def _maybe_decimate(scene: SarScene, max_dim: int | None):
    if not max_dim:
        return scene, 1.0
    h, w = scene.sigma0_db.shape
    longest = max(h, w)
    if longest <= max_dim:
        return scene, 1.0
    f = max_dim / float(longest)
    new_wh = (max(int(round(w * f)), 8), max(int(round(h * f)), 8))
    small_db = cv2.resize(scene.sigma0_db, new_wh, interpolation=cv2.INTER_AREA)
    small = SarScene(
        sigma0_db=small_db.astype(np.float32),
        bbox=scene.bbox,
        pixel_m=scene.pixel_m / f,
        transform=GeoTransform(tuple(scene.bbox), new_wh[0], new_wh[1]),
        acquisition=scene.acquisition,
        source=scene.source,
        bbox_source=scene.bbox_source,
        truth_mask=(
            None if scene.truth_mask is None
            else cv2.resize(scene.truth_mask.astype(np.uint8), new_wh,
                            interpolation=cv2.INTER_NEAREST).astype(bool)
        ),
        meta={**scene.meta, "decimated_from": [w, h]},
    )
    return small, f


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def run(
    source,
    *,
    sensor: str = "SAR",
    threshold: float | None = None,
    offset_k: float = 1.8,
    max_dim: int | None = 2048,
    ingest_kwargs: dict | None = None,
) -> dict:
    """Run STEP 1..5 and return a JSON-serialisable result dict."""
    timer = _Timer()

    # -- STEP 1: ingest ------------------------------------------------
    scene = load_scene(source, **(ingest_kwargs or {}))
    scene, scale = _maybe_decimate(scene, max_dim)
    transform = scene.transform
    h, w = scene.sigma0_db.shape
    timer.mark(
        "ingest",
        source=scene.source,
        size=[w, h],
        bbox_source=scene.bbox_source,
        acquisition=scene.acquisition.isoformat() if scene.acquisition else None,
        decimation=round(scale, 4),
    )

    # -- STEP 2: preprocess ------------------------------------------
    pre = preprocess(scene.sigma0_db)
    timer.mark("preprocess", **pre["stats"])

    # -- STEP 3: detect --------------------------------------------
    bg = estimate_background(pre["db"], scales=scales_for(sensor))
    offset = adaptive_offset(pre["db"], pre["sea"], k=offset_k, background=bg)
    px_km2 = (float(np.mean(transform.pixel_size_m())) / 1000.0) ** 2
    min_area_px = int(max(SAR.MIN_SPILL_AREA_PX, SAR.MIN_SPILL_AREA_KM2 / max(px_km2, 1e-12)))
    cands, seg = segment(
        pre["db"], pre["sea"], offset_low=offset, background=bg, min_area_px=min_area_px
    )
    timer.mark("segment", **seg["diagnostics"])

    X, feat_dicts = extract_batch(
        cands, pre["db"], bg, transform.pixel_size_m(), pre["land"]
    )
    timer.mark("features", n_candidates=len(cands),
               n_features=X.shape[1] if X.size else 0)

    clf = _classifier_for(sensor)
    thr = (
        float(threshold)
        if threshold is not None
        else (SAR.REAL_DECISION_THRESHOLD
              if str(scene.meta.get("source_kind", "")).lower() == "real"
              else SAR.OIL_PROBABILITY_THRESHOLD)
    )
    if clf is not None and len(cands):
        preds = clf.predict(X, threshold=thr)
        model_name = clf.metrics.get("model", "ensemble")
    else:
        preds = []
        for fd in feat_dicts:
            p = rule_based_score(fd)
            preds.append(
                Prediction(p, p >= thr, confidence_band(p), {"physics_rule": p})
            )
        model_name = "physics rule-based fallback (no trained model present)"
    timer.mark("classify", model=model_name, threshold=round(thr, 3),
               n_oil_raw=int(sum(1 for p in preds if p.is_oil)))

    # -- STEP 4 + STEP 5: filter + characterise -------------------
    detections: list[dict] = []
    oil_mask = np.zeros((h, w), bool)
    cand_mask = np.zeros((h, w), bool)
    n_forced = 0

    for cand, feats, pred in zip(cands, feat_dicts, preds):
        cand_mask |= cand.mask
        gates = _lookalike_gates(feats) if pred.is_oil else []
        is_oil = pred.is_oil and not gates
        n_forced += 1 if (pred.is_oil and gates) else 0

        label = L.label_for(is_oil, has_candidate=True)
        char = characterise(cand, feats, pred, transform, transform.pixel_area_m2())

        if is_oil:
            oil_mask |= cand.mask

        detections.append(
            {
                "classification": L.assert_canonical(label),
                "confidence": round(float(pred.probability), 4),
                "confidence_band": pred.confidence,
                "is_oil_like": is_oil,
                "look_alike_filter": {
                    "passed": not gates,
                    "gates_triggered": gates,
                },
                "area_px": int(cand.area_px),
                "bbox_pixels": list(cand.bbox),
                "centroid_pixel": [round(v, 1) for v in cand.centroid_rc],
                "characterization": char["properties"],
                "geojson": char["geojson"],
                "features": {k: round(float(v), 5) for k, v in feats.items()},
            }
        )

    # sort strongest first
    _order = {L.OIL_LIKE: 0, L.LOOKALIKE: 1, L.NONE: 2}
    detections.sort(key=lambda d: (_order[d["classification"]], -d["confidence"]))
    timer.mark("filter_characterise", n_detections=len(detections),
               n_forced_lookalike=n_forced)

    det_labels = [d["classification"] for d in detections]
    scene_label = L.scene_label(det_labels)
    oil_dets = [d for d in detections if d["classification"] == L.OIL_LIKE]
    if oil_dets:
        confidence = max(d["confidence"] for d in oil_dets)
    elif detections:
        confidence = max(d["confidence"] for d in detections)
    else:
        confidence = 0.0

    total_s = round(sum(s["seconds"] for s in timer.steps), 4)
    return {
        "status": "SPILL_DETECTED" if oil_dets else (
            "LOOKALIKE_ONLY" if detections else "NO_ANOMALY"
        ),
        "scene_classification": scene_label,
        "confidence": round(float(confidence), 4),
        "sensor": str(sensor).upper(),
        "bounding_box": [round(float(v), 6) for v in scene.bbox],
        "bbox_source": scene.bbox_source,
        "acquisition": scene.acquisition.isoformat() if scene.acquisition else None,
        "scene_size_px": [w, h],
        "pixel_size_m": round(float(np.mean(transform.pixel_size_m())), 2),
        "n_candidates": len(cands),
        "n_oil_like": len(oil_dets),
        "n_look_alike": sum(1 for d in det_labels if d == L.LOOKALIKE),
        "classifier": model_name,
        "decision_threshold": round(thr, 3),
        "detections": detections,
        "masks": {
            "oil_like": oil_mask,
            "candidates": cand_mask,
        },
        "polygons": {
            "type": "FeatureCollection",
            "features": [d["geojson"] for d in detections],
        },
        "preprocess_stats": pre["stats"],
        "segmentation_diagnostics": seg["diagnostics"],
        "timings": timer.steps,
        "runtime_seconds": total_s,
        "feature_names": FEATURE_NAMES,
        "labels_vocabulary": list(L.CANONICAL_LABELS),
    }
