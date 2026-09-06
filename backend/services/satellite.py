"""
Satellite (SAR) analysis service.

Thin orchestration layer over ``backend.ml.sar.pipeline``: it runs STEP 1..5 and
reshapes the result into the structured contract the rest of the backend
consumes - mask, polygons, bounding box, features, confidence - with the heavy
arrays kept as ndarrays for in-process callers and a compact RLE for transport.

The ML models load lazily on first call (``settings.ML_LAZY_LOAD``); this module
imports nothing from torch and nothing that touches the network.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from backend.core.logging import get_logger
from backend.ml.sar import labels as L
from backend.ml.sar.pipeline import run as _run_pipeline

log = get_logger("backend.services.satellite")

DEMO_SCENE = Path(__file__).resolve().parents[1] / "ml" / "sar" / "demo_data" / "demo_scene.npz"


# --------------------------------------------------------------------------- #
# Compact mask serialisation (row-major run-length; first run is False)
# --------------------------------------------------------------------------- #
def rle_encode(mask: np.ndarray) -> dict:
    flat = np.asarray(mask, bool).ravel(order="C")
    if flat.size == 0:
        return {"shape": list(mask.shape), "counts": []}
    change = np.flatnonzero(np.diff(flat))
    bounds = np.concatenate(([0], change + 1, [flat.size]))
    runs = np.diff(bounds).tolist()
    if flat[0]:                       # ensure the first run counts False pixels
        runs = [0, *runs]
    return {"shape": [int(s) for s in mask.shape], "counts": [int(r) for r in runs]}


def rle_decode(rle: dict) -> np.ndarray:
    shape = tuple(rle["shape"])
    flat = np.zeros(int(np.prod(shape)), bool)
    idx, value = 0, False
    for run in rle["counts"]:
        if value:
            flat[idx:idx + run] = True
        idx += run
        value = not value
    return flat.reshape(shape)


# --------------------------------------------------------------------------- #
# Service entry points
# --------------------------------------------------------------------------- #
def analyze_scene(
    source: Any,
    *,
    sensor: str = "SAR",
    threshold: float | None = None,
    max_dim: int | None = 2048,
    include_arrays: bool = True,
    ingest_kwargs: dict | None = None,
) -> dict:
    """
    Run the SAR pipeline on one scene and return the service contract.

    ``source`` may be a path (GeoTIFF / PNG / .npz / .npy), raw bytes, a numpy
    sigma0 array, or a pre-built ``SarScene``.
    """
    result = _run_pipeline(
        source,
        sensor=sensor,
        threshold=threshold,
        max_dim=max_dim,
        ingest_kwargs=ingest_kwargs,
    )

    w, s, e, n = result["bounding_box"]
    oil_mask = result["masks"]["oil_like"]
    cand_mask = result["masks"]["candidates"]

    detections = result["detections"]
    for d in detections:
        L.assert_canonical(d["classification"])  # invariant guard

    contract: dict = {
        "classification": L.assert_canonical(result["scene_classification"]),
        "status": result["status"],
        "confidence": result["confidence"],
        "sensor": result["sensor"],
        "acquisition": result["acquisition"],
        "bounding_box": {
            "west": w, "south": s, "east": e, "north": n,
            "source": result["bbox_source"],
        },
        "scene_size_px": result["scene_size_px"],
        "pixel_size_m": result["pixel_size_m"],
        # per-detection feature vectors (the 30 named SAR features)
        "features": [d["features"] for d in detections],
        # per-detection geometry / backscatter description (STEP 5)
        "characterization": [d["characterization"] for d in detections],
        "detections": detections,
        "polygons": result["polygons"],
        "mask_rle": {
            "oil_like": rle_encode(oil_mask),
            "candidates": rle_encode(cand_mask),
        },
        "counts": {
            "candidates": result["n_candidates"],
            "oil_like": result["n_oil_like"],
            "look_alike": result["n_look_alike"],
        },
        "pipeline": {
            "classifier": result["classifier"],
            "decision_threshold": result["decision_threshold"],
            "runtime_seconds": result["runtime_seconds"],
            "timings": result["timings"],
            "preprocess_stats": result["preprocess_stats"],
            "segmentation_diagnostics": result["segmentation_diagnostics"],
        },
        "labels_vocabulary": list(L.CANONICAL_LABELS),
    }

    if include_arrays:
        contract["mask"] = oil_mask            # HxW bool - oil-like union
        contract["candidate_mask"] = cand_mask

    log.info(
        "SAR analyze: %s (conf=%.3f, %d candidate(s), %.2fs)",
        contract["classification"],
        contract["confidence"],
        result["n_candidates"],
        result["runtime_seconds"],
    )
    return contract


def demo_scene_path() -> Path:
    """Path to the bundled offline demo scene (.npz)."""
    return DEMO_SCENE


def analyze_demo(**kwargs) -> dict:
    """Convenience: run :func:`analyze_scene` on the bundled demo scene."""
    if not DEMO_SCENE.exists():  # pragma: no cover - generated & committed
        raise FileNotFoundError(
            f"demo scene missing at {DEMO_SCENE}; run "
            "`python -m backend.ml.sar.demo_data._generate`"
        )
    return analyze_scene(DEMO_SCENE, **kwargs)
