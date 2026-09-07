"""
SAR ingestion + oil-like anomaly detection.

Ported from OceanTrace (the merge-selected SAR detector - see
docs/MERGE_ARCHITECTURE.md). Pure numpy / OpenCV / scikit, CPU-only, no U-Net.

Public entry point:

    from backend.ml.sar import run_pipeline
    result = run_pipeline("path/to/scene.tif")

Or go through the service wrapper: ``backend.services.satellite.analyze_scene``.
"""
from backend.ml.sar.labels import CANONICAL_LABELS, LOOKALIKE, NONE, OIL_LIKE
from backend.ml.sar.pipeline import run as run_pipeline

__all__ = [
    "run_pipeline",
    "CANONICAL_LABELS",
    "OIL_LIKE",
    "LOOKALIKE",
    "NONE",
]
