"""
(Re)train the SAR oil / look-alike RF+GB ensemble from the bundled dataset.

    python -m backend.ml.sar.train

The labelled dataset ``models/training_set.npz`` (X: 30 features, y: 0/1) was
harvested by OceanTrace's simulator sweep during the Phase 1 audit and is
committed so the model is reproducible offline without pulling in the simulator
or the drift stack. Training takes a few seconds on CPU.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from backend.ml.sar.classifier import OilClassifier, reset_cache
from backend.ml.sar.features import FEATURE_NAMES

MODELS_DIR = Path(__file__).resolve().parent / "models"
DATASET = MODELS_DIR / "training_set.npz"


def main(dataset: Path = DATASET, test_size: float = 0.25, seed: int = 42) -> int:
    if not dataset.exists():
        print(f"dataset not found: {dataset}", file=sys.stderr)
        return 1

    data = np.load(dataset, allow_pickle=True)
    X, y = np.asarray(data["X"], np.float32), np.asarray(data["y"], np.int64)
    if "feature_names" in data.files:
        names = [str(v) for v in data["feature_names"]]
        if names != FEATURE_NAMES:
            print("WARNING: dataset feature order differs from features.py", file=sys.stderr)

    print(f"dataset: {X.shape[0]} candidates, {int(y.sum())} oil, {int((1 - y).sum())} look-alike")
    clf = OilClassifier.train(X, y, test_size=test_size, seed=seed)
    out = clf.save(sensor="SAR")
    reset_cache()
    print(f"saved -> {out}")
    print("\ntop features:")
    for k, v in list(clf.importances.items())[:10]:
        print(f"  {k:24s} {v:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
