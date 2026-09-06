"""
Oil / look-alike classifier  (STEP 3, part 3).

Ported from SAMUDRA NETRA ``ml/sar/classifier.py`` - the merge-selected SAR
detector (docs/MERGE_ARCHITECTURE.md).

A soft-voting ensemble of a Random Forest (non-linear feature interactions,
interpretable importances) and a Gradient Boosting classifier (lower bias).
Both are trained on the 30 physically-motivated features; no CNN - it needs far
more labelled data and every decision here traces back to a named physical
quantity. The U-Net refiner from SN is deliberately NOT ported: the Phase 1
audit reproduced SN's own measurement that it *lowers* final-mask IoU
(0.751 -> 0.700) and costs ~2 s on CPU.

If no trained model file is present, ``rule_based_score`` provides a
deterministic physics fallback so the pipeline still returns a defensible
answer on a fresh checkout.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from backend.ml.sar.config import SAR
from backend.ml.sar.features import FEATURE_NAMES

MODELS_DIR = Path(__file__).resolve().parent / "models"
MODEL_PATH = MODELS_DIR / "oil_classifier.joblib"
METRICS_PATH = MODELS_DIR / "oil_classifier_metrics.json"


def model_path(sensor: str = "SAR") -> Path:
    sensor = str(sensor).upper()
    return MODEL_PATH if sensor == "SAR" else MODELS_DIR / f"oil_classifier_{sensor.lower()}.joblib"


def metrics_path(sensor: str = "SAR") -> Path:
    sensor = str(sensor).upper()
    return (
        METRICS_PATH
        if sensor == "SAR"
        else MODELS_DIR / f"oil_classifier_{sensor.lower()}_metrics.json"
    )


def confidence_band(p: float) -> str:
    """Map a probability to the operational confidence label."""
    for name, thr in sorted(SAR.CONFIDENCE_BANDS.items(), key=lambda kv: -kv[1]):
        if p >= thr:
            return name
    return "REJECTED"


@dataclass
class Prediction:
    probability: float
    is_oil: bool
    confidence: str
    contributions: dict          # feature -> signed contribution to the score

    def as_dict(self) -> dict:
        return {
            "oil_probability": round(float(self.probability), 4),
            "is_oil": bool(self.is_oil),
            "confidence": self.confidence,
            "top_evidence": [
                {"feature": k, "contribution": round(float(v), 4)}
                for k, v in sorted(
                    self.contributions.items(), key=lambda kv: -abs(kv[1])
                )[:6]
            ],
        }


class OilClassifier:
    def __init__(self, model=None, scaler=None, importances=None, metrics=None):
        self.model = model
        self.scaler = scaler
        self.importances = importances or {}
        self.metrics = metrics or {}

    # -- training ------------------------------------------------------
    @classmethod
    def train(cls, X, y, test_size=0.25, seed=42, verbose=True):
        from sklearn.ensemble import (
            GradientBoostingClassifier,
            RandomForestClassifier,
            VotingClassifier,
        )
        from sklearn.metrics import (
            accuracy_score,
            average_precision_score,
            confusion_matrix,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler

        X = np.asarray(X, np.float32)
        y = np.asarray(y, np.int64)
        strat = y if len(np.unique(y)) > 1 else None
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=strat
        )

        scaler = StandardScaler().fit(Xtr)
        Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)

        rf = RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        )
        gb = GradientBoostingClassifier(
            n_estimators=250, learning_rate=0.06, max_depth=3,
            subsample=0.85, random_state=seed,
        )
        model = VotingClassifier(
            [("rf", rf), ("gb", gb)], voting="soft", weights=[1.0, 1.0]
        )
        model.fit(Xtr_s, ytr)

        prob = model.predict_proba(Xte_s)[:, 1]
        pred = (prob >= SAR.OIL_PROBABILITY_THRESHOLD).astype(int)
        cm = confusion_matrix(yte, pred, labels=[0, 1])
        metrics = {
            "n_train": int(len(ytr)),
            "n_test": int(len(yte)),
            "n_oil_train": int(ytr.sum()),
            "n_oil_test": int(yte.sum()),
            "accuracy": round(float(accuracy_score(yte, pred)), 4),
            "precision": round(float(precision_score(yte, pred, zero_division=0)), 4),
            "recall": round(float(recall_score(yte, pred, zero_division=0)), 4),
            "f1": round(float(f1_score(yte, pred, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(yte, prob)), 4)
            if len(np.unique(yte)) > 1
            else None,
            "average_precision": round(float(average_precision_score(yte, prob)), 4)
            if len(np.unique(yte)) > 1
            else None,
            "confusion_matrix": {
                "tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
                "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
            },
            "decision_threshold": SAR.OIL_PROBABILITY_THRESHOLD,
            "model": "VotingClassifier(RandomForest-400 + GradientBoosting-250)",
        }

        rf_fitted = model.named_estimators_["rf"]
        imp = dict(
            sorted(
                zip(FEATURE_NAMES, [float(v) for v in rf_fitted.feature_importances_]),
                key=lambda kv: -kv[1],
            )
        )
        if verbose:
            print(json.dumps({k: metrics[k] for k in
                              ("accuracy", "precision", "recall", "f1", "roc_auc",
                               "confusion_matrix")}, indent=2))
        return cls(model, scaler, imp, metrics)

    # -- persistence ------------------------------------------------
    def save(self, path=None, sensor="SAR"):
        import joblib

        path = path or model_path(sensor)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "scaler": self.scaler,
                "importances": self.importances,
                "metrics": self.metrics,
                "sensor": sensor,
                "feature_names": FEATURE_NAMES,
            },
            path,
        )
        metrics_path(sensor).write_text(
            json.dumps(
                {"sensor": sensor, "metrics": self.metrics, "importances": self.importances},
                indent=2,
            ),
            encoding="utf-8",
        )
        return path

    @classmethod
    def load(cls, path=MODEL_PATH):
        import joblib

        path = Path(path)
        if not path.exists():
            return None
        d = joblib.load(path)
        return cls(d["model"], d["scaler"], d.get("importances"), d.get("metrics"))

    # -- inference -------------------------------------------------
    def predict(self, X, threshold=None):
        thr = SAR.OIL_PROBABILITY_THRESHOLD if threshold is None else float(threshold)
        X = np.atleast_2d(np.asarray(X, np.float32))
        probs = self.model.predict_proba(self.scaler.transform(X))[:, 1]
        return [
            Prediction(
                float(p),
                bool(p >= thr),
                confidence_band(float(p)),
                self._contributions(X[i]),
            )
            for i, p in enumerate(probs)
        ]

    def _contributions(self, x):
        """Importance x standardised deviation - a stable first-order 'why'."""
        z = (np.asarray(x, np.float64) - self.scaler.mean_) / np.maximum(
            np.sqrt(self.scaler.var_), 1e-9
        )
        return {
            name: float(self.importances.get(name, 0.0) * z[i])
            for i, name in enumerate(FEATURE_NAMES)
        }


_CACHE: dict = {}


def get_classifier(sensor: str = "SAR") -> OilClassifier | None:
    """Process-wide lazy singleton per sensor. ``None`` if no model file."""
    key = str(sensor).upper()
    if key not in _CACHE:
        clf = OilClassifier.load(model_path(key))
        if clf is None and key != "SAR":
            clf = OilClassifier.load(model_path("SAR"))
            if clf is not None:
                clf.metrics = {
                    **clf.metrics,
                    "sensor_fallback": f"{key} scored with the SAR model",
                }
        _CACHE[key] = clf
    return _CACHE[key]


def reset_cache() -> None:
    _CACHE.clear()


def rule_based_score(feat: dict) -> float:
    """Deterministic physics fallback used when no trained model is present."""
    s = 0.0
    s += 0.30 * np.clip((feat.get("mean_contrast_db", 0) - 2.5) / 6.0, 0, 1)
    s += 0.26 * np.clip((feat.get("border_gradient_db_px", 0) - 0.05) / 0.45, 0, 1)
    s += 0.16 * np.clip((feat.get("elongation", 1) - 1.5) / 8.0, 0, 1)
    s += 0.12 * np.clip((feat.get("complexity", 1) - 1.2) / 2.0, 0, 1)
    s += 0.10 * np.clip((0.85 - feat.get("solidity", 1)) / 0.45, 0, 1)
    s += 0.06 * np.clip((40.0 - feat.get("spreading", 50)) / 40.0, 0, 1)
    return float(np.clip(s, 0.0, 1.0))
