"""
AIS behavioural anomaly detection - POSEatSea autoencoder.

Ported from ``poseatsea/inference/ais.py``. An 11-feature undercomplete
autoencoder (11 -> 16 -> 8 -> 4 -> 8 -> 16 -> 11); a ping is flagged when its
mean reconstruction error clears ``AE_THRESHOLD``.

CRITICAL - the pre-trained ``StandardScaler`` is MANDATORY. The model was
trained on standardised features and returns confident nonsense on raw input.
Every scoring path here funnels through ``_reconstruction_errors``, which is the
single place ``scaler.transform`` is called; there is no code path that reaches
the network without it, and ``load_model`` refuses a scaler that is not a
fitted 11-feature transformer.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from backend.ml.ais.config import (
    AE_INPUT_DIM,
    AE_LATENT_DIM,
    AE_PRECISION,
    AE_RECALL,
    AE_THRESHOLD,
    AIS_AE_WEIGHTS,
    AIS_SCALER,
    SEVERITY_CRITICAL_RATIO,
    SEVERITY_HIGH_RATIO,
    resolve_device,
)
from backend.ml.ais.config import FEATURE_ORDER

RAW_COLUMNS = ["mmsi", "timestamp", "latitude", "longitude",
               "speed", "course", "rot", "msg_type", "status", "accuracy"]

DIFF_SOURCES = {
    "course_diff": "course",
    "rot_diff": "rot",
    "speed_diff": "speed",
    "lat_diff": "latitude",
    "long_diff": "longitude",
}


def _torch():
    import torch  # imported lazily so importing this module never needs torch
    import torch.nn as nn

    return torch, nn


def build_autoencoder():
    torch, nn = _torch()

    class Autoencoder(nn.Module):
        """Shapes are fixed by the checkpoint - do not change them."""

        def __init__(self, input_dim=AE_INPUT_DIM, latent_dim=AE_LATENT_DIM):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 16), nn.ReLU(),
                nn.Linear(16, 8), nn.ReLU(),
                nn.Linear(8, latent_dim),
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 8), nn.ReLU(),
                nn.Linear(8, 16), nn.ReLU(),
                nn.Linear(16, input_dim),
            )

        def forward(self, x):
            return self.decoder(self.encoder(x))

    return Autoencoder()


def _validate_scaler(scaler) -> None:
    """Refuse anything that is not a fitted 11-feature StandardScaler."""
    if scaler is None:
        raise ValueError(
            "AIS autoencoder requires its pre-trained StandardScaler - "
            "scoring raw, unscaled features is not permitted."
        )
    for attr in ("mean_", "scale_", "transform"):
        if not hasattr(scaler, attr):
            raise ValueError(f"scaler is missing {attr!r}; it is not a fitted StandardScaler")
    n = int(getattr(scaler, "n_features_in_", AE_INPUT_DIM))
    if n != AE_INPUT_DIM:
        raise ValueError(f"scaler expects {n} features, the autoencoder expects {AE_INPUT_DIM}")


def load_model(weights_path=None, scaler_path=None):
    """Return ``(model, scaler)``. Both are mandatory; the scaler is validated."""
    import joblib

    torch, _ = _torch()
    device = resolve_device()

    model = build_autoencoder()
    model.load_state_dict(torch.load(weights_path or AIS_AE_WEIGHTS, map_location=device),
                          strict=True)
    model.to(device)
    model.eval()

    with warnings.catch_warnings():
        # scaler pickled under scikit-learn 1.6.1; a newer runtime warns but
        # unpickles a StandardScaler correctly.
        warnings.filterwarnings("ignore", message=".*version.*")
        warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.*")
        scaler = joblib.load(scaler_path or AIS_SCALER)

    _validate_scaler(scaler)
    return model, scaler


# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #
def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the five per-vessel difference features, differenced within each MMSI."""
    out = df.copy()
    if "mmsi" not in out.columns:
        out["mmsi"] = 0
    sort_keys = ["mmsi", "timestamp"] if "timestamp" in out.columns else ["mmsi"]
    out = out.sort_values(sort_keys).reset_index(drop=True)

    for target, source in DIFF_SOURCES.items():
        if source not in out.columns:
            raise KeyError(f"column {source!r} is required to derive {target!r}")
        out[target] = out.groupby("mmsi")[source].diff().fillna(0.0)

    out["course_diff"] = ((out["course_diff"] + 180.0) % 360.0) - 180.0    # circular
    return out


def ensure_features(df: pd.DataFrame) -> pd.DataFrame:
    if all(c in df.columns for c in FEATURE_ORDER):
        return df.copy()
    return add_derived_features(df)


def feature_matrix(df: pd.DataFrame) -> np.ndarray:
    missing = [c for c in FEATURE_ORDER if c not in df.columns]
    if missing:
        raise KeyError(f"missing model features: {missing}")
    return df[FEATURE_ORDER].to_numpy(dtype=np.float64)


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
@dataclass
class AnomalyResult:
    is_anomaly: bool
    score: float
    threshold: float = AE_THRESHOLD
    per_feature_error: Optional[Dict[str, float]] = None

    @property
    def severity(self) -> str:
        if not self.is_anomaly:
            return "normal"
        ratio = self.score / self.threshold
        if ratio >= SEVERITY_CRITICAL_RATIO:
            return "critical"
        if ratio >= SEVERITY_HIGH_RATIO:
            return "high"
        return "elevated"

    def top_contributors(self, k: int = 3) -> List[tuple]:
        if not self.per_feature_error:
            return []
        return sorted(self.per_feature_error.items(), key=lambda kv: kv[1], reverse=True)[:k]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "is_anomaly": self.is_anomaly,
            "severity": self.severity,
            "score": round(self.score, 6),
            "threshold": self.threshold,
            "top_contributors": [
                {"feature": f, "error": round(e, 4)} for f, e in self.top_contributors()
            ],
        }


def _reconstruction_errors(model, scaler, x_raw: np.ndarray):
    """The ONLY place features reach the network. Scaling is not optional here."""
    _validate_scaler(scaler)
    torch, _ = _torch()
    x_scaled = scaler.transform(np.asarray(x_raw, dtype=np.float64))
    with torch.no_grad():
        tensor = torch.tensor(x_scaled, dtype=torch.float32).to(resolve_device())
        recon = model(tensor).cpu().numpy()
    sq = (x_scaled - recon) ** 2
    return sq.mean(axis=1), sq


def score_row(model, scaler, row: Dict[str, float]) -> AnomalyResult:
    """Score one ping given as a dict of the 11 features."""
    missing = [f for f in FEATURE_ORDER if f not in row]
    if missing:
        raise KeyError(f"missing model features: {missing}")
    x = np.array([[float(row[f]) for f in FEATURE_ORDER]], dtype=np.float64)
    means, sq = _reconstruction_errors(model, scaler, x)
    return AnomalyResult(
        is_anomaly=bool(means[0] >= AE_THRESHOLD),
        score=float(means[0]),
        per_feature_error={f: float(sq[0, i]) for i, f in enumerate(FEATURE_ORDER)},
    )


def score_frame(model, scaler, df: pd.DataFrame) -> pd.DataFrame:
    """Score every ping. Adds anomaly_score / is_anomaly / severity / top_feature."""
    prepared = ensure_features(df)
    means, sq = _reconstruction_errors(model, scaler, feature_matrix(prepared))
    out = prepared.copy()
    out["anomaly_score"] = means
    out["is_anomaly"] = means >= AE_THRESHOLD
    out["severity"] = [AnomalyResult(bool(s >= AE_THRESHOLD), float(s)).severity for s in means]
    out["top_feature"] = [FEATURE_ORDER[i] for i in sq.argmax(axis=1)]
    return out


def vessel_rollup(scored: pd.DataFrame) -> pd.DataFrame:
    if scored.empty:
        return pd.DataFrame()
    grouped = scored.groupby("mmsi").agg(
        pings=("anomaly_score", "size"),
        flagged=("is_anomaly", "sum"),
        max_score=("anomaly_score", "max"),
        mean_score=("anomaly_score", "mean"),
    ).reset_index()
    grouped["flagged_pct"] = (grouped["flagged"] / grouped["pings"] * 100).round(1)
    return grouped.sort_values("max_score", ascending=False).reset_index(drop=True)


def anomaly_score_0_1(max_reconstruction_error: float, threshold: float = AE_THRESHOLD) -> float:
    """Saturating map of the peak reconstruction error onto [0, 1] (POSEatSea)."""
    if max_reconstruction_error is None or max_reconstruction_error <= 0:
        return 0.0
    ratio = float(max_reconstruction_error) / float(threshold)
    return float(np.clip(np.log1p(ratio) / np.log1p(6.0), 0.0, 1.0))


def model_card() -> Dict[str, Any]:
    return {
        "precision": AE_PRECISION,
        "recall": AE_RECALL,
        "threshold": AE_THRESHOLD,
        "reading": (
            f"About {AE_PRECISION:.0%} of flags are genuine anomalies, but the model "
            f"catches only {AE_RECALL:.0%} of them. Treat a flag as strong evidence "
            f"and the absence of one as no evidence either way."
        ),
    }
