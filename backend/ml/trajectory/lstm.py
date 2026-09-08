"""
Vessel trajectory prediction - POSEatSea LSTM.

Ported from ``poseatsea/inference/trajectory.py``. A 2-layer LSTM (6 -> 128) maps
a vessel's last 8 AIS pings to its next position; ``rolling_predictions`` turns
that into a continuous actual-vs-predicted deviation trace.

Operating envelope (enforced by ``assess_inputs``, not left to be discovered):
  * geographic - outside the Mauritius AOI the normalisation saturates and the
    prediction is meaningless -> ``usable = False``;
  * kinematic - reliable near 60 s cadence on the NE-SW lane; NW-SE headings and
    slow cadence are marked ``degraded``.

The route-deviation score returned to the fusion engine is 0 whenever the model
is not usable, so an out-of-region vessel is never penalised for a bad guess.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from backend.ml.trajectory.config import (
    AOI_HARD_GATE,
    COVERAGE_GAP_FACTOR,
    DEVIATION_SCORE_SCALE_KM,
    LAT_MAX,
    LAT_MIN,
    LON_MAX,
    LON_MIN,
    SEQ_LEN,
    SPEED_MAX,
    TRAIN_LAT_SPAN,
    TRAIN_LON_SPAN,
    TRAJ_HIDDEN_DIM,
    TRAJ_INPUT_DIM,
    TRAJ_MEAN_ERROR_KM,
    TRAJ_MEDIAN_ERROR_KM,
    TRAJ_NOMINAL_INTERVAL_S,
    TRAJ_NUM_LAYERS,
    TRAJ_P90_ERROR_KM,
    TRAJ_RELIABLE_COURSE_BANDS,
    TRAJECTORY_WEIGHTS,
    resolve_device,
)

_AOI_MID = ((LAT_MIN + LAT_MAX) / 2.0, (LON_MIN + LON_MAX) / 2.0)

HISTORY_COLUMNS = ["latitude", "longitude", "speed", "course", "rot"]


def _torch():
    import torch
    import torch.nn as nn

    return torch, nn


def build_model():
    torch, nn = _torch()

    class LSTMTrajectoryModel(nn.Module):
        """Shapes are fixed by the checkpoint - do not change them."""

        def __init__(self, input_dim=TRAJ_INPUT_DIM, hidden_dim=TRAJ_HIDDEN_DIM,
                     num_layers=TRAJ_NUM_LAYERS):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                                batch_first=True, dropout=0.1)
            self.head = nn.Linear(hidden_dim, 2)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out[:, -1, :])

    return LSTMTrajectoryModel()


def load_model(weights_path=None):
    torch, _ = _torch()
    device = resolve_device()
    model = build_model()
    model.load_state_dict(torch.load(weights_path or TRAJECTORY_WEIGHTS, map_location=device),
                          strict=True)
    model.to(device)
    model.eval()
    return model


# --------------------------------------------------------------------------- #
# Normalisation
#
# The model was trained on coordinates scaled by the fixed Mauritius AOI box.
# Epic 2.2 keeps the same *span* but recentres it on the analysed window so the
# LSTM can run anywhere. ``frame = (lat0, lon0, lat_span, lon_span)`` maps
# ``lat0 -> 0.5``; with the AOI-centre frame this is bit-identical to the
# original ``(x - MIN) / (MAX - MIN)`` mapping.
# --------------------------------------------------------------------------- #
def _aoi_frame() -> tuple[float, float, float, float]:
    return (_AOI_MID[0], _AOI_MID[1], TRAIN_LAT_SPAN, TRAIN_LON_SPAN)


def frame_for(seq: np.ndarray) -> tuple[float, float, float, float]:
    """Normalisation frame for a window: the fixed AOI frame when the window's
    centroid is inside the training AOI, otherwise the same span recentred on it."""
    lat0 = float(np.mean(seq[..., 0]))
    lon0 = float(np.mean(seq[..., 1]))
    if in_aoi(lat0, lon0):
        return _aoi_frame()
    return (lat0, lon0, TRAIN_LAT_SPAN, TRAIN_LON_SPAN)


def normalize_features(seq: np.ndarray, frame: tuple | None = None) -> np.ndarray:
    """(..., 5) of (lat, lon, speed, course, rot) -> (..., 6) model input."""
    lat0, lon0, lat_span, lon_span = frame or _aoi_frame()
    out = seq.copy().astype(np.float32)
    out[..., 0] = (seq[..., 0] - lat0) / (lat_span + 1e-9) + 0.5
    out[..., 1] = (seq[..., 1] - lon0) / (lon_span + 1e-9) + 0.5
    out[..., 2] = np.clip(seq[..., 2], 0, SPEED_MAX) / SPEED_MAX
    course_rad = np.radians(seq[..., 3])
    sin_c, cos_c = np.sin(course_rad), np.cos(course_rad)
    rot_norm = np.clip(seq[..., 4], -128, 128) / 128.0
    return np.concatenate(
        [out[..., :3], sin_c[..., None], cos_c[..., None], rot_norm[..., None]], axis=-1
    )


def denorm_latlon(lat_norm: float, lon_norm: float, frame: tuple | None = None):
    lat0, lon0, lat_span, lon_span = frame or _aoi_frame()
    return (float(lat_norm - 0.5) * lat_span + lat0,
            float(lon_norm - 0.5) * lon_span + lon0)


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return float(2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1))))


def bearing_deg(lat1, lon1, lat2, lon2) -> float:
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return float((np.degrees(np.arctan2(y, x)) + 360.0) % 360.0)


def in_aoi(lat: float, lon: float) -> bool:
    return LAT_MIN <= lat <= LAT_MAX and LON_MIN <= lon <= LON_MAX


def _course_is_reliable(course: float) -> bool:
    c = course % 360.0
    return any(lo <= c <= hi for lo, hi in TRAJ_RELIABLE_COURSE_BANDS)


# --------------------------------------------------------------------------- #
# Input validation
# --------------------------------------------------------------------------- #
@dataclass
class InputAssessment:
    usable: bool
    warnings: List[str] = field(default_factory=list)
    blockers: List[str] = field(default_factory=list)
    confidence: str = "nominal"          # nominal | degraded | unreliable

    def as_dict(self) -> Dict[str, Any]:
        return {"usable": self.usable, "confidence": self.confidence,
                "warnings": self.warnings, "blockers": self.blockers}


def assess_inputs(history: pd.DataFrame) -> InputAssessment:
    blockers: List[str] = []
    warnings_: List[str] = []

    if len(history) != SEQ_LEN:
        blockers.append(f"Model consumes exactly {SEQ_LEN} pings; received {len(history)}.")
        return InputAssessment(False, warnings_, blockers, "unreliable")

    missing = [c for c in HISTORY_COLUMNS if c not in history.columns]
    if missing:
        blockers.append(f"History is missing columns: {missing}.")
        return InputAssessment(False, warnings_, blockers, "unreliable")

    outside = [
        (float(r.latitude), float(r.longitude))
        for r in history.itertuples() if not in_aoi(float(r.latitude), float(r.longitude))
    ]
    confidence = "nominal"
    if outside:
        msg = (
            f"{len(outside)} of {SEQ_LEN} pings are outside the Mauritius AOI the model was "
            f"trained on; the window is renormalised locally and the route-deviation score "
            f"is an unvalidated extrapolation (published 0.37 km error does not apply here)."
        )
        if AOI_HARD_GATE:
            blockers.append(msg)
            return InputAssessment(False, warnings_, blockers, "unreliable")
        warnings_.append(msg)
        confidence = "degraded"
    if "timestamp" in history.columns:
        ts = pd.to_datetime(history["timestamp"])
        gaps = ts.diff().dt.total_seconds().dropna()
        if len(gaps) and float(gaps.median()) > 2.5 * TRAJ_NOMINAL_INTERVAL_S:
            warnings_.append(
                f"Median ping gap {float(gaps.median()):.0f}s vs the ~{TRAJ_NOMINAL_INTERVAL_S:.0f}s "
                f"cadence this model was trained on; step size will be understated."
            )
            confidence = "degraded"

    mean_course = float(history["course"].astype(float).iloc[-3:].mean())
    if not _course_is_reliable(mean_course):
        warnings_.append(
            f"Recent heading ({mean_course:.0f} deg) is outside the NE-SW lane the model "
            f"learned well; bearing error is substantially higher on this axis."
        )
        confidence = "degraded"

    if float(history["speed"].astype(float).mean()) < 0.5:
        warnings_.append("Vessel effectively stationary; next-position prediction is uninformative.")
        confidence = "degraded"

    if float(history["speed"].astype(float).max()) > SPEED_MAX:
        warnings_.append(f"Speed exceeds the {SPEED_MAX} kn training ceiling and is clipped.")

    return InputAssessment(not blockers, warnings_, blockers, confidence)


# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #
@dataclass
class TrajectoryPrediction:
    predicted_lat: float
    predicted_lon: float
    last_lat: float
    last_lon: float
    assessment: InputAssessment
    actual_lat: Optional[float] = None
    actual_lon: Optional[float] = None

    @property
    def step_km(self) -> float:
        return haversine_km(self.last_lat, self.last_lon, self.predicted_lat, self.predicted_lon)

    @property
    def predicted_bearing(self) -> float:
        return bearing_deg(self.last_lat, self.last_lon, self.predicted_lat, self.predicted_lon)

    @property
    def deviation_km(self) -> Optional[float]:
        if self.actual_lat is None or self.actual_lon is None:
            return None
        return haversine_km(self.predicted_lat, self.predicted_lon, self.actual_lat, self.actual_lon)

    def deviation_verdict(self) -> Optional[str]:
        dev = self.deviation_km
        if dev is None:
            return None
        if dev <= TRAJ_MEDIAN_ERROR_KM:
            return f"Within the model's median error ({TRAJ_MEDIAN_ERROR_KM} km) - movement as expected."
        if dev <= TRAJ_P90_ERROR_KM:
            return f"Within the model's 90th-percentile error ({TRAJ_P90_ERROR_KM} km) - unremarkable."
        if dev <= 3 * TRAJ_P90_ERROR_KM:
            return "Above the model's usual error band - the vessel deviated from its expected track."
        return "Far outside the model's error band - a sharp, unmodelled manoeuvre."

    def as_dict(self) -> Dict[str, Any]:
        return {
            "predicted": {"lat": self.predicted_lat, "lon": self.predicted_lon},
            "last_known": {"lat": self.last_lat, "lon": self.last_lon},
            "step_km": round(self.step_km, 4),
            "predicted_bearing_deg": round(self.predicted_bearing, 1),
            "deviation_km": round(self.deviation_km, 4) if self.deviation_km is not None else None,
            "deviation_verdict": self.deviation_verdict(),
            "assessment": self.assessment.as_dict(),
        }


def predict_next_position(model, history: pd.DataFrame,
                          actual_next: Optional[Dict[str, float]] = None,
                          strict: bool = True) -> TrajectoryPrediction:
    torch, _ = _torch()
    assessment = assess_inputs(history)
    if strict and not assessment.usable:
        raise ValueError(" ".join(assessment.blockers))

    seq = history[HISTORY_COLUMNS].to_numpy(dtype=np.float64)[None, ...]
    frame = frame_for(seq[0])
    x = torch.tensor(normalize_features(seq, frame), dtype=torch.float32).to(resolve_device())
    with torch.no_grad():
        pred = model(x).cpu().numpy()
    lat, lon = denorm_latlon(pred[0, 0], pred[0, 1], frame)

    last = history.iloc[-1]
    return TrajectoryPrediction(
        predicted_lat=lat, predicted_lon=lon,
        last_lat=float(last["latitude"]), last_lon=float(last["longitude"]),
        assessment=assessment,
        actual_lat=actual_next.get("latitude") if actual_next else None,
        actual_lon=actual_next.get("longitude") if actual_next else None,
    )


def rolling_predictions(model, track: pd.DataFrame, stride: int = 1) -> pd.DataFrame:
    """Walk an 8-ping window along a track; predicted vs actual next position.

    Every window is normalised in its own recentred frame (``frame_for``) and
    the whole batch is pushed through the LSTM in a single forward pass, so the
    cost is ~O(1) torch calls regardless of track length.
    """
    torch, _ = _torch()
    n = len(track)
    if n < SEQ_LEN + 1:
        return pd.DataFrame()

    arr = track[HISTORY_COLUMNS].to_numpy(dtype=np.float64)          # (n, 5)
    starts = np.arange(0, n - SEQ_LEN, max(stride, 1))
    if starts.size == 0:
        return pd.DataFrame()
    windows = np.stack([arr[s:s + SEQ_LEN] for s in starts])         # (W, 8, 5)
    truth = arr[starts + SEQ_LEN]                                    # (W, 5)

    frames = [frame_for(w) for w in windows]
    norm = np.stack([normalize_features(windows[i], frames[i]) for i in range(len(starts))])
    with torch.no_grad():
        raw = model(torch.tensor(norm, dtype=torch.float32).to(resolve_device())).cpu().numpy()

    ts_all = None
    if "timestamp" in track.columns:
        ts_all = pd.to_datetime(track["timestamp"]).astype("int64").to_numpy() / 1e9  # seconds

    rows = []
    for i, s in enumerate(starts):
        lat_p, lon_p = denorm_latlon(raw[i, 0], raw[i, 1], frames[i])
        centroid = (float(np.mean(windows[i][:, 0])), float(np.mean(windows[i][:, 1])))
        conf = "nominal" if in_aoi(*centroid) else "degraded"
        gap_s = cadence_s = float("nan")
        coverage_gap = False
        if ts_all is not None:
            w_ts = ts_all[s:s + SEQ_LEN]
            cadence_s = float(np.median(np.diff(w_ts))) if SEQ_LEN > 1 else float("nan")
            gap_s = float(ts_all[s + SEQ_LEN] - w_ts[-1])
            if np.isfinite(cadence_s) and cadence_s > 0:
                coverage_gap = gap_s > COVERAGE_GAP_FACTOR * max(cadence_s, 5.0)
                if cadence_s > 2.5 * TRAJ_NOMINAL_INTERVAL_S:
                    conf = "degraded"
        rows.append({
            "index": int(s + SEQ_LEN),
            "actual_lat": float(truth[i, 0]), "actual_lon": float(truth[i, 1]),
            "pred_lat": lat_p, "pred_lon": lon_p,
            "deviation_km": haversine_km(lat_p, lon_p, float(truth[i, 0]), float(truth[i, 1])),
            "confidence": conf,
            "gap_s": gap_s, "cadence_s": cadence_s, "coverage_gap": coverage_gap,
        })
    return pd.DataFrame(rows)


def clean_trace(trace: pd.DataFrame) -> pd.DataFrame:
    if trace.empty or "coverage_gap" not in trace.columns:
        return trace
    return trace[~trace["coverage_gap"]]


# --------------------------------------------------------------------------- #
# Fusion signal: route-deviation score in [0, 1]
# --------------------------------------------------------------------------- #
def route_deviation_score(model, track_df: pd.DataFrame) -> tuple[float, Dict[str, Any]]:
    """Peak actual-vs-predicted deviation over the track, mapped to [0, 1].

    Returns 0 when the model is not usable on this track (too few pings, out of
    AOI): an out-of-region vessel is never penalised for a meaningless guess.
    """
    if track_df is None or len(track_df) < SEQ_LEN + 1:
        return 0.0, {"usable": False, "finding": "fewer than 9 pings - LSTM needs an 8-ping window plus a truth ping"}

    trace = rolling_predictions(model, track_df)      # batched - stride 1 is cheap
    if trace.empty:
        return 0.0, {"usable": False, "finding": "no window passed the LSTM operating envelope (likely out of the Mauritius AOI)"}

    clean = clean_trace(trace)
    used = clean if not clean.empty else trace
    dev = used["deviation_km"].astype(float)
    max_dev = float(dev.max())
    median_dev = float(dev.median())
    score = float(np.clip(max_dev / DEVIATION_SCORE_SCALE_KM, 0.0, 1.0))
    degraded = bool((used["confidence"] == "degraded").mean() > 0.5)
    in_region = bool(in_aoi(float(track_df["latitude"].mean()), float(track_df["longitude"].mean())))
    detail = {
        "usable": True,
        "aoi": in_region,
        "windows_scored": int(len(used)),
        "max_deviation_km": round(max_dev, 3),
        "median_deviation_km": round(median_dev, 3),
        "p90_reference_km": TRAJ_P90_ERROR_KM,
        "confidence": "nominal" if (in_region and not degraded) else "degraded",
        "coverage_gaps_excluded": int(len(trace) - len(clean)),
        "finding": (f"Peak departure from the predicted track was {max_dev:.2f} km "
                    f"(model p90 held-out error {TRAJ_P90_ERROR_KM} km)."
                    + ("" if in_region else
                       " Outside the Mauritius training AOI - locally renormalised, "
                       "treat as an indicative extrapolation.")),
    }
    if not in_region:
        detail["caveat"] = ("Route-deviation extrapolated outside the model's training "
                            "region; the 0.37 km published accuracy does not apply.")

    # Surface the model's predicted-vs-actual next-position path so the UI can
    # draw the predicted vessel route (dashed). Nothing here changes the model -
    # these are the positions rolling_predictions() already computed.
    _pp = used[["pred_lat", "pred_lon", "actual_lat", "actual_lon", "deviation_km"]].to_numpy(float)
    _step = max(1, len(_pp) // 40)
    detail["predicted_path"] = [
        {"lat": round(float(r[0]), 5), "lon": round(float(r[1]), 5),
         "actual_lat": round(float(r[2]), 5), "actual_lon": round(float(r[3]), 5),
         "deviation_km": round(float(r[4]), 3)}
        for r in _pp[::_step]
    ]
    return score, detail


def model_card() -> Dict[str, Any]:
    return {
        "mean_error_km": TRAJ_MEAN_ERROR_KM,
        "median_error_km": TRAJ_MEDIAN_ERROR_KM,
        "p90_error_km": TRAJ_P90_ERROR_KM,
        "sequence_length": SEQ_LEN,
        "aoi": {"lat": [LAT_MIN, LAT_MAX], "lon": [LON_MIN, LON_MAX]},
        "reading": (
            f"On unseen vessels, next-position error averages {TRAJ_MEAN_ERROR_KM} km "
            f"(median {TRAJ_MEDIAN_ERROR_KM} km). Valid only inside the Mauritius AOI."
        ),
    }
