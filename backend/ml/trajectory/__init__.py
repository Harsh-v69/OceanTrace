"""
Vessel trajectory / route-deviation LSTM (POSEatSea).

Ported from ``poseatsea/inference/trajectory.py``. A 2-layer LSTM that predicts
a vessel's next AIS position; ``route_deviation_score`` turns the actual-vs-
predicted trace into a [0, 1] fusion signal. Mauritius-AOI only - ``assess_inputs``
gates every out-of-region window so a meaningless prediction never reaches the
fusion engine.
"""
from backend.ml.trajectory.config import (
    LAT_MAX,
    LAT_MIN,
    LON_MAX,
    LON_MIN,
    SEQ_LEN,
    TRAJ_P90_ERROR_KM,
)
from backend.ml.trajectory.lstm import (
    InputAssessment,
    TrajectoryPrediction,
    assess_inputs,
    build_model,
    clean_trace,
    in_aoi,
    load_model,
    model_card,
    predict_next_position,
    rolling_predictions,
    route_deviation_score,
)

__all__ = [
    "LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX", "SEQ_LEN", "TRAJ_P90_ERROR_KM",
    "InputAssessment", "TrajectoryPrediction", "assess_inputs", "build_model",
    "clean_trace", "in_aoi", "load_model", "model_card", "predict_next_position",
    "rolling_predictions", "route_deviation_score",
]
