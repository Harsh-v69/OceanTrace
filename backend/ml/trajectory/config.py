"""
POSEatSea trajectory LSTM - constants.

Ported verbatim from ``poseatsea/config.py``. Fixed by the committed checkpoint
(``models/trajectory_lstm_baseline.pth``). The normalisation is anchored to the
Mauritius AOI + July-2020 speed distribution the model was trained on; the AOI
gate in ``assess_inputs`` is what keeps a meaningless out-of-region prediction
from ever reaching the fusion engine.
"""
from __future__ import annotations

import os
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent / "models"
TRAJECTORY_WEIGHTS = MODELS_DIR / "trajectory_lstm_baseline.pth"

# -- Mauritius AOI bounding box (model normalisation anchor) -------------
LAT_MIN = -20.565386666666665
LAT_MAX = -20.05773333333333
LON_MIN = 57.725333333333325
LON_MAX = 58.37872
SPEED_MAX = 19.3                 # 99.9th pct of training speed; input clip ceiling
SEQ_LEN = 8                      # the model consumes exactly 8 pings

TRAJ_INPUT_DIM = 6
TRAJ_HIDDEN_DIM = 128
TRAJ_NUM_LAYERS = 2

# Published held-out accuracy (surfaced so operators can calibrate trust).
TRAJ_MEAN_ERROR_KM = 0.37
TRAJ_MEDIAN_ERROR_KM = 0.19
TRAJ_P90_ERROR_KM = 0.63

# Empirically probed operating envelope (docs/MODEL_NOTES).
TRAJ_NOMINAL_INTERVAL_S = 60.0
TRAJ_RELIABLE_COURSE_BANDS = ((200.0, 290.0), (20.0, 110.0))

# A truth ping this many x the window cadence later is a reception gap, not a
# manoeuvre - excluded from the deviation trace before it is quoted.
COVERAGE_GAP_FACTOR = 4.0

# route-deviation -> [0, 1]: scaled against 5x the model's p90 held-out error.
DEVIATION_SCORE_SCALE_KM = 5.0 * TRAJ_P90_ERROR_KM


def resolve_device():
    import torch

    forced = os.getenv("POSEATSEA_DEVICE")
    return torch.device(forced) if forced else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
