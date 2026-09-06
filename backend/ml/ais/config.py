"""
POSEatSea AIS anomaly autoencoder - constants.

Ported verbatim from ``poseatsea/config.py``. Every value here is fixed by the
committed checkpoint (``models/ais_phase1_autoencoder.pth``) and its pickled
``StandardScaler`` (``models/ais_phase1_scaler.joblib``) - do NOT change them.
"""
from __future__ import annotations

import os
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent / "models"
AIS_AE_WEIGHTS = MODELS_DIR / "ais_phase1_autoencoder.pth"
AIS_SCALER = MODELS_DIR / "ais_phase1_scaler.joblib"

# The 11 features, in the exact positional order the autoencoder was trained on.
FEATURE_ORDER = [
    "speed", "course", "rot", "msg_type", "status", "accuracy",
    "course_diff", "rot_diff", "speed_diff", "lat_diff", "long_diff",
]
AE_INPUT_DIM = 11
AE_LATENT_DIM = 4
AE_THRESHOLD = 1.104481          # reconstruction-error cut, set at training time

# Held-out performance - deliberately precision-heavy.
AE_F1 = 0.567
AE_PRECISION = 0.930
AE_RECALL = 0.408

# Ping fields not in the canonical track schema get these Class-A defaults when
# the raw feed did not carry them.
DEFAULT_MSG_TYPE = 1            # Class-A position report
DEFAULT_ACCURACY = 1           # high-accuracy position flag

# Severity bands over score / AE_THRESHOLD.
SEVERITY_HIGH_RATIO = 2.0
SEVERITY_CRITICAL_RATIO = 5.0


def resolve_device():
    import torch

    forced = os.getenv("POSEATSEA_DEVICE")
    return torch.device(forced) if forced else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
