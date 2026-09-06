"""
AIS ingestion + intelligence.

* ``schema`` / ``tracks`` - pandas-free track processing (Phase 5): timestamp
  normalisation, duplicate / invalid-coordinate / impossible-speed removal,
  segmentation on transmission gaps.
* ``anomaly`` - the POSEatSea autoencoder (Phase 6). Its pre-trained
  ``StandardScaler`` is MANDATORY; ``anomaly.load_model`` validates it and every
  scoring path funnels through ``anomaly._reconstruction_errors``, the single
  place ``scaler.transform`` is called.
"""
from backend.ml.ais.anomaly import (
    AE_THRESHOLD,
    AnomalyResult,
    anomaly_score_0_1,
    load_model,
    score_frame,
    score_row,
)
from backend.ml.ais.schema import (
    CANONICAL_COLUMNS,
    NAV_STATUS,
    flag_from_mmsi,
    nav_status_label,
    type_group,
)
from backend.ml.ais.tracks import (
    MAX_PLAUSIBLE_SOG_KN,
    VesselTrack,
    build_tracks,
    clean_records,
    spatial_temporal_gate,
)

__all__ = [
    "CANONICAL_COLUMNS",
    "NAV_STATUS",
    "flag_from_mmsi",
    "nav_status_label",
    "type_group",
    "MAX_PLAUSIBLE_SOG_KN",
    "VesselTrack",
    "build_tracks",
    "clean_records",
    "spatial_temporal_gate",
    "AE_THRESHOLD",
    "AnomalyResult",
    "anomaly_score_0_1",
    "load_model",
    "score_frame",
    "score_row",
]
