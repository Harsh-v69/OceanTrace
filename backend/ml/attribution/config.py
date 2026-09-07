"""
Attribution scoring constants - the single configuration layer.

Ported / adapted from OceanTrace ``config.ATTRIB`` (attribution engine =
selected [SN] in docs/MERGE_ARCHITECTURE.md). Phase 5 ships the *baseline*
physical / spatiotemporal criteria; the behavioural criteria (AIS blackout,
speed anomaly, route deviation) fold in with the POSEatSea autoencoder + LSTM
in a later phase, as extra weighted components.

Every score component is normalised to [0, 1] by its primitive BEFORE being
weighted here. The weights sum to 1.0.
"""
from __future__ import annotations

from dataclasses import dataclass, fields


class _WeightsMixin:
    def as_dict(self) -> dict[str, float]:
        return {f.name: float(getattr(self, f.name)) for f in fields(self)}

    def normalized(self) -> dict[str, float]:
        d = self.as_dict()
        total = sum(d.values()) or 1.0
        return {k: v / total for k, v in d.items()}


@dataclass(frozen=True)
class AttributionWeights(_WeightsMixin):
    """Phase-5 BASELINE weights (physical + spatiotemporal only)."""

    spatiotemporal: float = 0.45   # origin<->vessel space-time coincidence in the window
    axis_alignment: float = 0.30   # vessel course vs the slick's reverse-drift axis
    proximity: float = 0.15        # closest point of approach to the reconstructed origin
    dwell: float = 0.10            # share of observed time spent inside the search radius


@dataclass(frozen=True)
class FusionWeights(_WeightsMixin):
    """
    UNIFIED FUSION weights (Epic 2.3 - the exact operating spec). Physical + AIS
    + behavioural evidence; every component is normalised to [0, 1] before
    weighting and the weights are renormalised over whatever components are
    available for a given vessel. Sum = 1.00.

    Note: the AIS autoencoder (``ais_anomaly``) is still computed and shown as
    transparent evidence in every ranking / dossier, but it is **not weighted**
    here - it flags a vessel for review, it does not move the score.
    """

    # -- physical evidence --------------------------------------------
    spatiotemporal: float = 0.30   # hindcast-origin distance/time consistency
    axis_alignment: float = 0.18   # slick reverse-drift axis vs vessel course
    # -- AIS evidence -----------------------------------------------
    proximity: float = 0.14        # closest point of approach (CPA) to the origin
    dwell: float = 0.10            # share of the vessel's observed time inside the search radius
    blackout: float = 0.10         # AIS dark period over the release window
    # -- behavioural evidence ------------------------------------------
    route_deviation: float = 0.09  # LSTM actual-vs-predicted track departure
    vessel_prior: float = 0.09     # a-priori discharge likelihood by vessel type


class ATTRIB:
    """Gates, kernels and reporting bands."""

    # default weights (dict mirror of AttributionWeights, for quick overrides)
    WEIGHTS = AttributionWeights().as_dict()

    # -- spatial / temporal gate --------------------------------------
    SEARCH_RADIUS_KM = 25.0        # spatial gate around the reconstructed origin
    TIME_PAD_H = 2.0              # release-window padding on each side

    # -- spatiotemporal consistency kernel -----------------------------
    ST_KERNEL_SIGMA_KM = 7.0     # Gaussian half-width for space-time overlap
    ST_PROXIMITY_SIGMA_KM = 5.0  # direct proximity term to the nearest origin parcel
    ST_CONSISTENCY_HALF_WIN_H = 1.5

    # -- axis alignment -----------------------------------------------
    AXIS_TOLERANCE_DEG = 55.0    # course within this of the slick axis scores > 0

    # -- proximity / CPA -------------------------------------------
    CPA_DECAY_KM = 8.0          # e-folding distance for the CPA score (unused by default)

    # -- proximity gate: a vessel with no space-time signal cannot be the source
    PROXIMITY_GATE_KNEE = 0.12  # score below which the total is scaled toward 0
    PROXIMITY_GATE_ENABLED = True

    # -- fusion (Phase 6) --------------------------------------------
    BLACKOUT_FULL_SCORE_MIN = 120.0   # a 2 h+ AIS dark period over the window scores 1.0
    AIS_ANOMALY_NEAR_RADIUS_KM = SEARCH_RADIUS_KM   # AE flags only count if near the slick
    VESSEL_PRIOR_DEFAULT = 0.30

    # -- reporting ------------------------------------------------
    MIN_TRACK_POINTS = 2
    MIN_SCORE_TO_REPORT = 8.0    # 0-100 scale
    BANDS = {                     # 0-100 score -> assessment band
        "PRIME_SUSPECT": 70.0,
        "PERSON_OF_INTEREST": 45.0,
        "WEAK_LEAD": 20.0,
    }

    # a-priori discharge likelihood by vessel-type group (kept for the fuller
    # fusion later; NOT used by the baseline weighted sum)
    TYPE_PRIOR = {
        "TANKER": 1.00, "CARGO": 0.72, "BULK": 0.70, "CONTAINER": 0.62,
        "PASSENGER": 0.35, "FISHING": 0.30, "TUG": 0.40, "HSC": 0.25,
        "PLEASURE": 0.15, "MILITARY": 0.20, "OTHER": 0.30, "UNKNOWN": 0.30,
    }


def assessment_band(score_0_100: float) -> str:
    for name, thr in sorted(ATTRIB.BANDS.items(), key=lambda kv: -kv[1]):
        if score_0_100 >= thr:
            return name
    return "CLEARED" if score_0_100 < ATTRIB.MIN_SCORE_TO_REPORT else "BACKGROUND_TRAFFIC"
