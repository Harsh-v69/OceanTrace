"""
Baseline vessel-attribution engine (Stage C core).

Ported from SAMUDRA NETRA ``ml/ais/scoring.py`` (attribution = selected [SN] in
docs/MERGE_ARCHITECTURE.md), plus POSEatSea's ``dwell`` term. Pure numpy,
CPU-only, deterministic.

Primitives:
    spatiotemporal_consistency  - origin<->vessel space-time coincidence
    axis_alignment              - vessel course vs the slick's reverse-drift axis
    closest_point_of_approach   - true CPA to the reconstructed origin (+ cpa_score)
    dwell_fraction              - share of observed time inside the search radius
    score_candidate             - the weighted, normalised, transparent combination

All physical constants + weights live in ``backend.ml.attribution.config``.
The investigation-level ranking wrapper is ``backend.services.attribution``.
"""
from backend.ml.attribution.axis import axis_alignment, slick_axis_from_origin
from backend.ml.attribution.config import ATTRIB, AttributionWeights, assessment_band
from backend.ml.attribution.proximity import (
    closest_point_of_approach,
    cpa_score,
    dwell_fraction,
)
from backend.ml.attribution.score import score_candidate
from backend.ml.attribution.spatiotemporal import spatiotemporal_consistency

__all__ = [
    "ATTRIB",
    "AttributionWeights",
    "assessment_band",
    "spatiotemporal_consistency",
    "axis_alignment",
    "slick_axis_from_origin",
    "closest_point_of_approach",
    "cpa_score",
    "dwell_fraction",
    "score_candidate",
]
