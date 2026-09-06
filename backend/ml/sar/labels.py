"""
Canonical anomaly labels.

The pipeline MUST classify strictly as one of these three strings - nothing
else may leak to an API response or a UI. Per the project rule, a SAR detection
is never asserted as "oil", only as an *oil-like anomaly*.
"""
from __future__ import annotations

OIL_LIKE = "Oil-like anomaly"
LOOKALIKE = "Likely look-alike"
NONE = "No significant anomaly"

CANONICAL_LABELS: tuple[str, str, str] = (OIL_LIKE, LOOKALIKE, NONE)

# Ranking for reducing many per-detection labels to one scene label
# (strongest wins).
_RANK = {OIL_LIKE: 2, LOOKALIKE: 1, NONE: 0}


def label_for(is_oil: bool, has_candidate: bool = True) -> str:
    """Map a per-candidate decision onto the canonical vocabulary."""
    if not has_candidate:
        return NONE
    return OIL_LIKE if is_oil else LOOKALIKE


def scene_label(detection_labels: list[str]) -> str:
    """Reduce per-detection labels to one scene-level label (strongest wins)."""
    if not detection_labels:
        return NONE
    return max(detection_labels, key=lambda label: _RANK.get(label, 0))


def assert_canonical(label: str) -> str:
    if label not in CANONICAL_LABELS:
        raise ValueError(
            f"{label!r} is not a canonical anomaly label; expected one of "
            f"{CANONICAL_LABELS}"
        )
    return label
