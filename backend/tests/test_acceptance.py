"""
Phase 9 - the final 24-point acceptance test, run as part of the suite.

The checklist logic lives in ``scripts/acceptance.py`` (also runnable on its own
for a human-readable report). Here we just assert every checkpoint passed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


@pytest.fixture(scope="module")
def acceptance():
    from scripts.acceptance import run_acceptance

    results, timings = run_acceptance()
    return results, timings


def test_all_24_checkpoints_pass(acceptance):
    results, _ = acceptance
    assert len(results) == 24, f"expected 24 checkpoints, ran {len(results)}"
    failed = [(n, label, detail) for n, label, ok, detail in results if not ok]
    assert not failed, "acceptance failures:\n" + "\n".join(
        f"  {n:>2}. {label} - {detail}" for n, label, detail in failed
    )


def test_stage_timings_present_and_bounded(acceptance):
    _, timings = acceptance
    for key in ("satellite_ingest", "preprocessing", "detection", "characterization",
                "hindcast_forecast", "ais_correlation", "attribution_fusion",
                "jurisdiction", "alert", "total_ms"):
        assert key in timings, f"missing stage timing {key!r}"
    # a single investigation on one CPU core must stay well under a minute
    assert 0 < timings["total_ms"] < 60_000, timings["total_ms"]
