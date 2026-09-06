"""
Deterministic demo scenarios.

Each scenario is a fully reproducible incident - a synthetic SAR scene plus a
set of AIS vessel tracks - that exercises the whole pipeline end to end
(detect -> characterise -> hindcast -> AIS fuse -> jurisdiction -> alert ->
evidence dossier). Fixed seeds mean the same inputs and the same ranking come
back on every run.
"""
from backend.simulator.scenarios import (
    SCENARIOS,
    build_scenario,
    list_scenarios,
    seed_scenario_zones,
)

__all__ = ["SCENARIOS", "build_scenario", "list_scenarios", "seed_scenario_zones"]
