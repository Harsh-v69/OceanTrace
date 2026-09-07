"""
Epic 4.2 - basic Mackay/Fay weathering (evaporation + spreading).
"""
from __future__ import annotations

import pytest

from backend.ml.drift.aging import evaporated_fraction, spread_area_km2, weather_slick
from backend.ml.drift.config import WEATHERING
from backend.services import jurisdiction as juris
from backend.tests._helpers import API, auth_header, make_user

MUMBAI = "mumbai-high-confidence"


def test_evaporation_is_monotone_bounded_and_zero_at_start():
    assert evaporated_fraction(0.0) == 0.0
    f6, f24, f48 = (evaporated_fraction(h) for h in (6, 24, 48))
    assert 0.0 < f6 < f24 < f48 <= WEATHERING.MAX_EVAPORATED_FRACTION
    # warmer water evaporates faster
    assert evaporated_fraction(12, temp_c=32) > evaporated_fraction(12, temp_c=18)


def test_fay_area_grows_with_time_and_volume():
    a1 = spread_area_km2(500.0, 6.0)
    a2 = spread_area_km2(500.0, 24.0)
    a3 = spread_area_km2(5000.0, 24.0)
    assert 0.0 < a1 < a2 < a3
    assert spread_area_km2(500.0, 0.0) == 0.0


def test_weather_slick_series_conserves_mass_and_thins():
    w = weather_slick(20.0, (6, 12, 24, 48))
    s = w["series"]
    assert [row["t_h"] for row in s] == [0.0, 6.0, 12.0, 24.0, 48.0]
    v0 = w["initial_volume_m3"]
    assert v0 == pytest.approx(20.0 * 1e6 * WEATHERING.INITIAL_SLICK_THICKNESS_M, rel=1e-6)
    vols = [row["volume_remaining_m3"] for row in s]
    assert vols == sorted(vols, reverse=True)                # never increases
    assert all(0 <= row["volume_remaining_m3"] <= v0 for row in s)
    thick = [row["mean_thickness_mm"] for row in s]
    assert thick[-1] < thick[0]                              # slick thins over time


def test_scenario_metadata_and_dossier_carry_weathering(client, db):
    juris.seed_demo_jurisdictions(db)
    make_user(client, email="nat@example.com", role="NATIONAL")
    h = auth_header(client, "nat@example.com")
    inv_id = client.post(f"{API}/scenarios/{MUMBAI}/run", headers=h).json()["investigation_id"]

    m = client.get(f"{API}/investigations/{inv_id}", headers=h).json()["summary_metrics"]
    wx = m.get("weathering")
    assert wx and wx["series"] and wx["oil_class"] == "medium_crude"
    assert wx["series"][0]["evaporated_fraction"] == 0.0
    assert wx["series"][-1]["evaporated_fraction"] > 0.0

    d = client.get(f"{API}/investigations/{inv_id}/dossier", headers=h).json()
    assert d["forward_forecast"]["weathering"]["series"]
    md = client.get(f"{API}/investigations/{inv_id}/dossier.md", headers=h).text
    assert "Weathering" in md
