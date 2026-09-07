"""
Epic 2.1 - bundled Indian coastline + RK4 land collision wired into the pipeline.
"""
from __future__ import annotations

import pytest

from backend.services import drift as drift_svc
from backend.services import jurisdiction as juris
from backend.tests._helpers import API, auth_header, make_user

MUMBAI = "mumbai-high-confidence"
AMBIGUOUS = "ambiguous-drift"
LOOKALIKE = "lookalike-darkpatch"


def test_simplified_coastline_loads_as_a_landmask():
    lm = drift_svc.load_indian_coastline()
    assert lm is not None and lm.any
    # a point well inland is land; a point in the Arabian Sea is not
    import numpy as np
    assert bool(lm.is_land(np.array([22.0]), np.array([78.0]))[0]) is True
    assert bool(lm.is_land(np.array([15.0]), np.array([68.0]))[0]) is False


@pytest.fixture
def nat(client, db):
    juris.seed_demo_jurisdictions(db)
    make_user(client, email="nat@example.com", role="NATIONAL")
    return auth_header(client, "nat@example.com")


def test_mumbai_scenario_beaches_with_eta_and_contact_point(client, nat):
    inv_id = client.post(f"{API}/scenarios/{MUMBAI}/run", headers=nat).json()["investigation_id"]
    ci = client.get(f"{API}/investigations/{inv_id}",
                    headers=nat).json()["summary_metrics"]["forecast"]["coastal_impact"]
    assert ci["will_beach"] is True
    assert 0.0 < ci["first_contact_eta_h"] <= 48.0
    lat, lon = ci["first_contact_point"]
    assert 15.0 < lat < 22.0 and 70.0 < lon < 74.0        # on / near the Maharashtra coast
    assert ci["contact_points"]
    assert 0.0 < ci["fraction_beached"] <= 1.0


def test_offshore_scenario_does_not_beach(client, nat):
    inv_id = client.post(f"{API}/scenarios/{AMBIGUOUS}/run", headers=nat).json()["investigation_id"]
    ci = client.get(f"{API}/investigations/{inv_id}",
                    headers=nat).json()["summary_metrics"]["forecast"]["coastal_impact"]
    assert ci["will_beach"] is False


def test_dossier_carries_the_forward_forecast_section(client, nat):
    inv_id = client.post(f"{API}/scenarios/{MUMBAI}/run", headers=nat).json()["investigation_id"]
    d = client.get(f"{API}/investigations/{inv_id}/dossier", headers=nat).json()
    assert "forward_forecast" in d
    sc = d["forward_forecast"]["shoreline_contact"]
    assert sc["will_beach"] is True and sc["first_contact_point"]
    md = client.get(f"{API}/investigations/{inv_id}/dossier.md", headers=nat).text
    assert "Shoreline contact" in md or "shoreline contact" in md.lower()
