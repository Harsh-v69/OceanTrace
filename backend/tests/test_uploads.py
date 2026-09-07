"""
Epic 3.3 - arbitrary SAR GeoTIFF/PNG + AIS CSV upload endpoints.
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest

from backend.services import jurisdiction as juris
from backend.tests._helpers import API, auth_header, make_user

_DEMO = Path(__file__).resolve().parents[1] / "ml" / "sar" / "demo_data"
MUMBAI = "mumbai-high-confidence"


@pytest.fixture
def nat(client, db):
    juris.seed_demo_jurisdictions(db)
    make_user(client, email="nat@example.com", role="NATIONAL")
    return auth_header(client, "nat@example.com")


def _scene_png() -> bytes:
    return (_DEMO / "demo_scene.png").read_bytes()


def _truth_mask_png() -> bytes:
    import cv2
    m = np.load(_DEMO / "demo_scene.npz")["truth_mask"].astype(np.uint8) * 255
    ok, buf = cv2.imencode(".png", m)
    assert ok
    return buf.tobytes()


def _demo_bbox() -> str:
    b = np.load(_DEMO / "demo_scene.npz")["bbox"]
    return ",".join(str(float(v)) for v in b)


# --------------------------------------------------------------------------- #
# SAR scene upload
# --------------------------------------------------------------------------- #
def test_upload_scene_runs_the_pipeline(client, nat):
    r = client.post(
        f"{API}/investigations/upload-scene", headers=nat,
        data={"title": "Op upload", "bbox": _demo_bbox()},
        files={"scene": ("demo_scene.png", _scene_png(), "image/png")},
    )
    assert r.status_code == 201, r.text
    inv = r.json()
    assert inv["reference"].startswith("INV-")
    m = inv["summary_metrics"]
    assert "sar" in m and m["sar"]["scene_classification"] in (
        "Oil-like anomaly", "Likely look-alike", "No significant anomaly")


def test_upload_scene_with_ground_truth_mask_gives_real_iou(client, nat):
    r = client.post(
        f"{API}/investigations/upload-scene", headers=nat,
        data={"title": "IoU check", "bbox": _demo_bbox()},
        files={
            "scene": ("demo_scene.png", _scene_png(), "image/png"),
            "ground_truth_mask": ("truth.png", _truth_mask_png(), "image/png"),
        },
    )
    assert r.status_code == 201, r.text
    iou = r.json()["summary_metrics"].get("iou")
    assert iou is not None
    assert 0.0 <= iou["value"] <= 1.0
    assert iou["truth_pixels"] > 0


def test_upload_scene_rejects_a_bad_bbox(client, nat):
    r = client.post(
        f"{API}/investigations/upload-scene", headers=nat,
        data={"bbox": "not-a-bbox"},
        files={"scene": ("s.png", _scene_png(), "image/png")},
    )
    assert r.status_code == 422


def test_upload_scene_is_jurisdiction_scoped(client, db):
    juris.seed_demo_jurisdictions(db)
    kl = juris.resolve_codes_to_ids(db, ["IN-KL"])[0]
    make_user(client, email="pilot.kl@example.com", role="PILOT", jurisdiction_ids=[kl])
    h = auth_header(client, "pilot.kl@example.com")
    # the demo bbox is off Maharashtra -> a Kerala pilot is refused
    r = client.post(
        f"{API}/investigations/upload-scene", headers=h,
        data={"bbox": _demo_bbox()},
        files={"scene": ("s.png", _scene_png(), "image/png")},
    )
    assert r.status_code == 403


# --------------------------------------------------------------------------- #
# AIS CSV ingestion
# --------------------------------------------------------------------------- #
def _ais_csv_near(lat: float, lon: float, mmsi: int = 412345678) -> bytes:
    """A short straight AIS track approaching (lat, lon) from the south-west."""
    lines = ["MMSI,BaseDateTime,LAT,LON,SOG,COG,Heading,VesselName,VesselType"]
    for i in range(30):
        t = f"2026-03-06T{2 + i // 12:02d}:{(i * 5) % 60:02d}:00Z"
        la = lat - 0.25 + i * (0.25 / 29)
        lo = lon - 0.25 + i * (0.25 / 29)
        lines.append(f"{mmsi},{t},{la:.5f},{lo:.5f},11.0,45.0,45,MV CSV RUNNER,80")
    return ("\n".join(lines)).encode()


def test_ingest_ais_reattributes_the_investigation(client, nat):
    inv = client.post(f"{API}/scenarios/{MUMBAI}/run", headers=nat).json()
    inv_id = inv["investigation_id"]
    m0 = client.get(f"{API}/investigations/{inv_id}", headers=nat).json()["summary_metrics"]
    origin = m0["hindcast"]["best_estimate"]

    r = client.post(
        f"{API}/vessels/ingest-ais", headers=nat,
        data={"investigation_id": str(inv_id), "reattribute": "true"},
        files={"csv_file": ("ais.csv", _ais_csv_near(origin[0], origin[1]), "text/csv")},
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["reattributed"] is True
    assert body["vessels"] >= 1
    assert body["rows_ingested"] == 30

    # the ingested vessel is now persisted + attached
    vs = {v["mmsi"] for v in client.get(f"{API}/vessels", headers=nat).json()}
    assert "412345678" in vs
    m1 = client.get(f"{API}/investigations/{inv_id}", headers=nat).json()["summary_metrics"]
    assert "412345678" in (m1.get("vessel_tracks") or {})


def test_ingest_ais_without_reattribution_just_attaches(client, nat):
    inv_id = client.post(f"{API}/scenarios/{MUMBAI}/run", headers=nat).json()["investigation_id"]
    m0 = client.get(f"{API}/investigations/{inv_id}", headers=nat).json()["summary_metrics"]
    origin = m0["hindcast"]["best_estimate"]
    r = client.post(
        f"{API}/vessels/ingest-ais", headers=nat,
        data={"investigation_id": str(inv_id), "reattribute": "false"},
        files={"csv_file": ("ais.csv", _ais_csv_near(origin[0], origin[1], mmsi=999888777), "text/csv")},
    )
    assert r.status_code == 201, r.text
    assert r.json()["reattributed"] is False
    m1 = client.get(f"{API}/investigations/{inv_id}", headers=nat).json()["summary_metrics"]
    assert "999888777" in (m1.get("vessel_tracks") or {})


def test_ingest_ais_rejects_empty_csv(client, nat):
    inv_id = client.post(f"{API}/scenarios/{MUMBAI}/run", headers=nat).json()["investigation_id"]
    r = client.post(
        f"{API}/vessels/ingest-ais", headers=nat,
        data={"investigation_id": str(inv_id)},
        files={"csv_file": ("empty.csv", b"MMSI,BaseDateTime,LAT,LON\n", "text/csv")},
    )
    assert r.status_code == 422
