"""
Phase 3 - SAR detection pipeline.

Feeds the bundled offline demo scene (and small synthetic arrays) through
``backend.services.satellite`` and the ``backend.ml.sar`` modules, and checks
output shapes, characterisation metrics, the STEP 4 look-alike filter, and the
strict anomaly-label vocabulary.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from backend.ml.sar import ingest
from backend.ml.sar import labels as L
from backend.ml.sar.characterize import characterise
from backend.ml.sar.classifier import Prediction
from backend.ml.sar.darkspot import DarkSpot
from backend.ml.sar.features import FEATURE_NAMES
from backend.ml.sar.geo import GeoTransform, bbox_for_scene, haversine_m
from backend.ml.sar.pipeline import _lookalike_gates
from backend.ml.sar.preprocess import preprocess, refined_lee, db_to_linear
from backend.ml.sar.synth import synth_scene
from backend.services import satellite


# --------------------------------------------------------------------------- #
# Shared: run the pipeline on the demo scene exactly once
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def demo_result() -> dict:
    return satellite.analyze_demo()


@pytest.fixture(scope="module")
def demo_truth() -> dict:
    d = np.load(satellite.demo_scene_path(), allow_pickle=True)
    return {
        "sigma0_db": d["sigma0_db"],
        "bbox": list(d["bbox"]),
        "pixel_m": float(d["pixel_m"]),
        "truth_mask": d["truth_mask"].astype(bool),
        "lookalike_mask": d["lookalike_mask"].astype(bool),
    }


# --------------------------------------------------------------------------- #
# Label vocabulary (strictness requirement)
# --------------------------------------------------------------------------- #
def test_canonical_labels_are_exactly_three():
    assert L.CANONICAL_LABELS == (
        "Oil-like anomaly",
        "Likely look-alike",
        "No significant anomaly",
    )


def test_label_for_mapping():
    assert L.label_for(True, has_candidate=True) == L.OIL_LIKE
    assert L.label_for(False, has_candidate=True) == L.LOOKALIKE
    assert L.label_for(True, has_candidate=False) == L.NONE


def test_scene_label_strongest_wins():
    assert L.scene_label([]) == L.NONE
    assert L.scene_label([L.LOOKALIKE, L.LOOKALIKE]) == L.LOOKALIKE
    assert L.scene_label([L.LOOKALIKE, L.OIL_LIKE, L.NONE]) == L.OIL_LIKE


def test_assert_canonical_rejects_anything_else():
    for bad in ("oil", "OIL_LIKE", "lookalike", "Detected", ""):
        with pytest.raises(ValueError):
            L.assert_canonical(bad)


# --------------------------------------------------------------------------- #
# STEP 1 - ingestion
# --------------------------------------------------------------------------- #
def test_ingest_demo_npz_extracts_bbox_and_time(demo_truth):
    scene = ingest.load_scene(str(satellite.demo_scene_path()))
    assert scene.source == "npz"
    assert scene.sigma0_db.shape == demo_truth["truth_mask"].shape
    assert scene.sigma0_db.dtype == np.float32
    assert scene.bbox == pytest.approx(demo_truth["bbox"], rel=1e-6)
    assert scene.acquisition is not None
    assert scene.acquisition.year == 2026 and scene.acquisition.tzinfo is not None
    assert scene.pixel_m == pytest.approx(60.0, rel=0.05)


def test_ingest_raw_array_needs_no_geo(demo_truth):
    scene = ingest.load_scene(demo_truth["sigma0_db"])
    assert scene.source == "array"
    assert scene.sigma0_db.shape == demo_truth["sigma0_db"].shape
    # No bbox given -> placeholder, but a valid GeoTransform is still produced.
    assert scene.bbox_source in ("placeholder", "center")
    assert isinstance(scene.transform, GeoTransform)


def test_ingest_png_quicklook_path():
    png = satellite.demo_scene_path().with_name("demo_scene.png")
    scene = ingest.load_scene(str(png))
    assert scene.source == "png"
    assert scene.sigma0_db.ndim == 2
    # inverted dB stretch lands in a plausible SAR range
    assert -30.0 < float(np.median(scene.sigma0_db)) < 5.0


def test_ingest_geotiff_derives_bbox_from_tags(tmp_path):
    import tifffile

    h, w = 200, 300
    px_deg = 60.0 / 111_320.0                       # ~60 m at the equator
    west, north = 72.5, 16.0
    arr = (synth_scene(seed=3, width=w, height=h, pixel_m=60.0).sigma0_db).astype(np.float32)
    path = tmp_path / "S1A_IW_GRDH_20260306T054200_demo.tif"
    tifffile.imwrite(
        path,
        arr,
        extratags=[
            (33550, "d", 3, (px_deg, px_deg, 0.0), True),        # ModelPixelScale
            (33922, "d", 6, (0, 0, 0, west, north, 0), True),    # ModelTiepoint
        ],
    )
    scene = ingest.load_scene(str(path))
    assert scene.source == "geotiff"
    assert scene.bbox_source == "geotiff"
    assert scene.sigma0_db.shape == (h, w)
    assert scene.bbox[0] == pytest.approx(west, abs=1e-6)
    assert scene.bbox[3] == pytest.approx(north, abs=1e-6)
    assert scene.bbox[2] > scene.bbox[0] and scene.bbox[3] > scene.bbox[1]
    assert scene.acquisition is not None and scene.acquisition.hour == 5


def test_parse_acquisition_from_sentinel1_name():
    dt = ingest.parse_acquisition("S1A_IW_GRDH_1SDV_20200725T153000_x.tif")
    assert (dt.year, dt.month, dt.day, dt.hour, dt.minute) == (2020, 7, 25, 15, 30)
    assert dt.tzinfo is not None
    assert ingest.parse_acquisition(None) is None


# --------------------------------------------------------------------------- #
# STEP 2 - preprocessing
# --------------------------------------------------------------------------- #
def test_preprocess_returns_the_expected_fields(demo_truth):
    pre = preprocess(demo_truth["sigma0_db"])
    assert set(pre) >= {"db", "raw_db", "linear", "land", "sea", "stats"}
    assert pre["db"].shape == demo_truth["sigma0_db"].shape
    assert pre["land"].dtype == bool and pre["sea"].dtype == bool
    assert "Refined Lee" in pre["stats"]["speckle_filter"]
    assert pre["stats"]["enl_estimated"] > 0


def test_refined_lee_suppresses_speckle_on_flat_sea():
    rng = np.random.default_rng(0)
    flat = np.full((256, 256), db_to_linear(np.float32(-11.0)), np.float32)
    speckled = flat * rng.gamma(4.4, 1.0 / 4.4, flat.shape).astype(np.float32)
    filtered = refined_lee(speckled, window=7, enl=4.4)
    # variance of a homogeneous region must drop substantially
    assert filtered.var() < 0.35 * speckled.var()
    assert abs(float(filtered.mean()) - float(speckled.mean())) < 0.1 * float(speckled.mean())


# --------------------------------------------------------------------------- #
# STEP 3/4/5 - end to end on the demo scene
# --------------------------------------------------------------------------- #
def test_scene_is_classified_oil_like(demo_result):
    assert demo_result["classification"] == L.OIL_LIKE
    assert demo_result["status"] == "SPILL_DETECTED"
    assert demo_result["confidence"] > 0.5
    assert demo_result["counts"]["oil_like"] == 1


def test_every_detection_uses_a_canonical_label(demo_result):
    assert demo_result["detections"], "expected at least one candidate"
    for d in demo_result["detections"]:
        assert d["classification"] in L.CANONICAL_LABELS
    assert demo_result["labels_vocabulary"] == list(L.CANONICAL_LABELS)


def test_mask_shape_and_rle_roundtrip(demo_result, demo_truth):
    mask = demo_result["mask"]
    assert mask.shape == demo_truth["truth_mask"].shape
    assert mask.dtype == bool and mask.any()
    decoded = satellite.rle_decode(demo_result["mask_rle"]["oil_like"])
    assert np.array_equal(decoded, mask)
    cand = satellite.rle_decode(demo_result["mask_rle"]["candidates"])
    assert cand.shape == mask.shape and cand.sum() >= mask.sum()


def test_polygons_are_a_feature_collection(demo_result):
    poly = demo_result["polygons"]
    assert poly["type"] == "FeatureCollection"
    assert len(poly["features"]) == len(demo_result["detections"])
    for feat in poly["features"]:
        assert feat["type"] == "Feature"
        assert feat["geometry"]["type"] in ("Polygon", "MultiPolygon")


def test_feature_vectors_have_all_thirty_named_features(demo_result):
    assert len(demo_result["features"]) == len(demo_result["detections"])
    for fv in demo_result["features"]:
        assert set(fv) == set(FEATURE_NAMES)
        assert all(math.isfinite(v) for v in fv.values())


def test_characterization_metrics_present_and_sane(demo_result):
    oil = next(d for d in demo_result["detections"] if d["classification"] == L.OIL_LIKE)
    c = oil["characterization"]
    for key in (
        "area_km2", "area_hectares", "perimeter_km", "centroid",
        "length_km", "width_km", "major_axis_km", "minor_axis_km",
        "orientation_deg",
    ):
        assert key in c, f"missing characterisation metric {key!r}"
    assert c["area_km2"] > 0
    assert 0.0 <= c["orientation_deg"] < 180.0
    assert c["major_axis_km"] >= c["minor_axis_km"] > 0
    assert c["length_km"] > c["width_km"] > 0
    lat, lon = c["centroid"]
    assert -90 <= lat <= 90 and -180 <= lon <= 180


def test_detected_slick_matches_ground_truth(demo_result, demo_truth):
    oil = next(d for d in demo_result["detections"] if d["classification"] == L.OIL_LIKE)
    c = oil["characterization"]

    # area: truth pixels * pixel area, within 25%
    px_area_km2 = (demo_truth["pixel_m"] / 1000.0) ** 2
    truth_area = float(demo_truth["truth_mask"].sum()) * px_area_km2
    assert c["area_km2"] == pytest.approx(truth_area, rel=0.25)

    # centroid: within 5 km of the truth-mask centroid
    tf = GeoTransform(tuple(demo_truth["bbox"]), *demo_truth["truth_mask"].shape[::-1])
    ys, xs = np.nonzero(demo_truth["truth_mask"])
    tlat, tlon = tf.pixel_to_ll(float(xs.mean()), float(ys.mean()))
    dlat, dlon = c["centroid"]
    assert haversine_m(dlat, dlon, float(tlat), float(tlon)) < 5_000.0

    # orientation: the synthetic slick's long axis maps to ~125 deg
    assert min(abs(c["orientation_deg"] - 125.0), 180 - abs(c["orientation_deg"] - 125.0)) < 25.0


def test_bounding_box_is_well_formed(demo_result):
    bb = demo_result["bounding_box"]
    assert bb["west"] < bb["east"]
    assert bb["south"] < bb["north"]
    assert bb["source"] in ("provided", "geotiff", "center", "placeholder")


def test_runtime_is_cpu_reasonable(demo_result):
    assert demo_result["pipeline"]["runtime_seconds"] < 15.0
    steps = [s["step"] for s in demo_result["pipeline"]["timings"]]
    assert steps[0] == "ingest" and "preprocess" in steps and "classify" in steps


def test_pipeline_is_deterministic():
    a = satellite.analyze_demo()
    b = satellite.analyze_demo()
    assert a["classification"] == b["classification"]
    assert a["confidence"] == pytest.approx(b["confidence"], abs=1e-9)
    assert len(a["detections"]) == len(b["detections"])


# --------------------------------------------------------------------------- #
# STEP 4 - look-alike context filter, in isolation
# --------------------------------------------------------------------------- #
def _clean_oil_features(**overrides) -> dict:
    base = {
        "local_wind_ms": 7.0,
        "dist_to_land_km": 40.0,
        "mean_contrast_db": 8.0,
        "border_gradient_db_px": 0.5,
        "spreading": 20.0,
    }
    base.update(overrides)
    return base


def test_lookalike_gates_pass_clean_oil():
    assert _lookalike_gates(_clean_oil_features()) == []


def test_lookalike_gate_wind_too_strong():
    assert "wind_too_strong" in _lookalike_gates(_clean_oil_features(local_wind_ms=16.0))


def test_lookalike_gate_surf_zone():
    assert "inside_surf_zone" in _lookalike_gates(_clean_oil_features(dist_to_land_km=0.2))


def test_lookalike_gate_weak_contrast():
    assert "contrast_too_weak" in _lookalike_gates(_clean_oil_features(mean_contrast_db=1.0))


def test_lookalike_gate_diffuse_round_cell():
    g = _lookalike_gates(_clean_oil_features(border_gradient_db_px=0.02, spreading=90.0))
    assert "diffuse_round_low_wind_cell" in g
    # a sharp edge alone (still round) must NOT trip it
    assert _lookalike_gates(_clean_oil_features(spreading=90.0)) == []


# --------------------------------------------------------------------------- #
# "No significant anomaly" path
# --------------------------------------------------------------------------- #
def test_clean_sea_yields_no_anomaly():
    clean = synth_scene(
        seed=7, with_oil=False, with_lowwind=False, with_biogenic=False,
        width=640, height=512,
    )
    out = satellite.analyze_scene(clean.sigma0_db, ingest_kwargs={"bbox": clean.meta["bbox"]})
    assert out["classification"] == L.NONE
    assert out["status"] == "NO_ANOMALY"
    assert out["detections"] == []
    assert out["confidence"] == 0.0
    assert satellite.rle_decode(out["mask_rle"]["oil_like"]).sum() == 0


# --------------------------------------------------------------------------- #
# Characterisation math, against a known synthetic geometry
# --------------------------------------------------------------------------- #
def test_characterise_geometry_against_known_rectangle():
    h, w = 400, 600
    mask = np.zeros((h, w), bool)
    # 200 px (along) x 40 px (across) rectangle, unrotated, centred
    mask[180:220, 200:400] = True

    pixel_m = 50.0
    bbox = bbox_for_scene(10.0, 70.0, w, h, pixel_m)
    tf = GeoTransform(tuple(bbox), w, h)
    cand = DarkSpot(label=1, mask=mask, area_px=int(mask.sum()),
                    bbox=(180, 200, 220, 400), centroid_rc=(199.5, 299.5))
    pred = Prediction(0.9, True, "HIGH", {})
    feats = {"elongation": 5.0, "complexity": 1.2, "solidity": 1.0,
             "spreading": 4.0, "mean_contrast_db": 7.0, "max_contrast_db": 9.0,
             "border_gradient_db_px": 0.4, "dist_to_land_km": 50.0,
             "local_wind_ms": 7.0}

    out = characterise(cand, feats, pred, tf, tf.pixel_area_m2())
    c = out["properties"]

    expected_km2 = (200 * 40) * (pixel_m ** 2) / 1e6      # 8000 px * 2500 m2
    assert c["area_km2"] == pytest.approx(expected_km2, rel=0.02)
    assert c["length_km"] == pytest.approx(200 * pixel_m / 1000.0, rel=0.1)
    assert c["width_km"] == pytest.approx(40 * pixel_m / 1000.0, rel=0.25)
    # long axis runs east-west -> bearing near 90 deg
    assert min(abs(c["orientation_deg"] - 90.0), 180 - abs(c["orientation_deg"] - 90.0)) < 12.0
    lat, lon = c["centroid"]
    assert lat == pytest.approx(10.0, abs=0.05)
    assert lon == pytest.approx(70.0, abs=0.05)
    # and it never asserts "oil" - only the oil-like vocabulary is used upstream
    assert "oil_probability" in c
