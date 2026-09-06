"""
Generate the bundled offline demo scene.

    python -m backend.ml.sar.demo_data._generate

Writes, next to this file:
    demo_scene.npz   sigma0_db (float32), bbox, pixel_m, acquisition, truth_mask,
                     lookalike_mask, meta
    demo_scene.png   8-bit dB quicklook (so a PNG-ingest path can be exercised)

Deterministic (fixed seed). Re-run only if the synth radiometry changes.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from backend.ml.sar.synth import db_to_uint8, synth_scene

HERE = Path(__file__).resolve().parent
NPZ = HERE / "demo_scene.npz"
PNG = HERE / "demo_scene.png"

# One documented, reproducible incident: a mid-ocean slick off the Konkan coast,
# 7 m/s wind (OPTIMAL detectability), plus a low-wind cell and a biogenic film
# as planted look-alikes.
ACQUISITION = datetime(2026, 3, 6, 5, 42, 0, tzinfo=timezone.utc)


def build() -> dict:
    scene = synth_scene(
        center_lat=15.60,
        center_lon=73.20,
        width=1024,
        height=768,
        pixel_m=60.0,
        wind_ms=7.0,
        seed=20260306,
        with_oil=True,
        with_lowwind=True,
        with_biogenic=True,
    )
    meta = {
        **scene.meta,
        "title": "Konkan offshore demo incident",
        "acquisition": ACQUISITION.isoformat(),
        "notes": "Synthetic Sentinel-1-like VV GRDH. 1 oil slick + 2 look-alikes.",
    }
    np.savez_compressed(
        NPZ,
        # float16 storage: ~0.01 dB quantisation, well below detection tolerance,
        # and it halves the committed asset. ingest upcasts to float32 on load.
        sigma0_db=scene.sigma0_db.astype(np.float16),
        bbox=np.asarray(scene.meta["bbox"], np.float64),
        pixel_m=np.float64(60.0),
        acquisition=np.asarray(ACQUISITION.isoformat()),
        truth_mask=scene.truth_mask,
        lookalike_mask=scene.lookalike_mask,
        meta=np.asarray(json.dumps(meta)),
    )
    cv2.imwrite(str(PNG), db_to_uint8(scene.sigma0_db))
    return {
        "npz": str(NPZ),
        "png": str(PNG),
        "oil_px": int(scene.truth_mask.sum()),
        "lookalike_px": int(scene.lookalike_mask.sum()),
        "bbox": scene.meta["bbox"],
    }


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
