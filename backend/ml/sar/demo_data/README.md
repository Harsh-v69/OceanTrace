# SAR demo data

Bundled so the SAR pipeline runs **fully offline** with no network and no
external imagery.

| File | What it is |
|---|---|
| `demo_scene.npz` | Synthetic Sentinel-1-like VV GRDH scene: `sigma0_db` (float32, 768×1024), `bbox` `[W,S,E,N]`, `pixel_m` (60), `acquisition` (ISO-8601 UTC), `truth_mask`, `lookalike_mask`, `meta`. |
| `demo_scene.png` | 8-bit dB quicklook of the same scene, so the PNG-ingest path can also be exercised. |
| `_generate.py` | Deterministic generator (`python -m backend.ml.sar.demo_data._generate`). Re-run only if `backend/ml/sar/synth.py` changes. |

## The incident

A mid-ocean oil slick off the Konkan coast (centre 15.60 N, 73.20 E), 7 m/s
wind (**OPTIMAL** SAR detectability), imaged 2026-03-06 05:42 UTC. Two
look-alikes are planted: a round low-wind cell and a sinuous biogenic film.
Ground truth: the pipeline should return **"Oil-like anomaly"** for the slick
(~74 km², oriented ≈125°, centroid ≈ 15.63 N, 73.26 E) and **"Likely
look-alike"** for the rest.

The radiometry is the calibrated CMOD-family geophysical model function ported
from OceanTrace's simulator, so a detector tuned here transfers to real
Sentinel-1 data.
