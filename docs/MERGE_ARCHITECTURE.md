# MERGE ARCHITECTURE — Phase 1 Repository Audit

**Problem statement:** SIH26143 (NTRO) — satellite oil-spill detection + AIS vessel attribution
**Target:** one unified FastAPI application (no separate Streamlit app)
**Audit date:** 2026-09-06
**Audit host:** Windows 11, Python 3.13.7, CPU only, no GPU

This document is the output of Phase 1 (audit only). **No application code has been
written or rewritten.** It records what each source repository contains, what actually
runs, what conflicts, and which implementation is selected for each capability of the
merged system.

---

## 0. Sources audited

| | SAMUDRA NETRA | POSEatSea |
|---|---|---|
| Repo | `github.com/YadavAnish56/SIH-2026-PROTOTYPE` | `github.com/23f2003521/SIH2026` |
| HEAD at audit | `203c0a3` "Revamp UI" (2026-08-31) | `fab46fa` "UI: Move Route Deviation…" (2026-09-02) |
| Python code | ~9,700 LOC (`ml/`, `backend/`, `simulator/`, `scripts/`) | ~6,200 LOC (`poseatsea/`, `ui/`, `api/`, `tests/`) |
| Frontend | ~1,900 LOC hand-written JS/CSS + vendored Leaflet | Streamlit console (`ui/`, ~1,400 LOC) |
| Web framework | **FastAPI** (`backend/main.py`) + static vanilla-JS SPA | **Streamlit** primary; **FastAPI** service optional (`api/main.py`) |
| Server entrypoint | `python run.py` → `uvicorn backend.main:app` :8000 | `streamlit run ui/app.py` :8501 / `uvicorn api.main:app` :8000 |
| Both target | SIH PS 26143, NTRO | SIH PS 26143, NTRO |

Both repositories are self-described as "working systems, not mockups", both are honest
about limitations in their READMEs, and both are considerably more mature than a typical
hackathon prototype. The merge is a genuine best-of-both exercise, not a rescue.

---

## 1. Component comparison

### 1.1 Capability matrix

| Capability | SAMUDRA NETRA | POSEatSea |
|---|---|---|
| **API style** | FastAPI, versioned REST `/api/v1`, 20+ endpoints, OpenAPI docs, wide-open CORS, frontend-agnostic by design | Streamlit (stateful re-run model) + a thin optional FastAPI mirror of the same inference modules |
| **SAR oil detection** | Classical chain: Refined-Lee speckle filter → adaptive dark-spot segmentation (hysteresis + watershed) → **30 physical features** → **RandomForest+GradientBoosting soft-voting ensemble** (+ physics rule fallback). Trained on simulator; separate SAR and EO models. Fragment re-merge for reporting. | **U-Net + MiT-B2 SegFormer**, 5-class semantic segmentation (sea/oil/look-alike/ship/land), 512×512. Trained on the Krestenitis/Sentinel-1 benchmark. |
| **Optical (EO) support** | Yes — `ml/eo/optical.py` converts an optical scene to a pseudo-dB anomaly index consumed by the *same* downstream chain. Dedicated EO classifier. | No |
| **Look-alike rejection** | Explicit — separate "Look-alike" outcome from the classifier + physics discriminators (border-gradient etc.); measured false-alarm rates | Explicit — dedicated "Look-alike" class in the segmenter |
| **Drift / hindcast physics** | **Yes — the differentiator.** Lagrangian RK4 advection (`ml/drift/particles.py`): current + 3% wind factor + Ekman deflection + Stokes + turbulent random walk. Backward hindcast = same code, negative timestep. Per-parcel ageing for continuous line-source releases. Coastline beaching. Weathering (Fay/Fingas/Mackay). | **None.** Explicitly declared: "No drift or hindcast modelling … Faking one would be worse than omitting it." Attribution assumes the slick lies where observed. |
| **Origin reconstruction** | Yes — origin-probability raster, release window in space and time, AIS-refined | No |
| **48 h forecast + shoreline impact** | Yes — forward RK4, first-contact ETA, % ashore | No |
| **Met-ocean data** | Deterministic synthesised ERA5/HYCOM-like fields (`ml/drift/metocean.py`), single seam `_build_grids()` for swapping in real NetCDF | n/a |
| **AIS ingestion / track building** | `ml/ais/tracks.py` — MarineCadastre schema, ITU-R M.1371 type codes, outlier removal, gap detection, resampling to a queryable `position(t)`, flag-from-MMSI | `poseatsea/scenario/real_ais.py` — parser for one specific real extract (Mauritius AOI, July 2020, 27,979 rows, 231 vessels), fractional-second timestamps, ROT sentinel handling |
| **AIS anomaly detection** | Rule/heuristic behavioural model (`ml/ais/anomaly.py`): AIS dark gap, slow-steaming vs the vessel's *own* median, loitering/course-deviation, true solar-elevation night check. 0–1 score + human-readable finding per detector. | **11-feature undercomplete autoencoder** (11→16→8→4→8→16→11), reconstruction-error threshold `AE_THRESHOLD=1.104481`. Precision 0.93 / recall 0.41. **Requires its pickled `StandardScaler`.** |
| **Trajectory prediction** | Not a dedicated model — course/loiter deviation is folded into the behavioural score | **2-layer LSTM** (6→128→128→2), last 8 pings → next position. Mean err 0.37 km. Hard operating-envelope guard (AOI box, ~60 s cadence, NE–SW heading band). |
| **Attribution / fusion** | **6-criterion transparent weighted model** (`ml/ais/scoring.py`), weights in `config.ATTRIB.WEIGHTS` sum to 1.0: spatiotemporal 0.32, axis-alignment 0.20, dark-gap 0.16, speed-anomaly 0.12, manoeuvre 0.08, vessel-prior 0.12. Two-pass (localise event in time, then score behaviour in that window). Proximity gate. **Feedback loop:** attribution pins release time → re-run hindcast → origin error 27→2 km. Type prior by vessel class. | **4-criterion transparent weighted sum** (`poseatsea/fusion.py`): proximity 0.40, anomaly 0.30, dwell 0.20, deviation 0.10. Single pass. "Anomaly counts only if near the slick." `drift_caveat` string attached to every result. |
| **RBAC / auth** | **None** | **None** |
| **Database** | **None** — bounded in-memory `LRUStore` (10 scenes / 10 analyses); light artefacts written as JSON to `data/jobs/` | **None** — models held in a lazy thread-safe registry; no persistence |
| **SMS / Twilio alerting** | **None** | **None** |
| **Model registry / lazy loading** | Per-sensor singleton cache in `classifier.py`; lazy torch import for optional U-Net | **`poseatsea/registry.py`** — mature: lazy, per-model load lock, per-model inference lock, load-time + param-count instrumentation, `strict=True` state-dict loading |
| **Frontend** | Vanilla JS SPA, no build step, no CDN (fonts + Leaflet vendored), light/dark themes, offline demo dataset baked in (`demo.js`), single network seam `js/api.js` | Streamlit multi-page console (Overview / AIS Anomaly / Route Deviation / SAR Segmenter), Esri satellite basemaps via `folium` |
| **Benchmark / self-validation** | `scripts/validate.py` — end-to-end sweep over N randomised incidents, scores itself vs ground truth; `scripts/evaluate_real.py` — real Zenodo Sentinel-1 test split | `pytest tests/` — 47 tests asserting weight/architecture match, scaler necessity, guard rails, attribution correctness |
| **Real-data readers** | Zenodo Sentinel-1 (`ml/data/zenodo.py`), MarineCadastre daily national file streamer (`ml/data/marinecadastre.py`) — parse local files only, **no auto-download** | One committed real AIS CSV; SAR scene library of 5 real Sentinel-1 frames + precomputed masks |

### 1.2 Overlap — where the two repos do the same job

| Overlapping function | SAMUDRA NETRA | POSEatSea | Verdict |
|---|---|---|---|
| SAR oil segmentation | Classical + RF/GB ensemble + optional small U-Net | U-Net/SegFormer (MiT-B2) | **Keep SN.** PS's checkpoint is not usable (§3.3). |
| Look-alike discrimination | Classifier + physics | Segmenter class | Keep SN; adopt PS's 5-class *vocabulary* for labels. |
| AIS behavioural anomaly | Heuristic multi-detector | Autoencoder + scaler | **Keep both — complementary.** PS AE as an additional weighted signal into SN's scorer (CLAUDE.md mandates the AE + scaler). |
| Vessel attribution / ranking | 6-criterion, 2-pass, feedback loop | 4-criterion, 1-pass | **Keep SN.** Superset of PS; PS's `dwell` term is the one idea worth porting in. |
| Attribution transparency (per-criterion breakdown returned) | Yes | Yes | Both good; keep SN's shape (already the API contract). |
| Trajectory / route deviation | Folded into behavioural score | Dedicated LSTM | **Keep PS LSTM** as a new capability; feed its deviation as SN's `manoeuvre`/route-deviation input. |
| FastAPI service | Primary, full contract | Secondary mirror | **Keep SN.** |
| Model lifecycle management | Ad-hoc per-module cache | Instrumented registry | **Adopt PS's registry pattern** for the merged `/ml` model loading. |
| Real AIS schema handling | General MarineCadastre normaliser | One bespoke extract parser | **Keep SN's normaliser;** keep PS's Mauritius/Wakashio parser as one concrete scenario adapter. |
| Deterministic demo scenario | `simulator/` (SAR + AIS + scenario) | Real Wakashio grounding replay | Keep both as selectable scenarios. |

---

## 2. Which ML checkpoints actually exist

### 2.1 SAMUDRA NETRA — **no checkpoints committed; all trained locally**

`.gitignore` excludes `data/models/*.joblib`, `*.npz`, `*.pt`. What is in the repo:

| File | Committed? | What it is |
|---|---|---|
| `data/models/oil_classifier_metrics.json` | Yes | Metrics + feature importances for the SAR classifier (no model) |
| `data/models/oil_classifier_eo_metrics.json` | Yes | Same for the EO classifier |
| `data/models/unet_metrics.json` | Yes | Held-out metrics for the optional U-Net (IoU 0.838) |
| `data/models/validation*.json` | Yes | Recorded sweep results (simulated + real Zenodo) |
| `oil_classifier.joblib`, `oil_classifier_eo.joblib`, `unet_oil.pt` | **No** | Generated by `python -m scripts.setup` |

**Audit result:** `scripts/setup --quick` ran clean in **98 s** on this host and produced
`oil_classifier.joblib` (RandomForest-400 + GradientBoosting-250 voting ensemble):
accuracy 0.983, precision 0.938, recall 0.882, ROC-AUC 0.97 on its held-out split.
The U-Net (`unet_oil.pt`) is **off by default** and only trains if torch is installed;
the authors' own measurement is that bolting it on *lowers* final-mask IoU (0.751→0.700)
and adds ~2 s, so it is an opt-in curiosity, not a dependency.

Implication for the merge: SN has **no binary model artefacts to carry** — the models
are reproducible from code + config in ~100 s. This is a strong point for offline-first.

### 2.2 POSEatSea — **3 of 4 checkpoints committed; the important one is not**

| File | In repo? | Size | State |
|---|---|---|---|
| `models/ais_phase1_autoencoder.pth` | **Yes** | 8 KB | Valid. Loads `strict=True`, 735 params. |
| `models/ais_phase1_scaler.joblib` | **Yes** | 863 B | Valid `StandardScaler`, `n_features_in_=11`. Pickled under scikit-learn **1.6.1**. |
| `models/trajectory_lstm_baseline.pth` | **Yes** | 811 KB | Valid. Loads `strict=True`. `lstm.weight_ih_l0` = (512, 6) as documented. |
| `models/best_sar_model.pth` (≈105 MB MiT-B2) | **No** | — | `.gitignore`d (over GitHub's 100 MB blob limit). Fetched at runtime from the Hugging Face Hub repo `23f2003521/poseatsea-weights` **only if `POSEATSEA_WEIGHTS_REPO` is set**. |

**Audit result — the SAR checkpoint is doubly unusable:**

1. **Not obtainable offline.** On a clean checkout with no `POSEATSEA_WEIGHTS_REPO`,
   `run.py --check` and 6 pytest tests fail with `FileNotFoundError: best_sar_model.pth`.
   Getting it needs an internet round-trip to Hugging Face.
2. **Even when downloaded, it is a broken checkpoint — per the authors.**
   `deploy/DEPLOY.md`, verbatim: *"`best_sar_model.pth` currently contains a trained
   MiT-B2 encoder with an **untrained decoder and segmentation head** … It emits
   incoherent noise regardless of input resolution or preprocessing."* Their own
   `deploy/DEPLOY.md` troubleshooting section: *"Only the SAR page costs real memory
   (342 MB). It is also not producing valid output at present."*

So POSEatSea's headline deep-learning component **does not work today**, by its authors'
own statement, and cannot be exercised in this audit without downloading a 105 MB file
that is known to be noise. The two small models (AE, LSTM) are genuine and load cleanly.

---

## 3. Selected implementation per capability (and what is rejected, and why)

Legend: **[SN]** = from SAMUDRA NETRA, **[PS]** = from POSEatSea, **[NEW]** = neither has it, must be built in Phase 2+.

| Capability | **Selected** | Rejected | Reason for the selection |
|---|---|---|---|
| **Unified web API** | **[SN]** FastAPI `backend/main.py` + `/api/v1` router | [PS] Streamlit `ui/app.py` | CLAUDE.md forbids a separate Streamlit app. SN is already a clean layered FastAPI service with a documented, versioned contract, an OpenAPI schema, CORS, and a UI-agnostic design (`serialize.py` is the only internal→wire boundary). PS's Streamlit model (full script re-run per interaction, `st.cache_resource`) is architecturally incompatible with "one unified API". PS's `api/main.py` is a thin mirror and confirms the inference modules are already API-shaped, but SN's is the fuller contract. |
| **SAR oil detection + look-alike rejection** | **[SN]** classical chain + RF/GB ensemble + physics fallback; **[PS]** 5-class label vocabulary adopted for output only | [PS] U-Net/MiT-B2 SegFormer as the detector | PS's checkpoint is unusable (§2.2): untrained decoder, emits noise, not in the repo, needs internet. SN's detector **runs in ~3 s on CPU with no downloads**, is fully reproducible from `scripts/setup`, exposes real IoU (median 0.79–0.89 simulated, 0.71 on real Zenodo), and every decision traces to a named physical feature (border-gradient, contrast) — which matters for "explainable to an investigator". SN also already handles the resolution-dependent minimum-area bug and real-vs-simulated threshold recalibration that PS's `deploy` notes are still grappling with. If a *genuinely trained* SegFormer becomes available later it can be added behind the same detector interface as an optional refiner (SN already has that slot for its U-Net). |
| **Optical (EO) imagery** | **[SN]** `ml/eo/optical.py` | (PS has nothing) | Problem statement names "SAR and EO". Only SN implements it, and it does so by feeding the same downstream chain — no second pipeline. |
| **Drift physics — hindcast + forecast** | **[SN]** `ml/drift/*` (RK4 particles, per-parcel ageing, weathering, coastline beaching, origin raster) | (PS deliberately has nothing) | This is the single biggest capability gap between the two repos and the core of the problem statement ("determine … where and when the oil entered the water"). PS explicitly chose not to build it. SN's implementation is measured: AIS-constrained origin error **median 3.5 km** vs physics-only 30 km in this audit's 12-run sweep. Non-negotiable keep. |
| **Met-ocean provider** | **[SN]** `ml/drift/metocean.py` (deterministic synthesised fields, single swap seam) | — | CLAUDE.md asks for "deterministic simulated met-ocean data". Exactly what SN provides, with `_build_grids()` as the documented seam for real ERA5/HYCOM later. |
| **AIS ingestion / normalisation / track reconstruction** | **[SN]** `ml/ais/tracks.py` + `ml/ais/schema.py` | [PS] `scenario/real_ais.py` as the general reader | SN's is a general MarineCadastre-schema normaliser with outlier removal, gap detection and a queryable `position(t)` interpolant that the hindcast needs (release times never align with ping times). PS's parser is hard-wired to one 2020 Mauritius extract. **Keep PS's parser as one scenario adapter** that emits SN's normalised frame. |
| **AIS behavioural anomaly (heuristics)** | **[SN]** `ml/ais/anomaly.py` (dark-gap, slow-steam-vs-own-median, loiter, solar-elevation night) | — | Directly feeds SN's 6-criterion scorer; produces human-readable findings. Keep. |
| **AIS anomaly (learned) — autoencoder + scaler** | **[PS]** `ais_phase1_autoencoder.pth` + `ais_phase1_scaler.joblib`, via `poseatsea/inference/ais.py` | Dropping it | **CLAUDE.md mandates** POSEatSea's AE and its pre-trained `StandardScaler` ("**Must** use their pre-trained `StandardScaler`"). Both checkpoints are valid and tiny (8 KB + 863 B). Integrate as an **additional normalised signal** into SN's attribution scorer (a learned complement to the heuristic dark-gap/speed terms), not as a replacement. Feature order is positional and the scaler is mandatory — `test_unscaled_input_gives_a_different_answer` proves skipping it changes the answer 100×. |
| **Trajectory / route-deviation prediction** | **[PS]** `trajectory_lstm_baseline.pth` via `poseatsea/inference/trajectory.py` | SN's implicit course-deviation heuristic (kept as fallback outside the AOI) | PS has a real, validated LSTM (mean err 0.37 km on unseen tracks; `test_prediction_matches_published_accuracy` passes) with an honest operating-envelope guard. It is a new capability SN lacks. Its output (deviation-from-predicted-track, km) becomes an input to SN's `manoeuvre`/route-deviation criterion. **Hard limitation:** valid **only inside the Mauritius AOI** it was normalised for (`LAT_MIN/MAX`, `LON_MIN/MAX` in `poseatsea/config.py`); outside, `assess_inputs()` must refuse and the merged scorer must fall back to SN's heuristic. |
| **Attribution / fusion / ranking** | **[SN]** `ml/ais/scoring.py` (6-criterion, two-pass, proximity gate, feedback loop) | [PS] `poseatsea/fusion.py` (4-criterion, one-pass) | SN's is a strict superset: it has spatiotemporal *volume* search (not a single assumed slick age), axis-alignment (does the vessel's course lie along the slick?), and the release-time feedback into the hindcast — none of which PS has because PS has no hindcast. **Port from PS:** the `dwell` term (share of *observed* time inside the radius, normalised against observed not nominal time — a genuinely good idea) and the always-attached `drift_caveat` string. Keep SN's weight vocabulary since it is already the API contract and already sums to 1.0. |
| **Model loading / lifecycle** | **[PS]** registry pattern (`poseatsea/registry.py`) generalised for `/ml` | [SN] per-module ad-hoc caches | PS's registry is the better engineering: lazy, per-model load + inference locks, `strict=True`, instrumented (load time, param count surfaced). Adopt its *shape* for lazy-loading the merged system's classifier, AE, LSTM and any future SAR net. Aligns with CLAUDE.md's "lazy model loading, caching". |
| **Frontend** | **[SN]** vanilla-JS SPA (`frontend/`), served as static files by FastAPI | [PS] Streamlit console | CLAUDE.md: "offline-first UI", "no separate Streamlit app". SN's UI has no build step, no CDN (everything vendored), a baked-in offline demo dataset, light/dark themes, and one network seam (`js/api.js`). PS's console is tied to the Streamlit runtime. PS's *page structure* (Overview / AIS / Route Deviation / SAR) is a useful reference for organising SN's views. |
| **Benchmark harness** | **[SN]** `scripts/validate.py` + `scripts/evaluate_real.py` | — | Produces the real IoU / top-1 attribution numbers CLAUDE.md wants exposed. Extend it to also exercise the PS AE and LSTM once integrated. |
| **Regression tests** | **[PS]** `pytest` suite (adapt) | — | SN ships **no unit tests at all** (only the end-to-end sweep). PS's 47-test suite is a model for what the merged repo needs: architecture/weight-shape asserts, "scaler is mandatory" guard, attribution-correctness, AOI-refusal. Carry its intent over to the merged code. |
| **RBAC (Pilot / Regional / National) + JWT/session auth** | **[NEW]** | — | Neither repo has any auth. Full build in a later phase, plus GeoJSON maritime-boundary polygons + point-in-polygon role mapping. |
| **Persistence — SQLite (PostGIS-ready)** | **[NEW]** | — | Neither repo has a database. SN's `LRUStore` interface (`put/get/…`, 4 methods) is the seam to implement against. |
| **Twilio SMS alerting (`SmsProvider` + Mock)** | **[NEW]** | — | Neither repo has it. Build the interface + mock first, real Twilio behind a flag, triggered on anomaly-confidence threshold. |

---

## 4. Dependency conflicts and model-compatibility issues

### 4.1 Installed-version reality (clean installs done during this audit, Python 3.13.7 / Windows)

| Package | SN `requirements.txt` | SN resolved | PS `requirements.txt` | PS resolved | Conflict? |
|---|---|---|---|---|---|
| Python | 3.11–3.14 stated | 3.13.7 | 3.11 (Streamlit Cloud) / 3.14 dev | 3.13.7 | none — both fine on 3.13 |
| **scikit-learn** | `>=1.4` | **1.9.0** | **`==1.6.1`** (hard pin) | **1.6.1** | **YES — see §4.2** |
| numpy | `>=1.26` | 2.5.2 | `>=1.26,<3` | 2.5.2 | none |
| pandas | `>=2.1` | 3.0.5 | `>=2.0` | 3.0.5 | none |
| scipy | `>=1.11` | 1.18.1 | (transitive) | 1.18.1 | none |
| **opencv** | `opencv-python>=4.9` | 5.0.0.93 | `opencv-python-headless>=4.8` | 5.0.0.93 (headless) | **Minor — see §4.3** |
| Pillow | `>=10.2` | 12.3.0 | `>=10.0` | 12.3.0 | none |
| joblib | `>=1.3` | 1.6.0 | `>=1.3` | 1.6.0 | none |
| **torch** | **optional** (`; extra == "deep"`) | not installed | **required** `>=2.2,<3` | **2.14.0+cpu** | **Semantic — see §4.4** |
| torchvision | — | — | (via smp) | 0.29.0+cpu | PS only |
| segmentation-models-pytorch | — | — | `>=0.3.3` | 0.5.0 | PS only (for the dead SAR net) |
| timm | — | — | `>=0.9` | 1.0.29 | PS only |
| fastapi | `>=0.110` | 0.141.1 | **absent from `requirements.txt`** | — (added manually in audit) | **YES — see §4.5** |
| uvicorn | `uvicorn[standard]>=0.27` | 0.52.4 | **absent** | — | **YES — see §4.5** |
| pydantic | `>=2.6` | 2.13.5 | (via fastapi) | 2.13.5 | none |
| python-multipart | `>=0.0.9` | 0.0.32 | `>=0.0` (listed) | 0.0.32 | none |
| streamlit | — | — | `>=1.40` | 1.63.0 | **Dropped in merge** (no Streamlit app) |
| altair / folium / streamlit-folium / pydeck | — | — | listed | installed | **Dropped in merge** (Streamlit-only) |
| huggingface-hub | — | — | `>=0.24` | 1.30.0 | Only needed for the dead SAR weights fetch — **drop** |
| shapely | `>=2.0` | 2.1.2 | — | — | SN only (geometry / point-in-polygon) |
| scikit-image | `>=0.22` | 0.26.0 | — | — | SN only |

### 4.2 scikit-learn — the one real hard conflict

- **SN** installs whatever is newest (`>=1.4`) and **trains its own** RF/GB classifier at
  `scripts/setup` time, so it pickles under whatever version is present — currently 1.9.0.
  It is version-tolerant by construction.
- **PS** hard-pins `scikit-learn==1.6.1` because its `ais_phase1_scaler.joblib` was pickled
  under exactly 1.6.1 and they want zero `InconsistentVersionWarning`.
- **A single merged environment can only install one version of scikit-learn.**

Assessment: **low-risk conflict.** `StandardScaler` is a trivial estimator (stores
`mean_`, `scale_`, `var_`, `n_features_in_`). It **unpickles and predicts correctly under
1.9.0** — verified in this audit: `joblib.load` of the PS scaler under sklearn 1.6.1 and
1.9.0 both give a working 11-feature `StandardScaler`. The only cost of *not* pinning is a
one-line `InconsistentVersionWarning`, which PS's own `inference/ais.py` already suppresses
with a `warnings.catch_warnings()` filter.

**Recommendation:** the merged repo pins **`scikit-learn>=1.6,<2`** (or a single chosen
version such as 1.6.1 for byte-for-byte reproducibility of the PS scaler), retrains SN's
classifier under whatever is chosen, and keeps the targeted warning filter around the
scaler load. Re-pickling the PS scaler under the chosen version and committing that is the
clean long-term fix (allowed — it's a re-serialisation, not a retrain).

### 4.3 opencv — headless vs GUI

SN depends on `opencv-python` (ships GUI/highgui libs), PS on `opencv-python-headless`.
Installing both into one env is a known footgun (the second silently wins; can also cause
`libGL.so.1` errors on servers — PS's `deploy/DEPLOY.md` calls this out). For an
offline-first server app **use `opencv-python-headless`**; nothing in either codebase calls
`cv2.imshow`/`cv2.waitKey` (verified — all usage is `imdecode`/`resize`/`dilate`/
`connectedComponentsWithStats`/`cvtColor`).

### 4.4 torch — optional in SN, mandatory in PS

- SN treats torch as an optional "deep" extra; the whole system runs without it (the U-Net
  simply reports "no checkpoint or torch" and is skipped). `ml/sar/unet.py` imports torch
  lazily inside functions.
- PS imports `torch` at module top-level in `poseatsea/config.py` (`DEVICE = resolve_device()`),
  so **any** use of the PS AE or LSTM drags torch in as a hard dependency.

Since the merge **keeps the PS AE and LSTM** (§3), **torch becomes a hard dependency of the
merged system.** Consequence: the merged `requirements.txt` must carry
`--extra-index-url https://download.pytorch.org/whl/cpu` and `torch>=2.2,<3` (CPU wheel,
~200 MB) as a *required* line, not an extra. This is compatible with "offline-first local
CPU" — torch-CPU runs fine without a GPU — but it is a ~200 MB install and a real change to
SN's current "pure-CPU, no-torch" footprint. `segmentation-models-pytorch`/`timm` are **not**
needed (they exist only for the dead SAR net) and should be dropped.

### 4.5 POSEatSea `requirements.txt` is missing FastAPI

`poseatsea/api/main.py` and `tests/test_api.py` both `import fastapi`, and `docs/VERSIONS.txt`
records `fastapi==0.141.1` / `uvicorn==0.52.4` in the dev environment — but **neither is
listed in `poseatsea/requirements.txt`**. On a clean install, `pytest tests/` aborts at
collection with `ModuleNotFoundError: No module named 'fastapi'` (reproduced in this audit).
Not a blocker for the merge (the merged repo takes SN's FastAPI stack, which is complete),
but it means PS's API tier and its `test_api.py` have **never run from its own stated deps** —
factor that into how much trust to place in that tier.

### 4.6 Model / architecture compatibility notes

| Issue | Detail | Impact on merge |
|---|---|---|
| PS AE feature order is **positional** | `FEATURE_ORDER` in `poseatsea/config.py` — 11 features, 5 of them per-vessel diffs (`course_diff` is circular-wrapped). Wrong order → silent nonsense. | The merged AIS pipeline must build the feature frame in exactly this order from SN's normalised AIS schema. A small adapter (`SN tracks → PS FEATURE_ORDER`) is Phase-2 work. |
| PS scaler is **mandatory** | Model trained on standardised inputs; raw input → error ~100× larger. | Load scaler with the AE always; never expose a "skip scaler" path. |
| PS LSTM is **AOI-locked** | Normalisation hard-coded to Mauritius box + July-2020 speed dist. Out-of-AOI `assess_inputs()` returns `usable=False`. | Merged scorer must gate LSTM use on AOI membership and fall back to SN's heuristic deviation elsewhere. Do **not** present LSTM output outside the AOI. |
| PS LSTM cadence / heading limits | Reliable near ~60 s ping cadence and on 045°/225° headings; degrades badly on NW–SE and at 5-min spacing. `TRAJ_RELIABLE_COURSE_BANDS`. | Surface the confidence band ("nominal/degraded/unreliable") through to the API, as PS already does. |
| SN classifier is **per-sensor** | Separate `oil_classifier.joblib` (SAR) and `oil_classifier_eo.joblib` (EO) over a shared 30-feature set. | Merged setup must train both; `get_classifier(sensor)` already handles fallback + caveat. |
| SN has **no committed model binaries** | All regenerated by `scripts/setup` (~100 s). | Merged CI/first-run must run setup (or ship the trained `.joblib`s as build artefacts). Keeps the repo light and offline-reproducible. |
| PS SAR `best_sar_model.pth` | Untrained decoder, emits noise, 105 MB, not in repo, needs HF Hub. | **Excluded from the merge entirely.** No `POSEATSEA_WEIGHTS_REPO`, no `huggingface-hub`, no `smp`/`timm`. |
| FastAPI `on_event("startup")` (SN) | `backend/main.py` uses the deprecated `@app.on_event` hook (works on installed 0.141, emits a DeprecationWarning). | Trivial migration to `lifespan=` (PS's `api/main.py` already shows the pattern) during the merge. |

---

## 5. Internet and GPU requirements

### 5.1 GPU

| Component | GPU needed? |
|---|---|
| SN — entire pipeline (SAR classical + RF/GB, drift RK4, AIS scoring, EO) | **No.** CPU only. NumPy/SciPy/scikit-learn/OpenCV. Audit sweep: mean 3.0 s/incident on this CPU box. |
| SN — optional U-Net | No (trains in ~5 min on CPU; disabled by default anyway). |
| PS — AIS autoencoder (735 params) | No. Runs on CPU in microseconds; `resolve_device()` picks CPU when no CUDA. |
| PS — trajectory LSTM (~200 K params) | No. CPU fine. |
| PS — SAR SegFormer (~24 M params, 110 MB) | Would *benefit* from GPU but runs on CPU (~seconds). **Moot — excluded from the merge.** |

**Merged system: no GPU required, anywhere.** Consistent with CLAUDE.md ("offline-first
local CPU execution").

### 5.2 Internet

| Need | SN | PS | Merged system |
|---|---|---|---|
| Fetch ML weights at runtime | **Never** (models built locally) | **Yes for SAR** — HF Hub download of `best_sar_model.pth` unless present | **Never** (SAR net excluded; AE + LSTM committed, tiny) |
| Python package install | Yes (one-time `pip install`) | Yes (one-time; torch from the CPU index) | Yes (one-time; torch-CPU index) |
| Basemap tiles in the UI | OpenStreetMap tiles, **graceful fallback** to a plain canvas on tile error — app fully usable offline | Esri World Imagery tiles; offline fallback only via `POSEATSEA_OFFLINE_MAPS=1` env flag | Take SN's UI: **works offline**, degraded basemap only |
| Fonts / JS libs (CDN) | **None** — Leaflet + fonts vendored in `frontend/vendor/` | Loaded by Streamlit runtime | SN's vendored approach — **no CDN** |
| Real datasets (Zenodo Sentinel-1, MarineCadastre daily file) | Manual download by the operator; readers parse local files only, **no auto-download** | One real AIS CSV committed; 5 SAR scenes + masks committed | Same — optional local files, no network calls in code |
| Twilio SMS (future) | n/a | n/a | **Yes when enabled** (real SMS); Mock provider is the offline default |

**Merged system runs fully offline** after a one-time dependency install, with the only
optional online features being (a) live basemap tiles and (b) real Twilio SMS when
explicitly switched on. This satisfies "no internet" for the core demo path.

---

## 6. Test / run results recorded in this audit

### 6.1 SAMUDRA NETRA

| Step | Command | Result |
|---|---|---|
| Dependency install | `pip install -r requirements.txt` | **OK** once the venv was placed at a short path. *First attempt failed* with `OSError [Errno 2] … sklearn\metrics\_pairwise_distances_reduction\…` — **Windows MAX_PATH (260 char)**; the deep scratchpad path + scikit-learn's long internal filenames exceed it. Fixed by installing to `C:\Users\harsh\.sihaudit\sn_venv`. **This is a real portability finding — see §7.** |
| Import smoke test | `import backend.main; from ml.* import …` | **OK** — all modules import (`ALL SN IMPORTS OK`). |
| Model setup | `python -m scripts.setup --quick` | **OK in 98 s.** Built 120-scene simulated training set, trained SAR + EO classifiers, built demo scenario. SAR classifier: acc 0.983 / prec 0.938 / recall 0.882 / ROC-AUC 0.97. |
| End-to-end validation | `python -m scripts.validate --runs 12` | **OK — 12/12 runs, 0 errored.** detection rate 1.00, top-1 attribution 1.00, top-3 1.00, seg IoU median 0.79 (mean 0.77), origin error median **3.52 km** (physics-only median 30.4 km), release-time error median **0.47 h**, score margin median 32.5, runtime mean **3.03 s**. Figures track the README's claimed 40/60-run numbers. |
| Server boot | `uvicorn backend.main:app` | **OK.** Starts, loads classifier, logs metrics. `GET /api/v1/health` → 200 with full metrics JSON. `GET /app/` → 200, 31 KB HTML (static SPA served). One `DeprecationWarning` for `@app.on_event("startup")`. |
| Unit tests | — | **None exist.** SN has no `tests/`, no `pytest.ini`, no `conftest.py`. Its "tests" are `scripts/validate.py` (above) and `scripts/evaluate_real.py` (needs the 450-scene Zenodo download, not run here). |

### 6.2 POSEatSea

| Step | Command | Result |
|---|---|---|
| Dependency install | `pip install -r requirements.txt pytest` | **OK** after relocating the venv to a short path (**same Windows MAX_PATH failure** as SN, this time on `torch\include\ATen\native\transformers\cuda\…\epilogue_rescale_output.h`). Installed torch 2.14.0+cpu, scikit-learn **1.6.1**, streamlit 1.63.0, smp 0.5.0, timm 1.0.29. |
| Import + small-model load | `from poseatsea.registry import get_registry; …get()` | **OK.** AIS autoencoder loads (735 params), trajectory LSTM loads, scaler `n_features_in_=11`. |
| `pytest tests/` (as-committed deps) | `pytest tests/ -q` | **Collection ERROR** — `ModuleNotFoundError: No module named 'fastapi'` (missing from `requirements.txt`, §4.5). 0 tests run. |
| `pytest tests/` (after adding fastapi/uvicorn) | `pytest tests/ -q --continue-on-collection-errors` | **41 passed, 6 failed** in 49 s. |
| — passing | | All AIS-anomaly, trajectory, attribution, scenario-integrity, real-AIS, precomputed-SAR-scene-library, and Streamlit-page-smoke tests. `test_prediction_matches_published_accuracy` (LSTM hits its published error envelope on unseen tracks) **passes**. `test_unscaled_input_gives_a_different_answer` **passes** (scaler necessity proven). |
| — failing | | `test_all_three_models_load_strict`, `test_sar_emits_five_classes`, `test_area_is_measured_at_source_resolution`, `test_precomputed_masks_match_live_inference`, `test_sar_checkpoint_is_trained`, `test_sar_segment_accepts_an_upload` — **all 6 are `FileNotFoundError: best_sar_model.pth`.** The SAR checkpoint is not in the repo and `POSEATSEA_WEIGHTS_REPO` was not set (offline audit). Per the authors, even downloading it would not make the SAR output valid (§2.2). |
| `run.py --check` (preflight) | | **FAILS** — reports `best_sar_model.pth` missing, `main()` returns 1. |
| Streamlit boot | `streamlit run ui/app.py --server.headless true` | **OK.** HTTP 200. Overview / AIS Anomaly / Route Deviation pages render and name the Wakashio as suspect; SAR page would raise on inference. Several `use_container_width` deprecation warnings (Streamlit ≥1.63). |
| FastAPI service | `uvicorn api.main:app` (via TestClient in tests) | **OK for non-SAR routes** (`/health`, `/ais/*`, `/trajectory/*`, `/attribution/*`, `/scenario`). `/sar/segment` fails (no weights). |

### 6.3 Cross-cutting finding — Windows long-path

Both installs failed on the first attempt with `OSError: [Errno 2] No such file or
directory` deep inside `site-packages`, because the combined path length exceeded Windows'
260-character limit. The trigger was the long working/scratch directory, not the repos
themselves. **The merged project must live at a short path** (e.g. its current
`C:\Users\harsh\Desktop\SIH\Oil_Spill_Prototype` is fine at 43 chars) **or enable
`LongPathsEnabled`**, and the README/setup docs should say so. scikit-learn and torch are
both offenders.

---

## 7. Runtime implications and known limitations

### 7.1 Implications of the selected architecture

1. **torch-CPU becomes a required dependency** (~200 MB install) because the merge keeps
   POSEatSea's autoencoder and LSTM. SN alone did not need torch. Still no GPU, still
   offline at run time, but the footprint grows. Drop `segmentation-models-pytorch`,
   `timm`, `huggingface-hub` (SAR-net-only).
2. **No binary model artefacts from SN** — first run (or CI) must execute
   `scripts/setup`-equivalent (~100 s) to produce the SAR/EO classifiers, or the build
   must ship them. The PS AE/LSTM/scaler (~820 KB total) are committed and carried as-is.
3. **One scikit-learn version** across the merged env; retrain SN's classifier under it and
   (ideally) re-pickle + commit the PS scaler under it.
4. **FastAPI-only web tier.** PS's Streamlit console is not carried; its 4-page structure
   informs the SN SPA's view layout. PS's `api/main.py` is a useful reference for the
   `lifespan` pattern and request/response schemas but is superseded by SN's `/api/v1`.
5. **New subsystems have no starting point in either repo:** RBAC + JWT/session auth,
   GeoJSON maritime-boundary polygons + point-in-polygon role mapping, SQLite persistence
   (PostGIS-ready), and the Twilio `SmsProvider`/Mock. All are greenfield Phase-2+ work.
   SN's `LRUStore` (4-method interface) and `config.py` (single source of tunables) are the
   seams to build against.
6. **AIS feature-order adapter needed** — SN's normalised AIS schema → PS's positional
   `FEATURE_ORDER` (with circular `course_diff`) is a small but must-be-exact shim.
7. **AOI gating** — the PS LSTM must be disabled (with fallback to SN's heuristic
   deviation) outside the Mauritius bounding box; the confidence band must reach the API.

### 7.2 Known limitations carried into the merged system

**From SAMUDRA NETRA:**
- Detection/attribution accuracy is validated primarily on SN's **own simulator**. Real
  Sentinel-1 performance is lower and needs a separate decision threshold
  (`REAL_DECISION_THRESHOLD = 0.75` vs simulated `0.30`): on the Zenodo real test set,
  recall 0.80 / specificity 0.83 at the recalibrated point (vs 0.96 / 0.39 at the
  simulated point). Origin reconstruction and attribution have **no real-world ground
  truth** — only the simulator knows the true culprit and release point.
- A vessel that never transmitted AIS cannot be attributed by AIS; the system says so
  rather than substituting a guess.
- Met-ocean fields are **synthesised**, not real ERA5/HYCOM (by design for the prototype;
  `_build_grids()` is the swap seam).
- Attribution is an investigative lead, not proof (every report must carry that caveat).
- SAR detectability has a genuine physical wind window (~5–8 m/s); young/thin slicks
  outside it are legitimately missed and reported as "no spill detected".
- No unit tests — only the end-to-end sweep. Regression safety net must be built.

**From POSEatSea:**
- **The SAR SegFormer checkpoint is non-functional** (untrained decoder → noise) and is
  **not carried into the merge**. If a correctly trained SegFormer is produced later it
  can slot in behind the detector interface as an optional refiner.
- **Trajectory LSTM is region-locked** to the Mauritius AOI + July-2020 speed
  distribution; invalid elsewhere without retraining. Degrades on NW–SE headings and at
  ping cadence slower than ~2.5× 60 s.
- **AIS autoencoder recall is 0.41** — it nominates vessels for review; it can never
  *clear* one. Every surface must say "flagged for review", never "confirmed".
- **No drift/hindcast** in PS itself — this is why SN's `ml/drift` is the backbone.
- `requirements.txt` omits FastAPI/uvicorn; the API tier and its tests have not run from
  PS's own declared dependencies.
- The 5 SAR demo scenes pair a real Sentinel-1 frame with a real vessel's operating area
  for presentation; the geographic pairing is illustrative, not measured (the segmentation
  output — when the model works — is genuine).
- Streamlit ≥1.63 emits `use_container_width` deprecation warnings throughout the console
  (cosmetic; console not carried anyway).

**New / structural:**
- No auth, no RBAC, no DB, no SMS in *either* repo — these are unbuilt.
- Windows MAX_PATH: deep install paths break `pip install` for scikit-learn and torch.
  Document a short project path or `LongPathsEnabled`.
- Merged offline story depends on: SAR classifiers built locally, PS weights committed,
  basemap tiles degrading gracefully, SMS defaulting to Mock. All achievable; none automatic.

---

## 8. One-paragraph summary

Take **SAMUDRA NETRA as the spine** — its FastAPI `/api/v1` service, its CPU-only
classical SAR + RF/GB detector with real IoU and per-feature explainability, its
optical-imagery path, its Lagrangian RK4 drift/hindcast/forecast physics with the
release-time feedback loop, its 6-criterion transparent attribution scorer, its AIS
schema normaliser, its vanilla-JS offline-first UI, and its self-scoring benchmark.
**Graft on from POSEatSea** exactly three things that are real and that SN lacks: the
11-feature **AIS anomaly autoencoder + its mandatory `StandardScaler`** (CLAUDE.md
mandates these) as an extra learned signal into SN's scorer, the **trajectory LSTM**
(AOI-gated) as the route-deviation input, and the **instrumented lazy model-registry
pattern**. Also port PS's `dwell` attribution term and its always-on `drift_caveat`.
**Reject** POSEatSea's Streamlit app (unified-API rule), its `fusion.py` (SN's is a
superset), its AIS parser as the general reader (keep it as one scenario adapter), and
its **SAR SegFormer entirely** — the checkpoint is 105 MB, not in the repo, needs
internet, and its authors state it emits noise. The only new hard dependency is
**torch-CPU** (~200 MB, no GPU). RBAC/JWT, SQLite/PostGIS, GeoJSON boundary polygons, and
the Twilio `SmsProvider`/Mock are greenfield — neither repo has any of them.

---

*Phase 1 complete. Awaiting approval before any code is written.*
