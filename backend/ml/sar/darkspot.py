"""
Dark-spot segmentation  (STEP 3, part 1).

Ported from OceanTrace ``ml/sar/darkspot.py``.

Oil suppresses Bragg scattering, so a slick is a DARK anomaly against the
wind-roughened sea. Finding it is a *local-contrast* problem, not a global
threshold problem.

1. Background estimation by multi-scale grayscale morphological closing (on a
   decimated copy for speed - the whole chain drops from ~8 s to well under 1 s).
2. Hysteresis thresholding: a high-contrast seed marks confident cores, a low
   threshold defines the growable region; only components containing a seed
   survive - recovering faint feathered edges without letting speckle create
   blobs.
3. Marker-controlled watershed splits a hysteresis mask back into one region per
   confident core, so a slick touching a low-wind cell is not reported as one
   sprawling blob the classifier (correctly) refuses to call oil.
4. Morphological cleanup + area gating + connected-component labelling.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage as ndi

from backend.ml.sar.config import SAR

# Background-closing scales in pixels. SAR sigma0 carries large wind-driven
# gradients, so the background estimate must reach across them.
BACKGROUND_SCALES = {
    "SAR": (20, 40, 80, 170),
    "EO": (8, 16, 32, 60),
}


def scales_for(sensor: str = "SAR"):
    return BACKGROUND_SCALES.get(str(sensor).upper(), BACKGROUND_SCALES["SAR"])


def _disk(radius: int):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)


def estimate_background(db, scales=(20, 40, 80, 170), blur=9.0, downsample=4):
    """Multi-scale morphological background (the 'slick-free' sea)."""
    img = np.asarray(db, np.float32)
    h, w = img.shape
    ds = max(int(downsample), 1)
    small = (
        cv2.resize(img, (max(w // ds, 8), max(h // ds, 8)), interpolation=cv2.INTER_AREA)
        if ds > 1
        else img
    )
    best = None
    for r in scales:
        rr = max(int(round(r / ds)), 1)
        closed = cv2.morphologyEx(small, cv2.MORPH_CLOSE, _disk(rr))
        best = closed if best is None else np.maximum(best, closed)
    best = cv2.GaussianBlur(best, (0, 0), max(blur / ds, 0.8))
    if ds > 1:
        best = cv2.resize(best, (w, h), interpolation=cv2.INTER_CUBIC)
    return best.astype(np.float32)


def hysteresis(low_mask, high_mask):
    """Keep only low-threshold components that contain a high-threshold seed."""
    lab, n = ndi.label(low_mask)
    if n == 0:
        return np.zeros_like(low_mask, bool)
    keep = np.zeros(n + 1, bool)
    seeded = np.unique(lab[high_mask])
    keep[seeded[seeded > 0]] = True
    return keep[lab]


def split_merged(mask, seeds, contrast, min_seed_px=60):
    """Split a hysteresis mask into one region per confident core (watershed)."""
    from skimage.segmentation import watershed

    mask = np.asarray(mask, bool)
    if not mask.any():
        return np.zeros(mask.shape, np.int32), 0

    markers, n_seeds = ndi.label(np.asarray(seeds, bool) & mask)
    if n_seeds <= 1:
        return ndi.label(mask)

    sizes = np.bincount(markers.ravel())
    keep = np.nonzero(sizes >= int(min_seed_px))[0]
    keep = keep[keep > 0]
    if keep.size <= 1:
        return ndi.label(mask)
    remap = np.zeros(sizes.size, np.int32)
    remap[keep] = np.arange(1, keep.size + 1, dtype=np.int32)
    markers = remap[markers]

    labels = watershed(-np.asarray(contrast, np.float32), markers, mask=mask)
    return labels.astype(np.int32), int(labels.max())


@dataclass
class DarkSpot:
    label: int
    mask: np.ndarray          # full-scene boolean mask
    area_px: int
    bbox: tuple               # (min_row, min_col, max_row, max_col)
    centroid_rc: tuple


def adaptive_offset(db, sea_mask=None, k=1.8, background=None):
    """Contrast threshold from the scene's own speckle statistics.

    Sits k robust standard deviations below the background, so it tightens in
    calm seas and relaxes in rough ones. Kept low enough that weak look-alikes
    ARE picked up as candidates - rejecting them is the classifier's job.
    """
    img = np.asarray(db, np.float32)
    sea = np.ones(img.shape, bool) if sea_mask is None else np.asarray(sea_mask, bool)
    bg = estimate_background(img) if background is None else background
    resid = (bg - img)[sea]
    if resid.size == 0:
        return SAR.ADAPTIVE_OFFSET_DB
    med = float(np.median(resid))
    mad = float(np.median(np.abs(resid - med))) * 1.4826
    return float(np.clip(med + k * mad, 1.2, 6.0))


def segment(db, sea_mask=None, offset_low=None, offset_high=None,
            min_area_px=None, max_area_frac=None, background=None,
            open_px=3, close_px=7):
    """Segment dark spots. Returns (candidates, diagnostics_dict)."""
    img = np.asarray(db, np.float32)
    h, w = img.shape
    sea = np.ones((h, w), bool) if sea_mask is None else np.asarray(sea_mask, bool)

    lo = SAR.ADAPTIVE_OFFSET_DB if offset_low is None else float(offset_low)
    hi = (lo + 1.4) if offset_high is None else float(offset_high)
    min_area = int(min_area_px if min_area_px is not None else SAR.MIN_SPILL_AREA_PX)
    max_frac = float(
        max_area_frac if max_area_frac is not None else SAR.MAX_SPILL_AREA_FRAC
    )

    bg = estimate_background(img) if background is None else background
    contrast = bg - img                      # positive where darker than sea

    m_low = (contrast > lo) & sea
    m_high = (contrast > hi) & sea
    m = hysteresis(m_low, m_high)

    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, _disk(open_px))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, _disk(close_px))
    m = ndi.binary_fill_holes(m.astype(bool))

    lab, n = split_merged(m, m_high, contrast)
    cands, rejected = [], {"too_small": 0, "too_large": 0}
    max_area = max_frac * h * w
    for i in range(1, n + 1):
        region = lab == i
        area = int(region.sum())
        if area < min_area:
            rejected["too_small"] += 1
            continue
        if area > max_area:
            rejected["too_large"] += 1
            continue
        ys, xs = np.nonzero(region)
        cands.append(
            DarkSpot(
                label=i,
                mask=region,
                area_px=area,
                bbox=(int(ys.min()), int(xs.min()), int(ys.max()) + 1, int(xs.max()) + 1),
                centroid_rc=(float(ys.mean()), float(xs.mean())),
            )
        )

    cands.sort(key=lambda c: -c.area_px)
    diag = {
        "threshold_low_db": round(lo, 2),
        "threshold_high_db": round(hi, 2),
        "components_found": int(n),
        "candidates_kept": len(cands),
        "rejected_too_small": rejected["too_small"],
        "rejected_too_large": rejected["too_large"],
        "min_area_px": min_area,
        "dark_pixel_fraction": round(float(m.mean()), 5),
    }
    return cands, {"background": bg, "contrast": contrast, "mask": m, "diagnostics": diag}
