"""
SAR pre-processing chain  (STEP 2).

Ported from SAMUDRA NETRA ``ml/sar/preprocess.py``. Mirrors the ESA SNAP graph
applied to a Sentinel-1 GRD product before oil-spill analysis:

    Apply-Orbit-File -> Thermal-Noise-Removal -> Calibration (sigma0)
    -> Speckle-Filter (Refined Lee) -> Terrain-Correction -> Land-Sea-Mask

Steps needing external data (precise orbits, DEM) are no-ops - the prototype
works on already-geocoded sigma0. Everything that affects detection (dB
calibration, speckle suppression, land masking) is implemented.

CPU notes
---------
* Refined Lee is expressed entirely as OpenCV ``boxFilter`` passes (separable,
  SIMD) - no per-pixel Python.
* ``estimate_enl`` scans on a stride so it is O(scene / patch^2), not O(scene).
Everything here runs comfortably under ~0.3 s on a 1280x1024 scene on one core.
"""
from __future__ import annotations

import cv2
import numpy as np

from backend.ml.sar.config import SAR


# --------------------------------------------------------------------------- #
# Calibration helpers
# --------------------------------------------------------------------------- #
def db_to_linear(db):
    return np.power(10.0, np.asarray(db, np.float32) / 10.0)


def linear_to_db(lin, floor=1e-8):
    return 10.0 * np.log10(np.maximum(np.asarray(lin, np.float32), floor))


def calibrate(raw, input_is_db=True):
    """Return (sigma0_db, sigma0_linear) from whatever the reader produced."""
    a = np.asarray(raw, np.float32)
    if input_is_db:
        return a, db_to_linear(a)
    return linear_to_db(a), a


def from_uint8_quicklook(img, lo=-26.0, hi=2.0):
    """Invert the standard dB quicklook stretch for imported 8-bit imagery.

    Absolute radiometry cannot be recovered from a PNG/JPEG quicklook, but oil
    detection is a *relative* contrast problem, so mapping the byte range back
    onto a plausible dB span preserves everything the detector uses.
    """
    x = np.asarray(img, np.float32)
    if x.ndim == 3:
        x = cv2.cvtColor(x.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    return lo + (x / 255.0) * (hi - lo)


# --------------------------------------------------------------------------- #
# Refined-Lee adaptive speckle filter
# --------------------------------------------------------------------------- #
def refined_lee(sigma0_linear, window=None, enl=None, cmax_factor=1.73):
    """Refined Lee adaptive speckle filter. Operates on linear intensity.

        Ci <= Cu          -> homogeneous  -> local mean
        Cu < Ci < Cmax    -> textured     -> mean + W*(pixel - mean)
        Ci >= Cmax        -> point target -> keep the pixel (preserves ships)
    """
    win = int(window or SAR.LEE_WINDOW)
    win = win if win % 2 else win + 1
    L = float(enl or SAR.EQUIV_NUM_LOOKS)

    img = np.asarray(sigma0_linear, np.float32)
    k = (win, win)
    mean = cv2.boxFilter(img, -1, k, normalize=True, borderType=cv2.BORDER_REFLECT)
    mean_sq = cv2.boxFilter(img * img, -1, k, normalize=True,
                            borderType=cv2.BORDER_REFLECT)
    var = np.maximum(mean_sq - mean * mean, 0.0)

    cu = 1.0 / np.sqrt(L)
    cmax = cu * cmax_factor
    with np.errstate(divide="ignore", invalid="ignore"):
        ci = np.sqrt(var) / np.maximum(mean, 1e-9)

    w = 1.0 - (cu * cu) / np.maximum(ci * ci, 1e-9)
    w = np.clip(w, 0.0, 1.0)
    out = mean + w * (img - mean)

    out = np.where(ci <= cu, mean, out)          # homogeneous -> smooth
    out = np.where(ci >= cmax, img, out)         # point target -> preserve
    return out.astype(np.float32)


def enhanced_lee_db(sigma0_db, window=None, enl=None):
    """Filter in linear power, return dB."""
    lin = db_to_linear(sigma0_db)
    return linear_to_db(refined_lee(lin, window, enl))


def estimate_enl(sigma0_linear, patch=64):
    """ENL = (mean/sd)^2 over the most homogeneous patch found on a stride."""
    img = np.asarray(sigma0_linear, np.float32)
    h, w = img.shape
    best, best_cv = None, np.inf
    for r in range(0, max(h - patch, 1), max(patch, 1)):
        for c in range(0, max(w - patch, 1), max(patch, 1)):
            blk = img[r:r + patch, c:c + patch]
            m, s = blk.mean(), blk.std()
            if m <= 1e-9:
                continue
            cv_ = s / m
            if cv_ < best_cv:
                best_cv, best = cv_, blk
    if best is None or best_cv <= 1e-9:
        return float(SAR.EQUIV_NUM_LOOKS)
    return float(np.clip(1.0 / (best_cv ** 2), 0.5, 100.0))


# --------------------------------------------------------------------------- #
# Land / sea mask
# --------------------------------------------------------------------------- #
def land_mask(sigma0_db, provided_mask=None, threshold_db=None,
              min_area_px=4000, close_px=15):
    """Radiometric land-sea mask (land is far brighter/rougher at C-band).

    Uses a median + MAD bright-tail threshold, not Otsu: a large slick is a
    strong dark mode that would make Otsu label the whole sea as land.
    """
    if provided_mask is not None:
        return np.asarray(provided_mask).astype(bool)

    db = np.asarray(sigma0_db, np.float32)
    smooth = cv2.GaussianBlur(db, (0, 0), 6.0)

    if threshold_db is not None:
        thr = float(threshold_db)
    else:
        med = float(np.median(smooth))
        mad = float(np.median(np.abs(smooth - med))) * 1.4826
        if mad < 1e-3:
            return np.zeros(db.shape, bool)
        thr = med + 2.5 * mad
        bright = smooth > thr
        if bright.mean() < 0.003:
            return np.zeros(db.shape, bool)
        if float(smooth[bright].mean() - med) < 3.0:
            return np.zeros(db.shape, bool)

    m = (smooth > thr).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((close_px, close_px), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))

    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    out = np.zeros_like(m, bool)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area_px:
            out |= (lab == i)
    return out


def coastal_buffer(land, buffer_px=6):
    """Dilate land - the surf zone is radiometrically unreliable."""
    if not np.any(land):
        return np.zeros_like(land, bool)
    k = np.ones((2 * buffer_px + 1, 2 * buffer_px + 1), np.uint8)
    return cv2.dilate(land.astype(np.uint8), k).astype(bool)


# --------------------------------------------------------------------------- #
# Full chain
# --------------------------------------------------------------------------- #
def preprocess(sigma0_db, provided_land=None, enl=None, window=None):
    """Return a dict every downstream stage consumes."""
    db = np.asarray(sigma0_db, np.float32)
    lin = db_to_linear(db)
    enl_est = estimate_enl(lin)
    filt_lin = refined_lee(lin, window=window, enl=enl or enl_est)
    filt_db = linear_to_db(filt_lin)

    land = land_mask(filt_db, provided_land)
    land_buf = coastal_buffer(land)
    sea = ~land_buf

    sea_vals = filt_db[sea]
    stats = {
        "enl_estimated": round(float(enl_est), 2),
        "sea_mean_db": round(float(sea_vals.mean()), 2) if sea_vals.size else None,
        "sea_std_db": round(float(sea_vals.std()), 2) if sea_vals.size else None,
        "sea_p05_db": round(float(np.percentile(sea_vals, 5)), 2) if sea_vals.size else None,
        "sea_p95_db": round(float(np.percentile(sea_vals, 95)), 2) if sea_vals.size else None,
        "land_fraction": round(float(land.mean()), 4),
        "speckle_filter": f"Refined Lee {window or SAR.LEE_WINDOW}x{window or SAR.LEE_WINDOW}",
    }
    return {"db": filt_db, "raw_db": db, "linear": filt_lin,
            "land": land, "sea": sea, "stats": stats}
