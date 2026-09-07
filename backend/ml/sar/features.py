"""
Candidate feature extraction  (STEP 3, part 2).

Ported verbatim from OceanTrace ``ml/sar/features.py`` - the 30 features and
their ORDER are fixed by the trained RF+GB ensemble and must not change.

GEOMETRY   - oil from a moving vessel is long, narrow, irregular; a low-wind
             cell is broad, round, convex.
BACKSCATTER- the border gradient is the strongest single feature: an oil film
             has a physical boundary (sigma0 steps within a few px); a low-wind
             cell's edge is diffuse. Contrast magnitude: biogenic 2-5 dB,
             mineral oil 6-13 dB.
TEXTURE    - GLCM statistics inside the spot; oil interiors are smoother.
CONTEXT    - distance to land and local wind speed.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import regionprops

FEATURE_NAMES = [
    # -- geometry (13)
    "area_km2", "perimeter_km", "complexity", "form_factor", "elongation",
    "eccentricity", "solidity", "extent", "spreading", "asymmetry",
    "length_km", "width_km", "n_holes",
    # -- backscatter (11)
    "mean_contrast_db", "max_contrast_db", "std_inside_db", "p10_contrast_db",
    "p90_contrast_db", "border_gradient_db_px", "border_gradient_std",
    "power_to_mean_ratio", "contrast_ratio_linear", "bg_std_db",
    "inside_bg_std_ratio",
    # -- texture (4)
    "glcm_contrast", "glcm_homogeneity", "glcm_energy", "glcm_correlation",
    # -- context (2)
    "dist_to_land_km", "local_wind_ms",
]


def _safe(x, default=0.0):
    x = float(x)
    return x if np.isfinite(x) else default


def _border_ring(mask, width=3):
    k = np.ones((3, 3), np.uint8)
    m = mask.astype(np.uint8)
    outer = cv2.dilate(m, k, iterations=width).astype(bool) & ~mask
    inner = mask & ~cv2.erode(m, k, iterations=width).astype(bool)
    return inner, outer


def _glcm_features(db, mask, bbox, levels=32):
    r0, c0, r1, c1 = bbox
    patch = db[r0:r1, c0:c1]
    sub = mask[r0:r1, c0:c1]
    if patch.size < 16 or sub.sum() < 16:
        return 0.0, 0.0, 0.0, 0.0

    vals = patch[sub]
    lo, hi = float(np.percentile(vals, 2)), float(np.percentile(vals, 98))
    if hi - lo < 1e-6:
        return 0.0, 1.0, 1.0, 0.0
    q = np.clip((patch - lo) / (hi - lo), 0, 1)
    q = (q * (levels - 1)).astype(np.uint8)
    q[~sub] = 0

    glcm = graycomatrix(
        q,
        distances=[1, 3],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=levels,
        symmetric=True,
        normed=True,
    )
    glcm = glcm[1:, 1:, :, :]
    s = glcm.sum(axis=(0, 1), keepdims=True)
    glcm = glcm / np.maximum(s, 1e-12)
    return (
        _safe(graycoprops(glcm, "contrast").mean()),
        _safe(graycoprops(glcm, "homogeneity").mean()),
        _safe(graycoprops(glcm, "energy").mean()),
        _safe(graycoprops(glcm, "correlation").mean()),
    )


def scene_context(db, land_mask=None):
    """Precompute scene-wide fields every candidate needs (compute once)."""
    a = np.asarray(db, np.float32)
    gy, gx = np.gradient(a)
    ctx = {"gmag": np.hypot(gy, gx)}
    if land_mask is not None and np.any(land_mask):
        ctx["dist_land_px"] = ndi.distance_transform_edt(~np.asarray(land_mask, bool))
    else:
        ctx["dist_land_px"] = None

    valid = np.ones(a.shape, bool) if land_mask is None else ~np.asarray(land_mask, bool)
    v = a[valid]
    ctx["sea_n"] = int(v.size)
    ctx["sea_s1"] = float(v.sum(dtype=np.float64))
    ctx["sea_s2"] = float(np.square(v, dtype=np.float64).sum(dtype=np.float64))
    ctx["valid"] = valid
    return ctx


def extract(candidate, db, background, pixel_size_m, land_mask=None,
            wind_field=None, ctx=None):
    """Full 30-feature vector for one candidate. Returns (vector, named_dict)."""
    if ctx is None:
        ctx = scene_context(db, land_mask)
    mask = candidate.mask
    px_m = float(np.mean(pixel_size_m))
    px_km2 = (px_m / 1000.0) ** 2

    # ---- geometry ------------------------------------------------------
    props = regionprops(mask.astype(np.uint8))[0]
    area_px = float(props.area)
    area_km2 = area_px * px_km2
    perim_px = float(props.perimeter) or 1.0
    perim_km = perim_px * px_m / 1000.0

    complexity = perim_px / (2.0 * np.sqrt(np.pi * max(area_px, 1.0)))
    form_factor = 4.0 * np.pi * area_px / max(perim_px ** 2, 1e-9)
    major = float(props.axis_major_length) or 1.0
    minor = float(props.axis_minor_length) or 1.0
    elongation = major / max(minor, 1e-6)
    eccentricity = float(props.eccentricity)
    solidity = float(props.solidity)
    extent = float(props.extent)

    lam = np.sort(np.asarray(props.inertia_tensor_eigvals, float))[::-1]
    spreading = 100.0 * lam[1] / max(lam[0] + lam[1], 1e-12)

    ys, xs = np.nonzero(mask)
    cy, cx = ys.mean(), xs.mean()
    asymmetry = float(
        np.abs(np.mean(((ys - cy) ** 3 + (xs - cx) ** 3))) ** (1 / 3.0)
        / max(np.sqrt(area_px), 1.0)
    )
    n_holes = int(max(1 - props.euler_number, 0))

    # ---- backscatter -------------------------------------------------
    inside = db[mask]
    bg_in = background[mask]
    contrast = bg_in - inside
    mean_contrast = float(np.mean(contrast))
    max_contrast = float(np.percentile(contrast, 98))
    std_inside = float(np.std(inside))
    p10_c = float(np.percentile(contrast, 10))
    p90_c = float(np.percentile(contrast, 90))

    inner, outer = _border_ring(mask, width=3)
    if inner.sum() > 4 and outer.sum() > 4:
        step = float(np.mean(db[outer]) - np.mean(db[inner]))
        border_grad = step / 6.0
        border_grad_std = float(np.std(ctx["gmag"][inner | outer]))
    else:
        border_grad, border_grad_std = 0.0, 0.0

    lin_in = np.power(10.0, inside / 10.0)
    lin_bg = np.power(10.0, bg_in / 10.0)
    ptm = float(np.std(lin_in) / max(np.mean(lin_in), 1e-12))
    contrast_ratio = float(np.mean(lin_in) / max(np.mean(lin_bg), 1e-12))

    sel = mask if ctx.get("valid") is None else (mask & ctx["valid"])
    vals = db[sel]
    n = ctx.get("sea_n", 0) - int(vals.size)
    if n > 1:
        s1 = ctx["sea_s1"] - float(vals.sum(dtype=np.float64))
        s2 = ctx["sea_s2"] - float(np.square(vals, dtype=np.float64).sum(dtype=np.float64))
        var = max(s2 / n - (s1 / n) ** 2, 0.0)
        bg_std = float(np.sqrt(var))
    else:
        bg_std = 1.0
    inside_bg_std_ratio = std_inside / max(bg_std, 1e-6)

    # ---- texture -----------------------------------------------------
    g_con, g_hom, g_ene, g_cor = _glcm_features(db, mask, candidate.bbox)

    # ---- context ---------------------------------------------------
    dl = ctx.get("dist_land_px")
    dist_land_km = float(dl[mask].min() * px_m / 1000.0) if dl is not None else 999.0
    local_wind = float(np.mean(wind_field[mask])) if wind_field is not None else 7.0

    values = [
        area_km2, perim_km, complexity, form_factor, elongation,
        eccentricity, solidity, extent, spreading, asymmetry,
        major * px_m / 1000.0, minor * px_m / 1000.0, n_holes,
        mean_contrast, max_contrast, std_inside, p10_c, p90_c,
        border_grad, border_grad_std, ptm, contrast_ratio, bg_std,
        inside_bg_std_ratio,
        g_con, g_hom, g_ene, g_cor,
        dist_land_km, local_wind,
    ]
    values = [_safe(v) for v in values]
    return np.array(values, np.float32), dict(zip(FEATURE_NAMES, values))


def extract_batch(candidates, db, background, pixel_size_m, land_mask=None,
                  wind_field=None):
    ctx = scene_context(db, land_mask)
    vecs, dicts = [], []
    for c in candidates:
        v, d = extract(c, db, background, pixel_size_m, land_mask, wind_field, ctx)
        vecs.append(v)
        dicts.append(d)
    if not vecs:
        return np.zeros((0, len(FEATURE_NAMES)), np.float32), []
    return np.vstack(vecs), dicts
