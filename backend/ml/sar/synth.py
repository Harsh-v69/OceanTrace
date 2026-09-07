"""
Physics-based Sentinel-1-like SAR radiometry, standalone.

Ported from OceanTrace ``simulator/sar_scene.py`` but decoupled from the
MetOcean/drift stack: it takes a scalar (or array) wind field directly. Used
ONLY to build the bundled offline demo scene - the real pipeline never imports
it. A detector tuned on this radiometry transfers to real Sentinel-1 data
because the model is the calibrated CMOD-family GMF.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from backend.ml.sar.config import SAR
from backend.ml.sar.geo import GeoTransform, bbox_for_scene


def sea_sigma0_db(u10, incidence_deg=33.0, rel_wind_deg=45.0):
    """C-band VV sea-surface NRCS in dB (CMOD5.N log-linear approximation)."""
    u = np.clip(np.asarray(u10, float), 0.6, 30.0)
    base = -25.5 + 15.8 * np.log10(u) - 0.19 * (incidence_deg - 33.0)
    phi = np.radians(rel_wind_deg)
    modulation = 1.15 * np.cos(phi) + 0.65 * np.cos(2 * phi)
    return base + modulation


def oil_damping_db(concentration, wind_speed, d_max=11.5, k=2.6):
    """Damping (positive dB to SUBTRACT) vs film load and wind.

    Only observable in a wind window: fades below ~2 m/s (specular sea) and
    above ~13 m/s (wave breaking disperses the film).
    """
    c = np.clip(np.asarray(concentration, float), 0.0, None)
    base = d_max * (1.0 - np.exp(-k * c))
    w = np.clip(np.asarray(wind_speed, float), 0.1, 30.0)
    lo = np.clip((w - 1.8) / 1.6, 0.0, 1.0)
    hi = np.clip((13.5 - w) / 3.0, 0.0, 1.0)
    return base * lo * hi


def detectability_window(u10: float) -> tuple[str, str]:
    """Human-readable SAR oil-spill detectability at this wind speed."""
    if u10 < 2.0:
        return "POOR", "Wind too light - specular sea gives no contrast against the film"
    if u10 < 3.0:
        return "MARGINAL", "Low wind; slick contrast reduced and look-alikes common"
    if u10 <= 10.0:
        return "OPTIMAL", "Wind in the 3-10 m/s window where Bragg damping is clearest"
    if u10 <= 13.0:
        return "MARGINAL", "High wind; wave breaking begins to disperse the film"
    return "POOR", "Wind too strong - the slick is mixed into the water column"


def db_to_uint8(db, lo=-26.0, hi=2.0):
    """Standard SAR quicklook: linear dB stretch to 8-bit."""
    x = np.clip((np.asarray(db, np.float32) - lo) / (hi - lo), 0, 1)
    return (x * 255).astype(np.uint8)


def _fractal_field(rng, h, w, octaves=6, persistence=0.55):
    """Fractional Brownian noise in [0, 1]."""
    out = np.zeros((h, w), np.float32)
    amp, total, size = 1.0, 0.0, 4
    for _ in range(octaves):
        layer = rng.random((max(size, 2), max(size, 2))).astype(np.float32)
        layer = cv2.resize(layer, (w, h), interpolation=cv2.INTER_CUBIC)
        out += amp * layer
        total += amp
        amp *= persistence
        size *= 2
    out /= max(total, 1e-6)
    return np.clip(out, 0, 1)


@dataclass
class SynthScene:
    sigma0_db: np.ndarray
    truth_mask: np.ndarray            # true oil pixels
    lookalike_mask: np.ndarray        # true look-alike pixels
    transform: GeoTransform
    meta: dict = field(default_factory=dict)


def _swept_ellipse(h, w, cx, cy, length, width, angle_deg, curve=0.0):
    """A rotated, slightly curved filled ellipse - a plausible drifted slick."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    th = np.radians(angle_deg)
    dx, dy = xx - cx, yy - cy
    a = dx * np.cos(th) + dy * np.sin(th)          # along axis
    b = -dx * np.sin(th) + dy * np.cos(th)         # across axis
    b = b - curve * (a / max(length, 1.0)) ** 2 * width   # gentle bow
    return ((a / (length / 2.0)) ** 2 + (b / (width / 2.0)) ** 2) <= 1.0


def synth_scene(
    *,
    center_lat: float = 15.6,
    center_lon: float = 73.2,
    width: int = 1024,
    height: int = 768,
    pixel_m: float = 60.0,
    wind_ms: float = 7.0,
    incidence_deg: float = 33.0,
    enl: float = SAR.EQUIV_NUM_LOOKS,
    seed: int = 20260306,
    with_oil: bool = True,
    with_lowwind: bool = True,
    with_biogenic: bool = True,
) -> SynthScene:
    """Build one deterministic offline demo scene (no land, open ocean)."""
    rng = np.random.default_rng(seed)
    bbox = bbox_for_scene(center_lat, center_lon, width, height, pixel_m)
    gt = GeoTransform(bbox, width, height)
    h, w = height, width

    # --- clean sea: mean wind + smooth mesoscale variation --------------
    wind = wind_ms * (0.9 + 0.2 * _fractal_field(rng, h, w, octaves=4))
    rel_wind = 40.0 + 20.0 * (_fractal_field(rng, h, w, octaves=3) - 0.5)
    sigma_db = sea_sigma0_db(wind, incidence_deg, rel_wind).astype(np.float32)
    sigma_db += np.linspace(-0.5, 0.5, w)[None, :].astype(np.float32)   # antenna ramp

    truth = np.zeros((h, w), bool)
    la_mask = np.zeros((h, w), bool)

    # --- oil slick: sharp edge, homogeneous dark interior --------------
    if with_oil:
        core = _swept_ellipse(h, w, cx=0.60 * w, cy=0.42 * h,
                              length=0.42 * w, width=0.055 * w,
                              angle_deg=35.0, curve=0.8)
        conc = cv2.GaussianBlur(core.astype(np.float32), (0, 0), 3.0)
        conc = np.clip((conc - 0.25) / 0.45, 0.0, 1.0) ** 0.7
        patch = cv2.GaussianBlur(_fractal_field(rng, h, w, octaves=6), (0, 0), 2.0)
        conc *= 0.75 + 0.4 * patch
        conc = np.clip(conc, 0.0, 1.0)
        sigma_db -= oil_damping_db(conc, wind, d_max=12.0).astype(np.float32)
        truth = conc > 0.16

    # --- look-alike 1: low-wind cell (round, soft edge, weak damping) --
    if with_lowwind:
        yy, xx = np.ogrid[:h, :w]
        cy, cx = 0.30 * h, 0.24 * w
        ry, rx = 0.16 * h, 0.19 * w
        g = np.exp(-(((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2))
        wind_local = wind * (1.0 - 0.42 * g)
        sigma_db += (sea_sigma0_db(wind_local, incidence_deg, rel_wind)
                     - sea_sigma0_db(wind, incidence_deg, rel_wind)).astype(np.float32)
        la_mask |= g > 0.5

    # --- look-alike 2: biogenic film (thin, sinuous, very weak) --------
    if with_biogenic:
        n = 240
        t = np.linspace(0, 1, n)
        x0, y0 = 0.12 * w, 0.72 * h
        xs = x0 + t * 0.6 * w + 40 * np.sin(6 * np.pi * t + 1.3)
        ys = y0 + t * 0.12 * h + 30 * np.cos(5 * np.pi * t + 0.7)
        layer = np.zeros((h, w), np.float32)
        cv2.polylines(layer, [np.stack([xs, ys], 1).astype(np.int32)], False, 1.0, 5)
        layer = cv2.GaussianBlur(layer, (0, 0), 3.0)
        sigma_db -= (3.4 * np.clip(layer, 0, 1)).astype(np.float32)
        la_mask |= layer > 0.25

    # --- linear power + Gamma(L) multiplicative speckle + noise floor --
    lin = np.power(10.0, sigma_db / 10.0).astype(np.float32)
    speckle = rng.gamma(shape=enl, scale=1.0 / enl, size=(h, w)).astype(np.float32)
    lin = lin * speckle
    lin += np.power(10.0, -22.0 / 10.0) * rng.gamma(2.0, 0.5, (h, w)).astype(np.float32)
    out_db = (10.0 * np.log10(np.maximum(lin, 1e-8))).astype(np.float32)

    detect, note = detectability_window(float(np.mean(wind)))
    meta = {
        "width": w, "height": h, "pixel_spacing_m": pixel_m,
        "bbox": list(bbox), "incidence_deg": incidence_deg, "polarisation": "VV",
        "band": "C-band (5.405 GHz)", "platform": "Sentinel-1 (simulated IW GRDH)",
        "enl": enl, "mean_wind_ms": round(float(np.mean(wind)), 2),
        "detectability": detect, "detectability_note": note,
        "scene_size_km": [round(w * pixel_m / 1000.0, 1), round(h * pixel_m / 1000.0, 1)],
        "center": [round(center_lat, 4), round(center_lon, 4)],
        "seed": seed,
    }
    return SynthScene(out_db, truth & ~la_mask, la_mask, gt, meta)
