"""
Slick characterisation  (STEP 5).

Turns a detected dark spot into a geographic + geometric description:

  * area (km2 and hectares), perimeter (km)
  * centroid (lat, lon)
  * major / minor axis length (km)
  * orientation of the principal axis (degrees, 0 = North, undirected 0-180)
  * a simplified GeoJSON polygon in WGS-84

Trimmed from OceanTrace ``ml/sar/characterize.py``: the weathering / age /
volume estimates depend on the drift model and arrive in Phase 4.
"""
from __future__ import annotations

import cv2
import numpy as np
from skimage.measure import regionprops

from backend.ml.sar.geo import GeoTransform, LocalFrame, feature, ring_to_geojson


def mask_to_polygons(mask, transform: GeoTransform, simplify_px=2.5, max_rings=4):
    """Contour the mask; return WGS-84 rings ([lon,lat] closed), largest first."""
    m = mask.astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:max_rings]

    rings = []
    for c in contours:
        if len(c) < 4:
            continue
        approx = cv2.approxPolyDP(c, simplify_px, True).reshape(-1, 2)
        if len(approx) < 4:
            approx = c.reshape(-1, 2)
        cols, rows = approx[:, 0].astype(float), approx[:, 1].astype(float)
        lats, lons = transform.pixel_to_ll(cols, rows)
        rings.append(ring_to_geojson(lats, lons))
    return rings


def oriented_extent(mask, transform: GeoTransform, n_bins=18):
    """Length, LOCAL width and principal-axis bearing of the slick.

    PCA on the pixel cloud (a slick is usually curved, so the principal axis
    describes it far better than a bounding box). Width is the median of per-bin
    cross-axis spans along the slick, so curvature does not inflate it.
    """
    ys, xs = np.nonzero(mask)
    if ys.size < 3:
        return 0.0, 0.0, 0.0, (0.0, 0.0)

    lats, lons = transform.pixel_to_ll(xs.astype(float), ys.astype(float))
    lat_c, lon_c = float(np.mean(lats)), float(np.mean(lons))
    frame = LocalFrame(lat_c, lon_c)
    x, y = frame.to_xy(lats, lons)
    pts = np.stack([np.asarray(x), np.asarray(y)], 1)
    pts = pts - pts.mean(0)

    cov = np.cov(pts.T)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]

    proj = pts @ vecs
    along, across = proj[:, 0], proj[:, 1]
    length_m = float(along.max() - along.min())

    edges = np.linspace(along.min(), along.max(), n_bins + 1)
    widths = []
    for i in range(n_bins):
        sel = (along >= edges[i]) & (along < edges[i + 1])
        if sel.sum() < 25:
            continue
        a = across[sel]
        widths.append(float(np.percentile(a, 95) - np.percentile(a, 5)))
    width_m = (
        float(np.median(widths))
        if widths
        else float(np.percentile(across, 95) - np.percentile(across, 5))
    )

    vx, vy = float(vecs[0, 0]), float(vecs[1, 0])
    bearing = (np.degrees(np.arctan2(vx, vy)) + 360.0) % 360.0
    if bearing >= 180.0:
        bearing -= 180.0                      # axis is undirected
    return length_m, width_m, bearing, (lat_c, lon_c)


def characterise(candidate, feats: dict, prediction, transform: GeoTransform,
                 pixel_area_m2: float) -> dict:
    """Geometric + geographic description of one detected slick."""
    mask = candidate.mask
    area_m2 = float(mask.sum()) * float(pixel_area_m2)
    length_m, width_m, bearing, (lat_c, lon_c) = oriented_extent(mask, transform)
    rings = mask_to_polygons(mask, transform)

    # regionprops axis lengths in pixels -> km (a second, ellipse-fit estimate)
    props = regionprops(mask.astype(np.uint8))[0]
    px_m = float(np.mean(transform.pixel_size_m()))
    major_km = float(props.axis_major_length) * px_m / 1000.0
    minor_km = float(props.axis_minor_length) * px_m / 1000.0
    perim_km = float(props.perimeter) * px_m / 1000.0

    geom = (
        {"type": "Polygon", "coordinates": rings[:1]}
        if len(rings) == 1
        else {"type": "MultiPolygon", "coordinates": [[r] for r in rings]}
    )

    props_out = {
        "area_km2": round(area_m2 / 1e6, 4),
        "area_hectares": round(area_m2 / 1e4, 1),
        "perimeter_km": round(perim_km, 3),
        "centroid": [round(lat_c, 5), round(lon_c, 5)],
        "length_km": round(length_m / 1000.0, 3),
        "width_km": round(width_m / 1000.0, 4),
        "major_axis_km": round(major_km, 3),
        "minor_axis_km": round(minor_km, 4),
        "orientation_deg": round(bearing, 1),
        "elongation": round(float(feats.get("elongation", 0.0)), 2),
        "complexity": round(float(feats.get("complexity", 0.0)), 3),
        "solidity": round(float(feats.get("solidity", 0.0)), 3),
        "spreading": round(float(feats.get("spreading", 0.0)), 1),
        "mean_contrast_db": round(float(feats.get("mean_contrast_db", 0.0)), 2),
        "max_contrast_db": round(float(feats.get("max_contrast_db", 0.0)), 2),
        "border_gradient_db_px": round(float(feats.get("border_gradient_db_px", 0.0)), 3),
        "distance_to_coast_km": round(float(feats.get("dist_to_land_km", 999.0)), 2),
        "local_wind_ms": round(float(feats.get("local_wind_ms", 0.0)), 1),
    }
    props_out.update(prediction.as_dict())

    return {
        "geojson": feature(geom, props_out),
        "properties": props_out,
        "centroid": (lat_c, lon_c),
        "length_m": length_m,
        "width_m": width_m,
        "bearing_deg": bearing,
        "area_m2": area_m2,
        "rings": rings,
    }
