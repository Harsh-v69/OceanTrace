"""
Geodesy + raster<->geographic mapping for a north-up SAR scene.

Ported subset of SAMUDRA NETRA ``backend/core/geo.py`` - only what the SAR
detector needs (the drift model's frame helpers arrive with Phase 4).

All bounding boxes are ``[west, south, east, north]`` in degrees.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EARTH_R = 6_371_008.8  # mean Earth radius, metres (IUGG)
_DEG = np.pi / 180.0


def m_per_deg(lat_deg: float) -> tuple[float, float]:
    """Metres per degree of (latitude, longitude) at a given latitude."""
    lat = lat_deg * _DEG
    m_lat = 111_132.92 - 559.82 * np.cos(2 * lat) + 1.175 * np.cos(4 * lat)
    m_lon = 111_412.84 * np.cos(lat) - 93.5 * np.cos(3 * lat)
    return float(m_lat), float(max(m_lon, 1e-6))


def haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance in metres, vectorised."""
    lat1, lon1, lat2, lon2 = map(np.asarray, (lat1, lon1, lat2, lon2))
    p1, p2 = lat1 * _DEG, lat2 * _DEG
    dp = (lat2 - lat1) * _DEG
    dl = (lon2 - lon1) * _DEG
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_R * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


@dataclass
class LocalFrame:
    """East-North metric frame tangent to the sphere at (lat0, lon0)."""

    lat0: float
    lon0: float

    def __post_init__(self):
        self.m_lat, self.m_lon = m_per_deg(self.lat0)

    def to_xy(self, lat, lon):
        lat, lon = np.asarray(lat, float), np.asarray(lon, float)
        return (lon - self.lon0) * self.m_lon, (lat - self.lat0) * self.m_lat

    def to_ll(self, x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        return self.lat0 + y / self.m_lat, self.lon0 + x / self.m_lon


def bbox_center(bbox) -> tuple[float, float]:
    w, s, e, n = bbox
    return (s + n) / 2.0, (w + e) / 2.0


def bbox_size_m(bbox) -> tuple[float, float]:
    w, s, e, n = bbox
    m_lat, m_lon = m_per_deg((s + n) / 2.0)
    return (e - w) * m_lon, (n - s) * m_lat


def bbox_for_scene(center_lat, center_lon, width_px, height_px, pixel_m):
    """Bbox giving exactly ``pixel_m`` ground sampling at this latitude."""
    m_lat, m_lon = m_per_deg(center_lat)
    half_w = width_px * pixel_m / 2.0 / m_lon
    half_h = height_px * pixel_m / 2.0 / m_lat
    return [center_lon - half_w, center_lat - half_h,
            center_lon + half_w, center_lat + half_h]


@dataclass
class GeoTransform:
    """North-up affine mapping between pixel (row, col) and (lat, lon)."""

    bbox: tuple
    width: int
    height: int

    @property
    def lon_res(self) -> float:
        w, s, e, n = self.bbox
        return (e - w) / self.width

    @property
    def lat_res(self) -> float:
        w, s, e, n = self.bbox
        return (n - s) / self.height

    def pixel_to_ll(self, col, row):
        w, s, e, n = self.bbox
        lon = w + (np.asarray(col, float) + 0.5) * self.lon_res
        lat = n - (np.asarray(row, float) + 0.5) * self.lat_res
        return lat, lon

    def ll_to_pixel(self, lat, lon):
        w, s, e, n = self.bbox
        col = (np.asarray(lon, float) - w) / self.lon_res - 0.5
        row = (n - np.asarray(lat, float)) / self.lat_res - 0.5
        return col, row

    def pixel_area_m2(self) -> float:
        wm, hm = bbox_size_m(self.bbox)
        return (wm / self.width) * (hm / self.height)

    def pixel_size_m(self) -> tuple[float, float]:
        wm, hm = bbox_size_m(self.bbox)
        return wm / self.width, hm / self.height


# --------------------------------------------------------------------------- #
# GeoJSON helpers
# --------------------------------------------------------------------------- #
def ring_to_geojson(lats, lons) -> list[list[float]]:
    """Closed ``[lon, lat]`` ring, as GeoJSON requires."""
    coords = [[float(lo), float(la)] for la, lo in zip(lats, lons)]
    if coords and coords[0] != coords[-1]:
        coords.append(coords[0])
    return coords


def feature(geometry: dict, properties: dict | None = None) -> dict:
    return {"type": "Feature", "geometry": geometry, "properties": properties or {}}


def feature_collection(features: list[dict]) -> dict:
    return {"type": "FeatureCollection", "features": features}
