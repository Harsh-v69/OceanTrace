"""
Scene ingestion  (STEP 1).

Accepts the shapes an operator actually has:

  * a numpy array of calibrated sigma0 (dB or linear)
  * a bundled ``.npz`` (our demo format: sigma0_db, bbox, pixel_m, acquisition)
  * an 8-bit PNG / JPEG quicklook  (dB stretch is inverted; radiometry is
    relative, which is all the detector uses)
  * a GeoTIFF - the raster is read with tifffile and the bounding box is derived
    from the ModelPixelScale / ModelTiepoint GeoTIFF tags when present

and normalises everything to ``SarScene``: a float32 sigma0-dB raster, a
``[west, south, east, north]`` bbox, a pixel size in metres, an acquisition
datetime (UTC) when one can be found, and a :class:`GeoTransform`.
"""
from __future__ import annotations

import io
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from backend.ml.sar.config import SAR
from backend.ml.sar.geo import GeoTransform, bbox_for_scene, bbox_size_m
from backend.ml.sar.preprocess import calibrate, from_uint8_quicklook

# ISO-ish datetime stamp used in Sentinel-1 product names, e.g. 20200725T153000.
_TS_RE = re.compile(r"(\d{8}T\d{6})")
# Default scene centre when neither a bbox nor a centre is supplied. Detection
# still works; only the geographic coordinates are placeholders.
_DEFAULT_CENTER = (15.0, 73.0)


@dataclass
class SarScene:
    sigma0_db: np.ndarray
    bbox: list                       # [west, south, east, north] degrees
    pixel_m: float
    transform: GeoTransform
    acquisition: datetime | None = None
    source: str = "array"
    bbox_source: str = "provided"    # provided | geotiff | center | placeholder
    truth_mask: np.ndarray | None = None
    meta: dict = field(default_factory=dict)

    @property
    def width(self) -> int:
        return int(self.sigma0_db.shape[1])

    @property
    def height(self) -> int:
        return int(self.sigma0_db.shape[0])


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def parse_acquisition(value) -> datetime | None:
    """Coerce a datetime / ISO string / product-name stamp to aware UTC."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    text = str(value)
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        pass
    m = _TS_RE.search(text)
    if m:
        return datetime.strptime(m.group(1), "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
    return None


def _looks_like_db(arr: np.ndarray) -> bool:
    """dB rasters are negative-ish floats; linear power is small positives."""
    if not np.issubdtype(arr.dtype, np.floating):
        return False
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return False
    return float(np.nanmedian(finite)) < 0.0 or float(finite.min()) < -1.0


def _bbox_from_geotiff(page, width: int, height: int) -> list | None:
    """Derive [w,s,e,n] from ModelPixelScale (33550) + ModelTiepoint (33922)."""
    try:
        tags = page.tags
        scale = tags.get("ModelPixelScaleTag")
        tie = tags.get("ModelTiepointTag")
        if scale is None or tie is None:
            return None
        sx, sy = float(scale.value[0]), float(scale.value[1])
        tp = list(tie.value)
        # tie point: (i, j, k, x, y, z) -> pixel (i,j) maps to world (x,y)
        i, j, _, x, y, _ = tp[:6]
        west = x - i * sx
        north = y + j * sy
        east = west + width * sx
        south = north - height * sy
        # Only trust it if it looks like degrees.
        if -180.0 <= west < east <= 180.0 and -90.0 <= south < north <= 90.0:
            return [float(west), float(south), float(east), float(north)]
    except Exception:  # noqa: BLE001 - a missing/odd tag just means "no bbox"
        return None
    return None


def _finalise(
    db: np.ndarray,
    *,
    bbox: list | None,
    pixel_m: float | None,
    center: tuple[float, float] | None,
    acquisition,
    source: str,
    bbox_source: str,
    truth_mask=None,
    meta: dict | None = None,
) -> SarScene:
    db = np.asarray(db, np.float32)
    h, w = db.shape

    if bbox is None:
        cy, cx = center or _DEFAULT_CENTER
        px = float(pixel_m or SAR.NOMINAL_PIXEL_M)
        bbox = bbox_for_scene(cy, cx, w, h, px)
        bbox_source = bbox_source if bbox_source != "provided" else (
            "center" if center else "placeholder"
        )

    transform = GeoTransform(tuple(bbox), w, h)
    if pixel_m is None:
        wm, hm = bbox_size_m(bbox)
        pixel_m = float(np.mean([wm / w, hm / h]))

    return SarScene(
        sigma0_db=db,
        bbox=[float(v) for v in bbox],
        pixel_m=float(pixel_m),
        transform=transform,
        acquisition=parse_acquisition(acquisition),
        source=source,
        bbox_source=bbox_source,
        truth_mask=None if truth_mask is None else np.asarray(truth_mask).astype(bool),
        meta=meta or {},
    )


# --------------------------------------------------------------------------- #
# Entry points
# --------------------------------------------------------------------------- #
def from_array(
    arr,
    *,
    bbox: list | None = None,
    pixel_m: float | None = None,
    center: tuple[float, float] | None = None,
    acquisition=None,
    input_is_db: bool | None = None,
    truth_mask=None,
    meta: dict | None = None,
) -> SarScene:
    a = np.asarray(arr)
    if a.ndim == 3:
        a = a.mean(axis=2)
    if a.dtype == np.uint8:
        db = from_uint8_quicklook(a)
    else:
        is_db = _looks_like_db(a) if input_is_db is None else bool(input_is_db)
        db, _ = calibrate(a.astype(np.float32), input_is_db=is_db)
    return _finalise(
        db, bbox=bbox, pixel_m=pixel_m, center=center, acquisition=acquisition,
        source="array", bbox_source="provided" if bbox is not None else "provided",
        truth_mask=truth_mask, meta=meta,
    )


def from_npz(path, *, meta: dict | None = None) -> SarScene:
    d = np.load(Path(path), allow_pickle=True)
    db = np.asarray(d["sigma0_db"], np.float32)
    bbox = list(d["bbox"]) if "bbox" in d.files else None
    pixel_m = float(d["pixel_m"]) if "pixel_m" in d.files else None
    acq = str(d["acquisition"]) if "acquisition" in d.files else None
    truth = d["truth_mask"] if "truth_mask" in d.files else None
    extra = dict(meta or {})
    if "meta" in d.files:
        try:
            raw = d["meta"].item()
            extra.update(raw if isinstance(raw, dict) else json.loads(str(raw)))
        except Exception:  # noqa: BLE001
            pass
    return _finalise(
        db, bbox=bbox, pixel_m=pixel_m, center=None, acquisition=acq,
        source="npz", bbox_source="provided" if bbox is not None else "placeholder",
        truth_mask=truth, meta=extra,
    )


def from_png(
    data_or_path,
    *,
    bbox: list | None = None,
    pixel_m: float | None = None,
    center: tuple[float, float] | None = None,
    acquisition=None,
    meta: dict | None = None,
) -> SarScene:
    import cv2

    if isinstance(data_or_path, (bytes, bytearray)):
        buf = np.frombuffer(data_or_path, np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
        name = ""
    else:
        p = Path(data_or_path)
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        name = p.name
    if img is None:
        raise ValueError("Could not decode image - expected a readable PNG/JPEG.")
    db = from_uint8_quicklook(img)
    return _finalise(
        db, bbox=bbox, pixel_m=pixel_m, center=center,
        acquisition=acquisition or name, source="png",
        bbox_source="provided" if bbox is not None else "placeholder", meta=meta,
    )


def from_geotiff(
    data_or_path,
    *,
    bbox: list | None = None,
    pixel_m: float | None = None,
    center: tuple[float, float] | None = None,
    acquisition=None,
    meta: dict | None = None,
) -> SarScene:
    import tifffile

    name = ""
    if isinstance(data_or_path, (bytes, bytearray)):
        tif = tifffile.TiffFile(io.BytesIO(data_or_path))
    else:
        p = Path(data_or_path)
        name = p.name
        tif = tifffile.TiffFile(str(p))

    with tif:
        page = tif.pages[0]
        arr = page.asarray()
        if arr.ndim == 3:
            arr = arr[..., 0] if arr.shape[-1] <= 4 else arr[0]
        derived = bbox if bbox is not None else _bbox_from_geotiff(
            page, arr.shape[1], arr.shape[0]
        )
        bbox_source = "provided" if bbox is not None else (
            "geotiff" if derived is not None else "placeholder"
        )

    if arr.dtype == np.uint8:
        db = from_uint8_quicklook(arr)
    else:
        is_db = _looks_like_db(arr.astype(np.float32))
        db, _ = calibrate(arr.astype(np.float32), input_is_db=is_db)

    return _finalise(
        db, bbox=derived, pixel_m=pixel_m, center=center,
        acquisition=acquisition or name, source="geotiff",
        bbox_source=bbox_source, meta=meta,
    )


def load_scene(source, **kwargs) -> SarScene:
    """Dispatch on the input type / file extension."""
    if isinstance(source, SarScene):
        return source
    if isinstance(source, np.ndarray):
        return from_array(source, **kwargs)
    if isinstance(source, (bytes, bytearray)):
        head = bytes(source[:4])
        if head[:2] in (b"II", b"MM"):
            return from_geotiff(source, **kwargs)
        return from_png(source, **kwargs)

    path = Path(source)
    ext = path.suffix.lower()
    if ext == ".npz":
        return from_npz(path, meta=kwargs.get("meta"))
    if ext in (".tif", ".tiff"):
        return from_geotiff(path, **kwargs)
    if ext in (".png", ".jpg", ".jpeg", ".bmp"):
        return from_png(path, **kwargs)
    if ext == ".npy":
        return from_array(np.load(path), **kwargs)
    raise ValueError(f"Unsupported SAR input: {path.name!r} ({ext})")
