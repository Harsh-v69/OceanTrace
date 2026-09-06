"""
AIS ingestion, cleaning and track reconstruction  (attribution, step 1).

Ported from SAMUDRA NETRA ``ml/ais/tracks.py`` and rewritten pandas-free
(pure numpy + stdlib) - pandas arrives with the POSEatSea / torch phase.

Raw AIS is messy in ways that matter for attribution: irregular cadence,
outlier fixes, mis-keyed MMSIs, default sentinels (COG/heading = 511), and
satellite dropouts. Before a vessel can be scored we need a continuous
``position(t)`` we can query at arbitrary instants - the hindcast's release
times will not line up with AIS timestamps - and a record of WHERE it
interpolated versus had data, because a transmission gap is itself evidence.

Processing:
  * timestamp normalisation (ISO-8601 / epoch / "YYYY-MM-DD HH:MM:SS") -> aware UTC
  * duplicate removal ((mmsi, timestamp) and consecutive identical fixes)
  * invalid-coordinate removal (out of range, NaN, null-island 0/0)
  * impossible-speed filtering (implied inter-ping speed > MAX_PLAUSIBLE_SOG_KN)
  * track segmentation on long transmission gaps
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

from backend.ml.ais.schema import HEADER_ALIASES, flag_from_mmsi, nav_status_label, type_group
from backend.ml.attribution.geo import haversine_m, initial_bearing_deg

MAX_PLAUSIBLE_SOG_KN = 45.0       # above this an implied speed is a GPS/decode error
HARD_SOG_CAP_KN = 80.0           # a reported SOG above this is a decode error
GAP_FLAG_MINUTES = 20.0         # silence longer than this is a recorded "gap"
GAP_SPLIT_MINUTES = 360.0       # silence longer than this splits the track into segments
_KN_PER_MS = 1.0 / 0.514444


# --------------------------------------------------------------------------- #
# Record normalisation
# --------------------------------------------------------------------------- #
def _canonical_key(key: str) -> str:
    k = str(key).strip().lower().replace(" ", "_")
    return HEADER_ALIASES.get(k, k)


def _parse_timestamp(value) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        # epoch seconds (or ms if implausibly large)
        v = float(value)
        if v > 1e11:
            v /= 1000.0
        try:
            return datetime.fromtimestamp(v, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    text = str(value).strip()
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y/%m/%d %H:%M:%S",
                "%d-%m-%Y %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _to_float(value):
    try:
        f = float(value)
        return f if np.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def clean_records(records, *, drop_null_island: bool = True) -> tuple[list[dict], dict]:
    """Normalise a list of raw AIS dicts. Returns ``(clean_rows, report)``.

    Impossible-speed filtering is per-vessel and time-ordered, so it happens in
    :func:`build_tracks`; ``report`` here counts the row-level drops.
    """
    report = {
        "input": 0, "kept": 0,
        "dropped_no_mmsi": 0, "dropped_unparseable_time": 0,
        "dropped_bad_coords": 0, "dropped_null_island": 0,
        "dropped_duplicates": 0,
    }
    seen: set[tuple] = set()
    out: list[dict] = []

    for raw in records or []:
        report["input"] += 1
        row = {_canonical_key(k): v for k, v in dict(raw).items()}

        mmsi = row.get("mmsi")
        try:
            mmsi = int(float(mmsi))
        except (TypeError, ValueError):
            report["dropped_no_mmsi"] += 1
            continue

        ts = _parse_timestamp(row.get("timestamp"))
        if ts is None:
            report["dropped_unparseable_time"] += 1
            continue

        lat = _to_float(row.get("latitude"))
        lon = _to_float(row.get("longitude"))
        if lat is None or lon is None or not (-90.0 <= lat <= 90.0) or not (-180.0 <= lon <= 180.0):
            report["dropped_bad_coords"] += 1
            continue
        if drop_null_island and abs(lat) < 1e-6 and abs(lon) < 1e-6:
            report["dropped_null_island"] += 1
            continue

        key = (mmsi, ts.isoformat(), round(lat, 6), round(lon, 6))
        if key in seen:
            report["dropped_duplicates"] += 1
            continue
        seen.add(key)

        sog = _to_float(row.get("sog"))
        cog = _to_float(row.get("cog"))
        heading = _to_float(row.get("heading"))
        rot = _to_float(row.get("rot"))
        out.append({
            "mmsi": mmsi,
            "timestamp": ts,
            "latitude": lat,
            "longitude": lon,
            "sog": sog if (sog is not None and 0.0 <= sog <= HARD_SOG_CAP_KN) else None,
            "cog": cog if (cog is not None and 0.0 <= cog < 360.0) else None,
            "heading": heading if (heading is not None and 0.0 <= heading < 360.0) else None,
            "rot": rot if (rot is not None and -128.0 <= rot <= 128.0) else None,
            "nav_status": row.get("nav_status"),
            "vessel_type": row.get("vessel_type"),
            "name": row.get("name"),
            "flag": row.get("flag"),
            "imo": row.get("imo"),
            "callsign": row.get("callsign"),
            "length": _to_float(row.get("length")),
            "width": _to_float(row.get("width")),
            "draft": _to_float(row.get("draft")),
        })

    report["kept"] = len(out)
    return out, report


# --------------------------------------------------------------------------- #
# Vessel track
# --------------------------------------------------------------------------- #
@dataclass
class VesselTrack:
    """One vessel's cleaned, time-ordered movement, queryable at any instant."""

    mmsi: int
    t_h: np.ndarray                 # hours relative to the reference (observation) time
    lat: np.ndarray
    lon: np.ndarray
    sog: np.ndarray
    cog: np.ndarray
    heading: np.ndarray
    rot: np.ndarray | None = None   # rate of turn (deg/min-ish); 0 where not reported
    timestamps: list | None = None  # aware UTC datetime per fix (for the AIS models)
    nav_status: int | None = None
    vessel_type_group: str = "UNKNOWN"
    name: str = "UNKNOWN"
    flag: str = "Unknown"
    imo: int | None = None
    callsign: str = ""
    length_m: float | None = None
    width_m: float | None = None
    draft_m: float | None = None
    gaps: list = field(default_factory=list)        # [(t0_h, t1_h, minutes), ...]
    segments: list = field(default_factory=list)    # [(i0, i1), ...] index ranges
    n_input: int = 0
    n_removed_speed: int = 0
    n_removed_time_order: int = 0

    # -- queries ------------------------------------------------------
    @property
    def n_points(self) -> int:
        return int(self.t_h.size)

    def covers(self, t_h, pad_h: float = 0.0) -> bool:
        return (self.t_h[0] - pad_h) <= float(t_h) <= (self.t_h[-1] + pad_h)

    def duration_h(self) -> float:
        return float(self.t_h[-1] - self.t_h[0]) if self.n_points else 0.0

    def position_at(self, t_h):
        """Linear interpolation. Scalar in -> scalar out; array in -> array out."""
        t = np.asarray(t_h, float)
        return np.interp(t, self.t_h, self.lat), np.interp(t, self.t_h, self.lon)

    def speed_at(self, t_h):
        return np.interp(np.asarray(t_h, float), self.t_h, self.sog)

    def course_at(self, t_h):
        """Interpolate course through the 0/360 wrap via unit vectors."""
        t = np.asarray(t_h, float)
        th = np.radians(self.cog)
        u = np.interp(t, self.t_h, np.sin(th))
        v = np.interp(t, self.t_h, np.cos(th))
        return (np.degrees(np.arctan2(u, v)) + 360.0) % 360.0

    def window_indices(self, t0_h: float, t1_h: float) -> np.ndarray:
        return np.nonzero((self.t_h >= float(t0_h)) & (self.t_h <= float(t1_h)))[0]

    def points_in_window(self, t0_h: float, t1_h: float):
        idx = self.window_indices(t0_h, t1_h)
        return self.t_h[idx], self.lat[idx], self.lon[idx]

    def course_over(self, t0_h: float, t1_h: float) -> float | None:
        """Straight-line bearing from the first to the last fix inside the window."""
        idx = self.window_indices(t0_h, t1_h)
        if idx.size < 2:
            return None
        return float(initial_bearing_deg(
            self.lat[idx[0]], self.lon[idx[0]], self.lat[idx[-1]], self.lon[idx[-1]]
        ))

    def in_gap(self, t_h) -> bool:
        return any(g[0] <= float(t_h) <= g[1] for g in self.gaps)

    def bbox(self) -> list[float]:
        return [float(self.lon.min()), float(self.lat.min()),
                float(self.lon.max()), float(self.lat.max())]

    def identity(self) -> dict:
        return {
            "mmsi": int(self.mmsi),
            "name": str(self.name),
            "flag": str(self.flag),
            "vessel_type": str(self.vessel_type_group),
            "nav_status": nav_status_label(self.nav_status) if self.nav_status is not None else None,
            "imo": int(self.imo) if self.imo else None,
            "callsign": str(self.callsign or ""),
            "length_m": self.length_m,
            "width_m": self.width_m,
        }


# --------------------------------------------------------------------------- #
# Impossible-speed filter (greedy walk, matches SN)
# --------------------------------------------------------------------------- #
def _drop_impossible_speed(t_h, lat, lon, max_kn=MAX_PLAUSIBLE_SOG_KN) -> np.ndarray:
    keep = np.ones(len(t_h), bool)
    i = 0
    for j in range(1, len(t_h)):
        dt_h = t_h[j] - t_h[i]
        if dt_h <= 0:
            keep[j] = False
            continue
        d_m = float(haversine_m(lat[i], lon[i], lat[j], lon[j]))
        kn = (d_m / (dt_h * 3600.0)) * _KN_PER_MS
        if kn > max_kn:
            keep[j] = False
        else:
            i = j
    return keep


def _segment_ranges(t_h, split_minutes) -> list[tuple[int, int]]:
    if len(t_h) == 0:
        return []
    breaks = np.nonzero(np.diff(t_h) * 60.0 > split_minutes)[0]
    starts = np.concatenate(([0], breaks + 1))
    ends = np.concatenate((breaks + 1, [len(t_h)]))
    return [(int(a), int(b)) for a, b in zip(starts, ends)]


# --------------------------------------------------------------------------- #
# build_tracks
# --------------------------------------------------------------------------- #
def build_tracks(
    records,
    reference_time,
    *,
    min_points: int = 2,
    max_sog_kn: float = MAX_PLAUSIBLE_SOG_KN,
    gap_flag_minutes: float = GAP_FLAG_MINUTES,
    gap_split_minutes: float = GAP_SPLIT_MINUTES,
    already_clean: bool = False,
) -> tuple[list[VesselTrack], dict]:
    """Reconstruct one :class:`VesselTrack` per MMSI. Returns ``(tracks, report)``."""
    if already_clean:
        rows, report = list(records), {"input": len(records), "kept": len(records)}
    else:
        rows, report = clean_records(records)

    ref = reference_time
    if not isinstance(ref, datetime):
        ref = _parse_timestamp(ref)
    if ref is None:
        raise ValueError("reference_time must be a datetime or a parseable timestamp")
    if ref.tzinfo is None:
        ref = ref.replace(tzinfo=timezone.utc)

    by_mmsi: dict[int, list[dict]] = {}
    for r in rows:
        by_mmsi.setdefault(r["mmsi"], []).append(r)

    tracks: list[VesselTrack] = []
    n_speed_dropped = 0
    for mmsi, group in by_mmsi.items():
        group.sort(key=lambda r: r["timestamp"])
        t_h = np.array([(r["timestamp"] - ref).total_seconds() / 3600.0 for r in group])
        lat = np.array([r["latitude"] for r in group])
        lon = np.array([r["longitude"] for r in group])

        # strictly increasing time (drop out-of-order / duplicate-time fixes)
        order_keep = np.concatenate(([True], np.diff(t_h) > 0))
        n_time = int((~order_keep).sum())
        t_h, lat, lon = t_h[order_keep], lat[order_keep], lon[order_keep]
        group = [g for g, k in zip(group, order_keep) if k]
        if len(t_h) < min_points:
            continue

        speed_keep = _drop_impossible_speed(t_h, lat, lon, max_sog_kn)
        n_speed = int((~speed_keep).sum())
        n_speed_dropped += n_speed
        t_h, lat, lon = t_h[speed_keep], lat[speed_keep], lon[speed_keep]
        group = [g for g, k in zip(group, speed_keep) if k]
        if len(t_h) < min_points:
            continue

        sog = np.array([g["sog"] if g["sog"] is not None else 0.0 for g in group])
        cog = np.array([g["cog"] if g["cog"] is not None else 0.0 for g in group])
        hdg = np.array([g["heading"] if g["heading"] is not None else np.nan for g in group])
        rot = np.array([g.get("rot") if g.get("rot") is not None else 0.0 for g in group])
        stamps = [g["timestamp"] for g in group]

        dt_min = np.diff(t_h) * 60.0
        gaps = [(float(t_h[i]), float(t_h[i + 1]), float(dt_min[i]))
                for i in np.nonzero(dt_min > gap_flag_minutes)[0]]
        segments = _segment_ranges(t_h, gap_split_minutes)

        last = group[-1]
        tracks.append(VesselTrack(
            mmsi=int(mmsi),
            t_h=t_h, lat=lat, lon=lon, sog=sog, cog=cog, heading=hdg,
            rot=rot, timestamps=stamps,
            nav_status=_coerce_int(last.get("nav_status")),
            vessel_type_group=type_group(last.get("vessel_type")),
            name=str(last.get("name") or "UNKNOWN"),
            flag=str(last.get("flag") or flag_from_mmsi(mmsi)),
            imo=_coerce_int(last.get("imo")),
            callsign=str(last.get("callsign") or ""),
            length_m=last.get("length"),
            width_m=last.get("width"),
            draft_m=last.get("draft"),
            gaps=gaps, segments=segments,
            n_input=len(group) + n_speed + n_time,
            n_removed_speed=n_speed,
            n_removed_time_order=n_time,
        ))

    tracks.sort(key=lambda tr: tr.mmsi)
    report = {
        **report,
        "vessels": len(tracks),
        "dropped_impossible_speed": n_speed_dropped,
        "reference_time": ref.isoformat(),
    }
    return tracks, report


def _coerce_int(value):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# Spatial / temporal gate
# --------------------------------------------------------------------------- #
def spatial_temporal_gate(
    tracks: list[VesselTrack],
    origin_lat: float,
    origin_lon: float,
    window_h,
    *,
    radius_km: float,
    pad_h: float,
    origin_spread_km: float = 0.0,
) -> tuple[list[VesselTrack], list[tuple], dict]:
    """Keep only vessels that could physically have been at the origin in the window."""
    t0 = float(min(window_h)) - pad_h
    t1 = float(max(window_h)) + pad_h
    reach_km = float(radius_km) + float(origin_spread_km)

    kept, rejected = [], []
    for tr in tracks:
        if tr.n_points < 2 or tr.t_h[-1] < t0 or tr.t_h[0] > t1:
            rejected.append((tr, "outside the release time window"))
            continue
        idx = tr.window_indices(t0, t1)
        if idx.size == 0:
            rejected.append((tr, "no reports inside the window"))
            continue
        d_km = haversine_m(tr.lat[idx], tr.lon[idx], origin_lat, origin_lon) / 1000.0
        if float(np.min(d_km)) > reach_km:
            rejected.append((tr, f"closest approach {float(np.min(d_km)):.0f} km > {reach_km:.0f} km gate"))
            continue
        kept.append(tr)

    gate = {
        "time_gate_h": [round(t0, 2), round(t1, 2)],
        "radius_km": round(reach_km, 1),
        "origin_centre": [round(float(origin_lat), 5), round(float(origin_lon), 5)],
        "n_input": len(tracks),
        "n_kept": len(kept),
        "n_rejected": len(rejected),
    }
    return kept, rejected, gate
