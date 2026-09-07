"""
Vessel / AIS service layer.

Orchestration over ``backend.ml.ais.tracks`` (the pandas-free track processor):
normalise raw AIS records, reconstruct one track per MMSI, detect
gap / blackout / loitering intervals, and adapt a track into the pandas frames
the POSEatSea autoencoder and LSTM consume.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np

from backend.core.logging import get_logger
from backend.ml.ais.config import DEFAULT_ACCURACY, DEFAULT_MSG_TYPE
from backend.ml.ais.tracks import (
    GAP_FLAG_MINUTES,
    VesselTrack,
    build_tracks,
    clean_records,
    spatial_temporal_gate,
)

log = get_logger("backend.services.vessels")

LOITER_SPEED_KN = 2.0
LOITER_MIN_MINUTES = 30.0


def normalise_ais(records) -> dict:
    """Row-level cleaning only (timestamps, coords, duplicates). No tracks yet."""
    rows, report = clean_records(records)
    return {"records": rows, "report": report}


def process_ais_records(
    records,
    reference_time: datetime | str,
    *,
    min_points: int = 2,
    max_sog_kn: float | None = None,
    gap_flag_minutes: float | None = None,
    gap_split_minutes: float | None = None,
) -> dict:
    """Clean + reconstruct tracks. Returns tracks plus a full processing report."""
    kwargs: dict = {"min_points": min_points}
    if max_sog_kn is not None:
        kwargs["max_sog_kn"] = max_sog_kn
    if gap_flag_minutes is not None:
        kwargs["gap_flag_minutes"] = gap_flag_minutes
    if gap_split_minutes is not None:
        kwargs["gap_split_minutes"] = gap_split_minutes

    tracks, report = build_tracks(records, reference_time, **kwargs)
    log.info(
        "AIS processed: %s rows -> %s kept -> %s vessel track(s)",
        report.get("input"), report.get("kept"), report.get("vessels"),
    )
    return {"tracks": tracks, "report": report, "summaries": [track_summary(t) for t in tracks]}


def track_summary(track: VesselTrack) -> dict:
    """A compact, JSON-serialisable view of one reconstructed track."""
    return {
        "identity": track.identity(),
        "n_points": track.n_points,
        "first_seen_h": round(float(track.t_h[0]), 3),
        "last_seen_h": round(float(track.t_h[-1]), 3),
        "duration_h": round(track.duration_h(), 3),
        "bbox": [round(v, 5) for v in track.bbox()],
        "n_segments": len(track.segments),
        "segments": [[int(a), int(b)] for a, b in track.segments],
        "gaps": [
            {"start_h": round(g[0], 2), "end_h": round(g[1], 2), "minutes": round(g[2], 1)}
            for g in track.gaps
        ],
        "removed": {
            "impossible_speed": track.n_removed_speed,
            "out_of_time_order": track.n_removed_time_order,
        },
        "mean_sog_kn": round(float(track.sog[track.sog > 0].mean()), 1) if (track.sog > 0).any() else 0.0,
    }


# --------------------------------------------------------------------------- #
# Gap / blackout / loitering detection
# --------------------------------------------------------------------------- #
def detect_blackouts(track: VesselTrack, *, window_h=None, min_minutes: float = GAP_FLAG_MINUTES) -> dict:
    """Transmission gaps ("dark periods"), and those overlapping a release window."""
    gaps = [
        {"start_h": round(g[0], 3), "end_h": round(g[1], 3), "minutes": round(g[2], 1)}
        for g in track.gaps if g[2] >= min_minutes
    ]
    over_window = []
    longest_over_window_min = 0.0
    if window_h is not None:
        t0, t1 = float(min(window_h)), float(max(window_h))
        for g in gaps:
            ov = max(0.0, min(g["end_h"], t1) - max(g["start_h"], t0))
            if ov > 0:
                over_window.append({**g, "overlap_minutes": round(ov * 60.0, 1)})
                longest_over_window_min = max(longest_over_window_min, ov * 60.0)
    return {
        "n_gaps": len(gaps),
        "gaps": gaps,
        "blackouts_over_release_window": over_window,
        "longest_blackout_over_window_min": round(longest_over_window_min, 1),
        "has_blackout_over_window": bool(over_window),
    }


def detect_loitering(track: VesselTrack, *, speed_kn: float = LOITER_SPEED_KN,
                     min_minutes: float = LOITER_MIN_MINUTES) -> list[dict]:
    """Contiguous spans where the vessel's speed stays below ``speed_kn``."""
    if track.n_points < 2:
        return []
    slow = track.sog < speed_kn
    spans, i = [], 0
    n = track.n_points
    while i < n:
        if not slow[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and slow[j + 1]:
            j += 1
        dur_min = float(track.t_h[j] - track.t_h[i]) * 60.0
        if dur_min >= min_minutes:
            spans.append({
                "start_h": round(float(track.t_h[i]), 3),
                "end_h": round(float(track.t_h[j]), 3),
                "minutes": round(dur_min, 1),
                "mean_sog_kn": round(float(track.sog[i:j + 1].mean()), 2),
            })
        i = j + 1
    return spans


# --------------------------------------------------------------------------- #
# Persistence + map-ready track views (Epic 1.4)
# --------------------------------------------------------------------------- #
def _iso(ts) -> str | None:
    if ts is None:
        return None
    try:
        return ts.isoformat()
    except AttributeError:
        return str(ts)


def persist_vessels(db, tracks, *, last_seen_at: datetime | None = None) -> dict[str, int]:
    """Idempotently upsert a :class:`~backend.models.vessel.Vessel` per track.

    Returns ``{mmsi_str: vessel_id}``.
    """
    from sqlalchemy import select

    from backend.models.vessel import Vessel

    out: dict[str, int] = {}
    for tr in tracks:
        ident = tr.identity()
        mmsi = str(ident["mmsi"])
        row = db.execute(select(Vessel).where(Vessel.mmsi == mmsi)).scalar_one_or_none()
        fields = dict(
            name=ident.get("name") or None,
            imo=str(ident["imo"]) if ident.get("imo") else None,
            call_sign=(ident.get("callsign") or None),
            flag=ident.get("flag") or None,
            vessel_type=ident.get("vessel_type") or None,
            length_m=ident.get("length_m"),
            width_m=ident.get("width_m"),
            last_seen_at=last_seen_at,
        )
        if row is None:
            row = Vessel(mmsi=mmsi, **fields)
            db.add(row)
        else:
            for k, v in fields.items():
                if v is not None:
                    setattr(row, k, v)
        db.flush()
        out[mmsi] = row.id
    return out


def track_view(track: VesselTrack, *, window_h=None, max_points: int = 80) -> dict:
    """A JSON-serialisable, map-ready view of one reconstructed track.

    ``pings`` is decimated to at most ``max_points`` fixes; ``loiter`` and
    ``blackouts`` carry the spans the UI draws as amber circles / dashed gaps.
    """
    n = track.n_points
    if n == 0:
        return {"mmsi": str(track.mmsi), "pings": [], "loiter": [], "blackouts": [], "metrics": {}}
    idx = np.unique(np.linspace(0, n - 1, min(max_points, n)).astype(int))
    ts = track.timestamps
    pings = [
        {
            "lat": round(float(track.lat[i]), 6),
            "lon": round(float(track.lon[i]), 6),
            "t_h": round(float(track.t_h[i]), 3),
            "ts": _iso(ts[i]) if ts is not None else None,
            "sog": round(float(track.sog[i]), 2),
            "cog": round(float(track.cog[i]), 1),
        }
        for i in idx
    ]
    bl = detect_blackouts(track, window_h=window_h)
    sog_pos = track.sog[track.sog > 0]
    cog = track.cog[np.isfinite(track.cog)]
    ident = track.identity()
    return {
        "mmsi": str(track.mmsi),
        "name": ident.get("name"),
        "vessel_type": ident.get("vessel_type"),
        "flag": ident.get("flag"),
        "pings": pings,
        "loiter": detect_loitering(track),
        "blackouts": bl["gaps"],
        "blackouts_over_window": bl["blackouts_over_release_window"],
        "metrics": {
            "n_points": n,
            "first_seen_h": round(float(track.t_h[0]), 2),
            "last_seen_h": round(float(track.t_h[-1]), 2),
            "duration_h": round(track.duration_h(), 2),
            "sog_min_kn": round(float(track.sog.min()), 2),
            "sog_max_kn": round(float(track.sog.max()), 2),
            "sog_mean_kn": round(float(sog_pos.mean()), 2) if sog_pos.size else 0.0,
            "mean_heading_deg": round(float(cog.mean()), 1) if cog.size else None,
            "n_gaps": bl["n_gaps"],
            "longest_blackout_over_window_min": bl["longest_blackout_over_window_min"],
            "bbox": [round(v, 5) for v in track.bbox()],
        },
    }


def build_track_views(tracks, *, window_h=None) -> dict[str, dict]:
    """``{mmsi_str: track_view}`` for every reconstructed track."""
    return {str(tr.mmsi): track_view(tr, window_h=window_h) for tr in tracks}


# --------------------------------------------------------------------------- #
# Adapters: VesselTrack -> the AIS-model pandas frames
# --------------------------------------------------------------------------- #
def ais_feature_frame(track: VesselTrack):
    """One row per fix, columns the autoencoder's feature engineering expects."""
    import pandas as pd

    n = track.n_points
    ts = track.timestamps if track.timestamps is not None else list(range(n))
    return pd.DataFrame({
        "mmsi": np.full(n, int(track.mmsi)),
        "timestamp": pd.to_datetime(ts, utc=True) if track.timestamps is not None else ts,
        "latitude": track.lat, "longitude": track.lon,
        "speed": track.sog, "course": track.cog,
        "rot": track.rot if track.rot is not None else np.zeros(n),
        "msg_type": np.full(n, DEFAULT_MSG_TYPE),
        "status": np.full(n, int(track.nav_status) if track.nav_status is not None else 0),
        "accuracy": np.full(n, DEFAULT_ACCURACY),
    })


def lstm_track_frame(track: VesselTrack):
    """Columns the LSTM's rolling-prediction walk expects (chronological)."""
    import pandas as pd

    n = track.n_points
    ts = track.timestamps if track.timestamps is not None else None
    frame = {
        "latitude": track.lat, "longitude": track.lon,
        "speed": track.sog, "course": track.cog,
        "rot": track.rot if track.rot is not None else np.zeros(n),
    }
    if ts is not None:
        frame["timestamp"] = pd.to_datetime(ts, utc=True)
    return pd.DataFrame(frame)


__all__ = [
    "normalise_ais",
    "process_ais_records",
    "track_summary",
    "spatial_temporal_gate",
    "detect_blackouts",
    "detect_loitering",
    "ais_feature_frame",
    "lstm_track_frame",
    "persist_vessels",
    "track_view",
    "build_track_views",
]
