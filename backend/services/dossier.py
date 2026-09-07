"""
Evidence dossier report engine.

Assembles a structured evidence report for one investigation from the persisted
``summary_metrics``, its anomalies and its alert audit trail. JSON is the
canonical form; ``dossier_markdown`` / ``dossier_html`` render a printable
structured view (PDF = the browser's print-to-PDF on the HTML).
"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.models.alert import Alert
from backend.models.anomaly import Anomaly
from backend.models.investigation import Investigation

DOSSIER_SCHEMA_VERSION = "1.0"


def build_dossier(db: Session, investigation_id: int) -> dict:
    inv = db.get(Investigation, investigation_id)
    if inv is None:
        raise KeyError(f"investigation {investigation_id} not found")

    m = inv.summary_metrics or {}
    anomalies = db.execute(
        select(Anomaly).where(Anomaly.investigation_id == inv.id).order_by(Anomaly.id)
    ).scalars().all()
    alerts = db.execute(
        select(Alert).where(Alert.investigation_id == inv.id).order_by(Alert.id)
    ).scalars().all()

    sar = m.get("sar", {})
    hind = m.get("hindcast", {})
    fore = m.get("forecast", {})
    attr = m.get("attribution", {})
    fb = m.get("feedback_loop", {})
    juris = m.get("jurisdiction", {})
    prime_det = (sar.get("primary_detection") or {})
    char = prime_det.get("characterization") or {}

    candidates = attr.get("candidates", [])
    prime = candidates[0] if candidates else None

    dossier = {
        "schema_version": DOSSIER_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),

        "incident_summary": {
            "reference": inv.reference,
            "title": inv.title,
            "status": inv.status.value,
            "scenario": m.get("scenario", {}).get("key"),
            "verdict": m.get("verdict"),
            "classification": sar.get("scene_classification"),
            "confidence": sar.get("confidence"),
            "detected_at": inv.detected_at.isoformat() if inv.detected_at else None,
            "centroid": [inv.centroid_lat, inv.centroid_lon],
            "created_by_user_id": inv.created_by_id,
        },

        "satellite_metadata": {
            "acquisition": sar.get("acquisition"),
            "bounding_box": sar.get("bounding_box"),
            "pixel_size_m": sar.get("pixel_size_m"),
            "classifier": (sar.get("pipeline") or {}).get("classifier"),
            "decision_threshold": (sar.get("pipeline") or {}).get("decision_threshold"),
            "preprocess": (sar.get("pipeline") or {}).get("preprocess_stats"),
            "segmentation": (sar.get("pipeline") or {}).get("segmentation_diagnostics"),
            "runtime_seconds": (sar.get("pipeline") or {}).get("runtime_seconds"),
            "candidate_counts": sar.get("counts"),
        },

        "spill_geometry": {
            "classification": prime_det.get("classification"),
            "detection_confidence": prime_det.get("confidence"),
            "area_km2": char.get("area_km2"),
            "area_hectares": char.get("area_hectares"),
            "perimeter_km": char.get("perimeter_km"),
            "length_km": char.get("length_km"),
            "width_km": char.get("width_km"),
            "major_axis_km": char.get("major_axis_km"),
            "minor_axis_km": char.get("minor_axis_km"),
            "orientation_deg": char.get("orientation_deg"),
            "centroid": char.get("centroid"),
            "mean_contrast_db": char.get("mean_contrast_db"),
            "border_gradient_db_px": char.get("border_gradient_db_px"),
            "look_alike_filter": prime_det.get("look_alike_filter"),
            "polygons": sar.get("polygons"),
        },

        "met_ocean": {
            "environmental_field": m.get("environmental_field"),
            "conditions": m.get("met_ocean"),
            "provenance": hind.get("provenance"),
        },

        "hindcast_origin": {
            "best_estimate": hind.get("best_estimate"),
            "centroid": hind.get("centroid"),
            "uncertainty_radius_km": hind.get("uncertainty_radius_km"),
            "origin_probability_field": hind.get("origin_probability_field"),
            "release_polygon": hind.get("release_polygon"),
            "confidence_ellipse": hind.get("confidence_ellipse"),
            "drift_params": hind.get("params"),
            "age_source": hind.get("age_source"),
        },

        "release_time_window": {
            "initial_window_h": fb.get("initial_release_window_h") or hind.get("age_prior_window_h"),
            "refined_window_h": fb.get("refined_release_window_h") or hind.get("release_window_h"),
            "age_point_estimate_h": hind.get("age_point_estimate_h"),
            "feedback_loop": {
                "converged": fb.get("converged"),
                "n_iterations": fb.get("n_iterations"),
                "iterations": fb.get("iterations"),
                "initial_origin": fb.get("initial_origin"),
                "refined_origin": fb.get("refined_origin"),
            },
        },

        "forward_forecast": {
            "horizons_h": fore.get("horizons_h"),
            "horizons": fore.get("horizons"),
            "shoreline_contact": fore.get("coastal_impact"),
        },

        "ais_evidence": {
            "traffic_gate": attr.get("gate"),
            "search": {
                "engine": attr.get("engine"),
                "weights": attr.get("base_weights"),
                "ai_engaged": attr.get("ai_engaged"),
            },
            "per_vessel": [
                {
                    "rank": c.get("rank"),
                    "mmsi": c["identity"]["mmsi"],
                    "name": c["identity"]["name"],
                    "vessel_type": c["identity"].get("vessel_type"),
                    "flag": c["identity"].get("flag"),
                    "cpa_km": c.get("cpa_km"),
                    "best_match_time_h": c.get("best_match_time_h"),
                    "blackout": _comp(c, "blackout"),
                    "proximity": _comp(c, "proximity"),
                    "ais_anomaly": _comp(c, "ais_anomaly"),
                }
                for c in candidates
            ],
        },

        "behaviour_metrics": {
            "per_vessel": [
                {
                    "rank": c.get("rank"),
                    "mmsi": c["identity"]["mmsi"],
                    "name": c["identity"]["name"],
                    "route_deviation": _comp(c, "route_deviation"),
                    "axis_alignment": _comp(c, "axis_alignment"),
                    "spatiotemporal": _comp(c, "spatiotemporal"),
                    "vessel_prior": _comp(c, "vessel_prior"),
                }
                for c in candidates
            ],
        },

        "candidate_ranking": {
            "summary": attr.get("summary"),
            "prime_suspect": _prime_explanation(prime),
            "candidates": [
                {
                    "rank": c.get("rank"),
                    "identity": c["identity"],
                    "score": c.get("score"),
                    "raw_score": c.get("raw_score"),
                    "assessment": c.get("assessment"),
                    "proximity_gate": c.get("proximity_gate"),
                    "components": {
                        k: {"value": v["value"], "weight": v["weight"],
                            "points": v["points"], "family": v.get("family"),
                            "finding": (v.get("detail") or {}).get("finding")}
                        for k, v in (c.get("components") or {}).items()
                    },
                }
                for c in candidates
            ],
        },

        "jurisdiction": {
            "primary_code": juris.get("primary_code"),
            "primary_name": juris.get("primary_name"),
            "chain_codes": juris.get("chain_codes"),
            "jurisdiction_ids": juris.get("jurisdiction_ids"),
            "note": ("Coordinate falls outside the seeded maritime zones "
                     "(international / unassigned waters)."
                     if not juris.get("primary_code") else None),
        },

        "anomalies": [
            {"id": a.id, "type": a.type.value, "source": a.source.value,
             "label": a.label, "confidence": a.confidence,
             "lat": a.lat, "lon": a.lon,
             "occurred_at": a.occurred_at.isoformat() if a.occurred_at else None,
             "details": a.details}
            for a in anomalies
        ],

        "audit_trail": {
            "investigation": {
                "created_at": inv.created_at.isoformat() if inv.created_at else None,
                "updated_at": inv.updated_at.isoformat() if inv.updated_at else None,
                "reference": inv.reference,
            },
            "pipeline_steps": m.get("pipeline_steps", []),
            "alerts": [
                {
                    "id": al.id, "status": al.status.value, "channel": al.channel.value,
                    "recipient": al.recipient, "provider": al.provider,
                    "attempts": al.attempts, "fingerprint": al.fingerprint,
                    "time_bucket": al.time_bucket, "dedup_of_id": al.dedup_of_id,
                    "triggered_by_confidence": al.triggered_by_confidence,
                    "sent_at": al.sent_at.isoformat() if al.sent_at else None,
                    "last_error": al.last_error,
                    "error_log": al.error_log,
                }
                for al in alerts
            ],
        },

        "caveats": m.get("caveats", []),
    }
    return dossier


def _comp(cand: dict, key: str):
    c = (cand.get("components") or {}).get(key)
    if not c:
        return None
    return {"value": c.get("value"), "points": c.get("points"),
            "finding": (c.get("detail") or {}).get("finding")}


def _prime_explanation(prime: dict | None) -> dict | None:
    if not prime:
        return None
    comps = prime.get("components") or {}
    contributions = sorted(
        ({"component": k, "family": v.get("family"), "value": v["value"],
          "weight": v["weight"], "points": v["points"],
          "finding": (v.get("detail") or {}).get("finding")}
         for k, v in comps.items()),
        key=lambda d: -d["points"],
    )
    return {
        "identity": prime["identity"],
        "score": prime.get("score"),
        "assessment": prime.get("assessment"),
        "why_ranked_first": contributions,
        "top_reasons": [c["finding"] for c in contributions[:3] if c["finding"]],
    }


# --------------------------------------------------------------------------- #
# Renderings
# --------------------------------------------------------------------------- #
def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.3f}".rstrip("0").rstrip(".")
    if isinstance(v, (list, tuple)):
        return ", ".join(_fmt(x) for x in v)
    return str(v)


def dossier_markdown(d: dict) -> str:
    inc = d["incident_summary"]
    L = [
        f"# Evidence Dossier - {inc['reference']}",
        "",
        f"**{inc['title']}**  ",
        f"Status: {inc['status']}  |  Classification: {inc['classification']}  |  "
        f"Confidence: {_fmt(inc['confidence'])}  ",
        f"Detected: {inc['detected_at']}  |  Centroid: {_fmt(inc['centroid'])}  ",
        f"Generated: {d['generated_at']}  |  Schema {d['schema_version']}",
        "",
        f"> {inc['verdict']}",
        "",
        "## 1. Satellite metadata",
    ]
    sm = d["satellite_metadata"]
    L += [f"- Acquisition: {sm['acquisition']}",
          f"- Bounding box: {_fmt(sm['bounding_box'])}",
          f"- Pixel size: {_fmt(sm['pixel_size_m'])} m",
          f"- Classifier: {sm['classifier']} (threshold {_fmt(sm['decision_threshold'])})",
          f"- Runtime: {_fmt(sm['runtime_seconds'])} s",
          f"- Candidate counts: {sm['candidate_counts']}", ""]

    g = d["spill_geometry"]
    L += ["## 2. Spill geometry",
          f"- Detection: {g['classification']} (conf {_fmt(g['detection_confidence'])})",
          f"- Area: {_fmt(g['area_km2'])} km2 ({_fmt(g['area_hectares'])} ha)",
          f"- Length x width: {_fmt(g['length_km'])} x {_fmt(g['width_km'])} km",
          f"- Orientation: {_fmt(g['orientation_deg'])} deg",
          f"- Mean contrast: {_fmt(g['mean_contrast_db'])} dB, "
          f"border gradient {_fmt(g['border_gradient_db_px'])} dB/px", ""]

    mo = d["met_ocean"]
    L += ["## 3. Met-ocean conditions",
          f"- Field: {mo['environmental_field']}",
          f"- Conditions: {mo['conditions']}", ""]

    ho = d["hindcast_origin"]
    rw = d["release_time_window"]
    L += ["## 4. Hindcast origin + release-time window",
          f"- Best-estimate origin: {_fmt(ho['best_estimate'])} "
          f"(+/- {_fmt(ho['uncertainty_radius_km'])} km)",
          f"- Initial release window: {_fmt(rw['initial_window_h'])} h",
          f"- Refined release window: {_fmt(rw['refined_window_h'])} h",
          f"- Feedback loop: {rw['feedback_loop']['n_iterations']} iteration(s), "
          f"converged={rw['feedback_loop']['converged']}", ""]

    ff = d.get("forward_forecast") or {}
    sc = ff.get("shoreline_contact") or {}
    L += ["## 4a. Forward forecast + shoreline contact"]
    if sc.get("will_beach"):
        fc = sc.get("first_contact_point") or []
        L += [f"- First landfall ETA: {_fmt(sc.get('first_contact_eta_h') or sc.get('eta_hours'))} h",
              f"- Contact coordinates: {_fmt(fc)}",
              f"- Oil ashore within {_fmt((ff.get('horizons_h') or [48])[-1])} h: "
              f"{_fmt(sc.get('fraction_beached'))}",
              f"- {sc.get('note', '')}", ""]
    else:
        L += [f"- {sc.get('note') or 'No shoreline contact modelled in the forecast window.'}", ""]

    cr = d["candidate_ranking"]
    L += ["## 5. Ranked candidate vessels", ""]
    if cr["prime_suspect"]:
        ps = cr["prime_suspect"]
        L += [f"**Prime suspect: {ps['identity']['name']} (MMSI {ps['identity']['mmsi']}, "
              f"{ps['identity'].get('vessel_type')}, {ps['identity'].get('flag')})** - "
              f"score {_fmt(ps['score'])}/100, {ps['assessment']}", "",
              "Why ranked #1:"]
        for c in ps["why_ranked_first"]:
            L.append(f"  - {c['component']} [{c['family']}] "
                     f"value {_fmt(c['value'])} x weight {_fmt(c['weight'])} "
                     f"= {_fmt(c['points'])} pts - {c['finding']}")
        L.append("")
    L += ["| Rank | Vessel | MMSI | Score | Assessment |", "|---|---|---|---|---|"]
    for c in cr["candidates"]:
        L.append(f"| {c['rank']} | {c['identity']['name']} | {c['identity']['mmsi']} | "
                 f"{_fmt(c['score'])} | {c['assessment']} |")
    L.append("")

    j = d["jurisdiction"]
    L += ["## 6. Jurisdictional mapping",
          f"- Primary: {j['primary_code']} ({j['primary_name']})",
          f"- Chain: {_fmt(j['chain_codes'])}"]
    if j.get("note"):
        L.append(f"- Note: {j['note']}")
    L.append("")

    at = d["audit_trail"]
    L += ["## 7. Audit trail", "", "Pipeline steps:"]
    for s in at["pipeline_steps"]:
        L.append(f"  - [{_fmt(s['seconds'])}s] {s['step']}: {s['label']}")
    if at["alerts"]:
        L += ["", "Alerts:"]
        for al in at["alerts"]:
            L.append(f"  - #{al['id']} {al['status']} via {al['provider']} to {al['recipient']} "
                     f"(attempts {al['attempts']}, fp {al['fingerprint']})")
    L += ["", "## 8. Caveats"]
    for cav in d["caveats"]:
        if cav:
            L.append(f"- {cav}")
    return "\n".join(L)


def dossier_html(d: dict) -> str:
    md = dossier_markdown(d)
    # very small markdown -> HTML (headings, lists, tables, bold, blockquote)
    import html as _h

    out, in_ul, in_tbl = [], False, False
    for line in md.splitlines():
        s = line.rstrip()
        if s.startswith("|"):
            cells = [c.strip() for c in s.strip("|").split("|")]
            if set("".join(cells)) <= set("-: "):
                continue
            if not in_tbl:
                out.append("<table>")
                in_tbl = True
            tag = "th" if all(c and not c.replace(".", "").isdigit() for c in cells) and len(out) and out[-1] == "<table>" else "td"
            out.append("<tr>" + "".join(f"<{tag}>{_h.escape(c)}</{tag}>" for c in cells) + "</tr>")
            continue
        if in_tbl:
            out.append("</table>")
            in_tbl = False
        if s.startswith("- ") or s.startswith("  - "):
            if not in_ul:
                out.append("<ul>")
                in_ul = True
            out.append(f"<li>{_inline(s.lstrip(' -'))}</li>")
            continue
        if in_ul:
            out.append("</ul>")
            in_ul = False
        if s.startswith("# "):
            out.append(f"<h1>{_h.escape(s[2:])}</h1>")
        elif s.startswith("## "):
            out.append(f"<h2>{_h.escape(s[3:])}</h2>")
        elif s.startswith("> "):
            out.append(f"<blockquote>{_inline(s[2:])}</blockquote>")
        elif s:
            out.append(f"<p>{_inline(s)}</p>")
    if in_ul:
        out.append("</ul>")
    if in_tbl:
        out.append("</table>")
    body = "\n".join(out)
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>Evidence Dossier - {_h.escape(d['incident_summary']['reference'])}</title>
<style>
 body{{font:14px/1.6 -apple-system,Segoe UI,Roboto,sans-serif;max-width:920px;
   margin:2rem auto;padding:0 1.5rem;color:#1a2332;background:#fff}}
 h1{{font-size:1.6rem;border-bottom:2px solid #2563eb;padding-bottom:.4rem}}
 h2{{font-size:1.15rem;margin-top:2rem;color:#2563eb}}
 blockquote{{border-left:3px solid #94a3b8;margin:1rem 0;padding:.4rem 1rem;
   background:#f1f5f9;color:#475569}}
 table{{border-collapse:collapse;width:100%;margin:1rem 0}}
 th,td{{border:1px solid #cbd5e1;padding:.4rem .6rem;text-align:left;font-size:13px}}
 th{{background:#f1f5f9}}
 ul{{margin:.3rem 0 .8rem 1.2rem}} code{{background:#f1f5f9;padding:.1rem .3rem;border-radius:3px}}
 @media print{{body{{margin:0;max-width:none}} h2{{page-break-after:avoid}}}}
</style></head><body>{body}
<hr><p><small>Generated {_h.escape(d['generated_at'])} - schema {d['schema_version']} -
print this page (Ctrl/Cmd-P) to produce a PDF.</small></p></body></html>"""


def _inline(s: str) -> str:
    import html as _h
    import re

    s = _h.escape(s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    return s
