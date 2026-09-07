"""
Geographical jurisdiction engine.

Defines a set of demo maritime responsibility zones along the Indian coastline
as GeoJSON polygons, seeds them into the database, and provides the
deterministic point-in-polygon lookup that maps an anomaly coordinate to the
jurisdictions it affects.

Hierarchy (parent -> child):

    IN-NATIONAL  (NATION)
      IN-WEST    (MARITIME_REGION)  -> IN-GJ, IN-MH, IN-GA-KA
      IN-SOUTH   (MARITIME_REGION)  -> IN-KL, IN-TN
      IN-EAST    (MARITIME_REGION)  -> IN-AP, IN-OD, IN-WB
      IN-ISLANDS (MARITIME_REGION)  -> IN-AN

State zones are latitude/longitude-tiled offshore boxes that do NOT overlap, so
a coastal point falls in exactly one state zone. A region's geometry is the
MultiPolygon of its member states; the nation's is the MultiPolygon of all
states - so containment is exact and strictly nested.

RBAC has two dimensions: the *role* (PILOT / REGIONAL / NATIONAL, in
``core/security.py``) and the *jurisdiction closure* (which zones a user may
see). A REGIONAL user assigned ``IN-WEST`` transitively sees every state under
it; a PILOT sees only the exact zone(s) assigned.
"""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.models.jurisdiction import Jurisdiction, JurisdictionType
from backend.models.user import User, UserRole

try:
    from shapely.geometry import Point, shape as _shape

    _HAVE_SHAPELY = True
except Exception:  # pragma: no cover
    _HAVE_SHAPELY = False


# --------------------------------------------------------------------------- #
# Zone definitions
# --------------------------------------------------------------------------- #
def _box(w: float, s: float, e: float, n: float) -> dict:
    """A GeoJSON Polygon rectangle [west, south, east, north]."""
    return {
        "type": "Polygon",
        "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]],
    }


def _multi(*polys: dict) -> dict:
    return {"type": "MultiPolygon", "coordinates": [p["coordinates"] for p in polys]}


# Offshore boxes tiled so no two state zones overlap (west coast split by
# latitude, east coast split by latitude, the two coasts split by longitude
# ~77.2 E, islands well to the east). Approximate but deterministic.
_STATE_BOXES: dict[str, dict] = {
    "IN-GJ":    _box(66.0, 20.4, 72.8, 24.5),    # Gujarat
    "IN-MH":    _box(68.5, 15.9, 73.6, 20.4),    # Maharashtra (Mumbai)
    "IN-GA-KA": _box(70.5, 13.5, 74.9, 15.9),    # Goa / Karnataka
    "IN-KL":    _box(72.0, 7.6, 77.2, 13.5),     # Kerala
    "IN-TN":    _box(77.2, 7.6, 82.2, 13.4),     # Tamil Nadu
    "IN-AP":    _box(79.6, 13.4, 86.2, 19.3),    # Andhra Pradesh
    "IN-OD":    _box(83.4, 19.3, 88.1, 21.2),    # Odisha
    "IN-WB":    _box(86.4, 21.2, 89.9, 22.9),    # West Bengal
    "IN-AN":    _box(91.0, 5.5, 94.6, 14.2),     # Andaman & Nicobar
}

_STATE_NAMES = {
    "IN-GJ": "Gujarat Coastal Zone",
    "IN-MH": "Maharashtra / Mumbai Coastal Zone",
    "IN-GA-KA": "Goa & Karnataka Coastal Zone",
    "IN-KL": "Kerala Coastal Zone",
    "IN-TN": "Tamil Nadu Coastal Zone",
    "IN-AP": "Andhra Pradesh Coastal Zone",
    "IN-OD": "Odisha Coastal Zone",
    "IN-WB": "West Bengal Coastal Zone",
    "IN-AN": "Andaman & Nicobar Islands Zone",
}

_REGIONS: dict[str, tuple[str, list[str]]] = {
    "IN-WEST": ("Western Maritime Region", ["IN-GJ", "IN-MH", "IN-GA-KA"]),
    "IN-SOUTH": ("Southern Maritime Region", ["IN-KL", "IN-TN"]),
    "IN-EAST": ("Eastern Maritime Region", ["IN-AP", "IN-OD", "IN-WB"]),
    "IN-ISLANDS": ("Island Territories Maritime Region", ["IN-AN"]),
}

NATIONAL_CODE = "IN-NATIONAL"


def zone_catalogue() -> list[dict]:
    """Flat, ordered list of every demo zone with its GeoJSON geometry + parent."""
    out: list[dict] = [{
        "code": NATIONAL_CODE,
        "name": "India National Maritime Jurisdiction",
        "type": JurisdictionType.NATION,
        "parent_code": None,
        "geometry": _multi(*_STATE_BOXES.values()),
    }]
    for rcode, (rname, members) in _REGIONS.items():
        out.append({
            "code": rcode, "name": rname, "type": JurisdictionType.MARITIME_REGION,
            "parent_code": NATIONAL_CODE,
            "geometry": _multi(*(_STATE_BOXES[m] for m in members)),
        })
    for rcode, (_rname, members) in _REGIONS.items():
        for scode in members:
            out.append({
                "code": scode, "name": _STATE_NAMES[scode],
                "type": JurisdictionType.COASTAL_STATE, "parent_code": rcode,
                "geometry": _STATE_BOXES[scode],
            })
    return out


# --------------------------------------------------------------------------- #
# Seeding
# --------------------------------------------------------------------------- #
def seed_demo_jurisdictions(db: Session) -> int:
    """Idempotently create every demo zone. Returns the number created."""
    existing = {
        j.code: j for j in db.execute(select(Jurisdiction)).scalars().all() if j.code
    }
    created = 0
    for spec in zone_catalogue():
        if spec["code"] in existing:
            continue
        j = Jurisdiction(
            code=spec["code"], name=spec["name"], type=spec["type"],
            geometry=spec["geometry"],
        )
        db.add(j)
        existing[spec["code"]] = j
        created += 1
    if created:
        db.flush()
    # wire parent links (safe to re-run)
    for spec in zone_catalogue():
        if spec["parent_code"]:
            child = existing.get(spec["code"])
            parent = existing.get(spec["parent_code"])
            if child is not None and parent is not None and child.parent_id != parent.id:
                child.parent_id = parent.id
    db.commit()
    return created


# --------------------------------------------------------------------------- #
# Point-in-polygon
# --------------------------------------------------------------------------- #
def point_in_geometry(geometry: dict | None, lat: float, lon: float) -> bool:
    """True if the GeoJSON ``geometry`` covers point (lat, lon).

    Uses ``covers`` rather than ``contains`` so a coordinate lying exactly on a
    zone boundary is still attributed to that zone.
    """
    if not geometry or not _HAVE_SHAPELY:
        return False
    try:
        return bool(_shape(geometry).covers(Point(float(lon), float(lat))))
    except Exception:  # noqa: BLE001 - a malformed geometry is "no match", not a crash
        return False


def jurisdictions_containing_point(db: Session, lat: float, lon: float) -> list[Jurisdiction]:
    """Every zone whose geometry covers the point, most-specific first."""
    rows = db.execute(select(Jurisdiction)).scalars().all()
    hits = [j for j in rows if point_in_geometry(j.geometry, lat, lon)]
    order = {
        JurisdictionType.COASTAL_STATE: 0, JurisdictionType.PORT: 0,
        JurisdictionType.MARITIME_REGION: 1, JurisdictionType.REGION: 1,
        JurisdictionType.EEZ: 2, JurisdictionType.ZONE: 2,
        JurisdictionType.NATION: 3,
    }
    hits.sort(key=lambda j: order.get(j.type, 5))
    return hits


def resolve_affected_jurisdictions(db: Session, lat: float, lon: float) -> dict:
    """Deterministic anomaly-coordinate -> jurisdiction mapping.

    Returns ``{primary, chain, codes, ids}`` where ``primary`` is the most
    specific zone (a coastal state, normally) and ``chain`` is
    ``[state, region, nation]``.
    """
    hits = jurisdictions_containing_point(db, lat, lon)
    if not hits:
        return {"primary": None, "chain": [], "codes": [], "ids": []}
    return {
        "primary": hits[0],
        "chain": hits,
        "codes": [j.code for j in hits if j.code],
        "ids": [j.id for j in hits],
    }


# --------------------------------------------------------------------------- #
# RBAC - jurisdiction closure + access checks
# --------------------------------------------------------------------------- #
def _descendant_ids(db: Session, root_ids: set[int]) -> set[int]:
    """Every jurisdiction transitively under ``root_ids`` (inclusive)."""
    rows = db.execute(select(Jurisdiction.id, Jurisdiction.parent_id)).all()
    children: dict[int, list[int]] = {}
    for jid, pid in rows:
        children.setdefault(pid, []).append(jid)
    closure: set[int] = set()
    stack = list(root_ids)
    while stack:
        cur = stack.pop()
        if cur in closure:
            continue
        closure.add(cur)
        stack.extend(children.get(cur, []))
    return closure


def accessible_jurisdiction_ids(db: Session, user: User) -> set[int] | None:
    """Ids the user may access. ``None`` means "all" (NATIONAL)."""
    if user.role == UserRole.NATIONAL:
        return None
    assigned = {int(i) for i in (user.jurisdiction_ids or [])}
    if not assigned:
        return set()
    return _descendant_ids(db, assigned)


def user_can_access_point(db: Session, user: User, lat: float, lon: float) -> bool:
    """Does the user's jurisdiction closure cover this coordinate?"""
    allowed = accessible_jurisdiction_ids(db, user)
    if allowed is None:
        return True
    if not allowed:
        return False
    return any(j.id in allowed for j in jurisdictions_containing_point(db, lat, lon))


def user_can_access_jurisdiction(db: Session, user: User, jurisdiction_id: int | None) -> bool:
    allowed = accessible_jurisdiction_ids(db, user)
    if allowed is None:
        return True
    if jurisdiction_id is None:
        return False
    return int(jurisdiction_id) in allowed


def user_can_access_any_code(db: Session, user: User, codes) -> bool:
    """True if any jurisdiction *code* in ``codes`` is inside the user's closure."""
    allowed = accessible_jurisdiction_ids(db, user)
    if allowed is None:
        return True
    if not codes:
        return False
    return any(int(i) in allowed for i in resolve_codes_to_ids(db, codes))


def user_can_access_coords_or_jurisdiction(
    db: Session, user: User, *, lat: float | None, lon: float | None,
    jurisdiction_id: int | None, jurisdiction_codes=None,
) -> bool:
    """A row is visible if its jurisdiction id, its codes, OR its coordinates are in scope."""
    if accessible_jurisdiction_ids(db, user) is None:
        return True
    if jurisdiction_id is not None and user_can_access_jurisdiction(db, user, jurisdiction_id):
        return True
    if jurisdiction_codes and user_can_access_any_code(db, user, jurisdiction_codes):
        return True
    if lat is not None and lon is not None and user_can_access_point(db, user, lat, lon):
        return True
    return False


def resolve_codes_to_ids(db: Session, codes) -> list[int]:
    if not codes:
        return []
    wanted = {str(c).strip().upper() for c in codes}
    rows = db.execute(
        select(Jurisdiction).where(Jurisdiction.code.in_(wanted))
    ).scalars().all()
    return [j.id for j in rows]
