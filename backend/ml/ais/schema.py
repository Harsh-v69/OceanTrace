"""
AIS data model + normalisation.

Ported from SAMUDRA NETRA ``ml/ais/schema.py``. The canonical field names used
throughout the backend are the ones named in the Phase 5 task:

    mmsi, timestamp, latitude, longitude, sog, cog, heading, nav_status, vessel_type

plus optional identity fields (name, flag, imo, callsign, length, width, draft).
Any feed (MarineCadastre, a national DGS/ICG export, Spire/exactEarth S-AIS)
plugs in with a header map.
"""
from __future__ import annotations

CANONICAL_COLUMNS = [
    "mmsi", "timestamp", "latitude", "longitude", "sog", "cog", "heading",
    "nav_status", "vessel_type", "name", "flag", "imo", "callsign",
    "length", "width", "draft",
]

#: Header aliases -> canonical name (case-insensitive match applied by the reader).
HEADER_ALIASES = {
    # MarineCadastre (NOAA/BOEM)
    "mmsi": "mmsi", "basedatetime": "timestamp", "lat": "latitude",
    "lon": "longitude", "sog": "sog", "cog": "cog", "heading": "heading",
    "vesselname": "name", "imo": "imo", "callsign": "callsign",
    "vesseltype": "vessel_type", "status": "nav_status", "length": "length",
    "width": "width", "draft": "draft",
    # common lowercase / alternate spellings
    "timestamp": "timestamp", "time": "timestamp", "datetime": "timestamp",
    "ts": "timestamp", "latitude": "latitude", "longitude": "longitude",
    "lng": "longitude", "long": "longitude", "speed": "sog",
    "speed_over_ground": "sog", "course": "cog", "course_over_ground": "cog",
    "true_heading": "heading", "hdg": "heading",
    "navstatus": "nav_status", "nav_status": "nav_status",
    "navigational_status": "nav_status", "shiptype": "vessel_type",
    "ship_type": "vessel_type", "type": "vessel_type", "type_code": "vessel_type",
    "vessel_name": "name", "shipname": "name", "flag": "flag",
}

# ITU-R M.1371 ship-and-cargo type codes -> coarse group used by scoring priors.
_TYPE_RANGES = [
    ((80, 89), "TANKER"),
    ((70, 79), "CARGO"),
    ((60, 69), "PASSENGER"),
    ((40, 49), "HSC"),
    ((30, 30), "FISHING"),
    ((31, 32), "TUG"),
    ((33, 35), "MILITARY"),
    ((36, 37), "PLEASURE"),
    ((50, 59), "TUG"),
    ((90, 99), "OTHER"),
]

NAV_STATUS = {
    0: "Under way using engine",
    1: "At anchor",
    2: "Not under command",
    3: "Restricted manoeuvrability",
    4: "Constrained by draught",
    5: "Moored",
    6: "Aground",
    7: "Engaged in fishing",
    8: "Under way sailing",
    11: "Under tow astern",
    12: "Under tow alongside",
    15: "Undefined",
}


def type_group(code) -> str:
    """AIS numeric ship type -> coarse group. Non-numeric / unknown -> 'UNKNOWN'."""
    try:
        c = int(float(code))
    except (TypeError, ValueError):
        return "UNKNOWN"
    for (lo, hi), name in _TYPE_RANGES:
        if lo <= c <= hi:
            return name
    return "UNKNOWN"


def nav_status_label(code) -> str:
    try:
        return NAV_STATUS.get(int(code), f"Status {int(code)}")
    except (TypeError, ValueError):
        return "Unknown"


# First three MMSI digits = Maritime Identification Digits -> flag state (ITU-R M.585).
MID_FLAG = {
    "201": "Albania", "202": "Andorra", "203": "Austria", "204": "Portugal",
    "205": "Belgium", "206": "Belarus", "207": "Bulgaria", "208": "Vatican",
    "209": "Cyprus", "210": "Cyprus", "211": "Germany", "212": "Cyprus",
    "213": "Georgia", "214": "Moldova", "215": "Malta", "216": "Armenia",
    "218": "Germany", "219": "Denmark", "220": "Denmark", "224": "Spain",
    "225": "Spain", "226": "France", "227": "France", "228": "France",
    "229": "Malta", "230": "Finland", "231": "Faroe Is.",
    "232": "United Kingdom", "233": "United Kingdom", "234": "United Kingdom",
    "235": "United Kingdom", "236": "Gibraltar", "237": "Greece",
    "238": "Croatia", "239": "Greece", "240": "Greece", "241": "Greece",
    "242": "Morocco", "243": "Hungary", "244": "Netherlands",
    "245": "Netherlands", "246": "Netherlands", "247": "Italy",
    "248": "Malta", "249": "Malta", "250": "Ireland", "251": "Iceland",
    "252": "Liechtenstein", "253": "Luxembourg", "254": "Monaco",
    "255": "Portugal", "256": "Malta", "257": "Norway", "258": "Norway",
    "259": "Norway", "261": "Poland", "262": "Montenegro", "263": "Portugal",
    "264": "Romania", "265": "Sweden", "266": "Sweden", "267": "Slovakia",
    "268": "San Marino", "269": "Switzerland", "270": "Czech Republic",
    "271": "Turkey", "272": "Ukraine", "273": "Russia",
    "274": "North Macedonia", "275": "Latvia", "276": "Estonia",
    "277": "Lithuania", "278": "Slovenia", "279": "Serbia",
    "303": "United States", "308": "Bahamas", "309": "Bahamas", "311": "Bahamas",
    "310": "Bermuda", "316": "Canada", "319": "Cayman Is.", "338": "United States",
    "351": "Panama", "352": "Panama", "353": "Panama", "354": "Panama",
    "355": "Panama", "356": "Panama", "357": "Panama", "370": "Panama",
    "371": "Panama", "372": "Panama", "373": "Panama", "374": "Panama",
    "366": "United States", "367": "United States", "368": "United States",
    "369": "United States", "375": "St Vincent", "376": "St Vincent",
    "377": "St Vincent",
    "403": "Saudi Arabia", "405": "Bangladesh", "408": "Bahrain", "412": "China",
    "413": "China", "414": "China", "416": "Taiwan", "417": "Sri Lanka",
    "419": "India", "422": "Iran", "423": "Azerbaijan", "425": "Iraq",
    "428": "Israel", "431": "Japan", "432": "Japan", "436": "Kazakhstan",
    "438": "Jordan", "440": "Korea", "441": "Korea", "445": "North Korea",
    "447": "Kuwait", "450": "Lebanon", "455": "Maldives", "457": "Mongolia",
    "459": "Nepal", "461": "Oman", "463": "Pakistan", "466": "Qatar",
    "470": "UAE", "471": "UAE", "473": "Yemen", "475": "Yemen",
    "477": "Hong Kong", "503": "Australia", "506": "Myanmar", "508": "Brunei",
    "512": "New Zealand", "525": "Indonesia", "533": "Malaysia",
    "538": "Marshall Is.", "548": "Philippines", "563": "Singapore",
    "564": "Singapore", "565": "Singapore", "566": "Singapore",
    "567": "Thailand", "574": "Vietnam", "576": "Vanuatu", "577": "Vanuatu",
    "601": "South Africa", "620": "Comoros", "636": "Liberia", "637": "Liberia",
    "642": "Libya", "645": "Mauritius", "657": "Nigeria", "664": "Seychelles",
    "667": "Sierra Leone", "710": "Brazil", "725": "Chile", "775": "Venezuela",
}


def flag_from_mmsi(mmsi) -> str:
    try:
        return MID_FLAG.get(str(int(mmsi))[:3], "Unknown")
    except (TypeError, ValueError):
        return "Unknown"
