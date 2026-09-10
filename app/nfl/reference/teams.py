"""Franchise identity, relocation handling and venue geography.

Canonical team codes follow the modern nflverse play-by-play codes (``LA``, ``LAC``,
``LV``, ``WAS``). Historical schedule codes (``STL``, ``SD``, ``OAK``) map onto the same
franchise so ratings and rolling features continue across relocations. Coordinates are
approximate stadium locations (accurate to well under a mile for travel-distance purposes).
"""
from __future__ import annotations
import math
from datetime import datetime
from zoneinfo import ZoneInfo

CANONICAL = {"STL": "LA", "LAR": "LA", "SD": "LAC", "OAK": "LV", "WSH": "WAS", "JAC": "JAX", "HST": "HOU", "BLT": "BAL", "CLV": "CLE", "ARZ": "ARI"}
DISPLAY = {"LA": "LAR", "WAS": "WSH"}  # ESPN style aliases used by the existing app
TEAMS = ("ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN", "DET", "GB", "HOU", "IND", "JAX", "KC", "LA", "LAC", "LV", "MIA", "MIN", "NE", "NO", "NYG", "NYJ", "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS")

DIVISIONS = {
    "BUF": "AFC East", "MIA": "AFC East", "NE": "AFC East", "NYJ": "AFC East",
    "BAL": "AFC North", "CIN": "AFC North", "CLE": "AFC North", "PIT": "AFC North",
    "HOU": "AFC South", "IND": "AFC South", "JAX": "AFC South", "TEN": "AFC South",
    "DEN": "AFC West", "KC": "AFC West", "LV": "AFC West", "LAC": "AFC West",
    "DAL": "NFC East", "NYG": "NFC East", "PHI": "NFC East", "WAS": "NFC East",
    "CHI": "NFC North", "DET": "NFC North", "GB": "NFC North", "MIN": "NFC North",
    "ATL": "NFC South", "CAR": "NFC South", "NO": "NFC South", "TB": "NFC South",
    "ARI": "NFC West", "LA": "NFC West", "SF": "NFC West", "SEA": "NFC West",
}


def canonical(team) -> str | None:
    if team is None:
        return None
    code = str(team).strip().upper()
    if not code or code == "NAN":
        return None
    return CANONICAL.get(code, code)


# (latitude, longitude, IANA time zone). Multiple entries per franchise cover relocations;
# the first entry whose ``until`` season is >= the game season applies.
HOME_VENUES: dict[str, list[dict]] = {
    "ARI": [{"until": 2005, "lat": 33.4255, "lon": -111.9325, "tz": "America/Phoenix", "name": "Sun Devil Stadium"}, {"until": 9999, "lat": 33.5276, "lon": -112.2626, "tz": "America/Phoenix", "name": "State Farm Stadium"}],
    "ATL": [{"until": 2016, "lat": 33.7575, "lon": -84.4009, "tz": "America/New_York", "name": "Georgia Dome"}, {"until": 9999, "lat": 33.7554, "lon": -84.4010, "tz": "America/New_York", "name": "Mercedes-Benz Stadium"}],
    "BAL": [{"until": 9999, "lat": 39.2780, "lon": -76.6227, "tz": "America/New_York", "name": "M&T Bank Stadium"}],
    "BUF": [{"until": 9999, "lat": 42.7738, "lon": -78.7870, "tz": "America/New_York", "name": "Highmark Stadium"}],
    "CAR": [{"until": 9999, "lat": 35.2258, "lon": -80.8528, "tz": "America/New_York", "name": "Bank of America Stadium"}],
    "CHI": [{"until": 9999, "lat": 41.8623, "lon": -87.6167, "tz": "America/Chicago", "name": "Soldier Field"}],
    "CIN": [{"until": 9999, "lat": 39.0955, "lon": -84.5161, "tz": "America/New_York", "name": "Paycor Stadium"}],
    "CLE": [{"until": 9999, "lat": 41.5061, "lon": -81.6995, "tz": "America/New_York", "name": "Huntington Bank Field"}],
    "DAL": [{"until": 2008, "lat": 32.8407, "lon": -96.9111, "tz": "America/Chicago", "name": "Texas Stadium"}, {"until": 9999, "lat": 32.7473, "lon": -97.0945, "tz": "America/Chicago", "name": "AT&T Stadium"}],
    "DEN": [{"until": 9999, "lat": 39.7439, "lon": -105.0201, "tz": "America/Denver", "name": "Empower Field at Mile High"}],
    "DET": [{"until": 2001, "lat": 42.6459, "lon": -83.2554, "tz": "America/New_York", "name": "Pontiac Silverdome"}, {"until": 9999, "lat": 42.3400, "lon": -83.0456, "tz": "America/New_York", "name": "Ford Field"}],
    "GB": [{"until": 9999, "lat": 44.5013, "lon": -88.0622, "tz": "America/Chicago", "name": "Lambeau Field"}],
    "HOU": [{"until": 9999, "lat": 29.6847, "lon": -95.4107, "tz": "America/Chicago", "name": "NRG Stadium"}],
    "IND": [{"until": 2007, "lat": 39.7640, "lon": -86.1636, "tz": "America/Indiana/Indianapolis", "name": "RCA Dome"}, {"until": 9999, "lat": 39.7601, "lon": -86.1639, "tz": "America/Indiana/Indianapolis", "name": "Lucas Oil Stadium"}],
    "JAX": [{"until": 9999, "lat": 30.3240, "lon": -81.6373, "tz": "America/New_York", "name": "EverBank Stadium"}],
    "KC": [{"until": 9999, "lat": 39.0489, "lon": -94.4839, "tz": "America/Chicago", "name": "GEHA Field at Arrowhead Stadium"}],
    "LA": [{"until": 2015, "lat": 38.6328, "lon": -90.1885, "tz": "America/Chicago", "name": "Edward Jones Dome"}, {"until": 2019, "lat": 34.0141, "lon": -118.2879, "tz": "America/Los_Angeles", "name": "Los Angeles Memorial Coliseum"}, {"until": 9999, "lat": 33.9535, "lon": -118.3392, "tz": "America/Los_Angeles", "name": "SoFi Stadium"}],
    "LAC": [{"until": 2016, "lat": 32.7831, "lon": -117.1196, "tz": "America/Los_Angeles", "name": "Qualcomm Stadium"}, {"until": 2019, "lat": 33.8644, "lon": -118.2611, "tz": "America/Los_Angeles", "name": "Dignity Health Sports Park"}, {"until": 9999, "lat": 33.9535, "lon": -118.3392, "tz": "America/Los_Angeles", "name": "SoFi Stadium"}],
    "LV": [{"until": 2019, "lat": 37.7516, "lon": -122.2005, "tz": "America/Los_Angeles", "name": "Oakland Coliseum"}, {"until": 9999, "lat": 36.0909, "lon": -115.1833, "tz": "America/Los_Angeles", "name": "Allegiant Stadium"}],
    "MIA": [{"until": 9999, "lat": 25.9580, "lon": -80.2389, "tz": "America/New_York", "name": "Hard Rock Stadium"}],
    "MIN": [{"until": 2013, "lat": 44.9740, "lon": -93.2581, "tz": "America/Chicago", "name": "Metrodome"}, {"until": 2015, "lat": 44.9764, "lon": -93.2247, "tz": "America/Chicago", "name": "TCF Bank Stadium"}, {"until": 9999, "lat": 44.9736, "lon": -93.2575, "tz": "America/Chicago", "name": "U.S. Bank Stadium"}],
    "NE": [{"until": 2001, "lat": 42.0909, "lon": -71.2643, "tz": "America/New_York", "name": "Foxboro Stadium"}, {"until": 9999, "lat": 42.0909, "lon": -71.2643, "tz": "America/New_York", "name": "Gillette Stadium"}],
    "NO": [{"until": 9999, "lat": 29.9511, "lon": -90.0812, "tz": "America/Chicago", "name": "Caesars Superdome"}],
    "NYG": [{"until": 2009, "lat": 40.8128, "lon": -74.0742, "tz": "America/New_York", "name": "Giants Stadium"}, {"until": 9999, "lat": 40.8135, "lon": -74.0745, "tz": "America/New_York", "name": "MetLife Stadium"}],
    "NYJ": [{"until": 2009, "lat": 40.8128, "lon": -74.0742, "tz": "America/New_York", "name": "Giants Stadium"}, {"until": 9999, "lat": 40.8135, "lon": -74.0745, "tz": "America/New_York", "name": "MetLife Stadium"}],
    "PHI": [{"until": 2002, "lat": 39.9061, "lon": -75.1717, "tz": "America/New_York", "name": "Veterans Stadium"}, {"until": 9999, "lat": 39.9008, "lon": -75.1675, "tz": "America/New_York", "name": "Lincoln Financial Field"}],
    "PIT": [{"until": 2000, "lat": 40.4468, "lon": -80.0134, "tz": "America/New_York", "name": "Three Rivers Stadium"}, {"until": 9999, "lat": 40.4468, "lon": -80.0158, "tz": "America/New_York", "name": "Acrisure Stadium"}],
    "SEA": [{"until": 2001, "lat": 47.6534, "lon": -122.3016, "tz": "America/Los_Angeles", "name": "Husky Stadium"}, {"until": 9999, "lat": 47.5952, "lon": -122.3316, "tz": "America/Los_Angeles", "name": "Lumen Field"}],
    "SF": [{"until": 2013, "lat": 37.7136, "lon": -122.3861, "tz": "America/Los_Angeles", "name": "Candlestick Park"}, {"until": 9999, "lat": 37.4030, "lon": -121.9700, "tz": "America/Los_Angeles", "name": "Levi's Stadium"}],
    "TB": [{"until": 9999, "lat": 27.9759, "lon": -82.5033, "tz": "America/New_York", "name": "Raymond James Stadium"}],
    "TEN": [{"until": 9999, "lat": 36.1665, "lon": -86.7713, "tz": "America/Chicago", "name": "Nissan Stadium"}],
    "WAS": [{"until": 9999, "lat": 38.9076, "lon": -76.8645, "tz": "America/New_York", "name": "Northwest Stadium"}],
}

# Neutral / international venues matched on stadium name fragments (lower case).
NEUTRAL_VENUES = [
    ("wembley", 51.5560, -0.2796, "Europe/London"), ("tottenham", 51.6043, -0.0664, "Europe/London"), ("twickenham", 51.4559, -0.3415, "Europe/London"),
    ("azteca", 19.3029, -99.1505, "America/Mexico_City"), ("allianz", 48.2188, 11.6247, "Europe/Berlin"), ("deutsche bank", 50.0686, 8.6455, "Europe/Berlin"), ("frankfurt", 50.0686, 8.6455, "Europe/Berlin"), ("munich", 48.2188, 11.6247, "Europe/Berlin"),
    ("corinthians", -23.5453, -46.4742, "America/Sao_Paulo"), ("neo qu", -23.5453, -46.4742, "America/Sao_Paulo"), ("sao paulo", -23.5453, -46.4742, "America/Sao_Paulo"), ("são paulo", -23.5453, -46.4742, "America/Sao_Paulo"),
    ("melbourne", -37.8200, 144.9834, "Australia/Melbourne"), ("bernab", 40.4531, -3.6883, "Europe/Madrid"), ("madrid", 40.4531, -3.6883, "Europe/Madrid"), ("croke", 53.3607, -6.2512, "Europe/Dublin"), ("dublin", 53.3607, -6.2512, "Europe/Dublin"),
    ("rogers centre", 43.6414, -79.3894, "America/Toronto"), ("toronto", 43.6414, -79.3894, "America/Toronto"), ("tokyo", 35.7056, 139.7519, "Asia/Tokyo"), ("estadio", 19.3029, -99.1505, "America/Mexico_City"),
    ("tom benson", 40.8163, -81.3990, "America/New_York"), ("canton", 40.8163, -81.3990, "America/New_York"), ("tempe", 33.4255, -111.9325, "America/Phoenix"), ("sun devil", 33.4255, -111.9325, "America/Phoenix"), ("liberty bowl", 35.1213, -89.9773, "America/Chicago"), ("tiger stadium", 30.4120, -91.1837, "America/Chicago"),
    ("alamodome", 29.4169, -98.4789, "America/Chicago"), ("ford field", 42.3400, -83.0456, "America/New_York"), ("champions field", 30.3240, -81.6373, "America/New_York"), ("estadio azteca", 19.3029, -99.1505, "America/Mexico_City"),
]


def home_venue(team: str, season: int) -> dict:
    code = canonical(team) or team
    entries = HOME_VENUES.get(code)
    if not entries:
        return {"lat": 39.8, "lon": -98.6, "tz": "America/Chicago", "name": "unknown"}
    for entry in entries:
        if season <= entry["until"]:
            return entry
    return entries[-1]


def venue_for_game(home_team: str, season: int, stadium: str | None, neutral: bool) -> dict:
    """Best-effort venue for a game: named neutral venue, otherwise the home team's stadium."""
    name = (stadium or "").lower()
    if neutral and name:
        for fragment, lat, lon, tz in NEUTRAL_VENUES:
            if fragment in name:
                return {"lat": lat, "lon": lon, "tz": tz, "name": stadium, "matched": fragment}
        # Super Bowls and other neutral games in NFL stadiums: match a franchise venue by name.
        for code, entries in HOME_VENUES.items():
            for entry in entries:
                if entry["name"].lower() in name or name in entry["name"].lower():
                    return entry | {"matched": code}
    return home_venue(home_team, season) | {"matched": "home"}


def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 3958.8
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def utc_offset_hours(tz: str, when: datetime) -> float:
    try:
        offset = when.astimezone(ZoneInfo(tz)).utcoffset()
        return offset.total_seconds() / 3600 if offset is not None else 0.0
    except Exception:  # noqa: BLE001 - unknown zone falls back to UTC
        return 0.0
