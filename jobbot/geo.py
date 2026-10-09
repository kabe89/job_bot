"""City coordinates + mile-radius location matching.

Adds an optional **radius filter** so the user can pick "within N miles of
Boston MA" or "within 75 mi of Rockville MD" and we expand the location-match
to include every city within that great-circle distance.

Coordinates are bundled (no live geocoding) — covers major US metropolitan hubs
and the user's home regions. To add more, append (name, lat, lon) tuples to
`CITIES` or `data/user_cities.txt`.
"""
from __future__ import annotations

import math
import re
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Tuple

USER_CITIES_FILE = Path("data/user_cities.txt")

# (lowercase-aliases, latitude, longitude, canonical_name)
CITIES: list[tuple[list[str], float, float, str]] = [
    # Chicago / Midwest
    (["chicago", "chicago il"],                                    41.8781, -87.6298, "Chicago, IL"),
    (["evanston", "evanston il"],                                  42.0451, -87.6877, "Evanston, IL"),
    (["naperville", "naperville il"],                              41.7508, -88.1535, "Naperville, IL"),
    # Seattle / Pacific Northwest
    (["seattle", "seattle wa"],                                    47.6062, -122.3321, "Seattle, WA"),
    (["bellevue", "bellevue wa"],                                  47.6101, -122.2015, "Bellevue, WA"),
    (["redmond", "redmond wa"],                                    47.6740, -122.1215, "Redmond, WA"),
    # NYC metro / Westchester / Suburbs
    (["tarrytown", "tarrytown ny"],                                41.0762, -73.8587, "Tarrytown, NY"),
    (["sleepy hollow"],                                            41.0859, -73.8590, "Sleepy Hollow, NY"),
    (["pearl river", "pearl river ny"],                            41.0590, -74.0218, "Pearl River, NY"),
    (["new york", "new york city", "nyc", "manhattan"],            40.7128, -74.0060, "New York, NY"),
    # Boston / Cambridge MA
    (["boston"],                                                    42.3601, -71.0589, "Boston, MA"),
    (["cambridge", "cambridge ma"],                                42.3736, -71.1097, "Cambridge, MA"),
    (["somerville", "somerville ma"],                              42.3876, -71.0995, "Somerville, MA"),
    (["waltham", "waltham ma"],                                    42.3765, -71.2356, "Waltham, MA"),
    (["lexington", "lexington ma"],                                42.4430, -71.2290, "Lexington, MA"),
    # Mid-Atlantic corridor — Rockville / Bethesda / Frederick MD + DC + Philly
    (["rockville", "rockville md"],                                39.0840, -77.1528, "Rockville, MD"),
    (["bethesda", "bethesda md"],                                  38.9847, -77.0947, "Bethesda, MD"),
    (["gaithersburg"],                                              39.1434, -77.2014, "Gaithersburg, MD"),
    (["frederick md", "frederick"],                                39.4143, -77.4105, "Frederick, MD"),
    (["baltimore"],                                                 39.2904, -76.6122, "Baltimore, MD"),
    (["college park"],                                              38.9897, -76.9378, "College Park, MD"),
    (["washington dc", "washington, dc", "washington d.c.", "dc"], 38.9072, -77.0369, "Washington, DC"),
    (["philadelphia", "philly"],                                   39.9526, -75.1652, "Philadelphia, PA"),
    (["princeton", "princeton nj"],                                40.3573, -74.6672, "Princeton, NJ"),
    # Research Triangle NC
    (["research triangle", "rtp", "durham", "durham nc"],          35.9940, -78.8986, "Durham, NC"),
    (["raleigh"],                                                   35.7796, -78.6382, "Raleigh, NC"),
    (["chapel hill"],                                               35.9132, -79.0558, "Chapel Hill, NC"),
    # SF Bay
    (["san francisco", "sf"],                                      37.7749, -122.4194, "San Francisco, CA"),
    (["south san francisco", "ssf"],                               37.6547, -122.4077, "South San Francisco, CA"),
    (["palo alto"],                                                 37.4419, -122.1430, "Palo Alto, CA"),
    (["berkeley"],                                                  37.8716, -122.2727, "Berkeley, CA"),
    (["emeryville"],                                                37.8313, -122.2852, "Emeryville, CA"),
    # San Diego / La Jolla
    (["san diego"],                                                 32.7157, -117.1611, "San Diego, CA"),
    (["la jolla"],                                                  32.8328, -117.2713, "La Jolla, CA"),
    # Additional major metros
    (["seattle"],                                                   47.6062, -122.3321, "Seattle, WA"),
    (["chicago"],                                                   41.8781, -87.6298, "Chicago, IL"),
    (["austin"],                                                    30.2672, -97.7431, "Austin, TX"),
    (["new haven"],                                                 41.3083, -72.9279, "New Haven, CT"),
    (["groton ct", "groton"],                                       41.3501, -72.0789, "Groton, CT"),
]


def _add_user_cities():
    """Append entries from data/user_cities.txt — format: name,lat,lon"""
    if not USER_CITIES_FILE.exists():
        return
    try:
        for line in USER_CITIES_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                name = parts[0]
                lat = float(parts[1]); lon = float(parts[2])
                aliases = [name.lower()] + [p.strip().lower() for p in parts[3:]]
                CITIES.append((aliases, lat, lon, name))
    except Exception:
        pass


_add_user_cities()


def _haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R_MILES = 3958.7613
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlam / 2) ** 2
    return 2 * R_MILES * math.asin(math.sqrt(a))


@lru_cache(maxsize=256)
def find_city(name: str) -> tuple[float, float, str] | None:
    """Resolve a user-supplied location string to (lat, lon, canonical)."""
    if not name:
        return None
    needle = name.lower().strip()
    # Exact alias match first
    for aliases, lat, lon, canon in CITIES:
        if needle in aliases:
            return (lat, lon, canon)
    # Substring fallback (e.g. "Rockville, MD 20850" -> rockville)
    for aliases, lat, lon, canon in CITIES:
        for a in aliases:
            if a in needle or needle.startswith(a):
                return (lat, lon, canon)
    return None


def cities_within(center: str, radius_miles: float) -> List[str]:
    """Return canonical city names within radius_miles of the center city."""
    c = find_city(center)
    if not c:
        return []
    lat, lon, _ = c
    out = []
    for aliases, lat2, lon2, canon in CITIES:
        if _haversine_miles(lat, lon, lat2, lon2) <= radius_miles:
            out.append(canon)
    return out


def expand_locations_with_radius(locations: Iterable[str], radius_miles: float) -> List[str]:
    """For each user-supplied location, add every known city within `radius_miles`.

    Prefer the canonical city name when a user alias resolves to a known city
    (so the output list has "Albany, NY" instead of both "albany" and "Albany, NY").
    """
    expanded: list[str] = []
    seen: set[str] = set()
    for loc in locations:
        if not loc:
            continue
        resolved = find_city(loc)
        canonical = resolved[2] if resolved else loc
        if canonical.lower() not in seen:
            seen.add(canonical.lower())
            expanded.append(canonical)
        if radius_miles > 0 and resolved:
            for nearby in cities_within(canonical, radius_miles):
                if nearby.lower() not in seen:
                    seen.add(nearby.lower())
                    expanded.append(nearby)
    return expanded


def location_in_radius(job_location: str, centers: Iterable[str], radius_miles: float) -> bool:
    """True if the job's location matches any city within radius of any center.

    Uses the bundled CITIES table for the job side too: we try to find a known
    city alias inside the job-location string.
    """
    if not job_location or not centers:
        return False
    # Find a known city in the job location string
    needle = job_location.lower()
    for aliases, lat2, lon2, canon in CITIES:
        if any(re.search(rf"\b{re.escape(a)}\b", needle) for a in aliases):
            # Found a known city — check distance to any center
            for center in centers:
                c = find_city(center)
                if c and _haversine_miles(c[0], c[1], lat2, lon2) <= radius_miles:
                    return True
    return False
