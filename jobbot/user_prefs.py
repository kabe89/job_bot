"""Persisted user preferences edited from the dashboard.

Currently:
  - `data/user_locations.txt` — extra locations the user wants to match
    (one per line, comma- or newline-separated; "anywhere"/"" disables filter).

Falls back to the baseline `SEARCH_LOCATIONS` from .env when the user file is
missing.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List

from .config import settings  # noqa: F401  (kept for callers + tests)

# Env-overridable so tests never read/write the real data/ files (conftest
# points these at a temp dir before jobbot is imported). The /locations and
# /tags save routes write here, so an un-isolated path lets the test suite
# clobber the user's real preferences.
USER_LOCATIONS_FILE = Path(os.environ.get("USER_LOCATIONS_FILE", "data/user_locations.txt"))
USER_TAGS_FILE = Path(os.environ.get("USER_TAGS_FILE", "data/user_tags.txt"))


def _normalize(raw: str) -> List[str]:
    """Parse multi-line OR comma-separated input.

    - Multi-line input is split on newlines only — commas stay intact so
      "Rockville, MD" is one location.
    - Single-line input is split on commas so "New York, Boston, Seattle"
      becomes three entries (since you'd lose state info either way).
    """
    # Split on newlines AND commas — the re-glue step below restores
    # 2-letter state suffixes back onto the preceding city.
    tokens = raw.replace("\r", "\n").replace(",", "\n").split("\n")
    # Re-glue 2-letter US-state codes onto the preceding city
    # (so "Rockville, MD" on one line stays "Rockville, MD" even though we
    # split on the comma).
    cleaned: list[str] = []
    for token in tokens:
        t = token.strip().strip(",").strip()
        if not t or t.startswith("#"):
            continue
        if len(t) == 2 and t.isupper() and cleaned:
            cleaned[-1] = f"{cleaned[-1]}, {t}"
        else:
            cleaned.append(t)
    out: list[str] = []
    seen: set[str] = set()
    for t in cleaned:
        key = t.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
    return out


def load_locations() -> List[str]:
    """User-edited locations if available, else baseline from settings."""
    if USER_LOCATIONS_FILE.exists():
        try:
            data = USER_LOCATIONS_FILE.read_text(encoding="utf-8")
            locs = _normalize(data)
            if locs:
                return locs
        except Exception:
            pass
    return settings.locations


def save_locations(raw: str) -> List[str]:
    locs = _normalize(raw)
    USER_LOCATIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    USER_LOCATIONS_FILE.write_text(
        "# Locations to match jobs against. One per line or comma-separated.\n"
        "# Tokens 'remote' / 'hybrid' have special meaning — they include those job types.\n"
        "# Save an empty file (or list just 'anywhere') to disable the location filter.\n\n"
        + "\n".join(locs) + ("\n" if locs else ""),
        encoding="utf-8",
    )
    return locs


def is_anywhere_mode(locations: List[str]) -> bool:
    if not locations:
        return True
    return any(l.lower() in ("anywhere", "any", "*") for l in locations)


def scrape_locations() -> List[str]:
    """The effective list of cities scrapers should query.

    1. Start with user_locations.txt (or .env baseline).
    2. Drop special tokens (remote/hybrid/anywhere) — those are job-type filters,
       not searchable cities.
    3. Expand with cities within `SCRAPE_RADIUS_MILES` of each.
    4. Cap to `MAX_SCRAPE_LOCATIONS` to avoid runaway HTTP cost.

    Used by LinkedIn, Indeed, and Google Jobs scrapers.
    """
    from .geo import expand_locations_with_radius
    base = load_locations()
    # Drop the job-type-token aliases — these aren't searchable cities
    cities = [l for l in base if l.lower() not in ("remote", "hybrid", "anywhere", "any", "*")]
    radius = max(0, int(getattr(settings, "scrape_radius_miles", 0)))
    expanded = expand_locations_with_radius(cities, radius) if cities else []
    cap = max(1, int(getattr(settings, "max_scrape_locations", 12)))
    return expanded[:cap]


def _normalize_tags(raw: str) -> List[str]:
    tokens = raw.replace("\r", "\n").replace(",", "\n").split("\n")
    out: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        t = token.strip()
        if not t or t.startswith("#"):
            continue
        key = t.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
    return out


def load_tags() -> List[str]:
    """User-edited scrape tags (keywords) if available, else baseline from settings."""
    if USER_TAGS_FILE.exists():
        try:
            data = USER_TAGS_FILE.read_text(encoding="utf-8")
            tags = _normalize_tags(data)
            if tags:
                return tags
        except Exception:
            pass
    return settings.keywords


def save_tags(raw: str) -> List[str]:
    tags = _normalize_tags(raw)
    USER_TAGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    USER_TAGS_FILE.write_text(
        "# Tags / keywords scrapers query for. One per line or comma-separated.\n"
        "# Save an empty file to fall back to SEARCH_KEYWORDS from .env.\n\n"
        + "\n".join(tags) + ("\n" if tags else ""),
        encoding="utf-8",
    )
    return tags


def includes_remote() -> bool:
    """True if the user-saved locations include the 'remote' token (so scrapers
    should also query the Remote/United States variant)."""
    return any(l.lower() == "remote" for l in load_locations())
