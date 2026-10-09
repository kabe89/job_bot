# jobbot/discovery/targets.py
"""Sole owner of the four live target-file formats: read existing coordinates
(for dedup) and append an approved DiscoveredTarget in each native format."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Set, Tuple

from .model import DiscoveredTarget

log = logging.getLogger("jobbot.discovery.targets")

ATS_TARGETS_FILE = Path("data/ats_targets.txt")        # provider,slug  (greenhouse|lever)
ASHBY_TARGETS_FILE = Path("data/ashby_targets.txt")    # slug,display_name
WORKDAY_TARGETS_FILE = Path("data/workday_targets.txt")  # tenant,board,name[,host]
COMPANIES_FILE = Path("data/companies.md")             # - Name


def _seed_coords() -> Set[Tuple[str, str]]:
    """Coordinates baked into the scrapers' DEFAULT_TARGETS."""
    out: Set[Tuple[str, str]] = set()
    from ..scrapers.company_pages import DEFAULT_TARGETS as GH
    for provider, slug in GH:
        out.add((provider, slug))
    from ..scrapers.ashby import DEFAULT_TARGETS as AS
    for slug, _name in AS:
        out.add(("ashby", slug))
    from ..scrapers.workday import DEFAULT_TARGETS as WD
    for tenant, board, _name, _host in WD:
        out.add(("workday", f"{tenant}/{board}"))
    return out


def _file_coords() -> Set[Tuple[str, str]]:
    out: Set[Tuple[str, str]] = set()
    if ATS_TARGETS_FILE.exists():
        for line in ATS_TARGETS_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "," in line:
                provider, slug = [x.strip() for x in line.split(",", 1)]
                out.add((provider, slug))
    if ASHBY_TARGETS_FILE.exists():
        for line in ASHBY_TARGETS_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                out.add(("ashby", line.split(",")[0].strip()))
    if WORKDAY_TARGETS_FILE.exists():
        for line in WORKDAY_TARGETS_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 2:
                    out.add(("workday", f"{parts[0]}/{parts[1]}"))
    return out


def existing_coords() -> Set[Tuple[str, str]]:
    return _seed_coords() | _file_coords()


def _watchlist_names() -> Set[str]:
    from ..watchlist import load_companies
    try:
        return {c.strip().lower() for c in load_companies()}
    except Exception:  # noqa: BLE001
        return set()


def _append_line(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def append(target: DiscoveredTarget) -> bool:
    """Append target to its native file. Idempotent. Returns True if written."""
    if target.coord() in existing_coords():
        return False
    p = target.provider
    if p in ("greenhouse", "lever"):
        _append_line(ATS_TARGETS_FILE, f"{p},{target.key}")
    elif p == "ashby":
        _append_line(ASHBY_TARGETS_FILE, f"{target.key},{target.display_name}")
    elif p == "workday":
        tenant, _, board = target.key.partition("/")
        host = target.wd_host or "wd1"
        _append_line(WORKDAY_TARGETS_FILE, f"{tenant},{board},{target.display_name},{host}")
    else:
        log.warning("append: unknown provider %r", p)
        return False
    return True


def append_watchlist(target: DiscoveredTarget) -> bool:
    """Append the display name to companies.md. Idempotent. Returns True if written."""
    name_lower = target.display_name.strip().lower()
    if name_lower in _watchlist_names():
        return False
    # Also check the file directly (covers cases where _watchlist_names doesn't
    # reflect lines written earlier in the same process, e.g. during tests).
    if COMPANIES_FILE.exists():
        for line in COMPANIES_FILE.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("- ") and stripped[2:].strip().lower() == name_lower:
                return False
    _append_line(COMPANIES_FILE, f"- {target.display_name}")
    # Invalidate the watchlist lru_cache so in-process consumers (matcher /
    # scrape cycle) see the newly-approved company without a restart (review m2).
    try:
        from ..watchlist import load_companies
        load_companies.cache_clear()
    except Exception:  # noqa: BLE001
        pass
    return True
