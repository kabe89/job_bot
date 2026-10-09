# jobbot/discovery/service.py
"""Orchestration shared by the CLI and the dashboard: run discovery sources
through validate + fit-score into the pending store, and approve/reject."""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

from ..config import settings
from ..profile import load_profile
from . import store, targets
from .discoverers import (harvest_from_jobs, llm_suggest_companies,
                          websearch_companies)
from .model import DiscoveredTarget
from .validate import fit_score, validate

log = logging.getLogger("jobbot.discovery.service")


def parse_coord(s: str) -> Tuple[str, str]:
    provider, _, key = s.partition(":")
    return (provider, key)


def _process(candidates: List[DiscoveredTarget]) -> dict:
    existing = targets.existing_coords()
    profile = load_profile()
    added = 0
    seen_invalid = 0
    for cand in candidates:
        if cand.coord() in existing:
            continue
        if store.get(cand.coord()) is not None:   # already pending/approved/rejected
            continue
        cand = validate(cand)
        if cand.valid:
            cand = fit_score(cand, profile)
        else:
            seen_invalid += 1
        if store.upsert(cand) and cand.valid:
            added += 1
    return {"candidates": len(candidates), "added": added, "invalid": seen_invalid}


def run_harvest() -> dict:
    try:
        return _process(harvest_from_jobs())
    except Exception as e:  # noqa: BLE001
        log.warning("run_harvest failed (%s).", e)
        return {"candidates": 0, "added": 0, "invalid": 0, "error": str(e)}


def run_discovery(sources: List[str]) -> dict:
    # Fail-open, symmetric with run_harvest: never raise into the CLI (review m1).
    try:
        cands: List[DiscoveredTarget] = []
        profile = load_profile()
        if "harvest" in sources:
            cands += harvest_from_jobs()
        if "websearch" in sources:
            cands += websearch_companies(profile)
        if "llm" in sources:
            cands += llm_suggest_companies(profile)
        return _process(cands)
    except Exception as e:  # noqa: BLE001
        log.warning("run_discovery failed (%s).", e)
        return {"candidates": 0, "added": 0, "invalid": 0, "error": str(e)}


def review(min_fit: Optional[float] = None) -> List[DiscoveredTarget]:
    floor = settings.discovery_min_fit_score if min_fit is None else min_fit
    return store.list_pending(min_fit=floor)


def approve(coords: List[Tuple[str, str]], watchlist: bool = False) -> dict:
    n = 0
    for coord in coords:
        t = store.get(coord)
        if t is None:
            continue
        targets.append(t)
        if watchlist:
            targets.append_watchlist(t)
        store.set_status(coord, "approved")
        n += 1
    return {"approved": n}


def reject(coords: List[Tuple[str, str]]) -> dict:
    n = sum(1 for coord in coords if store.set_status(coord, "rejected"))
    return {"rejected": n}
