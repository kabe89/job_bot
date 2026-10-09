# jobbot/discovery/store.py
"""Pending discovery queue (JSON). Atomic writes; coord-keyed dedup."""
from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import List, Optional, Tuple

from ..config import settings
from .model import DiscoveredTarget

log = logging.getLogger("jobbot.discovery.store")

# Serializes read-modify-write so the auto-harvest pass (in the /scrape worker
# thread) and a dashboard "Run discovery" thread can't interleave load/save and
# drop one side's additions (review m3).
_write_lock = threading.Lock()


def _load() -> List[DiscoveredTarget]:
    p = Path(settings.discovery_pending_path)
    if not p.exists():
        return []
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        return [DiscoveredTarget.from_dict(d) for d in raw]
    except Exception as e:  # noqa: BLE001
        log.warning("Could not read pending store (%s) — treating as empty.", e)
        return []


def _save(items: List[DiscoveredTarget]) -> None:
    p = Path(settings.discovery_pending_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps([t.to_dict() for t in items], indent=2), encoding="utf-8")
    os.replace(tmp, p)


def upsert(target: DiscoveredTarget) -> bool:
    with _write_lock:
        items = _load()
        if any(t.coord() == target.coord() for t in items):
            return False
        if not target.first_seen:
            from datetime import datetime
            target.first_seen = datetime.utcnow().isoformat()
        items.append(target)
        _save(items)
        return True


def get(coord: Tuple[str, str]) -> Optional[DiscoveredTarget]:
    for t in _load():
        if t.coord() == coord:
            return t
    return None


def set_status(coord: Tuple[str, str], status: str) -> bool:
    with _write_lock:
        items = _load()
        changed = False
        for t in items:
            if t.coord() == coord:
                t.status = status
                changed = True
        if changed:
            _save(items)
        return changed


def list_pending(min_fit: float = 0.0, valid_only: bool = True) -> List[DiscoveredTarget]:
    items = [t for t in _load() if t.status == "pending"]
    if valid_only:
        items = [t for t in items if t.valid]
    items = [t for t in items if t.fit_score >= min_fit]
    return sorted(items, key=lambda t: t.fit_score, reverse=True)
