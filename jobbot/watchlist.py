"""Parse the company watchlist Markdown and expose company names."""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import List

from .config import settings

BULLET_RE = re.compile(r"^[-*]\s+(.+?)(?:\s+[\-—(].*)?$")


@lru_cache(maxsize=1)
def load_companies() -> List[str]:
    p = Path(settings.company_watchlist)
    if not p.exists():
        return []
    names: List[str] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        m = BULLET_RE.match(line.strip())
        if not m:
            continue
        # Strip parenthetical and em-dash trailers, but keep internal hyphens (Bio-Techne, X-Chem).
        name = re.split(r"\s+[—–]\s+|\s*\(", m.group(1), maxsplit=1)[0].strip()
        if name and name not in names:
            names.append(name)
    return names


def refresh() -> None:
    load_companies.cache_clear()
