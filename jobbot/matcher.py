"""Lightweight local scoring + location filtering."""
from __future__ import annotations

import re
from collections import Counter
from typing import Iterable

WORD_RE = re.compile(r"[A-Za-z+#.\-]{2,}")
STOP = {
    "the", "and", "for", "with", "you", "your", "our", "are", "this", "that",
    "from", "have", "has", "will", "all", "any", "but", "not", "they", "their",
}

REMOTE_TOKENS = ("remote", "anywhere", "work from home", "wfh", "telecommute", "distributed")
HYBRID_TOKENS = ("hybrid", "flexible location")


def _tokens(text: str) -> list[str]:
    return [w.lower() for w in WORD_RE.findall(text or "") if w.lower() not in STOP]


def score(resume_text: str, job_text: str, keywords: Iterable[str]) -> float:
    """0..1 — token overlap (resume vs job) + keyword bonus."""
    r_tokens = set(_tokens(resume_text))
    j_tokens = Counter(_tokens(job_text))
    if not j_tokens:
        return 0.0
    top = [t for t, _ in j_tokens.most_common(80)]
    overlap = sum(1 for t in top if t in r_tokens) / max(len(top), 1)
    kw = [k.lower() for k in keywords]
    kw_hits = sum(1 for k in kw if k and k in (job_text or "").lower())
    kw_score = kw_hits / max(len(kw), 1) if kw else 0.0
    return round(0.6 * overlap + 0.4 * kw_score, 3)


def excluded(text: str, excludes: Iterable[str]) -> bool:
    """Word-boundary match — 'intern' must not match 'international'."""
    t = (text or "").lower()
    for e in excludes:
        if not e:
            continue
        pattern = r"\b" + re.escape(e.lower()) + r"\b"
        if re.search(pattern, t):
            return True
    return False


def location_match(job_location: str, job_description: str, allowed: Iterable[str]) -> bool:
    """True if job location/description mentions any allowed location OR is remote/hybrid."""
    allowed_lower = [a.lower() for a in allowed if a]
    if not allowed_lower:
        return True
    hay = f"{job_location} {job_description}".lower()
    # Always allow remote/hybrid if user listed those tokens
    user_allows_remote = any(a in ("remote", "anywhere") for a in allowed_lower)
    user_allows_hybrid = "hybrid" in allowed_lower
    if user_allows_remote and any(t in hay for t in REMOTE_TOKENS):
        return True
    if user_allows_hybrid and any(t in hay for t in HYBRID_TOKENS):
        return True
    return any(loc in hay for loc in allowed_lower)


def company_in_watchlist(company: str, watchlist: Iterable[str]) -> bool:
    c = (company or "").lower().strip()
    if not c:
        return False
    return any(w.lower() in c or c in w.lower() for w in watchlist if w)
