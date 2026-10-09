# jobbot/query_expansion.py
"""Derive a richer set of search queries from the candidate profile so the
scrapers cast a wider, better-targeted net. One Ollama call per profile; merged
with the user's static tags in the pipeline. Fail-open: returns [] on any error.
"""
from __future__ import annotations

import json
import logging
from typing import List

from . import gemini_client as _g
from . import ollama_client as oc
from .config import settings
from .profile import CandidateProfile

log = logging.getLogger("jobbot.query_expansion")

_PROMPT = """Given this candidate profile, list the best job-board SEARCH QUERIES
(job titles + close synonyms + role archetypes) to find relevant openings.
Short title-like phrases only — no boolean operators, no locations.
Return STRICT JSON: {{"queries": ["...", "..."]}} — at most {cap} items.

PROFILE:
titles: {titles}
archetypes: {archetypes}
skills: {skills}
domains: {domains}
summary: {summary}"""


def expand_queries(profile: CandidateProfile) -> List[str]:
    cap = int(settings.query_expansion_max or 25)
    try:
        out = oc._generate(
            _PROMPT.format(
                cap=cap,
                titles=", ".join(profile.role_titles),
                archetypes=", ".join(profile.role_archetypes),
                skills=", ".join(profile.skills[:20]),
                domains=", ".join(profile.domains),
                summary=profile.summary),
            temperature=0.3, include_profile=False)
        data = _g._parse_json_block(out) or {}
        raw = data.get("queries") or []
    except Exception as e:  # noqa: BLE001
        log.warning("query expansion failed (%s) — using static tags only.", e)
        return []

    seen: set[str] = set()
    result: List[str] = []
    for q in raw:
        s = str(q).strip()
        key = s.lower()
        if s and key not in seen:
            seen.add(key)
            result.append(s)
        if len(result) >= cap:
            break
    return result
