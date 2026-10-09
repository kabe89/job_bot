# jobbot/discovery/discoverers.py
"""The three discovery sources. Each returns raw (unvalidated, unscored)
DiscoveredTargets. websearch + llm are added in Task 6."""
from __future__ import annotations

import logging
import re
from typing import List, Optional, Tuple

from ..apply_questions import ASHBY_RE, GREENHOUSE_RE, LEVER_RE, WORKDAY_RE
from ..models import Job, session
from .model import DiscoveredTarget
from ..config import settings
from ..contacts import _web_search
from .. import ollama_client as oc
from .. import gemini_client as _g

log = logging.getLogger("jobbot.discovery")

# Board-ROOT fallbacks (review M2). The apply_questions regexes match only deep
# *posting* URLs (…/jobs/<id>, …/<uuid>, …/job/<slug>), but web/LLM search
# results are overwhelmingly careers landing / board-root URLs. These looser
# patterns capture the board slug/coordinate from a root URL; validation is the
# safety net for a wrong guess.
_GH_ROOT_RE = re.compile(r"(?:boards|job-boards)\.greenhouse\.io/([A-Za-z0-9_-]+)", re.I)
_LEVER_ROOT_RE = re.compile(r"jobs\.(?:eu\.)?lever\.co/([A-Za-z0-9_-]+)", re.I)
_ASHBY_ROOT_RE = re.compile(r"jobs\.ashbyhq\.com/([^/?#]+)", re.I)
_WD_ROOT_RE = re.compile(
    r"https?://([A-Za-z0-9_-]+)\.(wd\d+)\.myworkdayjobs\.com"
    r"(?:/[A-Za-z]{2}-[A-Za-z]{2})?/([A-Za-z0-9_%-]+)", re.I)


def _coord_from_url(url: str) -> Optional[Tuple[str, str, Optional[str]]]:
    """Extract (provider, key, wd_host) from an ATS job/careers URL, or None.

    Tries the strict posting-URL regexes first, then board-root fallbacks so
    careers-landing URLs from web/LLM search also resolve (review M2)."""
    if not url:
        return None
    m = GREENHOUSE_RE.search(url)
    if m:
        return ("greenhouse", m.group(1), None)
    m = LEVER_RE.search(url)
    if m:
        return ("lever", m.group(1), None)
    m = ASHBY_RE.search(url)
    if m:
        return ("ashby", m.group(1), None)
    m = WORKDAY_RE.match(url)
    if m:
        tenant, host, _lang, site = m.group(1), m.group(2), m.group(3), m.group(4)
        return ("workday", f"{tenant}/{site}", host)
    # --- board-root fallbacks ---
    m = _GH_ROOT_RE.search(url)
    if m:
        return ("greenhouse", m.group(1), None)
    m = _LEVER_ROOT_RE.search(url)
    if m:
        return ("lever", m.group(1), None)
    m = _ASHBY_ROOT_RE.search(url)
    if m:
        return ("ashby", m.group(1), None)
    m = _WD_ROOT_RE.search(url)
    if m:
        return ("workday", f"{m.group(1)}/{m.group(3)}", m.group(2))
    return None


def harvest_from_jobs() -> List[DiscoveredTarget]:
    """Scan stored Job URLs for ATS board coordinates (deduped by coord)."""
    out: dict = {}
    with session() as db:
        rows = db.query(Job.url, Job.company).all()
    for url, company in rows:
        c = _coord_from_url(url or "")
        if not c:
            continue
        provider, key, host = c
        coord = (provider, key)
        if coord in out:
            continue
        # A stored company may itself already be a bad coordinate slug (from the
        # era before _pretty_name); don't propagate that back out.
        name = (company or "").strip()
        if not name or "/" in name:
            name = _pretty_name(provider, key)
        out[coord] = DiscoveredTarget(
            provider=provider, key=key, wd_host=host,
            display_name=name, source="harvest")
    return list(out.values())


_ATS_SITES = "(site:greenhouse.io OR site:lever.co OR site:ashbyhq.com OR site:myworkdayjobs.com)"


def _known_tenant_names() -> dict:
    """tenant -> curated display name, from DEFAULT_TARGETS + workday_targets.txt.

    Reads the user's targets file too, so a name they curated there wins over a
    titlecased slug. Names that are themselves coordinates ("tenant_a/M_tx")
    are ignored -- that is the very bug this exists to stop.
    """
    from ..scrapers.workday import _load_targets
    out = {}
    try:
        for tenant, _board, name, _host in _load_targets():
            name = (name or "").strip()
            if name and "/" not in name:
                out[tenant.lower()] = name
    except Exception:  # noqa: BLE001 - never let config trouble break discovery
        pass
    return out


def _pretty_name(provider: str, key: str) -> str:
    """A human company name derived from an ATS coordinate.

    Web/LLM discovery finds a board URL with no company name attached. The
    coordinate must NEVER be used as the display name: for Workday the key is
    "tenant/board", which ends up in Job.company and then in generated letters
    ("Dear Acme/Careers") and breaks watchlist/referral matching.

    Prefers the curated DEFAULT_TARGETS name for a known tenant, else humanizes
    the slug. Best-effort by design -- discovery is a staged, human-approved
    queue, so the user fixes anything ugly at approval time.
    """
    key = (key or "").strip()
    if not key:
        return ""
    # Workday keys are "tenant/board"; every other provider is a flat slug.
    slug = key.split("/", 1)[0] if provider == "workday" else key
    slug = slug.strip()
    if not slug:
        return ""
    known = _known_tenant_names().get(slug.lower())
    if known:
        return known
    return " ".join(w.capitalize() for w in re.split(r"[-_\s]+", slug) if w)


def _candidates_from_urls(urls, source: str) -> List[DiscoveredTarget]:
    out: dict = {}
    for url in urls:
        c = _coord_from_url(url or "")
        if not c:
            continue
        provider, key, host = c
        coord = (provider, key)
        if coord in out:
            continue
        out[coord] = DiscoveredTarget(provider=provider, key=key, wd_host=host,
                                      display_name=_pretty_name(provider, key),
                                      source=source)
    return list(out.values())


def websearch_companies(profile) -> List[DiscoveredTarget]:
    """Search for ATS boards relevant to the profile; extract coords from result URLs."""
    domains = list(getattr(profile, "domains", []) or []) or ["technology"]
    # Geo scope is derived from the user's configured location (set in .env), so
    # no personal location is hard-coded here. Falls back to remote-only.
    _loc = (getattr(settings, "location_radius_hint", "")
            or getattr(settings, "applicant_location", "") or "").strip()
    geo = f'"{_loc}" OR remote' if _loc and "your" not in _loc.lower() else "remote"
    queries = [f'"{d}" careers ({geo}) {_ATS_SITES}' for d in domains]
    cap = int(getattr(settings, "discovery_websearch_queries", 8) or 8)
    urls: List[str] = []
    for q in queries[:cap]:
        try:
            urls += [r.get("url", "") for r in _web_search(q, max_results=10)]
        except Exception as e:  # noqa: BLE001
            log.warning("web search failed (%s) — skipping query.", e)
    return _candidates_from_urls(urls, "websearch")


def llm_suggest_companies(profile) -> List[DiscoveredTarget]:
    """Ask Ollama for fitting company names, resolve each to a board via web search."""
    cap = int(getattr(settings, "discovery_llm_max_companies", 15) or 15)
    try:
        out = oc._generate(
            "List real technology and industry companies that fit this candidate. "
            f"Return STRICT JSON: {{\"companies\": [\"Name\", ...]}} — at most {cap}.\n"
            f"Titles: {', '.join(getattr(profile, 'role_titles', []))}\n"
            f"Domains: {', '.join(getattr(profile, 'domains', []))}\n"
            f"Summary: {getattr(profile, 'summary', '')}",
            temperature=0.4, include_profile=False)
        names = (_g._parse_json_block(out) or {}).get("companies") or []
    except Exception as e:  # noqa: BLE001
        log.warning("LLM company suggestion failed (%s).", e)
        return []
    urls: List[str] = []
    for name in names[:cap]:
        try:
            hits = _web_search(f'{name} careers {_ATS_SITES}', max_results=5)
            urls += [r.get("url", "") for r in hits]
        except Exception as e:  # noqa: BLE001
            log.info("resolve failed for %r (%s)", name, e)
    return _candidates_from_urls(urls, "llm")
