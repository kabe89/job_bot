"""SmartRecruiters public-postings API scraper.

Public endpoint (no auth):
  GET https://api.smartrecruiters.com/v1/companies/<company>/postings

`data/smartrecruiters_targets.txt`: one slug per line.
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import List

import requests

from . import RawJob, registry

log = logging.getLogger("jobbot.scrapers.smartrecruiters")

TARGETS_FILE = Path("data/smartrecruiters_targets.txt")
DEFAULT_TARGETS = []  # empty by default — most biotechs don't use SR. Extend via data/smartrecruiters_targets.txt


def _load_targets():
    out = list(DEFAULT_TARGETS)
    if TARGETS_FILE.exists():
        for line in TARGETS_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 1:
                slug = parts[0]
                display = parts[1] if len(parts) > 1 else slug
                out.append((slug, display))
    return out


HEADERS = {"User-Agent": "JobBot/1.0", "Accept": "application/json"}


def _fetch(slug: str, query: str):
    url = f"https://api.smartrecruiters.com/v1/companies/{slug}/postings"
    try:
        r = requests.get(url, params={"q": query, "limit": 50}, headers=HEADERS, timeout=12)
        if r.status_code != 200:
            return []
        return r.json().get("content", []) or []
    except Exception as e:  # noqa: BLE001
        log.debug("smartrecruiters %s failed: %s", slug, e)
        return []


def scrape_smartrecruiters(keywords: List[str]) -> List[RawJob]:
    out: List[RawJob] = []
    queries = keywords or [""]
    for slug, display in _load_targets():
        seen = set()
        for q in queries[:3]:  # cap to keep request count reasonable
            for j in _fetch(slug, q):
                jid = j.get("id")
                if not jid or jid in seen:
                    continue
                seen.add(jid)
                title = j.get("name", "")
                loc = j.get("location") or {}
                location = ", ".join(filter(None, [loc.get("city"), loc.get("region"), loc.get("country")]))
                rd = j.get("refNumber") or ""
                description = (j.get("jobAd", {}).get("sections", {}).get("jobDescription", {}) or {}).get("text", "")
                if not description:
                    description = f"Apply at {j.get('applyUrl','')}\nRef: {rd}"
                released = j.get("releasedDate")
                posted_dt = None
                try:
                    if released:
                        posted_dt = datetime.fromisoformat(released.replace("Z", "+00:00"))
                except Exception:
                    pass
                out.append(RawJob(
                    source=f"smartrecruiters:{slug}",
                    title=title[:240],
                    company=display,
                    url=j.get("ref", "") or j.get("applyUrl", "") or f"https://jobs.smartrecruiters.com/{slug}/{jid}",
                    location=location[:200],
                    description=description[:8000],
                    posted_at=posted_dt,
                ))
    log.info("SmartRecruiters: %d postings across %d boards", len(out), len(_load_targets()))
    return out


registry.register(scrape_smartrecruiters)
