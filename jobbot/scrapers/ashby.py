"""Ashby HQ public job board scraper.

Many companies use Ashby. Public API:
  GET https://api.ashbyhq.com/posting-api/job-board/<slug>

`data/ashby_targets.txt`: one line per board:
  slug,display_name
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import List

import requests

from . import RawJob, registry

log = logging.getLogger("jobbot.scrapers.ashby")

TARGETS_FILE = Path("data/ashby_targets.txt")

DEFAULT_TARGETS: list[tuple[str, str]] = []



def _load_targets():
    out = list(DEFAULT_TARGETS)
    if TARGETS_FILE.exists():
        for line in TARGETS_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                out.append((parts[0], parts[1]))
    return out


HEADERS = {"User-Agent": "JobBot/1.0", "Accept": "application/json"}


def _fetch_board(slug: str):
    url = f"https://api.ashbyhq.com/posting-api/job-board/{slug}?includeCompensation=true"
    try:
        r = requests.get(url, headers=HEADERS, timeout=12)
        if r.status_code != 200:
            return []
        return r.json().get("jobs", []) or []
    except Exception as e:  # noqa: BLE001
        log.debug("ashby %s failed: %s", slug, e)
        return []


def scrape_ashby(keywords: List[str]) -> List[RawJob]:
    out: List[RawJob] = []
    kw_lower = [k.lower() for k in keywords] or [""]
    for slug, display in _load_targets():
        for j in _fetch_board(slug):
            title = j.get("title", "")
            desc = (j.get("descriptionPlain") or j.get("description") or "")
            haystack = f"{title} {desc}".lower()
            if not any(k in haystack for k in kw_lower):
                continue
            location = j.get("locationName", "") or j.get("location", "")
            published = j.get("publishedAt") or ""
            posted_dt = None
            try:
                if published:
                    posted_dt = datetime.fromisoformat(published.replace("Z", "+00:00"))
            except Exception:
                pass
            comp = j.get("compensation") or {}
            salary = ""
            if isinstance(comp, dict) and comp.get("summaryComponents"):
                # Best-effort comp string
                summary = comp.get("summaryComponents", [{}])[0]
                if isinstance(summary, dict):
                    salary = summary.get("label", "") or ""
            out.append(RawJob(
                source=f"ashby:{slug}",
                title=title[:240],
                company=display,
                url=j.get("jobUrl", "") or j.get("applyUrl", ""),
                location=location[:200],
                description=desc[:8000],
                salary=salary[:120],
                posted_at=posted_dt,
            ))
    log.info("Ashby: %d postings across %d boards", len(out), len(_load_targets()))
    return out


registry.register(scrape_ashby)
