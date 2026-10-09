"""Google Jobs scraper.

Google Jobs is rendered via the `htl;jobs` widget on standard search results,
which is fully JS-rendered and bot-protected. Instead of fighting that, this
scraper relies on the **JSON-LD JobPosting markup** that Google has indexed
across the web — many career sites embed it, and we can find them via Google
search using the `+` operator + site filters.

Two strategies are layered:

1. **Google CSE / Programmable Search**: if the user has set
   `GOOGLE_CSE_KEY` + `GOOGLE_CSE_CX` in `.env`, we use the official JSON API
   to find biochem job pages (clean, fast, costs ~$5/1000 queries).

2. **DuckDuckGo HTML fallback** (no API key needed): we issue a search like
   `site:linkedin.com/jobs OR site:lever.co biochemistry <your location>` and parse
   the result links. We then fetch each candidate URL and look for JSON-LD
   `@type=JobPosting` blobs.

Both strategies feed the same JSON-LD parser, so output is uniform.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Iterable, List

import requests
from bs4 import BeautifulSoup

from . import RawJob, registry
from ..config import settings

log = logging.getLogger("jobbot.scrapers.google_jobs")

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36")
# Explicit gzip-only Accept-Encoding avoids brotli decode failures in some envs.
HEADERS = {"User-Agent": UA, "Accept-Language": "en-US,en;q=0.9",
           "Accept-Encoding": "gzip, deflate"}

# Sites Google tends to index that publish JSON-LD JobPostings
TARGET_SITES = [
    "lever.co", "greenhouse.io", "myworkdayjobs.com",
    "smartrecruiters.com", "ashbyhq.com", "icims.com",
    "linkedin.com/jobs", "indeed.com/viewjob",
    "statejobs.ny.gov",
]


# ---------- Google CSE path (preferred when configured) ---------------------
def _cse_search(query: str, num: int = 10) -> list[str]:
    key = getattr(settings, "google_cse_key", "")
    cx = getattr(settings, "google_cse_cx", "")
    if not key or not cx:
        return []
    try:
        r = requests.get(
            "https://www.googleapis.com/customsearch/v1",
            params={"key": key, "cx": cx, "q": query, "num": min(num, 10)},
            headers=HEADERS, timeout=15,
        )
        if r.status_code != 200:
            return []
        return [item.get("link", "") for item in r.json().get("items", [])]
    except Exception as e:  # noqa: BLE001
        log.info("CSE failed: %s", e)
        return []


# ---------- Bing HTML fallback (works without an API key) ------------------
BING_BLOCK_RE = re.compile(
    r'<li class="b_algo"[^>]*>.*?<a\s+href="(https?://[^"]+)"',
    re.DOTALL | re.IGNORECASE,
)


def _bing_search(query: str, max_results: int | None = None) -> list[str]:
    if max_results is None:
        max_results = max(1, int(settings.max_search_engine_results))
    """Scrape Bing organic results via regex (BS4 mis-parses Bing's HTML)."""
    urls: list[str] = []
    # Paginate — Bing returns ~10 results per page; first=11 is page 2
    for first in range(1, max_results * 2, 10):
        try:
            r = requests.get(
                "https://www.bing.com/search",
                params={"q": query, "first": first},
                headers=HEADERS, timeout=15,
            )
            if r.status_code != 200:
                break
        except Exception as e:  # noqa: BLE001
            log.info("Bing fetch failed: %s", e)
            break
        found = BING_BLOCK_RE.findall(r.text)
        if not found:
            break
        for u in found:
            if u not in urls:
                urls.append(u)
        if len(urls) >= max_results:
            break
    return urls[:max_results]


# ---------- JSON-LD JobPosting parser --------------------------------------
JSON_LD_RE = re.compile(
    r'<script[^>]+type="application/ld\+json"[^>]*>(.+?)</script>',
    re.DOTALL | re.IGNORECASE,
)


def _extract_job_postings(url: str) -> list[RawJob]:
    try:
        r = requests.get(url, headers=HEADERS, timeout=12)
        if r.status_code != 200:
            return []
    except Exception:
        return []
    out: list[RawJob] = []
    for raw in JSON_LD_RE.findall(r.text):
        try:
            data = json.loads(raw.strip())
        except Exception:
            continue
        candidates = data if isinstance(data, list) else [data]
        # Some pages wrap in @graph
        for d in list(candidates):
            if isinstance(d, dict) and isinstance(d.get("@graph"), list):
                candidates.extend(d["@graph"])
        for d in candidates:
            if not isinstance(d, dict):
                continue
            if d.get("@type") != "JobPosting":
                continue
            title = (d.get("title") or "")[:240]
            if not title:
                continue
            org = d.get("hiringOrganization") or {}
            company = (org.get("name") if isinstance(org, dict) else org) or ""
            loc = d.get("jobLocation") or {}
            if isinstance(loc, list) and loc:
                loc = loc[0]
            addr = (loc.get("address") if isinstance(loc, dict) else None) or {}
            location_parts = [
                addr.get("addressLocality", ""),
                addr.get("addressRegion", ""),
                addr.get("addressCountry", "") if not addr.get("addressRegion") else "",
            ]
            location = ", ".join(p for p in location_parts if p)
            desc = re.sub(r"<[^>]+>", "", d.get("description", "") or "")[:8000]
            posted = d.get("datePosted") or ""
            posted_dt = None
            try:
                from datetime import datetime as _dt
                if posted:
                    posted_dt = _dt.fromisoformat(posted.replace("Z", "+00:00"))
            except Exception:
                pass
            out.append(RawJob(
                source="google_jobs",
                title=title,
                company=str(company)[:200],
                url=d.get("url") or url,
                location=location[:200],
                description=desc,
                posted_at=posted_dt,
            ))
    return out


def _build_queries() -> Iterable[str]:
    from .. import user_prefs
    base_kw = (settings.keywords or ["biochemistry"])[:2]
    locs = user_prefs.scrape_locations() or ["Remote"]
    site_clause = " OR ".join(f"site:{s}" for s in TARGET_SITES)
    for kw in base_kw:
        for loc in locs:
            yield f'({site_clause}) "{kw}" "{loc}"'
        if user_prefs.includes_remote():
            yield f'({site_clause}) "{kw}" remote'


def scrape_google_jobs(keywords: List[str]) -> List[RawJob]:
    from concurrent.futures import ThreadPoolExecutor, as_completed as _as_completed
    kw_lower = [k.lower() for k in keywords] or [""]
    seen_urls: set[str] = set()
    candidate_urls: list[str] = []

    using_cse = bool(getattr(settings, "google_cse_key", "")
                    and getattr(settings, "google_cse_cx", ""))
    for q in _build_queries():
        urls = _cse_search(q, num=10) if using_cse else _bing_search(q)
        for u in urls:
            if u not in seen_urls:
                seen_urls.add(u)
                candidate_urls.append(u)
    log.info("Google Jobs (%s): %d candidate posting URLs",
             "CSE" if using_cse else "Bing fallback", len(candidate_urls))

    cap = max(1, int(settings.max_google_jobs_urls))
    targets = candidate_urls[:cap]

    out: list[RawJob] = []
    seen_job_urls: set[str] = set()

    # Fetch up to 15 candidate URLs concurrently — turns a serial 5-10 min loop
    # into ~30 s while staying polite to individual domains.
    with ThreadPoolExecutor(max_workers=15, thread_name_prefix="gjobs") as ex:
        futures = {ex.submit(_extract_job_postings, url): url for url in targets}
        for fut in _as_completed(futures, timeout=120):
            try:
                postings = fut.result()
            except Exception:
                continue
            for rj in postings:
                if rj.url in seen_job_urls:
                    continue
                haystack = f"{rj.title} {rj.description}".lower()
                if not any(k in haystack for k in kw_lower):
                    continue
                seen_job_urls.add(rj.url)
                out.append(rj)

    log.info("Google Jobs: %d postings with JobPosting JSON-LD parsed", len(out))
    return out


registry.register(scrape_google_jobs)
