"""LinkedIn Jobs scraper via the public guest jobs endpoint.

  https://www.linkedin.com/jobs-guest/jobs/api/seeMoreJobPostings/search
    ?keywords=<term>&location=<location>&start=<offset>

Returns HTML cards (no auth). We parse each card for title/company/location/url.
Cycles through user-configured keywords × location combos.

LinkedIn rate-limits aggressively; expect occasional 429s — those are caught
and logged at INFO level rather than raised.
"""
from __future__ import annotations

import logging
import re
import time
from typing import List
from urllib.parse import urlencode

import requests
from bs4 import BeautifulSoup

from . import RawJob, registry
from ..config import settings

log = logging.getLogger("jobbot.scrapers.linkedin")

BASE = "https://www.linkedin.com/jobs-guest/jobs/api/seeMoreJobPostings/search"
DETAIL = "https://www.linkedin.com/jobs/view/{job_id}/"
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/131.0.0.0 Safari/537.36"),
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
}


def _fetch_search(keywords: str, location: str, start: int = 0) -> str | None:
    q = urlencode({"keywords": keywords, "location": location, "start": start, "f_TPR": "r604800"})
    url = f"{BASE}?{q}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code == 429:
            log.info("LinkedIn 429 (rate-limited) for %s @ %s start=%d", keywords, location, start)
            return None
        if r.status_code != 200:
            return None
        return r.text
    except Exception as e:  # noqa: BLE001
        log.info("LinkedIn fetch failed: %s", e)
        return None


def _parse_cards(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    out = []
    for card in soup.find_all("li") + soup.find_all("div", class_="base-card"):
        link = card.find("a", href=re.compile(r"linkedin\.com/jobs/view/"))
        if not link:
            continue
        href = link.get("href", "").split("?")[0]
        title_el = card.find(["h3", "h2"])
        company_el = card.find("h4") or card.find("a", class_=re.compile("hidden-nested-link"))
        loc_el = card.find("span", class_=re.compile("location|job-search-card__location"))
        title = title_el.get_text(strip=True) if title_el else ""
        company = company_el.get_text(strip=True) if company_el else ""
        location = loc_el.get_text(strip=True) if loc_el else ""
        if not title:
            continue
        m = re.search(r"/jobs/view/(\d+)", href)
        job_id = m.group(1) if m else ""
        out.append({
            "id": job_id,
            "title": title[:240],
            "company": company[:200],
            "location": location[:200],
            "url": href or DETAIL.format(job_id=job_id),
        })
    return out


def _is_disabled() -> bool:
    return not bool(getattr(settings, "linkedin_enabled", True))


def _keyword_terms() -> list[str]:
    """Run several biochem keyword searches in parallel to maximize coverage.
    LinkedIn's keyword field expects short phrases — long boolean strings under-perform.
    Pick the top-N most useful biochem terms from configured keywords."""
    raw = [k for k in (settings.keywords or []) if 1 < len(k) <= 30]
    if not raw:
        return ["biochemistry"]
    # Preferred biochem search phrases (best-coverage terms first)
    preferred = ["biochemistry", "molecular biology", "protein", "enzyme",
                 "drug discovery", "structural biology", "mass spectrometry"]
    picked: list[str] = []
    for p in preferred:
        if p in raw and p not in picked:
            picked.append(p)
        if len(picked) >= 5:
            break
    # Fill remaining slots with anything else from user keywords
    for k in raw:
        if len(picked) >= 5:
            break
        if k not in picked:
            picked.append(k)
    return picked[:5]


def _location_terms() -> list[str]:
    """Cities to query LinkedIn against. Pulls from user_locations.txt + radius."""
    from .. import user_prefs
    base = user_prefs.scrape_locations()
    if not base:
        base = ["Remote"]
    if user_prefs.includes_remote():
        base.append("Remote, United States")
    # Deduplicate (case-insensitive)
    seen, out = set(), []
    for l in base:
        if l.lower() not in seen:
            seen.add(l.lower()); out.append(l)
    return out


def scrape_linkedin(keywords: List[str]) -> List[RawJob]:
    if _is_disabled():
        log.info("LinkedIn scraping disabled via config")
        return []
    kw_lower = [k.lower() for k in keywords] or [""]
    out: List[RawJob] = []
    seen_urls: set[str] = set()

    queries = _keyword_terms()
    locations = _location_terms()
    pages_per_query = max(1, int(settings.max_linkedin_pages))

    for kw in queries:
        for loc in locations:
            for page in range(pages_per_query):
                html = _fetch_search(kw, loc, start=page * 25)
                if not html:
                    break
                cards = _parse_cards(html)
                if not cards:
                    break
                for c in cards:
                    if c["url"] in seen_urls:
                        continue
                    seen_urls.add(c["url"])
                    haystack = f"{c['title']} {c['company']}".lower()
                    if not any(k in haystack for k in kw_lower) and not any(
                        t in haystack for t in ("scientist", "biologist", "biochemist", "research", "phd", "postdoc")
                    ):
                        continue
                    out.append(RawJob(
                        source="linkedin",
                        title=c["title"],
                        company=c["company"] or "LinkedIn posting",
                        url=c["url"],
                        location=c["location"],
                        description=f"LinkedIn job. View full description at {c['url']}",
                        posted_at=None,
                    ))
                time.sleep(1.0)  # polite delay
    log.info("LinkedIn: %d unique postings (kw × loc × pages = %d combos)",
             len(out), len(queries) * len(locations) * pages_per_query)
    return out


registry.register(scrape_linkedin)
