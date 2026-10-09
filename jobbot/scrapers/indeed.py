"""Indeed.com scraper via the public job-search HTML endpoint.

Indeed throttles bots, but the search results page is fully server-rendered
when accessed with realistic browser headers. We paginate through several
keyword × location combos to maximize coverage, deduplicate by jobkey.
"""
from __future__ import annotations

import logging
import re
import time
from typing import List
from urllib.parse import quote_plus

import requests
from bs4 import BeautifulSoup

from . import RawJob, registry
from ..config import settings

log = logging.getLogger("jobbot.scrapers.indeed")

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/131.0.0.0 Safari/537.36"),
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
}


def _search_url(query: str, location: str, start: int) -> str:
    return (f"https://www.indeed.com/jobs?q={quote_plus(query)}"
            f"&l={quote_plus(location)}&fromage=14&start={start}")


def _parse(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    out = []
    # Indeed wraps each result in <div class="job_seen_beacon"> or similar
    for card in soup.select("div.job_seen_beacon, div.jobsearch-SerpJobCard, li.eu4oa1w0"):
        a = card.find("a", href=True)
        if not a:
            continue
        href = a["href"]
        # Normalize relative URL
        if href.startswith("/"):
            href = "https://www.indeed.com" + href
        title_el = card.find(["h2", "h3"])
        title = title_el.get_text(strip=True) if title_el else a.get_text(strip=True)
        company_el = (card.find(attrs={"data-testid": "company-name"})
                      or card.find("span", class_=re.compile("companyName"))
                      or card.find("a", class_=re.compile("companyName")))
        company = company_el.get_text(strip=True) if company_el else ""
        loc_el = card.find(attrs={"data-testid": "text-location"}) or card.find("div", class_=re.compile("companyLocation"))
        location = loc_el.get_text(strip=True) if loc_el else ""
        snippet_el = card.find("div", class_=re.compile("job-snippet")) or card.find("td", class_=re.compile("snippet"))
        snippet = snippet_el.get_text(" ", strip=True) if snippet_el else ""
        m = re.search(r"jk=([A-Za-z0-9]+)", href)
        jk = m.group(1) if m else href
        out.append({
            "id": jk, "title": title[:240], "company": company[:200],
            "location": location[:200], "url": href, "snippet": snippet[:2000],
        })
    return out


def scrape_indeed(keywords: List[str]) -> List[RawJob]:
    kw_lower = [k.lower() for k in keywords] or [""]
    queries = ["biochemistry", "molecular biology", "scientist phd", "protein"]
    from .. import user_prefs
    locs = user_prefs.scrape_locations() or ["Remote"]
    if user_prefs.includes_remote():
        locs = locs + ["Remote"]

    seen_jk: set[str] = set()
    out: List[RawJob] = []
    for q in queries:
        for loc in locs:
            # User-configurable pages (default 4) — 10 results per page
            max_start = max(1, int(settings.max_indeed_pages)) * 10
            for start in range(0, max_start, 10):
                try:
                    r = requests.get(_search_url(q, loc, start), headers=HEADERS, timeout=15)
                except Exception:
                    break
                if r.status_code != 200:
                    break
                cards = _parse(r.text)
                if not cards:
                    break
                added_this_page = 0
                for c in cards:
                    if c["id"] in seen_jk:
                        continue
                    seen_jk.add(c["id"])
                    haystack = f"{c['title']} {c['snippet']}".lower()
                    if not any(k in haystack for k in kw_lower):
                        continue
                    out.append(RawJob(
                        source="indeed",
                        title=c["title"],
                        company=c["company"] or "Indeed posting",
                        url=c["url"],
                        location=c["location"],
                        description=c["snippet"] or f"Open posting at {c['url']}",
                        posted_at=None,
                    ))
                    added_this_page += 1
                time.sleep(0.8)
                if added_this_page == 0:
                    # No new biochem cards on this page — assume tail of results
                    break
    log.info("Indeed: %d unique biochem postings", len(out))
    return out


registry.register(scrape_indeed)
